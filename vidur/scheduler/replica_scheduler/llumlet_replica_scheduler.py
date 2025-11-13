from typing import List, Optional, Tuple, Dict
import math

from vidur.entities import Request, Batch
from vidur.scheduler.replica_scheduler.base_replica_scheduler import BaseReplicaScheduler
from vidur.logger import init_logger

logger = init_logger(__name__)


class LlumletLocalScheduler(BaseReplicaScheduler):
    """
    Llumnix 'llumlet' — per-replica local scheduler with policy-faithful freeness.

    Freeness F = (M - ΣV) / B, where:
      - M: total KV blocks on the replica.
      - ΣV: virtual usage sum from multiple sources:
          * Physical KV in-use by running requests.
          * Head-of-line (HoL) queued *demand* in KV blocks (de-frag pressure).
          * Execution-priority headroom for high-priority requests.
          * Optional drain pressure (fake ∞) when replica is marked draining.
      - B: batch-normalization denominator (blocks per batch); defaults to 1.
    """

    # -------------------- Construction --------------------
    def __init__(
        self,
        replica_config,
        replica_scheduler_config,
        request_generator_config,
        replica,
        num_stages,
        execution_time_predictor,
    ):
        # BaseReplicaScheduler sets:
        #   self._config = replica_scheduler_config
        #   self._replica_id = replica.id
        #   self._num_stages
        #   self._allocation_map, self._num_allocated_blocks
        #   self._replica_stage_schedulers (for Chrome trace + stages)
        super().__init__(
            replica_config,
            replica_scheduler_config,
            request_generator_config,
            replica,
            num_stages,
            execution_time_predictor,
        )

        # Queue holds (priority, monotonic_seq, Request)
        # Lower priority value = "higher" priority (0 > 1 > 2...)
        self._priority_queue: List[Tuple[int, int, Request]] = []
        self._enqueue_seq: int = 0

        # Track number of running batches (BaseReplicaScheduler
        # also uses this; we only DECREMENT it in on_batch_end).
        self._num_running_batches: int = 0

        # Book-keeping for migration and priority-aware logic
        self._request_index: Dict[int, Request] = {}
        self._reservations: Dict[int, int] = {}     # req_id -> reserved_blocks (dest)
        self._migrations_out: Dict[int, int] = {}   # req_id -> blocks (source, running)

        # Optional drain flag (for scale-in)
        self._is_draining: bool = False

        # Tunables from replica scheduler config (stored in self._config)
        cfg = self._config
        self._headroom_blocks_per_hi: int = getattr(cfg, "priority_headroom_blocks", 0)
        self._high_priority_threshold: int = getattr(cfg, "high_priority_threshold", 1)
        # Batch normalization denominator B (blocks per batch)
        self._batch_normalizer_B: int = getattr(cfg, "batch_blocks", 1) or 1

    # -------------------- Queueing & batching --------------------
    def enqueue_request(self, request: Request) -> None:
        """
        Insert request into priority queue.
        """
        pr = getattr(request, "priority", 0)
        self._enqueue_seq += 1
        self._priority_queue.append((pr, self._enqueue_seq, request))
        # sort by (priority, seq): smaller priority first, then FIFO
        self._priority_queue.sort(key=lambda x: (x[0], x[1]))
        self._request_index[request.id] = request

    def _pop_next_request(self) -> Optional[Request]:
        if not self._priority_queue:
            return None
        _, _, req = self._priority_queue.pop(0)
        return req

    # Override free to be tolerant (pop(..., None))
    def free(self, *request_ids: int) -> None:
        for request_id in request_ids:
            num_blocks = self._allocation_map.pop(request_id, None)
            if num_blocks is not None:
                self._num_allocated_blocks -= num_blocks
        assert self._num_allocated_blocks >= 0

    def _peek_hol_request(self) -> Optional[Request]:
        if not self._priority_queue:
            return None
        return self._priority_queue[0][2]

    def _blocks_for_request_next_step(self, req: Request) -> int:
        """
        Map 'next num tokens' (prefill or 1 decode) to KV blocks.
        """
        num_tokens = self._get_request_next_num_tokens(req)
        block = max(1, getattr(self._config, "block_size", 1))
        return max(1, math.ceil(num_tokens / block))

    def _get_next_batch(self) -> Optional[Batch]:
        """
        Build a continuous-batching multi-request batch:
        • Greedily pick requests in priority/FIFO order
        • Include as many as fit in remaining KV memory
        • Stop when next request cannot be allocated
        • Remove chosen requests from the queue
        • Allocate KV for each request

        NOTE: DO NOT touch self._num_running_batches here.
        BaseReplicaScheduler.on_schedule() increments it
        once per returned batch. We only decrement in on_batch_end.
        """

        if not self._priority_queue:
            return None

        chosen_entries: List[Tuple[int, int, Request]] = []
        chosen_requests: List[Request] = []
        total_blocks = 0
        remaining_queue = list(self._priority_queue)  # snapshot

        for pr, seq, req in remaining_queue:
            blocks = self._blocks_for_request_next_step(req)

            # If adding this request exceeds KV capacity → stop packing
            if not self.can_allocate(total_blocks + blocks):
                break

            total_blocks += blocks
            chosen_entries.append((pr, seq, req))
            chosen_requests.append(req)

        if not chosen_requests:
            return None

        # Remove selected from real queue
        for item in chosen_entries:
            self._priority_queue.remove(item)

        # Allocate KV blocks and keep index
        for req in chosen_requests:
            blocks = self._blocks_for_request_next_step(req)
            self.allocate(req.id, blocks)
            self._request_index[req.id] = req

        # Per-request token counts for this iteration
        num_tokens = [self._get_request_next_num_tokens(req) for req in chosen_requests]

        # IMPORTANT: DO NOT increment _num_running_batches here
        # BaseReplicaScheduler.on_schedule() will do that.

        return Batch(
            replica_id=self._replica_id,
            requests=chosen_requests,
            num_tokens=num_tokens,
        )

    def get_batch_compute_cost(self, batch: Batch) -> float:
        """
        Return simulated compute duration (in seconds) to process this batch.
        vLLM/Llumnix semantics:
        - One decode iteration per batch
        - Cost grows with the number of active requests (batch size)
        """
        base_cost = getattr(self._config, "per_request_compute_cost", 0.01)
        batch_size = len(batch.requests)
        return base_cost * batch_size

    def on_batch_end(self, batch: Batch) -> None:
        """
        Called by BatchEndEvent *after* Batch.on_batch_end(...)
        has already updated all Request objects.

        We:
        - free KV
        - decrement running batch counter
        - re-enqueue non-completed requests
        - finalize migrations of running requests if needed
        """
        # 1) Free KV blocks for all requests in this batch
        self.free_batch(batch)
        self._num_running_batches = max(0, self._num_running_batches - 1)

        # 2) Handle each request in the batch independently
        for req in batch.requests:
            req_id = req.id

            # At this point, Batch.on_batch_end has already called
            # request.on_batch_end(time, num_tokens), so:
            #   - req.completed is correct
            #   - req.is_prefill_complete is updated
            #   - num_processed_tokens is updated

            if req.completed:
                self._request_index.pop(req_id, None)
                self._migrations_out.pop(req_id, None)
                self._reservations.pop(req_id, None)
                logger.debug(f"[Replica {self._replica_id}] Request {req_id} completed.")
            else:
                pr = getattr(req, "priority", 0)
                self._enqueue_seq += 1
                self._priority_queue.append((pr, self._enqueue_seq, req))
                self._priority_queue.sort(key=lambda x: (x[0], x[1]))
                logger.debug(
                    f"[Replica {self._replica_id}] Request {req_id} re-enqueued after batch."
                )

            # Migration handshake finalization for this request, if needed
            if req_id in self._migrations_out:
                blocks = self._migrations_out[req_id]
                if self._dest_commit_if_reserved(req_id, blocks):
                    self._allocation_map.pop(req_id, None)
                else:
                    self._abort_reservation(req_id)
                self._migrations_out.pop(req_id, None)

    # -------------------- Virtual-usage policy --------------------
    def _virtual_usage_physical(self) -> int:
        return int(self._num_allocated_blocks)

    def _virtual_usage_hol_demand(self) -> int:
        hol = self._peek_hol_request()
        if not hol:
            return 0
        return self._blocks_for_request_next_step(hol)

    def _virtual_usage_priority_headroom(self) -> int:
        if self._headroom_blocks_per_hi <= 0:
            return 0
        hi_thresh = self._high_priority_threshold
        hi_count = 0
        # queued
        for pr, _, _req in self._priority_queue:
            if pr <= hi_thresh:
                hi_count += 1
        # running (approximate: allocated requests with high priority)
        for rid in list(self._allocation_map.keys()):
            req = self._request_index.get(rid)
            if req and getattr(req, "priority", 0) <= hi_thresh:
                hi_count += 1
        if hi_count == 0:
            return 0
        return int(math.ceil(self._headroom_blocks_per_hi / max(1, hi_count)))

    def _virtual_usage_drain(self) -> int:
        if not self._is_draining:
            return 0
        return 10 * max(1, self._config.num_blocks)

    def _sum_virtual_usage(self) -> int:
        return (
            self._virtual_usage_physical()
            + self._virtual_usage_hol_demand()
            + self._virtual_usage_priority_headroom()
            + self._virtual_usage_drain()
        )

    def report_freeness(self) -> float:
        M = max(1, self._config.num_blocks)
        SigmaV = self._sum_virtual_usage()
        B = max(1, self._batch_normalizer_B)
        return (M - SigmaV) / B  # negative allowed

    def has_capacity(self, num_blocks: int = 1) -> bool:
        return self.can_allocate(num_blocks)

    # -------------------- Migration handshake --------------------
    def _reserve_on_dest(self, dest: "LlumletLocalScheduler", req_id: int, blocks: int) -> bool:
        if dest._reservations.get(req_id):
            return True
        if not dest.can_allocate(blocks):
            return False
        dest._reservations[req_id] = blocks
        return True

    def _abort_reservation(self, req_id: int) -> None:
        self._reservations.pop(req_id, None)

    def _dest_commit_if_reserved(self, req_id: int, blocks: int) -> bool:
        if self._reservations.get(req_id) != blocks:
            return False
        if not self.can_allocate(blocks):
            self._reservations.pop(req_id, None)
            return False
        self.allocate(req_id, blocks)
        self._reservations.pop(req_id, None)
        logger.info(
            f"[Migration] Request {req_id} successfully committed on destination "
            f"replica {self.replica_id} ({blocks} blocks)"
        )
        return True

    def decide_migration_candidate(self, target_capacity_blocks: int) -> Optional[int]:
        """
        Pick a running or queued request to move, preferring (low priority, small KV).
        """
        candidates: List[Tuple[int, int, int]] = []  # (priority, blocks, req_id)

        # Running
        for req_id, blocks in self._allocation_map.items():
            if blocks <= target_capacity_blocks:
                pr = getattr(self._request_index.get(req_id), "priority", 0)
                candidates.append((pr, blocks, req_id))

        # Queued — approximate blocks using next-step demand
        for pr, _, req in self._priority_queue:
            b = self._blocks_for_request_next_step(req)
            if b <= target_capacity_blocks:
                candidates.append((pr, b, req.id))

        if not candidates:
            return None
        # lowest priority (largest pr) first, then smallest KV
        candidates.sort(key=lambda t: (-t[0], t[1]))
        return candidates[0][2]

    def begin_migration_to(self, dest_scheduler: "LlumletLocalScheduler") -> Optional[Tuple[int, int, int]]:
        """
        Simulated live migration handshake:
        1) Choose candidate (low-pri, small KV).
        2) Reserve blocks on dest.
        3) If queued: move immediately.
        4) If running: mark as in-flight; final commit happens after the current
           batch step ends (in on_batch_end).
        """
        dest_free = dest_scheduler._config.num_blocks - (
            dest_scheduler._num_allocated_blocks + sum(dest_scheduler._reservations.values() or [])
        )
        if dest_free <= 0:
            return None

        cand_id = self.decide_migration_candidate(dest_free)
        if cand_id is None:
            logger.debug(
                f"[Replica {self.replica_id}] No suitable migration candidate "
                f"found for dest {dest_scheduler.replica_id}"
            )
            return None

        blocks = self._allocation_map.get(cand_id)
        # Queued path
        if blocks is None:
            req = self._request_index.get(cand_id)
            if not req:
                return None
            blocks = self._blocks_for_request_next_step(req)
            if self._reserve_on_dest(dest_scheduler, cand_id, blocks):
                # Move queued request immediately
                for i, (_pr, _seq, _r) in enumerate(list(self._priority_queue)):
                    if _r.id == cand_id:
                        self._priority_queue.pop(i)
                        break
                dest_scheduler.enqueue_request(req)
                dest_scheduler._dest_commit_if_reserved(cand_id, blocks)
                logger.info(
                    f"[Migration] Queued req {cand_id} moved from replica {self.replica_id} "
                    f"-> {dest_scheduler.replica_id} ({blocks} blocks)"
                )
                return (cand_id, self.replica_id, dest_scheduler.replica_id)
            return None

        # Running request path
        if not self._reserve_on_dest(dest_scheduler, cand_id, blocks):
            return None

        self._migrations_out[cand_id] = blocks
        logger.info(
            f"[Migration] Begin live-migration of req {cand_id} from replica {self.replica_id} "
            f"-> {dest_scheduler.replica_id} (blocks={blocks})"
        )
        return (cand_id, self.replica_id, dest_scheduler.replica_id)

    def set_draining(self, draining: bool) -> None:
        self._is_draining = draining
