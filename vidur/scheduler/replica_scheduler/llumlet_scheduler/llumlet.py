from typing import List, Optional, Tuple
import math

from vidur.entities import Request, Batch
from vidur.scheduler.replica_scheduler.base_replica_scheduler import BaseReplicaScheduler
from vidur.logger import init_logger

logger = init_logger(__name__)


from typing import List, Tuple, Optional, Dict
import math

from vidur.config import SimulationConfig
from vidur.entities import Request, Batch
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler
from vidur.scheduler.replica_scheduler.base_replica_scheduler import BaseReplicaScheduler
from vidur.execution_time_predictor import ExecutionTimePredictorRegistry
from vidur.logger import init_logger

logger = init_logger(__name__)

# ================================================================
# Replica scheduler: LlumletLocalScheduler — implements virtual-usage policy,
# negative freeness, priority headroom, de-frag pressure, and a simple
# (simulated) live-migration handshake compatible with Vidur.
# ================================================================

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
      - B: batch-normalization denominator (blocks per batch); defaults to 1
           if the framework lacks an explicit notion. Negative F is allowed.
    """

    # -------------------- Construction --------------------
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Queue holds (priority, monotonic_seq, Request). Higher priority first; FIFO within equal priority
        self._prio_queue: List[Tuple[int, int, Request]] = []
        self._enqueue_seq = 0
        self._num_running_batches = 0
        self._decode_progress: Dict[int, int] = {}

        # Book-keeping to enable priority-aware migration choices
        self._request_index: Dict[int, Request] = {}

        # Reservations for the migration handshake (dest side)
        self._reservations: Dict[int, int] = {}  # req_id -> reserved_blocks

        # Source-side migration in-flight (req_id -> blocks)
        self._migrations_out: Dict[int, int] = {}

        # Optional: replica draining pressure for scale-in
        self._is_draining = False

        # Tunables (read from replica scheduler config when present)
        rs_cfg = getattr(self, "_replica_scheduler_config", None)
        self._headroom_blocks_per_hi = getattr(rs_cfg, "priority_headroom_blocks", 0)
        self._high_priority_threshold = getattr(rs_cfg, "high_priority_threshold", 1)
        # Use 1 if framework has no explicit batch-size in blocks
        self._batch_normalizer_B = getattr(self._config, "batch_blocks", 1) or 1

    # -------------------- Queueing & batching --------------------
    def enqueue_request(self, request: Request) -> None:
        pr = getattr(request, "priority", 0)
        self._enqueue_seq += 1
        self._prio_queue.append((pr, self._enqueue_seq, request))
        self._prio_queue.sort(key=lambda x: (-x[0], x[1]))  # high→low priority, then FIFO
        self._request_index[request.id] = request

    def _pop_next_request(self) -> Optional[Request]:
        if not self._prio_queue:
            return None
        _, _, req = self._prio_queue.pop(0)
        return req

    def _peek_hol_request(self) -> Optional[Request]:
        if not self._prio_queue:
            return None
        return self._prio_queue[0][2]

    def _blocks_for_request_next_step(self, req: Request) -> int:
        # Convert the next scheduled token count into KV blocks required for the step
        num_tokens = self._get_request_next_num_tokens(req)
        block = max(1, getattr(self._config, "block_size", 1))
        return max(1, math.ceil(num_tokens / block))

    def _get_next_batch(self) -> Optional[Batch]:
        # Single-request batches, FCFS within priority.
        req = self._pop_next_request()
        if req is None:
            return None

        blocks_needed = self._blocks_for_request_next_step(req)

        # Try to allocate; if not possible now, push back to HoL to reflect pressure.
        if not self.can_allocate(blocks_needed):
            # Reinsert at front preserving priority order (acts as pressure source for F)
            self._prio_queue.insert(0, (getattr(req, "priority", 0), -1, req))
            return None

        # Allocate KV for this step and remember the request for migration scoring
        self.allocate(req.id, blocks_needed)
        self._request_index[req.id] = req

        return Batch(
            requests=[req],
            replica_id=self._replica_id,
            num_tokens=[self._get_request_next_num_tokens(req)],
        )

    def on_batch_end(self, batch: Batch) -> None:
        req = batch.requests[0]
        req_id = req.id

        # Only free if we still have an allocation for this request on this replica
        if req_id in self._allocation_map:
            self.free_batch(batch)
        else:
            logger.debug(
                f"[Replica {self._replica_id}] Skip free_batch for req {req_id} "
                f"(no allocation present; likely migrated or already freed)"
            )

        self._num_running_batches = max(0, self._num_running_batches - 1)

        # Skip if the request is completed
        if getattr(req, "completed", False):
            self._request_index.pop(req.id, None)
            self._migrations_out.pop(req.id, None)
            self._reservations.pop(req.id, None)
            logger.debug(f"[Replica {self._replica_id}] Request {req.id} completed.")
            return

        # --- New: Explicitly handle decoding steps ---
        remaining = self._decode_progress.get(req.id, getattr(req, "num_decode_tokens", 0))
        if remaining > 0:
            next_remaining = remaining - 1
            self._decode_progress[req.id] = next_remaining
            pr = getattr(req, "priority", 0)
            self._enqueue_seq += 1
            self._prio_queue.append((pr, self._enqueue_seq, req))
            self._prio_queue.sort(key=lambda x: (-x[0], x[1]))  # high→low priority, then FIFO
            logger.debug(
                f"[Replica {self._replica_id}] Decode step for request {req.id}: "
                f"{next_remaining} tokens remaining"
            )
        else:
            logger.debug(f"[Replica {self._replica_id}] Request {req.id} completed decode.")
            self._decode_progress.pop(req.id, None)


        # --- Keep your migration handshake logic below unchanged ---
        if req.id in self._migrations_out:
            blocks = self._migrations_out[req.id]
            if self._dest_commit_if_reserved(req.id, blocks):
                self._allocation_map.pop(req.id, None)
                self._migrations_out.pop(req.id, None)
            else:
                self._abort_reservation(req.id)
                self._migrations_out.pop(req.id, None)


    # -------------------- Virtual-usage policy --------------------
    def _virtual_usage_physical(self) -> int:
        # Physical KV-in use in blocks
        return int(self._num_allocated_blocks)

    def _virtual_usage_hol_demand(self) -> int:
        # Demand from the head-of-line (first queued) request — drives de-fragmentation
        hol = self._peek_hol_request()
        if not hol:
            return 0
        return self._blocks_for_request_next_step(hol)

    def _virtual_usage_priority_headroom(self) -> int:
        # Headroom for high-priority execution; spreads across number of high-priority requests
        if self._headroom_blocks_per_hi <= 0:
            return 0
        # Count high-priority running+queued
        hi_thresh = self._high_priority_threshold
        hi_count = 0
        # queued
        for pr, _, _req in self._prio_queue:
            if pr >= hi_thresh:
                hi_count += 1
        # running — approximate by number of allocated requests with priority >= thresh
        for rid in list(self._allocation_map.keys()):
            req = self._request_index.get(rid)
            if req and getattr(req, "priority", 0) >= hi_thresh:
                hi_count += 1
        if hi_count == 0:
            return 0
        # Distribute headroom across high-priority population
        return int(math.ceil(self._headroom_blocks_per_hi / max(1, hi_count)))

    def _virtual_usage_drain(self) -> int:
        # Fake ∞ when draining — use a large sentinel proportional to capacity
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
        # Allow negative F — conveys pressure to global layer
        return (M - SigmaV) / B

    def has_capacity(self, num_blocks: int = 1) -> bool:
        # For the global layer, capacity is a soft hint. Keep strict allocation check here.
        return self.can_allocate(num_blocks)

    # -------------------- Migration handshake --------------------
    def _reserve_on_dest(self, dest: "LlumletLocalScheduler", req_id: int, blocks: int) -> bool:
        if dest._reservations.get(req_id):
            return True
        if not dest.can_allocate(blocks):
            return False
        # Reserve without touching allocation map — prevents races with local allocator
        dest._reservations[req_id] = blocks
        return True

    def _abort_reservation(self, req_id: int) -> None:
        # Best-effort abort on *any* replica that may have reserved this id (caller ensures right dest)
        self._reservations.pop(req_id, None)

    def _dest_commit_if_reserved(self, req_id: int, blocks: int) -> bool:
        # If we are the *destination*, convert reservation -> allocation
        if self._reservations.get(req_id) != blocks:
            return False
        if not self.can_allocate(blocks):
            # Should not happen, but if it does, drop reservation
            self._reservations.pop(req_id, None)
            return False
        self.allocate(req_id, blocks)
        self._reservations.pop(req_id, None)
        logger.info(
            f"[Migration] Request {req_id} successfully committed on destination replica {self.replica_id} ({blocks} blocks)"
        )
        return True

    def decide_migration_candidate(self, target_capacity_blocks: int) -> Optional[int]:
        """Pick a running or queued request to move, preferring (low priority, small KV)."""
        candidates: List[Tuple[int, int, int]] = []  # (priority, blocks, req_id)

        # Running
        for req_id, blocks in self._allocation_map.items():
            if blocks <= target_capacity_blocks:
                pr = getattr(self._request_index.get(req_id), "priority", 0)
                candidates.append((pr, blocks, req_id))

        # Queued — approximate blocks using next-step demand
        for pr, _, req in self._prio_queue:
            b = self._blocks_for_request_next_step(req)
            if b <= target_capacity_blocks:
                candidates.append((pr, b, req.id))

        if not candidates:
            return None
        candidates.sort(key=lambda t: (t[0], t[1]))  # lowest priority first, then smallest KV
        return candidates[0][2]

    def begin_migration_to(self, dest_scheduler: "LlumletLocalScheduler") -> Optional[Tuple[int, int, int]]:
        """
        Simulated live migration handshake:
        1) Choose candidate (low-pri, small KV).
        2) Reserve blocks on dest.
        3) Mark as in-flight on source; final commit happens after the current batch step ends.
        """
        dest_free = dest_scheduler._config.num_blocks - (
            dest_scheduler._num_allocated_blocks + sum(dest_scheduler._reservations.values() or [])
        )
        if dest_free <= 0:
            return None

        cand_id = self.decide_migration_candidate(dest_free)
        if cand_id is None:
            logger.debug(f"[Replica {self.replica_id}] No suitable migration candidate found for dest {dest_scheduler.replica_id}")
            return None

        blocks = self._allocation_map.get(cand_id)
        if blocks is None:
            # Queued request path
            req = self._request_index.get(cand_id)
            if not req:
                return None
            blocks = self._blocks_for_request_next_step(req)
            if self._reserve_on_dest(dest_scheduler, cand_id, blocks):
                # Move queued request immediately
                for i, (_pr, _seq, _r) in enumerate(list(self._prio_queue)):
                    if _r.id == cand_id:
                        self._prio_queue.pop(i)
                        break
                dest_scheduler.enqueue_request(req)
                dest_scheduler._dest_commit_if_reserved(cand_id, blocks)
                logger.info(
                    f"[Migration] Queued req {cand_id} moved from replica {self.replica_id} -> {dest_scheduler.replica_id} ({blocks} blocks)"
                )
                return (cand_id, self.replica_id, dest_scheduler.replica_id)
            return None

        # Running request — do the handshake
        if not self._reserve_on_dest(dest_scheduler, cand_id, blocks):
            return None

        self._migrations_out[cand_id] = blocks
        logger.info(
            f"[Migration] Begin live-migration of req {cand_id} from replica {self.replica_id} "
            f"-> {dest_scheduler.replica_id} (blocks={blocks})"
        )
        return (cand_id, self.replica_id, dest_scheduler.replica_id)


    # -------------------- Scale-in/out hooks --------------------
    def set_draining(self, draining: bool) -> None:
        self._is_draining = draining