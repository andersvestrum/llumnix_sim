from typing import List, Optional, Tuple
import math

from vidur.entities import Request, Batch
from vidur.scheduler.replica_scheduler.base_replica_scheduler import BaseReplicaScheduler
from vidur.logger import init_logger

logger = init_logger(__name__)


class LlumletLocalScheduler(BaseReplicaScheduler):
    """
    Llumnix 'llumlet' — per-replica local scheduler.
    Handles request queuing, local load computation, and migration coordination.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._priority_queue: List[Tuple[int, Request]] = []
        self._migration_coordinator = None
        self._num_running_batches = 0  # Base.on_schedule() relies on this

    # ---------- Load Calculation ----------
    def compute_virtual_usage(self) -> float:
        """Simple virtual-usage: queue + running + memory fraction."""
        queue_len = len(self._request_queue)
        running_reqs = len(self._allocation_map)
        mem_ratio = (self._num_allocated_blocks / self._config.num_blocks) if self._config.num_blocks else 0.0
        alpha = beta = gamma = 1.0
        return alpha * queue_len + beta * running_reqs + gamma * mem_ratio

    # ---------- Queue and Scheduling ----------
    def enqueue_request(self, request: Request) -> None:
        """Priority-aware enqueue (higher number = higher priority)."""
        pr = getattr(request, "priority", 0)
        self._priority_queue.append((pr, request))
        # stable sort by priority, high→low
        self._priority_queue.sort(key=lambda x: x[0], reverse=True)

    def get_next_request(self) -> Optional[Request]:
        if not self._priority_queue:
            return None
        _, req = self._priority_queue.pop(0)
        return req

    # ---------- Migration Coordination ----------
    def decide_migration_candidate(self, target_capacity_blocks: int) -> Optional[int]:
        """
        Choose a running request (smaller KV first) that fits on target.
        """
        candidates = [
            (rid, blocks)
            for rid, blocks in self._allocation_map.items()
            if blocks <= target_capacity_blocks
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]

    def execute_migration(self, request_id: int, dest_scheduler: "LlumletLocalScheduler") -> bool:
        """
        Perform a simple migrate: pre-allocate on dest, free on source.
        """
        num_blocks = self._allocation_map.get(request_id, 0)
        if num_blocks <= 0 or not dest_scheduler.can_allocate(num_blocks):
            return False

        dest_scheduler.allocate(request_id, num_blocks)
        self.free(request_id)
        logger.debug(
            f"Migrated request {request_id} from {self.replica_id} to {dest_scheduler.replica_id}"
        )
        return True

    # ---------- BaseReplicaScheduler Overrides ----------
    def on_batch_end(self, batch: Batch) -> None:
        # Free KV blocks for finished prefill/decode step
        self.free_batch(batch)
        self._num_running_batches = max(0, self._num_running_batches - 1)

        req = batch.requests[0]
        remaining = getattr(req, "num_decode_tokens", 0)

        # Skip if the request is already completed
        if getattr(req, "completed", False):
            logger.debug(f"[Replica {self._replica_id}] Request {req.id} completed — no more decode batches.")
            return

        # If the request still has decode tokens remaining, enqueue for next step
        remaining = getattr(req, "num_decode_tokens", 0)
        if remaining > 0:
            pr = getattr(req, "priority", 0)
            self._priority_queue.append((pr, req))
            self._priority_queue.sort(key=lambda x: x[0], reverse=True)
            logger.debug(
                f"[Replica {self._replica_id}] Request {req.id} enqueued for decode ({remaining} tokens remaining)"
            )



    def _get_next_batch(self) -> Optional[Batch]:
        """
        Build a single-request batch (simple FCFS within priority).
        Allocate KV blocks before returning the Batch.
        """
        req = self.get_next_request()
        if req is None:
            return None

        # tokens to run next step; compute needed KV blocks
        num_tokens = self._get_request_next_num_tokens(req)
        blocks_needed = max(1, math.ceil(num_tokens / max(1, self._config.block_size)))

        if not self.can_allocate(blocks_needed):
            # can't run now; put it back at the front of the priority queue
            self._priority_queue.insert(0, (getattr(req, "priority", 0), req))
            return None

        # allocate KV for this request
        self.allocate(req.id, blocks_needed)

        # Base.on_schedule() will increment _num_running_batches after we return a batch
        return Batch(
            requests=[req],
            replica_id=self._replica_id,
            num_tokens=[num_tokens],
        )

    # ================================================================
    # Llumnix global ↔ llumlet interface methods
    # ================================================================
    def report_freeness(self) -> float:
        """
        Freeness F = (total_blocks - allocated_blocks) / total_blocks.
        Higher is "freer".
        """
        total = max(1, self._config.num_blocks)
        used = self._num_allocated_blocks
        return (total - used) / total

    def has_capacity(self, num_blocks: int = 1) -> bool:
        return self.can_allocate(num_blocks)

    def begin_migration_to(self, dest_scheduler: "LlumletLocalScheduler"):
        """
        Pick a running request to migrate (smallest KV first). If none running,
        try moving a queued request instead.
        """
        # 1) prefer running requests (they have allocation info)
        # try smallest that fits
        if self._allocation_map:
            # how many blocks can dest still take?
            dest_free = dest_scheduler._config.num_blocks - dest_scheduler._num_allocated_blocks
            candidate = self.decide_migration_candidate(dest_free)
            if candidate is not None and self.execute_migration(candidate, dest_scheduler):
                return (candidate, self._replica_id, dest_scheduler.replica_id)

        # 2) else move a queued request if any
        if self._priority_queue:
            _, req = self._priority_queue.pop(0)
            dest_scheduler.enqueue_request(req)
            return (req.id, self._replica_id, dest_scheduler.replica_id)

        return None
