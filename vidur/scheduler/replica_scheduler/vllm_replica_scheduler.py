from math import ceil
from typing import List

from vidur.entities.batch import Batch, Request
from vidur.scheduler.replica_scheduler.base_replica_scheduler import BaseReplicaScheduler
from vidur.logger import init_logger

logger = init_logger(__name__)


class VLLMReplicaScheduler(BaseReplicaScheduler):
    """
    Replica-level scheduler approximating vLLM behavior.
    Handles both prefill and decode requests without duplication,
    while respecting memory and batch-size constraints.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._preempted_requests: List[Request] = []
        self._num_running_batches = 0

        # Loose cap; memory is handled explicitly
        self._max_micro_batch_size = self._config.batch_size_cap // self._num_stages
        self._watermark_blocks = int(
            self._config.watermark_blocks_fraction * self._config.num_blocks
        )

    # -------------------------------------------------------------------------
    # Batch lifecycle
    # -------------------------------------------------------------------------
    def on_batch_end(self, batch: Batch) -> None:
        """Free completed requests; requeue unfinished ones for next decode."""
        self._num_running_batches -= 1

        for request in batch.requests:
            if request.completed:
                self.free(request.id)
            else:
                # avoid adding same request multiple times
                if all(r.id != request.id for r in self._preempted_requests):
                    self._preempted_requests.append(request)

    # -------------------------------------------------------------------------
    # Allocation helpers
    # -------------------------------------------------------------------------
    def _can_allocate_request(self, request: Request) -> bool:
        """Check if enough free blocks to admit this request."""
        if request.id not in self._allocation_map:
            # new request (prefill)
            num_required_blocks = ceil(
                request.num_prefill_tokens / self._config.block_size
            )
            return (
                self._config.num_blocks
                - self._num_allocated_blocks
                - num_required_blocks
                >= self._watermark_blocks
            )

        # ongoing decode requires at least one free block
        return self._config.num_blocks - self._num_allocated_blocks >= 1

    def _allocate_request(self, request: Request) -> None:
        """Allocate GPU blocks for the request if needed."""
        if request.id not in self._allocation_map:
            num_required_blocks = ceil(
                request.num_prefill_tokens / self._config.block_size
            )
            self.allocate(request.id, num_required_blocks)
            return

        num_tokens_reserved = self._allocation_map[request.id] * self._config.block_size
        num_tokens_required = max(0, request.num_processed_tokens - num_tokens_reserved)
        assert num_tokens_required in (0, 1), f"num_tokens_required={num_tokens_required}"

        if num_tokens_required == 1:
            self.allocate(request.id, 1)

    # -------------------------------------------------------------------------
    # Core scheduling loop
    # -------------------------------------------------------------------------
    def _get_next_batch(self) -> Batch | None:
        """Build the next executable batch (prefill or decode)."""
        requests: List[Request] = []
        num_tokens: List[int] = []
        phase_flags: List[str] = []
        seen: set[int] = set()

        # -------------------------------
        # 1. Handle queued requests (prefill or decode)
        # -------------------------------
        while self._request_queue:
            req = self._request_queue[0]
            if req.id in seen:
                # shouldn't happen; defensive
                self._request_queue.pop(0)
                continue

            next_num_tokens = self._get_request_next_num_tokens(req)
            if next_num_tokens <= 0:
                # nothing left to process
                self._request_queue.pop(0)
                continue

            if not self._can_allocate_request(req):
                break

            # simulate batch size and memory constraints
            new_num_tokens = num_tokens + [next_num_tokens]
            new_num_batch_tokens = len(new_num_tokens) * max(new_num_tokens)
            if new_num_batch_tokens > self._config.max_tokens_in_batch:
                break
            if len(self._allocation_map) >= self._config.batch_size_cap:
                break
            if len(requests) >= self._max_micro_batch_size:
                break

            # admit the request
            req = self._request_queue.pop(0)
            self._allocate_request(req)
            requests.append(req)
            num_tokens.append(next_num_tokens)
            phase_flags.append("decode" if req.is_prefill_complete else "prefill")
            seen.add(req.id)

        # -------------------------------
        # 2. If nothing queued, use preempted requests (decode phase)
        # -------------------------------
        if not requests and self._preempted_requests:
            self._preempted_requests.sort(key=lambda r: r.arrived_at)
            remaining_preempted: List[Request] = []

            while self._preempted_requests and len(requests) < self._max_micro_batch_size:
                req = self._preempted_requests.pop(0)
                if req.id in seen:
                    continue

                if not self._can_allocate_request(req):
                    # not enough memory now; keep for later
                    remaining_preempted.append(req)
                    continue

                next_num_tokens = self._get_request_next_num_tokens(req)
                if next_num_tokens <= 0:
                    continue

                self._allocate_request(req)
                requests.append(req)
                num_tokens.append(next_num_tokens)
                phase_flags.append("decode" if req.is_prefill_complete else "prefill")
                seen.add(req.id)

            # return leftover preempted requests for next cycle
            self._preempted_requests = remaining_preempted

        # -------------------------------
        # 3. Construct the batch
        # -------------------------------
        if not requests:
            return None

        batch = Batch(
            self._replica_id,
            requests,
            num_tokens,
            phase_flags=phase_flags,
        )

        return batch
