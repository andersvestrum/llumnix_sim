from typing import List

from vidur.entities.base_entity import BaseEntity
from vidur.entities.request import Request
from vidur.logger import init_logger

logger = init_logger(__name__)


def check_scheduled(func):
    """Ensure batch has been scheduled before accessing."""
    def wrapper(self, *args, **kwargs):
        if not self._scheduled:
            raise ValueError("Batch has not been scheduled yet")
        return func(self, *args, **kwargs)
    return wrapper


def check_completed(func):
    """Ensure batch has been completed before accessing."""
    def wrapper(self, *args, **kwargs):
        if not self._completed:
            raise ValueError("Batch has not been completed yet")
        return func(self, *args, **kwargs)
    return wrapper


class Batch(BaseEntity):
    def __init__(
        self,
        replica_id: int,
        requests: List[Request],
        num_tokens: List[int],
        phase_flags: List[str] | None = None,  # <--- added field for phase snapshot
    ) -> None:
        self._id = Batch.generate_id()
        self._replica_id = replica_id
        self._requests = requests
        self._num_tokens = num_tokens
        self._total_num_tokens = sum(num_tokens)

        # ✅ Snapshot the per-request phase (“prefill” or “decode”) at batch creation
        if phase_flags is None:
            phase_flags = [
                "decode" if r.is_prefill_complete else "prefill" for r in requests
            ]
        self._phase_flags = phase_flags

        # ✅ Compute prefill tokens using *snapshotted phase*, not live request state
        self._num_prefill_tokens = sum(
            t if phase == "prefill" else 0
            for t, phase in zip(self._num_tokens, self._phase_flags)
        )

        self._total_num_tokens_rounded = (self._total_num_tokens + 7) // 8 * 8

        self._scheduled_at = None
        self._completed_at = None
        self._scheduled = False
        self._completed = False

    # --- Properties ---
    @property
    def replica_id(self) -> int:
        return self._replica_id

    @property
    def num_tokens(self) -> List[int]:
        return self._num_tokens

    @property
    def total_num_tokens(self) -> int:
        return self._total_num_tokens

    @property
    def num_prefill_tokens(self) -> int:
        return self._num_prefill_tokens

    @property
    def num_decode_tokens(self) -> int:
        return self.total_num_tokens - self.num_prefill_tokens

    @property
    @check_scheduled
    def scheduled_at(self) -> float:
        return self._scheduled_at

    @property
    @check_completed
    def completed_at(self) -> float:
        return self._completed_at

    @property
    def completed(self) -> bool:
        return self._completed

    @property
    def scheduled(self) -> bool:
        return self._scheduled

    @property
    def size(self) -> int:
        return len(self._requests)

    @property
    def requests(self) -> List[Request]:
        return self._requests

    @property
    def request_ids(self) -> List[int]:
        return [r.id for r in self._requests]

    @property
    def phase_flags(self) -> List[str]:
        return self._phase_flags

    @property
    def all_requests_completed(self) -> bool:
        return all(r.completed for r in self._requests)

    # --- Lifecycle ---
    def on_schedule(self, time: float) -> None:
        self._scheduled_at = time
        self._scheduled = True
        for request in self._requests:
            request.on_batch_schedule(time)

    def on_batch_end(self, time: float) -> None:
        self._completed = True
        self._completed_at = time
        for request, num_tokens in zip(self._requests, self._num_tokens):
            request.on_batch_end(time, num_tokens)

    # --- Helpers ---
    @property
    def preempted_requests(self) -> List[Request]:
        return [r for r in self._requests if r.preempted]

    @property
    def completed_requests(self) -> List[Request]:
        return [r for r in self._requests if r.completed]

    def get_request_token_breakdown(self) -> List[dict]:
        """
        Return a per-request token breakdown, using the *snapshotted phase*.
        This avoids misclassifying decode batches later.
        """
        breakdown = []
        for request, num_tokens, phase in zip(
            self._requests, self._num_tokens, self._phase_flags
        ):
            prefill_tokens = num_tokens if phase == "prefill" else 0
            decode_tokens = num_tokens if phase == "decode" else 0
            breakdown.append(
                {
                    "request_id": request.id,
                    "phase": phase,
                    "prefill_tokens": prefill_tokens,
                    "decode_tokens": decode_tokens,
                    "total_tokens": num_tokens,
                    "completed": request.completed,
                    "preempted": request.preempted,
                }
            )
        return breakdown

    def to_dict(self) -> dict:
        return {
            "id": self._id,
            "size": self.size,
            "replica_id": self._replica_id,
            "scheduled_at": self._scheduled_at,
            "completed_at": self._completed_at,
            "scheduled": self._scheduled,
            "request_ids": self.request_ids,
            "num_tokens": self._num_tokens,
            "num_prefill_tokens": self.num_prefill_tokens,
            "num_decode_tokens": self.num_decode_tokens,
            "requests": self.get_request_token_breakdown(),
        }
