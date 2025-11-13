from typing import List

from vidur.entities.batch import Batch
from vidur.entities.batch_stage import BatchStage
from vidur.events import BaseEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType

logger = init_logger(__name__)


class BatchStageEndEvent(BaseEvent):
    def __init__(
        self,
        time: float,
        replica_id: int,
        stage_id: int,
        is_last_stage: bool,
        batch: Batch,
        batch_stage: BatchStage,
    ):
        super().__init__(time, EventType.BATCH_STAGE_END)

        self._replica_id = replica_id
        self._stage_id = stage_id
        self._is_last_stage = is_last_stage

        self._batch = batch
        self._batch_stage = batch_stage

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from vidur.events.batch_end_event import BatchEndEvent
        from vidur.events.batch_stage_arrival_event import BatchStageArrivalEvent
        from vidur.events.replica_stage_schedule_event import ReplicaStageScheduleEvent

        scheduler.get_replica_stage_scheduler(
            self._replica_id, self._stage_id
        ).on_stage_end()

        self._batch_stage.on_stage_end(self.time)
        metrics_store.on_batch_stage_end(
            self._batch_stage,
            self.time,
            self._replica_id,
            self._stage_id,
        )

        next_events = [
            ReplicaStageScheduleEvent(
                self.time,
                self._replica_id,
                self._stage_id,
            ),
        ]

        if self._is_last_stage:
            return next_events + [
                BatchEndEvent(self.time, self._replica_id, self._batch)
            ]

        return next_events + [
            BatchStageArrivalEvent(
                self.time,
                self._replica_id,
                self._stage_id + 1,
                self._batch,
            )
        ]

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "replica_id": self._replica_id,
            "stage_id": self._stage_id,
            "batch_id": self._batch.id,
            "batch_stage_id": self._batch_stage.id,
            "is_last_stage": self._is_last_stage,
        }

    def to_chrome_trace(self) -> list[dict]:
        # collect per-request priority if available
        request_priorities = [getattr(r, "priority", None) for r in self._batch.requests]
        # collect request IDs
        request_ids = [r.id for r in self._batch.requests]

        # choose a representative batch priority when all requests share the same priority
        batch_priority = None
        if request_priorities:
            unique_priorities = set(request_priorities)
            if len(unique_priorities) == 1:
                batch_priority = request_priorities[0]

        return [{
            "name": f"Batch {self._batch.id} Stage_id {self._stage_id} | Req_ids: {','.join(map(str, request_ids))}",
            "ph": "X",
            "ts": self._batch_stage.scheduled_at * 1e6,  # start time
            "dur": self._batch_stage.execution_time * 1e6,  # duration
            "pid": self._replica_id,
            "tid": self._stage_id,
            "args": {
                "batch_id": self._batch.id,
                "batch_stage_id": self._batch_stage.id,
                "replica_id": self._replica_id,
                "stage_id": self._stage_id,
                "is_last_stage": self._is_last_stage,
                "size": self._batch.size,
                "num_prefill_tokens": self._batch.num_prefill_tokens,
                "num_decode_tokens": self._batch.num_decode_tokens,
                "batch_priority": batch_priority,
                "request_priorities": request_priorities,
                "request_ids": request_ids,
            },
        }]



