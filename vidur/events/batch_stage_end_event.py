import json
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
        batch = self._batch
        stage = self._batch_stage
        replica_id = self._replica_id
        stage_id = self._stage_id
        is_last_stage = self._is_last_stage

        # --- compute per-request tokens from batch.num_tokens (actual values) ---
        per_req_entries = []
        total_prefill_tokens = 0
        total_decode_tokens = 0

        # zip ensures we use the per-request num_tokens from that batch
        for req, tokens in zip(batch.requests, batch.num_tokens):
            if req.is_prefill_complete:
                decode_tokens = tokens
                prefill_tokens = 0
            else:
                prefill_tokens = tokens
                decode_tokens = 0

            per_req_entries.append({
                "request_id": req.id,
                "prefill_tokens": prefill_tokens,
                "decode_tokens": decode_tokens,
                "total_tokens": tokens,
                "completed": req.completed,
                "preempted": req.preempted,
            })

            total_prefill_tokens += prefill_tokens
            total_decode_tokens += decode_tokens

        trace_entry = {
            "name": f"Batch {batch.id} Stage {stage_id}",
            "ph": "X",
            "ts": stage.scheduled_at * 1e6,
            "dur": stage.execution_time * 1e6,
            "pid": replica_id,
            "tid": stage_id,
            "args": {
                "batch_id": batch.id,
                "batch_stage_id": stage.id,
                "replica_id": replica_id,
                "stage_id": stage_id,
                "is_last_stage": is_last_stage,
                "size": batch.size,
                "num_prefill_tokens": total_prefill_tokens,
                "num_decode_tokens": total_decode_tokens,
                "requests": per_req_entries,
            },
        }

        return [trace_entry]




