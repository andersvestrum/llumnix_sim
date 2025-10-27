# vidur/scheduler/global_scheduler/llumnix_global_scheduler.py

from typing import Dict, List, Tuple
from vidur.config import SimulationConfig
from vidur.entities import Replica, Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler
from vidur.types import GlobalSchedulerType
from vidur.logger import init_logger

logger = init_logger(__name__)


class LlumnixGlobalScheduler(BaseGlobalScheduler):
    def __init__(self, config: SimulationConfig, replicas: Dict[int, Replica]):
        super().__init__(config, replicas)
        params = config.cluster_config.global_scheduler_config

        if not hasattr(self, "_inflight_requests"):
            self._inflight_requests: Dict[int, Request] = {}

        self._f_low = float(getattr(params, "freeness_low", 10.0))
        self._f_high = float(getattr(params, "freeness_high", 60.0))

        logger.info(
            f"Llumnix scheduler initialized (migration disabled): "
            f"f_low={self._f_low}, f_high={self._f_high}"
        )

    def schedule(self) -> List[Tuple[int, Request]]:
        placements: List[Tuple[int, Request]] = []

        # prune completed
        if self._inflight_requests:
            self._inflight_requests = {
                rid: r for rid, r in self._inflight_requests.items() if not r.completed
            }

        def _is_active(req_id: int) -> bool:
            for rs in self._replica_schedulers.values():
                if any(getattr(r, "id", None) == req_id for r in getattr(rs, "_request_queue", [])):
                    return True
                if req_id in getattr(rs, "_allocation_map", {}):
                    return True
            return False

        # prefill: new arrivals
        self.sort_requests()
        for req in list(self._request_queue):
            if _is_active(req.id):
                self._request_queue.remove(req)
                self._inflight_requests[req.id] = req
                continue
            rid = self._pick_replica_for_dispatch(req)
            self.get_replica_scheduler(rid).add_request(req)
            self._inflight_requests[req.id] = req
            placements.append((rid, req))
            self._request_queue.remove(req)

        # decode: re-enqueue when idle
        for req in list(self._inflight_requests.values()):
            if not req.is_prefill_complete or req.completed:
                continue
            if _is_active(req.id):
                continue
            rid = self._pick_replica_for_dispatch(req)
            self.get_replica_scheduler(rid).add_request(req)
            placements.append((rid, req))

        return placements

    def _freeness(self, rid: int) -> float:
        rs = self._replica_schedulers[rid]
        total_blocks = getattr(rs._config, "num_blocks", None)
        if not total_blocks:
            return 1.0
        used_blocks = getattr(rs, "num_allocated_blocks", 0)
        return max(0.0, min(1.0, (total_blocks - used_blocks) / total_blocks))

    def _pick_replica_for_dispatch(self, req: Request) -> int:
        freeness_scores = {
            rid: self._freeness(rid) for rid in self._replica_schedulers.keys()
        }
        best = max(freeness_scores.items(), key=lambda kv: kv[1])[0]
        logger.debug(f"[Llumnix] Dispatching req {req.id} → replica {best}")
        return best

    @staticmethod
    def get_type():
        return GlobalSchedulerType.LLUMNIX
