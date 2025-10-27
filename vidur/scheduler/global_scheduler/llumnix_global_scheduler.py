from typing import List, Tuple, Optional
import math
from vidur.config import SimulationConfig
from vidur.entities import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler
from vidur.scheduler.replica_scheduler.llumlet_replica_scheduler import LlumletLocalScheduler
from vidur.execution_time_predictor import ExecutionTimePredictorRegistry

class LlumnixGlobalScheduler(BaseGlobalScheduler):
    """
    Llumnix-style GLOBAL scheduler (refactored):
      - Does NOT inspect per-request or per-replica internals.
      - Uses llumlet-reported freeness (F = (M - ΣV)/B).
      - Dispatches to freest instance.
      - Triggers migrations by pairing source/dest; llumlets choose the request.
    """

    def __init__(self, config: SimulationConfig, replicas) -> None:

        # --- Manually initialize base fields (skip BaseGlobalScheduler.__init__) ---
        self._config = config
        self._replicas = replicas
        self._num_replicas = len(replicas)
        self._request_queue = []

        # Build predictor directly (normally done in BaseGlobalScheduler)
        execution_time_predictor = ExecutionTimePredictorRegistry.get(
            config.execution_time_predictor_config.get_type(),
            predictor_config=config.execution_time_predictor_config,
            replica_config=config.cluster_config.replica_config,
            replica_scheduler_config=config.cluster_config.replica_scheduler_config,
            metrics_config=config.metrics_config,
        )

        # Instantiate one Llumlet per replica
        self._replica_schedulers = {
            rid: LlumletLocalScheduler(
                config.cluster_config.replica_config,
                config.cluster_config.replica_scheduler_config,
                config.request_generator_config,
                replica,
                replica.num_pipeline_stages,
                execution_time_predictor,
            )
            for rid, replica in replicas.items()
        }

        # --- Llumnix-specific settings ---
        cfg = config.cluster_config.global_scheduler_config
        self._enable_migration = cfg.enable_migration
        self._rebalance_interval = cfg.rebalance_interval
        self._last_rebalance_time = 0.0
        self._num_priority_levels = cfg.num_priority_levels
        self._load_imbalance_threshold = getattr(cfg, "load_imbalance_threshold", 0.2)  # maxF - minF
        self._src_freeness_threshold = getattr(cfg, "src_freeness_threshold", 0.3)      # overloaded if F <= 0.3
        self._dst_freeness_threshold = getattr(cfg, "dst_freeness_threshold", 0.7)
        self._migration_count = 0


    # --------- Helpers that only use llumlet public API ----------
    def _all_freeness(self) -> List[Tuple[int, float]]:
        # llumlet must implement report_freeness()
        return [(rid, sch.report_freeness()) for rid, sch in self._replica_schedulers.items()]

    def _least_loaded_rid(self, require_capacity: bool = True, num_blocks: int = 1) -> Optional[int]:
        best = None
        best_F = -float("inf")
        for rid, sch in self._replica_schedulers.items():
            if require_capacity and not sch.has_capacity(num_blocks):
                continue
            F = sch.report_freeness()
            if F > best_F:
                best_F, best = F, rid
        return best

    def _imbalance(self) -> float:
        Fs = [F for _, F in self._all_freeness()]
        if not Fs:
            return 0.0
        mu = sum(Fs) / len(Fs)
        return math.sqrt(sum((f - mu) ** 2 for f in Fs) / len(Fs))

    # --------- New Request Placement (priority-aware FCFS) ----------
    def schedule(self) -> List[Tuple[int, Request]]:
        if not self._request_queue:
            return []

        # group by priority (higher numeric = higher priority)
        by_pr = {}
        for req in list(self._request_queue):
            pr = getattr(req, "priority", 0)
            by_pr.setdefault(pr, []).append(req)
        self._request_queue.clear()

        assignments: List[Tuple[int, Request]] = []
        for pr in sorted(by_pr.keys(), reverse=True):
            queue = by_pr[pr]
            while queue:
                req = queue[0]
                rid = self._least_loaded_rid(require_capacity=True, num_blocks=1)
                if rid is None:
                    break  # preserve priority; try again next tick
                self._replica_schedulers[rid].enqueue_request(req)
                assignments.append((rid, req))
                queue.pop(0)

        return assignments

    # --------- Migration Triggering (instance pairing only) ----------
    def should_rebalance(self, now: float) -> bool:
        if not self._enable_migration or self._num_replicas < 2:
            return False
        if (now - self._last_rebalance_time) < self._rebalance_interval:
            return False
        Fs = [F for _, F in self._all_freeness()]
        if not Fs:
            return False
        if (max(Fs) - min(Fs)) < self._load_imbalance_threshold:
            return False
        return True

    def rebalance(self, now: float) -> List[Tuple[int, int, int]]:
        self._last_rebalance_time = now
        migrations: List[Tuple[int, int, int]] = []

        # Rank replicas by freeness (0=full, 1=empty)
        freeness = sorted(self._all_freeness(), key=lambda x: x[1])
        if len(freeness) < 2:
            return migrations

        min_F = freeness[0][1]
        max_F = freeness[-1][1]
        imbalance_gap = max_F - min_F

        # Only rebalance if gap is meaningful
        if imbalance_gap < getattr(self, "_load_imbalance_threshold", 0.1):
            return migrations

        # Compute dynamic thresholds if not set
        src_thresh = getattr(self, "_src_freeness_threshold", min_F + 0.05)
        dst_thresh = getattr(self, "_dst_freeness_threshold", max_F - 0.05)

        # Select overloaded (low-F) and underloaded (high-F) replicas
        sources = [(rid, F) for rid, F in freeness if F <= src_thresh]
        dests   = [(rid, F) for rid, F in reversed(freeness) if F >= dst_thresh]

        for (src_rid, _), (dst_rid, _) in zip(sources, dests):
            src = self._replica_schedulers[src_rid]
            dst = self._replica_schedulers[dst_rid]
            mig = src.begin_migration_to(dst)
            if mig:
                migrations.append(mig)
                self._migration_count += 1

        return migrations


    # --------- Optional stats ----------
    def get_migration_stats(self) -> dict:
        return {
            "total_migrations": self._migration_count,
            "cluster_freeness": {rid: F for rid, F in self._all_freeness()},
            "imbalance_stddev": self._imbalance(),
        }
