from typing import List

from vidur.events import BaseEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType

logger = init_logger(__name__)


class AutoScaleEvent(BaseEvent):
    """
    Periodic event that checks cluster load and triggers auto-scaling decisions.
    
    When average freeness crosses thresholds:
    - avgF < autoscale_low: Scale out (add replicas)
    - avgF > autoscale_high: Scale in (drain replicas)
    
    Currently only supports scale-in (draining existing replicas).
    """
    
    def __init__(self, time: float, interval: float = 1.0):
        super().__init__(time, EventType.AUTOSCALE)
        self._interval = interval
        self._is_scaling_in = False
        self._draining_replicas = []
    
    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        """
        Check autoscaling conditions and trigger draining if needed.
        """
        # Only Llumnix scheduler supports autoscaling
        if not hasattr(scheduler, 'autoscale_recommendation'):
            return [AutoScaleEvent(self.time + self._interval, self._interval)]
        
        # Get autoscale recommendation
        recommendation = scheduler.autoscale_recommendation()
        
        if recommendation == "scale_in":
            # Find least loaded replica to drain
            freeness_list = scheduler._all_freeness()
            if freeness_list:
                # Sort by freeness (ascending) and drain the one with highest freeness
                # (least loaded = safest to drain)
                freeness_list.sort(key=lambda x: x[1])
                highest_freeness_rid, highest_freeness = freeness_list[-1]
                
                # Only trigger draining if there's capacity elsewhere
                if len(freeness_list) > 1:
                    scheduler.set_draining([highest_freeness_rid], draining=True)
                    self._draining_replicas = [highest_freeness_rid]
                    logger.info(
                        f"[AutoScale] Scale-in triggered: Replica {highest_freeness_rid} "
                        f"marked for draining (avgF={sum(f for _, f in freeness_list) / len(freeness_list):.3f}, "
                        f"high={scheduler._autoscale_high})"
                    )
        
        elif recommendation == "scale_out":
            logger.warning(
                f"[AutoScale] Scale-out recommended but not yet implemented. "
                f"Cluster would need more replicas."
            )
        
        # Schedule next autoscale check
        return [AutoScaleEvent(self.time + self._interval, self._interval)]
    
    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "recommendation": "scale_in" if self._is_scaling_in else "none",
            "draining_replicas": self._draining_replicas,
        }
    
    def to_chrome_trace(self):
        """
        Emit autoscale events to Chrome trace for visibility.
        """
        if not self._is_scaling_in:
            return []
        
        return [{
            "name": f"AutoScale (Drain Replica {self._draining_replicas[0]})",
            "cat": "autoscale",
            "ph": "i",  # Instant event
            "ts": self.time * 1e6,
            "pid": -1,  # Global scope
            "tid": 0,
            "s": "g",
            "args": {
                "draining_replicas": self._draining_replicas,
            }
        }]
