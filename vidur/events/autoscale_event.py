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
        self._is_scaling_out = False
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
            # Paper-compliant scale-in: select instance with fewest running requests
            # Per Section 4.4.3: "Llumnix chooses the instance with fewest running requests for termination"
            running_counts = scheduler._all_running_request_counts()
            if running_counts:
                # Sort by running request count (ascending) and drain the one with fewest
                running_counts.sort(key=lambda x: x[1])
                fewest_requests_rid, request_count = running_counts[0]
                
                # Only trigger draining if there's capacity elsewhere
                if len(running_counts) > 1:
                    scheduler.set_draining([fewest_requests_rid], draining=True)
                    self._draining_replicas = [fewest_requests_rid]
                    self._is_scaling_in = True
                    # Get normal-priority freeness for logging (paper-compliant metric)
                    normal_freeness = scheduler._all_normal_priority_freeness()
                    avg_freeness = sum(f for _, f in normal_freeness) / len(normal_freeness)
                    logger.info(
                        f"[AutoScale] Scale-in triggered: Replica {fewest_requests_rid} "
                        f"marked for draining ({request_count} running requests, "
                        f"avgF_normal={avg_freeness:.3f}, high={scheduler._autoscale_high})"
                    )
        
        elif recommendation == "scale_out":
            # Mark for trace emission
            self._is_scaling_out = True
            # Use normal-priority freeness for logging (paper-compliant metric)
            normal_freeness = scheduler._all_normal_priority_freeness()
            avg_freeness = sum(f for _, f in normal_freeness) / len(normal_freeness)
            logger.warning(
                f"[AutoScale] Scale-out recommended at avgF_normal={avg_freeness:.3f} "
                f"(low={scheduler._autoscale_low}) but not yet implemented. "
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
        events = []
        
        if self._is_scaling_in:
            events.append({
                "name": f"AutoScale: Scale-In (Drain Replica {self._draining_replicas[0]})",
                "cat": "autoscale",
                "ph": "i",  # Instant event
                "ts": self.time * 1e6,
                "pid": -1,  # Global scope
                "tid": 0,
                "s": "g",
                "args": {
                    "action": "scale_in",
                    "draining_replicas": self._draining_replicas,
                }
            })
        
        if self._is_scaling_out:
            events.append({
                "name": "AutoScale: Scale-Out Needed (not implemented)",
                "cat": "autoscale",
                "ph": "i",  # Instant event
                "ts": self.time * 1e6,
                "pid": -1,  # Global scope
                "tid": 0,
                "s": "g",
                "args": {
                    "action": "scale_out",
                    "note": "Scale-out not yet implemented"
                }
            })
        
        return events
