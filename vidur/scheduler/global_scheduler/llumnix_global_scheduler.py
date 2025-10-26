from typing import List, Tuple

from vidur.config import SimulationConfig
from vidur.entities import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler


class LlumnixGlobalScheduler(BaseGlobalScheduler):
    """
    Llumnix-style priority scheduling. It inspects each Request for an optional `priority` attribute
    (higher values mean higher priority). If no attribute is present, priority
    0 is assumed.
    """

    def __init__(self, config: SimulationConfig, replicas) -> None:
        super().__init__(config, replicas)
        # Number of priority levels to expect, default 
        self._num_priority_levels = getattr(config, "llumnix_num_priority_levels", 2)
        # Internal priority queues: mapping priority -> list[Request]
        # Higher numeric value -> higher priority
        # We initialize an empty dictionary and create queues when requests arrive
        self._priority_queues = {}
        # Simple round-robin pointer across replicas to avoid always picking replica 0 first for many requests
        self._next_replica_idx = 0

    def schedule(self) -> List[Tuple[int, Request]]:
        # If we have no queued requests, nothing to do
        if not self._request_queue:
            return []

        # Bucket requests by priority (higher numeric -> higher priority)
        for req in list(self._request_queue):
            pr = getattr(req, "priority", 0)
            self._priority_queues.setdefault(pr, []).append(req)
            # remove from base queue, we will manage via priority queues
            self._request_queue.remove(req)

        scheduled = []

        # Iterate priorities from high to low and try to schedule greedily
        for pr in sorted(self._priority_queues.keys(), reverse=True):
            queue = self._priority_queues.get(pr, [])
            # keep FIFO order inside a priority
            while queue and len(scheduled) < self._num_replicas:
                req = queue[0]

                # Try replicas starting from a rotating index to distribute load
                tried = 0
                scheduled_this_req = False
                while tried < self._num_replicas:
                    replica_ids = list(self._replica_schedulers.keys())
                    replica_id = replica_ids[self._next_replica_idx % len(replica_ids)]
                    self._next_replica_idx += 1
                    tried += 1

                    replica_scheduler = self._replica_schedulers[replica_id]
                    if replica_scheduler.can_allocate(1):
                        scheduled.append((replica_id, req))
                        queue.pop(0)
                        scheduled_this_req = True
                        break

                if not scheduled_this_req:
                    # could not find replica for this request right now; stop
                    # trying lower-priority requests because higher priority
                    # should be preferred
                    break

            # if queue is empty remove it from dict to keep keys small
            if not queue:
                del self._priority_queues[pr]

            # stop if we've scheduled as many requests as replicas
            if len(scheduled) >= self._num_replicas:
                break

        return scheduled
