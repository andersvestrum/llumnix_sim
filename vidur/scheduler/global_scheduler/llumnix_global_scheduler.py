from typing import List, Tuple, Optional
import math

from vidur.config import SimulationConfig
from vidur.entities import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler


class LlumnixGlobalScheduler(BaseGlobalScheduler):
    """
    Llumnix-style scheduler with priority-based scheduling and load-aware placement.
    
    Key features aligned with Llumnix paper (OSDI 2024):
    - Priority-based request queuing (higher priority served first)
    - Load-aware initial placement (uses least-loaded replica)
    - Periodic load rebalancing with request migration
    - Migration cost-benefit analysis
    """

    def __init__(self, config: SimulationConfig, replicas) -> None:
        super().__init__(config, replicas)
        
        # Get Llumnix-specific config
        llumnix_config = config.cluster_config.global_scheduler_config
        
        # Priority configuration
        self._num_priority_levels = llumnix_config.num_priority_levels
        self._priority_queues = {}
        
        # Load balancing configuration (from paper Section 3.2)
        self._alpha = llumnix_config.load_metric_alpha  # Weight for queue length
        self._beta = llumnix_config.load_metric_beta    # Weight for running requests
        self._gamma = llumnix_config.load_metric_gamma  # Weight for memory usage
        
        # Migration and rebalancing configuration (from paper Section 3.3)
        self._enable_migration = llumnix_config.enable_migration
        self._rebalance_interval = llumnix_config.rebalance_interval
        self._load_imbalance_threshold = llumnix_config.load_imbalance_threshold
        self._last_rebalance_time = 0.0
        
        # Migration cost parameters
        self._network_bandwidth_gbps = llumnix_config.network_bandwidth_gbps
        self._migration_overhead_ms = llumnix_config.migration_overhead_ms
        
        # Track migrations for metrics
        self._migration_count = 0
    
    def get_replica_load(self, replica_id: int) -> float:
        """
        Calculate weighted load for a replica using Llumnix's load formula.
        
        From paper Section 3.2: load considers queue length, running requests, 
        and memory usage with configurable weights.
        
        Returns:
            float: Weighted load metric for the replica
        """
        scheduler = self._replica_schedulers[replica_id]
        
        # Queue load: number of queued requests waiting to execute
        queue_load = len(scheduler._request_queue)
        
        # Running load: number of currently executing requests
        running_load = scheduler.num_pending_requests
        
        # Memory load: ratio of allocated to total KV cache blocks
        memory_load = 0.0
        if scheduler._config.num_blocks > 0:
            memory_load = scheduler._num_allocated_blocks / scheduler._config.num_blocks
        
        # Weighted combination (Llumnix load formula)
        return (self._alpha * queue_load + 
                self._beta * running_load + 
                self._gamma * memory_load)
    
    def find_least_loaded_replica(self, require_capacity: bool = True, 
                                   num_blocks: int = 1) -> Optional[int]:
        """
        Find the replica with minimum load.
        
        Args:
            require_capacity: If True, only consider replicas with available capacity
            num_blocks: Minimum blocks required if require_capacity is True
            
        Returns:
            Replica ID with minimum load, or None if no suitable replica found
        """
        min_load = float('inf')
        best_replica_id = None
        
        for replica_id in self._replica_schedulers.keys():
            scheduler = self._replica_schedulers[replica_id]
            
            # Check capacity if required
            if require_capacity and not scheduler.can_allocate(num_blocks):
                continue
            
            load = self.get_replica_load(replica_id)
            if load < min_load:
                min_load = load
                best_replica_id = replica_id
        
        return best_replica_id
    
    def calculate_load_imbalance(self) -> float:
        """
        Calculate load imbalance across replicas using standard deviation.
        
        Higher values indicate more imbalance. Used to trigger rebalancing.
        
        Returns:
            float: Standard deviation of loads across replicas
        """
        if not self._replica_schedulers:
            return 0.0
        
        loads = [self.get_replica_load(rid) for rid in self._replica_schedulers.keys()]
        
        if not loads:
            return 0.0
        
        mean_load = sum(loads) / len(loads)
        variance = sum((load - mean_load) ** 2 for load in loads) / len(loads)
        
        return math.sqrt(variance)
    
    def schedule(self) -> List[Tuple[int, Request]]:
        """
        Schedule requests using priority-based, load-aware placement.
        
        Algorithm:
        1. Organize requests by priority (higher first)
        2. For each priority level, schedule requests FIFO
        3. Place each request on the least-loaded replica with capacity
        
        This implements Llumnix's load-aware initial placement.
        """
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

        # Iterate priorities from high to low
        for pr in sorted(self._priority_queues.keys(), reverse=True):
            queue = self._priority_queues.get(pr, [])
            
            # Keep FIFO order within a priority
            while queue:
                req = queue[0]
                
                # LOAD-AWARE PLACEMENT: Find least loaded replica with capacity
                best_replica_id = self.find_least_loaded_replica(
                    require_capacity=True, 
                    num_blocks=1
                )
                
                if best_replica_id is not None:
                    scheduled.append((best_replica_id, req))
                    queue.pop(0)
                else:
                    # No replica has capacity for this request right now
                    # Stop trying for this priority to maintain priority ordering
                    break
            
            # Clean up empty queue
            if not queue:
                del self._priority_queues[pr]

        return scheduled
    
    def estimate_migration_time(self, request_id: int, source_id: int) -> float:
        """
        Estimate time to migrate a request's KV cache.
        
        Based on KV cache size and network bandwidth.
        
        Args:
            request_id: ID of request to migrate
            source_id: Source replica ID
            
        Returns:
            Estimated migration time in seconds
        """
        # Get KV cache size in blocks
        source_scheduler = self._replica_schedulers[source_id]
        kv_cache_blocks = source_scheduler._allocation_map.get(request_id, 0)
        
        if kv_cache_blocks == 0:
            return 0.0
        
        # Calculate transfer time based on block size and network bandwidth
        # Get block size from replica scheduler config
        block_size_bytes = source_scheduler._config.block_size
        total_bytes = kv_cache_blocks * block_size_bytes
        
        # Convert bandwidth from Gbps to bytes/second
        bandwidth_bytes_per_sec = self._network_bandwidth_gbps * 1e9 / 8
        transfer_time = total_bytes / bandwidth_bytes_per_sec
        
        # Add fixed overhead
        overhead = self._migration_overhead_ms / 1000.0
        
        return transfer_time + overhead
    
    def evaluate_migration_benefit(self, request_id: int, 
                                   source_id: int, 
                                   target_id: int) -> float:
        """
        Calculate net benefit of migrating a request.
        
        Benefit = load balancing improvement - migration cost
        Positive value means migration is beneficial (from paper Section 3.3).
        
        Args:
            request_id: ID of request to migrate
            source_id: Source replica ID
            target_id: Target replica ID
            
        Returns:
            Net benefit (positive = beneficial)
        """
        # Calculate current load imbalance
        source_load = self.get_replica_load(source_id)
        target_load = self.get_replica_load(target_id)
        
        # Estimate load after migration
        # Simplified: assume moving one request reduces source load and increases target load
        load_reduction = abs(source_load - target_load) / 2
        
        # Migration cost
        migration_time = self.estimate_migration_time(request_id, source_id)
        
        # Benefit formula: balance improvement minus migration overhead
        # Scale factor of 10 converts load units to time units
        return load_reduction * 10.0 - migration_time
    
    def should_rebalance(self, current_time: float) -> bool:
        """
        Check if load rebalancing should be performed.
        
        Rebalancing is triggered when:
        1. Sufficient time has passed since last rebalance
        2. Load imbalance exceeds threshold
        3. Migration is enabled
        4. Multiple replicas exist (can't rebalance with 1 replica)
        
        Args:
            current_time: Current simulation time
            
        Returns:
            True if rebalancing should occur
        """
        if not self._enable_migration:
            return False
        
        # Need at least 2 replicas to rebalance
        if self._num_replicas < 2:
            return False
        
        time_elapsed = current_time - self._last_rebalance_time
        if time_elapsed < self._rebalance_interval:
            return False
        
        imbalance = self.calculate_load_imbalance()
        return imbalance > self._load_imbalance_threshold
    
    def rebalance(self, current_time: float) -> List[Tuple[int, int, int]]:
        """
        Perform load rebalancing by migrating requests between replicas.
        
        This implements Llumnix's periodic rebalancing.
        
        Algorithm:
        1. Sort replicas by load (high to low)
        2. Try to migrate requests from overloaded to underloaded replicas
        3. Only migrate if benefit > cost
        
        Args:
            current_time: Current simulation time
            
        Returns:
            List of (request_id, source_replica, target_replica) migrations
        """
        self._last_rebalance_time = current_time
        migrations = []
        
        # Sort replicas by load
        replicas_by_load = sorted(
            self._replica_schedulers.keys(),
            key=lambda rid: self.get_replica_load(rid),
            reverse=True
        )
        
        # Try to migrate from overloaded to underloaded replicas
        num_replicas = len(replicas_by_load)
        for overloaded_id in replicas_by_load[:num_replicas // 2]:
            for underloaded_id in replicas_by_load[num_replicas // 2:]:
                # Find candidate request to migrate
                candidate = self._find_migration_candidate(overloaded_id, underloaded_id)
                
                if candidate is not None:
                    # Evaluate migration benefit
                    benefit = self.evaluate_migration_benefit(
                        candidate, overloaded_id, underloaded_id
                    )
                    
                    if benefit > 0:
                        migrations.append((candidate, overloaded_id, underloaded_id))
                        self._migration_count += 1
                        # Only migrate one request per pair to avoid over-correction
                        break
        
        return migrations
    
    def _find_migration_candidate(self, source_id: int, target_id: int) -> Optional[int]:
        """
        Find a request on source replica that can be migrated to target.
        
        Prefers requests with:
        - Smaller KV cache (lower migration cost)
        - Sufficient remaining execution time (worthwhile to migrate)
        
        Args:
            source_id: Source replica ID
            target_id: Target replica ID
            
        Returns:
            Request ID to migrate, or None if no suitable candidate
        """
        source_scheduler = self._replica_schedulers[source_id]
        target_scheduler = self._replica_schedulers[target_id]
        
        # Look for running requests on source that fit in target
        candidates = []
        for request_id, kv_cache_size in source_scheduler._allocation_map.items():
            if target_scheduler.can_allocate(kv_cache_size):
                candidates.append((request_id, kv_cache_size))
        
        if not candidates:
            return None
        
        # Prefer requests with smaller KV cache (cheaper to migrate)
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]
    
    def get_migration_stats(self) -> dict:
        """
        Get statistics about migrations performed.
        
        Returns:
            Dictionary with migration metrics
        """
        return {
            "total_migrations": self._migration_count,
            "current_load_imbalance": self.calculate_load_imbalance(),
            "replica_loads": {
                rid: self.get_replica_load(rid) 
                for rid in self._replica_schedulers.keys()
            }
        }
