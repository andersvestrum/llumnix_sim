"""
Preset Llumnix/Llumlet latency test scenarios.

Each test is a CLI string for `python3 -m vidur.main` configured to emit a
chrome trace. Overrides target the knobs that actually influence Llumnix
global scheduling (migration toggles, rebalance cadence/thresholds, priority
fan-out) and Llumlet local scheduling (KV capacity, block sizing, batch caps).
"""

from __future__ import annotations

# Base command shared by all scenarios.
BASE_COMMAND = [
    "python3 -m vidur.main",
    "--global_scheduler_config_type llumnix",
    "--llumnix_global_scheduler_config_num_priority_levels 3",
    "--llumnix_global_scheduler_config_enable_migration",
    "--llumnix_global_scheduler_config_rebalance_interval 0.05",
    "--cluster_config_num_replicas 4",
    "--replica_scheduler_config_type llumlet",
    "--synthetic_request_generator_config_num_priority_levels 3",
    "--synthetic_request_generator_config_num_requests 2000",
    "--length_generator_config_type zipf",
    "--zipf_request_length_generator_config_max_tokens 512",
    "--zipf_request_length_generator_config_theta 1.2",
    "--zipf_request_length_generator_config_min_tokens 64",
    "--zipf_request_length_generator_config_prefill_to_decode_ratio 2.0",
    "--interval_generator_config_type poisson",
    "--poisson_request_interval_generator_config_qps 100",
    "--llumlet_scheduler_config_num_blocks 128",
    "--llumlet_scheduler_config_block_size 16",
    "--llumlet_scheduler_config_batch_size_cap 8",
    "--replica_config_device a100",
    "--replica_config_model_name meta-llama/Llama-2-7b-hf",
    "--execution_time_predictor_config_type linear_regression",
    "--linear_regression_execution_time_predictor_config_prediction_max_batch_size 32",
    "--linear_regression_execution_time_predictor_config_prediction_max_tokens_per_request 8192",
    "--time_limit 60",
    "--metrics_config_enable_chrome_trace",
    "--metrics_config_write_metrics",
    "--metrics_config_store_request_metrics",
    "--log_level info",
]


def cmd_with_overrides(*overrides: str) -> str:
    """Build a runnable CLI string by appending overrides to the base command."""
    return " ".join(BASE_COMMAND + list(overrides))


# Ten targeted scenarios to generate chrome traces.
LATENCY_TESTS = [
    {
        "name": "baseline_migration_on",
        "description": "Reference: 4 replicas, migration enabled, 50ms rebalance, 3 priority levels.",
        "cmd": cmd_with_overrides(),
    },
    {
        "name": "migration_disabled",
        "description": "Global migration off to observe queuing under imbalance (no rebalancing).",
        "cmd": cmd_with_overrides("--no-llumnix_global_scheduler_config_enable_migration"),
    },
    {
        "name": "aggressive_rebalance",
        "description": "Fast 10ms rebalance interval with low 0.1 freeness gap to trigger frequent migrations.",
        "cmd": cmd_with_overrides(
            "--llumnix_global_scheduler_config_rebalance_interval 0.01",
            "--llumnix_global_scheduler_config_load_imbalance_threshold 0.1",
        ),
    },
    {
        "name": "lazy_rebalance",
        "description": "Sparse 200ms rebalance cadence and high 1.0 gap threshold to delay migrations.",
        "cmd": cmd_with_overrides(
            "--llumnix_global_scheduler_config_rebalance_interval 0.2",
            "--llumnix_global_scheduler_config_load_imbalance_threshold 1.0",
        ),
    },
    {
        "name": "tight_kv_capacity",
        "description": "Shrink KV pool to 64 blocks and batch cap 4 to stress Llumlet freeness under pressure.",
        "cmd": cmd_with_overrides(
            "--llumlet_scheduler_config_num_blocks 64",
            "--llumlet_scheduler_config_batch_size_cap 4",
        ),
    },
    {
        "name": "roomy_kv_capacity",
        "description": "Double KV pool to 256 blocks and widen batch cap to 16 for maximal packing.",
        "cmd": cmd_with_overrides(
            "--llumlet_scheduler_config_num_blocks 256",
            "--llumlet_scheduler_config_batch_size_cap 16",
        ),
    },
    {
        "name": "coarse_blocking",
        "description": "Larger 32-token blocks (fewer allocations) to see coarser freeness and migration choices.",
        "cmd": cmd_with_overrides("--llumlet_scheduler_config_block_size 32"),
    },
    {
        "name": "fine_blocking",
        "description": "Smaller 8-token blocks to allow finer packing and different freeness ordering.",
        "cmd": cmd_with_overrides("--llumlet_scheduler_config_block_size 8"),
    },
    {
        "name": "low_replica_high_qps",
        "description": "Only 2 replicas with higher 120 QPS arrival rate to stress global placement without headroom.",
        "cmd": cmd_with_overrides(
            "--cluster_config_num_replicas 2",
            "--poisson_request_interval_generator_config_qps 120",
        ),
    },
    {
        "name": "priority_stress_five_levels",
        "description": "5-level priority mix (generator + Llumnix) at 80 QPS to study cross-priority ordering.",
        "cmd": cmd_with_overrides(
            "--synthetic_request_generator_config_num_priority_levels 5",
            "--llumnix_global_scheduler_config_num_priority_levels 5",
            "--poisson_request_interval_generator_config_qps 80",
        ),
    },
]
