"""
Preset Llumnix/Llumlet latency test scenarios.

Each test is a CLI string for `python3 -m vidur.main` configured to emit a
chrome trace. Overrides target the knobs that actually influence Llumnix
global scheduling (migration toggles, rebalance cadence/thresholds, priority
fan-out) and Llumlet local scheduling (KV capacity, block sizing, batch caps).
"""

from __future__ import annotations

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
    "--linear_regression_execution_time_predictor_config_no_cache",
    # Keep caching fully disabled for latency tests.
    "--metrics_config_cache_dir /tmp/vidur_latency_no_cache",
    "--time_limit 60",
    "--metrics_config_enable_chrome_trace",
    "--metrics_config_write_metrics",
    "--metrics_config_store_request_metrics",
    "--log_level info",
]


def cmd_with_overrides(*overrides: str) -> str:
    """Build a runnable CLI string by appending overrides to the base command."""
    return " ".join(BASE_COMMAND + list(overrides))


LATENCY_TESTS = [
    {
        "name": "baseline_migration_on_easy",
        "description": "Baseline with migration enabled, slightly lower load (80 QPS) to keep queues light.",
        "cmd": cmd_with_overrides("--poisson_request_interval_generator_config_qps 80"),
    },
    {
        "name": "baseline_migration_on_medium",
        "description": "Baseline reference: migration enabled, 50ms rebalance, nominal 100 QPS.",
        "cmd": cmd_with_overrides(),
    },
    {
        "name": "baseline_migration_on_hard",
        "description": "Baseline but heavier load (140 QPS, 3000 reqs) to stress steady-state behavior.",
        "cmd": cmd_with_overrides(
            "--poisson_request_interval_generator_config_qps 140",
            "--synthetic_request_generator_config_num_requests 3000",
        ),
    },
    {
        "name": "migration_disabled_easy",
        "description": "Migration disabled with lower 70 QPS to observe queue buildup gently.",
        "cmd": cmd_with_overrides(
            "--no-llumnix_global_scheduler_config_enable_migration",
            "--poisson_request_interval_generator_config_qps 70",
        ),
    },
    {
        "name": "migration_disabled_medium",
        "description": "Migration disabled at nominal 100 QPS to see imbalance without rebalancing.",
        "cmd": cmd_with_overrides("--no-llumnix_global_scheduler_config_enable_migration"),
    },
    {
        "name": "aggressive_rebalance_easy",
        "description": "Aggressive rebalance with 20ms interval and 0.2 gap; moderate pressure.",
        "cmd": cmd_with_overrides(
            "--llumnix_global_scheduler_config_rebalance_interval 0.02",
            "--llumnix_global_scheduler_config_load_imbalance_threshold 0.2",
        ),
    },
    {
        "name": "aggressive_rebalance_medium",
        "description": "Aggressive rebalance: 10ms interval, 0.1 gap to trigger frequent migrations.",
        "cmd": cmd_with_overrides(
            "--llumnix_global_scheduler_config_rebalance_interval 0.01",
            "--llumnix_global_scheduler_config_load_imbalance_threshold 0.1",
        ),
    },
    {
        "name": "lazy_rebalance_easy",
        "description": "Lazy rebalance at 150ms and 0.8 gap; mild delay before migrations.",
        "cmd": cmd_with_overrides(
            "--llumnix_global_scheduler_config_rebalance_interval 0.15",
            "--llumnix_global_scheduler_config_load_imbalance_threshold 0.8",
        ),
    },
    {
        "name": "lazy_rebalance_medium",
        "description": "Lazy rebalance: 200ms cadence and 1.0 gap threshold to delay migrations.",
        "cmd": cmd_with_overrides(
            "--llumnix_global_scheduler_config_rebalance_interval 0.2",
            "--llumnix_global_scheduler_config_load_imbalance_threshold 1.0",
        ),
    },
    {
        "name": "lazy_rebalance_hard",
        "description": "Very lazy rebalance: 300ms cadence, 1.2 gap; raises risk of long-lived skew.",
        "cmd": cmd_with_overrides(
            "--llumnix_global_scheduler_config_rebalance_interval 0.3",
            "--llumnix_global_scheduler_config_load_imbalance_threshold 1.2",
        ),
    },
    {
        "name": "tight_kv_capacity_easy",
        "description": "Tighter KV: 96 blocks, batch cap 6 to lightly constrain freeness.",
        "cmd": cmd_with_overrides(
            "--llumlet_scheduler_config_num_blocks 96",
            "--llumlet_scheduler_config_batch_size_cap 6",
        ),
    },
    {
        "name": "tight_kv_capacity_medium",
        "description": "Tight KV: 64 blocks and batch cap 4 to stress freeness under pressure.",
        "cmd": cmd_with_overrides(
            "--llumlet_scheduler_config_num_blocks 64",
            "--llumlet_scheduler_config_batch_size_cap 4",
        ),
    },
    {
        "name": "tight_kv_capacity_hard",
        "description": "Severely tight KV: 48 blocks, batch cap 3, and higher 120 QPS to push spills.",
        "cmd": cmd_with_overrides(
            "--llumlet_scheduler_config_num_blocks 48",
            "--llumlet_scheduler_config_batch_size_cap 3",
            "--poisson_request_interval_generator_config_qps 120",
        ),
    },
    {
        "name": "roomy_kv_capacity_easy",
        "description": "Roomier KV: 192 blocks and batch cap 12 for better packing.",
        "cmd": cmd_with_overrides(
            "--llumlet_scheduler_config_num_blocks 192",
            "--llumlet_scheduler_config_batch_size_cap 12",
        ),
    },
    {
        "name": "roomy_kv_capacity_medium",
        "description": "Roomy KV: 256 blocks and batch cap 16 for maximal packing.",
        "cmd": cmd_with_overrides(
            "--llumlet_scheduler_config_num_blocks 256",
            "--llumlet_scheduler_config_batch_size_cap 16",
        ),
    },
    {
        "name": "roomy_kv_capacity_hard",
        "description": "Very roomy KV: 320 blocks, batch cap 20, and 5 priority levels to study mixing.",
        "cmd": cmd_with_overrides(
            "--llumlet_scheduler_config_num_blocks 320",
            "--llumlet_scheduler_config_batch_size_cap 20",
        ),
    },
    {
        "name": "priority_stress_five_levels_easy",
        "description": "5-level priority mix at 60 QPS to validate ordering under lighter load.",
        "cmd": cmd_with_overrides(
            "--poisson_request_interval_generator_config_qps 60",
        ),
    },
    {
        "name": "priority_stress_five_levels_medium",
        "description": "5-level priority mix (generator + Llumnix) at 80 QPS to study cross-priority ordering.",
        "cmd": cmd_with_overrides(
            "--poisson_request_interval_generator_config_qps 80",
        ),
    },
    {
        "name": "priority_stress_five_levels_hard",
        "description": "5-level priority mix at 110 QPS with 3000 requests to stress preemption and ordering.",
        "cmd": cmd_with_overrides(
            "--poisson_request_interval_generator_config_qps 110",
            "--synthetic_request_generator_config_num_requests 3000",
        ),
    },
    {
        "name": "load_metric_weights_easy",
        "description": "Load metric weights favor queue length lightly (alpha=0.8, beta=1.0, gamma=0.5).",
        "cmd": cmd_with_overrides(
            "--llumnix_global_scheduler_config_load_metric_alpha 0.8",
            "--llumnix_global_scheduler_config_load_metric_beta 1.0",
            "--llumnix_global_scheduler_config_load_metric_gamma 0.5",
        ),
    },
    {
        "name": "load_metric_weights_medium",
        "description": "Balanced load metric weights with slight queue emphasis (alpha=1.2, beta=1.0, gamma=1.0).",
        "cmd": cmd_with_overrides(
            "--llumnix_global_scheduler_config_load_metric_alpha 1.2",
            "--llumnix_global_scheduler_config_load_metric_beta 1.0",
            "--llumnix_global_scheduler_config_load_metric_gamma 1.0",
        ),
    },
    {
        "name": "load_metric_weights_hard",
        "description": "Heavier weighting on all dimensions (alpha=1.5, beta=1.5, gamma=1.5) to make imbalance triggers sensitive.",
        "cmd": cmd_with_overrides(
            "--llumnix_global_scheduler_config_load_metric_alpha 1.5",
            "--llumnix_global_scheduler_config_load_metric_beta 1.5",
            "--llumnix_global_scheduler_config_load_metric_gamma 1.5",
        ),
    },
    {
        "name": "migration_costs_easy",
        "description": "Cheap migrations: 200 Gbps bandwidth and 2 ms overhead.",
        "cmd": cmd_with_overrides(
            "--llumnix_global_scheduler_config_network_bandwidth_gbps 200",
            "--llumnix_global_scheduler_config_migration_overhead_ms 2.0",
        ),
    },
    {
        "name": "migration_costs_medium",
        "description": "Default-ish migration costs: 100 Gbps bandwidth and 5 ms overhead.",
        "cmd": cmd_with_overrides(
            "--llumnix_global_scheduler_config_network_bandwidth_gbps 100",
            "--llumnix_global_scheduler_config_migration_overhead_ms 5.0",
        ),
    },
    {
        "name": "migration_costs_hard",
        "description": "Expensive migrations: 40 Gbps bandwidth and 10 ms overhead to discourage moves.",
        "cmd": cmd_with_overrides(
            "--llumnix_global_scheduler_config_network_bandwidth_gbps 40",
            "--llumnix_global_scheduler_config_migration_overhead_ms 10.0",
        ),
    },
    {
        "name": "prefill_decode_ratio_easy",
        "description": "Lower prefill-to-decode ratio (1.5) for shorter prefill bursts.",
        "cmd": cmd_with_overrides("--zipf_request_length_generator_config_prefill_to_decode_ratio 1.5"),
    },
    {
        "name": "prefill_decode_ratio_medium",
        "description": "Baseline prefill-to-decode ratio (2.0).",
        "cmd": cmd_with_overrides("--zipf_request_length_generator_config_prefill_to_decode_ratio 2.0"),
    },
    {
        "name": "prefill_decode_ratio_hard",
        "description": "Higher prefill-to-decode ratio (3.0) to stress KV allocations up front.",
        "cmd": cmd_with_overrides("--zipf_request_length_generator_config_prefill_to_decode_ratio 3.0"),
    },
    {
        "name": "token_length_spread_easy",
        "description": "Narrower token length spread: max 384, theta 1.1, min 64.",
        "cmd": cmd_with_overrides(
            "--zipf_request_length_generator_config_max_tokens 384",
            "--zipf_request_length_generator_config_theta 1.1",
            "--zipf_request_length_generator_config_min_tokens 64",
        ),
    },
    {
        "name": "token_length_spread_medium",
        "description": "Moderate spread: max 512, theta 1.3, min 64.",
        "cmd": cmd_with_overrides(
            "--zipf_request_length_generator_config_max_tokens 512",
            "--zipf_request_length_generator_config_theta 1.3",
            "--zipf_request_length_generator_config_min_tokens 64",
        ),
    },
    {
        "name": "token_length_spread_hard",
        "description": "Wide spread: max 768, theta 1.4, min 32 to introduce heavy tails.",
        "cmd": cmd_with_overrides(
            "--zipf_request_length_generator_config_max_tokens 768",
            "--zipf_request_length_generator_config_theta 1.4",
            "--zipf_request_length_generator_config_min_tokens 32",
        ),
    },
    {
        "name": "batch_size_cap_only_easy",
        "description": "Batch cap nudged to 10 with default KV capacity.",
        "cmd": cmd_with_overrides("--llumlet_scheduler_config_batch_size_cap 10"),
    },
    {
        "name": "batch_size_cap_only_medium",
        "description": "Batch cap widened to 12 for higher packing.",
        "cmd": cmd_with_overrides("--llumlet_scheduler_config_batch_size_cap 12"),
    },
    {
        "name": "batch_size_cap_only_hard",
        "description": "Batch cap tightened to 6 to limit packing despite default KV.",
        "cmd": cmd_with_overrides("--llumlet_scheduler_config_batch_size_cap 6"),
    },
    {
        "name": "request_volume_easy",
        "description": "Smaller volume: 1200 requests at 90 QPS.",
        "cmd": cmd_with_overrides(
            "--synthetic_request_generator_config_num_requests 1200",
            "--poisson_request_interval_generator_config_qps 90",
        ),
    },
    {
        "name": "request_volume_medium",
        "description": "Baseline volume: 2000 requests at 100 QPS.",
        "cmd": cmd_with_overrides(
            "--synthetic_request_generator_config_num_requests 2000",
            "--poisson_request_interval_generator_config_qps 100",
        ),
    },
    {
        "name": "request_volume_hard",
        "description": "Heavy volume: 3500 requests at 130 QPS.",
        "cmd": cmd_with_overrides(
            "--synthetic_request_generator_config_num_requests 3500",
            "--poisson_request_interval_generator_config_qps 130",
        ),
    },
    {
        "name": "replica_scale_out_easy",
        "description": "Scale-out to 5 replicas at 100 QPS.",
        "cmd": cmd_with_overrides(
            "--cluster_config_num_replicas 5",
            "--poisson_request_interval_generator_config_qps 100",
        ),
    },
    {
        "name": "replica_scale_out_medium",
        "description": "Scale-out to 6 replicas at 110 QPS.",
        "cmd": cmd_with_overrides(
            "--cluster_config_num_replicas 6",
            "--poisson_request_interval_generator_config_qps 110",
        ),
    },
    {
        "name": "replica_scale_out_hard",
        "description": "Scale-out to 8 replicas at 130 QPS to test distribution fairness.",
        "cmd": cmd_with_overrides(
            "--cluster_config_num_replicas 8",
            "--poisson_request_interval_generator_config_qps 130",
        ),
    },
    {
        "name": "predictor_limits_easy",
        "description": "Looser predictor caps: max batch size 48, max tokens/request 12288.",
        "cmd": cmd_with_overrides(
            "--linear_regression_execution_time_predictor_config_prediction_max_batch_size 48",
            "--linear_regression_execution_time_predictor_config_prediction_max_tokens_per_request 12288",
        ),
    },
    {
        "name": "predictor_limits_medium",
        "description": "Default-like predictor caps: batch size 32, tokens/request 8192.",
        "cmd": cmd_with_overrides(
            "--linear_regression_execution_time_predictor_config_prediction_max_batch_size 32",
            "--linear_regression_execution_time_predictor_config_prediction_max_tokens_per_request 8192",
        ),
    },
    {
        "name": "predictor_limits_hard",
        "description": "Tighter predictor caps: batch size 24, tokens/request 4096 to constrain predictions.",
        "cmd": cmd_with_overrides(
            "--linear_regression_execution_time_predictor_config_prediction_max_batch_size 24",
            "--linear_regression_execution_time_predictor_config_prediction_max_tokens_per_request 4096",
        ),
    },
]
