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
    "--llumlet_scheduler_config_batch_size_cap 64",
    "--replica_config_device a100",
    "--replica_config_model_name meta-llama/Llama-2-7b-hf",
    "--execution_time_predictor_config_type linear_regression",
    "--linear_regression_execution_time_predictor_config_prediction_max_batch_size 32",
    "--linear_regression_execution_time_predictor_config_prediction_max_tokens_per_request 8192",
    "--linear_regression_execution_time_predictor_config_no_cache",
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


BASE_LATENCY_TESTS = [
    {
        "name": "baseline_migration_on",
        "description": "Baseline with migration enabled at nominal 100 QPS.",
        "cmd": cmd_with_overrides(),
    },

    # Test Type 1: Migration & Load Balancing Sensitivity
    {
        "name": "migration_disabled",
        "description": "Migration disabled to evaluate imbalance and preemption without rescheduling.",
        "cmd": cmd_with_overrides("--no-llumnix_global_scheduler_config_enable_migration"),
    },
    {
        "name": "rebalance_aggressive",
        "description": "Aggressive rebalance interval to trigger frequent migrations and stress the scheduler.",
        "cmd": cmd_with_overrides(
            "--llumnix_global_scheduler_config_rebalance_interval 0.01",
            "--llumnix_global_scheduler_config_load_imbalance_threshold 0.1",
        ),
    },

    # Test Type 2: KV Capacity & Fragmentation Stress
    {
        "name": "kv_capacity_tight",
        "description": "Tight KV capacity: 64 blocks and batch cap 16 to stress fragmentation and packing.",
        "cmd": cmd_with_overrides(
            "--llumlet_scheduler_config_num_blocks 64",
            "--llumlet_scheduler_config_batch_size_cap 16",
        ),
    },
]

PRIORITY_DISTRIBUTIONS = [
    #{"type": 1, "slug": "round_robin", "name": "ROUND_ROBIN"},
    {"type": 2, "slug": "uniform", "name": "UNIFORM"},
    {"type": 3, "slug": "normal", "name": "NORMAL"},
    {"type": 4, "slug": "power_law", "name": "POWER_LAW"},
    #{"type": 5, "slug": "enterprise", "name": "ENTERPRISE"},
    #{"type": 6, "slug": "burstier", "name": "BURSTIER"},
    #{"type": 7, "slug": "time_of_day", "name": "TIME_OF_DAY"},
    #{"type": 8, "slug": "traffic_class", "name": "TRAFFIC_CLASS"},
]

PRIORITY_LEVELS = [1, 2, 3, 4, 5]
REQUEST_COUNTS = [500, 2000]


def _apply_priority_distribution(cmd: str, dist_type: int) -> str:
    """Ensure the command sets the requested priority distribution, removing any existing override."""
    tokens = cmd.split()
    filtered = []
    skip = False
    for tok in tokens:
        if skip:
            skip = False
            continue
        if tok == "--synthetic_request_generator_config_priority_distribution_type":
            skip = True
            continue
        filtered.append(tok)
    filtered.append(f"--synthetic_request_generator_config_priority_distribution_type {dist_type}")
    return " ".join(filtered)


def _apply_priority_levels(cmd: str, num_levels: int) -> str:
    """Ensure the command sets the requested number of priority levels for both Llumnix and generator."""
    tokens = cmd.split()
    filtered = []
    skip = False
    for tok in tokens:
        if skip:
            skip = False
            continue
        if tok in (
            "--llumnix_global_scheduler_config_num_priority_levels",
            "--synthetic_request_generator_config_num_priority_levels",
        ):
            skip = True
            continue
        filtered.append(tok)
    filtered.append(f"--llumnix_global_scheduler_config_num_priority_levels {num_levels}")
    filtered.append(f"--synthetic_request_generator_config_num_priority_levels {num_levels}")
    return " ".join(filtered)


def _apply_num_requests(cmd: str, num_requests: int) -> str:
    """Ensure the command sets the requested number of synthetic requests."""
    tokens = cmd.split()
    filtered = []
    skip = False
    for tok in tokens:
        if skip:
            skip = False
            continue
        if tok == "--synthetic_request_generator_config_num_requests":
            skip = True
            continue
        filtered.append(tok)
    filtered.append(f"--synthetic_request_generator_config_num_requests {num_requests}")
    return " ".join(filtered)


def _expand_tests_with_distributions_levels_and_requests(base_tests):
    expanded = []
    for test in base_tests:
        for num_levels in PRIORITY_LEVELS:
            level_cmd = _apply_priority_levels(test["cmd"], num_levels)
            for num_requests in REQUEST_COUNTS:
                req_cmd = _apply_num_requests(level_cmd, num_requests)
                for dist in PRIORITY_DISTRIBUTIONS:
                    dist_suffix = f"dist{dist['type']}_{dist['slug']}"
                    level_suffix = f"lvl{num_levels}"
                    req_suffix = f"req{num_requests}"
                    expanded.append(
                        {
                            "name": f"{test['name']}_{level_suffix}_{req_suffix}_{dist_suffix}",
                            "description": (
                                f"{test['description']} "
                                f"Priority levels: {num_levels}. "
                                f"Requests: {num_requests}. "
                                f"Priority distribution: {dist['name']} (type={dist['type']})."
                            ),
                            "cmd": _apply_priority_distribution(req_cmd, dist["type"]),
                        }
                    )
    return expanded


LATENCY_TESTS = _expand_tests_with_distributions_levels_and_requests(BASE_LATENCY_TESTS)
