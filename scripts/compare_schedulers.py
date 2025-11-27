#!/usr/bin/env python3
"""Run short simulations for multiple replica schedulers and plot comparison.

This script is a convenience wrapper to run the simulator (`python -m vidur.main`)
for a small synthetic workload with different replica schedulers (vllm, orca, sarathi, llumlet)
and produce a small CSV + PNG comparing latency percentiles (P50, P90, P99).

Usage examples:
  python scripts/compare_schedulers.py --schedulers vllm orca sarathi llumlet

Notes:
 - The script disables wandb via the WANDB_MODE env var to avoid external logging.
 - By default it runs a tiny experiment (num_requests=64) so it is fast.

Fairness Considerations:
 - vLLM and llumlet both use max_tokens_in_batch constraints (set via --max_tokens_in_batch)
 - Orca and Sarathi only use batch_size_cap (no token limit per batch)
 - All schedulers respect the same batch_size_cap and num_blocks settings
 - The default max_tokens_in_batch=2048 ensures vLLM and llumlet have equal token budgets
"""
import argparse
import datetime
import os
import subprocess
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def run_simulation_for_scheduler(scheduler: str, num_replicas: int, out_dir: Path, args):
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "vidur.main",
        "--replica_config_device",
        args.device,
        "--replica_config_model_name",
        args.model,
        "--cluster_config_num_replicas",
        str(num_replicas),
        "--replica_config_tensor_parallel_size",
        str(args.tp),
        "--replica_config_num_pipeline_stages",
        str(args.pp),
        "--request_generator_config_type",
        "synthetic",
        "--synthetic_request_generator_config_num_requests",
        str(args.num_requests),
        "--length_generator_config_type",
        "fixed",
        "--fixed_request_length_generator_config_prefill_tokens",
        str(args.prefill_tokens),
        "--fixed_request_length_generator_config_decode_tokens",
        str(args.decode_tokens),
        "--interval_generator_config_type",
        "poisson",
        "--poisson_request_interval_generator_config_qps",
        str(args.qps),
        "--replica_scheduler_config_type",
        scheduler,
        "--metrics_config_output_dir",
        str(out_dir),
        "--metrics_config_wandb_project",
        "",
        "--metrics_config_wandb_group",
        "",
        "--no-metrics_config_enable_chrome_trace",
    ]

    # scheduler-specific options
    if scheduler.lower() == "sarathi":
        cmd += ["--sarathi_scheduler_config_chunk_size", str(args.sarathi_chunk_size)]

    # Set common parameters for fair comparison
    # All schedulers inherit from BaseReplicaSchedulerConfig and support these
    common_params = [
        f"--{scheduler}_scheduler_config_batch_size_cap", str(args.batch_cap),
        f"--{scheduler}_scheduler_config_block_size", str(args.block_size),
    ]
    
    # Only set num_blocks if explicitly provided (otherwise auto-computed from memory)
    if args.num_blocks is not None:
        common_params += [f"--{scheduler}_scheduler_config_num_blocks", str(args.num_blocks)]
    
    cmd += common_params

    # Scheduler-specific configurations
    if scheduler.lower() == "llumlet":
        # llumlet is a replica scheduler used with llumnix global scheduler
        cmd += [
            "--global_scheduler_config_type", "llumnix",
            "--llumlet_scheduler_config_max_tokens_in_batch", str(args.max_tokens_in_batch),
        ]
        # Add llumnix global scheduler config if migration is enabled
        if args.enable_migration:
            cmd += ["--llumnix_global_scheduler_config_enable_migration"]
    elif scheduler.lower() == "vllm":
        # vllm uses max_tokens_in_batch constraint
        cmd += [
            "--vllm_scheduler_config_max_tokens_in_batch", str(args.max_tokens_in_batch),
        ]
    elif scheduler.lower() == "orca":
        # orca only uses batch_size_cap (no additional configs needed)
        pass
    elif scheduler.lower() == "sarathi":
        # sarathi has no max_tokens_in_batch
        pass

    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, env=env)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Run failed for {scheduler}@{num_replicas}r: {e}")
        return False


def collect_metrics(out_dir: Path):
    # The simulator writes outputs into a timestamped subdirectory under the
    # provided output dir (e.g. simulator_output/<provided_ts>/<actual_ts>/request_metrics.csv).
    # Search recursively for the first request_metrics.csv and read it.
    matches = list(out_dir.rglob("request_metrics.csv"))
    if not matches:
        print(f"Warning: metrics CSV not found under {out_dir} (searched recursively)")
        return None

    csv_path = matches[0]
    df = pd.read_csv(csv_path)
    if "request_e2e_time" not in df.columns:
        print(f"Warning: request_e2e_time column not in {csv_path}")
        return None

    p50 = df["request_e2e_time"].quantile(0.5)
    p90 = df["request_e2e_time"].quantile(0.9)
    p99 = df["request_e2e_time"].quantile(0.99)
    mean = df["request_e2e_time"].mean()
    return {"p50": p50, "p90": p90, "p99": p99, "mean": mean}


def plot_results(results: dict, out_file: Path):
    if not results:
        raise ValueError("No results provided to plot_results()")

    df = pd.DataFrame(results).T
    missing = [c for c in ("p50", "p90", "p99") if c not in df.columns]
    if missing:
        raise KeyError(f"Missing percentile columns in results: {missing}")

    plot_df = df[["p50", "p90", "p99"]]

    # Try to extract replica count from index labels
    replica_count = None
    nice_labels = []
    for label in plot_df.index:
        if "@" in label and label.endswith("r"):
            parts = label.split("@")
            scheduler = parts[0]
            try:
                replicas = int(parts[1][:-1])
                replica_count = replicas
                nice_labels.append(f"{scheduler} (Replicas: {replicas})")
            except Exception:
                nice_labels.append(label)
        else:
            nice_labels.append(label)

    plot_df.index = nice_labels
    figsize = (max(10, len(plot_df) * 0.8), 6)
    ax = plot_df.plot(kind="bar", figsize=figsize, colormap="plasma")
    ax.set_ylabel("Request E2E latency (s)")
    # If replica_count is set, add it to the title
    if replica_count is not None:
        ax.set_title(f"Scheduler Comparison — Latency Percentiles\nReplicas: {replica_count}")
    else:
        ax.set_title("Scheduler Comparison — Latency Percentiles")
    ax.set_xlabel("Configuration (Scheduler and Replicas)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    y_min = plot_df.min().min() * 0.95
    y_max = plot_df.max().max() * 1.05
    ax.set_ylim(y_min, y_max)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=150)
    plt.close()


def plot_results_by_replica(results: dict, base_dir: Path, ts: str):
    if not results:
        print("No results to plot.")
        return

    # Parse labels like "scheduler@Nr" to group by N
    grouped: dict[int, dict] = {}
    for label, metrics in results.items():
        try:
            # Expect label format: name@{N}r
            parts = label.split('@')
            if len(parts) != 2 or not parts[1].endswith('r'):
                replica_count = None
            else:
                replica_count = int(parts[1][:-1])
        except Exception:
            replica_count = None

        if replica_count is None:
            # Put into a special group
            replica_count = -1

        grouped.setdefault(replica_count, {})[label] = metrics

    # Create one plot per replica count
    for rep_count, group in grouped.items():
        if not group:
            continue
        suffix = f"{rep_count}r" if rep_count >= 0 else "mixed"
        out_png = base_dir / f"{ts}_scheduler_comparison_{suffix}.png"
        out_csv = base_dir / f"{ts}_scheduler_comparison_{suffix}.csv"

        df = pd.DataFrame(group).T
        df.to_csv(out_csv)
        try:
            plot_results(group, out_png)
            print(f"Saved plot for {suffix} to {out_png}")
        except Exception as e:
            print(f"Could not plot for group {suffix}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedulers", nargs="+", default=["vllm", "orca", "sarathi", "llumlet"])
    parser.add_argument("--num_requests", type=int, default=50)
    parser.add_argument("--qps", type=float, default=1.0)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--device", type=str, default="a100")
    parser.add_argument("--replicas", nargs="+", type=int, default=[1, 2, 3], help="Number of replicas to test (can specify multiple values)")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--prefill_tokens", type=int, default=512)
    parser.add_argument("--decode_tokens", type=int, default=128)
    parser.add_argument("--sarathi_chunk_size", type=int, default=512)
    parser.add_argument("--batch_cap", type=int, default=64, help="Max batch size for all schedulers")
    parser.add_argument("--max_tokens_in_batch", type=int, default=2048, help="Max tokens in batch for schedulers that use it (vllm, llumlet)")
    parser.add_argument("--block_size", type=int, default=16, help="KV cache block size (tokens per block) for all schedulers")
    parser.add_argument("--num_blocks", type=int, default=None, help="Number of KV cache blocks (None = auto-compute from memory)")
    parser.add_argument("--enable_migration", action="store_true", help="Enable live migration for llumnix (only applies when using llumlet scheduler)")
    parser.add_argument("--results_dir", type=str, default="results/scheduler_cmp")
    parser.add_argument("--skip_run", action="store_true", help="Skip running sims; only plot from existing output dirs")
    parser.add_argument("--existing_output_dirs", nargs="*", help="If skipping run, pass a list of simulator output dirs to include (overrides default naming)")

    args = parser.parse_args()

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for scheduler in args.schedulers:
        for num_replicas in args.replicas:
            label = f"{scheduler}@{num_replicas}r"
            if args.skip_run and args.existing_output_dirs:
                # try to find matching output dir from provided list
                out_dir = Path(args.existing_output_dirs.pop(0))
            else:
                out_dir = Path("simulator_output") / f"{ts}_{scheduler}_{num_replicas}r"

                if not args.skip_run:
                    ok = run_simulation_for_scheduler(scheduler, num_replicas, out_dir, args)
                    if not ok:
                        # Skip metrics collection for failed runs
                        print(f"Skipping metrics collection for failed run {label}.")
                        continue

            metrics = collect_metrics(out_dir)
            if metrics is None:
                print(f"No metrics for {label} (looked in {out_dir})")
                continue

            results[label] = metrics

    # Write a combined CSV for reference
    df_all = pd.DataFrame(results).T
    csv_all = results_dir / f"{ts}_scheduler_comparison_all.csv"
    df_all.to_csv(csv_all)
    print(f"Saved combined CSV to {csv_all}")

    # Produce separate plots per replica count
    plot_results_by_replica(results, results_dir, ts)


if __name__ == "__main__":
    main()
