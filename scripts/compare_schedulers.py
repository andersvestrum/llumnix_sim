#!/usr/bin/env python3
"""Run short simulations for multiple replica schedulers and plot comparison.

This script is a convenience wrapper to run the simulator (`python -m vidur.main`)
for a small synthetic workload with different replica schedulers (vllm, orca, sarathi)
and produce a small CSV + PNG comparing latency percentiles (P50, P90, P99).

Usage examples:
  python scripts/compare_schedulers.py --schedulers vllm orca sarathi

Notes:
 - The script disables wandb via the WANDB_MODE env var to avoid external logging.
 - By default it runs a tiny experiment (num_requests=64) so it is fast.
"""
import argparse
import datetime
import os
import subprocess
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def run_simulation_for_scheduler(scheduler: str, out_dir: Path, args):
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
        str(args.replicas),
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
    ]

    # scheduler-specific options
    if scheduler.lower() == "sarathi":
        cmd += ["--sarathi_scheduler_config_chunk_size", str(args.sarathi_chunk_size)]

    # batch-cap flags for common schedulers
    cap_flag = f"--{scheduler}_scheduler_config_batch_size_cap"
    cmd += [cap_flag, str(args.batch_cap)]

    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


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
    # reorder columns for nicer layout
    missing = [c for c in ("p50", "p90", "p99") if c not in df.columns]
    if missing:
        raise KeyError(f"Missing percentile columns in results: {missing}")

    plot_df = df[["p50", "p90", "p99"]]

    ax = plot_df.plot(kind="bar", figsize=(8, 5), colormap="plasma")
    ax.set_ylabel("Request E2E latency (s)")
    ax.set_title("Scheduler comparison — latency percentiles")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedulers", nargs="+", default=["vllm", "orca", "sarathi"])
    parser.add_argument("--num_requests", type=int, default=64)
    parser.add_argument("--qps", type=float, default=1.0)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--device", type=str, default="a100")
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--prefill_tokens", type=int, default=2048)
    parser.add_argument("--decode_tokens", type=int, default=512)
    parser.add_argument("--sarathi_chunk_size", type=int, default=512)
    parser.add_argument("--batch_cap", type=int, default=64)
    parser.add_argument("--results_dir", type=str, default="results/scheduler_cmp")
    parser.add_argument("--skip_run", action="store_true", help="Skip running sims; only plot from existing output dirs")
    parser.add_argument("--existing_output_dirs", nargs="*", help="If skipping run, pass a list of simulator output dirs to include (overrides default naming)")

    args = parser.parse_args()

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for scheduler in args.schedulers:
        label = scheduler
        if args.skip_run and args.existing_output_dirs:
            # try to find matching output dir from provided list
            out_dir = Path(args.existing_output_dirs.pop(0))
        else:
            out_dir = Path("simulator_output") / f"{ts}_{scheduler}"

            if not args.skip_run:
                run_simulation_for_scheduler(scheduler, out_dir, args)

        metrics = collect_metrics(out_dir)
        if metrics is None:
            print(f"No metrics for {scheduler} (looked in {out_dir})")
            continue

        results[label] = metrics

    # write numeric CSV
    df = pd.DataFrame(results).T
    csv_out = results_dir / f"{ts}_scheduler_comparison.csv"
    df.to_csv(csv_out)
    print(f"Saved comparison CSV to {csv_out}")

    # plot
    png_out = results_dir / f"{ts}_scheduler_comparison.png"
    try:
        plot_results(results, png_out)
        print(f"Saved comparison plot to {png_out}")
    except Exception as e:
        print(f"Could not produce plot: {e}")
        print(f"If you have CSV at {csv_out} you can plot manually.")
    print(f"Saved comparison plot to {png_out}")


if __name__ == "__main__":
    main()
