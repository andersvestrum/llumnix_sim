#!/usr/bin/env python3
"""Compare Llumnix with llumlet vs round-robin scheduling across priority levels.

This script runs simulations comparing:
- Llumnix with llumlet (priority-aware scheduling)
- Llumnix with round-robin (baseline scheduling)

It produces multi-panel plots showing various metrics (P99, mean, preemption loss, etc.)
across different priority levels, similar to paper figures.

Usage:
  python scripts/compare_llumnix_priority.py --priority_levels 7 8
  python scripts/compare_llumnix_priority.py --num_requests 1000 --qps 3.0
"""
import argparse
import datetime
import os
import subprocess
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def run_simulation(config_name: str, num_priority_levels: int, out_dir: Path, args, global_scheduler: str, replica_scheduler: str):
    """Run a single simulation with specified configuration."""
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
        str(args.num_replicas),
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
        replica_scheduler,
        "--global_scheduler_config_type",
        global_scheduler,
        "--synthetic_request_generator_config_num_priority_levels",
        str(num_priority_levels),
        "--metrics_config_output_dir",
        str(out_dir),
        "--metrics_config_wandb_project",
        "",
        "--metrics_config_wandb_group",
        "",
        "--no-metrics_config_enable_chrome_trace",
    ]

    # Add scheduler-specific configs
    cmd += [
        f"--{replica_scheduler}_scheduler_config_batch_size_cap", str(args.batch_cap),
        f"--{replica_scheduler}_scheduler_config_block_size", str(args.block_size),
        f"--{replica_scheduler}_scheduler_config_max_tokens_in_batch", str(args.max_tokens_in_batch),
    ]

    if args.num_blocks is not None:
        cmd += [f"--{replica_scheduler}_scheduler_config_num_blocks", str(args.num_blocks)]
    
    # Add headroom decay mode for llumlet
    if replica_scheduler == "llumlet":
        cmd += [f"--llumlet_scheduler_config_headroom_decay_mode", args.headroom_decay_mode]

    # Enable migration if requested (only for llumnix global scheduler)
    if args.enable_migration and global_scheduler == "llumnix":
        cmd += ["--llumnix_global_scheduler_config_enable_migration"]

    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    print(f"\nRunning {config_name} with {num_priority_levels} priority levels...")
    print("Command:", " ".join(cmd))
    
    try:
        subprocess.run(cmd, check=True, env=env)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Run failed for {config_name}: {e}")
        return False


def collect_metrics(out_dir: Path):
    """Collect metrics from simulation output."""
    matches = list(out_dir.rglob("request_metrics.csv"))
    if not matches:
        print(f"Warning: metrics CSV not found under {out_dir}")
        return None

    csv_path = matches[0]
    df = pd.read_csv(csv_path)
    
    required_cols = ["request_e2e_time"]
    if not all(col in df.columns for col in required_cols):
        print(f"Warning: missing required columns in {csv_path}")
        return None

    metrics = {
        "request_p99": df["request_e2e_time"].quantile(0.99),
        "request_mean": df["request_e2e_time"].mean(),
    }

    # Collect prefill metrics if available
    if "prefill_e2e_time" in df.columns:
        metrics["prefill_p99"] = df["prefill_e2e_time"].quantile(0.99)
        metrics["prefill_mean"] = df["prefill_e2e_time"].mean()
    
    # Collect decode metrics (try different column names)
    decode_col = None
    if "decode_time_execution_plus_preemption_normalized" in df.columns:
        decode_col = "decode_time_execution_plus_preemption_normalized"
    elif "decode_time" in df.columns:
        decode_col = "decode_time"
    
    if decode_col:
        metrics["decode_p99"] = df[decode_col].quantile(0.99)
        metrics["decode_mean"] = df[decode_col].mean()

    # Collect preemption metrics if available
    if "num_restarts" in df.columns:
        # Preemption loss = number of restarts / total requests
        metrics["preemption_loss"] = df["num_restarts"].sum() / len(df)
    
    return metrics


def plot_comparison(results: dict, out_file: Path, args):
    """Create multi-panel comparison plot."""
    # Organize data by priority level
    llumlet_data = {}
    vllm_data = {}
    
    for label, metrics in results.items():
        parts = label.split("@")
        if len(parts) != 2:
            continue
        scheduler_type = parts[0]
        priority_level = int(parts[1].rstrip("p"))
        
        if scheduler_type == "llumlet":
            llumlet_data[priority_level] = metrics
        elif scheduler_type == "vllm":
            vllm_data[priority_level] = metrics
    
    priority_levels = sorted(set(list(llumlet_data.keys()) + list(vllm_data.keys())))
    
    if not priority_levels:
        print("No data to plot")
        return
    
    # Define metrics to plot
    metric_configs = [
        ("request_p99", "Request P99", "s"),
        ("request_mean", "Request Mean", "s"),
        ("prefill_p99", "Prefill P99", "s"),
        ("prefill_mean", "Prefill Mean", "s"),
        ("decode_p99", "Decode P99", "s"),
        ("decode_mean", "Decode Mean", "s"),
        ("preemption_loss", "Preemption Loss", ""),
    ]
    
    # Filter to only metrics we have data for
    available_metrics = []
    for metric_key, title, unit in metric_configs:
        has_data = any(metric_key in metrics for metrics in results.values())
        if has_data:
            available_metrics.append((metric_key, title, unit))
    
    if not available_metrics:
        print("No metrics available to plot")
        return
    
    # Create subplots
    n_plots = len(available_metrics)
    fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 4))
    if n_plots == 1:
        axes = [axes]
    
    for ax, (metric_key, title, unit) in zip(axes, available_metrics):
        llumlet_values = [llumlet_data.get(p, {}).get(metric_key, None) for p in priority_levels]
        vllm_values = [vllm_data.get(p, {}).get(metric_key, None) for p in priority_levels]
        
        # Plot with different styles
        if any(v is not None for v in llumlet_values):
            ax.plot(priority_levels, llumlet_values, 'o-', label='Llumnix', 
                   color='blue', linewidth=2, markersize=6)
        if any(v is not None for v in vllm_values):
            ax.plot(priority_levels, vllm_values, 's--', label='Round-Robin', 
                   color='orange', linewidth=2, markersize=6)
        
        ax.set_xlabel("Priority Levels", fontsize=11)
        ylabel = f"{title} ({unit})" if unit else title
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        
        # Set x-axis to show integer priority levels
        ax.set_xticks(priority_levels)
    
    plt.suptitle(f"Llumlet (Priority-Aware) vs vLLM (FCFS)\n{args.num_replicas} Replicas, {args.num_requests} Requests", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison plot to {out_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compare Llumnix with llumlet vs round-robin scheduling"
    )
    
    # Simulation parameters
    parser.add_argument("--num_requests", type=int, default=800, 
                       help="Total number of requests (local test: 800, production: 2000+)")
    parser.add_argument("--qps", type=float, default=10.0, 
                       help="Queries per second (very high QPS shows priority scheduling benefits)")
    parser.add_argument("--num_replicas", type=int, default=4, 
                       help="Number of replicas (balanced load distribution)")
    parser.add_argument("--priority_levels", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                       help="Priority levels to test (more levels shows scheduling benefits)")
    
    # Model/device parameters
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--device", type=str, default="a100")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    
    # Request parameters
    parser.add_argument("--prefill_tokens", type=int, default=512)
    parser.add_argument("--decode_tokens", type=int, default=128)
    
    # Scheduler parameters  
    parser.add_argument("--batch_cap", type=int, default=96, 
                       help="Batch size cap - allows better batch packing")
    parser.add_argument("--max_tokens_in_batch", type=int, default=4096,
                       help="Token limit per batch - allows mixed priority batching")
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--num_blocks", type=int, default=None)
    parser.add_argument("--headroom_decay_mode", type=str, default="exponential",
                       choices=["linear", "exponential"],
                       help="Headroom decay mode for llumlet: 'linear' or 'exponential'")
    parser.add_argument("--enable_migration", action="store_true",
                       help="Enable live migration for llumnix")
    
    # Output parameters
    parser.add_argument("--results_dir", type=str, default="results/llumnix_priority_cmp")
    parser.add_argument("--skip_run", action="store_true",
                       help="Skip simulations and only plot existing results")
    
    args = parser.parse_args()
    
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Run simulations for both schedulers across priority levels
    # Compare: llumlet (priority-aware) with llumnix global vs vllm (FCFS) with round-robin
    schedulers = [
        ("llumlet", "llumnix", "llumlet"),  # (label, global_scheduler, replica_scheduler)
        ("vllm", "round_robin", "vllm"),     # vllm is FCFS baseline
    ]
    
    for config_name, global_scheduler, replica_scheduler in schedulers:
        for num_priority_levels in args.priority_levels:
            label = f"{config_name}@{num_priority_levels}p"
            out_dir = Path("simulator_output") / f"{ts}_{config_name}_{num_priority_levels}p"
            
            if not args.skip_run:
                ok = run_simulation(config_name, num_priority_levels, out_dir, args, global_scheduler, replica_scheduler)
                if not ok:
                    print(f"Skipping metrics collection for failed run {label}")
                    continue
            
            metrics = collect_metrics(out_dir)
            if metrics is None:
                print(f"No metrics for {label} (looked in {out_dir})")
                continue
            
            results[label] = metrics
            print(f"  {label}: {metrics}")
    
    # Save combined results
    if results:
        df_all = pd.DataFrame(results).T
        csv_all = results_dir / f"{ts}_llumnix_comparison.csv"
        df_all.to_csv(csv_all)
        print(f"\nSaved combined CSV to {csv_all}")
        
        # Create comparison plot
        out_png = results_dir / f"{ts}_llumnix_comparison.png"
        plot_comparison(results, out_png, args)
    else:
        print("\nNo results to save or plot")


if __name__ == "__main__":
    main()
