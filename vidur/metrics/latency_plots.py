"""
Utility plots for latency-oriented analyses across simulation runs.

The functions below expect the aggregated ``analysis/stats.csv`` produced by
``vidur.config_optimizer.analyzer.stats_extractor`` and emit static PNGs for:
    - Tail latency elbow vs number of priority levels
    - End-to-end / prefill / decode mean latency vs QPS
    - P99 latency vs QPS
    - GPU cost vs P99 latency
    - Low/normal/high latency bands (P50/P95/P99) vs QPS
"""

import argparse
import os
from typing import Iterable, Optional

import matplotlib

# Use non-interactive backend to allow headless execution.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from vidur.config_optimizer.analyzer.constants import GPU_COSTS  # noqa: E402

from vidur.logger import init_logger

logger = init_logger(__name__)


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _first_available(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _plot_lines(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    outfile: str,
    xlabel: str,
    ylabel: str,
    title: str,
    color: str = "#1f77b4",
) -> None:
    data = df[[x_col, y_col]].dropna()
    if data.empty:
        logger.warning("Skipping plot %s because %s/%s has no data", outfile, x_col, y_col)
        return

    data = data.sort_values(x_col)
    plt.figure(figsize=(7, 4))
    plt.plot(data[x_col], data[y_col], marker="o", color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()
    logger.info("Wrote plot: %s", outfile)


def plot_tail_latency_elbow(
    df: pd.DataFrame,
    output_dir: str,
    latency_col: Optional[str] = None,
    priority_col: str = "num_priority_levels",
) -> None:
    col = latency_col or _first_available(
        df, ["request_e2e_time_normalized_99%", "request_e2e_time_99%", "ttft_99%"]
    )
    if col is None:
        logger.warning("No P99 latency column found; skipping tail latency elbow plot.")
        return
    if priority_col not in df.columns:
        logger.warning("Column %s not found; skipping tail latency elbow plot.", priority_col)
        return

    grouped = df[[priority_col, col]].dropna().groupby(priority_col)[col].median()
    if grouped.empty:
        logger.warning("No data for %s vs %s; skipping tail latency elbow plot.", priority_col, col)
        return

    grouped = grouped.reset_index().sort_values(priority_col)
    plt.figure(figsize=(6, 4))
    plt.plot(grouped[priority_col], grouped[col], marker="o", color="#d62728")
    plt.xlabel("Priority levels")
    plt.ylabel("P99 request latency (s)")
    plt.title("Tail latency vs priority levels (elbow)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    outfile = os.path.join(output_dir, "tail_latency_elbow.png")
    plt.savefig(outfile, dpi=150)
    plt.close()
    logger.info("Wrote plot: %s", outfile)


def plot_latency_means_vs_qps(df: pd.DataFrame, output_dir: str) -> None:
    qps_col = "poisson_request_interval_generator_qps"
    if qps_col not in df.columns:
        logger.warning("QPS column %s not found; skipping mean latency plots.", qps_col)
        return

    # End-to-end
    e2e_col = _first_available(
        df, ["request_e2e_time_normalized_mean", "request_e2e_time_mean"]
    )
    if e2e_col:
        _plot_lines(
            df,
            qps_col,
            e2e_col,
            os.path.join(output_dir, "e2e_mean_latency_vs_qps.png"),
            xlabel="QPS",
            ylabel="End-to-end mean latency (s)",
            title="End-to-end mean latency vs QPS",
        )

    # Prefill (TTFT)
    prefill_col = _first_available(df, ["ttft_mean", "prefill_e2e_time_mean"])
    if prefill_col:
        _plot_lines(
            df,
            qps_col,
            prefill_col,
            os.path.join(output_dir, "prefill_mean_latency_vs_qps.png"),
            xlabel="QPS",
            ylabel="Prefill mean latency (s)",
            title="Prefill mean latency vs QPS",
            color="#ff7f0e",
        )

    # Decode (fallback to available decode metric; TBT median as last resort)
    decode_col = _first_available(
        df,
        [
            "decode_time_execution_plus_preemption_normalized_mean",
            "decode_time_execution_plus_preemption_mean",
            "tbt_50%",
        ],
    )
    if decode_col:
        _plot_lines(
            df,
            qps_col,
            decode_col,
            os.path.join(output_dir, "decode_mean_latency_vs_qps.png"),
            xlabel="QPS",
            ylabel="Decode mean latency (s)",
            title="Decode mean latency vs QPS",
            color="#2ca02c",
        )
    else:
        logger.warning("No decode latency column found; skipped decode mean plot.")


def plot_p99_vs_qps(
    df: pd.DataFrame,
    output_dir: str,
    p99_col: Optional[str] = None,
    qps_col: str = "poisson_request_interval_generator_qps",
) -> None:
    col = p99_col or _first_available(
        df, ["request_e2e_time_normalized_99%", "request_e2e_time_99%"]
    )
    if col is None:
        logger.warning("No P99 latency column found; skipping P99 vs QPS plot.")
        return
    if qps_col not in df.columns:
        logger.warning("QPS column %s not found; skipping P99 vs QPS plot.", qps_col)
        return

    _plot_lines(
        df,
        qps_col,
        col,
        os.path.join(output_dir, "p99_latency_vs_qps.png"),
        xlabel="QPS",
        ylabel="P99 request latency (s)",
        title="P99 latency vs QPS",
        color="#9467bd",
    )


def plot_gpu_cost_vs_p99(
    df: pd.DataFrame,
    output_dir: str,
    p99_col: Optional[str] = None,
    cost_col: str = "cost",
    qps_col: str = "poisson_request_interval_generator_qps",
) -> None:
    col = p99_col or _first_available(
        df, ["request_e2e_time_normalized_99%", "request_e2e_time_99%"]
    )
    if col is None or cost_col not in df.columns:
        logger.warning(
            "Missing columns (%s or %s); skipping GPU cost vs P99 plot.", col, cost_col
        )
        return

    data = df[[col, cost_col, qps_col]].dropna()
    if data.empty:
        logger.warning("No data for GPU cost vs P99 plot.")
        return

    plt.figure(figsize=(7, 4))
    scatter = plt.scatter(
        data[col],
        data[cost_col],
        c=data[qps_col] if qps_col in data.columns else None,
        cmap="viridis",
        alpha=0.8,
        edgecolor="k",
    )
    plt.xlabel("P99 request latency (s)")
    plt.ylabel("GPU cost ($)")
    plt.title("GPU cost vs P99 latency")
    if qps_col in data.columns:
        cbar = plt.colorbar(scatter)
        cbar.set_label("QPS", rotation=270, labelpad=15)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    outfile = os.path.join(output_dir, "gpu_cost_vs_p99.png")
    plt.savefig(outfile, dpi=150)
    plt.close()
    logger.info("Wrote plot: %s", outfile)


def plot_latency_tiers(
    df: pd.DataFrame,
    output_dir: str,
    qps_col: str = "poisson_request_interval_generator_qps",
) -> None:
    p50_col = _first_available(df, ["request_e2e_time_normalized_50%", "request_e2e_time_50%"])
    p95_col = _first_available(df, ["request_e2e_time_normalized_95%", "request_e2e_time_95%"])
    p99_col = _first_available(df, ["request_e2e_time_normalized_99%", "request_e2e_time_99%"])

    if not all([p50_col, p95_col, p99_col]):
        logger.warning("Missing percentile columns; skipping latency tier plot.")
        return
    if qps_col not in df.columns:
        logger.warning("QPS column %s not found; skipping latency tier plot.", qps_col)
        return

    data = df[[qps_col, p50_col, p95_col, p99_col]].dropna().sort_values(qps_col)
    if data.empty:
        logger.warning("No data for latency tiers plot.")
        return

    plt.figure(figsize=(7, 4))
    plt.plot(data[qps_col], data[p50_col], marker="o", label="P50 (low)", color="#1f77b4")
    plt.plot(data[qps_col], data[p95_col], marker="o", label="P95 (normal)", color="#ff7f0e")
    plt.plot(data[qps_col], data[p99_col], marker="o", label="P99 (high)", color="#d62728")
    plt.xlabel("QPS")
    plt.ylabel("Latency (s)")
    plt.title("Latency tiers vs QPS (low/normal/high)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    outfile = os.path.join(output_dir, "latency_tiers_vs_qps.png")
    plt.savefig(outfile, dpi=150)
    plt.close()
    logger.info("Wrote plot: %s", outfile)


def main():
    parser = argparse.ArgumentParser(description="Generate latency plots from stats.csv.")
    parser.add_argument(
        "--stats-csv",
        type=str,
        default="analysis/stats.csv",
        help="Path to the aggregated stats.csv file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write plots (default: <stats_csv_dir>/latency_plots).",
    )
    parser.add_argument(
        "--p99-col",
        type=str,
        default=None,
        help="Override column name to use for P99 latency.",
    )
    parser.add_argument(
        "--priority-col",
        type=str,
        default="num_priority_levels",
        help="Column representing priority levels for the elbow plot.",
    )
    args = parser.parse_args()

    stats_path = args.stats_csv
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"stats.csv not found at {stats_path}")

    df = pd.read_csv(stats_path)
    out_dir = (
        args.output_dir
        if args.output_dir is not None
        else os.path.join(os.path.dirname(stats_path), "latency_plots")
    )
    _ensure_dir(out_dir)

    plot_tail_latency_elbow(df, out_dir, latency_col=args.p99_col, priority_col=args.priority_col)
    plot_latency_means_vs_qps(df, out_dir)
    plot_p99_vs_qps(df, out_dir, p99_col=args.p99_col)
    plot_gpu_cost_vs_p99(df, out_dir, p99_col=args.p99_col)
    plot_latency_tiers(df, out_dir)


def generate_latency_plots_from_run(sim_config) -> None:
    """
    Convenience entrypoint to generate latency plots directly after a single simulation run.

    Derives basic percentile/mean stats from per-request metrics and feeds them to the
    plotting routines in this module.
    """
    output_dir = sim_config.metrics_config.output_dir
    request_metrics_file = os.path.join(output_dir, "request_metrics.csv")
    completion_ts_file = os.path.join(output_dir, "plots", "request_completion_time_series.csv")

    if not os.path.exists(request_metrics_file):
        logger.warning("request_metrics.csv not found at %s; skipping latency plots.", request_metrics_file)
        return

    df_req = pd.read_csv(request_metrics_file)

    stats = {}

    def add_latency_stats(col: str):
        if col not in df_req.columns:
            return
        stats[f"{col}_mean"] = df_req[col].mean()
        stats[f"{col}_50%"] = df_req[col].quantile(0.50)
        stats[f"{col}_95%"] = df_req[col].quantile(0.95)
        stats[f"{col}_99%"] = df_req[col].quantile(0.99)

    for candidate in [
        "request_e2e_time_normalized",
        "request_e2e_time",
        "prefill_e2e_time",
        "prefill_time_execution_plus_preemption",
        "decode_time_execution_plus_preemption_normalized",
        "decode_time_execution_plus_preemption",
    ]:
        add_latency_stats(candidate)

    # Estimate runtime from completion time series if available.
    runtime = None
    if os.path.exists(completion_ts_file):
        df_completion = pd.read_csv(completion_ts_file)
        if "Time (sec)" in df_completion.columns:
            runtime = df_completion["Time (sec)"].max()
            stats["runtime"] = runtime

    # Add QPS from config if present; otherwise estimate from runtime.
    try:
        qps_cfg = getattr(
            getattr(sim_config.request_generator_config, "interval_generator_config", None),
            "qps",
            None,
        )
        if qps_cfg is not None:
            stats["poisson_request_interval_generator_qps"] = qps_cfg
        elif runtime and len(df_req) > 0:
            stats["poisson_request_interval_generator_qps"] = len(df_req) / runtime
    except Exception:
        pass

    # Priority levels (global scheduler or request generator)
    prio = getattr(
        getattr(sim_config.cluster_config, "global_scheduler_config", None),
        "num_priority_levels",
        None,
    )
    if prio is None:
        prio = getattr(sim_config.request_generator_config, "num_priority_levels", None)
    if prio is not None:
        stats["num_priority_levels"] = prio

    # Cost estimation if device is known.
    device = getattr(getattr(sim_config.cluster_config, "replica_config", None), "device", None)
    num_replicas = getattr(sim_config.cluster_config, "num_replicas", None)
    tp = getattr(getattr(sim_config.cluster_config, "replica_config", None), "tensor_parallel_size", None)
    pp = getattr(getattr(sim_config.cluster_config, "replica_config", None), "num_pipeline_stages", None)
    if runtime and device in GPU_COSTS and all(val is not None for val in [num_replicas, tp, pp]):
        num_gpus = num_replicas * tp * pp
        cost = runtime * num_gpus * GPU_COSTS[device] / 3600
        stats["cost"] = cost
        stats["gpu_hrs"] = runtime * num_gpus / 3600

    stats_df = pd.DataFrame([stats])

    plot_dir = _ensure_dir(os.path.join(output_dir, "latency_plots"))
    stats_df.to_csv(os.path.join(plot_dir, "stats_single_run.csv"), index=False)

    plot_tail_latency_elbow(stats_df, plot_dir)
    plot_latency_means_vs_qps(stats_df, plot_dir)
    plot_p99_vs_qps(stats_df, plot_dir)
    plot_gpu_cost_vs_p99(stats_df, plot_dir)
    plot_latency_tiers(stats_df, plot_dir)


if __name__ == "__main__":
    main()
