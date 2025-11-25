"""
Generate TTFT (time-to-first-token) and TBT (time-between-tokens) plots
bucketed by request priority for Llumnix runs. TTFT is read from
request_metrics.csv (prefill_e2e_time). TBT is derived from per-request
decode_time_execution_plus_preemption_normalized (seconds per decode token),
which more closely reflects time between tokens than batch durations.

Usage:
    python vidur/metrics/ttft_tbt_plots.py --run-dir <sim_output_dir>

If --run-dir is omitted, the most recent directory inside simulator_output/ is
used. Plots are written to <run-dir>/plots/.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

# Use a non-interactive backend for CLI / headless runs
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _find_latest_run(sim_output_root: Path) -> Path:
    """Pick the newest directory inside simulator_output/."""
    run_dirs: List[Path] = [p for p in sim_output_root.iterdir() if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(
            f"No simulator outputs found under {sim_output_root}. "
            "Pass --run-dir explicitly if outputs live elsewhere."
        )
    return max(run_dirs, key=lambda p: p.stat().st_mtime)


def _load_trace_events(trace_path: Path) -> List[dict]:
    with trace_path.open() as f:
        data = json.load(f)
    events = data.get("traceEvents", [])
    if not isinstance(events, list):
        raise ValueError(f"Unexpected chrome trace format in {trace_path}")
    return events


def _extract_request_priorities(trace_events: Iterable[dict]) -> Dict[int, int]:
    """
    Build request_id -> priority map from chrome trace events.
    Priority lives in args.request_priorities (one per request in the batch).
    """
    mapping: Dict[int, int] = {}
    for ev in trace_events:
        args = ev.get("args", {})
        req_ids = args.get("request_ids") or []
        req_prios = args.get("request_priorities") or []
        if not req_ids or not req_prios:
            continue

        for req_id, prio in zip(req_ids, req_prios):
            if req_id in mapping and mapping[req_id] != prio:
                # keep the first seen value and warn, but do not fail
                print(
                    f"[warn] Request {req_id} priority mismatch: "
                    f"{mapping[req_id]} vs {prio}. Using {mapping[req_id]}."
                )
                continue
            mapping[req_id] = prio
    return mapping


def _extract_tbt(trace_events: Iterable[dict]) -> pd.DataFrame:
    """
    Get per-batch execution durations (proxy for TBT) and associated priority.
    Duration is stored in microseconds in chrome trace, convert to seconds.
    """
    rows: List[Tuple[int, float]] = []
    for ev in trace_events:
        args = ev.get("args", {})
        prio = args.get("batch_priority")
        if prio is None:
            req_prios = args.get("request_priorities") or []
            if req_prios and len(set(req_prios)) == 1:
                prio = req_prios[0]
        if prio is None:
            continue

        dur_us = ev.get("dur")
        if dur_us is None:
            continue
        rows.append((int(prio), float(dur_us) / 1e6))

    return pd.DataFrame(rows, columns=["priority", "tbt_seconds"])


def _plot_cdf(
    df: pd.DataFrame,
    value_col: str,
    output_path: Path,
    title: str,
    xlabel: str,
) -> None:
    """Draw a simple CDF split by priority."""
    if df.empty:
        print(f"[warn] No data available for {title}; skipping plot.")
        return

    plt.figure(figsize=(8, 5))
    ax = sns.ecdfplot(data=df, x=value_col, hue="priority")
    plt.xlabel(xlabel)
    plt.ylabel("CDF")
    plt.title(title)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(title="Priority")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[info] Wrote {output_path}")


def _plot_hist(
    df: pd.DataFrame,
    value_col: str,
    output_path: Path,
    title: str,
    xlabel: str,
    log_x: bool = False,
) -> None:
    if df.empty:
        print(f"[warn] No data available for {title}; skipping plot.")
        return

    plt.figure(figsize=(8, 5))
    ax = sns.histplot(
        data=df, x=value_col, hue="priority", bins=30, element="step", stat="density"
    )
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(title="Priority")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[info] Wrote {output_path}")


def _print_stats(df: pd.DataFrame, value_col: str, label: str) -> None:
    if df.empty:
        print(f"[warn] No data for {label}; skipping stats.")
        return
    grouped = df.groupby("priority")[value_col]
    print(f"[info] {label} stats by priority:")
    for prio, series in grouped:
        desc = series.describe(percentiles=[0.5, 0.9, 0.95, 0.99])
        print(
            f"  prio {prio}: n={int(desc['count'])}, "
            f"mean={desc['mean']:.4f}, p50={desc['50%']:.4f}, "
            f"p90={desc['90%']:.4f}, p95={desc['95%']:.4f}, p99={desc['99%']:.4f}"
        )


def _plot_distribution(
    df: pd.DataFrame,
    value_col: str,
    output_path: Path,
    title: str,
    xlabel: str,
) -> None:
    """Plot histogram + KDE split by priority."""
    if df.empty:
        print(f"[warn] No data available for {title}; skipping distribution plot.")
        return

    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x=value_col, hue="priority", kde=True, stat="density", common_norm=False)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[info] Wrote {output_path}")


def _load_run_dir(run_dir: Optional[Path]) -> Path:
    if run_dir is not None:
        return run_dir
    sim_output_root = Path("simulator_output")
    return _find_latest_run(sim_output_root)


def main(run_dir: Optional[str] = None) -> None:
    run_dir_path = _load_run_dir(Path(run_dir) if run_dir else None)
    print(f"[info] Using run directory: {run_dir_path}")

    request_metrics_path = run_dir_path / "request_metrics.csv"
    chrome_trace_path = run_dir_path / "chrome_trace.json"
    if not request_metrics_path.exists():
        raise FileNotFoundError(
            f"request_metrics.csv not found in {run_dir_path}. "
            "Re-run the simulator with metrics enabled or pass a different --run-dir."
        )
    if not chrome_trace_path.exists():
        raise FileNotFoundError(
            f"chrome_trace.json not found in {run_dir_path}. "
            "Enable chrome tracing in metrics_config to generate it."
        )

    trace_events = _load_trace_events(chrome_trace_path)
    request_priorities = _extract_request_priorities(trace_events)
    if not request_priorities:
        raise ValueError("Could not find any request priorities in chrome_trace.json")

    request_df = pd.read_csv(request_metrics_path)
    request_df["priority"] = request_df["Request Id"].map(request_priorities)
    ttft_df = request_df[["priority", "prefill_e2e_time"]].dropna()
    if ttft_df.empty:
        raise ValueError("No TTFT data after joining priorities with request metrics.")

    # TBT: seconds per decode token, includes preemption.
    tbt_df = request_df[
        ["priority", "decode_time_execution_plus_preemption_normalized"]
    ].rename(columns={"decode_time_execution_plus_preemption_normalized": "tbt_seconds"})
    tbt_df = tbt_df.dropna()
    if tbt_df.empty:
        raise ValueError(
            "No TBT data found. Ensure decode_time_execution_plus_preemption_normalized is logged."
        )

    plots_dir = run_dir_path / "plots"
    _plot_cdf(
        ttft_df,
        value_col="prefill_e2e_time",
        output_path=plots_dir / "ttft_by_priority_cdf.png",
        title="TTFT by Priority (prefill_e2e_time)",
        xlabel="Seconds",
    )
    _plot_hist(
        ttft_df,
        value_col="prefill_e2e_time",
        output_path=plots_dir / "ttft_by_priority_hist.png",
        title="TTFT by Priority (prefill_e2e_time)",
        xlabel="Seconds",
    )
    _plot_cdf(
        tbt_df,
        value_col="tbt_seconds",
        output_path=plots_dir / "tbt_by_priority_cdf.png",
        title="TBT by Priority (decode_time_execution_plus_preemption_normalized)",
        xlabel="Seconds / token",
    )
    _plot_hist(
        tbt_df,
        value_col="tbt_seconds",
        output_path=plots_dir / "tbt_by_priority_hist.png",
        title="TBT by Priority (decode_time_execution_plus_preemption_normalized)",
        xlabel="Seconds / token",
        log_x=True,
    )

    _print_stats(ttft_df, "prefill_e2e_time", "TTFT")
    _print_stats(tbt_df, "tbt_seconds", "TBT")
    _plot_distribution(
        ttft_df,
        value_col="prefill_e2e_time",
        output_path=plots_dir / "ttft_distribution_by_priority.png",
        title="TTFT Distribution by Priority",
        xlabel="Seconds",
    )
    _plot_distribution(
        tbt_df,
        value_col="tbt_seconds",
        output_path=plots_dir / "tbt_distribution_by_priority.png",
        title="TBT Distribution by Priority",
        xlabel="Seconds",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot TTFT and TBT grouped by priority level for Llumnix runs."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Simulator output directory containing request_metrics.csv and chrome_trace.json. "
        "Defaults to the newest directory under simulator_output/.",
    )
    args = parser.parse_args()
    main(args.run_dir)
