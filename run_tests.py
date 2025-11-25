"""
Run all Llumnix latency scenarios, generate plots, and log results to Weights & Biases.

For each scenario in vidur.metrics.latency_config.LATENCY_TESTS:
 1) Execute the simulator with a scenario-specific output root.
 2) Run latency_analysis to produce plots under <run_dir>/plots.
 3) Log summaries + plots to wandb under a test-name namespace.

Environment:
  - Set WANDB_PROJECT / WANDB_ENTITY / WANDB_MODE as needed for logging.
  - Metrics tracing must be enabled (already set in the base command).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import wandb

from vidur.metrics.latency_config import LATENCY_TESTS
from vidur.metrics import latency_analysis as la


def _run_command(cmd: str) -> None:
    print(f"[info] Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def _latest_dir(root: Path) -> Path:
    candidates = [p for p in root.glob("*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No simulator outputs found under {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _find_new_run_dir(root: Path, before: set[Path]) -> Path:
    after = {p for p in root.glob("*") if p.is_dir()}
    new_dirs = after - before
    if new_dirs:
        return _latest_dir(Path(root))
    # fallback: no new dir detected; pick latest
    return _latest_dir(Path(root))


def _build_summary(run_dir: Path) -> Dict[str, float]:
    """Compute simple aggregates for wandb logging."""
    chrome_trace_path = run_dir / "chrome_trace.json"
    trace_events = la._load_trace_events(chrome_trace_path)
    priorities = la._extract_request_priorities(trace_events)
    request_df = la._load_request_metrics(run_dir, priorities)
    ttft_df = la._load_ttft(request_df)
    tbt_df = la._load_tbt(run_dir, priorities, trace_events, request_df)

    def qstats(series: pd.Series, prefix: str) -> Dict[str, float]:
        s = series.dropna()
        if s.empty:
            return {}
        return {
            f"{prefix}_mean": float(s.mean()),
            f"{prefix}_p50": float(s.quantile(0.50)),
            f"{prefix}_p90": float(s.quantile(0.90)),
            f"{prefix}_p95": float(s.quantile(0.95)),
            f"{prefix}_p99": float(s.quantile(0.99)),
        }

    summary = {}
    summary.update(qstats(ttft_df["prefill_e2e_time"], "ttft"))
    summary.update(qstats(tbt_df["tbt_seconds"], "tbt"))
    summary.update(qstats(request_df["request_e2e_time"], "e2e"))
    return summary


def _log_to_wandb(
    run,
    test_name: str,
    description: str,
    cmd: str,
    run_dir: Path,
    plots: List[Path],
    summary: Dict[str, float],
    step: int,
) -> None:
    if run is None:
        return
    images = [wandb.Image(str(p), caption=p.name) for p in plots]
    log_payload = {
        f"{test_name}/description": description,
        f"{test_name}/command": cmd,
        f"{test_name}/run_dir": str(run_dir),
        f"{test_name}/plots": images,
    }
    for k, v in summary.items():
        log_payload[f"{test_name}/{k}"] = v
    wandb.log(log_payload, step=step)


def _load_wandb_api_key(env_path: Path = Path(".env")) -> Optional[str]:
    """
    Read WANDB_API_KEY from a .env-style file.
    Keeps dependencies minimal (no python-dotenv requirement).
    """
    if not env_path.exists():
        return None
    key = None
    with env_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("WANDB_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    return key or None


def run_all_tests() -> None:
    api_key = _load_wandb_api_key()
    if api_key:
        wandb.login(key=api_key)

    wandb_run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "llumnix"),
        entity=os.getenv("WANDB_ENTITY"),
        mode=os.getenv("WANDB_MODE", "online"),
        name=os.getenv("WANDB_RUN_NAME", "latency_test_suite"),
        config={"num_tests": len(LATENCY_TESTS)},
    )

    for idx, test in enumerate(LATENCY_TESTS):
        name = test["name"]
        desc = test.get("description", "")

        # Direct outputs for this scenario under simulator_output/<name>/...
        base_root = Path("simulator_output") / name
        before_dirs = {p for p in base_root.glob("*") if p.is_dir()}
        base_root.mkdir(parents=True, exist_ok=True)

        cmd = f"{test['cmd']} --metrics_config_output_dir {base_root}"
        _run_command(cmd)

        run_dir = _find_new_run_dir(base_root, before_dirs)
        print(f"[info] Latest run dir for {name}: {run_dir}")

        # Generate plots
        la.main(str(run_dir))

        plots_dir = run_dir / "plots"
        plots = sorted(p for p in plots_dir.glob("*.png"))
        summary = _build_summary(run_dir)

        _log_to_wandb(
            wandb_run,
            test_name=name,
            description=desc,
            cmd=cmd,
            run_dir=run_dir,
            plots=plots,
            summary=summary,
            step=idx,
        )

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    run_all_tests()
