"""Load real benchmark artifacts for the CLDB dashboard.

The dashboard previously rendered hard-coded numbers. Everything here reads
the JSON exported by ``experiments/run_rl_evaluation.py`` so the UI reflects
the measurements that were actually run. When no artifact exists the callers
fall back to an explicit "run the benchmark" state instead of inventing data.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_RESULTS = Path(__file__).resolve().parent.parent / "experiments" / "results" / "cl_results.json"
DEFAULT_CWIL = Path(__file__).resolve().parent.parent / "experiments" / "results" / "cwil_ablation.json"


def results_path() -> Path:
    return Path(os.environ.get("CLDB_RESULTS_PATH", DEFAULT_RESULTS))


def _load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def load_results() -> Optional[Dict]:
    return _load_json(results_path())


def load_cwil_ablation() -> Optional[Dict]:
    return _load_json(Path(os.environ.get("CLDB_CWIL_PATH", DEFAULT_CWIL)))


def agent_names(results: Dict) -> List[str]:
    return list(results.get("agents", {}).keys())


def agent_metric(results: Dict, agent: str, metric: str) -> Optional[float]:
    entry = results.get("agents", {}).get(agent, {}).get("metrics", {}).get(metric)
    return None if entry is None else float(entry[0])


def agent_metric_rows(results: Dict) -> List[Dict]:
    metrics = ["ACC", "BWT", "FWT", "F", "Conflict"]
    rows = []
    for agent in agent_names(results):
        row = {"agent": agent}
        for metric in metrics:
            value = agent_metric(results, agent, metric)
            row[metric] = None if value is None else round(value, 4)
        rows.append(row)
    return rows


def best_agent(results: Dict, metric: str = "Conflict", higher_is_better: bool = True) -> Optional[str]:
    scored = [(agent_metric(results, a, metric), a) for a in agent_names(results)]
    scored = [(v, a) for v, a in scored if v is not None]
    if not scored:
        return None
    return (max if higher_is_better else min)(scored)[1]


def r_matrix(results: Dict, agent: str) -> Optional[List[List[Optional[float]]]]:
    return results.get("agents", {}).get(agent, {}).get("R_mean")


def tasks(results: Dict) -> List[int]:
    return results.get("config", {}).get("tasks", [])
