"""Track one suite across versions: pass rate with uncertainty, and which cases keep moving."""
from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import stats
from .runner import compare

Run = Dict[str, Any]


def _read(path: str) -> Optional[Run]:
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and "results" in data and "summary" in data else None


def load_runs(paths: Sequence[str]) -> List[Run]:
    """Load run files. Files keep the order given; a directory is sorted oldest first."""
    runs: List[Run] = []
    for path in paths:
        if os.path.isdir(path):
            found = [r for r in map(_read, glob.glob(os.path.join(path, "*.json"))) if r]
            runs.extend(sorted(found, key=lambda r: (r["timestamp"], r["label"])))
        else:
            run = _read(path)
            if run:
                runs.append(run)
    return runs


def build(runs: Sequence[Run]) -> Tuple[Dict[str, Any], List[str]]:
    """Summarize runs of the latest run's suite. Returns (history, labels of skipped runs).

    Each step is a paired McNemar test. Looking at many steps inflates the chance that one looks
    significant by luck, so the p-values are Holm-adjusted across the whole history.
    """
    if not runs:
        raise ValueError("no runs")
    suite = runs[-1]["suite"]
    kept = [r for r in runs if r["suite"] == suite]
    skipped = [r["label"] for r in runs if r["suite"] != suite]

    diffs = [compare(cur, prev) for prev, cur in zip(kept, kept[1:])]
    adjusted = stats.holm_adjust([d["mcnemar_p"] for d in diffs])
    rows = []
    for i, run in enumerate(kept):
        s = run["summary"]
        row: Dict[str, Any] = {
            "label": run["label"],
            "passed": s["passed"],
            "total": s["total"],
            "pass_rate": s["pass_rate"],
            "ci": s["ci"],
            "flaky": len(s["flaky"]),
            "ship": run.get("gate", {}).get("ship"),
            "diff": None,
        }
        if i:
            d = diffs[i - 1]
            p = adjusted[i - 1]
            row["diff"] = {"delta": d["delta"], "fixes": len(d["fixes"]),
                           "regressions": len(d["regressions"]), "p": p, "significant": p < 0.05}
        rows.append(row)

    outcomes = [{r["id"]: r["passed"] for r in run["results"]} for run in kept]
    ids = list(dict.fromkeys(i for o in outcomes for i in o))
    cases = []
    for case_id in ids:
        timeline = [o.get(case_id) for o in outcomes]
        if all(t is not False for t in timeline):
            continue  # only cases that have failed at least once are worth a reader's time
        seen = [t for t in timeline if t is not None]
        cases.append({"id": case_id, "timeline": timeline,
                      "flips": sum(a != b for a, b in zip(seen, seen[1:]))})
    return {"suite": suite, "confidence": kept[-1].get("confidence", 0.95),
            "rows": rows, "cases": cases}, skipped
