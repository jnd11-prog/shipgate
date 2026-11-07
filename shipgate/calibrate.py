"""Measure how well the LLM judge agrees with human labels before trusting it in the gate."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

from . import stats


class LabelJudge:
    """Offline judge that replays the verdicts recorded in the label file (for demos and tests)."""

    name = "mock"

    def judge(self, item: Dict[str, Any], output: str, rubric: str) -> Tuple[bool, str]:
        v = item.get("mock_judge", {})
        return bool(v.get("pass", False)), v.get("reason", "no mock verdict")


def calibrate(items: List[Dict[str, Any]], judge: Any, workers: int = 1,
              confidence: float = 0.95) -> Dict[str, Any]:
    def grade(item: Dict[str, Any]) -> Tuple[bool, str]:
        return judge.judge(item, item["output"], item["rubric"])

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        verdicts = list(pool.map(grade, items))

    pairs = [(bool(it["human_pass"]), v[0]) for it, v in zip(items, verdicts)]
    agreement = stats.cohen_kappa(pairs)
    agree_n = agreement["tp"] + agreement["tn"]
    agreement["agreement_ci"] = stats.wilson_interval(agree_n, agreement["n"], confidence)
    agreement["disagreements"] = [
        {
            "id": it["id"],
            "human_pass": it["human_pass"],
            "judge_pass": v[0],
            "judge_reason": v[1],
            "kind": "judge too lenient" if v[0] else "judge too strict",
        }
        for it, v in zip(items, verdicts)
        if bool(it["human_pass"]) != v[0]
    ]
    return agreement


def interpret_kappa(k: float) -> str:
    # Landis & Koch (1977) bands; conventional, not sacred.
    for bound, word in ((0.8, "almost perfect"), (0.6, "substantial"), (0.4, "moderate"),
                        (0.2, "fair"), (0.0, "slight")):
        if k > bound:
            return word
    return "no better than chance"
