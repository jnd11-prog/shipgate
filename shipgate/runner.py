"""Run a suite, summarize results, diff against a baseline, and apply the release gate."""
from __future__ import annotations

import statistics
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .checks import CheckResult, run_check


def run_suite(
    suite: Dict[str, Any], provider: Any, system_prompt: str, label: str
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    for case in suite["cases"]:
        start = time.perf_counter()
        try:
            output = provider.generate(case, system_prompt)
            error: Optional[str] = None
        except Exception as e:  # one broken case should not sink the whole run
            output, error = "", f"{type(e).__name__}: {e}"
        latency_ms = round((time.perf_counter() - start) * 1000, 1)

        if error:
            checks = [CheckResult("generation", False, error)]
        else:
            checks = [run_check(spec, output, case, provider.judge) for spec in case["checks"]]

        results.append(
            {
                "id": case["id"],
                "tag": case.get("tag", "untagged"),
                "severity": case.get("severity", "normal"),
                "input": case["input"],
                "output": output,
                "passed": all(c.passed for c in checks),
                "latency_ms": latency_ms,
                "checks": [c.to_dict() for c in checks],
            }
        )

    return {
        "label": label,
        "suite": suite["name"],
        "provider": provider.name,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": summarize(results),
        "results": results,
    }


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_tag: Dict[str, Dict[str, int]] = {}
    for r in results:
        t = by_tag.setdefault(r["tag"], {"total": 0, "passed": 0})
        t["total"] += 1
        t["passed"] += int(r["passed"])
    total = len(results)
    passed = sum(r["passed"] for r in results)
    latencies = [r["latency_ms"] for r in results]
    return {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "critical_failures": sum(
            1 for r in results if not r["passed"] and r["severity"] == "critical"
        ),
        "p50_latency_ms": round(statistics.median(latencies), 1) if latencies else 0.0,
        "by_tag": dict(sorted(by_tag.items())),
    }


def compare(run: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    before = {r["id"]: r["passed"] for r in baseline["results"]}
    regressions, fixes = [], []
    for r in run["results"]:
        if r["id"] not in before:
            continue
        if before[r["id"]] and not r["passed"]:
            regressions.append(r["id"])
        elif not before[r["id"]] and r["passed"]:
            fixes.append(r["id"])
    return {
        "baseline_label": baseline["label"],
        "baseline_pass_rate": baseline["summary"]["pass_rate"],
        "delta": round(run["summary"]["pass_rate"] - baseline["summary"]["pass_rate"], 4),
        "regressions": regressions,
        "fixes": fixes,
    }


def gate(
    run: Dict[str, Any],
    diff: Optional[Dict[str, Any]],
    min_pass: float,
    allow_regressions: bool = False,
) -> Tuple[bool, List[str]]:
    """Return (ship?, reasons it was blocked)."""
    reasons = []
    s = run["summary"]
    if s["pass_rate"] < min_pass:
        reasons.append(f"pass rate {s['pass_rate']:.0%} is below the {min_pass:.0%} bar")
    if s["critical_failures"]:
        reasons.append(f"{s['critical_failures']} critical case(s) failing")
    if diff and diff["regressions"] and not allow_regressions:
        reasons.append(
            f"{len(diff['regressions'])} regression(s) vs {diff['baseline_label']}: "
            + ", ".join(diff["regressions"])
        )
    return not reasons, reasons
