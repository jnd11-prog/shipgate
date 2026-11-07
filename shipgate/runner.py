"""Run a suite, summarize results with uncertainty, diff against a baseline, and apply the gate."""
from __future__ import annotations

import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from . import stats
from .checks import CheckResult, run_check

Run = Dict[str, Any]


def _run_one(case: Dict[str, Any], trial: int, provider: Any, system_prompt: str) -> Dict[str, Any]:
    start = time.perf_counter()
    try:
        output = provider.generate(case, system_prompt, trial)
        error: Optional[str] = None
    except Exception as e:  # one broken call should not sink the whole run
        output, error = "", f"{type(e).__name__}: {e}"
    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    if error:
        checks = [CheckResult("generation", False, error)]
    else:
        checks = [run_check(spec, output, case, provider.judge) for spec in case["checks"]]
    return {
        "trial": trial,
        "output": output,
        "passed": all(c.passed for c in checks),
        "latency_ms": latency_ms,
        "checks": [c.to_dict() for c in checks],
    }


def run_suite(
    suite: Dict[str, Any],
    provider: Any,
    system_prompt: str,
    label: str,
    trials: int = 1,
    workers: int = 1,
    confidence: float = 0.95,
) -> Run:
    """Run every case `trials` times. A case passes only if every trial passes.

    The strict rule is deliberate: a support bot that leaks an SSN on 1 in 5 tries is not
    "80% safe", it is unsafe.
    """
    if trials < 1:
        raise ValueError("trials must be >= 1")
    jobs = [(case, t) for case in suite["cases"] for t in range(trials)]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        samples = list(pool.map(lambda j: _run_one(j[0], j[1], provider, system_prompt), jobs))

    results: List[Dict[str, Any]] = []
    for i, case in enumerate(suite["cases"]):
        case_samples = samples[i * trials:(i + 1) * trials]
        passes = sum(s["passed"] for s in case_samples)
        # Show the first failing sample (the most informative one), else the first sample.
        shown = next((s for s in case_samples if not s["passed"]), case_samples[0])
        results.append(
            {
                "id": case["id"],
                "tag": case.get("tag", "untagged"),
                "severity": case.get("severity", "normal"),
                "input": case["input"],
                "trials": trials,
                "trial_passes": passes,
                "passed": passes == trials,
                "flaky": 0 < passes < trials,
                "output": shown["output"],
                "checks": shown["checks"],
                "shown_trial": shown["trial"],
                "latency_ms": statistics.median(s["latency_ms"] for s in case_samples),
                "distinct_outputs": len({s["output"] for s in case_samples}),
            }
        )

    return {
        "label": label,
        "suite": suite["name"],
        "provider": provider.name,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trials": trials,
        "confidence": confidence,
        "summary": summarize(results, confidence),
        "results": results,
    }


def summarize(results: List[Dict[str, Any]], confidence: float = 0.95) -> Dict[str, Any]:
    by_tag: Dict[str, Dict[str, Any]] = {}
    for r in results:
        t = by_tag.setdefault(r["tag"], {"total": 0, "passed": 0})
        t["total"] += 1
        t["passed"] += int(r["passed"])
    for t in by_tag.values():
        t["ci"] = stats.wilson_interval(t["passed"], t["total"], confidence)

    total = len(results)
    passed = sum(r["passed"] for r in results)
    trial_total = sum(r["trials"] for r in results)
    trial_passed = sum(r["trial_passes"] for r in results)
    latencies = [r["latency_ms"] for r in results]
    return {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "ci": stats.wilson_interval(passed, total, confidence),
        "sample_pass_rate": round(trial_passed / trial_total, 4) if trial_total else 0.0,
        "flaky": [r["id"] for r in results if r["flaky"]],
        "critical_failures": sum(
            1 for r in results if not r["passed"] and r["severity"] == "critical"
        ),
        "p50_latency_ms": round(statistics.median(latencies), 1) if latencies else 0.0,
        "by_tag": dict(sorted(by_tag.items())),
    }


def compare(run: Run, baseline: Run) -> Dict[str, Any]:
    """Paired comparison on the cases both runs share."""
    before = {r["id"]: r["passed"] for r in baseline["results"]}
    regressions, fixes, shared = [], [], 0
    for r in run["results"]:
        if r["id"] not in before:
            continue
        shared += 1
        if before[r["id"]] and not r["passed"]:
            regressions.append(r["id"])
        elif not before[r["id"]] and r["passed"]:
            fixes.append(r["id"])
    p = stats.mcnemar_exact(len(regressions), len(fixes))
    return {
        "baseline_label": baseline["label"],
        "baseline_pass_rate": baseline["summary"]["pass_rate"],
        "delta": round(run["summary"]["pass_rate"] - baseline["summary"]["pass_rate"], 4),
        "shared_cases": shared,
        "regressions": regressions,
        "fixes": fixes,
        "mcnemar_p": round(p, 4),
        "significant": p < 0.05,
    }


def gate(
    run: Run,
    diff: Optional[Dict[str, Any]],
    min_pass: float,
    allow_regressions: bool = False,
    on: str = "point",
) -> Tuple[bool, List[str]]:
    """Return (ship?, reasons it was blocked).

    on="point": compare the observed pass rate to the bar.
    on="lower": compare the lower confidence bound, i.e. ship only if we are confident the true
    pass rate clears the bar. Stricter, and needs a suite large enough to support the claim.
    """
    reasons = []
    s = run["summary"]
    if on == "lower":
        lo = s["ci"][0]
        if lo < min_pass:
            reasons.append(
                f"lower {run['confidence']:.0%} bound of pass rate is {lo:.0%}, below the "
                f"{min_pass:.0%} bar (observed {s['pass_rate']:.0%} on {s['total']} cases)"
            )
    elif s["pass_rate"] < min_pass:
        reasons.append(f"pass rate {s['pass_rate']:.0%} is below the {min_pass:.0%} bar")
    if s["critical_failures"]:
        reasons.append(f"{s['critical_failures']} critical case(s) failing")
    if diff and diff["regressions"] and not allow_regressions:
        reasons.append(
            f"{len(diff['regressions'])} regression(s) vs {diff['baseline_label']}: "
            + ", ".join(diff["regressions"])
        )
    return not reasons, reasons
