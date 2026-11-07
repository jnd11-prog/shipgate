"""Command-line entry point: `python -m shipgate {run,calibrate} ...`."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, List, Optional

from . import report, stats
from .calibrate import LabelJudge, calibrate, interpret_kappa
from .providers import AnthropicProvider, CachedProvider, MockProvider
from .runner import compare, gate, run_suite


def _load(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def _fmt_ci(ci: Any) -> str:
    return f"[{ci[0]:.0%}, {ci[1]:.0%}]"


def cmd_run(args: argparse.Namespace) -> int:
    suite = _load(args.suite)
    with open(args.prompt) as f:
        system_prompt = f.read()

    provider: Any
    if args.provider == "mock":
        if not args.mock:
            sys.exit("--mock FILE is required with --provider mock")
        provider = MockProvider(args.mock)
    else:
        provider = AnthropicProvider(model=args.model, judge_model=args.judge_model)
        if not args.no_cache:
            provider = CachedProvider(provider, os.path.join(args.out, ".cache"))

    label = args.label or os.path.splitext(os.path.basename(args.prompt))[0]
    run = run_suite(suite, provider, system_prompt, label, trials=args.trials,
                    workers=args.workers, confidence=args.confidence)
    diff = compare(run, _load(args.baseline)) if args.baseline else None
    if diff:
        run["comparison"] = diff
    ship, reasons = gate(run, diff, args.min_pass, args.allow_regressions, on=args.gate_on)
    power_n = stats.cases_needed(args.min_pass, args.confidence)
    run["gate"] = {"ship": ship, "reasons": reasons, "min_pass": args.min_pass,
                   "on": args.gate_on, "cases_needed": power_n}
    if isinstance(provider, CachedProvider):
        run["cache"] = {"hits": provider.hits, "misses": provider.misses}

    os.makedirs(args.out, exist_ok=True)
    run_path = os.path.join(args.out, f"{label}.json")
    with open(run_path, "w") as f:
        json.dump(run, f, indent=2)
    html_path = os.path.join(args.out, f"{label}.html")
    with open(html_path, "w") as f:
        f.write(report.render(run, diff, ship, reasons))

    s = run["summary"]
    conf = f"{args.confidence:.0%}"
    print(f"\n{suite['name']} · {label} · {provider.name} · {args.trials} trial(s)/case")
    print(f"  pass rate   {s['pass_rate']:.0%} ({s['passed']}/{s['total']}), "
          f"{conf} CI {_fmt_ci(s['ci'])}")
    if args.trials > 1:
        print(f"  per-sample  {s['sample_pass_rate']:.0%}  flaky: {', '.join(s['flaky']) or 'none'}")
    for tag, t in s["by_tag"].items():
        print(f"    {tag:<18} {t['passed']}/{t['total']}")
    if diff:
        sig = "significant" if diff["significant"] else "not significant"
        print(f"  vs {diff['baseline_label']:<8} {diff['delta'] * 100:+.0f} pts · "
              f"{len(diff['fixes'])} fixed · {len(diff['regressions'])} regressed · "
              f"McNemar p={diff['mcnemar_p']:.3f} ({sig})")
    if s["total"] < power_n:
        print(f"  power       {s['total']} cases can't certify {args.min_pass:.0%} at {conf}; "
              f"need ≥{power_n} all-passing cases")
    if "cache" in run:
        print(f"  cache       {run['cache']['hits']} hits, {run['cache']['misses']} misses")
    print(f"  verdict     {'SHIP' if ship else 'BLOCKED'}")
    for r in reasons:
        print(f"    - {r}")
    print(f"  results     {run_path}\n  report      {html_path}\n")
    return 0 if ship else 1


def cmd_calibrate(args: argparse.Namespace) -> int:
    items = _load(args.labels)["items"]
    judge: Any = LabelJudge() if args.provider == "mock" else AnthropicProvider(
        judge_model=args.judge_model)
    res = calibrate(items, judge, workers=args.workers)
    print(f"\nJudge calibration · {judge.name if args.provider == 'mock' else args.judge_model} "
          f"· {res['n']} human-labeled replies")
    print(f"  agreement    {res['agreement']:.0%}  95% CI {_fmt_ci(res['agreement_ci'])}")
    print(f"  Cohen's κ    {res['kappa']:.2f} ({interpret_kappa(res['kappa'])})")
    print(f"  fail recall  {res['fail_recall']:.0%}  (bad replies the judge catches)")
    print(f"  confusion    TP {res['tp']}  TN {res['tn']}  FP {res['fp']} (lenient)  "
          f"FN {res['fn']} (strict)")
    for d in res["disagreements"]:
        print(f"    - {d['id']}: {d['kind']} — {d['judge_reason']}")
    ok = res["kappa"] >= args.min_kappa and res["fail_recall"] >= args.min_fail_recall
    print(f"  verdict      {'TRUSTED' if ok else 'NOT TRUSTED'} "
          f"(needs κ ≥ {args.min_kappa}, fail recall ≥ {args.min_fail_recall:.0%})\n")
    return 0 if ok else 1


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="shipgate", description="Release gate for LLM features.")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a suite against a system prompt")
    r.add_argument("suite", help="path to suite JSON")
    r.add_argument("--prompt", required=True, help="system prompt file under test")
    r.add_argument("--provider", choices=["mock", "anthropic"], default="mock")
    r.add_argument("--mock", help="recorded outputs JSON (mock provider)")
    r.add_argument("--model", default="claude-opus-5-5")
    r.add_argument("--judge-model", default="claude-haiku-5-5")
    r.add_argument("--label", help="run name (defaults to prompt filename)")
    r.add_argument("--baseline", help="previous run JSON to diff against")
    r.add_argument("--trials", type=int, default=1, help="samples per case; a case passes only if all pass")
    r.add_argument("--workers", type=int, default=8, help="parallel API calls")
    r.add_argument("--confidence", type=float, default=0.95, choices=[0.80, 0.90, 0.95, 0.99])
    r.add_argument("--min-pass", type=float, default=0.85, help="minimum pass rate to ship")
    r.add_argument("--gate-on", choices=["point", "lower"], default="point",
                   help="gate on the observed rate or its lower confidence bound")
    r.add_argument("--allow-regressions", action="store_true")
    r.add_argument("--no-cache", action="store_true", help="disable the response cache")
    r.add_argument("--out", default="runs", help="output directory")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("calibrate", help="measure LLM-judge agreement with human labels")
    c.add_argument("labels", help="path to human-labeled replies JSON")
    c.add_argument("--provider", choices=["mock", "anthropic"], default="mock")
    c.add_argument("--judge-model", default="claude-haiku-5-5")
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--min-kappa", type=float, default=0.6)
    c.add_argument("--min-fail-recall", type=float, default=0.9)
    c.set_defaults(func=cmd_calibrate)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
