"""Command-line entry point: `python -m shipgate run ...`."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from . import report
from .providers import AnthropicProvider, MockProvider
from .runner import compare, gate, run_suite


def _load(path: str):
    with open(path) as f:
        return json.load(f)


def cmd_run(args: argparse.Namespace) -> int:
    suite = _load(args.suite)
    with open(args.prompt) as f:
        system_prompt = f.read()

    if args.provider == "mock":
        if not args.mock:
            sys.exit("--mock FILE is required with --provider mock")
        provider = MockProvider(args.mock)
    else:
        provider = AnthropicProvider(model=args.model, judge_model=args.judge_model)

    label = args.label or os.path.splitext(os.path.basename(args.prompt))[0]
    run = run_suite(suite, provider, system_prompt, label)
    diff = compare(run, _load(args.baseline)) if args.baseline else None
    if diff:
        run["comparison"] = diff
    ship, reasons = gate(run, diff, args.min_pass, args.allow_regressions)
    run["gate"] = {"ship": ship, "reasons": reasons, "min_pass": args.min_pass}

    os.makedirs(args.out, exist_ok=True)
    run_path = os.path.join(args.out, f"{label}.json")
    with open(run_path, "w") as f:
        json.dump(run, f, indent=2)
    html_path = os.path.join(args.out, f"{label}.html")
    with open(html_path, "w") as f:
        f.write(report.render(run, diff, ship, reasons))

    s = run["summary"]
    print(f"\n{suite['name']} · {label} · {provider.name}")
    print(f"  pass rate   {s['pass_rate']:.0%} ({s['passed']}/{s['total']})")
    for tag, t in s["by_tag"].items():
        print(f"    {tag:<18} {t['passed']}/{t['total']}")
    if diff:
        print(f"  vs {diff['baseline_label']:<8} {diff['delta'] * 100:+.0f} pts · "
              f"{len(diff['fixes'])} fixed · {len(diff['regressions'])} regressed")
    print(f"  verdict     {'SHIP' if ship else 'BLOCKED'}")
    for r in reasons:
        print(f"    - {r}")
    print(f"  results     {run_path}\n  report      {html_path}\n")
    return 0 if ship else 1


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="shipgate", description="Release gate for LLM features.")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a suite against a system prompt")
    r.add_argument("suite", help="path to suite JSON")
    r.add_argument("--prompt", required=True, help="system prompt file under test")
    r.add_argument("--provider", choices=["mock", "anthropic"], default="mock")
    r.add_argument("--mock", help="canned outputs JSON (mock provider)")
    r.add_argument("--model", default="claude-opus-5-5")
    r.add_argument("--judge-model", default="claude-haiku-5-5")
    r.add_argument("--label", help="run name (defaults to prompt filename)")
    r.add_argument("--baseline", help="previous run JSON to diff against")
    r.add_argument("--min-pass", type=float, default=0.85, help="minimum pass rate to ship")
    r.add_argument("--allow-regressions", action="store_true")
    r.add_argument("--out", default="runs", help="output directory")
    args = p.parse_args(argv)
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
