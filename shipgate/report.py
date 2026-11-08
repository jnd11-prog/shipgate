"""Render a run as a single self-contained HTML release report."""
from __future__ import annotations

from html import escape
from typing import Any, Dict, List, Optional

CSS = """
:root{--bg:#f7f7f5;--card:#fff;--ink:#1c1c1a;--mute:#6b6b66;--line:#e4e3de;
--ok:#1f7a4d;--okbg:#e3f3ea;--bad:#b3261e;--badbg:#fbe7e5;--bar:#2f5bd3;--track:#ecebe6}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--card:#1e1e1c;--ink:#ecebe6;--mute:#9b9a94;
--line:#33332f;--ok:#5cc58f;--okbg:#17301f;--bad:#f2867c;--badbg:#3a1d1a;--bar:#7f9cf5;--track:#2a2a27}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif}
main{max-width:960px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:17px;margin:36px 0 12px}
.meta{color:var(--mute);font-size:13px}
.verdict{margin:20px 0;padding:16px 18px;border-radius:12px;font-weight:600}
.verdict.ship{background:var(--okbg);color:var(--ok)}.verdict.block{background:var(--badbg);color:var(--bad)}
.verdict ul{margin:6px 0 0;padding-left:20px;font-weight:400}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}
.kpi b{display:block;font-size:26px;font-variant-numeric:tabular-nums}.kpi span{color:var(--mute);font-size:13px}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}
td,th{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;font-size:14px}
th{color:var(--mute);font-weight:500}td.num{font-variant-numeric:tabular-nums;white-space:nowrap}
.track{background:var(--track);height:8px;border-radius:4px;min-width:80px}.fill{background:var(--bar);height:8px;border-radius:4px}
.case{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;margin-bottom:12px}
.case h3{margin:0 0 6px;font-size:15px}.tag{font-size:12px;color:var(--mute);font-weight:400}
.pill{display:inline-block;font-size:12px;padding:1px 8px;border-radius:99px;margin-left:6px}
.pill.reg{background:var(--badbg);color:var(--bad)}.pill.crit{background:var(--badbg);color:var(--bad)}
.pill.flaky{background:var(--track);color:var(--ink)}
ul.notes{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 14px 14px 34px;margin:0}
ul.notes li{margin:4px 0}
.io{white-space:pre-wrap;background:var(--bg);border-radius:8px;padding:8px 10px;font-size:13px;margin:6px 0;overflow-wrap:anywhere}
.lbl{font-size:12px;color:var(--mute)}.fail{color:var(--bad)}.pass{color:var(--ok)}
ul.checks{margin:6px 0 0;padding-left:18px;font-size:13px}
"""


def _pct(x: float) -> str:
    return f"{x:.0%}"


def _ci(ci: Any) -> str:
    return f"{ci[0]:.0%}–{ci[1]:.0%}"


def _kpis(run: Dict[str, Any], diff: Optional[Dict[str, Any]]) -> str:
    s = run["summary"]
    conf = f"{run.get('confidence', 0.95):.0%}"
    cards = [
        (_pct(s["pass_rate"]), f"pass rate ({s['passed']}/{s['total']}) · {conf} CI {_ci(s['ci'])}"),
        (str(s["critical_failures"]), "critical failures"),
    ]
    if run.get("trials", 1) > 1:
        cards.append((str(len(s["flaky"])), f"flaky cases ({run['trials']} trials each)"))
    if run["provider"] != "mock":
        cards.append((f"{s['p50_latency_ms']:.0f} ms", "median latency"))
    if diff:
        sign = "+" if diff["delta"] >= 0 else ""
        sig = "significant" if diff["significant"] else "not significant"
        cards.insert(1, (f"{sign}{diff['delta'] * 100:.0f} pts",
                         f"vs {escape(diff['baseline_label'])} · p={diff['mcnemar_p']:.2f}, {sig}"))
        cards.append((f"{len(diff['fixes'])} / {len(diff['regressions'])}", "fixes / regressions"))
    return '<div class="kpis">' + "".join(
        f'<div class="kpi"><b>{v}</b><span>{label}</span></div>' for v, label in cards
    ) + "</div>"


def _stats_notes(run: Dict[str, Any], diff: Optional[Dict[str, Any]]) -> str:
    s, g = run["summary"], run.get("gate", {})
    notes = []
    if diff:
        b, c = len(diff["regressions"]), len(diff["fixes"])
        if diff["significant"]:
            notes.append(f"The change is statistically significant (exact McNemar test on {b + c} "
                         f"cases that changed, p={diff['mcnemar_p']:.3f}).")
        else:
            notes.append(f"The {diff['delta'] * 100:+.0f} pt change is <b>not</b> statistically "
                         f"significant: only {b + c} cases changed outcome ({c} fixed, {b} broke), "
                         f"exact McNemar p={diff['mcnemar_p']:.2f}. Treat it as directional.")
    need = g.get("cases_needed")
    if need and s["total"] < need:
        notes.append(f"With {s['total']} cases the {run.get('confidence', 0.95):.0%} interval is "
                     f"{_ci(s['ci'])}. Certifying the {g['min_pass']:.0%} bar needs at least "
                     f"{need} cases, all passing.")
    if s["flaky"]:
        notes.append(f"Flaky (passed some trials, failed others): {escape(', '.join(s['flaky']))}. "
                     "A single-sample eval would have missed these.")
    flake = g.get("flake")
    if flake and run.get("trials", 1) < flake["trials_needed"]:
        notes.append(f"{run.get('trials', 1)} trial(s) per case would catch a behavior that fails "
                     f"{flake['rate']:.0%} of the time with {flake['detect_prob']:.0%} probability. "
                     f"Reaching 95% needs {flake['trials_needed']} trials.")
    if not notes:
        return ""
    return "<h2>Statistical read</h2><ul class='notes'>" + "".join(f"<li>{n}</li>" for n in notes) + "</ul>"


def _tag_table(run: Dict[str, Any]) -> str:
    rows = []
    for tag, t in run["summary"]["by_tag"].items():
        rate = t["passed"] / t["total"]
        ci = _ci(t["ci"]) if "ci" in t else ""
        rows.append(
            f"<tr><td>{escape(tag)}</td><td class='num'>{t['passed']}/{t['total']}</td>"
            f"<td style='width:40%'><div class='track'><div class='fill' style='width:{rate * 100:.0f}%'></div></div></td>"
            f"<td class='num'>{_pct(rate)}</td><td class='num lbl'>{ci}</td></tr>"
        )
    return ("<table><tr><th>Category</th><th>Passed</th><th></th><th>Rate</th><th>95% CI</th></tr>"
            + "".join(rows) + "</table>")


def _case(r: Dict[str, Any], regressions: List[str]) -> str:
    pills = ""
    if r["id"] in regressions:
        pills += '<span class="pill reg">regression</span>'
    if r["severity"] == "critical":
        pills += '<span class="pill crit">critical</span>'
    if r.get("flaky"):
        pills += (f'<span class="pill flaky">flaky · {r["trial_passes"]}/{r["trials"]} trials</span>')
    checks = "".join(
        f"<li class='{'pass' if c['passed'] else 'fail'}'>{'✓' if c['passed'] else '✗'} "
        f"{escape(c['name'])} <span class='lbl'>— {escape(c['detail'])}</span></li>"
        for c in r["checks"]
    )
    return (
        f"<div class='case'><h3>{escape(r['id'])} <span class='tag'>{escape(r['tag'])}</span>{pills}</h3>"
        f"<div class='lbl'>User</div><div class='io'>{escape(r['input'])}</div>"
        f"<div class='lbl'>Model</div><div class='io'>{escape(r['output'])}</div>"
        f"<ul class='checks'>{checks}</ul></div>"
    )


def render(
    run: Dict[str, Any], diff: Optional[Dict[str, Any]], ship: bool, reasons: List[str]
) -> str:
    regressions = diff["regressions"] if diff else []
    failed = [r for r in run["results"] if not r["passed"]]
    failed.sort(key=lambda r: (r["id"] not in regressions, r["severity"] != "critical"))
    if ship:
        verdict = '<div class="verdict ship">SHIP — all release gates passed.</div>'
    else:
        items = "".join(f"<li>{escape(x)}</li>" for x in reasons)
        verdict = f'<div class="verdict block">BLOCKED — do not ship.<ul>{items}</ul></div>'
    fixes = ""
    if diff and diff["fixes"]:
        fixes = "<h2>Fixed since baseline</h2><p>" + ", ".join(escape(f) for f in diff["fixes"]) + "</p>"
    failures = "".join(_case(r, regressions) for r in failed) or "<p>No failing cases.</p>"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shipgate · {escape(run['label'])}</title><style>{CSS}</style></head><body><main>
<h1>{escape(run['suite'])} · {escape(run['label'])}</h1>
<div class="meta">{escape(run['provider'])} · {run.get('trials', 1)} trial(s) per case · {escape(run['timestamp'])}</div>
{verdict}{_kpis(run, diff)}
{_stats_notes(run, diff)}
<h2>Pass rate by category</h2>{_tag_table(run)}
{fixes}
<h2>Failing cases ({len(failed)})</h2>{failures}
</main></body></html>"""
