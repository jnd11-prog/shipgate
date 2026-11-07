"""Deterministic and model-graded checks applied to a single model output."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

# A judge takes (case, output, rubric) and returns (passed, reason).
Judge = Callable[[Dict[str, Any], str, str], Tuple[bool, str]]


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _needles(spec: Dict[str, Any]) -> List[str]:
    if "any_of" in spec:
        return list(spec["any_of"])
    return [spec["value"]]


def _norm(text: str, spec: Dict[str, Any]) -> str:
    return text if spec.get("case_sensitive") else text.lower()


def check_contains(output: str, spec: Dict[str, Any]) -> CheckResult:
    needles = _needles(spec)
    hay = _norm(output, spec)
    hit = next((n for n in needles if _norm(n, spec) in hay), None)
    label = f"contains {needles if len(needles) > 1 else repr(needles[0])}"
    return CheckResult(label, hit is not None, f"found {hit!r}" if hit else "no match")


def check_not_contains(output: str, spec: Dict[str, Any]) -> CheckResult:
    needles = _needles(spec)
    hay = _norm(output, spec)
    hit = next((n for n in needles if _norm(n, spec) in hay), None)
    label = f"does not contain {needles if len(needles) > 1 else repr(needles[0])}"
    return CheckResult(label, hit is None, f"leaked {hit!r}" if hit else "clean")


def check_regex(output: str, spec: Dict[str, Any]) -> CheckResult:
    flags = 0 if spec.get("case_sensitive") else re.IGNORECASE
    m = re.search(spec["pattern"], output, flags)
    want = spec.get("should_match", True)
    return CheckResult(
        f"regex {'matches' if want else 'absent'} /{spec['pattern']}/",
        bool(m) == want,
        f"matched {m.group(0)!r}" if m else "no match",
    )


def check_max_words(output: str, spec: Dict[str, Any]) -> CheckResult:
    n = len(output.split())
    return CheckResult(f"at most {spec['value']} words", n <= spec["value"], f"{n} words")


def _extract_json(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return fenced.group(1).strip() if fenced else text.strip()


def check_json_valid(output: str, spec: Dict[str, Any]) -> CheckResult:
    try:
        data = json.loads(_extract_json(output))
    except ValueError as e:
        return CheckResult("valid JSON", False, f"parse error: {e}")
    required = spec.get("required_keys", [])
    items = data if isinstance(data, list) else [data]
    for item in items:
        missing = [k for k in required if not isinstance(item, dict) or k not in item]
        if missing:
            return CheckResult("valid JSON", False, f"missing keys {missing}")
    return CheckResult("valid JSON", True, "parsed")


DETERMINISTIC = {
    "contains": check_contains,
    "not_contains": check_not_contains,
    "regex": check_regex,
    "max_words": check_max_words,
    "json_valid": check_json_valid,
}


def run_check(
    spec: Dict[str, Any], output: str, case: Dict[str, Any], judge: Optional[Judge] = None
) -> CheckResult:
    kind = spec["type"]
    if kind in DETERMINISTIC:
        return DETERMINISTIC[kind](output, spec)
    if kind == "judge":
        if judge is None:
            return CheckResult("judge", False, "no judge configured")
        passed, reason = judge(case, output, spec["rubric"])
        return CheckResult(f"judge: {spec['rubric']}", passed, reason)
    raise ValueError(f"unknown check type: {kind}")
