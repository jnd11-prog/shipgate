"""Model providers: a live Claude provider and an offline mock for demos and CI."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Tuple

JUDGE_SYSTEM = (
    "You are a strict QA grader for a customer-support assistant. "
    "Decide whether the assistant's reply satisfies the rubric. "
    'Respond with only a JSON object: {"pass": true|false, "reason": "<one short sentence>"}'
)


class MockProvider:
    """Replays canned outputs (and judge verdicts) keyed by case id. No network, no cost."""

    name = "mock"

    def __init__(self, path: str):
        with open(path) as f:
            self.data: Dict[str, Any] = json.load(f)

    def generate(self, case: Dict[str, Any], system_prompt: str) -> str:
        if case["id"] not in self.data:
            raise KeyError(f"mock file has no output for case {case['id']!r}")
        return self.data[case["id"]]["output"]

    def judge(self, case: Dict[str, Any], output: str, rubric: str) -> Tuple[bool, str]:
        verdict = self.data.get(case["id"], {}).get("judge", {})
        return bool(verdict.get("pass", False)), verdict.get("reason", "no mock verdict")


class AnthropicProvider:
    """Runs the system under test on Claude and grades rubric checks with a cheaper judge model."""

    name = "anthropic"

    def __init__(self, model: str = "claude-opus-5-5", judge_model: str = "claude-haiku-5-5"):
        import anthropic  # imported lazily so the mock path needs no SDK

        self.client = anthropic.Anthropic()
        self.model = model
        self.judge_model = judge_model
        self.name = f"anthropic:{model}"

    def _text(self, resp: Any) -> str:
        if resp.stop_reason == "refusal":
            return "[REFUSED BY MODEL]"
        return "".join(getattr(b, "text", "") for b in resp.content if b.type == "text")

    def generate(self, case: Dict[str, Any], system_prompt: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": case["input"]}],
            # Server-side refusal fallback: if a safety classifier declines, the API retries on
            # a suitable fallback model inside the same call.
            extra_headers={"anthropic-beta": "server-side-fallback-2026-07-01"},
            extra_body={"output_config": {"effort": "medium"}, "fallbacks": "default"},
        )
        return self._text(resp)

    def judge(self, case: Dict[str, Any], output: str, rubric: str) -> Tuple[bool, str]:
        prompt = (
            f"<rubric>{rubric}</rubric>\n"
            f"<user_message>{case['input']}</user_message>\n"
            f"<assistant_reply>{output}</assistant_reply>"
        )
        resp = self.client.messages.create(
            model=self.judge_model,
            max_tokens=1024,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            extra_body={"output_config": {"effort": "low"}},
        )
        raw = self._text(resp)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        try:
            verdict = json.loads(m.group(0)) if m else {}
        except ValueError:
            verdict = {}
        if "pass" not in verdict:
            return False, f"judge returned unparseable output: {raw[:120]!r}"
        return bool(verdict["pass"]), str(verdict.get("reason", ""))
