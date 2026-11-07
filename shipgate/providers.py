"""Model providers: a live Claude provider, an offline mock, and a content-addressed cache."""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from typing import Any, Dict, Optional, Tuple

JUDGE_SYSTEM = (
    "You are a strict QA grader for a customer-support assistant. "
    "Decide whether the assistant's reply satisfies the rubric. Judge only the rubric; "
    "if the reply is ambiguous, fail it. "
    'Respond with only a JSON object: {"pass": true|false, "reason": "<one short sentence>"}'
)

Case = Dict[str, Any]


class MockProvider:
    """Replays recorded outputs and judge verdicts keyed by case id. No network, no cost.

    An entry is either {"output": ..., "judge": {...}} or {"samples": [entry, entry, ...]}.
    Samples are cycled by trial index, which lets the demo model a non-deterministic system.
    """

    name = "mock"

    def __init__(self, path: str):
        with open(path) as f:
            self.data: Dict[str, Any] = json.load(f)

    def _entry(self, case: Case, trial: int) -> Dict[str, Any]:
        if case["id"] not in self.data:
            raise KeyError(f"mock file has no output for case {case['id']!r}")
        entry = self.data[case["id"]]
        samples = entry.get("samples")
        return samples[trial % len(samples)] if samples else entry

    def generate(self, case: Case, system_prompt: str, trial: int = 0) -> str:
        return self._entry(case, trial)["output"]

    def judge(self, case: Case, output: str, rubric: str) -> Tuple[bool, str]:
        for trial in range(len(self.data.get(case["id"], {}).get("samples", [None]))):
            entry = self._entry(case, trial)
            if entry["output"] == output:
                verdict = entry.get("judge", {})
                return bool(verdict.get("pass", False)), verdict.get("reason", "no mock verdict")
        return False, "no mock verdict"


class AnthropicProvider:
    """Runs the system under test on Claude and grades rubric checks with a cheaper judge model."""

    def __init__(self, model: str = "claude-opus-5-5", judge_model: str = "claude-haiku-5-5"):
        import anthropic  # imported lazily so the mock path needs no SDK

        # The SDK retries 408/409/429/5xx and connection errors with exponential backoff.
        self.client = anthropic.Anthropic(max_retries=4)
        self.model = model
        self.judge_model = judge_model
        self.name = f"anthropic:{model}"

    @staticmethod
    def _text(resp: Any) -> str:
        if resp.stop_reason == "refusal":
            return "[REFUSED BY MODEL]"
        return "".join(getattr(b, "text", "") for b in resp.content if b.type == "text")

    def generate(self, case: Case, system_prompt: str, trial: int = 0) -> str:
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

    def judge(self, case: Case, output: str, rubric: str) -> Tuple[bool, str]:
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
        return parse_verdict(self._text(resp))


def parse_verdict(raw: str) -> Tuple[bool, str]:
    """Parse the judge's JSON verdict. Anything unparseable fails closed."""
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        verdict = json.loads(m.group(0)) if m else {}
    except ValueError:
        verdict = {}
    if not isinstance(verdict, dict) or not isinstance(verdict.get("pass"), bool):
        return False, f"judge returned unparseable output: {raw[:120]!r}"
    return verdict["pass"], str(verdict.get("reason", ""))


class CachedProvider:
    """Content-addressed on-disk cache around any provider.

    Keys hash everything that determines a response (provider, prompt, input, trial index; or
    judge model, rubric, input, output). Re-running an unchanged suite is free and bit-for-bit
    reproducible; changing the prompt only re-queries what changed.
    """

    def __init__(self, inner: Any, cache_dir: str):
        self.inner = inner
        self.name = inner.name
        self.dir = cache_dir
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()
        os.makedirs(cache_dir, exist_ok=True)

    def _key(self, *parts: Any) -> str:
        blob = json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(blob).hexdigest()

    def _get(self, key: str) -> Optional[Any]:
        path = os.path.join(self.dir, key + ".json")
        if os.path.exists(path):
            with open(path) as f:
                with self._lock:
                    self.hits += 1
                return json.load(f)
        with self._lock:
            self.misses += 1
        return None

    def _put(self, key: str, value: Any) -> None:
        tmp = os.path.join(self.dir, f"{key}.{threading.get_ident()}.tmp")
        with open(tmp, "w") as f:
            json.dump(value, f)
        os.replace(tmp, os.path.join(self.dir, key + ".json"))  # atomic: no torn reads

    def generate(self, case: Case, system_prompt: str, trial: int = 0) -> str:
        key = self._key("gen", self.inner.name, system_prompt, case["input"], trial)
        hit = self._get(key)
        if hit is not None:
            return hit
        out = self.inner.generate(case, system_prompt, trial)
        self._put(key, out)
        return out

    def judge(self, case: Case, output: str, rubric: str) -> Tuple[bool, str]:
        key = self._key("judge", getattr(self.inner, "judge_model", self.inner.name), rubric,
                        case["input"], output)
        hit = self._get(key)
        if hit is not None:
            return bool(hit[0]), hit[1]
        verdict = self.inner.judge(case, output, rubric)
        self._put(key, list(verdict))
        return verdict
