import json
import os
import tempfile
import unittest

from shipgate.checks import run_check
from shipgate.cli import main
from shipgate.providers import MockProvider
from shipgate.runner import compare, gate, run_suite

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.join(HERE, "..", "examples", "support_bot")
CASE = {"id": "x", "input": "hi"}


class ChecksTest(unittest.TestCase):
    def test_contains_case_insensitive(self):
        self.assertTrue(run_check({"type": "contains", "value": "30 DAYS"}, "within 30 days", CASE).passed)

    def test_not_contains_any_of(self):
        r = run_check({"type": "not_contains", "any_of": ["a1", "b2"]}, "has b2", CASE)
        self.assertFalse(r.passed)
        self.assertIn("b2", r.detail)

    def test_regex_should_not_match(self):
        spec = {"type": "regex", "pattern": r"\b[A-Z]{4,}\d{2,}\b", "should_match": False, "case_sensitive": True}
        self.assertFalse(run_check(spec, "use code FREEBIE100", CASE).passed)
        self.assertTrue(run_check(spec, "no codes here", CASE).passed)

    def test_max_words(self):
        self.assertFalse(run_check({"type": "max_words", "value": 2}, "one two three", CASE).passed)

    def test_json_valid_with_fence_and_keys(self):
        spec = {"type": "json_valid", "required_keys": ["name"]}
        self.assertTrue(run_check(spec, '```json\n[{"name": "a"}]\n```', CASE).passed)
        self.assertFalse(run_check(spec, '[{"nom": "a"}]', CASE).passed)
        self.assertFalse(run_check(spec, "Here you go: [1]", CASE).passed)

    def test_judge_without_judge_fails_closed(self):
        self.assertFalse(run_check({"type": "judge", "rubric": "r"}, "x", CASE).passed)


class EndToEndTest(unittest.TestCase):
    def _run(self, version):
        with open(os.path.join(EX, "suite.json")) as f:
            suite = json.load(f)
        provider = MockProvider(os.path.join(EX, "mock", f"{version}.json"))
        return run_suite(suite, provider, "", version)

    def test_v1_fails_safety(self):
        run = self._run("v1")
        self.assertEqual(run["summary"]["passed"], 8)
        self.assertEqual(run["summary"]["critical_failures"], 2)
        ship, _ = gate(run, None, 0.85)
        self.assertFalse(ship)

    def test_v2_improves_but_regression_blocks(self):
        v1, v2 = self._run("v1"), self._run("v2")
        diff = compare(v2, v1)
        self.assertEqual(diff["regressions"], ["plan-price"])
        self.assertEqual(len(diff["fixes"]), 4)
        ship, reasons = gate(v2, diff, 0.85)
        self.assertFalse(ship)
        self.assertTrue(any("regression" in r for r in reasons))
        ship, _ = gate(v2, diff, 0.85, allow_regressions=True)
        self.assertTrue(ship)

    def test_cli_writes_report_and_exit_code(self):
        with tempfile.TemporaryDirectory() as out:
            args = ["run", os.path.join(EX, "suite.json"), "--prompt", os.path.join(EX, "prompts", "v1.txt"),
                    "--mock", os.path.join(EX, "mock", "v1.json"), "--out", out]
            self.assertEqual(main(args), 1)
            self.assertTrue(os.path.exists(os.path.join(out, "v1.html")))


if __name__ == "__main__":
    unittest.main()
