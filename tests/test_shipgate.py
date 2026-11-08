import json
import os
import random
import tempfile
import threading
import unittest

from shipgate import stats
from shipgate.calibrate import LabelJudge, calibrate
from shipgate.checks import run_check
from shipgate.cli import main
from shipgate.history import build, load_runs
from shipgate.providers import CachedProvider, MockProvider, parse_verdict
from shipgate.runner import compare, gate, run_suite

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.join(HERE, "..", "examples", "support_bot")
CASE = {"id": "x", "input": "hi"}


def load(*parts):
    with open(os.path.join(EX, *parts)) as f:
        return json.load(f)


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


class VerdictParsingTest(unittest.TestCase):
    def test_parses_json_inside_prose(self):
        self.assertEqual(parse_verdict('Sure: {"pass": true, "reason": "ok"}'), (True, "ok"))

    def test_fails_closed_on_garbage_and_non_bool(self):
        self.assertFalse(parse_verdict("I think it passes")[0])
        self.assertFalse(parse_verdict('{"pass": "yes"}')[0])  # a string is not a verdict
        self.assertFalse(parse_verdict('{"pass": tru')[0])


class StatsTest(unittest.TestCase):
    def test_wilson_known_values(self):
        lo, hi = stats.wilson_interval(8, 12)
        self.assertAlmostEqual(lo, 0.3906, places=3)
        self.assertAlmostEqual(hi, 0.8619, places=3)
        self.assertEqual(stats.wilson_interval(0, 0), (0.0, 1.0))

    def test_wilson_properties(self):
        rng = random.Random(0)
        for _ in range(500):
            n = rng.randint(1, 400)
            k = rng.randint(0, n)
            lo, hi = stats.wilson_interval(k, n)
            self.assertTrue(0 <= lo <= k / n <= hi <= 1)
            # Higher confidence never gives a narrower interval.
            lo99, hi99 = stats.wilson_interval(k, n, 0.99)
            self.assertTrue(lo99 <= lo + 1e-12 and hi99 >= hi - 1e-12)

    def test_mcnemar_exact(self):
        self.assertAlmostEqual(stats.mcnemar_exact(2, 4), 44 / 64)
        self.assertAlmostEqual(stats.mcnemar_exact(0, 6), 2 / 64)
        self.assertEqual(stats.mcnemar_exact(0, 0), 1.0)
        self.assertEqual(stats.mcnemar_exact(3, 3), 1.0)  # capped
        self.assertEqual(stats.mcnemar_exact(1, 9), stats.mcnemar_exact(9, 1))  # symmetric

    def test_cases_needed_matches_closed_form(self):
        # With k = n, the Wilson lower bound is n / (n + z^2).
        for target in (0.5, 0.8, 0.85, 0.9, 0.95, 0.99):
            n = stats.cases_needed(target)
            self.assertGreaterEqual(stats.wilson_interval(n, n)[0], target)
            self.assertLess(stats.wilson_interval(n - 1, n - 1)[0], target)
        self.assertEqual(stats.cases_needed(0.85), 22)

    def test_kappa(self):
        perfect = stats.cohen_kappa([(True, True), (False, False)] * 5)
        self.assertEqual(perfect["kappa"], 1.0)
        chance = stats.cohen_kappa([(True, True), (True, False), (False, True), (False, False)])
        self.assertAlmostEqual(chance["kappa"], 0.0)


class CacheTest(unittest.TestCase):
    def test_hits_and_misses_are_keyed_by_content(self):
        calls = []

        class Counting:
            name = "counting"

            def generate(self, case, system_prompt, trial=0):
                calls.append((case["input"], trial))
                return f"out-{trial}"

            def judge(self, case, output, rubric):
                return True, "ok"

        with tempfile.TemporaryDirectory() as d:
            p = CachedProvider(Counting(), d)
            self.assertEqual(p.generate(CASE, "sys", 0), "out-0")
            self.assertEqual(p.generate(CASE, "sys", 0), "out-0")  # hit
            p.generate(CASE, "sys", 1)  # different trial: miss
            p.generate(CASE, "other prompt", 0)  # different prompt: miss
            self.assertEqual(len(calls), 3)
            self.assertEqual((p.hits, p.misses), (1, 3))
            # A fresh wrapper over the same dir reuses the disk cache.
            p2 = CachedProvider(Counting(), d)
            p2.generate(CASE, "sys", 0)
            self.assertEqual(len(calls), 3)

    def test_parallel_run_is_deterministic(self):
        suite = load("suite.json")
        provider = MockProvider(os.path.join(EX, "mock", "v2.json"))
        a = run_suite(suite, provider, "", "a", trials=5, workers=1)
        b = run_suite(suite, provider, "", "b", trials=5, workers=16)
        strip = lambda run: [(r["id"], r["trial_passes"], r["output"]) for r in run["results"]]
        self.assertEqual(strip(a), strip(b))


class EndToEndTest(unittest.TestCase):
    def _run(self, version, trials=1):
        provider = MockProvider(os.path.join(EX, "mock", f"{version}.json"))
        return run_suite(load("suite.json"), provider, "", version, trials=trials)

    def test_v1_fails_safety(self):
        run = self._run("v1")
        self.assertEqual(run["summary"]["passed"], 8)
        self.assertEqual(run["summary"]["critical_failures"], 2)
        self.assertFalse(gate(run, None, 0.85)[0])

    def test_single_trial_misses_flaky_case(self):
        v1, v2 = self._run("v1"), self._run("v2")
        diff = compare(v2, v1)
        self.assertEqual(diff["regressions"], ["plan-price"])
        self.assertEqual(v2["summary"]["flaky"], [])

    def test_trials_expose_flaky_regression(self):
        v1, v2 = self._run("v1", 5), self._run("v2", 5)
        self.assertEqual(v2["summary"]["flaky"], ["angry-customer"])
        angry = next(r for r in v2["results"] if r["id"] == "angry-customer")
        self.assertEqual((angry["trial_passes"], angry["trials"]), (4, 5))
        self.assertFalse(angry["passed"])
        diff = compare(v2, v1)
        self.assertEqual(sorted(diff["regressions"]), ["angry-customer", "plan-price"])
        self.assertEqual(len(diff["fixes"]), 4)
        self.assertAlmostEqual(diff["mcnemar_p"], 0.6875)
        self.assertFalse(diff["significant"])

    def test_gate_on_lower_bound_is_stricter(self):
        run = self._run("v2")  # 11/12 observed, lower bound ~65%
        self.assertTrue(gate(run, None, 0.85, on="point")[0])
        ship, reasons = gate(run, None, 0.85, on="lower")
        self.assertFalse(ship)
        self.assertIn("lower 95% bound", reasons[0])

    def test_cli_run_and_calibrate_exit_codes(self):
        with tempfile.TemporaryDirectory() as out:
            args = ["run", os.path.join(EX, "suite.json"), "--prompt", os.path.join(EX, "prompts", "v1.txt"),
                    "--mock", os.path.join(EX, "mock", "v1.json"), "--out", out]
            self.assertEqual(main(args), 1)
            self.assertTrue(os.path.exists(os.path.join(out, "v1.html")))
        self.assertEqual(main(["calibrate", os.path.join(EX, "judge_labels.json")]), 1)


class CalibrationTest(unittest.TestCase):
    def test_demo_judge_is_lenient(self):
        res = calibrate(load("judge_labels.json")["items"], LabelJudge())
        self.assertEqual((res["tp"], res["tn"], res["fp"], res["fn"]), (5, 4, 2, 1))
        self.assertAlmostEqual(res["kappa"], 0.5)
        self.assertAlmostEqual(res["fail_recall"], 4 / 6)
        kinds = sorted(d["kind"] for d in res["disagreements"])
        self.assertEqual(kinds, ["judge too lenient", "judge too lenient", "judge too strict"])


class FlakePowerTest(unittest.TestCase):
    def test_known_values(self):
        self.assertAlmostEqual(stats.flake_detection_prob(0.2, 5), 1 - 0.8**5)
        self.assertEqual(stats.trials_needed(0.2, 0.95), 14)
        self.assertEqual(stats.trials_needed(0.5, 0.75), 2)  # exact boundary

    def test_trials_needed_is_minimal(self):
        rng = random.Random(7)
        for _ in range(500):
            f, d = rng.uniform(0.01, 0.95), rng.uniform(0.05, 0.99)
            n = stats.trials_needed(f, d)
            self.assertGreaterEqual(stats.flake_detection_prob(f, n), d)
            if n > 1:
                self.assertLess(stats.flake_detection_prob(f, n - 1), d)

    def test_rejects_bad_rate(self):
        with self.assertRaises(ValueError):
            stats.trials_needed(0.0)
        with self.assertRaises(ValueError):
            stats.flake_detection_prob(1.0, 3)

    def test_cli_rejects_bad_rate_before_running(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["run", "missing.json", "--prompt", "missing.txt", "--flake-rate", "1.5"])
        self.assertIn("flake_rate", str(ctx.exception.code))

    def test_holm(self):
        adj = stats.holm_adjust([0.01, 0.04, 0.03])
        for got, want in zip(adj, [0.03, 0.06, 0.06]):
            self.assertAlmostEqual(got, want)
        self.assertEqual(stats.holm_adjust([]), [])
        self.assertEqual(stats.holm_adjust([0.9, 0.8]), [1.0, 1.0])


class HistoryTest(unittest.TestCase):
    def _demo(self, out):
        for v, baseline in (("v1", []), ("v2", ["--baseline", os.path.join(out, "v1.json")])):
            main(["run", os.path.join(EX, "suite.json"), "--prompt", os.path.join(EX, "prompts", v + ".txt"),
                  "--mock", os.path.join(EX, "mock", v + ".json"), "--trials", "5", "--out", out] + baseline)

    def test_demo_history(self):
        with tempfile.TemporaryDirectory() as out:
            self._demo(out)
            with open(os.path.join(out, "v2.json")) as f:
                self.assertEqual(json.load(f)["gate"]["flake"]["trials_needed"], 14)
            # Junk JSON and non-run JSON in the directory are ignored.
            with open(os.path.join(out, "notes.json"), "w") as f:
                f.write("{not json")
            with open(os.path.join(out, "other.json"), "w") as f:
                json.dump({"hello": 1}, f)
            h, skipped = build(load_runs([os.path.join(out, "v1.json"), os.path.join(out, "v2.json")]))
            self.assertEqual(main(["history", out]), 0)
        self.assertEqual(skipped, [])
        self.assertEqual([r["label"] for r in h["rows"]], ["v1", "v2"])
        d = h["rows"][1]["diff"]
        self.assertAlmostEqual(d["delta"], 2 / 12, places=3)
        self.assertEqual((d["fixes"], d["regressions"]), (4, 2))
        self.assertAlmostEqual(d["p"], 0.6875)  # one comparison, so Holm changes nothing
        by_id = {c["id"]: c for c in h["cases"]}
        self.assertEqual(by_id["pii-ssn"]["timeline"], [False, True])
        self.assertEqual(by_id["plan-price"]["timeline"], [True, False])
        self.assertEqual(by_id["plan-price"]["flips"], 1)
        self.assertEqual(len(h["cases"]), 6)

    def _run(self, label, suite, outcomes, ts):
        results = [{"id": i, "passed": ok} for i, ok in outcomes.items()]
        passed = sum(outcomes.values())
        summary = {"total": len(results), "passed": passed, "pass_rate": passed / len(results),
                   "ci": [0.0, 1.0], "flaky": []}
        return {"label": label, "suite": suite, "timestamp": ts, "summary": summary, "results": results}

    def test_new_case_and_other_suite(self):
        runs = [self._run("a", "S", {"x": True}, "1"), self._run("zz", "Other", {"x": True}, "2"),
                self._run("b", "S", {"x": True, "new": False}, "3")]
        h, skipped = build(runs)
        self.assertEqual(skipped, ["zz"])
        self.assertEqual(h["cases"], [{"id": "new", "timeline": [None, False], "flips": 0}])

    def test_no_runs_exit_code(self):
        with tempfile.TemporaryDirectory() as out:
            self.assertEqual(main(["history", out]), 2)


if __name__ == "__main__":
    unittest.main()
