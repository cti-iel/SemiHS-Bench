"""Tests for --rerun-errors recovery and circuit breaker."""

from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from bench.run_benchmark import (
    _compute_retry_sleep,
    _is_error_prediction,
    _is_rate_limit_error,
    _load_errored_frozen_ids,
    load_existing_predictions,
)


# ---------------------------------------------------------------------------
# _is_rate_limit_error
# ---------------------------------------------------------------------------

class TestIsRateLimitError(unittest.TestCase):
    def _exc(self, msg: str) -> Exception:
        return Exception(msg)

    def test_detects_429(self):
        self.assertTrue(_is_rate_limit_error(self._exc("HTTP 429 Too Many Requests")))

    def test_detects_resource_exhausted(self):
        self.assertTrue(_is_rate_limit_error(self._exc("Resource exhausted: quota exceeded")))

    def test_detects_rate_limit(self):
        self.assertTrue(_is_rate_limit_error(self._exc("RateLimitExceeded for model")))

    def test_ignores_other_errors(self):
        self.assertFalse(_is_rate_limit_error(self._exc("Connection timeout")))
        self.assertFalse(_is_rate_limit_error(self._exc("Invalid API key")))
        self.assertFalse(_is_rate_limit_error(self._exc("Internal server error 500")))


# ---------------------------------------------------------------------------
# _compute_retry_sleep
# ---------------------------------------------------------------------------

class TestComputeRetrySleep(unittest.TestCase):
    def test_rate_limited_floors_at_30(self):
        sleep = _compute_retry_sleep(0, base_s=1.0, rate_limited=True)
        self.assertGreaterEqual(sleep, 30.0)

    def test_rate_limited_grows_exponentially(self):
        s0 = _compute_retry_sleep(0, base_s=5.0, rate_limited=True)
        s1 = _compute_retry_sleep(1, base_s=5.0, rate_limited=True)
        self.assertGreater(s1, s0)

    def test_rate_limited_caps_at_120(self):
        sleep = _compute_retry_sleep(10, base_s=5.0, rate_limited=True)
        self.assertLessEqual(sleep, 125.0)  # 120 cap + up to 5 jitter

    def test_normal_error_grows_from_base(self):
        sleep = _compute_retry_sleep(0, base_s=10.0, rate_limited=False)
        self.assertGreaterEqual(sleep, 10.0)

    def test_normal_error_caps_at_60(self):
        sleep = _compute_retry_sleep(10, base_s=10.0, rate_limited=False)
        self.assertLessEqual(sleep, 61.0)  # 60 cap + up to 1 jitter


# ---------------------------------------------------------------------------
# _is_error_prediction
# ---------------------------------------------------------------------------

class TestIsErrorPrediction(unittest.TestCase):
    def test_sentinel_is_error(self):
        self.assertTrue(_is_error_prediction({"ranked_codes": ["000000"]}))

    def test_real_prediction_is_not_error(self):
        self.assertFalse(_is_error_prediction({"ranked_codes": ["854231", "854239", "854290", "854300"]}))

    def test_empty_codes_is_not_flagged(self):
        # Empty is ambiguous - don't accidentally clear it
        self.assertFalse(_is_error_prediction({"ranked_codes": []}))

    def test_multi_code_starting_with_000000_is_not_flagged(self):
        # Only the pure sentinel ["000000"] should match
        self.assertFalse(_is_error_prediction({"ranked_codes": ["000000", "854231"]}))


# ---------------------------------------------------------------------------
# _load_errored_frozen_ids
# ---------------------------------------------------------------------------

class TestLoadErroredFrozenIds(unittest.TestCase):
    def _write_calls_gz(self, path: Path, entries: list) -> None:
        with gzip.open(path, "wt", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

    def test_returns_empty_when_file_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = _load_errored_frozen_ids(Path(tmp) / "calls.jsonl.gz")
            self.assertEqual(result, set())

    def test_returns_errored_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls_path = Path(tmp) / "calls.jsonl.gz"
            self._write_calls_gz(calls_path, [
                {"frozen_id": "v2.0.eval.0001", "status": "error"},
                {"frozen_id": "v2.0.eval.0002", "status": "ok"},
                {"frozen_id": "v2.0.eval.0003", "status": "error"},
            ])
            result = _load_errored_frozen_ids(calls_path)
            self.assertEqual(result, {"v2.0.eval.0001", "v2.0.eval.0003"})

    def test_uses_last_status_per_frozen_id(self):
        # If a record appears twice (e.g. from a prior partial run), last entry wins
        with tempfile.TemporaryDirectory() as tmp:
            calls_path = Path(tmp) / "calls.jsonl.gz"
            self._write_calls_gz(calls_path, [
                {"frozen_id": "v2.0.eval.0001", "status": "error"},
                {"frozen_id": "v2.0.eval.0001", "status": "ok"},   # retry succeeded
            ])
            result = _load_errored_frozen_ids(calls_path)
            self.assertEqual(result, set())

    def test_ignores_malformed_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls_path = Path(tmp) / "calls.jsonl.gz"
            with gzip.open(calls_path, "wt", encoding="utf-8") as f:
                f.write("not json\n")
                f.write(json.dumps({"frozen_id": "v2.0.eval.0001", "status": "error"}) + "\n")
            result = _load_errored_frozen_ids(calls_path)
            self.assertEqual(result, {"v2.0.eval.0001"})


# ---------------------------------------------------------------------------
# rerun_errors integration: predictions cleared correctly
# ---------------------------------------------------------------------------

class TestRerunErrorsIntegration(unittest.TestCase):
    """Simulate the run_combo rerun_errors logic without invoking the full harness."""

    def _make_submission(self, path: Path, predictions: list) -> None:
        path.write_text(json.dumps({
            "submission": {"name": "test", "mode": "constrained", "tier": 1, "schema_version": "2.0.0"},
            "predictions": predictions,
        }), encoding="utf-8")

    def _write_calls_gz(self, path: Path, entries: list) -> None:
        with gzip.open(path, "wt", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

    def test_sentinel_predictions_cleared(self):
        with tempfile.TemporaryDirectory() as tmp:
            submission = Path(tmp) / "submission.json"
            calls = Path(tmp) / "calls.jsonl.gz"
            self._make_submission(submission, [
                {"frozen_id": "v2.0.eval.0001", "ranked_codes": ["854231", "854239"]},
                {"frozen_id": "v2.0.eval.0002", "ranked_codes": ["000000"]},  # sentinel
                {"frozen_id": "v2.0.eval.0003", "ranked_codes": ["000000"]},  # sentinel
            ])
            self._write_calls_gz(calls, [
                {"frozen_id": "v2.0.eval.0001", "status": "ok"},
                {"frozen_id": "v2.0.eval.0002", "status": "error"},
                {"frozen_id": "v2.0.eval.0003", "status": "error"},
            ])

            predictions_by_id = load_existing_predictions(submission)
            errored_ids = _load_errored_frozen_ids(calls)
            sentinel_ids = {fid for fid, p in predictions_by_id.items() if _is_error_prediction(p)}
            to_rerun = errored_ids | sentinel_ids
            for fid in to_rerun:
                predictions_by_id.pop(fid, None)

            self.assertIn("v2.0.eval.0001", predictions_by_id)
            self.assertNotIn("v2.0.eval.0002", predictions_by_id)
            self.assertNotIn("v2.0.eval.0003", predictions_by_id)

    def test_good_predictions_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            submission = Path(tmp) / "submission.json"
            calls = Path(tmp) / "calls.jsonl.gz"
            self._make_submission(submission, [
                {"frozen_id": "v2.0.eval.0001", "ranked_codes": ["854231", "854239"]},
                {"frozen_id": "v2.0.eval.0002", "ranked_codes": ["390110", "390120"]},
            ])
            self._write_calls_gz(calls, [
                {"frozen_id": "v2.0.eval.0001", "status": "ok"},
                {"frozen_id": "v2.0.eval.0002", "status": "ok"},
            ])

            predictions_by_id = load_existing_predictions(submission)
            errored_ids = _load_errored_frozen_ids(calls)
            sentinel_ids = {fid for fid, p in predictions_by_id.items() if _is_error_prediction(p)}
            to_rerun = errored_ids | sentinel_ids
            for fid in to_rerun:
                predictions_by_id.pop(fid, None)

            self.assertEqual(len(predictions_by_id), 2)

    def test_no_calls_log_falls_back_to_sentinel_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            submission = Path(tmp) / "submission.json"
            calls = Path(tmp) / "missing_calls.jsonl.gz"  # doesn't exist
            self._make_submission(submission, [
                {"frozen_id": "v2.0.eval.0001", "ranked_codes": ["854231"]},
                {"frozen_id": "v2.0.eval.0002", "ranked_codes": ["000000"]},
            ])

            predictions_by_id = load_existing_predictions(submission)
            errored_ids = _load_errored_frozen_ids(calls)  # returns set() - file missing
            sentinel_ids = {fid for fid, p in predictions_by_id.items() if _is_error_prediction(p)}
            to_rerun = errored_ids | sentinel_ids

            self.assertEqual(to_rerun, {"v2.0.eval.0002"})


# ---------------------------------------------------------------------------
# Circuit breaker logic (consecutive error counter)
# ---------------------------------------------------------------------------

class TestCircuitBreaker(unittest.TestCase):
    """Unit-test the consecutive-error / abort logic in isolation."""

    def _run_circuit_breaker(self, error_sequence: list, threshold: int) -> tuple:
        """Simulate write_result's circuit breaker for a list of bool error flags."""
        consecutive_errors = 0
        abort = False
        tripped_at = None
        for i, is_error in enumerate(error_sequence):
            if is_error:
                consecutive_errors += 1
                if consecutive_errors >= threshold:
                    abort = True
                    tripped_at = i
                    break
            else:
                consecutive_errors = 0
        return abort, tripped_at

    def test_trips_after_threshold_consecutive_errors(self):
        errors = [True, True, True, True, True]
        aborted, idx = self._run_circuit_breaker(errors, threshold=5)
        self.assertTrue(aborted)
        self.assertEqual(idx, 4)

    def test_does_not_trip_below_threshold(self):
        errors = [True, True, True, True, False]
        aborted, _ = self._run_circuit_breaker(errors, threshold=5)
        self.assertFalse(aborted)

    def test_reset_on_success_prevents_trip(self):
        # 4 errors, 1 success, 4 errors - should not trip with threshold=5
        errors = [True, True, True, True, False, True, True, True, True]
        aborted, _ = self._run_circuit_breaker(errors, threshold=5)
        self.assertFalse(aborted)

    def test_reset_on_success_then_new_run_trips(self):
        # 4 errors, 1 success, 5 errors - trips on the second run
        errors = [True, True, True, True, False, True, True, True, True, True]
        aborted, idx = self._run_circuit_breaker(errors, threshold=5)
        self.assertTrue(aborted)
        self.assertEqual(idx, 9)

    def test_threshold_of_1_trips_on_first_error(self):
        errors = [True]
        aborted, idx = self._run_circuit_breaker(errors, threshold=1)
        self.assertTrue(aborted)
        self.assertEqual(idx, 0)

    def test_all_successes_never_trips(self):
        errors = [False] * 20
        aborted, _ = self._run_circuit_breaker(errors, threshold=5)
        self.assertFalse(aborted)


if __name__ == "__main__":
    unittest.main()
