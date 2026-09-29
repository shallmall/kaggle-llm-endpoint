"""Offline tests for ktl_serve.push_and_watch (no network, no TPU).

Regression coverage for the `watch` bool/function name-shadowing bug that
made `launch.py serve` crash with "TypeError: 'bool' object is not callable"
right after a successful kernel push.
"""
import json
import tempfile
from pathlib import Path
from unittest import mock
import unittest

import ktl_serve


class PushAndWatchDispatchesWatchTest(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        self.state_file = self.root / "state.json"
        self.ktl = ktl_serve

    def tearDown(self):
        self._td.cleanup()

    def _run_push(self, **kw):
        """Drive the real push_and_watch with subprocess + watch mocked.

        Returns (state, watched_calls)."""
        pushed = mock.Mock(returncode=0, stdout="successfully pushed\n",
                           stderr="")
        with mock.patch.object(self.ktl, "STATE_FILE", self.state_file), \
             mock.patch("ktl_common.run_kaggle", return_value=pushed) as rk, \
             mock.patch.object(self.ktl, "watch") as wk:
            state = self.ktl.push_and_watch("qwen", "maxishallmall", **kw)
        return state, wk, rk

    def test_watch_progress_true_calls_watch_with_state(self):
        state, wk, rk = self._run_push(watch_progress=True)
        wk.assert_called_once()
        a, k = wk.call_args
        self.assertEqual(a[0], state["kernel"])
        self.assertEqual(a[1], state["topic"])
        self.assertEqual(a[2], None)  # relay
        self.assertEqual(a[3], "qwen")  # model_key
        self.assertFalse(k["stop_after_ready"])

    def test_watch_progress_false_skips_watch(self):
        state, wk, rk = self._run_push(watch_progress=False)
        wk.assert_not_called()

    def test_state_file_written_and_returned(self):
        state, wk, rk = self._run_push(watch_progress=False)
        saved = json.loads(self.state_file.read_text())
        for key in ("kernel", "topic", "api_key", "model"):
            self.assertEqual(saved[key], state[key], key)
        self.assertTrue(saved["kernel"].startswith("maxishallmall/"))

    def test_stop_after_ready_forwarded(self):
        _, wk, _ = self._run_push(watch_progress=True, stop_after_ready=True)
        self.assertTrue(wk.call_args.kwargs["stop_after_ready"])


if __name__ == "__main__":
    unittest.main()