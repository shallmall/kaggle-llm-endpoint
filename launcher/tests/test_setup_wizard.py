"""Offline integration tests for ktl_setup: the setup wizard and doctor.

External commands (npx/wrangler, kaggle, node) are faked with tiny shell
scripts on a temp PATH; KTL_HOME, HOME and Kaggle credentials live under a
temp dir. The TPU push is monkeypatched; the verify/doctor HTTP checks run
against a local mock server (no network).
"""
import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ktl_common  # noqa: E402
import ktl_env  # noqa: E402
import ktl_serve  # noqa: E402
import ktl_setup  # noqa: E402
from ktl_common import SubprocessError, load_config  # noqa: E402

from test_env_testmatrix import _Handler  # noqa: E402

IS_POSIX = os.name == "posix"


def make_args(**kw):
    base = dict(yes=True, dry_run=False, only=None, model=None,
                worker_name=None, adopt_worker=None, rotate=False,
                reset=False, verbose=False)
    base.update(kw)
    return argparse.Namespace(**base)


def fake_rows(r):
    rows = []
    for i, name in enumerate(["GET /v1/models", "POST chat", "POST stream",
                              "POST messages Bearer", "POST messages x-api-key",
                              "POST responses", "tools probe", "latency"]):
        rows.append({"name": name, "critical": i < 6, "ok": True,
                     "status": 200, "detail": "200 1 ms"})
    return rows


@unittest.skipUnless(IS_POSIX, "fake executables need a POSIX shell")
class WizardBase(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        root = Path(self._td.name)
        self.root = root
        self.bin = root / "bin"
        self.bin.mkdir()
        self.ktl_home = root / "ktl"
        self.home = root / "home"
        self.home.mkdir()
        (self.home / ".kaggle").mkdir()
        (self.home / ".kaggle" / "kaggle.json").write_text(json.dumps(
            {"username": "fakeuser", "key": "fake-key", "token": "fake-token"}))
        self.work = root / "work"
        self.record = root / "record"
        self.stdin_keys = root / "stdin_keys"
        self.whoami_file = root / "whoami_ok"

        self._make("node", 'if [ "$1" = "--version" ]; then echo "v22.3.7"; fi\n')
        self._make("npx", r"""#!/bin/sh
printf 'npx|%s\n' "$*" >> "${FAKE_RECORD:?}"
case "$*" in
  *whoami*)
    if [ "${FAKE_WHOAMI_TABLE:-0}" = "1" ]; then
      echo "┌────────────────────────────────────┬──────────────────────────────────┐"
      echo "│ Account Name                       │ Account ID                       │"
      echo "├────────────────────────────────────┼──────────────────────────────────┤"
      echo "│ Fake Account                       │ 1234567890abcdef1234567890abcdef │"
      echo "└────────────────────────────────────┴──────────────────────────────────┘"
      exit 0
    fi
    if [ -n "$CLOUDFLARE_API_TOKEN" ] && [ "${FAKE_TOKEN_INVALID:-0}" != "1" ]; then
      echo "Account Name: Token Account"
      echo "Logged in as: token@example.com"
      exit 0
    fi
    if [ ! -e "${FAKE_WHOAMI_FILE:?}" ]; then
      echo "You are not logged in. Run 'wrangler login' to log in." >&2
      exit 1
    fi
    echo "Account Name: Fake Account"
    echo "Logged in as: fake@example.com"
    ;;
  *deploy*)
    echo "Uploaded fake-worker (2.00 GB)"
    if [ "${FAKE_DEPLOY_NOURL:-0}" = "1" ]; then
      echo "(no URL line in this scenario)"
    else
      echo "Deployed fake-worker:"
      echo "  https://fake-worker.fakeacct.workers.dev"
    fi
    ;;
  *"secret put"*)
    key="$5"
    val="$(cat)"
    printf '%s=%s\n' "$key" "$val" >> "${FAKE_STDIN_KEYS:?}"
    echo "Success! The environment variable $key has been updated."
    ;;
  *"login --device"*)
    echo "Attempting to login via OAuth Device Authorization Grant..."
    echo "To authorize Wrangler, please visit:"
    echo "  https://dash.cloudflare.com/oauth2/device"
    echo "and enter the code: ABCDEF12"
    if [ "${FAKE_DEVICE_FAIL:-0}" = "1" ]; then
      echo "Error: Timed out waiting for device verification." >&2
      exit 1
    fi
    touch "${FAKE_WHOAMI_FILE:?}"
    echo "Successfully logged in."
    ;;
  *login*)
    if [ "${FAKE_LOGIN_FAIL:-0}" = "1" ]; then
      echo "Timed out waiting for authorization code, please try again." >&2
      exit 1
    fi
    touch "${FAKE_WHOAMI_FILE:?}"
    echo "Logged in to Fake Account."
    ;;
  *)
    echo "fake npx ok"
    ;;
esac
""")
        self._make("kaggle", r"""#!/bin/sh
printf 'kaggle|%s\n' "$*" >> "${FAKE_RECORD:?}"
case "$*" in
  *--version*)
    echo "Kaggle CLI 9.9.9 (fake)"
    ;;
  *"kernels list"*)
    if [ "${FAKE_KAGGLE_FAIL:-0}" = "1" ]; then
      echo "Error: Not authenticated." >&2
      exit 1
    fi
    echo "kernel-ref"
    ;;
  *"config view"*)
    echo "username: fakeuser"
    ;;
  *"auth login"*)
    echo "Logged in."
    ;;
  *)
    echo "ok"
    ;;
esac
""")
        self._make("codex", 'echo "fake-codex"\n')

        env = {
            "PATH": str(self.bin) + os.pathsep + os.environ.get("PATH", ""),
            "KTL_HOME": str(self.ktl_home),
            "HOME": str(self.home),
            "FAKE_RECORD": str(self.record),
            "FAKE_STDIN_KEYS": str(self.stdin_keys),
            "FAKE_WHOAMI_FILE": str(self.whoami_file),
            "FAKE_KAGGLE_FAIL": "0",
            "FAKE_TOKEN_INVALID": "0",
            "FAKE_LOGIN_FAIL": "0",
            "FAKE_DEVICE_FAIL": "0",
            "FAKE_WHOAMI_TABLE": "0",
            "KTL_RELAY_URL": "",
            "KTL_CLIENT_API_KEY": "",
            "KTL_API_KEY": "",
            "CLOUDFLARE_API_TOKEN": "",
            "CLOUDFLARE_ACCOUNT_ID": "",
        }
        self._p = mock.patch.dict(os.environ, env)
        self._p.start()
        self._ss = mock.patch.object(
            ktl_env, "_SERVE_STATE_FILE",
            root / "home" / ".kaggle-tpu-lab.json")
        self._ss.start()

    def tearDown(self):
        self._ss.stop()
        self._p.stop()
        self._td.cleanup()

    def _make(self, name, body):
        p = self.bin / name
        p.write_text(body if body.startswith("#!") else "#!/bin/sh\n" + body)
        p.chmod(0o755)
        return p

    # -- recording helpers -------------------------------------------------
    def record_lines(self, prefix=""):
        if not self.record.exists():
            return []
        lines = self.record.read_text().splitlines()
        return [ln for ln in lines if prefix == "" or ln.startswith(prefix)]

    def stdin_secret_map(self):
        out = {}
        if self.stdin_keys.exists():
            for ln in self.stdin_keys.read_text().splitlines():
                if "=" in ln:
                    k, v = ln.split("=", 1)
                    out[k] = v
        return out

    def patch_serve_and_matrix(self, push=mock.DEFAULT):
        if push is mock.DEFAULT:
            push = mock.Mock(return_value=None)
        p1 = mock.patch.object(ktl_serve, "push_and_watch", push)
        p2 = mock.patch.object(ktl_env, "run_matrix", side_effect=fake_rows)
        p1.start()
        p2.start()
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)
        return push

    def run_setup(self, **kw):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = ktl_setup.cmd_setup(make_args(**kw))
        return rc, buf.getvalue()


class TestDryRun(WizardBase):
    def test_dry_run_changes_nothing(self):
        self.patch_serve_and_matrix()
        rc, out = self.run_setup(dry_run=True)
        self.assertEqual(rc, 0)
        self.assertIn("DRY RUN", out)
        # no state written
        self.assertFalse((self.ktl_home / "config.json").exists())
        self.assertFalse((self.ktl_home / "setup.log").exists())
        # no wrangler calls, no kaggle mutations
        self.assertEqual(self.record_lines("npx|"), [])
        for ln in self.record_lines("kaggle|"):
            self.assertNotIn("auth login", ln)
            self.assertNotIn("kernels list", ln)
        # no client config written under the temp HOME
        self.assertFalse((self.home / ".codex").exists())

    def test_dry_run_reset_keeps_config(self):
        cfg = load_config(migrate=False)
        cfg["secrets"]["client_api_key"] = "x" * 43
        ktl_common.save_config(cfg)
        rc, _ = self.run_setup(dry_run=True, reset=True)
        self.assertEqual(rc, 0)
        self.assertEqual(
            load_config(migrate=False)["secrets"]["client_api_key"], "x" * 43)


class TestCloudflareTokenAuth(WizardBase):
    """setup accepts a CLOUDFLARE_API_TOKEN env var (non-interactive); the only
    interactive login is `wrangler login --device` — the OAuth 2.0 device flow
    (RFC 8628) that needs no localhost callback server, so it works identically
    on a desktop, over SSH, or on any headless/VM machine. No SSH detection,
    no plain-flow branch."""

    def test_good_token_skips_browser_login(self):
        # no whoami_file.touch(): a CLOUDFLARE_API_TOKEN env var must work
        # WITHOUT any OAuth login
        self.patch_serve_and_matrix()
        with mock.patch.dict(os.environ,
                             {"CLOUDFLARE_API_TOKEN": "tok_123",
                              "CLOUDFLARE_ACCOUNT_ID": "acct_1"}):
            rc, out = self.run_setup()
        self.assertEqual(rc, 0, out)
        for ln in self.record_lines("npx|"):
            self.assertNotIn("login", ln)

    def test_bad_token_fails_with_clear_message(self):
        self.patch_serve_and_matrix()
        with mock.patch.dict(os.environ,
                             {"CLOUDFLARE_API_TOKEN": "tok_123",
                              "CLOUDFLARE_ACCOUNT_ID": "acct_1",
                              "FAKE_TOKEN_INVALID": "1"}):
            rc, out = self.run_setup()
        self.assertEqual(rc, 1)
        self.assertIn("CLOUDFLARE_API_TOKEN is set", out)
        for ln in self.record_lines("npx|"):
            self.assertNotIn("login", ln)

    def test_device_login_is_used_everywhere(self):
        """Even on a plain desktop with a display and no SSH, the wizard logs
        in with `wrangler login --device` — no detection, no plain login."""
        self.patch_serve_and_matrix()
        with mock.patch.dict(os.environ,
                             {"DISPLAY": ":0", "WAYLAND_DISPLAY": ":0",
                              "SSH_CONNECTION": "", "SSH_CLIENT": ""}):
            rc, out = self.run_setup(only="cloudflare")
        self.assertEqual(rc, 0, out)
        self.assertIn("using the DEVICE login (OAuth 2.0 device flow)", out)
        self.assertIn("https://dash.cloudflare.com/oauth2/device", out)
        self.assertIn("logged in as Fake Account", out)
        lines = "|".join(self.record_lines())
        self.assertIn("login --device", lines)
        plain = [ln for ln in self.record_lines()
                 if "wrangler@4 login" in ln and "wrangler@4 login --device" not in ln]
        self.assertEqual(plain, [], lines)

    def test_device_login_fails_with_clear_message(self):
        self.patch_serve_and_matrix()
        with mock.patch.dict(os.environ, {"FAKE_DEVICE_FAIL": "1"}):
            rc, out = self.run_setup(only="cloudflare")
        self.assertEqual(rc, 1)
        self.assertIn("device login did not complete", out)

    def test_whoami_table_format_is_recognized(self):
        """wrangler 4.x prints the account as a box-drawing table (not the old
        `Account Name:` line), even to a pipe: an already-logged-in account
        must be detected from that output without any login."""
        self.patch_serve_and_matrix()
        with mock.patch.dict(os.environ, {"FAKE_WHOAMI_TABLE": "1"}):
            rc, out = self.run_setup(only="cloudflare")
        self.assertEqual(rc, 0, out)
        self.assertIn("logged in as Fake Account", out)
        for ln in self.record_lines("npx|"):
            self.assertNotIn("login", ln)


class TestParseWhoami(unittest.TestCase):
    def test_plain_line(self):
        self.assertEqual(
            ktl_setup._parse_whoami("Account Name: Fake Account\n"
                                    "Logged in as: fake@example.com"),
            "Fake Account")

    def test_table_format(self):
        out = ("┌──────────────────┬──────────────────────────────────┐\n"
               "│ Account Name     │ Account ID                       │\n"
               "├──────────────────┼──────────────────────────────────┤\n"
               "│ Example Account  │ 1234567890abcdef1234567890abcdef │\n"
               "└──────────────────┴──────────────────────────────────┘\n")
        self.assertEqual(
            ktl_setup._parse_whoami(out),
            "Example Account")

    def test_logged_in_as_fallback(self):
        self.assertEqual(
            ktl_setup._parse_whoami("Logged in as: someone@example.com"),
            "someone@example.com")
        self.assertEqual(ktl_setup._parse_whoami("not logged in"), "")


class TestHappyPath(WizardBase):
    def test_full_run(self):
        self.whoami_file.touch()
        push = self.patch_serve_and_matrix()
        rc, out = self.run_setup()
        self.assertEqual(rc, 0, out)

        cfg = load_config(migrate=False)
        self.assertEqual(set(cfg["steps"]), set(ktl_common.STEP_NAMES))
        for step in ktl_setup.WIZARD_STEPS:
            self.assertEqual(cfg["steps"][step], "done", f"step {step}: {out}")
        # serve is no longer part of the wizard — it stays untried until the
        # user explicitly boots a TPU with `python launch.py serve`
        self.assertEqual(cfg["steps"]["serve"], "pending")
        self.assertEqual(cfg["model"], "qwen")
        self.assertEqual(cfg["kaggle"]["username"], "fakeuser")
        self.assertEqual(cfg["cloudflare"]["relay_url"],
                         "https://fake-worker.fakeacct.workers.dev")
        self.assertEqual(cfg["cloudflare"]["worker_name"], "kaggle-tpu-relay")
        self.assertEqual(len(cfg["secrets"]["client_api_key"]), 43)
        self.assertEqual(len(cfg["secrets"]["update_secret"]), 43)

        # config-only wizard: never boots a TPU and never runs the matrix
        push.assert_not_called()
        self.assertNotIn("start the TPU session", out)
        self.assertNotIn("compatibility matrix", out)

        # the summary prints the endpoint URL (paired with the key) so clients
        # know exactly where to talk to and what to send
        self.assertIn("endpoint   : https://fake-worker.fakeacct.workers.dev", out)
        self.assertIn("clients use https://fake-worker.fakeacct.workers.dev/v1", out)
        self.assertIn(f"key        : {ktl_common.mask(cfg['secrets']['client_api_key'])}", out)

        # secrets went to wrangler on STDIN, never argv
        secrets_map = self.stdin_secret_map()
        self.assertEqual(secrets_map.get("CLIENT_API_KEY"),
                         cfg["secrets"]["client_api_key"])
        self.assertEqual(secrets_map.get("UPDATE_SECRET"),
                         cfg["secrets"]["update_secret"])
        for ln in self.record_lines("npx|"):
            self.assertNotIn(cfg["secrets"]["client_api_key"], ln)
            self.assertNotIn(cfg["secrets"]["update_secret"], ln)

        # log written, secrets not in it
        log = (self.ktl_home / "setup.log")
        self.assertTrue(log.exists())
        logtext = log.read_text()
        self.assertNotIn(cfg["secrets"]["client_api_key"], logtext)
        self.assertNotIn(cfg["secrets"]["update_secret"], logtext)
        self.assertIn("auto-approved", logtext)

        # client config written (codex was the only detected client)
        toml = (self.home / ".codex" / "config.toml").read_text()
        self.assertIn("kaggle-tpu", toml)
        self.assertIn("fake-worker.fakeacct.workers.dev", toml)
        self.assertIn("KTL_CLIENT_API_KEY", toml)
        self.assertNotIn(cfg["secrets"]["client_api_key"], toml)

    def test_model_flag_is_honoured(self):
        self.whoami_file.touch()
        self.patch_serve_and_matrix()
        rc, out = self.run_setup(model="glm")
        self.assertEqual(rc, 0, out)
        self.assertEqual(load_config(migrate=False)["model"], "glm")

    def test_rerun_all_done_changes_nothing(self):
        self.whoami_file.touch()
        self.patch_serve_and_matrix()
        rc, _ = self.run_setup()
        self.assertEqual(rc, 0)
        before = ktl_common.config_path().read_bytes()
        rc, out = self.run_setup()
        self.assertEqual(rc, 0, out)
        self.assertEqual(ktl_common.config_path().read_bytes(), before)
        for step in ktl_setup.WIZARD_STEPS:
            self.assertIn(f"skip  : {step}", out)
        self.assertEqual(out.count("  skip  : "), len(ktl_setup.WIZARD_STEPS))


class TestGates(WizardBase):
    def test_gate_blocks_without_yes(self):
        self.whoami_file.touch()
        self.patch_serve_and_matrix()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), \
                mock.patch("builtins.input", side_effect=["n"]):
            rc = ktl_setup.cmd_setup(make_args(yes=False))
        self.assertEqual(rc, 1)
        cfg = load_config(migrate=False)
        for step in ("prereqs", "kaggle", "cloudflare", "secrets"):
            self.assertEqual(cfg["steps"][step], "done")
        self.assertEqual(cfg["steps"]["worker"], "failed")
        for step in ("serve", "clients"):
            self.assertEqual(cfg["steps"][step], "pending")
        self.assertEqual([ln for ln in self.record_lines("npx|")
                          if "deploy" in ln], [])

    def test_worker_url_paste_fallback(self):
        self.whoami_file.touch()
        self.patch_serve_and_matrix()
        cfg = ktl_common.load_config(migrate=False)
        cfg["secrets"]["client_api_key"] = "c" * 43
        cfg["secrets"]["update_secret"] = "u" * 43
        ktl_common.save_config(cfg)
        with mock.patch.dict(os.environ, {"FAKE_DEPLOY_NOURL": "1"}):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), \
                    mock.patch("builtins.input",
                               side_effect=["y",
                                            "https://pasted.acct.workers.dev",
                                            "y"]):
                rc = ktl_setup.cmd_setup(make_args(only="worker", yes=False))
        self.assertEqual(rc, 0, buf.getvalue())
        cfg = load_config(migrate=False)
        self.assertEqual(cfg["cloudflare"]["relay_url"],
                         "https://pasted.acct.workers.dev")
        self.assertEqual(cfg["steps"]["worker"], "done")

    def test_eof_proceeds_with_gate_defaults_but_no_clients(self):
        """Closed stdin: confirms fall back to their Y/n default (gates pass,
        matching `--yes` semantics), while choose() picks nothing — so the
        wizard completes but writes no client config."""
        self.whoami_file.touch()
        self.patch_serve_and_matrix()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), \
                mock.patch("builtins.input", side_effect=EOFError):
            rc = ktl_setup.cmd_setup(make_args(yes=False))
        self.assertEqual(rc, 0, buf.getvalue())
        cfg = load_config(migrate=False)
        self.assertEqual(cfg["steps"]["worker"], "done")
        self.assertEqual(cfg["steps"]["serve"], "pending")
        self.assertFalse((self.home / ".codex").exists())


class TestResume(WizardBase):
    def test_failed_serve_is_retried_via_only(self):
        """serve is not part of the config-only wizard, but `--only serve`
        still boots a TPU and keeps the resume/failure semantics."""
        self.whoami_file.touch()
        # run 1: serve fails
        boom = mock.Mock(side_effect=SubprocessError("fake push failed", "hint"))
        self.patch_serve_and_matrix(push=boom)
        rc, out1 = self.run_setup(only="serve")
        self.assertEqual(rc, 1)
        cfg = load_config(migrate=False)
        self.assertEqual(cfg["steps"]["serve"], "failed")

        # run 2: everything green
        self.patch_serve_and_matrix()  # replace the patch with a good one
        rc, out2 = self.run_setup(only="serve")
        self.assertEqual(rc, 0, out2)
        self.assertEqual(load_config(migrate=False)["steps"]["serve"], "done")


class TestAdoptAndRotate(WizardBase):
    def test_adopt_worker_skips_deploy(self):
        self.whoami_file.touch()
        self.patch_serve_and_matrix()
        import ktl_common
        cfg = ktl_common.load_config(migrate=False)
        cfg["secrets"]["client_api_key"] = "c" * 43
        cfg["secrets"]["update_secret"] = "u" * 43
        ktl_common.save_config(cfg)
        rc, out = self.run_setup(only="worker",
                                 adopt_worker="https://existing.acct.workers.dev/")
        self.assertEqual(rc, 0, out)
        cfg = load_config(migrate=False)
        self.assertEqual(cfg["cloudflare"]["relay_url"],
                         "https://existing.acct.workers.dev")
        self.assertEqual([ln for ln in self.record_lines("npx|")
                          if "deploy" in ln], [])
        self.assertEqual(set(self.stdin_secret_map()),
                         {"CLIENT_API_KEY", "UPDATE_SECRET"})

    def test_adopt_rejects_non_workers_url(self):
        self.whoami_file.touch()
        self.patch_serve_and_matrix()
        cfg = ktl_common.load_config(migrate=False)
        cfg["secrets"]["client_api_key"] = "c" * 43
        cfg["secrets"]["update_secret"] = "u" * 43
        ktl_common.save_config(cfg)
        rc, out = self.run_setup(only="worker", adopt_worker="http://example.com")
        self.assertEqual(rc, 1)
        # the refusal message must be printed, not swallowed by cmd_setup
        self.assertIn("does not look like a workers.dev URL", out)
        self.assertEqual([ln for ln in self.record_lines("npx|")
                          if "secret" in ln], [])

    def test_worker_without_secrets_is_guarded(self):
        """`setup --only worker` on a fresh machine must not deploy / put empty
        secrets — it should fail fast with a pointer to the secrets step."""
        self.whoami_file.touch()
        self.patch_serve_and_matrix()
        rc, out = self.run_setup(only="worker")
        self.assertEqual(rc, 1)
        self.assertIn("no relay secrets generated yet", out)
        self.assertEqual([ln for ln in self.record_lines("npx|")
                          if "deploy" in ln], [])
        self.assertEqual([ln for ln in self.record_lines("npx|")
                          if "secret put" in ln], [])

    def test_rotate_reputs_secrets_when_worker_done(self):
        self.whoami_file.touch()
        self.patch_serve_and_matrix()
        rc, _ = self.run_setup()
        self.assertEqual(rc, 0)
        old = load_config(migrate=False)["secrets"]
        # mark where run 1's puts ended so we can isolate run 2's
        run1_put_count = len([ln for ln in self.record_lines("npx|")
                              if "secret put" in ln])
        self.assertEqual(run1_put_count, 2)
        rc, out = self.run_setup(only="secrets", rotate=True)
        self.assertEqual(rc, 0, out)
        new = load_config(migrate=False)["secrets"]
        self.assertNotEqual(new["client_api_key"], old["client_api_key"])
        self.assertNotEqual(new["update_secret"], old["update_secret"])
        # exactly two fresh puts (one per secret) in the rotate run
        puts = [ln for ln in self.record_lines("npx|") if "secret put" in ln]
        self.assertEqual(len(puts), run1_put_count + 2)
        # the values that hit wrangler stdin are the NEW ones
        secret_map = self.stdin_secret_map()
        self.assertEqual(secret_map["CLIENT_API_KEY"], new["client_api_key"])
        self.assertEqual(secret_map["UPDATE_SECRET"], new["update_secret"])
        # old values never printed to stdout
        self.assertNotIn(old["client_api_key"], out)
        self.assertNotIn(old["update_secret"], out)


class TestReset(WizardBase):
    def test_reset_wipes_config_and_restarts(self):
        import ktl_common
        cfg = ktl_common.load_config(migrate=False)
        cfg["secrets"]["client_api_key"] = "z" * 43
        ktl_common.save_config(cfg)
        self.whoami_file.touch()
        self.patch_serve_and_matrix()
        rc, out = self.run_setup(reset=True)
        self.assertEqual(rc, 0, out)
        self.assertIn("deleted", out)
        cfg = load_config(migrate=False)
        self.assertNotEqual(cfg["secrets"]["client_api_key"], "z" * 43)
        self.assertEqual(cfg["steps"]["prereqs"], "done")


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------

class DoctorBase(WizardBase):
    def _tree(self):
        """Snapshot (path, mtime_ns, size) for every file under HOME and KTL_HOME."""
        out = {}
        for base in (self.home, self.ktl_home):
            if not base.exists():
                continue
            for p in base.rglob("*"):
                if p.is_file():
                    st = p.stat()
                    out[str(p)] = (st.st_mtime_ns, st.st_size)
        return out

    def start_mock(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        import threading
        self._t = threading.Thread(target=self.server.serve_forever, daemon=True)
        self._t.start()
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.server.server_close)
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def run_doctor(self, **kw):
        kw.setdefault("json", False)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = ktl_setup.cmd_doctor(argparse.Namespace(**kw))
        return rc, buf.getvalue()


class TestDoctor(DoctorBase):
    def test_no_config_unhealthy(self):
        rc, out = self.run_doctor()
        self.assertEqual(rc, 1)
        self.assertIn("problem(s)", out)
        self.assertIn("python launch.py setup", out)
        self.assertIn("no endpoint", out)
        # zero side effects
        self.assertFalse((self.ktl_home / "config.json").exists())
        self.assertFalse((self.ktl_home / "setup.log").exists())

    def test_healthy_with_mock_endpoint(self):
        self.whoami_file.touch()
        url = self.start_mock()
        cfg = ktl_common.load_config(migrate=False)
        cfg["cloudflare"]["relay_url"] = url
        cfg["secrets"]["client_api_key"] = "d" * 43
        cfg["secrets"]["update_secret"] = "u" * 43
        for s in ktl_common.STEP_NAMES:
            ktl_common.set_step(cfg, s, "done")
        ktl_common.save_config(cfg)
        before = self._tree()
        rc, out = self.run_doctor()
        self.assertEqual(rc, 0, out)
        self.assertIn("RESULT: healthy", out)
        self.assertIn("GET /v1/models -> 200", out)
        self.assertIn("all critical routes pass", out)
        # zero side effects: the whole HOME + KTL_HOME tree is untouched
        self.assertEqual(self._tree(), before)
        self.assertFalse((self.ktl_home / "setup.log").exists())
        # no secret value leaked into the report
        self.assertNotIn("d" * 43, out)

    def test_unreachable_relay(self):
        self.whoami_file.touch()
        cfg = ktl_common.load_config(migrate=False)
        cfg["cloudflare"]["relay_url"] = "http://127.0.0.1:1"
        cfg["secrets"]["client_api_key"] = "d" * 43
        ktl_common.save_config(cfg)
        rc, out = self.run_doctor()
        self.assertEqual(rc, 1)
        self.assertIn("unreachable", out)
        self.assertIn("how to fix", out)

    def test_json_output(self):
        self.whoami_file.touch()
        url = self.start_mock()
        cfg = ktl_common.load_config(migrate=False)
        cfg["cloudflare"]["relay_url"] = url
        cfg["secrets"]["client_api_key"] = "d" * 43
        ktl_common.save_config(cfg)
        rc, out = self.run_doctor(json=True)
        self.assertEqual(rc, 0, out)
        start = out.index("\n{") + 1
        doc = json.loads(out[start:])
        self.assertTrue(doc["healthy"])
        self.assertIn("checks", doc)
        self.assertEqual(len(doc["matrix"]), 8)
        # JSON must be parseable on its own: nothing but the blob after \n{
        self.assertNotIn("PASS", out[start:])


if __name__ == "__main__":
    unittest.main()
