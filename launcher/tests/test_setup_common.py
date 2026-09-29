"""Offline unit tests for ktl_common: config, keys, redaction, prompts, tools.

No network. External commands are exercised through tiny fake executables on
a temporary PATH (POSIX only — skipped elsewhere).
"""
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ktl_common as C  # noqa: E402
import ktl_env  # noqa: E402

IS_POSIX = os.name == "posix"


# ---------------------------------------------------------------------------
# config file
# ---------------------------------------------------------------------------

class ConfigBase(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.ktl_home = Path(self._td.name) / "ktl"
        self._p = mock.patch.dict(os.environ, {"KTL_HOME": str(self.ktl_home)})
        self._p.start()

    def tearDown(self):
        self._p.stop()
        self._td.cleanup()


class TestConfigPaths(ConfigBase):
    def test_ktl_home_prefers_env_override(self):
        self.assertEqual(C.ktl_home(), self.ktl_home)
        self.assertEqual(C.config_path(), self.ktl_home / "config.json")
        self.assertEqual(C.legacy_state_path(), self.ktl_home / "state.json")
        self.assertEqual(C.log_path(), self.ktl_home / "setup.log")

    def test_load_defaults_when_absent(self):
        cfg = C.load_config(migrate=False)
        self.assertEqual(cfg["version"], C.CONFIG_VERSION)
        self.assertEqual(cfg["model"], "qwen")
        self.assertEqual(set(cfg["steps"]), set(C.STEP_NAMES))
        self.assertTrue(all(v == "pending" for v in cfg["steps"].values()))
        self.assertFalse(C.config_path().exists())  # load must not create the file

    def test_load_corrupt_returns_defaults(self):
        self.ktl_home.mkdir(parents=True, exist_ok=True)
        (self.ktl_home / "config.json").write_text("{not json")
        cfg = C.load_config(migrate=False)
        self.assertEqual(cfg["model"], "qwen")

    def test_save_is_atomic_0600_and_roundtrips(self):
        cfg = C.load_config(migrate=False)
        cfg["model"] = "glm"
        cfg["secrets"]["client_api_key"] = "k" * 43
        C.save_config(cfg)
        p = C.config_path()
        self.assertTrue(p.exists())
        if IS_POSIX:
            self.assertEqual(stat.S_IMODE(p.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(p.parent.stat().st_mode), 0o700)
        back = C.load_config(migrate=False)
        self.assertEqual(back["model"], "glm")
        self.assertEqual(back["secrets"]["client_api_key"], "k" * 43)
        self.assertTrue(back["updated_at"])

    def test_save_replaces_no_partial_file(self):
        cfg = C.load_config(migrate=False)
        C.save_config(cfg)
        cfg["model"] = "glm"
        C.save_config(cfg)
        self.assertEqual(json.loads(C.config_path().read_text())["model"], "glm")
        # no temp litter
        leftovers = [f for f in self.ktl_home.iterdir() if f.name.startswith("config.json.tmp")]
        self.assertEqual(leftovers, [])

    def test_update_config_toplevel(self):
        C.update_config(model="glm")
        self.assertEqual(C.load_config(migrate=False)["model"], "glm")

    def test_set_step_and_status(self):
        cfg = C.load_config(migrate=False)
        C.set_step(cfg, "prereqs", "done")
        self.assertEqual(C.step_status(cfg, "prereqs"), "done")
        self.assertEqual(C.step_status(cfg, "serve"), "pending")
        C.set_step(cfg, "serve", "failed")
        self.assertEqual(C.step_status(cfg, "serve"), "failed")


class TestLegacyMigration(ConfigBase):
    def test_migration_copies_model_and_relay_url(self):
        self.ktl_home.mkdir(parents=True, exist_ok=True)
        (self.ktl_home / "state.json").write_text(json.dumps(
            {"base_url": "https://old-relay.workers.dev", "mode": "relay",
             "model": "glm"}))
        cfg = C.load_config()  # migrate=True default
        self.assertTrue(C.config_path().exists())
        self.assertEqual(cfg["model"], "glm")
        self.assertEqual(cfg["cloudflare"]["relay_url"], "https://old-relay.workers.dev")
        # legacy file left untouched
        self.assertEqual(json.loads((self.ktl_home / "state.json").read_text())["mode"], "relay")

    def test_migration_skips_direct_mode_url(self):
        self.ktl_home.mkdir(parents=True, exist_ok=True)
        (self.ktl_home / "state.json").write_text(json.dumps(
            {"base_url": "https://tmp.trycloudflare.com", "mode": "direct",
             "model": "qwen"}))
        cfg = C.load_config()
        self.assertEqual(cfg["cloudflare"]["relay_url"], "")

    def test_migration_never_overwrites_existing_config(self):
        self.ktl_home.mkdir(parents=True, exist_ok=True)
        (self.ktl_home / "config.json").write_text(json.dumps(
            {"version": 1, "model": "glm", "cloudflare": {"relay_url": "https://a.b"}}))
        (self.ktl_home / "state.json").write_text(json.dumps(
            {"base_url": "https://other.workers.dev", "mode": "relay", "model": "qwen"}))
        cfg = C.load_config()
        self.assertEqual(cfg["cloudflare"]["relay_url"], "https://a.b")
        self.assertEqual(cfg["model"], "glm")


# ---------------------------------------------------------------------------
# Cloudflare credential file (cloudflare.env)
# ---------------------------------------------------------------------------

class TestCloudflareEnvFile(ConfigBase):
    def test_save_roundtrip_0600(self):
        p = C.save_cloudflare_env({"token": "tok_abc", "account_id": "acct_1"})
        self.assertEqual(p, C.cloudflare_env_path())
        self.assertTrue(p.exists())
        if IS_POSIX:
            self.assertEqual(stat.S_IMODE(p.stat().st_mode), 0o600)
        self.assertEqual(C.load_cloudflare_env(),
                         {"token": "tok_abc", "account_id": "acct_1"})

    def test_save_requires_token(self):
        with self.assertRaises(ValueError):
            C.save_cloudflare_env({"account_id": "acct_1"})

    def test_load_missing_and_tokenless_returns_empty(self):
        self.assertEqual(C.load_cloudflare_env(), {})
        self.ktl_home.mkdir(parents=True, exist_ok=True)
        (self.ktl_home / "cloudflare.env").write_text(
            "CLOUDFLARE_ACCOUNT_ID=acct_1\n")
        self.assertEqual(C.load_cloudflare_env(), {})

    def test_load_corrupt_returns_empty(self):
        self.ktl_home.mkdir(parents=True, exist_ok=True)
        (self.ktl_home / "cloudflare.env").write_text("\x00\x01garbage")
        self.assertEqual(C.load_cloudflare_env(), {})

    def test_load_skips_blank_and_unrecognized_lines(self):
        self.ktl_home.mkdir(parents=True, exist_ok=True)
        (self.ktl_home / "cloudflare.env").write_text(
            "\nCLOUDFLARE_API_TOKEN= tok_x \nFOO=bar\n")
        self.assertEqual(C.load_cloudflare_env(), {"token": "tok_x"})

    def test_apply_sets_env_but_real_env_wins(self):
        C.apply_cloudflare_env({"token": "tok_a", "account_id": "acct_a"})
        self.assertEqual(os.environ.get("CLOUDFLARE_API_TOKEN"), "tok_a")
        self.assertEqual(os.environ.get("CLOUDFLARE_ACCOUNT_ID"), "acct_a")
        os.environ["CLOUDFLARE_API_TOKEN"] = "tok_real"
        self.assertTrue(C.apply_cloudflare_env({"token": "tok_b"}))
        self.assertEqual(os.environ.get("CLOUDFLARE_API_TOKEN"), "tok_real")
        os.environ.pop("CLOUDFLARE_API_TOKEN", None)
        os.environ.pop("CLOUDFLARE_ACCOUNT_ID", None)

    def test_apply_no_token_returns_false(self):
        self.assertFalse(C.apply_cloudflare_env({}))


# ---------------------------------------------------------------------------
# keys + redaction
# ---------------------------------------------------------------------------

class TestKeysAndRedaction(unittest.TestCase):
    def test_gen_key_shape(self):
        keys = {C.gen_key() for _ in range(50)}
        self.assertEqual(len(keys), 50)
        for k in keys:
            self.assertEqual(len(k), 43)  # 32 bytes urlsafe base64
            self.assertTrue(k.isascii())

    def test_mask(self):
        self.assertEqual(C.mask(""), "(none)")
        self.assertEqual(C.mask("abcdefg"), "*******")
        v = "a" * 20 + "z" * 23
        self.assertEqual(C.mask(v), "aaaa…zzzz")

    def test_redact_name_equals_value(self):
        self.assertEqual(C.redact("CLIENT_API_KEY=abcdef123456\n"),
                         "CLIENT_API_KEY=abcd…3456\n")
        self.assertEqual(C.redact("export KTL_API_KEY=0123456789abcdef\n"),
                         "export KTL_API_KEY=0123…cdef\n")
        # multi-line: every secret line masked
        out = C.redact("UPDATE_SECRET=aaaaaaaaaaaaaaaa\nOTHER=x\nMY_TOKEN=bbbbbbbbbbbbbbbb\n")
        self.assertNotIn("aaaaaaaaaaaaaaaa", out)
        self.assertNotIn("bbbbbbbbbbbbbbbb", out)
        self.assertIn("OTHER=x", out)

    def test_redact_leaves_placeholder_and_nonsecrets(self):
        self.assertEqual(C.redact("<YOUR_CLIENT_API_KEY>"),
                         ktl_env.PLACEHOLDER)
        self.assertEqual(C.redact("model=qwen base_url=https://x.dev"),
                         "model=qwen base_url=https://x.dev")

    def test_client_key_from_config_precedence(self):
        cfg = {"secrets": {"client_api_key": "c" * 43}}
        self.assertEqual(C.client_key_from_config(cfg, "")[0], "c" * 43)
        self.assertEqual(C.client_key_from_config(cfg, "")[1], "config")
        env = "e" * 43
        self.assertEqual(C.client_key_from_config(cfg, env)[0], env)
        self.assertEqual(C.client_key_from_config(cfg, env)[1], "env")
        key, src = C.client_key_from_config({"secrets": {}}, "")
        self.assertEqual(key, ktl_env.PLACEHOLDER)
        self.assertEqual(src, "placeholder")

    def test_resolve_precedence(self):
        self.assertEqual(C.resolve("cli", "env", "cfg", "def"), "cli")
        self.assertEqual(C.resolve(None, "env", "cfg", "def"), "env")
        self.assertEqual(C.resolve(None, None, "cfg", "def"), "cfg")
        self.assertEqual(C.resolve(None, None, None, "def"), "def")
        self.assertEqual(C.resolve(None, None, "", "def"), "def")  # empty string = unset


# ---------------------------------------------------------------------------
# subprocess helper
# ---------------------------------------------------------------------------

@unittest.skipUnless(IS_POSIX, "needs POSIX shell for fake executables")
class FakeBinBase(unittest.TestCase):
    """Base that puts a temp bin dir (with fake executables) first on PATH."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.bin = Path(self._td.name) / "bin"
        self.bin.mkdir()
        self.work = Path(self._td.name) / "work"
        self.work.mkdir()

    def tearDown(self):
        self._td.cleanup()

    def make_bin(self, name, script):
        p = self.bin / name
        p.write_text("#!/bin/sh\n" + script)
        p.chmod(0o755)
        return p

    def env_with_bin(self, **extra):
        env = {
            "PATH": str(self.bin) + os.pathsep + os.environ.get("PATH", ""),
        }
        env.update(extra)
        return mock.patch.dict(os.environ, env)

    def env_only_bin(self):
        """PATH restricted to the (possibly empty) temp bin: real tools absent."""
        return mock.patch.dict(os.environ, {"PATH": str(self.bin)})


class TestToolDiscovery(FakeBinBase):
    def test_node_version_parses_fake(self):
        self.make_bin("node", 'echo "v22.3.7"\n')
        with self.env_with_bin():
            self.assertEqual(C.node_version(), (22, 3, 7))

    def test_node_version_bad_output_is_none(self):
        self.make_bin("node", 'echo "garbage"\n')
        with self.env_with_bin():
            self.assertIsNone(C.node_version())

    def test_node_absent_is_none(self):
        with self.env_only_bin():  # empty bin dir, no node on PATH
            self.assertIsNone(C.node_version())

    def test_kaggle_cmd_prefers_path_executable(self):
        self.make_bin("kaggle", 'echo "Kaggle CLI 9.9.9"\n')
        with self.env_with_bin():
            self.assertEqual(C.kaggle_cmd(), [str(self.bin / "kaggle")])
            self.assertIn("9.9.9", C.kaggle_version())

    def test_kaggle_cmd_fallback_to_module(self):
        with self.env_only_bin():  # no kaggle on PATH -> module fallback
            self.assertEqual(C.kaggle_cmd(), [sys.executable, "-m", "kaggle"])
            self.assertEqual(C.kaggle_version(), "")  # no kaggle -> empty, no raise

    def test_run_kaggle_corrupt_binary_is_synthetic_failure(self):
        p = self.bin / "kaggle"
        p.write_bytes(b"\x7fELF" + b"\x00" * 20)   # not a runnable binary -> ENOEXEC
        p.chmod(0o755)
        with self.env_with_bin():
            r = C.run_kaggle("--version")
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("not runnable", r.stderr)
            ok, _detail = C.kaggle_authed()   # must not raise, must report False
            self.assertFalse(ok)

    def test_wrangler_cmd_shape(self):
        self.make_bin("npx", 'echo "npx ok"\n')
        with self.env_only_bin():
            self.assertEqual(C.wrangler_cmd("whoami"),
                             [str(self.bin / "npx"), "-y", "wrangler@4", "whoami"])

    def test_wrangler_cmd_without_npx_raises(self):
        with self.env_only_bin():
            with self.assertRaises(C.SubprocessError):
                C.wrangler_cmd("whoami")


class TestRunLogged(FakeBinBase):
    def test_success_returns_completed_process(self):
        self.make_bin("tool", 'echo "hello out"\necho "hello err" 1>&2\n')
        with self.env_with_bin():
            r = C.run_logged([str(self.bin / "tool")])
            self.assertEqual(r.returncode, 0)
            self.assertEqual(r.stdout.strip(), "hello out")

    def test_failure_raises_with_hint(self):
        self.make_bin("tool", 'echo "Error: ENOTFOUND example" 1>&2\nexit 7\n')
        with self.env_with_bin():
            with self.assertRaises(C.SubprocessError) as cm:
                C.run_logged([str(self.bin / "tool")])
            self.assertIn("failed (exit 7)", str(cm.exception))
            self.assertIn("DNS", cm.exception.hint)  # ENOTFOUND -> DNS hint

    def test_stdin_is_passed(self):
        self.make_bin("tool", 'read v; echo "got:$v"\n')
        with self.env_with_bin():
            r = C.run_logged([str(self.bin / "tool")], stdin="s3cr3t\n")
            self.assertIn("got:s3cr3t", r.stdout)

    def test_output_is_redacted_before_logging(self):
        self.make_bin("tool", 'echo "MY_SECRET_TOKEN=supersecretvalue"\n')
        lines = []
        with self.env_with_bin():
            C.run_logged([str(self.bin / "tool")], log_line=lines.append)
        self.assertTrue(lines)
        for ln in lines:
            self.assertNotIn("supersecretvalue", ln)
        # masked but still recognisable: first4 + ellipsis + last4
        self.assertTrue(any("MY_SECRET_TOKEN=supe…alue" in ln for ln in lines))

    def test_timeout_raises(self):
        self.make_bin("tool", 'sleep 2\n')
        with self.env_with_bin():
            with self.assertRaises(C.SubprocessError):
                C.run_logged([str(self.bin / "tool")], timeout=1)


class TestHintForOutput(unittest.TestCase):
    def test_known_patterns(self):
        self.assertIn("cloudflare", C.hint_for_output("Error: not logged in.").lower())
        # "unauthorized" matches before "401", so feed a bare 401 for the 401 hint
        self.assertIn("401", C.hint_for_output("Error: 401"))
        self.assertIn("Node", C.hint_for_output("Error: engines: node >= 22.0.0"))
        self.assertIn("quota", C.hint_for_output("kaggle: quota exceeded").lower())
        self.assertEqual(C.hint_for_output("totally unknown text"), "")


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------

class TestPrompts(unittest.TestCase):
    def test_yes_mode(self):
        ctx = C.Ctx(yes=True)
        self.assertTrue(C.confirm(ctx, "proceed?", default=False))  # --yes overrides
        self.assertEqual(C.ask(ctx, "name?", default="d"), "d")
        self.assertEqual(C.choose(ctx, "pick?", ["a", "b"]), ["a", "b"])
        self.assertTrue(any("auto-approved" in ln for ln in ctx.log_lines))

    def test_dry_run_mode(self):
        ctx = C.Ctx(dry_run=True)
        self.assertTrue(C.confirm(ctx, "proceed?", default=True))
        self.assertFalse(C.confirm(ctx, "proceed?", default=False))
        self.assertEqual(C.choose(ctx, "pick?", ["a", "b"]), [])
        self.assertTrue(any("would ask" in ln for ln in ctx.log_lines))

    def test_manual_answers_via_input(self):
        ctx = C.Ctx()
        with mock.patch("builtins.input", side_effect=["n", "y", "b", "1, 2"]):
            self.assertFalse(C.confirm(ctx, "first?"))
            self.assertTrue(C.confirm(ctx, "second?"))
            self.assertEqual(C.ask(ctx, "value?", default="d"), "b")
            self.assertEqual(C.choose(ctx, "pick?", ["a", "b"]), ["a", "b"])

    def test_manual_eof_uses_defaults(self):
        ctx = C.Ctx()
        with mock.patch("builtins.input", side_effect=EOFError):
            self.assertFalse(C.confirm(ctx, "ask?", default=False))
            self.assertEqual(C.ask(ctx, "value?", default="d"), "d")
            self.assertEqual(C.choose(ctx, "pick?", ["a"]), [])

    def test_wait_for_enter_eof_safe(self):
        ctx = C.Ctx()
        with mock.patch("builtins.input", side_effect=EOFError):
            C.wait_for_enter(ctx, "enter?")


# ---------------------------------------------------------------------------
# git-tree secret guard
# ---------------------------------------------------------------------------

@unittest.skipUnless(IS_POSIX and subprocess.run(
    ["git", "--version"], capture_output=True).returncode == 0,
    "needs git")
class TestGitignoreGuard(FakeBinBase, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.repo = Path(self._td.name) / "repo"
        self.repo.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=self.repo, check=True)

    def test_guard_outside_git_is_noop(self):
        ctx = C.Ctx()
        target = self.work / "opencode.json"
        C.gitignore_guard(ctx, str(target), "literalkey1234567890")
        self.assertFalse((self.work / ".gitignore").exists())

    def test_guard_adds_pattern_with_yes(self):
        ctx = C.Ctx(yes=True)
        target = self.repo / "opencode.json"
        C.gitignore_guard(ctx, str(target), "literalkey1234567890")
        gi = (self.repo / ".gitignore").read_text()
        self.assertIn("opencode.json", gi)

    def test_guard_skips_placeholder(self):
        ctx = C.Ctx(yes=True)
        target = self.repo / "opencode.json"
        C.gitignore_guard(ctx, str(target), ktl_env.PLACEHOLDER)
        self.assertFalse((self.repo / ".gitignore").exists())

    def test_guard_decline_leaves_gitignore(self):
        ctx = C.Ctx()
        target = self.repo / "opencode.json"
        with mock.patch("builtins.input", return_value="n"):
            C.gitignore_guard(ctx, str(target), "literalkey1234567890")
        self.assertFalse((self.repo / ".gitignore").exists())


if __name__ == "__main__":
    unittest.main()
