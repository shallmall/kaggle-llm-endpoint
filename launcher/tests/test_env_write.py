"""Write-mode: merge semantics, idempotency, refuse-unparseable, backups, restore."""
import json
import os
import tempfile
import unittest
from _helpers import mk_resolved, placeholder_resolved
import ktl_env


class WriteModeBase(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.dir = self._td.name

    def tearDown(self):
        self._td.cleanup()

    def path(self, name):
        return os.path.join(self.dir, name)

    def read(self, p):
        with open(p) as f:
            return f.read()

    def write(self, p, content):
        with open(p, "w") as f:
            f.write(content)

    def jload(self, p):
        with open(p) as f:
            return json.load(f)


class TestOpencodeWrite(WriteModeBase):
    def test_new_file(self):
        p = self.path("opencode.json")
        ktl_env._write_json_provider(p, mk_resolved(client="opencode"), False)
        data = self.jload(p)
        self.assertIn("qwen3.8-27b", data["provider"]["kaggle-tpu"]["models"])
        self.assertEqual(data["provider"]["kaggle-tpu"]["options"]["baseURL"],
                         "https://ex.dev/v1")

    def test_preserves_other_models_and_real_key(self):
        p = self.path("opencode.json")
        existing = {"provider": {"kaggle-tpu": {
            "options": {"baseURL": "https://old.dev/v1", "apiKey": "sk-OLD-REAL"},
            "models": {"glm-5.3-flash": {"name": "glm-5.3-flash"}}}}}
        self.write(p, json.dumps(existing))
        ktl_env._write_json_provider(p, placeholder_resolved(client="opencode"), False)
        ktl = self.jload(p)["provider"]["kaggle-tpu"]
        self.assertIn("glm-5.3-flash", ktl["models"])
        self.assertIn("qwen3.8-27b", ktl["models"])
        self.assertEqual(ktl["options"]["apiKey"], "sk-OLD-REAL")   # NOT clobbered
        self.assertEqual(ktl["options"]["baseURL"], "https://ex.dev/v1")

    def test_real_key_overwrites_with_reveal(self):
        p = self.path("opencode.json")
        existing = {"provider": {"kaggle-tpu": {
            "options": {"baseURL": "https://old.dev/v1", "apiKey": "sk-OLD"}, "models": {}}}}
        self.write(p, json.dumps(existing))
        ktl_env._write_json_provider(p, mk_resolved(client="opencode", reveal=True), False)
        self.assertEqual(self.jload(p)["provider"]["kaggle-tpu"]["options"]["apiKey"],
                         "sk-test-123")

    def test_no_reveal_preserves_existing_key(self):
        p = self.path("opencode.json")
        existing = {"provider": {"kaggle-tpu": {
            "options": {"baseURL": "https://old.dev/v1", "apiKey": "sk-OLD"}, "models": {}}}}
        self.write(p, json.dumps(existing))
        ktl_env._write_json_provider(p, mk_resolved(client="opencode"), False)
        self.assertEqual(self.jload(p)["provider"]["kaggle-tpu"]["options"]["apiKey"], "sk-OLD")

    def test_new_file_no_reveal_writes_placeholder(self):
        p = self.path("opencode.json")
        ktl_env._write_json_provider(p, mk_resolved(client="opencode"), False)
        self.assertEqual(self.jload(p)["provider"]["kaggle-tpu"]["options"]["apiKey"],
                         "<YOUR_CLIENT_API_KEY>")

    def test_idempotent(self):
        p = self.path("opencode.json")
        r = mk_resolved(client="opencode")
        ktl_env._write_json_provider(p, r, False)
        first = self.read(p)
        self.assertIn("no change", ktl_env._write_json_provider(p, r, False))
        self.assertEqual(self.read(p), first)

    def test_refuse_unparseable(self):
        p = self.path("opencode.json")
        self.write(p, "{ not valid json ")
        with self.assertRaises(ktl_env.RefuseWrite):
            ktl_env._write_json_provider(p, mk_resolved(client="opencode"), False)
        self.assertTrue(any(".ktl-bak-" in b for b in os.listdir(self.dir)))


class TestClaudeWrite(WriteModeBase):
    def test_preserves_other_env_keys(self):
        p = self.path("settings.json")
        self.write(p, json.dumps({"env": {"OTHER_VAR": "keep"}, "theme": "dark"}))
        ktl_env._write_json_env(p, placeholder_resolved(client="claude-code"), False)
        data = self.jload(p)
        self.assertEqual(data["env"]["OTHER_VAR"], "keep")
        self.assertEqual(data["theme"], "dark")
        self.assertEqual(data["env"]["ANTHROPIC_BASE_URL"], "https://ex.dev")
        self.assertEqual(data["env"]["ANTHROPIC_AUTH_TOKEN"], "<YOUR_CLIENT_API_KEY>")

    def test_real_key_set_with_reveal(self):
        p = self.path("settings.json")
        ktl_env._write_json_env(p, mk_resolved(client="claude-code", reveal=True), False)
        self.assertEqual(self.jload(p)["env"]["ANTHROPIC_AUTH_TOKEN"], "sk-test-123")

    def test_no_reveal_new_file_writes_placeholder(self):
        p = self.path("settings.json")
        ktl_env._write_json_env(p, mk_resolved(client="claude-code"), False)
        self.assertEqual(self.jload(p)["env"]["ANTHROPIC_AUTH_TOKEN"],
                         "<YOUR_CLIENT_API_KEY>")


class TestCodexWrite(WriteModeBase):
    def test_new_file(self):
        p = self.path("config.toml")
        ktl_env._write_toml_block(p, mk_resolved(client="codex"), False)
        content = self.read(p)
        self.assertIn("# >>> ktl managed >>>", content)
        self.assertIn('base_url = "https://ex.dev/v1"', content)

    def test_idempotent_replaces_block(self):
        p = self.path("config.toml")
        r = mk_resolved(client="codex")
        ktl_env._write_toml_block(p, r, False)
        first = self.read(p)
        self.assertIn("no change", ktl_env._write_toml_block(p, r, False))
        self.assertEqual(self.read(p), first)
        self.assertEqual(first.count("# >>> ktl managed >>>"), 1)

    def test_updates_block_when_changed(self):
        p = self.path("config.toml")
        ktl_env._write_toml_block(p, mk_resolved(client="codex", base_url="https://a.dev"), False)
        ktl_env._write_toml_block(p, mk_resolved(client="codex", base_url="https://b.dev"), False)
        content = self.read(p)
        self.assertIn('base_url = "https://b.dev/v1"', content)
        self.assertNotIn("https://a.dev", content)
        self.assertEqual(content.count("# >>> ktl managed >>>"), 1)

    def test_preserves_user_content(self):
        p = self.path("config.toml")
        self.write(p, 'model = "gpt-4"\nnotify = ["x"]\n')
        ktl_env._write_toml_block(p, mk_resolved(client="codex"), False)
        content = self.read(p)
        self.assertIn('model = "gpt-4"', content)
        self.assertIn("# >>> ktl managed >>>", content)

    def test_refuse_unparseable(self):
        p = self.path("config.toml")
        self.write(p, "this = = = not toml\n[[[broken\n")
        with self.assertRaises(ktl_env.RefuseWrite):
            ktl_env._write_toml_block(p, mk_resolved(client="codex"), False)


class TestHermesWrite(WriteModeBase):
    def test_new_file_with_reveal(self):
        p = self.path(".env")
        ktl_env._write_hermes_env(p, mk_resolved(client="hermes", reveal=True), False)
        content = self.read(p)
        self.assertIn("KTL_CLIENT_API_KEY=sk-test-123", content)

    def test_new_file_no_reveal_writes_placeholder(self):
        p = self.path(".env")
        ktl_env._write_hermes_env(p, mk_resolved(client="hermes"), False)
        content = self.read(p)
        self.assertIn("KTL_CLIENT_API_KEY=<YOUR_CLIENT_API_KEY>", content)

    def test_placeholder_keeps_existing_key(self):
        p = self.path(".env")
        self.write(p, "KTL_CLIENT_API_KEY=sk-EXISTING\nFOO=bar\n")
        ktl_env._write_hermes_env(p, placeholder_resolved(client="hermes"), False)
        lines = self.read(p).splitlines()
        self.assertIn("KTL_CLIENT_API_KEY=sk-EXISTING", lines)
        self.assertIn("FOO=bar", lines)

    def test_stale_openai_lines_left_untouched(self):
        p = self.path(".env")
        self.write(p, "OPENAI_API_KEY=sk-OLD\nOPENAI_BASE_URL=http://x/v1\n")
        ktl_env._write_hermes_env(p, mk_resolved(client="hermes", reveal=True), False)
        lines = self.read(p).splitlines()
        self.assertIn("OPENAI_API_KEY=sk-OLD", lines)          # not ours to remove
        self.assertIn("OPENAI_BASE_URL=http://x/v1", lines)
        self.assertIn("KTL_CLIENT_API_KEY=sk-test-123", lines)

    def test_idempotent(self):
        p = self.path(".env")
        r = mk_resolved(client="hermes")
        ktl_env._write_hermes_env(p, r, False)
        first = self.read(p)
        self.assertIn("no change", ktl_env._write_hermes_env(p, r, False))
        self.assertEqual(self.read(p), first)


class TestHermesModelWrite(WriteModeBase):
    def test_new_config_gets_block(self):
        p = self.path("config.yaml")
        ktl_env._write_hermes_model(p, mk_resolved(client="hermes"), False)
        content = self.read(p)
        self.assertIn("provider: custom", content)
        self.assertIn("api_mode: chat_completions", content)
        self.assertIn("default: qwen3.8-27b", content)
        self.assertIn("base_url: https://ex.dev/v1", content)
        self.assertIn("api_key: ${KTL_CLIENT_API_KEY}", content)

    def test_replaces_only_the_model_block(self):
        p = self.path("config.yaml")
        self.write(p,
                   "model:\n"
                   "  default: auto/best-coding\n"
                   "  provider: custom\n"
                   "  base_url: http://localhost:20128/v1\n"
                   "\n"
                   "database:\n"
                   "  journal_mode: wal\n")
        ktl_env._write_hermes_model(p, mk_resolved(client="hermes"), False)
        content = self.read(p)
        self.assertIn("default: qwen3.8-27b", content)
        self.assertIn("base_url: https://ex.dev/v1", content)
        self.assertNotIn("localhost:20128", content)
        self.assertNotIn("auto/best-coding", content)
        self.assertIn("journal_mode: wal", content)      # other sections untouched

    def test_preserves_indented_model_like_lines(self):
        p = self.path("config.yaml")
        self.write(p,
                   "web:\n"
                   "  search_backend: searxng\n"
                   "  models:\n"
                   "    model: embedded\n")
        ktl_env._write_hermes_model(p, mk_resolved(client="hermes"), False)
        content = self.read(p)
        self.assertIn("models:", content)                # indented `models:` kept
        self.assertIn("model: embedded", content)
        self.assertIn("provider: custom", content)       # block appended

    def test_idempotent(self):
        p = self.path("config.yaml")
        r = mk_resolved(client="hermes")
        ktl_env._write_hermes_model(p, r, False)
        first = self.read(p)
        self.assertIn("no change", ktl_env._write_hermes_model(p, r, False))
        self.assertEqual(self.read(p), first)


class TestBackupRestore(WriteModeBase):
    def test_backup_keeps_three(self):
        p = self.path("f.txt")
        self.write(p, "v0")
        for i in range(5):
            self.write(p, f"v{i+1}")
            ktl_env.backup(p)
        backs = [b for b in os.listdir(self.dir) if ".ktl-bak-" in b]
        self.assertEqual(len(backs), 3)

    def test_restore_latest(self):
        p = self.path("f.txt")
        self.write(p, "original")
        ktl_env.backup(p)
        self.write(p, "modified")
        msg = ktl_env.restore(p)
        self.assertIn("restored", msg)
        self.assertEqual(self.read(p), "original")

    def test_restore_no_backup(self):
        p = self.path("nope.txt")
        self.assertIn("no backup", ktl_env.restore(p))


class TestDryRun(WriteModeBase):
    def test_dry_run_does_not_write(self):
        p = self.path("opencode.json")
        report = ktl_env._write_json_provider(p, mk_resolved(client="opencode"), True)
        self.assertTrue(report.startswith("[dry-run]"))
        self.assertFalse(os.path.exists(p))

    def test_dry_run_diff(self):
        p = self.path("opencode.json")
        self.write(p, "{}\n")
        report = ktl_env._write_json_provider(p, mk_resolved(client="opencode"), True)
        self.assertIn("would update", report)
        self.assertIn("+", report)
        self.assertEqual(self.read(p), "{}\n")


class TestRedact(unittest.TestCase):
    def test_masks_real_key_keeps_placeholder(self):
        old = '{"apiKey": "sk-real-abc"}\n'
        new = '{"apiKey": "<YOUR_CLIENT_API_KEY>"}\n'
        d = ktl_env.unified_diff(old, new, "x.json")
        self.assertNotIn("sk-real-abc", d)
        self.assertIn("***REDACTED***", d)          # old real key masked
        self.assertIn("<YOUR_CLIENT_API_KEY>", d)   # placeholder slot stays visible

    def test_masks_env_style(self):
        d = ktl_env._redact("+OPENAI_API_KEY=sk-live-123\n")
        self.assertNotIn("sk-live-123", d)
        self.assertIn("OPENAI_API_KEY=***REDACTED***", d)

    def test_masks_ktl_key_env_style(self):
        d = ktl_env._redact("+KTL_CLIENT_API_KEY=sk-live-456\n")
        self.assertNotIn("sk-live-456", d)
        self.assertIn("KTL_CLIENT_API_KEY=***REDACTED***", d)

    def test_masks_unrelated_credentials_in_context(self):
        # dotenv diff context lines carry the user's other secrets
        d = ktl_env._redact(" GITHUB_TOKEN=github_pat_11SECRET\n+KTL_CLIENT_API_KEY=<YOUR_CLIENT_API_KEY>\n")
        self.assertNotIn("github_pat_11SECRET", d)
        self.assertIn("GITHUB_TOKEN=***REDACTED***", d)
        self.assertIn("<YOUR_CLIENT_API_KEY>", d)      # placeholder stays visible

    def test_non_secret_names_not_masked(self):
        d = ktl_env._redact(" SEARXNG_URL=http://127.0.0.1:8080\n OPENAI_BASE_URL=https://x.dev/v1\n")
        self.assertIn("http://127.0.0.1:8080", d)
        self.assertIn("https://x.dev/v1", d)
        self.assertNotIn("***REDACTED***", d)


if __name__ == "__main__":
    unittest.main()
