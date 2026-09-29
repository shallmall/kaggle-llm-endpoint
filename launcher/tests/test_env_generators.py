"""Per-client generator output: structure + per-shell syntax."""
import unittest
from _helpers import mk_resolved, placeholder_resolved
import ktl_env


class TestShellSyntax(unittest.TestCase):
    SHELLS = ["bash", "zsh", "fish", "powershell", "cmd"]

    def test_export_reference_per_shell(self):
        cases = {
            "bash": "export OPENAI_API_KEY=$KTL_CLIENT_API_KEY",
            "zsh": "export OPENAI_API_KEY=$KTL_CLIENT_API_KEY",
            "fish": "set -x OPENAI_API_KEY $KTL_CLIENT_API_KEY",
            "powershell": "$env:OPENAI_API_KEY = $env:KTL_CLIENT_API_KEY",
            "cmd": "set OPENAI_API_KEY=%KTL_CLIENT_API_KEY%",
        }
        for shell, want in cases.items():
            self.assertEqual(ktl_env.shell_export(shell, "OPENAI_API_KEY", True, "x"), want)

    def test_export_literal_per_shell(self):
        self.assertIn("sk-a", ktl_env.shell_export("bash", "K", False, "sk-a"))
        self.assertTrue(ktl_env.shell_export("bash", "K", False, "sk-a").startswith("export K='"))
        self.assertIn("$env:K = \"sk-a\"", ktl_env.shell_export("powershell", "K", False, "sk-a"))
        self.assertEqual(ktl_env.shell_export("cmd", "K", False, "sk-a"), "set K=sk-a")

    def test_detect_shell_default(self):
        self.assertIn(ktl_env.detect_shell(), self.SHELLS)


class TestClientGenerators(unittest.TestCase):
    def _out(self, client, **kw):
        return ktl_env.render(mk_resolved(client=client, **kw))

    def test_claude_code_root_no_v1(self):
        out = self._out("claude-code")
        self.assertIn('"ANTHROPIC_BASE_URL": "https://ex.dev"', out)
        self.assertNotIn("https://ex.dev/v1\"", out)          # must NOT be /v1
        # bare model id (vLLM validates the name; no provider prefix)
        self.assertIn('"ANTHROPIC_MODEL": "qwen3.8-27b"', out)
        self.assertNotIn("kaggle-tpu/qwen3.8-27b", out)
        # background (haiku) tasks routed at the same model, both var spellings
        self.assertIn('"ANTHROPIC_SMALL_FAST_MODEL": "qwen3.8-27b"', out)
        self.assertIn('"ANTHROPIC_DEFAULT_HAIKU_MODEL": "qwen3.8-27b"', out)
        # sonnet/opus aliases pinned too — otherwise /model sonnet|opus would
        # switch to default Anthropic ids the relay does not serve
        self.assertIn('"ANTHROPIC_DEFAULT_SONNET_MODEL": "qwen3.8-27b"', out)
        self.assertIn('"ANTHROPIC_DEFAULT_OPUS_MODEL": "qwen3.8-27b"', out)
        # file client: placeholder by default, no real secret in output
        self.assertIn('"ANTHROPIC_AUTH_TOKEN": "<YOUR_CLIENT_API_KEY>"', out)
        self.assertNotIn("sk-test-123", out)

    def test_claude_code_reveal_literal(self):
        out = self._out("claude-code", reveal=True)
        self.assertIn('"ANTHROPIC_AUTH_TOKEN": "sk-test-123"', out)

    def test_claude_code_placeholder_not_leaked(self):
        out = self._out("claude-code", client_key="<YOUR_CLIENT_API_KEY>",
                        key_is_placeholder=True, key_source="placeholder")
        self.assertIn("<YOUR_CLIENT_API_KEY>", out)

    def test_codex_responses_and_env_key(self):
        out = self._out("codex")
        self.assertIn('base_url = "https://ex.dev/v1"', out)
        self.assertIn('wire_api = "responses"', out)
        self.assertIn('env_key = "KTL_CLIENT_API_KEY"', out)
        self.assertIn("# >>> ktl managed >>>", out)
        self.assertIn("codex -c model_provider=kaggle-tpu -c model=qwen3.8-27b", out)
        # no self-referencing export, no leaked literal without --reveal
        self.assertNotIn("export KTL_CLIENT_API_KEY=$KTL_CLIENT_API_KEY", out)
        self.assertNotIn("sk-test-123", out)

    def test_codex_reveal_shows_literal(self):
        out = self._out("codex", reveal=True)
        self.assertIn("sk-test-123", out)

    def test_codex_unsupported_model_warns(self):
        # e.g. GLM: backend has no /v1/responses -> must warn, not advertise
        out = self._out("codex", responses_api=False)
        self.assertIn("NOT SUPPORTED FOR THIS MODEL", out)
        self.assertIn("/v1/responses", out)

    def test_codex_supported_model_no_warning(self):
        out = self._out("codex", responses_api=True)
        self.assertNotIn("NOT SUPPORTED FOR THIS MODEL", out)

    def test_opencode_v1_and_literal(self):
        out = self._out("opencode")
        self.assertIn('"baseURL": "https://ex.dev/v1"', out)
        self.assertIn('"kaggle-tpu"', out)
        self.assertIn('"apiKey": "<YOUR_CLIENT_API_KEY>"', out)  # placeholder default
        self.assertNotIn("sk-test-123", out)

    def test_opencode_reveal_literal(self):
        out = self._out("opencode", reveal=True)
        self.assertIn('"apiKey": "sk-test-123"', out)

    def test_hermes_model_block(self):
        out = self._out("hermes")
        # .env line: placeholder by default, no real secret
        self.assertIn("KTL_CLIENT_API_KEY=<YOUR_CLIENT_API_KEY>", out)
        # config.yaml model block (provider: custom -> chat/completions)
        self.assertIn("provider: custom", out)
        self.assertIn("api_mode: chat_completions", out)
        self.assertIn("default: qwen3.8-27b", out)
        self.assertIn("base_url: https://ex.dev/v1", out)
        self.assertIn("api_key: ${KTL_CLIENT_API_KEY}", out)
        # no OPENAI_* config lines (that mechanism does not route; see comments)
        self.assertNotIn("OPENAI_BASE_URL=", out)
        self.assertNotIn("OPENAI_API_KEY=", out)
        self.assertNotIn("sk-test-123", out)

    def test_hermes_reveal_literal(self):
        out = self._out("hermes", reveal=True)
        self.assertIn("KTL_CLIENT_API_KEY=sk-test-123", out)
        # the YAML always references the env var — never the literal
        self.assertIn("api_key: ${KTL_CLIENT_API_KEY}", out)
        self.assertNotIn("api_key: sk-test-123", out)

    def test_aider_print_only(self):
        out = self._out("aider")
        self.assertIn("ex.dev/v1", out)
        self.assertIn("AIDER_MODEL", out)
        # endpoint var is OPENAI_API_BASE (verified: aider openai-compat docs),
        # and the model needs the openai/ prefix to route to the compat endpoint
        self.assertIn("OPENAI_API_BASE", out)
        self.assertNotIn("OPENAI_BASE_URL", out)
        self.assertIn("openai/qwen3.8-27b", out)

    def test_curl_two_examples(self):
        out = self._out("curl")
        self.assertIn("ex.dev/v1/chat/completions", out)
        self.assertIn("ex.dev/v1/messages", out)
        self.assertIn("x-api-key:", out)
        self.assertIn("anthropic-version:", out)

    def test_python_base_urls(self):
        out = self._out("python")
        self.assertIn('base_url="https://ex.dev/v1"', out)   # openai -> /v1
        self.assertIn('base_url="https://ex.dev"', out)      # anthropic -> root

    def test_print_clients_use_reference_by_default(self):
        for client in ("aider", "curl", "python"):
            out = self._out(client)
            self.assertIn("KTL_CLIENT_API_KEY", out)          # the reference
            self.assertNotIn("sk-test-123", out)              # not the literal

    def test_print_clients_reveal_literal(self):
        for client in ("aider", "curl", "python"):
            out = self._out(client, reveal=True)
            self.assertIn("sk-test-123", out)

    def test_all_clients_render_without_error(self):
        for client in ktl_env.CLIENTS:
            for shell in TestShellSyntax.SHELLS:
                self.assertTrue(ktl_env.render(mk_resolved(client=client, shell=shell)).strip())


if __name__ == "__main__":
    unittest.main()
