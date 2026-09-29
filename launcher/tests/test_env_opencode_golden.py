"""Golden test for the opencode generator.

Locks the provider block (id `kaggle-tpu`) byte-for-byte so any future
refactor that changes the emitted shape is caught. Values are synthetic;
the *structure and formatting* are what's pinned.
"""
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import ktl_env  # noqa: E402


def resolve(**kw):
    base = dict(
        client="opencode",
        model_key="qwen",
        api_model="qwen3.8-27b",
        context=262144,
        max_output=65536,
        cost={"input": 0.45, "output": 3.2, "cache_read": 0.05},
        base_url="https://ex.workers.dev",
        client_key="sk-test",
        key_is_placeholder=False,
        key_source="env",
        url_source="cli",
        shell="bash",
        reveal=False,
    )
    base.update(kw)
    return ktl_env.Resolved(**base)


class TestOpencodeGolden(unittest.TestCase):
    def test_opencode_provider_json_matches_golden(self):
        golden = (HERE / "tests/goldens/opencode/qwen.reveal.json").read_text()
        got = ktl_env.opencode_provider_json(resolve(reveal=True))
        self.assertEqual(got, golden)

    def test_base_url_has_v1(self):
        prov = ktl_env.opencode_provider(resolve())
        self.assertEqual(prov["kaggle-tpu"]["options"]["baseURL"],
                         "https://ex.workers.dev/v1")

    def test_api_key_placeholder_by_default(self):
        # opencode is a static JSON file (no env-ref support). By default we
        # write the placeholder so no real secret is emitted; --reveal opts in.
        prov = ktl_env.opencode_provider(resolve(reveal=False))
        self.assertEqual(prov["kaggle-tpu"]["options"]["apiKey"],
                         "<YOUR_CLIENT_API_KEY>")

    def test_api_key_literal_with_reveal(self):
        prov = ktl_env.opencode_provider(resolve(reveal=True))
        self.assertEqual(prov["kaggle-tpu"]["options"]["apiKey"], "sk-test")

    def test_api_key_placeholder_when_unavailable(self):
        prov = ktl_env.opencode_provider(
            resolve(client_key="<YOUR_CLIENT_API_KEY>", key_is_placeholder=True))
        self.assertEqual(prov["kaggle-tpu"]["options"]["apiKey"],
                         "<YOUR_CLIENT_API_KEY>")

    def test_glm_model_block(self):
        prov = ktl_env.opencode_provider(resolve(
            model_key="glm", api_model="glm-5.3-flash",
            cost={"input": 0.15, "output": 0.5, "cache_read": 0.03}))
        self.assertIn("glm-5.3-flash", prov["kaggle-tpu"]["models"])
        self.assertEqual(prov["kaggle-tpu"]["models"]["glm-5.3-flash"]["cost"],
                         {"input": 0.15, "output": 0.5, "cache_read": 0.03})


if __name__ == "__main__":
    unittest.main()
