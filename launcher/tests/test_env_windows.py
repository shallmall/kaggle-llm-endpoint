"""Shell detection + Windows (os.name mocked) behavior."""
import os
import unittest
from unittest import mock
from _helpers import mk_resolved
import ktl_env


class TestDetectShell(unittest.TestCase):
    def test_windows_defaults_powershell(self):
        with mock.patch.object(os, "name", "nt"), \
                mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SHELL", None)
            os.environ.pop("PSModulePath", None)
            self.assertEqual(ktl_env.detect_shell(), "powershell")

    def test_psmodulepath_wins(self):
        with mock.patch.dict(os.environ, {"PSModulePath": "C:\\x"}):
            self.assertEqual(ktl_env.detect_shell(), "powershell")

    def test_sHELL_zsh(self):
        with mock.patch.dict(os.environ, {"SHELL": "/usr/bin/zsh"}):
            self.assertEqual(ktl_env.detect_shell(), "zsh")

    def test_sHELL_fish(self):
        with mock.patch.dict(os.environ, {"SHELL": "/usr/local/bin/fish"}):
            self.assertEqual(ktl_env.detect_shell(), "fish")


class TestWindowsRendering(unittest.TestCase):
    def test_powershell_exports(self):
        out = ktl_env.render(mk_resolved(client="aider", shell="powershell"))
        self.assertIn("$env:AIDER_MODEL", out)
        self.assertIn("$env:OPENAI_API_KEY", out)
        self.assertIn("$env:OPENAI_API_BASE", out)

    def test_cmd_exports(self):
        out = ktl_env.render(mk_resolved(client="aider", shell="cmd"))
        self.assertIn("set OPENAI_API_KEY=", out)

    def test_fish_exports(self):
        out = ktl_env.render(mk_resolved(client="aider", shell="fish"))
        self.assertIn("set -x OPENAI_API_KEY", out)

    def test_normalize_root(self):
        self.assertEqual(ktl_env.normalize_root("https://a.dev/v1"), "https://a.dev")
        self.assertEqual(ktl_env.normalize_root("https://a.dev/"), "https://a.dev")
        self.assertEqual(ktl_env.normalize_root("https://a.dev/v1/"), "https://a.dev")
        self.assertEqual(ktl_env.normalize_root("  https://a.dev  "), "https://a.dev")


if __name__ == "__main__":
    unittest.main()
