#!/usr/bin/env python3
"""
kaggle-tpu-lab launcher — serve LLMs on a free Kaggle TPU from your terminal.

    python launch.py setup                          # guided install (fresh machine)
    python launch.py doctor                         # read-only health report
    python launch.py serve                          # Qwen3.8-27B (default)
    python launch.py serve --model glm              # GLM-5.3-Flash (high reasoning)
    python launch.py env                            # AI-client config for the relay
    python launch.py status                         # one-shot status + recent events
    python launch.py stop                           # kill the TPU session

This file is a thin argparse dispatcher. Command logic lives in:
    ktl_setup.py  — setup wizard + doctor
    ktl_env.py    — AI-client config generator + compatibility matrix
    ktl_serve.py  — kernel build/push, ntfy watch, status/stop
    ktl_common.py — config file, keys, redaction, prompts, subprocess helpers

Requires the Kaggle CLI, authenticated (see `python launch.py setup`).
Only the Python standard library is used.
"""
import argparse
import sys

import ktl_common
import ktl_env
import ktl_serve
import ktl_setup


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    # --- serve ---
    s = sub.add_parser("serve", help="push the serving kernel and watch it come up")
    s.add_argument("--model", choices=list(ktl_common.MODELS), default="qwen",
                   help="which model to serve (default: qwen)")
    s.add_argument("--user", help="Kaggle username (auto-detected if possible)")
    s.add_argument("--slug", default=None,
                   help="kernel name (default: per-model slug)")
    s.add_argument("--max-model-len", type=int, default=None,
                   help="context length (Qwen default: 262144; GLM default: 262144)")
    s.add_argument("--max-num-seqs", type=int, default=4,
                   help="max concurrent sequences (Qwen only)")
    s.add_argument("--streams", type=int, default=None,
                   help="concurrent decode streams (GLM only)")
    s.add_argument("--mtp", type=int, default=3,
                   help="MTP speculative tokens (Qwen only; 0 disables)")
    s.add_argument("--reasoning-effort", default=None,
                   help="server-side default reasoning effort "
                        "(Qwen: xhigh|medium|low; GLM: low|high)")
    s.add_argument("--keepalive-min", type=int, default=480,
                   help="auto-shutdown after this many minutes of serving")
    s.add_argument("--weights-dataset", default=None,
                   help="override the weights dataset (Qwen only)")
    s.add_argument("--no-tools", action="store_true",
                   help="disable tool-calling support (Qwen only)")
    s.add_argument("--text-only", action="store_true",
                   help="skip the vision tower (Qwen only)")
    s.add_argument("--vision", type=lambda x: x.lower() in ("true", "1", "yes"),
                   default=None,
                   help="enable/disable vision tower (GLM only)")
    s.add_argument("--verbose", action="store_true",
                   help="show verbose log lines in the kernel log")
    s.add_argument("--fast-start", action="store_true",
                   help="skip TPU graph precompile (Qwen only)")
    s.set_defaults(fn=ktl_serve.cmd_serve)

    # --- build-env (Qwen maintainer flow) ---
    s = sub.add_parser("build-env", help="(maintainers) push a kernel that builds the "
                        "env dataset: venv + XLA cache + cloudflared")
    s.add_argument("--user", help="Kaggle username (auto-detected if possible)")
    s.add_argument("--slug", default="qwen38-env-bundle")
    s.add_argument("--weights-dataset", default="rahim3/qwen3-8-27b-bf16")
    s.set_defaults(fn=ktl_serve.cmd_build_env)

    # --- status ---
    s = sub.add_parser("status", help="show current kernel status + recent events")
    s.add_argument("--follow", "-f", action="store_true", help="keep watching")
    s.set_defaults(fn=ktl_serve.cmd_status)

    # --- stop ---
    s = sub.add_parser("stop", help="terminate the TPU session")
    s.set_defaults(fn=ktl_serve.cmd_stop)

    # --- env (AI-client config generator) ---
    s = sub.add_parser("env", help="print / write AI-client config for the relay",
                       description="Generate ready-to-paste (or auto-written) config for "
                                   "AI clients pointing at this endpoint.")
    s.add_argument("client", nargs="?", default="all",
                   choices=["all"] + ktl_env.CLIENTS,
                   help="which client to configure (default: all)")
    s.add_argument("--restore", metavar="CLIENT", default=None,
                   choices=["all"] + ktl_env.CLIENTS,
                   help="restore the most recent backup for CLIENT (instead of generating)")
    s.add_argument("--model", choices=list(ktl_common.MODELS), default=None,
                   help="model to target (default: last served, else qwen)")
    s.add_argument("--url", default=None,
                   help="base URL (root, /v1 optional); overrides KTL_RELAY_URL and config")
    s.add_argument("--shell", choices=["bash", "zsh", "fish", "powershell", "cmd"],
                   default=None, help="shell syntax (default: autodetected)")
    s.add_argument("--write", action="store_true",
                   help="write the config file(s) (with a .ktl-bak-* backup)")
    s.add_argument("--scope", default=None,
                   help="comma list of clients to limit --write to (e.g. opencode,claude-code)")
    s.add_argument("--dry-run", action="store_true",
                   help="show the diff of what --write would change, without writing")
    s.add_argument("--reveal", action="store_true",
                   help="embed the literal API key in printed output (default: env reference)")
    s.add_argument("--test", action="store_true",
                   help="run the live compatibility matrix against the endpoint")
    s.add_argument("--wait", type=int, default=0, metavar="SECONDS",
                   help="with --test: keep polling every 10 s while booting/unreachable")
    s.add_argument("--json", action="store_true", help="machine-readable output")
    s.set_defaults(fn=ktl_env.cmd_env)

    # --- setup (guided installer) ---
    s = sub.add_parser("setup", help="guided installer: prerequisites -> Worker -> clients",
                       description="Configures a fresh machine (prerequisites, Cloudflare "
                                   "login, Worker relay, secrets, AI-client configs) "
                                   "without starting a TPU session. Each step is "
                                   "idempotent and resumable; state lives in "
                                   "~/.ktl/config.json. Boot the TPU later with "
                                   "`python launch.py serve`.")
    s.add_argument("--yes", action="store_true",
                   help="answer yes to every prompt (quota warnings are still logged)")
    s.add_argument("--dry-run", action="store_true",
                   help="print every action, change nothing")
    s.add_argument("--only", choices=list(ktl_common.STEP_NAMES) + ["verify"],
                   default=None, help="run just this step")
    s.add_argument("--model", choices=list(ktl_common.MODELS), default=None,
                   help="model to serve/configure (default: config.json, else qwen)")
    s.add_argument("--worker-name", default=None,
                   help="Cloudflare Worker name (default: kaggle-tpu-relay)")
    s.add_argument("--adopt-worker", metavar="URL", default=None,
                   help="skip deploy: point at an existing worker.dev URL, "
                        "just set its secrets and save it")
    s.add_argument("--rotate", action="store_true",
                   help="regenerate the relay secrets (with approval) and re-put them")
    s.add_argument("--reset", action="store_true",
                   help="delete ~/.ktl/config.json after confirmation, then start fresh "
                        "(does NOT delete the Worker)")
    s.add_argument("--verbose", action="store_true", help="more log detail")
    s.set_defaults(fn=ktl_setup.cmd_setup)

    # --- doctor (read-only health report) ---
    s = sub.add_parser("doctor", help="read-only health report (safe to paste in bug reports)")
    s.add_argument("--json", action="store_true", help="machine-readable output")
    s.set_defaults(fn=ktl_setup.cmd_doctor)

    args = ap.parse_args()

    # Set model-specific defaults for reasoning_effort
    if args.cmd == "serve" and args.reasoning_effort is None:
        args.reasoning_effort = "xhigh" if args.model == "qwen" else "high"
    if args.cmd == "serve" and args.max_model_len is None:
        args.max_model_len = 262144

    # Commands return an exit code (env/doctor) or None.
    # sys.exit(None) -> 0, so this is safe for every subcommand.
    rc = args.fn(args)
    if rc is not None:
        sys.exit(rc)


if __name__ == "__main__":
    main()
