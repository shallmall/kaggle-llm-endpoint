#!/usr/bin/env bash
# Local (CPU-only) verification for the GLM-5.3-Flash serving stack.
# No Kaggle account, no TPU needed.
#
#   verify_local.sh          launcher build check + full kernel smoke test
#   verify_local.sh --full   + the engine unit test suite (slow on a laptop)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB="$(cd "$HERE/../.." && pwd)"     # kaggle-llm-endpoint/
ROOT="$(cd "$LAB/.." && pwd)"        # repo root
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"
FULL=0
[ "${1:-}" = "--full" ] && FULL=1

if ! "$PY" -c "import jax, numpy, torch, transformers" 2>/dev/null; then
  echo "missing python deps — install them into the repo venv:"
  echo "  uv pip install 'jax[cpu]' numpy transformers pytest gguf"
  echo "  uv pip install torch --index-url https://download.pytorch.org/whl/cpu"
  exit 1
fi

echo "== 1/3 launcher build check (CFG injection + embedded engine) =="
"$PY" - "$LAB" <<'EOF'
import sys, re, io, tarfile, base64, subprocess, tempfile, pathlib
lab = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(lab))
import launch
cfg = {"ntfy_topic": "ktl-verify", "api_key": "verify", "served_model_name": "glm-5.3-flash",
       "keepalive_min": 480, "reasoning_effort_default": "low", "max_len": 262144,
       "streams": 3, "vision": True}
src, name = launch.build_kernel(launch.MODELS["glm"], cfg)
if not re.search(r"^CFG = \{.*\}$", src, re.M):
    sys.exit("CFG line missing from built kernel")
m = re.search(r'^ENGINE_B64 = "([A-Za-z0-9+/=]+)"', src, re.M)
if not m:
    sys.exit("ENGINE_B64 line missing from built kernel")
tf = tarfile.open(fileobj=io.BytesIO(base64.b64decode(m.group(1))), mode="r:gz")
names = tf.getnames()
if len(names) < 15:
    sys.exit(f"engine tarball has {len(names)} files, expected 15")
with tempfile.TemporaryDirectory() as td:
    tf.extractall(td)
    (pathlib.Path(td) / "built.py").write_text(src)
    checks = [("kernel", pathlib.Path(td) / "built.py", ["-m", "py_compile"]),
              ("engine", pathlib.Path(td) / "glm53", ["-m", "compileall", "-q"])]
    for label, path, args in checks:
        r = subprocess.run([sys.executable, *args, str(path)], capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"{label} compile failed:\n{r.stderr}")
print(f"   OK: {name} ({len(src) // 1024} KB), {len(names)} engine files, all compile")
EOF

echo "== 2/3 kernel smoke test (CPU, tiny engine, real GLM tokenizer) =="
SNAP="$HOME/.cache/glm_hf/models--zai-org--GLM-5.3-Flash/snapshots"
if [ ! -d "$SNAP" ] || [ -z "$(ls -A "$SNAP" 2>/dev/null)" ]; then
  echo "   downloading the GLM tokenizer (a few MB, one-time)..."
  "$PY" - <<'EOF'
import os
from huggingface_hub import snapshot_download
p = snapshot_download("zai-org/GLM-5.3-Flash", allow_patterns=["*.json", "*.jinja", "tokenizer*"])
d = os.path.expanduser("~/.cache/glm_hf/models--zai-org--GLM-5.3-Flash/snapshots")
os.makedirs(d, exist_ok=True)
link = os.path.join(d, "main")
if not os.path.exists(link):
    os.symlink(p, link)
print("   tokenizer at", d)
EOF
fi
(cd "$LAB" && "$PY" glm53-flash/tools/harness_serve.py)

if [ "$FULL" = 1 ]; then
  echo "== 3/3 engine unit test suite (CPU, slow) =="
  (cd "$LAB/glm53-flash/engine" && "$PY" -m pytest glm53/tests -q)
else
  echo "== 3/3 engine unit test suite: skipped (pass --full to run it) =="
fi
echo
echo "ALL CHECKS PASSED"
