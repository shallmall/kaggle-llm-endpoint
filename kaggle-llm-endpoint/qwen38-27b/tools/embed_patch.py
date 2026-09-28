#!/usr/bin/env python3
"""Re-embed patches/*.diff into kernel/serve_qwen38.py (paths relative to this model folder).

Currently embedded (applied in this order by apply_mtp_patch):
  - mtp-rollback-v0280.diff      GDN state rollback on rejected draft tokens
  - spec-draft-rows-v0280.diff   coerce draft-token rows to mutable lists so
                                 async scheduling + structured outputs don't
                                 crash on immutable rows

Run after editing a patch file:  python qwen38-27b/tools/embed_patch.py
"""
import base64
import gzip
import re
from pathlib import Path

repo = Path(__file__).resolve().parent.parent
patches_dir = repo / "patches"
diff = b"".join((patches_dir / name).read_bytes() for name in (
    "mtp-rollback-v0280.diff",
    "spec-draft-rows-v0280.diff",
))
blob = base64.b64encode(gzip.compress(diff, 9)).decode()

script_path = repo / "kernel" / "serve_qwen38.py"
src = script_path.read_text()
new, n = re.subn(r'^MTP_PATCH_B64 = .*$',
                 f'MTP_PATCH_B64 = "{blob}"  # __EMBEDDED_PATCH__',
                 src, count=1, flags=re.M)
if n != 1:
    raise SystemExit("marker line MTP_PATCH_B64 = ... not found")
script_path.write_text(new)
print(f"embedded {len(diff)} bytes of diff as {len(blob)} chars of base64")
