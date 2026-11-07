import base64
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent
ckpt_path = root / "model" / "hybrid_isp_final.pt"
out_path = root / "backend" / "app" / "core" / "embedded_weights.py"

if not ckpt_path.exists():
    raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

raw = ckpt_path.read_bytes()
b64 = base64.b64encode(raw).decode("utf-8")

content = f'''"""Embedded fallback weights for cloud deployments where *.pt files are excluded from Git."""
EMBEDDED_MODEL_BASE64 = "{b64}"
'''

out_path.write_text(content, encoding="utf-8")
print(f"Generated {out_path} ({len(b64):,} chars from {len(raw):,} bytes)")
