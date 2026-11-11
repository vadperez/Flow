#!/usr/bin/env python3
"""Fetch the files of a published model, by default WITHOUT its trained weights.

    python scripts/fetch_model_files.py --model Qwen/Qwen3.5-0.8B --dry-run        # list files and sizes
    python scripts/fetch_model_files.py --model Qwen/Qwen3.5-0.8B                  # config + tokenizer, about 23 MB
    python scripts/fetch_model_files.py --model Qwen/Qwen3.5-4B --with-weights     # everything, about 9.3 GB

Files land in models/<name>/. Without weights you get everything needed to build the network
with random weights (scripts/make_blank_model.py) and to check the tokenizer
(scripts/check_tokenizer.py). A frozen "borrowed" core needs --with-weights.

Needs: pip install huggingface_hub   (already there if Transformers is installed)
Limits: downloads without an account are rate-limited by huggingface.co. If you hit the limit,
create a free read token there and set the environment variable HF_TOKEN before running.
The token is read from the environment only and is never printed or stored by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHT_PATTERNS = ["*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt", "*.gguf", "*.onnx", "*.h5", "*.msgpack", "*.ot",
                   "*.tflite", "*.mlmodel", "*.npz"]


def human(n: float) -> str:
    for unit in ("B", "kB", "MB", "GB"):
        if n < 1000 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1000
    return f"{n:.2f} GB"


def explain(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}"
    low = text.lower()
    if any(w in low for w in ("proxy", "connection", "timed out", "ssl", "resolve", "unreachable")):
        return text + ("\n-> huggingface.co cannot be reached from this machine. Check the connection; a proxy, a firewall or a "
                       "security suite may be blocking python.exe: allow it for outgoing HTTPS.")
    if "429" in low or "rate limit" in low or "too many requests" in low:
        return text + "\n-> huggingface.co is rate-limiting this address. Wait a few minutes, or set HF_TOKEN (free read token)."
    if "401" in low or "403" in low or "gated" in low:
        return text + "\n-> access refused. Open the model page in a browser, accept its terms if asked, and set HF_TOKEN."
    if "404" in low or "not found" in low:
        return text + "\n-> no such model. Check the spelling of --model (for example Qwen/Qwen3.5-0.8B)."
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B", help="repository id on huggingface.co")
    parser.add_argument("--revision", default=None, help="commit hash or tag to pin (recommended once the project runs)")
    parser.add_argument("--out", default=None, help="target folder (default: models/<name>)")
    parser.add_argument("--with-weights", action="store_true", help="also download the trained weights")
    parser.add_argument("--dry-run", action="store_true", help="list files and sizes, download nothing")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("This script needs the package huggingface_hub:  pip install huggingface_hub")
        return 2
    name = args.model.split("/")[-1]
    target = Path(args.out) if args.out else ROOT / "models" / name
    ignore = None if args.with_weights else WEIGHT_PATTERNS
    common = dict(repo_id=args.model, revision=args.revision, ignore_patterns=ignore)

    try:
        plan = snapshot_download(dry_run=True, **common)
    except TypeError:
        plan = None  # an older huggingface_hub without dry_run
    except Exception as exc:  # noqa: BLE001 - every failure gets a readable explanation
        print(explain(exc))
        return 1
    if plan is not None:
        total = sum(f.file_size or 0 for f in plan)
        print(f"{args.model}: {len(plan)} files, {human(total)} " + ("(with weights)" if args.with_weights else "(weights left out)"))
        for f in sorted(plan, key=lambda f: f.filename):
            print(f"  {human(f.file_size or 0):>10}  {f.filename}")
        free = shutil.disk_usage(target.parent if target.parent.exists() else ROOT).free
        if total * 1.1 > free:
            print(f"Not enough free disk space: need about {human(total * 1.1)}, have {human(free)}.")
            return 1
        if args.dry_run:
            return 0
    elif args.dry_run:
        print("This huggingface_hub version cannot list without downloading. Update it:  pip install -U huggingface_hub")
        return 2

    try:
        snapshot_download(local_dir=str(target), **common)
    except Exception as exc:  # noqa: BLE001
        print(explain(exc))
        return 1

    files = sorted(p for p in target.rglob("*") if p.is_file() and ".cache" not in p.parts and p.name != "FILES.json")
    listing = []
    for p in files:
        digest = hashlib.sha256()
        with open(p, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        listing.append({"file": str(p.relative_to(target)).replace("\\", "/"), "bytes": p.stat().st_size, "sha256": digest.hexdigest()})
    (target / "FILES.json").write_text(json.dumps({"model": args.model, "revision": args.revision, "with_weights": args.with_weights,
                                                   "files": listing}, indent=1), encoding="utf-8")
    print(f"\n{len(listing)} files in {target} ({human(sum(f['bytes'] for f in listing))}); list with hashes: FILES.json")
    config = target / "config.json"
    if config.exists() and name.lower().startswith("qwen3.5"):
        copy = ROOT / "configs" / "qwen3_5" / f"{name}.config.json"
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(config, copy)
        print(f"published configuration copied to {copy.relative_to(ROOT)}")
    if (target / "LICENSE").exists():
        print("License file kept next to the model files. Keep it with any copy or modified version you pass on.")
    print("Next:  python scripts/check_tokenizer.py --model", target.relative_to(ROOT) if target.is_relative_to(ROOT) else target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
