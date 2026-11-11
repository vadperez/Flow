#!/usr/bin/env python3
"""Pack the project into one zip file, without outputs and caches.

    python ops/make_archive.py                       # ../flow-starter-<version>.zip
    python ops/make_archive.py --out D:/backup/flow.zip
    python ops/make_archive.py --with-runs           # also runs/ (cores, fitted packs, reports)

Left out: .git, caches, models/ (fetch them again), registry/*.sqlite and reports/ (rebuilt by
the tools), ops/status.json unless --with-state is given. Needs only the standard library.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".hypothesis", ".pytest_cache", "__pycache__", "models", ".venv", "venv"}
KEEP_EMPTY = {".gitkeep"}


def wanted(path: Path, with_runs: bool, with_state: bool) -> bool:
    rel = path.relative_to(ROOT)
    parts = rel.parts
    if any(p in SKIP_DIRS or p.endswith(".egg-info") for p in parts) or path.suffix == ".pyc":
        return False
    if parts[0] == "runs":
        return with_runs
    if parts[0] == "reports" or (parts[0] == "registry" and path.suffix.startswith(".sqlite")):
        return path.name in KEEP_EMPTY
    if rel.as_posix() == "ops/status.json":
        return with_state
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=None, help="zip file to write")
    parser.add_argument("--with-runs", action="store_true", help="include runs/")
    parser.add_argument("--with-state", action="store_true", help="include ops/status.json")
    args = parser.parse_args()

    version = "0.0"
    match = re.search(r'^version\s*=\s*"([^"]+)"', (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.M)
    if match:
        version = match.group(1)
    out = Path(args.out) if args.out else ROOT.parent / f"{ROOT.name}-{version}.zip"
    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and wanted(p, args.with_runs, args.with_state))
    if out.resolve() in {p.resolve() for p in files}:
        print("The archive cannot be written inside the folders it packs. Choose another --out.")
        return 1
    raw = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for p in files:
            archive.write(p, (Path(ROOT.name) / p.relative_to(ROOT)).as_posix())
            raw += p.stat().st_size
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"{out}  {len(files)} files, {raw / 1e6:.1f} MB unpacked, {out.stat().st_size / 1e6:.2f} MB packed")
    print(f"sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
