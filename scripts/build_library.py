#!/usr/bin/env python3
"""Write the bodies of the exact functions as module packs into library/.

    python scripts/build_library.py

The weights are constructed, not trained, so this is deterministic and takes about a second.
The packs contain a body only. A trigger and a binder are added when a pack is fitted to a core
(scripts/toy_walkthrough.py does that for a toy core and writes the fitted packs to runs/toy/library).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from flow.bodies import AddExact, CompareNumeric
    from flow.runtime.minimal import Library

    library = Library(ROOT / "library")
    for body in (AddExact(), CompareNumeric()):
        folder = library.write_pack(f"fn.{body.function}.v1", body, builder="scripts/build_library.py")
        m = library.manifest(folder.name)
        print(f"{m['id']}: {m['cost']['params']:,} weights ({m['cost']['nonzero']:,} non-zero), {m['cost']['bytes']:,} bytes, "
              f"sha256 {m['sha256']['body.safetensors'][:16]}")
    problems = library.problems()
    print("all packs valid" if not problems else problems)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
