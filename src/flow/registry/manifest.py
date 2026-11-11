"""Module manifests: reading, writing, validating against specs/manifest.schema.json, hashing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

SCHEMA_PATH = Path(__file__).resolve().parents[3] / "specs" / "manifest.schema.json"
STATES = ["proposed", "prototype", "active", "differentiating", "suppressed", "eliminated", "archived"]
# Lifecycle of spec section 5: which status may follow which.
TRANSITIONS = {
    "proposed": {"prototype", "archived"},
    "prototype": {"active", "eliminated", "archived"},
    "active": {"differentiating", "suppressed", "eliminated"},
    "differentiating": {"active", "suppressed"},
    "suppressed": {"active", "eliminated"},
    "eliminated": {"archived"},
    "archived": set(),
}


def schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def problems(manifest: dict) -> "list[str]":
    """Human-readable list of what is wrong with a manifest (empty when valid)."""
    try:
        import jsonschema
    except ImportError:
        return ["cannot validate: the package 'jsonschema' is not installed (pip install jsonschema)"]
    validator = jsonschema.Draft202012Validator(schema())
    return [f"{'/'.join(map(str, e.path)) or '(manifest)'}: {e.message}" for e in sorted(validator.iter_errors(manifest), key=str)]


def read(path: "str | Path") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def write(manifest: dict, path: "str | Path") -> None:
    Path(path).write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")


def file_hash(path: "str | Path") -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pack_problems(folder: "str | Path") -> "list[str]":
    """Check one pack folder: the manifest is valid, the named files exist and their hashes match."""
    folder = Path(folder)
    if not (folder / "manifest.yaml").exists():
        return [f"{folder}: no manifest.yaml"]
    manifest = read(folder / "manifest.yaml")
    found = problems(manifest)
    if manifest.get("id") != folder.name:
        found.append(f"id {manifest.get('id')!r} does not match the folder name {folder.name!r}")
    hashes = manifest.get("sha256") if isinstance(manifest.get("sha256"), dict) else {}
    for part in ("trigger", "binder", "body"):
        name = (manifest.get(part) or {}).get("file")
        if not name:
            continue
        if not (folder / name).exists():
            found.append(f"{part}: file {name} is missing")
        elif name not in hashes:
            found.append(f"{part}: no sha256 recorded for {name}")
        elif hashes[name] != file_hash(folder / name):
            found.append(f"{part}: {name} does not match its recorded sha256")
    return found
