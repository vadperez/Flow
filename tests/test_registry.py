"""Manifests, pack checks, the registry and its lifecycle rules (pipeline step S1.4, spec section 5)."""
import json
import shutil
from pathlib import Path

import pytest
import yaml

from flow.bodies import CompareNumeric
from flow.registry import manifest as mf
from flow.registry.__main__ import main as cli
from flow.registry.registry import Registry, RegistryError
from flow.runtime.minimal import Library

ROOT = Path(__file__).resolve().parents[1]

ANNEX_C = """
id: fn.add_exact.v1
kind: register_fn
function: add_exact
signature: {args: [NUMBER, NUMBER], result: NUMBER}
parent: null
trigger: {file: trigger.safetensors, threshold: 0.71}
binder: {file: binder.safetensors, pointers: 2}
body: {file: body.safetensors, route: programmed}
portable_body: true
fitted_to: "core-release-1.0@sha256:abc"
swap: {any_token: true, kv_write: false}
cost: {params: 4529079, bytes: 9059022, flops_per_call: 290000}
tests: tests/functions/add_exact/
provenance: {builder: agent-PRG, license: Apache-2.0}
status: prototype
version: 1.0.0
sha256: abc
---
id: fn.fact_lookup.v1
kind: lookup_fn
function: fact_lookup
signature: {args: [SYMBOL, SYMBOL], result: [SYMBOL, NUMBER]}
packs: [kb.general.v1, kb.domain.v1]
portable_body: true
fitted_to: "core-release-1.0@sha256:abc"
swap: {any_token: true, kv_write: false, paging: by_entity}
provenance: {builder: agent-PRG, license: Apache-2.0}
status: proposed
version: 1.0.0
---
id: fn.extract_dosing.clinical.v1
kind: delta_skill
function: extract_slots
domain: biomed.clinical_text
parent: fn.extract_slots.v1
surfaces: {layers: [loop.0, loop.1, loop.2, loop.3], matrices: [query, output, mlp_in, mlp_out], rank: 16}
portable_body: false
fitted_to: "core-release-1.0@sha256:abc"
swap: {any_token: true, kv_write: false}
provenance: {builder: agent-TRN, license: Apache-2.0}
status: proposed
version: 1.0.0
"""


def test_schema_accepts_the_pipeline_examples():
    for manifest in yaml.safe_load_all(ANNEX_C):
        assert mf.problems(manifest) == []


def test_schema_rejects_bad_manifests():
    good = next(yaml.safe_load_all(ANNEX_C))
    assert mf.problems(dict(good, swap={"any_token": True, "kv_write": True}))  # modules never write the key/value cache
    assert mf.problems(dict(good, status="finished"))
    assert mf.problems(dict(good, id="add exact"))
    assert mf.problems({k: v for k, v in good.items() if k != "fitted_to"})  # a trigger without the core it was fitted to
    skill = list(yaml.safe_load_all(ANNEX_C))[2]
    assert mf.problems(dict(skill, portable_body=True))  # a delta skill is never portable across cores


def test_repository_library_is_valid():
    library = Library(ROOT / "library")
    assert library.ids() == ["fn.add_exact.v1", "fn.compare_numeric.v1"]
    assert library.problems() == {}


@pytest.fixture()
def packs(tmp_path):
    library = Library(tmp_path / "library")
    library.write_pack("fn.compare_numeric.v1", CompareNumeric(), builder="test")
    return library


def test_pack_damage_is_detected(packs):
    folder = packs.root / "fn.compare_numeric.v1"
    assert mf.pack_problems(folder) == []
    with open(folder / "body.safetensors", "r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"\x01")
    assert any("sha256" in p for p in mf.pack_problems(folder))
    with pytest.raises(RuntimeError, match="damaged"):
        packs.load_module("fn.compare_numeric.v1")


def test_registry_lifecycle(packs, tmp_path):
    registry = Registry(tmp_path / "registry.sqlite")
    added, failed = registry.scan(packs.root)
    assert added == ["fn.compare_numeric.v1"] and not failed
    assert registry.list()[0]["status"] == "prototype"
    with pytest.raises(RegistryError, match="lifecycle allows"):
        registry.set_status("fn.compare_numeric.v1", "suppressed")  # a prototype cannot be suppressed
    registry.set_status("fn.compare_numeric.v1", "active", "passed its tests")
    registry.set_status("fn.compare_numeric.v1", "suppressed", "unused for 30 days")
    assert mf.read(packs.root / "fn.compare_numeric.v1" / "manifest.yaml")["status"] == "suppressed"
    registry.record_usage("fn.compare_numeric.v1", 5, "2026-11-10")
    registry.record_usage("fn.compare_numeric.v1", 7, "2026-11-10")
    info = registry.show("fn.compare_numeric.v1")
    assert info["calls_total"] == 12 and [e["event"] for e in info["events"]] == ["added", "status", "status"]
    assert registry.verify() == {}
    data = registry.export(tmp_path / "out" / "registry.json")
    assert data["by_status"] == {"suppressed": 1}
    assert json.loads((tmp_path / "out" / "registry.json").read_text())["modules"][0]["id"] == "fn.compare_numeric.v1"
    shutil.rmtree(packs.root / "fn.compare_numeric.v1")
    assert "gone" in registry.verify()["fn.compare_numeric.v1"][0]
    registry.close()


def test_command_line(packs, tmp_path, capsys):
    db = str(tmp_path / "r.sqlite")
    assert cli(["--db", db, "scan", str(packs.root)]) == 0
    assert cli(["--db", db, "status", "fn.compare_numeric.v1", "archived"]) == 0
    assert cli(["--db", db, "status", "fn.compare_numeric.v1", "active"]) == 1  # archived is final
    assert cli(["--db", db, "verify"]) == 0
    assert cli(["--db", db, "list"]) == 0
    out = capsys.readouterr().out
    assert "final state" in out and "fn.compare_numeric.v1" in out
