"""Minimal runtime: module packs on disk, and a session that loads and unloads them.

This is the reference for "keep functions separate and load them dynamically":

    Library   a folder with one sub-folder per module pack (manifest.yaml + weight files)
    Session   a core, a controller block and the modules that are resident right now.
              load() and unload() may be called between any two tokens. Session state is kept in
              registers and in the text, never inside a module, so unloading loses nothing.

Not here yet (pipeline stage S8): memory tiers, prefetch, eviction by budget, the dwell rule.
Every load and unload is recorded in Session.events, which is what a residency monitor shows.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from ..bodies import BODIES
from ..registry import manifest as mf

CONTROLLER_DIR = "_controller"
PROJECT_LICENSE = "Apache-2.0"  # license written into the packs this project builds (decision D-006)


class Library:
    def __init__(self, root: "str | Path") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def ids(self) -> "list[str]":
        return sorted(p.name for p in self.root.iterdir() if (p / "manifest.yaml").exists())

    def manifest(self, module_id: str) -> dict:
        return mf.read(self.root / module_id / "manifest.yaml")

    def problems(self) -> "dict[str, list[str]]":
        return {i: p for i in self.ids() if (p := mf.pack_problems(self.root / i))}

    # ---- writing ----------------------------------------------------------------------------------
    def write_pack(self, module_id: str, body, heads=None, fitted_to: "str | None" = None, builder: str = "",
                   status: str = "prototype", threshold: float = 0.5, tests: str = "") -> Path:
        """Write a pack: the body (16-bit floats) and, if given, the trigger and binder fitted to a core."""
        folder = self.root / module_id
        folder.mkdir(parents=True, exist_ok=True)
        body.save(folder / "body.safetensors")
        circuit = body.circuit
        manifest = {
            "id": module_id, "kind": "register_fn", "function": body.function, "operations": list(body.operations),
            "signature": {"args": [k.value for k in body.signature.args], "result": body.signature.result.value},
            "parent": None,
        }
        files = ["body.safetensors"]
        if heads is not None:
            from safetensors.torch import save_file

            state = {k: v.detach().cpu().contiguous() for k, v in heads.state_dict().items()}
            # binder.safetensors: the pointer queries; trigger.safetensors: the trigger and the module's feature layer
            save_file({k: v for k, v in state.items() if not k.startswith("query.")}, str(folder / "trigger.safetensors"))
            save_file({k: v for k, v in state.items() if k.startswith("query.")}, str(folder / "binder.safetensors"))
            manifest["trigger"] = {"file": "trigger.safetensors", "threshold": threshold}
            manifest["binder"] = {"file": "binder.safetensors", "pointers": len(body.signature.args)}
            manifest["heads_config"] = dict(heads.config)
            files += ["trigger.safetensors", "binder.safetensors"]
        manifest.update({
            "body": {"file": "body.safetensors", "route": "programmed", "format": "flow-circuit-1", "dtype": "float16"},
            "portable_body": True, "fitted_to": fitted_to,
            "swap": {"any_token": True, "kv_write": False},
            "cost": {"params": int(circuit.n_params), "nonzero": int(circuit.n_nonzero),
                     "bytes": int(sum((folder / f).stat().st_size for f in files)),
                     "flops_per_call": float(2 * circuit.n_nonzero)},
            "tests": tests or f"tests/test_{'add_exact' if body.function == 'add_exact' else 'compare'}.py",
            "provenance": {"builder": builder or "flow.runtime.minimal", "license": PROJECT_LICENSE,
                           "built": datetime.now(timezone.utc).strftime("%Y-%m-%d")},
            "status": status, "version": "1.0.0",
            "sha256": {f: mf.file_hash(folder / f) for f in files},
        })
        found = mf.problems(manifest)
        if found:
            raise ValueError(f"manifest of {module_id} is not valid: {found}")
        mf.write(manifest, folder / "manifest.yaml")
        return folder

    def write_controller(self, controller, fitted_to: str) -> Path:
        from safetensors.torch import save_file

        folder = self.root / CONTROLLER_DIR
        folder.mkdir(parents=True, exist_ok=True)
        save_file({k: v.detach().cpu().contiguous() for k, v in controller.state_dict().items()}, str(folder / "controller.safetensors"))
        info = {"config": controller.config, "fitted_to": fitted_to, "sha256": mf.file_hash(folder / "controller.safetensors")}
        (folder / "controller.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        return folder

    # ---- reading ----------------------------------------------------------------------------------
    def load_controller(self):
        from safetensors.torch import load_file

        from ..function_layer.model import ControllerBlock

        folder = self.root / CONTROLLER_DIR
        info = json.loads((folder / "controller.json").read_text(encoding="utf-8"))
        controller = ControllerBlock(**info["config"])
        controller.load_state_dict(load_file(str(folder / "controller.safetensors")))
        return controller.eval(), info["fitted_to"]

    def load_module(self, module_id: str):
        """Returns (manifest, body, heads). Verifies the recorded hashes first."""
        from safetensors.torch import load_file

        from ..function_layer.model import FunctionHeads

        folder = self.root / module_id
        found = mf.pack_problems(folder)
        if found:
            raise RuntimeError(f"module pack {module_id} is damaged or incomplete: {found}")
        manifest = mf.read(folder / "manifest.yaml")
        if manifest["function"] not in BODIES:
            raise RuntimeError(f"no body class for function {manifest['function']!r}")
        body = BODIES[manifest["function"]].load(folder / manifest["body"]["file"])
        heads = None
        if "trigger" in manifest:
            heads = FunctionHeads(**manifest["heads_config"])
            state = load_file(str(folder / manifest["trigger"]["file"]))
            state.update(load_file(str(folder / manifest["binder"]["file"])))
            heads.load_state_dict(state)
        return manifest, body, heads


class Session:
    def __init__(self, core, tokenizer, library: Library, core_id: "str | None" = None) -> None:
        from ..function_layer.loop import FunctionLayer

        self.core, self.tokenizer, self.library = core, tokenizer, library
        controller, self.fitted_to = library.load_controller()
        if core_id is not None and core_id != self.fitted_to:
            raise RuntimeError(f"this library was fitted to {self.fitted_to}, not to {core_id}")
        self.layer = FunctionLayer(controller)
        self.events: "list[dict]" = []
        self._bytes: "dict[str, int]" = {}
        self._calls: "dict[str, int]" = {}  # calls made by modules that were unloaded since

    def resident(self) -> "list[str]":
        return sorted(self.layer.modules)

    def resident_bytes(self) -> int:
        return sum(self._bytes.values())

    def load(self, module_id: str) -> dict:
        """Read a pack from disk and attach it. Returns timing and size."""
        if module_id in self.layer.modules:
            return {"module": module_id, "ms": 0.0, "bytes": self._bytes[module_id], "already_resident": True}
        t0 = time.perf_counter()
        manifest, body, heads = self.library.load_module(module_id)
        if heads is None:
            raise RuntimeError(f"{module_id} has a body but no trigger and binder: it is not fitted to any core yet")
        if manifest["fitted_to"] != self.fitted_to:
            raise RuntimeError(f"{module_id} was fitted to {manifest['fitted_to']}, the controller block to {self.fitted_to}")
        self.layer.attach(module_id, heads, body, manifest["trigger"]["threshold"])
        ms = (time.perf_counter() - t0) * 1000
        self._bytes[module_id] = manifest["cost"]["bytes"]
        event = {"time": time.time(), "action": "load", "module": module_id, "ms": round(ms, 2), "bytes": manifest["cost"]["bytes"],
                 "resident_bytes": self.resident_bytes()}
        self.events.append(event)
        return event

    def usage(self) -> "dict[str, int]":
        """Calls made by every module during this session (input to the registry's usage table)."""
        counts = dict(self._calls)
        for module_id, resident in self.layer.modules.items():
            counts[module_id] = counts.get(module_id, 0) + resident.calls
        return counts

    def unload(self, module_id: str) -> None:
        if module_id in self.layer.modules:
            self._calls[module_id] = self._calls.get(module_id, 0) + self.layer.modules[module_id].calls
            self.layer.detach(module_id)
            self._bytes.pop(module_id, None)
            self.events.append({"time": time.time(), "action": "unload", "module": module_id, "resident_bytes": self.resident_bytes()})
