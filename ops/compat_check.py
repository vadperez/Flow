#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Check whether this machine can run the Flow starter, and which parts.

    python ops/compat_check.py            # table of findings
    python ops/compat_check.py --json     # the same as JSON (written to reports/compat.json as well)

This file is written to run on old Python versions too, so that an unsuitable machine gets a
clear list of what is missing instead of a crash. It changes nothing on the machine.
"""
from __future__ import print_function

import json
import os
import platform
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIN_PY, GOOD_PY = (3, 11), ((3, 12), (3, 13))
NEEDS = {  # package: (minimum version, tier, pip name)
    "numpy": ("2.0", "base", "numpy"), "yaml": ("6.0", "base", "pyyaml"), "jsonschema": ("4.20", "base", "jsonschema"),
    "safetensors": ("0.5", "base", "safetensors"),
    "torch": ("2.7", "model", "torch"), "transformers": ("5.22", "model", "transformers"),
    "tokenizers": ("0.21", "model", "tokenizers"), "huggingface_hub": ("1.0", "model", "huggingface_hub"),
    "pytest": ("8.0", "dev", "pytest"), "hypothesis": ("6.100", "dev", "hypothesis"),
}


def version_tuple(text):
    parts = []
    for piece in str(text).split("+")[0].split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def memory_bytes():
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo") as handle:
                for line in handle:
                    if line.startswith("MemTotal"):
                        return int(line.split()[1]) * 1024
        if sys.platform == "win32":
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("avail", ctypes.c_ulonglong), ("page_total", ctypes.c_ulonglong), ("page_avail", ctypes.c_ulonglong),
                            ("virtual_total", ctypes.c_ulonglong), ("virtual_avail", ctypes.c_ulonglong),
                            ("extended", ctypes.c_ulonglong)]

            status = Status()
            status.length = ctypes.sizeof(Status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return int(status.total)
        if sys.platform == "darwin":
            import subprocess

            return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())
    except Exception:  # noqa: BLE001 - the check must never crash
        pass
    return None


def main():
    rows, facts = [], {}

    def add(name, found, needed, verdict, note=""):
        rows.append((name, str(found), str(needed), verdict, note))
        facts[name] = {"found": str(found), "needed": str(needed), "verdict": verdict, "note": note}

    # ---- Python and machine -----------------------------------------------------------------------
    py = sys.version_info[:3]
    machine = platform.machine() or "?"
    bits = 64 if sys.maxsize > 2 ** 32 else 32
    if py[:2] < MIN_PY:
        add("Python", "%d.%d.%d" % py, ">= 3.11 (3.12 recommended)", "MISSING",
            "PyTorch 2.15 and Transformers 5.22 need Python 3.11 or newer. Install Python 3.12 (64-bit).")
    elif py[:2] in GOOD_PY:
        add("Python", "%d.%d.%d" % py, ">= 3.11 (3.12 recommended)", "OK")
    else:
        add("Python", "%d.%d.%d" % py, ">= 3.11 (3.12 recommended)", "WARNING",
            "Works where wheels exist; 3.12 and 3.13 are the versions with wheels for every dependency on every target.")
    add("Python build", "%d-bit" % bits, "64-bit", "OK" if bits == 64 else "MISSING")
    system = "%s %s" % (platform.system(), platform.release())
    add("System", "%s, %s" % (system, machine), "Windows 10/11, Linux or macOS, 64-bit", "OK")
    if sys.platform == "win32":
        native = os.environ.get("PROCESSOR_ARCHITEW6432") or os.environ.get("PROCESSOR_ARCHITECTURE", machine)
        if "ARM" in native.upper() and "ARM" not in machine.upper():
            add("Windows on Arm", "x64 Python under emulation", "ARM64 build of Python", "WARNING",
                "Install the ARM64 build of Python 3.12 or 3.13; PyTorch has native ARM64 wheels (CPU only) on its own index.")
        elif "ARM" in machine.upper():
            add("Windows on Arm", "native ARM64 Python", "ARM64 build of Python", "OK",
                "Install PyTorch with: pip install torch --index-url https://download.pytorch.org/whl/cpu (CPU only).")
        add("Security software", "not checked", "-", "NOTE",
            "If a script is blocked or 'contained' by security software, add this folder and python.exe "
            "to the trusted applications, and allow python.exe outgoing HTTPS for downloads.")

    # ---- memory and disk --------------------------------------------------------------------------
    ram = memory_bytes()
    if ram is None:
        add("Memory", "unknown", ">= 8 GB", "WARNING")
    else:
        gb = ram / 1e9
        add("Memory", "%.1f GB" % gb, ">= 8 GB", "OK" if gb >= 7.5 else "WARNING",
            "" if gb >= 7.5 else "Enough for tests and the toy walkthrough; too little for a borrowed core.")
    free = shutil.disk_usage(ROOT).free / 1e9
    add("Free disk here", "%.1f GB" % free, ">= 2 GB (12 GB with Qwen3.5-4B weights)", "OK" if free >= 2 else "MISSING")

    # ---- packages ---------------------------------------------------------------------------------
    missing = {"base": [], "model": [], "dev": []}
    versions = {}
    for module, (minimum, tier, pip_name) in sorted(NEEDS.items(), key=lambda kv: ("base", "model", "dev").index(kv[1][1])):
        try:
            mod = __import__(module)
            try:
                from importlib import metadata

                found = metadata.version(pip_name)
            except Exception:  # noqa: BLE001 - old Python, or an unusual install
                found = getattr(mod, "__version__", "?")
            versions[module] = found
            ok = version_tuple(found) >= version_tuple(minimum)
            add(pip_name, found, ">= " + minimum + " (" + tier + ")", "OK" if ok else "MISSING", "" if ok else "pip install -U " + pip_name)
            if not ok:
                missing[tier].append(pip_name)
        except Exception as exc:  # noqa: BLE001
            add(pip_name, "not installed", ">= " + minimum + " (" + tier + ")", "MISSING", type(exc).__name__)
            missing[tier].append(pip_name)

    # ---- PyTorch details and the Qwen3.5 model code --------------------------------------------------
    gpu_gb = 0.0
    if "torch" in versions:
        try:
            import torch

            threads = torch.get_num_threads()
            if torch.cuda.is_available():
                gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                add("GPU", "%s, %.1f GB" % (torch.cuda.get_device_name(0), gpu_gb), "optional; >= 12 GB for Qwen3.5-4B", "OK")
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                add("GPU", "Apple MPS", "optional", "OK")
            else:
                add("GPU", "none (CPU, %d threads)" % threads, "optional", "NOTE",
                    "Tests and the toy walkthrough run on CPU. Core pretraining (stage S4) needs a GPU, rented if necessary.")
        except Exception as exc:  # noqa: BLE001
            add("GPU", "check failed", "optional", "WARNING", str(exc)[:80])
    if "transformers" in versions:
        try:
            from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig  # noqa: F401

            add("Qwen3.5 model code", "present", "in Transformers", "OK")
        except Exception:  # noqa: BLE001
            add("Qwen3.5 model code", "absent", "in Transformers", "MISSING", "pip install -U transformers")
            missing["model"].append("transformers")

    # ---- what can run here ------------------------------------------------------------------------
    py_ok = py[:2] >= MIN_PY and bits == 64
    base_ok = py_ok and not missing["base"]
    model_ok = base_ok and not missing["model"]
    ram_gb = (ram or 0) / 1e9
    tiers = [
        ("Type system, written weights, registry", base_ok, "pip install -r ops/requirements-base.txt"),
        ("Tests (pytest)", base_ok and not missing["dev"], "pip install -r ops/requirements-dev.txt"),
        ("Blank cores and the toy walkthrough (CPU)", model_ok and ram_gb >= 3.5, "pip install -r ops/requirements-model.txt"),
        ("Borrowed core Qwen3.5-0.8B (frozen)", model_ok and (ram_gb >= 7.5 or gpu_gb >= 4), "8 GB memory or a 4 GB GPU; 2 GB disk"),
        ("Borrowed core Qwen3.5-4B (frozen)", model_ok and (ram_gb >= 30 or gpu_gb >= 11), "a 12 GB GPU, or 32 GB memory on CPU (slow); 10 GB disk"),
    ]

    if "--json" in sys.argv:
        out = {"checks": facts, "can_run": dict((name, ok) for name, ok, _ in tiers)}
        print(json.dumps(out, indent=1))
    else:
        widths = [max(len(r[i]) for r in rows + [("Check", "Found", "Needed", "Verdict", "")]) for i in range(4)]
        line = "  ".join("%-" + str(w) + "s" for w in widths)
        print(line % ("Check", "Found", "Needed", "Verdict"))
        print(line % tuple("-" * w for w in widths))
        for r in rows:
            print(line % r[:4])
            if r[4]:
                print("      " + r[4])
        print("")
        print("What this machine can run:")
        for name, ok, need in tiers:
            print("  [%s] %s%s" % ("x" if ok else " ", name, "" if ok else "   <- needs: " + need))
    try:
        target = os.path.join(ROOT, "reports")
        if not os.path.isdir(target):
            os.makedirs(target)
        with open(os.path.join(target, "compat.json"), "w") as handle:
            json.dump({"checks": facts, "can_run": dict((name, ok) for name, ok, _ in tiers)}, handle, indent=1)
    except Exception:  # noqa: BLE001
        pass
    return 0 if base_ok else 1


if __name__ == "__main__":
    sys.exit(main())
