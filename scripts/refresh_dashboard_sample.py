#!/usr/bin/env python3
"""Put your own latest data into the console mockup, so that it shows it when opened as a file.

    python scripts/refresh_dashboard_sample.py

Reads, where they exist:
    ops/status.json                                    (python ops/next_steps.py)
    reports/registry.json, else runs/toy/registry.json (python -m flow.registry export reports/registry.json)
    runs/toy/report.json                               (python scripts/toy_walkthrough.py)
and rewrites the embedded sample inside dashboards/mockups/index.html. Nothing else changes.
A page opened from disk cannot read neighbouring files by itself; this script, the "Open files"
button, or serving the folder (python -m http.server, then /dashboards/mockups/) are the three ways in.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "dashboards" / "mockups" / "index.html"
START, END = "/*SAMPLE-START*/", "/*SAMPLE-END*/"


def read(*candidates: str):
    for name in candidates:
        path = ROOT / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8")), name
    return None, None


def main() -> int:
    if not (ROOT / "ops" / "status.json").exists():
        subprocess.run([sys.executable, str(ROOT / "ops" / "next_steps.py"), "--json"], stdout=subprocess.DEVNULL, check=False)
    status, status_from = read("ops/status.json")
    registry, registry_from = read("reports/registry.json", "runs/toy/registry.json")
    report, report_from = read("runs/toy/report.json")
    if status is not None:
        status = {"gates": status.get("gates", {}), "summary": status.get("summary", {})}  # the log stays out
    if report is not None:
        report = {"dynamic_loading": report.get("dynamic_loading", {})}  # the part the console shows
    sample = json.dumps({"status": status, "registry": registry, "report": report}, ensure_ascii=False).replace("</", "<\\/")
    page = PAGE.read_text(encoding="utf-8")
    if START not in page or END not in page:
        print(f"{PAGE} has no sample markers; it was not changed.")
        return 1
    head, rest = page.split(START, 1)
    PAGE.write_text(head + START + sample + END + rest.split(END, 1)[1], encoding="utf-8")
    for label, source in (("pipeline status", status_from), ("module registry", registry_from), ("session report", report_from)):
        print(f"  {label:16} {source or 'not found: the panel will say how to produce it'}")
    print(f"written: {PAGE.relative_to(ROOT)} ({len(sample):,} characters of data)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
