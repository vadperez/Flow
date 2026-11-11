#!/usr/bin/env python3
"""Pipeline status tool for the orchestrator (pipeline Annex D, rule 1).

    python ops/next_steps.py                      # what is done, what can start now, what waits for a gate
    python ops/next_steps.py --done S0.1 S0.2     # mark steps as done
    python ops/next_steps.py --skip S5B.3 --note "no GPU hours left"     # skip an optional step (counts as done)
    python ops/next_steps.py --gate G1 go --note "approved"       # record a gate decision: go | stop | pivot
    python ops/next_steps.py --undo S0.2          # back to pending
    python ops/next_steps.py --json               # machine-readable summary (also what the dashboard reads)

Reads specs/task_graph.yaml, keeps the state in ops/status.json. Needs only PyYAML.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRAPH, STATUS = ROOT / "specs" / "task_graph.yaml", ROOT / "ops" / "status.json"


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load():
    try:
        import yaml
    except ImportError:
        sys.exit("This tool needs PyYAML:  pip install pyyaml")
    graph = yaml.safe_load(GRAPH.read_text(encoding="utf-8"))
    status = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {"steps": {}, "gates": {}, "log": []}
    return graph, status


def summary(graph: dict, status: dict) -> dict:
    steps = {s["id"]: s for s in graph["steps"]}
    state = lambda i: status["steps"].get(i, {}).get("state", "pending")
    finished = {i for i in steps if state(i) in ("done", "skipped")}
    passed = {g for g, v in status["gates"].items() if v.get("decision") == "go"}
    stopped = {g: v["decision"] for g, v in status["gates"].items() if v.get("decision") in ("stop", "pivot")}
    ready, waiting_gate, blocked = [], [], []
    for i, s in steps.items():
        if i in finished:
            continue
        missing = [n for n in s["needs"] if n not in finished and n not in passed]
        if not missing:
            ready.append(i)
        elif all(n in graph["gates"] for n in missing):
            waiting_gate.append((i, missing))
        else:
            blocked.append(i)
    open_gates = [g for g, v in graph["gates"].items() if g not in status["gates"] and all(a in finished for a in v["after"])]
    stages = {}
    for i in steps:
        stage = i.split(".")[0].rstrip("ABCDEF")
        entry = stages.setdefault(stage, {"title": graph.get("stages", {}).get(stage, ""), "total": 0, "done": 0})
        entry["total"] += 1
        entry["done"] += i in finished
    return {"updated": status.get("updated"), "total": len(steps), "done": len(finished), "ready": ready,
            "ready_info": [{"id": i, "title": steps[i]["title"], "agent": steps[i]["agent"]} for i in ready],
            "waiting_for_gate": [{"step": i, "gates": m} for i, m in waiting_gate], "blocked": len(blocked),
            "gates": {g: status["gates"].get(g, {"decision": "open" if g in open_gates else "not reached"}) for g in graph["gates"]},
            "gates_awaiting_decision": open_gates, "stopped": stopped, "stages": stages,
            "steps": {i: {"title": s["title"], "agent": s["agent"], "state": state(i), "optional": bool(s.get("optional"))} for i, s in steps.items()}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--done", nargs="+", default=[], metavar="STEP")
    parser.add_argument("--skip", nargs="+", default=[], metavar="STEP")
    parser.add_argument("--undo", nargs="+", default=[], metavar="STEP")
    parser.add_argument("--gate", nargs=2, metavar=("GATE", "DECISION"))
    parser.add_argument("--note", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    graph, status = load()
    steps = {s["id"]: s for s in graph["steps"]}
    changed = False
    for ids, new in ((args.done, "done"), (args.skip, "skipped"), (args.undo, "pending")):
        for i in ids:
            if i not in steps:
                sys.exit(f"unknown step {i}. Known ids are in specs/task_graph.yaml.")
            if new == "skipped" and not steps[i].get("optional"):
                sys.exit(f"{i} is not optional and cannot be skipped.")
            if new == "done":
                before = summary(graph, status)
                if i not in before["ready"] and status["steps"].get(i, {}).get("state") != "done":
                    missing = [n for n in steps[i]["needs"] if n not in {k for k, v in status['steps'].items() if v['state'] in ('done', 'skipped')}
                               and status["gates"].get(n, {}).get("decision") != "go"]
                    sys.exit(f"{i} cannot be done yet: it needs {', '.join(missing)}.")
            status["steps"][i] = {"state": new, "time": now(), "note": args.note}
            status["log"].append({"time": now(), "step": i, "state": new, "note": args.note})
            changed = True
    if args.gate:
        gate, decision = args.gate
        if gate not in graph["gates"] or decision not in ("go", "stop", "pivot"):
            sys.exit("usage: --gate G1|G2|G3|G4 go|stop|pivot")
        if gate not in summary(graph, status)["gates_awaiting_decision"] and gate not in status["gates"]:
            sys.exit(f"{gate} is not reached yet: it comes after {', '.join(graph['gates'][gate]['after'])}.")
        status["gates"][gate] = {"decision": decision, "time": now(), "note": args.note}
        status["log"].append({"time": now(), "gate": gate, "decision": decision, "note": args.note})
        changed = True
    if changed or not STATUS.exists():
        status["updated"] = now()
        status["summary"] = {k: v for k, v in summary(graph, status).items() if k != "steps"}
        STATUS.write_text(json.dumps(status, indent=1), encoding="utf-8")
    info = summary(graph, status)
    if args.json:
        print(json.dumps(info, indent=1))
        return 0
    print(f"Flow pipeline v{graph['version']}: {info['done']} of {info['total']} steps done")
    for stage, e in info["stages"].items():
        bar = "#" * e["done"] + "." * (e["total"] - e["done"])
        print(f"  {stage:4} {bar:12} {e['done']}/{e['total']}  {e['title']}")
    print("Gates: " + ", ".join(f"{g} {v['decision']}" for g, v in info["gates"].items()))
    for g in info["gates_awaiting_decision"]:
        print(f"  -> {g} is waiting for your decision (prepare the packet: ops/gate_packet_template.md)")
    print("Can start now:")
    for i in info["ready"]:
        s = steps[i]
        print(f"  {i:6} [{'+'.join(s['agent']):11}] {s['title']}" + ("  (optional)" if s.get("optional") else ""))
    if info["waiting_for_gate"]:
        waiting = {}
        for w in info["waiting_for_gate"]:
            waiting.setdefault("+".join(w["gates"]), []).append(w["step"])
        for gate, ids in waiting.items():
            print(f"Waiting only for {gate}: {', '.join(ids)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
