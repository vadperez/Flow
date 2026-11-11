#!/usr/bin/env python3
"""The whole Flow mechanism in miniature, on a CPU, in a few minutes.

    1. Build a tiny BLANK core (Qwen3.5-style layers, random weights) and pretrain it on toy text
       with answer masking: it learns the language and where a result is needed, never the result.
    2. Freeze it. Fit the controller block and the heads of add_exact (trigger + binder).
    3. Freeze the controller block too. Add compare_numeric alone, touching nothing else.
    4. Score: exact answers, numbers longer than any seen in training, two calls chained through
       RES, and one pair of calls that was never shown in training (subtract, then greater).
    5. Remove modules and see which answers disappear (separation). Load and unload a module
       between two tokens of one text (dynamic loading). A chain that needs a module which is not
       loaded is dropped: the model abstains, it does not print a half-finished result.

Everything is written to runs/toy/: the core, the fitted module packs, a registry of them
(registry.sqlite, registry.json), report.json, report.md.

    python scripts/toy_walkthrough.py            # about 10 minutes on 2 CPU cores
    python scripts/toy_walkthrough.py --quick    # under a minute: checks that the code runs; scores are meaningless
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(ROOT / "runs" / "toy"))
    parser.add_argument("--quick", action="store_true", help="tiny run that only checks the code path")
    parser.add_argument("--core-steps", type=int, default=900)
    parser.add_argument("--fit-steps", type=int, default=500)
    parser.add_argument("--add-steps", type=int, default=350)
    parser.add_argument("--train-records", type=int, default=6000)
    parser.add_argument("--test-records", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0, help="CPU threads (0 = library default)")
    args = parser.parse_args()
    if args.quick:
        args.core_steps, args.fit_steps, args.add_steps, args.train_records, args.test_records = 30, 20, 15, 400, 60

    try:
        import torch
    except ImportError:
        from flow.core.blank import REQUIREMENTS
        print(REQUIREMENTS)
        return 2
    from flow.bodies import AddExact, CompareNumeric
    from flow.core import blank
    from flow.core.pretrain import pretrain
    from flow.data import align as al
    from flow.data.tokenizer import ToyTokenizer
    from flow.data.world import HELD_OUT, ToyWorld, execute, vocabulary
    from flow.function_layer import evaluate as ev
    from flow.function_layer.loop import FunctionLayer, generate
    from flow.function_layer.model import ControllerBlock, FunctionHeads
    from flow.function_layer.train import FunctionSpec, fit
    from flow.runtime.minimal import Library, Session
    from flow.typesys import PieceMap

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    report = {"settings": vars(args), "torch": torch.__version__}

    def say(title):
        print(f"\n=== {title} " + "=" * max(0, 70 - len(title)))

    # ---- data ------------------------------------------------------------------------------------
    say("1. Data")
    tok = ToyTokenizer.from_texts([], extra=vocabulary())
    worlds = {"train": ToyWorld(args.seed + 1), "test": ToyWorld(args.seed + 2),
              "long": ToyWorld(args.seed + 3, min_digits=8, max_digits=11), "heldout": ToyWorld(args.seed + 4)}
    records = {"train": worlds["train"].records(args.train_records), "test": worlds["test"].records(args.test_records),
               "long": worlds["long"].records(args.test_records), "heldout": worlds["heldout"].records(args.test_records // 2, split="heldout")}
    assert all(execute(r) for rs in records.values() for r in rs), "a generated record does not reproduce"
    assert not any(r["template"] in HELD_OUT for r in records["train"] + records["test"] + records["long"])
    data = {k: [al.align(r, tok) for r in rs] for k, rs in records.items()}
    print(f"vocabulary {len(tok)} tokens; records: " + ", ".join(f"{k} {len(v)}" for k, v in records.items()))
    print("example:", records["train"][0]["text"])
    (out / "sample_records.jsonl").write_text("\n".join(json.dumps(r) for r in records["train"][:50]) + "\n", encoding="utf-8")

    # ---- blank core, answer-masked pretraining ---------------------------------------------------
    say("2. Blank core, pretrained with answer masking")
    spec = blank.CoreSpec(vocab_size=len(tok), hidden_size=128, intermediate_size=256, heads=4, kv_heads=2, head_dim=32,
                          linear_heads=4, linear_head_dim=32, entry="L", loop="F", loop_passes=2, exit="F", max_positions=256,
                          pad_token_id=tok.pad_id, bos_token_id=tok.bos_id, eos_token_id=tok.eos_id)
    core = blank.build(spec, seed=args.seed)
    counts = blank.count_parameters(core)
    print(f"layer passes {spec.pattern} (L = linear attention, F = full attention; the F in the middle runs twice, shared)")
    print(f"parameters: {counts['total']:,} (random to start with)")
    gen = torch.Generator().manual_seed(args.seed)
    examples = [al.lm_example(a, tok.bos_id, tok.eos_id, tok.abstain_id, abstained=bool(torch.rand(1, generator=gen) < 0.15))
                for a in data["train"]]
    t0 = time.time()
    report["core_pretraining"] = pretrain(core, examples, tok.pad_id, tok.abstain_id, steps=args.core_steps, seed=args.seed,
                                          log_every=max(1, args.core_steps // 6))
    report["core"] = {"pattern": spec.pattern, "parameters": counts["total"], "seconds": round(time.time() - t0, 1)}
    blank.save(core, out / "core")
    alone = ev.summarize(ev.answers(core, tok, None, data["test"]), by="function")
    report["core_alone"] = alone
    print(f"core alone on the test set: exact {alone['all']['exact']}/{alone['all']['n']}, "
          f"abstained {alone['all']['abstained']}, wrong {alone['all']['wrong']}")

    # ---- function layer ---------------------------------------------------------------------------
    say("3. Fit the controller block and add_exact (core frozen)")
    bodies = {"add_exact": AddExact(), "compare_numeric": CompareNumeric()}
    specs = {name: FunctionSpec.of(body) for name, body in bodies.items()}
    controller = ControllerBlock(core_width=spec.hidden_size, n_taps=len(spec.pattern) + 1)
    make_heads = lambda s: FunctionHeads(spec.hidden_size, len(spec.pattern) + 1, controller.width, controller.key_dim,
                                         controller.ctrl_dim, len(s.operations), len(s.arg_kinds))
    heads = {name: make_heads(s) for name, s in specs.items()}
    t0 = time.time()
    report["fit_add_exact"] = fit(core, controller, {"add_exact": heads["add_exact"]}, specs, data["train"], tok.bos_id, tok.pad_id,
                                  steps=args.fit_steps, seed=args.seed, log_every=max(1, args.fit_steps // 5))
    say("4. Add compare_numeric alone (core and controller block frozen)")
    report["fit_compare_numeric"] = fit(core, controller, {"compare_numeric": heads["compare_numeric"]}, specs, data["train"],
                                        tok.bos_id, tok.pad_id, steps=args.add_steps, train_controller=False, seed=args.seed + 1,
                                        log_every=max(1, args.add_steps // 4))
    report["fit_seconds"] = round(time.time() - t0, 1)
    n_ctrl = sum(p.numel() for p in controller.parameters())
    n_heads = {k: sum(p.numel() for p in h.parameters()) for k, h in heads.items()}
    print(f"learned parameters: controller block {n_ctrl:,}; heads " + ", ".join(f"{k} {v:,}" for k, v in n_heads.items()))
    report["learned_parameters"] = {"controller_block": n_ctrl, **{f"heads.{k}": v for k, v in n_heads.items()}}

    # ---- module packs on disk, then a session that loads them -----------------------------------------
    library = Library(out / "library")
    core_id = f"toy-core@{blank_hash(out / 'core' / 'model.safetensors')}"
    library.write_controller(controller, fitted_to=core_id)
    for name in bodies:
        library.write_pack(f"fn.{name}.v1", bodies[name], heads[name], fitted_to=core_id, builder="scripts/toy_walkthrough.py")
    session = Session(core, tok, library)
    for module_id in library.ids():
        session.load(module_id)
    layer = session.layer

    # ---- scores -------------------------------------------------------------------------------------
    say("5. Scores with both modules loaded")
    scores = {}
    for name in ("test", "long", "heldout"):
        rows = ev.answers(core, tok, layer, data[name])
        scores[name] = {"by_template": ev.summarize(rows), "by_function": ev.summarize(rows, by="function")}
        s = scores[name]["by_function"]["all"]
        print(f"{name:8s} exact {s['exact']}/{s['n']} = {s['exact_rate']:.3f}   abstained {s['abstained']}   wrong {s['wrong']}")
    scores["false_calls"] = ev.false_calls(core, tok, layer, data["test"])
    print("calls at positions where none belongs:", scores["false_calls"])
    report["scores"] = scores
    print("per template (test | long numbers):")
    for template, s in scores["test"]["by_template"].items():
        long = scores["long"]["by_template"].get(template, {"exact_rate": float("nan")})
        print(f"  {template:9s} {s['exact_rate']:.3f} | {long['exact_rate']:.3f}")

    # ---- separation: remove one module, see what disappears -------------------------------------------
    say("6. Separation: unload one module")
    full = ev.summarize(ev.answers(core, tok, layer, data["test"]), by="function")
    removal = {}
    for gone in ("fn.compare_numeric.v1", "fn.add_exact.v1"):
        session.unload(gone)
        removal[gone] = ev.summarize(ev.answers(core, tok, layer, data["test"]), by="function")
        session.load(gone)
    floor = alone
    report["separation"] = {"all_loaded": full, "removed": removal, "core_alone": floor, "localization": {}}
    for gone, name in (("fn.add_exact.v1", "add_exact"), ("fn.compare_numeric.v1", "compare_numeric")):
        other = "compare_numeric" if name == "add_exact" else "add_exact"
        only_other = lambda rows_by, fn=other, own=name: rows_by  # readability below
        w, wo, fl = full[name]["exact_rate"], removal[gone][name]["exact_rate"], floor[name]["exact_rate"]
        loc = round((w - wo) / (w - fl), 4) if w > fl else float("nan")
        report["separation"]["localization"][name] = loc
        print(f"without {gone}: {name} {w:.3f} -> {wo:.3f} (abstained {removal[gone][name]['abstained']}/{removal[gone][name]['n']}, "
              f"wrong {removal[gone][name]['wrong']}); localization L = {loc}")
    # records that use only the other function must be untouched
    for gone, name in (("fn.add_exact.v1", "add_exact"), ("fn.compare_numeric.v1", "compare_numeric")):
        other = "compare_numeric" if name == "add_exact" else "add_exact"
        pure = [a for a in data["test"] if a.sites and all(s.fn == other for site in a.sites for s in site.steps)]
        before = ev.summarize(ev.answers(core, tok, layer, pure), by="function")["all"]["exact"]
        session.unload(gone)
        after = ev.summarize(ev.answers(core, tok, layer, pure), by="function")["all"]["exact"]
        session.load(gone)
        report["separation"][f"{other}_only_records_without_{name}"] = {"n": len(pure), "exact_before": before, "exact_after": after}
        print(f"records that need only {other}: {before}/{len(pure)} with everything loaded, {after}/{len(pure)} without {gone}")

    # ---- dynamic loading between tokens ---------------------------------------------------------------
    say("7. Dynamic loading: one text, modules changed between tokens")
    piece_map = PieceMap(lambda s: tok.encode(s).ids, overrides={"-": tok.token_id("<neg>"), ".": tok.token_id("<dot>")})
    stop = (tok.eos_id, tok.abstain_id, tok.token_id("."))
    prompts = ["mara has 417 coins and gets 85 more . now mara has",
               "ines has 90 shells and gets 25 more . bruno has 170 . does ines have more than bruno ?"]
    demo = []
    session.unload("fn.add_exact.v1")
    session.unload("fn.compare_numeric.v1")
    g = generate(core, tok, session.layer, prompts[0], piece_map, stop_ids=stop)
    demo.append({"loaded": [], "text": g.text})
    print("nothing loaded      :", g.text)
    info = session.load("fn.add_exact.v1")
    g = generate(core, tok, session.layer, prompts[0], piece_map, stop_ids=stop)
    demo.append({"loaded": session.resident(), "text": g.text, "load_ms": info["ms"], "trace": [t.__dict__ for t in g.traces]})
    print(f"add_exact loaded    : {g.text}    (load took {info['ms']:.1f} ms, {info['bytes']:,} bytes)")
    g = generate(core, tok, session.layer, prompts[1], piece_map, stop_ids=stop)
    demo.append({"loaded": session.resident(), "text": g.text, "trace": [t.__dict__ for t in g.traces]})
    print("compare not loaded  :", g.text, "   (add_exact ran, said its result is not the answer, nothing continued)")
    info = session.load("fn.compare_numeric.v1")
    g = generate(core, tok, session.layer, prompts[1], piece_map, stop_ids=stop)
    demo.append({"loaded": session.resident(), "text": g.text, "load_ms": info["ms"], "trace": [t.__dict__ for t in g.traces]})
    print(f"compare loaded      : {g.text}    (load took {info['ms']:.1f} ms)")
    for t in g.traces:
        print(f"    call {t.step}: {t.fn}.{t.op}({', '.join(t.args)}) -> {t.result}   pointers {t.pointers}   score {t.score}"
              f"   {'final' if t.final else 'not final'}")
    report["dynamic_loading"] = {"demo": demo, "events": session.events}

    # ---- registry: index the fitted packs, record how often each was called ---------------------------
    from flow.registry.registry import Registry

    (out / "registry.sqlite").unlink(missing_ok=True)
    registry = Registry(out / "registry.sqlite")
    indexed, failed = registry.scan(out / "library")
    for module_id, calls in session.usage().items():
        registry.record_usage(module_id, calls)
    report["registry"] = {"indexed": indexed, "problems": failed, "usage": session.usage()}
    registry.export(out / "registry.json")
    registry.close()

    report["seconds_total"] = round(time.time() - t_start, 1)
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    (out / "report.md").write_text(markdown(report), encoding="utf-8")
    say(f"done in {report['seconds_total']} s; report in {out / 'report.md'}")
    return 0


def blank_hash(path: Path) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def markdown(r: dict) -> str:
    s = r["scores"]
    lines = ["# Toy walkthrough report", "",
             f"Core: {r['core']['parameters']:,} parameters, layer passes {r['core']['pattern']}, pretrained from random weights "
             f"in {r['core']['seconds']} s with answer masking. Learned on top: "
             + ", ".join(f"{k} {v:,}" for k, v in r["learned_parameters"].items()) + ".", "",
             "| Set | Exact | Abstained | Wrong | n |", "|---|---|---|---|---|"]
    a = r["core_alone"]["all"]
    lines.append(f"| Core alone, test | {a['exact_rate']:.3f} | {a['abstained']} | {a['wrong']} | {a['n']} |")
    for name, label in (("test", "Both modules, test"), ("long", "Both modules, numbers of 8-11 digits (training: 1-6)"),
                        ("heldout", "Both modules, pair of calls never shown in training (subtract, then greater)")):
        x = s[name]["by_function"]["all"]
        lines.append(f"| {label} | {x['exact_rate']:.3f} | {x['abstained']} | {x['wrong']} | {x['n']} |")
    lines += ["", f"Calls at positions where none belongs: {s['false_calls']['false_calls']} in {s['false_calls']['positions']:,} positions.", "",
              "## Separation", "", "| Module unloaded | Function | Exact before | Exact after | Localization L |", "|---|---|---|---|---|"]
    sep = r["separation"]
    for gone, name in (("fn.add_exact.v1", "add_exact"), ("fn.compare_numeric.v1", "compare_numeric")):
        lines.append(f"| {gone} | {name} | {sep['all_loaded'][name]['exact_rate']:.3f} | {sep['removed'][gone][name]['exact_rate']:.3f} | {sep['localization'][name]} |")
    lines += ["", "## Per template", "", "| Template | Test | Long numbers |", "|---|---|---|"]
    for t, x in s["test"]["by_template"].items():
        long = s["long"]["by_template"].get(t)
        lines.append(f"| {t} | {x['exact_rate']:.3f} | {long['exact_rate']:.3f} |" if long else f"| {t} | {x['exact_rate']:.3f} | |")
    lines += ["", "## Dynamic loading", ""]
    for d in r["dynamic_loading"]["demo"]:
        lines.append(f"- loaded {d['loaded'] or 'nothing'}: `{d['text']}`")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
