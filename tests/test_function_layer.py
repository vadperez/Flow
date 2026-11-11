"""Function layer, call loop and runtime on a tiny blank core.

These tests check mechanics (shapes, causality, loading and unloading, type checks), not quality:
fitting takes minutes and is what scripts/toy_walkthrough.py does.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from flow.bodies import AddExact, CompareNumeric
from flow.core import blank
from flow.data import align as al
from flow.data.tokenizer import ToyTokenizer
from flow.data.world import ToyWorld, vocabulary
from flow.function_layer import evaluate as ev
from flow.function_layer.loop import FunctionLayer, generate
from flow.function_layer.model import ControllerBlock, FunctionHeads
from flow.function_layer.train import FunctionSpec, collate, core_hidden, fit, losses
from flow.runtime.minimal import Library, Session
from flow.typesys import Kind, PieceMap


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    tok = ToyTokenizer.from_texts([], extra=vocabulary())
    spec = blank.CoreSpec(vocab_size=len(tok), hidden_size=32, intermediate_size=64, heads=2, kv_heads=1, head_dim=16,
                          linear_heads=2, linear_head_dim=16, entry="L", loop="F", loop_passes=2, exit="", max_positions=256,
                          pad_token_id=tok.pad_id, bos_token_id=tok.bos_id, eos_token_id=tok.eos_id)
    core = blank.build(spec).eval()
    bodies = {"add_exact": AddExact(), "compare_numeric": CompareNumeric()}
    specs = {k: FunctionSpec.of(b) for k, b in bodies.items()}
    controller = ControllerBlock(core_width=32, n_taps=len(spec.pattern) + 1, width=32, heads=2, key_dim=16)
    heads = {k: FunctionHeads(32, len(spec.pattern) + 1, 32, 16, controller.ctrl_dim, len(s.operations), len(s.arg_kinds), width=16, heads=2)
             for k, s in specs.items()}
    data = [al.align(r, tok) for r in ToyWorld(seed=3).records(40)]
    return dict(tok=tok, spec=spec, core=core, bodies=bodies, specs=specs, controller=controller, heads=heads, data=data)


def test_losses_and_one_fit_step(setup):
    s = setup
    batch = s["data"][:8]
    ids, pad = collate(batch, s["tok"].bos_id, s["tok"].pad_id)
    hidden = core_hidden(s["core"], ids, pad)
    loss, counts = losses(s["controller"], s["heads"], s["specs"], batch, hidden, pad)
    assert loss.requires_grad and counts["sites"] == sum(len(a.sites) for a in batch) and counts["pointer_examples"] > 0
    before = [p.clone() for p in s["core"].parameters()]
    history = fit(s["core"], s["controller"], s["heads"], s["specs"], s["data"], s["tok"].bos_id, s["tok"].pad_id, steps=3,
                  batch_size=8, log=lambda row: None)
    assert history and all(torch.equal(a, b) for a, b in zip(before, s["core"].parameters()))  # the core is frozen


def test_features_are_causal(setup):
    s = setup
    a = s["data"][0]
    ids, pad = collate([a], s["tok"].bos_id, s["tok"].pad_id)
    changed = ids.clone()
    changed[0, -1] = s["tok"].token_id("coins")
    layer = FunctionLayer(s["controller"])
    layer.attach("fn.add_exact.v1", s["heads"]["add_exact"], s["bodies"]["add_exact"])
    f1 = layer.features(core_hidden(s["core"], ids, pad), pad)
    f2 = layer.features(core_hidden(s["core"], changed, pad), pad)
    assert torch.allclose(f1["c"][0, :-1], f2["c"][0, :-1], atol=1e-5)
    assert torch.allclose(f1["z"]["fn.add_exact.v1"][0, :-1], f2["z"]["fn.add_exact.v1"][0, :-1], atol=1e-5)


def test_call_loop_respects_types_and_thresholds(setup):
    s = setup
    layer = FunctionLayer(s["controller"])
    a = next(x for x in s["data"] if x.sites)
    ids, pad = collate([a], s["tok"].bos_id, s["tok"].pad_id)
    hidden = core_hidden(s["core"], ids, pad)
    from flow.typesys import Anchor
    anchors = [Anchor(x.token + 1, x.start, x.end, x.value) for x in a.anchors]
    p = a.sites[0].position + 1
    layer.attach("fn.add_exact.v1", s["heads"]["add_exact"], s["bodies"]["add_exact"], threshold=1.01)  # can never fire
    value, traces = layer.calls_at(layer.features(hidden, pad), 0, p, anchors)
    assert value is None and traces == []
    layer.attach("fn.add_exact.v1", s["heads"]["add_exact"], s["bodies"]["add_exact"], threshold=0.0)  # always fires
    value, traces = layer.calls_at(layer.features(hidden, pad), 0, p, anchors)
    assert len(traces) == s["controller"].max_calls and all(t.result is not None for t in traces)
    assert (value is None) == (not traces[-1].final)  # a chain whose last call is not final is dropped
    if value is not None:
        assert value.kind is Kind.NUMBER
    assert all(t.pointers and all(q == "RES" or q <= p for q in t.pointers) for t in traces)  # only earlier values or RES
    value, traces = layer.calls_at(layer.features(hidden, pad), 0, p, [])  # nothing typed to point at
    assert value is None and traces[0].error.startswith("no NUMBER")


def test_library_session_load_unload(setup, tmp_path):
    s = setup
    library = Library(tmp_path / "library")
    library.write_controller(s["controller"], fitted_to="tiny-core@test")
    for name, body in s["bodies"].items():
        library.write_pack(f"fn.{name}.v1", body, s["heads"][name], fitted_to="tiny-core@test", builder="test")
    library.write_pack("fn.orphan_exact.v1".replace("orphan_exact", "add_exact").replace("v1", "v2"), s["bodies"]["add_exact"])  # body only
    assert library.problems() == {}
    session = Session(s["core"], s["tok"], library, core_id="tiny-core@test")
    with pytest.raises(RuntimeError, match="not fitted"):
        session.load("fn.add_exact.v2")
    info = session.load("fn.add_exact.v1")
    assert session.resident() == ["fn.add_exact.v1"] and info["bytes"] > 9_000_000 and info["ms"] > 0
    session.load("fn.compare_numeric.v1")
    assert session.resident_bytes() == sum(library.manifest(i)["cost"]["bytes"] for i in session.resident())
    rows_before = ev.answers(s["core"], s["tok"], session.layer, s["data"])
    session.unload("fn.compare_numeric.v1")
    session.unload("fn.add_exact.v1")
    assert session.resident() == [] and [e["action"] for e in session.events] == ["load", "load", "unload", "unload"]
    session.load("fn.add_exact.v1")
    session.load("fn.compare_numeric.v1")
    rows_after = ev.answers(s["core"], s["tok"], session.layer, s["data"])
    assert [r["got"] for r in rows_before] == [r["got"] for r in rows_after]  # reloading changes nothing
    with pytest.raises(RuntimeError, match="fitted to"):
        Session(s["core"], s["tok"], library, core_id="another-core")


def test_generate_runs_with_and_without_modules(setup):
    s = setup
    tok = s["tok"]
    piece_map = PieceMap(lambda text: tok.encode(text).ids, overrides={"-": tok.token_id("<neg>"), ".": tok.token_id("<dot>")})
    assert piece_map.missing == []
    layer = FunctionLayer(s["controller"])
    plain = generate(s["core"], tok, None, "mara has 4 coins and gets 5 more . now mara has", piece_map, max_new=4)
    assert len(plain.ids) > 0 and plain.traces == []
    import copy
    heads = copy.deepcopy(s["heads"]["add_exact"])
    with torch.no_grad():
        heads.trigger[-1].bias[-1] = 50.0  # untrained heads: force "final" so that the result is printed
    layer.attach("fn.add_exact.v1", heads, s["bodies"]["add_exact"], threshold=0.0)
    forced = generate(s["core"], tok, layer, "mara has 4 coins and gets 5 more . now mara has", piece_map, max_new=3)
    assert forced.traces and forced.printed_by_emitter  # with threshold 0 a call always happens and its result is printed
    printed = tok.render(forced.ids)[0].split("now mara has ")[1].split()[0]
    assert printed.lstrip("-").replace(".", "").isdigit()
