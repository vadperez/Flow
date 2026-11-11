"""Toy generator, reference interpreter and alignment (pipeline step S2.1, in miniature)."""
import pytest

from flow.data import align as al
from flow.data.tokenizer import ToyTokenizer
from flow.data.world import HELD_OUT, TEMPLATES, ToyWorld, execute, vocabulary
from flow.typesys import Kind

SIGNATURES = {"add_exact": (Kind.NUMBER, Kind.NUMBER), "compare_numeric": (Kind.NUMBER, Kind.NUMBER)}


@pytest.fixture(scope="module")
def tok():
    return ToyTokenizer.from_texts([], extra=vocabulary())


def test_records_reproduce_and_align(tok):
    """1,000 records, re-executed by the reference interpreter, reproduce their results exactly;
    every pointer lands on a token of the right type; the tokenizer reproduces the text."""
    world = ToyWorld(seed=1)
    seen = set()
    for record in world.records(1000):
        assert execute(record)
        a = al.align(record, tok)
        al.check_pointers(a, SIGNATURES)
        assert tok.render(a.ids)[0] == record["text"] and tok.token_id("<unk>") not in a.ids
        for site in a.sites:
            assert "".join(record["text"][s:e] for s, e in a.offsets[site.first:site.last + 1]) == \
                record["text"][record["result_spans"][0][0]:record["result_spans"][0][1]]
        seen.add(record["template"])
    assert seen == {t[0] for t in TEMPLATES if t[0] not in HELD_OUT} | {"none"}


def test_long_numbers_and_held_out_split(tok):
    world = ToyWorld(seed=2, min_digits=9, max_digits=12)
    for record in world.records(300, split="heldout"):
        assert record["template"] in HELD_OUT | {"none"} and execute(record)
        al.check_pointers(al.align(record, tok), SIGNATURES)
    for name in ("add-1", "sub-4", "add>sub", "eq-1", "add-5"):
        for record in world.records(100, template=name):
            assert execute(record)


def test_interpreter_catches_a_wrong_record():
    record = ToyWorld(seed=3).record(template="add-1", distractors=False)
    s, e = record["result_spans"][0]
    broken = dict(record, text=record["text"][:s] + "1" + record["text"][s:])
    assert execute(record) and not execute(broken)


def test_answer_masking_labels(tok):
    record = ToyWorld(seed=4).record(template="add>sub", distractors=False)
    a = al.align(record, tok)
    inputs, labels = al.lm_example(a, tok.bos_id, tok.eos_id, tok.abstain_id)
    site = a.sites[0]
    assert len(inputs) == len(labels) == len(a.ids) + 1
    assert labels[site.position + 1] == tok.abstain_id  # the call position predicts <abstain>
    result_tokens = a.ids[site.first:site.last + 1]
    visible = [l for l in labels if l != al.IGNORE]
    kept = len(labels) - len(visible)
    assert kept == len(result_tokens) - 1  # every result token is hidden from the loss
    assert labels[site.last + 1] == a.ids[site.last + 1]  # the text after the result is learned normally
    assert [s.op for s in site.steps] == ["add", "subtract"] and site.steps[1].pointers[0] == al.RES
    inputs2, labels2 = al.lm_example(a, tok.bos_id, tok.eos_id, tok.abstain_id, abstained=True)
    assert inputs2.count(tok.abstain_id) == 1 and len(inputs2) == len(inputs) - len(result_tokens) + 1
    assert al.IGNORE not in labels2
