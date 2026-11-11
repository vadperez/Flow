"""From a record with character spans to token positions for one tokenizer.

Works with any tokenizer object whose encode(text) returns .ids, .pieces and .offsets
(character span per token). ToyTokenizer does; flow.borrowed.hf_core wraps Hugging Face tokenizers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..typesys import Anchor, Kind, Value, flag, number, scan

RES = -1  # pointer value meaning "the result of the previous call"
IGNORE = -100  # label ignored by the loss


class AlignmentError(ValueError):
    """A record does not fit the tokenizer (a span does not end on a token boundary, ...)."""


@dataclass
class Step:
    fn: str
    op: str
    pointers: "list[int]"  # index into Aligned.anchors, or RES
    result: Value


@dataclass
class Site:
    """One place where a result is needed."""

    position: int  # token index of the call: the last token before the result
    first: int  # first and last token of the printed result
    last: int
    steps: "list[Step]" = field(default_factory=list)


@dataclass
class Aligned:
    record: dict
    ids: "list[int]"
    pieces: "list[str]"
    offsets: "list[tuple[int, int]]"
    anchors: "list[Anchor]"
    sites: "list[Site]"


def _value(spec: dict) -> Value:
    if spec["type"] == "NUMBER":
        return number(spec["value"])
    if spec["type"] == "FLAG":
        return flag(spec["value"] == "yes")
    raise AlignmentError(f"unsupported result type {spec['type']}")


def align(record: dict, tokenizer, table=None) -> Aligned:
    text = record["text"]
    enc = tokenizer.encode(text)
    anchors = scan(text, enc.offsets, enc.pieces, table)
    by_span = {(a.start, a.end): i for i, a in enumerate(anchors)}
    sites = []
    for index, (start, end) in enumerate(record["result_spans"]):
        inside = [t for t, (s, e) in enumerate(enc.offsets) if s >= start and e <= end]
        if not inside or enc.offsets[inside[0]][0] != start or enc.offsets[inside[-1]][1] != end or inside[0] == 0:
            raise AlignmentError(f"{record['id']}: result span {start}-{end} does not match token boundaries")
        site = Site(position=inside[0] - 1, first=inside[0], last=inside[-1])
        for call in sorted((c for c in record["calls"] if c["for"] == index), key=lambda c: c["step"]):
            pointers = []
            for arg in call["args"]:
                if "ptr" in arg:
                    if not site.steps:
                        raise AlignmentError(f"{record['id']}: RES used in the first call")
                    pointers.append(RES)
                    continue
                span = tuple(arg["span"])
                if span not in by_span or anchors[by_span[span]].token > site.position:
                    raise AlignmentError(f"{record['id']}: argument {span} is not a typed value before the call")
                pointers.append(by_span[span])
            site.steps.append(Step(call["fn"], call["op"], pointers, _value(call["result"])))
        sites.append(site)
    return Aligned(record, enc.ids, enc.pieces, enc.offsets, anchors, sites)


def lm_example(a: Aligned, bos: int, eos: int, abstain: int, abstained: bool = False) -> "tuple[list[int], list[int]]":
    """Inputs and labels for answer-masked pretraining of the core.

    The label at the call position is <abstain>; the labels that are result tokens are ignored.
    The core therefore learns where a result is needed, never the result itself.
    With abstained=True the result is replaced by <abstain> in the input as well, so the core
    also learns how text continues after it has abstained.
    """
    if abstained and a.sites:
        ids, cut = [], {t for s in a.sites for t in range(s.first, s.last + 1)}
        firsts = {s.first for s in a.sites}
        for t, tok in enumerate(a.ids):
            if t in firsts:
                ids.append(abstain)
            elif t not in cut:
                ids.append(tok)
        seq = [bos] + ids + [eos]
        return seq[:-1], seq[1:]
    seq = [bos] + list(a.ids) + [eos]
    inputs, labels = seq[:-1], seq[1:]
    for s in a.sites:
        labels[s.position + 1] = abstain  # input index = token index + 1 because of <bos>
        for t in range(s.first, s.last):
            labels[t + 1] = IGNORE
    return inputs, labels


def check_pointers(a: Aligned, signatures: "dict[str, tuple[Kind, ...]]") -> None:
    """Every pointer must land on a value of the type the function expects."""
    for site in a.sites:
        previous = None
        for step in site.steps:
            for pointer, kind in zip(step.pointers, signatures[step.fn]):
                got = previous.kind if pointer == RES else a.anchors[pointer].value.kind
                if got is not kind:
                    raise AlignmentError(f"{a.record['id']}: {step.fn} needs {kind.value}, pointer gives {got.value}")
            previous = step.result
