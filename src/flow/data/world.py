"""Toy world: templated task text with call traces.

Small on purpose: enough to exercise the whole mechanism on a laptop. The real generator, with
6,000 entities, 8 relations and teacher-written paraphrases, is pipeline step S2.1.

A record follows Annex B of the pipeline, with one change: positions are stored as character
spans, not token indices, so the same record serves any tokenizer (see align.py).

    {"id": "toy-000012",
     "text": "mara has 412 coins and gets 35 more . now mara has 447 .",
     "calls": [{"for": 0, "step": 1, "fn": "add_exact", "op": "add",
                "args": [{"span": [9, 12]}, {"span": [28, 30]}],
                "result": {"type": "NUMBER", "value": "447"}, "printed": true}],
     "result_spans": [[51, 54]],
     "combination": ["add_exact.add"], "template": "add-1", "split": "train"}

An argument is either {"span": [start, end]} (a value in the text) or {"ptr": "RES"} (the result
of the previous call). All calls for one result happen at the token just before the result.
"""
from __future__ import annotations

import random
from decimal import Decimal

from ..typesys.numcode import N_DIGITS, Number

NAMES = ["mara", "tomas", "ines", "pablo", "lucia", "diego", "elena", "bruno", "carla", "hugo", "nora", "omar"]
ITEMS = ["coins", "apples", "stamps", "books", "shells", "beads", "cards", "seeds"]

FUNCTION_OF = {"add": "add_exact", "subtract": "add_exact", "greater": "compare_numeric", "less": "compare_numeric",
               "equal": "compare_numeric"}

# (template id, text with slots, plan). Slots: {n} {m} names, {i} item, {a} {b} {c} numbers, {=} the result.
# Plan: the calls, in order; "RES" is the result of the previous call.
TEMPLATES = [
    ("add-1", "{n} has {a} {i} and gets {b} more . now {n} has {=} .", [("add", "a", "b")]),
    ("add-2", "{n} had {a} {i} . then {n} found {b} . total : {=} .", [("add", "a", "b")]),
    ("add-3", "there are {a} {i} in the box and {b} in the bag . together : {=} .", [("add", "a", "b")]),
    ("add-4", "{n} gets {b} {i} . before that {n} had {a} . now {n} has {=} .", [("add", "a", "b")]),
    ("add-5", "the price was {a} . it rose by {b} . new price : {=} .", [("add", "a", "b")]),
    ("sub-1", "{n} has {a} {i} and gives away {b} . now {n} has {=} .", [("subtract", "a", "b")]),
    ("sub-2", "{n} had {a} {i} . then {n} lost {b} . left : {=} .", [("subtract", "a", "b")]),
    ("sub-3", "{n} lost {b} {i} . before that {n} had {a} . left : {=} .", [("subtract", "a", "b")]),
    ("sub-4", "the balance is {a} . a charge of {b} arrives . new balance : {=} .", [("subtract", "a", "b")]),
    ("gt-1", "{n} has {a} {i} and {m} has {b} . does {n} have more ? {=} .", [("greater", "a", "b")]),
    ("lt-1", "{n} has {a} {i} and {m} has {b} . does {n} have fewer ? {=} .", [("less", "a", "b")]),
    ("eq-1", "{n} has {a} {i} and {m} has {b} . do they have the same ? {=} .", [("equal", "a", "b")]),
    ("gt-2", "is {a} more than {b} ? {=} .", [("greater", "a", "b")]),
    ("lt-2", "is {a} less than {b} ? {=} .", [("less", "a", "b")]),
    ("add>sub", "{n} has {a} {i} , gets {b} more and then gives away {c} . now {n} has {=} .",
     [("add", "a", "b"), ("subtract", "RES", "c")]),
    ("sub>add", "{n} had {a} {i} , lost {b} and then found {c} . now {n} has {=} .",
     [("subtract", "a", "b"), ("add", "RES", "c")]),
    ("add>add", "{n} has {a} {i} , gets {b} more and then gets {c} more . now {n} has {=} .",
     [("add", "a", "b"), ("add", "RES", "c")]),
    ("add>gt", "{n} has {a} {i} and gets {b} more . {m} has {c} . does {n} have more than {m} ? {=} .",
     [("add", "a", "b"), ("greater", "RES", "c")]),
    ("add>lt", "{n} has {a} {i} and gets {b} more . {m} has {c} . does {n} have fewer than {m} ? {=} .",
     [("add", "a", "b"), ("less", "RES", "c")]),
    ("sub>lt", "{n} has {a} {i} and gives away {b} . {m} has {c} . does {n} have fewer than {m} ? {=} .",
     [("subtract", "a", "b"), ("less", "RES", "c")]),
    ("add>eq", "{n} has {a} {i} and gets {b} more . {m} has {c} . do they have the same ? {=} .",
     [("add", "a", "b"), ("equal", "RES", "c")]),
    ("sub>eq", "{n} has {a} {i} and gives away {b} . {m} has {c} . do they have the same ? {=} .",
     [("subtract", "a", "b"), ("equal", "RES", "c")]),
    ("found>gt", "{n} had {a} {i} and found {b} . {m} has {c} . does {n} have more than {m} ? {=} .",
     [("add", "a", "b"), ("greater", "RES", "c")]),
    ("found>lt", "{n} had {a} {i} and found {b} . {m} has {c} . does {n} have fewer than {m} ? {=} .",
     [("add", "a", "b"), ("less", "RES", "c")]),
    ("found>eq", "{n} had {a} {i} and found {b} . {m} has {c} . do they have the same ? {=} .",
     [("add", "a", "b"), ("equal", "RES", "c")]),
    ("sub>gt", "{n} has {a} {i} and gives away {b} . {m} has {c} . does {n} have more than {m} ? {=} .",
     [("subtract", "a", "b"), ("greater", "RES", "c")]),
]
# Never in training: "subtract, then greater". Its parts are seen in other pairs: add is followed by
# greater, less and equal; subtract by less and equal. So the question, not the first call, decides
# the comparison, and this set tests whether calls combine in a way that was never shown.
HELD_OUT = {"sub>gt"}
DISTRACTORS = ["{m} is {k} years old .", "there are {k} {i} on the shelf .", "it is day {k} .", "{m} lives at number {k} ."]
MAY_BE_NEGATIVE = {"sub-4"}
DECIMAL = {"add-5"}


def vocabulary() -> "list[str]":
    """Every word the generator can produce (for the toy tokenizer)."""
    words = set(NAMES) | set(ITEMS)
    for _, text, _ in TEMPLATES:
        words.update(w for w in text.split() if not w.startswith("{"))
    for text in DISTRACTORS:
        words.update(w for w in text.split() if not w.startswith("{"))
    return sorted(words)


def apply(op: str, x: Decimal, y: Decimal):
    """Reference semantics of every operation, in plain decimal arithmetic."""
    return {"add": lambda: x + y, "subtract": lambda: x - y, "greater": lambda: x > y, "less": lambda: x < y,
            "equal": lambda: x == y}[op]()


def result_text(value) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    return Number.from_value(value).text()


class ToyWorld:
    def __init__(self, seed: int = 0, min_digits: int = 1, max_digits: int = 6) -> None:
        self.rng = random.Random(seed)
        self.min_digits, self.max_digits = min_digits, max_digits
        self.count = 0

    def _number(self, decimals: int = 0) -> Decimal:
        d = self.rng.randint(self.min_digits, self.max_digits)
        whole = self.rng.randint(0 if d == 1 else 10 ** (d - 1), 10 ** d - 1)
        return Decimal(whole).scaleb(-decimals) if decimals else Decimal(whole)

    def _values(self, template: str, plan) -> "dict[str, Decimal]":
        decimals = self.rng.choice([1, 2]) if template in DECIMAL else 0
        for _ in range(1000):
            v = {k: self._number(decimals) for k in "abc"}
            ops = [op for op, _, _ in plan]
            if "equal" in ops and self.rng.random() < 0.5:
                v["b"] = v["a"]
            if ops[0] == "subtract" and template not in MAY_BE_NEGATIVE and v["a"] < v["b"]:
                v["a"], v["b"] = v["b"], v["a"]
            if len(ops) == 2 and ops[1] == "equal" and self.rng.random() < 0.5:
                v["c"] = apply(ops[0], v["a"], v["b"])  # half of the chained "same?" questions are true
            res, ok = None, True
            for op, x, y in plan:  # every intermediate and final number must fit the NUMBER type
                res = apply(op, res if x == "RES" else v[x], res if y == "RES" else v[y])
                if not isinstance(res, bool):
                    ok &= abs(res) < 10 ** (N_DIGITS - decimals)
                    ok &= res >= 0 or template in MAY_BE_NEGATIVE
            if ok:
                return v
        raise RuntimeError("could not sample values")

    def _fill(self, text: str, slots: "dict[str, str]") -> "tuple[str, dict[str, list[int]]]":
        """Fill the slots; return the text and the character span of every number slot and of {=}."""
        out, spans, pos = [], {}, 0
        for word in text.split():
            shown = slots[word[1:-1]] if word.startswith("{") else word
            if word.startswith("{") and word[1:-1] in ("a", "b", "c", "="):
                spans[word[1:-1]] = [pos, pos + len(shown)]
            out.append(shown)
            pos += len(shown) + 1
        return " ".join(out), spans

    def distractor(self) -> str:
        slots = {"m": self.rng.choice(NAMES), "i": self.rng.choice(ITEMS), "k": str(self.rng.randint(1, 999))}
        return self._fill(self.rng.choice(DISTRACTORS), slots)[0]

    def record(self, template: "str | None" = None, split: str = "train", distractors: bool = True) -> dict:
        """One record. template=None picks any template that is allowed in the split."""
        self.count += 1
        rid = f"toy-{self.count:06d}"
        if template is None and distractors and self.rng.random() < 0.08:  # text without any call
            text = " ".join(self.distractor() for _ in range(self.rng.randint(1, 3)))
            return {"id": rid, "text": text, "calls": [], "result_spans": [], "combination": [], "template": "none", "split": split}
        if template is None:
            allowed = [t for t in TEMPLATES if (t[0] in HELD_OUT) == (split == "heldout")]
            tid, text, plan = self.rng.choice(allowed)
        else:
            tid, text, plan = next(t for t in TEMPLATES if t[0] == template)
        values = self._values(tid, plan)
        n, m = self.rng.sample(NAMES, 2)
        results, res = [], None
        for op, x, y in plan:
            res = apply(op, res if x == "RES" else values[x], res if y == "RES" else values[y])
            results.append(res)
        slots = {"n": n, "m": m, "i": self.rng.choice(ITEMS), "=": result_text(results[-1])}
        slots.update({k: result_text(v) for k, v in values.items()})
        body, spans = self._fill(text, slots)
        before = self.distractor() + " " if distractors and self.rng.random() < 0.5 else ""
        after = " " + self.distractor() if distractors and self.rng.random() < 0.3 else ""
        shift = len(before)
        calls = []
        for step, ((op, x, y), value) in enumerate(zip(plan, results), start=1):
            args = [{"ptr": "RES"} if k == "RES" else {"span": [spans[k][0] + shift, spans[k][1] + shift]} for k in (x, y)]
            calls.append({"for": 0, "step": step, "fn": FUNCTION_OF[op], "op": op, "args": args,
                          "result": {"type": "FLAG" if isinstance(value, bool) else "NUMBER", "value": result_text(value)},
                          "printed": step == len(plan)})
        return {"id": rid, "text": before + body + after, "calls": calls,
                "result_spans": [[spans["="][0] + shift, spans["="][1] + shift]],
                "combination": [f"{FUNCTION_OF[op]}.{op}" for op, _, _ in plan], "template": tid, "split": split}

    def records(self, n: int, **kwargs) -> "list[dict]":
        return [self.record(**kwargs) for _ in range(n)]


def execute(record: dict) -> bool:
    """Reference interpreter: re-run the calls from the text alone and compare with what is written."""
    text = record["text"]
    for index, (start, end) in enumerate(record["result_spans"]):
        res = None
        for call in sorted((c for c in record["calls"] if c["for"] == index), key=lambda c: c["step"]):
            args = [res if "ptr" in a else Decimal(text[a["span"][0]:a["span"][1]]) for a in call["args"]]
            if any(a is None or isinstance(a, bool) for a in args):
                return False  # RES used before it exists, or a FLAG used as a number
            res = apply(call["op"], *args)
            if result_text(res) != call["result"]["value"]:
                return False
        if res is None or result_text(res) != text[start:end]:
            return False
    return True
