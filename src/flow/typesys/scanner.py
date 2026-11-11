"""Typed scanner: finds the typed value anchored at each token.

A number is anchored at its last token. A token listed in the symbol table carries a SYMBOL.
The scanner is deterministic and works on any tokenizer that reports character offsets,
so it also runs on text the model has just generated.

Reference implementation: it rescans the text on every call. An incremental version is a
later optimization (pipeline step S8.1).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from . import numcode as num
from .symcode import SymbolTable
from .values import Kind, Value

# Optional sign, digits (with optional thousands separators), optional decimal part.
# Not glued to a preceding letter, digit, dot or comma ("H2O", "v1.2" are not numbers).
NUMBER_RE = re.compile(
    r"(?<![\w.,])[-−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\d|[.,]\d)"
)


@dataclass(frozen=True)
class Anchor:
    token: int  # index of the token that carries the value
    start: int  # character span of the value in the text
    end: int
    value: Value


def find_numbers(text: str) -> "list[tuple[int, int, num.Number]]":
    """Character spans of the numbers in `text` that fit the NUMBER type."""
    found = []
    for m in NUMBER_RE.finditer(text):
        try:
            found.append((m.start(), m.end(), num.Number.from_value(m.group(0))))
        except num.NumberRangeError:
            continue  # too long for the type: it stays plain text
    return found


def _token_at(offsets: "list[tuple[int, int]]", char_index: int) -> "int | None":
    """Index of the token whose span contains `char_index`."""
    for i, (s, e) in enumerate(offsets):
        if s <= char_index < e:
            return i
    return None


def scan(
    text: str,
    offsets: "list[tuple[int, int]]",
    token_texts: "list[str] | None" = None,
    table: "SymbolTable | None" = None,
) -> "list[Anchor]":
    """All typed anchors of a tokenized text, ordered by token index.

    offsets[i] is the (start, end) character span of token i.
    token_texts[i] is the text of token i; it is needed only to find symbols.
    """
    anchors: list[Anchor] = []
    for start, end, n in find_numbers(text):
        tok = _token_at(offsets, end - 1)
        if tok is not None:
            anchors.append(Anchor(tok, start, end, Value(Kind.NUMBER, n)))
    if table is not None and token_texts is not None:
        number_tokens = {a.token for a in anchors}
        for i, piece in enumerate(token_texts):
            word = piece.strip()
            if word in table and i not in number_tokens:
                s, e = offsets[i]
                anchors.append(Anchor(i, s, e, Value(Kind.SYMBOL, word)))
    return sorted(anchors, key=lambda a: a.token)


def anchors_before(anchors: "list[Anchor]", position: int, kind: "Kind | None" = None) -> "list[Anchor]":
    """Anchors at tokens <= position, optionally of one kind. These are a binder's candidates."""
    return [a for a in anchors if a.token <= position and (kind is None or a.value.kind is kind)]
