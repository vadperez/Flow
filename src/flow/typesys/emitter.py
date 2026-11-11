"""Emitter: prints a typed result as tokens.

NUMBER: one piece per character of the canonical text (sign, digits, decimal point).
SYMBOL: the symbol as one piece.
FLAG:   "yes" or "no".

While the emitter is printing, its token replaces the core's choice. When it has nothing
to print, the core's logits pass through unchanged.
"""
from __future__ import annotations

from .values import Kind, Value


def pieces(value: Value) -> "list[str]":
    """The pieces to print for a value, in order."""
    if value.kind is Kind.NUMBER:
        return list(value.payload.text())
    if value.kind is Kind.SYMBOL:
        return [str(value.payload)]
    if value.kind is Kind.FLAG:
        return ["yes" if value.payload else "no"]
    return []


class Emitter:
    """Small state machine holding the pieces still to be printed."""

    def __init__(self) -> None:
        self._pending: list[str] = []
        self.printed: list[str] = []

    def start(self, value: Value) -> None:
        self._pending = pieces(value)
        self.printed = []

    @property
    def active(self) -> bool:
        return bool(self._pending)

    def next_piece(self) -> str:
        piece = self._pending.pop(0)
        self.printed.append(piece)
        return piece

    def cancel(self) -> None:
        self._pending = []


class PieceMap:
    """Maps emitter pieces to token ids of one tokenizer.

    Build it once per tokenizer with `encode_piece`, a function that returns the token ids of a
    short string. A piece must map to exactly one token; otherwise it is reported in `missing`.
    """

    def __init__(self, encode_piece, extra_pieces: "list[str] | None" = None,
                 overrides: "dict[str, int] | None" = None) -> None:
        self.ids: dict[str, int] = dict(overrides or {})  # pieces whose token is known in advance
        self.missing: list[str] = []
        self._encode = encode_piece
        for p in list("0123456789") + ["-", ".", "yes", "no"] + list(extra_pieces or []):
            self.add(p)

    def add(self, piece: str) -> "int | None":
        if piece in self.ids:
            return self.ids[piece]
        ids = list(self._encode(piece))
        if len(ids) == 1:
            self.ids[piece] = int(ids[0])
            return self.ids[piece]
        if piece not in self.missing:
            self.missing.append(piece)
        return None

    def token_id(self, piece: str) -> int:
        tid = self.add(piece)
        if tid is None:
            raise KeyError(f"piece {piece!r} is not a single token in this tokenizer")
        return tid
