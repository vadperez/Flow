"""Typed values: a 32-dimensional code plus a 2-dimensional type tag (34 dimensions)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from . import numcode as num
from . import symcode as sym

CODE_DIM = 32
VALUE_DIM = 34


class Kind(str, Enum):
    NONE = "NONE"
    NUMBER = "NUMBER"
    SYMBOL = "SYMBOL"
    FLAG = "FLAG"


_TAGS = {Kind.NONE: (0.0, 0.0), Kind.NUMBER: (1.0, 0.0), Kind.SYMBOL: (0.0, 1.0), Kind.FLAG: (1.0, 1.0)}


@dataclass(frozen=True)
class Value:
    """A typed value in readable form. `payload` is a Number, a symbol string, a bool, or None."""

    kind: Kind
    payload: object = None

    def text(self) -> str:
        if self.kind is Kind.NUMBER:
            return self.payload.text()
        if self.kind is Kind.SYMBOL:
            return str(self.payload)
        if self.kind is Kind.FLAG:
            return "yes" if self.payload else "no"
        return ""


NONE = Value(Kind.NONE)


def number(value) -> Value:
    return Value(Kind.NUMBER, value if isinstance(value, num.Number) else num.Number.from_value(value))


def symbol(name: str) -> Value:
    return Value(Kind.SYMBOL, name)


def flag(truth: bool) -> Value:
    return Value(Kind.FLAG, bool(truth))


def pack(value: Value, table: "sym.SymbolTable | None" = None) -> np.ndarray:
    """Value -> 34-dimensional register content."""
    v = np.zeros(VALUE_DIM, dtype=np.float32)
    v[CODE_DIM:] = _TAGS[value.kind]
    if value.kind is Kind.NUMBER:
        v[:CODE_DIM] = num.encode(value.payload)
    elif value.kind is Kind.SYMBOL:
        if table is None:
            raise ValueError("a symbol table is needed to pack a SYMBOL")
        v[:CODE_DIM] = table.encode(value.payload)
    elif value.kind is Kind.FLAG:
        v[0] = 1.0 if value.payload else -1.0
    return v


def kind_of(v: np.ndarray) -> Kind:
    tag = (bool(v[CODE_DIM] > 0.5), bool(v[CODE_DIM + 1] > 0.5))
    return {(False, False): Kind.NONE, (True, False): Kind.NUMBER, (False, True): Kind.SYMBOL, (True, True): Kind.FLAG}[tag]


def unpack(v: np.ndarray, table: "sym.SymbolTable | None" = None) -> Value:
    """34-dimensional register content -> Value."""
    kind = kind_of(v)
    if kind is Kind.NUMBER:
        return Value(kind, num.decode(v[:CODE_DIM]))
    if kind is Kind.SYMBOL:
        if table is None:
            raise ValueError("a symbol table is needed to unpack a SYMBOL")
        return Value(kind, table.decode(v[:CODE_DIM]))
    if kind is Kind.FLAG:
        return Value(kind, bool(v[0] > 0))
    return NONE
