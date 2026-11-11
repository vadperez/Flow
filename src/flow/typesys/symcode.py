"""SYMBOL type: a 16-bit symbol id in the second-order Reed-Muller code RM(2,5).

RM(2,5) maps 16 information bits to 32 code bits with minimum Hamming distance 8.
In +-1 form the correlation between two different codewords is at most 16 out of 32,
so decoding is one matrix product followed by an argmax, with a wide margin.

A symbol id is the index of a symbol (entity name, unit, relation word) in the symbol table.
"""
from __future__ import annotations

import itertools
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

ID_BITS = 16
CODE_DIM = 32
MAX_SYMBOLS = 1 << ID_BITS


@lru_cache(maxsize=1)
def generator() -> np.ndarray:
    """16 x 32 generator matrix over GF(2): rows are 1, x1..x5 and all products xi*xj."""
    points = np.array(list(itertools.product([0, 1], repeat=5)), dtype=np.uint8)  # 32 x 5
    rows = [np.ones(32, dtype=np.uint8)]
    rows += [points[:, i] for i in range(5)]
    rows += [points[:, i] & points[:, j] for i in range(5) for j in range(i + 1, 5)]
    g = np.stack(rows)
    assert g.shape == (ID_BITS, CODE_DIM)
    return g


@lru_cache(maxsize=1)
def codebook() -> np.ndarray:
    """All 65,536 codewords in +-1 form, as int8, row i = code of symbol id i."""
    ids = np.arange(MAX_SYMBOLS, dtype=np.uint32)
    bits = ((ids[:, None] >> np.arange(ID_BITS)) & 1).astype(np.uint8)  # 65536 x 16
    words = (bits @ generator()) % 2  # 65536 x 32
    return (1 - 2 * words.astype(np.int8)).astype(np.int8)


def encode_id(symbol_id: int) -> np.ndarray:
    if not 0 <= symbol_id < MAX_SYMBOLS:
        raise ValueError(f"symbol id {symbol_id} outside 0..{MAX_SYMBOLS - 1}")
    return codebook()[symbol_id].astype(np.float32)


def decode_id(v: np.ndarray) -> int:
    """Nearest codeword by correlation."""
    return int(np.argmax(codebook().astype(np.float32) @ np.asarray(v, dtype=np.float32)))


class SymbolTable:
    """Ordered list of symbols. The position of a symbol is its id."""

    def __init__(self, symbols: "list[str] | None" = None) -> None:
        self.symbols: list[str] = []
        self.index: dict[str, int] = {}
        for s in symbols or []:
            self.add(s)

    def add(self, symbol: str) -> int:
        if symbol in self.index:
            return self.index[symbol]
        if len(self.symbols) >= MAX_SYMBOLS:
            raise ValueError("symbol table is full")
        self.index[symbol] = len(self.symbols)
        self.symbols.append(symbol)
        return self.index[symbol]

    def __contains__(self, symbol: str) -> bool:
        return symbol in self.index

    def __len__(self) -> int:
        return len(self.symbols)

    def id_of(self, symbol: str) -> int:
        return self.index[symbol]

    def symbol_of(self, symbol_id: int) -> str:
        return self.symbols[symbol_id]

    def encode(self, symbol: str) -> np.ndarray:
        return encode_id(self.index[symbol])

    def decode(self, v: np.ndarray) -> str:
        return self.symbols[decode_id(v)]

    def save(self, path: "str | Path") -> None:
        Path(path).write_text(json.dumps({"symbols": self.symbols}, ensure_ascii=False, indent=1), encoding="utf-8")

    @staticmethod
    def load(path: "str | Path") -> "SymbolTable":
        return SymbolTable(json.loads(Path(path).read_text(encoding="utf-8"))["symbols"])
