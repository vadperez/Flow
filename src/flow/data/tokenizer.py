"""A tiny word-level tokenizer for the walkthrough and the tests.

Every digit is its own token, as required by the type system (pipeline step S2.3). Words,
punctuation marks and the reserved tokens are single tokens too. A minus sign glued to a number
and a decimal point inside a number are the tokens <neg> and <dot>, so that tokens can be turned
back into the same text. It reports character offsets, which is all the typed scanner needs.
The real project trains a BPE tokenizer in S2.3.

Text for this tokenizer is "spaced": words, numbers and punctuation marks separated by single
spaces, for example "mara has 1250 coins . the change is -3.5 ." For such text,
render(encode(text).ids) returns the text unchanged.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>", "<abstain>", "<unknown>", "<refuse>", "<neg>", "<dot>"]
PIECE_RE = re.compile(r"<[a-z]+>|[A-Za-z_]+|\d|[^\sA-Za-z\d]")


@dataclass
class Encoding:
    ids: "list[int]"
    pieces: "list[str]"
    offsets: "list[tuple[int, int]]"


class ToyTokenizer:
    def __init__(self, vocab: "list[str]") -> None:
        self.vocab = list(vocab)
        self.index = {p: i for i, p in enumerate(self.vocab)}
        for s in SPECIALS:
            if s not in self.index:
                raise ValueError(f"vocabulary lacks the reserved token {s}")

    @classmethod
    def from_texts(cls, texts, extra=()) -> "ToyTokenizer":
        seen = set()
        for t in texts:
            seen.update(PIECE_RE.findall(t))
        seen.update(extra)
        base = SPECIALS + list("0123456789") + ["-", ".", "yes", "no"]
        rest = sorted(seen - set(base))
        return cls(base + rest)

    def __len__(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.index["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.index["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.index["<eos>"]

    @property
    def abstain_id(self) -> int:
        return self.index["<abstain>"]

    def token_id(self, piece: str) -> int:
        return self.index.get(piece, self.index["<unk>"])

    def encode(self, text: str) -> Encoding:
        ids, pieces, offsets = [], [], []
        for m in PIECE_RE.finditer(text):
            piece, s, e = m.group(0), m.start(), m.end()
            digit_next = e < len(text) and text[e].isdigit()
            if piece == "-" and digit_next and (s == 0 or text[s - 1].isspace()):
                piece = "<neg>"
            elif piece == "." and digit_next and s > 0 and text[s - 1].isdigit():
                piece = "<dot>"
            pieces.append(piece)
            ids.append(self.token_id(piece))
            offsets.append((s, e))
        return Encoding(ids, pieces, offsets)

    def render(self, ids) -> "tuple[str, list[tuple[int, int]]]":
        """Token ids -> (text, character span of every token). Inverse of encode() on spaced text."""
        text, offsets, previous = "", [], ""
        for i in ids:
            piece = self.vocab[i]
            shown = {"<neg>": "-", "<dot>": "."}.get(piece, piece)
            glued = bool(previous) and (
                (piece.isdigit() and (previous.isdigit() or previous in ("<neg>", "<dot>")))
                or (piece == "<dot>" and previous.isdigit()))
            if previous and not glued:
                text += " "
            offsets.append((len(text), len(text) + len(shown)))
            text += shown
            previous = piece
        return text, offsets

    def decode(self, ids) -> str:
        return self.render(ids)[0]

    def save(self, path: "str | Path") -> None:
        Path(path).write_text(json.dumps({"vocab": self.vocab}, ensure_ascii=False, indent=1), encoding="utf-8")

    @staticmethod
    def load(path: "str | Path") -> "ToyTokenizer":
        return ToyTokenizer(json.loads(Path(path).read_text(encoding="utf-8"))["vocab"])
