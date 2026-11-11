"""Written weights: a small builder for exact ReLU circuits.

A body of an exact function is a plain feed-forward network

    h1 = relu(W1 x + b1), h2 = relu(W2 h1 + b2), ..., y = Wout h_last + bout

whose weights are constructed, never trained. This module lets a constructor script name every
unit and write each one as an affine expression of units of the previous layer.

Signals are bits. A bit is carried by a pair of units ("hi" and "lo") whose difference is the bit:

    bit(s) = relu(2 s - 0.5) - relu(2 s - 1.5)      0 for s <= 0.25, 1 for s >= 0.75

so every bit() removes an input error of up to 0.25 instead of passing it on. This is what keeps
the circuits exact in 16-bit floats and stops errors from growing over layers or over chained calls.

    AND(p1..pk, not q1..qm) = bit(sum(p) - sum(q) - (k - 1))
    OR of exclusive bits    = their sum              (an expression, no extra unit)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PRECISIONS = ("fp32", "fp16", "fp16_strict", "fp16_strict_bias_first")


class Expr:
    """Affine combination of named units of one layer: sum(coef * unit) + const."""

    __slots__ = ("terms", "const")

    def __init__(self, terms: "dict[str, float] | None" = None, const: float = 0.0) -> None:
        self.terms = dict(terms or {})
        self.const = float(const)

    @staticmethod
    def unit(name: str) -> "Expr":
        return Expr({name: 1.0})

    @staticmethod
    def total(exprs) -> "Expr":
        out = Expr()
        for e in exprs:
            out = out + e
        return out

    def _coerce(self, other) -> "Expr":
        return other if isinstance(other, Expr) else Expr(const=float(other))

    def __add__(self, other) -> "Expr":
        other = self._coerce(other)
        terms = dict(self.terms)
        for k, v in other.terms.items():
            terms[k] = terms.get(k, 0.0) + v
        return Expr(terms, self.const + other.const)

    __radd__ = __add__

    def __neg__(self) -> "Expr":
        return self * -1.0

    def __sub__(self, other) -> "Expr":
        return self + (-self._coerce(other))

    def __rsub__(self, other) -> "Expr":
        return self._coerce(other) + (-self)

    def __mul__(self, k: float) -> "Expr":
        return Expr({n: v * float(k) for n, v in self.terms.items()}, self.const * float(k))

    __rmul__ = __mul__


@dataclass
class Circuit:
    """Compiled circuit: dense layers plus a linear readout."""

    input_names: "list[str]"
    output_names: "list[str]"
    layers: "list[tuple[np.ndarray, np.ndarray]]"  # (W [n_out, n_in], b [n_out]) per ReLU layer
    readout: "tuple[np.ndarray, np.ndarray]"
    _rows: dict = field(default_factory=dict, repr=False, compare=False)

    # ---- running ------------------------------------------------------------------------------
    def forward(self, x: np.ndarray, precision: str = "fp32", return_hidden: bool = False):
        """x: [batch, n_inputs] -> [batch, n_outputs].

        precision
            fp32                     plain single precision
            fp16                     weights and every layer's output rounded to 16-bit floats;
                                     products summed in single precision (the usual GPU behaviour)
            fp16_strict              every product and every partial sum rounded to 16-bit floats,
                                     bias added last
            fp16_strict_bias_first   the same, bias added first (the least favourable order)
        """
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}")
        hidden = []
        if precision == "fp32":
            h = np.asarray(x, dtype=np.float32)
            for w, b in self.layers:
                h = np.maximum(h @ w.T + b, 0.0)
                hidden.append(h)
            out = h @ self.readout[0].T + self.readout[1]
        elif precision == "fp16":
            cast = lambda a: a.astype(np.float16).astype(np.float32)
            if "w16" not in self._rows:  # weights rounded once
                self._rows["w16"] = [(cast(w), cast(b)) for w, b in self.layers + [self.readout]]
            rounded = self._rows["w16"]
            h = cast(np.asarray(x, dtype=np.float32))
            for w, b in rounded[:-1]:
                h = cast(np.maximum(h @ w.T + b, 0.0))
                hidden.append(h)
            out = cast(h @ rounded[-1][0].T + rounded[-1][1])
        else:
            bias_first = precision.endswith("bias_first")
            h = np.asarray(x).astype(np.float16)
            for k in range(len(self.layers)):
                h = np.maximum(self._strict(k, h, bias_first), np.float16(0))
                hidden.append(h.astype(np.float32))
            out = self._strict(len(self.layers), h, bias_first).astype(np.float32)
        return (out, hidden) if return_hidden else out

    def _strict(self, k: int, h: np.ndarray, bias_first: bool) -> np.ndarray:
        """One affine map with every operation rounded to 16-bit floats (no fused multiply-add)."""
        if k not in self._rows:
            w, b = self.layers[k] if k < len(self.layers) else self.readout
            nonzero = w != 0
            width = max(1, int(nonzero.sum(axis=1).max()))
            idx = np.zeros((w.shape[0], width), dtype=np.int64)
            val = np.zeros((w.shape[0], width), dtype=np.float16)
            rows, cols = np.nonzero(nonzero)
            slot = np.concatenate([np.arange(n) for n in np.bincount(rows, minlength=w.shape[0])]) if len(rows) else rows
            idx[rows, slot] = cols
            val[rows, slot] = w[rows, cols].astype(np.float16)
            self._rows[k] = (idx, val, b.astype(np.float16))
        idx, val, bias = self._rows[k]
        acc = np.zeros((h.shape[0], idx.shape[0]), dtype=np.float16)
        if bias_first:
            acc = acc + bias
        for t in range(idx.shape[1]):
            acc = acc + h[:, idx[:, t]] * val[:, t]
        return acc if bias_first else acc + bias

    # ---- facts --------------------------------------------------------------------------------
    @property
    def widths(self) -> "list[int]":
        return [len(self.input_names)] + [w.shape[0] for w, _ in self.layers] + [len(self.output_names)]

    @property
    def n_params(self) -> int:
        return sum(w.size + b.size for w, b in self.layers + [self.readout])

    @property
    def n_nonzero(self) -> int:
        return int(sum(np.count_nonzero(w) + np.count_nonzero(b) for w, b in self.layers + [self.readout]))

    # ---- storage ------------------------------------------------------------------------------
    def state_dict(self, dtype=np.float32) -> "dict[str, np.ndarray]":
        sd = {}
        for i, (w, b) in enumerate(self.layers):
            sd[f"layers.{i}.weight"] = w.astype(dtype)
            sd[f"layers.{i}.bias"] = b.astype(dtype)
        sd["readout.weight"] = self.readout[0].astype(dtype)
        sd["readout.bias"] = self.readout[1].astype(dtype)
        return sd

    @staticmethod
    def from_state_dict(sd: "dict[str, np.ndarray]", input_names=None, output_names=None) -> "Circuit":
        n = len([k for k in sd if k.startswith("layers.") and k.endswith(".weight")])
        layers = [(np.asarray(sd[f"layers.{i}.weight"], np.float32), np.asarray(sd[f"layers.{i}.bias"], np.float32))
                  for i in range(n)]
        readout = (np.asarray(sd["readout.weight"], np.float32), np.asarray(sd["readout.bias"], np.float32))
        return Circuit(list(input_names or [f"in{i}" for i in range(layers[0][0].shape[1])]),
                       list(output_names or [f"out{i}" for i in range(readout[0].shape[0])]), layers, readout)


class CircuitBuilder:
    """Collects named ReLU units layer by layer and compiles them to dense matrices."""

    def __init__(self, input_names: "list[str]") -> None:
        self.input_names = list(input_names)
        self._names: list[list[str]] = [list(input_names)]
        self._index: list[dict[str, int]] = [{n: i for i, n in enumerate(input_names)}]
        self._rows: list[list[Expr]] = [[]]
        self._outputs: list[tuple[str, Expr]] = []
        self.bit_units: list[list[tuple[str, str]]] = [[]]  # (hi, lo) names per layer, for tests

    def layer(self) -> "CircuitBuilder":
        """Start the next ReLU layer."""
        self._names.append([])
        self._index.append({})
        self._rows.append([])
        self.bit_units.append([])
        return self

    def _check(self, expr: Expr, where: str) -> None:
        prev = self._index[-2] if where == "unit" else self._index[-1]
        unknown = [n for n in expr.terms if n not in prev]
        if unknown:
            raise KeyError(f"{where} refers to units that are not in the previous layer: {unknown[:5]}")

    def relu(self, name: str, expr: Expr) -> Expr:
        """Add the unit relu(expr) to the current layer and return it as an expression."""
        if len(self._names) < 2:
            raise RuntimeError("call layer() before adding units")
        if name in self._index[-1]:
            raise KeyError(f"duplicate unit name {name!r} in this layer")
        self._check(expr, "unit")
        self._index[-1][name] = len(self._names[-1])
        self._names[-1].append(name)
        self._rows[-1].append(expr)
        return Expr.unit(name)

    keep = relu  # carries a non-negative signal to the next layer unchanged

    def bit(self, name: str, expr: Expr, low: float = 0.25, high: float = 0.75) -> Expr:
        """Restoring bit: 0 where expr <= low, 1 where expr >= high. Uses two units."""
        gain = 1.0 / (high - low)
        hi = self.relu(name + ".hi", gain * (expr - low))
        lo = self.relu(name + ".lo", gain * (expr - high))
        self.bit_units[-1].append((name + ".hi", name + ".lo"))
        return hi - lo

    def all_of(self, name: str, true: "list[Expr]", false: "list[Expr] | None" = None) -> Expr:
        """Bit that is 1 exactly when every bit in `true` is 1 and every bit in `false` is 0."""
        s = Expr.total(true) - Expr.total(false or []) - (len(true) - 1)
        return self.bit(name, s)

    def output(self, name: str, expr: Expr) -> None:
        self._check(expr, "output")
        self._outputs.append((name, expr))

    def unit_index(self, layer: int, name: str) -> int:
        """Position of a unit in a layer (layer 1 is the first ReLU layer)."""
        return self._index[layer][name]

    def _dense(self, rows: "list[Expr]", prev: "dict[str, int]") -> "tuple[np.ndarray, np.ndarray]":
        w = np.zeros((len(rows), len(prev)), dtype=np.float32)
        b = np.zeros(len(rows), dtype=np.float32)
        for r, expr in enumerate(rows):
            for n, v in expr.terms.items():
                w[r, prev[n]] = v
            b[r] = expr.const
        return w, b

    def compile(self) -> Circuit:
        layers = [self._dense(self._rows[i], self._index[i - 1]) for i in range(1, len(self._names))]
        readout = self._dense([e for _, e in self._outputs], self._index[-1])
        return Circuit(self.input_names, [n for n, _ in self._outputs], layers, readout)
