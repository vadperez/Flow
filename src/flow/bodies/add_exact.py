"""add_exact: exact signed addition and subtraction of 12-digit numbers as written weights.

Inputs  (65): code of a (32), code of b (32), op (0 = add, 1 = subtract)
Outputs (33): code of the result (32), overflow flag

The circuit is logic on bits, in four ReLU layers:

    L1  read every digit as a one-hot bit; read the signs, the operation, the decimal-point position
    L2  one bit per (place, digit of a, digit of b); every per-place fact (starts a carry, passes
        it on, a > b, a < b, a = b, digit of a+b, a-b, b-a) is a sum of these bits.
        One bit per combination of (sign of a, sign of b, operation).
    L3  carry and borrow look-ahead; comparison of the two magnitudes
    L4  result digits: for each of the three candidates (a+b, a-b, b-a), the digit with the
        incoming carry or borrow applied, selected by the signs and the comparison; result sign;
        overflow

Every logic unit is a restoring bit (see circuit.py), so input noise and 16-bit rounding are
removed at each layer and a result can be fed straight into the next call.

Both numbers must have the same number of decimal places. `AddExact` aligns them first, in
plain code; writing that alignment as weights is left to pipeline step S5A.2.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..typesys import numcode as num
from ..typesys.numcode import Number, NumberRangeError
from ..typesys.values import Kind, Value
from .base import Body, BodyError, Signature
from .circuit import Circuit, CircuitBuilder, Expr

P = num.N_DIGITS
# A digit is read as v when the dot product of its (cos, sin) pair with prototype v exceeds T_HI;
# the nearest other prototype gives cos(36 deg) = 0.809, below T_LO. Both gains are exact in binary.
T_LO, T_HI = 0.875, 0.955
INPUT_DIM = 2 * num.CODE_DIM + 1
OUTPUT_DIM = num.CODE_DIM + 1


def build(return_builder: bool = False):
    names = [f"a.{k}" for k in range(num.CODE_DIM)] + [f"b.{k}" for k in range(num.CODE_DIM)] + ["op"]
    cb = CircuitBuilder(names)
    code = lambda arg, k: Expr.unit(f"{arg}.{k}")

    # ---- L1: read the inputs as bits ------------------------------------------------------------
    cb.layer()
    onehot = {}
    for arg in "ab":
        for i in range(P):
            c, s = code(arg, 1 + 2 * i), code(arg, 2 + 2 * i)
            for v in range(10):
                dot = float(num.PROTO[v, 0]) * c + float(num.PROTO[v, 1]) * s
                onehot[arg, i, v] = cb.bit(f"{arg}{i}={v}", dot, T_LO, T_HI)
    a_neg = cb.bit("a_neg", (1 - code("a", 0)) * 0.5)
    b_neg = cb.bit("b_neg", (1 - code("b", 0)) * 0.5)
    sub = cb.bit("sub", Expr.unit("op"))
    scale = [cb.bit(f"scale{k}", code("a", num.SCALE.start + k)) for k in range(num.MAX_SCALE + 1)]

    # ---- L2: one bit per (place, digit of a, digit of b); one bit per sign case -----------------
    cb.layer()
    pair = {(i, v, w): cb.all_of(f"p{i}.{v}{w}", [onehot["a", i, v], onehot["b", i, w]])
            for i in range(P) for v in range(10) for w in range(10)}
    case = {}
    for an in (0, 1):
        for bn in (0, 1):
            for op in (0, 1):
                bits = [(a_neg, an), (b_neg, bn), (sub, op)]
                case[an, bn, op] = cb.all_of(f"case{an}{bn}{op}", [e for e, on in bits if on], [e for e, on in bits if not on]) \
                    if any(on for _, on in bits) else cb.bit("case000", 1 - a_neg - b_neg - sub)
    scale = [cb.keep(f"scale{k}", e) for k, e in enumerate(scale)]

    def where(i, pred) -> Expr:
        return Expr.total(pair[i, v, w] for v in range(10) for w in range(10) if pred(v, w))

    def cases(pred) -> Expr:  # b counts as negative when exactly one of (b is negative, subtract) holds
        return Expr.total(e for (an, bn, op), e in case.items() if pred(an, bn ^ op))

    gen = [where(i, lambda v, w: v + w >= 10) for i in range(P)]
    prop = [where(i, lambda v, w: v + w == 9) for i in range(P)]
    gt = [where(i, lambda v, w: v > w) for i in range(P)]
    lt = [where(i, lambda v, w: v < w) for i in range(P)]
    eq = [where(i, lambda v, w: v == w) for i in range(P)]
    z = [[where(i, lambda v, w, r=r: (v + w) % 10 == r) for r in range(10)] for i in range(P)]  # a + b
    y = [[where(i, lambda v, w, r=r: (v - w) % 10 == r) for r in range(10)] for i in range(P)]  # a - b
    x = [[where(i, lambda v, w, r=r: (w - v) % 10 == r) for r in range(10)] for i in range(P)]  # b - a

    # ---- L3: look-ahead and comparison -----------------------------------------------------------
    cb.layer()

    def chain(tag, start, through, j, i) -> Expr:
        """Bit: place j starts a carry (or borrow) and places j+1 .. i-1 pass it on to place i."""
        return cb.all_of(f"{tag}{j}>{i}", [start[j]] + [through[k] for k in range(j + 1, i)])

    carry = {i: Expr.total(chain("C", gen, prop, j, i) for j in range(i)) for i in range(1, P + 1)}
    bor_ab = {i: Expr.total(chain("Bab", lt, eq, j, i) for j in range(i)) for i in range(1, P)}
    bor_ba = {i: Expr.total(chain("Bba", gt, eq, j, i) for j in range(i)) for i in range(1, P)}
    # |a| > |b|: some place has a > b and all higher places are equal
    a_gt_b = Expr.total(cb.all_of(f"GT{i}", [gt[i]] + [eq[j] for j in range(i + 1, P)]) for i in range(P))
    a_lt_b = Expr.total(cb.all_of(f"LT{i}", [lt[i]] + [eq[j] for j in range(i + 1, P)]) for i in range(P))
    z = [[cb.keep(f"z{i}.{r}", z[i][r]) for r in range(10)] for i in range(P)]
    y = [[cb.keep(f"y{i}.{r}", y[i][r]) for r in range(10)] for i in range(P)]
    x = [[cb.keep(f"x{i}.{r}", x[i][r]) for r in range(10)] for i in range(P)]
    same = cb.bit("same", cases(lambda an, bn: an == bn))  # effective signs agree: add magnitudes
    differ = cb.bit("differ", cases(lambda an, bn: an != bn))  # subtract the smaller magnitude
    both_neg = cb.bit("both_neg", cases(lambda an, bn: an and bn))
    only_a_neg = cb.bit("only_a_neg", cases(lambda an, bn: an and not bn))
    only_b_neg = cb.bit("only_b_neg", cases(lambda an, bn: bn and not an))
    both_zero = cb.all_of("both_zero", [pair[i, 0, 0] for i in range(P)])  # keeps "-0 + -0" from giving "-0"
    scale = [cb.keep(f"scale{k}", e) for k, e in enumerate(scale)]

    # ---- L4: result digits, sign, overflow ------------------------------------------------------
    cb.layer()
    digit = [[Expr() for _ in range(10)] for _ in range(P)]
    candidates = [("S", z, carry, +1, [same], []),  # a + b
                  ("A", y, bor_ab, -1, [differ], [a_lt_b]),  # a - b when |a| >= |b|
                  ("B", x, bor_ba, -1, [differ, a_lt_b], [])]  # b - a when |a| < |b|
    for tag, bits, into, step, on, off in candidates:
        for i in range(P):
            for r in range(10):
                if i not in into:  # place 0 receives no carry or borrow
                    digit[i][r] += cb.all_of(f"{tag}{i}.{r}", [bits[i][r]] + on, off)
                    continue
                digit[i][r] += cb.all_of(f"{tag}{i}.{r}.n", [bits[i][r]] + on, [into[i]] + off)
                digit[i][r] += cb.all_of(f"{tag}{i}.{r}.m", [bits[i][(r - step) % 10], into[i]] + on, off)
    negative = (cb.all_of("neg_sum", [both_neg], [both_zero]) + cb.all_of("neg_ab", [only_a_neg, a_gt_b])
                + cb.all_of("neg_ba", [only_b_neg, a_lt_b]))
    overflow = cb.all_of("overflow", [carry[P], same])
    scale = [cb.keep(f"scale{k}", e) for k, e in enumerate(scale)]

    # ---- readout: back to the NUMBER code ----------------------------------------------------------
    cb.output("r.0", 1 - 2 * negative)
    for i in range(P):
        cb.output(f"r.{1 + 2 * i}", Expr.total(float(num.PROTO[r, 0]) * digit[i][r] for r in range(10)))
        cb.output(f"r.{2 + 2 * i}", Expr.total(float(num.PROTO[r, 1]) * digit[i][r] for r in range(10)))
    for k, e in enumerate(scale):
        cb.output(f"r.{num.SCALE.start + k}", e)
    cb.output("overflow", overflow)
    circuit = cb.compile()
    return (circuit, cb) if return_builder else circuit


def build_circuit() -> Circuit:
    return build()


def pack_inputs(a_codes: np.ndarray, b_codes: np.ndarray, subtract: np.ndarray) -> np.ndarray:
    """[batch, 32], [batch, 32], [batch] -> [batch, 65]."""
    return np.concatenate([a_codes, b_codes, np.asarray(subtract, np.float32)[:, None]], axis=1).astype(np.float32)


def align(a: Number, b: Number) -> "tuple[Number, Number]":
    """Give both numbers the same number of decimal places. Plain code for now (see module docstring)."""
    scale = max(a.scale, b.scale)
    return a.rescaled(scale), b.rescaled(scale)


class AddExact(Body):
    """The add_exact body: operations "add" and "subtract" on two NUMBERs."""

    function = "add_exact"
    operations = ("add", "subtract")
    signature = Signature((Kind.NUMBER, Kind.NUMBER), Kind.NUMBER)

    def __init__(self, circuit: "Circuit | None" = None) -> None:
        self.circuit = circuit or build_circuit()

    def compute(self, a: Number, b: Number, subtract: bool = False) -> "tuple[Number, bool]":
        """Returns (result, overflow). On overflow the result is not meaningful."""
        try:
            a2, b2 = align(a, b)
        except NumberRangeError:
            return Number.from_parts(False, 0, 0), True
        xin = pack_inputs(num.encode(a2)[None], num.encode(b2)[None], np.array([1.0 if subtract else 0.0]))
        out = self.circuit.forward(xin)[0]
        return num.decode(out[: num.CODE_DIM]).normalized(), bool(out[num.CODE_DIM] > 0.5)

    def run(self, operation: str, args: "list[Value]") -> Value:
        result, overflow = self.compute(args[0].payload, args[1].payload, subtract=(operation == "subtract"))
        if overflow:
            raise BodyError("add_exact: the result does not fit 12 digits")
        return Value(Kind.NUMBER, result)

    def save(self, path: "str | Path", dtype=np.float16) -> None:
        from safetensors.numpy import save_file

        save_file(self.circuit.state_dict(dtype), str(path), metadata={"format": f"flow-circuit-1:{self.function}"})

    @classmethod
    def load(cls, path: "str | Path") -> "AddExact":
        from safetensors.numpy import load_file

        return cls(Circuit.from_state_dict(load_file(str(path))))
