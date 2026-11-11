"""compare_numeric: exact comparison of two signed numbers as written weights.

Inputs  (67): code of a (32), code of b (32), operation one-hot (greater, less, equal)
Output   (1): FLAG code: +1 for yes, -1 for no

Same construction as add_exact: one-hot digits, one bit per (place, digit of a, digit of b),
then "some place decides and all higher places are equal", then the signs.
Both numbers must have the same number of decimal places (aligned by the wrapper, in plain code).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..typesys import numcode as num
from ..typesys.numcode import Number, NumberRangeError
from ..typesys.values import Kind, Value
from .add_exact import T_HI, T_LO, align
from .base import Body, BodyError, Signature
from .circuit import Circuit, CircuitBuilder, Expr

P = num.N_DIGITS
OPERATIONS = ("greater", "less", "equal")


def build(return_builder: bool = False):
    names = [f"a.{k}" for k in range(num.CODE_DIM)] + [f"b.{k}" for k in range(num.CODE_DIM)] + [f"op.{o}" for o in OPERATIONS]
    cb = CircuitBuilder(names)
    code = lambda arg, k: Expr.unit(f"{arg}.{k}")

    cb.layer()  # L1: read the inputs as bits
    onehot = {}
    for arg in "ab":
        for i in range(P):
            c, s = code(arg, 1 + 2 * i), code(arg, 2 + 2 * i)
            for v in range(10):
                onehot[arg, i, v] = cb.bit(f"{arg}{i}={v}", float(num.PROTO[v, 0]) * c + float(num.PROTO[v, 1]) * s, T_LO, T_HI)
    a_neg = cb.bit("a_neg", (1 - code("a", 0)) * 0.5)
    b_neg = cb.bit("b_neg", (1 - code("b", 0)) * 0.5)
    op = {o: cb.bit(f"op.{o}", Expr.unit(f"op.{o}")) for o in OPERATIONS}

    cb.layer()  # L2: one bit per (place, digit of a, digit of b); one bit per sign case
    pair = {(i, v, w): cb.all_of(f"p{i}.{v}{w}", [onehot["a", i, v], onehot["b", i, w]])
            for i in range(P) for v in range(10) for w in range(10)}
    signs = {"pp": cb.bit("pp", 1 - a_neg - b_neg), "pn": cb.all_of("pn", [b_neg], [a_neg]),
             "np": cb.all_of("np", [a_neg], [b_neg]), "nn": cb.all_of("nn", [a_neg, b_neg])}
    op = {o: cb.keep(f"op.{o}", e) for o, e in op.items()}
    where = lambda i, pred: Expr.total(pair[i, v, w] for v in range(10) for w in range(10) if pred(v, w))
    gt = [where(i, lambda v, w: v > w) for i in range(P)]
    lt = [where(i, lambda v, w: v < w) for i in range(P)]
    eq = [where(i, lambda v, w: v == w) for i in range(P)]

    cb.layer()  # L3: comparison of the magnitudes
    mag_gt = Expr.total(cb.all_of(f"GT{i}", [gt[i]] + [eq[j] for j in range(i + 1, P)]) for i in range(P))
    mag_lt = Expr.total(cb.all_of(f"LT{i}", [lt[i]] + [eq[j] for j in range(i + 1, P)]) for i in range(P))
    signs = {k: cb.bit(k, e) for k, e in signs.items()}
    op = {o: cb.keep(f"op.{o}", e) for o, e in op.items()}

    cb.layer()  # L4: apply the signs, select the asked relation
    greater = [[signs["pp"], mag_gt], [signs["pn"]], [signs["nn"], mag_lt]]
    less = [[signs["pp"], mag_lt], [signs["np"]], [signs["nn"], mag_gt]]
    yes = Expr()
    for k, terms in enumerate(greater):
        yes += cb.all_of(f"greater{k}", terms + [op["greater"]])
    for k, terms in enumerate(less):
        yes += cb.all_of(f"less{k}", terms + [op["less"]])
    yes += cb.all_of("equal_pp", [signs["pp"], op["equal"]], [mag_gt, mag_lt])
    yes += cb.all_of("equal_nn", [signs["nn"], op["equal"]], [mag_gt, mag_lt])
    cb.output("flag", 2 * yes - 1)
    circuit = cb.compile()
    return (circuit, cb) if return_builder else circuit


def pack_inputs(a_codes: np.ndarray, b_codes: np.ndarray, operation: "np.ndarray | int") -> np.ndarray:
    """[batch, 32], [batch, 32], operation index [batch] or int -> [batch, 67]."""
    n = a_codes.shape[0]
    onehot = np.zeros((n, len(OPERATIONS)), np.float32)
    onehot[np.arange(n), np.broadcast_to(np.asarray(operation, np.int64), (n,))] = 1.0
    return np.concatenate([a_codes, b_codes, onehot], axis=1).astype(np.float32)


class CompareNumeric(Body):
    """The compare_numeric body: operations "greater", "less", "equal" on two NUMBERs -> FLAG."""

    function = "compare_numeric"
    operations = OPERATIONS
    signature = Signature((Kind.NUMBER, Kind.NUMBER), Kind.FLAG)

    def __init__(self, circuit: "Circuit | None" = None) -> None:
        self.circuit = circuit or build()

    def compute(self, a: Number, b: Number, operation: str) -> bool:
        a2, b2 = align(a.normalized(), b.normalized())  # may raise NumberRangeError
        x = pack_inputs(num.encode(a2)[None], num.encode(b2)[None], OPERATIONS.index(operation))
        return bool(self.circuit.forward(x)[0, 0] > 0)

    def run(self, operation: str, args: "list[Value]") -> Value:
        try:
            return Value(Kind.FLAG, self.compute(args[0].payload, args[1].payload, operation))
        except NumberRangeError as exc:
            raise BodyError("compare_numeric: the numbers cannot be brought to the same decimal places") from exc

    def save(self, path: "str | Path", dtype=np.float16) -> None:
        from safetensors.numpy import save_file

        save_file(self.circuit.state_dict(dtype), str(path), metadata={"format": f"flow-circuit-1:{self.function}"})

    @classmethod
    def load(cls, path: "str | Path") -> "CompareNumeric":
        from safetensors.numpy import load_file

        return cls(Circuit.from_state_dict(load_file(str(path))))
