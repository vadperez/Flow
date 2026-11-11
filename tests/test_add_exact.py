"""Tests of the written adder (pipeline step S5A.1): exactness, not a score.

The carry into a place depends only on which class each lower place falls in (starts a carry,
passes it on, stops it), so covering every class pattern covers the carry logic. The same holds
for borrows and for the comparison of magnitudes with the classes (less, equal, greater).
"""
import itertools
from decimal import Decimal

import numpy as np
import pytest

from flow.bodies.add_exact import AddExact, build, pack_inputs
from flow.bodies.base import BodyError
from flow.typesys import flag, number
from flow.bodies.circuit import CircuitBuilder, Expr
from flow.typesys import numcode as num
from flow.typesys.numcode import Number

LIMIT = 10 ** num.N_DIGITS
POW10 = 10 ** np.arange(num.N_DIGITS, dtype=np.int64)


@pytest.fixture(scope="module")
def built():
    return build(return_builder=True)


@pytest.fixture(scope="module")
def circuit(built):
    return built[0]


def check(circuit, a, a_neg, b, b_neg, sub, precision="fp32", chunk=20000):
    """Run the circuit on integer operands and compare with integer arithmetic. Returns raw outputs."""
    a, b = np.asarray(a, np.int64), np.asarray(b, np.int64)
    a_neg, b_neg, sub = (np.broadcast_to(np.asarray(v, bool), a.shape) for v in (a_neg, b_neg, sub))
    want = np.where(a_neg, -a, a) + np.where(b_neg ^ sub, -b, b)
    outs = []
    for s in range(0, len(a), chunk):
        e = slice(s, s + chunk)
        x = pack_inputs(num.encode_arrays(a_neg[e], a[e]), num.encode_arrays(b_neg[e], b[e]), sub[e])
        outs.append(circuit.forward(x, precision=precision))
    out = np.concatenate(outs)
    neg, mant, scale = num.decode_arrays(out[:, : num.CODE_DIM])
    got = np.where(neg, -mant, mant)
    overflow = out[:, num.CODE_DIM] > 0.5
    fits = np.abs(want) < LIMIT
    assert (overflow == ~fits).all(), "overflow flag wrong"
    wrong = np.flatnonzero(fits & (got != want))
    assert len(wrong) == 0, f"{len(wrong)} wrong, first: {a[wrong[0]]} {b[wrong[0]]} -> {got[wrong[0]]}, want {want[wrong[0]]}"
    assert (neg[fits] == (want[fits] < 0)).all(), "sign of a zero result must be positive"
    assert (scale == 0).all()
    return out


def from_classes(patterns: np.ndarray, pools, rng) -> "tuple[np.ndarray, np.ndarray]":
    """patterns [n, 12] of class ids -> operand pairs whose places fall in those classes."""
    a = np.zeros(len(patterns), np.int64)
    b = np.zeros(len(patterns), np.int64)
    for i in range(num.N_DIGITS):
        for cls, pool in enumerate(pools):
            rows = np.flatnonzero(patterns[:, i] == cls)
            pick = np.asarray(pool)[rng.integers(0, len(pool), len(rows))]
            a[rows] += pick[:, 0] * POW10[i]
            b[rows] += pick[:, 1] * POW10[i]
    return a, b


ALL_PATTERNS = np.array(list(itertools.product(range(3), repeat=num.N_DIGITS)), dtype=np.int8)  # 3^12 = 531,441
PAIRS = [(v, w) for v in range(10) for w in range(10)]
CARRY_POOLS = [[p for p in PAIRS if sum(p) >= 10], [p for p in PAIRS if sum(p) == 9], [p for p in PAIRS if sum(p) <= 8]]
ORDER_POOLS = [[p for p in PAIRS if p[0] < p[1]], [p for p in PAIRS if p[0] == p[1]], [p for p in PAIRS if p[0] > p[1]]]


def test_circuit_shape(circuit):
    assert circuit.widths == [65, 500, 2423, 847, 1395, 33]
    assert 4_500_000 < circuit.n_params < 4_600_000 and circuit.n_nonzero < 150_000


def test_builder_rejects_wrong_layer():
    cb = CircuitBuilder(["x"])
    cb.layer()
    cb.relu("h", Expr.unit("x"))
    cb.layer()
    with pytest.raises(KeyError):
        cb.relu("bad", Expr.unit("x"))  # x is two layers back


def test_restoring_bit():
    cb = CircuitBuilder(["x"])
    cb.layer()
    cb.output("y", cb.bit("b", Expr.unit("x")))
    c = cb.compile()
    x = np.array([[-1.0], [0.0], [0.25], [0.5], [0.75], [1.0], [1.2]], np.float32)
    assert c.forward(x)[:, 0].tolist() == [0, 0, 0, 0.5, 1, 1, 1]


def test_small_table_all_signs_and_ops(circuit):
    """Every pair from a set of special values, with every sign and both operations."""
    values = [0, 1, 2, 5, 9, 10, 11, 19, 99, 100, 101, 999, 1000, LIMIT - 1, LIMIT - 2, 5 * 10 ** 11, 123456789012,
              10 ** 11, 10 ** 11 - 1, 909090909090, 90909090909]
    rows = np.array([(a, an, b, bn, s) for a, b in itertools.product(values, repeat=2)
                     for an in (0, 1) for bn in (0, 1) for s in (0, 1)], dtype=np.int64)
    check(circuit, rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3], rows[:, 4])


@pytest.mark.slow
def test_every_carry_pattern(circuit):
    """All 3^12 patterns of (starts a carry, passes it on, stops it). Run with: pytest -m slow"""
    a, b = from_classes(ALL_PATTERNS, CARRY_POOLS, np.random.default_rng(0))
    check(circuit, a, False, b, False, False)


@pytest.mark.slow
def test_every_borrow_and_comparison_pattern(circuit):
    """All 3^12 patterns of (a digit < b digit, equal, greater): a - b, then (-a) + b."""
    a, b = from_classes(ALL_PATTERNS, ORDER_POOLS, np.random.default_rng(1))
    check(circuit, a, False, b, False, True)
    half = slice(0, None, 2)
    check(circuit, a[half], True, b[half], False, False)


def test_pattern_sample(circuit):
    """A sample of the patterns that the two slow tests cover in full."""
    rng = np.random.default_rng(9)
    rows = rng.choice(len(ALL_PATTERNS), 40000, replace=False)
    a, b = from_classes(ALL_PATTERNS[rows], CARRY_POOLS, rng)
    check(circuit, a, False, b, False, False)
    a, b = from_classes(ALL_PATTERNS[rows], ORDER_POOLS, rng)
    check(circuit, a, rng.random(len(a)) < 0.5, b, False, True)


@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_random_pairs(circuit, precision):
    rng = np.random.default_rng(2)
    n = 30000
    a = rng.integers(0, 10 ** rng.integers(1, 13, n))
    b = rng.integers(0, 10 ** rng.integers(1, 13, n))
    check(circuit, a, rng.random(n) < 0.4, b, rng.random(n) < 0.4, rng.random(n) < 0.5, precision)


def test_pattern_sample_in_16_bit(circuit):
    rng = np.random.default_rng(3)
    rows = rng.choice(len(ALL_PATTERNS), 30000, replace=False)
    a, b = from_classes(ALL_PATTERNS[rows], CARRY_POOLS, rng)
    check(circuit, a, False, b, False, False, "fp16")
    a, b = from_classes(ALL_PATTERNS[rows], ORDER_POOLS, rng)
    check(circuit, a, rng.random(len(a)) < 0.5, b, False, True, "fp16")


@pytest.mark.parametrize("precision", ["fp16_strict", "fp16_strict_bias_first"])
def test_strict_16_bit_arithmetic(circuit, precision):
    """Every product and partial sum rounded to 16 bits: the hardest patterns plus random ones."""
    rng = np.random.default_rng(4)
    rows = np.concatenate([rng.choice(len(ALL_PATTERNS), 300, replace=False), [0, len(ALL_PATTERNS) // 2, len(ALL_PATTERNS) - 1]])
    a, b = from_classes(ALL_PATTERNS[rows], CARRY_POOLS, rng)
    check(circuit, a, False, b, False, False, precision)
    a, b = from_classes(ALL_PATTERNS[rows], ORDER_POOLS, rng)
    check(circuit, a, rng.random(len(a)) < 0.5, b, rng.random(len(a)) < 0.5, True, precision)
    long_carry = np.array([LIMIT - 1, LIMIT - 1, 0, 1, 10 ** 11], np.int64)  # carry or borrow through all places
    other = np.array([1, LIMIT - 1, 1, LIMIT - 1, 1], np.int64)
    for sub in (False, True):
        check(circuit, long_carry, False, other, False, sub, precision)


@pytest.mark.parametrize("precision", ["fp32", "fp16", "fp16_strict_bias_first"])
def test_every_bit_is_clean(built, precision):
    """Inside the circuit every bit stays close to 0 or 1, so no error can add up over layers."""
    circuit, cb = built
    rng = np.random.default_rng(5)
    n = 4000 if precision != "fp16_strict_bias_first" else 200
    a, b = rng.integers(0, LIMIT, n), rng.integers(0, LIMIT, n)
    x = pack_inputs(num.encode_arrays(rng.random(n) < 0.5, a), num.encode_arrays(rng.random(n) < 0.5, b), rng.random(n) < 0.5)
    _, hidden = circuit.forward(x, precision=precision, return_hidden=True)
    for layer, h in enumerate(hidden, start=1):
        hi = [cb.unit_index(layer, name) for name, _ in cb.bit_units[layer]]
        lo = [cb.unit_index(layer, name) for _, name in cb.bit_units[layer]]
        bits = h[:, hi] - h[:, lo]
        worst = np.abs(bits - np.round(bits)).max()
        assert worst < (1e-6 if precision == "fp32" else 0.02), f"layer {layer}: bit off by {worst}"


def test_outputs_are_clean_codes(circuit):
    """Results land on exact code points."""
    rng = np.random.default_rng(6)
    n = 5000
    a, b = rng.integers(0, LIMIT // 2, n), rng.integers(0, LIMIT // 2, n)
    for precision, tolerance in (("fp32", 1e-6), ("fp16", 2e-3)):
        out = check(circuit, a, False, b, rng.random(n) < 0.5, False, precision)
        neg, mant, scale = num.decode_arrays(out[:, : num.CODE_DIM])
        assert np.abs(out[:, : num.CODE_DIM] - num.encode_arrays(neg, mant, scale)).max() < tolerance


def test_tolerates_noisy_inputs(circuit):
    """Noise on the input codes is removed by the first layer."""
    rng = np.random.default_rng(7)
    n = 3000
    a, b = rng.integers(0, LIMIT // 2, n), rng.integers(0, LIMIT // 2, n)
    a_neg, sub = rng.random(n) < 0.5, rng.random(n) < 0.5
    x = pack_inputs(num.encode_arrays(a_neg, a), num.encode_arrays(False, b), sub)
    x[:, : 2 * num.CODE_DIM] += rng.uniform(-0.03, 0.03, (n, 2 * num.CODE_DIM)).astype(np.float32)
    out = circuit.forward(x)
    neg, mant, _ = num.decode_arrays(out[:, : num.CODE_DIM])
    want = np.where(a_neg, -a, a) + np.where(sub, -b, b)
    assert (np.where(neg, -mant, mant) == want).all()
    clean = circuit.forward(pack_inputs(num.encode_arrays(a_neg, a), num.encode_arrays(False, b), sub))
    assert np.abs(out - clean).max() < 1e-5  # the noise does not reach the output at all


@pytest.mark.parametrize("precision", ["fp16", "fp16_strict_bias_first"])
def test_chained_calls_do_not_drift(circuit, precision):
    """The result code is fed back as an argument, call after call (composition without text)."""
    rng = np.random.default_rng(8)
    steps = 300 if precision == "fp16" else 40
    total, code = 0, num.encode(Number.from_value(0))[None]
    for _ in range(steps):
        step = int(rng.integers(-10 ** 9, 10 ** 9))
        b = num.encode_arrays([step < 0], [abs(step)])
        code = circuit.forward(pack_inputs(code, b, np.zeros(1)), precision=precision)[:, : num.CODE_DIM]
        total += step
        neg, mant, _ = num.decode_arrays(code)
        assert (-mant[0] if neg[0] else mant[0]) == total
        assert np.abs(code - num.encode_arrays([total < 0], [abs(total)])).max() < 2e-3


def test_wrapper_aligns_decimal_places_and_saves(tmp_path):
    body = AddExact()
    cases = [("1.5", "0.5", False), ("100", "0.001", True), ("-250.75", "1000", False), ("0.000001", "0.999999", False),
             ("48700", "500", False), ("3", "3", True), ("-0.5", "0.5", False)]
    for a, b, sub in cases:
        got, ovf = body.compute(Number.from_value(a), Number.from_value(b), sub)
        want = Decimal(a) - Decimal(b) if sub else Decimal(a) + Decimal(b)
        assert not ovf and got.to_decimal() == want
    assert body.compute(Number.from_value("999999999999"), Number.from_value("1"))[1]  # overflow
    assert body.compute(Number.from_value("123456789012"), Number.from_value("0.5"))[1]  # cannot align: out of range
    body.save(tmp_path / "body.safetensors")  # stored as 16-bit floats
    again = AddExact.load(tmp_path / "body.safetensors")
    got, ovf = again.compute(Number.from_value("999999.5"), Number.from_value("0.5"))
    assert not ovf and got.text() == "1000000"
    got, ovf = again.compute(Number.from_value("-123456.789"), Number.from_value("0.000001"), True)
    assert not ovf and got.text() == "-123456.789001"
    assert again("subtract", [number("5"), number("7.5")]) == number("-2.5")  # the typed interface
    with pytest.raises(BodyError):
        again("add", [number("999999999999"), number("1")])
    with pytest.raises(BodyError):
        again("add", [number("1"), flag(True)])
