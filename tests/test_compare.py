"""Tests of the written comparison body."""
import itertools

import numpy as np
import pytest

from flow.bodies.base import BodyError
from flow.bodies.compare import OPERATIONS, CompareNumeric, build, pack_inputs
from flow.typesys import Kind, number
from flow.typesys import numcode as num
from test_add_exact import ALL_PATTERNS, LIMIT, ORDER_POOLS, from_classes


@pytest.fixture(scope="module")
def circuit():
    return build()


def check(circuit, a, a_neg, b, b_neg, precision="fp32", chunk=20000):
    a, b = np.asarray(a, np.int64), np.asarray(b, np.int64)
    a_neg = np.broadcast_to(np.asarray(a_neg, bool), a.shape) & (a != 0)  # canonical input: zero is never negative
    b_neg = np.broadcast_to(np.asarray(b_neg, bool), b.shape) & (b != 0)
    for s in range(0, len(a), chunk):  # in chunks: the whole pattern set at once needs about 6 GB of memory
        e = slice(s, s + chunk)
        sa, sb = np.where(a_neg[e], -a[e], a[e]), np.where(b_neg[e], -b[e], b[e])
        ca, cb = num.encode_arrays(a_neg[e], a[e]), num.encode_arrays(b_neg[e], b[e])
        for k, want in enumerate([sa > sb, sa < sb, sa == sb]):
            out = circuit.forward(pack_inputs(ca, cb, k), precision=precision)[:, 0]
            assert (np.abs(np.abs(out) - 1) < 5e-3).all(), "flag code must be +1 or -1"
            assert ((out > 0) == want).all(), OPERATIONS[k]


def test_shape(circuit):
    assert circuit.widths[0] == 67 and circuit.widths[-1] == 1 and len(circuit.layers) == 4


def test_special_values_all_signs(circuit):
    values = [0, 1, 2, 9, 10, 11, 99, 100, LIMIT - 1, LIMIT - 2, 10 ** 11, 123456789012, 123456789013]
    rows = np.array([(a, an, b, bn) for a, b in itertools.product(values, repeat=2) for an in (0, 1) for bn in (0, 1)])
    check(circuit, rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3])


@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_order_patterns_sample(circuit, precision):
    rng = np.random.default_rng(0)
    rows = rng.choice(len(ALL_PATTERNS), 20000, replace=False)
    a, b = from_classes(ALL_PATTERNS[rows], ORDER_POOLS, rng)
    check(circuit, a, rng.random(len(a)) < 0.5, b, rng.random(len(a)) < 0.5, precision)


@pytest.mark.slow
def test_every_order_pattern(circuit):
    a, b = from_classes(ALL_PATTERNS, ORDER_POOLS, np.random.default_rng(1))
    check(circuit, a, False, b, False)
    check(circuit, a, True, b, True)


def test_strict_16_bit(circuit):
    rng = np.random.default_rng(2)
    rows = rng.choice(len(ALL_PATTERNS), 200, replace=False)
    a, b = from_classes(ALL_PATTERNS[rows], ORDER_POOLS, rng)
    check(circuit, a, rng.random(len(a)) < 0.5, b, rng.random(len(a)) < 0.5, "fp16_strict_bias_first")


def test_typed_interface(tmp_path):
    body = CompareNumeric()
    assert body("greater", [number("2.5"), number("2.49")]).payload is True
    assert body("less", [number("-3"), number("0.001")]).payload is True
    assert body("equal", [number("7.50"), number("7.5")]).payload is True
    assert body("equal", [number("7"), number("-7")]).payload is False
    assert body("greater", [number("0"), number("-0.000001")]).kind is Kind.FLAG
    with pytest.raises(BodyError):
        body("greater", [number("123456789012"), number("0.5")])  # cannot be aligned within 12 digits
    body.save(tmp_path / "body.safetensors")
    assert CompareNumeric.load(tmp_path / "body.safetensors")("less", [number("1"), number("2")]).payload is True
