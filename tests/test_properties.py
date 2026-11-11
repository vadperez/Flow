"""Property tests of the exact bodies (pipeline step S1.5): algebraic identities on random numbers."""
from decimal import Decimal

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings
from hypothesis import strategies as st

from flow.bodies import AddExact, CompareNumeric
from flow.bodies.base import BodyError
from flow.typesys import number

ADD, CMP = AddExact(), CompareNumeric()
SMALL = st.integers(min_value=-(10 ** 11), max_value=10 ** 11)
SCALE = st.integers(min_value=0, max_value=3)


def num(value: int, scale: int = 0):
    return number(Decimal(value).scaleb(-scale))


@settings(max_examples=150, deadline=None)
@given(SMALL, SMALL)
def test_addition_commutes(a, b):
    assert ADD("add", [num(a), num(b)]) == ADD("add", [num(b), num(a)])


@settings(max_examples=150, deadline=None)
@given(SMALL, SMALL)
def test_subtraction_undoes_addition(a, b):
    total = ADD("add", [num(a), num(b)])
    assert ADD("subtract", [total, num(b)]) == num(a)
    assert ADD("add", [num(a), num(0)]) == num(a) and ADD("subtract", [num(a), num(a)]) == num(0)


@settings(max_examples=150, deadline=None)
@given(st.integers(-10 ** 8, 10 ** 8), SCALE, st.integers(-10 ** 8, 10 ** 8), SCALE)
def test_matches_decimal_arithmetic_with_decimal_places(a, sa, b, sb):
    x, y = Decimal(a).scaleb(-sa), Decimal(b).scaleb(-sb)
    assert ADD("add", [number(x), number(y)]).payload.to_decimal() == x + y
    assert ADD("subtract", [number(x), number(y)]).payload.to_decimal() == x - y
    assert CMP("greater", [number(x), number(y)]).payload == (x > y)
    assert CMP("equal", [number(x), number(y)]).payload == (x == y)


@settings(max_examples=150, deadline=None)
@given(SMALL, SMALL)
def test_comparison_is_antisymmetric_and_complete(a, b):
    greater, less, equal = (CMP(op, [num(a), num(b)]).payload for op in ("greater", "less", "equal"))
    assert greater + less + equal == 1  # exactly one of the three holds
    assert greater == CMP("less", [num(b), num(a)]).payload


@settings(max_examples=60, deadline=None)
@given(st.integers(10 ** 11, 10 ** 12 - 1), st.integers(9 * 10 ** 11, 10 ** 12 - 1))
def test_overflow_is_reported_not_wrapped(a, b):
    with pytest.raises(BodyError):
        ADD("add", [num(a), num(b)])
