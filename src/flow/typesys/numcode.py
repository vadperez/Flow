"""NUMBER type: signed fixed-point decimals in a designed 32-dimensional code.

Layout of the code vector (pipeline v0.3, section 2.3):
    [0]       sign: +1 or -1
    [1:25]    12 decimal digits of the mantissa, least significant first,
              each as (cos, sin) of 2*pi*digit/10
    [25:32]   position of the decimal point, one-hot over 0..6 decimal places

value = sign * mantissa * 10**(-scale)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

import numpy as np

N_DIGITS = 12
MAX_SCALE = 6
CODE_DIM = 32
SIGN = 0
DIGITS = slice(1, 1 + 2 * N_DIGITS)
SCALE = slice(1 + 2 * N_DIGITS, CODE_DIM)

# The ten digit prototypes on the unit circle, 36 degrees apart.
PROTO = np.array(
    [[math.cos(2 * math.pi * d / 10), math.sin(2 * math.pi * d / 10)] for d in range(10)],
    dtype=np.float32,
)


class NumberRangeError(ValueError):
    """The value does not fit 12 digits with at most 6 decimal places."""


@dataclass(frozen=True)
class Number:
    negative: bool
    digits: tuple[int, ...]  # N_DIGITS mantissa digits, least significant first
    scale: int  # decimal places, 0..MAX_SCALE

    def __post_init__(self) -> None:
        if len(self.digits) != N_DIGITS or any(not 0 <= d <= 9 for d in self.digits):
            raise NumberRangeError(f"need {N_DIGITS} digits in 0..9, got {self.digits!r}")
        if not 0 <= self.scale <= MAX_SCALE:
            raise NumberRangeError(f"scale {self.scale} outside 0..{MAX_SCALE}")

    @property
    def mantissa(self) -> int:
        return sum(d * 10**i for i, d in enumerate(self.digits))

    def to_decimal(self) -> Decimal:
        value = Decimal(self.mantissa).scaleb(-self.scale)
        return -value if self.negative else value

    def normalized(self) -> "Number":
        """Canonical form: no trailing fractional zeros, and zero is never negative."""
        mantissa, scale = self.mantissa, self.scale
        while scale > 0 and mantissa % 10 == 0:
            mantissa //= 10
            scale -= 1
        return Number.from_parts(self.negative and mantissa != 0, mantissa, scale)

    def rescaled(self, scale: int) -> "Number":
        """Same value with more decimal places (digits shift up). Raises if it no longer fits."""
        if scale < self.scale:
            raise NumberRangeError("rescaled() only increases the number of decimal places")
        return Number.from_parts(self.negative, self.mantissa * 10 ** (scale - self.scale), scale)

    def text(self) -> str:
        """Canonical text, the form the emitter prints and the scanner reads back."""
        n = self.normalized()
        s = str(n.mantissa).rjust(n.scale + 1, "0")
        body = s if n.scale == 0 else f"{s[:-n.scale]}.{s[-n.scale:]}"
        return ("-" if n.negative else "") + body

    @staticmethod
    def from_parts(negative: bool, mantissa: int, scale: int) -> "Number":
        if mantissa < 0 or mantissa >= 10**N_DIGITS:
            raise NumberRangeError(f"mantissa {mantissa} does not fit {N_DIGITS} digits")
        digits = tuple((mantissa // 10**i) % 10 for i in range(N_DIGITS))
        return Number(bool(negative), digits, scale)

    @staticmethod
    def from_value(value: "int | str | Decimal") -> "Number":
        """Build a canonical Number from an int, a Decimal, or text such as '-1,250.50'."""
        if isinstance(value, str):
            value = value.replace(",", "").replace("−", "-").strip()
        try:
            dec = Decimal(value)
        except (InvalidOperation, ValueError) as exc:
            raise NumberRangeError(f"not a number: {value!r}") from exc
        if not dec.is_finite():
            raise NumberRangeError(f"not a finite number: {value!r}")
        sign, digits, exponent = dec.as_tuple()
        mantissa = int("".join(map(str, digits)) or "0")
        if exponent > 0:
            mantissa *= 10**exponent
            scale = 0
        else:
            scale = -int(exponent)
        while scale > 0 and mantissa % 10 == 0:  # strip trailing fractional zeros first
            mantissa //= 10
            scale -= 1
        if scale > MAX_SCALE:
            raise NumberRangeError(f"{value!r} has more than {MAX_SCALE} decimal places")
        return Number.from_parts(bool(sign) and mantissa != 0, mantissa, scale)


def encode(n: Number) -> np.ndarray:
    """Number -> code vector of CODE_DIM float32 values. The number is encoded as given (not normalized)."""
    v = np.zeros(CODE_DIM, dtype=np.float32)
    v[SIGN] = -1.0 if n.negative else 1.0
    v[DIGITS] = PROTO[list(n.digits)].reshape(-1)
    v[SCALE.start + n.scale] = 1.0
    return v


def decode(v: np.ndarray) -> Number:
    """Code vector -> Number, by nearest prototype. Tolerates noise up to about +-18 degrees per digit."""
    v = np.asarray(v, dtype=np.float32)
    pairs = v[DIGITS].reshape(N_DIGITS, 2)
    digits = tuple(int(i) for i in np.argmax(pairs @ PROTO.T, axis=1))
    scale = int(np.argmax(v[SCALE]))
    return Number(bool(v[SIGN] < 0), digits, scale)


def encode_batch(numbers: "list[Number]") -> np.ndarray:
    return np.stack([encode(n) for n in numbers])


# ---- array forms, for large batches (training data, exhaustive tests) ---------------------------
_POW10 = 10 ** np.arange(N_DIGITS, dtype=np.int64)


def encode_arrays(negative, mantissa, scale=0) -> np.ndarray:
    """negative [n] bool, mantissa [n] int64 (0 .. 10**12 - 1), scale [n] or int -> codes [n, 32]."""
    mantissa = np.asarray(mantissa, dtype=np.int64)
    n = mantissa.shape[0]
    digits = (mantissa[:, None] // _POW10[None, :]) % 10
    v = np.zeros((n, CODE_DIM), dtype=np.float32)
    v[:, SIGN] = np.where(np.asarray(negative, dtype=bool), -1.0, 1.0)
    v[:, DIGITS] = PROTO[digits].reshape(n, -1)
    v[np.arange(n), SCALE.start + np.broadcast_to(np.asarray(scale, dtype=np.int64), (n,))] = 1.0
    return v


def decode_arrays(v: np.ndarray) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """codes [n, 32] -> (negative [n] bool, mantissa [n] int64, scale [n] int64)."""
    v = np.asarray(v, dtype=np.float32)
    pairs = v[:, DIGITS].reshape(v.shape[0], N_DIGITS, 2)
    digits = np.argmax(pairs @ PROTO.T, axis=2).astype(np.int64)
    return v[:, SIGN] < 0, (digits * _POW10[None, :]).sum(axis=1), np.argmax(v[:, SCALE], axis=1)
