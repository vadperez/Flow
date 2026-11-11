# Flow type system, as implemented in this starter

Status: draft for gate G1 (pipeline step S1.3). Code: `src/flow/typesys/`. Tests: `tests/test_typesys.py`, `tests/test_properties.py`.

## Values

A typed value is 34 numbers: a 32-dimensional code and a 2-dimensional type tag.

| Type | Tag | Code (32 dimensions) | File |
|---|---|---|---|
| NONE | (0, 0) | zeros | `values.py` |
| NUMBER | (1, 0) | `[0]` sign, +1 or −1 · `[1:25]` 12 decimal digits, least significant first, each as (cos, sin) of 2π·digit/10 · `[25:32]` one-hot position of the decimal point, 0 to 6 places | `numcode.py` |
| SYMBOL | (0, 1) | 16-bit symbol id as a ±1 codeword of the Reed–Muller code RM(2,5): 32 bits, minimum distance 8. The id is the symbol's index in the symbol table | `symcode.py` |
| FLAG | (1, 1) | `[0]` +1 for yes, −1 for no | `values.py` |

Limits of NUMBER: 12 significant digits, at most 6 decimal places, sign. A value outside these limits raises `NumberRangeError`; a body that would produce one raises `BodyError` and the model abstains.

Canonical text of a NUMBER: no thousands separators, no trailing zeros after the decimal point, `-` for negative, `0` never negative. What the emitter prints, the scanner reads back as the same value (tested on random values).

## Where values come from

- **Scanner** (`scanner.py`): finds every number in the text and anchors its value at the number's **last token**; anchors a SYMBOL at every token that is listed in the symbol table. It needs only the character span of each token, so it works with any tokenizer that reports offsets.
- **Bodies**: write their result to RES.

Values of earlier tokens are not stored. The scanner recomputes them when a binder points at them.

## Where values go

- **Emitter** (`emitter.py`): prints RES piece by piece in place of the core's own token: a NUMBER digit by digit, most significant first; a SYMBOL as its token; a FLAG as `yes` or `no`. `PieceMap` checks that every piece is exactly one token of the tokenizer in use and lists the pieces that are not.

## Registers (`registers.py`)

| Register | Holds | Persists | Written by |
|---|---|---|---|
| ARG0, ARG1, ARG2 | arguments of the current call | no | binder |
| RES | result of the latest call | until printed or replaced | body |
| STATE | 16 key → value slots; overwrite on the same key, the oldest slot is evicted when full | for the session | state bodies only |
| CTRL | calls made at this position, miss and overflow flags, last function | for the session | runtime |

Session state lives in the registers and in the text, never inside a module. This is what makes unloading and reloading a module safe at any token.

## Precision

All codes are exact in 16-bit floats. The written bodies restore every bit at every layer (`src/flow/bodies/circuit.py`), so a result can be the argument of the next call without drift: tested for 300 chained calls in 16-bit arithmetic, and for 40 with every product and partial sum rounded to 16 bits in the least favourable order.

## Open points for G1

1. Symbols are single tokens in v0. Multi-token names need a span rule (planned for v1).
2. Decimal alignment of two NUMBERs is done in plain code before the written weights run (`add_exact.align`). Writing it as weights is step S5A.2.
3. Numbers written with thousands separators are read (`1,250`), but always printed without them.
