"""Type system tests (pipeline step S1.3): codecs, scanner, emitter, and the print -> scan closure."""
import random
from decimal import Decimal

import numpy as np
import pytest

from flow.typesys import (Emitter, Kind, Number, NumberRangeError, PieceMap, Registers, SymbolTable, find_numbers,
                          flag, number, pack, pieces, scan, symbol, unpack)
from flow.typesys import numcode as num
from flow.typesys import symcode as sym
from flow.data.tokenizer import ToyTokenizer

BOUNDARY = ["0", "9", "10", "-1", "999999999999", "-999999999999", "0.000001", "-0.5", "123456.654321", "1000000",
            "1.5", "2.50", "007", "-0"]


def random_number(rng: random.Random) -> Number:
    scale = rng.choice([0, 0, 0, 1, 2, 3, 6])
    mantissa = rng.randrange(10 ** rng.randint(1, num.N_DIGITS))
    return Number.from_parts(rng.random() < 0.3 and mantissa != 0, mantissa, scale)


def test_number_roundtrip_boundaries():
    for text in BOUNDARY:
        n = Number.from_value(text)
        assert num.decode(num.encode(n)) == n
        assert n.to_decimal() == Decimal(text.replace(",", ""))
        assert Number.from_value(n.text()) == n  # canonical text reads back to the same number


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_number_roundtrip_random(dtype):
    rng = random.Random(0)
    for _ in range(20000):
        n = random_number(rng)
        code = num.encode(n).astype(dtype)
        assert num.decode(code) == n


@pytest.mark.slow
def test_number_roundtrip_million():
    rng = random.Random(1)
    for _ in range(1_000_000):
        n = random_number(rng)
        assert num.decode(num.encode(n).astype(np.float16)) == n


def test_number_decode_tolerates_noise():
    rng = np.random.default_rng(0)
    n = Number.from_value("-12345.678")
    for _ in range(200):
        noisy = num.encode(n) + rng.normal(0, 0.08, num.CODE_DIM).astype(np.float32)
        assert num.decode(noisy) == n


def test_number_canonical_text_and_limits():
    assert Number.from_value("2.50").text() == "2.5"
    assert Number.from_value("1,250").text() == "1250"
    assert Number.from_value("-0").text() == "0"
    assert Number.from_value("0.5").text() == "0.5"
    assert Number.from_value(1500).rescaled(2).mantissa == 150000
    with pytest.raises(NumberRangeError):
        Number.from_value("1234567890123")  # 13 digits
    with pytest.raises(NumberRangeError):
        Number.from_value("0.1234567")  # 7 decimal places
    with pytest.raises(NumberRangeError):
        Number.from_value("abc")


def test_reed_muller_code_properties():
    g = sym.generator()
    assert g.shape == (16, 32)
    book = sym.codebook()
    assert book.shape == (65536, 32) and len({row.tobytes() for row in book}) == 65536
    # Linear code: minimum distance = minimum weight of a non-zero codeword = 8.
    weights = ((1 - book.astype(np.int32)) // 2).sum(axis=1)
    assert weights[0] == 0 and weights[1:].min() == 8


def test_symbol_roundtrip_all_ids_and_noise():
    rng = np.random.default_rng(0)
    book = sym.codebook().astype(np.float32)
    assert (np.argmax(book[:4096] @ book.T, axis=1) == np.arange(4096)).all()
    for sid in rng.integers(0, 65536, 300):
        noisy = sym.encode_id(int(sid)) + rng.normal(0, 0.35, 32).astype(np.float32)
        assert sym.decode_id(noisy) == sid
        assert sym.decode_id(sym.encode_id(int(sid)).astype(np.float16)) == sid


def test_symbol_table(tmp_path):
    table = SymbolTable(["Vorland", "capital", "Tarsis"])
    assert table.add("capital") == 1 and table.add("mg") == 3 and len(table) == 4
    assert table.decode(table.encode("Tarsis")) == "Tarsis"
    table.save(tmp_path / "symbols.json")
    assert SymbolTable.load(tmp_path / "symbols.json").symbols == table.symbols


def test_pack_unpack_values():
    table = SymbolTable(["Tarsis"])
    for v in [number("-48,700.25"), symbol("Tarsis"), flag(True), flag(False)]:
        assert unpack(pack(v, table), table) == v
    assert unpack(np.zeros(34, dtype=np.float32)).kind is Kind.NONE


def test_scanner_finds_numbers():
    text = "Mara has 1,250 mg, adds 300.50 and loses -5. H2O v1.2 3-5 500mg."
    found = [(text[s:e], n.text()) for s, e, n in find_numbers(text)]
    assert found == [("1,250", "1250"), ("300.50", "300.5"), ("-5", "-5"), ("3", "3"), ("5", "5"), ("500", "500")]


def test_scanner_anchors_on_last_token_and_symbols():
    tok = ToyTokenizer.from_texts(["what is the capital of Vorland plus 500 ? it is 48700 ."])
    text = "what is the capital of Vorland plus 500 ? it is 48700 ."
    enc = tok.encode(text)
    table = SymbolTable(["Vorland", "capital"])
    anchors = scan(text, enc.offsets, enc.pieces, table)
    got = [(enc.pieces[a.token], a.value.kind.value, a.value.text()) for a in anchors]
    assert got == [("capital", "SYMBOL", "capital"), ("Vorland", "SYMBOL", "Vorland"), ("0", "NUMBER", "500"),
                   ("0", "NUMBER", "48700")]
    assert anchors[2].token == 9 and anchors[3].token == 17  # the last digit of each number


def test_emitter_pieces_and_closure():
    """What the emitter prints, the scanner reads back as the same value."""
    rng = random.Random(2)
    emitter = Emitter()
    for _ in range(3000):
        n = random_number(rng)
        emitter.start(number(n))
        out = []
        while emitter.active:
            out.append(emitter.next_piece())
        text = "".join(out)
        assert text == n.text()
        (s, e, back), = find_numbers("it is " + text + " .")
        assert back == n.normalized()
    assert pieces(symbol("Tarsis")) == ["Tarsis"] and pieces(flag(False)) == ["no"]


def test_piece_map_reports_missing_pieces():
    tok = ToyTokenizer.from_texts(["yes no - . 0 1 2 3 4 5 6 7 8 9 Tarsis"])
    pm = PieceMap(lambda s: tok.encode(s).ids, extra_pieces=["Tarsis"])
    assert pm.missing == [] and pm.token_id("7") == tok.encode("7").ids[0]
    assert pm.add("no such piece") is None and "no such piece" in pm.missing


def test_registers_state_slots():
    r = Registers()
    for i in range(20):
        r.bind(f"k{i}", number(i))
    assert len(r.state) == 16 and r.query("k0").kind is Kind.NONE and r.miss
    assert r.query("k19") == number(19) and not r.miss
    r.bind("k5", number(500))
    assert r.query("k5") == number(500) and len(r.state) == 16
