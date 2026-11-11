#!/usr/bin/env python3
"""Check how a model's tokenizer treats numbers (pipeline steps S2.3 and S3.1).

    python scripts/check_tokenizer.py --model models/Qwen3.5-0.8B

Needs only the tokenizer files (scripts/fetch_model_files.py without --with-weights).
Prints the findings and writes them to reports/tokenizer-<name>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="folder with the tokenizer files")
    args = parser.parse_args()
    folder = Path(args.model)
    if not folder.exists():
        print(f"{folder} does not exist. Fetch the files first:  python scripts/fetch_model_files.py --model Qwen/{folder.name}")
        return 1
    try:
        from transformers import AutoTokenizer
        from flow.borrowed.hf_core import HFTokenizer, number_report
    except ImportError as exc:
        from flow.core.blank import REQUIREMENTS
        print(f"{REQUIREMENTS}\n(import failed: {exc})")
        return 2
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(folder))
    except Exception as exc:  # noqa: BLE001
        print(f"Could not load a tokenizer from {folder}: {type(exc).__name__}: {exc}")
        return 1
    report = number_report(tokenizer)
    print(f"tokenizer of {folder.name}: {report['vocabulary_size']:,} tokens, fast: {report['fast_tokenizer']}")
    print(f"start token {report['bos_token']!r}, end token {report['eos_token']!r}, padding {report['pad_token']!r}")
    print(f"each digit is one token: {report['each_digit_is_one_token']}")
    print(f"numbers are split into single digits: {report['numbers_split_into_single_digits']}")
    for sample, pieces in report["tokens_of_samples"].items():
        print(f"  {sample!r:16} -> {pieces}")
    print("answer words (number of tokens):", report["tokens_per_answer_word"])
    if report["fast_tokenizer"]:  # the scanner on this tokenizer: values must land on the last token of each number
        from flow.typesys import scan
        adapter = HFTokenizer(tokenizer)
        text = "Mara has 1250 coins and gets 35 more. The change is -3.5 today."
        enc = adapter.encode(text)
        found = [(a.value.text(), enc.pieces[a.token]) for a in scan(text, enc.offsets, enc.pieces)]
        report["scanner_check"] = found
        print("scanner on this tokenizer (value, token it is anchored at):", found)
        back, _ = adapter.render(enc.ids)
        report["render_reproduces_text"] = back == text
        print("decoding the tokens reproduces the text:", back == text)
    print("verdict:", report["verdict"])
    for p in report["problems"]:
        print("  -", p)
    out = ROOT / "reports" / f"tokenizer-{folder.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print("written:", out.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
