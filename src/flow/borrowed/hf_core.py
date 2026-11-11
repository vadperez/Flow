"""Borrowed core: a published Qwen3.5 model used frozen, language model only (pipeline step S3.1).

The published checkpoints are multimodal (text, image, video). Flow needs the text network and
the output layer only, so the weights of the vision part are skipped while reading.

    core = load_text_core("models/Qwen3.5-0.8B")       # needs the weights (fetch with --with-weights)
    tok  = HFTokenizer.load("models/Qwen3.5-0.8B")

The returned core has the same interface as a blank core from flow.core.blank, so the function
layer, the fitting code and the call loop work on both without change.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..core.blank import REQUIREMENTS, published_text_config

TEXT_PREFIX = "model.language_model."


def load_text_core(folder: "str | Path", dtype: str = "bfloat16", device: str = "cpu"):
    """Qwen3_5ForCausalLM with the published text weights, frozen. `folder` holds config.json and *.safetensors."""
    try:
        import torch
        from safetensors.torch import load_file
        from transformers import Qwen3_5ForCausalLM
    except ImportError as exc:
        raise RuntimeError(f"{REQUIREMENTS}\n(import failed: {exc})") from exc
    folder = Path(folder)
    weight_files = sorted(folder.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(
            f"{folder} has no weight files. A borrowed core needs the trained weights:\n"
            f"  python scripts/fetch_model_files.py --model Qwen/{folder.name} --with-weights")
    config = published_text_config(folder / "config.json")
    model = Qwen3_5ForCausalLM(config).to(getattr(torch, dtype))
    state, skipped = {}, 0
    for file in weight_files:
        for key, tensor in load_file(str(file)).items():
            if key.startswith(TEXT_PREFIX):
                state["model." + key[len(TEXT_PREFIX):]] = tensor
            elif key == "lm_head.weight":
                state[key] = tensor
            else:
                skipped += 1  # vision tower, extra prediction heads
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if k != "lm_head.weight"]
    if missing or unexpected:
        raise RuntimeError(f"the weights in {folder} do not fit the text network: missing {missing[:5]}, unexpected {list(unexpected)[:5]}")
    model.tie_weights()
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.flow_source = {"folder": str(folder), "tensors_used": len(state), "tensors_skipped": skipped}
    return model


@dataclass
class Encoding:
    ids: "list[int]"
    pieces: "list[str]"
    offsets: "list[tuple[int, int]]"


class HFTokenizer:
    """Adapter that gives a Hugging Face tokenizer the small interface Flow uses.

    A start token is always put in front of a sequence by the training and scoring code:
    the tokenizer's <bos> if it has one, otherwise its end-of-text token.
    """

    def __init__(self, tokenizer) -> None:
        self.tk = tokenizer
        if not getattr(tokenizer, "is_fast", False):
            raise RuntimeError("Flow needs a 'fast' tokenizer (it reports character offsets); this one is not.")
        start = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if start is None or pad is None:
            raise RuntimeError("the tokenizer defines neither a start nor an end-of-text token")
        self.bos_id, self.pad_id, self.eos_id = int(start), int(pad), tokenizer.eos_token_id

    @staticmethod
    def load(folder: "str | Path") -> "HFTokenizer":
        from transformers import AutoTokenizer

        return HFTokenizer(AutoTokenizer.from_pretrained(str(folder)))

    def __len__(self) -> int:
        return len(self.tk)

    @property
    def vocab(self):  # id -> piece, as needed by the scoring code
        return _Pieces(self.tk)

    def encode(self, text: str) -> Encoding:
        out = self.tk(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = list(out["input_ids"])
        offsets = [tuple(o) for o in out["offset_mapping"]]
        return Encoding(ids, [text[s:e] for s, e in offsets], offsets)

    def render(self, ids) -> "tuple[str, list[tuple[int, int]]]":
        """Token ids -> (text, character span per token), by decoding growing prefixes."""
        text, offsets, done = "", [], 0
        for i in range(len(ids)):
            text = self.tk.decode(ids[: i + 1], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            offsets.append((done, len(text)))
            done = len(text)
        return text, offsets

    def token_id(self, piece: str) -> int:
        ids = self.tk(piece, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise KeyError(f"{piece!r} is {len(ids)} tokens in this tokenizer, not one")
        return int(ids[0])


class _Pieces:
    def __init__(self, tk) -> None:
        self.tk = tk

    def __getitem__(self, i: int) -> str:
        return self.tk.decode([int(i)], skip_special_tokens=False)


def number_report(tokenizer) -> dict:
    """How a tokenizer treats numbers, which decides how the scanner and the emitter are set up.

    `tokenizer` is a Hugging Face fast tokenizer. Returns a dict with the findings and a verdict.
    """
    enc = lambda s: tokenizer(s, add_special_tokens=False)["input_ids"]
    show = lambda s: [tokenizer.decode([i]) for i in enc(s)]
    digits_single = {d: len(enc(d)) == 1 for d in "0123456789"}
    samples = ["447", " 447", "1250", " 31415926", "12.50", " -3.5", "1,250", " 2026-11-10", "0.000001"]
    split = {s: show(s) for s in samples}
    # A number is "split into digits" when no token holds more than one digit.
    per_digit = all(sum(ch.isdigit() for ch in piece) <= 1 for s in samples for piece in split[s])
    words = {w: len(enc(w)) for w in ["yes", " yes", "no", " no", "Yes", "No"]}
    report = {
        "vocabulary_size": len(tokenizer), "fast_tokenizer": bool(getattr(tokenizer, "is_fast", False)),
        "bos_token": tokenizer.bos_token, "eos_token": tokenizer.eos_token, "pad_token": tokenizer.pad_token,
        "each_digit_is_one_token": all(digits_single.values()), "numbers_split_into_single_digits": per_digit,
        "tokens_of_samples": split, "tokens_per_answer_word": words,
        "first_digit_after_space": {s: show(s)[:2] for s in [" 4", " 44", " -4"]},
    }
    problems = []
    if not report["fast_tokenizer"]:
        problems.append("not a fast tokenizer: no character offsets, the scanner cannot place values on tokens")
    if not report["each_digit_is_one_token"]:
        problems.append("some digit is not a single token: the emitter cannot print digit by digit")
    if not per_digit:
        problems.append("numbers are merged into multi-digit tokens: the scanner still works (a number is anchored at its "
                        "last token), but the emitter must print one digit per token and the core sees mixed spellings")
    if min(words["yes"], words[" yes"]) != 1 or min(words["no"], words[" no"]) != 1:
        problems.append("'yes' or 'no' is not a single token: map FLAG answers to other pieces")
    report["problems"] = problems
    report["verdict"] = "fits the type system as is" if not problems else "usable with the adjustments listed"
    return report
