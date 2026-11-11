"""The borrowed-core harness and the tokenizer check, on stand-ins built locally.

The real Qwen3.5 files are not part of this repository. These tests build a tiny multimodal
checkpoint with random weights and a small tokenizer, in the same file layout as a published model,
and run the same code on them.
"""
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from flow.borrowed.hf_core import HFTokenizer, load_text_core, number_report
from flow.typesys import scan

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "qwen3_5"


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """A tiny Qwen3.5 checkpoint (text + vision), random weights, saved the way published models are."""
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    raw = json.loads((CONFIGS / "Qwen3.5-0.8B.config.json").read_text())
    text = dict(raw["text_config"], vocab_size=400, hidden_size=64, intermediate_size=128, num_hidden_layers=4, head_dim=16,
                num_attention_heads=4, num_key_value_heads=2, linear_key_head_dim=16, linear_value_head_dim=16,
                linear_num_key_heads=4, linear_num_value_heads=4, eos_token_id=2, max_position_embeddings=256,
                layer_types=["linear_attention"] * 3 + ["full_attention"])
    text["rope_parameters"] = dict(text["rope_parameters"], mrope_section=[1, 1, 0])
    vision = dict(raw["vision_config"], depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                  num_position_embeddings=64)
    keep = {k: v for k, v in raw.items() if k in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id")}
    keep = {k: v % 400 for k, v in keep.items()}
    config = Qwen3_5Config(text_config=text, vision_config=vision, tie_word_embeddings=True, **keep)
    torch.manual_seed(0)
    full = Qwen3_5ForConditionalGeneration(config).float().eval()
    folder = tmp_path_factory.mktemp("Qwen3.5-tiny")
    full.save_pretrained(folder)
    return folder, full


def test_text_core_from_multimodal_checkpoint(checkpoint):
    folder, full = checkpoint
    assert any(folder.glob("*.safetensors"))
    core = load_text_core(folder, dtype="float32")
    assert core.flow_source["tensors_skipped"] > 0  # the vision part was left out
    assert not any(p.requires_grad for p in core.parameters())
    ids = torch.randint(3, 400, (2, 12))
    with torch.no_grad():
        mine = core(input_ids=ids, use_cache=False).logits
        hidden = full.model.language_model(input_ids=ids, use_cache=False).last_hidden_state
        theirs = full.lm_head(hidden)
    assert torch.allclose(mine, theirs, atol=1e-5)
    out = core.model(input_ids=ids, output_hidden_states=True, use_cache=False)
    assert len(out.hidden_states) == 5 and hasattr(core, "lm_head")


def test_missing_weights_message(tmp_path):
    (tmp_path / "config.json").write_text((CONFIGS / "Qwen3.5-0.8B.config.json").read_text())
    with pytest.raises(FileNotFoundError, match="--with-weights"):
        load_text_core(tmp_path)


@pytest.fixture(scope="module")
def hf_tokenizer(tmp_path_factory):
    """A small byte-level BPE tokenizer that keeps every digit as its own token, saved and reloaded."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    tk = Tokenizer(models.BPE(unk_token=None))
    tk.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Digits(individual_digits=True),
                                                pre_tokenizers.ByteLevel(add_prefix_space=False)])
    tk.decoder = decoders.ByteLevel()
    corpus = ["Mara has 1250 coins and gets 35 more. The change is -3.5 today. yes no yes no"] * 50
    tk.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=400, special_tokens=["<|endoftext|>"],
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    folder = tmp_path_factory.mktemp("tok")
    PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|endoftext|>").save_pretrained(folder)
    return AutoTokenizer.from_pretrained(folder)


def test_tokenizer_adapter_and_scanner(hf_tokenizer):
    adapter = HFTokenizer(hf_tokenizer)
    text = "Mara has 1250 coins and gets 35 more. The change is -3.5 today."
    enc = adapter.encode(text)
    assert len(enc.ids) == len(enc.offsets) and adapter.bos_id == adapter.eos_id  # no <bos>: the end token starts a sequence
    anchors = scan(text, enc.offsets, enc.pieces)
    assert [a.value.text() for a in anchors] == ["1250", "35", "-3.5"]
    assert [enc.pieces[a.token].strip() for a in anchors] == ["0", "5", "5"]  # anchored at the last digit
    back, offsets = adapter.render(enc.ids)
    assert back == text and offsets == enc.offsets
    assert adapter.token_id("7") == hf_tokenizer("7", add_special_tokens=False)["input_ids"][0]


def test_number_report(hf_tokenizer):
    report = number_report(hf_tokenizer)
    assert report["each_digit_is_one_token"] and report["numbers_split_into_single_digits"] and report["fast_tokenizer"]
    assert report["tokens_of_samples"]["447"] == ["4", "4", "7"]
