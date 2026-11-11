"""Blank cores (the Qwen3.5 text architecture without trained weights). Needs torch and transformers."""
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from flow.core import blank

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "qwen3_5"
TINY = dict(vocab_size=300, hidden_size=64, intermediate_size=128, heads=4, kv_heads=2, head_dim=16,
            linear_heads=4, linear_head_dim=16, max_positions=128)


@pytest.fixture(scope="module")
def tiny():
    return blank.build(blank.CoreSpec(entry="L", loop="LF", loop_passes=3, exit="F", **TINY)).eval()


def test_pattern_and_owners():
    spec = blank.CoreSpec(vocab_size=16384)
    assert spec.pattern == "LLLF" * 4 and spec.unique_layers == 8
    assert spec.owner() == [0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 14, 15]
    with pytest.raises(ValueError):
        blank.CoreSpec(vocab_size=10, loop="LX")


def test_default_core_size():
    spec = blank.CoreSpec(vocab_size=16384)
    model = blank.build(spec)
    counts = blank.count_parameters(model)
    assert counts["embeddings"] == 16384 * 512
    assert 30e6 < counts["total"] < 40e6  # the "small resident core" of the spec: 10-50M parameters
    untied = blank.describe(blank.text_config(spec))["total"]
    assert untied > counts["total"] * 1.5  # the loop saves parameters


def test_loop_passes_share_parameters(tiny):
    layers, owners = tiny.model.layers, tiny.flow_spec.owner()
    assert owners == [0, 1, 2, 1, 2, 1, 2, 7]
    assert layers[3].mlp.gate_proj.weight is layers[1].mlp.gate_proj.weight
    assert layers[6].self_attn.q_proj.weight is layers[2].self_attn.q_proj.weight
    assert layers[0].mlp.gate_proj.weight is not layers[1].mlp.gate_proj.weight


def test_forward_is_causal_and_trains(tiny):
    ids = torch.randint(3, 300, (2, 20))
    with torch.no_grad():
        a = tiny(input_ids=ids, use_cache=False).logits
        changed = ids.clone()
        changed[:, 12] = (changed[:, 12] + 5) % 300
        b = tiny(input_ids=changed, use_cache=False).logits
    assert a.shape == (2, 20, 300)
    assert torch.equal(a[:, :12], b[:, :12]) and not torch.equal(a[:, 12:], b[:, 12:])
    hs = blank.hidden_states(tiny, ids)
    assert len(hs) == len(tiny.flow_spec.pattern) + 1 and hs[-1].shape == (2, 20, 64)
    loss = tiny(input_ids=ids, labels=ids, use_cache=False).loss
    loss.backward()
    assert tiny.model.layers[1].mlp.gate_proj.weight.grad.abs().sum() > 0
    tiny.zero_grad()


def test_cache_works_with_shared_layers(tiny):
    ids = torch.randint(3, 300, (1, 8))
    with torch.no_grad():
        with_cache = tiny.generate(ids, max_new_tokens=5, do_sample=False, use_cache=True)
        without = tiny.generate(ids, max_new_tokens=5, do_sample=False, use_cache=False)
    assert torch.equal(with_cache, without)


def test_save_and_load(tiny, tmp_path):
    blank.save(tiny, tmp_path / "core")
    again = blank.load(tmp_path / "core").eval()
    ids = torch.randint(3, 300, (1, 16))
    with torch.no_grad():
        assert torch.equal(tiny(input_ids=ids, use_cache=False).logits, again(input_ids=ids, use_cache=False).logits)
    assert blank.count_parameters(again)["total"] == blank.count_parameters(tiny)["total"]
    stored = (tmp_path / "core" / "model.safetensors").stat().st_size
    assert stored < blank.count_parameters(tiny)["total"] * 4 * 1.1  # shared layers are stored once


@pytest.mark.parametrize("name,billions", [("Qwen3.5-0.8B", 0.752), ("Qwen3.5-2B", 1.882), ("Qwen3.5-4B", 4.206)])
def test_published_shapes_without_weights(name, billions):
    """The published configuration alone is enough to build the network (here: only to count it)."""
    config = blank.published_text_config(CONFIGS / f"{name}.config.json")
    assert abs(blank.describe(config)["total"] / 1e9 - billions) < 0.001


def test_published_shape_modified():
    config = blank.published_text_config(CONFIGS / "Qwen3.5-0.8B.config.json", vocab_size=4096, num_hidden_layers=4,
                                         eos_token_id=2, max_position_embeddings=512)
    assert config.layer_types == ["linear_attention"] * 3 + ["full_attention"]
    model = blank.build_from_config(config).eval()
    with torch.no_grad():
        assert model(input_ids=torch.randint(0, 4096, (1, 8)), use_cache=False).logits.shape == (1, 8, 4096)
