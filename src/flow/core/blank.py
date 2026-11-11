"""Blank cores: the Qwen3.5 text architecture built from a configuration, without trained weights.

What a published model consists of
    config.json           the shape of the network (a few kB)
    tokenizer files       text <-> token ids (about 23 MB for Qwen3.5)
    model code            in the Transformers library (Apache-2.0)
    *.safetensors         the trained weights (1.75 GB for Qwen3.5-0.8B, 9.3 GB for Qwen3.5-4B)

Only the last item is the trained model. This module builds the network from the first three:
either with the published shape, or with a modified shape, for example the small Flow core.
A network built this way has random weights and knows nothing until it is trained.

Loop: Flow's core reuses a block of layers several times (spec Annex A). Here the reused passes
are separate layer objects that share the same parameters, so every pass keeps its own cache slot.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

REQUIREMENTS = (
    "This part of Flow needs PyTorch >= 2.7 and Transformers >= 5.0 (the first major version that "
    "contains the Qwen3.5 model code).\n"
    "Install them with:  pip install -r ops/requirements-model.txt\n"
    "Then check the machine with:  python ops/compat_check.py"
)

KINDS = {"L": "linear_attention", "F": "full_attention"}


def _libs():
    """Import the heavy libraries, or stop with a message that says what is needed."""
    try:
        import torch
        from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    except ImportError as exc:  # missing library, or a Transformers version without Qwen3.5
        raise RuntimeError(f"{REQUIREMENTS}\n(import failed: {exc})") from exc
    return torch, Qwen3_5TextConfig, Qwen3_5ForCausalLM


@dataclass
class CoreSpec:
    """Shape of a Flow core. Layer kinds: L = linear attention (Gated DeltaNet), F = full attention."""

    vocab_size: int
    hidden_size: int = 512
    intermediate_size: int = 1344  # three MLP matrices (gate, up, down): same size as a 2,048 two-matrix MLP
    heads: int = 8  # full attention
    kv_heads: int = 2
    head_dim: int = 64
    linear_heads: int = 8  # linear attention (key heads = value heads)
    linear_head_dim: int = 64
    conv_kernel: int = 4
    entry: str = "LL"  # run once
    loop: str = "LFLL"  # run `loop_passes` times with shared parameters
    loop_passes: int = 3
    exit: str = "LF"  # run once
    max_positions: int = 4096
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    extra: dict = field(default_factory=dict)  # further Qwen3_5TextConfig fields, if needed

    def __post_init__(self) -> None:
        for name in ("entry", "loop", "exit"):
            if set(getattr(self, name)) - set(KINDS):
                raise ValueError(f"{name} may contain only L and F, got {getattr(self, name)!r}")
        if self.loop_passes < 1 or (self.loop_passes > 1 and not self.loop):
            raise ValueError("loop_passes must be >= 1, and a loop needs at least one layer")
        if self.head_dim % 8:
            raise ValueError("head_dim must be a multiple of 8 (a quarter of it is rotated, in pairs)")

    @property
    def pattern(self) -> str:
        """Layer kinds in the order they run, for example 16 letters for 2 + 4 x 3 + 2."""
        return self.entry + self.loop * self.loop_passes + self.exit

    @property
    def unique_layers(self) -> int:
        return len(self.entry) + len(self.loop) + len(self.exit)

    def owner(self) -> "list[int]":
        """For every pass, the index of the pass that owns its parameters (itself if not shared)."""
        e, n = len(self.entry), len(self.loop)
        owners = list(range(e))
        for p in range(self.loop_passes):
            owners += [e + k for k in range(n)]
        first_exit = e + n * self.loop_passes
        return owners + [first_exit + k for k in range(len(self.exit))]


def text_config(spec: CoreSpec):
    """CoreSpec -> Qwen3_5TextConfig (the same class the published models use)."""
    _, Qwen3_5TextConfig, _ = _libs()
    pairs = spec.head_dim // 8  # rotated pairs: a quarter of the head, two numbers per pair
    fields = dict(
        vocab_size=spec.vocab_size, hidden_size=spec.hidden_size, intermediate_size=spec.intermediate_size,
        num_hidden_layers=len(spec.pattern), layer_types=[KINDS[k] for k in spec.pattern],
        num_attention_heads=spec.heads, num_key_value_heads=spec.kv_heads, head_dim=spec.head_dim,
        linear_num_key_heads=spec.linear_heads, linear_num_value_heads=spec.linear_heads,
        linear_key_head_dim=spec.linear_head_dim, linear_value_head_dim=spec.linear_head_dim,
        linear_conv_kernel_dim=spec.conv_kernel, max_position_embeddings=spec.max_positions,
        tie_word_embeddings=True, pad_token_id=spec.pad_token_id, bos_token_id=spec.bos_token_id,
        eos_token_id=spec.eos_token_id,
        rope_parameters={"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25,
                         "mrope_interleaved": True, "mrope_section": [(pairs + 2 - k) // 3 for k in range(3)]},
    )
    fields.update(spec.extra)
    return Qwen3_5TextConfig(**fields)


def published_text_config(config_json: "str | Path | dict", **changes):
    """Text part of a published Qwen3.5 config.json -> Qwen3_5TextConfig, with optional changes.

    Example: published_text_config("configs/qwen3_5/Qwen3.5-0.8B.config.json", vocab_size=16384)
    """
    _, Qwen3_5TextConfig, _ = _libs()
    raw = config_json if isinstance(config_json, dict) else json.loads(Path(config_json).read_text(encoding="utf-8"))
    fields = dict(raw.get("text_config", raw))
    for key in ("model_type", "dtype", "torch_dtype", "transformers_version", "architectures"):
        fields.pop(key, None)
    fields.update(changes)
    if "num_hidden_layers" in changes and "layer_types" not in changes:
        fields.pop("layer_types", None)  # let the config rebuild the 3 linear : 1 full pattern
    return Qwen3_5TextConfig(**fields)


def _share_parameters(target, source) -> None:
    """Make `target` use the parameter objects of `source` (same shapes, same module tree)."""
    for name, parameter in source.named_parameters():
        path, _, leaf = name.rpartition(".")
        setattr(target.get_submodule(path) if path else target, leaf, parameter)


def build(spec: CoreSpec, seed: int = 0):
    """A blank core: random weights, loop passes sharing parameters. Returns Qwen3_5ForCausalLM."""
    torch, _, Qwen3_5ForCausalLM = _libs()
    torch.manual_seed(seed)
    model = Qwen3_5ForCausalLM(text_config(spec))
    layers = model.model.layers
    for index, owner in enumerate(spec.owner()):
        if owner != index:
            _share_parameters(layers[index], layers[owner])
    model.flow_spec = spec
    return model


def build_from_config(config, seed: int = 0):
    """A blank network from any Qwen3_5TextConfig (published shape or modified). No sharing."""
    torch, _, Qwen3_5ForCausalLM = _libs()
    torch.manual_seed(seed)
    return Qwen3_5ForCausalLM(config)


def count_parameters(model) -> dict:
    """Unique parameters (shared ones counted once), split into embeddings and the rest."""
    seen, total = set(), 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
    embeddings = model.get_input_embeddings().weight.numel()
    return {"total": total, "embeddings": embeddings, "layers_and_norms": total - embeddings}


def describe(config) -> dict:
    """Parameter count of a configuration without allocating memory for the weights."""
    torch, _, Qwen3_5ForCausalLM = _libs()
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(config)
    counts = count_parameters(model)
    counts["bytes_16bit"] = counts["total"] * 2
    return counts


def hidden_states(model, input_ids, attention_mask=None):
    """All hidden states: the embedding output, then the output of every pass. Tuple of [B, T, H]."""
    out = model.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
    return out.hidden_states


def _unique_state(model, spec: "CoreSpec | None") -> dict:
    state = model.state_dict()
    state.pop("lm_head.weight", None)  # tied to the embeddings
    if spec is not None:
        for index, owner in enumerate(spec.owner()):
            if owner != index:
                for key in [k for k in state if k.startswith(f"model.layers.{index}.")]:
                    del state[key]
    return {k: v.contiguous() for k, v in state.items()}


def save(model, folder: "str | Path") -> Path:
    """Write config.json, flow_core.json (if the model has a CoreSpec) and model.safetensors."""
    from safetensors.torch import save_file

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    spec = getattr(model, "flow_spec", None)
    model.config.to_json_file(str(folder / "config.json"))
    if spec is not None:
        (folder / "flow_core.json").write_text(json.dumps(asdict(spec), indent=2), encoding="utf-8")
    save_file(_unique_state(model, spec), str(folder / "model.safetensors"), metadata={"format": "pt"})
    return folder


def load(folder: "str | Path"):
    """Read a core written by save()."""
    from safetensors.torch import load_file

    torch, Qwen3_5TextConfig, Qwen3_5ForCausalLM = _libs()
    folder = Path(folder)
    spec_file = folder / "flow_core.json"
    if spec_file.exists():
        spec = CoreSpec(**json.loads(spec_file.read_text(encoding="utf-8")))
        model = build(spec)
    else:
        spec = None
        model = Qwen3_5ForCausalLM(Qwen3_5TextConfig.from_json_file(str(folder / "config.json")))
    state = load_file(str(folder / "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    shared = set()
    if spec is not None:
        shared = {i for i, owner in enumerate(spec.owner()) if owner != i}
    really_missing = [k for k in missing if k != "lm_head.weight"
                      and not any(k.startswith(f"model.layers.{i}.") for i in shared)]
    if really_missing or unexpected:
        raise RuntimeError(f"weights do not fit this core: missing {really_missing[:5]}, unexpected {list(unexpected)[:5]}")
    model.tie_weights()
    return model
