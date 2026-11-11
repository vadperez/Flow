#!/usr/bin/env python3
"""Build a network WITHOUT trained weights, from a published or modified configuration.

    # 1. Count a published shape (no memory needed, nothing downloaded):
    python scripts/make_blank_model.py --config configs/qwen3_5/Qwen3.5-4B.config.json --describe

    # 2. The Flow native core (small Qwen3.5-style network with a shared loop), random weights:
    python scripts/make_blank_model.py --spec configs/core_v0.yaml --out runs/blank-flow-core

    # 3. A published shape, modified, random weights:
    python scripts/make_blank_model.py --config configs/qwen3_5/Qwen3.5-0.8B.config.json \
        --set hidden_size=512 num_hidden_layers=8 intermediate_size=1344 head_dim=64 vocab_size=16384 \
              linear_key_head_dim=64 linear_value_head_dim=64 linear_num_key_heads=8 linear_num_value_heads=8 \
              eos_token_id=2 max_position_embeddings=4096 --out runs/blank-small

A blank network knows nothing: it has to be trained before it is of any use. Small ones (tens of
millions of parameters) can be trained within this project's budget; a blank 4B network cannot.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_changes(pairs: "list[str]") -> dict:
    changes = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        try:
            changes[key] = json.loads(raw)
        except json.JSONDecodeError:
            changes[key] = raw
    return changes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="a published config.json (full or text part)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="changes to the text configuration")
    parser.add_argument("--spec", help="a Flow core description (YAML with the fields of CoreSpec), e.g. configs/core_v0.yaml")
    parser.add_argument("--flow-core", action="store_true", help="the Flow native core with built-in defaults (see --vocab)")
    parser.add_argument("--vocab", type=int, default=16384)
    parser.add_argument("--describe", action="store_true", help="only count parameters; allocate and write nothing")
    parser.add_argument("--out", help="folder for config.json and model.safetensors (random weights)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        from flow.core import blank
        blank._libs()
    except RuntimeError as exc:
        print(exc)
        return 2

    if args.spec:
        import yaml

        fields = yaml.safe_load(Path(args.spec).read_text(encoding="utf-8"))
        fields.update(parse_changes(args.set))
        try:
            spec = blank.CoreSpec(**fields)
        except (TypeError, ValueError) as exc:
            print(f"{args.spec} is not a valid core description: {exc}")
            return 1
        config = blank.text_config(spec)
        label = f"Flow core from {Path(args.spec).name}, layer passes {spec.pattern} ({spec.unique_layers} unique layers)"
    elif args.flow_core:
        spec = blank.CoreSpec(vocab_size=args.vocab, **parse_changes(args.set))
        config = blank.text_config(spec)
        label = f"Flow core, layer passes {spec.pattern} ({spec.unique_layers} unique layers)"
    elif args.config:
        spec = None
        config = blank.published_text_config(args.config, **parse_changes(args.set))
        label = f"{Path(args.config).name}" + (" with changes" if args.set else "")
    else:
        parser.error("give --spec, --config or --flow-core")

    if args.describe or not args.out:
        if spec is not None:
            counts = blank.count_parameters(blank.build(spec, args.seed))
        else:
            counts = blank.describe(config)
        print(label)
        print(f"  parameters      {counts['total']:>15,}   (embeddings {counts['embeddings']:,})")
        print(f"  size, 16-bit    {counts['total'] * 2 / 1e9:>15.3f} GB")
        print(f"  size, 32-bit    {counts['total'] * 4 / 1e9:>15.3f} GB")
        print(f"  layers          {config.num_hidden_layers}: " + "".join("L" if t == "linear_attention" else "F" for t in config.layer_types)
              + "   (L = linear attention, F = full attention)")
        print(f"  width {config.hidden_size}, MLP {config.intermediate_size}, vocabulary {config.vocab_size:,}")
        if not args.out:
            return 0

    model = blank.build(spec, args.seed) if spec is not None else blank.build_from_config(config, args.seed)
    folder = blank.save(model, args.out)
    size = (folder / "model.safetensors").stat().st_size
    print(f"blank network written to {folder} ({size / 1e6:.1f} MB, random weights, seed {args.seed})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
