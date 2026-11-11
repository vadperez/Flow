# Budget and spend

Cap for v0: **to be set** (decision D-003). Plan: 118–200 GPU-hours, about 40–120 USD at 0.35–0.60 USD per hour for a 24 GB card (pipeline draft, section 6; prices seen on 2026-11-10 and to be re-checked on the day of renting).

Rules (pipeline Annex D, rule 5): pause at 80% of a stage budget; stop and open a gate above 100%.

## What needs no GPU

Everything in this starter: tests, written bodies, registry, blank small cores, the toy walkthrough (about 10 minutes on 2 CPU cores), the function layer on Qwen3.5-0.8B at small batch sizes.

## What limits speed

| Activity | Limit | Remedy |
|---|---|---|
| Fitting triggers and binders on a frozen core | One forward pass of the core per batch; no backward pass through it | Cache the hidden states of the training set once (S3.2) |
| Qwen3.5-style layers without the optional kernels | Transformers falls back to slower pure-PyTorch code | Install `flash-linear-attention` and `causal-conv1d` on the Linux GPU machine |
| Downloads from huggingface.co | Anonymous rate limits; 9.3 GB for Qwen3.5-4B | Set `HF_TOKEN` (free read token); pin the revision |
| Native core pretraining | About 7 × 10^17 FLOPs per seed: 4–8 hours on a 24 GB card | Rented GPU; checkpoints every 2 × 10^8 tokens |
| Your time at the four gates | Work stops until a decision is recorded | Gate packets prepared in advance (ops/gate_packet_template.md) |

## Ledger

| Date | Stage | Item | GPU-hours | USD | Running total USD | Note |
|---|---|---|---|---|---|---|
| | | | | | 0 | |
