# Model choice, and model files without the weights

Date: 2026-11-10. Status: proposal for decision D-002 (ops/decisions.md).

## 1. Recommendation

Use the **Qwen3.5 small family** in three roles:

| Role | Model | Text network | Download | Runs on |
|---|---|---|---|---|
| Target borrowed core (frozen, with its weights) | **Qwen3.5-4B** | 4.206B parameters | 9.3 GB | GPU with 12 GB or more; 8.4 GB of weights at 16 bits |
| Development borrowed core (frozen, with its weights) | **Qwen3.5-0.8B** | 0.752B parameters | 1.75 GB | CPU with 8 GB memory, or any GPU |
| Native Flow core (blank, pretrained by this project) | Qwen3.5 architecture, scaled down | 34.7M parameters | nothing | CPU for tests; one 24 GB GPU for pretraining |

The 4B model is the largest of the family that is clearly below 9B parameters. The 9B model is at the limit: its text network, built here from its published shape, has 8.95B parameters, and the published checkpoint adds a vision encoder on top.

Reasons for this family:

1. **License.** Apache-2.0 for the weights, and Apache-2.0 for the model code in Transformers. Use, modification and redistribution are allowed, also commercially; the license text and notices stay with any copy, and changed files say that they were changed. (The license file in each model repository is the authority; this is not legal advice.)
2. **The architecture already contains what Flow's spec lists** (Annex A of the spec): gated attention, a causal convolution with kernel 4 in the linear-attention layers, rotary positions on a quarter of each head, and linear attention with a constant-size state. The published layout is 3 linear-attention layers followed by 1 full-attention layer, repeated.
3. **A text-only model class exists.** The checkpoints are multimodal, but `Qwen3_5ForCausalLM` builds the text network alone, so the vision encoder is left out (about 0.10B parameters of the 0.8B checkpoint).
4. **One family, one tokenizer, three sizes.** Function bodies are portable across cores; triggers and binders are refitted per core. Work done on the 0.8B model moves to the 4B model by refitting small heads.
5. **It runs on the machines at hand.** The 0.8B model needs no GPU.

## 2. Can a model's files be obtained and modified without its weights?

Yes. A published model consists of four separable things:

| Part | Size (Qwen3.5-0.8B) | What it is | Needed for a blank network |
|---|---|---|---|
| `config.json` | 2.9 kB | The shape of the network: widths, depths, layer order | Yes; can be edited freely |
| Model code | in the Transformers library | The layers themselves | Yes |
| Tokenizer files | about 23 MB | Text to token ids and back | Only if the same vocabulary is kept |
| `*.safetensors` | 1.75 GB | The trained weights | No |

Only the last part is the trained model. A network built from the first two has random weights.

### What was checked here

| Check | Result |
|---|---|
| Build the published shapes without weights (no memory allocated) | 0.752B, 1.882B and 4.206B parameters for the 0.8B, 2B and 4B text networks |
| Build a modified shape with random weights | Width 512, 16 layer passes from 8 unique layers, vocabulary 16,384: 34.7M parameters. Forward pass runs; changing a token changes nothing before it; save and reload give the same network |
| Train a blank network from scratch | A 0.52M-parameter network of this architecture is pretrained on the CPU by `scripts/toy_walkthrough.py` |
| Read a multimodal checkpoint and keep the text network only | Tested on a stand-in checkpoint built locally with the published file layout (`tests/test_borrowed.py`) |
| Download from huggingface.co | **Not tested.** `scripts/fetch_model_files.py` does it and leaves the weight files out by default |

Because of the last row, three things are still unverified and are the first to run:

```
python scripts/fetch_model_files.py --model Qwen/Qwen3.5-0.8B          # about 23 MB, no weights
python scripts/check_tokenizer.py   --model models/Qwen3.5-0.8B        # does it split numbers into digits?
python scripts/make_blank_model.py  --config models/Qwen3.5-0.8B/config.json --describe
```

The configurations in `configs/qwen3_5/` were typed in from the published values (0.8B in full; 2B and 4B from their published sizes). The fetch script replaces them with the originals.

### What a blank network is good for

A blank network knows nothing. Training cost grows with parameters × tokens (about 6 × parameters × tokens operations):

| Blank network | Tokens to train on | Order of cost |
|---|---|---|
| Flow native core, 34.7M | 2 × 10^9 | 4–8 hours on one 24 GB GPU (pipeline section 6) |
| 4B, weak | 10^11 | thousands of GPU-hours: several thousand USD |
| 4B, comparable to the published weights | 10^12 to 10^13 | tens to hundreds of thousands of USD |

So: build **blank** only the small native core, and use the 0.8B and 4B models **with their weights, frozen**. The function layer never changes them.

## 3. What limits each option

| Limit | Affects | Note |
|---|---|---|
| GPU memory | Qwen3.5-4B | 8.4 GB of weights at 16 bits plus activations. A 12 GB card for inference and for fitting heads; a free Kaggle T4 (16 GB) also holds it |
| CPU speed | Qwen3.5-0.8B on a laptop | Usable for fitting heads at small batch sizes. Cache the hidden states once instead of recomputing them |
| Optional kernels | Linear-attention layers | Without `flash-linear-attention` and `causal-conv1d`, Transformers uses slower pure-PyTorch code. They are built for Linux with CUDA; on Windows expect the slow path |
| Windows on Arm | PyTorch | Native ARM64 builds are CPU-only and come from the PyTorch package index, not from PyPI: `pip install torch --index-url https://download.pytorch.org/whl/cpu` (Python 3.11–3.13) |
| Security software | Downloads | A firewall or security suite may block `python.exe` from reaching huggingface.co; allow it for outgoing HTTPS. The fetch script says so when the connection fails |
| Download limits | huggingface.co | Anonymous downloads are rate-limited. A free read token in the environment variable `HF_TOKEN` lifts this |
| Tokenizer | Emitter | If the tokenizer merges digits into multi-digit tokens, the emitter must print digit tokens only. `check_tokenizer.py` reports this |

## 4. Alternatives below 9B parameters

| Model | Size | License | Note |
|---|---|---|---|
| Gemma 4 E2B, E4B | 2.3B, 4.5B active | Apache 2.0 | Multimodal with audio |
| Phi-4-mini | 3.8B | MIT | Text |
| SmolLM3-3B | 3B | Apache 2.0 | 64K context |
| Ministral 3B, 8B | 3B, 8B | Apache 2.0 | Vision |
| Llama 3.2 1B, 3B | 1B, 3B | Llama Community License | Text; license with use conditions |
| InternLM3-8B | 8B | Apache 2.0 | |
| LFM2.5-1.2B | 1.2B | LFM Open v1.0 | Own license |

Any of them can serve as a borrowed core: the function layer needs only hidden states and a tokenizer that reports character offsets. `src/flow/borrowed/hf_core.py` is written for Qwen3.5 (weight names, configuration class); another family needs a loader of about the same length. A blank native core of another architecture needs its own `core/blank.py`.

## 5. Sources

- Qwen3.5-4B model card (license, sizes, layer layout, context length): https://huggingface.co/Qwen/Qwen3.5-4B
- Qwen3.5-9B configuration: https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/config.json
- Qwen3.5 in Transformers (classes, date added, optional kernels): https://huggingface.co/docs/transformers/model_doc/qwen3_5
- Release of the small series, 2026-03-02: https://gigazine.net/gsc_news/en/20260303-qwen-3-5-small
- Downloading part of a repository (`allow_patterns`, `ignore_patterns`, dry run): https://huggingface.co/docs/huggingface_hub/en/guides/download
- PyTorch on Windows on Arm: https://learn.arm.com/install-guides/pytorch-woa and the wheel index https://download.pytorch.org/whl/cpu/torch/
- Open-weight models below 9B: https://github.com/xigh/open-weight-models (list updated June 2026)
