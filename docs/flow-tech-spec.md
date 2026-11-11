# Flow — Technical Specification (summary)

**Version** 0.2 · **Date** 2026-11-03 · **Status** design specification for a research prototype · **Owner** the project author

**Companions:** `flow-build-pipeline.md` (how to build it) · [Function catalog](https://claude.ai/artifact/Lt5ZCKRD3zBsw1Bey2pHA9) (functions already identified in trained networks)

> **Definition.** Flow is a function-first, modular neural architecture. A small resident core composes functional modules (generic functions and their domain-bound variants) loaded on demand, even mid-generation, from a large, tiered parameter library. Modules are created, differentiated, suppressed and eliminated by a developmental program. Because every module sits behind a typed interface contract, modules can be trained, distilled, programmed or grafted, including by a builder model.

---

## 1. Scope

| In scope (v0–v1) | Out of scope for now |
|---|---|
| Function-first module library (functions × domains) | Neuromorphic or custom silicon |
| Small resident core with recurrent depth | Frontier-scale general competence in v0 |
| Tiered memory (fast memory, RAM, SSD) with residency control | Lifetime learning inside the base product (optional extension, §11) |
| Developmental lifecycle: differentiate, suppress, eliminate | Domain-only experts without functional structure |
| Machine-buildable modules with contracts and unit tests | |
| Local deployment: desktop GPU, unified-memory laptop, Pi-class edge | |
| Loading and unloading modules during generation (hot swap, §4.9) | |

## 2. Design principles

1. **Function first.** The unit of organization is a function (an operation), not a topic. Domain knowledge enters as domain-bound variants of functions, with fallback to the generic function.
2. **Small active, large retained.** Active compute stays small; retained competence grows in the library, mostly outside fast memory.
3. **Residency by design.** Module granularity is chosen so that modules stay loaded long enough to amortize transfers (§4.6), and modules switch in and out mid-generation without stalling it (§4.9).
4. **Developmental organization.** Broad potential first, then differentiation, reversible suppression, and elimination only after shift tests.
5. **Machine-buildable.** Every module has a manifest, a contract and tests, so a builder model can write, train, distill, repair or remove it piece by piece.
6. **Measured, not assumed.** Every claim maps to a metric with a kill criterion (§9).
7. **Form follows the computation.** Each module's architecture matches what its function computes: lookup → table; exact algorithm → programmed weights; state carried across a sequence → recurrent block; composition depth → loops in the core; access to context → attention in the core (Annex A).

## 3. Architecture

```
                      ┌───────────────────────────────────────────────┐
  tokens ──► Core ◄───┤ Module slots (typed, declared per layer)      │
  (resident,   │      └──────────────▲────────────────────────────────┘
  recurrent)   │                     │ load · evict · prefetch
               │      ┌──────────────┴─────────┐     ┌────────────────────────┐
               ├────► │ Controller             │ ──► │ Residency manager      │
               │      │ options + gating       │     │ T0 fast · T1 RAM ·     │
               │      └────────────────────────┘     │ T2 SSD · T3 generate   │
               ▼                                     └───────────▲────────────┘
       output + trace (modules used, scores)                     │
                                                     ┌───────────┴────────────┐
                                                     │ Module library         │
                                                     │ registry + module packs│
                                                     └────────────────────────┘
```

## 4. Components

### 4.1 Core (always resident)

| Property | Specification |
|---|---|
| Role | Holds the shared representation contract and the general primitives (Level 1 of the catalog) |
| Form | Decoder transformer with looped blocks (recurrent depth), gated attention (sigmoid gate after attention, so no attention sinks), Canon-style local token mixing, a final LayerNorm, and slot hooks at declared layers |
| Number input | Fourier number embedding (cos/sin per digit position, FoNE-style), so the number code is designed rather than discovered |
| Resident primitives | Previous-token, token and concept induction, copy/move, long-range retrieval, no-op gating (replaces attention sinks), copy suppression, confidence regulation |
| Size | v0: 10–50M parameters · v1: ~10⁸ · hypothesis band for broad competence: 10⁸–10⁹ active |
| Lifecycle | Pretrained once, then frozen; its contract changes only with a major version |

### 4.2 Module library

| Module type | Content | Payload | Build route | Size guide |
|---|---|---|---|---|
| Generic function | An operation usable across domains (compare, extract, summarize a span, track an entity) | LoRA delta or small block at a slot | Trained or distilled | 1–10% of core |
| Domain-bound function | A function specialized to a domain (extract dosing from clinical text) | LoRA delta stacked on its generic parent, with fallback to the parent | Distilled, or programmed when the domain rule is exact | 0.5–5% of core |
| Programmed | Exact algorithms (add, compare, count, copy, sort, unit conversion) | Compiled or written weights | Programmed | 10⁴–10⁶ parameters |
| Lookup memory | Static associations (terms, facts, n-grams) | Hashed embedding tables with deterministic addressing | Trained | Large, sparse access |
| Regulator | Know/don't-know gating, refusal gating, confidence | Small heads, neurons or directions | Trained or edited | Tiny |
| Recurrent state | Carries state across a sequence (entities, beliefs, counters, game state) | Small delta-rule linear RNN with β ∈ [0, 2], so transition eigenvalues span [−1, 1] | Trained | 1–5% of core |
| Generated (optional) | Variants synthesized on demand from a description | Hypernetwork output | Generated | Not stored |

Size guides are design targets, not measurements.

### 4.3 Module manifest (required for every module)

```yaml
id: fn.compare_numeric.v1
type: programmed            # generic | domain | programmed | lookup | regulator | state | generated
function: compare_numeric
domain: null                # e.g. biomed.clinical_text for domain-bound variants
level: 2                    # catalog level 1–3; a domain variant takes its parent's level
parent: null                # generic function a domain-bound variant stacks on and falls back to
interface:
  reads: [number_code]
  writes: [comparison_flag]
  slot: L4.post_attn
  mode: additive_delta      # additive_delta | output_mix | sequential
  kv_write: false           # true: switches only at span starts, and the span's keys and values are recomputed
cost: {params: 117000, bytes: 234000, flops_per_token: 2.3e5}   # 16-bit weights; FLOPs ≈ 2 × params
tests: tests/functions/compare_numeric/   # exhaustive for programmed modules
provenance: {route: programmed, tool: tracr, builder: agent-PRG, license: Apache-2.0}
compat: {core_contract: ">=1.0,<2.0"}
status: active              # proposed | prototype | differentiating | active | suppressed | eliminated | archived
version: 1.0.0
sha256: <content hash>
```

### 4.4 Interface contract

- **Representation contract.** Fixed residual width *d* and reserved subspaces for the shared data types (Level 0 of the catalog): number code, ordinal/count manifold, binding IDs, task/function vector, concept directions. The bases are measured on the frozen core and published with the contract; the number code is imposed at the input by the Fourier number embedding.
- **Call semantics.** Each module declares its slot, what it reads and writes, and whether it adds a parameter delta, mixes outputs or runs in sequence. Order-sensitive compositions are declared explicitly.
- **Invariants.** A disabled module has exactly zero effect. Write norms are bounded. Execution is deterministic. Slot compatibility is checked at load.
- **Versioning.** Semantic versioning on the contract; each module declares the contract range it supports.
- **Swap semantics.** Switch points, the key/value rule and the fallback chain are part of the contract (§4.9).

### 4.5 Controller

| Level | Decision | Granularity | Method |
|---|---|---|---|
| Option selection | Which module set serves the current span | Task or span | Options policy with a switching (deliberation) cost |
| Gating | How much each resident module contributes | Per step | Small learned gate within the resident set |
| Prefetch | What to load next | Ahead of need | Predictor over plan and state; lookup modules prefetch exact rows from the input tokens |
| Fallback | What runs while a module loads, or when it is missing or confidence is low | Any | Chain: domain variant → generic parent → core only |

Budget: controller compute ≤ 15% of total FLOPs. Every output carries a trace of the modules used, their scores and every switch.

### 4.6 Memory hierarchy and residency

| Tier | Medium | Holds |
|---|---|---|
| T0 active | GPU VRAM or unified RAM | Core, controller, active modules, KV state |
| T1 cache | System RAM | Hot and prefetched modules |
| T2 library | NVMe SSD | All validated and suppressed modules |
| T3 synthesize (optional) | Compute | Modules generated on demand instead of fetched |
| T4 archive | Offline storage | Eliminated modules, for audit and rollback |

Rules (each measured per hardware profile during the build):

- **Decode ceiling:** tokens/s ≤ fast-memory bandwidth ÷ active bytes per token.
- **Dwell rule:** with perfect prefetch, a module must stay in use for at least *f·R* steps; without prefetch, at least *10·f·R* steps for ≤ 10% overhead. Here *f* = module size ÷ core size and *R* = fast-memory bandwidth ÷ link bandwidth.
- **Cache sizing:** start from the skew reported in MoE offloading, where roughly 15–20% of experts handle about 80% of tokens; re-measure on Flow.

| Profile (approximate) | Fast memory | Library link | R |
|---|---|---|---|
| Desktop GPU | ~1 TB/s VRAM | RAM over PCIe 4: ~25 GB/s · NVMe Gen4: ~7 GB/s | ~40 · ~150 |
| Unified-memory laptop | ~100–135 GB/s | NVMe: ~5–7 GB/s | ~20 |
| Raspberry Pi 5 | ~17 GB/s, ≤ 16 GB RAM | NVMe over PCIe x1: ~0.4–0.9 GB/s | ~20–40 |

### 4.7 Developmental lifecycle

| State | Meaning | Enter when | Exit when |
|---|---|---|---|
| Proposed | Listed in the vocabulary, no weights | Vocabulary review | Build starts |
| Prototype | Built; unit tests pass | Tests pass | Integration tests pass |
| Active | Registered and routable | Integration tests pass | Usage below θ → Suppressed; split → Differentiating |
| Differentiating | Being split into specialized variants | Usage forms separable clusters | Variants beat the parent on held-out tests |
| Suppressed | In the cold tier, still routable on demand | Usage < θ over window W | Used again → Active; passes elimination tests → Eliminated |
| Eliminated | Removed from the library | Change ≤ ε on standard and shift suites | — |
| Archived | Stored offline | On elimination | Restore request |

Duplicates are merged before elimination. Elimination requires shift suites as well as standard suites, because self-repair can hide a removed module's role on standard tests. Modules flagged critical (safety regulators) are never eliminated automatically.

### 4.8 Build routes

| Route | Use when | Tools | Verification |
|---|---|---|---|
| Programmed | The function has a short algorithmic description | Tracr (RASP to weights), Neural Comprehension CoNNs, weight-space programming by a frontier LLM | Exhaustive or property tests |
| Trained | The function is learned from data | PyTorch + PEFT (LoRA) against the frozen core | Held-out tests |
| Distilled | A teacher model already has the function | Teacher-generated data per function | Teacher agreement + held-out tests |
| Generated | Many variants are needed on demand | Hypernetwork in the Text-to-LoRA style | Recovery vs stored modules |
| Grafted or edited | Repair or transplant | Weight-space programming in a sandbox | Before/after evaluation, logged diffs |

Rule of thumb: a function is programmable when its description is much shorter than its weights (a short RASP program). Algorithms with short RASP-L programs are also learnable with length generalization, so they may be trained instead. Knowledge does not compress that way, so it is trained, distilled or stored in lookup memory. Annex A lists the architecture known to produce each catalog function.

### 4.9 Dynamic loading (hot swap)

| Element | Specification |
|---|---|
| Separation | Each function is its own module pack, built against the frozen core and writing only to its declared slots and subspaces; a disabled module has exactly zero effect |
| Slot buffers | Fixed-size buffers per slot are reserved in T0 at startup; loading is a copy plus a pointer switch, with no allocation during generation |
| Load path | T2 (memory-mapped safetensors) → T1 (pinned buffer) → T0, asynchronous and double-buffered; SHA-256 and contract-range check before activation; packs hold tensors only, never executable code |
| Switch points | Modules switch on or off only between tokens. A module switched on at token t acts on tokens t onward; earlier tokens keep their cached keys and values |
| Key/value rule | Default `kv_write: false`: the module does not change key or value projections and can switch at any token. With `kv_write: true` it switches only at span starts, and the span's keys and values are recomputed |
| Never wait | While a module loads, or if it is missing, the fallback chain runs (domain variant → generic parent → core only); the switch happens when the load completes. Generation never blocks on a load |
| Training for swaps | Gates and controller are trained with module dropout: modules randomly absent, switched on late or switched off early |
| Session pinning | A session pins the library version; library changes apply to new sessions after registry validation |
| Telemetry | Load latency, stalls (target zero), fallback rate, time until the specialized module is active, hit rate, dwell |

Cost at the sizes of §4.2: a rank-16 LoRA module on a 100M-parameter core is about 5 MB. On a desktop GPU it moves from RAM in about one decode step and from NVMe in about four, so it must stay in use for 1–4 tokens with prefetch and 10–40 without (dwell rule, §4.6).

## 5. Functional vocabulary v0

| ID | Function | Level | Reads → writes | Route |
|---|---|---|---|---|
| prim.* | Previous-token, induction (token, concept), copy/move, retrieval, no-op gating, suppression, confidence | 1 | Core-internal | Core pretraining, verified by probes |
| add_exact | Exact multi-digit addition and subtraction | 2 | number_code → number_code | Programmed |
| compare_numeric | Greater, less, equal | 2 | number_code → comparison_flag | Programmed |
| successor | Next item in ordered sequences | 2 | ordinal → ordinal | Programmed |
| count_len | Character and item counting | 2 | tokens → count manifold | Programmed or trained |
| bind_track | Entity–attribute binding and state tracking | 2 | tokens → binding IDs | State module (trained) |
| fact_lookup | Static fact and term recall | 2 | n-grams → concept directions | Lookup memory |
| agree_syntax | Grammatical agreement | 2 | syntax features → token logits | Trained |
| summarize_span | Compress a span to its key content | 2 | span → summary state | Distilled |
| extract_slots | Extract typed fields from text into slots | 2 | text → structured slots | Distilled |
| unit_convert | Unit conversion with SI prefixes and dimension checks | 2 | number_code + unit → number_code | Programmed |
| two_hop | Compose two lookups through a bridge entity | 3 | Composed by the controller | Composition |
| rule_induce | Infer an abstract rule from examples and apply it | 3 | task vector → output | Trained |
| know_gate | Answer vs "don't know" | 3 | entity familiarity → gate | Regulator |
| refuse_gate | Refusal decision | 3 | harm features → gate | Regulator |
| extract_dosing@clinical | Dose, unit, route and frequency from clinical text; variant of extract_slots | 2 · domain | text → structured slots | Distilled |
| summarize_methods@biomed | Methods-section summary; variant of summarize_span | 2 · domain | span → summary state | Distilled |
| unit_convert@lab | Analyte-specific laboratory conversions (e.g., glucose mg/dL ↔ mmol/L); variant of unit_convert | 2 · domain | number_code + unit → number_code | Programmed |

Each domain-bound variant names its generic parent in its manifest and falls back to it when the domain is absent. The example domain is replaceable. Any clinical data used for domain modules must be public and de-identified.

## 6. Data and file formats

| Artifact | Format |
|---|---|
| Module pack | Directory: `manifest.yaml`, `weights.safetensors` (tensors only), `tests/`, `card.md`, `LICENSE` |
| Library index | SQLite `registry.db` + JSON export; SHA-256 content hashes |
| Datasets | Parquet or JSONL + data cards; split manifests with hashes |
| Traces | JSONL per run: span, modules, scores, residency and swap events |
| Checkpoints | safetensors + library snapshot manifest (for rollback) |
| Reference runtime | PyTorch + safetensors; runs every module type and the residency manager |
| llama.cpp export (after a port) | GGUF core once the core graph (looped blocks, slot hooks) is ported; LoRA-type modules as GGUF adapters with per-request scales; programmed and lookup modules need custom operators |
| Reports | Markdown + CSV |

**File flow:** sources → importers (license and hash checks) → datasets → build jobs → module packs → registry (validation) → packaging → runtime → traces and telemetry → dashboards → lifecycle decisions written back to the registry.

## 7. Operations and administration

| Tool | Function |
|---|---|
| Module registry and library manager (CLI + web UI) | Register, validate, promote, suppress, eliminate, restore; license and provenance check on import |
| Residency monitor | Tier occupancy, hits and misses, dwell histograms, link bandwidth, load latency, stalls, fallback rate, energy proxy |
| Training and evaluation dashboard | Per-module tests, module × suite regression matrix, baselines |
| Trace viewer | Which modules produced an output; diff of two runs |
| Experiment tracker | MLflow runs, parameters and artifacts |
| Checkpoint and rollback service | Snapshot of core + library state; one-command rollback |
| Profiler | Latency, memory, FLOPs, energy where measurable |
| Remote access | Authenticated web UI, local-only by default |
| Audit log | Append-only record of every library change (who, what, why) |
| Import and export | Datasets, module packs, library snapshots, reports |

Dashboards use a light theme by default. They render a skeleton or cached state immediately, then stream fresh data as it arrives.

## 8. Resource limits (what bounds performance)

1. **Fast-memory bandwidth:** sets decode speed.
2. **Fast-memory capacity:** sets what stays resident.
3. **Link bandwidth to RAM and SSD:** decides whether offloading works (dwell rule, §4.6).
4. **Controller compute:** budgeted at ≤ 15% of FLOPs.
5. **Interference between composed modules:** caps composition quality.
6. **Training data per function and teacher cost:** set module quality and budget.
7. **Energy:** per-token SSD access costs several times more than fast memory, so loads must be amortized.
8. **Prefetch accuracy:** decides whether modules arrive before they are needed or the fallback chain runs instead.

## 9. Metrics and acceptance

| # | Metric | Target (v0) | Kill criterion |
|---|---|---|---|
| M1 | Accuracy on held-out combinations of known functions | ≥ 5 points above a token-routed MoE matched on active and total parameters | Not above the matched MoE |
| M2 | Capability per active parameter, per FLOP, per joule | Reported against all baselines | — |
| M3 | Add a function | Old-suite change ≤ 1 point | > 1 point of forgetting |
| M4 | Remove a function | Unrelated-suite change ≤ 1 point | > 1 point of collateral damage |
| M5 | Residency | Median dwell ≥ f·R; swap overhead ≤ 10%; prefetch hit ≥ 90% | Dwell < f·R → drop the SSD tier |
| M6 | Controller | ≤ 15% of FLOPs | > 15% with no accuracy gain |
| M7 | Traceability | 100% of outputs carry a module trace | — |
| M8 | Lookup memory | Downstream gain, not only lower perplexity | No downstream gain at v0 scale → defer to v1 |
| M9 | Programmed modules | 100% pass on exhaustive tests; accuracy gain over the same system without programmed modules | — |
| M10 | Robustness after elimination | Shift-suite change ≤ 1 point | > 1 point |
| M11 | Dynamic loading | A module loaded at token t matches the same module resident and off until t (within tolerance); zero generation stalls; switching mid-sequence costs ≤ 1 point vs modules resident from the start | > 1 point, or any stall |

Targets are initial design values; they are fixed at gate G2 of the build pipeline.

## 10. Risks

| Risk | Why it matters | Mitigation |
|---|---|---|
| Imposed functions may cost quality | Trained experts specialize in superposed micro-domains, not clean functions | Measure against a matched MoE (M1); allow learned sub-structure inside modules |
| Composition interference | Added deltas can conflict | Contract subspaces, interference tests, ordering rules |
| Controller paradox | Choosing functions requires understanding the task | Hierarchical controller, fallback path, compute budget |
| Lookup memory may not help small models | An independent test at 0.5–4B parameters found lower perplexity but no benchmark gain | Early check M8 |
| Scale beats structure | Hand-designed structure often loses to scale | Keep only structure that wins on the metrics |
| Prior-art overlap | SkillNet, DEMix and DeepThink cover parts of the design | Novelty claims limited to the combination (§12) |
| Teacher terms | Some commercial APIs restrict training on their outputs | Prefer open-weight teachers with permissive licenses; check terms |
| Runtime support | Local runtimes do not natively support looped cores, slot hooks, programmed or lookup modules, or tiered module loading | PyTorch reference runtime first; port to llama.cpp after a go decision |
| Mid-sequence switching | A module switched on mid-sequence creates states never seen in training | Module dropout in training; M11 |
| Self-repair hides removal effects | Downstream components compensate for a removed module on standard tests | Elimination also requires shift suites (M10) |

## 11. Optional extension: lifetime learning

User-side adapters live in a separate library namespace. They never modify the base library and can be exported, imported or removed independently.

## 12. Position relative to prior work

| Work | Shares with Flow | Flow differs by |
|---|---|---|
| Mixture of experts | Sparse activation | Designed functions; span-level residency |
| SkillNet, SkillNet-X | Predefined skills; two skill axes | Developmental lifecycle; tiered library; machine-built modules |
| DEMix | Adding and removing modules | Functions with domain-bound variants, not domains alone |
| DeepThink | Bank, selector, recurrent core, typed interfaces | Residency; lifecycle; programmed modules (DeepThink reports no experiments) |
| Engram | Offloadable lookup memory | One module type among several |
| Temporal MoE, StickyMoE | Trained locality for offloading | Locality of designed functions |
| Neural Comprehension | Compiled modules inside a language model | One build route among several |
| Weight-space programming | LLM-written and LLM-edited weights | Used as a build and repair route |
| Multi-adapter serving (vLLM, S-LoRA) | Loading adapters at runtime | Function-typed modules, fallback chain, residency-aware prefetch |

## 13. Evidence anchors

- [Function catalog (Flow)](https://claude.ai/artifact/Lt5ZCKRD3zBsw1Bey2pHA9)
- [Engram: conditional memory via scalable lookup](https://arxiv.org/html/2601.07372v1)
- [Engram at 0.5–4B: a negative downstream result](https://github.com/t-hertel/Engram)
- [Temporal Mixture-of-Experts](https://neurips.cc/virtual/2026/170072)
- [Expert skew and offloading in llama.cpp](https://github.com/ggml-org/llama.cpp/issues/20757)
- [SSD offloading and energy](https://arxiv.org/html/2508.06978v1)
- [Knowledge capacity: about 2 bits per parameter](https://arxiv.org/abs/2404.05405)
- [Tiny Recursive Models](https://arxiv.org/abs/2510.04871)
- [Neural Comprehension (compiled modules)](https://arxiv.org/abs/2304.01665)
- [Text-to-LoRA](https://sakana.ai/text-to-lora/)
- [Weight-space programming](https://rikiyat.github.io/blog/weight-space-programming/)
- [RouterInterp: superposed specialization of experts](https://icml.cc/virtual/2026/79470)
- [SkillNet-NLU](https://arxiv.org/abs/2203.03312) · [DEMix](https://aclanthology.org/2022.naacl-main.407) · [DeepThink](https://arxiv.org/abs/2610.16171)
- [Runtime LoRA loading in vLLM](https://docs.vllm.ai/en/latest/features/lora/) · [S-LoRA](https://arxiv.org/abs/2311.03285) · [LLM in a flash](https://arxiv.org/abs/2312.11514)
- Architecture-to-function evidence: Annex A

## 14. Glossary

| Term | Meaning |
|---|---|
| Active parameters | Parameters read for one step of computation |
| Dwell | Number of consecutive steps a loaded module stays in use |
| f | Module size ÷ core size |
| R | Fast-memory bandwidth ÷ link bandwidth to the tier the module comes from |
| Slot | Declared insertion point in the core where a module reads and writes |
| Contract | The fixed representation, slots, call semantics and invariants modules must respect |
| Residency | Which tier a module currently occupies |
| Suppression | Reversible demotion of a module to the cold tier |
| Elimination | Removal of a module after it passes regression and shift tests |
| Option | A module set chosen by the controller for a span of computation |
| Hot swap | Loading or unloading a module between tokens during generation |
| Fallback chain | Domain variant → generic parent → core only; runs while a module loads or when it is missing |
| Module dropout | Training with modules randomly absent or switched mid-sequence, so swaps stay in distribution |
| State module | Small recurrent block that carries state across a sequence |
| CoNN | Compiled neural network: a module whose weights encode a program |
| LoRA | Low-rank adapter; the default payload for trained modules |
| GGUF | File format used by llama.cpp for models and adapters |

## 15. Version history

| Version | Date | Change |
|---|---|---|
| 0.1 | 2026-11-03 | First summary specification |
| 0.2 | 2026-11-03 | Dynamic loading (§4.9), recurrent state modules, core design choices, M11, Annex A |

## Annex A. Architectures known to produce catalog functions

Architecture sets what can form cheaply; data and training decide whether it forms. Levels refer to the [function catalog](https://claude.ai/artifact/Lt5ZCKRD3zBsw1Bey2pHA9).

| Catalog function (level) | What produces it | Place in Flow | Sources |
|---|---|---|---|
| Number helix (L0), addition (L2) | Numbers coded as cos/sin at periods 2, 5, 10, 100; addition as rotation. Can be imposed: FoNE (one token, 2 dimensions per digit) reaches 99% on 6-digit addition with 64× less data; Abacus digit-position embeddings, trained on 20 digits, reach ~99% on 100 | Number embedding in the contract; add_exact programmed | [Kantamneni & Tegmark 2025](https://arxiv.org/abs/2502.00873) · [FoNE](https://arxiv.org/html/2502.09741v1) · [Abacus](https://arxiv.org/abs/2405.17399) |
| Count manifold (L0), line breaking (L2) | Count on a curved low-dimensional manifold; heads compare two counts by rotating one manifold against the other in the QK product | Core + count_len | [Gurnee et al. 2025](https://transformer-circuits.pub/2025/linebreaks/index.html) |
| Binding IDs, entity and belief tracking (L0–L3) | Attention with ordering IDs and pointer lookbacks. General state tracking is beyond fixed-depth transformers and SSMs (under standard complexity assumptions); it needs recurrence, or linear RNNs with negative transition eigenvalues | State module (bind_track) or looped core | [Prakash et al. 2025](https://arxiv.org/abs/2505.14685) · [Merrill et al. 2024](https://arxiv.org/abs/2404.08819) · [Grazzi et al. 2024](https://arxiv.org/abs/2411.12537) |
| Previous-token and induction heads (L1) | Position signal, then two attention layers composing (the previous-token head feeds the induction head's keys); local token mixing before attention (Canon layers) gives the same shortcut and ~2× reasoning depth; data with repeats | Core | [Olsson et al. 2022](https://arxiv.org/abs/2209.11895) · [Allen-Zhu 2025](https://arxiv.org/abs/2512.17351) |
| Retrieval, copying, name movers (L1) | Softmax attention over the full KV cache; fixed-state models fail to copy beyond their state size | Core | [Jelassi et al. 2024](https://arxiv.org/abs/2402.01032) |
| Successor heads (L1) | Mod-10 features read by an OV circuit | successor programmed | [Gould et al. 2023](https://arxiv.org/abs/2312.09230) |
| Key–value memory, fact storage and recall (L1–L2) | MLP as key–value memory (~2 bits per parameter); product-key memory layers and hashed n-gram tables give sparse reads that can live off the GPU | fact_lookup (lookup memory) | [Geva et al. 2021](https://arxiv.org/abs/2012.14913) · [Berges et al. 2024](https://arxiv.org/abs/2412.09764) · [Engram](https://arxiv.org/html/2601.07372v1) |
| Arithmetic heuristics (L1) | Sparse MLP neurons, each a range or modulo rule: heuristics, not an algorithm | Replaced by programmed modules | [Nikankin et al. 2024](https://arxiv.org/abs/2410.21272) |
| Attention sink (L1) | Softmax must sum to 1, so idle heads park attention on one token; a sigmoid gate after attention removes sinks and massive activations | Gated attention in the core | [Gu et al. 2024](https://arxiv.org/abs/2410.10781) · [Qiu et al. 2025](https://arxiv.org/abs/2505.06708) |
| Copy suppression (L1) | Late heads whose OV circuit maps a token to its own negative logit | Core | [McDougall et al. 2023](https://arxiv.org/abs/2310.04625) |
| Entropy neurons (L1) | Write into the unembedding's null space; the final LayerNorm rescales, changing confidence but not ranking | Core; final LayerNorm kept | [Stolfo et al. 2024](https://arxiv.org/abs/2406.16254) |
| IOI, greater-than (L2) | IOI: three chained head stages (duplicate-token → S-inhibition → name movers); greater-than: MLPs boost years above the start year | compare_numeric programmed | [Wang et al. 2022](https://arxiv.org/abs/2211.00593) · [Hanna et al. 2023](https://arxiv.org/abs/2305.00586) |
| In-context learning, function vectors (L1–L2) | Attention plus bursty, Zipf-distributed data; one linear-attention layer ≈ one gradient-descent step; mid-layer heads emit the function vector | Core; task vector as a contract subspace | [Chan et al. 2022](https://arxiv.org/abs/2205.05055) · [von Oswald et al. 2023](https://arxiv.org/abs/2212.07677) · [Todd et al. 2024](https://arxiv.org/abs/2310.15213) |
| Two-hop reasoning (L3) | k hops need ≈ log k layers; fails when the bridge entity resolves too late; non-recurrent stacks fail out of distribution, and weight sharing across layers improves it | Looped core + two_hop | [Sanford et al. 2024](https://arxiv.org/abs/2402.09268) · [Biran et al. 2024](https://arxiv.org/abs/2406.12775) · [Wang et al. 2024](https://arxiv.org/abs/2405.15071) |
| Abstract rule induction (L3) | Symbol-abstraction heads → symbolic induction heads → retrieval heads | rule_induce on core attention | [Yang et al. 2025](https://arxiv.org/abs/2502.20332) |
| Knowledge and refusal gating (L3) | Single linear directions for entity familiarity and for refusal | Regulators | [Ferrando et al. 2024](https://arxiv.org/abs/2411.14257) · [Arditi et al. 2024](https://arxiv.org/abs/2406.11717) |
| Multilingual processing (L3) | Shared language-independent features in middle layers; language-specific circuits at input and output | Language at the ends, functions in the middle | [Lindsey et al. 2025](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) |

No minimal recipe is known for forward planning or self-repair; both appear with scale.

**Training triggers.** Repeated spans (induction), bursty sequences (in-context learning), weight decay (clean Fourier arithmetic instead of memorization). Algorithms with short [RASP-L](https://arxiv.org/abs/2310.16028) programs length-generalize when trained; the rest are programmed.
