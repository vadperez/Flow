# Flow

**Flow** is a proposal for a function-first modular neural model: a small core, typed registers, and function modules that are kept as separate packs of files and loaded and unloaded while the model generates. Exact functions (arithmetic, comparison, lookup) are written directly as weights instead of being learned, and the core is trained not to learn them.

## What is new

**How models are built today.** A conventional neural model is one block of weights. Its designers choose layer types and sizes, then train everything on a broad objective. Whatever functions the model needs (arithmetic, comparison, recall of facts, keeping state) emerge somewhere inside those weights, approximately, and mixed with everything else. Three consequences follow:

- All weights are loaded and run for every input, whatever the task needs.
- No function can be inspected, corrected, replaced or removed by itself, because none lives in one place.
- Exact operations are learned as approximations, so a model with billions of parameters can still miscalculate.

Existing ways around this each solve one part. Mixture-of-experts models route between blocks, but the blocks have no assigned function. Adapter libraries add swappable parts, but those parts are learned and entangled with the base model. Tool calling gives exact results, but through a text round trip outside the network. Compiled-network research writes programs as weights, but for one function at a time.

**What Flow does differently.** It decides the functional organization before training, and makes it a property of the files on disk:

1. **Functions come first.** The design starts from a vocabulary of functions and gives each one a module with a declared interface: typed arguments in, one typed result out.
2. **Exact functions are written, not learned.** Their weights are constructed and tested exhaustively. They cost no training and make no arithmetic errors.
3. **The core is trained not to know them.** During pretraining the core learns where a result is needed and learns to abstain there. This keeps each function in its module and nowhere else, and it is measured: remove the module and its function must disappear while the others stay.
4. **Modules are loaded while the model runs.** Each is a separate pack of files. Only the modules a task needs sit in fast memory; the rest stay in RAM or on disk. Loading one between two output tokens took 12–24 ms in the prototype.
5. **Modules compose inside the network.** One module's result goes to the next through typed registers, with no conversion to text, and every call is recorded in a trace.
6. **The library has a lifecycle.** A function can be added without retraining anything else, specialized into a domain variant, suppressed, and eliminated when tests show it is safe.

**What this is meant to give.** A small active network whose competence grows by adding files, lower memory and compute per task, exact results where exactness is possible, and a record of which module produced each output.

**What is and is not established.** Points 2 to 5 work in the miniature run in this repository. Each idea has precedents (see *Related work*); the combination, the abstaining core and the localization measure are, to our knowledge, new. Whether the approach gives broad competence at a fraction of the active parameters is a hypothesis that the pipeline's baselines are designed to test.

**Status: untrained research prototype.** This repository holds the design, tested building blocks and a miniature end-to-end run on a CPU. No model has been trained at useful scale, and no comparison against baselines has been run. Anyone with GPU time is welcome to take the pipeline further; see *Next steps*.

Version 0.3 · 2026-11-10 · implements the first slices of `docs/flow-build-pipeline-v0.3-draft.md` (stages S0, S1, S3 and S5A in miniature).

## Best for

Flow should pay off where the work breaks into known, typed functions and only a few are needed at any moment. The saving comes from functions that are written instead of learned, and from modules that stay on disk until needed. The figures below are estimates; none has been measured yet.

| Application | Why it fits | Core to use | Expected saving in active parameters |
|---|---|---|---|
| Machine and process control | Typed sensor inputs; many modes, one active at a time; known functions (limits, filters, sequencing) | A small native core without language (a few million parameters or fewer) | Large: only selection and sequencing are learned |
| Robot skill libraries, excluding vision | One module per skill, loaded per task; kinematics and transforms are written | A small native core; a separate vision encoder if cameras are used | 100–1,000× against large vision-language-action models, for the control part |
| Laboratory and instrument automation | Fixed command sets, units, calibration tables, exact protocols | A small native core | Large: mostly lookup and state |
| Telemetry and protocol handling | Typed and rule-based; rules change by device, so swappable packs matter | A small native core | Large: nearly all of it can be written |
| Small on-device assistants for calculation-heavy domains | A small language core for the wording, modules for the exact part | A frozen published language model (Qwen3.5-0.8B or 4B) with the function layer | Moderate, on those tasks only |
| Devices serving many domains with little memory | One core, domain packs paged in from storage | Either, by whether language is needed | Active memory shrinks; total stored size does not |
| Safety monitors beside a larger model | A small module checks outputs against exact limits, with a trace | No core: a module beside the host model | An addition, not a reduction |

Not a fit: perception (vision and audio encoders do not decompose into typed functions), open-ended language and reasoning (the core stays the bottleneck), and tasks with one fixed function.

Two benefits apart from size: every output carries a trace of the module that produced it, and a function can be corrected or replaced without retraining anything else.

## Models and weights

Nothing here is trained at useful scale. The table gives what the prototype used and what a trained system is expected to weigh. Sizes are at 16 bits per weight unless noted.

| Part | Architecture | Parameters | File size | State |
|---|---|---|---|---|
| Prototype core (toy run) | Qwen3.5 text architecture, 4 layer passes, width 128 | 0.52M | 2.1 MB (32-bit) | Trained on toy text in 6 minutes on a CPU; rebuilt by `scripts/toy_walkthrough.py` |
| Flow native core v0 | Qwen3.5 text architecture, scaled down: width 512, 16 layer passes from 8 unique layers (`configs/core_v0.yaml`) | 34.7M | 69 MB | Blank; pretraining is pipeline stage S4 |
| Controller block | One small transformer layer over the core's hidden states | 0.22M in the toy run; about 0.9M planned | about 2 MB | Fitted with the first function |
| Trigger and binder, per function | Small heads on the controller block | 0.13M in the toy run | under 1 MB | Fitted per function and per core |
| Body of `add_exact` | Written ReLU circuit, 4 layers | 4.53M stored, 0.14M non-zero | 9.1 MB dense; under 1 MB if stored sparse | Final; in `library/` |
| Body of `compare_numeric` | Written ReLU circuit, 4 layers | 1.36M stored, 0.02M non-zero | 2.7 MB dense | Final; in `library/` |
| Borrowed core, development | Qwen3.5-0.8B, text network, frozen | 0.75B | 1.5 GB | Published weights, downloaded, never changed |
| Borrowed core, target | Qwen3.5-4B, text network, frozen | 4.21B | 8.4 GB | Published weights, downloaded, never changed |

So a trained v0 system with the native core is expected to be on the order of 100 MB in all: 69 MB of core, plus a library that grows by roughly 1–10 MB per exact function. With a borrowed core, the project adds only megabytes to the published model. These totals are estimates.

Why Qwen3.5: Apache-2.0, a text-only model class in Transformers, and layers that already contain what the spec asks for. Details and alternatives: `docs/MODEL_CHOICE.md`.

## Adding functions

The number of functions can grow, and growing it without retraining anything else is one of the six claims (`specs/claims.md`, C4). The v0 plan has 12 modules; the type system addresses up to 65,536 symbols and puts no limit on the number of modules. What limits growth in practice is trigger accuracy: every added function is one more candidate at each call site.

To add a function:

1. **Declare it** in `specs/vocabulary.yaml`: name, operations, argument and result types (NUMBER, SYMBOL, FLAG).
2. **Write the body** as a subclass of `flow.bodies.base.Body`. For an exact function, build it with `flow.bodies.circuit.CircuitBuilder`, as `add_exact.py` and `compare.py` do. A learned or computed body uses the same interface.
3. **Test the body alone**, exhaustively where the input classes allow it (`tests/test_add_exact.py` is the model).
4. **Build the pack**: add the body to `scripts/build_library.py`, which writes `library/<id>/manifest.yaml` and `body.safetensors`.
5. **Generate task records** with call traces for the new function, varying every operation in every context (pipeline step S2.1, rule 6).
6. **Fit its trigger and binder** with `flow.function_layer.train.fit(..., train_controller=False)`: the core, the controller block and all other modules stay frozen. The toy walkthrough adds `compare_numeric` this way.
7. **Register and check**: `python -m flow.registry scan library/`, then confirm that the old suites moved by at most 1 point (M3) and that removing the new module removes only its function (M12).

Not yet shown: how far this holds beyond two functions. The pipeline measures it at 12 (stages S5 to S9).

## Requirements

| To do this | You need |
|---|---|
| Tests, written bodies, registry | Python 3.11 or newer (3.12 recommended), 64-bit Windows, Linux or macOS, 8 GB memory; the packages in `ops/requirements-base.txt` and `ops/requirements-dev.txt` |
| Blank cores, function layer, toy walkthrough | The above plus PyTorch 2.7 or newer and Transformers 5.22 or newer (`ops/requirements-model.txt`); a CPU is enough |
| Qwen3.5-0.8B as a frozen core | 8 GB memory and 2 GB disk; CPU or any GPU |
| Qwen3.5-4B as a frozen core | A GPU with 12 GB or more, and 10 GB disk |
| Pretraining the native core | One 24 GB GPU for 4–8 hours per run (estimate); 118–200 GPU-hours for the whole v0 pipeline |
| Exhaustive tests | 1.3 GB of free memory, about 4 minutes |

Exact tested versions are in `ops/lockfile.txt`. `python ops/compat_check.py` reports what a given machine can run; it also runs on older Pythons and says what is missing. On Windows on Arm, PyTorch is CPU-only and installs from its own index: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Start here

```
python ops/compat_check.py                 # what this machine can run; changes nothing
pip install -r ops/requirements-model.txt -r ops/requirements-dev.txt
pip install -e .                           # makes "python -m flow.registry" available
python -m pytest                           # 72 tests, about one minute
python scripts/toy_walkthrough.py          # the whole mechanism on a CPU, about 10 minutes
python ops/next_steps.py                   # where the pipeline stands and what can start
```

Then fetch the model files, without the weights, and check them:

```
python scripts/fetch_model_files.py --model Qwen/Qwen3.5-0.8B --dry-run     # list files and sizes
python scripts/fetch_model_files.py --model Qwen/Qwen3.5-0.8B               # about 23 MB, no weights
python scripts/check_tokenizer.py   --model models/Qwen3.5-0.8B
python scripts/make_blank_model.py  --spec configs/core_v0.yaml --describe  # the Flow core, 34.7M parameters
```

Model choice and the answer to "model files without the weights": `docs/MODEL_CHOICE.md`.

## What is here

| Folder | Content | Pipeline step |
|---|---|---|
| `src/flow/typesys/` | NUMBER, SYMBOL and FLAG codes, scanner, emitter, registers | S1.3 |
| `src/flow/bodies/` | Builder for exact ReLU circuits; `add_exact` (add, subtract) and `compare_numeric` (greater, less, equal) as written weights | S5A.1 |
| `src/flow/core/` | Blank cores: the Qwen3.5 text architecture built from a configuration, without weights; answer-masked pretraining | S4.1, S4.2 |
| `src/flow/borrowed/` | A published Qwen3.5 model as a frozen core, text network only; tokenizer adapter and number check | S3.1 |
| `src/flow/function_layer/` | Controller block, module heads (trigger with call, operation and final; binder with hard pointers), call loop, fitting, scoring | S3.2 |
| `src/flow/runtime/` | Module packs on disk; a session that loads and unloads them between tokens and records every event | S3.4 |
| `src/flow/registry/` | Manifest schema check, SQLite registry with the lifecycle rules, command line | S1.4 |
| `src/flow/data/` | Toy tokenizer, toy task generator with call traces, alignment of traces to any tokenizer | S2.1 in miniature |
| `library/` | The two body packs, built by `scripts/build_library.py` (bit-for-bit reproducible) | S5A |
| `specs/` | Vocabulary, type system, module contract, claims, manifest schema, task graph of the pipeline | S1 |
| `configs/` | Flow core v0, borrowed cores, published Qwen3.5 configurations | S1, S4 |
| `ops/` | Compatibility check, pipeline status tool, pinned versions, decisions, budget, gate packet template | S0 |
| `dashboards/mockups/` | Mockup of the operations console: pipeline, registry, residency, traces (`preview.png` shows it) | S0.5 |
| `docs/` | Spec v0.2, pipeline v0.3 draft, model choice, the original note, reference run | |
| `tests/` | 72 fast tests and 4 exhaustive ones (`python -m pytest -m slow`: about 4 minutes, 1.3 GB of memory) | |

## What the toy walkthrough shows

A 0.52M-parameter blank core is pretrained on toy text with answer masking, frozen, and given two modules. Reference run on 2 CPU cores, Python 3.13, PyTorch 2.15 (`docs/reference_run/`):

| Check | Result |
|---|---|
| Core alone | 0 of 556 exact. It abstains on all 556 and gives no wrong answer |
| Both modules loaded | 556 of 556 exact |
| Numbers of 8–11 digits (training had 1–6) | 555 of 555 exact |
| A pair of calls never shown in training (subtract, then greater) | 274 of 274 exact |
| Calls at positions where none belongs | 0 in 18,850 positions |
| Unload `fn.add_exact.v1` | Its answers go from 1.000 to 0.000 (abstains on 469 of 469, 0 wrong). Records that need only the comparison stay at 87 of 87. Localization L = 1.0 |
| Unload `fn.compare_numeric.v1` | 1.000 to 0.000 (abstains on 292 of 292, 0 wrong). Records that need only addition stay at 264 of 264. L = 1.0 |
| Load a module between two tokens of one text | 12–24 ms. The answer appears with the module and is `<abstain>` without it |
| Learned parameters | Core 523,816; controller block 224,133; heads 127,241 and 127,370. The bodies are written, not learned |
| Time | 518 s in all, 383 s of it core pretraining |

The toy text comes from 26 templates, so these scores show that the mechanism works and that the code is right. They do not predict scores on natural text; that is what pipeline stages S2 and S3 measure.

## What was verified, and what was not

Verified here (Linux x86-64, CPU): all tests; the exhaustive tests of both bodies (all 531,441 carry patterns, all borrow and comparison patterns); exactness in 16-bit floats including the least favourable rounding order; blank networks of the published 0.8B, 2B and 4B shapes (parameter counts) and of modified shapes (forward pass, causality, save and reload); the walkthrough end to end; the tests, the quick walkthrough and the tools from a fresh unpack of this archive.

Not verified, and therefore the first things to run:

1. Downloading from huggingface.co. The fetch script was tested up to its error messages only.
2. The real Qwen3.5 weights and tokenizer. The loader was tested on a stand-in checkpoint with the published file layout. Whether the tokenizer splits numbers into single digits is reported by `check_tokenizer.py`.
3. Windows, macOS and GPUs. Paths and file handling are written to be portable; everything was run on Linux x86-64 on a CPU only.

## How files move

```
huggingface.co ──fetch_model_files.py──▶ models/<name>/           (config, tokenizer; weights only with --with-weights)
configs/*.yaml ──make_blank_model.py───▶ runs/<name>/              (blank network: config.json + model.safetensors)
src/flow/bodies ──build_library.py─────▶ library/<id>/             (manifest.yaml + body.safetensors)
toy data + blank core ──toy_walkthrough.py──▶ runs/toy/            (core, fitted packs, registry, report.json, report.md)
library/ ──python -m flow.registry scan──▶ registry/registry.sqlite ──export──▶ reports/registry.json
ops/status.json, reports/registry.json, runs/toy/report.json ──▶ dashboards/mockups/index.html
```

Nothing is written outside this folder. `models/`, `runs/`, `registry/*.sqlite` and `reports/` are outputs and can be deleted.

## Administration

| Tool | Use |
|---|---|
| `python ops/compat_check.py` | Machine check with a clear list of what is missing; writes `reports/compat.json` |
| `python ops/next_steps.py` | Pipeline status: done, ready, waiting for a gate. `--done`, `--skip`, `--gate G1 go` record progress in `ops/status.json` |
| `python -m flow.registry scan library/` · `list` · `show` · `status` · `verify` · `export` | Module registry: index packs, move them along the lifecycle, verify file hashes, export for reports |
| `dashboards/mockups/index.html` | Console mockup. Open it in a browser. It shows embedded sample data at once; use "Open files" or `python scripts/refresh_dashboard_sample.py` for your own data, or serve the folder with `python -m http.server` |
| `Session.events`, `Session.usage()` | Every load and unload with duration and resident bytes; calls per module |
| `python ops/make_archive.py` | Pack the project into one zip without outputs and caches, for backup or hand-over; `--with-runs` adds cores, fitted packs and reports |

The console is a mockup, not a finished tool (pipeline rule 9: mockups before builds).

## Next steps in the pipeline

1. Read `docs/flow-build-pipeline-v0.3-draft.md` and `ops/decisions.md`; decide the open items for gate G1.
2. S0: put this folder under version control, add CI running `python -m pytest`, pin the Qwen3.5 revision in `configs/borrowed_core.yaml`.
3. S3.1: fetch Qwen3.5-0.8B with weights, run the tokenizer check, load it with `flow.borrowed.hf_core.load_text_core`.
4. S3.2: generate task records in that tokenizer's token space (replace the toy generator by the S2.1 generator) and fit `add_exact` on the frozen 0.8B core with `flow.function_layer.train.fit`.

## Related work

Flow combines ideas that exist separately. The closest published work:

| Work | What it shares with Flow | What Flow adds |
|---|---|---|
| Neural Comprehension (arXiv 2304.01665), OccamLLM (2406.06576), IGC (2501.00684) | Exact arithmetic attached to a frozen language model | A typed library of many functions, composition through registers, modules as loadable files |
| Composable differentiable libraries (2407.04899), The Neural Compiler (2605.22498), Tracr | Programs compiled into network modules | Loading and unloading during generation; a core trained to abstain |
| Adapter and expert libraries such as DMPEL (2506.05985), LoRA hubs, mixture of experts | A growing library with a learned router | Function-typed modules with exact bodies and a measured localization index |
| Tool calling | Exact results from outside the model | Results stay inside the network as typed values, with no text round trip |

Not found in one system elsewhere, to our knowledge: answer-masked core pretraining, a per-module localization metric with a kill criterion, and a lifecycle (differentiate, suppress, eliminate) for the library. These are hypotheses until the pipeline's baselines are run.

## How this was made

The idea and direction are the author's (`docs/functionally_organized_dynamic_parameter_network.md`). The specification, pipeline, code and tests were drafted with an AI assistant (Claude) and checked by running them; the results quoted above are from those runs.

## License

Apache-2.0 (`LICENSE`, `NOTICE`). The Qwen3.5 configuration values and the Transformers model code used here are also Apache-2.0. No trained weights of any published model are included.
