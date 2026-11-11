# Flow module contract, as implemented in this starter

Status: draft for gate G1 (pipeline step S1.4). It states what every module may rely on and what it must guarantee. Code: `src/flow/bodies/base.py`, `src/flow/function_layer/`, `src/flow/runtime/minimal.py`. Manifest schema: `specs/manifest.schema.json`.

## 1. A module pack

A folder `library/<module-id>/` with:

| File | Content | Depends on the core |
|---|---|---|
| `manifest.yaml` | identity, signature, files, hashes, cost, status, provenance | — |
| `body.safetensors` | the function itself (written or trained weights) | no, for register, lookup and state functions |
| `trigger.safetensors` | the module's private feature layer and its trigger | yes |
| `binder.safetensors` | the pointer queries, one per argument | yes |

A pack with a body only is valid and portable. It can be loaded for generation once a trigger and a binder have been fitted to the core in use; `fitted_to` in the manifest names that core release.

## 2. What a body sees and does

- Input: the operation name and the argument values, already type-checked against the signature.
- Output: one typed value, written to RES. A state body may also read and write STATE.
- It never sees the text, the core's activations, other modules, or the key/value cache.
- If it cannot produce a result (overflow, missing value), it raises `BodyError`. The call loop then prints nothing and the core's own token is used; at a place where a result is needed that token is `<abstain>`.

## 3. What a trigger and a binder see and do

Inputs: the module's private features of the core's hidden states at the current position, the shared features of the controller block, and CTRL (the number of calls already made at this position and the type held in RES).

- **Trigger**: one score per operation. The module is called when its best score is at or above the threshold in its manifest and no other resident module scores higher.
- **Final**: one more output, read only when the module is called. It says that this call completes the chain. If the last call of a chain is not final, the result is not printed and the model abstains. This is the behaviour when the module that should continue is not loaded.
- **Binder**: one hard pointer per argument. Candidates are the typed values at or before the current position that have the type the signature asks for, plus RES when it holds that type. Soft mixtures of values are not allowed.

## 4. Call loop (per position)

Up to 4 calls. Each call: arbitration → binder → body → RES. Then the emitter prints RES if the last call was final; otherwise the core's token is used. A printed result becomes ordinary text, and the scanner gives it a typed value again.

Every call is recorded: position, step, module, operation, score, pointers, arguments, result, final, error.

## 5. Loading and unloading

- A module may be attached or detached between any two tokens.
- A module never writes to the core's weights or to its key/value cache (`swap.kv_write: false` is enforced by the manifest schema).
- Loading verifies the SHA-256 of every file against the manifest and refuses a damaged pack.
- A module fitted to another core release is refused with a message naming both releases.
- Unloading loses nothing: session state is in the registers and in the text.
- Every load and unload is recorded with its duration and the bytes resident afterwards (`Session.events`).

## 6. Fitting rules (separation)

1. The core is frozen while any module is fitted. Tests check that no core parameter changes.
2. The controller block is fitted once, with the first modules, and then frozen. A module added later brings its own feature layer, trigger and binder and changes nothing that exists.
3. Training examples for a trigger include every call step at every call site, also those of other functions, where it must stay silent.
4. Pretraining of a native core excludes result tokens from the loss and trains `<abstain>` at the position before a result (`flow.data.align.lm_example`).

## 7. Lifecycle

Status values and allowed moves are in `src/flow/registry/manifest.py` (`TRANSITIONS`) and enforced by `python -m flow.registry status …`: proposed → prototype → active ⇄ differentiating / suppressed → eliminated → archived.

## 8. Not covered by this starter

Lookup functions with paged rows, state bodies as weights, delta skills (LoRA surfaces), regulators, fallback from a domain variant to its parent, memory tiers, prefetch and the dwell rule. The manifest schema already has their fields.
