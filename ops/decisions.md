# Decision log

One entry per decision. Gate decisions (G1–G4) are yours; record them with
`python ops/next_steps.py --gate G1 go --note "..."` and add the reasoning here.

## Open: needed at gate G1

| # | Decision | Default proposed | Where it matters |
|---|---|---|---|
| D-001 | Example domain | Clinical dosing text and laboratory units | S2 data, domain variants in S9.2 |
| D-002 | Borrowed core for the skeleton | Qwen3.5-0.8B for development, Qwen3.5-4B as the target (docs/MODEL_CHOICE.md) | S3, S13 |
| D-003 | Compute | One rented 24 GB GPU; a spending cap set by the project owner | ops/budget.md |
| D-004 | Teacher model for distilled data | An open-weight model of the same family, run on the rented GPU | S2.4, S5D |
| D-005 | Spec amendments | The nine items in section 9 of the pipeline draft, plus the three starter findings below | spec v0.3 |
| D-006 | License of this project's own code and module packs | Apache-2.0, the license of Qwen3.5 and of Transformers (`PROJECT_LICENSE` in src/flow/runtime/minimal.py) | every manifest |

## Made while building the starter (for your confirmation at G1)

| # | Decision | Reason |
|---|---|---|
| S-001 | The native core uses the Qwen3.5 text architecture, scaled down, instead of custom layer code | The published architecture already has the listed ingredients (gated attention, causal convolution with kernel 4, rotary positions on a quarter of each head, constant-state linear attention). Maintained code under Apache-2.0; the borrowed and the native core share one interface. 34.7M parameters with the defaults in configs/core_v0.yaml |
| S-002 | Loop passes are separate layer objects that share parameters | Every pass keeps its own cache slot; the weights are stored once |
| S-003 | Code lives in `src/flow/…`; the type system package is `typesys` | A top-level folder named `types` hides Python's standard module of that name |
| S-004 | Every trigger has a "final" output; a chain whose last call is not final is dropped | Without it, unloading the module that completes a chain printed the intermediate result. With it, the model abstains (measured in the toy run) |
| S-005 | Trigger training weighs call sites as much as all plain positions together | Otherwise the few hard negatives (the right place, the wrong operation) are ignored and chains of different operations fail (measured: 3% → 100% exact on mixed chains in the toy run) |
| S-006 | The body of add_exact has 4.53M weights, 144,046 of them non-zero, not the 0.61M of the illustration in Annex C | Every logic unit is a restoring pair, so results stay exact in 16-bit floats over chained calls. A sparse storage format would cut the 9 MB file to under 1 MB (S5A.3) |

## Log

| Date | Decision | By | Notes |
|---|---|---|---|
| | | | |
