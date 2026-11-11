# Claims

What Flow claims, how each claim is tested, and the result that would refute it.
Source: pipeline draft v0.3, sections 1 and 5; spec v0.2, section 9. Targets are design values
until they are fixed at gate G2.

## The six properties

| # | Claim | Test | Metric | Refuted when |
|---|---|---|---|---|
| C1 | **Separate.** Each function is one pack of files. The core never changes when the library does | The core's hash is identical before and after every library change | Registry audit | The hash changes |
| C2 | **Localized.** A function lives in its module and nowhere else | Remove the module: its own suite falls to the floor, the others do not move | M12 ≥ 0.95, M4 ≤ 1 point | M12 below 0.90: the function is not modular; rebuild it or declare it part of the core |
| C3 | **Composable.** Modules built apart work together, also in combinations never trained | Sealed held-out combinations | M1 ≥ 5 points above the matched mixture of experts | No gain over the matched baselines |
| C4 | **Extensible.** A function is added without retraining anything else | Add held-back functions | M3 ≤ 1 point change on old suites | Old suites move by more than 1 point |
| C5 | **Loadable.** Modules load and unload during generation | Swap tests, residency profile | M5, M11 (equivalence, zero stalls, ≤ 1 point cost) | Any stall, or more than 1 point lost |
| C6 | **Developmental.** The library differentiates, suppresses and eliminates safely | Lifecycle run | M10 ≤ 1 point change after elimination | Shift suites move by more than 1 point |

## The two claims behind them

| # | Claim | Test | Refuted when |
|---|---|---|---|
| C7 | Exact functions written as weights are exact, not approximately right | Exhaustive and property tests of every body (M9 = 100%) | One failing case |
| C8 | Capability grows without the active network growing in proportion | Capability per active parameter, per FLOP and per joule against all baselines (M2), including the scratchpad baseline S10.5 | A resource-matched conventional model does as well |

## Localization index (M12)

    L_f = (accuracy with module f − accuracy without it) ÷ (accuracy with module f − floor)

The floor is the accuracy of an answer that uses no computation (abstaining, or the most frequent answer).
L_f = 1: the function lives only in its module. L_f = 0: the core does it alone.

## What the starter kit already shows, in miniature

`python scripts/toy_walkthrough.py` measures C1, C2, C4, C5 and C7 on a 0.5M-parameter blank core
with two modules (`docs/reference_run/report.md`). It says nothing about C3 at scale, C6 or C8:
those need the full pipeline and its baselines.

## What is not claimed

- That a blank network of the size of a published model can be trained on this budget.
- That the toy results carry over to natural text. The toy text is generated from templates.
- Any fixed compression ratio. Reductions of orders of magnitude are an aspiration to test.
