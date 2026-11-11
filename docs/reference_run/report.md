# Toy walkthrough report

Core: 523,816 parameters, layer passes LFFF, pretrained from random weights in 383.4 s with answer masking. Learned on top: controller_block 224,133, heads.add_exact 127,241, heads.compare_numeric 127,370.

| Set | Exact | Abstained | Wrong | n |
|---|---|---|---|---|
| Core alone, test | 0.000 | 556 | 0 | 556 |
| Both modules, test | 1.000 | 0 | 0 | 556 |
| Both modules, numbers of 8-11 digits (training: 1-6) | 1.000 | 0 | 0 | 555 |
| Both modules, pair of calls never shown in training (subtract, then greater) | 1.000 | 0 | 0 | 274 |

Calls at positions where none belongs: 0 in 18,850 positions.

## Separation

| Module unloaded | Function | Exact before | Exact after | Localization L |
|---|---|---|---|---|
| fn.add_exact.v1 | add_exact | 1.000 | 0.000 | 1.0 |
| fn.compare_numeric.v1 | compare_numeric | 1.000 | 0.000 | 1.0 |

## Per template

| Template | Test | Long numbers |
|---|---|---|
| add-1 | 1.000 | 1.000 |
| add-2 | 1.000 | 1.000 |
| add-3 | 1.000 | 1.000 |
| add-4 | 1.000 | 1.000 |
| add-5 | 1.000 | 1.000 |
| add>add | 1.000 | 1.000 |
| add>eq | 1.000 | 1.000 |
| add>gt | 1.000 | 1.000 |
| add>lt | 1.000 | 1.000 |
| add>sub | 1.000 | 1.000 |
| all | 1.000 | 1.000 |
| eq-1 | 1.000 | 1.000 |
| found>eq | 1.000 | 1.000 |
| found>gt | 1.000 | 1.000 |
| found>lt | 1.000 | 1.000 |
| gt-1 | 1.000 | 1.000 |
| gt-2 | 1.000 | 1.000 |
| lt-1 | 1.000 | 1.000 |
| lt-2 | 1.000 | 1.000 |
| sub-1 | 1.000 | 1.000 |
| sub-2 | 1.000 | 1.000 |
| sub-3 | 1.000 | 1.000 |
| sub-4 | 1.000 | 1.000 |
| sub>add | 1.000 | 1.000 |
| sub>eq | 1.000 | 1.000 |
| sub>lt | 1.000 | 1.000 |

## Dynamic loading

- loaded nothing: `mara has 417 coins and gets 85 more . now mara has <abstain>`
- loaded ['fn.add_exact.v1']: `mara has 417 coins and gets 85 more . now mara has 502 .`
- loaded ['fn.add_exact.v1']: `ines has 90 shells and gets 25 more . bruno has 170 . does ines have more than bruno ? <abstain>`
- loaded ['fn.add_exact.v1', 'fn.compare_numeric.v1']: `ines has 90 shells and gets 25 more . bruno has 170 . does ines have more than bruno ? no .`
