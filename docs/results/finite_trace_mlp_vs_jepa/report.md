# Finite-trace successor prediction: MLP vs JEPA

Complete fixed-protocol run: 24 maps x 3 seeds x 5 budgets, 360 cases, 720 from-scratch fits and 1,080 model-case records. All training data are finite real trace steps. This evaluates unseen transitions within each map, not transfer to previously unseen maps.

Both models receive identical four-channel state images, action indices and observed steps (including duplicates). The MLP is flatten + 8-D action embedding → Linear(152,64) → GELU → Linear(64,|S|), with direct successor CE. JEPA reuses ActionJEPA (32-D latent, original CNN and fixed position subspace, 64-unit predictor), EMA 0.99, existing prediction/variance/covariance losses plus ranking weight 0.1. All known states can be candidates; only observed successors label examples.

Both methods use AdamW, lr 0.0003, weight decay 0.0001, 300 full-batch epochs and final-epoch selection. No pretrained checkpoints, early stopping, oracle repair, heldout selection or hyperparameter search.

Training pools exactly match the original sanity experiment. Each seed/map has 64 independently sampled held-out real 16-step traces, unique and disjoint from all 16 training-pool traces. The pool is fixed across budgets.

## Interpretation of the baselines and denominators

The observed-edge baseline is partial. It abstains on unseen queries; 0% unseen accuracy means no correct unseen predictions, not a neural model selecting wrong successors. Its overall score is edge recall. Its raw held-out acceptance is action-sensitive membership in R_G.

Only its explicitly named observed_self_loop_completion is sent to formal evaluation. Structural and CTL/LTL results for this baseline hold **under this explicit totalization policy**, not for raw R_G. Completion is evaluation-only, not learned inference.

Transition accuracy counts all known S x A once, including unreachable states. Identity metrics also include unreachable states. Reachable and unreachable unseen subsets are reported separately. Macro values are unweighted means over cases; pooled counts and min/max/std are in JSON. Nested budgets, rotations and reused seeds are not independent trials.

## Main comparison: action_insensitive

| Budget | Model | Observed acc | Unseen acc | Overall acc | Held-out trace acceptance | Initial bisim | Identity bisim |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | observed_edge | 100.00% | 0.00% | 9.37% | 0.26% | 24/72 | 0/72 |
| 1 | mlp | 62.43% | 0.88% | 6.60% | 0.17% | 24/72 | 0/72 |
| 1 | jepa | 100.00% | 26.79% | 33.64% | 0.76% | 24/72 | 0/72 |
| 2 | observed_edge | 100.00% | 0.00% | 15.35% | 2.65% | 24/72 | 0/72 |
| 2 | mlp | 42.98% | 0.45% | 6.85% | 0.26% | 24/72 | 0/72 |
| 2 | jepa | 100.00% | 27.83% | 38.87% | 8.01% | 24/72 | 0/72 |
| 4 | observed_edge | 100.00% | 0.00% | 23.34% | 11.74% | 24/72 | 0/72 |
| 4 | mlp | 30.96% | 0.15% | 7.16% | 0.26% | 24/72 | 0/72 |
| 4 | jepa | 100.00% | 28.56% | 45.19% | 21.66% | 24/72 | 0/72 |
| 8 | observed_edge | 100.00% | 0.00% | 32.04% | 36.74% | 24/72 | 0/72 |
| 8 | mlp | 23.97% | 0.05% | 7.47% | 0.67% | 24/72 | 0/72 |
| 8 | jepa | 99.62% | 28.88% | 51.50% | 43.62% | 24/72 | 0/72 |
| 16 | observed_edge | 100.00% | 0.00% | 41.06% | 61.18% | 24/72 | 0/72 |
| 16 | mlp | 18.54% | 0.00% | 7.35% | 0.91% | 24/72 | 0/72 |
| 16 | jepa | 98.14% | 30.07% | 57.92% | 64.67% | 24/72 | 0/72 |

## All primary structural metrics: action_insensitive

| Budget | Model | Initial sim M → model | Initial sim model → M | Initial bisim | Identity sim M → model | Identity sim model → M | Identity bisim |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | observed_edge | 24/72 | 66/72 | 24/72 | 0/72 | 66/72 | 0/72 |
| 1 | mlp | 24/72 | 68/72 | 24/72 | 0/72 | 68/72 | 0/72 |
| 1 | jepa | 24/72 | 67/72 | 24/72 | 0/72 | 67/72 | 0/72 |
| 2 | observed_edge | 24/72 | 60/72 | 24/72 | 0/72 | 60/72 | 0/72 |
| 2 | mlp | 24/72 | 67/72 | 24/72 | 0/72 | 66/72 | 0/72 |
| 2 | jepa | 24/72 | 64/72 | 24/72 | 0/72 | 64/72 | 0/72 |
| 4 | observed_edge | 24/72 | 60/72 | 24/72 | 0/72 | 60/72 | 0/72 |
| 4 | mlp | 24/72 | 68/72 | 24/72 | 0/72 | 67/72 | 0/72 |
| 4 | jepa | 24/72 | 62/72 | 24/72 | 0/72 | 60/72 | 0/72 |
| 8 | observed_edge | 24/72 | 58/72 | 24/72 | 0/72 | 57/72 | 0/72 |
| 8 | mlp | 24/72 | 72/72 | 24/72 | 0/72 | 71/72 | 0/72 |
| 8 | jepa | 24/72 | 60/72 | 24/72 | 0/72 | 57/72 | 0/72 |
| 16 | observed_edge | 24/72 | 55/72 | 24/72 | 0/72 | 52/72 | 0/72 |
| 16 | mlp | 24/72 | 72/72 | 24/72 | 0/72 | 72/72 | 0/72 |
| 16 | jepa | 24/72 | 60/72 | 24/72 | 0/72 | 57/72 | 0/72 |

## Main comparison: action_sensitive

| Budget | Model | Observed acc | Unseen acc | Overall acc | Held-out trace acceptance | Initial bisim | Identity bisim |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | observed_edge | 100.00% | 0.00% | 9.37% | 0.26% | 24/72 | 0/72 |
| 1 | mlp | 62.43% | 0.88% | 6.60% | 0.17% | 24/72 | 0/72 |
| 1 | jepa | 100.00% | 26.79% | 33.64% | 0.76% | 24/72 | 0/72 |
| 2 | observed_edge | 100.00% | 0.00% | 15.35% | 2.65% | 24/72 | 0/72 |
| 2 | mlp | 42.98% | 0.45% | 6.85% | 0.26% | 24/72 | 0/72 |
| 2 | jepa | 100.00% | 27.83% | 38.87% | 8.01% | 24/72 | 0/72 |
| 4 | observed_edge | 100.00% | 0.00% | 23.34% | 11.74% | 24/72 | 0/72 |
| 4 | mlp | 30.96% | 0.15% | 7.16% | 0.26% | 24/72 | 0/72 |
| 4 | jepa | 100.00% | 28.56% | 45.19% | 21.66% | 24/72 | 0/72 |
| 8 | observed_edge | 100.00% | 0.00% | 32.04% | 36.74% | 24/72 | 0/72 |
| 8 | mlp | 23.97% | 0.05% | 7.47% | 0.67% | 24/72 | 0/72 |
| 8 | jepa | 99.62% | 28.88% | 51.50% | 43.62% | 24/72 | 0/72 |
| 16 | observed_edge | 100.00% | 0.00% | 41.06% | 61.18% | 24/72 | 0/72 |
| 16 | mlp | 18.54% | 0.00% | 7.35% | 0.91% | 24/72 | 0/72 |
| 16 | jepa | 98.14% | 30.07% | 57.92% | 64.67% | 24/72 | 0/72 |

## All primary structural metrics: action_sensitive

| Budget | Model | Initial sim M → model | Initial sim model → M | Initial bisim | Identity sim M → model | Identity sim model → M | Identity bisim |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | observed_edge | 24/72 | 24/72 | 24/72 | 0/72 | 0/72 | 0/72 |
| 1 | mlp | 24/72 | 24/72 | 24/72 | 0/72 | 0/72 | 0/72 |
| 1 | jepa | 24/72 | 24/72 | 24/72 | 0/72 | 0/72 | 0/72 |
| 2 | observed_edge | 24/72 | 24/72 | 24/72 | 0/72 | 0/72 | 0/72 |
| 2 | mlp | 24/72 | 24/72 | 24/72 | 0/72 | 0/72 | 0/72 |
| 2 | jepa | 24/72 | 24/72 | 24/72 | 0/72 | 0/72 | 0/72 |
| 4 | observed_edge | 24/72 | 24/72 | 24/72 | 0/72 | 0/72 | 0/72 |
| 4 | mlp | 24/72 | 24/72 | 24/72 | 0/72 | 0/72 | 0/72 |
| 4 | jepa | 24/72 | 24/72 | 24/72 | 0/72 | 0/72 | 0/72 |
| 8 | observed_edge | 24/72 | 24/72 | 24/72 | 0/72 | 0/72 | 0/72 |
| 8 | mlp | 24/72 | 24/72 | 24/72 | 0/72 | 0/72 | 0/72 |
| 8 | jepa | 24/72 | 24/72 | 24/72 | 0/72 | 0/72 | 0/72 |
| 16 | observed_edge | 24/72 | 24/72 | 24/72 | 0/72 | 0/72 | 0/72 |
| 16 | mlp | 24/72 | 24/72 | 24/72 | 0/72 | 0/72 | 0/72 |
| 16 | jepa | 24/72 | 24/72 | 24/72 | 0/72 | 0/72 | 0/72 |

## Additional CTL/LTL diagnostics

The existing six CTL and six universal-path LTL formulas remain diagnostic. LTL verdicts are actually executed with nuXmv on saved predicted graphs; exact exported-model hashes cache identical queries. Every all-state AF goal/F goal and AG !danger/G !danger identity is audited.

| Budget | Model | Initial CTL formula agreement | All-six CTL cases | Initial LTL formula agreement | All-six LTL cases | All-state CTL | All-state LTL |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | observed_edge | 63.89% | 24/72 | 70.83% | 30/72 | 49.85% | 59.06% |
| 1 | mlp | 62.96% | 24/72 | 68.52% | 24/72 | 51.27% | 58.80% |
| 1 | jepa | 63.89% | 24/72 | 70.83% | 30/72 | 49.85% | 59.09% |
| 2 | observed_edge | 68.06% | 24/72 | 77.08% | 39/72 | 50.81% | 60.56% |
| 2 | mlp | 63.43% | 24/72 | 68.98% | 25/72 | 51.76% | 59.47% |
| 2 | jepa | 68.06% | 24/72 | 76.62% | 37/72 | 50.81% | 60.51% |
| 4 | observed_edge | 71.30% | 24/72 | 81.94% | 46/72 | 51.91% | 62.30% |
| 4 | mlp | 63.43% | 24/72 | 69.68% | 27/72 | 51.75% | 59.54% |
| 4 | jepa | 71.30% | 24/72 | 81.94% | 46/72 | 52.02% | 62.49% |
| 8 | observed_edge | 74.54% | 24/72 | 86.81% | 53/72 | 53.64% | 64.98% |
| 8 | mlp | 61.11% | 24/72 | 66.67% | 24/72 | 49.52% | 56.91% |
| 8 | jepa | 74.54% | 24/72 | 86.81% | 53/72 | 53.69% | 65.06% |
| 16 | observed_edge | 75.93% | 25/72 | 87.50% | 54/72 | 55.86% | 67.51% |
| 16 | mlp | 61.11% | 24/72 | 66.67% | 24/72 | 49.46% | 56.81% |
| 16 | jepa | 75.46% | 25/72 | 87.50% | 54/72 | 55.26% | 66.91% |

## Unseen accuracy paired comparison

| Budget | JEPA minus MLP (percentage points) | JEPA wins | MLP wins | Ties |
|---:|---:|---:|---:|---:|
| 1 | 25.91 | 72 | 0 | 0 |
| 2 | 27.37 | 72 | 0 | 0 |
| 4 | 28.41 | 72 | 0 | 0 |
| 8 | 28.83 | 72 | 0 | 0 |
| 16 | 30.07 | 72 | 0 | 0 |

## Reachability and held-out difficulty

| Budget | Model | Reachable unseen acc | Unreachable unseen acc | Held-out containing unseen pairs | Acceptance among those traces | Observed-unseen gap |
|---:|---|---:|---:|---:|---:|---:|
| 1 | observed_edge | 0.00% | 0.00% | 4596/4608 | 0.00% | 100.00 pp |
| 1 | mlp | 1.44% | 0.00% | 4596/4608 | 0.09% | 61.54 pp |
| 1 | jepa | 26.49% | 29.54% | 4596/4608 | 0.50% | 73.21 pp |
| 2 | observed_edge | 0.00% | 0.00% | 4486/4608 | 0.00% | 100.00 pp |
| 2 | mlp | 0.70% | 0.00% | 4486/4608 | 0.00% | 42.53 pp |
| 2 | jepa | 28.39% | 29.60% | 4486/4608 | 5.63% | 72.17 pp |
| 4 | observed_edge | 0.00% | 0.00% | 4067/4608 | 0.00% | 100.00 pp |
| 4 | mlp | 0.21% | 0.00% | 4067/4608 | 0.03% | 30.81 pp |
| 4 | jepa | 29.50% | 30.12% | 4067/4608 | 11.49% | 71.44 pp |
| 8 | observed_edge | 0.00% | 0.00% | 2915/4608 | 0.00% | 100.00 pp |
| 8 | mlp | 0.05% | 0.00% | 2915/4608 | 0.03% | 23.92 pp |
| 8 | jepa | 29.84% | 29.75% | 2915/4608 | 11.80% | 70.74 pp |
| 16 | observed_edge | 0.00% | 0.00% | 1789/4608 | 0.00% | 100.00 pp |
| 16 | mlp | 0.00% | 0.00% | 1789/4608 | 0.00% | 18.54 pp |
| 16 | jepa | 32.52% | 29.72% | 1789/4608 | 15.23% | 68.07 pp |

## by_seed

| Group | Budget | Model | Observed acc | Unseen acc | Overall acc | Held-out acceptance | Initial bisim (ignore / respect actions) | Identity bisim (ignore / respect actions) |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 20260804 | 1 | observed_edge | 100.00% | 0.00% | 9.52% | 0.39% | 8/24 / 8/24 | 0/24 / 0/24 |
| 20260804 | 1 | mlp | 61.74% | 0.82% | 6.67% | 0.07% | 8/24 / 8/24 | 0/24 / 0/24 |
| 20260804 | 1 | jepa | 100.00% | 27.11% | 34.04% | 0.85% | 8/24 / 8/24 | 0/24 / 0/24 |
| 20260804 | 2 | observed_edge | 100.00% | 0.00% | 16.24% | 2.41% | 8/24 / 8/24 | 0/24 / 0/24 |
| 20260804 | 2 | mlp | 42.02% | 0.57% | 7.21% | 0.20% | 8/24 / 8/24 | 0/24 / 0/24 |
| 20260804 | 2 | jepa | 100.00% | 28.46% | 40.03% | 7.16% | 8/24 / 8/24 | 0/24 / 0/24 |
| 20260804 | 4 | observed_edge | 100.00% | 0.00% | 23.97% | 13.15% | 8/24 / 8/24 | 0/24 / 0/24 |
| 20260804 | 4 | mlp | 30.41% | 0.14% | 7.24% | 0.26% | 8/24 / 8/24 | 0/24 / 0/24 |
| 20260804 | 4 | jepa | 100.00% | 28.90% | 45.91% | 22.40% | 8/24 / 8/24 | 0/24 / 0/24 |
| 20260804 | 8 | observed_edge | 100.00% | 0.00% | 32.81% | 38.28% | 8/24 / 8/24 | 0/24 / 0/24 |
| 20260804 | 8 | mlp | 23.50% | 0.05% | 7.63% | 0.52% | 8/24 / 8/24 | 0/24 / 0/24 |
| 20260804 | 8 | jepa | 99.72% | 29.10% | 52.26% | 43.29% | 8/24 / 8/24 | 0/24 / 0/24 |
| 20260804 | 16 | observed_edge | 100.00% | 0.00% | 41.75% | 61.52% | 8/24 / 8/24 | 0/24 / 0/24 |
| 20260804 | 16 | mlp | 17.78% | 0.00% | 7.34% | 0.65% | 8/24 / 8/24 | 0/24 / 0/24 |
| 20260804 | 16 | jepa | 98.44% | 29.93% | 58.51% | 64.26% | 8/24 / 8/24 | 0/24 / 0/24 |
| 20260805 | 1 | observed_edge | 100.00% | 0.00% | 9.39% | 0.13% | 8/24 / 8/24 | 0/24 / 0/24 |
| 20260805 | 1 | mlp | 62.51% | 1.01% | 6.71% | 0.26% | 8/24 / 8/24 | 0/24 / 0/24 |
| 20260805 | 1 | jepa | 100.00% | 26.65% | 33.53% | 0.52% | 8/24 / 8/24 | 0/24 / 0/24 |
| 20260805 | 2 | observed_edge | 100.00% | 0.00% | 15.34% | 2.02% | 8/24 / 8/24 | 0/24 / 0/24 |
| 20260805 | 2 | mlp | 46.00% | 0.39% | 7.15% | 0.07% | 8/24 / 8/24 | 0/24 / 0/24 |
| 20260805 | 2 | jepa | 100.00% | 27.92% | 38.94% | 7.29% | 8/24 / 8/24 | 0/24 / 0/24 |
| 20260805 | 4 | observed_edge | 100.00% | 0.00% | 23.12% | 8.72% | 8/24 / 8/24 | 0/24 / 0/24 |
| 20260805 | 4 | mlp | 30.85% | 0.17% | 6.96% | 0.00% | 8/24 / 8/24 | 0/24 / 0/24 |
| 20260805 | 4 | jepa | 100.00% | 28.35% | 44.85% | 17.38% | 8/24 / 8/24 | 0/24 / 0/24 |
| 20260805 | 8 | observed_edge | 100.00% | 0.00% | 31.58% | 34.83% | 8/24 / 8/24 | 0/24 / 0/24 |
| 20260805 | 8 | mlp | 23.81% | 0.00% | 7.20% | 0.59% | 8/24 / 8/24 | 0/24 / 0/24 |
| 20260805 | 8 | jepa | 99.59% | 29.08% | 51.29% | 43.36% | 8/24 / 8/24 | 0/24 / 0/24 |
| 20260805 | 16 | observed_edge | 100.00% | 0.00% | 40.74% | 60.61% | 8/24 / 8/24 | 0/24 / 0/24 |
| 20260805 | 16 | mlp | 19.36% | 0.00% | 7.44% | 0.98% | 8/24 / 8/24 | 0/24 / 0/24 |
| 20260805 | 16 | jepa | 98.45% | 30.54% | 57.98% | 65.82% | 8/24 / 8/24 | 0/24 / 0/24 |
| 20260806 | 1 | observed_edge | 100.00% | 0.00% | 9.21% | 0.26% | 8/24 / 8/24 | 0/24 / 0/24 |
| 20260806 | 1 | mlp | 63.03% | 0.82% | 6.42% | 0.20% | 8/24 / 8/24 | 0/24 / 0/24 |
| 20260806 | 1 | jepa | 100.00% | 26.60% | 33.36% | 0.91% | 8/24 / 8/24 | 0/24 / 0/24 |
| 20260806 | 2 | observed_edge | 100.00% | 0.00% | 14.46% | 3.52% | 8/24 / 8/24 | 0/24 / 0/24 |
| 20260806 | 2 | mlp | 40.91% | 0.39% | 6.21% | 0.52% | 8/24 / 8/24 | 0/24 / 0/24 |
| 20260806 | 2 | jepa | 100.00% | 27.10% | 37.64% | 9.57% | 8/24 / 8/24 | 0/24 / 0/24 |
| 20260806 | 4 | observed_edge | 100.00% | 0.00% | 22.92% | 13.35% | 8/24 / 8/24 | 0/24 / 0/24 |
| 20260806 | 4 | mlp | 31.62% | 0.13% | 7.28% | 0.52% | 8/24 / 8/24 | 0/24 / 0/24 |
| 20260806 | 4 | jepa | 100.00% | 28.43% | 44.80% | 25.20% | 8/24 / 8/24 | 0/24 / 0/24 |
| 20260806 | 8 | observed_edge | 100.00% | 0.00% | 31.74% | 37.11% | 8/24 / 8/24 | 0/24 / 0/24 |
| 20260806 | 8 | mlp | 24.60% | 0.10% | 7.59% | 0.91% | 8/24 / 8/24 | 0/24 / 0/24 |
| 20260806 | 8 | jepa | 99.55% | 28.45% | 50.96% | 44.21% | 8/24 / 8/24 | 0/24 / 0/24 |
| 20260806 | 16 | observed_edge | 100.00% | 0.00% | 40.69% | 61.39% | 8/24 / 8/24 | 0/24 / 0/24 |
| 20260806 | 16 | mlp | 18.47% | 0.00% | 7.27% | 1.11% | 8/24 / 8/24 | 0/24 / 0/24 |
| 20260806 | 16 | jepa | 97.53% | 29.75% | 57.29% | 63.93% | 8/24 / 8/24 | 0/24 / 0/24 |

## by_family

| Group | Budget | Model | Observed acc | Unseen acc | Overall acc | Held-out acceptance | Initial bisim (ignore / respect actions) | Identity bisim (ignore / respect actions) |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| danger_gate | 1 | observed_edge | 100.00% | 0.00% | 9.61% | 0.33% | 0/24 / 0/24 | 0/24 / 0/24 |
| danger_gate | 1 | mlp | 63.29% | 0.86% | 6.82% | 0.00% | 0/24 / 0/24 | 0/24 / 0/24 |
| danger_gate | 1 | jepa | 100.00% | 26.95% | 33.97% | 0.91% | 0/24 / 0/24 | 0/24 / 0/24 |
| danger_gate | 2 | observed_edge | 100.00% | 0.00% | 15.62% | 2.93% | 0/24 / 0/24 | 0/24 / 0/24 |
| danger_gate | 2 | mlp | 42.98% | 0.36% | 6.96% | 0.39% | 0/24 / 0/24 | 0/24 / 0/24 |
| danger_gate | 2 | jepa | 100.00% | 27.58% | 38.88% | 7.62% | 0/24 / 0/24 | 0/24 / 0/24 |
| danger_gate | 4 | observed_edge | 100.00% | 0.00% | 23.42% | 13.80% | 0/24 / 0/24 | 0/24 / 0/24 |
| danger_gate | 4 | mlp | 28.32% | 0.04% | 6.52% | 0.39% | 0/24 / 0/24 | 0/24 / 0/24 |
| danger_gate | 4 | jepa | 100.00% | 28.76% | 45.40% | 22.66% | 0/24 / 0/24 | 0/24 / 0/24 |
| danger_gate | 8 | observed_edge | 100.00% | 0.00% | 32.43% | 33.14% | 0/24 / 0/24 | 0/24 / 0/24 |
| danger_gate | 8 | mlp | 22.31% | 0.16% | 7.16% | 0.46% | 0/24 / 0/24 | 0/24 / 0/24 |
| danger_gate | 8 | jepa | 99.50% | 28.77% | 51.68% | 39.84% | 0/24 / 0/24 | 0/24 / 0/24 |
| danger_gate | 16 | observed_edge | 100.00% | 0.00% | 41.73% | 56.45% | 0/24 / 0/24 | 0/24 / 0/24 |
| danger_gate | 16 | mlp | 17.55% | 0.00% | 7.26% | 0.91% | 0/24 / 0/24 | 0/24 / 0/24 |
| danger_gate | 16 | jepa | 98.28% | 30.33% | 58.67% | 61.13% | 0/24 / 0/24 | 0/24 / 0/24 |
| safe_detour | 1 | observed_edge | 100.00% | 0.00% | 8.85% | 0.39% | 0/24 / 0/24 | 0/24 / 0/24 |
| safe_detour | 1 | mlp | 61.45% | 0.71% | 5.96% | 0.39% | 0/24 / 0/24 | 0/24 / 0/24 |
| safe_detour | 1 | jepa | 100.00% | 25.58% | 32.16% | 0.78% | 0/24 / 0/24 | 0/24 / 0/24 |
| safe_detour | 2 | observed_edge | 100.00% | 0.00% | 15.14% | 2.21% | 0/24 / 0/24 | 0/24 / 0/24 |
| safe_detour | 2 | mlp | 42.64% | 0.54% | 6.80% | 0.33% | 0/24 / 0/24 | 0/24 / 0/24 |
| safe_detour | 2 | jepa | 100.00% | 26.59% | 37.70% | 5.99% | 0/24 / 0/24 | 0/24 / 0/24 |
| safe_detour | 4 | observed_edge | 100.00% | 0.00% | 23.96% | 10.74% | 0/24 / 0/24 | 0/24 / 0/24 |
| safe_detour | 4 | mlp | 30.76% | 0.30% | 7.49% | 0.26% | 0/24 / 0/24 | 0/24 / 0/24 |
| safe_detour | 4 | jepa | 100.00% | 26.83% | 44.34% | 17.71% | 0/24 / 0/24 | 0/24 / 0/24 |
| safe_detour | 8 | observed_edge | 100.00% | 0.00% | 33.46% | 33.79% | 0/24 / 0/24 | 0/24 / 0/24 |
| safe_detour | 8 | mlp | 22.99% | 0.00% | 7.55% | 0.91% | 0/24 / 0/24 | 0/24 / 0/24 |
| safe_detour | 8 | jepa | 99.61% | 27.97% | 51.89% | 39.71% | 0/24 / 0/24 | 0/24 / 0/24 |
| safe_detour | 16 | observed_edge | 100.00% | 0.00% | 43.42% | 59.57% | 0/24 / 0/24 | 0/24 / 0/24 |
| safe_detour | 16 | mlp | 15.87% | 0.00% | 6.74% | 0.85% | 0/24 / 0/24 | 0/24 / 0/24 |
| safe_detour | 16 | jepa | 97.09% | 29.64% | 58.85% | 60.68% | 0/24 / 0/24 | 0/24 / 0/24 |
| sealed_region | 1 | observed_edge | 100.00% | 0.00% | 9.65% | 0.07% | 24/24 / 24/24 | 0/24 / 0/24 |
| sealed_region | 1 | mlp | 62.55% | 1.08% | 7.01% | 0.13% | 24/24 / 24/24 | 0/24 / 0/24 |
| sealed_region | 1 | jepa | 100.00% | 27.83% | 34.79% | 0.59% | 24/24 / 24/24 | 0/24 / 0/24 |
| sealed_region | 2 | observed_edge | 100.00% | 0.00% | 15.28% | 2.80% | 24/24 / 24/24 | 0/24 / 0/24 |
| sealed_region | 2 | mlp | 43.31% | 0.46% | 6.81% | 0.07% | 24/24 / 24/24 | 0/24 / 0/24 |
| sealed_region | 2 | jepa | 100.00% | 29.30% | 40.03% | 10.42% | 24/24 / 24/24 | 0/24 / 0/24 |
| sealed_region | 4 | observed_edge | 100.00% | 0.00% | 22.64% | 10.68% | 24/24 / 24/24 | 0/24 / 0/24 |
| sealed_region | 4 | mlp | 33.79% | 0.10% | 7.47% | 0.13% | 24/24 / 24/24 | 0/24 / 0/24 |
| sealed_region | 4 | jepa | 100.00% | 30.08% | 45.83% | 24.61% | 24/24 / 24/24 | 0/24 / 0/24 |
| sealed_region | 8 | observed_edge | 100.00% | 0.00% | 30.24% | 43.29% | 24/24 / 24/24 | 0/24 / 0/24 |
| sealed_region | 8 | mlp | 26.60% | 0.00% | 7.71% | 0.65% | 24/24 / 24/24 | 0/24 / 0/24 |
| sealed_region | 8 | jepa | 99.75% | 29.89% | 50.94% | 51.30% | 24/24 / 24/24 | 0/24 / 0/24 |
| sealed_region | 16 | observed_edge | 100.00% | 0.00% | 38.02% | 67.51% | 24/24 / 24/24 | 0/24 / 0/24 |
| sealed_region | 16 | mlp | 22.19% | 0.00% | 8.06% | 0.98% | 24/24 / 24/24 | 0/24 / 0/24 |
| sealed_region | 16 | jepa | 99.06% | 30.24% | 56.25% | 72.20% | 24/24 / 24/24 | 0/24 / 0/24 |

## Reproduce

```bash
PYTHONPATH=src NUXMV_BINARY=/path/to/nuXmv \
  python -m experiments.finite_trace_mlp_vs_jepa \
  --output-dir outputs/finite_trace_mlp_vs_jepa/new_run
```

The destination must not exist. experiment_config.json and run_started.json lock the protocol, sources and prior result hashes. cases.jsonl records all predictions, training-data/parameter hashes, observed/unseen scores, held-out failure steps/edges and both structural semantics. trace_pools.jsonl saves exact training and held-out pools. ltl_backend.jsonl.gz saves all unique symbolic exports and actual backend output/verdicts. Predictions permit formal replay without retraining; model weights can be reproduced from the recorded config/seeds/source/runtime, with no checkpoint search.

This is a comparison of these specified architectures and this fixed optimization budget. It does not establish superiority over every simple supervised predictor or identify an optimal learning rule.

## Evaluator-only unseen edge-type diagnostic

Accuracies use all unseen pairs, including unreachable states. Moving edges have a real successor different from their source. The self-loop completion is an **evaluation-only context**, not learned inference or the raw partial observed-edge baseline. Its moving-edge accuracy is 0% by construction.

| Budget | Model / evaluation policy | Unseen accuracy | True self-loop unseen accuracy | Moving unseen accuracy |
|---:|---|---:|---:|---:|
| 1 | observed_edge | 0.00% | 0.00% | 0.00% |
| 1 | mlp | 0.88% | 1.22% | 0.75% |
| 1 | jepa | 26.79% | 94.64% | 1.14% |
| 1 | observed_self_loop_completion_evaluation_only | 27.47% | 100.00% | 0.00% |
| 2 | observed_edge | 0.00% | 0.00% | 0.00% |
| 2 | mlp | 0.45% | 0.60% | 0.38% |
| 2 | jepa | 27.83% | 96.16% | 1.40% |
| 2 | observed_self_loop_completion_evaluation_only | 27.91% | 100.00% | 0.00% |
| 4 | observed_edge | 0.00% | 0.00% | 0.00% |
| 4 | mlp | 0.15% | 0.22% | 0.12% |
| 4 | jepa | 28.56% | 96.71% | 2.03% |
| 4 | observed_self_loop_completion_evaluation_only | 28.05% | 100.00% | 0.00% |
| 8 | observed_edge | 0.00% | 0.00% | 0.00% |
| 8 | mlp | 0.05% | 0.06% | 0.05% |
| 8 | jepa | 28.88% | 96.86% | 1.92% |
| 8 | observed_self_loop_completion_evaluation_only | 28.42% | 100.00% | 0.00% |
| 16 | observed_edge | 0.00% | 0.00% | 0.00% |
| 16 | mlp | 0.00% | 0.00% | 0.00% |
| 16 | jepa | 30.07% | 97.26% | 2.05% |
| 16 | observed_self_loop_completion_evaluation_only | 29.43% | 100.00% | 0.00% |
