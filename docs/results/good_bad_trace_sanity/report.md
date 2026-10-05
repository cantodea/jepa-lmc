# Good / bad trace formulation: sanity experiment

Known S, I, A, L; hidden R; visible state identities. No JEPA training or inference.

The fixed October benchmark has 24 maps, seeds 20260804/20260805/20260806, 16-step trajectories and nested budgets 1/2/4/8/16. Each budget has 72 cases.

## Partial-relation sample results

Coverage and recall use the entire action-labelled R, including unreachable states. Means are unweighted over the 72 seed-map cases. Trajectory counts are observation budgets, not transition coverage. Bisimulation columns below refer ONLY to the separately completed total model.

### action_insensitive

| Trajectory budget | Mean transition coverage | Observed-edge good acceptance | Observed-edge bad rejection | Observed-edge transition recall | Self-loop completion initial bisimulation | Complete-model bad rejection |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.37% | 100.00% | 100.00% | 9.37% | 24/72 (33.33%) | 0.00% |
| 2 | 15.35% | 100.00% | 100.00% | 15.35% | 24/72 (33.33%) | 0.00% |
| 4 | 23.34% | 100.00% | 100.00% | 23.34% | 24/72 (33.33%) | 0.00% |
| 8 | 32.04% | 100.00% | 100.00% | 32.04% | 24/72 (33.33%) | 0.00% |
| 16 | 41.06% | 100.00% | 100.00% | 41.06% | 24/72 (33.33%) | 0.00% |

### action_sensitive

| Trajectory budget | Mean transition coverage | Observed-edge good acceptance | Observed-edge bad rejection | Observed-edge transition recall | Self-loop completion initial bisimulation | Complete-model bad rejection |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.37% | 100.00% | 100.00% | 9.37% | 24/72 (33.33%) | 0.00% |
| 2 | 15.35% | 100.00% | 100.00% | 15.35% | 24/72 (33.33%) | 0.00% |
| 4 | 23.34% | 100.00% | 100.00% | 23.34% | 24/72 (33.33%) | 0.00% |
| 8 | 32.04% | 100.00% | 100.00% | 32.04% | 24/72 (33.33%) | 0.00% |
| 16 | 41.06% | 100.00% | 100.00% | 41.06% | 24/72 (33.33%) | 0.00% |

| Budget | Observed-edge precision min/max | Complete-model good acceptance min/max |
|---:|---:|---:|
| 1 | 100.00% / 100.00% | 100.00% / 100.00% |
| 2 | 100.00% / 100.00% | 100.00% / 100.00% |
| 4 | 100.00% / 100.00% | 100.00% / 100.00% |
| 8 | 100.00% / 100.00% | 100.00% / 100.00% |
| 16 | 100.00% / 100.00% | 100.00% / 100.00% |

| Budget | Observed steps per case | Mean unique (s,a) | Mean unique triples | Recall min-max |
|---:|---:|---:|---:|---:|
| 1 | 16 | 11.61 | 11.61 | 5.00%-12.10% |
| 2 | 32 | 19.03 | 19.03 | 9.17%-22.50% |
| 4 | 64 | 28.96 | 28.96 | 12.50%-36.29% |
| 8 | 128 | 39.78 | 39.78 | 17.50%-43.75% |
| 16 | 256 | 50.99 | 50.99 | 21.67%-57.81% |

## Formal evaluation of total models

The observed-edge relation is partial.
Self-loop completion is introduced only to obtain a total transition system
compatible with the existing formal evaluation pipeline.

For observed (s,a), retain its observed successor; for every unobserved (s,a), insert (s,a,s). All conclusions in this section hold **under this explicit totalization policy**. No CTL or bisimulation claim is assigned to partial R_G.

The unchanged evaluate_model_pair calls the existing behavioral_relations fixed points, audits the greatest relations and cross-checks bisimulation with existing partition refinement. CTL uses the unchanged six-formula native diagnostic. LTL is not run.

### action_insensitive

| Model | Budget | Initial sim M → model | Initial sim model → M | Initial bisimulation | Identity bisimulation |
|---|---:|---:|---:|---:|---:|
| observed_self_loop_completion | 1 | 24/72 | 66/72 | 24/72 | 0/72 |
| observed_self_loop_completion | 2 | 24/72 | 60/72 | 24/72 | 0/72 |
| observed_self_loop_completion | 4 | 24/72 | 60/72 | 24/72 | 0/72 |
| observed_self_loop_completion | 8 | 24/72 | 58/72 | 24/72 | 0/72 |
| observed_self_loop_completion | 16 | 24/72 | 55/72 | 24/72 | 0/72 |
| maximally_permissive | all | 72/72 | 0/72 | 0/72 | 0/72 |
| real_reference | all | 72/72 | 72/72 | 72/72 | 72/72 |

### action_sensitive

| Model | Budget | Initial sim M → model | Initial sim model → M | Initial bisimulation | Identity bisimulation |
|---|---:|---:|---:|---:|---:|
| observed_self_loop_completion | 1 | 24/72 | 24/72 | 24/72 | 0/72 |
| observed_self_loop_completion | 2 | 24/72 | 24/72 | 24/72 | 0/72 |
| observed_self_loop_completion | 4 | 24/72 | 24/72 | 24/72 | 0/72 |
| observed_self_loop_completion | 8 | 24/72 | 24/72 | 24/72 | 0/72 |
| observed_self_loop_completion | 16 | 24/72 | 24/72 | 24/72 | 0/72 |
| maximally_permissive | all | 72/72 | 0/72 | 0/72 | 0/72 |
| real_reference | all | 72/72 | 72/72 | 72/72 | 72/72 |

## Six-formula CTL diagnostic

Fractions below mean all six initial verdicts agree with M. Full per-formula verdicts, all-state counts and structural mismatch states are in cases.jsonl.

| Budget | Self-loop completion | Maximally permissive | Real reference |
|---:|---:|---:|---:|
| 1 | 24/72 | 24/72 | 72/72 |
| 2 | 24/72 | 24/72 | 72/72 |
| 4 | 24/72 | 24/72 | 72/72 |
| 8 | 24/72 | 24/72 | 72/72 |
| 16 | 25/72 | 24/72 | 72/72 |

## Seed and family breakdown

| Seed | Budget | Mean transition recall | Initial bisimulation (ignore / respect actions) |
|---:|---:|---:|---:|
| 20260804 | 1 | 9.52% | 8/24 / 8/24 |
| 20260804 | 2 | 16.24% | 8/24 / 8/24 |
| 20260804 | 4 | 23.97% | 8/24 / 8/24 |
| 20260804 | 8 | 32.81% | 8/24 / 8/24 |
| 20260804 | 16 | 41.75% | 8/24 / 8/24 |
| 20260805 | 1 | 9.39% | 8/24 / 8/24 |
| 20260805 | 2 | 15.34% | 8/24 / 8/24 |
| 20260805 | 4 | 23.12% | 8/24 / 8/24 |
| 20260805 | 8 | 31.58% | 8/24 / 8/24 |
| 20260805 | 16 | 40.74% | 8/24 / 8/24 |
| 20260806 | 1 | 9.21% | 8/24 / 8/24 |
| 20260806 | 2 | 14.46% | 8/24 / 8/24 |
| 20260806 | 4 | 22.92% | 8/24 / 8/24 |
| 20260806 | 8 | 31.74% | 8/24 / 8/24 |
| 20260806 | 16 | 40.69% | 8/24 / 8/24 |

| Family | Budget | Mean transition recall | Initial bisimulation (ignore / respect actions) |
|---|---:|---:|---:|
| danger_gate | 1 | 9.61% | 0/24 / 0/24 |
| danger_gate | 2 | 15.62% | 0/24 / 0/24 |
| danger_gate | 4 | 23.42% | 0/24 / 0/24 |
| danger_gate | 8 | 32.43% | 0/24 / 0/24 |
| danger_gate | 16 | 41.73% | 0/24 / 0/24 |
| safe_detour | 1 | 8.85% | 0/24 / 0/24 |
| safe_detour | 2 | 15.14% | 0/24 / 0/24 |
| safe_detour | 4 | 23.96% | 0/24 / 0/24 |
| safe_detour | 8 | 33.46% | 0/24 / 0/24 |
| safe_detour | 16 | 43.42% | 0/24 / 0/24 |
| sealed_region | 1 | 9.65% | 24/24 / 24/24 |
| sealed_region | 2 | 15.28% | 24/24 / 24/24 |
| sealed_region | 4 | 22.64% | 24/24 / 24/24 |
| sealed_region | 8 | 30.24% | 24/24 / 24/24 |
| sealed_region | 16 | 38.02% | 24/24 / 24/24 |

## What this establishes

Expected perfect separation with incomplete R recovery: 360/360 cases.

This is also a direct mathematical consequence: every good execution uses only edges in R, hence R_G is a subset of R. Every good trace is accepted by construction. Every R-invalid negative with the designated initial contains an edge outside R and therefore outside R_G. No bad traces need to be consulted to fit this separator. The result is stronger than merely rejecting the sampled negatives.

R_G is minimal by edge inclusion for accepting these identified good traces; it can still recombine observed edges into additional traces. It does not reconstruct unseen transitions. This uses partial hypotheses. If totality is required of the learner, R_G is not itself an admissible total solution, and completion may invalidate sample separation. We make no such claim about the completed model.

In sealed_region, the initial reachable component has only safe labels; both real and completed systems are action-total on a safe component. They can be initially bisimilar despite extensive missing edges. Bisimulation is observational equivalence, not equality of R. Identity bisimulation in the existing evaluator includes unreachable states and does not mean strict edge equality either.

The complete model illustrates good-only over-generalization: it accepts all syntactically valid negatives, including the same-label corruptions.

For Luca's formulation, finite good/bad separation alone does not define transition reconstruction in the known-state setting. A reconstruction objective needs requirements beyond sample consistency, such as held-out real transition/trace recovery or explicit structural comparison, and a clearly stated partial/total hypothesis class and inductive assumptions. This does not rule out useful learning under additional assumptions, and the fixed rotated maps and nested samples are not independent statistical trials.

## Reproduce and inspect

```bash
PYTHONPATH=src python -m experiments.good_bad_trace_sanity \
  --output-dir outputs/good_bad_trace_sanity/new_run
```

The output directory must not exist. Files: report.json (aggregates, provenance, hashes), cases.jsonl (360 full records), trace_pools.jsonl (72 paired pools with corrupted-step metadata), benchmark.json (24 exact reference graphs) and report.md. Controls are evaluated once per map and reused, with the cache and timing recorded explicitly.
