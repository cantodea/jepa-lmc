# Primary learned-model evaluation: simulation and bisimulation

Evaluation date: 2026-10-02. Base `main`: `88e116c`. This report re-evaluates the
frozen ranking JEPA graphs and oracle-local candidate graphs on the same three
seeds × 24 stress maps (72 cases per construction; 8,928 state-action queries).
Simulation and bisimulation are the **primary formal comparisons**. Top-1 accuracy
and the existing six CTL/six LTL formulas are **additional diagnostic checks**.

No training or inference was performed for this replay. Saved graph exports were
read directly. All 3,825 restored original artifacts and all six pinned
ranking/regression checkpoints remain byte-identical. Models, selection protocols,
data, labels, initial states, CTL/LTL semantics and `nuxmv.py` are unchanged.

## Definitions

All relations preserve atomic-proposition labels. `simulation_real_to_learned`
means that the learned graph can match every real step. Its initial metric is
`forall i in I_real, exists j in I_learned: (i,j)` belongs to the greatest forward
simulation. `simulation_learned_to_real` uses the opposite direction.

`initial_bisimulation` asks whether the greatest bisimulation covers both designated
initial sets: every initial on either side has a related initial on the other.
For singleton initial sets this is exactly `(i_real, i_learned) in ~`. It is computed
by the existing bisimulation fixed point, not by combining two simulations.

`identity_bisimulation` requires shared state identifiers and means
`forall s in S: (s,s) in ~`. It includes unreachable states. The two
`identity_simulation_*` metrics apply that same diagonal-membership test to each
greatest simulation. These metrics are `null` when state spaces differ; their
aggregate denominators exclude such cases. Here “global” means this all-state
identity metric, not merely initial equivalence.

There are two semantics, always evaluated separately:

- `action_insensitive` (`action_sensitive=False`): forget action labels, matching
  the project's unchanged Kripke/CTL semantics.
- `action_sensitive` (`action_sensitive=True`): every matching reply must carry
  the same action, testing action-respecting dynamics.

The additional `strict_identity_*` fields test whether the **diagonal alone** is
a relation (label equality plus edge inclusion/equality). This is stronger than
requiring diagonal pairs to belong to the greatest relation: matching successors
may be distinct but behaviorally equivalent. Historical `identity_relation_holds`
and the oracle's direct `identity_simulation` refer to this strict check; they
have not been silently renamed or changed.

## Uniform structural results

All fractions below count whole seed-map cases, not individual states or edges.

| Metric | Ranking, ignore actions | Oracle-local, ignore actions | Ranking, respect actions | Oracle-local, respect actions |
|---|---:|---:|---:|---:|
| Initial simulation real → learned | 60/72 | 72/72 | 40/72 | 72/72 |
| Initial simulation learned → real | 58/72 | 55/72 | 40/72 | 40/72 |
| Initial bisimulation | 40/72 | 49/72 | 40/72 | 40/72 |
| Identity simulation real → learned | 52/72 | 72/72 | 33/72 | 72/72 |
| Identity simulation learned → real | 54/72 | 51/72 | 33/72 | 33/72 |
| Identity bisimulation (global) | 33/72 | 46/72 | 33/72 | 33/72 |
| Strict identity simulation real → learned | 38/72 | 72/72 | 33/72 | 72/72 |
| Strict identity simulation learned → real | 45/72 | 45/72 | 33/72 | 33/72 |
| Strict identity bisimulation | 33/72 | 45/72 | 33/72 | 33/72 |

Oracle-local has sound forward simulation on every case, but this does not imply
reverse simulation or bisimulation. On `20260804 / sealed_region_layout1_rot0`,
ordinary identity bisimulation holds although the unlabelled edge sets differ.
This explains **46/72** identity bisimulation versus **45/72** strict identity
bisimulation. Action-sensitive identity bisimulation remains **33/72**.

## Additional transition and temporal diagnostics

| Diagnostic | Ranking Top-1 | Ranking + oracle-local |
|---|---:|---:|
| Underlying predictor Top-1 | 8832/8928 (98.9247%) | Same frozen predictor (98.9247%) |
| Candidate successor coverage | Not an overapproximation | 8,928/8,928 (historical oracle geometry) |
| All-six initial CTL agreement | 68/72 | 69/72 |
| All-six initial LTL agreement | 70/72 | 70/72 |
| All-state CTL agreement | 13197/13392 (98.5439%) | 13230/13392 (98.7903%) |
| All-state LTL agreement | 13282/13392 (99.1786%) | 13279/13392 (99.1562%) |
| Six CTL formulas agree, but initial bisimulation fails | 28 cases | 20 cases |

The CTL counts reproduce **68/72** and **69/72** exactly. The corresponding
bisimulation counts are **40/72** and **49/72**. Thus the finite formula suite
misses 28 and 20 structurally inequivalent cases. Bisimulation preserves these
formulas; agreement on only these formulas does not establish bisimulation.
Two-way simulation is also not substituted for bisimulation.

Native CTL was recomputed and checked against every saved state/property verdict.
LTL values were reused from the original nuXmv runs after checking graph-export
hashes, query hashes, and complete state/property coverage; this replay did **not**
rerun all experimental LTL backend calls. The full test suite did run its real
nuXmv integration tests. `backend_sanity.py` remains an additional backend
consistency audit and `nuxmv.py` remains solely the exporter/external backend.

The oracle-local Top-1 diagnostic refers to its underlying frozen point predictor.
The set-valued graph's 100% coverage is a different metric and is never reported as
100% Top-1 accuracy. Candidate geometry is unchanged: mean size 1.011201,
maximum 3, 98.9247% singletons. These oracle bounds still consume true successors.

## Bisimulation by seed

Each denominator is 24 maps.

| Seed | Ranking initial / identity | Oracle-local initial / identity | Ranking action initial / identity | Oracle-local action initial / identity |
|---|---:|---:|---:|---:|
| 20260804 | 10/24 / 8/24 | 14/24 / 13/24 | 10/24 / 8/24 | 10/24 / 8/24 |
| 20260805 | 10/24 / 6/24 | 15/24 / 13/24 | 10/24 / 6/24 | 10/24 / 6/24 |
| 20260806 | 20/24 / 19/24 | 20/24 / 20/24 | 20/24 / 19/24 | 20/24 / 19/24 |

## Recompute without training

Restore the existing private experiment archives into the checkout. Use a fresh
output directory (the commands refuse to overwrite prior evidence):

```bash
python -m experiments.evaluate_structural \
  --run-dir outputs/top1_quality/round2/behavioral/best \
  --ltl-dir outputs/top1_quality/round2/temporal/best_ltl \
  --output-dir outputs/model_evaluation/ranking_replay --expected-cases 72

python -m experiments.evaluate_structural \
  --run-dir outputs/local_abstraction/oracle_ranking_v1/behavioral/ranking_local \
  --ltl-dir outputs/local_abstraction/oracle_ranking_v1/temporal/ranking_local_ltl \
  --predictor-run-dir outputs/top1_quality/round2/behavioral/best \
  --output-dir outputs/model_evaluation/oracle_local_replay --expected-cases 72
```

The common implementation is `evaluation/structural.py`; it calls the existing
`verification/behavioral_relations.py`. Every result is audited for closure and
maximality, and both bisimulation semantics are cross-checked with the independent
partition-refinement implementation. For each construction this replay reproduced
**432 saved complete greatest relations**, audited **432 certificates**, checked
**144 partitions**, and reproduced **26,784 saved native CTL verdicts**.

New runs of `evaluate_jepa`, `evaluate_multibackend`, the ranking/Top-1 evaluator,
`quality_behavioral`, `top1_behavioral_relations`, and `oracle_local_abstraction`
now save the common scorecard for each graph pair. The ranking and oracle summary
scripts include it alongside the existing diagnostics. No training/selection rule
was changed. The older pilot scripts still train when explicitly invoked; use the
saved-graph command above for evaluation-only work.

The JSON schema exposes `behavioral.action_insensitive` and
`behavioral.action_sensitive` per case. Reports contain `primary_metrics`,
`diagnostics`, per-seed/family summaries and `mismatch_cases`. The mixed-split
ranking evaluator also reports `by_split`. Missing LTL diagnostics are not counted
as successes. Full state indices excluded from each identity relation are retained.

## Validation and records

**132 tests passed; 0 failures, 0 errors, 0 skipped**, including all actual nuXmv
integration tests. The 13 new tests cover different-size bisimilar systems,
finite-suite CTL agreement without bisimulation, action semantics, simulation
direction, unreachable states, multiple initials, strict identity versus diagonal
membership, mutual simulation without bisimulation, unknown denominators, saved
artifact integrity, ranking evaluation and oracle-local pipeline integration.

All changed Python files pass Ruff; `git diff --check` passes. The full-repository
Ruff scan still reports two pre-existing `RUF005` style suggestions in unchanged
`envs/gridworld.py` and `verification/witness.py`.

- [Ranking complete report](results/structural_evaluation/ranking/report.json),
  [case table](results/structural_evaluation/ranking/cases.csv),
  [mismatches](results/structural_evaluation/ranking/mismatches.json).
- [Oracle-local complete report](results/structural_evaluation/oracle_local/report.json),
  [case table](results/structural_evaluation/oracle_local/cases.csv),
  [mismatches](results/structural_evaluation/oracle_local/mismatches.json).
- [Integrity and runtime audit](results/structural_evaluation/audit.json),
  [full test log](results/structural_evaluation/test_results.txt).

Historical reports remain unchanged. These are exact results on the saved finite
graphs in an already inspected stress suite, not a guarantee for unseen instances.
