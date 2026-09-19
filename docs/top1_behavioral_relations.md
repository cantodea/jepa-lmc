# Exact simulation and bisimulation of the existing Top-1 graphs

The maximal-relation check finds more corresponding initial states than literal
edge equality does. However, **initial bisimulation passes only in sealed safe
regions**: 8/24, 7/24 and 8/24 maps for the three checkpoints. Neither the dangerous
gate nor the safe-detour family has a passing initial bisimulation. The current
Top-1 graphs therefore do not support a broad behavioral-equivalence claim on
this suite.

This is an exact comparison of two fully constructed finite graphs. The checker
uses no radius, changes no transitions, and requires no retraining or JEPA
architecture/loss changes. The CEGAR experiment remains available separately.

## Inputs and direction convention

The [protocol](../configs/top1_relations_protocol.json) reuses the three frozen
checkpoints and all 24 existing topology maps. Each map has 30–32 states; a seed
covers 744 states and 2,976 state/action pairs. States, initial states and exact
labels are identical between the real and learned graph. The only differing
component is the transition relation.

Top-1 reconstruction calls the existing `evaluate_gridworld_transitions` and
`transition_system_from_retrievals`. This is the normal float32 `torch.cdist` /
`argsort` decoder, not an epsilon candidate model. The earlier oracle-radius
diagnostic used float64 distances. Neither decoder supplies a certified error
bound; once a finite graph has been decoded, relation checking is discrete.

We write **M <= Top1** to mean that Top1 can match every step of M. Thus the
right-hand model simulates the left-hand one. The primary check ignores action
labels, matching the repository's ordinary CTL checker. A separate, stronger
check requires equal action labels on matching edges.

The comparison starts from all label-compatible cross-model pairs, not just
`(s,s)`. The result records initial correspondence, all same-coordinate pairs,
the complete surviving relation, and an explanation for every excluded pair.

## Initial-state results

Each entry counts maps out of 24 whose corresponding initial pair belongs to
the indicated **greatest** relation.

| Seed | M <= Top1 | Top1 <= M | Bisimulation | All six existing CTL formulas agree at the start |
|---|---:|---:|---:|---:|
| 20260804 | 13/24 | 8/24 | **8/24** | 21/24 |
| 20260805 | 10/24 | 16/24 | **7/24** | 13/24 |
| 20260806 | 10/24 | 22/24 | **8/24** | 18/24 |

There are 29 model/map cases where all six checked formulas agree at the start
but bisimulation fails. Agreement on a finite property suite is insufficient
to establish the stronger relation. Conversely, passing bisimulation provides
CTL equivalence over the fixed atomic propositions, not merely agreement on
those six formulas.

For reference, literal full-graph edge inclusion/equality gives:

| Seed | R subset of R_hat | R_hat subset of R | R = R_hat |
|---|---:|---:|---:|
| 20260804 | 2/24 | 4/24 | 2/24 |
| 20260805 | 0/24 | 5/24 | 0/24 |
| 20260806 | 0/24 | 13/24 | 0/24 |

These are the conditions for the **identity relation itself** to be a simulation
or bisimulation over every state. They differ from asking whether individual
diagonal pairs belong to a larger relation containing off-diagonal pairs.
They also inspect the whole graph, whereas an initial-state result need not
depend on unreachable components.

## Same-coordinate states throughout the graph

The counts below test `(s,s)` membership in each maximal relation, including
states unreachable from the designated initial state. They are not the number
of pairs in the identity relation that independently form a closed simulation.

| Seed | M <= Top1: states | Top1 <= M: states | Bisimilar states | Maps where every diagonal pair is bisimilar |
|---|---:|---:|---:|---:|
| 20260804 | 422/744 (56.72%) | 242/744 (32.53%) | 144/744 (19.35%) | 2/24 |
| 20260805 | 246/744 (33.06%) | 497/744 (66.80%) | 102/744 (13.71%) | 0/24 |
| 20260806 | 211/744 (28.36%) | 675/744 (90.73%) | 120/744 (16.13%) | 0/24 |

Top-1 action accuracy is 95.09%, 92.07% and 95.43%, respectively. These local
prediction rates do not imply preservation of branching behavior.

## Why the passing cases need careful interpretation

| Family | Seed 20260804: initial bisimulation | Seed 20260805 | Seed 20260806 |
|---|---:|---:|---:|
| Sealed region | 8/8 | 7/8 | 8/8 |
| Dangerous mandatory gate | 0/8 | 0/8 | 0/8 |
| Safe detour | 0/8 | 0/8 | 0/8 |

In **all 23 passing cases**, every state reachable from the start in either
graph has exactly the label `{safe}`. Neither goal nor danger is reachable.
Both reachable graphs are total. The Cartesian product of these two reachable
sets is therefore a bisimulation: labels match and each successor can be matched
by some successor still in the other reachable set.

This is valid equivalence for the given observation language, even when the
graphs have different edges. It is weak evidence that the detailed movement
dynamics were learned correctly. For example, seed 20260805 fails the remaining
sealed case, `sealed_region_layout1_rot90`: the real start reaches 18 safe states,
while the Top-1 graph reaches 20 states including danger.

The action-labelled check gives initial simulation in either direction and
bisimulation on 8/24, 7/24 and 8/24 maps. In this run its complete bisimulation
pair sets equal the ordinary-CTL bisimulation pair sets on every map. This is an
observed result, not a general equivalence of the two semantics. Both GridWorld
models have a unique successor for every action; in that special setting an
action-preserving simulation with equal state labels also satisfies the reverse
step condition. This does not hold for general nondeterministic systems.

## Actual mutual-simulation failures

The following initial pairs satisfy simulation in both directions but fail
bisimulation:

- 20260805: `danger_gate_layout0_rot180`.
- 20260806: `danger_gate_layout0_rot270`.
- 20260806: `danger_gate_layout1_rot270`.

For the first case, the bisimulation checker removes the initial pair
`((5,5),(5,5))` at round 8. The real edge `(5,5) -> (4,5)` has no bisimilar reply:

| Candidate Top-1 successor of (5,5) | Pair with real successor (4,5) was removed at round |
|---|---:|
| (4,5) | 7 |
| (5,4) | 5 |
| (5,5) | 2 |

Even the same-coordinate reply has incompatible subsequent branching behavior.
The full removal evidence records the recursive reasons for all these failures.
It is a finite proof DAG covering **every** candidate reply, not necessarily a
single path. Independently computed simulation relations cannot be combined by
assuming their intersection is automatically a bisimulation.

## Algorithm and what its certificate establishes

For simulation, initialize `Q0 = {(s,t): L(s)=L(t)}` and repeatedly remove a pair
if some left successor has no right successor related by the current `Q`.
For bisimulation, also impose the converse successor condition on the **same Q**.
Action-sensitive mode additionally requires matching actions. These are the
standard step-matching definitions; see the
[simulation/bisimulation definitions and algorithms](https://arxiv.org/pdf/1301.1638).

The finite product contains at most 1,024 pairs per map. The algorithm uses
synchronous deletion: all decisions in a round refer to the previous pair set.
Each deletion records a challenging edge and a round number; label mismatches
are rejected at round zero.

The certificate checker establishes two separate facts. First, all surviving
pairs satisfy labels and the required successor conditions, so the result is
a valid relation. Second, each excluded pair either mismatches labels or has
a real challenging edge whose every possible reply was excluded at a strictly
earlier round. Induction on that round shows no valid relation could contain
the excluded pair. Together these facts establish **greatestness**, rather than
just checking that some possibly incomplete relation is valid.

Bisimulation is also recomputed independently by partition refinement on the
disjoint union of the two graphs. Partitions split by labels and successor-block
signatures, without using the pair-deletion implementation.

For a related initial pair, M <= Top1 allows universal CTL truths of Top1 to
transfer to M, and existential CTL truths of M to transfer to Top1. Reversing the
simulation reverses those roles. These statements concern the universal and
existential fragments in negation normal form; arbitrary mixed CTL does not
follow from one-way simulation. Bisimulation preserves all CTL formulas over
the supplied labels in both directions. Under the primary unlabelled semantics,
a matching real path may use different actions and coordinates; a predicted
action sequence is not thereby certified for literal replay.

The results require the complete real graph for the compared domain. They do
not establish a relation on unseen maps or solve verification with unknown R.
Only three checkpoints and six base layouts were examined; rotations are
correlated. This is a diagnostic result, with no new statistical acceptance gate.

## Optional nuXmv LTL evaluation of the same graphs

The subsequent [backend sanity check](backend_sanity_check.md) has now run all
six CTL properties and the two equivalent CTL/LTL pairs through actual nuXmv on
these saved graphs. Every backend and paired-formula comparison agrees, and the
52/72 initial CTL score is unchanged. The other four LTL properties are outside
that sanity check's scope.

The recorded results above include the six CTL checks, but **no external LTL
run**. Having nuXmv installed does not automatically add LTL to
`top1_behavioral_relations.py`. The separate command below reads that run's
`report.json` and `relations.jsonl`, verifies their graph-export hash and map
coverage, and checks those exact saved graphs. It does not train, load weights,
or decode predictions again.

After running the Top-1 experiment, use PowerShell from the repository root.
Replace the executable path with your installed binary. If nuXmv is on PATH,
the environment assignment can be omitted.

```powershell
$env:NUXMV_BINARY = "D:\tools\nuXmv\bin\nuXmv.exe"
& .\.venv\Scripts\python.exe -m unittest tests.test_nuxmv tests.test_top1_ltl_experiment -v
& .\.venv\Scripts\python.exe experiments\evaluate_top1_ltl.py --run-dir outputs/behavioral_relations/top1_local --output-dir outputs/behavioral_relations/top1_local_ltl
```

Set `--run-dir` to your actual previous output directory (`top1` if you used
the reproduction command below). An explicit `--executable PATH` is also
supported. Each LTL output directory must be fresh. Missing executables and
incomplete backend verdicts are errors; they never produce a completed report.

The existing LTL suite contains `X !danger`, `F goal`, `G !danger`,
`safe U goal`, `G F safe`, and `F G !danger`. They are interpreted universally
over infinite paths, with action labels ignored and no fairness constraints.
In particular, `F goal` corresponds to CTL `AF goal`, **not** `EF goal`.
`G !danger` and `F goal` are retained as consistency diagnostics but excluded
from the existing primary LTL score. Missing positive or negative examples
produce a null balanced score, rather than an artificial perfect score.

For all 72 saved model/map cases, the command requests 26,784 backend verdicts:
two graphs times 2,232 states times six formulas. The resulting 13,392
same-coordinate formula comparisons include 432 initial-state comparisons.
Progress is printed after each map. Outputs are:

- `report.json`: overall, per-seed and per-map summaries, with initial-state
  and all-state scores separated; per-property confusion counts, safety false
  positives, source hashes and backend identity.
- `queries.csv`: every state/formula comparison, including original coordinates.
- `backend/`: the exact `.smv` models and nuXmv output, including any printed
  counterexample traces. File hashes are included in the report.

Bisimulation also preserves these LTL properties. Agreement on this finite
suite alone does not establish bisimulation. The older
`evaluate_multibackend.py` trains a new model and evaluates the random pilot
test maps; it is a separate experiment and does not reproduce this graph suite.
The new runner has mocked-backend regression checks and an installed-backend
integration test. The latter requires a local nuXmv/NuSMV installation; no
full-suite external LTL result is claimed here. For this addition, 28 targeted
tests ran: 25 passed and three external-backend tests were skipped because
nuXmv was unavailable. Reading and reconstructing all 72 saved graph pairs
also passed, including the source-export hash and map-coverage checks.

## Reproduce and inspect

Reuse the three existing checkpoints described in the
[earlier checkpoint instructions](latent_radius_stress_results.md#reproduce-and-inspect).
No training is needed when those files exist. After the normal editable install:

```powershell
& .\.venv\Scripts\python.exe experiments\top1_behavioral_relations.py --checkpoints outputs/latent_radius/oracle/model.pt outputs/latent_radius/stress_training/seed_20260805/model.pt outputs/latent_radius/stress_training/seed_20260806/model.pt --output-dir outputs/behavioral_relations/top1
```

Choose a fresh directory for another run. On Linux use `python` and `/` in the
script path. The recorded source is commit `a86b4ccc`; the run used Python
3.12.14, PyTorch 2.14.0+cpu and four CPU threads. The recorded processing times
are 3.55 / 3.23 / 3.21 seconds per seed, including reconstruction and in-memory
audits but excluding training, installation and checkpoint loading. Windows
commands are documented but have not been executed on Windows.

The API works directly with existing `ExplicitTransitionSystem` objects:

```python
from jepa_lmc.verification.behavioral_relations import (
    greatest_relation, audit_greatest_relation,
)

forward = greatest_relation(real, top1)
backward = greatest_relation(top1, real)
bisim = greatest_relation(real, top1, kind="bisimulation")
audit_greatest_relation(real, top1, bisim)
print(bisim.relates_initials(real, top1))
```

The [public aggregate](results/top1_relations/summary.json) records seed/family
counts, hashes, validation and selected diagnostic examples. The separately
delivered `jepa-lmc-top1-behavioral-relations-records.zip` contains both graphs,
all surviving and rejected pairs, initial failure evidence, 432 CSV rows, the
original report, export-audit script/results, test log and checksums. Raw graphs
and pair tables are not uploaded to the public repository.

Original relation-experiment validation: **98 unittest cases, 96 passed and two
existing external nuXmv
checks skipped**; Ruff passes. New tests include exhaustive enumeration of all
candidate relations on small graphs, nonidentity correspondence, mutual
simulation without bisimulation, action semantics, unreachable states, multiple
initials, and rejection of a forged closed but nonmaximal result. The experiment
passes 432 greatest-relation certificate audits, 144 independent partition
comparisons and 338,406 CTL pair/formula preservation checks. A separate replay
also validates every serialized certificate and reconciles the CSV aggregates.
