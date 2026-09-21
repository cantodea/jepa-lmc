# Oracle local successor abstraction on frozen ranking JEPA

This study freezes the three `ranking` checkpoints selected at commit `c15c457`.
It asks whether the learned geometry can yield useful, tight overapproximations
if correct local error bounds are supplied. It does not estimate such bounds.
The original regression checkpoints and every previous result are preserved.

The [fixed protocol](../configs/local_abstraction_protocol.json) uses the same
24 stress maps and three seeds. Primary results use ranking/local balls;
ranking/global maximum balls isolate the effect of locality on the same geometry.
Regression/global balls must reproduce every saved historical maximum-radius
candidate set and CTL aggregate. Regression/local balls isolate the geometry gain.
Frozen Top-1, exact and all-states graphs provide additional controls. No maps,
weights, formula suites, candidate topology filters or training recipes change.

## Conditional claim and best-case interpretation

Let $M=(S,I,R,L)$ be finite with a total deterministic successor $T(s,a)$.
Keep $S,I,L$ fixed. For distances $D_{sa,u}$ to all target-encoder states of the
same map, construct

$$b_{oracle}(s,a)=D_{sa,T(s,a)},\qquad
C_{local}(s,a)=\{u\in S:D_{sa,u}\le b_{oracle}(s,a)\}.$$

The closed comparison retains every boundary tie. Thus each true successor is
present, every action has a successor, and $R\subseteq R_C$ with actions retained.
The identity relation is an action-preserving simulation from $M$ to $M_C$:
each true step is matched by that same step, with identical endpoint labels.
Forgetting actions gives the corresponding Kripke simulation and trace inclusion.
Consequently, $M_C,s\models\varphi$ implies $M,s\models\varphi$ for ACTL in
universal negation-normal form, and for universally interpreted LTL. Existential
CTL false verdicts transfer in the other truth direction. Mixed arbitrary CTL
and bisimulation do not follow merely from edge inclusion.

Within this fixed distance-ball family, the oracle local radius is the smallest
one containing the true successor. Any valid larger radius adds candidates and
cannot improve universal-True/existential-False proving power. With one maximum
radius pooled over the entire stress domain for each checkpoint,
$C_{local}(s,a)\subseteq C_{global}(s,a)$ necessarily holds. These are best-case
diagnostics, not a deployable method when $T$ is unknown. They also do not rule out
other, non-ball abstraction families.

Distances use the historical evaluator's direct float64 Euclidean arithmetic on
float32 network outputs; complete rankings are independently checked by direct
subtraction and against canonical float32 Top-1. The finite-graph edge audits are
exact comparisons of state/action identifiers. The distances themselves are not
certified real-arithmetic enclosures. Both encoder outputs and complete distances
are retained for numerical replay. If there are no boundary ties, local set size
equals the true successor's rank; ties can make the set larger than its reported
rank under deterministic index-based tie breaking.

## Metrics and interpretation

Candidate metrics count action-labelled triples. The main spurious-edge rate is
$|R_C\setminus R|/|R_C|$; extra edges per true edge use $|R|$ as denominator and
are reported separately. Ordinary Kripke edge counts deduplicate different actions.
We report every requested metric by seed, family and action, with complete raw
rankings and candidate memberships available in the private record archive.

The first build stopped on a bitwise comparison of the recomputed historical
maximum radius, before constructing any candidate graphs. The unchanged seed-04
baseline gave 4.763254719625307 versus the saved 4.763255292832166. The failed
attempt is retained. The corrected replay records the numerical difference and
allows a relative tolerance of `1e-5` for this scalar only. Every historical
candidate membership and CTL aggregate must still match exactly, and every new
oracle ball uses its own saved distance table with an exact closed comparison.
The protocol, checkpoints, distance definition and decision thresholds are unchanged.

For temporal proving power, the historical primary score excludes immediate
source-label cases. It has 480 one-sided proof opportunities per seed over
`AG !danger`, `EF goal` and `E[!danger U goal]`. Universal safety true-proof recall
is reported separately. `AF goal` has no non-immediate positive opportunities in
this suite: a perfect all-state agreement for that formula is not evidence of
useful liveness proving power. Every LTL formula is interpreted universally, with
no fairness. The existing backend's legacy `top1` storage key explicitly aliases
the selected candidate graph; set-valued action edges are not collapsed to one.

The engineering continuation gate is fixed before the full run: every seed needs
full inclusion and identity simulation, no one-sided violations, mean set size
at most 1.10, maximum size at most 5, at least 95% singletons, and at least 75%
non-immediate safety and primary proof recovery. Both required proof families
must have at least four maps with a proof. Relative to historical global balls,
the pooled set count must fall by at least 50% and primary proof recall rise by
at least ten percentage points, without per-seed proof regression. These are
research screening thresholds, not statistical confidence guarantees.

## Reproduce

Use a checkout of this branch and extract the private records/checkpoints into
it. Install with `python -m pip install -e .`, and use fresh output directories.
The build refuses to overwrite previous outputs and performs no training:

```bash
python -m experiments.oracle_local_abstraction --output-dir outputs/local_abstraction/local_recheck
```

For each of `ranking_local`, `ranking_global`, `regression_global`, run the two
unchanged external-backend scripts on its saved graph directory. For example:

```bash
python -m experiments.backend_sanity --run-dir outputs/local_abstraction/local_recheck/behavioral/ranking_local --output-dir outputs/local_abstraction/local_recheck/temporal/ranking_local_sanity --executable /path/to/nuXmv
python -m experiments.evaluate_top1_ltl --run-dir outputs/local_abstraction/local_recheck/behavioral/ranking_local --output-dir outputs/local_abstraction/local_recheck/temporal/ranking_local_ltl --executable /path/to/nuXmv
```

Repeat those backend commands for `ranking_global` and `regression_global`, then:

```bash
python -m experiments.summarize_local_abstraction --run-dir outputs/local_abstraction/local_recheck
```

The private archive includes all six frozen checkpoints, the model bank, historical
global candidates and the saved ranking Top-1 temporal records required by these
commands. Complete reproduction creates a fresh run and a new preservation manifest;
the completed run's manifest also names older experiment artifacts outside this
archive. No checkpoint is selected, updated or retrained during these commands.

On Windows use the virtual environment's `python.exe` and your installed
`nuXmv.exe`. The same states, labels, designated initial state and state-specific
query encoding are used for both logics. The executable is not redistributed.

## Candidate geometry results

The fixed domain has 8,928 state-action queries: 2,976 per seed, over all valid
states of each of the 24 maps, including states unreachable from the designated
initial state. This is exhaustive evaluation on these finite maps, not a claim
about unseen environments.

| Seed | Coverage | Mean size | Median / max | Size 1 | Size 2 | Spurious / candidate edges |
|---|---:|---:|---:|---:|---:|---:|
| 20260804 | 100% | 1.009745 | 1 / 2 | 99.0255% | 0.9745% | 29 / 3,005 (0.9651%) |
| 20260805 | 100% | 1.020161 | 1 / 3 | 98.1183% | 1.7473% | 60 / 3,036 (1.9763%) |
| 20260806 | 100% | 1.003696 | 1 / 2 | 99.6304% | 0.3696% | 11 / 2,987 (0.3683%) |
| Pooled | 100% | **1.011201** | **1 / 3** | **98.9247%** | **1.0305%** | **100 / 9,028 (1.1077%)** |

The ranking true-successor histogram is 8,832 at rank 1, 92 at rank 2 and four
at rank 3. Top-1/2/3 coverage is **98.9247% / 99.9552% / 100%**. No boundary ties
occur. Thus the oracle ball has exactly the true successor's rank in every query.
Coverage is guaranteed by the oracle construction; the informative findings are
its small size and retained temporal proving power.

| Predictor and bound | Mean size | Max | Singletons | Spurious edges | Spurious-edge rate |
|---|---:|---:|---:|---:|---:|
| Ranking, local oracle | **1.011201** | 3 | **98.9247%** | **100** | **1.1077%** |
| Ranking, pooled global maximum | 2.940748 | 8 | 10.1703% | 17,327 | 65.9950% |
| Regression, local oracle | 1.059140 | 3 | 94.1980% | 528 | 5.5838% |
| Regression, historical global maximum | 3.120744 | 9 | 9.4198% | 18,934 | 67.9564% |
| All states | 31.021505 | 32 | 0% | 268,032 | 96.7764% |
| Exact graph | 1 | 1 | 100% | 0 | 0% |

Ranking/local reduces candidate count by 67.60% and spurious action edges by
99.47% relative to the historical global baseline. Against the same ranking
predictor's global bound, candidate count falls by 65.61%. Against regression/local,
it removes 81.06% of spurious edges. This separates the effect of local bounds
from the improvement in the predictor.

For ranking/local, every family and action group has 100% coverage and median
size 1. The full seed-by-family-by-action breakdown, plus Top-k coverage, is in
the [aggregate JSON](results/oracle_local_abstraction/summary.json).

| Group | Queries | Mean size | Max | Size 1 | Size 2 | Spurious-edge rate |
|---|---:|---:|---:|---:|---:|---:|
| sealed_region | 2,880 | 1.010764 | 2 | 98.9236% | 1.0764% | 1.0649% |
| danger_gate | 2,976 | 1.010417 | 3 | 99.0255% | 0.9073% | 1.0309% |
| safe_detour | 3,072 | 1.012370 | 3 | 98.8281% | 1.1068% | 1.2219% |
| up (0) | 2,232 | 1.006720 | 2 | 99.3280% | 0.6720% | 0.6676% |
| down (1) | 2,232 | 1.022849 | 3 | 97.8943% | 1.9265% | 2.2339% |
| left (2) | 2,232 | 1.007616 | 2 | 99.2384% | 0.7616% | 0.7559% |
| right (3) | 2,232 | 1.007616 | 2 | 99.2384% | 0.7616% | 0.7559% |

## Behavioral and CTL results

Action-labelled inclusion and the identity simulation from the real graph to the
candidate graph hold on **72/72 maps**, at every state and for every action.
All one-sided CTL violations are zero. Exact identity simulation is checked
directly, separately from membership in the greatest simulation relation.

The primary non-immediate proof opportunities are 120 true `AG !danger`, 120
false `EF goal`, and 240 false `E[!danger U goal]` per seed. The false existential
verdicts are valid absence proofs from an overapproximation; a true existential
verdict alone is not transferred.

| Construction | Seed 04 | Seed 05 | Seed 06 | Pooled primary proof recall |
|---|---:|---:|---:|---:|
| Ranking/local | 426/480 | 408/480 | 480/480 | **91.25%** |
| Ranking/global | 408/480 | 132/480 | 480/480 | 70.83% |
| Regression/local | 408/480 | 408/480 | 480/480 | 90.00% |
| Historical regression/global | 264/480 | 138/480 | 480/480 | 61.25% |
| All states | 0/480 | 0/480 | 0/480 | 0% |
| Exact graph | 480/480 | 480/480 | 480/480 | 100% |

Ranking/local retains **324/360 (90%)** non-immediate safety proofs: 102/120,
102/120 and 120/120 by seed. It has proofs on 7/8, 7/8 and 8/8 sealed-region
maps, and 8/8, 7/8 and 8/8 danger-gate maps. Safe-detour has no primary proof
opportunities and is N/A for that score. `AF goal` similarly has no non-immediate
true-proof opportunity; its agreement must not be treated as liveness evidence.

| Graph | Initial real-to-candidate simulation | Initial candidate-to-real simulation | Initial ordinary bisimulation | Initial all-six CTL agreement |
|---|---:|---:|---:|---:|
| Frozen ranking Top-1 | 60/72 | 58/72 | 40/72 | 68/72 |
| Ranking/local | **72/72** | 55/72 | **49/72** | **69/72** |
| Ranking/global | 72/72 | 18/72 | 18/72 | 59/72 |
| Regression/local | 72/72 | 42/72 | 33/72 | 68/72 |
| Historical regression/global | 72/72 | 16/72 | 16/72 | 54/72 |

These simulations use ordinary Kripke semantics. With actions preserved, ranking/local
initial forward simulation still holds on 72/72, while reverse simulation and
bisimulation each hold on 40/72. Sound overapproximation does not imply bisimulation.
Pooled all-state CTL agreement for ranking/local is 98.7903%.

The largest proving-power gain comes from using local bounds: even the older
regression geometry recovers 90% of primary opportunities under oracle local bounds.
Ranking further tightens candidates and raises primary recovery to 91.25%; the
full global-to-local gain should not be attributed solely to ranking training.

## External backend and decision

The unchanged nuXmv 2.2.0 backend independently checks ranking/local,
ranking/global and historical regression/global. For **each** variant:

- Native CTL versus nuXmv CTL: **26,784/26,784 agreement**.
- `AG !danger` versus `G !danger`: **4,464/4,464 agreement**.
- `AF goal` versus `F goal`: **4,464/4,464 agreement**.
- No mismatches. State-specific encodings, labels, initial states and graphs
  match, with no fairness. All six original LTL formulas are also evaluated.

| Graph | Initial all-six LTL agreement | All-state LTL agreement | Non-immediate LTL true-proof recovery | Unsound LTL True claims |
|---|---:|---:|---:|---:|
| Frozen ranking Top-1 (saved run) | 70/72 | 13,282/13,392 (99.1786%) | 2,963/3,072 | 1 |
| Ranking/local | **70/72** | **13,279/13,392 (99.1562%)** | **2,959/3,072 (96.3216%)** | **0** |
| Ranking/global | 66/72 | 12,833/13,392 (95.8259%) | 2,513/3,072 (81.8034%) | 0 |
| Historical regression/global | 64/72 | 12,733/13,392 (95.0792%) | 2,413/3,072 (78.5482%) | 0 |

The Top-1 recovery column counts empirically correct True verdicts; Top-1 does
not have the overapproximation guarantee. Ranking/local removes its single
unsound `X !danger` True verdict and loses four previously correct True verdicts
through conservatism. Consequently all-state LTL agreement falls by three
verdicts even though the abstraction becomes sound. Initial LTL agreement is
unchanged. Better coverage does not imply higher agreement for every metric.

For ranking/local, non-immediate LTL proof recall is 90% for each of `G !danger`,
`G F safe` and `F G !danger`, and 1,987/1,992 (99.7490%) for `X !danger`.
`F goal` and `safe U goal` have zero non-immediate positive opportunities here;
their true-proof recall is N/A. The three safety-related LTL formulas share
verdicts on this suite, so they are not independent evidence.

**Decision: pass the fixed oracle feasibility gate on all three seeds.**
Relative to the historical global baseline, pooled primary proof recall rises
from 61.25% to 91.25% (+30 percentage points), with no seed regression, while
candidate count falls by 67.60%. The geometry is sufficiently tight and useful
to justify a separate future study of how to obtain valid local bounds when
the true transition relation is unknown. This study stops at that decision:
it does not fit bounds, use conformal/PAC, or certify unseen maps. Because the
oracle is the smallest sound ball, larger valid bounds may lose proving power.

The audit independently replays 144 full embedding/distance tables (two models,
72 seed-map cases), 17,856 rankings and 62,496 candidate memberships across all
seven variants. All 8,928 historical global candidate sets and 72 saved ranking
Top-1 graphs are reproduced exactly. It checks 3,024 greatest-relation
certificates, 1,008 independent bisimulation partitions and 26,784 nested-graph
temporal monotonicity comparisons. All 22 focused unit tests pass. All six
checkpoint hashes and 2,983 pre-existing files, including the retained failed
replay attempt, are unchanged. The complete machine-readable results are in the
[aggregate summary](results/oracle_local_abstraction/summary.json); full raw
embeddings, distances, memberships, verdicts and backend logs are in the private
record archive.
