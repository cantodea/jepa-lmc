# CEGAR pivot: JEPA saves successor queries on the fixed screen

**Continue with the oracle-backed, property-directed experiment.** All three
frozen models pass the prespecified query-efficiency screen. JEPA guidance uses
51.03%, 51.42% and 52.26% fewer exact successor queries than direct BFS, with no
incorrect verdicts or lost true transitions. The observed gain comes entirely
from finding concrete paths sooner. Proving absence of a path costs the same as
the controls on this suite.

This supports a bounded research direction: use JEPA to prioritize expensive
concrete checks while a trusted refinement mechanism supplies correctness.
It does not establish an autonomous neural safety certificate, faster safety
proofs, generalization to unknown state spaces, or a novel CEGAR theory. The
previous [uniform-radius/Yang screening decision](latent_radius_stress_results.md)
remains STOP for that construction.

## Literature and the implemented change

The [literature review and conditional proof](cegar_literature_and_design.md)
connect the design to Clarke et al.'s CEGAR and Vinzent et al.'s AAAI 2023 neural
policy verification. Their concrete feasibility checks require trustworthy
environment semantics. The AAAI paper verifies a neural action policy with known
environment operators; it does not solve unknown learned-dynamics certification.

Here, known finite state identities and exact labels are fixed. Unknown actions
initially admit **every** state. Frozen JEPA distances only rank which upper-graph
path to check. A counted exact deterministic simulator query replaces one
state/action candidate set by its true singleton. All other sets remain intact.
The maintained relation is

\[
R_k^-\subseteq R^A\subseteq R_k^+.
\]

A confirmed lower path supplies a witness or safety counterexample. Absence of
the relevant path in the upper graph supplies the complementary verdict. Only
`AG !danger`, `EF goal`, and `E[!danger U goal]` are implemented in this loop.
Exhausting a query budget returns `unknown`. The observed lower graph is never
totalized with invented self-loops. Neither an oracle error radius nor neural
confidence authorizes edge deletion.

## Fixed experiment and decision

The [protocol](../configs/cegar_protocol.json) was committed before the experiment
(published protocol commit `67b7aced`; implementation `2994c382`). It reuses the
three existing 40-map, 100-epoch checkpoints, without changing the encoder,
predictor, loss, training code or model weights.

There are 24 maps: three topology families, two layouts per family, four rotations
per layout. At each map's start, check the three properties independently with a
fresh oracle cache for each method. This gives **72 tasks per seed/method and
864 full-budget tasks overall**. The full-table budget is 120, 124 or 128 queries
depending on the map. All start-state queries are non-immediate label cases.
Exhaustive ground truth is built only after the compared tasks on a map finish,
and used for auditing, not guidance.

The engineering screen requires zero incorrect conclusions and preserved
inclusion, plus at least 20% fewer total queries than both uniform CEGAR and
direct BFS on **every** seed, and strictly fewer than shuffled JEPA rankings.
The three seeds all pass. No ranking, map, threshold or checkpoint was adjusted
after seeing the result.

| Model seed | Uniform CEGAR: queries/task | JEPA CEGAR: queries/task | Shuffled CEGAR: queries/task | Direct BFS: queries/task | Reduction vs uniform | Reduction vs BFS | Reduction vs shuffled |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20260804 | 71.14 | **34.93** | 67.72 | 71.33 | 50.90% | 51.03% | 48.42% |
| 20260805 | 71.14 | **34.65** | 65.69 | 71.33 | 51.29% | 51.42% | 47.25% |
| 20260806 | 71.14 | **34.06** | 67.57 | 71.33 | 52.13% | 52.26% | 49.60% |

Exact totals per seed are 5,122 for uniform CEGAR, 5,136 for BFS, and
2,515 / 2,495 / 2,452 for JEPA. Shuffled rankings use 4,876 / 4,730 / 4,865.
Every method resolves all 72 tasks per seed correctly at the full budget;
864/864 verdicts agree with the existing exact CTL checker. Every intermediate
and final action-labelled inclusion audit passes. All three model-state hashes
are unchanged after evaluation.

### Resolution with limited queries

Budget percentages are relative to each map's complete `|S||A|` table, rounded
down. These are proportions of tasks with a conclusive correct answer, not
accuracy with `unknown` treated as a Boolean.

| Method / seed | Resolved at 10% budget | At 25% | At 50% | At 100% |
|---|---:|---:|---:|---:|
| Uniform CEGAR, each seed | 0.00% | 11.11% | 34.72% | 100% |
| Direct BFS, each seed | 0.00% | 11.11% | 36.11% | 100% |
| JEPA, 20260804 | 16.67% | **55.56%** | **77.78%** | 100% |
| JEPA, 20260805 | 26.39% | **55.56%** | **77.78%** | 100% |
| JEPA, 20260806 | 29.17% | **55.56%** | **77.78%** | 100% |
| Shuffled, 20260804 | 1.39% | 11.11% | 43.06% | 100% |
| Shuffled, 20260805 | 4.17% | 11.11% | 41.67% | 100% |
| Shuffled, 20260806 | 2.78% | 11.11% | 37.50% | 100% |

The runner calculates these rates from the deterministic full-run query counts.
The separate audit actually replays every task at all four budgets, checks the
query prefixes, inclusion and certificates, and compares the bounded verdicts
with those rates: **3,456 bounded replays pass**.

## Where the savings occur

Every seed has 40 path-existence tasks and 32 path-absence tasks. Path existence
includes both existential-True witnesses and `AG !danger`-False counterexamples.

| Certificate required | Tasks per seed | Uniform total queries | BFS total queries | JEPA totals, seeds 04 / 05 / 06 |
|---|---:|---:|---:|---:|
| Concrete path | 40 | 3,202 | 3,216 | **595 / 575 / 532** |
| Absence of a path | 32 | 1,920 | 1,920 | **1,920 / 1,920 / 1,920** |

Against BFS, each seed has 40 task-level query wins, 32 ties, and zero losses.
The aggregate improvement therefore must not be described as faster proofs of
safety or unreachability. It is evidence for finding real witnesses and real
safety violations with fewer queries on these maps.

| Topology family | Uniform mean queries | BFS mean queries | JEPA means, seeds 04 / 05 / 06 |
|---|---:|---:|---:|
| Sealed region | 60.00 | 60.00 | 60.00 / 60.00 / 60.00 |
| Dangerous mandatory gate | 66.58 | 66.83 | 28.88 / 28.33 / 28.00 |
| Safe detour | 86.83 | 87.17 | 15.92 / 15.63 / 14.17 |

This limitation follows from the chosen information model. With a nonempty
target set, any still-unqueried action at a permitted reachable state admits a
direct edge to a target. To exclude all target paths, all such outgoing actions
must therefore be checked. Without additional certified restrictions on possible
successors, neural ordering cannot remove this requirement. This is a statement
about the current all-states initialization, not all forms of CEGAR.

The final mean upper candidate size is actually **larger** for JEPA:
22.55 / 22.62 / 22.76, versus uniform CEGAR's 13.79 and BFS's 13.74.
It stops after obtaining a sufficient property certificate while leaving more
irrelevant actions unknown. Globally shrinking candidate sets is not the goal.
JEPA's mean singleton fractions are 28.57% / 28.35% / 27.88%; here they equal
the queried-pair fractions, since every other action still admits all states.
Initial true-successor coverage is already guaranteed by the full catalogue;
the meaningful question is how many exact queries are needed for a verdict.

### The previously failing wall jump

For seed 20260805 on `danger_gate_layout0_rot0`, the JEPA-guided safe-until task
initially considers action right from `(0,1)` toward `(0,3)`. At **query 2**, the
simulator returns `(0,1)`: the wall blocks the move. Refinement removes `(0,3)`
from that action's set and keeps `(0,1)`. This is the specific spurious edge from
the earlier radius experiment.

The algorithm continues checking remaining paths. It returns
`E[!danger U goal] = False` after **48 queries**, equal to all three controls.
Removing the first false edge is useful, but does not itself prove the property.
The regression test separately checks that other unqueried actions remain
unrestricted after this local update.

## Costs, scope and the next research step

On this cheap simulator, BFS is faster in elapsed verification time. The recorded
72-task BFS totals are 0.106 / 0.013 / 0.017 seconds; JEPA-guided checking takes
0.701 / 0.598 / 0.589 seconds, plus 2.234 / 0.402 / 0.119 seconds of neural
preparation across each seed's 24 maps. Preparation includes observations,
encoding, distances and rank shuffling; it is reused across the three properties.
Training and checkpoint loading are excluded. These are single-run diagnostic
timings; concurrent validation and first-use initialization affected them, so
they are not controlled runtime benchmarks. They provide no speedup claim.

The state catalogue, wall layout and labels are given. Known GridWorld movement
rules would permit direct symbolic successor computation from that information.
Treating every successor as an expensive black-box query is an experimental
access restriction; it is not intrinsic to this toy environment. The learned
method also constructs all candidate distances, which costs quadratic space in
the number of states. A complete finite catalogue, exact labels, deterministic
stationary total dynamics, and trusted resettable oracle access remain required.

Only three model seeds and six base layouts were used; rotations are correlated.
This start-state experiment is not directly comparable to the previous radius
experiment's all-state accuracy denominators. No unseen-world certificate,
general CTL equivalence, scalability result or statistical significance follows.

The next justified study is whether the savings survive when concrete checks
are genuinely expensive and the baselines use equally informative guidance.
Predeclare a new suite and compare JEPA-guided direct concrete search, a geometry
or domain heuristic, and a simpler learned transition predictor. Those controls
would separate the value of the learned ranking from the value of this CEGAR
loop. For safety-proof acceleration, additional sound structural constraints
are needed; this run supplies no evidence that ranking alone achieves it.

## Reproduce and inspect

Use the unchanged three checkpoints from the
[earlier reproducibility instructions](latent_radius_stress_results.md#reproduce-and-inspect).
If they already exist, no training is required. With the normal editable install:

```powershell
& .\.venv\Scripts\python.exe experiments\cegar_refinement.py --checkpoints outputs/latent_radius/oracle/model.pt outputs/latent_radius/stress_training/seed_20260805/model.pt outputs/latent_radius/stress_training/seed_20260806/model.pt --output-dir outputs/refinement/cegar
& .\.venv\Scripts\python.exe experiments\audit_cegar_refinement.py --run-dir outputs/refinement/cegar --checkpoints outputs/latent_radius/oracle/model.pt outputs/latent_radius/stress_training/seed_20260805/model.pt outputs/latent_radius/stress_training/seed_20260806/model.pt
```

Choose a fresh output directory for a new experiment. On Linux replace the
interpreter with `python` and script separators with `/`. This run used Python
3.12.14, PyTorch 2.14.0+cpu and four CPU threads. The Windows commands have not
been executed on Windows; other runtimes may change rankings through numerics.

The [aggregate summary](results/cegar/summary.json) records all seed/property/
family metrics, the decision, model/source/export hashes and bounded audit.
The complete `report.json`, 864 task rows, 864 full query traces, audit output
and validation record are delivered separately as
`jepa-lmc-cegar-refinement-records.zip`. Raw traces and checkpoint weights are
not uploaded to the public repository. Local runs generate the same formats.

Validation: **90 unittest cases, zero failures, two existing external nuXmv
checks skipped** because the executable is unavailable. Ruff passes on all new
Python files. Tests include adversarial priorities, false-witness replacement,
the original wall jump, zero-budget behavior, exact threshold arithmetic, and
all topology families. The bounded replay audit also recomputes aggregate
metrics and the screening decision from the saved task CSV, and checks source
and export checksums.
