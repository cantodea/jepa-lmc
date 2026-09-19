# Topology stress result: stop the current uniform-radius route

The prespecified decision is **STOP**. Seeds 20260804 and 20260806 pass; seed
20260805 fails the required consistency across topology families. Following the
agreed stopping rule, this project should stop investing in Yang-style uniform
latent-radius certification for the current JEPA. Keep the experiment and
conditional theorem as research results.

There is useful information in these candidates: all three seeds outperform the
all-states control, and one recovers every primary proof opportunity. The reason
to stop is insufficient consistency under the fixed screening rule. These data
do not establish that Yang 2025, or every possible JEPA abstraction, is impossible.

## What was fixed and run

The [protocol](latent_radius_stress_protocol.md) and its
[executable thresholds](../configs/latent_radius_stress_protocol.json) were fixed
before stress inference. No threshold, map or model seed was selected from these
results. Each of three frozen checkpoints uses the same 40 training maps, 100
epochs, batch size 512, latent dimension 32 and original training code. No encoder,
predictor, loss or training implementation changed. Evaluation checks the model
state digest before and after inference.

The 24 maps contain sealed safe regions, dangerous mandatory gates and safe
detours: two layouts and four rotations per family. For each checkpoint we
enumerate 744 states and **2,976 state/action pairs**. One radius is pooled over
all 24 maps. The candidate catalogue includes every valid state in the same map.
There is no topology filter, truncation or edge repair.

Each variant has 4,464 CTL queries across six formulas. The primary score uses
2,136 queries after removing immediate source-label cases: `AG !danger` (720),
`EF goal` (720), and `E[!danger U goal]` (696). Its balanced score is the mean of
the three per-property balanced accuracies. The exact graph supplies **480**
non-immediate universal-True / existential-False proof opportunities: 360 in
sealed regions and 120 at dangerous gates. A recovered proof additionally
requires the complete unlabelled transition-inclusion audit on its map.

## The four requested measurements

Coverage and singleton rates use all state/action pairs. Precision below is
`TP/(TP+FP)` over all six CTL formulas, including source-label base cases. The
primary decision table in the next section excludes those base cases.

| Seed | Radius | Epsilon | Successor coverage | Mean candidate size | Singleton rate | CTL positive precision | One-sided violations |
|---|---|---:|---:|---:|---:|---:|---:|
| 20260804 | 95th | 2.971292 | 94.9933% | 1.2258 | 67.7419% | 100.00% | 0 |
| 20260804 | 99th | 4.177052 | 98.9919% | 2.3810 | 15.6922% | 97.28% | 0 |
| 20260804 | Maximum | 4.763255 | 100% | 3.0397 | 8.7030% | 92.16% | 0 |
| 20260805 | 95th | 3.109665 | 94.9933% | 1.3044 | 60.7191% | 100.00% | 18 |
| 20260805 | 99th | 3.894577 | 98.9919% | 2.1032 | 22.8159% | 97.28% | 0 |
| 20260805 | Maximum | 5.291539 | 100% | 4.0215 | 1.9825% | 88.03% | 0 |
| 20260806 | 95th | 3.102031 | 94.9933% | 1.3797 | 53.9651% | 100.00% | 0 |
| 20260806 | 99th | 3.615674 | 98.9919% | 1.9368 | 27.8898% | 100.00% | 0 |
| 20260806 | Maximum | 3.978182 | 100% | 2.3011 | 17.5739% | 100.00% | 0 |

At the maximum, all three seeds have **24/24 action-labelled inclusions and
24/24 unlabelled inclusions**, no empty action sets, and zero one-sided CTL
violations. Empirical one-sided soundness is 100%. Maximum-radius coverage is
an oracle consistency check: the same enumerated errors define the radius.

Quantiles omit 149 (95th) and 30 (99th) true action successors per seed. Unlabelled
inclusion still holds on respectively 22/17/23 maps at the 95th radius, and 24
maps for every seed at the 99th radius. Another action can supply a missing edge
in ordinary CTL. This does not restore action-labelled inclusion. The 18 one-sided
violations at seed 20260805's 95th radius occur without complete inclusion; a high
positive precision alone does not establish soundness. All variants are total
as unlabelled graphs here; no maps were skipped.

## Decision and controls

| Seed / control | Primary balanced score | Gain over all-states | Recovered proofs / 480 | Sealed maps with a proof / 8 | Dangerous-gate maps with a proof / 8 | Gate |
|---|---:|---:|---:|---:|---:|---|
| All states, every seed | 50.00% | 0 pp | 0 | 0 | 0 | Negative control |
| Exact graph, every seed | 100.00% | 50 pp | 480 | 8 | 8 | Positive control |
| 20260804, maximum | 77.50% | 27.50 pp | 264 (55.00%) | 5 | 5 | Pass |
| 20260805, maximum | 65.42% | 15.42 pp | 138 (28.75%) | **3** | **1** | **Fail** |
| 20260806, maximum | 100.00% | 50 pp | 480 (100.00%) | 8 | 8 | Pass |

The all-states control has mean candidate size 31.0215. Candidate sets at the
maximum are much smaller for all seeds. All seeds meet the 10-percentage-point
balanced gain and 10% proof-recovery thresholds. The failing seed does not meet
the separately fixed requirement of **at least 4/8 maps in each required family**.
The overall gate requires every seed to pass. Pooling seeds, choosing the best
checkpoint or substituting a quantile would change that rule after seeing results.

The primary per-property scores and proof counts show the limitation:

| Maximum-radius seed | AG !danger balanced / recovered | EF goal balanced / recovered | E[!danger U goal] balanced / recovered |
|---|---|---|---|
| 20260804 | 77.50% / 66 | 77.50% / 66 | 77.50% / 132 |
| 20260805 | 67.50% / 42 | 67.50% / 42 | 61.25% / 54 |
| 20260806 | 100.00% / 120 | 100.00% / 120 | 100.00% / 240 |

For seed 20260805, recovered proofs are 126/360 in sealed regions and 12/120 at
dangerous gates. A count of zero opportunities for the safe-detour family is
expected: its positive reachability and negative universal safety verdicts do
not transfer from an overapproximation. Those cases still contribute to Boolean
scoring. All six formulas match exact truth for seed 20260806 on this suite;
agreement on six formulas does not establish arbitrary CTL equivalence.

## A concrete failure

In `danger_gate_layout0_rot0`, the wall's only opening is the dangerous cell
`(1,2)`. On the exact graph, the start `(0,0)` cannot reach goal `(5,5)` without
danger: `E[!danger U goal]` is False.

At seed 20260805's maximum radius, a candidate path is

\[
(0,0)\to(0,1)\to(0,3)\to(2,4)\to(4,4)\to(5,5).
\]

This path avoids danger. In particular, action **right** from `(0,1)` really
stays at `(0,1)` because `(0,2)` is a wall. Nevertheless `(0,3)` is included:

\[
d(\hat z((0,1),\mathrm{right}),E_t(o_{(0,3)}))
=5.165100\leq\epsilon_{\max}=5.291539.
\]

The true successor error for that pair is 3.311397, so both successors are
included. This creates an existential True that cannot transfer to the exact
graph. The failure is a loss of proving power, consistent with zero violations
of the correct one-sided theorem. A sealed-region counterexample and both
paths' complete coordinates are retained in the machine-readable summary.

For the same frozen network and uniform-ball construction, a valid bound on a
domain containing these inputs cannot be smaller than their maximum error.
Increasing the radius preserves this spurious edge and path. Certification of
a larger uniform radius therefore cannot fix this observed failure for this
checkpoint. The [conditional theorem](latent_radius_conditional_theorem.md)
states the assumptions and proves transition inclusion and one-sided CTL transfer.

## Scope and research action

Stop the current Yang-style uniform-radius route under the agreed screen. Do not
add certification machinery or retune this suite to rescue the decision. Retain
the reusable oracle evaluator, the conditional theorem and these mixed but
insufficiently stable results.

This is an engineering screening decision, not a statistical rejection or an
impossibility theorem. There are only three model seeds and six base layouts;
rotations are correlated. The training pilot fixes start/goal at opposite corners,
whereas three stress rotations also move the goal. The experiment tests that
combined topology/orientation shift, and cannot isolate their causal effects.
All states and labels are known, the error maximum uses test truth, and numerical
distances are not certified real-arithmetic enclosures. No generalization error
bound, full implementation of Yang 2025, or bisimulation claim follows.

## Reproduce and inspect

After the normal editable installation, reuse an existing matching seed-20260804
checkpoint at `outputs/latent_radius/oracle/model.pt`. If it is absent, create it
with the first command. Train only missing checkpoints; each output directory
must be fresh. These commands preserve the original recipe.

```powershell
& .\.venv\Scripts\python.exe experiments\oracle_latent_radius.py --seed 20260804 --train-maps 40 --epochs 100 --output-dir outputs/latent_radius/oracle
& .\.venv\Scripts\python.exe experiments\oracle_latent_radius.py --seed 20260805 --train-maps 40 --epochs 100 --output-dir outputs/latent_radius/stress_training/seed_20260805
& .\.venv\Scripts\python.exe experiments\oracle_latent_radius.py --seed 20260806 --train-maps 40 --epochs 100 --output-dir outputs/latent_radius/stress_training/seed_20260806
& .\.venv\Scripts\python.exe experiments\stress_latent_radius.py --checkpoints outputs/latent_radius/oracle/model.pt outputs/latent_radius/stress_training/seed_20260805/model.pt outputs/latent_radius/stress_training/seed_20260806/model.pt --output-dir outputs/latent_radius/stress
```

On Linux, replace the interpreter path with `python` and use `/` in script paths.
The recorded run used Python 3.12.14, PyTorch 2.14.0+cpu and four CPU threads.
Other runtime versions can change numerical results; checkpoint and model-state
hashes are recorded. Checkpoints can be regenerated with the commands above and
are not committed to Git.

The [aggregate summary](results/latent_radius_stress/summary.json) contains the
exact radii, per-seed decisions, per-property/family scores and audit results.
The complete records are delivered separately to the requester as
`jepa-lmc-latent-radius-stress-records.zip`: every pair's error, every candidate
set, every CTL outcome, the original report, per-map results, failure paths,
checkpoint/source hashes and the test log. Raw per-query outputs are not uploaded
to the public repository. Running the commands above generates the same export
formats locally under the chosen output directory.

The archive includes SHA-256 checksums for every file. The recorded build ID is
local commit `61c58ba`; published commit `2ff70f9` has the identical source tree.

Validation: **80 unittest cases, zero failures, two existing external nuXmv
cases skipped** because the executable was absent. Ruff passes for added Python
files. Independent auditing verified 8,928 errors, 26,784 candidate records,
80,352 CTL records, quantiles, set nesting, inclusion and all primary aggregates.
An independent breadth-first search checked 66,960 exact/candidate primary truth
values from the exported graphs, covering the three radii and both graph controls.
The test log is included in the delivered archive. Execution was on Linux CPU;
the Windows commands were documented but not run on Windows.
