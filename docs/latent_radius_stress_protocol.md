# Fixed decision experiment

This protocol is fixed before running the stress maps through JEPA. It is the
last screening experiment for the current uniform-radius route, as requested.
The executable thresholds are in
[`configs/latent_radius_stress_protocol.json`](../configs/latent_radius_stress_protocol.json).

## Cases and controls

There are 24 maps: three topology families, two layouts per family, four rotations
per layout. All are 6x6 and use the unchanged four-channel encoder input.

| Family | Construction | Exact truth at the start: AG !danger / EF goal / E[!danger U goal] |
|---|---|---|
| Sealed region | A wall separates the safe start component from goal and danger | True / False / False |
| Dangerous gate | The only passage through the wall is dangerous | False / True / False |
| Safe detour | A second safe passage bypasses the dangerous gate | False / True / True |

The layouts use `(wall column, dangerous gate row)` equal to `(2, 1)` and `(3, 4)`.
Each layout is rotated by 0, 90, 180 and 270 degrees. Rotations are related cases,
not independent statistical samples. Start, goal and hazards rotate with the map.
We score every valid state, not just the start. The start truth table is checked
using the exact checker before any neural inference. Cases are not selected by
model performance or removed after failures.

The three models use seeds 20260804, 20260805 and 20260806 with the original
40-map, 100-epoch recipe. The first checkpoint is reused from the previous pilot;
the other two use identical architecture, training maps, optimizer settings and
budget. None trains on the stress maps. Each model is frozen before evaluation.

For each checkpoint, the primary radius is the maximum error over all pairs of
all 24 maps. The two quantiles and Top-1 are diagnostics only: they cannot rescue
a failing maximum-radius result by dropping coverage. The all-states control is
the uninformative superset; the exact graph is the positive control. There is no
extra training, grid-distance filtering, edge repair or per-state radius tuning.

## What counts as useful?

The primary properties are `AG !danger`, `EF goal` and `E[!danger U goal]`.
Remove source-label base cases before scoring: goal already makes `EF goal` and
the until formula true; danger already makes `AG !danger` and the until formula
false. A safe non-goal state's inability to reach the goal, or its ability to
avoid danger on every path, depends on topology and counts as non-immediate.

Report both Boolean fidelity and sound proving power. A **transfer opportunity**
is a non-immediate primary query whose exact truth is in the direction that can
transfer from an overapproximation: universal True or existential False. A
recovered opportunity must have that same verdict in the candidate graph and
pass the complete relation-inclusion audit. The all-states graph should recover
zero such opportunities on this suite; the exact graph must recover all of them.
This guards against counting a spurious existential True as a useful proof.

Continue only if **all three** seeds satisfy every condition:

- All 24 maps have full action successor coverage and unlabelled inclusion.
- No one-sided transfer violation occurs.
- The macro balanced accuracy over the three non-immediate primary property
  groups exceeds the all-states control by at least **10 percentage points**.
- At least **10%** of exact non-immediate transfer opportunities are recovered.
- At least **4 of 8 maps in each** of the sealed-region and dangerous-gate families
  recover at least one opportunity. Safe-detour cases test the positive reachability
  side; they are not expected to supply these one-sided proof opportunities.

The numerical thresholds are prespecified engineering stopping rules, not
claims of statistical significance. Results must include each seed, property
and family so that a pooled score cannot hide a failure. If any seed fails, stop
investing in the current Yang-style uniform-radius route and record the negative
result. The scope is these models and this construction, not every method in
Yang 2025 or every possible JEPA representation.
