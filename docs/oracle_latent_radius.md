# Oracle latent-radius experiment

This experiment asks whether the **existing frozen JEPA** can produce small
successor sets that contain the true successor, and whether those sets retain
useful CTL information. It changes no encoder, predictor, loss or training code.

## Run on Windows

From the repository root, after the normal editable installation:

```powershell
# Train once using the same defaults as evaluate_jepa.py, then evaluate.
& .\.venv\Scripts\python.exe experiments\oracle_latent_radius.py --output-dir outputs/latent_radius/oracle

# Reuse that exact checkpoint without training again.
& .\.venv\Scripts\python.exe experiments\oracle_latent_radius.py --checkpoint outputs/latent_radius/oracle/model.pt --output-dir outputs/latent_radius/replay

# Optional next diagnostic: choose radii on validation maps, then freeze them.
& .\.venv\Scripts\python.exe experiments\oracle_latent_radius.py --checkpoint outputs/latent_radius/oracle/model.pt --radius-source validation --output-dir outputs/latent_radius/validation
```

Defaults: 40 training maps, 100 epochs, 10 test maps, batch size 512, latent
dimension 32, seed 20260804, CPU with four threads. Map identities come from the
existing fixed pilot splits; changing a map count does not move split boundaries.
Validation mode uses 20 disjoint validation maps. Each run needs a fresh output
directory, to preserve previous results. The original repository contains no
trained checkpoint; the default command therefore trains one first.

The checkpoint bundle contains `model_state_dict`, `model_kwargs` and training
provenance. A plain compatible `state_dict` is also accepted, using the existing
default architecture and `--latent-dim`. For other architecture configurations,
supply the bundle with the original `model_kwargs`. Unknown training provenance
is explicitly reported; recorded training/evaluation map overlap is rejected.
Checkpoint evaluation never updates weights, the EMA target, or model modes.

## Precise protocol

For every **valid** state and all four actions on each test map:

\[
\hat z(s,a)=P(E_c(o_s),a),\qquad
e(s,a)=\|\hat z(s,a)-E_t(o_{T(s,a)})\|_2.
\]

Here `o` is the existing four-channel observation and `E_t` is the frozen EMA
target encoder. The distance is unnormalized Euclidean distance in the **whole**
latent vector, including the existing position subspace. No new normalization,
position scaling, coordinate filtering or nearest-neighbour repair is applied.
Distances are computed directly in float64 from frozen network outputs to reduce
cancellation near zero; this does not certify real-arithmetic numerical bounds.

Pool errors over all test pairs, weighting each pair equally, to compute
`epsilon_max`, `epsilon_95` and `epsilon_99`. Quantiles use linear interpolation.
For each one, use **one global scalar radius** on all test maps:

\[
C_\epsilon(s,a)=\{s'\in S_{\mathrm{same\ map}}:
\|\hat z(s,a)-E_t(o_{s'})\|_2\le\epsilon\}.
\]

The ball is closed, and ties are retained. Candidate states retain their original
identities and labels even if latent embeddings collide. All valid states are
candidates, including states unreachable from the initial state. This is a known
finite-state experiment, not discovery of an unknown state space.

**Oracle mode intentionally uses test truth to choose the radii.** In particular,
maximum-radius coverage on these enumerated pairs must be 100% by construction.
That is a consistency check, not independent evidence for a general error bound.
In validation mode, the same quantile function sees only validation errors; test
truth is subsequently used for scoring and relation audits, never radius fitting.

## Outputs and metrics

| File | Contents |
|---|---|
| `report.json` | Protocol, model provenance, runtime/source hashes, radii, pooled and per-map results |
| `errors.csv` | One error for every test pair; also validation errors in validation mode |
| `candidate_sets.jsonl` | Exact candidate state list for each pair at each of the three radii |
| `ctl_outcomes.csv` | Both truth values for each variant/map/state/property |
| `model.pt` | Reusable model bundle, written only when training |

The four requested measurements are:

1. **Successor coverage:** fraction of pairs with `T(s,a)` in the candidate set.
2. **Mean candidate size:** includes empty sets; also report size as a fraction
   of that map's candidate universe, to expose near-complete graphs.
3. **Singleton fraction:** fraction with exactly one candidate. Singleton
   precision separately checks how often that sole candidate is correct.
4. **CTL:** per-property and pooled confusion counts, agreement, positive
   precision, false-safe count, and the one-sided transfer diagnostics below.

`positive_precision = TP / (TP + FP)` measures positive Boolean predictions.
`false_safe_count` is specific to `AG !danger`; zero false-safe does not establish
soundness of the other formulas. `primary_balanced_score` is the mean of per-formula
balanced accuracies for the existing three primary properties (`EF goal`,
`E[!danger U goal]`, `AG !danger`), matching the existing evaluator's convention.
If a required denominator is zero, the metric is `null`, not a perfect score.

`top1` and `all_states` are controls using the same distance table, states and
labels. The former chooses one argmin (state-order tie break); the latter retains
every state. Their candidates are not additional radius choices.

Empty action sets are counted literally. Ordinary CTL uses the union of actions,
so a missing action does not necessarily make the state a deadlock. If any state
has **no successor at all**, the map's CTL results are marked `not_total` and
skipped. No edges are added. Aggregates show evaluated/skipped map counts and
actual query counts; comparisons across different evaluated maps need care.

## What relation has actually been checked?

The action-labelled relation is

\[
\hat R^A_\epsilon=\{(s,a,s'):s'\in C_\epsilon(s,a)\}.
\]

`action_inclusion` exhaustively checks `R^A` is a subset of this relation on that
map. `relation_inclusion` instead checks the **unlabelled** relation used by CTL:

\[
R\subseteq\hat R_\epsilon,\qquad
\hat R_\epsilon=\{(s,s'):\exists a,\ s'\in C_\epsilon(s,a)\}.
\]

Action coverage can be below 100% while unlabelled inclusion still holds: a true
edge can be supplied under another action. Neither relation implies the other
direction of inclusion. Missing and extra edges are reported. `relation_equal`
checks exact unlabelled equality, not a search for arbitrary bisimulations.

With fixed states, labels and initial states, inclusion gives an identity
simulation **from the exact graph into the candidate graph**. It supports these
directions for the existing property suite:

| Formula family | Verdict on candidate graph transferable to exact graph |
|---|---|
| `AG !danger`, `AF goal` | True |
| `EF danger`, `EF goal`, `E[!danger U goal]`, `EG safe` | False |

The opposite verdicts remain inconclusive using this overapproximation alone.
Thus a spurious existential True is compatible with a valid overapproximation.
The syntactic classifier handles monotone CTL fragments conservatively and does
not infer a transfer direction for mixed formulas such as `AG(EF goal)`.

`one_sided_claims` counts verdicts in those transfer directions.
`empirical_one_sided_soundness` is the fraction of such verdicts agreeing with
truth; `one_sided_decisive_fraction` divides claims by all evaluated queries.
These remain **empirical conditional claims** when relation inclusion fails.
`audited_transfer_claims` and `audited_transfer_fraction` count them only on maps
where the complete unlabelled inclusion audit passes. Their scope is those finite
graphs, not future maps. The CLI fails if a one-sided violation occurs despite
audited inclusion, or if an oracle maximum misses a true action successor.

## Connection to Yang 2025

The target reference is Yang, Wang and Xiang, *Neural transition system
abstraction for neural network dynamical system models and its application to
Computational Tree Logic verification*, Neural Networks 186, 107261,
[DOI](https://doi.org/10.1016/j.neunet.2025.107261).

The openly available
[2024 predecessor, Proposition 1](https://arxiv.org/html/2402.11739v1)
explicitly connects transition correspondence to a **guaranteed** model error
bound and a geometric margin within a target partition. Its set-valued neural
reachability calculation and the error between learned and real dynamics play
different roles. This implementation is a feasibility diagnostic; it does not
implement or claim to verify the 2025 paper's complete theorem conditions.

For this project, the relevant discrepancy is between
`P(E_c(o_s),a)` and `E_t(o_T(s,a))`. A bound in physical state coordinates cannot
simply be substituted with an empirical latent radius. Certifying the neural
predictor's own output set also does not bound its error against unknown dynamics.

The experimental decision has two parts:

- If even the oracle maximum needs most candidate states, or loses useful CTL
  distinctions, an eventual certified bound is unlikely to help this construction.
- If candidates stay small and useful distinctions survive, developing a bound
  valid on the intended domain is a worthwhile next question. A validation-radius
  run tests transfer empirically but still supplies no universal bound.

Singletons plus coverage identify exact successors on those pairs. A good
singleton percentage alone proves neither action equivalence nor bisimulation.
No architecture changes are needed to run this diagnostic or inspect its failure
cases. See [the recorded pilot](oracle_latent_radius_pilot.md) for one actual run.
