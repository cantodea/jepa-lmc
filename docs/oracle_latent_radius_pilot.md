# Recorded pilot: oracle radius and validation transfer

This is a new run of the repository's unchanged training recipe, not a recovered
historical checkpoint. Base revision: `7484a8f3629a5560c529d8adceef476b55b5019a`.
Training: 40 maps, 100 epochs, batch size 512, latent dimension 32, seed 20260804.
Runtime: CPU, four threads, Python 3.12.14, PyTorch 2.14.0+cpu. Other versions or
hardware can produce different numerical results.

Both evaluations use **the same checkpoint** and ten held-out maps (seeds
20260922–20260931): 309 states, 1,236 state-action pairs, and 1,854 CTL queries
per variant. Every state's four actions are enumerated. There are no skipped
CTL maps in this run.

See the [protocol](oracle_latent_radius.md) for definitions and Windows commands.
The [machine-readable results](results/oracle_latent_radius_pilot.json) preserve
full-precision summaries, per-property confusion counts, map relation audits,
training provenance and source/output hashes. Large raw outputs and the model
are generated locally by the experiment rather than checked into Git.

## Oracle radii: fitted on the test errors

| Variant | Epsilon | Successor coverage | Mean candidate size | Singleton fraction | CTL agreement | Primary balanced CTL |
|---|---:|---:|---:|---:|---:|---:|
| Top-1 | — | 94.26% | 1.000 | 100.00% | 94.66% | 68.51% |
| 95th percentile | 3.364647 | 94.98% | 1.532 | 47.41% | 99.35% | 73.48% |
| 99th percentile | 4.270068 | 98.95% | 2.471 | 13.19% | 98.81% | 61.62% |
| Maximum | 5.098717 | 100.00% | 3.789 | 2.02% | 98.81% | 61.62% |
| All states | — | 100.00% | 31.065 | 0.00% | 98.81% | 61.62% |

The 95th/99th percentile names describe the interpolated thresholds. Finite
sample coverage need not equal exactly 95%/99%.

| Variant | Maps with action inclusion | Maps with unlabelled inclusion | One-sided violations / claims | Queries with an audited transferable verdict |
|---|---:|---:|---:|---:|
| Top-1 | 0/10 | 1/10 | 87/153 | 5/1,854 (0.27%) |
| 95th percentile | 0/10 | 8/10 | 0/66 | 52/1,854 (2.80%) |
| 99th percentile | 3/10 | 10/10 | 0/56 | 56/1,854 (3.02%) |
| Maximum | 10/10 | 10/10 | 0/56 | 56/1,854 (3.02%) |
| All states | 10/10 | 10/10 | 0/56 | 56/1,854 (3.02%) |

For all three radius variants, empirical one-sided soundness is 100% over their
respective claims. At the 95th percentile, inclusion fails on two maps, so the
zero observed violations do not establish a transfer guarantee there.

The maximum radius uses only about **12.2% of the candidate universe per pair**
on average. It is therefore a nontrivial transition overapproximation. However,
only **25 of 1,236** action pairs retain a single candidate.

The maximum-radius, 99th-percentile and all-states controls yield the same
truth-value vector on this suite. Their high 98.81% agreement is not evidence of
useful temporal precision. For example:

- `EF goal` is already true at 305/309 exact states; an always-True answer looks
  98.71% accurate.
- `AG !danger` is true at just 4/309 exact states. All four positives are lost at
  the maximum radius; reporting zero false-safe hides that loss of proving power.
- The 56 transferable verdicts at the maximum radius are label-immediate:
  `AF goal` at the ten goal states, `EG safe` False at the 23 danger states, and
  `E[!danger U goal]` False at those 23 danger states. This pilot demonstrates no
  additional nontrivial temporal conclusion from that radius.

## Validation radii: fitted without test errors

Twenty fixed validation maps (seeds 20260902–20260921) supply the radii. Training
and the ten test maps remain identical to the oracle run.

| Radius | Epsilon | Test successor coverage | Mean candidate size | Singleton fraction | Maps with action inclusion | Maps with unlabelled inclusion |
|---|---:|---:|---:|---:|---:|---:|
| 95th percentile | 3.726978 | 96.60% | 1.902 | 28.88% | 0/10 | 10/10 |
| 99th percentile | 4.241027 | 98.87% | 2.447 | 13.75% | 3/10 | 10/10 |
| Maximum | 5.028475 | 99.92% | 3.638 | 2.91% | 9/10 | 10/10 |

The validation maximum misses **one** of 1,236 test action successors. This
directly demonstrates why a maximum observed error is not a guaranteed global
bound. Unlabelled inclusion still passes because the same source-target edge is
present under another action. That distinction matters if later work reasons
about a specific action or policy.

The validation 95th-percentile graph retains 66 audited transferable verdicts
(3.56%) and a 73.48% primary balanced score. It is empirically more precise here,
but choosing it after inspecting these test results would itself be test-driven
tuning. It is not a certified radius for future environments.

## Decision supported by this run

**Continue investigating small set-valued successors, but do not claim the Yang
guarantee has been obtained.** Geometry is encouraging: a worst-error oracle ball
reduces the candidate universe substantially. Temporal proving power is weak:
the fully covered graph loses the interesting positive safety cases, and all
its transferable verdicts in this suite follow immediately from labels.

Before attempting a general guaranteed-bound construction, a useful next
experiment would keep this model frozen and use preselected maps/properties with
both true and false nontrivial safety/reachability cases. The error bound would
then still need separate justification on the intended domain. This single-seed
pilot does not establish unseen-map coverage, equivalence or bisimulation.

## Verification of the implementation

- Existing baseline: 62 tests, two external nuXmv tests skipped (binary absent).
- With the new module: 73 tests, the same two skips; no failures.
- New cases cover exact reconstruction, quantile boundaries, latent collisions,
  empty sets, missing edges, both CTL transfer directions, aggregation, and frozen
  model parameters/modes with a deliberately different target encoder.
- Checkpoint replay reproduced all radii and summary values exactly. Exported
  error/candidate records independently reproduce coverage, mean size and singleton
  fractions; both result bundles match the final evaluator's source hash.
- Ruff passes for all added Python files. No learning/model/training files changed.

Tests and experiments ran on Linux CPU. Windows commands are provided but were
not executed in a Windows runtime. External nuXmv is not required by this experiment.
