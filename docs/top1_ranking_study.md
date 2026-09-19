# Bounded ranking and predictor-depth study (round 2)

This user-authorized second round preserves the original checkpoints and all
round-1 results. It tests a concrete objective mismatch: the baseline minimizes
latent regression error, while evaluation chooses the closest same-map target.
In the frozen baseline, 508 of 518 stress errors have the true successor ranked
second. Direct float64 and canonical float32 retrieval arithmetic agreed; this
motivates training the ordering, not replacing the decoder with an oracle.

The [fixed protocol](../configs/top1_ranking_protocol.json) predeclares exactly
three configurations and three seeds. All warm-start the original epoch-100
weights, restart AdamW, use a cosine learning rate from 1e-4 to 1e-5 over at most
150 additional epochs, and retain the same 40 training maps (4,736 transitions).
Validation is checked every five epochs. Each seed retains its best validation
checkpoint, including epoch zero; eight checks without improvement trigger early
stopping after at least 40 epochs. A single configuration is then selected across
all three seeds using mean original-validation accuracy, with baseline eligible.
Selection is frozen **before** any new stress evaluation. The previous round's
final-epoch-only rule is not reused.

| Candidate | Objective | Predictor change |
| --- | --- | --- |
| `earlystop_control` | Original JEPA loss | None |
| `ranking` | Original loss + 0.1 ranking CE | None |
| `ranking_depth` | Same ranking objective | One residual hidden block |

For prediction $p_i$, detached EMA targets $z_j$, and true successor index $t_i$:

$$
\ell_{rank}(i)=-\log\frac{\exp(-\|p_i-z_{t_i}\|_2^2/\tau)}
{\sum_{j\in S_{map(i)}}\exp(-\|p_i-z_j\|_2^2/\tau)},\qquad \tau=1.
$$

The source state is a candidate, and negatives are all other valid states from
the same **training** map. Targets are detached; context encoder and predictor
receive gradients, and the target encoder keeps its original EMA update. This
adds a contrastive ranking term to JEPA; it is not the original negative-free
objective. The four-channel observation, two-convolution encoder, latent size
32, fixed position features, action embedding and full-latent Euclidean decoder
stay unchanged. The new hidden block is LayerNorm/Linear/GELU/Linear with a
zero-initialized final layer inside a residual connection. It initially preserves
the baseline function exactly. Default model arguments retain old checkpoint keys.

Training groups four complete maps per batch, with ten optimizer updates per
epoch and every original transition exactly once. Shared states are encoded once
per batch, then their contexts are repeated for four actions. All candidates use
this batching, including the control; ranking comparisons are therefore matched.
The control differs from the historical baseline continuation in batching,
learning-rate budget and validation selection, so it isolates the new objective
only within this round. There is no data augmentation or test-transition training.
Stress remains an already-inspected diagnostic holdout, not a new blind test.

## Reproduce

Install the project in a fresh checkout and extract the private record archive
there to restore the original checkpoints. Use a fresh output directory for any
new run; commands refuse to overwrite completed results. From the repository root:

```bash
python -m pip install -e .
python -m experiments.improve_top1_ranking --run-dir outputs/top1_quality/round2_local --phase initialize
```

For each candidate above and each seed `20260804`, `20260805`, `20260806`, run:

```bash
python -m experiments.improve_top1_ranking --run-dir outputs/top1_quality/round2_local --phase train --candidate ranking --seed 20260804
```

After all nine training runs, freeze selection once, then evaluate every saved
validation-best model using the same candidate/seed combinations:

```bash
python -m experiments.improve_top1_ranking --run-dir outputs/top1_quality/round2_local --phase select
python -m experiments.improve_top1_ranking --run-dir outputs/top1_quality/round2_local --phase evaluate --candidate ranking --seed 20260804
python -m experiments.quality_behavioral --run-dir outputs/top1_quality/round2_local --model best
```

The existing `backend_sanity.py` and `evaluate_top1_ltl.py` accept the resulting
`outputs/top1_quality/round2_local/behavioral/best` graph directory. No formula,
label, graph semantics, fairness or verification algorithm is changed here.

Results will be recorded after completing the predeclared runs.
