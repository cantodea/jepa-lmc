# Bounded Top-1 model-quality study

This separate study pauses new abstraction work and asks whether modest training
changes can lift JEPA Top-1 accuracy to 99–100%, and whether a validation-selected
improvement also improves behavioral and temporal fidelity. Previous checkpoints
and formal experiment outputs are preserved. The learning, decoder, verification
and environment source modules are unchanged.

The [fixed protocol](../configs/top1_quality_protocol.json) limits the study to
three configurations and three seeds, with no subsequent tuning. The complete
results and decision will be recorded here when these runs finish.

## Baseline diagnosis

All transitions are enumerated, including unreachable states. Training uses
exactly the original 40 maps / 4,736 transitions. Validation uses the original
20 held-out maps / 2,404 transitions. The 24 topology stress maps contribute
2,976 transitions per seed. No stress or validation transitions are trained on.
The 50 pilot test maps remain unscored in this study.

| Seed | Train | Original validation | Stress | Stress errors |
| --- | ---: | ---: | ---: | ---: |
| 20260804 | 94.57% | 92.80% | 95.09% | 146 |
| 20260805 | 94.47% | 92.10% | 92.07% | 236 |
| 20260806 | 96.01% | 93.72% | 95.43% | 136 |

Three rotations of each validation map are additional diagnostics, never a
selection criterion. Original training goals occupy one corner, whereas the
stress suite rotates goals. This is a possible distribution difference; the
diagnostic does not assume it explains the errors.

Exhaustive float32 `torch.cdist` decoding is compared with independent float64
direct subtraction on the same embeddings. All 51,984 transition predictions
agree. Target-embedding self retrieval succeeds on all 12,996 checked map states;
there are no target-state collisions. Stress Top-2 accuracy is 99.90%, 99.76%,
and 100%. The dominant issue is therefore which nearby target wins, rather than
an approximate search or a floating-point retrieval bug. Position-only and
learned-only distance ablations are reported as diagnostics; neither replaces
the decoder or enters model selection.

Training boundary-blocked actions have zero errors for all three seeds.
Wall-blocked actions have 207/729, 100/729, and 74/729 errors; free moves have
50/3,196, 162/3,196, and 115/3,196 errors. At epoch 100 the original loss still
decreases, motivating the bounded epoch/LR/capacity tests. These observations
alone do not separate optimization, representation, and generalization limits.

## Fixed candidate budget and selection

| Configuration | Initialization | Architecture | Epochs | Learning rate after epoch 100 |
| --- | --- | --- | ---: | --- |
| Baseline | Existing saved weights | latent 32, encoder width 32, predictor 64 | 100 | Original 3e-4 |
| `long_constant` | Corresponding baseline | Same | 100 + 200 | 3e-4 |
| `long_cosine` | Corresponding baseline | Same | 100 + 200 | Cosine 3e-4 to 3e-5 |
| `capacity_cosine` | Fresh, same fixed seed | latent 64, encoder width 48, predictor 128 | 300 | Same cosine schedule |

All three use exactly the same training examples, batch size 512, AdamW weight
decay 1e-4, EMA 0.99, and original JEPA loss. Observations are cached to reduce
data-loading cost; no examples are added. No augmentation is used.

The saved baseline bundles lack optimizer and RNG state, so the continuations
explicitly restart AdamW and the data-loader generator at epoch 100. They are
warm starts, not exact resumptions of the old run. The fresh capacity control
also restarts these at epoch 100. Learning rates are constant through epoch 100,
then follow their declared schedules over epochs 101–300.

Only epoch 300 is eligible for a candidate. Intermediate validation at epochs
100, 150, 200 and 250 diagnoses convergence, without selecting checkpoints.
One configuration is selected for all three seeds by mean original-validation
accuracy, including baseline as an eligible option. Ties prefer fewer trainable
parameters and then the listed order. Clear improvement requires at least one
percentage point in mean validation and a gain on every seed. The 99% target is
reported separately for validation and stress. Previously inspected stress maps
are a diagnostic holdout, not a fresh blind test.

## Reproduce in a separate output directory

Clone or update `agent/oracle-latent-radius`, install the project, and extract the
private checkpoint/record archive into the repository root. Raw records and
weights are supplied separately; the repository contains code and aggregate
results. Preserve the original `outputs/latent_radius/...` checkpoint paths and
`outputs/behavioral_relations/top1` graph export. Use a fresh run directory:

```powershell
$run = "outputs/top1_quality/local_run"
python experiments/top1_quality_diagnosis.py --output-dir "$run/diagnosis"
foreach ($candidate in @("long_constant", "long_cosine", "capacity_cosine")) {
    foreach ($seed in @(20260804, 20260805, 20260806)) {
        python experiments/improve_top1.py --run-dir $run --candidate $candidate --seed $seed
        if ($LASTEXITCODE -ne 0) { throw "Training or evaluation failed" }
    }
}
python experiments/improve_top1.py --run-dir $run --select
```

The reports distinguish train, original validation, rotated validation and
stress. Each contains exhaustive state-action predictions, map/family/action/
motion summaries, retrieval margins, loss histories, and checkpoint hashes.
Model bundles retain the original `ActionJEPA(**payload['model_kwargs'])` loading
interface and additionally save optimizer and Torch/data-loader RNG state.
They are compatible with the configurable checkpoint path in
`oracle_latent_radius.py`; historical fixed-checkpoint protocols remain strict.

For the selected model, the behavior runner uses the existing greatest-relation
algorithms, elimination-certificate audits, independent partition refinement,
and six CTL formulas. Baseline replay must reproduce every original graph and
entire relation record. The same graph exports feed the existing six-formula
nuXmv LTL runner and strict CTL/paired-formula sanity checker, with no fairness:

```powershell
$env:NUXMV_BINARY = "C:/path/to/nuXmv.exe"
foreach ($model in @("baseline", "best")) {
    python experiments/quality_behavioral.py --run-dir $run --model $model
    python experiments/evaluate_top1_ltl.py --run-dir "$run/behavioral/$model" --output-dir "$run/temporal/${model}_ltl"
    python experiments/backend_sanity.py --run-dir "$run/behavioral/$model" --output-dir "$run/temporal/${model}_sanity"
}
python experiments/summarize_top1_quality.py --run-dir $run
```

Only claims about the enumerated finite graphs follow from these checks.
Higher average Top-1 accuracy alone does not imply simulation, bisimulation,
monotone temporal improvement, or a certified local uncertainty bound. This
round does not construct or tune a new abstraction.
