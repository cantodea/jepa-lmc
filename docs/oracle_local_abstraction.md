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

On Windows use the virtual environment's `python.exe` and your installed
`nuXmv.exe`. The same states, labels, designated initial state and state-specific
query encoding are used for both logics. The executable is not redistributed.

Results and the continuation decision are recorded after the complete fixed run.
