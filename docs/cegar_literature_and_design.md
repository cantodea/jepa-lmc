# CEGAR for JEPA-LMC: literature and a bounded feasibility experiment

The literature supports a **conditional pivot to property-directed transition
refinement**. The indispensable component is a trustworthy way to check concrete
transitions or paths. CEGAR does not infer the real transition relation from a
neural prediction. In this repository, the exact GridWorld simulator supplies
that component, with every successor query counted explicitly.

The branch now studies whether a frozen JEPA can reduce those queries. The
uniform-radius result remains a historical baseline; its certification route
remains stopped. No JEPA architecture or training change is needed for this pivot.

## What the primary literature actually supports

| Source | Relevant result | Boundary for this project |
|---|---|---|
| Clarke, Grumberg, Jha, Lu and Veith, CAV 2000; expanded JACM 2003 | Start from an overapproximation, check a specification, analyze an abstract counterexample against the concrete system, and refine the abstraction when necessary | Concrete-system semantics are available for the feasibility check; neural prediction disagreement is insufficient |
| Vinzent, Sharma and Hoffmann, AAAI 2023 | Predicate abstraction and CEGAR for neural action policies; distinguishes transition-induced and policy-induced spurious paths; also evaluates heuristic search and incremental computation | Environment operators have known guards and updates. The NN selects actions; it is not an unknown-dynamics world model |
| Peled, Vardi and Yannakakis, Black Box Checking, 1999 | Combine experiments with model checking when the implementation's structure is initially unknown | Learning/checking requires an interaction model and assumptions; it does not make finite observations an unconditional safety proof |
| Pellen et al., author-hosted 2026 manuscript, Quick Bug Detection through Black-Box Checking | Studies checking intermediate learned hypotheses and measures queries needed to find bugs | Bug-finding efficiency and proof of absence of bugs are separate; its protocol-learning setting differs from a finite, fully observable GridWorld |

Primary sources:

- [CAV 2000 paper](https://orna.cswp.cs.technion.ac.il/wp-content/uploads/sites/72/2022/12/CAV00-automatic-abstraction.pdf), [expanded JACM 2003 paper](https://www.cs.cmu.edu/~emc/papers/Papers%20In%20Refereed%20Journals/Counterexample-guided%20abstraction%20refinement.pdf), DOI [10.1145/876638.876643](https://doi.org/10.1145/876638.876643).
- [AAAI 2023 publication](https://ojs.aaai.org/index.php/AAAI/article/view/26772), [author PDF](https://fai.cs.uni-saarland.de/vinzent/papers/aaai23.pdf), especially sections 2, 4, 5 and 6; [proof supplement](https://fai.cs.uni-saarland.de/vinzent/papers/aaai23-tr.pdf).
- [Black Box Checking](https://link.springer.com/chapter/10.1007/978-0-387-35578-8_13).
- [Quick Bug Detection through Black-Box Checking, author manuscript](https://sws.cs.ru.nl/publications/papers/fvaan/PellenEtAlASE26.pdf).

The following design and guarantees are derived for our finite deterministic
setting. They are not a reproduction of the AAAI algorithm, a claim to new CEGAR
theory, or a claim that the cited systems directly support JEPA.

## What changes mathematically

Let the complete finite state catalogue `S`, nonempty action set `A`, initial
state and exact labels be known. The unknown dynamics is a fixed total function
`T: S x A -> S`. A trusted oracle can return `T(s,a)` for a requested concrete
state/action. Direct access/reset to such a state is assumed; resetting and
replaying an episode are not free capabilities in a general physical system.

Maintain an upper successor set `C_k(s,a)` and observed edges `R_k^-`. Initially,

\[
C_0(s,a)=S,\qquad R_0^-=\varnothing.
\]

Every unqueried action remains completely unknown. After querying exactly one
pair and obtaining `t=T(s,a)`, replace only its upper set:

\[
C_{k+1}(s,a)=\{t\},\qquad
R_{k+1}^-=R_k^-\cup\{(s,a,t)\}.
\]

All other pairs keep their sets. Let `R_k^+` contain the action-labelled candidate
edges. Induction immediately gives

\[
R_k^-\subseteq R^A\subseteq R_k^+,\qquad
R_{k+1}^-\supseteq R_k^-,\quad R_{k+1}^+\subseteq R_k^+.
\]

Base case: total dynamics has a successor in the complete catalogue. Inductive
step: an exact deterministic query identifies that pair's unique successor,
which remains in the upper set and is added to the lower relation. No other pair
changes. This proof uses the oracle assumptions, not a bound on JEPA error.

For an unqueried pair, every target remains possible even if its neural score is
poor. Thus incorrect rankings can cost queries but cannot invalidate inclusion.
The all-states initialization is deliberately imprecise; the experiment measures
how much trusted information must be acquired before a particular property is
decided. It does not count the initial all-states inclusion as an achievement.

## Which CTL verdicts this prototype supports

The initial scope is the existing three primary properties, all reducible to
finite reachability. Source labels are exact and stay fixed.

| Property | Confirmed concrete path proves | Absence of a path in the upper graph proves |
|---|---|---|
| `AG !danger` | False, via a path to danger | True |
| `EF goal` | True, via a path to goal | False |
| `E[!danger U goal]` | True, via a path to goal with safe preceding states | False |

The observed lower graph may have deadlocks. It is used only for finite path
evidence; it is never fed into a total-Kripke CTL checker by adding self-loops.
The upper graph is total because every action set is nonempty. Arbitrary mixed
CTL, liveness lassos, hidden state and nondeterministic successor sampling are
outside this prototype. One sampled outcome cannot justify singleton refinement
for a nondeterministic action.

If an upper path has no confirmed counterpart, validate its actions from the
start, using cached exact transitions when available. Query the first unknown
pair, refine that pair, and continue only while the candidate prefix matches.
Re-run checking at the first mismatch. Validate a new path after each mismatch;
removing one witness need not remove another route to the same target.

Every continuing round queries at least one previously unknown pair. With no
budget limit there are at most `|S||A|` such queries. At that point the upper and
lower action relations equal the real relation. All returned True/False verdicts
are sound under the assumptions; a budget can instead yield `unknown`.

With state identities fixed, matching all labelled edges suffices to realize a
path. General predicate abstraction merges concrete states: individually feasible
abstract edges can still form an infeasible combined path. Applying this simple
edge check to such merged states would be invalid.

## The old false edge

At `(0,1)` with action right, a trusted query returns `(0,1)` because of the wall.
Refinement removes `(0,3)` from that **action's** candidate set while retaining
the true successor. It says nothing about unqueried actions, so the algorithm
checks for remaining witnesses again. The neural distance alone cannot authorize
this removal. If no trusted local verifier or oracle is available, this step
cannot support a safety claim.

## JEPA's role and the experiment

JEPA supplies a ranking of same-map successor candidates, computed from frozen
predictions and target encodings. It sees the known state observations and labels,
but no true successor queries or test errors during ranking. The default unknown
edge cost is `1 + rank`; known edges cost 1. A minimum-cost upper path is chosen
for validation. These positive costs are search priorities, not probabilities,
error bounds, certified transitions or physical path lengths.

The [fixed protocol](../configs/cegar_protocol.json) reuses the 24 topology maps
and three checkpoints. Each property starts with a fresh query cache at the map's
start state: 72 independent verification tasks per seed/method. Compare:

1. Uniform-cost CEGAR, without JEPA.
2. CEGAR with the fixed JEPA rankings.
3. CEGAR with each ranking row shuffled, testing the value of the learned order.
4. Direct, on-demand exact BFS with the same oracle access and action order.

Exhaustively reading `|S||A|` successors is a reference budget. Count exact
queries, resolved tasks at 10/25/50/100% of that budget, invalid conclusions,
refinement rounds, spurious paths, remaining candidate size and elapsed time.
Report neural preparation time separately. The fixed exploratory support rule
requires correctness and inclusion, plus at least 20% fewer total queries than
both uniform CEGAR and BFS on **each** seed, and fewer than shuffled rankings.
This is an engineering screen, not a significance test.

The fundamental research question is now: **does JEPA help obtain a trustworthy
answer using fewer expensive concrete checks?** The answer cannot be inferred
merely from CEGAR terminating or removing the recorded false edge. On this small
GridWorld, direct checking may be faster in wall-clock time even if JEPA saves
oracle calls. Training cost and complete catalogue enumeration also remain real
costs. Any broader claim must specify which operations are expensive and which
state/label/reset assumptions are available.
