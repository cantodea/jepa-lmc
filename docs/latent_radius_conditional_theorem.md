# Conditional transition inclusion and CTL transfer

This is a theorem for the finite-state construction implemented in this branch.
It makes the missing premise explicit; it does not certify the premise for a
trained neural network. It is an adaptation for JEPA, not a claim that the complete
method or all theorem conditions of Yang 2025 have been implemented.

## Definitions and assumptions

Let `S` be a nonempty finite state set, `A` a nonempty finite action set, `I` a
nonempty subset of `S`, and `L: S -> 2^AP` a fixed labelling function. Let the true
dynamics `T: S x A -> S` be deterministic and total. For a collection of maps,
apply the statement to each map, or tag each state with its map identity.

Fix the observation function `o`, context encoder `E_c`, target encoder `E_t`,
predictor `P`, latent metric `d` and a finite radius `epsilon >= 0`. No weights,
target EMA, observations or labels change while the graph is built or checked.
Write

\[
g(s,a)=P(E_c(o(s)),a),\qquad h(s)=E_t(o(s)).
\]

The crucial assumption is a uniform bound **on the entire intended domain**:

\[
\forall s\in S,\ \forall a\in A:\quad
d\bigl(g(s,a),h(T(s,a))\bigr)\leq\epsilon. \tag{B}
\]

For each input, define the candidate set over the complete state catalogue:

\[
C_\epsilon(s,a)=\{t\in S:d(g(s,a),h(t))\leq\epsilon\}.
\]

No true successor may be removed by a later top-k cut, topology heuristic,
clustering decision or other filter. Distinct states keep their identities even
when their target embeddings coincide. Injectivity of `h` is **not required**
for inclusion, although collisions can harm precision.

Define

\[
R^A=\{(s,a,T(s,a)):s\in S,a\in A\},\quad
\hat R^A_\epsilon=\{(s,a,t):t\in C_\epsilon(s,a)\},
\]
\[
R=\{(s,t):\exists a\ (s,a,t)\in R^A\},\quad
\hat R_\epsilon=\{(s,t):\exists a\ (s,a,t)\in\hat R^A_\epsilon\}.
\]

The exact and candidate Kripke structures are `M=(S,I,R,L)` and
`M_hat=(S,I,R_hat_epsilon,L)`. Thus the states, initial states and labels agree.

## Theorem

Under (B):

1. `T(s,a)` belongs to `C_epsilon(s,a)` for every input. Every candidate action set
   is nonempty; the candidate Kripke relation is total.
2. `R^A` is contained in `R_hat^A_epsilon`, hence `R` is contained in
   `R_hat_epsilon`.
3. The identity relation on `S` is a label-preserving simulation from the exact
   structure into the candidate structure.
4. For every state `s`, universal-fragment CTL truth transfers from the candidate
   structure to the exact structure. Existential-fragment CTL falsity transfers
   in that same direction.

Here the universal fragment permits literals `p`, `!p`, conjunction/disjunction
and `AX`, `AF`, `AG` recursively. The existential fragment permits literals,
conjunction/disjunction and `EX`, `EF`, `EG`, `E[phi U psi]` recursively. These
fragments cover the six properties currently evaluated. They exclude arbitrary
mixed nesting such as `AG(EF goal)`. Additional standard universal operators can
be treated separately; they are not implemented by this checker.

### Proof

For any `s,a`, assumption (B) is precisely the membership test for `T(s,a)` in
`C_epsilon(s,a)`. This proves nonemptiness and labelled inclusion. Erasing the
action from each labelled edge proves unlabelled inclusion. Every exact edge
`s -> t` is therefore matched by the same candidate edge, with identical labels
at `s` and `t`; this proves the identity simulation, including initial states.

Inclusion implies every infinite exact path is also a candidate path. A structural
induction on formulas proves

\[
\operatorname{Sat}_{\hat M}(\phi_\forall)
\subseteq\operatorname{Sat}_{M}(\phi_\forall),\qquad
\operatorname{Sat}_{M}(\phi_\exists)
\subseteq\operatorname{Sat}_{\hat M}(\phi_\exists).
\]

Literals have identical satisfaction sets because labels agree. Conjunction and
disjunction preserve inclusion. For universal operators, the candidate structure
quantifies over every exact successor/path and possibly more, while the inductive
hypothesis transfers the required subformula truth to the exact states. For
existential operators, a witnessing exact successor/path remains in the candidate
structure and the inductive hypothesis preserves its subformula truth, including
the condition and target along an until witness. Taking the contrapositive of the
existential inclusion gives the stated falsity transfer. QED.

## What is not concluded

- Candidate existential True need not be true in the exact structure. Universal
  False in the candidate structure need not be false in the exact structure.
- Inclusion alone does not establish equivalence, arbitrary CTL preservation or
  bisimulation. If both transition relations are equal, the identity is a
  bisimulation; that stronger premise is not assumed here.
- A high singleton rate does not establish that the singletons are correct. If
  (B) holds and **every** action set is a singleton, then each singleton must be
  the true successor, giving equality of the action-labelled relations.
- Certifying the predictor's own output set does not establish (B), which compares
  predictions with the target encoding of the **true** successor.
- Changing/learning the state catalogue or its labels requires extra assumptions;
  this theorem does not handle unknown states or learned labelling errors.

## Oracle experiments and the missing premise

If all true transitions of a particular finite domain are available, its maximum
error satisfies (B) on that domain by definition. This is an exhaustive oracle
construction. The implementation separately audits graph inclusion by exact set
comparison. Neither step proves a bound on unenumerated inputs or future maps.
Quantile radii can omit successors. A maximum fitted to validation errors can also
omit test successors. Good empirical CTL agreement cannot replace premise (B).

The theorem uses exact metric comparisons. The implementation uses floating-point
neural outputs and distances, and does not provide a certified numerical error
enclosure. The discrete relation audit validates the actual constructed finite
graph; it should not be described as real-arithmetic neural-network certification.

For the **same fixed network, metric and uniform-radius construction**, any valid
bound on a domain containing the test inputs is at least their true maximum
error. A larger radius produces a superset of every candidate set. Consequently
it can only maintain or lose universal-True / existential-False proving power.
If even the oracle maximum loses the useful conclusions, replacing it with a
larger certified uniform bound will not restore those conclusions. This statement
does not cover different local bounds, partitions, architectures or abstractions.

The related [2024 predecessor, Proposition 1](https://arxiv.org/html/2402.11739v1)
uses a guaranteed model discrepancy and a geometric margin for transition
correspondence. The relevant discrepancy here is `d(g(s,a),h(T(s,a)))` in target
latent space. A physical-state error bound cannot simply be substituted for it.
The target reference remains [Yang et al. (2025)](https://doi.org/10.1016/j.neunet.2025.107261);
the theorem above is stated and proved directly for this repository's construction.
