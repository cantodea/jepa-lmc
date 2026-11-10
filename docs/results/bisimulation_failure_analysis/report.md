# Bisimulation failure mechanisms in Cross-Map Ranking JEPA

## 1. Research question and scope

This report archives the completed independent post-hoc analysis of the original Cross-Map Ranking Round 2 JEPA graphs. It asks why some successor-prediction errors destroy initial bisimulation whereas others do not, despite aggregate Top-1 accuracy approaching 99%.

The source implementation is fixed at `539410dec1d5c4e4433bd9a57b10a309cd9a850d`. The study uses the already evaluated 24 stress maps and three model seeds, yielding 72 seed–map cases. It does not train or run inference with JEPA or MLP, retune a model, introduce maps, or alter original predictions. True transitions are used only for verification and post-hoc counterfactual correction. This publication step packages the completed results; it does not repeat or extend the experiments.

The main finding contradicts a possible expectation that a few critical errors can be corrected while other errors remain in a failing instance. In this dataset, every failing instance requires correction of **all** its errors. Conversely, every tested correction subset in the seven initially bisimilar cases with errors preserves initial bisimulation.

## 2. Experimental protocol

### 2.1 Data identity and original-result reproduction

The recovered graph export is 29,962,156 bytes, with SHA-256:

`8d51e45e76bf1f5a0c70732fdcfa90f72b08780c0558e6d8bea39793b2eb5d98`.

The behavioral report, CTL query file, LTL report, and LTL query file also match the hashes in the original structural-evaluation provenance. Graphs contain complete state catalogues, labels, designated initial states, and four deterministic action successors per state. Their identities match the original 72 cases, using seeds 20260804, 20260805, and 20260806.

The true graphs were checked against the existing stress-map definitions, including coordinates, labels, initial states, and every transition. The predicted graphs were checked against all 8,928 stress rows in the three original transition-prediction CSVs. Existing map reconstruction was used only to validate the original data, not to create a new test set. The training configuration specifies 40 training maps; no training reproduction was performed.

| Reproduced quantity | Result |
|---|---:|
| Seed–map cases | 72 |
| State–action transitions | 8,928 |
| Correct / incorrect successors | 8,832 / 96 |
| Top-1 accuracy | 98.9247311827957% |
| Initial bisimulation, ignoring actions | 40/72 |
| Identity bisimulation, ignoring actions | 33/72 |
| Initial / identity bisimulation, respecting actions | 40/72 / 33/72 |
| Initial agreement on all six CTL formulas | 68/72 |
| Initial agreement on all six LTL formulas | 70/72 |
| Initial CTL agreement without initial bisimulation | 28 cases |

The complete greatest bisimulation pair sets, not just their initial verdicts, coincide between the two action semantics in all 72 original cases. Their elimination depths can differ. This observation is not a general equivalence between action-sensitive and action-insensitive bisimulation.

The formal labels are `{safe}`, `{danger}`, and `{goal,safe}`. In particular, goal implies safe in this encoding. An earlier summary-only draft incorrectly described these labels as mutually exclusive; the original graphs resolve that ambiguity, and the present results use their exact proposition sets.

### 2.2 Formal comparison and certificates

Initial bisimulation tests membership of the designated initial pair in the greatest relation. Identity bisimulation requires every shared-state diagonal pair to belong to that greatest relation, including unreachable states. It does not require the diagonal alone to form a bisimulation.

The analysis reuses the existing `greatest_relation`, `audit_greatest_relation`, and `bisimulation_by_partition` implementations. Both directions of bisimulation use the same surviving pair set. Action-insensitive checking ignores action labels and duplicate targets; action-sensitive checking requires a response with the same action.

All 432 saved greatest simulation/bisimulation relations were reproduced, including their complete surviving pair sets and the removal rank of every eliminated pair. Some historical certificates choose a different valid challenging edge or list eliminations in a different order. Both historical and newly computed certificates pass the audit; uniqueness of the challenge text is not required for validity.

The completed main run performed 2,524 relation-certificate audits, 1,804 independent partition checks, and 64 rooted initial-failure DAG checks. These counts include original relations, historical certificates, within-true-graph comparisons, and counterfactual graphs; they are not independent experimental samples. The supplementary local-state example adds one audited relation, partition comparison, and rooted DAG. Eight upstream algorithm tests and four analysis tests passed.

Native CTL was recomputed at every state in both graphs, reproducing 26,784 original Boolean verdicts. LTL uses original nuXmv verdicts after checking graph/query hashes and complete coverage of 13,392 state–property comparisons. The LTL backend was not rerun.

### 2.3 Counterfactual operation and exact search

For each original error set E, a repair subset F⊆E replaces only the predicted successor of each selected state–action pair by its true successor. All other edges, labels, and initial states remain fixed. Every subset is enumerated; no greedy procedure or monotonicity assumption is used. Both action semantics are recomputed, audited, and independently checked by partition refinement for each subset.

The reported minimum is therefore exact within this restricted correction space. It is not the minimum number of arbitrary graph edits, relabelings, changes to correct edges, or initial-state changes. Oracle access to the true relation makes this an explanatory intervention, not an online learning or deployable error-repair algorithm.

## 3. Error distribution

An error is called reachable when its **source state** is reachable from the designated initial state.

| Quantity | Initially bisimilar cases | Failing cases |
|---|---:|---:|
| Cases | 40 | 32 |
| Transitions | 4,904 | 4,024 |
| Errors | 28 | 68 |
| Error rate | 0.5710% | 1.6899% |
| Cases with errors | 7 | 32 |
| Errors reachable in the true graph | 10 | 68 |
| Errors reachable in the predicted graph | 10 | 68 |
| Errors unreachable in both graphs | 18 | 0 |
| Moving transition predicted as self-loop | 27 | 47 |
| True self-loop predicted as movement | 1 | 21 |
| Different successor proposition sets | 3 | 7 |
| Successors bisimilar within the true graph | 10 | 0 |

Overall, 74 errors replace a move with a self-loop; 22 replace a wall-blocked self-loop with movement. No error predicts a different non-loop movement for a true movement. Error-source reachability agrees between the two graphs for all 96 errors. The complete reachable-state sets agree in 70 cases; the other two are failing sealed-region cases with newly accessible regions.

The true-to-predicted successor label pairs are `{safe}→{safe}` for 86 errors, `{danger}→{safe}` for six, `{safe}→{danger}` for two, `{goal,safe}→{safe}` for one, and `{safe}→{goal,safe}` for one. Thus 61 of the 68 errors in failing cases have two `{safe}` successors. Immediate label disagreement is not necessary for failure.

Counting a proposition if it appears at the source, true successor, or predicted successor, two errors involve goal, eight involve danger, and all 96 involve safe. These categories overlap and are not a partition.

| Map family | Cases | Errors | Initial passes | Identity passes | Reachable errors | Unreachable errors |
|---|---:|---:|---:|---:|---:|---:|
| sealed_region | 24 | 31 | 22 | 15 | 13 | 18 |
| danger_gate | 24 | 29 | 10 | 10 | 29 | 0 |
| safe_detour | 24 | 36 | 8 | 8 | 36 | 0 |

These distributions are descriptive associations. Reachability alone is not used as a causal explanation; the exhaustive interventions establish the effects of retaining the observed errors in the specified correction space.

## 4. Bisimulation failure mechanisms

### 4.1 Reachability and successor behavior classes

Let t be a true successor and p its predicted replacement. The analysis additionally computes whether t and p are bisimilar **within the same true graph**. This is stronger than equality of immediate labels and different from equality of state identifiers.

| Structural situation | Errors | Original-case outcome | Intervention evidence |
|---|---:|---|---|
| Unreachable source; different true-graph behavior classes | 18 | Pass | Every correction/retention subset passes |
| Reachable source; same true-graph behavior class | 10 | Pass | Every correction/retention subset passes |
| Reachable source; different true-graph behavior classes | 68 | Fail | Retaining any one error in isolation still fails |

This separation holds exactly for the observed data. It is not a universal necessary-and-sufficient rule for arbitrary labeled transition systems. Duplicate successors, alternative branches, interactions between edits, and non-diagonal matching can change the effect of an error elsewhere.

### 4.2 Why seven imperfect reconstructions remain initially bisimilar

All seven belong to sealed_region. In each, both reachable sets coincide and contain only states labeled exactly `{safe}`. They are closed under all actions and have a successor for every action. The analysis explicitly checks that the Cartesian product of the two reachable sets is a bisimulation, including same-action responses.

Consequently, ten reachable errors alter state identities without leaving the common behavior class. The other 18 errors originate outside the reachable region. This rules out the explanation that all harmless errors were unreachable.

| Seed | sealed_region map | Errors | Reachable / unreachable errors | Tested subsets | Passing subsets |
|---|---|---:|---:|---:|---:|
| 20260804 | layout0_rot180 | 4 | 2 / 2 | 16 | 16 |
| 20260804 | layout1_rot0 | 4 | 1 / 3 | 16 | 16 |
| 20260805 | layout0_rot0 | 3 | 0 / 3 | 8 | 8 |
| 20260805 | layout0_rot180 | 3 | 1 / 2 | 8 | 8 |
| 20260805 | layout0_rot270 | 9 | 4 / 5 | 512 | 512 |
| 20260805 | layout1_rot180 | 3 | 2 / 1 | 8 | 8 |
| 20260806 | layout0_rot180 | 2 | 0 / 2 | 4 | 4 |

All seven fail identity bisimulation. Initial equivalence does not imply correct reconstruction of every shared state, especially in isolated regions.

### 4.3 Formal failure evidence

A pair removed at round k has an existing challenging edge for which every legal response leads to a pair removed earlier. Recursively following all responses yields a finite branching DAG ending in label mismatches. The archive contains a complete root DAG for each of the 32 failures under each action semantics.

Action-insensitive initial removals occur at rounds 5–11; their DAGs contain 18–291 nodes. Action-sensitive initial removals occur at rounds 4–11, with 5–12 DAG nodes. All initial labels match. All roots are therefore eliminated for successor-matching failure, not an immediate initial label mismatch. The action-insensitive certificates select a true-side challenge in 30 cases and a predicted-side challenge in the two sealed-region failures. This selection is a valid proof choice, not a unique causal attribution.

The exported roots, challenges, complete response sets, and decreasing ranks were independently checked and checked again after serialization. These are formal fixed-point elimination certificates, not ordinary error paths, shortest traces, or generated LTL counterexamples.

### 4.4 Representative verified cases

**A single same-label error opens a sealed region.** In `20260804 / sealed_region_layout1_rot90`, the sole error concerns state s12=(2,0), action down. The true successor is (2,0) because of a wall; the predicted successor is (4,0). Both are labeled `{safe}`, but the spurious crossing makes goal and danger reachable from the predicted initial state. Top-1 is 119/120, or 99.1667%.

The initial pair (s5,s5), at coordinate (0,5), is removed at round 9. A predicted challenge s5→s11 has true responses with distinct targets s5, s11, and s4. The resulting pairs (5,11), (11,11), and (4,11) were all removed at round 8. The complete DAG has 117 nodes. The root challenge itself is not the original prediction error: the violation propagates through recursive matching conditions. Correcting (s12,down) restores both initial-bisimulation verdicts.

**Nine errors remain compatible with initial equivalence.** In `20260805 / sealed_region_layout0_rot270`, nine of 120 predictions are wrong, giving only 92.5% accuracy. Both initial reachable regions contain 12 `{safe}` states. Four reachable move-to-loop errors stay within the same behavior class, while five sources are unreachable. All 512 subsets pass; the minimum repair count is zero. Identity bisimulation still fails, including effects in the isolated goal/danger region.

**Equal labels and shortest distances do not determine behavior equivalence.** In `20260804 / safe_detour_layout1_rot180`, three of 128 predictions are wrong. Both six-formula suites agree at every state, but initial bisimulation fails. Of eight correction subsets, only correcting all three succeeds.

One error changes the down successor of s23=(4,2) from s23 to s28=(5,1). Both successor labels are `{safe}`. Within the true graph, both are six steps from goal and five from danger, yet they are not bisimilar. At round 6, the local pair (23,28) is removed because s23 can move to s24=(4,3), whereas the distinct response targets from s28 are s22, s28, and s27. The pairs (24,22), (24,28), and (24,27) were removed at rounds 5, 4, and 4. A 63-node local DAG certifies this distinction.

The cross-system initial pair is removed at round 9, with a 124-node DAG. Correcting the other two errors while retaining this one still fails. The local structural difference therefore has both formal evidence and an isolated intervention. Among all 68 failing-case errors, 67 change at least one true-graph shortest distance to goal or danger; this verified exception shows why distance summaries cannot replace recursive branch matching.

## 5. Exact counterfactual repair results

| Minimum corrections | Failing cases | Original errors in each such case |
|---:|---:|---:|
| 1 | 13 | 1 |
| 2 | 9 | 2 |
| 3 | 6 | 3 |
| 4 | 1 | 4 |
| 5 | 3 | 5 |

All 222 subsets across the 32 failures were evaluated. Under both action semantics, each instance has exactly one successful subset: its complete original error set. Every minimum set is therefore unique, and its size equals the original number of errors.

There are 68 single-edge correction attempts in failing cases. Thirteen succeed, exactly in the single-error cases; 55 fail. No single-edge correction succeeds in the 19 multi-error cases. The exact successful transitions are listed in `single_edge_repairs_13.csv`.

For each of those 68 errors, correcting every other error and retaining only that error still yields failure. Thus correcting each is necessary for a successful correction set, and each retained error alone is sufficient to destroy initial equivalence relative to the exact true graph. This does **not** mean each error is a necessary cause of the original failure: in a multi-error case, removing one may leave other sufficient causes.

All 572 subsets in the seven originally passing cases pass under both semantics. The 28 errors can therefore be retained in any combination within their respective original cases and this correction space, not merely in the originally observed combination.

The evidence supports the statement that one to five corrections restore every failing instance because each has at most five errors. It does not support a claim that a selectively chosen proper subset of errors suffices in a multi-error failure.

## 6. Relation to CTL/LTL agreement

| Property agreement | Initially bisimilar / 40 | Failing / 32 |
|---|---:|---:|
| All six initial CTL formulas | 40 | 28 |
| All six initial LTL formulas | 40 | 30 |
| Both initial suites | 40 | 28 |
| Both suites at every state | 39 | 27 |

The 28 CTL-agreeing failures consist of 13 danger_gate and 15 safe_detour cases. They also agree on the six CTL formulas at every state and on all six LTL formulas initially. In 27 cases both suites agree at every state. The missed behavior differences therefore cannot be explained solely by restricting evaluation to the initial state.

One initially bisimilar case has a non-initial property disagreement. This is consistent with initial bisimulation: it does not assert that every same-ID state pair is related.

The CTL suite comprises `EF danger`, `EF goal`, `E[!danger U goal]`, `AG !danger`, `AF goal`, and `EG safe`. The universal-path LTL suite comprises `X !danger`, `F goal`, `G !danger`, `safe U goal`, `G F safe`, and `F G !danger`. No new fairness assumptions were introduced. Agreement on these fixed suites does not establish equivalence for all CTL/LTL formulas. Failure of bisimulation is not, by itself, evidence that some LTL formula distinguishes the systems. This analysis does not claim to synthesize a new distinguishing formula.

## 7. Findings and limitations

**Why can a single error destroy initial bisimulation?** A reachable error can change a successor's future behavior class, including opening a previously inaccessible labeled region. The matching violation propagates to the initial pair. All 13 single-error failures have audited certificates and successful one-edge corrections; immediate successor labels need not differ.

**Why can several errors preserve it?** Unreachable changes and changes within a closed, action-total `{safe}` behavior class need not affect initial behavior. The explicit reachable-product relation and all 572 subset checks jointly establish this explanation for the seven observed cases.

**Which structural features matter?** Reachability, successor behavior classes, closure of same-label regions, access to goal/danger regions, and recursive matching of all branches explain these results. Error count, immediate labels, motion categories, and shortest-distance summaries are individually insufficient. Greatest bisimulation remains the actual decision procedure.

These are verified finite-graph findings and restricted counterfactual results. They do not establish an unrestricted theorem about error locations or a repair algorithm that operates without ground truth. The 72 cases share 24 maps, map families, layouts, rotations, and three model seeds; they are not 72 independent random samples. The stress suite had already been inspected and is not a fresh blind test. No unseen-map formal guarantee, training reproducibility claim, or new checkpoint-inference result follows.

The raw graph and companion files are not included in ordinary Git. Public integrity checks can verify archived identity, arithmetic, subset completeness, and internal certificate structure. Complete source-graph validation or reproduction requires the original hash-matching files, obtainable only through a maintainer-approved transfer as described in the [README](README.md).
