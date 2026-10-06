## Evaluator-only unseen edge-type diagnostic

Accuracies use all unseen pairs, including unreachable states. Moving edges have a real successor different from their source. The self-loop completion is an **evaluation-only context**, not learned inference or the raw partial observed-edge baseline. Its moving-edge accuracy is 0% by construction.

| Budget | Model / evaluation policy | Unseen accuracy | True self-loop unseen accuracy | Moving unseen accuracy |
|---:|---|---:|---:|---:|
| 1 | observed_edge | 0.00% | 0.00% | 0.00% |
| 1 | mlp | 0.88% | 1.22% | 0.75% |
| 1 | jepa | 26.79% | 94.64% | 1.14% |
| 1 | observed_self_loop_completion_evaluation_only | 27.47% | 100.00% | 0.00% |
| 2 | observed_edge | 0.00% | 0.00% | 0.00% |
| 2 | mlp | 0.45% | 0.60% | 0.38% |
| 2 | jepa | 27.83% | 96.16% | 1.40% |
| 2 | observed_self_loop_completion_evaluation_only | 27.91% | 100.00% | 0.00% |
| 4 | observed_edge | 0.00% | 0.00% | 0.00% |
| 4 | mlp | 0.15% | 0.22% | 0.12% |
| 4 | jepa | 28.56% | 96.71% | 2.03% |
| 4 | observed_self_loop_completion_evaluation_only | 28.05% | 100.00% | 0.00% |
| 8 | observed_edge | 0.00% | 0.00% | 0.00% |
| 8 | mlp | 0.05% | 0.06% | 0.05% |
| 8 | jepa | 28.88% | 96.86% | 1.92% |
| 8 | observed_self_loop_completion_evaluation_only | 28.42% | 100.00% | 0.00% |
| 16 | observed_edge | 0.00% | 0.00% | 0.00% |
| 16 | mlp | 0.00% | 0.00% | 0.00% |
| 16 | jepa | 30.07% | 97.26% | 2.05% |
| 16 | observed_self_loop_completion_evaluation_only | 29.43% | 100.00% | 0.00% |
