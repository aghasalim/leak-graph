## Leakage inflation

| dataset | model | transductive | inductive | **inflation** | resolved? |
|---|---|---|---|---|---|
| Cora | GCN | 81.3 ± 0.7 | 80.3 ± 0.7 | **1.0 ± 0.9** | yes |
| Cora | GraphSAGE | 80.4 ± 0.7 | 78.9 ± 0.8 | **1.5 ± 1.0** | yes |
| Cora | MLP | 56.9 ± 0.8 | 56.9 ± 0.8 | **0.0 ± 0.0** | **no** |
| Cora | LabelProp | 60.3 ± 0.0 | 60.3 ± 0.0 | **0.0 ± 0.0** | **no** |
| CiteSeer | GCN | 69.2 ± 0.7 | 68.5 ± 0.8 | **0.7 ± 1.2** | **no** |
| CiteSeer | GraphSAGE | 68.5 ± 1.3 | 67.5 ± 1.1 | **1.1 ± 1.4** | yes |
| CiteSeer | MLP | 52.7 ± 1.0 | 52.7 ± 1.0 | **0.0 ± 0.0** | **no** |
| CiteSeer | LabelProp | 36.6 ± 0.0 | 36.6 ± 0.0 | **0.0 ± 0.0** | **no** |
| PubMed | GCN | 78.6 ± 0.7 | 78.3 ± 0.6 | **0.3 ± 1.2** | **no** |
| PubMed | GraphSAGE | 76.8 ± 0.9 | 76.7 ± 0.9 | **0.1 ± 1.2** | **no** |
| PubMed | MLP | 72.5 ± 0.4 | 72.5 ± 0.4 | **0.0 ± 0.0** | **no** |
| PubMed | LabelProp | 42.0 ± 0.0 | 42.0 ± 0.0 | **0.0 ± 0.0** | **no** |
| chameleon | GCN | 37.6 ± 2.8 | 40.0 ± 2.6 | **-2.5 ± 2.0** | yes |
| chameleon | GraphSAGE | 51.6 ± 1.6 | 51.0 ± 1.7 | **0.6 ± 1.3** | **no** |
| chameleon | MLP | 50.5 ± 3.0 | 50.5 ± 3.0 | **0.0 ± 0.0** | **no** |
| chameleon | LabelProp | 21.8 ± 1.5 | 21.8 ± 1.5 | **0.0 ± 0.0** | **no** |
| squirrel | GCN | 26.2 ± 1.4 | 27.4 ± 1.8 | **-1.2 ± 1.3** | yes |
| squirrel | GraphSAGE | 37.4 ± 2.1 | 37.3 ± 1.9 | **0.1 ± 0.9** | **no** |
| squirrel | MLP | 36.2 ± 1.7 | 36.2 ± 1.7 | **0.0 ± 0.0** | **no** |
| squirrel | LabelProp | 17.8 ± 1.3 | 17.8 ± 1.3 | **0.0 ± 0.0** | **no** |

## Duplicate component

| dataset | model | inflation (all test) | inflation (no straddling duplicates) | duplicate component |
|---|---|---|---|---|
| Cora | GCN | 1.0 ± 0.9 | 1.1 ± 0.9 | -0.0 |
| Cora | GraphSAGE | 1.5 ± 1.0 | 1.6 ± 1.0 | -0.0 |
| CiteSeer | GCN | 0.7 ± 1.2 | 0.6 ± 1.3 | 0.1 |
| CiteSeer | GraphSAGE | 1.1 ± 1.4 | 1.0 ± 1.3 | 0.1 |
| PubMed | GCN | 0.3 ± 1.2 | 0.3 ± 1.2 | 0.0 |
| PubMed | GraphSAGE | 0.1 ± 1.2 | 0.1 ± 1.2 | -0.0 |
| chameleon | GCN | -2.5 ± 2.0 | -3.2 ± 2.7 | 0.8 |
| chameleon | GraphSAGE | 0.6 ± 1.3 | -0.2 ± 1.9 | 0.8 |
| squirrel | GCN | -1.2 ± 1.3 | -1.6 ± 2.3 | 0.4 |
| squirrel | GraphSAGE | 0.1 ± 0.9 | 0.9 ± 1.0 | -0.8 |

## Baselines

| dataset | MLP (features, no graph) | neighbour vote (graph, no features, no learning) | GCN inductive | GCN inductive, same nodes as the vote |
|---|---|---|---|---|
| Cora | 56.9 | 79.6 (covers 21.1% of test) | 80.3 | 86.1 ± 1.0 |
| CiteSeer | 52.7 | 76.0 (covers 9.6% of test) | 68.5 | 73.5 ± 1.2 |
| PubMed | 72.5 | 73.7 (covers 1.9% of test) | 78.3 | 78.9 ± 0.0 |
| chameleon | 50.5 | 33.0 (covers 39.3% of test) | 40.0 | 36.2 ± 5.0 |
| squirrel | 36.2 | 20.7 (covers 40.3% of test) | 27.4 | 27.8 ± 2.3 |

## Density control

| dataset | model | total inflation | density cost | test-node-specific |
|---|---|---|---|---|
| Cora | GCN | 1.0 ± 0.9 | 0.1 ± 0.8 | 1.0 ± 0.9 |
| Cora | GraphSAGE | 1.5 ± 1.0 | 2.4 ± 1.2 | -0.8 ± 1.6 |
| CiteSeer | GCN | 0.7 ± 1.2 | 1.2 ± 1.0 | -0.5 ± 1.4 |
| CiteSeer | GraphSAGE | 1.1 ± 1.4 | 2.8 ± 1.8 | -1.8 ± 1.8 |
| PubMed | GCN | 0.3 ± 1.2 | -0.3 ± 0.8 | 0.6 ± 0.5 |
| PubMed | GraphSAGE | 0.1 ± 1.2 | -0.6 ± 1.2 | 0.7 ± 1.0 |
| chameleon | - | _not measurable: density_control needs 456 spare unlabelled nodes but the split leaves only 0; this split partitions the whole graph_ | | |
| squirrel | - | _not measurable: density_control needs 1041 spare unlabelled nodes but the split leaves only 0; this split partitions the whole graph_ | | |

## Duplicate detector

| dataset | nodes | exact duplicate nodes | all-zero feature rows | near-dup pairs | cutoff | straddling pairs | test nodes with a train twin |
|---|---|---|---|---|---|---|---|
| Cora | 2708 | 27 (1.0%) | 0 | 440 | 0.474 | 10 | 8 / 1000 (0.8%) |
| CiteSeer | 3327 | 35 (1.1%) | 15 | 2790 | 0.292 | 51 | 45 / 1000 (4.5%) |
| PubMed | 19717 | 7 (0.0%) | 0 | 188 | 0.904 | 1 | 1 / 1000 (0.1%) |
| chameleon | 2277 | 375 (16.5%) | 233 | 3395 | 0.378 | 683 | 128 / 456 (28.1%) |
| squirrel | 5201 | 265 (5.1%) | 165 | 7348 | 0.354 | 1580 | 377 / 1041 (36.2%) |

## Feature-label detector

| dataset | logistic regression on features alone | majority class | giveaway features | expected under label-permuted null | test nodes covered |
|---|---|---|---|---|---|
| Cora | 57.6 | 13.0 | 3 / 1433 | 0.0 | 7.2% |
| CiteSeer | 59.3 | 7.7 | 0 / 3703 | 0.0 | 0.0% |
| PubMed | 72.9 | 18.0 | 9 / 500 | 0.4 | 50.1% |
| chameleon | 44.7 | 22.4 | 61 / 2325 | 1.0 | 9.6% |
| squirrel | 35.4 | 19.3 | 4 / 2089 | 0.2 | 1.3% |

