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

| dataset | model | total inflation | density cost | resolved? | test-node-specific | resolved? |
|---|---|---|---|---|---|---|
| Cora | GCN | 1.0 ± 0.9 | 0.1 ± 0.8 | **no** | 1.0 ± 0.9 | yes |
| Cora | GraphSAGE | 1.5 ± 1.0 | 2.4 ± 1.2 | yes | -0.8 ± 1.6 | **no** |
| Cora | MLP | 0.0 ± 0.0 | 0.0 ± 0.0 | **no** | 0.0 ± 0.0 | **no** |
| Cora | LabelProp | 0.0 ± 0.0 | 0.0 ± 0.0 | **no** | 0.0 ± 0.0 | **no** |
| CiteSeer | GCN | 0.7 ± 1.2 | 1.2 ± 1.0 | yes | -0.5 ± 1.4 | **no** |
| CiteSeer | GraphSAGE | 1.1 ± 1.4 | 2.8 ± 1.8 | yes | -1.8 ± 1.8 | yes |
| CiteSeer | MLP | 0.0 ± 0.0 | 0.0 ± 0.0 | **no** | 0.0 ± 0.0 | **no** |
| CiteSeer | LabelProp | 0.0 ± 0.0 | 0.0 ± 0.0 | **no** | 0.0 ± 0.0 | **no** |
| PubMed | GCN | 0.3 ± 1.2 | -0.3 ± 0.8 | **no** | 0.6 ± 0.5 | yes |
| PubMed | GraphSAGE | 0.1 ± 1.2 | -0.6 ± 1.2 | **no** | 0.7 ± 1.0 | yes |
| PubMed | MLP | 0.0 ± 0.0 | 0.0 ± 0.0 | **no** | 0.0 ± 0.0 | **no** |
| PubMed | LabelProp | 0.0 ± 0.0 | 0.0 ± 0.0 | **no** | 0.0 ± 0.0 | **no** |

## Density control, bisected test set

| dataset | model | total inflation | density cost | resolved? | test-node-specific | resolved? |
|---|---|---|---|---|---|---|
| Cora | GCN | 0.3 ± 1.1 | 0.3 ± 1.2 | **no** | 0.0 ± 0.5 | **no** |
| Cora | GraphSAGE | 1.3 ± 1.3 | 1.1 ± 1.4 | yes | 0.2 ± 1.6 | **no** |
| Cora | MLP | 0.0 ± 0.0 | 0.0 ± 0.0 | **no** | 0.0 ± 0.0 | **no** |
| Cora | LabelProp | 0.0 ± 0.0 | 0.0 ± 0.0 | **no** | 0.0 ± 0.0 | **no** |
| CiteSeer | GCN | 0.2 ± 1.2 | 0.4 ± 1.3 | **no** | -0.2 ± 1.4 | **no** |
| CiteSeer | GraphSAGE | 0.5 ± 1.4 | 0.7 ± 1.1 | **no** | -0.2 ± 1.4 | **no** |
| CiteSeer | MLP | 0.0 ± 0.0 | 0.0 ± 0.0 | **no** | 0.0 ± 0.0 | **no** |
| CiteSeer | LabelProp | 0.0 ± 0.0 | 0.0 ± 0.0 | **no** | 0.0 ± 0.0 | **no** |
| PubMed | GCN | -0.2 ± 0.9 | -0.1 ± 0.9 | **no** | -0.1 ± 0.8 | **no** |
| PubMed | GraphSAGE | -0.1 ± 1.0 | -0.3 ± 1.3 | **no** | 0.2 ± 0.8 | **no** |
| PubMed | MLP | 0.0 ± 0.0 | 0.0 ± 0.0 | **no** | 0.0 ± 0.0 | **no** |
| PubMed | LabelProp | 0.0 ± 0.0 | 0.0 ± 0.0 | **no** | 0.0 ± 0.0 | **no** |
| chameleon | GCN | -1.0 ± 2.5 | -0.5 ± 2.9 | **no** | -0.5 ± 4.5 | **no** |
| chameleon | GraphSAGE | 0.0 ± 2.7 | -0.5 ± 3.0 | **no** | 0.5 ± 2.3 | **no** |
| chameleon | MLP | 0.0 ± 0.0 | 0.0 ± 0.0 | **no** | 0.0 ± 0.0 | **no** |
| chameleon | LabelProp | 0.0 ± 0.0 | 0.0 ± 0.0 | **no** | 0.0 ± 0.0 | **no** |
| squirrel | GCN | -1.1 ± 0.9 | -0.5 ± 1.6 | **no** | -0.7 ± 2.2 | **no** |
| squirrel | GraphSAGE | -0.5 ± 1.5 | 0.2 ± 1.8 | **no** | -0.7 ± 1.4 | **no** |
| squirrel | MLP | 0.0 ± 0.0 | 0.0 ± 0.0 | **no** | 0.0 ± 0.0 | **no** |
| squirrel | LabelProp | 0.0 ± 0.0 | 0.0 ± 0.0 | **no** | 0.0 ± 0.0 | **no** |

## Density control, random splits

| dataset | model | total inflation | density cost | resolved? | test-node-specific | resolved? |
|---|---|---|---|---|---|---|
| Cora | GCN | -0.1 ± 1.7 | 0.4 ± 1.7 | **no** | -0.5 ± 1.9 | **no** |
| Cora | GraphSAGE | 0.1 ± 1.7 | 1.0 ± 2.2 | **no** | -0.9 ± 2.6 | **no** |
| Cora | MLP | 0.0 ± 0.0 | 0.0 ± 0.0 | **no** | 0.0 ± 0.0 | **no** |
| Cora | LabelProp | 0.0 ± 0.0 | 0.0 ± 0.0 | **no** | 0.0 ± 0.0 | **no** |
| CiteSeer | GCN | 0.6 ± 1.5 | -0.2 ± 1.9 | **no** | 0.8 ± 1.2 | yes |
| CiteSeer | GraphSAGE | 0.4 ± 2.2 | 0.3 ± 1.2 | **no** | 0.1 ± 2.1 | **no** |
| CiteSeer | MLP | 0.0 ± 0.0 | 0.0 ± 0.0 | **no** | 0.0 ± 0.0 | **no** |
| CiteSeer | LabelProp | 0.0 ± 0.0 | 0.0 ± 0.0 | **no** | 0.0 ± 0.0 | **no** |
| PubMed | GCN | -0.7 ± 1.5 | -0.1 ± 1.1 | **no** | -0.6 ± 1.0 | **no** |
| PubMed | GraphSAGE | -0.2 ± 1.5 | 0.1 ± 1.1 | **no** | -0.4 ± 1.3 | **no** |
| PubMed | MLP | 0.0 ± 0.0 | 0.0 ± 0.0 | **no** | 0.0 ± 0.0 | **no** |
| PubMed | LabelProp | 0.0 ± 0.0 | 0.0 ± 0.0 | **no** | 0.0 ± 0.0 | **no** |
| chameleon | GCN | 0.0 ± 1.9 | -0.1 ± 1.9 | **no** | 0.1 ± 1.8 | **no** |
| chameleon | GraphSAGE | 0.8 ± 1.3 | -0.5 ± 1.3 | **no** | 1.3 ± 2.0 | yes |
| chameleon | MLP | 0.0 ± 0.0 | 0.0 ± 0.0 | **no** | 0.0 ± 0.0 | **no** |
| chameleon | LabelProp | 0.0 ± 0.0 | 0.0 ± 0.0 | **no** | 0.0 ± 0.0 | **no** |
| squirrel | GCN | 0.2 ± 1.7 | -0.2 ± 1.4 | **no** | 0.4 ± 1.2 | **no** |
| squirrel | GraphSAGE | -0.1 ± 1.4 | 0.1 ± 1.4 | **no** | -0.2 ± 1.9 | **no** |
| squirrel | MLP | 0.0 ± 0.0 | 0.0 ± 0.0 | **no** | 0.0 ± 0.0 | **no** |
| squirrel | LabelProp | 0.0 ± 0.0 | 0.0 ± 0.0 | **no** | 0.0 ± 0.0 | **no** |

## Duplicate definitions

| definition | Cora | CiteSeer | CiteSeer / Cora | PubMed | chameleon | squirrel |
|---|---|---|---|---|---|---|
| _quoted by OGB from Zou et al._ | _1.0_ | _5.0_ | _5.0_ | _-_ | _-_ | _-_ |
| `exact_but_conflicting_label` | 0.00 | 0.12 | ∞ | 0.02 | 13.22 | 4.42 |
| `all_zero_feature_rows` | 0.00 | 0.45 | ∞ | 0.00 | 10.23 | 3.17 |
| `exact_duplicate_pairs_over_nodes_excluding_zero_rows` | 0.81 | 0.30 | 0.37 | 0.03 | 10.36 | 1.50 |
| `exact_nodes_excluding_all_zero_rows` | 1.00 | 0.60 | 0.60 | 0.04 | 6.24 | 1.92 |
| `cosine_0.99` | 1.00 | 0.60 | 0.60 | 0.04 | 6.24 | 1.92 |
| `jaccard_0.95` | 1.14 | 0.96 | 0.84 | 0.04 | 6.24 | 1.92 |
| `cosine_0.95` | 1.37 | 1.26 | 0.92 | 0.10 | 6.24 | 1.92 |
| `exact_and_same_label` | 1.00 | 0.93 | 0.93 | 0.02 | 3.25 | 0.67 |
| `exact_nodes_in_a_duplicate_group` | 1.00 | 1.05 | 1.06 | 0.04 | 16.47 | 5.10 |
| `exact_on_binarised_features` | 1.00 | 1.05 | 1.06 | 0.04 | 16.47 | 5.10 |
| `jaccard_0.8` | 2.47 | 2.89 | 1.17 | 0.12 | 6.24 | 1.92 |
| `cosine_0.9` | 2.18 | 2.58 | 1.19 | 0.47 | 6.24 | 1.92 |
| `exact_redundant_copies` | 0.59 | 0.72 | 1.22 | 0.02 | 14.36 | 4.25 |
| `cosine_0.8` | 4.10 | 5.17 | 1.26 | 5.24 | 6.94 | 2.04 |
| `cosine_0.7` | 5.80 | 7.94 | 1.37 | 23.73 | 13.48 | 6.81 |
| `jaccard_0.5` | 7.02 | 9.86 | 1.41 | 2.06 | 14.27 | 6.96 |
| `identical_neighbour_set` | 7.75 | 17.64 | 2.28 | 44.61 | 57.09 | 49.63 |
| `exact_duplicate_pairs_over_nodes` | 0.81 | 3.46 | 4.25 | 0.03 | 1197.36 | 261.64 |

| similarity | cutoff that puts CiteSeer at 5% | CiteSeer there | Cora there | Cora quoted |
|---|---|---|---|---|
| cosine | 0.8006 | 5.05 | 3.95 | 1.0 |
| jaccard | 0.6667 | 5.17 | 4.10 | 1.0 |

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

