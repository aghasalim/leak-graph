# LeakGraph

An audit kit for leakage in transductive GNN node-classification benchmarks. It measures
duplicate nodes, feature–label leakage and neighbourhood-label leakage on the standard
datasets, re-evaluates the standard models under a genuinely inductive split, and reports one
number per (dataset, model): **leakage inflation**, the accuracy that transductive evaluation
adds over inductive evaluation.

## This is tooling, not a discovery

Every phenomenon measured here is already documented in the literature. I did not find any of
it. What this repository contributes is a single harness that measures all of it on the same
splits with the same seeds under the same protocol, a headline metric that makes the cost
comparable across datasets and models, and a test suite that checks the instrument rather than
the result. Treat it as a measuring device and a replication, not as a novel claim.

See [Prior work](#prior-work) for what was already known, with sources I verified before
citing them.

## The headline number

`leakage inflation = transductive test accuracy − inductive test accuracy`, for the same
model, the same split, the same seed, the same initialisation and the same epoch budget. The
only thing that differs between the two arms is whether the test nodes existed in the graph
the model trained on.

The difference is computed **per seed and then averaged**, not as a difference of two averages.
Initialisation noise is shared between the arms and cancels in the paired difference, which is
what makes an effect of one accuracy point resolvable at all with ten seeds.

<!--BEGIN:inflation-->
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
<!--END:inflation-->

`resolved?` asks whether the mean paired difference exceeds two standard errors of that same
paired difference. Where it says **no**, I am not claiming an effect. A row reading `0.4 ± 1.1`
with `no` means I measured nothing that ten seeds can distinguish from zero, and it is reported
at the same size and in the same table as the rows that did resolve.

### What I actually found, including the parts that argue against the premise

I built this expecting transductive evaluation to be buying the GNNs a visible amount of
accuracy. Mostly it is not.

- **The effect is small everywhere.** Across ten (dataset, model) GNN cells, no resolved
  inflation exceeds 2.5 accuracy points in magnitude, and only five cells resolve at all. On **PubMed
  neither GNN shows anything** — 0.30 ± 1.20 and 0.07 ± 1.19 points, both unresolved. That is a
  negative result and I am stating it as plainly as the positive ones.
- **On chameleon and squirrel the GCN's inflation is negative and resolved** (−2.48 ± 2.05 and
  −1.19 ± 1.28 points). Training on the graph *including* test nodes makes it measurably
  **worse** there. I would not read that as "transductive evaluation is safe on heterophilous
  data": the GCN is badly mis-specified on both (37.6% and 26.2%, against an MLP that scores
  50.5% and 36.2% without using the graph at all), so extra neighbours mostly buy it more of a
  smoothing operation that was already hurting. The sign is real; the mechanism is a guess.
- **The duplicate component is nearly nothing.** Excluding test nodes that have a near-duplicate
  twin in training moves inflation by at most 0.8 points anywhere, including on squirrel, which
  has 1,580 straddling near-duplicate pairs. Duplicates are abundant; their contribution to
  *this particular* gap is small.
- **Most of what inflation there is turns out not to be leakage at all.** The
  [density control](#is-the-gap-actually-leakage) removes an equal number of *unlabelled*
  nodes instead of the test nodes. For GraphSAGE on Cora and CiteSeer that alone costs more
  accuracy than the full inflation does, leaving a negative test-node-specific remainder. Of
  the six (dataset, model) cells where the control can be built, only **Cora GCN** keeps a
  positive test-specific component of the same size as its headline number. The headline metric
  is, for the most part, measuring graph size.
- **Both controls read exactly `0.00` in all ten of their cells.** That is the only reason I trust the
  small numbers above, and it is the thing that was broken (see
  [Finding I1](#finding-i1-the-mlp-control-was-reporting-leakage-that-was-actually-a-dropout-rng-offset)).

The honest summary is that `transductive − inductive` is a much weaker instrument than I assumed
when I started building it, and the density control is what showed that. I have left the
headline metric in place because it is what the literature's framing implies, and put the
control immediately after it.

A caution on the third point: Platonov et al. showed that *removing* duplicate nodes changes GNN
performance and model rankings on squirrel and chameleon. That is a different measurement from
mine. They compare a dataset against a de-duplicated version of itself; I compare transductive
against inductive evaluation with the duplicates present in both arms. My small duplicate
component does not contradict their finding, and nothing here should be read as doing so.

**MLP and LabelProp are not results. They are the instrument's calibration.** Neither model can
have a transductive/inductive gap: the MLP never reads the graph, and label propagation has no
parameters and propagates the same training labels over the same full graph in both arms. Both
must read exactly `0.0`. They do. When they did not, the harness was broken — see
[Finding I1](#finding-i1-the-mlp-control-was-reporting-leakage-that-was-actually-a-dropout-rng-offset).

## What the number means

An accuracy gap on its own is uninterpretable. Two baselines bound it.

<!--BEGIN:baselines-->
| dataset | MLP (features, no graph) | neighbour vote (graph, no features, no learning) | GCN inductive | GCN inductive, same nodes as the vote |
|---|---|---|---|---|
| Cora | 56.9 | 79.6 (covers 21.1% of test) | 80.3 | 86.1 ± 1.0 |
| CiteSeer | 52.7 | 76.0 (covers 9.6% of test) | 68.5 | 73.5 ± 1.2 |
| PubMed | 72.5 | 73.7 (covers 1.9% of test) | 78.3 | 78.9 ± 0.0 |
| chameleon | 50.5 | 33.0 (covers 39.3% of test) | 40.0 | 36.2 ± 5.0 |
| squirrel | 36.2 | 20.7 (covers 40.3% of test) | 27.4 | 27.8 ± 2.3 |
<!--END:baselines-->

The neighbour vote has no features, no parameters and no training: it labels a test node by
majority vote over its training-set neighbours. It can only predict test nodes that *have* a
training neighbour, so the last column scores the GCN on exactly that subset, which is the only
way the two numbers answer the same question.

## Is the gap actually leakage?

Transductive minus inductive is confounded. The inductive training graph is missing the test
nodes' information, which is the effect I want to price — but it is also simply a smaller and
sparser graph, and that costs accuracy on its own for reasons that have nothing to do with
leakage.

So I ran a third arm. `density_control` removes the same *number* of nodes from the inductive
view, but draws them from the unlabelled pool — nodes the model was never going to be scored on
— instead of the test set. That splits the headline gap in two:

    transductive − density_control  = the cost of a smaller, sparser training graph
    density_control − inductive     = what is specific to hiding the test nodes

<!--BEGIN:density-->
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
<!--END:density-->

**This is the most consequential table in the repository, and it undercuts the headline metric.**
For GraphSAGE on Cora and CiteSeer the density cost (2.4 and 2.8 points) is *larger than the
total inflation*, and the test-node-specific remainder is negative. In other words, essentially
none of GraphSAGE's apparent leakage inflation on the citation networks is leakage: it is the
model reacting to a training graph with 1,000 fewer nodes in it. The same is true of CiteSeer
GCN. Only **Cora GCN** has its inflation survive the control roughly intact (1.0 total, 0.1
density, 1.0 test-specific).

Two smaller notes. On PubMed the density cost is slightly *negative* — the smaller graph trained
marginally better — so the test-specific component (0.6 and 0.7 points) is larger than the
unresolved total, which is a reminder that the two components need not have the same sign.
And I have not computed a resolution test for the decomposed components, so treat these as
directional rather than as effects I am claiming.

The control needs spare unlabelled nodes to remove. The Planetoid public splits leave most of
the graph unlabelled, so it works there. The geom-gcn splits of chameleon and squirrel assign
every single node to train, val or test, so no pool exists and the control cannot be built at
all. That is recorded in the table rather than silently skipped.

## Decomposing by duplicates

A duplicate pair only leaks if it *straddles* the train/test boundary. Two identical nodes both
sitting in the training set teach a model nothing it could not have learned from either copy;
it is the pair with one foot in train and one in test that hands over the answer.

So the same paired difference is recomputed while scoring only test nodes that have **no**
near-duplicate twin anywhere in the training set. Whatever inflation disappears is what the
straddling duplicates were worth.

<!--BEGIN:components-->
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
<!--END:components-->

## The detectors

### Duplicate and near-duplicate nodes

<!--BEGIN:duplicates-->
| dataset | nodes | exact duplicate nodes | all-zero feature rows | near-dup pairs | cutoff | straddling pairs | test nodes with a train twin |
|---|---|---|---|---|---|---|---|
| Cora | 2708 | 27 (1.0%) | 0 | 440 | 0.474 | 10 | 8 / 1000 (0.8%) |
| CiteSeer | 3327 | 35 (1.1%) | 15 | 2790 | 0.292 | 51 | 45 / 1000 (4.5%) |
| PubMed | 19717 | 7 (0.0%) | 0 | 188 | 0.904 | 1 | 1 / 1000 (0.1%) |
| chameleon | 2277 | 375 (16.5%) | 233 | 3395 | 0.378 | 683 | 128 / 456 (28.1%) |
| squirrel | 5201 | 265 (5.1%) | 165 | 7348 | 0.354 | 1580 | 377 / 1041 (36.2%) |
<!--END:duplicates-->

Two of these line up with the literature and one does not. My **1.0% exact duplicate nodes on
Cora** matches the "1% of the nodes are duplicated" figure that OGB quotes from Zou et al. My
**16.5% on chameleon and 5.1% on squirrel** are consistent with Platonov et al.'s "large number
of duplicate nodes" in exactly those two datasets. But I measure **1.1% on CiteSeer against the
5% quoted**, which I cannot reconcile — most likely they count something I do not (their figure
may include near-duplicates, or a different feature representation). I am reporting my number
and the disagreement rather than quietly adopting theirs.

Note also how weakly duplicate abundance predicts leakage here. squirrel has 1,580 straddling
pairs and 36.2% of its test nodes have a training twin, the worst in the table — and its GCN
inflation is *negative*.

The cosine cutoff is **calibrated, not chosen**. I build a null by permuting each feature column
independently, which preserves every feature's marginal frequency exactly while destroying all
co-occurrence structure between nodes, then take the largest cosine similarity any pair achieves
under that null. A real pair at or above that line is more similar than every pair drawn from
independent nodes with identical marginals.

What that justifies is a threshold, not a per-pair p-value. The real graph has far more pairs
than the null subsample did, so it gets more chances to clear the bar by luck, and the counts
should be read as an upper bound. Counts at fixed cutoffs of 0.95 and 0.99 are in
`reports/detectors.json` so the sensitivity to that choice is visible.

The `all-zero feature rows` column exists because of a real inconsistency. A node whose feature
vector is entirely zero is an exact duplicate of every other such node, so `torch.unique` counts
it — but cosine similarity is undefined for a zero vector, so it can never appear among the
near-duplicate pairs and is invisible to the straddling analysis. Rather than let the two
duplicate measurements disagree for an unstated reason, both are reported.

### Feature–label leakage

How much of the label is already in a node's own features, with no graph at all.

<!--BEGIN:features-->
| dataset | logistic regression on features alone | majority class | giveaway features | expected under label-permuted null | test nodes covered |
|---|---|---|---|---|---|
| Cora | 57.6 | 13.0 | 3 / 1433 | 0.0 | 7.2% |
| CiteSeer | 59.3 | 7.7 | 0 / 3703 | 0.0 | 0.0% |
| PubMed | 72.9 | 18.0 | 9 / 500 | 0.4 | 50.1% |
| chameleon | 44.7 | 22.4 | 61 / 2325 | 1.0 | 9.6% |
| squirrel | 35.4 | 19.3 | 4 / 2089 | 0.2 | 1.3% |
<!--END:features-->

The striking column is `giveaway features`: **3 of Cora's 1,433 features and 0 of CiteSeer's
3,703**, against a label-permuted null of essentially zero. By this definition — a single feature
whose presence pins the label down with 95% purity and at least 5 training nodes' support —
CiteSeer has no feature–label leakage at all. That sits oddly beside the "62% leakage rate"
quoted for CiteSeer, and is the clearest evidence that the quoted figure measures something
other than what I measure here. PubMed is the opposite case: 9 giveaway features covering half
the test set, on only 500 features.

`logistic regression on features alone` is the ceiling: whatever a GNN scores, the part below
this line was never graph learning. Note that on **PubMed the MLP reaches 72.5% against the
GCN's 78.3%**, so most of PubMed's headline accuracy is available without the graph at all. `giveaway features` is the mechanism — individual feature
dimensions (single vocabulary words, for the citation datasets) whose mere presence pins the
label down. Purity is measured on the training set only and then applied to test nodes, so it is
leakage that actually transfers rather than an in-sample artefact, and the count is compared
against a label-permuted null so that "how many would you expect by chance" is measured rather
than assumed.

### Neighbourhood label leakage

Covered by the neighbour-vote column in [What the number means](#what-the-number-means).
Full output, including coverage and the accuracy restricted to covered nodes, is in
`reports/detectors.json`.

## The inductive split is the thing that has to be right

Everything above is worthless if the inductive split is not actually inductive, and it is easy
to get subtly wrong. The common shortcut is to keep the full node feature matrix and merely drop
the edges touching test nodes. That is *probably* fine for a plain GCN, since isolated rows do
not influence anyone else's representation — but it is not provable, it breaks the moment a
model uses any node-set-level statistic (BatchNorm over all nodes, a global readout, degree
normalisation over the full index), and it cannot be tested by construction.

So `induced_subgraph` physically removes the nodes and relabels the survivors to `0..k-1`. The
test nodes do not exist during training. They are re-attached, with all their edges, only at
inference, exactly as an unseen node would arrive in deployment.

That property is asserted as an experiment rather than a claim about shapes:

- `test_inductive_training_is_blind_to_test_features` overwrites every test node's features with
  noise at 100x scale and asserts the inductive training loss is **bit-identical**, while the
  transductive loss moves.
- `test_inductive_training_is_blind_to_test_edges` rewires every test-node edge endpoint to a
  different random test node and asserts the same.

Validation also happens inside the training view, so in the inductive arm the model is never
scored on a graph containing test nodes until inference and model selection cannot leak either.

`make test` runs the suite. It builds a synthetic planted-partition graph in process, touches no
network and downloads nothing, which is what lets CI run it. CI additionally asserts that the
test run left `data/` empty.

## Instrument bugs

Both of these were found by the controls, not by inspection. They are recorded here rather than
quietly patched, because a harness that has never been caught being wrong is a harness nobody
has checked.

### Finding I1: the MLP control was reporting leakage that was actually a dropout RNG offset

The MLP cannot have a transductive/inductive gap. It never reads `edge_index`, and the inductive
view leaves every training node's features untouched, so the two arms are the same computation.
It was nevertheless reporting a non-zero one.

Cause: the MLP was being handed the full node feature matrix and its output sliced afterwards.
`F.dropout` therefore drew its mask at the *view's* shape — and the inductive view has fewer
rows than the transductive one. The two arms consumed the RNG stream differently, so the
training nodes received different dropout masks, and the resulting accuracy difference had
nothing to do with leakage.

Measured before the fix, over 10 seeds (`reports/instrument_bug_mlp_rng.json`):

| dataset | mean spurious gap | std | largest single seed |
|---|---|---|---|
| Cora | +0.89 pp | 1.51 | 3.9 pp |
| CiteSeer | +0.69 pp | 1.92 | 3.0 pp |

The mean is small, but it is the same order of magnitude as the real GCN inflation this
repository measures on Cora, and on individual seeds it was several times larger. Had the
control not been there, I would have reported a fabricated effect at roughly the size of the
genuine one.

Fix: `uses_graph = False` on graph-free models, and the harness runs them on the masked rows
only, so the dropout draw cannot depend on how many other nodes happen to be in the view. The
control now reads exactly `0.0`, and `test_mlp_has_exactly_zero_inflation` asserts it.

### Finding I2: `random_split` silently produced an empty test set

Asking the Planetoid defaults (500 val, 1000 test) of a graph too small to supply them made the
test slice come out empty. Every downstream accuracy then became `NaN` — reproducibly, and with
no error. A metric that is silently `NaN` is worse than a crash, because it looks like a result.
`random_split` now raises, and `test_random_split_refuses_to_silently_produce_an_empty_test_set`
pins it. Found by a test that was trying to check something else.

### Finding I3: two duplicate measurements disagreed for an unstated reason

All-zero feature rows are exact duplicates of one another and `torch.unique` counts them as
such, but cosine similarity is undefined for a zero vector, so they can never appear among the
near-duplicate pairs and were invisible to the straddling analysis. Rather than silently pick a
convention, `DuplicateReport` now carries `zero_feature_nodes`, so the gap between the two
counts is visible in the table rather than being an unexplained inconsistency.

### Finding I4: the neighbourhood-leakage comparison was invalid as first written

The neighbour vote can only predict test nodes that have at least one training-set neighbour.
On Cora's public split that is a minority of the test set. Comparing the vote's accuracy on
that subset against a GCN's accuracy on the *whole* test set is not a comparison — it is two
different questions with one label. The harness now also records GNN accuracy restricted to
exactly the covered subset, which is the last column of the baselines table.

The pre-fix numbers in `reports/instrument_bug_mlp_rng.json` cannot be reproduced by the code in
this repository, and that is deliberate: the current harness returns exactly `0.0`, which is the
whole point of the fix. The artifact is committed as the record of the measurement.

## Limitations

- **Hyperparameters are fixed, not tuned.** One setting (2 layers, hidden 64, dropout 0.5, Adam
  at lr 0.01 and weight decay 5e-4, up to 300 epochs, best-validation checkpoint) is used for
  every dataset, model and regime. Absolute accuracies are therefore below published numbers,
  especially on chameleon and squirrel where these homophily-oriented defaults are a poor fit.
  Inflation is a paired difference under identical hyperparameters, which is what it is designed
  to measure, but I have not tested whether tuning each regime separately would change it. It
  might: a model trained on a smaller graph may want different regularisation.
- **Only the standard splits.** Planetoid results use the single public split, so the reported
  variance is over initialisation only and does not include split variance. chameleon and
  squirrel use the ten geom-gcn splits, one per seed, so their variance includes both and is
  correspondingly larger. The two protocols are not directly comparable to each other.
- **The near-duplicate threshold is permissive.** It controls for "how similar do independent
  nodes get by chance" but not for the fact that the real graph has more pairs and therefore
  more chances to clear the bar. Counts are an upper bound.
- **Inference re-attaches test nodes to the full graph**, so test–test edges are visible at
  inference. That matches the usual "a batch of new nodes arrives together" deployment story,
  not the stricter "one node at a time" one, which would be a different and lower number.
- **Nothing here is causal.** Inflation is an accounting difference between two protocols. The
  density control separates the size effect from the test-node-specific effect, but neither
  component is a claim about mechanism inside the model.

## What I could not measure

- **The density control on chameleon and squirrel.** The geom-gcn splits partition every node
  into train/val/test, leaving no unlabelled pool to draw a size-matched control from. The
  headline inflation for those datasets is therefore *not* decomposed into density cost and
  test-specific cost, and I cannot say which it is.
- **A reproduction of the 42%/62% feature–label figures.** Those papers report "42% of nodes
  leak information between features and labels" for Cora without, in what I read, defining the
  measurement. My feature–label detector reports related quantities (logistic-regression
  accuracy, giveaway feature counts) but they are not the same statistic, so the numbers here
  neither confirm nor contradict theirs. My giveaway count on CiteSeer is zero; theirs is 62%
  of nodes. Those two things can both be true of different definitions.
- **Why my CiteSeer duplicate rate (1.1%) disagrees with the quoted 5%.** I could not retrieve
  the section of the primary source that would say how they counted, so I cannot tell whether
  this is a methodological difference or a genuine conflict.
- **Exact duplicate counts from the primary source.** I verified that Zou et al. claim
  duplicates and feature–label leaks in Cora and CiteSeer, and I verified OGB's quotation of
  their percentages verbatim, but I could not retrieve the section of Zou et al. containing the
  per-dataset numbers themselves. The duplicate counts in this README are my own measurements.
- **Anything about node ordering or dataset provenance.** This kit takes the datasets as
  `torch_geometric` ships them.
- **Whether inflation transfers to other architectures.** Two GNNs, one hidden size, one
  optimiser, one learning rate. No hyperparameter sweep, so I cannot claim the gap is invariant
  to architecture or tuning.
- **Anything at OGB scale.** Everything here fits on a laptop CPU. Whether inflation behaves the
  same on graphs three orders of magnitude larger is untested, and I would not extrapolate.

## Prior work

Each of these I fetched and checked before citing. Where I could not verify a claim, I say so
above rather than repeat it.

- **Zou et al., *Dimensional Reweighting Graph Convolutional Networks* (arXiv:1907.02237)** —
  the primary source for the Cora/CiteSeer data-quality claims. Its abstract describes "several
  fixes on duplicates, information leaks, and wrong labels" of standard node-classification
  benchmarks, and section 1 states that Cora and CiteSeer "suffer from duplicates and
  feature-label information leaks".
- **Hu et al., *Open Graph Benchmark* (arXiv:2005.00687, NeurIPS 2020)** — quotes those figures:
  "in Cora, 42% of the nodes leak information between their features and labels, and 1% of the
  nodes are duplicated. The situation for CiteSeer is even worse, with leakage rates of 62% and
  duplication rates of 5%." Note that OGB **attributes this to Zou et al.**; it is not OGB's own
  measurement, and the widely-repeated "OGB found 42% leakage in Cora" is a misattribution.
- **Platonov et al., *A critical look at the evaluation of GNNs under heterophily: are we really
  making progress?* (arXiv:2302.11640, ICLR 2023)** — the source for the chameleon/squirrel
  duplicates. Its abstract: "The most significant of these drawbacks is the presence of a large
  number of duplicate nodes in the datasets Squirrel and Chameleon, which leads to train-test
  data leakage", and "removing duplicate nodes strongly affects GNN performance on these
  datasets."
- **Guo & Vanden Broucke, *A Critique on Transductive Evaluation for GNN Node Classification*
  (DataMod 2024, LNCS 15556, 2025, doi:10.1007/978-3-031-87908-1_1)** — argues that transductive
  evaluation "suppresses the potential of GNNs to generalize", on the grounds that masking
  labels still leaves the graph attributes of the masked nodes visible during training, and
  proposes an inductive splitting scheme for single-graph datasets. This repository's inductive
  arm is the same idea; the contribution here is pricing it rather than proposing it.

## Layout

    src/leakgraph/
      data.py        dataset loading + the synthetic graph CI uses
      splits.py      Split, induced_subgraph, the three training views
      detectors.py   duplicates, feature-label, neighbour-label
      models.py      GCN, GraphSAGE, MLP, LabelProp
      experiment.py  the paired transductive/inductive harness
    experiments/
      run_audit.py            detectors + the full audit -> reports/
      run_density_control.py  the third arm
      make_tables.py          renders every table in this README from reports/
    tests/           32 tests, synthetic graph only, no network
    reports/         committed result artifacts

## Reproducing

    make venv        # .venv on python 3.12
    make test        # 32 tests, seconds, no downloads
    make detectors   # detectors only, no training
    make audit       # the full thing: downloads ~200MB, hours on a laptop CPU
    make control     # the density control
    make tables      # regenerate every table in this README from reports/

Every table above is written by `make tables` from the committed artifacts in `reports/`, into
the `<!--BEGIN:...-->` regions of this file. Nothing is typed in by hand. If a number here ever
stops matching the artifacts, `make tables` produces a dirty git diff and says so.

## Licence

MIT.
