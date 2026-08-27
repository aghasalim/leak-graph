# LeakGraph

**How much accuracy does transductive evaluation add on standard GNN node-classification benchmarks, and where does it come from?**

[![tests](https://img.shields.io/badge/tests-36%20passing-brightgreen.svg)](tests/)
[![licence](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)

---

## Abstract

Transductive node classification evaluates a model on nodes that were present in
the graph it trained on. The standard concern is that this leaks. This work
measures the size of that leak on five benchmarks and four models under one
protocol: identical splits, seeds, initialisation and epoch budget, differing
only in whether the test nodes existed in the training graph. The difference is
taken per seed and then averaged, so shared initialisation noise cancels.

Inflation is real on the homophilous citation graphs and small, at most 1.5
accuracy points, on Cora with GraphSAGE. On the two heterophilous graphs it
reverses: GCN *loses* 2.5 points to the transductive split. LabelProp and MLP
read exactly zero everywhere, which is the instrument check rather than a result,
since neither can distinguish the two splits. Exposure and leakage turn out to be
different quantities: squirrel exposes 40% of its test nodes to a labelled
training neighbour against Cora's 21% and still inflates less, because a vote
over those labels scores 21% there against a 19% majority baseline, versus 80%
against 13% on Cora. An attempt to decompose inflation into a density term and a
test-specific term mostly fails to resolve, and is reported as failing.

**Contributions.** (i) One harness measuring duplicate, feature, label and
neighbourhood-label leakage on the same splits with the same seeds. (ii) A
headline metric, leakage inflation, comparable across datasets and models. (iii)
Negative and density-matched controls, including a random split that reads zero.
(iv) Evidence that the duplicate rate is largely a choice of definition, ranging
from 0.04% to 45% on PubMed depending which is used.

---

## 1. Scope: this is tooling, not a discovery


Every phenomenon measured here is already documented in the literature. I did not find any of
it. What this repository contributes is a single harness that measures all of it on the same
splits with the same seeds under the same protocol, a headline metric that makes the cost
comparable across datasets and models, and a test suite that checks the instrument rather than
the result. Treat it as a measuring device and a replication, not as a novel claim.

See [Prior work](#prior-work) for what was already known, with sources I verified before
citing them.

## 2. The headline number
Inflation is real on the homophilous citation graphs and small: 1.5 accuracy points at most, on Cora with GraphSAGE.

![leakage inflation by dataset and model](reports/figures/leakage-inflation.png)

Full detail in [notes/METHODS.md](notes/METHODS.md#2-the-headline-number).
### What I actually found, including the parts that argue against the premise
I built this expecting transductive evaluation to be buying the GNNs a visible amount of accuracy.

Full detail in [notes/METHODS.md](notes/METHODS.md#what-i-actually-found-including-the-parts-that-argue-against-the-premise).
## 3. What the number means
An accuracy gap on its own is uninterpretable.

Full detail in [notes/METHODS.md](notes/METHODS.md#3-what-the-number-means).
## 4. Is the gap actually leakage?
The random split is the negative control: it has no temporal or structural reason to leak, so a non-zero reading there would be a fault in the harness rather than a finding.

![the same measurement under three conditions](reports/figures/controls.png)

Full detail in [notes/METHODS.md](notes/METHODS.md#4-is-the-gap-actually-leakage).
### Recovering the control where the split leaves no pool
A pool of unlabelled nodes can be manufactured out of the test set itself.

Full detail in [notes/METHODS.md](notes/METHODS.md#recovering-the-control-where-the-split-leaves-no-pool).
### A second split scheme
The other route is to stop using the shipped splits.

Full detail in [notes/METHODS.md](notes/METHODS.md#a-second-split-scheme).
## 5. Decomposing by duplicates
Adding test nodes does two things at once: it makes the graph denser, helping any node, and it exposes the test nodes' own neighbourhoods.

![inflation split into a density term and a test-specific term](reports/figures/inflation-decomposition.png)

Full detail in [notes/METHODS.md](notes/METHODS.md#5-decomposing-by-duplicates).
## 6. The detectors
"Duplicate node" has no single meaning, and the choice dominates the answer.

![duplicate rate under sixteen definitions](reports/figures/duplicate-definitions.png)
![test-set exposure and what it is worth](reports/figures/leakage-channels.png)

Full detail in [notes/METHODS.md](notes/METHODS.md#6-the-detectors).
### Duplicate and near-duplicate nodes
<!--BEGIN:duplicates--> | dataset | nodes | exact duplicate nodes | all-zero feature rows | near-dup pairs | cutoff | straddling pairs | test nodes with a train twin | |---|---|---|---|---|---|---|---| | Cora | 2708 | 27 (1.0%) | 0 | 440 | 0.474 | 10 | 8 / 1000 (0.8%) | | CiteSeer | 3327 | 35 (1.1%) | 15 | 2790 | 0.292 | 51 | 45 / 1000 (4.5%) | | PubMed | 19717 | 7 (0.0%) | 0 | 188 | 0.904 | 1 | 1 / 1000 (0.1%) | | chameleon | 2277 | 375 (16.5%) | 233 | 3395 | 0.378 | 683 | 128 / 456 (28.1%) | | squirrel | 5201 | 265 (5.1%) | 165 | 7348 | 0.354 | 1580 | 377 / 1041 (36.2%) | <!--END:duplicates--> Two of these line up with the literature and one does not.

Full detail in [notes/METHODS.md](notes/METHODS.md#duplicate-and-near-duplicate-nodes).
### Trying to reconcile the CiteSeer duplicate rate
"Duplicate" is not one definition, so the disagreement might be a definitional difference rather than a conflict.`make duplicate-definitions` evaluates eighteen readings of it on all five datasets: exact feature match counted four different ways, with and without labels, with and without the all-zero rows, near-duplicates at five cosine cutoffs and three Jaccard cutoffs, and duplication defined on the graph instead of the features.

Full detail in [notes/METHODS.md](notes/METHODS.md#trying-to-reconcile-the-citeseer-duplicate-rate).
### Feature, label leakage
How much of the label is already in a node's own features, with no graph at all.

Full detail in [notes/METHODS.md](notes/METHODS.md#feature-label-leakage).
### Neighbourhood label leakage

Covered by the neighbour-vote column in [What the number means](#what-the-number-means).
Full output, including coverage and the accuracy restricted to covered nodes, is in
`reports/detectors.json`.

## 7. The inductive split is the thing that has to be right
Everything above is worthless if the inductive split is not actually inductive, and it is easy to get subtly wrong.

Full detail in [notes/METHODS.md](notes/METHODS.md#7-the-inductive-split-is-the-thing-that-has-to-be-right).
## 8. Instrument bugs

Both of these were found by the controls, not by inspection. They are recorded here rather than
quietly patched, because a harness that has never been caught being wrong is a harness nobody
has checked.

### Finding I1: the MLP control was reporting leakage that was actually a dropout RNG offset
The MLP cannot have a transductive/inductive gap.

Full detail in [notes/METHODS.md](notes/METHODS.md#finding-i1-the-mlp-control-was-reporting-leakage-that-was-actually-a-dropout-rng-offset).
### Finding I2:`random_split` silently produced an empty test set

Asking the Planetoid defaults (500 val, 1000 test) of a graph too small to supply them made the
test slice come out empty. Every downstream accuracy then became`NaN`, reproducibly, and with
no error. A metric that is silently`NaN` is worse than a crash, because it looks like a result.
`random_split` now raises, and`test_random_split_refuses_to_silently_produce_an_empty_test_set`
pins it. Found by a test that was trying to check something else.

### Finding I3: two duplicate measurements disagreed for an unstated reason

All-zero feature rows are exact duplicates of one another and`torch.unique` counts them as
such, but cosine similarity is undefined for a zero vector, so they can never appear among the
near-duplicate pairs and were invisible to the straddling analysis. Rather than silently pick a
convention,`DuplicateReport` now carries`zero_feature_nodes`, so the gap between the two
counts is visible in the table rather than being an unexplained inconsistency.

### Finding I4: the neighbourhood-leakage comparison was invalid as first written
The neighbour vote can only predict test nodes that have at least one training-set neighbour.

Full detail in [notes/METHODS.md](notes/METHODS.md#finding-i4-the-neighbourhood-leakage-comparison-was-invalid-as-first-written).
## 9. Limitations

- **Hyperparameters are fixed, not tuned.** One setting (2 layers, hidden 64, dropout 0.5, Adam
  at lr 0.01 and weight decay 5e-4, up to 300 epochs, best-validation checkpoint) is used for
  every dataset, model and regime. Absolute accuracies are therefore below published numbers,
  especially on chameleon and squirrel where these homophily-oriented defaults are a poor fit.
  Inflation is a paired difference under identical hyperparameters, which is what it is designed
  to measure, but I have not tested whether tuning each regime separately would change it. It
  might: a model trained on a smaller graph may want different regularisation.
- **The headline tables use the standard splits.** Planetoid results use the single public
  split, so their variance is over initialisation only and does not include split variance.
  chameleon and squirrel use the ten geom-gcn splits, one per seed, so their variance includes
  both and is correspondingly larger. The two protocols are not directly comparable to each
  other. A second, uniform random-split scheme is now measured alongside them
  ([above](#a-second-split-scheme)), and it is a *different* protocol rather than a robustness
  check of the same one: 20 labels per class is far more label-scarce than the geom-gcn splits'
  48%, so on chameleon and squirrel it trains a much weaker model. Where the two schemes
  disagree, label scarcity is confounded with the split.
- **Still one hyperparameter setting and one scale.** No tuning per regime, nothing at OGB
  scale.
- **The near-duplicate threshold is permissive.** It controls for "how similar do independent
  nodes get by chance" but not for the fact that the real graph has more pairs and therefore
  more chances to clear the bar. Counts are an upper bound.
- **Inference re-attaches test nodes to the full graph**, so test, test edges are visible at
  inference. That matches the usual "a batch of new nodes arrives together" deployment story,
  not the stricter "one node at a time" one, which would be a different and lower number.
- **Nothing here is causal.** Inflation is an accounting difference between two protocols. The
  density control separates the size effect from the test-node-specific effect, but neither
  component is a claim about mechanism inside the model.

## 10. What I could not measure
- ~~**The density control on chameleon and squirrel.**~~ Now measured, two ways, by reserving half the test set as the removal pool, and again under a second split scheme that leaves a pool of its own.

Full detail in [notes/METHODS.md](notes/METHODS.md#10-what-i-could-not-measure).
## 11. Prior work

Each of these I fetched and checked before citing. Where I could not verify a claim, I say so
above rather than repeat it.

- **Zou et al., *Dimensional Reweighting Graph Convolutional Networks* (arXiv:1907.02237)**
  the primary source for the Cora/CiteSeer data-quality claims. Its abstract describes "several
  fixes on duplicates, information leaks, and wrong labels" of standard node-classification
  benchmarks, and section 1 states that Cora and CiteSeer "suffer from duplicates and
  feature-label information leaks".
- **Hu et al., *Open Graph Benchmark* (arXiv:2005.00687, NeurIPS 2020)**, quotes those figures:
  "in Cora, 42% of the nodes leak information between their features and labels, and 1% of the
  nodes are duplicated. The situation for CiteSeer is even worse, with leakage rates of 62% and
  duplication rates of 5%." Note that OGB **attributes this to Zou et al.**; it is not OGB's own
  measurement, and the widely-repeated "OGB found 42% leakage in Cora" is a misattribution.
- **Platonov et al., *A critical look at the evaluation of GNNs under heterophily: are we really
  making progress?* (arXiv:2302.11640, ICLR 2023)**, the source for the chameleon/squirrel
  duplicates. Its abstract: "The most significant of these drawbacks is the presence of a large
  number of duplicate nodes in the datasets Squirrel and Chameleon, which leads to train-test
  data leakage", and "removing duplicate nodes strongly affects GNN performance on these
  datasets."
- **Guo & Vanden Broucke, *A Critique on Transductive Evaluation for GNN Node Classification*
  (DataMod 2024, LNCS 15556, 2025, doi:10.1007/978-3-031-87908-1_1)**, argues that transductive
  evaluation "suppresses the potential of GNNs to generalize", on the grounds that masking
  labels still leaves the graph attributes of the masked nodes visible during training, and
  proposes an inductive splitting scheme for single-graph datasets. This repository's inductive
  arm is the same idea; the contribution here is pricing it rather than proposing it.

## 12. Repository layout

    src/leakgraph/
      data.py        dataset loading + the synthetic graph CI uses
      splits.py      Split, induced_subgraph, the three training views
      detectors.py   duplicates, feature-label, neighbour-label
      models.py      GCN, GraphSAGE, MLP, LabelProp
      experiment.py  the paired transductive/inductive harness
    experiments/
      run_audit.py                  detectors + the full audit -> reports/
      run_density_control.py        the third arm; --bisect and --random-splits
      run_duplicate_definitions.py  eighteen readings of "duplicate", against the quotes
      make_tables.py                renders every table in this README from reports/
    tests/           36 tests, synthetic graph only, no network
    reports/         committed result artifacts

## 13. Reproducing

    make venv                    # .venv on python 3.12
    make test                    # 36 tests, seconds, no downloads
    make detectors               # detectors only, no training
    make audit                   # the full thing: downloads ~200MB, hours on a laptop CPU
    make control                 # the density control, where the split leaves a pool
    make control-bisected        # the same control everywhere, via a halved test set
    make control-random-splits   # a second split scheme, with its own control
    make duplicate-definitions   # no training: which definition matches the quoted rates
    make tables                  # regenerate every table in this README from reports/

Every table above is written by`make tables` from the committed artifacts in`reports/`, into
the`<!--BEGIN:...-->` regions of this file. Nothing is typed in by hand. If a number here ever
stops matching the artifacts,`make tables` produces a dirty git diff and says so.

## 14. Licence

MIT.
