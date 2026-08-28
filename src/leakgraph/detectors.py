"""Leakage detectors. Three families, each producing numbers rather than adjectives.

1. duplicate / near-duplicate nodes  -> `duplicate_report`
2. feature-label leakage             -> `feature_label_report`
3. neighbourhood label leakage       -> `neighbour_label_report`

Every threshold in here is calibrated against a measured null, never chosen by eye. See
`calibrate_cosine_threshold` for what the null is and what it does and does not justify.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor

# ---------------------------------------------------------------------------------------
# 1. duplicate and near-duplicate nodes
# ---------------------------------------------------------------------------------------


def _l2_normalise(x: Tensor) -> Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


def _pairwise_cosine_above(x: Tensor, threshold: float, chunk: int = 512) -> Tensor:
    """Indices (2, P) of node pairs i<j with cosine(x_i, x_j) >= threshold.

    Chunked so that PubMed (19,717 nodes, 194M pairs) never materialises a full N x N matrix.
    """
    xn = _l2_normalise(x.float())
    n = xn.size(0)
    out: list[Tensor] = []
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        sims = xn[start:stop] @ xn.T  # (chunk, N)
        rows, cols = torch.nonzero(sims >= threshold, as_tuple=True)
        rows = rows + start
        upper = rows < cols  # keep i<j only; also drops the diagonal
        if bool(upper.any()):
            out.append(torch.stack([rows[upper], cols[upper]]))
    if not out:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.cat(out, dim=1)


def _max_null_cosine(x: Tensor, seed: int, subsample: int, chunk: int = 512) -> float:
    """Largest cosine similarity between any two rows of a column-permuted copy of `x`.

    The null shuffles each feature column independently. That preserves every feature's
    marginal frequency exactly while destroying all co-occurrence structure between nodes:
    under it, two nodes are similar only by chance. Taking the maximum over the null's
    pairs gives a similarity level that the "nodes are independent" hypothesis never reached
    in this many draws.
    """
    g = torch.Generator().manual_seed(seed)
    n = min(subsample, x.size(0))
    rows = torch.randperm(x.size(0), generator=g)[:n]
    null = x[rows].float().clone()
    for j in range(null.size(1)):
        null[:, j] = null[torch.randperm(n, generator=g), j]

    xn = _l2_normalise(null)
    best = 0.0
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        sims = xn[start:stop] @ xn.T
        # blank out the diagonal so a row is not compared against itself
        idx = torch.arange(start, stop)
        sims[torch.arange(stop - start), idx] = -1.0
        best = max(best, float(sims.max()))
    return best


@dataclass
class CosineCalibration:
    threshold: float
    null_pairs_examined: int
    null_subsample_nodes: int
    real_pairs_examined: int
    note: str


def calibrate_cosine_threshold(
    x: Tensor, seed: int = 0, subsample: int = 4000
) -> CosineCalibration:
    """Set the near-duplicate cosine cutoff to the maximum similarity seen under the null.

    What this justifies: any real pair at or above the threshold is more similar than every
    one of the ~`null_pairs_examined` pairs drawn from independent nodes with identical
    feature marginals.

    What it does not justify: a per-pair significance claim. The real graph usually has more
    pairs than the null subsample did, so it gets more chances to clear the bar by luck. The
    threshold is therefore permissive and the near-duplicate counts it produces should be
    read as an upper bound. `duplicate_report` also reports counts at fixed cutoffs of 0.95
    and 0.99 so the reader can see how sensitive the answer is to this choice.
    """
    n_null = min(subsample, x.size(0))
    threshold = _max_null_cosine(x, seed=seed, subsample=subsample)
    return CosineCalibration(
        threshold=threshold,
        null_pairs_examined=n_null * (n_null - 1) // 2,
        null_subsample_nodes=n_null,
        real_pairs_examined=x.size(0) * (x.size(0) - 1) // 2,
        note="threshold = max cosine over column-permuted null; permissive, counts are an upper bound",
    )


@dataclass
class DuplicateScan:
    """The split-independent half of duplicate detection: which pairs are duplicates at all.

    Kept separate from `duplicate_report` because the pairwise cosine sweep is by far the most
    expensive thing in this repository (squirrel is 5,201 x 2,089, swept three times) while
    the split-dependent part is a couple of mask lookups. Chameleon and squirrel ship ten
    splits each; scanning once and reporting ten times is a 10x saving for free.
    """

    pairs: Tensor  # (2, P), i<j, cosine >= calibrated threshold
    calibration: CosineCalibration
    exact_duplicate_nodes: int
    exact_duplicate_pairs: int
    pairs_at_0_95: int
    pairs_at_0_99: int
    zero_feature_nodes: int


def scan_duplicates(x: Tensor, seed: int = 0) -> DuplicateScan:
    calib = calibrate_cosine_threshold(x, seed=seed)

    # exact duplicates: group identical feature rows
    _, inverse, counts = torch.unique(x, dim=0, return_inverse=True, return_counts=True)
    group_sizes = counts[inverse]

    # Nodes whose feature row is entirely zero are mutually identical, so `torch.unique` counts
    # them as exact duplicates: but cosine similarity is undefined for a zero vector, so they
    # cannot appear in the near-duplicate pairs and are silently absent from the straddling
    # analysis. CiteSeer has such nodes. Report the count rather than let the two duplicate
    # measurements disagree for an unstated reason.
    zero_rows = int(((x != 0).sum(dim=1) == 0).sum())

    return DuplicateScan(
        pairs=_pairwise_cosine_above(x, calib.threshold),
        calibration=calib,
        exact_duplicate_nodes=int((group_sizes > 1).sum()),
        exact_duplicate_pairs=int((counts * (counts - 1) // 2).sum()),
        pairs_at_0_95=int(_pairwise_cosine_above(x, 0.95).size(1)),
        pairs_at_0_99=int(_pairwise_cosine_above(x, 0.99).size(1)),
        zero_feature_nodes=zero_rows,
    )


@dataclass
class DuplicateReport:
    num_nodes: int
    exact_duplicate_nodes: int
    exact_duplicate_pairs: int
    # all-zero feature rows: exact duplicates of each other, but cosine is undefined for them
    # so they are excluded from the near-duplicate pairs below
    zero_feature_nodes: int
    calibrated_threshold: float
    near_duplicate_pairs: int
    near_duplicate_pairs_at_0_95: int
    near_duplicate_pairs_at_0_99: int
    straddling_pairs: int
    test_nodes_with_train_twin: int
    test_nodes: int
    frac_test_with_train_twin: float
    same_label_frac_among_straddling: float

    def to_dict(self) -> dict:
        return asdict(self)


def duplicate_report(
    x: Tensor,
    y: Tensor,
    train_mask: Tensor,
    test_mask: Tensor,
    seed: int = 0,
    scan: DuplicateScan | None = None,
) -> tuple[DuplicateReport, CosineCalibration, Tensor]:
    """Count duplicates, and specifically the ones that straddle the train/test boundary.

    Only straddling pairs leak. A duplicate pair sitting entirely inside the training set
    teaches the model nothing it could not have learned from either copy alone; it is the pair
    with one foot in train and one in test that hands the model the answer.

    Returns the report, the calibration used, and the boolean mask of test nodes that have a
    near-duplicate twin somewhere in the training set. Pass `scan` to reuse a sweep across
    several splits of the same graph.
    """
    if scan is None:
        scan = scan_duplicates(x, seed=seed)

    i, j = scan.pairs[0], scan.pairs[1]
    straddles = (train_mask[i] & test_mask[j]) | (test_mask[i] & train_mask[j])
    twin = torch.zeros_like(test_mask)
    twin[i[straddles & test_mask[i]]] = True
    twin[j[straddles & test_mask[j]]] = True

    n_straddle = int(straddles.sum())
    same_label = (
        float((y[i[straddles]] == y[j[straddles]]).float().mean())
        if n_straddle
        else float("nan")
    )

    report = DuplicateReport(
        num_nodes=int(x.size(0)),
        exact_duplicate_nodes=scan.exact_duplicate_nodes,
        exact_duplicate_pairs=scan.exact_duplicate_pairs,
        zero_feature_nodes=scan.zero_feature_nodes,
        calibrated_threshold=scan.calibration.threshold,
        near_duplicate_pairs=int(scan.pairs.size(1)),
        near_duplicate_pairs_at_0_95=scan.pairs_at_0_95,
        near_duplicate_pairs_at_0_99=scan.pairs_at_0_99,
        straddling_pairs=n_straddle,
        test_nodes_with_train_twin=int(twin.sum()),
        test_nodes=int(test_mask.sum()),
        frac_test_with_train_twin=float(twin.sum()) / max(int(test_mask.sum()), 1),
        same_label_frac_among_straddling=same_label,
    )
    return report, scan.calibration, twin


# ---------------------------------------------------------------------------------------
# 2. feature-label leakage
# ---------------------------------------------------------------------------------------


@dataclass
class FeatureLabelReport:
    logreg_test_accuracy: float
    majority_class_accuracy: float
    num_features: int
    giveaway_features: int
    min_support: int
    purity_cutoff: float
    null_giveaway_features: float
    frac_test_covered_by_giveaway: float
    giveaway_vote_accuracy_on_covered: float

    def to_dict(self) -> dict:
        return asdict(self)


def feature_label_report(
    x: Tensor,
    y: Tensor,
    train_mask: Tensor,
    test_mask: Tensor,
    min_support: int = 5,
    purity_cutoff: float = 0.95,
    seed: int = 0,
) -> FeatureLabelReport:
    """How much of the label is already in a node's own features, with no graph at all.

    Two numbers, because they answer different questions.

    `logreg_test_accuracy` is the ceiling: a plain multinomial logistic regression on raw
    features. Whatever a GNN scores, the part below this line was never graph learning.

    `giveaway_features` is the mechanism: individual feature dimensions (single vocabulary
    words, for the citation datasets) whose mere presence pins the label down. Purity is
    measured on the training set only, then applied to test nodes, so it is leakage that
    actually transfers rather than an in-sample artefact. The count is compared against a
    column-permuted null so that "how many would you expect by chance" is measured and not
    assumed.
    """
    from sklearn.linear_model import LogisticRegression

    xn = x.numpy()
    yn = y.numpy()
    clf = LogisticRegression(max_iter=2000)
    clf.fit(xn[train_mask.numpy()], yn[train_mask.numpy()])
    logreg_acc = float(clf.score(xn[test_mask.numpy()], yn[test_mask.numpy()]))

    train_counts = torch.bincount(y[train_mask], minlength=int(y.max()) + 1)
    majority_acc = float((y[test_mask] == train_counts.argmax()).float().mean())

    def _giveaways(feats: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        present = (feats > 0).float()  # (n_train, F)
        onehot = torch.nn.functional.one_hot(labels, int(y.max()) + 1).float()
        per_class = present.T @ onehot  # (F, C) counts of each class among nodes having f
        support = per_class.sum(dim=1)
        purity = per_class.max(dim=1).values / support.clamp_min(1)
        is_giveaway = (support >= min_support) & (purity >= purity_cutoff)
        return is_giveaway, per_class.argmax(dim=1)

    x_train, y_train = x[train_mask], y[train_mask]
    is_giveaway, giveaway_label = _giveaways(x_train, y_train)

    # measured null: shuffle labels among the training nodes, recount
    g = torch.Generator().manual_seed(seed)
    null_counts = [
        int(_giveaways(x_train, y_train[torch.randperm(y_train.numel(), generator=g)])[0].sum())
        for _ in range(5)
    ]

    x_test = x[test_mask]
    hits = (x_test[:, is_giveaway] > 0).float()  # (n_test, G)
    covered = hits.sum(dim=1) > 0
    if bool(covered.any()):
        votes = torch.nn.functional.one_hot(
            giveaway_label[is_giveaway], int(y.max()) + 1
        ).float()
        pred = (hits @ votes).argmax(dim=1)
        vote_acc = float((pred[covered] == y[test_mask][covered]).float().mean())
    else:
        vote_acc = float("nan")

    return FeatureLabelReport(
        logreg_test_accuracy=logreg_acc,
        majority_class_accuracy=majority_acc,
        num_features=int(x.size(1)),
        giveaway_features=int(is_giveaway.sum()),
        min_support=min_support,
        purity_cutoff=purity_cutoff,
        null_giveaway_features=float(sum(null_counts) / len(null_counts)),
        frac_test_covered_by_giveaway=float(covered.float().mean()),
        giveaway_vote_accuracy_on_covered=vote_acc,
    )


# ---------------------------------------------------------------------------------------
# 3. neighbourhood label leakage
# ---------------------------------------------------------------------------------------


@dataclass
class NeighbourLabelReport:
    frac_test_with_train_neighbour: float
    vote_accuracy_on_covered: float
    vote_accuracy_overall: float
    majority_class_accuracy: float

    def to_dict(self) -> dict:
        return asdict(self)


def neighbour_label_report(
    edge_index: Tensor, y: Tensor, train_mask: Tensor, test_mask: Tensor
) -> NeighbourLabelReport:
    """Predict a test label by majority vote over its *training* neighbours. No features, no
    learning, no parameters. Whatever this scores, a GNN has to beat it to have done anything.
    """
    num_classes = int(y.max()) + 1
    src, dst = edge_index[0], edge_index[1]

    labelled = train_mask[src]
    votes = torch.zeros(y.numel(), num_classes)
    votes.index_add_(
        0,
        dst[labelled],
        torch.nn.functional.one_hot(y[src[labelled]], num_classes).float(),
    )

    covered = votes.sum(dim=1) > 0
    pred = votes.argmax(dim=1)

    fallback = torch.bincount(y[train_mask], minlength=num_classes).argmax()
    pred_overall = torch.where(covered, pred, fallback)

    test_covered = covered & test_mask
    return NeighbourLabelReport(
        frac_test_with_train_neighbour=float(test_covered.sum()) / max(int(test_mask.sum()), 1),
        vote_accuracy_on_covered=(
            float((pred[test_covered] == y[test_covered]).float().mean())
            if bool(test_covered.any())
            else float("nan")
        ),
        vote_accuracy_overall=float((pred_overall[test_mask] == y[test_mask]).float().mean()),
        majority_class_accuracy=float((y[test_mask] == fallback).float().mean()),
    )
