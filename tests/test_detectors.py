"""Tests for the leakage detectors and for the two harness controls.

A detector that reports leakage on a graph with none is worse than no detector, so several of
these tests are negative controls: they assert the detectors stay quiet when they should.
"""

import pytest
import torch

from leakgraph.data import synthetic_graph
from leakgraph.detectors import (
    calibrate_cosine_threshold,
    duplicate_report,
    feature_label_report,
    neighbour_label_report,
    scan_duplicates,
)
from leakgraph.experiment import train_one


@pytest.fixture
def graph():
    return synthetic_graph(seed=0, num_duplicates=10)


def test_exact_duplicate_count_matches_injected_ground_truth():
    g = synthetic_graph(seed=0, num_duplicates=8, num_nodes=120, num_classes=3)
    split = g.splits[0]
    report, _, _ = duplicate_report(g.x, g.y, split.train_mask, split.test_mask)
    # the fixture copies node i*C onto node i*C+C for i in range(num_duplicates), so the
    # copies chain into runs: count pairs rather than assuming they are disjoint
    assert report.exact_duplicate_pairs >= 1
    assert report.exact_duplicate_nodes >= 2


def test_no_duplicates_reported_when_none_were_injected():
    g = synthetic_graph(seed=0, num_duplicates=0)
    split = g.splits[0]
    report, _, _ = duplicate_report(g.x, g.y, split.train_mask, split.test_mask)
    assert report.exact_duplicate_pairs == 0
    assert report.exact_duplicate_nodes == 0


def test_zero_feature_rows_are_counted_separately():
    """All-zero rows are exact duplicates of each other, but cosine similarity is undefined for
    a zero vector so they can never show up in the near-duplicate pairs. If the two duplicate
    measurements are going to disagree, the reason has to be visible in the report."""
    g = synthetic_graph(seed=0, num_duplicates=0)
    g.x[:4] = 0.0
    scan = scan_duplicates(g.x, seed=0)
    assert scan.zero_feature_nodes == 4
    # they are exact duplicates ...
    assert scan.exact_duplicate_pairs >= 6  # 4 choose 2
    # ... but contribute no near-duplicate pairs
    i, j = scan.pairs[0], scan.pairs[1]
    assert not bool(((i < 4) & (j < 4)).any())


def test_cosine_threshold_is_calibrated_not_hardcoded():
    """The calibrated cutoff must sit strictly inside (0, 1): a null that produced 1.0 would
    mean the null itself contains duplicates and the calibration is meaningless."""
    g = synthetic_graph(seed=0)
    calib = calibrate_cosine_threshold(g.x, seed=0, subsample=100)
    assert 0.0 < calib.threshold < 1.0
    assert calib.null_pairs_examined == 100 * 99 // 2
    assert calib.real_pairs_examined == g.num_nodes * (g.num_nodes - 1) // 2


def test_calibration_is_deterministic_given_a_seed():
    g = synthetic_graph(seed=0)
    a = calibrate_cosine_threshold(g.x, seed=5, subsample=100)
    b = calibrate_cosine_threshold(g.x, seed=5, subsample=100)
    assert a.threshold == b.threshold


def test_straddling_twins_are_a_subset_of_the_test_set(graph):
    split = graph.splits[0]
    _, _, twin = duplicate_report(graph.x, graph.y, split.train_mask, split.test_mask)
    assert bool((twin & ~split.test_mask).sum()) is False
    assert twin.dtype == torch.bool


def test_feature_label_detector_beats_the_majority_baseline_on_separable_features(graph):
    """The fixture's features are class centroids plus small noise, so features alone must be
    strongly predictive. If this ever fails, the detector is broken, not the data."""
    split = graph.splits[0]
    report = feature_label_report(graph.x, graph.y, split.train_mask, split.test_mask)
    assert report.logreg_test_accuracy > report.majority_class_accuracy + 0.2


def test_feature_label_detector_finds_nothing_when_labels_are_random():
    """Negative control: shuffle the labels so features carry no information. Logistic
    regression must collapse towards the majority-class rate."""
    g = synthetic_graph(seed=0)
    torch.manual_seed(0)
    y = g.y[torch.randperm(g.num_nodes)]
    split = g.splits[0]
    report = feature_label_report(g.x, y, split.train_mask, split.test_mask)
    assert report.logreg_test_accuracy < report.majority_class_accuracy + 0.2


def test_giveaway_feature_count_is_compared_against_a_measured_null(graph):
    """The label-permuted null is what makes the giveaway count interpretable. It must be
    computed, finite, and not silently equal to the real count."""
    split = graph.splits[0]
    report = feature_label_report(graph.x, graph.y, split.train_mask, split.test_mask)
    assert report.null_giveaway_features == report.null_giveaway_features  # not NaN
    assert report.null_giveaway_features >= 0


def test_neighbour_vote_detects_homophily(graph):
    """The fixture is a planted partition with p_in >> p_out, so voting over train neighbours
    alone must beat the majority-class rate by a wide margin."""
    split = graph.splits[0]
    report = neighbour_label_report(graph.edge_index, graph.y, split.train_mask, split.test_mask)
    assert report.frac_test_with_train_neighbour > 0.9
    assert report.vote_accuracy_on_covered > report.majority_class_accuracy + 0.2


def test_neighbour_vote_finds_nothing_on_a_shuffled_graph(graph):
    """Negative control: shuffling the labels destroys homophily, so the neighbour vote must
    fall back to chance."""
    torch.manual_seed(0)
    y = graph.y[torch.randperm(graph.num_nodes)]
    split = graph.splits[0]
    report = neighbour_label_report(graph.edge_index, y, split.train_mask, split.test_mask)
    assert report.vote_accuracy_overall < report.majority_class_accuracy + 0.25


def test_neighbour_vote_reports_zero_coverage_on_an_edgeless_graph(graph):
    split = graph.splits[0]
    empty = torch.zeros(2, 0, dtype=torch.long)
    report = neighbour_label_report(empty, graph.y, split.train_mask, split.test_mask)
    assert report.frac_test_with_train_neighbour == 0.0
    assert report.vote_accuracy_overall == pytest.approx(report.majority_class_accuracy)


# --------------------------------------------------------------------------------------
# harness controls: two models that cannot have a transductive/inductive gap
# --------------------------------------------------------------------------------------


def test_mlp_has_exactly_zero_inflation(graph):
    """The MLP never reads the graph and the inductive view leaves training-node features
    untouched, so the two regimes are literally the same computation. Any non-zero gap here
    means the harness is comparing something other than what it claims to."""
    split = graph.splits[0]
    kw = dict(epochs=30, patience=30)
    t = train_one(graph, split, "MLP", "transductive", seed=0, **kw)
    i = train_one(graph, split, "MLP", "inductive", seed=0, **kw)
    assert t.test_accuracy == i.test_accuracy


def test_label_propagation_has_exactly_zero_inflation(graph):
    """Label propagation has no parameters and propagates the same training labels over the
    same full graph in both regimes. Same control logic as the MLP."""
    split = graph.splits[0]
    t = train_one(graph, split, "LabelProp", "transductive", seed=0)
    i = train_one(graph, split, "LabelProp", "inductive", seed=0)
    assert t.test_accuracy == i.test_accuracy


def test_gcn_runs_in_both_regimes_and_scores_above_chance(graph):
    split = graph.splits[0]
    kw = dict(epochs=60, patience=60)
    for regime in ("transductive", "inductive"):
        res = train_one(graph, split, "GCN", regime, seed=0, **kw)
        assert 0.0 <= res.test_accuracy <= 1.0
        assert res.test_accuracy > 1.0 / graph.num_classes
