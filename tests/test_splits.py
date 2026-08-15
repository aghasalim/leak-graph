"""Tests for the split logic.

The property that matters most in this repository is that the inductive training view really
does not contain the test nodes. `test_inductive_training_is_blind_to_test_features` is the
strongest statement of it: it corrupts the hidden nodes' features and asserts the inductive
training loss is bit-identical, while the transductive one moves.
"""

import pytest
import torch

from leakgraph.data import synthetic_graph
from leakgraph.models import GCN
from leakgraph.splits import Split, induced_subgraph, make_training_view, random_split


@pytest.fixture
def graph():
    return synthetic_graph(seed=0)


def test_split_rejects_overlapping_masks():
    m = torch.zeros(5, dtype=torch.bool)
    both = m.clone()
    both[0] = True
    with pytest.raises(ValueError, match="overlap"):
        Split(both, both.clone(), m.clone())


def test_split_rejects_non_bool_masks():
    m = torch.zeros(5, dtype=torch.bool)
    with pytest.raises(ValueError, match="bool"):
        Split(torch.zeros(5, dtype=torch.long), m, m.clone())


def test_induced_subgraph_drops_nodes_and_relabels(graph):
    keep = ~graph.splits[0].test_mask
    x, y, ei, old_to_new = induced_subgraph(graph.x, graph.y, graph.edge_index, keep)

    assert x.size(0) == int(keep.sum())
    assert y.size(0) == int(keep.sum())
    # every surviving edge endpoint is a valid index into the new node set
    assert int(ei.max()) < x.size(0)
    assert int(ei.min()) >= 0
    # dropped nodes have no new index; kept nodes map to a contiguous 0..k-1 range
    assert (old_to_new[~keep] == -1).all()
    assert sorted(old_to_new[keep].tolist()) == list(range(int(keep.sum())))


def test_induced_subgraph_preserves_features_and_labels_of_kept_nodes(graph):
    keep = ~graph.splits[0].test_mask
    x, y, _, old_to_new = induced_subgraph(graph.x, graph.y, graph.edge_index, keep)
    kept_ids = torch.nonzero(keep).flatten()
    assert torch.equal(x[old_to_new[kept_ids]], graph.x[kept_ids])
    assert torch.equal(y[old_to_new[kept_ids]], graph.y[kept_ids])


def test_induced_subgraph_keeps_exactly_the_edges_with_both_endpoints_kept(graph):
    keep = ~graph.splits[0].test_mask
    _, _, ei, _ = induced_subgraph(graph.x, graph.y, graph.edge_index, keep)
    expected = int((keep[graph.edge_index[0]] & keep[graph.edge_index[1]]).sum())
    assert ei.size(1) == expected
    assert ei.size(1) < graph.edge_index.size(1), "fixture should have cross-boundary edges"


def test_inductive_view_has_no_test_nodes(graph):
    split = graph.splits[0]
    view = make_training_view(graph.x, graph.y, graph.edge_index, split, "inductive")
    assert view.x.size(0) == graph.num_nodes - int(split.test_mask.sum())
    assert int(view.train_mask.sum()) == int(split.train_mask.sum())
    assert int(view.val_mask.sum()) == int(split.val_mask.sum())


def test_transductive_view_is_the_whole_graph(graph):
    split = graph.splits[0]
    view = make_training_view(graph.x, graph.y, graph.edge_index, split, "transductive")
    assert torch.equal(view.x, graph.x)
    assert torch.equal(view.edge_index, graph.edge_index)


def test_inductive_training_is_blind_to_test_features(graph):
    """The core correctness property, stated as an experiment rather than an assertion about
    shapes: replace every test node's features with noise. The inductive training loss must
    not move at all. The transductive one must."""
    split = graph.splits[0]
    corrupted = graph.x.clone()
    torch.manual_seed(99)
    corrupted[split.test_mask] = torch.randn_like(corrupted[split.test_mask]) * 100

    def training_loss(x, regime):
        view = make_training_view(x, graph.y, graph.edge_index, split, regime)
        torch.manual_seed(0)
        net = GCN(x.size(1), 8, graph.num_classes, dropout=0.0)
        out = net(view.x, view.edge_index)
        return torch.nn.functional.cross_entropy(
            out[view.train_mask], view.y[view.train_mask]
        ).item()

    assert training_loss(graph.x, "inductive") == training_loss(corrupted, "inductive")
    assert training_loss(graph.x, "transductive") != training_loss(corrupted, "transductive")


def test_inductive_training_is_blind_to_test_edges(graph):
    """Same idea for topology. Every test-node endpoint is rewired to a different, randomly
    chosen test node, which changes the graph substantially while keeping each edge's
    test/non-test pattern intact. The inductive training loss must not move."""
    split = graph.splits[0]
    ei = graph.edge_index
    test_ids = torch.nonzero(split.test_mask).flatten()
    torch.manual_seed(7)
    rewired = ei.clone()
    for row in (0, 1):
        is_test = split.test_mask[ei[row]]
        rewired[row, is_test] = test_ids[
            torch.randint(0, test_ids.numel(), (int(is_test.sum()),))
        ]
    assert not torch.equal(rewired, ei), "rewiring must actually change the graph"

    def training_loss(edge_index):
        view = make_training_view(graph.x, graph.y, edge_index, split, "inductive")
        torch.manual_seed(0)
        net = GCN(graph.x.size(1), 8, graph.num_classes, dropout=0.0)
        out = net(view.x, view.edge_index)
        return torch.nn.functional.cross_entropy(
            out[view.train_mask], view.y[view.train_mask]
        ).item()

    assert training_loss(ei) == training_loss(rewired)


def test_unknown_regime_raises(graph):
    with pytest.raises(ValueError, match="unknown regime"):
        make_training_view(graph.x, graph.y, graph.edge_index, graph.splits[0], "semi")


def _unlabelled_split(graph, num_spare=30):
    """The synthetic fixture partitions every node, so carve out a spare pool for the control."""
    split = graph.splits[0]
    test_ids = torch.nonzero(split.test_mask).flatten()
    val_ids = torch.nonzero(split.val_mask).flatten()
    test = torch.zeros_like(split.test_mask)
    test[test_ids[: len(test_ids) // 2]] = True
    val = torch.zeros_like(split.val_mask)
    val[val_ids[:num_spare]] = True
    return Split(split.train_mask.clone(), val, test)


def test_density_control_removes_as_many_nodes_as_the_inductive_view(graph):
    """The control must be the same size as the inductive view, or it is not controlling for
    size."""
    split = _unlabelled_split(graph)
    ind = make_training_view(graph.x, graph.y, graph.edge_index, split, "inductive")
    ctl = make_training_view(graph.x, graph.y, graph.edge_index, split, "density_control")
    assert ctl.x.size(0) == ind.x.size(0)


def test_density_control_keeps_every_training_and_validation_node(graph):
    """It may only drop nodes from the unlabelled pool. Dropping a train node would remove
    supervision and confound the control with less training data."""
    split = _unlabelled_split(graph)
    ctl = make_training_view(graph.x, graph.y, graph.edge_index, split, "density_control")
    assert int(ctl.train_mask.sum()) == int(split.train_mask.sum())
    assert int(ctl.val_mask.sum()) == int(split.val_mask.sum())


def test_density_control_keeps_the_test_nodes_in_the_training_graph(graph):
    """This is what makes it a control rather than a second inductive split: the test nodes
    are still there, so any accuracy it loses is a pure density effect."""
    split = _unlabelled_split(graph)
    ind = make_training_view(graph.x, graph.y, graph.edge_index, split, "inductive")
    ctl = make_training_view(graph.x, graph.y, graph.edge_index, split, "density_control")
    n_test = int(split.test_mask.sum())
    # the inductive view lost every test node; the control lost none of them
    assert ind.x.size(0) == graph.num_nodes - n_test
    test_rows = graph.x[split.test_mask]
    present = sum(
        1 for r in test_rows if bool((ctl.x == r).all(dim=1).any())
    )
    assert present == n_test


def test_density_control_raises_when_the_split_partitions_the_whole_graph(graph):
    """chameleon and squirrel are exactly this case, so the failure must be explicit rather
    than a silently degenerate control."""
    with pytest.raises(ValueError, match="partitions the whole graph"):
        make_training_view(
            graph.x, graph.y, graph.edge_index, graph.splits[0], "density_control"
        )


def test_random_split_is_disjoint_and_class_balanced():
    y = torch.arange(300) % 4
    split = random_split(300, y, seed=1, train_per_class=10, num_val=40, num_test=60)
    assert int(split.train_mask.sum()) == 40
    assert int(split.val_mask.sum()) == 40
    assert int(split.test_mask.sum()) == 60
    for c in range(4):
        assert int((y[split.train_mask] == c).sum()) == 10


def test_random_split_is_deterministic_given_a_seed():
    y = torch.arange(300) % 4
    kw = dict(train_per_class=10, num_val=40, num_test=60)
    a = random_split(300, y, seed=3, **kw)
    b = random_split(300, y, seed=3, **kw)
    c = random_split(300, y, seed=4, **kw)
    assert torch.equal(a.test_mask, b.test_mask)
    assert not torch.equal(a.test_mask, c.test_mask)


def test_random_split_refuses_to_silently_produce_an_empty_test_set():
    """Found while writing these tests: with the Planetoid defaults on a 300-node graph the
    test slice came out empty and every downstream accuracy became NaN, reproducibly. An
    empty test set must be an error, not a quiet result."""
    y = torch.arange(300) % 4
    with pytest.raises(ValueError, match="cannot supply"):
        random_split(300, y, seed=0)  # defaults ask for 500 val + 1000 test
