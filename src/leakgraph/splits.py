"""Train/val/test splits and the transductive vs inductive training views.

The single most important thing in this repository is `induced_subgraph`. An inductive
split is only inductive if the test nodes are *physically absent* from the graph the model
trains on. A common shortcut is to keep the full node feature matrix and merely drop the
edges that touch test nodes. That is *probably* fine for a plain GCN -- isolated rows do not
influence anyone else's representation -- but it is not provable, it breaks the moment a
model uses any node-set-level statistic (BatchNorm over all nodes, a global readout,
degree normalisation over the full index), and it cannot be tested by construction.

So we relabel instead. `induced_subgraph` returns a graph with `keep_mask.sum()` nodes and
no trace of the removed ones. `tests/test_splits.py` then asserts the property that matters:
overwriting the hidden nodes' features with garbage does not change the inductive training
loss by even one bit, while it does change the transductive one.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class Split:
    """A node-level split. All three masks are boolean, length = number of nodes."""

    train_mask: Tensor
    val_mask: Tensor
    test_mask: Tensor

    def __post_init__(self) -> None:
        n = self.train_mask.numel()
        if self.val_mask.numel() != n or self.test_mask.numel() != n:
            raise ValueError("masks must all have the same length")
        for name in ("train_mask", "val_mask", "test_mask"):
            if getattr(self, name).dtype != torch.bool:
                raise ValueError(f"{name} must be a bool tensor")
        overlap = (
            (self.train_mask & self.val_mask)
            | (self.train_mask & self.test_mask)
            | (self.val_mask & self.test_mask)
        )
        if bool(overlap.any()):
            raise ValueError("train/val/test masks overlap")

    @property
    def num_nodes(self) -> int:
        return int(self.train_mask.numel())


@dataclass(frozen=True)
class TrainingView:
    """The graph a model is allowed to see during training, plus its masks.

    For the transductive regime this is the whole graph. For the inductive regime it is the
    subgraph induced on the non-test nodes, with node ids relabelled to 0..k-1.
    """

    x: Tensor
    y: Tensor
    edge_index: Tensor
    train_mask: Tensor
    val_mask: Tensor
    regime: str


def induced_subgraph(
    x: Tensor, y: Tensor, edge_index: Tensor, keep_mask: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Physically drop every node outside `keep_mask` and relabel the survivors to 0..k-1.

    Returns (x_sub, y_sub, edge_index_sub, old_to_new) where `old_to_new[i]` is the new index
    of old node i, or -1 if node i was dropped. Only edges with *both* endpoints kept survive.
    """
    if keep_mask.dtype != torch.bool:
        raise ValueError("keep_mask must be a bool tensor")
    if keep_mask.numel() != x.size(0):
        raise ValueError("keep_mask length must equal the number of nodes")

    old_to_new = torch.full((x.size(0),), -1, dtype=torch.long, device=x.device)
    old_to_new[keep_mask] = torch.arange(int(keep_mask.sum()), device=x.device)

    both_kept = keep_mask[edge_index[0]] & keep_mask[edge_index[1]]
    edge_index_sub = old_to_new[edge_index[:, both_kept]]

    return x[keep_mask], y[keep_mask], edge_index_sub, old_to_new


def bisect_test_split(split: Split, seed: int = 0) -> tuple[Split, Tensor]:
    """Halve the test set and hand the other half back as a spare pool.

    The density control needs unlabelled nodes it can remove, and the geom-gcn splits of
    chameleon and squirrel assign every node to train, val or test, so no such pool exists.
    Reserving half of the test set manufactures one. A reserved node keeps its features and
    its edges in the graph, its label is never used for anything, and it is never scored --
    which is precisely the status of an unlabelled Planetoid node. The cost is that the
    scored test set is half the size, so every measurement made on it is correspondingly
    noisier, and the number of nodes removed by the inductive arm halves too.

    Returns (split with the halved test set, boolean mask of the reserved pool).
    """
    test_ids = torch.nonzero(split.test_mask).flatten()
    perm = test_ids[torch.randperm(test_ids.numel(), generator=torch.Generator().manual_seed(seed))]
    half = perm.numel() // 2
    scored = torch.zeros_like(split.test_mask)
    scored[perm[:half]] = True
    reserved = torch.zeros_like(split.test_mask)
    reserved[perm[half:]] = True
    return Split(split.train_mask.clone(), split.val_mask.clone(), scored), reserved


def make_training_view(
    x: Tensor,
    y: Tensor,
    edge_index: Tensor,
    split: Split,
    regime: str,
    seed: int = 0,
    pool_mask: Tensor | None = None,
) -> TrainingView:
    """Build the graph the model trains on.

    transductive: the full graph. Test node features and edges are present during training;
        only their labels are withheld. This is what the standard benchmarks do.
    inductive: the subgraph induced on train+val nodes. Test nodes do not exist during
        training. They are re-attached (with all their edges) only at inference time.
    density_control: the subgraph with an equal number of *unlabelled non-test* nodes removed
        at random instead of the test nodes.

    The density control exists because transductive minus inductive is confounded. The
    inductive graph is missing the test nodes' information, which is the effect we want, but
    it is also simply smaller and sparser, which hurts training on its own and has nothing to
    do with leakage. Removing the same number of nodes that the model was never going to be
    scored on separates the two: transductive minus density_control is the density cost, and
    density_control minus inductive is what is left over that is specific to hiding the test
    nodes themselves.

    It needs spare nodes to remove, so it only works on splits that leave part of the graph
    unlabelled. The Planetoid public splits do. The geom-gcn splits of chameleon and squirrel
    partition every node into train/val/test, so there is no pool to draw from and this raises
    -- unless `pool_mask` names one explicitly, which is how `bisect_test_split` recovers the
    control on those two datasets. Passing `pool_mask` also lets the Planetoid runs use the
    identical pool construction, so the two protocols can be compared rather than assumed
    equivalent.
    """
    if regime == "transductive":
        return TrainingView(x, y, edge_index, split.train_mask, split.val_mask, regime)

    if regime == "inductive":
        keep = ~split.test_mask
    elif regime == "density_control":
        pool = (
            ~(split.train_mask | split.val_mask | split.test_mask)
            if pool_mask is None
            else pool_mask
        )
        if bool((pool & (split.train_mask | split.val_mask | split.test_mask)).any()):
            raise ValueError("pool_mask overlaps train/val/test; the control would drop labels")
        n_remove = int(split.test_mask.sum())
        pool_idx = torch.nonzero(pool).flatten()
        if pool_idx.numel() < n_remove:
            raise ValueError(
                f"density_control needs {n_remove} spare unlabelled nodes but the split "
                f"leaves only {pool_idx.numel()}; this split partitions the whole graph"
            )
        g = torch.Generator().manual_seed(seed)
        drop = pool_idx[torch.randperm(pool_idx.numel(), generator=g)[:n_remove]]
        keep = torch.ones_like(split.test_mask)
        keep[drop] = False
    else:
        raise ValueError(f"unknown regime {regime!r}")

    x_s, y_s, ei_s, _ = induced_subgraph(x, y, edge_index, keep)
    return TrainingView(
        x_s, y_s, ei_s, split.train_mask[keep], split.val_mask[keep], regime
    )


def random_split(
    num_nodes: int,
    y: Tensor,
    seed: int,
    train_per_class: int = 20,
    num_val: int = 500,
    num_test: int = 1000,
) -> Split:
    """Planetoid-style random split: `train_per_class` per class, then val/test from the rest."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_nodes, generator=g)

    train = torch.zeros(num_nodes, dtype=torch.bool)
    for c in y.unique():
        cls_nodes = perm[y[perm] == c][:train_per_class]
        train[cls_nodes] = True

    rest = perm[~train[perm]]
    if rest.numel() < num_val + num_test:
        # Without this the slicing below silently yields an empty test set, which then reads
        # as a perfectly reproducible accuracy of NaN. Fail loudly instead.
        raise ValueError(
            f"{num_nodes} nodes cannot supply {int(train.sum())} train + {num_val} val + "
            f"{num_test} test; only {rest.numel()} remain after the train draw"
        )
    val = torch.zeros(num_nodes, dtype=torch.bool)
    test = torch.zeros(num_nodes, dtype=torch.bool)
    val[rest[:num_val]] = True
    test[rest[num_val : num_val + num_test]] = True
    return Split(train, val, test)
