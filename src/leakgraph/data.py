"""Dataset loading, plus the synthetic graph that lets CI run without any download.

Real datasets come from torch_geometric's loaders. The synthetic graph is a planted-partition
graph with a controllable number of injected duplicate nodes -- it is what the tests use, and
it is the only thing CI ever touches.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .splits import Split

PLANETOID = ("Cora", "CiteSeer", "PubMed")
WIKIPEDIA = ("chameleon", "squirrel")
DATASETS = PLANETOID + WIKIPEDIA


@dataclass
class GraphData:
    name: str
    x: Tensor
    y: Tensor
    edge_index: Tensor
    splits: list[Split]  # one per provided split; index by seed

    @property
    def num_nodes(self) -> int:
        return int(self.x.size(0))

    @property
    def num_classes(self) -> int:
        return int(self.y.max()) + 1


def load(name: str, root: str = "data") -> GraphData:
    """Load a benchmark. Planetoid ships one public split; the Wikipedia ones ship ten."""
    if name in PLANETOID:
        from torch_geometric.datasets import Planetoid

        data = Planetoid(root=root, name=name)[0]
        splits = [Split(data.train_mask, data.val_mask, data.test_mask)]
    elif name in WIKIPEDIA:
        from torch_geometric.datasets import WikipediaNetwork

        # geom_gcn_preprocess=True is the variant the heterophily literature reports on, and
        # the one Platonov et al. (arXiv:2302.11640) found the duplicate nodes in.
        data = WikipediaNetwork(root=root, name=name, geom_gcn_preprocess=True)[0]
        splits = [
            Split(data.train_mask[:, i], data.val_mask[:, i], data.test_mask[:, i])
            for i in range(data.train_mask.size(1))
        ]
    else:
        raise ValueError(f"unknown dataset {name!r}")

    return GraphData(name, data.x, data.y, data.edge_index, splits)


def synthetic_graph(
    num_nodes: int = 120,
    num_classes: int = 3,
    num_features: int = 16,
    p_in: float = 0.25,
    p_out: float = 0.02,
    num_duplicates: int = 10,
    seed: int = 0,
) -> GraphData:
    """A planted-partition graph with `num_duplicates` exact feature-duplicate node pairs.

    Small, deterministic, and instant, so the tests can assert on known ground truth: exactly
    `num_duplicates` duplicate pairs exist, and edges are homophilous by construction.
    """
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(num_nodes) % num_classes

    centres = torch.randn(num_classes, num_features, generator=g)
    x = centres[y] + 0.3 * torch.randn(num_nodes, num_features, generator=g)

    # inject exact duplicates: node i copies node i + num_duplicates, same class
    for i in range(num_duplicates):
        src = i * num_classes
        dst = src + num_classes  # same class, since classes cycle with period num_classes
        if dst < num_nodes:
            x[dst] = x[src]

    same = y.unsqueeze(0) == y.unsqueeze(1)
    probs = torch.where(same, torch.tensor(p_in), torch.tensor(p_out))
    adj = torch.rand(num_nodes, num_nodes, generator=g) < probs
    adj = torch.triu(adj, diagonal=1)
    src, dst = torch.nonzero(adj, as_tuple=True)
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])

    perm = torch.randperm(num_nodes, generator=g)
    train = torch.zeros(num_nodes, dtype=torch.bool)
    val = torch.zeros(num_nodes, dtype=torch.bool)
    test = torch.zeros(num_nodes, dtype=torch.bool)
    n_train = int(0.5 * num_nodes)
    n_val = int(0.2 * num_nodes)
    train[perm[:n_train]] = True
    val[perm[n_train : n_train + n_val]] = True
    test[perm[n_train + n_val :]] = True

    return GraphData("synthetic", x, y, edge_index, [Split(train, val, test)])
