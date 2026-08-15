"""The four models. Two GNNs and two baselines that exist to bound what the GNN numbers mean.

MLP (features, no graph) and label propagation (graph, no features) are not filler. A GCN
scoring 81% on Cora means one thing if an MLP scores 55% and another thing entirely if it
scores 74%. Both baselines are also *controls* for the harness itself: neither can have a
non-zero transductive/inductive gap, for reasons given in their docstrings, so if the audit
ever reports one, the audit is broken.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import GCNConv, LabelPropagation, SAGEConv


class GCN(nn.Module):
    uses_graph = True

    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class GraphSAGE(nn.Module):
    uses_graph = True

    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class MLP(nn.Module):
    """Features only. `edge_index` is accepted and ignored so the harness can treat it
    uniformly with the GNNs.

    Control property: the MLP never reads `edge_index`, and the inductive view keeps every
    training node's features untouched, so its transductive and inductive runs are the same
    computation. Its measured inflation must be exactly 0.0.

    `uses_graph = False` is load-bearing, not decoration. It tells the harness to run this
    model on the masked rows only, so that its dropout draw does not depend on how many other
    nodes happen to be in the view. Getting that wrong cost us a real false positive -- see
    "Finding I1" in the README.
    """

    uses_graph = False

    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.5):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden)
        self.lin2 = nn.Linear(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor | None = None) -> Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x)


class LabelProp:
    """Label propagation: graph only, no features, no parameters, no training.

    Control property: it has nothing to fit, and at inference both regimes propagate the same
    training labels over the same full graph. Its measured inflation must be exactly 0.0.
    Its accuracy is the floor a GNN has to clear to be doing more than smearing train labels
    across edges.
    """

    def __init__(self, num_layers: int = 3, alpha: float = 0.9):
        self.prop = LabelPropagation(num_layers=num_layers, alpha=alpha)

    def predict(self, y: Tensor, edge_index: Tensor, train_mask: Tensor) -> Tensor:
        return self.prop(y, edge_index, train_mask).argmax(dim=1)


def build(name: str, in_dim: int, out_dim: int, hidden: int = 64) -> nn.Module:
    if name == "GCN":
        return GCN(in_dim, hidden, out_dim)
    if name == "GraphSAGE":
        return GraphSAGE(in_dim, hidden, out_dim)
    if name == "MLP":
        return MLP(in_dim, hidden, out_dim)
    raise ValueError(f"unknown model {name!r}")


TRAINED_MODELS = ("GCN", "GraphSAGE", "MLP")
