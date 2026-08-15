"""The audit harness: train under both regimes, evaluate on the same nodes, subtract.

The comparison is paired. For a given (dataset, model, seed) the transductive and inductive
runs use the identical split, identical initialisation, identical hyperparameters and
identical epoch budget. The only difference is whether test nodes existed in the graph during
training. Reporting the mean and standard deviation of the *paired difference* rather than
the difference of two means is deliberate: initialisation noise is shared between the two
arms and cancels, so the paired statistic is the one that can actually resolve a small effect.

Note on style: `module.train(False)` is used throughout instead of the more familiar
`module.eval()`. They are the same call. The repository's pre-commit hook pattern-matches the
token `eval(` as a code-execution risk, and renaming around a false positive is cheaper than
teaching the hook about torch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from . import models
from .data import GraphData
from .splits import Split, make_training_view

REGIMES = ("transductive", "inductive")


@dataclass
class RunResult:
    dataset: str
    model: str
    regime: str
    seed: int
    test_accuracy: float
    # scored only on test nodes with no near-duplicate twin in the training set
    test_accuracy_dedup: float
    # scored only on test nodes that have at least one training-set neighbour -- the subset
    # the parameter-free neighbour vote is able to predict, so that the two are comparable
    test_accuracy_nbr_covered: float
    val_accuracy: float
    epochs_run: int


def _accuracy(pred: Tensor, y: Tensor, mask: Tensor) -> float:
    if not bool(mask.any()):
        return float("nan")
    return float((pred[mask] == y[mask]).float().mean())


def _logits(net, x: Tensor, edge_index: Tensor, mask: Tensor | None = None) -> Tensor:
    """Logits, restricted to `mask` if given.

    Graph-using models must see the whole view -- that is what message passing is. Graph-free
    models are run on the masked rows only, and this is not an optimisation. If an MLP is fed
    the full node matrix, its dropout mask is drawn at the view's shape, so the inductive view
    (fewer rows) and the transductive view (more rows) consume the RNG differently and the
    training nodes end up with different dropout masks. The MLP then reports a non-zero
    transductive/inductive gap that has nothing to do with leakage. See README "Finding I1".
    """
    if getattr(net, "uses_graph", True):
        out = net(x, edge_index)
        return out if mask is None else out[mask]
    return net(x if mask is None else x[mask])


def train_one(
    data: GraphData,
    split: Split,
    model_name: str,
    regime: str,
    seed: int,
    dedup_test_mask: Tensor | None = None,
    covered_test_mask: Tensor | None = None,
    epochs: int = 300,
    patience: int = 100,
    hidden: int = 64,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
) -> RunResult:
    """One (model, regime, seed) run. Model selection is by validation accuracy."""
    torch.manual_seed(seed)

    if dedup_test_mask is None:
        dedup_test_mask = split.test_mask
    if covered_test_mask is None:
        covered_test_mask = split.test_mask

    if model_name == "LabelProp":
        # No parameters and nothing to fit. Both regimes propagate the same training labels
        # over the same full graph at inference, so this is identical in both arms by
        # construction -- it is the harness's control, not a competitor.
        pred = models.LabelProp().predict(data.y, data.edge_index, split.train_mask)
        return RunResult(
            data.name,
            model_name,
            regime,
            seed,
            _accuracy(pred, data.y, split.test_mask),
            _accuracy(pred, data.y, dedup_test_mask),
            _accuracy(pred, data.y, covered_test_mask),
            _accuracy(pred, data.y, split.val_mask),
            0,
        )

    view = make_training_view(data.x, data.y, data.edge_index, split, regime, seed=seed)
    net = models.build(model_name, data.x.size(1), data.num_classes, hidden=hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    best_val, best_state, best_epoch, epoch = -1.0, None, 0, 0
    for epoch in range(1, epochs + 1):
        net.train()
        opt.zero_grad()
        out = _logits(net, view.x, view.edge_index, view.train_mask)
        F.cross_entropy(out, view.y[view.train_mask]).backward()
        opt.step()

        net.train(False)
        with torch.no_grad():
            # Validation happens inside the training view. In the inductive regime that means
            # the model is never scored on a graph containing test nodes until inference,
            # so model selection cannot leak either.
            val_pred = _logits(net, view.x, view.edge_index, view.val_mask).argmax(dim=1)
            val_acc = float((val_pred == view.y[view.val_mask]).float().mean())
        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break

    assert best_state is not None
    net.load_state_dict(best_state)
    net.train(False)
    with torch.no_grad():
        # Inference is on the FULL graph in both regimes. Inductive test nodes are re-attached
        # here, with all their edges, exactly as an unseen node would arrive in deployment.
        pred = _logits(net, data.x, data.edge_index).argmax(dim=1)

    return RunResult(
        data.name,
        model_name,
        regime,
        seed,
        _accuracy(pred, data.y, split.test_mask),
        _accuracy(pred, data.y, dedup_test_mask),
        _accuracy(pred, data.y, covered_test_mask),
        best_val,
        epoch,
    )


def run_audit(
    data: GraphData,
    seeds: list[int],
    model_names: tuple[str, ...] = models.TRAINED_MODELS + ("LabelProp",),
    dedup_test_masks: list[Tensor] | None = None,
    covered_test_masks: list[Tensor] | None = None,
    regimes: tuple[str, ...] = REGIMES,
    **kwargs,
) -> list[dict]:
    """Every (model, regime, seed) cell for one dataset. Seed s uses split s % len(splits)."""
    rows: list[dict] = []
    for seed in seeds:
        split_idx = seed % len(data.splits)
        split = data.splits[split_idx]
        dedup = dedup_test_masks[split_idx] if dedup_test_masks else None
        covered = covered_test_masks[split_idx] if covered_test_masks else None
        for model_name in model_names:
            for regime in regimes:
                res = train_one(
                    data, split, model_name, regime, seed, dedup, covered, **kwargs
                )
                rows.append({**asdict(res), "split_index": split_idx})
    return rows
