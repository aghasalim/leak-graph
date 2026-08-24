"""Draw the README figures from the saved reports.

Reads ``reports/`` only -- no training, no downloads.  The first figure is the
headline metric with its instrument controls visible; the second is the leakage
channel that explains the sign and size of the first.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

MODELS = ["GCN", "GraphSAGE", "LabelProp", "MLP"]
COLOURS = {
    "GCN": "#2166ac",
    "GraphSAGE": "#67a9cf",
    "LabelProp": "#bdbdbd",
    "MLP": "#969696",
}
HOMOPHILOUS = ["Cora", "CiteSeer", "PubMed"]


def inflation(out: Path) -> Path:
    """Leakage inflation per dataset and model, with the seed standard error.

    The two graph-free models are the control: they cannot see the difference
    between a transductive and an inductive split, so anything other than exactly
    zero for them would mean the harness itself is leaking.
    """
    table = pd.read_csv(REPORTS / "inflation.csv")
    datasets = HOMOPHILOUS + [d for d in table.dataset.unique() if d not in HOMOPHILOUS]

    figure, ax = plt.subplots(figsize=(10.5, 5.0))
    width = 0.2
    base = np.arange(len(datasets))
    for offset, model in enumerate(MODELS):
        rows = table[table.model == model].set_index("dataset").loc[datasets]
        ax.bar(
            base + (offset - 1.5) * width,
            rows.inflation_mean * 100,
            width,
            yerr=rows.inflation_stderr * 100,
            capsize=2.5,
            label=model,
            color=COLOURS[model],
            edgecolor="0.3",
            linewidth=0.5,
        )
    ax.axhline(0, color="0.2", lw=1.0)
    ax.axvline(len(HOMOPHILOUS) - 0.5, color="0.75", lw=1.0, ls="--")
    ax.text(
        len(HOMOPHILOUS) - 0.42, ax.get_ylim()[1] * 0.92,
        "heterophilous →", fontsize=9, color="0.4",
    )
    ax.set_xticks(base)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("leakage inflation (accuracy points)")
    ax.set_title(
        "Transductive minus inductive test accuracy, 10 seeds.\n"
        "LabelProp and MLP sit at exactly zero because neither can tell the "
        "splits apart — that is the instrument check.",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=9, ncol=4)
    ax.spines[["top", "right"]].set_visible(False)

    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def leakage_channels(out: Path) -> Path:
    """The exposure that transductive evaluation buys, per dataset.

    Neighbour-label leakage is the channel that matters: how many test nodes touch
    a labelled training node, and how far a bare vote over those labels gets you.
    """
    records = json.loads((REPORTS / "detectors.json").read_text())
    datasets = [r["dataset"] for r in records]
    base = np.arange(len(datasets))

    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4.4))

    left.bar(
        base - 0.2,
        [r["neighbour_label"]["frac_test_with_train_neighbour"] * 100 for r in records],
        0.4, label="has a labelled train neighbour", color="#2166ac",
    )
    left.bar(
        base + 0.2,
        [r["duplicates"]["frac_test_with_train_twin"] * 100 for r in records],
        0.4, label="has a near-duplicate in train", color="#b2182b",
    )
    left.set_xticks(base)
    left.set_xticklabels(datasets, rotation=20, ha="right")
    left.set_ylabel("% of test nodes")
    left.set_title("how exposed is the test set?", fontsize=10)
    left.legend(frameon=False, fontsize=8)
    left.spines[["top", "right"]].set_visible(False)

    vote = [r["neighbour_label"]["vote_accuracy_on_covered"] * 100 for r in records]
    majority = [r["neighbour_label"]["majority_class_accuracy"] * 100 for r in records]
    right.bar(base - 0.2, vote, 0.4, label="neighbour-label vote", color="#2166ac")
    right.bar(base + 0.2, majority, 0.4, label="majority class", color="#bdbdbd")
    right.set_xticks(base)
    right.set_xticklabels(datasets, rotation=20, ha="right")
    right.set_ylabel("accuracy on covered test nodes (%)")
    right.set_title(
        "how much does that exposure actually tell you?", fontsize=10
    )
    right.legend(frameon=False, fontsize=8)
    right.spines[["top", "right"]].set_visible(False)

    figure.suptitle(
        "Exposure is not leakage. Squirrel exposes twice as many test nodes as "
        "Cora,\nand the labels it exposes are worth almost nothing.",
        fontsize=10, y=0.02,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for path in (
        inflation(FIGURES / "leakage-inflation.png"),
        leakage_channels(FIGURES / "leakage-channels.png"),
    ):
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
