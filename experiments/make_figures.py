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
    width = 0.26
    base = np.arange(len(datasets))
    for offset, model in enumerate(MODELS):
        rows = table[table.model == model].set_index("dataset").loc[datasets]
        ax.bar(
            base + (offset - 1.0) * width,
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


def decomposition(out: Path) -> Path:
    """Split inflation into what graph density buys and what the test nodes buy.

    Adding test nodes to the graph does two things at once: it makes the graph
    denser, which helps any node, and it exposes the test nodes' own
    neighbourhoods. Only the second is leakage in the sense people usually mean,
    and the density control separates them by adding an equal number of *non-test*
    nodes instead.

    The honest result is that the split mostly does not resolve. The CSV carries a
    ``*_resolved`` flag per component -- whether the effect clears its own noise --
    and half the components fail it, with the test-specific term coming out
    negative on three of six. Bars that did not resolve are drawn hollow so the
    figure cannot be read as a clean attribution.
    """
    table = pd.read_csv(REPORTS / "density_control.csv")
    table = table[table.model.isin(["GCN", "GraphSAGE"])]
    labels = [f"{r.dataset}\n{r.model}" for _, r in table.iterrows()]
    base = np.arange(len(table))

    figure, ax = plt.subplots(figsize=(11.5, 4.8))
    for offset, (mean, std, flag, colour, label) in enumerate([
        ("density_cost_mean", "density_cost_std", "density_resolved", "#bdbdbd",
         "explained by graph density"),
        ("test_specific_mean", "test_specific_std", "test_specific_resolved", "#2166ac",
         "specific to the test nodes"),
    ]):
        for index, (_, row) in enumerate(table.iterrows()):
            resolved = bool(row[flag])
            ax.bar(
                index + (offset - 0.5) * 0.4,
                row[mean] * 100, 0.4,
                yerr=row[std] * 100, capsize=2.5,
                color=colour if resolved else "none",
                edgecolor=colour if not resolved else "0.3",
                hatch="" if resolved else "///",
                lw=1.2 if not resolved else 0.5,
                label=label if index == 0 else None,
            )
    ax.axhline(0, color="0.2", lw=1.0)
    ax.set_xticks(base)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("accuracy points")
    ax.set_title(
        "Hollow bars did not clear their own noise. Six of twelve components "
        "fail that test,\nand the test-specific term is negative in three "
        "cases -- the decomposition mostly does not resolve.",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def controls(out: Path) -> Path:
    """The same measurement under three controls, including a negative one.

    A random split has no temporal or structural reason to leak, so the harness
    reading anything other than zero there would be a fault in the harness.
    """
    # density_control only ran on the three homophilous datasets, so it is not
    # in this comparison -- it is the subject of the decomposition figure instead.
    frames = {
        "measured": "inflation.csv",
        "random split (negative control)": "random_split_control.csv",
        "bisected graph": "bisected_control.csv",
    }
    figure, ax = plt.subplots(figsize=(11, 4.6))
    width = 0.26
    datasets = None
    for offset, (label, filename) in enumerate(frames.items()):
        table = pd.read_csv(REPORTS / filename)
        table = table[table.model == "GCN"].set_index("dataset")
        if datasets is None:
            datasets = list(table.index)
        column = "inflation_mean" if "inflation_mean" in table else "total_inflation_mean"
        error = "inflation_stderr" if "inflation_stderr" in table else "total_inflation_std"
        rows = table.loc[datasets]
        ax.bar(np.arange(len(datasets)) + (offset - 1.0) * width,
               rows[column] * 100, width, yerr=rows[error] * 100, capsize=2,
               label=label, edgecolor="0.3", lw=0.4)
    ax.axhline(0, color="0.2", lw=1.0)
    ax.set_xticks(np.arange(len(datasets)))
    ax.set_xticklabels(datasets)
    ax.set_ylabel("leakage inflation (accuracy points)")
    ax.set_title("GCN under four conditions. The negative control sits on zero.",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def duplicate_definitions(out: Path) -> Path:
    """How much of the duplicate rate is the definition rather than the data.

    "Duplicate node" has no single meaning. Exact-match and cosine-0.99 agree on
    Cora at 1%; cosine-0.7 puts PubMed at 24% and identical-neighbour-set puts it
    at 45%. Any headline duplicate figure is a choice of threshold.
    """
    table = pd.read_csv(REPORTS / "duplicate_definitions.csv", index_col=0)
    # Two rows count *pairs* normalised by node count, so they exceed 1 whenever a
    # node sits in several pairs -- chameleon reads 12.0 there. That is a different
    # quantity from the node fractions in every other row, and putting both on one
    # colour scale would imply a 1197% duplicate rate. Node fractions only.
    table = table.drop(index=[i for i in table.index if "pairs_over_nodes" in i])

    figure, ax = plt.subplots(figsize=(11, 6.2))
    image = ax.imshow(table.values * 100, cmap="YlOrRd", aspect="auto",
                      vmin=0, vmax=50)
    ax.set_xticks(np.arange(table.shape[1]))
    ax.set_xticklabels(table.columns, rotation=20, ha="right")
    ax.set_yticks(np.arange(table.shape[0]))
    ax.set_yticklabels([i.replace("_", " ") for i in table.index], fontsize=8)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            value = table.values[i, j] * 100
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=7,
                    color="white" if value > 28 else "0.2")
    figure.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="% of nodes")
    ax.set_title(
        "The same graphs under eighteen definitions of \"duplicate\".\n"
        "PubMed is 0.04% duplicated or 45% duplicated depending which you pick.",
        fontsize=10,
    )
    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for path in (
        inflation(FIGURES / "leakage-inflation.png"),
        leakage_channels(FIGURES / "leakage-channels.png"),
        decomposition(FIGURES / "inflation-decomposition.png"),
        controls(FIGURES / "controls.png"),
        duplicate_definitions(FIGURES / "duplicate-definitions.png"),
    ):
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
