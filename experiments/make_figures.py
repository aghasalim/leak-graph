"""Draw the README figures from the saved reports.

Reads ``reports/`` only: no training, no downloads, no network. Every number in
every figure comes out of a committed CSV or JSON, so a figure cannot disagree
with the tables in the README.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image

from style import PALETTE, titled

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

MODELS = ["GCN", "GraphSAGE", "LabelProp", "MLP"]

# The two GNNs take palette colours. The two graph-free models are grey on
# purpose: they are the instrument check, not a result, and grey says so before
# the caption does.
COLOURS = {
    "GCN": PALETTE[0],
    "GraphSAGE": PALETTE[3],
    "LabelProp": "#c9c9c9",
    "MLP": "#9a9a9a",
}
NEIGHBOUR, DUPLICATE, BASELINE = PALETTE[0], PALETTE[1], "#bdbdbd"

HOMOPHILOUS = ["Cora", "CiteSeer", "PubMed"]
HETEROPHILOUS = ["chameleon", "squirrel"]
ORDER = HOMOPHILOUS + HETEROPHILOUS


def _outside(ax, **kwargs):
    """Legend parked to the right of the axes, where it cannot cover a bar."""
    ax.legend(loc="center left", bbox_to_anchor=(1.015, 0.5), **kwargs)


def inflation(out: Path) -> Path:
    """Leakage inflation per dataset and model, with the seed standard error.

    The two graph-free models are the control: they cannot see the difference
    between a transductive and an inductive split, so anything other than exactly
    zero for them would mean the harness itself is leaking.
    """
    table = pd.read_csv(REPORTS / "inflation.csv")
    datasets = [d for d in ORDER if d in set(table.dataset)]
    gnn = table[table.model.isin(["GCN", "GraphSAGE"])]
    best = gnn.inflation_mean.max() * 100
    worst = gnn.loc[gnn.inflation_mean.idxmin()]
    # Only claim a sign flip if one is actually in the file.
    claim = (f"The leak tops out at {best:.1f} accuracy points, and on "
             f"{worst.dataset} it reverses") if worst.inflation_mean < 0 else (
             f"The leak tops out at {best:.1f} accuracy points and never reverses")

    figure, ax = plt.subplots(figsize=(10.5, 5.0))
    width = 0.21
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
            edgecolor="0.35",
            linewidth=0.5,
        )
    ax.axhline(0, color="0.2", lw=1.0)
    if set(HETEROPHILOUS) & set(datasets):
        ax.axvline(len(HOMOPHILOUS) - 0.5, color="0.75", lw=1.0, ls="--")
        ax.text(len(HOMOPHILOUS) - 0.42, ax.get_ylim()[1] * 0.78,
                "heterophilous ->", fontsize=9, color="#777777")
    ax.set_xticks(base)
    ax.set_xticklabels(datasets)
    ax.set_xlabel("benchmark graph")
    ax.set_ylabel("leakage inflation (accuracy points)")
    titled(
        ax,
        claim,
        "transductive minus inductive test accuracy, differenced per seed over 10 "
        "seeds, whiskers are one standard error",
    )
    _outside(ax, title="model")

    figure.savefig(out)
    plt.close(figure)
    return out


def leakage_channels(out: Path) -> Path:
    """The exposure that transductive evaluation buys, per dataset.

    Neighbour-label leakage is the channel that matters: how many test nodes touch
    a labelled training node, and how far a bare vote over those labels gets you.
    """
    records = json.loads((REPORTS / "detectors.json").read_text())
    records = sorted(records, key=lambda r: ORDER.index(r["dataset"]))
    datasets = [r["dataset"] for r in records]
    base = np.arange(len(datasets))

    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 4.8))

    left.bar(
        base - 0.2,
        [r["neighbour_label"]["frac_test_with_train_neighbour"] * 100 for r in records],
        0.4, label="has a labelled training neighbour", color=NEIGHBOUR,
    )
    left.bar(
        base + 0.2,
        [r["duplicates"]["frac_test_with_train_twin"] * 100 for r in records],
        0.4, label="has a near-duplicate in training", color=DUPLICATE,
    )
    left.set_xticks(base)
    left.set_xticklabels(datasets)
    left.set_xlabel("benchmark graph")
    left.set_ylabel("test nodes exposed (% of the test set)")
    titled(left, "The heterophilous graphs expose twice as many test nodes",
           "two channels a transductive split opens, on split 0")
    left.legend(loc="upper left")

    vote = [r["neighbour_label"]["vote_accuracy_on_covered"] * 100 for r in records]
    majority = [r["neighbour_label"]["majority_class_accuracy"] * 100 for r in records]
    right.bar(base - 0.2, vote, 0.4, label="vote over neighbour labels", color=NEIGHBOUR)
    right.bar(base + 0.2, majority, 0.4, label="majority class", color=BASELINE)
    right.set_xticks(base)
    right.set_xticklabels(datasets)
    right.set_xlabel("benchmark graph")
    right.set_ylabel("accuracy on covered test nodes (%)")
    titled(right, "And there those labels are worth almost nothing",
           "a bare majority vote over those neighbours, scored only where it can predict")
    right.legend(loc="upper right")

    figure.tight_layout()
    figure.savefig(out)
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
    ``*_resolved`` flag per component, whether the effect clears its own noise, and
    half the components fail it. Bars that did not resolve are drawn hollow so the
    figure cannot be read as a clean attribution.
    """
    table = pd.read_csv(REPORTS / "density_control.csv")
    table = table[table.model.isin(["GCN", "GraphSAGE"])]
    labels = [f"{r.dataset}\n{r.model}" for _, r in table.iterrows()]
    base = np.arange(len(table))

    components = [
        ("density_cost_mean", "density_cost_std", "density_resolved", "#9a9a9a",
         "explained by graph density"),
        ("test_specific_mean", "test_specific_std", "test_specific_resolved", PALETTE[0],
         "specific to the test nodes"),
    ]
    total = len(table) * len(components)
    hollow = sum(int(not row[flag]) for _, _, flag, _, _ in components
                 for _, row in table.iterrows())
    negative = int((table.test_specific_mean < 0).sum())

    figure, ax = plt.subplots(figsize=(11.5, 5.0))
    for offset, (mean, std, flag, colour, label) in enumerate(components):
        for index, (_, row) in enumerate(table.iterrows()):
            resolved = bool(row[flag])
            ax.bar(
                index + (offset - 0.5) * 0.4,
                row[mean] * 100, 0.4,
                yerr=row[std] * 100, capsize=2.5,
                color=colour if resolved else "none",
                edgecolor=colour if not resolved else "0.35",
                hatch="" if resolved else "///",
                lw=1.2 if not resolved else 0.5,
                label=label if index == 0 else None,
            )
    ax.axhline(0, color="0.2", lw=1.0)
    ax.set_xticks(base)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("accuracy points")
    titled(
        ax,
        f"{hollow} of the {total} components never clear their own noise",
        f"hollow bars failed the two standard error test, and the test-specific "
        f"term is negative in {negative} of {len(table)} cases",
    )
    _outside(ax)
    figure.savefig(out)
    plt.close(figure)
    return out


def controls(out: Path) -> Path:
    """The same measurement under three conditions, including a negative control.

    A random split has no temporal or structural reason to leak, so the harness
    reading anything other than zero there would be a fault in the harness.
    """
    # density_control only ran on the three homophilous datasets, so it is not
    # in this comparison; it is the subject of the decomposition figure instead.
    frames = {
        "measured": ("inflation.csv", PALETTE[0]),
        "random split (negative control)": ("random_split_control.csv", PALETTE[3]),
        "bisected graph": ("bisected_control.csv", PALETTE[2]),
    }
    figure, ax = plt.subplots(figsize=(11, 4.8))
    width = 0.26
    datasets = None
    random_mean = None
    for offset, (label, (filename, colour)) in enumerate(frames.items()):
        table = pd.read_csv(REPORTS / filename)
        table = table[table.model == "GCN"].set_index("dataset")
        if datasets is None:
            datasets = [d for d in ORDER if d in table.index]
        column = "inflation_mean" if "inflation_mean" in table else "total_inflation_mean"
        error = "inflation_stderr" if "inflation_stderr" in table else "total_inflation_std"
        rows = table.loc[datasets]
        if "random" in label:
            random_mean = rows[column].abs().max() * 100
        ax.bar(np.arange(len(datasets)) + (offset - 1.0) * width,
               rows[column] * 100, width, yerr=rows[error] * 100, capsize=2,
               label=label, color=colour, edgecolor="0.35", lw=0.4)
    ax.axhline(0, color="0.2", lw=1.0)
    ax.set_xticks(np.arange(len(datasets)))
    ax.set_xticklabels(datasets)
    ax.set_xlabel("benchmark graph")
    ax.set_ylabel("leakage inflation (accuracy points)")
    titled(
        ax,
        "The negative control reads zero, so the harness is not the leak",
        f"GCN under three conditions; the random split never moves more than "
        f"{random_mean:.1f} accuracy points from zero",
    )
    _outside(ax, title="condition")
    figure.savefig(out)
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
    # node sits in several pairs, and chameleon reads 12.0 there. That is a
    # different quantity from the node fractions in every other row, and putting
    # both on one colour scale would imply a 1197% duplicate rate. Node fractions
    # only.
    table = table.drop(index=[i for i in table.index if "pairs_over_nodes" in i])
    table = table[[c for c in ORDER if c in table.columns]]
    values = table.values * 100
    top = float(values.max())
    column = list(table.columns).index("PubMed")
    rows = list(table.index)
    strictest = values[rows.index("exact_nodes_in_a_duplicate_group"), column]
    loosest_row = int(values[:, column].argmax())

    figure, ax = plt.subplots(figsize=(11, 6.4))
    image = ax.imshow(values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=top)
    ax.set_xticks(np.arange(table.shape[1]))
    ax.set_xticklabels(table.columns)
    ax.set_xlabel("benchmark graph")
    ax.set_yticks(np.arange(table.shape[0]))
    ax.set_yticklabels([i.replace("_", " ") for i in table.index], fontsize=8.5)
    ax.grid(False)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.1f}", ha="center", va="center", fontsize=7.5,
                    color="white" if values[i, j] > 0.62 * top else "0.15")
    bar = figure.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    bar.set_label("% of nodes")
    titled(
        ax,
        f"PubMed is {strictest:.2f}% duplicated by exact match and "
        f"{values[loosest_row, column]:.1f}% by "
        f"{rows[loosest_row].replace('_', ' ')}",
        f"the same {table.shape[1]} graphs read under {table.shape[0]} "
        f"definitions of \"duplicate node\"",
    )
    figure.savefig(out)
    plt.close(figure)
    return out


def _paired(runs: pd.DataFrame) -> pd.DataFrame:
    """Per-seed transductive minus inductive accuracy, the way the CSV builds it."""
    wide = runs.pivot_table(index=["dataset", "model", "seed"], columns="regime",
                            values="test_accuracy")
    return (wide["transductive"] - wide["inductive"]).rename("paired")


def seed_accumulation(out: Path, fps: int = 14) -> Path:
    """Watch the headline effect fight its own noise as seeds are added.

    runs.csv holds the per-seed accuracies behind inflation.csv, so the running
    paired mean at ten seeds lands exactly on the committed inflation number. The
    two standard error band is the same test the ``resolved?`` column applies, and
    the animation is the honest reason most of that column says no. Only the three
    citation graphs have per-seed rows committed, so only they are drawn.
    """
    runs = pd.read_csv(REPORTS / "runs.csv")
    paired = _paired(runs)
    datasets = [d for d in HOMOPHILOUS if d in runs.dataset.unique()]
    models = ["GCN", "GraphSAGE"]
    series = {(d, m): paired.loc[(d, m)].sort_index().values * 100
              for d in datasets for m in models}
    seeds = min(len(v) for v in series.values())
    counts = np.arange(2, seeds + 1)

    means = {k: np.array([v[:n].mean() for n in counts]) for k, v in series.items()}
    errors = {k: np.array([v[:n].std(ddof=1) / np.sqrt(n) for n in counts])
              for k, v in series.items()}

    figure, axes = plt.subplots(1, len(datasets), figsize=(11.5, 4.1), sharey=True)
    # Scaled from three seeds on. Two seeds give a band so wide it runs off the
    # top of every panel, which is the point being made rather than something to
    # fit in.
    span = max(abs(means[k][i]) + 2 * errors[k][i]
               for k in means for i in range(1, len(counts)))
    art = {}
    for ax, dataset in zip(axes, datasets):
        ax.set_xlim(counts[0] - 0.4, counts[-1] + 0.4)
        ax.set_ylim(-span * 1.2, span * 1.35)
        ax.axhline(0, color="0.2", lw=1.0)
        ax.set_xticks(counts[::2])
        ax.set_xlabel("seeds averaged (count)")
        ax.text(0.03, 0.965, dataset, transform=ax.transAxes, ha="left",
                va="top", fontsize=10.5, color="#333333")
        for model in models:
            colour = COLOURS[model]
            art[(dataset, model, "band")] = ax.fill_between(
                counts, np.zeros_like(counts, dtype=float),
                np.zeros_like(counts, dtype=float), color=colour, alpha=0.16,
                linewidth=0)
            art[(dataset, model, "line")] = ax.plot(
                [], [], color=colour, lw=2.0,
                label=model if dataset == datasets[0] else None)[0]
            art[(dataset, model, "head")] = ax.plot(
                [], [], "o", color=colour, markersize=5.5)[0]
            # The band needs two points to draw anything, so the leading
            # interval is also drawn as a whisker. Without it the first frame
            # reads as if two seeds had no spread at all.
            art[(dataset, model, "whisker")] = ax.plot(
                [], [], color=colour, lw=1.2, alpha=0.75)[0]
        art[(dataset, "note")] = ax.text(
            0.5, 0.03, "", transform=ax.transAxes, fontsize=8.5, color="#5a5a5a",
            ha="center", va="bottom")
    axes[0].set_ylabel("running leakage inflation\n(accuracy points)")
    # The legend lives in the last panel, where the data never reaches the top
    # right. An animation is saved without the tight bounding box, so anything
    # parked outside the axes would be cropped off.
    handles, names = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, names, loc="upper right", fontsize=9)
    # Lay out first, title second. The house title is left-aligned and long, so
    # tight_layout counts it as overhang off the right of the first panel and
    # crushes all three panels to make room for it.
    figure.tight_layout()
    figure.subplots_adjust(top=0.80)
    titled(axes[0], "Ten seeds is not enough to resolve one accuracy point",
           "running paired mean, band is two standard errors, seeds in the order "
           "they were run")

    hold, tail = 4, 22
    order = list(np.repeat(np.arange(len(counts)), hold)) + [len(counts) - 1] * tail

    def draw(frame: int):
        i = order[frame]
        n = counts[i]
        for dataset in datasets:
            for model in models:
                key = (dataset, model)
                x, y, e = counts[:i + 1], means[key][:i + 1], errors[key][:i + 1]
                art[(dataset, model, "band")].remove()
                art[(dataset, model, "band")] = art[(dataset, model, "line")].axes.fill_between(
                    x, y - 2 * e, y + 2 * e, color=COLOURS[model], alpha=0.16, linewidth=0)
                art[(dataset, model, "line")].set_data(x, y)
                art[(dataset, model, "head")].set_data(x[-1:], y[-1:])
                art[(dataset, model, "whisker")].set_data(
                    [x[-1], x[-1]], [y[-1] - 2 * e[-1], y[-1] + 2 * e[-1]])
            cleared = [m for m in models
                       if abs(means[(dataset, m)][i]) > 2 * errors[(dataset, m)][i]]
            if len(cleared) == len(models):
                verdict = "both clear zero"
            elif cleared:
                verdict = f"{cleared[0]} clears zero"
            else:
                verdict = "neither clears zero"
            art[(dataset, "note")].set_text(f"{n} seeds: {verdict}")
        return []

    animation = FuncAnimation(figure, draw, frames=len(order),
                              interval=1000 // fps, blit=False)
    animation.save(out, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(figure)
    _shrink(out)
    return out


def _shrink(path: Path) -> None:
    """Requantise every frame onto one shared palette. Roughly halves the file."""
    source = Image.open(path)
    frames, durations = [], []
    try:
        while True:
            frames.append(source.convert("RGB"))
            durations.append(source.info.get("duration", 62))
            source.seek(source.tell() + 1)
    except EOFError:
        pass
    shared = frames[len(frames) // 2].quantize(64, method=Image.Quantize.MEDIANCUT)
    quantised = [f.quantize(palette=shared, dither=Image.Dither.NONE) for f in frames]
    quantised[0].save(path, save_all=True, append_images=quantised[1:], loop=0,
                      duration=durations, optimize=True)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for path in (
        inflation(FIGURES / "leakage-inflation.png"),
        leakage_channels(FIGURES / "leakage-channels.png"),
        decomposition(FIGURES / "inflation-decomposition.png"),
        controls(FIGURES / "controls.png"),
        duplicate_definitions(FIGURES / "duplicate-definitions.png"),
        seed_accumulation(FIGURES / "seeds-versus-noise.gif"),
    ):
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
