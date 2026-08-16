"""The density control for the headline metric.

Transductive minus inductive is not a clean measurement of leakage. The inductive training
graph is missing the test nodes' information, which is the thing we want to price, but it is
also just a smaller and sparser graph, and that costs accuracy by itself. This script removes
an equal number of *unlabelled non-test* nodes instead -- nodes the model was never going to
be scored on -- so that the size change is held constant and only the identity of what was
removed differs.

    transductive - density_control  = the cost of a smaller training graph
    density_control - inductive     = what is specific to hiding the test nodes

Two modes.

Default: draw the removed nodes from whatever the split leaves unlabelled. That works on the
Planetoid public splits and fails on chameleon and squirrel, whose geom-gcn splits assign
every node to train, val or test, so there is no pool at all.

`--bisect`: reserve half of the test set as the pool and score on the other half. A reserved
node keeps its features and edges in the graph, its label is never used, and it is never
scored -- the exact status of an unlabelled Planetoid node -- so the control becomes
constructible on every dataset. It is run on the Planetoid splits too, where the ordinary
control also works, so the two can be compared instead of the bisection being assumed valid.

`--random-splits`: throw away the shipped splits entirely and draw ten random Planetoid-style
splits per dataset (20 per class train, 500 val, 500 test). This is a second split scheme,
so its variance includes split variance rather than initialisation only, and it leaves a
large unlabelled pool on every dataset -- an independent route to the control on chameleon
and squirrel that shares no construction with `--bisect`.

    python experiments/run_density_control.py
    python experiments/run_density_control.py --bisect
    python experiments/run_density_control.py --random-splits
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import asdict, replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leakgraph.data import DATASETS, PLANETOID, load  # noqa: E402
from leakgraph.experiment import train_one  # noqa: E402
from leakgraph.splits import bisect_test_split, random_split  # noqa: E402

REPORTS = Path(__file__).resolve().parents[1] / "reports"
REGIMES = ("transductive", "density_control", "inductive")
# MLP and LabelProp cannot have a gap in any arm: neither reads the graph the density control
# modifies. They are here as the instrument's calibration, exactly as in the headline table.
MODELS = ("GCN", "GraphSAGE", "MLP", "LabelProp")


def run_dataset(data, seeds: list[int], models: tuple[str, ...], epochs: int, bisect: bool):
    """Every (model, regime, seed) cell for one dataset. Seed s uses split s % len(splits)."""
    rows = []
    for seed in seeds:
        idx = seed % len(data.splits)
        split, pool = data.splits[idx], None
        if bisect:
            # Seeded by the split index, not the run seed: which half is reserved is a
            # property of the split, so every seed sharing a split must reserve the same half.
            split, pool = bisect_test_split(split, seed=idx)
        for model in models:
            for regime in REGIMES:
                res = train_one(
                    data, split, model, regime, seed, pool_mask=pool, epochs=epochs
                )
                rows.append({**asdict(res), "split_index": idx})
    return rows


def summarise(name: str, rows: list[dict], models: tuple[str, ...]) -> list[dict]:
    out = []
    for model in models:
        cell = [r for r in rows if r["model"] == model]
        seeds = sorted({r["seed"] for r in cell})

        def arm(regime):
            return [
                next(r["test_accuracy"] for r in cell if r["seed"] == s and r["regime"] == regime)
                for s in seeds
            ]

        trans, ctl, ind = arm("transductive"), arm("density_control"), arm("inductive")
        density = [t - c for t, c in zip(trans, ctl)]
        specific = [c - i for c, i in zip(ctl, ind)]
        total = [t - i for t, i in zip(trans, ind)]

        def ms(v):
            return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)

        def resolved(v):
            """Two standard errors of the paired component, same test as the headline table."""
            m, s = ms(v)
            return abs(m) > 2 * s / (len(v) ** 0.5) if s > 0 else m != 0

        d_m, d_s = ms(density)
        s_m, s_s = ms(specific)
        t_m, t_s = ms(total)
        out.append(
            {
                "dataset": name,
                "model": model,
                "seeds": len(seeds),
                "transductive_mean": ms(trans)[0],
                "density_control_mean": ms(ctl)[0],
                "inductive_mean": ms(ind)[0],
                "total_inflation_mean": t_m,
                "total_inflation_std": t_s,
                "density_cost_mean": d_m,
                "density_cost_std": d_s,
                "test_specific_mean": s_m,
                "test_specific_std": s_s,
                "total_resolved": resolved(total),
                "density_resolved": resolved(density),
                "test_specific_resolved": resolved(specific),
                "status": "measured",
            }
        )
        print(
            f"[{name}] {model:<10} total={t_m:+.4f}+-{t_s:.4f} "
            f"density={d_m:+.4f}+-{d_s:.4f}{'*' if resolved(density) else ' '} "
            f"test-specific={s_m:+.4f}+-{s_s:.4f}{'*' if resolved(specific) else ' '}",
            flush=True,
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument(
        "--bisect",
        action="store_true",
        help="reserve half the test set as the removal pool; works on every dataset",
    )
    ap.add_argument(
        "--random-splits",
        action="store_true",
        help="replace the shipped splits with ten random ones (20/class, 500 val, 500 test)",
    )
    args = ap.parse_args()
    if args.bisect and args.random_splits:
        ap.error("--bisect and --random-splits are two different controls; run them separately")
    datasets = args.datasets or list(
        PLANETOID if not (args.bisect or args.random_splits) else DATASETS
    )
    models = tuple(args.models)

    rows, summary = [], []
    for name in datasets:
        data = load(name)
        if args.random_splits:
            data = replace(
                data,
                splits=[
                    random_split(data.num_nodes, data.y, seed=s, num_val=500, num_test=500)
                    for s in range(args.seeds)
                ],
            )
        try:
            got = run_dataset(data, list(range(args.seeds)), models, args.epochs, args.bisect)
        except ValueError as exc:
            # Expected for chameleon/squirrel without --bisect. Record it rather than skipping.
            print(f"[{name}] control not constructible: {exc}", flush=True)
            summary.append({"dataset": name, "model": "-", "status": f"not measurable: {exc}"})
            continue
        rows.extend(got)
        summary.extend(summarise(name, got, models))

    import pandas as pd

    REPORTS.mkdir(parents=True, exist_ok=True)
    stem = (
        "bisected_control"
        if args.bisect
        else "random_split_control"
        if args.random_splits
        else "density_control"
    )
    pd.DataFrame(rows).to_csv(REPORTS / f"{stem}_runs.csv", index=False)
    pd.DataFrame(summary).to_csv(REPORTS / f"{stem}.csv", index=False)
    print(f"wrote {REPORTS}/{stem}.csv", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
