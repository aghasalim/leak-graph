"""The density control for the headline metric.

Transductive minus inductive is not a clean measurement of leakage. The inductive training
graph is missing the test nodes' information, which is the thing we want to price, but it is
also just a smaller and sparser graph, and that costs accuracy by itself. This script removes
an equal number of *unlabelled non-test* nodes instead -- nodes the model was never going to
be scored on -- so that the size change is held constant and only the identity of what was
removed differs.

    transductive - density_control  = the cost of a smaller training graph
    density_control - inductive     = what is specific to hiding the test nodes

Only runs on splits that leave part of the graph unlabelled. chameleon and squirrel use the
geom-gcn splits, which assign every node to train, val or test, so there is no pool of spare
nodes and the control cannot be constructed for them at all.

    python experiments/run_density_control.py
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leakgraph.data import PLANETOID, load  # noqa: E402
from leakgraph.experiment import run_audit  # noqa: E402

REPORTS = Path(__file__).resolve().parents[1] / "reports"
REGIMES = ("transductive", "density_control", "inductive")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(PLANETOID))
    ap.add_argument("--models", nargs="+", default=["GCN", "GraphSAGE"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()

    rows, summary = [], []
    for name in args.datasets:
        data = load(name)
        try:
            got = run_audit(
                data,
                seeds=list(range(args.seeds)),
                model_names=tuple(args.models),
                regimes=REGIMES,
                epochs=args.epochs,
            )
        except ValueError as exc:
            # Expected for chameleon/squirrel. Record it rather than skipping silently.
            print(f"[{name}] control not constructible: {exc}", flush=True)
            summary.append({"dataset": name, "model": "-", "status": f"not measurable: {exc}"})
            continue
        rows.extend(got)

        for model in args.models:
            cell = [r for r in got if r["model"] == model]
            seeds = sorted({r["seed"] for r in cell})

            def arm(regime):
                return [
                    next(
                        r["test_accuracy"]
                        for r in cell
                        if r["seed"] == s and r["regime"] == regime
                    )
                    for s in seeds
                ]

            trans, ctl, ind = arm("transductive"), arm("density_control"), arm("inductive")
            density = [t - c for t, c in zip(trans, ctl)]
            specific = [c - i for c, i in zip(ctl, ind)]
            total = [t - i for t, i in zip(trans, ind)]

            def ms(v):
                return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)

            d_m, d_s = ms(density)
            s_m, s_s = ms(specific)
            t_m, t_s = ms(total)
            summary.append(
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
                    "status": "measured",
                }
            )
            print(
                f"[{name}] {model:<10} total={t_m:+.4f}+-{t_s:.4f} "
                f"density={d_m:+.4f}+-{d_s:.4f} test-specific={s_m:+.4f}+-{s_s:.4f}",
                flush=True,
            )

    import pandas as pd

    REPORTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(REPORTS / "density_control_runs.csv", index=False)
    pd.DataFrame(summary).to_csv(REPORTS / "density_control.csv", index=False)
    print(f"wrote {REPORTS}/density_control.csv", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
