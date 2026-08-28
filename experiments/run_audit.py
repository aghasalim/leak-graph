"""Run the full audit: detectors, then transductive vs inductive training, then the tables.

    python experiments/run_audit.py --datasets Cora CiteSeer --seeds 10

Writes raw per-run rows to reports/runs.csv, detector output to reports/detectors.json, and
the aggregated inflation table to reports/inflation.csv.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leakgraph.data import DATASETS, load  # noqa: E402
from leakgraph.detectors import (  # noqa: E402
    duplicate_report,
    feature_label_report,
    neighbour_label_report,
    scan_duplicates,
)
from leakgraph.experiment import run_audit  # noqa: E402
from leakgraph.models import TRAINED_MODELS  # noqa: E402

REPORTS = Path(__file__).resolve().parents[1] / "reports"


def covered_by_train_neighbour(edge_index, train_mask, test_mask):
    """Test nodes with at least one training-set neighbour, the subset the parameter-free
    neighbour vote can actually predict. GNN accuracy is also reported on exactly this subset
    so the two numbers answer the same question."""
    has_train_nbr = torch.zeros_like(test_mask)
    src, dst = edge_index[0], edge_index[1]
    has_train_nbr[dst[train_mask[src]]] = True
    return test_mask & has_train_nbr


def detectors_for(data, split_index: int = 0, scan=None):
    """Run all three detector families on one split.

    Returns (payload, dedup_test_mask, covered_test_mask)."""
    split = data.splits[split_index]
    dup, calib, twin = duplicate_report(
        data.x, data.y, split.train_mask, split.test_mask, scan=scan
    )
    payload = {
        "dataset": data.name,
        "split_index": split_index,
        "num_nodes": data.num_nodes,
        "num_edges": int(data.edge_index.size(1)),
        "num_classes": data.num_classes,
        "duplicates": dup.to_dict(),
        "cosine_calibration": {
            "threshold": calib.threshold,
            "null_pairs_examined": calib.null_pairs_examined,
            "null_subsample_nodes": calib.null_subsample_nodes,
            "real_pairs_examined": calib.real_pairs_examined,
            "note": calib.note,
        },
        "feature_label": feature_label_report(
            data.x, data.y, split.train_mask, split.test_mask
        ).to_dict(),
        "neighbour_label": neighbour_label_report(
            data.edge_index, data.y, split.train_mask, split.test_mask
        ).to_dict(),
    }
    # the de-duplicated test set: test nodes that have NO near-duplicate twin in training
    return (
        payload,
        split.test_mask & ~twin,
        covered_by_train_neighbour(data.edge_index, split.train_mask, split.test_mask),
    )


def summarise(rows: list[dict]) -> list[dict]:
    """Aggregate to one row per (dataset, model), with the paired inflation statistic."""
    out = []
    keys = sorted({(r["dataset"], r["model"]) for r in rows})
    for dataset, model in keys:
        cell = [r for r in rows if r["dataset"] == dataset and r["model"] == model]
        seeds = sorted({r["seed"] for r in cell})

        def arm(regime, field="test_accuracy"):
            return [
                next(
                    r[field]
                    for r in cell
                    if r["seed"] == s and r["regime"] == regime
                )
                for s in seeds
            ]

        trans, ind = arm("transductive"), arm("inductive")
        trans_d, ind_d = arm("transductive", "test_accuracy_dedup"), arm(
            "inductive", "test_accuracy_dedup"
        )
        ind_cov = arm("inductive", "test_accuracy_nbr_covered")
        paired = [t - i for t, i in zip(trans, ind)]
        paired_dedup = [t - i for t, i in zip(trans_d, ind_d)]

        def ms(v):
            return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)

        t_m, t_s = ms(trans)
        i_m, i_s = ms(ind)
        p_m, p_s = ms(paired)
        pd_m, pd_s = ms(paired_dedup)
        out.append(
            {
                "dataset": dataset,
                "model": model,
                "seeds": len(seeds),
                "transductive_mean": t_m,
                "transductive_std": t_s,
                "inductive_mean": i_m,
                "inductive_std": i_s,
                # inductive accuracy restricted to the nodes the neighbour vote can predict,
                # so it can be compared with the neighbour vote directly
                "inductive_nbr_covered_mean": ms(ind_cov)[0],
                "inductive_nbr_covered_std": ms(ind_cov)[1],
                "inflation_mean": p_m,
                "inflation_std": p_s,
                "inflation_stderr": p_s / (len(seeds) ** 0.5),
                # the same paired difference, but scored only on test nodes that have no
                # near-duplicate twin in the training set
                "inflation_dedup_mean": pd_m,
                "inflation_dedup_std": pd_s,
                # what the duplicates were worth: how much of the inflation disappears once
                # the straddling test nodes are excluded from scoring
                "duplicate_component": p_m - pd_m,
                # Two standard errors of the PAIRED difference. Anything that fails this is
                # reported as unresolved rather than dressed up as a small effect.
                "resolved": abs(p_m) > 2 * p_s / (len(seeds) ** 0.5) if p_s > 0 else p_m != 0,
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--out", default=str(REPORTS))
    ap.add_argument(
        "--detectors-only",
        action="store_true",
        help="run the detectors and rewrite detectors.json without retraining anything",
    )
    args = ap.parse_args()

    reports = Path(args.out)
    reports.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    all_detectors: list[dict] = []

    for name in args.datasets:
        print(f"[{name}] loading", flush=True)
        data = load(name)

        # one pairwise sweep for the whole dataset, reused across its splits
        scan = scan_duplicates(data.x)

        dedup_masks, covered_masks = [], []
        for split_index in range(len(data.splits)):
            payload, dedup, covered = detectors_for(data, split_index, scan=scan)
            dedup_masks.append(dedup)
            covered_masks.append(covered)
            if split_index == 0:
                all_detectors.append(payload)
                print(
                    f"[{name}] detectors: "
                    f"near-dup pairs={payload['duplicates']['near_duplicate_pairs']} "
                    f"straddling={payload['duplicates']['straddling_pairs']} "
                    f"logreg={payload['feature_label']['logreg_test_accuracy']:.3f} "
                    f"nbr-vote={payload['neighbour_label']['vote_accuracy_overall']:.3f}",
                    flush=True,
                )

        if args.detectors_only:
            continue

        rows = run_audit(
            data,
            seeds=list(range(args.seeds)),
            model_names=TRAINED_MODELS + ("LabelProp",),
            dedup_test_masks=dedup_masks,
            covered_test_masks=covered_masks,
            epochs=args.epochs,
        )
        all_rows.extend(rows)
        for s in summarise(rows):
            print(
                f"[{name}] {s['model']:<10} "
                f"trans={s['transductive_mean']:.4f}+-{s['transductive_std']:.4f} "
                f"ind={s['inductive_mean']:.4f}+-{s['inductive_std']:.4f} "
                f"inflation={s['inflation_mean']:+.4f}+-{s['inflation_std']:.4f}",
                flush=True,
            )

        # Checkpoint after every dataset rather than once at the end. The full sweep takes
        # hours, and an earlier version of this script wrote only on completion: when that
        # run was killed partway, every completed dataset was lost with it. Rewriting the
        # whole file each time is O(datasets^2) in IO and completely irrelevant at five rows.
        _write(reports, all_rows, all_detectors)
        print(f"[{name}] checkpointed {len(all_rows)} rows", flush=True)

    _write(reports, all_rows, all_detectors)
    written = "detectors.json" if args.detectors_only else "runs.csv, inflation.csv, detectors.json"
    print(f"wrote {reports}/{written}", flush=True)


def _write(reports: Path, rows: list[dict], detectors: list[dict]) -> None:
    (reports / "detectors.json").write_text(json.dumps(detectors, indent=2))
    if not rows:
        return
    import pandas as pd

    pd.DataFrame(rows).to_csv(reports / "runs.csv", index=False)
    pd.DataFrame(summarise(rows)).to_csv(reports / "inflation.csv", index=False)


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
