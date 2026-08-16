"""What definition of "duplicate" reproduces the duplication rates quoted in the literature?

OGB (arXiv:2005.00687), quoting Zou et al. (arXiv:1907.02237), states that "1% of the nodes
are duplicated" in Cora and that CiteSeer's "duplication rate" is 5%. My exact-feature-match
detector reproduces Cora's 1% and reads 1.1% on CiteSeer, and the README recorded that as an
unreconciled disagreement.

This script tries to reconcile it. Both figures come from the same sentence of the same
source, so a candidate definition has to hit *both*: roughly 1% on Cora and roughly 5% on
CiteSeer at the same setting. A definition that lands on 5% for CiteSeer while sending Cora
to 30% has not explained anything.

Fourteen definitions are evaluated on all five datasets, plus a solve: the cosine threshold
at which CiteSeer reads exactly 5%, and what Cora reads at that same threshold.

    python experiments/run_duplicate_definitions.py

Writes reports/duplicate_definitions.csv and reports/duplicate_definitions.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leakgraph.data import DATASETS, load  # noqa: E402

REPORTS = Path(__file__).resolve().parents[1] / "reports"

# The two numbers a candidate definition has to reproduce simultaneously.
QUOTED = {"Cora": 0.01, "CiteSeer": 0.05}
TOLERANCE = 0.005  # a definition "matches" if it lands within half a point of the quote


def _twinned(x: torch.Tensor, threshold: float, kind: str, chunk: int = 512) -> torch.Tensor:
    """Boolean mask of nodes having at least one other node at similarity >= threshold.

    Counts *nodes*, not pairs, because the quoted figures are a percentage of nodes. Computed
    chunk-wise so PubMed's 194M pairs never materialise.

    kind="cosine" on the raw feature rows; kind="jaccard" on their binary support, which is
    the natural similarity for a bag-of-words and is not monotone in cosine.
    """
    n = x.size(0)
    out = torch.zeros(n, dtype=torch.bool)
    if kind == "cosine":
        v = x.float()
        v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)
    elif kind == "jaccard":
        v = (x > 0).float()
        sizes = v.sum(dim=1)
    else:
        raise ValueError(kind)

    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        inter = v[start:stop] @ v.T
        if kind == "cosine":
            sims = inter
        else:
            union = sizes[start:stop, None] + sizes[None, :] - inter
            sims = inter / union.clamp_min(1e-12)
        sims[torch.arange(stop - start), torch.arange(start, stop)] = -1.0
        out[start:stop] = (sims >= threshold).any(dim=1)
    return out


def _exact_groups(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(group size of each node's feature row, mask of all-zero rows)."""
    _, inverse, counts = torch.unique(x, dim=0, return_inverse=True, return_counts=True)
    return counts[inverse], (x != 0).sum(dim=1) == 0


def _structural_twins(edge_index: torch.Tensor, n: int) -> torch.Tensor:
    """Nodes sharing their exact neighbour set with another node. Features ignored entirely.

    Isolated nodes are excluded: they all trivially share the empty neighbourhood, which
    would be an artefact of the encoding rather than a duplicate document.
    """
    acc: list[set] = [set() for _ in range(n)]
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    for u, v in zip(src, dst):
        acc[u].add(v)
        acc[v].add(u)
    nbrs = [frozenset(s) for s in acc]
    seen: dict[frozenset, int] = {}
    for s in nbrs:
        if s:
            seen[s] = seen.get(s, 0) + 1
    return torch.tensor([bool(s) and seen[s] > 1 for s in nbrs])


def definitions(data) -> dict[str, float]:
    """Every candidate reading of "the duplication rate", as a fraction of nodes."""
    x, y, n = data.x, data.y, data.num_nodes
    group, zero = _exact_groups(x)
    dup = group > 1
    _, inv, counts = torch.unique(x, dim=0, return_inverse=True, return_counts=True)

    # identical features AND identical label / identical features but conflicting labels
    same_label = torch.zeros(n, dtype=torch.bool)
    conflict = torch.zeros(n, dtype=torch.bool)
    for gid in torch.nonzero(counts > 1).flatten().tolist():
        members = torch.nonzero(inv == gid).flatten()
        labels = y[members]
        if bool((labels == labels[0]).all()):
            same_label[members] = True
        else:
            conflict[members] = True

    out = {
        # ---- exact feature match, counted several ways -------------------------------
        "exact_nodes_in_a_duplicate_group": float(dup.sum()) / n,
        "exact_nodes_excluding_all_zero_rows": float((dup & ~zero).sum()) / n,
        "exact_redundant_copies": sum(int(c) - 1 for c in counts if c > 1) / n,
        "exact_duplicate_pairs_over_nodes": sum(int(c) * (int(c) - 1) // 2 for c in counts) / n,
        # the same, with the all-zero rows dropped first: CiteSeer's 15 empty feature vectors
        # are mutually identical and contribute 105 pairs on their own, so the pair-counting
        # definition needs to be checked with and without them
        "exact_duplicate_pairs_over_nodes_excluding_zero_rows": sum(
            int(c) * (int(c) - 1) // 2
            for c in torch.unique(x[~zero], dim=0, return_counts=True)[1]
        )
        / n,
        "exact_and_same_label": float(same_label.sum()) / n,
        "exact_but_conflicting_label": float(conflict.sum()) / n,
        "exact_on_binarised_features": float(
            (_exact_groups((x > 0).float())[0] > 1).sum()
        )
        / n,
        # ---- near duplicates at fixed cosine cutoffs ---------------------------------
        **{
            f"cosine_{t}": float(_twinned(x, t, "cosine").sum()) / n
            for t in (0.99, 0.95, 0.90, 0.80, 0.70)
        },
        # ---- near duplicates by Jaccard on the bag-of-words support ------------------
        **{
            f"jaccard_{t}": float(_twinned(x, t, "jaccard").sum()) / n
            for t in (0.95, 0.80, 0.50)
        },
        # ---- no features at all ------------------------------------------------------
        "identical_neighbour_set": float(_structural_twins(data.edge_index, n).sum()) / n,
        "all_zero_feature_rows": float(zero.sum()) / n,
    }
    return out


def solve_threshold(x: torch.Tensor, target: float, kind: str) -> tuple[float, float]:
    """Bisect for the similarity cutoff at which the node-level rate equals `target`.

    The rate is monotone non-increasing in the threshold, so bisection is well posed.
    Returns (threshold, rate achieved).
    """
    n = x.size(0)
    lo, hi = 0.0, 1.0
    best = (1.0, 0.0)
    for _ in range(24):
        mid = (lo + hi) / 2
        rate = float(_twinned(x, mid, kind).sum()) / n
        best = (mid, rate)
        if rate > target:
            lo = mid
        else:
            hi = mid
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    args = ap.parse_args()

    table: dict[str, dict[str, float]] = {}
    for name in args.datasets:
        data = load(name)
        table[name] = definitions(data)
        print(f"[{name}] {len(table[name])} definitions computed", flush=True)

    # Which definitions reproduce BOTH quoted figures at the same setting?
    verdicts = {}
    if set(QUOTED) <= set(table):
        for key in next(iter(table.values())):
            hits = {
                ds: abs(table[ds][key] - want) <= TOLERANCE for ds, want in QUOTED.items()
            }
            verdicts[key] = {
                "cora": table["Cora"][key],
                "citeseer": table["CiteSeer"][key],
                # The quote implies CiteSeer is 5x Cora. That ratio is the sharpest test of a
                # candidate definition, because it does not depend on either absolute level.
                "citeseer_over_cora": (
                    table["CiteSeer"][key] / table["Cora"][key]
                    if table["Cora"][key] > 0
                    else float("inf")
                ),
                "reproduces_both": all(hits.values()),
                "reproduces_cora_only": hits["Cora"] and not hits["CiteSeer"],
                "reproduces_citeseer_only": hits["CiteSeer"] and not hits["Cora"],
            }

    # The solve: what cutoff makes CiteSeer read 5%, and what does Cora read there?
    solves = {}
    if "CiteSeer" in table and "Cora" in table:
        cs, cora = load("CiteSeer"), load("Cora")
        for kind in ("cosine", "jaccard"):
            thr, rate = solve_threshold(cs.x, QUOTED["CiteSeer"], kind)
            cora_rate = float(_twinned(cora.x, thr, kind).sum()) / cora.num_nodes
            solves[kind] = {
                "threshold_giving_citeseer_5pct": thr,
                "citeseer_rate_there": rate,
                "cora_rate_at_the_same_threshold": cora_rate,
                "cora_quoted": QUOTED["Cora"],
            }
            print(
                f"[solve/{kind}] CiteSeer hits {rate:.3%} at threshold {thr:.4f}; "
                f"Cora reads {cora_rate:.3%} there (quoted 1%)",
                flush=True,
            )

    payload = {
        "quoted": QUOTED,
        "tolerance": TOLERANCE,
        "rates": table,
        "verdicts": verdicts,
        "solves": solves,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "duplicate_definitions.json").write_text(json.dumps(payload, indent=2))

    import pandas as pd

    pd.DataFrame(table).to_csv(REPORTS / "duplicate_definitions.csv")
    winners = [k for k, v in verdicts.items() if v["reproduces_both"]]
    print(f"definitions reproducing both quoted figures: {winners or 'none'}", flush=True)
    print(f"wrote {REPORTS}/duplicate_definitions.csv", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(2)
    main()
