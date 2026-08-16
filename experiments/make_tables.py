"""Render the README's tables straight from reports/*.csv and reports/detectors.json.

Every number in the README is produced by this script from a committed artifact. Nothing is
typed in by hand, because a hand-typed number is a number nobody can check.

The script does two things: it writes reports/tables.md, and it rewrites the regions of
README.md delimited by <!--BEGIN:name--> / <!--END:name--> markers. So the claim "these
numbers came from the committed artifacts" is checkable by anyone: re-run it and see whether
git reports a diff.

    python experiments/make_tables.py > reports/tables.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPORTS = Path(__file__).resolve().parents[1] / "reports"
ORDER = ["Cora", "CiteSeer", "PubMed", "chameleon", "squirrel"]
MODELS = ["GCN", "GraphSAGE", "MLP", "LabelProp"]


def pp(x: float) -> str:
    """Percentage points, to one decimal."""
    return f"{100 * x:.1f}"


def sort_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_d"] = df["dataset"].apply(lambda d: ORDER.index(d) if d in ORDER else 99)
    df["_m"] = df["model"].apply(lambda m: MODELS.index(m) if m in MODELS else 99)
    return df.sort_values(["_d", "_m"]).drop(columns=["_d", "_m"])


def inflation_table(df: pd.DataFrame) -> str:
    lines = [
        "| dataset | model | transductive | inductive | **inflation** | resolved? |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in sort_key(df).iterrows():
        mark = "yes" if r["resolved"] else "**no**"
        lines.append(
            f"| {r['dataset']} | {r['model']} | "
            f"{pp(r['transductive_mean'])} ± {pp(r['transductive_std'])} | "
            f"{pp(r['inductive_mean'])} ± {pp(r['inductive_std'])} | "
            f"**{pp(r['inflation_mean'])} ± {pp(r['inflation_std'])}** | {mark} |"
        )
    return "\n".join(lines)


def component_table(df: pd.DataFrame) -> str:
    lines = [
        "| dataset | model | inflation (all test) | inflation (no straddling duplicates) "
        "| duplicate component |",
        "|---|---|---|---|---|",
    ]
    for _, r in sort_key(df[df["model"].isin(["GCN", "GraphSAGE"])]).iterrows():
        lines.append(
            f"| {r['dataset']} | {r['model']} | "
            f"{pp(r['inflation_mean'])} ± {pp(r['inflation_std'])} | "
            f"{pp(r['inflation_dedup_mean'])} ± {pp(r['inflation_dedup_std'])} | "
            f"{pp(r['duplicate_component'])} |"
        )
    return "\n".join(lines)


def baseline_table(df: pd.DataFrame, det: list[dict]) -> str:
    """Inductive GNN accuracy against the two baselines that bound its interpretation."""
    lines = [
        "| dataset | MLP (features, no graph) | neighbour vote (graph, no features, no "
        "learning) | GCN inductive | GCN inductive, same nodes as the vote |",
        "|---|---|---|---|---|",
    ]
    by_ds = {d["dataset"]: d for d in det}
    for name in ORDER:
        sub = df[df["dataset"] == name]
        if sub.empty:
            continue
        mlp = sub[sub["model"] == "MLP"]["inductive_mean"].iloc[0]
        gcn = sub[sub["model"] == "GCN"]
        nbr = by_ds[name]["neighbour_label"]
        lines.append(
            f"| {name} | {pp(mlp)} | "
            f"{pp(nbr['vote_accuracy_on_covered'])} "
            f"(covers {pp(nbr['frac_test_with_train_neighbour'])}% of test) | "
            f"{pp(gcn['inductive_mean'].iloc[0])} | "
            f"{pp(gcn['inductive_nbr_covered_mean'].iloc[0])} ± "
            f"{pp(gcn['inductive_nbr_covered_std'].iloc[0])} |"
        )
    return "\n".join(lines)


def detector_table(det: list[dict]) -> str:
    lines = [
        "| dataset | nodes | exact duplicate nodes | all-zero feature rows | near-dup pairs "
        "| cutoff | straddling pairs | test nodes with a train twin |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for d in sorted(det, key=lambda d: ORDER.index(d["dataset"])):
        u = d["duplicates"]
        pct = 100 * u["exact_duplicate_nodes"] / u["num_nodes"]
        lines.append(
            f"| {d['dataset']} | {u['num_nodes']} | "
            f"{u['exact_duplicate_nodes']} ({pct:.1f}%) | "
            f"{u.get('zero_feature_nodes', 'n/a')} | "
            f"{u['near_duplicate_pairs']} | {u['calibrated_threshold']:.3f} | "
            f"{u['straddling_pairs']} | "
            f"{u['test_nodes_with_train_twin']} / {u['test_nodes']} "
            f"({pp(u['frac_test_with_train_twin'])}%) |"
        )
    return "\n".join(lines)


def feature_table(det: list[dict]) -> str:
    lines = [
        "| dataset | logistic regression on features alone | majority class | giveaway "
        "features | expected under label-permuted null | test nodes covered |",
        "|---|---|---|---|---|---|",
    ]
    for d in sorted(det, key=lambda d: ORDER.index(d["dataset"])):
        f = d["feature_label"]
        lines.append(
            f"| {d['dataset']} | {pp(f['logreg_test_accuracy'])} | "
            f"{pp(f['majority_class_accuracy'])} | "
            f"{f['giveaway_features']} / {f['num_features']} | "
            f"{f['null_giveaway_features']:.1f} | "
            f"{pp(f['frac_test_covered_by_giveaway'])}% |"
        )
    return "\n".join(lines)


def density_table(path: Path) -> str:
    """Splits the headline gap into "the graph got smaller" and "the test nodes specifically
    went away". Datasets where the control cannot be built are listed as such, not omitted.

    Each component carries its own two-standard-error resolution mark, the same test the
    headline table uses. Without it a decomposed component is only a direction.
    """
    if not path.exists():
        return "_not run_"
    dc = pd.read_csv(path)
    lines = [
        "| dataset | model | total inflation | density cost | resolved? | test-node-specific "
        "| resolved? |",
        "|---|---|---|---|---|---|---|",
    ]

    def mark(row, key):
        if key not in row or pd.isna(row[key]):
            return "n/a"
        return "yes" if bool(row[key]) else "**no**"

    for _, r in sort_key(dc).iterrows():
        if r["status"] != "measured":
            lines.append(f"| {r['dataset']} | - | _{r['status']}_ | | | | |")
            continue
        lines.append(
            f"| {r['dataset']} | {r['model']} | "
            f"{pp(r['total_inflation_mean'])} ± {pp(r['total_inflation_std'])} | "
            f"{pp(r['density_cost_mean'])} ± {pp(r['density_cost_std'])} | "
            f"{mark(r, 'density_resolved')} | "
            f"{pp(r['test_specific_mean'])} ± {pp(r['test_specific_std'])} | "
            f"{mark(r, 'test_specific_resolved')} |"
        )
    return "\n".join(lines)


def duplicate_definition_table(path: Path) -> str:
    """Which reading of "duplicate" reproduces the quoted 1% (Cora) / 5% (CiteSeer) pair.

    The CiteSeer/Cora ratio is the sharpest column: the quote implies 5x, and it does not
    depend on either absolute level.
    """
    if not path.exists():
        return "_not run_"
    payload = json.loads(path.read_text())
    rates, verdicts = payload["rates"], payload["verdicts"]
    lines = [
        "| definition | Cora | CiteSeer | CiteSeer / Cora | PubMed | chameleon | squirrel |",
        "|---|---|---|---|---|---|---|",
        "| _quoted by OGB from Zou et al._ | _1.0_ | _5.0_ | _5.0_ | _-_ | _-_ | _-_ |",
    ]
    for key in sorted(verdicts, key=lambda k: -abs(verdicts[k]["citeseer_over_cora"] - 5.0)):
        ratio = verdicts[key]["citeseer_over_cora"]
        cells = " | ".join(
            f"{100 * rates[d][key]:.2f}" if d in rates else "-"
            for d in ("PubMed", "chameleon", "squirrel")
        )
        lines.append(
            f"| `{key}` | {100 * rates['Cora'][key]:.2f} | {100 * rates['CiteSeer'][key]:.2f} | "
            f"{'∞' if ratio == float('inf') else f'{ratio:.2f}'} | {cells} |"
        )
    solves = payload.get("solves", {})
    if solves:
        lines.append("")
        lines.append(
            "| similarity | cutoff that puts CiteSeer at 5% | CiteSeer there | Cora there "
            "| Cora quoted |"
        )
        lines.append("|---|---|---|---|---|")
        for kind, s in solves.items():
            lines.append(
                f"| {kind} | {s['threshold_giving_citeseer_5pct']:.4f} | "
                f"{100 * s['citeseer_rate_there']:.2f} | "
                f"{100 * s['cora_rate_at_the_same_threshold']:.2f} | "
                f"{100 * s['cora_quoted']:.1f} |"
            )
    return "\n".join(lines)


def splice(text: str, name: str, body: str) -> str:
    """Replace whatever sits between <!--BEGIN:name--> and <!--END:name--> with `body`.

    This is what makes "every number in the README came from a run" checkable rather than
    merely asserted: `make tables` rewrites them all, so a stale or hand-edited number shows
    up as a dirty git diff.
    """
    begin, end = f"<!--BEGIN:{name}-->", f"<!--END:{name}-->"
    if begin not in text or end not in text:
        raise SystemExit(f"README is missing the {name} markers")
    head, rest = text.split(begin, 1)
    _, tail = rest.split(end, 1)
    return f"{head}{begin}\n{body}\n{end}{tail}"


def main() -> None:
    df = pd.read_csv(REPORTS / "inflation.csv")
    det = json.loads((REPORTS / "detectors.json").read_text())

    tables = {
        "inflation": inflation_table(df),
        "components": component_table(df),
        "baselines": baseline_table(df, det),
        "duplicates": detector_table(det),
        "features": feature_table(det),
        "density": density_table(REPORTS / "density_control.csv"),
        "bisected": density_table(REPORTS / "bisected_control.csv"),
        "randomsplit": density_table(REPORTS / "random_split_control.csv"),
        "dupdefs": duplicate_definition_table(REPORTS / "duplicate_definitions.json"),
    }

    for title, key in [
        ("Leakage inflation", "inflation"),
        ("Duplicate component", "components"),
        ("Baselines", "baselines"),
        ("Density control", "density"),
        ("Density control, bisected test set", "bisected"),
        ("Density control, random splits", "randomsplit"),
        ("Duplicate definitions", "dupdefs"),
        ("Duplicate detector", "duplicates"),
        ("Feature-label detector", "features"),
    ]:
        print(f"## {title}\n")
        print(tables[key])
        print()

    readme = REPORTS.parent / "README.md"
    if readme.exists():
        text = readme.read_text()
        for name, body in tables.items():
            text = splice(text, name, body)
        readme.write_text(text)
        print(f"<!-- spliced {len(tables)} tables into {readme.name} -->", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
