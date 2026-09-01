// Re-render every generated table in the docs, in JavaScript, and require the
// committed text to match.
//
// experiments/make_tables.py writes nine tables into README.md and
// notes/METHODS.md from reports/. Its --check mode compares the committed docs
// against its own output, which catches a hand edit and nothing else: if the
// renderer itself dropped a column, rounded the wrong way or read the wrong
// field, --check would agree with the mistake and pass.
//
// This is a second renderer, written from the summary artifacts rather than from
// make_tables.py, and it compares against the committed docs directly. It is the
// last link in the chain the rest of verify/ checks: SQL and C get from the raw
// runs to the summary CSVs, this gets from the summary CSVs to the sentence a
// reader actually sees.
//
//   node verify/tables.mjs [root]

import { readFileSync } from "node:fs";
import { argv, exit } from "node:process";

const root = argv[2] ?? ".";
const ORDER = ["Cora", "CiteSeer", "PubMed", "chameleon", "squirrel"];
const MODELS = ["GCN", "GraphSAGE", "MLP", "LabelProp"];

/** Percentage points, to one decimal. */
const pp = (x) => (100 * Number(x)).toFixed(1);

function readCSV(name) {
  const text = readFileSync(`${root}/reports/${name}`, "utf8");
  const lines = text.split("\n").filter((l) => l.length > 0);
  const header = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const cells = line.split(",");
    return Object.fromEntries(header.map((h, i) => [h, cells[i]]));
  });
}

// Python's json module writes bare NaN and Infinity, which JSON.parse rejects.
// Both appear for real reasons, documented in verify/gocheck/main.go, and both
// are swapped for null here: nothing this file renders reads either of them.
function readJSON(name) {
  const raw = readFileSync(`${root}/reports/${name}`, "utf8");
  return JSON.parse(raw.replace(/\b(-?Infinity|NaN)\b/g, (m) => (m === "NaN" ? "null" : m === "-Infinity" ? "-1e999" : "1e999")));
}

const rank = (df) =>
  df
    .map((r, i) => [r, ORDER.indexOf(r.dataset) < 0 ? 99 : ORDER.indexOf(r.dataset), MODELS.indexOf(r.model) < 0 ? 99 : MODELS.indexOf(r.model), i])
    .sort((a, b) => a[1] - b[1] || a[2] - b[2] || a[3] - b[3])
    .map((t) => t[0]);

function inflationTable(df) {
  const lines = [
    "| dataset | model | transductive | inductive | **inflation** | resolved? |",
    "|---|---|---|---|---|---|",
  ];
  for (const r of rank(df)) {
    const mark = r.resolved === "True" ? "yes" : "**no**";
    lines.push(
      `| ${r.dataset} | ${r.model} | ${pp(r.transductive_mean)} ± ${pp(r.transductive_std)} | ` +
        `${pp(r.inductive_mean)} ± ${pp(r.inductive_std)} | ` +
        `**${pp(r.inflation_mean)} ± ${pp(r.inflation_std)}** | ${mark} |`
    );
  }
  return lines.join("\n");
}

function componentTable(df) {
  const lines = [
    "| dataset | model | inflation (all test) | inflation (no straddling duplicates) | duplicate component |",
    "|---|---|---|---|---|",
  ];
  for (const r of rank(df.filter((r) => r.model === "GCN" || r.model === "GraphSAGE"))) {
    lines.push(
      `| ${r.dataset} | ${r.model} | ${pp(r.inflation_mean)} ± ${pp(r.inflation_std)} | ` +
        `${pp(r.inflation_dedup_mean)} ± ${pp(r.inflation_dedup_std)} | ${pp(r.duplicate_component)} |`
    );
  }
  return lines.join("\n");
}

function baselineTable(df, det) {
  const lines = [
    "| dataset | MLP (features, no graph) | neighbour vote (graph, no features, no learning) | GCN inductive | GCN inductive, same nodes as the vote |",
    "|---|---|---|---|---|",
  ];
  const byDs = Object.fromEntries(det.map((d) => [d.dataset, d]));
  for (const name of ORDER) {
    const sub = df.filter((r) => r.dataset === name);
    if (sub.length === 0) continue;
    const mlp = sub.find((r) => r.model === "MLP").inductive_mean;
    const gcn = sub.find((r) => r.model === "GCN");
    const nbr = byDs[name].neighbour_label;
    lines.push(
      `| ${name} | ${pp(mlp)} | ${pp(nbr.vote_accuracy_on_covered)} ` +
        `(covers ${pp(nbr.frac_test_with_train_neighbour)}% of test) | ` +
        `${pp(gcn.inductive_mean)} | ${pp(gcn.inductive_nbr_covered_mean)} ± ${pp(gcn.inductive_nbr_covered_std)} |`
    );
  }
  return lines.join("\n");
}

function detectorTable(det) {
  const lines = [
    "| dataset | nodes | exact duplicate nodes | all-zero feature rows | near-dup pairs | cutoff | straddling pairs | test nodes with a train twin |",
    "|---|---|---|---|---|---|---|---|",
  ];
  for (const d of [...det].sort((a, b) => ORDER.indexOf(a.dataset) - ORDER.indexOf(b.dataset))) {
    const u = d.duplicates;
    const pct = ((100 * u.exact_duplicate_nodes) / u.num_nodes).toFixed(1);
    const zero = "zero_feature_nodes" in u ? u.zero_feature_nodes : "n/a";
    lines.push(
      `| ${d.dataset} | ${u.num_nodes} | ${u.exact_duplicate_nodes} (${pct}%) | ${zero} | ` +
        `${u.near_duplicate_pairs} | ${u.calibrated_threshold.toFixed(3)} | ${u.straddling_pairs} | ` +
        `${u.test_nodes_with_train_twin} / ${u.test_nodes} (${pp(u.frac_test_with_train_twin)}%) |`
    );
  }
  return lines.join("\n");
}

function featureTable(det) {
  const lines = [
    "| dataset | logistic regression on features alone | majority class | giveaway features | expected under label-permuted null | test nodes covered |",
    "|---|---|---|---|---|---|",
  ];
  for (const d of [...det].sort((a, b) => ORDER.indexOf(a.dataset) - ORDER.indexOf(b.dataset))) {
    const f = d.feature_label;
    lines.push(
      `| ${d.dataset} | ${pp(f.logreg_test_accuracy)} | ${pp(f.majority_class_accuracy)} | ` +
        `${f.giveaway_features} / ${f.num_features} | ${f.null_giveaway_features.toFixed(1)} | ` +
        `${pp(f.frac_test_covered_by_giveaway)}% |`
    );
  }
  return lines.join("\n");
}

function densityTable(name) {
  const dc = readCSV(name);
  const lines = [
    "| dataset | model | total inflation | density cost | resolved? | test-node-specific | resolved? |",
    "|---|---|---|---|---|---|---|",
  ];
  const mark = (r, key) => (r[key] === undefined || r[key] === "" ? "n/a" : r[key] === "True" ? "yes" : "**no**");
  for (const r of rank(dc)) {
    if (r.status !== "measured") {
      lines.push(`| ${r.dataset} | - | _${r.status}_ | | | | |`);
      continue;
    }
    lines.push(
      `| ${r.dataset} | ${r.model} | ${pp(r.total_inflation_mean)} ± ${pp(r.total_inflation_std)} | ` +
        `${pp(r.density_cost_mean)} ± ${pp(r.density_cost_std)} | ${mark(r, "density_resolved")} | ` +
        `${pp(r.test_specific_mean)} ± ${pp(r.test_specific_std)} | ${mark(r, "test_specific_resolved")} |`
    );
  }
  return lines.join("\n");
}

function dupDefTable() {
  const payload = readJSON("duplicate_definitions.json");
  const { rates, verdicts, solves } = payload;
  const lines = [
    "| definition | Cora | CiteSeer | CiteSeer / Cora | PubMed | chameleon | squirrel |",
    "|---|---|---|---|---|---|---|",
    "| _quoted by OGB from Zou et al._ | _1.0_ | _5.0_ | _5.0_ | _-_ | _-_ | _-_ |",
  ];
  // Furthest from the quoted 5x ratio first, ties in insertion order, which is
  // what Python's stable sorted() over the JSON object gives.
  const keys = Object.keys(verdicts)
    .map((k, i) => [k, i])
    .sort((a, b) => -Math.abs(verdicts[a[0]].citeseer_over_cora - 5.0) + Math.abs(verdicts[b[0]].citeseer_over_cora - 5.0) || a[1] - b[1])
    .map((t) => t[0]);
  for (const key of keys) {
    const ratio = verdicts[key].citeseer_over_cora;
    const cells = ["PubMed", "chameleon", "squirrel"]
      .map((d) => (d in rates ? (100 * rates[d][key]).toFixed(2) : "-"))
      .join(" | ");
    lines.push(
      `| \`${key}\` | ${(100 * rates.Cora[key]).toFixed(2)} | ${(100 * rates.CiteSeer[key]).toFixed(2)} | ` +
        `${!Number.isFinite(ratio) ? "∞" : ratio.toFixed(2)} | ${cells} |`
    );
  }
  if (solves && Object.keys(solves).length > 0) {
    lines.push("");
    lines.push("| similarity | cutoff that puts CiteSeer at 5% | CiteSeer there | Cora there | Cora quoted |");
    lines.push("|---|---|---|---|---|");
    for (const [kind, s] of Object.entries(solves)) {
      lines.push(
        `| ${kind} | ${s.threshold_giving_citeseer_5pct.toFixed(4)} | ${(100 * s.citeseer_rate_there).toFixed(2)} | ` +
          `${(100 * s.cora_rate_at_the_same_threshold).toFixed(2)} | ${(100 * s.cora_quoted).toFixed(1)} |`
      );
    }
  }
  return lines.join("\n");
}

const inflation = readCSV("inflation.csv");
const detectors = readJSON("detectors.json");

const tables = {
  inflation: inflationTable(inflation),
  components: componentTable(inflation),
  baselines: baselineTable(inflation, detectors),
  duplicates: detectorTable(detectors),
  features: featureTable(detectors),
  density: densityTable("density_control.csv"),
  bisected: densityTable("bisected_control.csv"),
  randomsplit: densityTable("random_split_control.csv"),
  dupdefs: dupDefTable(),
};

let failures = 0;
let compared = 0;
const seen = Object.fromEntries(Object.keys(tables).map((k) => [k, 0]));

for (const doc of ["README.md", "notes/METHODS.md"]) {
  const text = readFileSync(`${root}/${doc}`, "utf8");
  for (const [name, body] of Object.entries(tables)) {
    const begin = `<!--BEGIN:${name}-->`;
    const end = `<!--END:${name}-->`;
    const i = text.indexOf(begin);
    if (i < 0) continue;
    const j = text.indexOf(end, i);
    if (j < 0) {
      console.log(`  ${doc}: ${begin} has no matching ${end}  FAIL`);
      failures++;
      continue;
    }
    seen[name]++;
    compared++;
    const committed = text.slice(i + begin.length, j).trim();
    if (committed === body.trim()) {
      console.log(`  ${doc.padEnd(16)} ${name.padEnd(12)} ${body.split("\n").length} lines match`);
      continue;
    }
    failures++;
    console.log(`  ${doc} ${name}: FAIL`);
    const a = committed.split("\n");
    const b = body.trim().split("\n");
    for (let k = 0; k < Math.max(a.length, b.length); k++) {
      if (a[k] !== b[k]) {
        console.log(`    line ${k + 1}\n      committed: ${a[k] ?? "<end>"}\n      recomputed: ${b[k] ?? "<end>"}`);
        break;
      }
    }
  }
}

const orphans = Object.entries(seen).filter(([, n]) => n === 0).map(([n]) => n);
if (orphans.length > 0) {
  console.log(`  no marker in either document for: ${orphans.join(", ")}  FAIL`);
  failures += orphans.length;
}

if (failures > 0) {
  console.log(`\n${failures} generated regions disagree with the artifacts they claim to come from`);
  exit(1);
}
console.log(`\nJavaScript re-renders all ${compared} generated regions from reports/ and the committed text matches`);
