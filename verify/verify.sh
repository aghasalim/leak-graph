#!/usr/bin/env bash
# Recompute what this repository publishes, in every language installed here,
# and require the answers to agree.
#
# The chain of evidence runs raw runs -> summary CSV -> generated table ->
# sentence in the README, and until now every link of it was made by the same
# pandas code. If the aggregation in experiments/run_audit.py were wrong, the
# summary, the figures, the generated tables and the README would all be wrong
# together and consistently, and nothing would notice.
#
# Each check below re-derives one link from the artifact behind it, in a
# language that shares no code with the others:
#
#   SQL         reports/*_runs.csv -> all four summary tables, 956 statistics
#   C           reports/runs.csv   -> reports/inflation.csv, resolution flag included
#   Go          reports/           -> structure, grids, and the three control tables
#   R           the resolution rule against a t test and a bootstrap
#   Rust        the resolution rule against an exact enumeration of every sign flip
#   JavaScript  summary CSVs -> the generated tables committed in the docs
#   Ruby        summary CSVs -> the hand written figures in the README prose
#   Java        the four experiment files are the experiments the README describes
#
# Anything whose toolchain is missing is skipped with a message rather than
# quietly passing. CI has all of them.
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

pass=0 fail=0 skip=0

run () {
    local name="$1" tool="$2"; shift 2
    printf '\n=== %s ===\n' "$name"
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'skipped: %s is not installed\n' "$tool"
        skip=$((skip + 1)); return
    fi
    if "$@"; then pass=$((pass + 1)); else fail=$((fail + 1)); fi
}

# Melt one published summary CSV into file,dataset,model,statistic,value, which
# is the shape verify/summaries.sql emits. Booleans become 1 and 0; the status
# column is prose and is not a statistic.
melt () {
    awk -F, -v file="$1" '
        NR == 1 {
            for (i = 1; i <= NF; i++) {
                h[i] = $i
                if ($i == "dataset") d = i
                if ($i == "model")   m = i
            }
            next
        }
        /"/ { print "quoted field in " file ", this melt cannot parse it" > "/dev/stderr"; exit 3 }
        NF > 1 {
            for (i = 1; i <= NF; i++) {
                if (h[i] == "dataset" || h[i] == "model" || h[i] == "status") continue
                v = $i
                if (v == "True")  v = 1
                if (v == "False") v = 0
                print file "," $d "," $m "," h[i] "," v
            }
        }' "reports/$1"
}

check_sql () {
    for f in inflation.csv density_control.csv bisected_control.csv random_split_control.csv; do
        melt "$f" || return 1
    done > "$tmp/published.csv"
    sqlite3 -init verify/summaries.sql :memory: "" > "$tmp/sql.csv" 2>/dev/null || return 1

    awk -F, -v tol=1e-12 '
        NR == FNR { want[$1 "," $2 "," $3 "," $4] = $5; seen[$1 "," $2 "," $3 "," $4] = 0; next }
        {
            k = $1 "," $2 "," $3 "," $4
            if (!(k in want)) { print "  SQL produced " k ", which is not published"; bad++; next }
            seen[k] = 1
            d = $5 - want[k]; if (d < 0) d = -d
            if (d > worst) { worst = d; at = k }
            if (d > tol) { printf "  %s: SQL %.17g, published %.17g, |d| %.1e\n", k, $5, want[k], d; bad++ }
            n++
        }
        END {
            for (k in seen) if (!seen[k]) { print "  " k " is published but SQL did not produce it"; bad++ }
            printf "SQL recomputes %d statistics from the per run rows, worst disagreement %.1e at %s\n", n, worst, at
            if (bad) { printf "%d disagreements\n", bad; exit 1 }
        }' "$tmp/published.csv" "$tmp/sql.csv"
}

check_c () {
    cc -std=c99 -O2 -Wall -Wextra -Wpedantic -Werror \
       -o "$tmp/inflation" verify/inflation.c -lm || return 1
    "$tmp/inflation" "$root"
}

check_go () { ( cd verify/gocheck && go run . -root "$root" ); }

check_rust () { ( cd verify/permute && cargo run --release --quiet -- "$root" ); }

run "SQL, all four summary tables"   sqlite3 check_sql
run "C, the headline table"          cc      check_c
run "Go, structure and the controls" go      check_go
run "R, the resolution rule"         Rscript Rscript verify/verify.R "$root"
run "Rust, exact sign flip test"     cargo   check_rust
run "JavaScript, the generated docs" node    node verify/tables.mjs "$root"
run "Ruby, the README prose"         ruby    ruby verify/claims.rb "$root"
run "Java, the experiment files"     java    java verify/Protocol.java "$root"

printf '\n%s\n' "----------------------------------------"
printf '%d passed, %d failed, %d skipped\n' "$pass" "$fail" "$skip"
[ "$fail" -eq 0 ] || exit 1
[ "$pass" -gt 0 ] || { echo "nothing ran"; exit 1; }
