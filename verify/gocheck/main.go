// Structural validation of every artifact under reports/, plus an independent
// recomputation of the three density control tables.
//
// Everything the README says is read off the files in reports/. Nothing checked
// that those files are well formed. A truncated write, a column that drifted, a
// NaN out of a zero division or a run that is present in one arm and missing in
// the other would all survive to the rendered table looking like a result. This
// walks every CSV and every JSON there and refuses all of it, then recomputes
// density_control.csv, bisected_control.csv and random_split_control.csv from
// their per-run rows.
package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

const tol = 1e-12

// The three control tables, each with the per-run file it is derived from.
var controls = []struct{ summary, runs string }{
	{"density_control.csv", "density_control_runs.csv"},
	{"bisected_control.csv", "bisected_control_runs.csv"},
	{"random_split_control.csv", "random_split_control_runs.csv"},
}

var regimes = []string{"transductive", "density_control", "inductive"}

func readCSV(path string) ([]string, [][]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.FieldsPerRecord = 0 // a ragged file is an error, which is the point
	rows, err := r.ReadAll()
	if err != nil {
		return nil, nil, err
	}
	if len(rows) < 2 {
		return nil, nil, fmt.Errorf("only %d rows", len(rows))
	}
	return rows[0], rows[1:], nil
}

func col(header []string, name string) int {
	for i, h := range header {
		if h == name {
			return i
		}
	}
	return -1
}

// validate reports every structural problem in one file rather than the first,
// so a broken run is diagnosed in a single pass.
func validate(path string) []string {
	var problems []string
	header, rows, err := readCSV(path)
	if err != nil {
		return []string{fmt.Sprintf("unreadable: %v", err)}
	}

	seen := map[string]bool{}
	for i, h := range header {
		// duplicate_definitions.csv has an unnamed index column, which pandas
		// writes and reads back as the row label. Only later columns must be named.
		if h == "" && i > 0 {
			problems = append(problems, fmt.Sprintf("column %d has an empty name", i+1))
		}
		if seen[h] && h != "" {
			problems = append(problems, fmt.Sprintf("duplicate column %q", h))
		}
		seen[h] = true
	}

	for i, row := range rows {
		for j, cell := range row {
			low := strings.ToLower(strings.TrimSpace(cell))
			if low == "nan" || low == "inf" || low == "-inf" || low == "infinity" {
				problems = append(problems,
					fmt.Sprintf("row %d column %s is %s", i+2, header[j], cell))
			}
			// An accuracy that parses but sits outside [0, 1] is not a rounding
			// question, it is a broken run.
			if strings.HasPrefix(header[j], "test_accuracy") || header[j] == "val_accuracy" {
				v, err := strconv.ParseFloat(cell, 64)
				if err != nil {
					problems = append(problems,
						fmt.Sprintf("row %d column %s is not a number: %q", i+2, header[j], cell))
				} else if v < 0 || v > 1 {
					problems = append(problems,
						fmt.Sprintf("row %d column %s is %v, outside [0, 1]", i+2, header[j], v))
				}
			}
		}
	}
	return problems
}

// Every (dataset, model, regime, seed) has to be present exactly once, or a
// paired difference is silently pairing the wrong things.
func validateGrid(path string, want []string) []string {
	var problems []string
	header, rows, err := readCSV(path)
	if err != nil {
		return []string{fmt.Sprintf("unreadable: %v", err)}
	}
	d, m := col(header, "dataset"), col(header, "model")
	g, s := col(header, "regime"), col(header, "seed")
	if d < 0 || m < 0 || g < 0 || s < 0 {
		return []string{"missing one of dataset, model, regime, seed"}
	}

	count := map[string]int{}
	cells := map[string]bool{}
	seeds := map[string]map[string]bool{}
	for _, r := range rows {
		count[strings.Join([]string{r[d], r[m], r[g], r[s]}, "|")]++
		cells[r[d]+"|"+r[m]] = true
		if seeds[r[d]+"|"+r[m]] == nil {
			seeds[r[d]+"|"+r[m]] = map[string]bool{}
		}
		seeds[r[d]+"|"+r[m]][r[s]] = true
	}
	for k, n := range count {
		if n != 1 {
			problems = append(problems, fmt.Sprintf("%s appears %d times", k, n))
		}
	}
	var names []string
	for c := range cells {
		names = append(names, c)
	}
	sort.Strings(names)
	for _, c := range names {
		for _, g := range want {
			for sd := range seeds[c] {
				if count[c+"|"+g+"|"+sd] == 0 {
					problems = append(problems,
						fmt.Sprintf("%s has no %s row for seed %s", c, g, sd))
				}
			}
		}
	}
	return problems
}

// Python's json module writes bare NaN and Infinity, which no other JSON parser
// accepts. Both appear here for real reasons: CiteSeer has no giveaway features
// at all, so the accuracy of a vote over them is undefined, and two duplicate
// definitions read exactly zero on Cora, so the CiteSeer over Cora ratio is
// infinite. Those two keys are allowed to carry a non finite value. Anywhere
// else it would mean a division that was not supposed to happen, so the file is
// checked with the literals swapped out and their positions confirmed first.
var allowedNonFinite = map[string]bool{
	"giveaway_vote_accuracy_on_covered": true,
	"citeseer_over_cora":                true,
}

var nonFinitePattern = regexp.MustCompile(`"([^"]+)"\s*:\s*(-?Infinity|NaN)`)

func validateJSON(path string) ([]string, int) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return []string{err.Error()}, 0
	}
	var problems []string
	hits := nonFinitePattern.FindAllSubmatch(raw, -1)
	for _, h := range hits {
		if !allowedNonFinite[string(h[1])] {
			problems = append(problems,
				fmt.Sprintf("%q is %s, which is not a value this key may take",
					h[1], h[2]))
		}
	}
	cleaned := nonFinitePattern.ReplaceAll(raw, []byte(`"$1": null`))
	var any interface{}
	if err := json.Unmarshal(cleaned, &any); err != nil {
		problems = append(problems, err.Error())
	}
	return problems, len(hits)
}

func mean(v []float64) float64 {
	s := 0.0
	for _, x := range v {
		s += x
	}
	return s / float64(len(v))
}

// Sample standard deviation, n-1, two pass, matching statistics.stdev.
func stdev(v []float64) float64 {
	if len(v) < 2 {
		return 0
	}
	m := mean(v)
	s := 0.0
	for _, x := range v {
		s += (x - m) * (x - m)
	}
	return math.Sqrt(s / float64(len(v)-1))
}

// The repository's resolution rule: two standard errors of the paired component.
func resolved(v []float64) bool {
	m, s := mean(v), stdev(v)
	if s > 0 {
		return math.Abs(m) > 2*s/math.Sqrt(float64(len(v)))
	}
	return m != 0
}

func sub(a, b []float64) []float64 {
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] - b[i]
	}
	return out
}

func recomputeControl(root, summary, runs string) int {
	bad := 0
	header, rows, err := readCSV(filepath.Join(root, "reports", runs))
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s: %v\n", runs, err)
		return 1
	}
	dC, mC := col(header, "dataset"), col(header, "model")
	gC, sC := col(header, "regime"), col(header, "seed")
	aC := col(header, "test_accuracy")
	if dC < 0 || mC < 0 || gC < 0 || sC < 0 || aC < 0 {
		fmt.Fprintf(os.Stderr, "%s: missing a required column\n", runs)
		return 1
	}

	type key struct {
		dataset, model, regime string
		seed                   int
	}
	acc := map[key]float64{}
	seedsOf := map[string]map[int]bool{}
	for _, r := range rows {
		seed, err := strconv.Atoi(r[sC])
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s: seed %q is not an integer\n", runs, r[sC])
			return 1
		}
		v, err := strconv.ParseFloat(r[aC], 64)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s: accuracy %q is not a number\n", runs, r[aC])
			return 1
		}
		acc[key{r[dC], r[mC], r[gC], seed}] = v
		cell := r[dC] + "|" + r[mC]
		if seedsOf[cell] == nil {
			seedsOf[cell] = map[int]bool{}
		}
		seedsOf[cell][seed] = true
	}

	pubHeader, pubRows, err := readCSV(filepath.Join(root, "reports", summary))
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s: %v\n", summary, err)
		return 1
	}
	c := func(name string) int { return col(pubHeader, name) }

	for _, r := range pubRows {
		dataset, model := r[c("dataset")], r[c("model")]
		cell := dataset + "|" + model
		var seeds []int
		for s := range seedsOf[cell] {
			seeds = append(seeds, s)
		}
		sort.Ints(seeds)
		if len(seeds) == 0 {
			fmt.Printf("  %s %s/%s is published but has no runs\n", summary, dataset, model)
			bad++
			continue
		}

		arms := map[string][]float64{}
		for _, g := range regimes {
			for _, s := range seeds {
				v, ok := acc[key{dataset, model, g, s}]
				if !ok {
					fmt.Printf("  %s %s/%s seed %d has no %s row\n", summary, dataset, model, s, g)
					bad++
				}
				arms[g] = append(arms[g], v)
			}
		}
		total := sub(arms["transductive"], arms["inductive"])
		density := sub(arms["transductive"], arms["density_control"])
		specific := sub(arms["density_control"], arms["inductive"])

		want := func(name string) float64 {
			v, _ := strconv.ParseFloat(r[c(name)], 64)
			return v
		}
		wantFlag := func(name string) bool { return r[c(name)] == "True" }

		cellBad := 0
		for _, chk := range []struct {
			name string
			got  float64
		}{
			{"seeds", float64(len(seeds))},
			{"transductive_mean", mean(arms["transductive"])},
			{"density_control_mean", mean(arms["density_control"])},
			{"inductive_mean", mean(arms["inductive"])},
			{"total_inflation_mean", mean(total)},
			{"total_inflation_std", stdev(total)},
			{"density_cost_mean", mean(density)},
			{"density_cost_std", stdev(density)},
			{"test_specific_mean", mean(specific)},
			{"test_specific_std", stdev(specific)},
		} {
			if d := math.Abs(chk.got - want(chk.name)); d > tol {
				fmt.Printf("      %s %s/%s %s got %.17g want %.17g |d| %.1e FAIL\n",
					summary, dataset, model, chk.name, chk.got, want(chk.name), d)
				cellBad++
			}
		}
		for _, chk := range []struct {
			name string
			got  bool
		}{
			{"total_resolved", resolved(total)},
			{"density_resolved", resolved(density)},
			{"test_specific_resolved", resolved(specific)},
		} {
			if chk.got != wantFlag(chk.name) {
				fmt.Printf("      %s %s/%s %s got %v want %v FAIL\n",
					summary, dataset, model, chk.name, chk.got, wantFlag(chk.name))
				cellBad++
			}
		}
		// The decomposition is only meaningful if the two components add back
		// up to the total, which is arithmetic rather than a published claim.
		if d := math.Abs(mean(density) + mean(specific) - mean(total)); d > tol {
			fmt.Printf("      %s %s/%s components do not sum to the total, |d| %.1e FAIL\n",
				summary, dataset, model, d)
			cellBad++
		}
		bad += cellBad
	}
	fmt.Printf("  %-26s %2d rows recomputed from %s\n", summary, len(pubRows), runs)
	return bad
}

func main() {
	root := flag.String("root", ".", "repository root")
	flag.Parse()

	reports := filepath.Join(*root, "reports")
	csvs, err := filepath.Glob(filepath.Join(reports, "*.csv"))
	if err != nil || len(csvs) == 0 {
		fmt.Fprintf(os.Stderr, "no CSVs under %s\n", reports)
		os.Exit(2)
	}
	sort.Strings(csvs)

	bad := 0
	fmt.Printf("validating %d CSV files under reports/\n", len(csvs))
	for _, path := range csvs {
		if problems := validate(path); len(problems) > 0 {
			bad += len(problems)
			for _, p := range problems {
				fmt.Printf("  %s: %s\n", filepath.Base(path), p)
			}
		}
	}
	if bad == 0 {
		fmt.Printf("  no ragged rows, duplicate columns, NaN, Inf or out of range accuracy\n")
	}

	jsons, _ := filepath.Glob(filepath.Join(reports, "*.json"))
	sort.Strings(jsons)
	fmt.Printf("validating %d JSON files under reports/\n", len(jsons))
	nonFinite := 0
	for _, path := range jsons {
		problems, seen := validateJSON(path)
		nonFinite += seen
		bad += len(problems)
		for _, p := range problems {
			fmt.Printf("  %s: %s\n", filepath.Base(path), p)
		}
	}
	if bad == 0 {
		fmt.Printf("  all parse, %d non finite values and all of them at keys where the\n"+
			"  quantity is genuinely undefined\n", nonFinite)
	}

	fmt.Printf("\nchecking the per-run grids are complete\n")
	grids := []struct {
		file string
		want []string
	}{
		{"runs.csv", []string{"transductive", "inductive"}},
		{"density_control_runs.csv", regimes},
		{"bisected_control_runs.csv", regimes},
		{"random_split_control_runs.csv", regimes},
	}
	for _, g := range grids {
		problems := validateGrid(filepath.Join(reports, g.file), g.want)
		bad += len(problems)
		for _, p := range problems {
			fmt.Printf("  %s: %s\n", g.file, p)
		}
		if len(problems) == 0 {
			fmt.Printf("  %-30s every cell has one row per regime per seed\n", g.file)
		}
	}

	fmt.Printf("\nrecomputing the density control tables from their runs\n")
	for _, c := range controls {
		bad += recomputeControl(*root, c.summary, c.runs)
	}

	if bad > 0 {
		fmt.Printf("\n%d problems\n", bad)
		os.Exit(1)
	}
	fmt.Printf("\nGo reproduces the three control tables and reports/ is well formed\n")
}
