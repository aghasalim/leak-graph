/* Recompute reports/inflation.csv from reports/runs.csv, in C.
 *
 * inflation.csv is the headline table: the transductive/inductive gap per
 * dataset and model, its spread, its standard error, and the two standard
 * error resolution flag that decides which cells the README is allowed to
 * quote. All of it comes out of one pandas aggregation in
 * experiments/run_audit.py, and every figure and generated table downstream
 * reads that aggregation rather than the runs it came from.
 *
 * This is a second implementation from the per-run rows. Columns are resolved
 * by name, so a column added or reordered upstream cannot silently shift what
 * is read. Exits non-zero on the first disagreement past the tolerance.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LINE 4096
#define MAX_CELLS 64
#define MAX_SEEDS 64
#define TOL 1e-12

/* One (dataset, model) cell, with the arms held per seed so the difference can
 * be taken pairwise before anything is averaged. */
typedef struct {
    char dataset[64], model[64];
    int seeds;
    int seed_id[MAX_SEEDS];
    double trans[MAX_SEEDS], ind[MAX_SEEDS];
    double trans_dedup[MAX_SEEDS], ind_dedup[MAX_SEEDS];
    double ind_covered[MAX_SEEDS];
    int have_trans[MAX_SEEDS], have_ind[MAX_SEEDS];
} Cell;

static Cell cells[MAX_CELLS];
static int n_cells;

static int column_of(const char *header, const char *name)
{
    char buf[LINE];
    strncpy(buf, header, sizeof buf - 1);
    buf[sizeof buf - 1] = '\0';
    int i = 0;
    for (char *tok = strtok(buf, ",\r\n"); tok; tok = strtok(NULL, ",\r\n"), i++)
        if (strcmp(tok, name) == 0)
            return i;
    return -1;
}

static const char *field(const char *line, int index)
{
    static char out[256];
    int col = 0;
    const char *p = line;
    while (col < index) {
        p = strchr(p, ',');
        if (!p)
            return "";
        p++;
        col++;
    }
    const char *end = strchr(p, ',');
    size_t n = end ? (size_t)(end - p) : strlen(p);
    if (n >= sizeof out)
        n = sizeof out - 1;
    memcpy(out, p, n);
    out[n] = '\0';
    char *nl = strpbrk(out, "\r\n");
    if (nl)
        *nl = '\0';
    return out;
}

static double mean_of(const double *v, int n)
{
    double s = 0.0;
    for (int i = 0; i < n; i++)
        s += v[i];
    return s / n;
}

/* Sample standard deviation, n-1, two pass. statistics.stdev is also two pass,
 * so the two agree to the last bit rather than merely to a tolerance. */
static double stdev_of(const double *v, int n)
{
    if (n < 2)
        return 0.0;
    const double m = mean_of(v, n);
    double s = 0.0;
    for (int i = 0; i < n; i++)
        s += (v[i] - m) * (v[i] - m);
    return sqrt(s / (n - 1));
}

static Cell *cell_for(const char *dataset, const char *model)
{
    for (int i = 0; i < n_cells; i++)
        if (strcmp(cells[i].dataset, dataset) == 0 && strcmp(cells[i].model, model) == 0)
            return &cells[i];
    if (n_cells >= MAX_CELLS) {
        fprintf(stderr, "more than %d cells\n", MAX_CELLS);
        exit(2);
    }
    Cell *c = &cells[n_cells++];
    snprintf(c->dataset, sizeof c->dataset, "%s", dataset);
    snprintf(c->model, sizeof c->model, "%s", model);
    return c;
}

static int slot_for(Cell *c, int seed)
{
    for (int i = 0; i < c->seeds; i++)
        if (c->seed_id[i] == seed)
            return i;
    if (c->seeds >= MAX_SEEDS) {
        fprintf(stderr, "more than %d seeds in %s/%s\n", MAX_SEEDS, c->dataset, c->model);
        exit(2);
    }
    c->seed_id[c->seeds] = seed;
    return c->seeds++;
}

static int check(const char *what, double got, double want, int *failures)
{
    const double d = fabs(got - want);
    if (d > TOL) {
        printf("      %-28s got %+.17g want %+.17g  |d| %.1e  FAIL\n",
               what, got, want, d);
        (*failures)++;
        return 1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    const char *root = argc > 1 ? argv[1] : ".";
    char path[1024], line[LINE], header[LINE];

    snprintf(path, sizeof path, "%s/reports/runs.csv", root);
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }
    if (!fgets(header, sizeof header, f)) { fclose(f); return 2; }

    const char *needed[] = { "dataset", "model", "regime", "seed", "test_accuracy",
                             "test_accuracy_dedup", "test_accuracy_nbr_covered" };
    int c_of[7];
    for (int i = 0; i < 7; i++) {
        c_of[i] = column_of(header, needed[i]);
        if (c_of[i] < 0) {
            fprintf(stderr, "runs.csv has no %s column\n", needed[i]);
            fclose(f);
            return 2;
        }
    }

    long rows = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '\n' || line[0] == '\r' || line[0] == '\0')
            continue;
        char dataset[64], model[64], regime[64];
        snprintf(dataset, sizeof dataset, "%s", field(line, c_of[0]));
        snprintf(model, sizeof model, "%s", field(line, c_of[1]));
        snprintf(regime, sizeof regime, "%s", field(line, c_of[2]));
        const int seed = atoi(field(line, c_of[3]));
        const double acc = atof(field(line, c_of[4]));
        const double dedup = atof(field(line, c_of[5]));
        const double covered = atof(field(line, c_of[6]));

        Cell *c = cell_for(dataset, model);
        const int k = slot_for(c, seed);
        if (strcmp(regime, "transductive") == 0) {
            c->trans[k] = acc;
            c->trans_dedup[k] = dedup;
            c->have_trans[k] = 1;
        } else if (strcmp(regime, "inductive") == 0) {
            c->ind[k] = acc;
            c->ind_dedup[k] = dedup;
            c->ind_covered[k] = covered;
            c->have_ind[k] = 1;
        } else {
            fprintf(stderr, "unexpected regime %s in runs.csv\n", regime);
            fclose(f);
            return 2;
        }
        rows++;
    }
    fclose(f);
    printf("read %ld runs, %d (dataset, model) cells\n", rows, n_cells);

    for (int i = 0; i < n_cells; i++)
        for (int k = 0; k < cells[i].seeds; k++)
            if (!cells[i].have_trans[k] || !cells[i].have_ind[k]) {
                fprintf(stderr, "%s/%s seed %d is missing an arm\n",
                        cells[i].dataset, cells[i].model, cells[i].seed_id[k]);
                return 2;
            }

    snprintf(path, sizeof path, "%s/reports/inflation.csv", root);
    f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }
    if (!fgets(header, sizeof header, f)) { fclose(f); return 2; }

    const char *stats[] = {
        "dataset", "model", "seeds", "transductive_mean", "transductive_std",
        "inductive_mean", "inductive_std", "inductive_nbr_covered_mean",
        "inductive_nbr_covered_std", "inflation_mean", "inflation_std",
        "inflation_stderr", "inflation_dedup_mean", "inflation_dedup_std",
        "duplicate_component", "resolved"
    };
    const int n_stats = (int)(sizeof stats / sizeof stats[0]);
    int col[16];
    for (int i = 0; i < n_stats; i++) {
        col[i] = column_of(header, stats[i]);
        if (col[i] < 0) {
            fprintf(stderr, "inflation.csv has no %s column\n", stats[i]);
            fclose(f);
            return 2;
        }
    }

    int failures = 0, checked = 0, matched[MAX_CELLS] = {0};
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '\n' || line[0] == '\r' || line[0] == '\0')
            continue;
        char dataset[64], model[64];
        snprintf(dataset, sizeof dataset, "%s", field(line, col[0]));
        snprintf(model, sizeof model, "%s", field(line, col[1]));

        int idx = -1;
        for (int i = 0; i < n_cells; i++)
            if (strcmp(cells[i].dataset, dataset) == 0 && strcmp(cells[i].model, model) == 0)
                idx = i;
        if (idx < 0) {
            printf("  %s/%s is published but has no runs\n", dataset, model);
            failures++;
            continue;
        }
        matched[idx] = 1;
        Cell *c = &cells[idx];
        const int n = c->seeds;

        double paired[MAX_SEEDS], paired_dedup[MAX_SEEDS];
        for (int k = 0; k < n; k++) {
            paired[k] = c->trans[k] - c->ind[k];
            paired_dedup[k] = c->trans_dedup[k] - c->ind_dedup[k];
        }
        const double p_m = mean_of(paired, n), p_s = stdev_of(paired, n);
        const double pd_m = mean_of(paired_dedup, n), pd_s = stdev_of(paired_dedup, n);
        const double stderr_ = p_s / sqrt((double)n);
        /* Two standard errors of the paired difference. A cell with no spread
         * resolves only if its mean is non-zero, which is how the graph-free
         * models come out unresolved rather than infinitely resolved. */
        const double resolved = p_s > 0.0 ? (fabs(p_m) > 2.0 * stderr_ ? 1.0 : 0.0)
                                          : (p_m != 0.0 ? 1.0 : 0.0);

        int bad = 0;
        bad |= check("seeds", (double)n, atof(field(line, col[2])), &failures);
        bad |= check("transductive_mean", mean_of(c->trans, n), atof(field(line, col[3])), &failures);
        bad |= check("transductive_std", stdev_of(c->trans, n), atof(field(line, col[4])), &failures);
        bad |= check("inductive_mean", mean_of(c->ind, n), atof(field(line, col[5])), &failures);
        bad |= check("inductive_std", stdev_of(c->ind, n), atof(field(line, col[6])), &failures);
        bad |= check("inductive_nbr_covered_mean", mean_of(c->ind_covered, n),
                     atof(field(line, col[7])), &failures);
        bad |= check("inductive_nbr_covered_std", stdev_of(c->ind_covered, n),
                     atof(field(line, col[8])), &failures);
        bad |= check("inflation_mean", p_m, atof(field(line, col[9])), &failures);
        bad |= check("inflation_std", p_s, atof(field(line, col[10])), &failures);
        bad |= check("inflation_stderr", stderr_, atof(field(line, col[11])), &failures);
        bad |= check("inflation_dedup_mean", pd_m, atof(field(line, col[12])), &failures);
        bad |= check("inflation_dedup_std", pd_s, atof(field(line, col[13])), &failures);
        bad |= check("duplicate_component", p_m - pd_m, atof(field(line, col[14])), &failures);

        const char *flag = field(line, col[15]);
        const double want_flag = strcmp(flag, "True") == 0 ? 1.0 : 0.0;
        if (strcmp(flag, "True") != 0 && strcmp(flag, "False") != 0) {
            printf("      resolved is %s, not True or False  FAIL\n", flag);
            failures++;
            bad = 1;
        } else if (resolved != want_flag) {
            printf("      resolved got %s want %s  FAIL\n",
                   resolved > 0 ? "True" : "False", flag);
            failures++;
            bad = 1;
        }

        printf("  %-10s %-10s n=%2d  inflation %+.6f +- %.6f  se %.6f  %s  %s\n",
               dataset, model, n, p_m, p_s, stderr_,
               resolved > 0 ? "resolved " : "unresolved", bad ? "FAIL" : "ok");
        checked++;
    }
    fclose(f);

    for (int i = 0; i < n_cells; i++)
        if (!matched[i]) {
            printf("  %s/%s has runs but no published row\n", cells[i].dataset, cells[i].model);
            failures++;
        }

    if (checked != n_cells || failures) {
        printf("\n%d disagreements over %d published rows and %d cells of runs\n",
               failures, checked, n_cells);
        return 1;
    }
    printf("\nC reproduces all %d rows of reports/inflation.csv from reports/runs.csv,\n"
           "13 statistics and the resolution flag each, tolerance %.0e\n", checked, TOL);
    return 0;
}
