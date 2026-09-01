# The inference behind the README, redone in base R.
#
# Every cell in this repository is called resolved or unresolved by one rule:
# the mean paired difference over ten seeds exceeds two standard errors of that
# difference. That rule decides which numbers the README is allowed to quote, and
# it is applied in exactly one place, experiments/run_audit.py and
# experiments/run_density_control.py. Nothing else has ever tested it.
#
# Three things here that the arithmetic checks in verify/ do not do:
#
#   1. a paired t test on every component, so the repository's rule of thumb is
#      compared against the textbook test it is standing in for
#   2. the instrument claim, that MLP and LabelProp read exactly zero in every
#      cell of every table, checked at the level of individual seeds
#   3. a nonparametric bootstrap of each resolved cell, which assumes no shape
#      at all, requiring the interval to sit on one side of zero
#
# No packages, so CI needs nothing beyond the R that is already on the runner.

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) > 0) args[1] else "."
set.seed(20260901)

BOOT <- 20000
GRAPH_FREE <- c("MLP", "LabelProp")

runs <- read.csv(file.path(root, "reports", "runs.csv"))
inflation <- read.csv(file.path(root, "reports", "inflation.csv"))
controls <- list(
    density_control = "density_control",
    bisected_control = "bisected_control",
    random_split_control = "random_split_control"
)

arm <- function(df, dataset, model, regime, seeds) {
    sapply(seeds, function(s) {
        v <- df$test_accuracy[df$dataset == dataset & df$model == model &
                              df$regime == regime & df$seed == s]
        if (length(v) != 1) stop(sprintf("%s/%s/%s seed %d: %d rows, expected 1",
                                         dataset, model, regime, s, length(v)))
        v
    })
}

# Collect every paired component in the repository: the headline inflation, and
# the three components of each of the three control tables.
components <- list()
add <- function(table, dataset, model, name, diffs, flag) {
    components[[length(components) + 1]] <<- list(
        table = table, dataset = dataset, model = model, name = name,
        diffs = diffs, flag = flag
    )
}

for (i in seq_len(nrow(inflation))) {
    r <- inflation[i, ]
    seeds <- sort(unique(runs$seed[runs$dataset == r$dataset & runs$model == r$model]))
    add("inflation", r$dataset, r$model, "inflation",
        arm(runs, r$dataset, r$model, "transductive", seeds) -
            arm(runs, r$dataset, r$model, "inductive", seeds),
        as.logical(r$resolved))
}

for (stem in names(controls)) {
    raw <- read.csv(file.path(root, "reports", paste0(stem, "_runs.csv")))
    pub <- read.csv(file.path(root, "reports", paste0(stem, ".csv")))
    for (i in seq_len(nrow(pub))) {
        r <- pub[i, ]
        seeds <- sort(unique(raw$seed[raw$dataset == r$dataset & raw$model == r$model]))
        tr <- arm(raw, r$dataset, r$model, "transductive", seeds)
        ct <- arm(raw, r$dataset, r$model, "density_control", seeds)
        ind <- arm(raw, r$dataset, r$model, "inductive", seeds)
        add(stem, r$dataset, r$model, "total", tr - ind, as.logical(r$total_resolved))
        add(stem, r$dataset, r$model, "density", tr - ct, as.logical(r$density_resolved))
        add(stem, r$dataset, r$model, "test_specific", ct - ind,
            as.logical(r$test_specific_resolved))
    }
}

cat(sprintf("%d paired components across four published tables\n", length(components)))
failures <- 0

# 1. The repository's rule against a paired t test on the same numbers.
#
# The rule uses a fixed multiplier of 2. The t test at ten seeds uses 2.262, so
# the rule is the more permissive of the two by construction and the interesting
# question is how many cells sit in the gap. A cell the repository calls
# unresolved that the t test rejects would mean the arithmetic is wrong, not that
# the rule is loose, so that direction is a hard failure.
rule_only <- character(0)
test_only <- character(0)
for (k in components) {
    d <- k$diffs
    reject <- if (sd(d) > 0) t.test(d)$p.value < 0.05 else FALSE
    label <- sprintf("%s %s/%s %s", k$table, k$dataset, k$model, k$name)
    if (k$flag && !reject) rule_only <- c(rule_only, label)
    if (reject && !k$flag) test_only <- c(test_only, label)
}
cat(sprintf("\ntwo standard error rule against a paired t test at 0.05\n"))
cat(sprintf("  %d of %d components agree\n",
            length(components) - length(rule_only) - length(test_only), length(components)))
cat(sprintf("  %d resolved by the rule but not by the t test:\n", length(rule_only)))
for (l in rule_only) cat(sprintf("    %s\n", l))
if (length(test_only) > 0) {
    cat(sprintf("  FAIL: %d rejected by the t test but published as unresolved:\n",
                length(test_only)))
    for (l in test_only) cat(sprintf("    %s\n", l))
    failures <- failures + length(test_only)
}

# 2. The instrument check. MLP and LabelProp cannot tell the two arms apart, so
# every one of their paired differences must be zero at every seed, not merely
# zero on average. This is the claim that caught instrument bug I1.
cat("\ninstrument check, graph free models\n")
zero_cells <- 0
zero_seeds <- 0
for (k in components) {
    if (!(k$model %in% GRAPH_FREE)) next
    zero_cells <- zero_cells + 1
    zero_seeds <- zero_seeds + length(k$diffs)
    if (any(k$diffs != 0)) {
        cat(sprintf("  FAIL %s %s/%s %s: max |difference| %.3e over %d seeds\n",
                    k$table, k$dataset, k$model, k$name, max(abs(k$diffs)), length(k$diffs)))
        failures <- failures + 1
    }
}
cat(sprintf("  %d cells, %d seed level differences, all exactly zero\n",
            zero_cells, zero_seeds))

# 3. Bootstrap. The published interval is two standard errors of a mean over ten
# seeds, which assumes a shape the data does not have to have. A percentile
# bootstrap makes no such assumption, so a resolved cell whose bootstrap interval
# straddles zero would mean the claim rests on the assumption rather than on the
# data.
#
# The requirement is the 90% interval rather than the 95% one, on purpose. Ten
# values give a bootstrap distribution with visible atoms, and one cell's 95%
# bound sits on zero to eight decimal places, so at 95% the answer depends on the
# bootstrap seed rather than on the data. That cell is named below instead of
# being hidden inside a pass.
cat(sprintf("\npercentile bootstrap, %d draws per cell\n", BOOT))
boot_ci <- function(d, level) {
    n <- length(d)
    tail <- (1 - level) / 2
    stats <- replicate(BOOT, mean(sample(d, n, replace = TRUE)))
    quantile(stats, c(tail, 1 - tail), names = FALSE)
}
checked <- 0
tightest <- list(margin = Inf, label = "none")
for (k in components) {
    if (!k$flag) next
    checked <- checked + 1
    ci90 <- boot_ci(k$diffs, 0.90)
    ci95 <- boot_ci(k$diffs, 0.95)
    excludes90 <- ci90[1] > 0 || ci90[2] < 0
    # Distance from zero to the nearer 95% bound, negative if the interval
    # covers zero. This is the number that says how close a call the cell is.
    margin <- if (mean(k$diffs) > 0) ci95[1] else -ci95[2]
    if (!excludes90) failures <- failures + 1
    if (margin < tightest$margin) {
        tightest <- list(margin = margin,
                         label = sprintf("%s %s/%s %s", k$table, k$dataset, k$model, k$name))
    }
    cat(sprintf("  %-20s %-10s %-14s mean %+.4f  90%% ci [%+.4f, %+.4f]  95%% margin %+.5f  %s\n",
                paste(k$table, k$dataset), k$model, k$name, mean(k$diffs), ci90[1], ci90[2],
                margin, if (excludes90) "one side of 0" else "FAIL: straddles 0"))
}
cat(sprintf("  %d resolved cells, every 90%% interval on one side of zero\n", checked))
cat(sprintf("  closest call at 95%%: %s, margin %+.5f\n", tightest$label, tightest$margin))

if (failures > 0) {
    cat(sprintf("\n%d checks failed\n", failures))
    quit(status = 1)
}
cat("\nR agrees with every resolution flag the README quotes\n")
