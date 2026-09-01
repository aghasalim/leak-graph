-- Recompute every published summary table from the per-run rows, in SQL.
--
-- reports/inflation.csv, density_control.csv, bisected_control.csv and
-- random_split_control.csv are all produced by the same pandas/statistics code
-- in experiments/. Each one is a group-by over a *_runs.csv file in the same
-- directory, and nothing checked that the group-by was right: the README, the
-- figures and the generated tables all read the summary, not the runs.
--
-- This derives all four from the raw rows with nothing but SQL. Output is long
-- format, one line per (file, dataset, model, statistic), and verify/verify.sh
-- melts the published CSVs the same way and diffs the two.
--
-- Run: sqlite3 -init verify/summaries.sql :memory: ""

.mode csv
.headers off
.import --csv reports/runs.csv runs
.import --csv reports/density_control_runs.csv density_runs
.import --csv reports/bisected_control_runs.csv bisected_runs
.import --csv reports/random_split_control_runs.csv random_runs

-- One row per (dataset, model, seed) with the arms side by side. Pairing on the
-- seed is the whole point of the metric: the two arms share initialisation, so
-- the difference is taken per seed and only then averaged.
CREATE TEMP VIEW paired AS
    SELECT t.dataset, t.model, t.seed,
           CAST(t.test_accuracy AS REAL)               AS trans,
           CAST(i.test_accuracy AS REAL)               AS ind,
           CAST(t.test_accuracy_dedup AS REAL)         AS trans_dedup,
           CAST(i.test_accuracy_dedup AS REAL)         AS ind_dedup,
           CAST(i.test_accuracy_nbr_covered AS REAL)   AS ind_covered
    FROM runs t
    JOIN runs i ON i.dataset = t.dataset AND i.model = t.model AND i.seed = t.seed
    WHERE t.regime = 'transductive' AND i.regime = 'inductive';

-- Sample standard deviation, n-1 in the denominator, which is what
-- statistics.stdev gives. Written out rather than hidden in a helper because
-- SQLite has no user defined aggregate here and the naive sum-of-squares form
-- is the thing being checked.
CREATE TEMP VIEW inflation AS
    SELECT dataset, model, COUNT(*) AS n,
           AVG(trans) AS t_m, AVG(ind) AS i_m, AVG(ind_covered) AS c_m,
           AVG(trans - ind) AS p_m, AVG(trans_dedup - ind_dedup) AS pd_m,
           SQRT((SUM(trans * trans) - COUNT(*) * AVG(trans) * AVG(trans))
                / (COUNT(*) - 1.0)) AS t_s,
           SQRT((SUM(ind * ind) - COUNT(*) * AVG(ind) * AVG(ind))
                / (COUNT(*) - 1.0)) AS i_s,
           SQRT((SUM(ind_covered * ind_covered)
                 - COUNT(*) * AVG(ind_covered) * AVG(ind_covered))
                / (COUNT(*) - 1.0)) AS c_s,
           SQRT((SUM((trans - ind) * (trans - ind))
                 - COUNT(*) * AVG(trans - ind) * AVG(trans - ind))
                / (COUNT(*) - 1.0)) AS p_s,
           SQRT((SUM((trans_dedup - ind_dedup) * (trans_dedup - ind_dedup))
                 - COUNT(*) * AVG(trans_dedup - ind_dedup)
                   * AVG(trans_dedup - ind_dedup))
                / (COUNT(*) - 1.0)) AS pd_s
    FROM paired GROUP BY dataset, model;

SELECT 'inflation.csv', dataset, model, stat, value FROM (
    SELECT dataset, model, 'seeds' AS stat, CAST(n AS REAL) AS value FROM inflation
    UNION ALL SELECT dataset, model, 'transductive_mean', t_m FROM inflation
    UNION ALL SELECT dataset, model, 'transductive_std', t_s FROM inflation
    UNION ALL SELECT dataset, model, 'inductive_mean', i_m FROM inflation
    UNION ALL SELECT dataset, model, 'inductive_std', i_s FROM inflation
    UNION ALL SELECT dataset, model, 'inductive_nbr_covered_mean', c_m FROM inflation
    UNION ALL SELECT dataset, model, 'inductive_nbr_covered_std', c_s FROM inflation
    UNION ALL SELECT dataset, model, 'inflation_mean', p_m FROM inflation
    UNION ALL SELECT dataset, model, 'inflation_std', p_s FROM inflation
    UNION ALL SELECT dataset, model, 'inflation_stderr', p_s / SQRT(n) FROM inflation
    UNION ALL SELECT dataset, model, 'inflation_dedup_mean', pd_m FROM inflation
    UNION ALL SELECT dataset, model, 'inflation_dedup_std', pd_s FROM inflation
    UNION ALL SELECT dataset, model, 'duplicate_component', p_m - pd_m FROM inflation
    -- The resolution rule: two standard errors of the paired difference. A cell
    -- with no spread at all is resolved only if its mean is non-zero, which is
    -- how the graph-free controls come out unresolved rather than infinitely so.
    UNION ALL SELECT dataset, model, 'resolved',
        CASE WHEN p_s > 0 THEN (CASE WHEN ABS(p_m) > 2 * p_s / SQRT(n) THEN 1.0 ELSE 0.0 END)
             ELSE (CASE WHEN p_m <> 0 THEN 1.0 ELSE 0.0 END) END
        FROM inflation
);

-- The three control tables share a shape: three arms, three paired components.
CREATE TEMP VIEW control_rows AS
    SELECT 'density_control.csv' AS file, * FROM density_runs
    UNION ALL SELECT 'bisected_control.csv', * FROM bisected_runs
    UNION ALL SELECT 'random_split_control.csv', * FROM random_runs;

CREATE TEMP VIEW control_paired AS
    SELECT t.file, t.dataset, t.model, t.seed,
           CAST(t.test_accuracy AS REAL) AS trans,
           CAST(c.test_accuracy AS REAL) AS ctl,
           CAST(i.test_accuracy AS REAL) AS ind
    FROM control_rows t
    JOIN control_rows c ON c.file = t.file AND c.dataset = t.dataset
                       AND c.model = t.model AND c.seed = t.seed
    JOIN control_rows i ON i.file = t.file AND i.dataset = t.dataset
                       AND i.model = t.model AND i.seed = t.seed
    WHERE t.regime = 'transductive' AND c.regime = 'density_control'
      AND i.regime = 'inductive';

CREATE TEMP VIEW control AS
    SELECT file, dataset, model, COUNT(*) AS n,
           AVG(trans) AS t_m, AVG(ctl) AS c_m, AVG(ind) AS i_m,
           AVG(trans - ind) AS tot_m, AVG(trans - ctl) AS den_m, AVG(ctl - ind) AS spe_m,
           SQRT((SUM((trans - ind) * (trans - ind))
                 - COUNT(*) * AVG(trans - ind) * AVG(trans - ind))
                / (COUNT(*) - 1.0)) AS tot_s,
           SQRT((SUM((trans - ctl) * (trans - ctl))
                 - COUNT(*) * AVG(trans - ctl) * AVG(trans - ctl))
                / (COUNT(*) - 1.0)) AS den_s,
           SQRT((SUM((ctl - ind) * (ctl - ind))
                 - COUNT(*) * AVG(ctl - ind) * AVG(ctl - ind))
                / (COUNT(*) - 1.0)) AS spe_s
    FROM control_paired GROUP BY file, dataset, model;

SELECT file, dataset, model, stat, value FROM (
    SELECT file, dataset, model, 'seeds' AS stat, CAST(n AS REAL) AS value FROM control
    UNION ALL SELECT file, dataset, model, 'transductive_mean', t_m FROM control
    UNION ALL SELECT file, dataset, model, 'density_control_mean', c_m FROM control
    UNION ALL SELECT file, dataset, model, 'inductive_mean', i_m FROM control
    UNION ALL SELECT file, dataset, model, 'total_inflation_mean', tot_m FROM control
    UNION ALL SELECT file, dataset, model, 'total_inflation_std', tot_s FROM control
    UNION ALL SELECT file, dataset, model, 'density_cost_mean', den_m FROM control
    UNION ALL SELECT file, dataset, model, 'density_cost_std', den_s FROM control
    UNION ALL SELECT file, dataset, model, 'test_specific_mean', spe_m FROM control
    UNION ALL SELECT file, dataset, model, 'test_specific_std', spe_s FROM control
    UNION ALL SELECT file, dataset, model, 'total_resolved',
        CASE WHEN tot_s > 0 THEN (CASE WHEN ABS(tot_m) > 2 * tot_s / SQRT(n) THEN 1.0 ELSE 0.0 END)
             ELSE (CASE WHEN tot_m <> 0 THEN 1.0 ELSE 0.0 END) END FROM control
    UNION ALL SELECT file, dataset, model, 'density_resolved',
        CASE WHEN den_s > 0 THEN (CASE WHEN ABS(den_m) > 2 * den_s / SQRT(n) THEN 1.0 ELSE 0.0 END)
             ELSE (CASE WHEN den_m <> 0 THEN 1.0 ELSE 0.0 END) END FROM control
    UNION ALL SELECT file, dataset, model, 'test_specific_resolved',
        CASE WHEN spe_s > 0 THEN (CASE WHEN ABS(spe_m) > 2 * spe_s / SQRT(n) THEN 1.0 ELSE 0.0 END)
             ELSE (CASE WHEN spe_m <> 0 THEN 1.0 ELSE 0.0 END) END FROM control
);
