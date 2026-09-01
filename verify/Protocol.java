// Do the four experiment files describe the protocols the README says they do?
//
// Everything else in verify/ checks arithmetic: a summary against the runs it
// aggregates, a rendered table against the summary. None of it can tell whether
// the runs themselves are the runs being claimed. Three of these facts are
// stated in the README and in the docstrings of experiments/, and none was ever
// asserted anywhere:
//
//   1. The density control is the headline experiment with a third arm added.
//      Same splits, same seeds, same initialisation, same epoch budget, so its
//      transductive and inductive arms must be the SAME NUMBERS as runs.csv,
//      bit for bit, not merely close. If they drifted, the decomposition would
//      be splitting a gap that the headline table never measured.
//
//   2. The bisected and random split controls are different protocols, not the
//      same one relabelled. They must NOT reproduce runs.csv.
//
//   3. MLP and LabelProp cannot tell the arms apart, and LabelProp has no
//      parameters at all, so it cannot differ between seeds on one split. This
//      is the instrument check that caught the dropout RNG bug in Finding I1.
//
// Run with the single file launcher, so there is nothing to compile or ignore:
//
//   java verify/Protocol.java [root]

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

public class Protocol {

    /** One row of a *_runs.csv, kept as text so comparisons are bit for bit. */
    record Run(String dataset, String model, String regime, String seed, Map<String, String> fields) {
        String key() {
            return dataset + "|" + model + "|" + regime + "|" + seed;
        }
    }

    static final String[] GRAPH_FREE = {"MLP", "LabelProp"};
    /** The epoch budget the README states, and the patience floor early stopping allows. */
    static final int MAX_EPOCHS = 300;

    static int failures = 0;

    static void fail(String message) {
        System.out.println("  FAIL " + message);
        failures++;
    }

    static List<Run> read(Path path) throws IOException {
        List<String> lines = Files.readAllLines(path);
        if (lines.size() < 2) {
            throw new IOException(path + " has " + lines.size() + " lines");
        }
        String[] header = lines.get(0).split(",", -1);
        List<Run> out = new ArrayList<>();
        for (int i = 1; i < lines.size(); i++) {
            if (lines.get(i).isBlank()) {
                continue;
            }
            String[] cells = lines.get(i).split(",", -1);
            if (cells.length != header.length) {
                throw new IOException(path + " row " + (i + 1) + " has " + cells.length
                        + " cells, header has " + header.length);
            }
            Map<String, String> fields = new HashMap<>();
            for (int j = 0; j < header.length; j++) {
                fields.put(header[j], cells[j]);
            }
            for (String needed : new String[] {"dataset", "model", "regime", "seed"}) {
                if (!fields.containsKey(needed)) {
                    throw new IOException(path + " has no " + needed + " column");
                }
            }
            out.add(new Run(fields.get("dataset"), fields.get("model"),
                    fields.get("regime"), fields.get("seed"), fields));
        }
        return out;
    }

    static Map<String, Run> index(List<Run> runs) {
        Map<String, Run> out = new HashMap<>();
        for (Run r : runs) {
            if (out.put(r.key(), r) != null) {
                fail(r.key() + " appears more than once");
            }
        }
        return out;
    }

    /** How many rows two files share a key with, and how many of those are identical. */
    static int[] overlap(Map<String, Run> a, Map<String, Run> b, String[] columns) {
        int shared = 0;
        int identical = 0;
        for (Map.Entry<String, Run> e : b.entrySet()) {
            Run other = a.get(e.getKey());
            if (other == null) {
                continue;
            }
            shared++;
            boolean same = true;
            for (String c : columns) {
                if (!other.fields().get(c).equals(e.getValue().fields().get(c))) {
                    same = false;
                }
            }
            if (same) {
                identical++;
            }
        }
        return new int[] {shared, identical};
    }

    public static void main(String[] args) throws IOException {
        String root = args.length > 0 ? args[0] : ".";
        Path reports = Path.of(root, "reports");

        List<Run> headline = read(reports.resolve("runs.csv"));
        List<Run> densityRuns = read(reports.resolve("density_control_runs.csv"));
        List<Run> bisectedRuns = read(reports.resolve("bisected_control_runs.csv"));
        List<Run> randomRuns = read(reports.resolve("random_split_control_runs.csv"));

        Map<String, Run> headlineIndex = index(headline);
        // Everything the two experiments would have to agree on if they really
        // ran the same model on the same split with the same seed.
        String[] shared = {"test_accuracy", "val_accuracy", "epochs_run", "split_index"};

        System.out.println("the density control is the headline experiment with a third arm");
        int[] dens = overlap(headlineIndex, index(densityRuns), shared);
        if (dens[0] == 0) {
            fail("density_control_runs.csv shares no run with runs.csv at all");
        } else if (dens[1] != dens[0]) {
            fail(String.format("only %d of %d shared runs are identical to runs.csv; "
                    + "the two experiments are not on the same splits and seeds",
                    dens[1], dens[0]));
        } else {
            System.out.printf("  %d shared runs, all identical to runs.csv in "
                    + "accuracy, validation, epochs and split%n", dens[0]);
        }

        System.out.println("\nthe other two controls are different protocols, not relabelled copies");
        for (var pair : new Object[][] {
                {"bisected_control_runs.csv", bisectedRuns},
                {"random_split_control_runs.csv", randomRuns}}) {
            @SuppressWarnings("unchecked")
            List<Run> rows = (List<Run>) pair[1];
            int[] o = overlap(headlineIndex, index(rows), new String[] {"test_accuracy"});
            if (o[0] == 0) {
                fail(pair[0] + " shares no run key with runs.csv, so it cannot be compared");
            } else if (o[1] * 2 >= o[0]) {
                fail(String.format("%s reproduces %d of %d of runs.csv, so it is not a "
                        + "second protocol", pair[0], o[1], o[0]));
            } else {
                System.out.printf("  %-32s %d of %d shared runs coincide, as a different "
                        + "split scheme should%n", pair[0], o[1], o[0]);
            }
        }

        System.out.println("\nthe graph free models cannot tell the arms apart");
        String[] accuracyColumns = {"test_accuracy", "test_accuracy_dedup", "test_accuracy_nbr_covered"};
        int compared = 0;
        for (var pair : new Object[][] {
                {"runs.csv", headline},
                {"density_control_runs.csv", densityRuns},
                {"bisected_control_runs.csv", bisectedRuns},
                {"random_split_control_runs.csv", randomRuns}}) {
            @SuppressWarnings("unchecked")
            List<Run> rows = (List<Run>) pair[1];
            Map<String, Run> byKey = index(rows);
            Set<String> regimes = new HashSet<>();
            for (Run r : rows) {
                regimes.add(r.regime());
            }
            for (Run r : rows) {
                boolean graphFree = false;
                for (String m : GRAPH_FREE) {
                    graphFree |= r.model().equals(m);
                }
                if (!graphFree || !r.regime().equals("transductive")) {
                    continue;
                }
                for (String regime : regimes) {
                    if (regime.equals("transductive")) {
                        continue;
                    }
                    Run other = byKey.get(r.dataset() + "|" + r.model() + "|" + regime + "|" + r.seed());
                    if (other == null) {
                        fail(pair[0] + " " + r.dataset() + "/" + r.model() + " seed " + r.seed()
                                + " has no " + regime + " row");
                        continue;
                    }
                    for (String c : accuracyColumns) {
                        compared++;
                        if (!r.fields().get(c).equals(other.fields().get(c))) {
                            fail(pair[0] + " " + r.dataset() + "/" + r.model() + " seed " + r.seed()
                                    + " " + c + ": transductive " + r.fields().get(c) + " but "
                                    + regime + " " + other.fields().get(c));
                        }
                    }
                }
            }
        }
        System.out.printf("  %d accuracies compared across arms, every one identical to the "
                + "last digit%n", compared);

        System.out.println("\nLabelProp has no parameters, so one split is one answer");
        int groups = 0;
        for (var pair : new Object[][] {
                {"runs.csv", headline},
                {"density_control_runs.csv", densityRuns},
                {"bisected_control_runs.csv", bisectedRuns},
                {"random_split_control_runs.csv", randomRuns}}) {
            @SuppressWarnings("unchecked")
            List<Run> rows = (List<Run>) pair[1];
            Map<String, Set<String>> bySplit = new TreeMap<>();
            for (Run r : rows) {
                if (!r.model().equals("LabelProp")) {
                    continue;
                }
                bySplit.computeIfAbsent(r.dataset() + " split " + r.fields().get("split_index"),
                        k -> new HashSet<>()).add(r.fields().get("test_accuracy"));
            }
            for (Map.Entry<String, Set<String>> e : bySplit.entrySet()) {
                groups++;
                if (e.getValue().size() != 1) {
                    fail(pair[0] + " LabelProp on " + e.getKey() + " gives "
                            + e.getValue().size() + " different accuracies: " + e.getValue());
                }
            }
        }
        System.out.printf("  %d dataset and split groups, one answer each%n", groups);

        System.out.println("\nthe epoch budget the README states");
        int checked = 0;
        for (var pair : new Object[][] {
                {"runs.csv", headline},
                {"density_control_runs.csv", densityRuns},
                {"bisected_control_runs.csv", bisectedRuns},
                {"random_split_control_runs.csv", randomRuns}}) {
            @SuppressWarnings("unchecked")
            List<Run> rows = (List<Run>) pair[1];
            for (Run r : rows) {
                int epochs = Integer.parseInt(r.fields().get("epochs_run").trim());
                checked++;
                if (r.model().equals("LabelProp")) {
                    if (epochs != 0) {
                        fail(pair[0] + " LabelProp ran " + epochs + " epochs, it has nothing to fit");
                    }
                } else if (epochs < 1 || epochs > MAX_EPOCHS) {
                    fail(pair[0] + " " + r.dataset() + "/" + r.model() + " seed " + r.seed()
                            + " ran " + epochs + " epochs, outside 1 to " + MAX_EPOCHS);
                }
            }
        }
        System.out.printf("  %d runs, every trained one inside 1 to %d epochs and every "
                + "LabelProp run at 0%n", checked, MAX_EPOCHS);

        if (failures > 0) {
            System.out.println("\n" + failures + " protocol claims are not true of the files");
            System.exit(1);
        }
        System.out.println("\nJava confirms the four experiment files are the experiments the "
                + "README describes");
    }
}
