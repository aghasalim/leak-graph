# The sentences in the README, recomputed from the artifacts they describe.
#
# Every table in this repository is generated: `make tables` rewrites the marked
# regions of README.md and notes/METHODS.md from reports/, so a stale table shows
# up as a dirty git diff. The prose around those tables is not generated. The
# abstract, the headline paragraph and the discussion all quote figures that were
# typed in by hand once, from a run that is now months old, and nothing has ever
# checked them again.
#
# This recomputes each of those figures from reports/, renders it exactly as the
# sentence renders it, and requires the sentence to still contain it. A claim
# whose number has drifted fails here rather than being read by someone as fact.
#
#   ruby verify/claims.rb [root]

require "csv"
require "json"

ROOT = ARGV[0] || "."
README = File.read(File.join(ROOT, "README.md"))
# The README is hard wrapped, so a sentence is a line and a half. Collapsing
# whitespace lets a claim be written here as one string.
FLAT = README.gsub(/\s+/, " ")

def csv_rows(name)
  CSV.read(File.join(ROOT, "reports", name), headers: true).map(&:to_h)
end

def load_json(name)
  # Python writes bare NaN and Infinity. Ruby's parser takes them with allow_nan.
  JSON.parse(File.read(File.join(ROOT, "reports", name)), allow_nan: true)
end

# Percentage points, one decimal, exactly as the prose writes them.
def pp1(x)
  format("%.1f", 100 * x)
end

def pp2(x)
  format("%.2f", 100 * x)
end

def pp0(x)
  format("%.0f", 100 * x)
end

def commas(n)
  n.to_s.reverse.scan(/\d{1,3}/).join(",").reverse
end

WORDS = {
  0 => "not one", 1 => "one", 2 => "two", 3 => "three", 4 => "four", 5 => "five",
  6 => "six", 7 => "seven", 8 => "eight", 9 => "nine", 10 => "ten",
  11 => "eleven", 12 => "twelve", 13 => "thirteen", 14 => "fourteen",
  15 => "fifteen", 16 => "sixteen", 17 => "seventeen", 18 => "eighteen"
}.freeze

def word(n)
  WORDS.fetch(n) { raise "no word for #{n}, the claim needs rewriting" }
end

inflation = csv_rows("inflation.csv")
density   = csv_rows("density_control.csv")
bisected  = csv_rows("bisected_control.csv")
randomsp  = csv_rows("random_split_control.csv")
detectors = load_json("detectors.json")
dupdefs   = load_json("duplicate_definitions.json")

gnn = inflation.select { |r| %w[GCN GraphSAGE].include?(r["model"]) }
cell = lambda do |rows, dataset, model|
  rows.find { |r| r["dataset"] == dataset && r["model"] == model } ||
    raise("no #{dataset}/#{model} row")
end
det = ->(name) { detectors.find { |d| d["dataset"] == name } || raise("no #{name} detector row") }
num = ->(r, k) { Float(r[k]) }

# Largest inflation anywhere, and where it is. The abstract and the headline
# paragraph both quote it, in two different word orders.
top = gnn.max_by { |r| num.(r, "inflation_mean") }
top_pp = pp1(num.(top, "inflation_mean"))

resolved_gnn = gnn.select { |r| r["resolved"] == "True" }
worst_resolved = pp1(resolved_gnn.map { |r| num.(r, "inflation_mean").abs }.max)

cham = pp1(-num.(cell.(inflation, "chameleon", "GCN"), "inflation_mean"))
squir = pp1(-num.(cell.(inflation, "squirrel", "GCN"), "inflation_mean"))

pubmed_gcn = pp1(num.(cell.(inflation, "PubMed", "GCN"), "inflation_mean"))
pubmed_sage = pp1(num.(cell.(inflation, "PubMed", "GraphSAGE"), "inflation_mean"))

rs_gnn = randomsp.select { |r| %w[GCN GraphSAGE].include?(r["model"]) }
rs_resolved = rs_gnn.count { |r| r["total_resolved"] == "True" }

dens_gnn = density.select { |r| %w[GCN GraphSAGE].include?(r["model"]) }
dens_clearing = dens_gnn.count { |r| r["density_resolved"] == "True" } +
                dens_gnn.count { |r| r["test_specific_resolved"] == "True" }
dens_negative = dens_gnn.count { |r| num.(r, "test_specific_mean") < 0 }

cora_gcn_d = cell.(density, "Cora", "GCN")
sage_cora_d = cell.(density, "Cora", "GraphSAGE")
sage_cite_d = cell.(density, "CiteSeer", "GraphSAGE")

bis_cham = cell.(bisected, "chameleon", "GCN")
bis_squir = cell.(bisected, "squirrel", "GCN")

dup_component = pp1(gnn.map { |r| num.(r, "duplicate_component").abs }.max)

cora_det = det.("Cora")
sq_det = det.("squirrel")
pub_det = det.("PubMed")
cite_det = det.("CiteSeer")
cham_det = det.("chameleon")

ratios = dupdefs["verdicts"].map { |k, v| [k, v["citeseer_over_cora"]] }
within = ratios.count { |_, r| r.is_a?(Float) && r.finite? && r.abs <= 1.5 }

solve = dupdefs["solves"]["cosine"]

# Each claim: a label, the string rebuilt from the artifacts, and the file it
# came out of. The test is that the README still contains the rebuilt string.
CLAIMS = [
  ["largest inflation, abstract wording", "inflation.csv",
   "at most #{top_pp} accuracy points, on #{top['dataset']} with #{top['model']}"],
  ["largest inflation, headline wording", "inflation.csv",
   "#{top_pp} accuracy points at most, on #{top['dataset']} with #{top['model']}"],
  ["the two negative GCN readings", "inflation.csv",
   "*loses* #{cham} points on chameleon and #{squir} on squirrel"],
  ["how many GNN cells resolve", "inflation.csv",
   "Only #{word(resolved_gnn.length)} of the #{word(gnn.length)} GNN cells resolve at two standard errors"],
  ["the largest resolved reading", "inflation.csv",
   "no resolved reading exceeds #{worst_resolved} accuracy points"],
  ["PubMed reads nothing", "inflation.csv",
   "PubMed shows nothing at all for either GNN, #{pubmed_gcn} and #{pubmed_sage} points"],
  ["random splits resolve nothing", "random_split_control.csv",
   "leaves #{word(rs_resolved)} of the #{word(rs_gnn.length)} cells resolved"],
  ["the smaller-graph term for GraphSAGE", "density_control.csv",
   "the smaller-graph term alone is #{pp1(num.(sage_cora_d, 'density_cost_mean'))} points on Cora " \
   "and #{pp1(num.(sage_cite_d, 'density_cost_mean'))} on CiteSeer"],
  ["Cora GCN survives the control", "density_control.csv",
   "#{pp1(num.(cora_gcn_d, 'total_inflation_mean'))} point total, " \
   "#{pp1(num.(cora_gcn_d, 'density_cost_mean'))} of it density, and a resolved test-specific " \
   "#{pp1(num.(cora_gcn_d, 'test_specific_mean'))}"],
  ["the bisected control on the heterophilous pair", "bisected_control.csv",
   "chameleon GCN reading #{pp1(num.(bis_cham, 'density_cost_mean'))} density and " \
   "#{pp1(num.(bis_cham, 'test_specific_mean'))} test-specific and squirrel GCN " \
   "#{pp1(num.(bis_squir, 'density_cost_mean'))} and #{pp1(num.(bis_squir, 'test_specific_mean'))}"],
  ["what the second split scheme does to the negatives", "random_split_control.csv",
   "chameleon moves from -#{cham} to #{pp1(num.(cell.(randomsp, 'chameleon', 'GCN'), 'total_inflation_mean'))} " \
   "and squirrel from -#{squir} to #{pp1(num.(cell.(randomsp, 'squirrel', 'GCN'), 'total_inflation_mean'))}"],
  ["what the duplicates are worth", "inflation.csv",
   "moves inflation by at most #{dup_component} points anywhere"],
  ["squirrel straddling pairs", "detectors.json",
   "squirrel has #{commas(sq_det['duplicates']['straddling_pairs'])} straddling pairs"],
  ["density components clearing two standard errors", "density_control.csv",
   "#{word(dens_clearing)} of its #{word(2 * dens_gnn.length)} GNN components clearing two"],
  ["negative test-specific terms", "density_control.csv",
   "negative in #{word(dens_negative)} of the #{word(dens_gnn.length)} cells"],
  ["PubMed under two definitions", "duplicate_definitions.json",
   "PubMed is #{pp2(dupdefs['rates']['PubMed']['exact_nodes_in_a_duplicate_group'])}% duplicated under exact " \
   "feature match and #{pp1(dupdefs['rates']['PubMed']['identical_neighbour_set'])}% duplicated under " \
   "identical neighbour set"],
  ["how many definitions land near Cora", "duplicate_definitions.json",
   "#{word(within).capitalize} of them put CiteSeer within a factor of 1.5 of Cora"],
  ["the cutoff solved for", "duplicate_definitions.json",
   "gives #{format('%.4f', solve['threshold_giving_citeseer_5pct'])}, at which Cora reads " \
   "#{pp2(solve['cora_rate_at_the_same_threshold'])}%"],
  ["my CiteSeer duplicate rate", "duplicate_definitions.json",
   "My CiteSeer number is #{pp2(dupdefs['rates']['CiteSeer']['exact_nodes_in_a_duplicate_group'])}%"],
  ["exposure against Cora", "detectors.json",
   "exposes #{pp0(sq_det['neighbour_label']['frac_test_with_train_neighbour'])}% of its test nodes to a " \
   "labelled training neighbour against Cora's " \
   "#{pp0(cora_det['neighbour_label']['frac_test_with_train_neighbour'])}%"],
  ["what the neighbour vote is worth", "detectors.json",
   "scores #{pp0(sq_det['neighbour_label']['vote_accuracy_on_covered'])}% there against a " \
   "#{pp0(sq_det['feature_label']['majority_class_accuracy'])}% majority baseline"],
  ["the same vote on Cora", "detectors.json",
   "#{pp0(cora_det['neighbour_label']['vote_accuracy_on_covered'])}% against " \
   "#{pp0(cora_det['feature_label']['majority_class_accuracy'])}%"],
  ["the two baselines that bound Cora", "inflation.csv",
   "scores #{pp1(num.(cell.(inflation, 'Cora', 'MLP'), 'inductive_mean'))}% against the inductive GCN's " \
   "#{pp1(num.(cell.(inflation, 'Cora', 'GCN'), 'inductive_mean'))}%"],
  ["the vote and the GCN on the same nodes", "detectors.json + inflation.csv",
   "scores #{pp1(cora_det['neighbour_label']['vote_accuracy_on_covered'])}% on the " \
   "#{pp0(cora_det['neighbour_label']['frac_test_with_train_neighbour'])}% of Cora test nodes"],
  ["the GCN on the covered subset", "inflation.csv",
   "the GCN scores #{pp1(num.(cell.(inflation, 'Cora', 'GCN'), 'inductive_nbr_covered_mean'))}% on that same subset"],
  ["giveaway features", "detectors.json",
   "Cora has #{cora_det['feature_label']['giveaway_features']} of " \
   "#{commas(cora_det['feature_label']['num_features'])} and CiteSeer has " \
   "#{cite_det['feature_label']['giveaway_features']} of " \
   "#{commas(cite_det['feature_label']['num_features'])}"],
  ["PubMed giveaway features", "detectors.json",
   "#{pub_det['feature_label']['giveaway_features']} of its " \
   "#{pub_det['feature_label']['num_features']} features"],
  ["exact duplicate rate on Cora", "detectors.json",
   "My #{pp1(cora_det['duplicates']['exact_duplicate_nodes'].to_f / cora_det['duplicates']['num_nodes'])}% " \
   "exact duplicate nodes on Cora"],
  ["exact duplicate rates on the heterophilous pair", "detectors.json",
   "#{pp1(cham_det['duplicates']['exact_duplicate_nodes'].to_f / cham_det['duplicates']['num_nodes'])}% on " \
   "chameleon with #{pp1(sq_det['duplicates']['exact_duplicate_nodes'].to_f / sq_det['duplicates']['num_nodes'])}% " \
   "on squirrel"]
].freeze

failures = 0
puts "#{CLAIMS.length} sentences in README.md, rebuilt from reports/"
CLAIMS.each do |label, source, rebuilt|
  if FLAT.include?(rebuilt)
    puts format("  ok   %-46s <- %s", label, source)
  else
    failures += 1
    puts format("  FAIL %-46s <- %s", label, source)
    puts "         the artifacts now say: #{rebuilt.inspect}"
    puts "         the README does not contain that string"
  end
end

if failures.positive?
  puts "\n#{failures} of #{CLAIMS.length} README sentences no longer match the artifacts"
  exit 1
end
puts "\nRuby rebuilt every one of the #{CLAIMS.length} hand written figures in the README\n" \
     "from reports/ and found the sentence still saying it"
