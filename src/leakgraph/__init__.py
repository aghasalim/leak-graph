"""LeakGraph: a reproducible audit of leakage in transductive GNN node-classification
benchmarks.

This is tooling, not a discovery. The individual leakage phenomena it measures are already
documented in the literature (see the README's Prior work section). What is here is a single
harness that measures all of them on the same splits with the same seeds, plus one number --
leakage inflation, that summarises the cost of transductive evaluation per (dataset, model).
"""

from .data import DATASETS, GraphData, load, synthetic_graph
from .detectors import duplicate_report, feature_label_report, neighbour_label_report
from .experiment import run_audit, train_one
from .splits import Split, induced_subgraph, make_training_view, random_split

__all__ = [
    "DATASETS",
    "GraphData",
    "Split",
    "duplicate_report",
    "feature_label_report",
    "induced_subgraph",
    "load",
    "make_training_view",
    "neighbour_label_report",
    "random_split",
    "run_audit",
    "synthetic_graph",
    "train_one",
]

__version__ = "0.1.0"
