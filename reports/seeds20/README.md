# 20-seed replication

`python experiments/run_audit.py --seeds 20 --out reports/seeds20` on a Mac laptop CPU, python 3.12, the pinned `requirements.txt`. Same code and epoch budget as the 10-seed run in `reports/`; only `--seeds` differs.

Result: the 10-seed numbers hold. Every GCN and GraphSAGE inflation mean moved by less than one standard error of the 10-seed estimate (largest shift: 0.99 SE, chameleon GraphSAGE, 0.0061 to 0.0021), and the standard errors shrank as expected. Signs are unchanged except squirrel GraphSAGE, which is zero within noise at both counts (0.0012, then -0.0002). LabelProp and MLP are exactly 0 in both.

The published README tables still come from the 10-seed artifacts in `reports/`. This folder is an added replication, not a replacement, and `make tables` and `make verify` do not read it.
