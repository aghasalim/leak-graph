PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: venv install test audit audit-quick detectors control control-bisected \ tables-check
	control-random-splits duplicate-definitions tables figures clean

venv:
	/Users/salim/.local/bin/python3.12 -m venv .venv || python3.12 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install:
	$(PIP) install -r requirements.txt

# Tests run on a synthetic graph only. No downloads, no network. This is what CI runs.
test:
	$(PY) -m pytest tests -q

# The full audit. Downloads ~100MB of benchmark data on first run and takes roughly an hour
# on a laptop CPU.
audit:
	$(PY) experiments/run_audit.py --seeds 10

# Citation networks only, fewer seeds. Useful for checking the pipeline end to end.
audit-quick:
	$(PY) experiments/run_audit.py --datasets Cora CiteSeer --seeds 3 --epochs 100

# Detectors only: no training, so this is minutes rather than an hour.
detectors:
	$(PY) experiments/run_audit.py --detectors-only

# Separates the "smaller training graph" cost from the "test nodes specifically hidden" cost.
# Only constructible on the Planetoid splits; see the README.
control:
	$(PY) experiments/run_density_control.py

# The same control on every dataset, by reserving half the test set as the removal pool, and
# again under a second split scheme. Together these are what make chameleon and squirrel
# measurable at all.
control-bisected:
	$(PY) experiments/run_density_control.py --bisect

control-random-splits:
	$(PY) experiments/run_density_control.py --random-splits

# No training: sweeps eighteen readings of "duplicate" against the rates quoted in the
# literature. Minutes.
duplicate-definitions:
	$(PY) experiments/run_duplicate_definitions.py

# Regenerates every table in the README from the committed artifacts.
tables:
	$(PY) experiments/make_tables.py > reports/tables.md

figures:
	$(PY) experiments/make_figures.py

clean:
	rm -rf reports/*.csv reports/*.json .pytest_cache **/__pycache__

tables-check:  ## fail if the generated tables no longer match the CSVs
	$(PY) experiments/make_tables.py --check
