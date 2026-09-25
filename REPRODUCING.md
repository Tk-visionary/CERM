# Reproducing CERM 1.0.0a1

This repository contains the installable CERM library and its package/release
validation. Research-scale experiments and external benchmark protocols are
maintained in the separate research workspace.

Run commands from the repository root with CPython 3.10–3.13.

## 1. Create an isolated environment

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 2. Install development dependencies

```bash
python -m pip install -e ".[dev,docs]"
```

## 3. Run package and API checks

```bash
python scripts/check_package.py
python scripts/check_documentation.py
python scripts/check_public_api.py
python scripts/check_runtime_dependencies.py
```

## 4. Run the test suite

```bash
python -m pytest
python scripts/run_estimator_checks.py
```

CI repeats the package tests on Linux, macOS, and Windows across the supported
Python range, with separate coverage, estimator-check, lint, documentation, and
distribution jobs.

## 5. Build documentation

```bash
python -m mkdocs build --strict
```

## 6. Build and verify wheel/sdist

Build into a temporary directory if you want to keep the source checkout clean:

```bash
CERM_DIST_DIR="$(mktemp -d)"
python -m build --outdir "$CERM_DIST_DIR"
python -m twine check "$CERM_DIST_DIR"/*
python scripts/verify_distributions.py "$CERM_DIST_DIR"/*
```

For a fresh-wheel smoke test:

```bash
CERM_WHEEL="$(find "$CERM_DIST_DIR" -maxdepth 1 -name '*.whl' -print -quit)"
python -m venv "$CERM_DIST_DIR/fresh-wheel"
"$CERM_DIST_DIR/fresh-wheel/bin/python" -m pip install --upgrade pip
"$CERM_DIST_DIR/fresh-wheel/bin/python" -m pip install "$CERM_WHEEL"
"$CERM_DIST_DIR/fresh-wheel/bin/python" scripts/smoke_install.py
```

The distribution metadata must identify the project as `CERM 1.0.0a1` and the
license expression as `Apache-2.0`.

## Research reproduction

OpenML audits, GBDT comparisons, UNDR experiments, frozen task manifests,
preregistrations, and scaling studies are intentionally not part of this
library checkout. They are maintained in a separate research workspace and will
be published separately after its own provenance and privacy audit.

The public CERM repository intentionally starts from a clean release snapshot.
Pre-public development and research records are retained privately for provenance
and audit; see `PROVENANCE.md`.
