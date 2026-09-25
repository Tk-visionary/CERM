# Installation

CERM targets Python 3.10–3.13. The PyPI distribution name is `CERM`; the
Python import name is `cerm`.

## Install from PyPI

After the first public release is available on PyPI:

```bash
python -m pip install CERM
```

Verify the environment with:

```bash
python -m pip check
python -c "import cerm; print(cerm.__version__)"
```

## Install from a source checkout

Until the first PyPI release is visible, or when testing unreleased changes,
install the repository directly:

```bash
python -m pip install -e .
```

Then follow [Getting started](getting_started.md).

## Development installation

```bash
python -m pip install -e ".[test]"
python -m pytest
```

For lint, build, and distribution checks:

```bash
python -m pip install -e ".[dev]"
```

For the documentation site:

```bash
python -m pip install -e ".[docs]"
python -m mkdocs build --strict
```

Research-only dependencies, OpenML audit runners, and cross-library benchmark
workflows are maintained in a separate research workspace rather than as an
installation extra of the CERM package. Those artifacts will be published
separately.

## Install a locally built wheel

Build a wheel with:

```bash
python -m pip install build
python -m build
```

Then install the generated wheel from `dist/`:

```bash
python -m pip install dist/cerm-*.whl
```

The core wheel is pure Python. Native prediction compilation is optional and
requires a compatible local C++ compiler. Semantic and optimized Python
prediction backends remain available without a compiler.

## Supported input scope

The current package supports dense NumPy arrays and pandas DataFrames. Sparse
matrices, GPU training, and out-of-core training are outside the supported
scope. See [limitations](limitations.md) before deployment evaluation.
