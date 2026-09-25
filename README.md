# CERM

[![CI](https://github.com/Tk-visionary/CERM/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Tk-visionary/CERM/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](./LICENSE)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-compatible-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/status-alpha-orange)](docs/api_stability.md)

[Documentation](docs/index.md) |
[Getting Started](docs/getting_started.md) |
[API Reference](docs/api_reference.md) |
[Changelog](CHANGELOG.md)

**Compiler-oriented finite-state models for tabular classification and regression.**

CERM is an experimental, scikit-learn-compatible learning library for dense
tabular data. It learns explicit finite-state main and interaction effects, then
keeps statistical fitting separate from prediction-program optimization and
native deployment. The design emphasizes **compact**, **auditable**, and
**deployable** tabular models without hiding the learned program behind a large
ensemble.

> **Alpha.** CERM is suitable for experimentation and independent evaluation.
> Sparse matrices, GPU training, and out-of-core training are not currently
> supported. Experimental APIs may change before a stable release.

## Highlights

- **Familiar API.** Classification and regression estimators follow the normal
  scikit-learn `fit`, `predict`, and `predict_proba` workflow.
- **Fused regression by default.** `CERMRegressor` uses fused residual V2: a
  finite-state mean baseline plus gated residual-distribution correction. The
  previous finite-state Ridge estimator is retained as `CERMRidgeRegressor`.
- **Explicit finite-state structure.** Main effects and interactions can be
  inspected directly instead of being distributed across a large tree ensemble.
- **Compiler-oriented deployment.** Training, prediction optimization, export,
  and native compilation are separate stages.
- **Auditable behavior.** Resolved parameters, fit diagnostics, feature tables,
  interaction tables, and model summaries are available through public
  inspection helpers where the fitted representation has that structure.
- **Research is separated from releases.** Experimental studies, OpenML audits,
  preregistrations, and external benchmark protocols are maintained separately
  from the installable library. Research artifacts will be published separately.

## Installation

The PyPI distribution name is `CERM`; the Python import name is `cerm`.
After the first public release is visible on PyPI:

```bash
python -m pip install CERM
```

For an unreleased source checkout:

```bash
git clone https://github.com/Tk-visionary/CERM.git
cd CERM
python -m pip install -e .
```

For development and tests:

```bash
python -m pip install -e ".[test]"
python -m pytest
```

## Quick start

```python
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from cerm import CERMClassifier, model_summary

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

model = CERMClassifier(random_state=42).fit(X_train, y_train)
probability = model.predict_proba(X_test)[:, 1]

print("log loss:", log_loss(y_test, probability))
print(model_summary(model))
```

For pandas DataFrames, non-numeric columns are categorical by default and the
fitted input schema is retained. Reordered prediction columns are restored to
the fitted order; missing or unexpected columns fail explicitly.

## Estimators

```python
from cerm import (
    CERMClassifier,
    CERMRegressor,
    CERMRidgeRegressor,
    CERMGeneralizedRegressor,
    CERMMultiLabelClassifier,
    CERMMultiOutputRegressor,
)
```

`CERMClassifier` supports binary and multiclass classification.

`CERMRegressor` is the ordinary regression estimator and now uses fused residual
V2. Its validated default capacity is `max_features=24`,
`max_interaction_features=12`, and `max_pairs=4`. The fused procedure contains a
finite-state mean baseline and may select `current_mean`, `fused`, or the
small-sample raw fused correction according to its fitted gate.

`CERMRidgeRegressor` preserves the previous `CERMRegressor` finite-state Ridge
implementation and its historical parameter vocabulary. `CERMFusedRegressor`
remains available as a compatibility name for code that explicitly opted into
fused V2 before it became the default.

`CERMGeneralizedRegressor` exposes generalized objectives such as quantile and
count-family losses. Multilabel and multi-output targets use dedicated
estimators. See [task support](docs/tasks.md).

## Parameters

Estimator families intentionally do not share a fake universal HPO vocabulary.
Use `get_tunable_params(model)` for the active estimator.

For default fused regression:

```python
from cerm import CERMRegressor

model = CERMRegressor(
    n_bins=10,
    max_bins=16,
    max_features=24,
    max_interaction_features=12,
    max_pairs=4,
    random_state=42,
)
```

For classification and historical Ridge regression, direct finite-state controls
include `max_bins`, `max_features`, `max_interaction_features`,
`max_interactions`, `interaction_order`, `reg_lambda`, `subsample`, and
`colsample` where supported. Human-language aliases remain compatibility
controls for those historical surfaces.

Use `get_tunable_params(model)` for the HPO-oriented view,
`get_search_params(model)` for search-family choices, `get_resource_params(model)`
for execution limits, and `get_convenience_params(model)` for the compatibility
alias view. See [parameters](docs/parameters.md).

## Inspect what CERM fitted

```python
from cerm import inspect_model, feature_table, interaction_table, structure_table

inspection = inspect_model(model)
features = feature_table(model)
interactions = interaction_table(model)
structure = structure_table(model)
```

For compact logs, use `model_summary(model)`. Fused regression reports its engine,
selected branch, and public capacity through `fit_diagnostics_`; historical
finite-state estimators additionally expose their explicit selected structure.
See [model inspection](docs/inspection.md).

## Save, export, and deploy

Python persistence:

```python
path = model.save("cerm_model.joblib")
```

Versioned CERM export:

```python
manifest = model.export("cerm_export")
```

`CERMRegressor.compile_native(...)` uses the fused native lowering.
`CERMRidgeRegressor` keeps the previous Ridge semantic/native contracts.
Prediction optimization and native compilation remain separate from statistical
training, so deployment rewrites do not silently reselect the fitted model. See
[advanced usage](docs/advanced_usage.md) and
[compatibility](docs/compatibility.md).

## Documentation

- [Getting started](docs/getting_started.md)
- [API reference](docs/api_reference.md)
- [Default regression / fused V2](docs/fused_regression.md)
- [Model inspection](docs/inspection.md)
- [Validation errors](docs/errors.md)
- [Installation](docs/installation.md)
- [Parameters](docs/parameters.md)
- [Task support](docs/tasks.md)
- [Advanced usage](docs/advanced_usage.md)
- [Architecture](docs/architecture.md)
- [Limitations](docs/limitations.md)
