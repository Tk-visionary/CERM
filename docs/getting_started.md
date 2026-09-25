# Getting started

CERM follows the scikit-learn estimator pattern: construct an estimator, call
`fit`, then use `predict`, `predict_proba`, or `score`.

## Install

After the first public PyPI release is available:

```bash
python -m pip install CERM
```

For an unreleased source checkout:

```bash
python -m pip install -e .
```

For tests and development tools:

```bash
python -m pip install -e ".[test]"
```

The distribution name is `CERM`; the Python import name is `cerm`.

## Binary classification

The default constructor is the recommended starting point. CERM accepts dense
NumPy arrays and pandas DataFrames.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

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

`CERMClassifier()` uses the full validated search semantics. You do not need to
choose state counts, interaction budgets, or solver controls to get started.

## DataFrames, categories, and missing values

With a DataFrame, `categorical_features="auto"` is the default. Non-numeric
columns are treated as categorical, and the fitted input schema is retained.
Prediction accepts the same columns in a different order and restores the fitted
order internally; missing or extra columns are rejected.

```python
import pandas as pd
from cerm import CERMClassifier

X = pd.DataFrame(
    {
        "age": [22, 41, 35, 58, 29, 47],
        "city": ["A", "B", "A", "C", "B", "C"],
        "income": [4.2, 7.5, None, 9.8, 5.1, 8.4],
    }
)
y = [0, 1, 0, 1, 0, 1]

model = CERMClassifier(random_state=42).fit(X, y)
print(model.feature_names_in_)
```

## Regression

The ordinary `CERMRegressor` now uses fused residual V2. It starts from a
finite-state mean baseline and promotes a residual correction only when the V2
selection gate chooses it.

```python
from cerm import CERMRegressor, CERMRidgeRegressor, model_summary

regressor = CERMRegressor(random_state=42).fit(X_train, y_train)
prediction = regressor.predict(X_test)
print(model_summary(regressor))
print(regressor.selected_kind_)
```

The previous finite-state Ridge estimator is still available explicitly:

```python
ridge = CERMRidgeRegressor(random_state=42).fit(X_train, y_train)
```

`CERMFusedRegressor` is retained as a compatibility name for code that adopted
fused V2 before it became the default. For quantile, Huber, Poisson, Gamma,
Tweedie, or multi-quantile objectives, use `CERMGeneralizedRegressor`. See
[task support](tasks.md) for the exact contracts.

## The parameters most users should consider

Classification and historical Ridge regression use the established finite-state
capacity vocabulary. For those estimators, direct controls include
`max_bins`, `max_features`, `max_interaction_features`, `max_interactions`,
`interaction_order`, `reg_lambda`, `subsample`, and `colsample`.

Default fused regression has a smaller dedicated HPO surface:

| Parameter | Default | Meaning |
|---|---:|---|
| `n_bins` | 10 | Residual-survival threshold count. |
| `max_bins` | 16 | Maximum nested finite-state resolution. |
| `max_features` | 24 | Retained main-effect feature cap. |
| `max_interaction_features` | 12 | Feature cap for pair search. |
| `max_pairs` | 4 | Shared fused pair-prefix cap. |

Use `get_tunable_params(model)` to avoid mixing estimator-specific vocabularies.
`CERMRegressor` intentionally does not expose Ridge-only controls such as
`max_interactions`, `reg_lambda`, presets, subsampling, or column sampling.
Choose `CERMRidgeRegressor` when you specifically need those historical
semantics.

For classification, convenience aliases such as `search_effort`, `state_detail`,
and `feature_budget` remain public for compatibility and interactive use, while
direct numeric controls remain preferred for HPO.

```python
from cerm import get_tunable_params

model = CERMClassifier(
    search_effort="balanced",
    interaction_order=2,
    n_jobs=-1,
    random_state=42,
)

print(get_tunable_params(model))
print(model.parameter_summary())
```

## Inspect a fitted model

Use the level of detail appropriate to the task:

```python
from cerm import get_tunable_params, model_summary

print(model_summary(model))       # compact notebook/log summary
print(get_tunable_params(model))  # recommended HPO/model view
print(model.fit_diagnostics_)     # detailed fit/search diagnostics
```

Some historical estimator families also expose `resolved_params_`; fused
regression instead reports its selected branch and effective capacity in
`fit_diagnostics_`. `get_params()` and `set_params()` remain the canonical
scikit-learn parameter API and work with cloning, pipelines, and search tools.

## Save or export

For Python-to-Python persistence:

```python
path = model.save("cerm_model.joblib")
restored = CERMClassifier.load(path)
```

For a versioned CERM package with checksummed JSON/NPZ state:

```python
manifest = model.export("cerm_export")
print(manifest)
```

Optimized and native deployment paths are separate from fitting, so deployment
rewrites do not silently change the selected statistical model. See
[advanced usage](advanced_usage.md).

## Where to go next

- [API reference](api_reference.md) for common classes, methods, and fitted attributes.
- [Default regression](fused_regression.md) for fused V2 and historical Ridge compatibility.
- [Parameters](parameters.md) for the full recommended and compatibility surfaces.
- [Task support](tasks.md) for multiclass, multilabel, and regression behavior.
- [Limitations](limitations.md) before production evaluation.
- Research artifacts for experimental studies, external audits, and benchmark
  protocols will be published separately.
