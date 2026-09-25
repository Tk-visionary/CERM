# CERM

CERM is alpha-stage Apache-2.0 software for scikit-learn-compatible finite-state
classification and regression on dense tabular data.

If you are new to the project, start with [Getting started](getting_started.md).
The default estimator path is intentionally conventional:

```python
from cerm import CERMClassifier

model = CERMClassifier().fit(X_train, y_train)
prediction = model.predict(X_test)
```

## Documentation paths

- [Getting started](getting_started.md): first model, DataFrames, regression, inspection, persistence.
- [API reference](api_reference.md): estimators, methods, fitted attributes, inspection helpers.
- [Parameters](parameters.md): recommended semantic controls and the full compatibility surface.
- [Task support](tasks.md): classification, regression, multiclass, multilabel, and generalized objectives.
- [Advanced usage](advanced_usage.md): optimization, compilation, export, and advanced controls.
- [Architecture](architecture.md): semantic learning, prediction optimization, and backend lowering.
- [Limitations](limitations.md): unsupported data and deployment boundaries.
- [API stability](api_stability.md): public/private and experimental API expectations.
- [Benchmark policy](benchmarking.md): how external performance claims are managed.

Research-scale experiments, preregistrations, OpenML audits, and cross-library
benchmark workflows are maintained in a separate research workspace and will be
published separately.
The CERM repository is focused on the installable library, package tests, user
documentation, examples, and release engineering.

CERM separates the fitted statistical program from training-graph rewrites,
optimized prediction backends, and native deployment.

CERM is independently maintained by Taishi Kawahara and was developed with
substantial assistance from OpenAI's ChatGPT. It is not an official project of
the University of Tokyo or OpenAI.
