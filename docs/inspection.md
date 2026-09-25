# Model inspection

CERM exposes one task-neutral inspection workflow for the three primary
estimators:

```python
from cerm import inspect_model

inspection = inspect_model(model)
inspection
```

In a notebook, the object renders a compact model summary followed by selected
feature and model-structure tables. The same information can be used separately:

```python
from cerm import feature_table, interaction_table, model_summary, structure_table

print(model_summary(model))
feature_table(model)
interaction_table(model)
structure_table(model)
```

## Selected feature table

`feature_table(model)` reports the selected finite-state feature positions using
human-readable adapted feature names when a single fitted representation is
available. The columns are:

- `rank`: position inside the selected finite-state representation;
- `adapted_index`: index in CERM's adapted input matrix;
- `feature_name`: adapted feature name;
- `used_in_pair`: whether the feature participates in a selected pair term;
- `used_in_block`: whether it participates in a directed conditional block.

The table describes the fitted representation. It is not a feature-importance
ranking and does not claim causal importance.

## Interaction table

`interaction_table(model)` reports representation-level interactions with named
features. Current row kinds are `pair`, `fine_pair`, and `block`. Regression and
generalized regression expose selected pair terms. Binary classification can
additionally expose fine pairs and conditional blocks.

## One compact main/pair/block table

`structure_table(model)` combines selected main terms and interactions into one
short table. `kind` is one of `main`, `pair`, `fine_pair`, or `block`; `term`
contains the named feature expression and `detail` retains fitted structural
metadata such as the directed block gate state.

This is intended for notebooks, experiment review, and debugging. It is a
structural description, not an importance ranking.

Composite estimators such as independent one-vs-rest or multi-output bundles may
not have one globally meaningful structure; in that case the compact structural
tables are empty rather than combining incompatible per-head representations
silently.

## Operational summary

`model_summary(model)` is intentionally small. It reports configuration,
selected counts, fit time, estimated model bytes, and resource risk when those
quantities are available. Detailed research and search diagnostics remain in
`fit_diagnostics_`.

`inspect_model(model).to_dict()` converts the summary, feature table,
interaction table, and combined structure table into records suitable for
experiment logs or bug reports.
