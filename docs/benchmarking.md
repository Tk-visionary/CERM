# Benchmarking and performance claims

CERM keeps package validation separate from research-scale performance studies.
The installable-library repository contains correctness, compatibility, and
release checks; external benchmark protocols and results live in
the separate research workspace.

## Claims supported by this repository

The package tests and CI are designed to support claims such as:

- the documented Python API imports and follows its tested estimator contracts;
- semantic, optimized, exported, and native paths satisfy their documented
  equivalence or error-bound checks where implemented;
- wheel/sdist metadata, installation, and package hygiene are reproducible;
- supported Python and operating-system CI jobs pass for the release revision.

These checks are not evidence of general predictive or systems superiority over
other tabular learners.

## External performance claims

Claims comparing CERM with XGBoost, LightGBM, CatBoost, scikit-learn, or other
learners should be backed by an exact, immutable revision of the separately
published research record that documents:

- public task identifiers and fixed splits;
- preprocessing and tuning budgets;
- seeds and software versions;
- failures and timeouts;
- quality metrics at task level;
- fit time, prediction time, memory, and model size where relevant;
- statistical summaries and any preregistered promotion gates.

README headline claims should not be copied from an exploratory or superseded
experiment without a stable research revision supporting them.

## Repository boundary

The CERM library repository intentionally does not carry OpenML task manifests,
raw benchmark tables, preregistrations, cross-library benchmark runners, or UNDR
research runners. Those assets are maintained separately and will be published
after their own release audit. Stable methods may later move into CERM through
an explicit pull request.
