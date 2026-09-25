# Changelog

## 1.0.0a1

Initial public repository snapshot for CERM 1.0.0a1. PyPI publication is a separate release step.

- Publish the distribution as `CERM` while keeping the Python import package `cerm`.
- Use the canonical GitHub repository `Tk-visionary/CERM` and PyPI Trusted Publishing.
- License the public release under the Apache License 2.0 (`Apache-2.0`).
- Provide scikit-learn-compatible classification, regression, generalized regression,
  multilabel, and multi-output estimator surfaces.
- Make fused residual V2 the ordinary top-level `CERMRegressor` implementation after
  maturing its typed-input, sample-weight, semantic-export, portable, and native-runtime
  contracts. Preserve the former finite-state Ridge regressor explicitly as
  `CERMRidgeRegressor`; keep `CERMFusedRegressor` as a compatibility name for the fused
  family.
- Keep the default fused regression capacity separate from Ridge-only parameter
  vocabulary instead of adding an ambiguous `regression_strategy` switch.
- Expose finite-state model inspection through model summaries and structure tables.
- Keep semantic fitting, optimized prediction, export, and native compilation as
  explicit separate stages.
- Keep research-scale experiments, OpenML audits, preregistrations, and external
  benchmark tooling in a separate research workspace so the library repository stays
  focused on package code, tests, documentation, examples, and release engineering.
- Keep unpromoted research mechanisms explicit and opt-in; promoted fused regression V2
  is the intentional exception reflected in the ordinary regression estimator.
- Ship the cleaned public documentation, release metadata, multi-platform CI,
  estimator checks, wheel/sdist verification, and installed-wheel smoke tests.
- Preserve the pre-public CERM development and reconstruction record privately for
  provenance and audit, while starting the public repository from a clean snapshot.

## Pre-public development history

Versions before `1.0.0a1` were internal research and release-preparation builds and
were not part of the public PyPI release line. Their detailed changelog, restoration
record, benchmark notes, patches, and research workspace are retained privately for
provenance and audit rather than published as part of the clean public repository
history.
