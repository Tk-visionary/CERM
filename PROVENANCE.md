# Code provenance

CERM is an independently developed research software project maintained by
Taishi Kawahara. It is not an official project of the University of Tokyo,
OpenAI, or another institution.

Development was substantially assisted by OpenAI's ChatGPT. The maintainer
specified the architecture, algorithms, tests, experiments, acceptance gates,
and API, and reviewed the resulting implementation.

## Source classification

- `src/cerm/`: public package implementation.
- `src/cerm/_internal/`: private CERM implementation modules retained behind the
  public API. The directory name does not indicate bundled third-party code.
- `src/cerm/_compat/`: narrowly scoped compatibility code for supported
  scikit-learn versions.
- `tests/`, `scripts/`, `examples/`, and `docs/`: project-owned validation and
  documentation assets.
- Research-scale experiments, external audits, preregistrations, and benchmark
  protocols are maintained in a separate private research workspace and will be
  published separately after their own release audit.

The available pre-public version snapshots from 0.5.5 onward show incremental
development, but some internal modules predate the oldest retained snapshot.
Those files were included in the pre-release source review and no
high-confidence public code match was found.

## Pre-public 0.12.0a11 reconstruction history

CERM 1.0.0a1 descends from an evidence-backed reconstruction of the pre-public
0.12.0a11 tree, performed from the retained 0.12.0a10 source plus the recovered
a10→a11 patch contract. Before adding UNDR, that reconstructed tree passed the
recorded 189/189 package tests. The project does **not** claim byte identity with
an original 0.12.0a11 archive that was unavailable at restoration time.

The public source tree does not carry private restoration artifacts, generated
distributions, large patch/evidence files, the full pre-public 0.x changelog, or
research-only benchmark/runner trees.

The pre-public development history and reconstruction evidence are retained
privately for provenance and audit. The public Git repository intentionally
starts from a clean release snapshot rather than exposing the complete private
development history.

## Public license

CERM 1.0.0a1 is prepared for public release under the Apache License 2.0
(`Apache-2.0`). The relicensing was completed before publication. Pre-public
development records are retained privately rather than rewritten to make the
historical license state appear different from what it was during development.

## Third-party code

CERM depends on third-party packages through separately installed Python
distributions. Their licenses remain governed by those projects. No third-party
source package is intentionally vendored into CERM.

Provenance or licensing concerns may be reported using the route documented in
`SECURITY.md`.
