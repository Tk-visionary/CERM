# Security policy

## Supported versions

CERM is alpha research software. Only the most recent alpha or development
release receives security fixes.

## Reporting

Do not publish suspected vulnerability details in a public issue.

Use GitHub Private Vulnerability Reporting for security-sensitive reports:

**Security → Report a vulnerability**

Please include enough detail to reproduce and assess the issue, but do not
include credentials, private data, or unrelated confidential artifacts.

Ordinary non-security bugs may be reported through the normal issue tracker.

## Untrusted artifacts

CERM uses `joblib` persistence and can load native shared libraries produced by
its compiler backend. Both mechanisms can execute code. Never load a model,
training cache, compiled package, manifest, or shared library from an untrusted
source.

Checksum verification detects accidental corruption; it is not a sandbox and
does not establish that an artifact is trustworthy.

## Native compilation

Native compilation invokes a local C++ compiler and loads the resulting shared
library into the Python process. Run this feature only in a controlled build
environment. The semantic and optimized Python prediction backends remain
available when native compilation is disabled or unavailable.
