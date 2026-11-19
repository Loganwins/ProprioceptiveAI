# Changelog

All notable changes to `proprioceptive-lens`. The format follows
[Keep a Changelog](https://keepachangelog.com/); versions follow [Semantic
Versioning](https://semver.org/).

## [0.1.17] - 2026-11-19

First release published via **PyPI Trusted Publishing** (OIDC, no long-lived
token). No client feature changes — this is a packaging / supply-chain update.

### Added
- **Trusted Publishing** — releases are now published from GitHub Actions
  using an OIDC token that PyPI verifies against the repo. A compromised upload
  token can no longer ship a trojaned wheel.
- **IP-leak gate on publish** — the build fails if any 9+ float array (steering
  vector / dark-head theta) is ever inlined into the client wheel. The client
  stays IP-free; the proprietary vectors remain server-side.
  (`scripts/check_ip_leak.py`, wired into `.github/workflows/publish.yml`.)
- **Provenance metadata** — the package now declares its author and maintainer.

## [0.1.0] to [0.1.16] - 2026-11-08 to 2026-11-09

Early iteration of the client (interpretability dashboard + deep mode +
verified-answer route + local memory + MCP). Versions 0.1.0 through 0.1.9 were
**yanked** (early bugs, including the 0.1.9 deep-mode space-dropping issue);
0.1.10 through 0.1.16 are the live line. This section is not expanded — the
detailed per-version history predates the adoption of a changelog and the
Trusted-Publishing pipeline.

## Unreleased

### Changed
- (add entries here before the next release)
