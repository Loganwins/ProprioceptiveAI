# Changelog

All notable changes to `proprioceptive-lens`. The format follows
[Keep a Changelog](https://keepachangelog.com/); versions follow [Semantic
Versioning](https://semver.org/).

## [0.1.19] - 2026-11-25

### Changed
- **Apache License 2.0.** The Lens client and its Claude Code plugin are now licensed
  under Apache 2.0 (was MIT), with a NOTICE file. The hosted backend, Behaviour probes
  and private probe assets remain proprietary and are not part of this repository.
- README: a clearer quick start, demos, a contributing guide and a link to
  proprioceptiveai.com.
- PyPI page: links to the website, the repository, issues and this changelog.

### Fixed
- Tests that need the optional deep-mode packages (torch, tokenizers) now skip when
  those are not installed, so the publish workflow's test step passes in CI.

### Note
- 0.1.18 was tagged but never published to PyPI. Its changes (free Deluxe with no time
  limit, and the D1-D6 fixes) reach PyPI in this release.

## [0.1.18] - 2026-11-20

### Changed
- **Lens Deluxe is free, with no time limit.** The Steer panel now says "Free, no card
  and no time limit" and its button reads "Get free key" (was "Free for 7 days" /
  "Start free trial"). The activation token emailed at sign-up no longer expires.
  README and docstrings updated to match.

### Fixed
- Six externally audited defects (D1-D6), first released here: contradiction-aware
  `answers_agree`; explicit truncation fields in `check_claims`; single-source
  `__version__`; origin wall on GET/POST/DELETE (loopback-only origins); verified-answer
  retry bounded at 8; the publish workflow runs the test suite before building.

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
