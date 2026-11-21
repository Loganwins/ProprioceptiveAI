# Contributing to Proprioceptive Lens

Help make local model inspection easier to use. Useful contributions include clearer setup instructions, reproducible bug reports, UI improvements, client tests, and examples for terminal and MCP workflows.

## Report a bug

[Open an issue](https://github.com/Loganwins/ProprioceptiveAI/issues/new) with:

- Your operating system, Python version, and installed Lens version (`python -m pip show proprioceptive-lens`).
- The exact model name and whether you used Ollama or deep mode.
- Steps to reproduce, expected behaviour, and actual behaviour.
- A minimal public example and relevant error output, with credentials and private conversations removed.

For model-quality observations, include the prompt, model settings if known, and the original output. Distinguish token confidence, Behaviour readings, steering, and verified answers so the report can be reproduced in the right mode.

## Propose an improvement

For a substantial change, open an issue describing the user problem and proposed approach first. Small documentation and client fixes can go straight to a pull request.

Keep a pull request focused. Explain the user-visible change and how you checked it. Add a regression test when fixing a reproducible code bug; for documentation, check commands, links, and rendered formatting.

## Public client and private backend

This repository's client is licensed under Apache 2.0. The proprietary Behaviour probes and scoring service are separate, as described in [the README](README.md#license). Client contributions do not require publishing backend code, model probe assets, service credentials, or internal deployment records.

## Share what you build

Working integrations and clear demonstrations are welcome in issues. If Lens helps your workflow, a star or a link from your example helps others discover it.
