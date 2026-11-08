#!/usr/bin/env bash
# Build release artifacts with a current toolchain (system pip on some distros builds "UNKNOWN" packages).
set -euo pipefail
cd "$(dirname "$0")/.."
ENV=$(mktemp -d)/build-env
python3 -m venv "$ENV" && "$ENV/bin/pip" install -q --upgrade pip build twine
rm -rf dist && "$ENV/bin/python" -m build --outdir dist .
"$ENV/bin/twine" check dist/*
# Smoke-test the wheel in a clean environment
TEST=$(mktemp -d)/test-env
python3 -m venv "$TEST" && "$TEST/bin/pip" install -q dist/*.whl && "$TEST/bin/lens" --version
echo "Ready. Publish with: $ENV/bin/twine upload dist/*   (needs a PyPI API token)"
