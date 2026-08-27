#!/usr/bin/env bash
# ARGUS test runner (Unix shell)
#
# Usage:
#   ./scripts/run_tests.sh
#
# Assumes a virtual environment is already active with the `dev-test`
# (and whichever other) dependency groups installed:
#   pip install -e ".[core,dev-test]"

set -euo pipefail

echo "ARGUS: running test suite..."

python -m pytest tests/ -v

if [ $? -ne 0 ]; then
    echo "ARGUS: tests FAILED" >&2
    exit 1
fi

echo "ARGUS: tests PASSED"