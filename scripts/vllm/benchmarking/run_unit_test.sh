#!/bin/bash
# Script to run unit tests inside the container
set -euo pipefail

NIGHTLY=${1:-"false"} # Default is not nightly

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$REPO_DIR"

# Mark git directory as safe to avoid dubious ownership error in container
git config --global --add safe.directory "$REPO_DIR"

# Dependencies are installed in the workflow file (.github/workflows/tests.yml)

TEST_DIR="$REPO_DIR/tests"

# Run tests from a temp dir to force Python to test the installed package
cd "$(mktemp -d)"

echo "Running pytest..."
if [ "$NIGHTLY" = "yes" ]; then
  pytest "$TEST_DIR" -sv --timeout=300
else
  pytest "$TEST_DIR" -sv --timeout=300 -m "not nightly"
fi
