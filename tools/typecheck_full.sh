#!/usr/bin/env bash
# Type-check src/ with third-party packages resolved.
#
# Runs inside the CI image, where torch, jax and vllm are installed, so calls
# into those packages are checked against their real signatures. The
# pre-commit hook runs the same checker with no packages installed; every
# third-party import is `Any` there, so it sees fewer errors and keeps its own
# baseline. This script uses the full-environment baseline instead. Extra
# arguments are passed through, so `--update-baseline` regenerates it.
set -euo pipefail

cd "$(dirname "$0")/.."

uv pip install --system --quiet pyrefly==1.3.0

site_packages="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

pyrefly check \
  --site-package-path "${site_packages}" \
  --baseline .pyrefly-baseline-full.json \
  "$@"
