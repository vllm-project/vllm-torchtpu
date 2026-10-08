#!/usr/bin/env bash
# Type-check src/ with third-party packages resolved.
#
# Runs inside the CI image, where torch, jax and vllm are installed, so calls
# into those packages are checked against their real signatures. The
# pre-commit hook runs the same checker with no packages installed; every
# third-party import is `Any` there, so it sees fewer errors and keeps its own
# baseline. This script uses the full-environment baseline instead. Extra
# arguments are passed through, so `--update-baseline` regenerates it.
#
# Outside a container the script reruns itself inside the nightly build of
# the image Buildkite uses (the Dockerfile's `dev` target), because other local
# package versions report different errors. It pulls the nightly on every run
# so the check never uses a stale one; the pull downloads only when a newer
# build exists, and removes the copy it replaces unless something else still
# uses it. That needs Docker and
# `gcloud auth configure-docker us-central1-docker.pkg.dev`. Set
# TYPECHECK_IMAGE to check against a different image; it is pulled only when
# it is not present locally. Set TYPECHECK_IN_PLACE=1 to skip Docker in an
# environment that already has the CI image's packages.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -z "${TYPECHECK_IN_PLACE:-}" && ! -f /.dockerenv && ! -f /run/.containerenv ]]; then
  if ! command -v docker >/dev/null; then
    echo "tools/typecheck_full.sh runs in the CI image and needs Docker." >&2
    exit 1
  fi
  nightly="us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu/vllm-torchtpu-dev:latest"
  image="${TYPECHECK_IMAGE:-${nightly}}"
  old_id="$(docker image inspect --format '{{.Id}}' "${image}" 2>/dev/null || true)"
  if [[ "${image}" == "${nightly}" || -z "${old_id}" ]]; then
    if ! docker pull --platform linux/amd64 "${image}" >&2; then
      echo "Could not pull ${image}. For an authentication error, run:" \
        "gcloud auth configure-docker us-central1-docker.pkg.dev." \
        "Without access to the registry, build the image locally:" \
        "see docs/developers_guide/type_checking.md." >&2
      exit 1
    fi
    # Each nightly would otherwise leave several GB behind. Remove the
    # replaced copy only once no tag points at it.
    new_id="$(docker image inspect --format '{{.Id}}' "${image}")"
    if [[ -n "${old_id}" && "${old_id}" != "${new_id}" ]] &&
      [[ "$(docker image inspect --format '{{len .RepoTags}}' "${old_id}")" == 0 ]]; then
      docker image rm "${old_id}" >/dev/null ||
        echo "Could not remove the replaced image ${old_id}; see the error above." >&2
    fi
  fi
  created="$(docker image inspect --format '{{.Created}}' "${image}")"
  echo "Checking in ${image} (built ${created})." >&2
  # --init makes Ctrl-C reach pyrefly; bash as PID 1 would ignore it.
  tty_flag=""
  if [[ -t 0 && -t 1 ]]; then
    tty_flag="-it"
  fi
  exec docker run --rm --init --platform linux/amd64 ${tty_flag:+"${tty_flag}"} \
    -v "${PWD}:/workspace/vllm-torchtpu" \
    -w /workspace/vllm-torchtpu \
    "${image}" bash tools/typecheck_full.sh "$@"
fi

uv pip install --system --quiet pyrefly==1.3.0

site_packages="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

status=0
pyrefly check \
  --site-package-path "${site_packages}" \
  --baseline .pyrefly-baseline-full.json \
  "$@" || status=$?

# As root in a container, `--update-baseline` writes the file as root; give it
# the owner of the checkout.
if [[ "$(id -u)" == 0 && -f .pyrefly-baseline-full.json ]]; then
  owner="$(stat -c %u:%g .)"
  if [[ "$(stat -c %u:%g .pyrefly-baseline-full.json)" != "${owner}" ]]; then
    chown "${owner}" .pyrefly-baseline-full.json ||
      echo "Could not give .pyrefly-baseline-full.json back to ${owner}." >&2
  fi
fi
exit "${status}"
