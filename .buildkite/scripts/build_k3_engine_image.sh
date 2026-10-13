#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The engine image of the Kimi-K3 megakernel lane, published as the build's
# K3_ENGINE_IMAGE metadata, which common.sh hands the AgentX manifest. The
# engines serve code from another branch than the one this build runs (the
# megakernel integration branch does not merge into main cleanly), while the
# client and the pair's scripts stay on this build's commit.
#
#   K3_ENGINE_IMAGE=<image>  use that image as it is
#   K3_ENGINE_REF=<ref>      build the image from that ref of this repo - a
#                            branch, a tag, a full 40-character commit SHA (git
#                            cannot fetch a short one) or refs/pull/<N>/head;
#                            default inferact/megakernel-integration - or reuse
#                            it if the registry already has it
set -euo pipefail

if [ -n "${K3_ENGINE_IMAGE:-}" ]; then
  echo "--- Engine image given: ${K3_ENGINE_IMAGE}"
  buildkite-agent meta-data set K3_ENGINE_IMAGE "${K3_ENGINE_IMAGE}"
  exit 0
fi

REF="${K3_ENGINE_REF:-inferact/megakernel-integration}"
# setup_docker_env.sh's repository; its tags are <commit>-<vllm>, so a k3engine-
# prefix keeps these apart from the per-build CI images.
IMAGE_REPO="us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu-ci/vllm-torchtpu"

echo "--- Resolving ${REF}"
git fetch -q origin "${REF}"
SHA="$(git rev-parse FETCH_HEAD)"
# The vLLM that ref pins, as build_image.sh reads it, so the tag names both.
VLLM_PIN="$(git show "${SHA}:pyproject.toml" \
  | sed -nE 's/.*vllm @ git\+https:\/\/github\.com\/vllm-project\/vllm\.git@([a-zA-Z0-9.-]+).*/\1/p' | head -n 1)"
[ -n "${VLLM_PIN}" ] || { echo "[FATAL] no vLLM pin in ${REF}'s pyproject.toml" >&2; exit 1; }
TAG="${IMAGE_REPO}:k3engine-${SHA}-${VLLM_PIN}"

if docker manifest inspect "${TAG}" >/dev/null 2>&1; then
  echo "--- ${TAG} is already in the registry"
else
  echo "--- Building ${TAG} from ${REF} (${SHA})"
  bash .buildkite/scripts/cleanup_docker.sh
  SRC="$(mktemp -d)/engine"
  git worktree add -q --detach "${SRC}" "${SHA}"
  trap 'git worktree remove --force "${SRC}" >/dev/null 2>&1 || true' EXIT
  # That ref's own build script, Dockerfile and vLLM pin: the image is the
  # branch's, built the way its own pipelines build it.
  ( cd "${SRC}" && ./docker/build_image.sh --target dev -t "${TAG}" )
  echo "--- Pushing ${TAG}"
  docker push "${TAG}"
  docker rmi "${TAG}" || true
fi

buildkite-agent meta-data set K3_ENGINE_IMAGE "${TAG}"
echo "--- K3_ENGINE_IMAGE=${TAG}"
