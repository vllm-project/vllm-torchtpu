#!/usr/bin/env bash
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

# Sourced by every script that names the CI image; leaves IMAGE_REPO set.
#
# The builder (setup_docker_env.sh) and the consumers have to agree on this
# string, and each of the five used to carry its own copy. A registry move that
# misses one leaves that lane pulling from the old project, which surfaces as a
# permission error on the pull rather than as anything naming the cause.
#
# CI_IMAGE_REPO overrides the default, so a fork or a one-off build can push
# somewhere its own service account can write without editing the tree.
# shellcheck disable=SC2034  # read by the scripts that source this file
IMAGE_REPO="${CI_IMAGE_REPO:-us-central1-docker.pkg.dev/inferact-vllm-tpu/vllm-tpu-ci/vllm-torchtpu}"

# Repos that get the same tag pushed to them, space separated. Only the builder
# reads this; nothing pulls from a mirror, because CI_IMAGE_TAG names IMAGE_REPO
# and that is what every consumer asks for.
#
# The old repo is a mirror rather than gone so that a pull someone has written
# down - a reproduction on a workstation, a manifest outside this tree - keeps
# resolving through the move. Unset it (CI_IMAGE_MIRROR_REPOS=) to push once.
# shellcheck disable=SC2034  # read by setup_docker_env.sh
CI_IMAGE_MIRROR_REPOS="${CI_IMAGE_MIRROR_REPOS-us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu-ci/vllm-torchtpu}"
