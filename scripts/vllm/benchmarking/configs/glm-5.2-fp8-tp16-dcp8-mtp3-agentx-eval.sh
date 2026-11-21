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
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# The GSM8K arm of glm-5.2-fp8-tp16-dcp8-mtp3-agentx.sh: the same server with
# real MTP verification instead of synthetic acceptance (accuracy has to come
# from the tokens the target model actually accepts), and no replay. The nightly
# GSM8K step runs agentx_sweep.sh with this config, the full 1319-question split
# and --gsm8k-min-score 0.90, the floor InferenceX holds a GLM-5.2 AgentX
# submission to (strict-match and flexible-extract both).
GLM52_AGENTX_SYNTHETIC_AL=""
# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/glm-5.2-fp8-tp16-dcp8-mtp3-agentx.sh"
# GSM8K only: agentx_sweep.sh passes "none" to agentx_client.sh, which then
# skips the replay.
AGENTX_CONC_LIST="none"
