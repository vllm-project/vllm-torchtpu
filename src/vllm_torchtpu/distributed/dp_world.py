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
"""Whether the DP engines share one distributed world and step in lockstep.

Expert parallelism needs it: its MoE combine reduces across every DP*TP
worker. A model can need it for other collectives too (for example an
embedding sharded across DP); its config hook (VerifyAndUpdateConfig, which
vLLM runs before the platform's check_and_update_config) calls
`require_unified_dp_world`. The mark travels with the ParallelConfig to every
worker, so the checks below need nothing but the ParallelConfig.

Kept free of torch and model imports: process binding runs before the
accelerator runtime is initialized.
"""

UNIFIED_DP_WORLD_ATTR = "tpu_unified_dp_world"


def require_unified_dp_world(parallel_config) -> None:
    """Mark that this model's DP engines must share one world."""
    setattr(parallel_config, UNIFIED_DP_WORLD_ATTR, True)


def unified_dp_world(parallel_config) -> bool:
    """EP, or a model that asked for it, couples the DP engines."""
    return bool(
        parallel_config.enable_expert_parallel
        or getattr(parallel_config, UNIFIED_DP_WORLD_ATTR, False)
    )
