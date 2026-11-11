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
"""Opt-in MSA kernel variants, set from env flags before the first trace.

The kernels are plain JAX and read module-level flags; the attention layer
calls `apply` from its constructor. Every flag defaults off (the kernels'
default paths), and an enabled one logs once per process with its env name.
"""

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.minimax_m3.kernels import msa_attention

logger = init_logger(__name__)

_logged: set[str] = set()


def _set(env_name: str, module, attr: str) -> None:
    value = bool(getattr(envs, env_name))
    setattr(module, attr, value)
    if value and env_name not in _logged:
        _logged.add(env_name)
        logger.info("MiniMax-M3 MSA kernel variant on: %s=1", env_name)


def apply() -> None:
    _set("TPU_MINIMAX_M3_MSA_UNSTABLE_SORT", msa_attention, "UNSTABLE_UNION_SORT")
