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
"""The dense sequence-parallelism switch, read from `--additional-config`.

Kept apart from `dense_sequence_parallel` and `gemma4_patch` so the platform
can validate the value at startup without importing vLLM model layers.
"""

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# Sequence parallelism for dense tensor-parallel decoder layers (wired for
# dense per-channel/per-tensor FP8 or unquantized bf16 Gemma 4; unquantized
# linears gather bf16 rows without FP8 quantization or the single-shot decode
# kernel). Off by default.
KEY = "tpu_dense_sequence_parallel"


def dense_sequence_parallel_enabled(vllm_config: "VllmConfig | None") -> bool:
    """Returns the setting; a non-bool value raises."""
    if vllm_config is None:
        return False
    additional_config = cast("dict[str, Any]", vllm_config.additional_config)
    value = additional_config.get(KEY, False)
    if not isinstance(value, bool):
        raise ValueError(f"additional_config[{KEY!r}] must be a bool, got {value!r}")
    return value
