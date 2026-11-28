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
"""The in-graph RoPE switch, read from `--additional-config`.

Kept apart from `rope_in_graph` so the platform can validate the value at
startup without importing vLLM's rotary layers.
"""

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# Compute rotary cos/sin in-graph from positions instead of indexing a
# precomputed [max_position, rotary_dim] HBM table. On by default; set to
# false in `--additional-config` to keep the precomputed table.
# TODO(#2747): remove this key together with TPU_ROPE_CACHE_TRUNCATE and
# TPU_ROPE_CACHE_ROW_MAJOR once the DeepSeek-V4 Pallas kernels no longer read
# the table.
KEY = "tpu_rope_in_graph"


def in_graph_enabled(vllm_config: "VllmConfig | None") -> bool:
    """Returns whether in-graph RoPE is enabled; a non-bool value raises.

    Returns ``False`` when ``vllm_config`` is ``None`` so callers without a
    ``VllmConfig`` keep the precomputed table path, while an omitted
    ``tpu_rope_in_graph`` key in ``additional_config`` (such as ``{}``)
    defaults to ``True``.
    """
    if vllm_config is None:
        return False
    additional_config = cast("dict[str, Any]", vllm_config.additional_config)
    value = additional_config.get(KEY, True)
    if not isinstance(value, bool):
        raise ValueError(f"additional_config[{KEY!r}] must be a bool, got {value!r}")
    return value
