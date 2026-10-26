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
"""Host PLE lookup boundary for vLLM piecewise compilation."""

import torch
from vllm.forward_context import get_forward_context

from .ple_storage import HostPLETable


@torch.library.custom_op("qwen4_exp::host_lookup", mutates_args=())
def host_lookup(
    ids: torch.Tensor,
    local_tokens: int,
    embedding_dim: int,
    dp_size: int,
    fp8: bool,
    prefix: str,
) -> torch.Tensor:
    layer = get_forward_context().no_compile_layers[prefix]
    table = layer.ple_embedding.ngram_embedding
    assert isinstance(table, HostPLETable)
    raw = table.lookup_host(ids.cpu())
    return raw.reshape(dp_size, local_tokens, embedding_dim).to(ids.device)


@host_lookup.register_fake
def _host_lookup_fake(ids, local_tokens, embedding_dim, dp_size, fp8, prefix):
    return ids.new_empty(
        (dp_size, local_tokens, embedding_dim),
        dtype=torch.int8 if fp8 else torch.int16,
    )
