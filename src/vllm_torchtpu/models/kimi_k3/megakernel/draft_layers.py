# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

# Source provenance: inferact/k3-megakernel@30c9ba966b src/vllm_torchtpu/megakernel/kimi_k3/draft_layers.py
"""The MLA DSpark draft's latent layers as vLLM KV cache layers.

The megakernel runner has no vLLM drafter, so nothing declares the draft's
KV cache. With the MLA draft (``Inferact/Kimi-K3-DSpark``) both roles build
its five MLA attention layers on the meta device (no weights), named after the
target's layers as vLLM numbers a speculator's (``model.layers.93..97``).
Their spec equals the target's MLA layers', so vLLM groups them with those:
the same block tables and DCP pages, the KV budget counts them, and the KV
connector transfers them with the target's pages. The draft reads and writes
them as its pools (``pools.draft_raw_pools``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from vllm_torchtpu.models.kimi_k3.megakernel.pools import DRAFT_LAYER0


def register_draft_layers(vllm_config: Any, checkpoint: Path) -> list[torch.nn.Module]:
    """Build (and register) the draft's MLA layers; keep the returned modules
    alive for the runner's lifetime."""
    from vllm.config import set_current_vllm_config
    from vllm.transformers_utils.configs.k3_dspark import K3DSparkConfig

    from vllm_torchtpu.models.vllm.kimi_k3.attention import MultiHeadLatentAttention

    config = K3DSparkConfig.from_pretrained(str(checkpoint))
    layers: list[torch.nn.Module] = []
    # The target's quantization does not apply to the draft (its projections
    # are never materialized: meta device).
    saved = vllm_config.quant_config
    vllm_config.quant_config = None
    try:
        with set_current_vllm_config(vllm_config), torch.device("meta"):
            for index in range(int(config.num_hidden_layers)):
                layers.append(
                    MultiHeadLatentAttention(
                        config,
                        vllm_config,
                        prefix=f"model.layers.{DRAFT_LAYER0 + index}.self_attn",
                        use_rope=True,
                        non_causal_multi_token_decode=True,
                    )
                )
    finally:
        vllm_config.quant_config = saved
    return layers
