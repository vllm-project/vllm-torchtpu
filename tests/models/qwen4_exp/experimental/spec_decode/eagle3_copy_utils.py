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
"""Builders for tests of the model-owned EAGLE/MTP proposer copy."""

from types import SimpleNamespace
from unittest import mock

import torch

from vllm_torchtpu.models.qwen4_exp.experimental.spec_decode.eagle3 import (
    DraftChunkInputs,
    Eagle3Proposer,
)


def make_proposer(
    draft_tp: int | None = 1, target_tp: int = 1, method: str = "mtp"
) -> Eagle3Proposer:
    speculative_config = SimpleNamespace(
        draft_tensor_parallel_size=draft_tp, method=method, draft_model_config=None
    )
    parallel_config = SimpleNamespace(
        tensor_parallel_size=target_tp, pipeline_parallel_size=1
    )
    vllm_config = SimpleNamespace(
        speculative_config=speculative_config, parallel_config=parallel_config
    )
    return Eagle3Proposer(
        runner=mock.MagicMock(
            _pcp_enabled=False, _mtp_enabled=method == "mtp", _mtp_prefill_only=False
        ),
        vllm_config=vllm_config,
    )


def make_chunk(
    *,
    input_ids,
    position_ids,
    query_start_loc_np,
    start_index,
    num_reqs,
    hidden=8,
    device,
    padded_tokens=None,
    attn_ctx=None,
    hidden_states=None,
    draft_lengths=None,
    sequence_layout_plan=None,
):
    """Build a DraftChunkInputs for tests. aux/attn_ctx are only needed by
    paths that consume them; _prepare_draft_inputs does not touch attn_ctx, so
    it defaults to an inert placeholder. Pass a real attn_ctx for the propose
    loop, which reads attn_ctx.use_max_model_len."""
    padded_tokens = padded_tokens or input_ids.shape[0]
    chunk = DraftChunkInputs(
        input_ids=input_ids,
        position_ids=position_ids,
        query_start_loc_np=query_start_loc_np,
        attn_ctx=attn_ctx,
        start_index=start_index,
        num_reqs=num_reqs,
        # Eagle3 always exposes exactly 3 aux hidden states (the draft combine
        # consumes 3 * hidden); the compiled _draft_combine_hidden_states wrapper
        # takes them positionally, so provide 3.
        aux_hidden_states=[
            torch.zeros((padded_tokens, hidden), device=device) for _ in range(3)
        ],
        hidden_states=hidden_states,
        draft_lengths=draft_lengths,
    )
    if sequence_layout_plan is not None:
        chunk.sequence_layout_plan = sequence_layout_plan
    return chunk
