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
"""Unit tests for prompt row caching, token padding skip, and runner H2D reuse."""

from __future__ import annotations

import pickle
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.layernorm import RMSNorm

from vllm_torchtpu.layers.adapter.token_padding import TokenPaddingState
from vllm_torchtpu.layers.core.attention_metadata import (
    AttentionMetadataBuilderContext,
    stage_block_table_uploads,
)
from vllm_torchtpu.runner import utils as runner_utils
from vllm_torchtpu.runner.prompt_rows import PromptTokenIds, cache_prompt_rows
from vllm_torchtpu.runner.tpu_runner import (
    TPUModelRunner,
    _req_exceeds_most_model_len,
)


def test_prompt_token_ids_array_view_and_mutation_invalidation():
    src = np.array([10, 20, 30], dtype=np.int32)
    ids = PromptTokenIds([10, 20, 30], src)
    arr = np.asarray(ids)
    assert not arr.flags.writeable
    assert arr.tolist() == [10, 20, 30]

    ids.append(40)
    assert np.asarray(ids).tolist() == [10, 20, 30, 40]
    ids[1] = 99
    assert np.asarray(ids, dtype=np.int64).tolist() == [10, 99, 30, 40]
    assert pickle.loads(pickle.dumps(ids)) == [10, 99, 30, 40]


def test_cache_prompt_rows_skips_multimodal_and_caches_text():
    token_ids_cpu = np.zeros((2, 8), dtype=np.int32)
    token_ids_cpu[0, :4] = [1, 2, 3, 4]
    token_ids_cpu[1, :3] = [7, 8, 9]
    reqs = {
        "text": SimpleNamespace(prompt_token_ids=[1, 2, 3, 4], mm_features=[]),
        "vlm": SimpleNamespace(prompt_token_ids=[7, 8, 9], mm_features=[object()]),
    }
    batch = SimpleNamespace(
        req_id_to_index={"text": 0, "vlm": 1}, token_ids_cpu=token_ids_cpu
    )
    cache_prompt_rows(reqs, batch, ["text", "vlm", "missing"])
    assert isinstance(reqs["text"].prompt_token_ids, PromptTokenIds)
    assert reqs["text"].prompt_token_ids._array is None
    assert not isinstance(reqs["vlm"].prompt_token_ids, PromptTokenIds)
    token_ids_cpu[0, 0] = -1
    arr = np.asarray(reqs["text"].prompt_token_ids)
    assert arr.tolist() == [1, 2, 3, 4]
    assert reqs["text"].prompt_token_ids._array is arr


def test_token_padding_state_skips_unchanged_upload():
    state = TokenPaddingState.create(8, device="cpu")
    with patch.object(
        torch.Tensor, "copy_", wraps=state.local_padding_mask.copy_
    ) as spy:
        state.update(3, 6)
        state.update(3, 6)
        state.update(6, 6)
        state.update(8, 4)  # empty slice normalizes identically to (0, 0)
    assert spy.call_count == 2
    assert state.get_local_padding_mask(8).tolist() == [False] * 8


def test_upcast_norm_weights_converts_learned_rmsnorm_to_fp32():
    class TinyBlock(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = RMSNorm(16, eps=1e-6)
            self.norm.weight = torch.nn.Parameter(
                torch.ones(16, dtype=torch.bfloat16) * 0.5
            )
            self.linear = torch.nn.Linear(16, 16, bias=False, dtype=torch.bfloat16)

    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        block = TinyBlock()
    runner_utils.upcast_norm_weights(block)
    assert block.norm.weight.dtype == torch.float32
    assert block.linear.weight.dtype == torch.bfloat16


def test_upload_unless_unchanged_handles_bf16_and_retains_in_flight_source():
    runner = TPUModelRunner.__new__(TPUModelRunner)
    runner.device = torch.device("cpu")
    runner._h2d_reuse = {}
    runner._unfenced_h2d_sources = []

    scratch = torch.tensor([1.0, 0.5], dtype=torch.bfloat16)
    first = runner._upload_unless_unchanged("temp", scratch)
    second = runner._upload_unless_unchanged("temp", scratch.clone())
    assert first is second
    assert len(runner._unfenced_h2d_sources) == 1
    first_src = runner._unfenced_h2d_sources[0][0]

    scratch[0] = 2.0
    third = runner._upload_unless_unchanged("temp", scratch)
    assert third is not first
    assert len(runner._unfenced_h2d_sources) == 2
    assert runner._unfenced_h2d_sources[0][0] is first_src
    assert first[0].item() == 1.0
    assert third[0].item() == 2.0


def test_stage_block_table_uploads_reuses_and_invalidates_on_change():
    cpu_table = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]], dtype=torch.int32)
    bt_obj = SimpleNamespace(get_cpu_tensor=lambda: cpu_table)
    builder = SimpleNamespace(
        kv_cache_group_id=0,
        is_mamba_group=False,
        _block_table_geometry=lambda _ctx: (bt_obj, 4),
    )
    group = SimpleNamespace(metadata_builders=[builder])
    uploads: list[torch.Tensor] = []

    runner = SimpleNamespace(
        attn_groups=[[group]],
        _reuse_staged_block_tables=True,
        _staged_block_table_reuse={},
        _unified_kv_layout=False,
        device=torch.device("cpu"),
    )

    def _ctx() -> AttentionMetadataBuilderContext:
        ctx = MagicMock(spec=AttentionMetadataBuilderContext)
        ctx.position_ids_override = None
        ctx.start_index = 0
        ctx.num_reqs = 2
        ctx.mamba_state_indices = None
        ctx.staged_block_tables = None
        return ctx

    with (
        patch(
            "vllm_torchtpu.layers.core.attention_metadata.zeros_h2d_source",
            side_effect=lambda _r, n: torch.zeros(n, dtype=torch.int32),
        ),
        patch(
            "vllm_torchtpu.layers.core.attention_metadata.stage_h2d_upload",
            side_effect=lambda _r, t: uploads.append(t.clone()) or t.clone(),
        ),
    ):
        ctx1 = _ctx()
        stage_block_table_uploads(runner, ctx1, num_reqs_padded=4)
        ctx2 = _ctx()
        stage_block_table_uploads(runner, ctx2, num_reqs_padded=4)
        assert len(uploads) == 1
        assert ctx2.staged_block_tables[0] is ctx1.staged_block_tables[0]

        cpu_table[1, 3] = 6
        ctx3 = _ctx()
        stage_block_table_uploads(runner, ctx3, num_reqs_padded=4)
        assert len(uploads) == 2
        assert not torch.equal(ctx3.staged_block_tables[0], ctx1.staged_block_tables[0])


def test_build_padded_sampling_params_isolates_sampling_fence():
    runner = TPUModelRunner.__new__(TPUModelRunner)
    runner.device = torch.device("cpu")
    runner.vocab_size = 128
    runner._h2d_reuse = {}
    runner._unfenced_h2d_sources = []
    runner._input_staging_fence = "pre_forward_fence"
    runner._sampling_h2d_in_flight = {}
    runner._steady_sampling_h2d_reuse = False
    runner.input_batch = SimpleNamespace(
        temperature_cpu_tensor=torch.tensor([0.7, 0.8], dtype=torch.float32),
        top_k_cpu_tensor=torch.tensor([32, 128], dtype=torch.int32),
        top_p_cpu_tensor=torch.tensor([0.9, 1.0], dtype=torch.float32),
    )
    logits = torch.zeros((4, 128), dtype=torch.float32)
    t, k, p = runner._build_padded_sampling_params(0, 2, logits)
    assert runner._unfenced_h2d_sources == []
    assert runner._input_staging_fence == "pre_forward_fence"
    assert k[:, 0].tolist() == [32, 0, 0, 0]
    assert t.shape == (4, 1) and p.shape == (4, 1)


def test_most_model_len_accounts_for_cached_prefix_tokens():
    runner = TPUModelRunner.__new__(TPUModelRunner)
    runner.most_model_len = 8192
    runner.num_reqs_most_model_len = 4
    runner.num_reqs_max_model_len = 1
    runner.input_batch = SimpleNamespace(
        num_reqs=2,
        req_ids=["r0", "r1"],
        num_computed_tokens_cpu=np.array([8000, 0], dtype=np.int32),
    )
    sched = SimpleNamespace(
        total_num_scheduled_tokens=257,
        num_scheduled_tokens={"r0": 256, "r1": 1},
    )
    assert _req_exceeds_most_model_len(runner, 0, 256)
    assert not _req_exceeds_most_model_len(runner, 1, 1)
    assert runner._count_input_chunks(sched) == (2, 1)
