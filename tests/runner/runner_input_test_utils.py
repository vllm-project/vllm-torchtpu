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

import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.layers.core.sequence_layout import create_sequence_layout_planner
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


class _BlockTable:
    def __init__(self, table: torch.Tensor):
        self._table = table
        self.max_num_blocks_per_req = table.shape[1]

    def get_cpu_tensor(self):
        return self._table


def _make_runner(
    *,
    num_computed_tokens,
    prompt_tokens,
    scheduled_tokens,
    token_paddings,
    uses_mrope=False,
    mtp_enabled=False,
    pcp_size=2,
    max_num_reqs=8,
    block_size=16,
):
    num_reqs = len(scheduled_tokens)
    max_model_len = max(64, max(prompt_tokens) + 16)
    target_num_reqs = max_num_reqs
    max_num_tokens = max(token_paddings)

    runner = SimpleNamespace()
    runner._stage_timing = MagicMock()
    runner.device = torch.device("cpu")
    runner.most_model_len = None
    runner.num_reqs_max_model_len = target_num_reqs
    runner.num_reqs_most_model_len = None
    runner.num_tokens_paddings = list(token_paddings)
    runner.max_num_reqs = max_num_reqs
    runner.max_num_tokens = max_num_tokens
    runner.max_model_len = max_model_len
    runner.block_size = block_size
    runner.uses_mrope = uses_mrope
    runner.supports_mm_inputs = False
    runner.lora_config = None
    runner.speculative_config = None
    runner.reorder_batch_threshold = 1
    runner._pcp_enabled = pcp_size > 1
    runner._mtp_enabled = mtp_enabled
    runner._mtp_prefill_only = mtp_enabled and pcp_size > 1
    runner._pp_is_first = True
    runner._pp_is_last = True
    runner._pp_wave = None
    runner.scheduler_config = SimpleNamespace(async_scheduling=False)
    runner.parallel_config = SimpleNamespace(
        prefill_context_parallel_size=pcp_size,
        cp_kv_cache_interleave_size=block_size,
        decode_context_parallel_size=1,
        pipeline_parallel_size=1,
    )
    runner.vllm_config = SimpleNamespace(
        parallel_config=runner.parallel_config,
        scheduler_config=runner.scheduler_config,
        speculative_config=None,
        kv_transfer_config=None,
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    runner.sequence_layout_planner = create_sequence_layout_planner(runner.vllm_config)
    runner.cache_config = SimpleNamespace(
        num_gpu_blocks=8, num_gpu_blocks_override=None
    )
    runner.mesh = SimpleNamespace(
        shape={
            "attn_dp": 1,
            "expert": 1,
            "model": 1,
        }
    )

    runner.input_ids_cpu = torch.zeros(max_num_tokens, dtype=torch.int32)
    runner.positions_cpu = torch.zeros(max_num_tokens, dtype=torch.int32)
    runner.positions_np = runner.positions_cpu.numpy()
    runner.query_start_loc_cpu = torch.zeros(max_num_tokens + 1, dtype=torch.int32)
    runner.query_start_loc_np = runner.query_start_loc_cpu.numpy()
    runner.seq_lens_cpu = torch.zeros(max_num_tokens, dtype=torch.int32)
    runner.seq_lens_np = runner.seq_lens_cpu.numpy()
    runner.arange_np = np.arange(max_num_tokens, dtype=np.int64)
    runner._request_distribution_cpu = torch.zeros(3, dtype=torch.int32)
    runner._decode_device_cache_key = None
    # The staging-tensor fence is recorded by execute_model; these tests
    # call _prepare_inputs directly, so the wait must find an empty fence.
    runner._input_staging_fence = None
    runner._wait_input_staging_fence = types.MethodType(
        TPUModelRunner._wait_input_staging_fence, runner
    )
    runner._cached_query_start_loc = None
    runner._cached_logits_indices = None
    runner._cached_request_distribution = None
    runner._h2d_reuse = {}
    runner._upload_unless_unchanged = types.MethodType(
        TPUModelRunner._upload_unless_unchanged, runner
    )
    runner._dp_target_bucket = None
    runner._has_mamba_state = False
    # One pipeline stage: the runner embeds its own inputs.
    runner._pp_is_first = True
    runner._pp_is_last = True
    # The real schedule check; the stub registers no attention layer, so no
    # kernel bounds the step.
    runner._attention_capacity = {}
    runner._attention_runs_batched_kernel = None
    runner._attention_schedule_capacity = types.MethodType(
        TPUModelRunner._attention_schedule_capacity, runner
    )
    runner._check_attention_schedule = types.MethodType(
        TPUModelRunner._check_attention_schedule, runner
    )
    # Spec-decode mamba rollback is inactive in these PCP layout tests; the
    # runner leaves the read-offset buffer unset (None) unless speculative
    # decoding is enabled, so _prepare_inputs / dummy_run skip the windowed
    # GDN distribution path and just pass None through to the metadata.
    runner.mamba_slot_read_offsets = None
    # Affine checkpoint addressing (no per-checkpoint blocks) in these tests.
    runner._mamba_ckpt_window = 1
    runner._unified_kv_layout = False
    # No GDN internal prefill checkpoints in these layout tests.
    runner._prefill_ckpt_units = None
    runner._prefill_ckpt_zero_offsets = {}
    # Default to layer-major KV cache (None bundle) for PCP integration layout tests.
    runner._kv_cache_bundle = None
    # Bind the real seed-copy collector; an empty copy plan makes it a no-op
    # (no mamba align mode in these layout tests).
    runner._mamba_copy_plan = []
    runner._collect_mamba_state_seed_copies = (
        TPUModelRunner._collect_mamba_state_seed_copies.__get__(runner)
    )
    runner._prepare_async_token_substitution_indices = lambda *_args, **_kwargs: (
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
    )
    runner.empty_slot_mappings = {0: torch.empty(0)}
    if uses_mrope:

        def make_buffer(rows, cols, *, dtype):
            return SimpleNamespace(cpu=torch.zeros((rows, cols), dtype=dtype))

        runner._make_buffer = make_buffer
        runner.mrope_positions = make_buffer(3, max_num_tokens + 1, dtype=torch.int32)

        def fill_mrope_positions(_scheduler_output):
            positions = torch.arange(max_num_tokens + 1, dtype=torch.int32)
            runner.mrope_positions.cpu[0] = positions
            runner.mrope_positions.cpu[1] = positions + 1000
            runner.mrope_positions.cpu[2] = positions + 2000

        runner._calc_mrope_positions = fill_mrope_positions

    token_ids_cpu = torch.zeros((num_reqs, max_model_len), dtype=torch.int32)
    for req_idx in range(num_reqs):
        token_ids_cpu[req_idx] = torch.arange(
            100 * (req_idx + 1),
            100 * (req_idx + 1) + max_model_len,
            dtype=torch.int32,
        )

    pages_per_req = max(4, (max(prompt_tokens) + block_size - 1) // block_size)
    block_tables = torch.zeros((num_reqs, pages_per_req), dtype=torch.int32)
    next_block = 0
    for req_idx, seq_len in enumerate(prompt_tokens):
        num_blocks = (
            max(seq_len, num_computed_tokens[req_idx] + scheduled_tokens[req_idx])
            + block_size
            - 1
        ) // block_size
        block_tables[req_idx, :num_blocks] = torch.arange(
            next_block, next_block + num_blocks, dtype=torch.int32
        )
        next_block += num_blocks

    input_batch = SimpleNamespace(
        num_reqs=num_reqs,
        req_ids=[f"req{i}" for i in range(num_reqs)],
        req_id_to_index={f"req{i}": i for i in range(num_reqs)},
        num_computed_tokens_cpu=np.asarray(num_computed_tokens, dtype=np.int32),
        num_prompt_tokens=np.asarray(prompt_tokens, dtype=np.int32),
        token_ids_cpu=token_ids_cpu.numpy(),
        token_ids_cpu_tensor=token_ids_cpu,
        block_table=[_BlockTable(block_tables)],
    )
    runner.input_batch = input_batch

    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        page_size_padded=block_size * 128 * 2,
    )
    runner.kv_cache_config = SimpleNamespace(
        num_blocks=8,
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["layer.0"], kv_cache_spec=spec)],
        has_mamba_layers=True,
    )

    def fake_build_attention_metadata(**_kwargs):
        ctx = runner._attn_metadata_builder_ctx
        md = AttentionMetadata(
            input_positions=(
                ctx.position_ids_override
                if ctx.position_ids_override is not None
                else runner.position_ids
            ),
            block_tables=torch.zeros((target_num_reqs * 4,), dtype=torch.int32),
            seq_lens=ctx.seq_lens,
            query_start_loc=ctx.query_start_loc,
            request_distribution=ctx.request_distribution,
            mamba_state_indices=ctx.mamba_state_indices,
            sequence_layout_kind=ctx.sequence_layout_descriptor.kind.value,
            sequence_layout_protocol=ctx.sequence_layout_descriptor.protocol,
            sequence_layout_version=ctx.sequence_layout_descriptor.version,
        )
        return {"layer.0": md}, None

    runner._build_attention_metadata = fake_build_attention_metadata
    return runner


def _scheduler_output(scheduled_tokens):
    return SimpleNamespace(
        total_num_scheduled_tokens=sum(scheduled_tokens),
        num_scheduled_tokens={
            f"req{i}": int(n) for i, n in enumerate(scheduled_tokens)
        },
        scheduled_spec_decode_tokens={},
    )
