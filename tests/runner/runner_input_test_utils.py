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
from vllm.config.reasoning import ReasoningConfig
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec
from vllm.v1.worker.gpu_input_batch import CachedRequestState

from vllm_torchtpu.layers.adapter.sample.rejection_sampler import RejectionSampler
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.layers.core.sequence_layout import create_sequence_layout_planner
from vllm_torchtpu.runner import min_tokens
from vllm_torchtpu.runner.thinking_budget_tracker import ThinkingBudgetTracker
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.structured_decoding.structured_decoding_manager import (
    StructuredDecodingManager,
)


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
    runner.max_num_local_reqs = max_num_reqs
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
    runner._pp_handoff = None
    runner._adaptive_verification = None
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
    runner.pcp_manager = None
    runner.get_pcp_planner = types.MethodType(TPUModelRunner.get_pcp_planner, runner)
    runner.cache_config = SimpleNamespace(
        block_size=block_size, num_gpu_blocks=8, num_gpu_blocks_override=None
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
    # _dummy_run records one itself (see _fake_tpu_events).
    runner._input_staging_fence = None
    runner._unfenced_h2d_sources = []
    runner._fenced_h2d_sources = []
    runner._free_h2d_sources = {}
    runner._registered_h2d_sources = set()
    runner._wait_input_staging_fence = types.MethodType(
        TPUModelRunner._wait_input_staging_fence, runner
    )
    runner._record_input_staging_fence = types.MethodType(
        TPUModelRunner._record_input_staging_fence, runner
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


def _fake_tpu_events(monkeypatch):
    """Stand in for `torch.tpu.Event` so a CPU test can record the staging fence."""
    event = SimpleNamespace(record=lambda: None, synchronize=lambda: None)
    monkeypatch.setattr(
        torch, "tpu", SimpleNamespace(Event=lambda: event), raising=False
    )


def _scheduler_output(scheduled_tokens):
    return SimpleNamespace(
        total_num_scheduled_tokens=sum(scheduled_tokens),
        num_scheduled_tokens={
            f"req{i}": int(n) for i, n in enumerate(scheduled_tokens)
        },
        scheduled_spec_decode_tokens={},
    )


def _make_sample_runner(
    reasoning_cfg: ReasoningConfig,
    req_ids: list[str],
    requests: dict[str, CachedRequestState],
    *,
    vocab_size: int = 256,
    max_num_reqs: int = 4,
    all_greedy: bool = True,
    async_scheduling: bool = False,
    top_tokens: bool = False,
    spec_k: int | None = None,
    spec_method: str = "ngram",
) -> SimpleNamespace:
    """Stand-in ``TPUModelRunner`` for ``sample_tokens`` and its chunk samplers.

    Binds the real sampling methods (``_sample_chunk_logits``,
    ``_sample_plain_chunk``, ``_sample_spec_verify_chunk``, min_tokens, async
    output, verify gathers) to a CPU ``SimpleNamespace`` with real
    ``MinTokensTracker`` and ``ThinkingBudgetTracker`` instances fed from
    ``requests``.
    """
    tracker = ThinkingBudgetTracker(
        reasoning_cfg, max_num_reqs=max_num_reqs, device=torch.device("cpu")
    )
    min_tracker = min_tokens.MinTokensTracker()
    for rid, rstate in requests.items():
        tracker.add(rid, rstate.sampling_params, rstate.prompt_token_ids)
        min_tracker.add(rid, rstate.sampling_params)

    spec_cfg = (
        None
        if spec_k is None
        else SimpleNamespace(
            method=spec_method,
            num_speculative_tokens=spec_k,
            draft_tensor_parallel_size=None,
            rejection_sample_method="strict",
            synthetic_acceptance_rates=None,
        )
    )
    is_eagle3 = spec_k is not None and spec_method == "eagle3"
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        model_config=SimpleNamespace(max_logprobs=2, seed=7),
        execute_model_state=None,
        input_batch=SimpleNamespace(
            max_num_logprobs=None,
            num_reqs=len(req_ids),
            req_ids=list(req_ids),
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            all_greedy=all_greedy,
            no_top_k=True,
            no_top_p=True,
            generators={},
            num_tokens_no_spec=[
                len(requests[rid].prompt_token_ids)
                + len(requests[rid].output_token_ids)
                for rid in req_ids
            ],
            token_ids_cpu=np.zeros((max(len(req_ids), 1), 64), dtype=np.int32),
        ),
        requests=requests,
        num_prompt_logprobs={},
        _pre_async_results=None,
        _dp_step_other_decode=False,
        _min_tokens=min_tracker,
        _thinking_budget=tracker,
        _greedy_head_owner=object() if top_tokens else None,
        speculative_config=spec_cfg,
        rejection_sampler=(
            None
            if spec_cfg is None
            else RejectionSampler(spec_cfg, torch.device("cpu"))
        ),
        drafter=SimpleNamespace(bucketed_drafts=False),
        spec_decode_manager=SimpleNamespace(
            propose_draft_token_ids=(
                (lambda **_kw: torch.tensor([[41, 42]], dtype=torch.int32))
                if is_eagle3
                else (lambda **_kw: None)
            ),
            stage_draft_token_ids_for_host=lambda *_a, **_kw: None,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        _is_async_drafter=is_eagle3,
        _async_drafter=SimpleNamespace(draft_chunks=None, bucketed_drafts=False),
        _adaptive_verification=None,
        _mtp_prefill_only=False,
        _tp_rank0_mask=None,
        _fence_spec_kv_under_token_copy=False,
        _is_dspark_pd_producer=False,
        is_pooling_model=False,
        max_num_reqs=max_num_reqs,
        num_tokens_paddings=[16, 32, 64, 128],
        vocab_size=vocab_size,
        _get_sampling_generator=lambda: torch.Generator().manual_seed(7),
        _greedy_sample=lambda x: torch.argmax(x, dim=-1, keepdim=True),
        _sampler_topk_candidates=lambda *_a: 0,
        _update_mamba_slot_read_offsets=lambda *_args: None,
        _update_num_xla_graphs=lambda *_args: None,
        _drafts_per_chunk=TPUModelRunner._drafts_per_chunk,
        _stage_timing=MagicMock(),
    )
    runner._stage_timing.take_samples.return_value = ([], [])
    for fn in (
        TPUModelRunner._mask_min_tokens,
        TPUModelRunner._min_tokens_batch_state,
        TPUModelRunner._settle_draft_placeholders_for_min_tokens,
        TPUModelRunner._num_output_tokens,
        TPUModelRunner._greedy_top_tokens_this_step,
        TPUModelRunner._apply_temperature,
        TPUModelRunner._sampler_applies_topk_topp,
        TPUModelRunner._modify_prev_results,
        TPUModelRunner._update_placeholder,
        TPUModelRunner._sync_replicated_drafts_across_tp,
        TPUModelRunner._sync_one_replicated_draft,
        TPUModelRunner._scheduled_draft_counts,
        TPUModelRunner._assemble_async_spec_substitution,
        TPUModelRunner._assemble_async_prefill_bootstrap,
        TPUModelRunner._sample_chunk_logits,
        TPUModelRunner._sample_plain_chunk,
        TPUModelRunner._sample_spec_verify_chunk,
        TPUModelRunner.spec_bonus_and_target_logits,
        TPUModelRunner.spec_bonus_and_target_tokens,
        TPUModelRunner.spec_gather_bonus_and_target_logits,
        TPUModelRunner.spec_logprob_row_tokens,
        TPUModelRunner.gather_logprobs,
        TPUModelRunner.maybe_wait_for_kv_save,
        TPUModelRunner.get_finished_kv_transfers,
        TPUModelRunner._build_kv_connector_output,
    ):
        setattr(runner, fn.__name__, fn.__get__(runner))
    runner.sample_from_logits_func = TPUModelRunner.sample_from_logits.__get__(runner)
    runner.structured_decoding_manager = StructuredDecodingManager(
        runner  # type: ignore[arg-type]
    )
    return runner
