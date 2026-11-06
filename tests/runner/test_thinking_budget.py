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
"""Tests for per-request ``thinking_token_budget`` on ``TPUModelRunner``."""

import inspect
from collections.abc import Iterable
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm import SamplingParams
from vllm.config.reasoning import ReasoningConfig
from vllm.sampling_params import StructuredOutputsParams
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.logits_processor.interface import BatchUpdate
from vllm.v1.sample.thinking_budget_state import ThinkingBudgetStateHolder
from vllm.v1.worker import gpu_model_runner
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

from vllm_torchtpu.layers.adapter.sample import thinking_budget as tb_kernels
from vllm_torchtpu.layers.adapter.sample.rejection_sampler import RejectionSampler
from vllm_torchtpu.layers.adapter.sample.thinking_budget import (
    COL_IN_THINK,
    COL_THINK_COUNT,
    INT32_MAX,
    NO_FORCED_TOKEN,
    UNTRACKED_STATE,
    StateTuple,
    ThinkingBudgetMarkers,
    _advance_state_tensor,
    _broadcast_amax_scatter,
    advance_plain_states,
    advance_sequence_state,
    advance_spec_states,
    advance_state,
    compute_kmp_failure,
    compute_sequence_state,
    force_bonus_logits,
    force_bonus_tokens,
    force_spec_target_logits,
    force_spec_target_tokens,
    mask_plain_logits,
    override_and_advance_plain,
    prepare_spec_verify_drafts,
    query_forced_token,
    seed_states,
)
from vllm_torchtpu.runner import greedy_top_tokens, min_tokens, thinking_budget
from vllm_torchtpu.runner.thinking_budget import (
    ThinkingBudgetStepState,
    ThinkingBudgetTracker,
)
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.runner.tpu_runner_async_output import AsyncTPUCopyState
from vllm_torchtpu.structured_decoding.structured_decoding_manager import (
    StructuredDecodingManager,
)
from vllm_torchtpu.utils import synchronize_tensors

QWEN3_START, QWEN3_END = (151667,), (151668,)
DEEPSEEK_START, DEEPSEEK_END = (128798,), (128799,)
GEMMA4_START, GEMMA4_END = (105, 2364, 107), (106, 107)


@pytest.fixture(autouse=True)
def _eager_cpu_kernels(request: pytest.FixtureRequest):
    """Run ``@torch.compile(backend="tpu")`` kernels eagerly in CPU tests."""
    if request.node.get_closest_marker("tpu_test") is not None:
        yield
        return
    with torch.compiler.set_stance("force_eager"):
        yield


def _reasoning_config(
    start_ids: list[int] | None,
    end_ids: list[int] | None,
    *,
    natural_end_ids: list[int] | None = None,
    enabled: bool = True,
) -> ReasoningConfig:
    cfg = ReasoningConfig(reasoning_parser="qwen3" if enabled else "")
    cfg._reasoning_start_token_ids = start_ids
    cfg._reasoning_end_token_ids = end_ids
    cfg._natural_reasoning_end_token_ids = (
        end_ids if natural_end_ids is None else natural_end_ids
    )
    cfg._enabled = enabled
    return cfg


def _request_state(
    req_id: str,
    prompt_token_ids: list[int],
    params: SamplingParams | None,
    output_token_ids: list[int] | None = None,
    num_computed_tokens: int | None = None,
) -> CachedRequestState:
    return CachedRequestState(
        req_id=req_id,
        prompt_token_ids=list(prompt_token_ids),
        mm_features=[],
        sampling_params=params,
        generator=None,
        block_ids=([0],),
        num_computed_tokens=(
            len(prompt_token_ids)
            if num_computed_tokens is None
            else num_computed_tokens
        ),
        output_token_ids=[] if output_token_ids is None else list(output_token_ids),
    )


def _pack_grammar_bitmask(
    vocab_size: int, allowed_rows: list[Iterable[int]]
) -> np.ndarray:
    num_words = (vocab_size + 31) // 32
    mask = np.zeros((len(allowed_rows), num_words), dtype=np.uint32)
    for row_i, toks in enumerate(allowed_rows):
        for tok in toks:
            mask[row_i, tok // 32] |= np.uint32(1) << np.uint32(tok % 32)
    return mask.view(np.int32)


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
    tracker = ThinkingBudgetTracker(reasoning_cfg, max_num_reqs=max_num_reqs)
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
        _pre_async_results=None,
        _pp_wave=None,
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
        TPUModelRunner._sample_spec_verify_chunk,
        TPUModelRunner.spec_bonus_and_target_logits,
        TPUModelRunner.spec_bonus_and_target_tokens,
        TPUModelRunner.spec_gather_bonus_and_target_logits,
        TPUModelRunner.spec_logprob_row_tokens,
        TPUModelRunner.gather_logprobs,
        TPUModelRunner.maybe_wait_for_kv_save,
        TPUModelRunner.get_finished_kv_transfers,
    ):
        setattr(runner, fn.__name__, fn.__get__(runner))
    runner.sample_from_logits_func = TPUModelRunner.sample_from_logits.__get__(runner)
    runner.structured_decoding_manager = StructuredDecodingManager(
        runner  # type: ignore[arg-type]
    )
    return runner


def _set_exec_state(
    runner: SimpleNamespace,
    logits_list: list[torch.Tensor],
    num_scheduled: dict[str, int],
    *,
    num_reqs_list: list[int] | None = None,
    md_list: list[object] | None = None,
    spec_tokens: dict[str, list[int]] | None = None,
    top_tokens: bool = False,
    has_structured: bool = False,
) -> None:
    runner.execute_model_state = SimpleNamespace(
        scheduler_output=SimpleNamespace(
            num_scheduled_tokens=num_scheduled,
            scheduled_spec_decode_tokens=spec_tokens or {},
            finished_req_ids=set(),
            has_structured_output_requests=has_structured,
        ),
        draft_chunks=[],
        logits_list=logits_list,
        num_reqs_list=num_reqs_list or [len(num_scheduled)],
        spec_decode_metadata_list=md_list or [None] * len(logits_list),
        mamba_state_indices_list=None,
        pooler_output_list=[],
        top_tokens=top_tokens,
        kv_save_nan_check=None,
    )


def _make_input_batch(
    max_model_len: int = 128,
    max_num_reqs: int = 4,
    reasoning_config: ReasoningConfig | None = None,
) -> InputBatch:
    return InputBatch(
        max_num_reqs=max_num_reqs,
        max_model_len=max_model_len,
        max_num_batched_tokens=32,
        device=torch.device("cpu"),
        vocab_size=160000,
        block_sizes=[16],
        max_num_blocks_per_req=[max_model_len // 16],
        logitsprocs=LogitsProcessors(),
        logitsprocs_need_output_token_ids=False,
        reasoning_config=reasoning_config,
    )


def _make_lifecycle_runner(
    cfg: ReasoningConfig, max_num_reqs: int = 4
) -> TPUModelRunner:
    runner = object.__new__(TPUModelRunner)
    runner.device = torch.device("cpu")
    runner.speculative_config = None
    runner.scheduler_config = SimpleNamespace(async_scheduling=True)
    runner.parallel_config = SimpleNamespace(
        data_parallel_size=1, enable_expert_parallel=False
    )
    runner.use_async_scheduling = True
    runner.use_async_spec_decode = False
    runner.requests, runner.num_prompt_logprobs = {}, {}
    runner._pre_async_results = None
    runner._last_sequence_layout_plan = None
    runner._mamba_state_pos, runner._mamba_slot_by_req_id = {}, {}
    runner._free_mamba_slots, runner._mamba_offset_seeded = [], set()
    runner._install_spec_token_room_guard = lambda: None
    runner._min_tokens = min_tokens.MinTokensTracker()
    runner._thinking_budget = ThinkingBudgetTracker(cfg, max_num_reqs=max_num_reqs)
    runner.input_batch = _make_input_batch(128, max_num_reqs=max_num_reqs)
    runner.late_interaction_runner = SimpleNamespace(
        on_requests_finished=lambda _ids: None,
        register_request=lambda _rid, _params: None,
    )
    runner.encoder_cache = {}
    runner.is_pooling_model = runner.uses_mrope = False
    runner.uses_xdrope_dim = 0
    runner._may_reorder_batch = lambda _output: None
    return runner


def _verify_md(
    drafts_per_req: list[list[int]], pad_tokens: int = 4, pad_reqs: int = 4
) -> SimpleNamespace:
    flat_drafts, tgt_idx, bonus_idx, seg, grp = [], [], [], [], []
    row_cursor = 0
    for req_i, drafts in enumerate(drafts_per_req):
        flat_drafts.extend(drafts)
        tgt_idx.extend(range(row_cursor, row_cursor + len(drafts)))
        bonus_idx.append(row_cursor + len(drafts))
        seg.extend([req_i] * len(drafts))
        grp.extend(range(len(drafts)))
        row_cursor += len(drafts) + 1
    pad = pad_tokens - len(flat_drafts)
    lengths = [len(d) for d in drafts_per_req]
    return SimpleNamespace(
        draft_token_ids=torch.tensor(flat_drafts + [0] * pad, dtype=torch.int32),
        draft_lengths=torch.tensor(
            lengths + [0] * (pad_reqs - len(lengths)), dtype=torch.int32
        ),
        draft_lengths_cpu=np.array(lengths),
        target_logits_indices=torch.tensor(tgt_idx + [0] * pad, dtype=torch.int32),
        bonus_logits_indices=torch.tensor(
            bonus_idx + [0] * (pad_reqs - len(bonus_idx)), dtype=torch.int32
        ),
        final_logits_indices=torch.arange(pad_tokens, dtype=torch.int32),
        segment_ids=torch.tensor(seg + [len(drafts_per_req)] * pad, dtype=torch.int64),
        group_indices=torch.tensor(grp + [0] * pad, dtype=torch.int32),
    )


def _run_sample_tokens(runner: SimpleNamespace, grammar_output: object = None):
    def host_copy(chunks, lengths, spec_logprobs=None):
        return AsyncTPUCopyState(
            chunks_cpu=[c.detach().cpu() for c in chunks],
            chunk_real_lens=list(lengths),
            copy_ready_event=MagicMock(),
            spec_logprobs=spec_logprobs,
        )

    with (
        patch(
            "vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group", return_value=False
        ),
        patch.object(AsyncTPUCopyState, "from_device_chunks", side_effect=host_copy),
        torch.compiler.set_stance("force_eager"),
    ):
        return TPUModelRunner.sample_tokens(
            runner,  # type: ignore[arg-type]
            grammar_output,  # type: ignore[arg-type]
        )


class TestUpstreamParityAndBoundaries:
    """Differential parity against upstream ``ThinkingBudgetStateHolder``."""

    @pytest.mark.parametrize(
        ("start_ids", "end_ids"),
        [
            (list(QWEN3_START), list(QWEN3_END)),
            (list(DEEPSEEK_START), list(DEEPSEEK_END)),
            (list(GEMMA4_START), list(GEMMA4_END)),
        ],
    )
    @pytest.mark.parametrize("budget", [0, 1, 3, None])
    @pytest.mark.parametrize("prompt_open", [True, False])
    def test_differential_parity_with_upstream(
        self,
        start_ids: list[int],
        end_ids: list[int],
        budget: int | None,
        prompt_open: bool,
    ) -> None:
        vocab_size = max(*start_ids, *end_ids, 64) + 16
        cfg = _reasoning_config(start_ids, end_ids)
        prompt = [1, 2, *start_ids] if prompt_open else [1, 2, 3]
        params = SamplingParams(temperature=0.0, thinking_token_budget=budget)

        holder = ThinkingBudgetStateHolder(cfg, 4, 0, torch.device("cpu"), False)
        v1_out: list[int] = []
        holder.sync_batch(BatchUpdate(1, (), [(0, params, list(prompt), v1_out)], ()))
        tracker = ThinkingBudgetTracker(cfg, max_num_reqs=4)
        tracker.add("req0", params, prompt)
        req_state = _request_state("req0", prompt, params)
        requests = {"req0": req_state}

        planned = (
            list(range(20, 32)) if prompt_open else [11, 12, *start_ids, *range(20, 29)]
        )
        tpu_out: list[int] = []
        for base_tok in planned:
            holder.update_state([v1_out], spec_token_ids=None)
            ref_logits = torch.zeros((1, vocab_size), dtype=torch.float32)
            ref_logits[0, base_tok] = 5.0
            ref_logits = holder.apply_to_logits(ref_logits, False, None)
            v1_out.append(int(torch.argmax(ref_logits, dim=-1).item()))

            state_table = tracker.prepare_step(torch.device("cpu"), ["req0"], requests)
            tpu_tok = base_tok
            if state_table is not None:
                assert tracker.markers is not None
                slot_ids = tracker.build_chunk_slot_ids(
                    torch.device("cpu"), ["req0"], 0, 1, padded_num_reqs=4
                )
                assert slot_ids is not None
                row_logits = torch.zeros((4, vocab_size), dtype=torch.float32)
                row_logits[0, base_tok] = 5.0
                row_logits = mask_plain_logits(
                    state_table, slot_ids, row_logits, tracker.markers.end_ids
                )
                sel = torch.argmax(row_logits, dim=-1, keepdim=True)
                state_table = advance_plain_states(
                    state_table, slot_ids, sel, *tracker.markers.kernel_args()
                )
                tracker.commit_state_table(state_table)
                tpu_tok = int(sel[0, 0].item())
            tpu_out.append(tpu_tok)
            req_state.output_token_ids.append(tpu_tok)

        if not (prompt_open and len(start_ids) > 1 and budget not in (0, None)):
            assert tpu_out == v1_out
        else:
            assert budget is not None
            prefix = [*planned[:budget], *end_ids]
            assert tpu_out == prefix + planned[len(prefix) :]

    def test_marker_resolution_and_boundary_semantics(self) -> None:
        assert ThinkingBudgetMarkers.from_reasoning_config(None) is None
        assert (
            ThinkingBudgetMarkers.from_reasoning_config(
                _reasoning_config(list(QWEN3_START), list(QWEN3_END), enabled=False)
            )
            is None
        )
        tracker_disabled = ThinkingBudgetTracker(_reasoning_config([], []), 4)
        with patch.object(thinking_budget.logger, "warning_once") as warn:
            tracker_disabled.add("plain", SamplingParams(), [1])
            warn.assert_not_called()
            tracker_disabled.add("b0", SamplingParams(thinking_token_budget=2), [1])
            warn.assert_called_once()
        assert not tracker_disabled

        # Distinct natural_end_ids vs forced end_ids & overlapping KMP fallback.
        assert compute_kmp_failure((10, 10, 11)) == (0, 0, 1, 0)
        distinct = ThinkingBudgetMarkers.from_reasoning_config(
            _reasoning_config([90], [92, 93, 94], natural_end_ids=[91])
        )
        assert distinct is not None
        st_nat = compute_sequence_state([1, 90, 10, 91, 20, 21], 2, distinct)
        assert st_nat[COL_IN_THINK] == 0 and query_forced_token(st_nat, distinct) == -1
        st_force = compute_sequence_state([1, 90, 10, 11, 92], 2, distinct)
        assert query_forced_token(st_force, distinct) == 93
        st_force = advance_state(advance_state(st_force, 93, distinct), 94, distinct)
        assert st_force[COL_IN_THINK] == 0

        # Partial forced-end prefix in prompt, multi-turn reopen, clamping & -1.
        gemma = ThinkingBudgetMarkers.from_reasoning_config(
            _reasoning_config(list(GEMMA4_START), list(GEMMA4_END))
        )
        assert gemma is not None
        st_partial = compute_sequence_state(
            [1, *GEMMA4_START, 20, 21, GEMMA4_END[0]], 3, gemma
        )
        assert query_forced_token(st_partial, gemma) == GEMMA4_END[1]
        st_reopen = compute_sequence_state(
            [1, *GEMMA4_START, 10, *GEMMA4_END, 13, *GEMMA4_START, 14, 15, 16],
            3,
            gemma,
        )
        assert query_forced_token(st_reopen, gemma) == GEMMA4_END[0]

        tracker = ThinkingBudgetTracker(
            _reasoning_config(list(QWEN3_START), list(QWEN3_END)), max_num_reqs=4
        )
        tracker.add("huge", SamplingParams(thinking_token_budget=2**40), [1])
        assert tracker.requests["huge"] == INT32_MAX
        neg = SamplingParams()
        neg.thinking_token_budget = -1
        tracker.add("neg", neg, [1])
        assert "neg" not in tracker.requests

    @pytest.mark.parametrize("end_ids", [[151668, 198], [151668, 198, 198]])
    def test_natural_end_prefix_of_end_ids_in_multiturn_prompt(
        self, end_ids: list[int]
    ) -> None:
        """When ``natural_end_ids`` is a proper prefix of ``end_ids`` (e.g. Qwen3
        ``[151668]`` vs ``[151668, 198]``), an earlier turn exceeding ``budget``
        that closed with ``151668`` + non-``198`` answer tokens must exit
        ``in_think`` and still allow a subsequent turn's ``<think>`` to reopen."""
        markers = ThinkingBudgetMarkers.from_reasoning_config(
            _reasoning_config([151667], end_ids, natural_end_ids=[151668])
        )
        assert markers is not None
        kernel_args = markers.kernel_args()

        # Turn 1 exceeds budget=2 (3 think tokens: 10, 11, 12), emits 151668,
        # then emits answer tokens 50, 51, 52 without trailing 198.
        turn1 = [1, 151667, 10, 11, 12, 151668, 50, 51, 52]
        st_t1 = compute_sequence_state(turn1, 2, markers)
        assert st_t1[COL_IN_THINK] == 0
        assert query_forced_token(st_t1, markers) == NO_FORCED_TOKEN

        # Tensor KMP step-by-step replay matches host replay identically.
        t_st = torch.tensor([[2, 0, 0, 0, 0, 0, 0]], dtype=torch.int32)
        for tok in turn1:
            t_st = _advance_state_tensor(
                t_st, torch.tensor([tok], dtype=torch.int32), *kernel_args
            )
        assert tuple(t_st[0].tolist()) == st_t1

        # Turn 2 reopens <think> and hits budget=2 after 20, 21 -> forces end_ids.
        multiturn = [*turn1, 60, 151667, 20, 21]
        st_t2 = compute_sequence_state(multiturn, 2, markers)
        assert st_t2[COL_IN_THINK] == 1 and st_t2[COL_THINK_COUNT] == 2
        assert query_forced_token(st_t2, markers) == end_ids[0]
        for expected_tok in end_ids:
            assert query_forced_token(st_t2, markers) == expected_tok
            st_t2 = advance_state(st_t2, expected_tok, markers)
        assert st_t2[COL_IN_THINK] == 0
        assert query_forced_token(st_t2, markers) == NO_FORCED_TOKEN

        # Back-to-back boundary: 151668 immediately followed by 151667 (or multi-token
        # start prefix after partial end_ids) reopens thinking without dropping start.
        st_adj = compute_sequence_state(
            [151667, 10, 11, 12, 151668, 151667, 30], 2, markers
        )
        assert st_adj[COL_IN_THINK] == 1 and st_adj[COL_THINK_COUNT] == 1

        multi_start = ThinkingBudgetMarkers.from_reasoning_config(
            _reasoning_config([105, 107], [151668, 198, 198], natural_end_ids=[151668])
        )
        assert multi_start is not None
        st_shadow = compute_sequence_state(
            [105, 107, 10, 11, 12, 151668, 198, 105, 107, 20], 2, multi_start
        )
        assert st_shadow[COL_IN_THINK] == 1 and st_shadow[COL_THINK_COUNT] == 1


class TestKMPKernelsFuzz:
    """Randomized KMP parity across host sequences, plain decode, and spec verify."""

    @pytest.mark.parametrize(
        ("start_ids", "end_ids", "nat_ids"),
        [
            ([10], [11], [11]),
            ([105, 2364, 107], [106, 107], [106, 107]),
            ([10, 10, 11], [20, 20, 21], [20, 20, 21]),
            ([10, 20], [30, 31, 32], [29]),
            ([151667], [151668, 198], [151668]),
        ],
    )
    def test_randomized_kmp_fuzz_plain_and_spec_verify(
        self, start_ids: list[int], end_ids: list[int], nat_ids: list[int]
    ) -> None:
        markers = ThinkingBudgetMarkers.from_reasoning_config(
            _reasoning_config(start_ids, end_ids, natural_end_ids=nat_ids)
        )
        assert markers is not None
        kernel_args = markers.kernel_args()
        rng = np.random.default_rng(12345)
        vocab_pool = list({*start_ids, *end_ids, *nat_ids, 1, 2, 3, 4, 5})

        for length in (0, 7, 128, 2048):
            seq = (
                [int(x) for x in rng.choice(vocab_pool, size=length)] if length else []
            )
            ref: StateTuple = (3, 0, 0, 0, 0, 0, 0)
            for tok in seq:
                ref = advance_state(ref, tok, markers)
            assert advance_sequence_state((3, 0, 0, 0, 0, 0, 0), seq, markers) == ref

        state_table = torch.zeros((9, 7), dtype=torch.int32)
        state_table[:, 0] = -1
        budgets = {1: 0, 2: 1, 3: 2, 4: 4, 5: 0, 6: 3}
        scalar_states: dict[int, StateTuple] = {0: UNTRACKED_STATE}
        seed_slots = torch.zeros(6, dtype=torch.int32)
        seed_vals = torch.zeros((6, 7), dtype=torch.int32)
        for i, (slot, b) in enumerate(budgets.items()):
            st = compute_sequence_state(start_ids if slot % 2 else [1, 2], b, markers)
            scalar_states[slot], seed_slots[i] = st, slot
            seed_vals[i] = torch.tensor(st, dtype=torch.int32)
        state_table = seed_states(state_table, seed_slots, seed_vals)

        vocab_sz = max(vocab_pool) + 8
        for _ in range(30):
            chosen_slots = [1, 2, 3, 0, 4, 5, 6, 0]
            rng.shuffle(chosen_slots)
            slot_t = torch.tensor(chosen_slots, dtype=torch.int32)
            raw_choices = [int(rng.choice(vocab_pool)) for _ in range(8)]
            sampled = torch.tensor([[c] for c in raw_choices], dtype=torch.int64)
            plain_logits = torch.full((8, vocab_sz), -20.0)
            for r_i, c in enumerate(raw_choices):
                plain_logits[r_i, c] = 5.0

            expected_out: list[int] = []
            for row_i, slot in enumerate(chosen_slots):
                raw_tok = raw_choices[row_i]
                if slot == 0:
                    expected_out.append(raw_tok)
                else:
                    st = scalar_states[slot]
                    forced = query_forced_token(st, markers)
                    chosen = forced if forced >= 0 else raw_tok
                    expected_out.append(chosen)
                    scalar_states[slot] = advance_state(st, chosen, markers)

            masked_l = mask_plain_logits(
                state_table, slot_t, plain_logits, markers.end_ids
            )
            sampled_from_logits = torch.argmax(masked_l, dim=-1, keepdim=True)
            next_tbl_logits = advance_plain_states(
                state_table, slot_t, sampled_from_logits, *kernel_args
            )
            got, state_table = override_and_advance_plain(
                state_table, slot_t, sampled, *kernel_args
            )
            assert sampled_from_logits.squeeze(-1).tolist() == expected_out
            assert got.squeeze(-1).tolist() == expected_out
            assert torch.equal(next_tbl_logits, state_table)
            assert tuple(state_table[0].tolist()) == UNTRACKED_STATE
            for slot, st in scalar_states.items():
                assert tuple(state_table[slot].tolist()) == st

        k, padded_tokens = 3, 16
        for _ in range(20):
            picked = [
                int(x) for x in rng.choice([1, 2, 3, 4, 5, 6], size=3, replace=False)
            ]
            real_slots = [*picked, 0]
            rng.shuffle(real_slots)
            active_slots = [*real_slots, 0, 0]
            slot_tensor = torch.tensor(active_slots, dtype=torch.int32)
            draft_np = np.full(padded_tokens, 77, dtype=np.int32)
            seg_np = np.full(padded_tokens, 4, dtype=np.int64)
            grp_np = np.zeros(padded_tokens, dtype=np.int32)
            exp_forced = np.full(padded_tokens, NO_FORCED_TOKEN, dtype=np.int32)
            exp_bonus_forced = np.full(4, NO_FORCED_TOKEN, dtype=np.int32)
            exp_eff = draft_np.copy()
            cursor = 0
            for req_i in range(4):
                slot = active_slots[req_i]
                st_cur = None if slot == 0 else scalar_states[slot]
                for g_i in range(int(rng.integers(0, k + 1))):
                    tok = int(rng.choice(vocab_pool))
                    draft_np[cursor], seg_np[cursor], grp_np[cursor] = tok, req_i, g_i
                    if st_cur is not None:
                        f_tok = query_forced_token(st_cur, markers)
                        exp_forced[cursor] = f_tok
                        exp_eff[cursor] = -1 if f_tok >= 0 and tok != f_tok else tok
                        st_cur = advance_state(st_cur, tok, markers)
                    else:
                        exp_eff[cursor] = tok
                    cursor += 1
                if st_cur is not None:
                    exp_bonus_forced[req_i] = query_forced_token(st_cur, markers)

            draft_t = torch.from_numpy(draft_np)
            forced_tgt, forced_bonus = prepare_spec_verify_drafts(
                state_table,
                slot_tensor,
                draft_t,
                torch.from_numpy(seg_np),
                torch.from_numpy(grp_np),
                k,
                *kernel_args,
            )
            raw_logits = torch.zeros((padded_tokens, vocab_sz), dtype=torch.float32)
            raw_logits[:, 2] = 4.0
            flogits, eff_l = force_spec_target_logits(raw_logits, draft_t, forced_tgt)
            ftokens, eff_t = force_spec_target_tokens(
                torch.full((padded_tokens,), 2, dtype=torch.int64),
                draft_t,
                forced_tgt,
            )
            raw_bonus_logits = torch.zeros((4, vocab_sz), dtype=torch.float32)
            raw_bonus_logits[:, 3] = 4.0
            fbonus_l = force_bonus_logits(raw_bonus_logits, forced_bonus)
            fbonus_t = force_bonus_tokens(
                torch.full((4,), 3, dtype=torch.int64), forced_bonus
            )
            assert eff_l.tolist() == exp_eff.tolist() == eff_t.tolist()
            assert torch.argmax(flogits, dim=-1).tolist() == ftokens.tolist()
            assert ftokens.tolist() == [
                int(f) if f >= 0 else 2 for f in exp_forced.tolist()
            ]
            assert torch.argmax(fbonus_l, dim=-1).tolist() == fbonus_t.tolist()
            assert fbonus_t.tolist() == [
                int(f) if f >= 0 else 3 for f in exp_bonus_forced.tolist()
            ]

            next_toks = torch.full((4, k + 1), -1, dtype=torch.int32)
            for req_i in range(4):
                slot = active_slots[req_i]
                for col_i in range(int(rng.integers(1, k + 2))):
                    tok_c = int(rng.choice(vocab_pool))
                    next_toks[req_i, col_i] = tok_c
                    if slot != 0:
                        scalar_states[slot] = advance_state(
                            scalar_states[slot], tok_c, markers
                        )
            state_table = advance_spec_states(
                state_table, slot_tensor, next_toks, k, *kernel_args
            )
            for slot, st in scalar_states.items():
                assert tuple(state_table[slot].tolist()) == st

    def test_logits_immutability_neg_inf_guard_and_amax_sentinel(self) -> None:
        cfg = _reasoning_config(list(GEMMA4_START), list(GEMMA4_END))
        tracker = ThinkingBudgetTracker(cfg, max_num_reqs=512)
        assert tracker.markers is not None
        reqs: dict[str, CachedRequestState] = {}
        batch_ids = [f"r_{i}" for i in range(300)]
        for i, rid in enumerate(batch_ids):
            sp = SamplingParams(thinking_token_budget=None if i % 3 == 0 else 0)
            reqs[rid] = _request_state(rid, [1, *GEMMA4_START], sp)
            tracker.add(rid, sp, [1, *GEMMA4_START])
        table = tracker.prepare_step(torch.device("cpu"), batch_ids, reqs)
        slots = tracker.build_chunk_slot_ids(
            torch.device("cpu"), batch_ids, 0, 300, 512
        )
        assert table is not None and slots is not None and slots[300].item() == 0
        seg = torch.full((1024,), 300, dtype=torch.int64)
        seg[0] = 1
        seg[1] = 2
        forced_tgt, _ = prepare_spec_verify_drafts(
            table,
            slots,
            torch.full((1024,), 77, dtype=torch.int32),
            seg,
            torch.zeros(1024, dtype=torch.int32),
            3,
            *tracker.markers.kernel_args(),
        )
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            logits = torch.ones((1024, 128), dtype=dtype)
            # Row 0: finite negative logit at forced token -> masked to one-hot 0.0.
            logits[0, GEMMA4_END[0]] = -10.0
            # Row 1: already -inf (e.g. disallowed by grammar/min_tokens) -> untouched.
            logits[1, GEMMA4_END[0]] = float("-inf")
            snap = logits.clone()
            out_l, eff_d = force_spec_target_logits(
                logits,
                torch.full((1024,), 77, dtype=torch.int32),
                forced_tgt,
            )
            assert torch.equal(logits, snap) and torch.equal(out_l[2:], snap[2:])
            assert out_l[0, GEMMA4_END[0]].item() == 0.0 and eff_d[0].item() == -1
            assert torch.equal(out_l[1], snap[1]) and eff_d[1].item() == 77
            assert (eff_d[2:] == 77).all()

        # _AMAX_SENTINEL == INT32_MIN preserves even INT32_MIN in active scatter rows.
        i32_min = torch.iinfo(torch.int32).min
        base = torch.zeros((4, 2), dtype=torch.int32)
        vals = torch.tensor([[i32_min, 7], [9, i32_min]], dtype=torch.int32)
        scattered = _broadcast_amax_scatter(
            base,
            torch.tensor([2, 0], dtype=torch.int32),
            vals,
            torch.tensor([True, False]),
        )
        assert scattered[2].tolist() == [i32_min, 7]
        assert scattered[0].tolist() == [0, 0]

        with pytest.raises(AssertionError, match="Duplicate active indices"):
            seed_states(
                table,
                torch.tensor([1, 1], dtype=torch.int32),
                torch.zeros((2, 7), dtype=torch.int32),
            )


class TestLifecyclePreemptionAndChunkedPrefill:
    """Lifecycle hooks, wave slot recycling, H2D reuse, and real ``_update_states``."""

    def test_update_states_lifecycle_slot_reuse_and_batch_reorder(self) -> None:
        cfg = _reasoning_config(list(QWEN3_START), list(QWEN3_END))
        tracker = ThinkingBudgetTracker(cfg, max_num_reqs=2)
        tracker.add("plain", SamplingParams(), [1, QWEN3_START[0]])
        tracker.add("pool", None, [1])
        assert tracker.enabled and not tracker and tracker._state_table is None

        sp2 = SamplingParams(thinking_token_budget=2)
        reqs = {
            rid: _request_state(rid, [1, QWEN3_START[0]], sp2)
            for rid in ("r0", "r1", "r2")
        }
        for rid in reqs:
            tracker.add(rid, sp2, [1, QWEN3_START[0]])
        assert (
            tracker.prepare_step(
                torch.device("cpu"), ["r0", "r1"], {"r0": reqs["r0"], "r1": reqs["r1"]}
            )
            is not None
        )
        # Steady-state build_chunk_slot_ids reuses the cached tensor without H2D copy.
        s1 = tracker.build_chunk_slot_ids(torch.device("cpu"), ["r0", "r1"], 0, 2, 4)
        s2 = tracker.build_chunk_slot_ids(torch.device("cpu"), ["r0", "r1"], 0, 2, 4)
        s_pad8 = tracker.build_chunk_slot_ids(
            torch.device("cpu"), ["r0", "r1"], 0, 2, 8
        )
        s_reord = tracker.build_chunk_slot_ids(
            torch.device("cpu"), ["r1", "r0"], 0, 2, 4
        )
        assert s1 is not None and s2 is s1
        assert s_pad8 is not None and s_pad8 is not s1 and s_pad8.shape == (8,)
        assert s_reord is not None and s_reord.tolist()[:2] == s1.tolist()[:2][::-1]

        with pytest.raises(AssertionError, match="slot pool exhausted"):
            tracker.prepare_step(torch.device("cpu"), ["r0", "r1", "r2"], reqs)
        assert (
            tracker.prepare_step(
                torch.device("cpu"), ["r1", "r2"], {"r1": reqs["r1"], "r2": reqs["r2"]}
            )
            is not None
        )
        assert "r0" not in tracker._slot_by_req_id and "r2" in tracker._slot_by_req_id

        # InputBatch swap_states + upstream thinking_budget_state_holder cleanup.
        batch = _make_input_batch(64, reasoning_config=cfg)
        assert batch.thinking_budget_state_holder is not None
        reinit_runner = object.__new__(TPUModelRunner)
        reinit_runner.input_batch = batch
        reinit_runner.max_num_reqs = 4
        reinit_runner.max_num_blocks_per_req = 4
        reinit_runner.block_table_cpu = torch.zeros((4, 4), dtype=torch.int32)
        reinit_runner._refresh_smem_request_caps = lambda _cfg: None
        with (
            patch.object(
                gpu_model_runner.GPUModelRunner,
                "may_reinitialize_input_batch",
                lambda self, _kv: None,
            ),
            patch(
                "vllm_torchtpu.runner.tpu_runner.install_kernel_block_tables",
                lambda *_a: None,
            ),
        ):
            reinit_runner.may_reinitialize_input_batch(
                MagicMock(), kernel_block_sizes=[16]
            )
        assert batch.thinking_budget_state_holder is None

        tb_swap = ThinkingBudgetTracker(cfg, max_num_reqs=4)
        pa = SamplingParams(thinking_token_budget=1)
        ra = _request_state("a", [1, QWEN3_START[0], 10], pa)
        rb = _request_state("b", [1, QWEN3_START[0], 10], SamplingParams())
        batch.add_request(ra)
        batch.add_request(rb)
        batch.refresh_metadata()
        batch.swap_states(0, 1)
        tb_swap.add("a", pa, ra.prompt_token_ids)
        tb_swap.add("b", rb.sampling_params, rb.prompt_token_ids)
        st = tb_swap.prepare_step(
            torch.device("cpu"), batch.req_ids[:2], {"a": ra, "b": rb}
        )
        slots = tb_swap.build_chunk_slot_ids(
            torch.device("cpu"), batch.req_ids, 0, 2, 4
        )
        assert st is not None and slots is not None and tb_swap.markers is not None
        out, _ = override_and_advance_plain(
            st,
            slots,
            torch.tensor([[55], [66], [0], [0]], dtype=torch.int64),
            *tb_swap.markers.kernel_args(),
        )
        assert out[:2, 0].tolist() == [55, QWEN3_END[0]]

    def test_more_concurrent_requests_than_max_num_reqs_across_waves(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With > ``max_num_reqs`` live budgeted requests rotating across unscheduled
        waves (``preempted_req_ids=None``), ``release_unscheduled`` frees slots on
        wave swaps and re-entering requests reseed from committed outputs."""
        monkeypatch.setattr(
            gpu_model_runner, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True)
        )
        cfg = _reasoning_config(list(GEMMA4_START), list(GEMMA4_END))
        runner = _make_lifecycle_runner(cfg, max_num_reqs=2)
        tracker = runner._thinking_budget
        assert tracker is not None and tracker.markers is not None
        prompt = [1, *GEMMA4_START]

        # Admit 5 concurrent budgeted requests across waves of size <= 2.
        budgets = {"r0": 2, "r1": 1, "r2": 0, "r3": 2, "r4": 1}

        def schedule_wave(wave_ids: list[str], new_ids: list[str]) -> None:
            so = SchedulerOutput.make_empty()
            so.num_scheduled_tokens = {
                rid: (len(prompt) if rid in new_ids else 1) for rid in wave_ids
            }
            so.total_num_scheduled_tokens = sum(so.num_scheduled_tokens.values())
            so.scheduled_new_reqs = [
                NewRequestData(
                    req_id=rid,
                    prompt_token_ids=list(prompt),
                    mm_features=[],
                    sampling_params=SamplingParams(
                        temperature=0.0, thinking_token_budget=budgets[rid]
                    ),
                    pooling_params=None,
                    block_ids=([1],),
                    num_computed_tokens=0,
                    lora_request=None,
                )
                for rid in new_ids
            ]
            cached = [rid for rid in wave_ids if rid not in new_ids]
            if cached:
                resumed = {
                    rid
                    for rid in cached
                    if rid not in runner.input_batch.req_id_to_index
                }
                so.scheduled_cached_reqs = CachedRequestData(
                    req_ids=cached,
                    resumed_req_ids=resumed,
                    new_token_ids=[],
                    all_token_ids={
                        rid: [*prompt, *runner.requests[rid].output_token_ids]
                        for rid in resumed
                    },
                    new_block_ids=[
                        ([1],) if rid in resumed else None for rid in cached
                    ],
                    num_computed_tokens=[
                        len(prompt) + len(runner.requests[rid].output_token_ids)
                        for rid in cached
                    ],
                    num_output_tokens=[
                        len(runner.requests[rid].output_token_ids) for rid in cached
                    ],
                )
            runner._update_states(so)

        def decode_wave(wave_ids: list[str], raw_toks: list[int]) -> list[int]:
            tbl = tracker.prepare_step(torch.device("cpu"), wave_ids, runner.requests)
            sl = tracker.build_chunk_slot_ids(
                torch.device("cpu"), wave_ids, 0, len(wave_ids), 2
            )
            assert tbl is not None and sl is not None and tracker.markers is not None
            sel = torch.tensor([[t] for t in raw_toks], dtype=torch.int64)
            out, tbl = override_and_advance_plain(
                tbl, sl, sel, *tracker.markers.kernel_args()
            )
            tracker.commit_state_table(tbl)
            toks = [int(out[i, 0].item()) for i in range(len(wave_ids))]
            for rid, tok in zip(wave_ids, toks):
                runner.requests[rid].output_token_ids.append(tok)
            return toks

        # Wave 1: admit r0 (budget=2), r1 (budget=1); emit 1 token each.
        schedule_wave(["r0", "r1"], ["r0", "r1"])
        assert decode_wave(["r0", "r1"], [20, 30]) == [20, 30]

        # Wave 2: evict r0, r1 implicitly (preempted_req_ids=None) & admit r2, r3.
        schedule_wave(["r2", "r3"], ["r2", "r3"])
        assert set(tracker._slot_by_req_id).isdisjoint({"r0", "r1"})
        assert decode_wave(["r2", "r3"], [40, 50]) == [GEMMA4_END[0], 50]

        # Wave 3: admit r4 (5th live request!) alongside re-entering r0.
        schedule_wave(["r4", "r0"], ["r4"])
        assert len(runner.requests) == 5 and len(tracker.requests) == 5
        # r0 already has [20] (1/2 budget used): next token [21] hits 2/2, then forces!
        assert decode_wave(["r4", "r0"], [60, 21]) == [60, 21]
        assert decode_wave(["r4", "r0"], [61, 22]) == [GEMMA4_END[0], GEMMA4_END[0]]
        assert decode_wave(["r4", "r0"], [62, 23]) == [GEMMA4_END[1], GEMMA4_END[1]]

    @pytest.mark.parametrize(
        ("explicit_preempt", "settled_outputs"),
        [(True, [20, 21]), (False, [20, 21]), (True, [])],
    )
    def test_preemption_and_resume_with_real_gpu_model_runner(
        self,
        monkeypatch: pytest.MonkeyPatch,
        explicit_preempt: bool,
        settled_outputs: list[int],
    ) -> None:
        """Run real ``GPUModelRunner._update_states`` on preempt/resume with
        unsettled ``[0]`` async placeholders in ``output_token_ids``."""
        monkeypatch.setattr(
            gpu_model_runner, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True)
        )
        # Set natural_end_ids=[0] so any surviving [0] placeholder closes thinking!
        cfg = _reasoning_config(
            list(GEMMA4_START), list(GEMMA4_END), natural_end_ids=[0]
        )
        runner = _make_lifecycle_runner(cfg, max_num_reqs=4)
        tracker = runner._thinking_budget
        assert tracker is not None

        prompt = [1, *GEMMA4_START]
        params = SamplingParams(temperature=0.0, thinking_token_budget=3)
        so_new = SchedulerOutput.make_empty()
        so_new.num_scheduled_tokens = {"p": len(prompt)}
        so_new.total_num_scheduled_tokens = len(prompt)
        so_new.scheduled_new_reqs = [
            NewRequestData(
                req_id="p",
                prompt_token_ids=list(prompt),
                mm_features=[],
                sampling_params=params,
                pooling_params=None,
                block_ids=([1],),
                num_computed_tokens=0,
                lora_request=None,
            )
        ]
        runner._update_states(so_new)

        table = tracker.prepare_step(torch.device("cpu"), ["p"], runner.requests)
        slots = tracker.build_chunk_slot_ids(torch.device("cpu"), ["p"], 0, 1, 4)
        assert table is not None and slots is not None and tracker.markers is not None
        for tok in (20, 21, 22):
            _, table = override_and_advance_plain(
                table,
                slots,
                torch.tensor([[tok], [0], [0], [0]], dtype=torch.int64),
                *tracker.markers.kernel_args(),
            )
            tracker.commit_state_table(table)
        runner.requests["p"].output_token_ids = [20, 0, 0, 0]

        so_preempt = SchedulerOutput.make_empty()
        so_preempt.preempted_req_ids = {"p"} if explicit_preempt else None
        runner._update_states(so_preempt)
        assert "p" not in runner.input_batch.req_id_to_index
        assert "p" not in tracker._slot_by_req_id

        all_toks = [*prompt, *settled_outputs]
        so_resume = SchedulerOutput.make_empty()
        so_resume.num_scheduled_tokens = {"p": len(all_toks)}
        so_resume.total_num_scheduled_tokens = len(all_toks)
        so_resume.scheduled_cached_reqs = CachedRequestData(
            req_ids=["p"],
            resumed_req_ids={"p"},
            new_token_ids=[],
            all_token_ids={"p": all_toks},
            new_block_ids=[([2],)],
            num_computed_tokens=[0],
            num_output_tokens=[len(settled_outputs)],
        )
        runner._update_states(so_resume)
        assert runner.requests["p"].output_token_ids == settled_outputs

        table = tracker.prepare_step(torch.device("cpu"), ["p"], runner.requests)
        slots = tracker.build_chunk_slot_ids(torch.device("cpu"), ["p"], 0, 1, 4)
        assert table is not None and slots is not None
        emitted: list[int] = []
        for tok in (30, 31, 32, 33, 34):
            out, table = override_and_advance_plain(
                table,
                slots,
                torch.tensor([[tok], [0], [0], [0]], dtype=torch.int64),
                *tracker.markers.kernel_args(),
            )
            tracker.commit_state_table(table)
            emitted.append(int(out[0, 0].item()))
        expected_prefix = [30, 31, 32][: 3 - len(settled_outputs)] + list(GEMMA4_END)
        assert emitted[: len(expected_prefix)] == expected_prefix

        so_resub = SchedulerOutput.make_empty()
        so_resub.finished_req_ids = {"p"}
        so_resub.num_scheduled_tokens = {"p": len(prompt)}
        so_resub.total_num_scheduled_tokens = len(prompt)
        so_resub.scheduled_new_reqs = [
            NewRequestData(
                req_id="p",
                prompt_token_ids=list(prompt),
                mm_features=[],
                sampling_params=SamplingParams(thinking_token_budget=5),
                pooling_params=None,
                block_ids=([3],),
                num_computed_tokens=0,
                lora_request=None,
            )
        ]
        runner._update_states(so_resub)
        assert dict(tracker.requests) == {"p": 5}

    def test_async_cold_reseed_and_chunked_prefill_discard(self) -> None:
        # 1. Cold reseed in sample_tokens settles in-flight [0] placeholder first.
        p0 = SamplingParams(temperature=0.0, thinking_token_budget=0)
        req0 = _request_state("r0", [5, 6], p0, output_token_ids=[0])
        runner0 = _make_sample_runner(
            _reasoning_config([0], [99]),
            ["r0"],
            {"r0": req0},
            vocab_size=128,
            async_scheduling=True,
        )
        runner0._thinking_budget._dirty_states.clear()
        runner0._thinking_budget._slot_by_req_id.clear()
        runner0.input_batch.num_tokens_no_spec[0] = 3
        runner0._pre_async_results = SimpleNamespace(
            req_ids=["r0"],
            request_seq_lens=[(0, req0, 2, "r0")],
            discard_sampled_tokens_req_indices=[],
            num_draft_per_req={},
            wait_for_copy=lambda: torch.tensor([[42]], dtype=torch.int64),
        )
        logits0 = torch.zeros((4, 128))
        logits0[0, 55] = 8.0
        _set_exec_state(runner0, [logits0], {"r0": 1})
        assert _run_sample_tokens(runner0).get_output().sampled_token_ids == [[55]]
        runner0._modify_prev_results()
        assert req0.output_token_ids == [42, 55]

        # 2. Chunked prefill (4/8 then 4/8 tokens) suppresses forcing & rewinds rng.
        prompt = [1, 2, 3, 4, 5, *GEMMA4_START]
        req = _request_state(
            "r0",
            prompt,
            SamplingParams(temperature=0.0, thinking_token_budget=0),
            num_computed_tokens=0,
        )
        runner = _make_sample_runner(
            _reasoning_config(list(GEMMA4_START), list(GEMMA4_END)), ["r0"], {"r0": req}
        )

        class _OffsetGen:
            offset = 16

            def get_offset(self) -> int:
                return self.offset

            def set_offset(self, val: int) -> None:
                self.offset = val

        gen = _OffsetGen()
        runner.input_batch.generators = {0: gen}
        logits = torch.zeros((1, 256))
        logits[0, 42] = 5.0
        emitted: list[list[int]] = []
        for scheduled in (4, 4, 1, 1):
            _set_exec_state(runner, [logits], {"r0": scheduled})
            emitted.append(_run_sample_tokens(runner).sampled_token_ids[0])
            req.num_computed_tokens += scheduled
        assert emitted == [[], [GEMMA4_END[0]], [GEMMA4_END[1]], [42]]
        assert gen.offset == 12


class TestSampleTokens:
    """Plain and speculative ``sample_tokens`` integration tests."""

    @pytest.mark.parametrize(
        "mode", ["random_logprobs_min_tokens", "greedy_top_tokens", "async_dynamic"]
    )
    def test_plain_sample_tokens(self, mode: str) -> None:
        cfg = _reasoning_config(list(GEMMA4_START), list(GEMMA4_END))
        if mode == "random_logprobs_min_tokens":
            params = SamplingParams(
                temperature=0.7,
                top_k=4,
                top_p=0.95,
                min_tokens=5,
                stop_token_ids=[99],
                thinking_token_budget=1,
                logprobs=2,
            )
            req = _request_state("r0", [1, *GEMMA4_START], params)
            runner = _make_sample_runner(cfg, ["r0"], {"r0": req}, all_greedy=False)
            runner.input_batch.no_top_k = runner.input_batch.no_top_p = False
            runner.input_batch.max_num_logprobs = 2
            runner._build_padded_sampling_params = lambda *_a: (
                torch.full((4, 1), 0.7),
                torch.full((4, 1), 4, dtype=torch.int32),
                torch.full((4, 1), 0.95),
            )
            base = torch.full((4, 256), -100.0)
            base[0, 99], base[0, 50] = 20.0, 10.0
            emitted: list[int] = []
            ranks: list[int] = []
            top1_raw_ids: list[int] = []
            for _ in range(4):
                _set_exec_state(runner, [base.clone()], {"r0": 1})
                res = _run_sample_tokens(runner)
                tok = res.sampled_token_ids[0][0]
                emitted.append(tok)
                req.num_computed_tokens += 1
                lp = res.logprobs
                assert lp is not None
                assert np.isfinite(lp.logprobs[0]).all()
                # Raw pre-mask top-1 token is always 99 (logit 20.0 > 10.0 > -100.0).
                top1_raw_ids.append(int(lp.logprob_token_ids[0, 1]))
                ranks.append(int(lp.sampled_token_ranks[0]))
            assert emitted == [50, GEMMA4_END[0], GEMMA4_END[1], 50]
            assert top1_raw_ids == [99, 99, 99, 99]
            assert ranks[0] == 2 and all(1 <= r <= 256 for r in ranks)
        elif mode == "greedy_top_tokens":
            params = SamplingParams(temperature=0.0, thinking_token_budget=2)
            req = _request_state("r0", [1, *GEMMA4_START], params)
            runner = _make_sample_runner(cfg, ["r0"], {"r0": req}, top_tokens=True)
            assert greedy_top_tokens.step_eligible(runner.input_batch, runner.requests)
            assert runner._greedy_top_tokens_this_step()
            emitted = []
            for raw_tok in (30, 31, 32, 33, 34):
                top_ids = torch.tensor([[raw_tok], [0], [0], [0]], dtype=torch.int64)
                _set_exec_state(runner, [top_ids], {"r0": 1}, top_tokens=True)
                emitted.append(_run_sample_tokens(runner).sampled_token_ids[0][0])
                req.num_computed_tokens += 1
            assert emitted == [30, 31, GEMMA4_END[0], GEMMA4_END[1], 34]
        else:
            params = SamplingParams(temperature=0.0, thinking_token_budget=2)
            req = _request_state("r0", [1, 2], params, num_computed_tokens=0)
            runner = _make_sample_runner(
                cfg, ["r0"], {"r0": req}, vocab_size=4096, async_scheduling=True
            )
            stream = [10, *GEMMA4_START, 50, 51, 52, 53, 60]
            next_tpu: list[int] = []
            for idx, raw_tok in enumerate(stream):
                logits = torch.zeros((4, 4096))
                logits[0, raw_tok] = 9.0
                sched = 2 if idx == 0 else 1
                _set_exec_state(runner, [logits], {"r0": sched})
                out = _run_sample_tokens(runner)
                assert runner._pre_async_results is not None
                next_tpu.append(
                    int(runner._pre_async_results.next_tokens_tpu[0].item())
                )
                req.output_token_ids[-1] = -1
                out.get_output()
                req.num_computed_tokens += sched
            runner._modify_prev_results()
            expected = [10, *GEMMA4_START, 50, 51, *GEMMA4_END, 60]
            assert req.output_token_ids == expected == next_tpu

    def test_structured_decoding_min_tokens_and_real_gather_logprobs(self) -> None:
        """Combine ``thinking_token_budget``, real ``StructuredDecodingManager``,
        ``min_tokens``, and real ``TPUModelRunner.gather_logprobs`` when ``end_ids``
        is first disallowed by grammar (and by ``min_tokens``) and then permitted."""
        cfg = _reasoning_config(list(GEMMA4_START), list(GEMMA4_END))
        params = SamplingParams(
            temperature=0.6,
            top_k=2,
            top_p=0.9,
            min_tokens=4,
            stop_token_ids=[98],
            thinking_token_budget=1,
            logprobs=2,
            structured_outputs=StructuredOutputsParams(regex=".*"),
        )
        req = _request_state("r0", [1, *GEMMA4_START], params)
        runner = _make_sample_runner(cfg, ["r0"], {"r0": req}, all_greedy=False)
        runner.input_batch.no_top_k = runner.input_batch.no_top_p = False
        runner.input_batch.max_num_logprobs = 2
        runner._build_padded_sampling_params = lambda *_a: (
            torch.full((4, 1), 0.6),
            torch.full((4, 1), 2, dtype=torch.int32),
            torch.full((4, 1), 0.9),
        )

        # Raw logits prefer grammar-disallowed 99 (25.0) and stop 98 (18.0);
        # grammar-valid 50 has 8.0 and GEMMA4_END tokens have -5.0.
        raw_base = torch.full((4, 256), -30.0)
        raw_base[0, 99] = 25.0
        raw_base[0, 98] = 18.0
        raw_base[0, 50] = 8.0
        raw_base[0, GEMMA4_END[0]] = -5.0
        raw_base[0, GEMMA4_END[1]] = -5.0

        # Step 0: budget not yet reached; grammar allows {50, 98}, min_tokens masks 98
        # -> samples 50.
        # Step 1: budget=1 reached, but grammar ONLY allows {50} (GEMMA4_END[0] is -inf)
        # -> forcing yields to grammar and samples 50 with finite raw logprobs!
        # Steps 2..3: grammar allows {50, GEMMA4_END[0], GEMMA4_END[1]} -> forces
        # GEMMA4_END[0] then GEMMA4_END[1].
        # Step 4: thinking closed & min_tokens=4 reached -> grammar allows {98} -> 98.
        allowed_steps = [
            [50, 98],
            [50],
            [50, *GEMMA4_END],
            [50, *GEMMA4_END],
            [98],
        ]
        emitted: list[int] = []
        for allowed in allowed_steps:
            g_out = GrammarOutput(
                structured_output_request_ids=["r0"],
                grammar_bitmask=_pack_grammar_bitmask(256, [allowed]),
            )
            _set_exec_state(runner, [raw_base.clone()], {"r0": 1}, has_structured=True)
            res = _run_sample_tokens(runner, g_out)
            tok = res.sampled_token_ids[0][0]
            emitted.append(tok)
            req.num_computed_tokens += 1
            lp = res.logprobs
            assert lp is not None
            assert np.isfinite(lp.logprobs[0]).all()
            assert int(lp.logprob_token_ids[0, 0]) == tok
            # Pre-grammar raw logits keep token 99 (25.0) & 98 (18.0) as top-2!
            assert lp.logprob_token_ids[0, 1:3].tolist() == [99, 98]
            assert 1 <= int(lp.sampled_token_ranks[0]) <= 256
        assert emitted == [50, 50, GEMMA4_END[0], GEMMA4_END[1], 98]

        # Also verify unstructured min_tokens stop-masking when end_ids[0] is a stop:
        # below min_tokens=2, end_ids[0] is -inf so forcing defers until step 2.
        stop_params = SamplingParams(
            temperature=0.0,
            min_tokens=2,
            stop_token_ids=[GEMMA4_END[0]],
            thinking_token_budget=0,
        )
        req_s = _request_state("s0", [1, *GEMMA4_START], stop_params)
        runner_s = _make_sample_runner(cfg, ["s0"], {"s0": req_s})
        logits_s = torch.full((4, 256), -20.0)
        logits_s[0, 50] = 6.0
        emitted_s: list[int] = []
        for _ in range(4):
            _set_exec_state(runner_s, [logits_s.clone()], {"s0": 1})
            tok_s = _run_sample_tokens(runner_s).sampled_token_ids[0][0]
            emitted_s.append(tok_s)
            req_s.num_computed_tokens += 1
        assert emitted_s == [50, 50, GEMMA4_END[0], GEMMA4_END[1]]

    @pytest.mark.parametrize(
        "mode",
        [
            "strict_greedy",
            "top_tokens",
            "random_temp",
            "async_eagle3",
            "grammar_partial_drafts",
            "mixed_draftless_chunk",
        ],
    )
    def test_spec_verify_sample_tokens(self, mode: str) -> None:
        cfg = _reasoning_config(list(GEMMA4_START), list(GEMMA4_END))
        if mode in ("strict_greedy", "top_tokens", "random_temp"):
            greedy = mode != "random_temp"
            use_top = mode == "top_tokens"
            reqs = {
                f"r{i}": _request_state(
                    f"r{i}",
                    [1, *GEMMA4_START],
                    SamplingParams(
                        temperature=0.0 if greedy else 0.7,
                        top_k=-1 if greedy else 2,
                        thinking_token_budget=b,
                    ),
                )
                for i, b in enumerate((1, 3, 2))
            }
            runner = _make_sample_runner(
                cfg,
                ["r0", "r1", "r2"],
                reqs,
                all_greedy=greedy,
                top_tokens=use_top,
                spec_k=3,
            )
            if not greedy:
                runner.input_batch.no_top_k = False
            runner._build_padded_sampling_params = lambda *_a: (
                torch.full((4, 1), 0.7),
                torch.full((4, 1), 2, dtype=torch.int32),
                torch.ones((4, 1)),
            )
            # Notice r1's bonus row (index 7) prefers 53 (10.0) while GEMMA4_END[0] is
            # -50.0: pre-sampling bonus_logits forcing must mask row 7 to GEMMA4_END[0]
            # BEFORE top-k=2 / temperature=0.7 sampling!
            prefs = [40, 55, 56, 88, 50, 51, 52, 53, 69, 61, 62, 63]
            logits = torch.full((16, 256), -50.0)
            for idx, p in enumerate(prefs):
                logits[idx, p] = 10.0
            values = torch.argmax(logits, dim=-1, keepdim=True) if use_top else logits
            drafts_per_req = [
                [40, GEMMA4_END[0], GEMMA4_END[1]],
                [50, 51, 52],
                [60, 61, 62],
            ]
            _set_exec_state(
                runner,
                [values],
                {"r0": 4, "r1": 4, "r2": 4},
                md_list=[_verify_md(drafts_per_req, pad_tokens=16)],
                spec_tokens={f"r{i}": d for i, d in enumerate(drafts_per_req)},
                top_tokens=use_top,
            )
            assert _run_sample_tokens(runner).sampled_token_ids == [
                [40, GEMMA4_END[0], GEMMA4_END[1], 88],
                [50, 51, 52, GEMMA4_END[0]],
                [69],
            ]
            st_tbl = runner._thinking_budget.ensure_state_table(torch.device("cpu"))
            slot_r2 = runner._thinking_budget._slot_by_req_id["r2"]
            assert int(st_tbl[slot_r2, COL_THINK_COUNT].item()) == 1
        elif mode == "async_eagle3":
            req = _request_state(
                "r0",
                [1, *GEMMA4_START],
                SamplingParams(temperature=0.0, thinking_token_budget=2),
            )
            runner = _make_sample_runner(
                cfg,
                ["r0"],
                {"r0": req},
                vocab_size=4096,
                async_scheduling=True,
                spec_k=2,
                spec_method="eagle3",
            )
            for dev_drafts, row_prefs, exp in (
                ([41, 42], [41, 42, 43], [41, 42, GEMMA4_END[0]]),
                ([44, 45], [44, 45, 46], [GEMMA4_END[1]]),
            ):
                logits = torch.zeros((4, 4096))
                for r_i, tok in enumerate(row_prefs):
                    logits[r_i, tok] = 7.0
                _set_exec_state(
                    runner,
                    [logits],
                    {"r0": 3},
                    md_list=[_verify_md([dev_drafts])],
                    spec_tokens={"r0": [-1, -1]},
                )
                assert _run_sample_tokens(runner).get_output().sampled_token_ids == [
                    exp
                ]
                req.output_token_ids[-3:] = [-1, -1, -1]
                req.num_computed_tokens = req.num_tokens - 3
            runner._modify_prev_results()
            assert req.output_token_ids == [41, 42, *GEMMA4_END]
        elif mode == "grammar_partial_drafts":
            req = _request_state(
                "r0",
                [1, *GEMMA4_START],
                SamplingParams(
                    temperature=0.0,
                    thinking_token_budget=2,
                    structured_outputs=StructuredOutputsParams(regex=".*"),
                ),
            )
            runner = _make_sample_runner(
                cfg, ["r0"], {"r0": req}, vocab_size=256, spec_k=3
            )
            logits = torch.full((4, 256), -20.0)
            for idx, tok in enumerate((30, 31, 32, 33)):
                logits[idx, tok] = 6.0
            _set_exec_state(
                runner,
                [logits],
                {"r0": 4},
                md_list=[_verify_md([[30, 31, 32]])],
                spec_tokens={"r0": [30, 31, 32]},
                has_structured=True,
            )
            all_toks = list(range(256))
            grammar_out = GrammarOutput(
                structured_output_request_ids=["r0"],
                grammar_bitmask=_pack_grammar_bitmask(256, [all_toks] * 4),
                num_acceptable_drafts=[1],
            )
            assert _run_sample_tokens(runner, grammar_out).sampled_token_ids == [
                [30, 31]
            ]
        else:
            r0 = _request_state(
                "r0", [1, *GEMMA4_START], SamplingParams(thinking_token_budget=1)
            )
            r1 = _request_state(
                "r1",
                [1, *GEMMA4_START],
                SamplingParams(
                    thinking_token_budget=0,
                    logprobs=1,
                    structured_outputs=StructuredOutputsParams(regex=".*"),
                ),
            )
            runner = _make_sample_runner(
                cfg, ["r0", "r1"], {"r0": r0, "r1": r1}, vocab_size=256, spec_k=2
            )
            runner.input_batch.max_num_logprobs = 1
            v_logits, p_logits = (
                torch.full((4, 256), -20.0),
                torch.full((2, 256), -20.0),
            )
            for idx, tok in enumerate((30, 31, 32)):
                v_logits[idx, tok] = 6.0
            # On the draft-less chunk (r1), token 99 is disallowed by grammar (logit
            # 15.0) while GEMMA4_END[0] is allowed: logprob_logits saved pre-grammar
            # must report 99 as raw top-1 while sampling emits forced GEMMA4_END[0].
            p_logits[0, 99] = 15.0
            p_logits[0, 77] = 6.0
            g_out = GrammarOutput(
                structured_output_request_ids=["r1"],
                grammar_bitmask=_pack_grammar_bitmask(
                    256, [[77, *GEMMA4_END], [77], [77]]
                ),
            )
            _set_exec_state(
                runner,
                [v_logits, p_logits],
                {"r0": 3, "r1": 1},
                num_reqs_list=[1, 1],
                md_list=[_verify_md([[30, 31]]), None],
                spec_tokens={"r0": [30, 31]},
                has_structured=True,
            )
            res = _run_sample_tokens(runner, g_out)
            assert res.sampled_token_ids == [[30, GEMMA4_END[0]], [GEMMA4_END[0]]]
            assert res.logprobs is not None
            assert int(res.logprobs.logprob_token_ids[2, 1]) == 99


class TestPrecompileAndTPU:
    """Precompile shape coverage under ``fullgraph=True`` plus TPU parity."""

    @pytest.mark.parametrize("use_top_tokens", [False, True])
    def test_precompile_covers_every_runtime_shape_and_dtype(
        self, monkeypatch: pytest.MonkeyPatch, use_top_tokens: bool
    ) -> None:
        torch._dynamo.reset()
        try:
            with (
                torch.compiler.set_stance("default"),
                torch._dynamo.config.patch(
                    cache_size_limit=64,
                    accumulated_recompile_limit=64,
                    recompile_limit=64,
                ),
            ):
                calls: set[tuple[object, ...]] = set()
                seen_plain_order: list[int] = []
                for orig_fn in (
                    seed_states,
                    mask_plain_logits,
                    advance_plain_states,
                    override_and_advance_plain,
                    prepare_spec_verify_drafts,
                    force_spec_target_logits,
                    force_bonus_logits,
                    force_spec_target_tokens,
                    force_bonus_tokens,
                    advance_spec_states,
                ):
                    fn_name = orig_fn.__name__
                    compiled_fn = torch.compile(
                        inspect.unwrap(orig_fn),
                        backend="eager",
                        fullgraph=True,
                        dynamic=False,
                    )

                    def spy(*args, _name=fn_name, _compiled=compiled_fn):
                        if _name == "mask_plain_logits":
                            seen_plain_order.append(int(args[1].shape[0]))
                        sig = tuple(
                            (tuple(a.shape), a.dtype)
                            if isinstance(a, torch.Tensor)
                            else a
                            for a in args
                        )
                        calls.add((_name, *sig))
                        return _compiled(*args)

                    monkeypatch.setattr(tb_kernels, fn_name, spy)

                cfg = _reasoning_config(list(GEMMA4_START), list(GEMMA4_END))
                runner = object.__new__(TPUModelRunner)
                runner.max_num_reqs = 64
                runner.num_reqs_paddings = [8, 16, 32, 64]
                runner.num_tokens_paddings = [16, 32, 64, 128]
                runner.speculative_config = SimpleNamespace(num_speculative_tokens=3)
                runner._mtp_prefill_only = False
                runner._greedy_head_owner = object() if use_top_tokens else None
                runner.device = torch.device("cpu")
                runner.vocab_size = 128
                runner._hidden_states_dtype = torch.bfloat16
                runner._precompile_timed = lambda _label: nullcontext()
                runner.structured_decoding_manager = SimpleNamespace(
                    target_grammar_bitmask_cpu=torch.zeros((64, 1), dtype=torch.int32)
                )
                runner._thinking_budget = ThinkingBudgetTracker(cfg, max_num_reqs=64)
                monkeypatch.setenv("TPU_PARALLEL_PRECOMPILE", "1")
                monkeypatch.setenv("LOCAL_RANK", "4")

                runner._precompile_thinking_budget()
                assert seen_plain_order == [32, 64, 8, 16]
                precompiled, calls = calls, set()
                top_only = {
                    "override_and_advance_plain",
                    "force_spec_target_tokens",
                    "force_bonus_tokens",
                }
                seen_names = {c[0] for c in precompiled}
                if not use_top_tokens:
                    assert seen_names.isdisjoint(top_only)
                else:
                    assert top_only <= seen_names

                tb = runner._thinking_budget
                assert tb.markers is not None
                sp = SamplingParams(thinking_token_budget=2)
                tb.add("r0", sp, [1, *GEMMA4_START])
                req = _request_state("r0", [1, *GEMMA4_START], sp)
                st_table = tb.prepare_step(torch.device("cpu"), ["r0"], {"r0": req})
                assert st_table is not None

                step = ThinkingBudgetStepState(tb, st_table, tb.markers, frozenset())
                for num_reqs in runner.num_reqs_paddings:
                    slots = step.chunk_slot_ids(
                        torch.device("cpu"), ["r0"], 0, 1, num_reqs
                    )
                    assert slots is not None
                    step.mask_plain_logits(
                        torch.zeros((num_reqs, 128), dtype=torch.bfloat16), slots
                    )
                    step.advance_plain(
                        slots, torch.zeros((num_reqs, 1), dtype=torch.int64)
                    )
                    if use_top_tokens:
                        step.override_and_advance_plain(
                            torch.device("cpu"),
                            ["r0"],
                            0,
                            1,
                            torch.zeros((num_reqs, 1), dtype=torch.int64),
                        )

                for num_tokens in runner._verify_logits_paddings():
                    for num_reqs in runner.num_reqs_paddings:
                        if not runner._spec_verify_bucket_reachable(
                            num_tokens, num_reqs
                        ):
                            continue
                        chunk = step.spec_chunk(
                            torch.device("cpu"), ["r0"], 0, 1, num_reqs
                        )
                        assert chunk is not None
                        md = SimpleNamespace(
                            segment_ids=torch.full((num_tokens,), 1, dtype=torch.int64),
                            group_indices=torch.zeros(num_tokens, dtype=torch.int32),
                        )
                        chunk.prepare_logits(
                            torch.zeros(num_tokens, dtype=torch.int32),
                            torch.zeros((num_tokens, 128), dtype=torch.bfloat16),
                            torch.zeros((num_reqs, 128), dtype=torch.bfloat16),
                            md,  # type: ignore[arg-type]
                            3,
                        )
                        if use_top_tokens:
                            chunk.prepare_tokens(
                                torch.zeros(num_tokens, dtype=torch.int32),
                                torch.zeros(num_tokens, dtype=torch.int64),
                                torch.zeros(num_reqs, dtype=torch.int64),
                                md,  # type: ignore[arg-type]
                                3,
                            )
                        chunk.finish(
                            torch.full((num_reqs, 4), -1, dtype=torch.int32), 3
                        )

                assert calls == precompiled
        finally:
            torch._dynamo.reset()

    @pytest.mark.tpu_test
    def test_tpu_compiled_kernels_match_cpu_reference(self) -> None:
        import torch_tpu  # noqa: F401

        tpu = torch.device("tpu")
        cfg = _reasoning_config(list(GEMMA4_START), list(GEMMA4_END))
        markers = ThinkingBudgetMarkers.from_reasoning_config(cfg)
        assert markers is not None
        kernel_args = markers.kernel_args()

        cpu_tb = ThinkingBudgetTracker(cfg, max_num_reqs=4)
        tpu_tb = ThinkingBudgetTracker(cfg, max_num_reqs=4)
        reqs: dict[str, CachedRequestState] = {}
        batch_ids = ["a", "b", "c", "d"]
        for rid, b in (("a", 0), ("b", 1), ("c", 2), ("d", None)):
            sp = SamplingParams(thinking_token_budget=b)
            reqs[rid] = _request_state(rid, [1, *GEMMA4_START], sp)
            cpu_tb.add(rid, sp, [1, *GEMMA4_START])
            tpu_tb.add(rid, sp, [1, *GEMMA4_START])

        with torch.compiler.set_stance("force_eager"):
            cpu_tbl = cpu_tb.prepare_step(torch.device("cpu"), batch_ids, reqs)
            cpu_sl = cpu_tb.build_chunk_slot_ids(
                torch.device("cpu"), batch_ids, 0, 4, 4
            )
        tpu_tbl = tpu_tb.prepare_step(tpu, batch_ids, reqs)
        tpu_sl = tpu_tb.build_chunk_slot_ids(tpu, batch_ids, 0, 4, 4)
        assert cpu_tbl is not None and tpu_tbl is not None
        assert cpu_sl is not None and tpu_sl is not None
        synchronize_tensors(tpu_tbl)
        assert torch.equal(tpu_tbl.cpu(), cpu_tbl)

        drafts_cpu = torch.tensor(
            [106, 99, 0, 50, 51, 52, 0, 60, 61, 0, 70, 0, 0, 0, 0, 0], dtype=torch.int32
        )
        seg_cpu = torch.tensor(
            [0, 0, 4, 1, 1, 1, 4, 2, 2, 4, 3, 4, 4, 4, 4, 4], dtype=torch.int64
        )
        grp_cpu = torch.tensor(
            [0, 1, 0, 0, 1, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.int32
        )
        logits_cpu = torch.randn((16, 128), dtype=torch.bfloat16)
        logits_cpu[1, GEMMA4_END[1]] = float("-inf")
        bonus_logits_cpu = torch.randn((4, 128), dtype=torch.bfloat16)

        with torch.compiler.set_stance("force_eager"):
            ftgt_cpu, fbon_cpu = prepare_spec_verify_drafts(
                cpu_tbl, cpu_sl, drafts_cpu, seg_cpu, grp_cpu, 3, *kernel_args
            )
            fl_cpu, ed_cpu = force_spec_target_logits(logits_cpu, drafts_cpu, ftgt_cpu)
            fbl_cpu = force_bonus_logits(bonus_logits_cpu, fbon_cpu)
            ft_cpu, edt_cpu = force_spec_target_tokens(
                torch.arange(16, dtype=torch.int64), drafts_cpu, ftgt_cpu
            )
            fbt_cpu = force_bonus_tokens(torch.arange(4, dtype=torch.int64), fbon_cpu)

        ftgt_tpu, fbon_tpu = prepare_spec_verify_drafts(
            tpu_tbl,
            tpu_sl,
            drafts_cpu.to(tpu),
            seg_cpu.to(tpu),
            grp_cpu.to(tpu),
            3,
            *kernel_args,
        )
        fl_tpu, ed_tpu = force_spec_target_logits(
            logits_cpu.to(tpu), drafts_cpu.to(tpu), ftgt_tpu
        )
        fbl_tpu = force_bonus_logits(bonus_logits_cpu.to(tpu), fbon_tpu)
        ft_tpu, edt_tpu = force_spec_target_tokens(
            torch.arange(16, dtype=torch.int64).to(tpu),
            drafts_cpu.to(tpu),
            ftgt_tpu,
        )
        fbt_tpu = force_bonus_tokens(
            torch.arange(4, dtype=torch.int64).to(tpu), fbon_tpu
        )
        synchronize_tensors([fl_tpu, ed_tpu, fbl_tpu, ft_tpu, edt_tpu, fbt_tpu])
        assert torch.equal(fl_tpu.cpu(), fl_cpu) and torch.equal(ed_tpu.cpu(), ed_cpu)
        assert torch.equal(fbl_tpu.cpu(), fbl_cpu)
        assert torch.equal(ft_tpu.cpu(), ft_cpu) and torch.equal(edt_tpu.cpu(), edt_cpu)
        assert torch.equal(fbt_tpu.cpu(), fbt_cpu)

        next_cpu = torch.tensor(
            [[106, 107, -1, -1], [50, 106, -1, -1], [60, 61, 82, -1], [70, -1, -1, -1]],
            dtype=torch.int32,
        )
        plain_logits_cpu = torch.randn((4, 128), dtype=torch.bfloat16)
        sel_cpu = torch.tensor([[40], [41], [42], [43]], dtype=torch.int64)
        with torch.compiler.set_stance("force_eager"):
            cpu_tbl = advance_spec_states(cpu_tbl, cpu_sl, next_cpu, 3, *kernel_args)
            mpl_cpu = mask_plain_logits(
                cpu_tbl, cpu_sl, plain_logits_cpu, markers.end_ids
            )
            cpu_tbl_adv = advance_plain_states(cpu_tbl, cpu_sl, sel_cpu, *kernel_args)
            out_p_cpu, cpu_tbl = override_and_advance_plain(
                cpu_tbl, cpu_sl, sel_cpu, *kernel_args
            )

        tpu_tbl = advance_spec_states(
            tpu_tbl, tpu_sl, next_cpu.to(tpu), 3, *kernel_args
        )
        mpl_tpu = mask_plain_logits(
            tpu_tbl, tpu_sl, plain_logits_cpu.to(tpu), markers.end_ids
        )
        tpu_tbl_adv = advance_plain_states(
            tpu_tbl, tpu_sl, sel_cpu.to(tpu), *kernel_args
        )
        out_p_tpu, tpu_tbl = override_and_advance_plain(
            tpu_tbl, tpu_sl, sel_cpu.to(tpu), *kernel_args
        )
        synchronize_tensors([mpl_tpu, tpu_tbl_adv, out_p_tpu, tpu_tbl])
        assert torch.equal(mpl_tpu.cpu(), mpl_cpu)
        assert torch.equal(tpu_tbl_adv.cpu(), cpu_tbl_adv)
        assert torch.equal(out_p_tpu.cpu(), out_p_cpu)
        assert torch.equal(tpu_tbl.cpu(), cpu_tbl)
