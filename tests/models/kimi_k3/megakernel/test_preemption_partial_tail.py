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

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request

from vllm_torchtpu import (
    _cache_only_prefill_tokens,
    _disable_partial_hits_without_block_copies,
)
from vllm_torchtpu.models.kimi_k3.megakernel import runner as runner_mod
from vllm_torchtpu.models.kimi_k3.megakernel.config import MegakernelEngineConfig
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3 import program as k3
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.aggregate import (
    KimiAggregate,
    role_config,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.prefill import (
    KimiPrefiller,
    skips_step,
)
from vllm_torchtpu.models.kimi_k3.megakernel.program import (
    RequestInput,
    StepInput,
    StepOutput,
)
from vllm_torchtpu.models.kimi_k3.megakernel.runner import MegakernelModelRunner
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def _engine_config(role: str = "aggregate") -> MegakernelEngineConfig:
    return MegakernelEngineConfig(
        role=role,
        tensor_parallel_size=32,
        decode_context_parallel_size=32,
        block_size=512,
        kv_cache_dtype="fp8",
        max_num_seqs=8,
        max_model_len=163840,
        max_num_batched_tokens=16448,
        num_speculative_tokens=0,
    )


def _vllm_config(kv_role: str | None) -> SimpleNamespace:
    kv = None
    if kv_role is not None:
        kv = SimpleNamespace(
            is_kv_producer=(kv_role == "kv_producer"),
            is_kv_consumer=(kv_role == "kv_consumer"),
        )
    return SimpleNamespace(
        load_config=SimpleNamespace(
            load_format="megakernel",
            model_loader_extra_config={"presharded": "/p"},
        ),
        kv_transfer_config=kv,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=32,
            decode_context_parallel_size=32,
            cp_kv_cache_interleave_size=512,
            data_parallel_size=1,
        ),
        cache_config=SimpleNamespace(
            block_size=512,
            cache_dtype="fp8",
            mamba_ssm_cache_dtype="auto",
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=8,
            max_num_batched_tokens=16384 if kv_role == "kv_producer" else 16448,
            async_scheduling=True,
        ),
        model_config=SimpleNamespace(
            max_model_len=163840,
            hf_config=SimpleNamespace(architectures=["KimiK3ForConditionalGeneration"]),
        ),
        speculative_config=None,
    )


def _make_hybrid_coordinator() -> HybridKVCacheCoordinator:
    hash_fn = get_hash_fn_by_name("sha256")
    init_none_hash(hash_fn)
    fa_spec = FullAttentionSpec(
        block_size=512,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.float8_e4m3fn,
    )
    mamba_spec = MambaSpec(
        shapes=((3, 128, 128), (3, 3, 384)),
        dtypes=(torch.float32, torch.bfloat16),
        block_size=512,
        page_size_padded=327680,
        mamba_cache_mode="align",
        num_speculative_blocks=0,
        num_prefill_checkpoint_blocks=0,
    )
    kv_cfg = KVCacheConfig(
        num_blocks=546,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["layers.3.self_attn"], fa_spec),
            KVCacheGroupSpec(["layers.0.linear_attn"], mamba_spec),
            KVCacheGroupSpec(["layers.1.linear_attn"], mamba_spec),
            KVCacheGroupSpec(["layers.2.linear_attn"], mamba_spec),
        ],
    )
    return HybridKVCacheCoordinator(
        kv_cache_config=kv_cfg,
        max_model_len=163840,
        max_in_flight_tokens=16448,
        use_eagle=False,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=32,
        pcp_world_size=1,
        scheduler_block_size=16384,
        hash_block_size=512,
    )


def test_should_disable_block_copies_on_aggregate_and_keep_enabled_on_prefill() -> None:
    agg_cfg = _engine_config("aggregate")
    agg_caps = k3.capabilities(agg_cfg)
    assert agg_caps.page_aligned_chunks is True
    assert agg_caps.block_copies is False

    prefill_caps = k3.capabilities(role_config(agg_cfg, "prefill"))
    assert prefill_caps.page_aligned_chunks is True
    assert prefill_caps.block_copies is True

    decode_caps = k3.capabilities(role_config(agg_cfg, "decode"))
    assert decode_caps.page_aligned_chunks is False
    assert decode_caps.block_copies is False


def test_should_round_prefix_hits_to_16k_blocks_on_aggregate_only() -> None:
    agg_coord = _make_hybrid_coordinator()
    agg_sched = SimpleNamespace(
        vllm_config=_vllm_config(None),
        kv_cache_manager=SimpleNamespace(coordinator=agg_coord),
        mamba_partial_cache_hit=True,
    )
    assert agg_coord.enable_partial_hash_hits is True
    assert _disable_partial_hits_without_block_copies(agg_sched) is True
    assert agg_coord.enable_partial_hash_hits is False
    assert agg_sched.mamba_partial_cache_hit is False
    for manager in agg_coord.single_type_managers:
        assert manager.cache_hit_alignment_tokens == 16384

    pf_coord = _make_hybrid_coordinator()
    pf_sched = SimpleNamespace(
        vllm_config=_vllm_config("kv_producer"),
        kv_cache_manager=SimpleNamespace(coordinator=pf_coord),
        mamba_partial_cache_hit=True,
    )
    assert _disable_partial_hits_without_block_copies(pf_sched) is False
    assert pf_coord.enable_partial_hash_hits is True
    assert pf_sched.mamba_partial_cache_hit is True


def test_should_never_skip_partial_tail_hash_boundary_in_skips_step() -> None:
    assert skips_step(98304, 114176, 114200, 98304, 16384, 16384, 512) is False
    assert skips_step(98304, 114176, 114199, 98304, 16384, 16384, 512) is False
    assert skips_step(0, 4096, 6000, 0, 16384, 16384, 512) is True
    assert skips_step(98304, 113664, 114200, 98304, 16384, 16384, 512) is True
    assert skips_step(0, 16384, 20000, 0, 16384, 16384, 512) is False
    assert skips_step(98304, 114200, 114200, 98304, 16384, 16384, 512) is False
    assert skips_step(0, 8192, 20000, 0, 16384, 16384, 512) is False


class _RecordingProgram:
    def __init__(self) -> None:
        self.seen_steps: list[StepInput] = []

    def step(self, step_input: StepInput) -> StepOutput:
        self.seen_steps.append(
            StepInput(
                requests=list(step_input.requests),
                finished=set(step_input.finished),
                copies=list(step_input.copies),
                mask=step_input.mask,
            )
        )
        return StepOutput(
            torch.zeros((max(len(step_input.requests), 1), 1), dtype=torch.int32)
        )


def test_should_forward_preempted_req_ids_to_program_finished_across_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_tpu_init(
        self: Any, vllm_config: Any, device: Any, *args: Any, **kwargs: Any
    ) -> None:
        self.vllm_config = vllm_config
        self.scheduler_config = SimpleNamespace(async_scheduling=False)
        self.kv_cache_config = SimpleNamespace(kv_cache_groups=[object()])
        self.speculative_config = None
        self.execute_model_state = None
        table = np.zeros((2, 4), dtype=np.int32)
        self.input_batch = SimpleNamespace(
            num_reqs=1,
            req_ids=["active"],
            block_table=[
                SimpleNamespace(get_cpu_tensor=lambda: torch.from_numpy(table))
            ],
            token_ids_cpu=np.zeros((2, 32), dtype=np.int32),
        )
        self.requests = {
            "active": SimpleNamespace(
                num_computed_tokens=0,
                num_tokens=16,
                num_prompt_tokens=16,
                sampling_params=SamplingParams(temperature=0.0, max_tokens=4),
            )
        }
        self.maybe_setup_kv_connector = lambda *a, **kw: None

    monkeypatch.setattr(TPUModelRunner, "__init__", fake_tpu_init)
    monkeypatch.setattr(TPUModelRunner, "_update_states", lambda self, out: None)
    monkeypatch.setattr(runner_mod, "has_kv_transfer_group", lambda: False)
    monkeypatch.setattr(
        runner_mod, "set_forward_context", lambda *a, **kw: nullcontext()
    )
    monkeypatch.setattr(runner_mod.dist_utils, "get_raiden_inline_load", lambda: False)

    runner = MegakernelModelRunner(_vllm_config(None), device="cpu")
    program = _RecordingProgram()
    runner.program = program

    zero_step = SchedulerOutput.make_empty()
    zero_step.preempted_req_ids = {"preempt_on_empty"}
    zero_step.finished_req_ids = {"done_on_empty"}
    runner.execute_model(zero_step)
    assert len(program.seen_steps) == 0

    active_step = SchedulerOutput.make_empty()
    active_step.num_scheduled_tokens = {"active": 16}
    active_step.total_num_scheduled_tokens = 16
    active_step.preempted_req_ids = {"victim_in_step"}
    runner.execute_model(active_step)
    runner.execute_model_state = None

    assert len(program.seen_steps) == 1
    assert set(program.seen_steps[0].finished) == {
        "done_on_empty",
        "preempt_on_empty",
        "victim_in_step",
    }

    next_step = SchedulerOutput.make_empty()
    next_step.num_scheduled_tokens = {"active": 1}
    next_step.total_num_scheduled_tokens = 1
    next_step.preempted_req_ids = None
    runner.execute_model(next_step)
    assert len(program.seen_steps) == 2
    assert set(program.seen_steps[1].finished) == set()


class _FakePrefillerForAggregate:
    def __init__(self) -> None:
        self._requests: dict[str, Any] = {"stale_pf": object()}
        self.seen_steps: list[StepInput] = []

    def step(self, step_input: StepInput) -> StepOutput:
        for req_id in step_input.finished:
            self._requests.pop(req_id, None)
        self.seen_steps.append(step_input)
        return StepOutput(torch.empty((0, 1), dtype=torch.int32))

    def active_request_ids(self) -> set[str]:
        return set(self._requests)


class _FakeSlotMap:
    def __init__(self, initial: set[str]) -> None:
        self.active = set(initial)

    def __contains__(self, req_id: str) -> bool:
        return req_id in self.active

    def release(self, finished: set[str]) -> None:
        self.active -= set(finished)


class _FakeDecoderForAggregate:
    def __init__(self, tracked: set[str]) -> None:
        self.tracked = set(tracked)
        self.slot_map = _FakeSlotMap(tracked)
        self.seen_steps: list[StepInput] = []

    def forget(self, finished: set[str]) -> None:
        ids = set(finished or ())
        self.tracked -= ids
        self.slot_map.release(ids)

    def step(self, step_input: StepInput) -> StepOutput:
        self.forget(set(step_input.finished or ()))
        for row in step_input.requests:
            if row.scheduled > 0:
                self.tracked.add(row.req_id)
                self.slot_map.active.add(row.req_id)
        self.seen_steps.append(step_input)
        return StepOutput(torch.zeros((len(step_input.requests), 1), dtype=torch.int32))


def test_should_forget_preempted_decoder_request_before_split_rows_in_aggregate() -> (
    None
):
    agg = object.__new__(KimiAggregate)
    agg.prefiller = _FakePrefillerForAggregate()
    agg.decoder = _FakeDecoderForAggregate({"resumed_full_hit", "still_decoding"})

    rows = [
        RequestInput(
            req_id="resumed_full_hit",
            computed=16384,
            scheduled=16,
            token_ids=np.arange(16408, dtype=np.int32),
            num_tokens=16400,
            num_prompt_tokens=16384,
            block_tables=(np.zeros(2, dtype=np.int32),),
        ),
        RequestInput(
            req_id="still_decoding",
            computed=20000,
            scheduled=1,
            token_ids=np.arange(20008, dtype=np.int32),
            num_tokens=20001,
            num_prompt_tokens=16384,
            block_tables=(np.zeros(2, dtype=np.int32),),
        ),
    ]
    agg.step(StepInput(requests=rows, finished={"resumed_full_hit", "stale_pf"}))

    assert agg.prefiller.active_request_ids() == set()
    assert len(agg.prefiller.seen_steps) == 1
    pf_rows = agg.prefiller.seen_steps[0].requests
    dec_rows = agg.decoder.seen_steps[0].requests
    assert (pf_rows[0].computed, pf_rows[0].scheduled) == (16384, 15)
    assert (dec_rows[0].computed, dec_rows[0].scheduled) == (16399, 1)
    assert (pf_rows[1].scheduled, dec_rows[1].scheduled) == (0, 1)

    agg.step(StepInput(requests=[rows[1]], finished=()))
    assert len(agg.decoder.seen_steps) == 2


class _MiniPrefillerHarness(KimiPrefiller):
    def __init__(self) -> None:
        self.config = _engine_config("prefill")
        self.page_tokens = 512
        self.slots = 32
        self.compute_logits = False
        self.draft_latents = None
        self.program = SimpleNamespace(windows=(512, 16384), capacity=16384)
        self._writer = SimpleNamespace(copy_blocks=lambda pairs: None)
        self._requests = {}
        self._mla_group = 0
        self._kda_groups = [(1, 512)]
        self.executed_ranges: list[tuple[int, int]] = []

    def _blocks(self, values: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(values, np.int32))

    def _read_hit_states(self, rows: Any, hits: Any) -> None:
        return None

    def _run_window(
        self,
        rows: Any,
        by_row: Any,
        window: int,
        members: Any,
        *,
        need_logits: bool,
    ) -> None:
        for piece, _ in members:
            self.executed_ranges.append((piece.start, piece.end))
        return None


def test_should_execute_tail_boundary_in_prefiller_and_coalesce_mid_chunk() -> None:
    prefiller = _MiniPrefillerHarness()
    tables = (np.arange(1, 9, dtype=np.int32), np.arange(10, 250, dtype=np.int32))

    prefiller.step(
        StepInput(
            requests=[
                RequestInput(
                    req_id="coalesced",
                    computed=0,
                    scheduled=4096,
                    token_ids=np.arange(6008, dtype=np.int32),
                    num_tokens=6000,
                    num_prompt_tokens=6000,
                    block_tables=tables,
                )
            ]
        )
    )
    assert prefiller.executed_ranges == []

    prefiller.step(
        StepInput(
            requests=[
                RequestInput(
                    req_id="tail_boundary",
                    computed=98304,
                    scheduled=15872,
                    token_ids=np.arange(114208, dtype=np.int32),
                    num_tokens=114200,
                    num_prompt_tokens=114200,
                    block_tables=tables,
                )
            ],
            finished={"coalesced"},
        )
    )
    assert prefiller.executed_ranges == [(98304, 114176)]


def _simulate_double_preemption_multi_block(*, role: str, patched: bool) -> list[str]:
    coord = _make_hybrid_coordinator()
    hash_fn = get_hash_fn_by_name("sha256")
    sched = SimpleNamespace(
        vllm_config=_vllm_config(None if role == "aggregate" else "kv_producer"),
        kv_cache_manager=SimpleNamespace(coordinator=coord),
        mamba_partial_cache_hit=True,
    )
    if patched:
        _disable_partial_hits_without_block_copies(sched)
    if role == "aggregate":
        _cache_only_prefill_tokens(sched)

    rng = np.random.default_rng(42)
    req = Request(
        request_id="req-double-preempt",
        prompt_token_ids=rng.integers(10, 10000, size=114200).tolist(),
        sampling_params=SamplingParams(temperature=0.0, max_tokens=16384),
        pooling_params=None,
        arrival_time=0.0,
        block_hasher=get_request_block_hasher(512, hash_fn),
    )
    phys: dict[int, str] = {b: "UNWRITTEN_RECYCLED_F32_NAN" for b in range(1, 546)}
    phys[0] = "NULL"

    pf_limit = (
        req.num_prompt_tokens - 1 if role == "aggregate" else req.num_prompt_tokens
    )
    hit_blocks, hit_len, _ = coord.find_longest_cache_hit(req.block_hashes, pf_limit)
    coord.allocate_new_computed_blocks(req.request_id, hit_blocks, hit_len, 0)
    prefill_state: dict[str, dict[str, Any]] = {}

    chunk_ends = (
        [16384, 32768, 49152, 65536, 81920, 98304, 114176]
        if sched.mamba_partial_cache_hit
        else [16384, 32768, 49152, 65536, 81920, 98304]
    )
    prev = 0
    for end in chunk_ends:
        if prev > 0:
            for manager in coord.single_type_managers:
                manager.remove_skipped_blocks(
                    req.request_id, prev, req.num_prompt_tokens
                )
        coord.allocate_new_blocks(req.request_id, end, end)
        for manager in coord.single_type_managers:
            manager.take_pending_cow_copies()
        fa_tbl = [
            b.block_id
            for b in coord.single_type_managers[0].req_to_blocks[req.request_id]
        ]
        kda_bs = [
            coord.single_type_managers[g]
            .req_to_blocks[req.request_id][(end - 1) // 512]
            .block_id
            for g in (1, 2, 3)
        ]
        st = prefill_state.setdefault(req.request_id, {"start": 0, "skipped": []})
        should_skip = (
            skips_step(prev, end, pf_limit, st["start"], 16384, 16384, 512)
            if patched
            else (
                end < pf_limit
                and pf_limit - st["start"] <= 16384
                and end // 16384 == prev // 16384
            )
        )
        if should_skip:
            st["skipped"].append((end, kda_bs))
        else:
            phys[fa_tbl[(end - 1) // 16384]] = f"VALID_MLA({st['start']}..{end})"
            for block_id in kda_bs:
                phys[block_id] = f"VALID_KDA({end})"
            st["start"] = end
        req.num_computed_tokens = end
        coord.cache_blocks(req, end)
        for manager in coord.single_type_managers:
            manager.new_step_starts()
        prev = end

    if patched:
        prefill_state.pop(req.request_id, None)
    freed1 = coord.pop_blocks_for_free(req.request_id)
    coord.block_pool.free_blocks(reversed(freed1))
    req.num_computed_tokens = 0

    hb1, hl1, _ = coord.find_longest_cache_hit(req.block_hashes, pf_limit)
    coord.allocate_new_computed_blocks(req.request_id, hb1, hl1, 0)
    coord.allocate_new_blocks(
        req.request_id, req.num_prompt_tokens, req.num_prompt_tokens
    )
    for src_b, dst_b in coord.single_type_managers[0].take_pending_cow_copies():
        phys[dst_b.block_id] = phys[src_b.block_id]
    fa_tbl_r1 = [
        b.block_id for b in coord.single_type_managers[0].req_to_blocks[req.request_id]
    ]
    st1 = prefill_state.get(req.request_id)
    expected1 = (
        None
        if st1 is None
        else (st1["skipped"][-1][0] if st1["skipped"] else st1["start"])
    )
    if st1 is None or hl1 != expected1:
        st1 = {"start": hl1 if hl1 % 512 == 0 else 0, "skipped": []}
    for blk_idx in range(st1["start"] // 16384, (pf_limit - 1) // 16384 + 1):
        if (
            phys[fa_tbl_r1[blk_idx]] == "UNWRITTEN_RECYCLED_F32_NAN"
            and st1["start"] <= blk_idx * 16384
        ):
            phys[fa_tbl_r1[blk_idx]] = f"VALID_MLA({st1['start']}..{pf_limit})"
        elif (
            st1["start"] > blk_idx * 16384
            and phys[fa_tbl_r1[blk_idx]] == "UNWRITTEN_RECYCLED_F32_NAN"
        ):
            phys[fa_tbl_r1[blk_idx]] = "UNWRITTEN_RECYCLED_F32_NAN"
        else:
            phys[fa_tbl_r1[blk_idx]] = f"VALID_MLA(0..{pf_limit})"
    prefill_state.pop(req.request_id, None)
    req.num_computed_tokens = req.num_prompt_tokens
    coord.cache_blocks(req, req.num_prompt_tokens)
    for manager in coord.single_type_managers:
        manager.new_step_starts()
    req.append_output_token_ids(999)

    freed2 = coord.pop_blocks_for_free(req.request_id)
    coord.block_pool.free_blocks(reversed(freed2))
    req.num_computed_tokens = 0

    pf_limit2 = req.num_tokens - 1 if role == "aggregate" else req.num_prompt_tokens
    hb2, hl2, _ = coord.find_longest_cache_hit(req.block_hashes, pf_limit2)
    coord.allocate_new_computed_blocks(req.request_id, hb2, hl2, 0)
    coord.allocate_new_blocks(req.request_id, req.num_tokens, req.num_tokens)
    for src_b, dst_b in coord.single_type_managers[0].take_pending_cow_copies():
        phys[dst_b.block_id] = phys[src_b.block_id]
    fa_tbl_r2 = [
        b.block_id for b in coord.single_type_managers[0].req_to_blocks[req.request_id]
    ]
    num_hit_mla_blocks = (hl2 + 16383) // 16384
    return [phys[fa_tbl_r2[idx]] for idx in range(num_hit_mla_blocks)]


@pytest.mark.parametrize("role", ["aggregate", "prefill"])
def test_should_never_publish_or_read_unwritten_blocks_across_double_preemption(
    role: str,
) -> None:
    unpatched_reads = _simulate_double_preemption_multi_block(role=role, patched=False)
    assert "UNWRITTEN_RECYCLED_F32_NAN" in unpatched_reads

    patched_reads = _simulate_double_preemption_multi_block(role=role, patched=True)
    assert len(patched_reads) >= 6
    assert "UNWRITTEN_RECYCLED_F32_NAN" not in patched_reads
    assert all(tag.startswith("VALID_MLA") for tag in patched_reads)
