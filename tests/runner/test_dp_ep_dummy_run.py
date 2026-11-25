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
"""Tests for DP+EP _dummy_run metadata and precompiled idle logits buckets."""

import contextlib
import types
from types import SimpleNamespace

import pytest
import torch
from runner_input_test_utils import (
    _fake_tpu_events,
    _make_runner,
)
from vllm import forward_context as fc_mod
from vllm.forward_context import get_forward_context

from vllm_torchtpu.runner import tpu_runner as tpu_runner_mod
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def test_should_populate_dp_metadata_and_clamp_idle_logits_indices_in_dummy_run(
    monkeypatch,
):
    _fake_tpu_events(monkeypatch)
    monkeypatch.setattr(
        tpu_runner_mod,
        "set_vllm_model_wrapper_context",
        lambda **_kwargs: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        tpu_runner_mod, "synchronize_tensors", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        fc_mod.current_platform,
        "set_additional_forward_context",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        fc_mod,
        "coordinate_batch_across_dp",
        lambda **_kwargs: pytest.fail("unexpected coordinate_batch_across_dp call"),
    )

    runner = _make_runner(
        num_computed_tokens=[0],
        prompt_tokens=[16],
        scheduled_tokens=[1],
        token_paddings=[16, 32, 64],
        pcp_size=1,
        dp_size=4,
        dp_rank=2,
        enable_ep=True,
        max_num_reqs=48,
        block_size=16,
    )
    runner.num_reqs_paddings = tpu_runner_mod._get_req_paddings(
        tpu_runner_mod.MIN_NUM_SEQS, runner.max_num_reqs
    )
    runner._profile_inputs = {}
    runner.vocab_size = 128
    runner.model_config = SimpleNamespace(get_hidden_size=lambda: 8)
    runner._hidden_states_dtype = torch.bfloat16
    runner._precompile_timed = lambda *_args: contextlib.nullcontext()
    runner._profile_input = types.MethodType(TPUModelRunner._profile_input, runner)
    runner.maybe_select_dummy_loras = lambda *_args, **_kwargs: contextlib.nullcontext()
    dummy_run = TPUModelRunner._dummy_run
    precompile_logits = TPUModelRunner._precompile_compute_selected_logits

    seen_dp_metadata = []
    seen_batch_descriptors = []
    seen_logits_shapes = []
    seen_logits_pairs = []

    def fake_forward_model(
        *, input_ids, positions, inputs_embeds=None, intermediate_tensors=None
    ):
        del positions, inputs_embeds, intermediate_tensors
        ctx = get_forward_context()
        seen_dp_metadata.append(ctx.dp_metadata)
        seen_batch_descriptors.append(ctx.batch_descriptor)
        return torch.zeros((input_ids.shape[0], 8), dtype=torch.bfloat16), None

    def fake_compute_selected_logits(hidden_states, logits_indices):
        seen_logits_shapes.append(tuple(logits_indices.shape))
        seen_logits_pairs.append((hidden_states.shape[0], logits_indices.shape[0]))
        return torch.zeros((logits_indices.shape[0], 16), dtype=torch.bfloat16)

    runner.forward_model = fake_forward_model
    runner.compute_selected_logits = fake_compute_selected_logits

    precompile_logits(runner)
    precompiled_grid = set(seen_logits_pairs)
    seen_logits_shapes.clear()
    seen_logits_pairs.clear()

    # 1. Startup warmup / profiling (dp_lockstep=False) sets DPMetadata on DP+EP.
    elapsed = dummy_run(
        runner,
        num_tokens=16,
        num_reqs=48,
        num_blocks=4,
        dp_lockstep=False,
        profile_prefix=32,
    )
    assert isinstance(elapsed, float)
    assert seen_dp_metadata[-1] is not None
    assert seen_dp_metadata[-1].num_tokens_across_dp_cpu.tolist() == [16, 16, 16, 16]
    assert seen_logits_shapes == []

    # 2. Idle lockstep (dp_lockstep=True) sizes logits rows via
    # _prompt_logprob_rows_per_call(num_tokens) so every shape stays in-grid:
    # - num_tokens=16 -> min(16, max_num_reqs=48)=16 -> padded 16
    # - num_tokens=64, num_reqs=24 -> min(64, max_num_reqs=48)=48 -> padded 48
    # - num_tokens=64, num_reqs=64 -> min(64, max_num_reqs=48)=48 -> padded 48
    for num_tokens, num_reqs, expected_logits_reqs in (
        (16, 48, 16),
        (64, 24, 48),
        (64, 64, 48),
    ):
        res = dummy_run(
            runner,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            num_blocks=4,
            dp_lockstep=True,
        )
        assert res is None
        assert seen_dp_metadata[-1] is not None
        assert (
            seen_dp_metadata[-1].num_tokens_across_dp_cpu.tolist() == [num_tokens] * 4
        )
        assert seen_logits_shapes[-1] == (expected_logits_reqs,)
        assert seen_logits_pairs[-1] in precompiled_grid

    # 3. Non-lockstep configuration (dp_size=1, enable_ep=False) keeps
    # dp_metadata=None while still passing num_tokens to set_forward_context
    # (matching execute_model, including for dp_size=1 sequence-parallel MoE).
    non_dp_runner = _make_runner(
        num_computed_tokens=[0],
        prompt_tokens=[16],
        scheduled_tokens=[1],
        token_paddings=[16, 32],
        pcp_size=1,
        dp_size=1,
        dp_rank=0,
        enable_ep=False,
        max_num_reqs=16,
        block_size=16,
    )
    non_dp_runner.maybe_select_dummy_loras = runner.maybe_select_dummy_loras
    non_dp_runner.forward_model = fake_forward_model
    non_dp_runner.compute_selected_logits = fake_compute_selected_logits

    dummy_run(
        non_dp_runner,
        num_tokens=16,
        num_reqs=16,
        num_blocks=4,
        dp_lockstep=False,
    )
    assert seen_dp_metadata[-1] is None
    assert seen_batch_descriptors[-1] is not None
    assert seen_batch_descriptors[-1].num_tokens == 16

    sp_moe_runner = _make_runner(
        num_computed_tokens=[0],
        prompt_tokens=[16],
        scheduled_tokens=[1],
        token_paddings=[16, 32],
        pcp_size=1,
        dp_size=1,
        dp_rank=0,
        enable_ep=True,
        max_num_reqs=16,
        block_size=16,
    )
    sp_moe_runner.parallel_config.use_sequence_parallel_moe = True
    sp_moe_runner.maybe_select_dummy_loras = runner.maybe_select_dummy_loras
    sp_moe_runner.forward_model = fake_forward_model
    sp_moe_runner.compute_selected_logits = fake_compute_selected_logits

    dummy_run(
        sp_moe_runner,
        num_tokens=16,
        num_reqs=16,
        num_blocks=4,
        dp_lockstep=False,
    )
    assert seen_dp_metadata[-1] is not None
    assert seen_dp_metadata[-1].num_tokens_across_dp_cpu.tolist() == [16]
    assert seen_batch_descriptors[-1] is not None
    assert seen_batch_descriptors[-1].num_tokens == 16


@pytest.mark.tpu_test
def test_dp_ep_dummy_run_tpu_zero_recompilation(device, monkeypatch):
    """After precompiling logits and running a dp_lockstep=False warmup on TPU,
    an idle dp_lockstep=True dummy run must compile zero new XLA graphs
    (covering both the precompiled request-bucket rule and CPU .to(out.device)
    index construction)."""
    runner = _make_runner(
        num_computed_tokens=[0, 0],
        prompt_tokens=[128, 256],
        scheduled_tokens=[3, 2],
        token_paddings=[16, 32],
        pcp_size=1,
        dp_size=2,
        dp_rank=1,
        enable_ep=True,
        max_num_reqs=32,
        block_size=64,
    )
    runner.device = device
    runner.num_reqs_paddings = tpu_runner_mod._get_req_paddings(
        tpu_runner_mod.MIN_NUM_SEQS, runner.max_num_reqs
    )
    runner.is_pooling_model = False
    runner.model_config = SimpleNamespace(get_hidden_size=lambda: 8)
    runner._hidden_states_dtype = torch.bfloat16
    runner._precompile_timed = lambda *_args: contextlib.nullcontext()
    lm_weight = torch.zeros((8, 16), dtype=torch.bfloat16).to(device)
    runner.model = SimpleNamespace(compute_logits=lambda x: x @ lm_weight)
    runner.compute_selected_logits = types.MethodType(
        TPUModelRunner.compute_selected_logits, runner
    )
    runner.maybe_select_dummy_loras = lambda *_args, **_kwargs: contextlib.nullcontext()

    monkeypatch.setattr(
        tpu_runner_mod,
        "set_vllm_model_wrapper_context",
        lambda **_kwargs: contextlib.nullcontext(),
    )

    preallocated_hidden = torch.zeros((16, 8), dtype=torch.bfloat16).to(device)
    seen_dp_dev_types = []

    def tpu_forward_model(
        *, input_ids, positions, inputs_embeds=None, intermediate_tensors=None
    ):
        del input_ids, positions, inputs_embeds, intermediate_tensors
        dp_md = get_forward_context().dp_metadata
        assert dp_md is not None
        seen_dp_dev_types.append(dp_md.num_tokens_across_dp_cpu.device.type)
        assert dp_md.num_tokens_across_dp_cpu.cpu().tolist() == [16, 16]
        return preallocated_hidden, None

    runner.forward_model = tpu_forward_model
    dummy_run = TPUModelRunner._dummy_run

    TPUModelRunner._precompile_compute_selected_logits(runner)
    dummy_run(
        runner,
        num_tokens=16,
        num_reqs=32,
        num_blocks=4,
        dp_lockstep=False,
    )
    tpu_runner_mod.synchronize_tensors(None)
    graphs_before_idle = len(torch.tpu._get_cache_stats().per_entry_stats)

    dummy_run(
        runner,
        num_tokens=16,
        num_reqs=32,
        num_blocks=4,
        dp_lockstep=True,
    )
    tpu_runner_mod.synchronize_tensors(None)
    graphs_after_idle = len(torch.tpu._get_cache_stats().per_entry_stats)

    assert seen_dp_dev_types == ["tpu", "tpu"]
    assert graphs_after_idle == graphs_before_idle
