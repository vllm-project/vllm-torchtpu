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
"""The device-decode controller: the flag, the configurations it refuses,
the programs it compiles and warms up, and the stock steps' sampling noise
and slot write-back."""

import contextlib

import pytest
import torch
from device_decode_test_utils import (
    FLAG,
    MAX_LOGPROBS,
    SimEngine,
    make_runner,
)

from vllm_torchtpu import envs
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform
from vllm_torchtpu.runner.device_decode import controller as controller_mod
from vllm_torchtpu.runner.device_decode.controller import (
    DeviceDecodeController,
    stock_uniform,
    stock_uniform_key,
)
from vllm_torchtpu.runner.device_decode.head import SamplerVariant, head_key
from vllm_torchtpu.runner.device_decode.inputs import DecodeShape
from vllm_torchtpu.runner.device_decode.rng import counter_uniform_impl
from vllm_torchtpu.runner.device_decode.slots import (
    NO_SLOT,
    block_write_key,
    scalar_write_key,
    write_back_key,
)


def test_the_flag_creates_the_controller(monkeypatch):
    runner = make_runner(monkeypatch)
    monkeypatch.delenv(FLAG, raising=False)
    assert DeviceDecodeController.maybe_create(runner) is None
    monkeypatch.setenv(FLAG, "0")
    assert DeviceDecodeController.maybe_create(runner) is None
    monkeypatch.setenv(FLAG, "1")
    assert isinstance(
        DeviceDecodeController.maybe_create(runner), DeviceDecodeController
    )


@pytest.mark.parametrize(("name", "value"), [(FLAG, "maybe")])
def test_bad_flag_values_raise(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        getattr(envs, name)


def test_every_rank_gets_the_flag():
    assert FLAG in TpuPlatform.additional_env_vars


def test_the_flag_stays_out_of_the_compile_cache_key(monkeypatch):
    """The flag changes no program the compile cache holds, so it is not one
    of the key's env factors."""
    from vllm_torchtpu.compilation import tpu_compiler

    assert FLAG in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    monkeypatch.setenv(FLAG, "1")
    assert FLAG not in tpu_compiler._tpu_compile_env_factors()


EXCLUDED = {
    "pooling": lambda r: setattr(r, "is_pooling_model", True),
    "speculative decoding": lambda r: setattr(r, "speculative_config", object()),
    "pipeline parallelism": lambda r: setattr(
        r.parallel_config, "pipeline_parallel_size", 2
    ),
    "prefill context parallelism": lambda r: setattr(r, "pcp_manager", object()),
    "decode context parallelism": lambda r: setattr(
        r.parallel_config, "decode_context_parallel_size", 2
    ),
    "M-RoPE": lambda r: setattr(r, "uses_mrope", True),
    "multimodal": lambda r: setattr(r, "supports_mm_inputs", True),
    "LoRA": lambda r: setattr(r, "lora_config", object()),
    "KV transfer": lambda r: setattr(r.vllm_config, "kv_transfer_config", object()),
    "DP lockstep": lambda r: (
        setattr(r.parallel_config, "data_parallel_size", 2),
        setattr(r.parallel_config, "enable_expert_parallel", True),
    ),
    "VLLM_TPU_MOST_MODEL_LEN": lambda r: setattr(r, "most_model_len", 64),
    "TPU_GREEDY_TOP_TOKENS": lambda r: setattr(r, "_greedy_head_owner", object()),
    "TPU_DP_VOCAB_PARALLEL": lambda r: setattr(r, "_dp_vocab_logits", True),
    "enforce_eager": lambda r: setattr(r.model_config, "enforce_eager", True),
}


@pytest.mark.parametrize("name", sorted(EXCLUDED))
def test_unsupported_configurations_stay_on_the_stock_path(monkeypatch, name):
    runner = make_runner(monkeypatch)
    monkeypatch.setenv(FLAG, "1")
    controller = DeviceDecodeController.maybe_create(runner)
    assert controller.static_gate_failures() == []
    EXCLUDED[name](runner)
    assert controller.static_gate_failures() == [name]
    controller.precompile_heads()
    controller.initialize_and_precompile()
    assert controller.slots is None and name in controller.disabled_reason
    assert controller.registry == set()


def test_startup_compiles_every_program_a_step_can_run(monkeypatch):
    engine = SimEngine(monkeypatch, async_scheduling=False, device=True)
    runner, controller = engine.runner, engine.controller
    paddings = runner.num_reqs_paddings
    variants = SamplerVariant.all_variants(runner)
    assert {v.num_logprobs for v in variants} == {-1, 0, MAX_LOGPROBS}
    expected = {head_key(rows, v) for rows in paddings for v in variants}
    expected |= {stock_uniform_key(rows, runner.vocab_size) for rows in paddings}
    expected |= {s.program_key for s in DecodeShape.all_decode_shapes(runner)}
    for rows in paddings:
        expected |= {scalar_write_key(rows), block_write_key(0, rows)}
        expected.add(write_back_key(rows))
    assert controller.registry == expected


def test_dry_steps_run_each_decode_shape_on_the_null_block(monkeypatch):
    """The warmup's device steps (on TPU only) feed the backbone the metadata
    of `num_reqs` one-token requests on the null block, and leave the slots
    as they were."""
    engine = SimEngine(monkeypatch, async_scheduling=False, device=True)
    runner, controller = engine.runner, engine.controller
    slots = controller.slots
    metadata = []

    def record(attn_metadata, *_args, **_kwargs):
        metadata.append(attn_metadata["attn.0"])
        return contextlib.nullcontext()

    monkeypatch.setattr(controller_mod, "set_forward_context", record)
    shapes = DecodeShape.all_decode_shapes(runner)
    before = [t.clone() for t in (slots.num_computed, slots.last_token)]
    controller._dry_steps(shapes)
    assert len(metadata) == len(shapes)
    for shape, md in zip(shapes, metadata):
        num_reqs = shape.num_reqs
        assert md.request_distribution.tolist() == [num_reqs] * 3
        assert md.seq_lens[:num_reqs].tolist() == [1] * num_reqs
        assert not md.seq_lens[num_reqs:].any() and not md.block_tables.any()
    assert all(
        torch.equal(a, b)
        for a, b in zip(before, (slots.num_computed, slots.last_token))
    )


def test_stock_noise_is_the_head_noise(monkeypatch):
    engine = SimEngine(monkeypatch, async_scheduling=False, device=True)
    engine.add("a", 5, temperature=0.8)
    engine.add("b", 3, temperature=0.8)
    engine.run(1)
    controller, slots = engine.controller, engine.controller.slots
    runner = engine.runner
    output = engine.sched.schedule()
    runner._update_states(output)
    controller.on_states_updated(output)
    chunk = controller.stock_chunk_plan(output, 0, 2, 8)
    noise = chunk.uniform(runner.vocab_size)
    for row, req_id in enumerate(runner.input_batch.req_ids[:2]):
        slot = slots.slot_of(req_id)
        position = int(runner.input_batch.num_computed_tokens_cpu[row])
        want = counter_uniform_impl(
            slots.seed_lo[slot : slot + 1],
            slots.seed_hi[slot : slot + 1],
            torch.tensor([position], dtype=torch.int32),
            runner.vocab_size,
        )
        assert torch.equal(noise[row : row + 1], want)


def test_the_stock_noise_program_slices_the_plan_itself(monkeypatch):
    """The stock-path noise program takes a chunk's [P, 4] plan whole, as
    uploaded, with the layout its precompile used: no strided column view of
    the plan enters the compiled program."""
    plans = []
    stock_uniform_impl = controller_mod.stock_uniform_impl

    def spy(seed_lo, seed_hi, plan, vocab_size):
        plans.append(plan)
        return stock_uniform_impl(seed_lo, seed_hi, plan, vocab_size)

    monkeypatch.setattr(controller_mod, "stock_uniform_impl", spy)
    engine = SimEngine(monkeypatch, async_scheduling=False, device=True)
    precompiled = {(tuple(p.shape), p.stride(), p.dtype) for p in plans}
    assert {shape[0] for shape, _, _ in precompiled} == set(
        engine.runner.num_reqs_paddings
    )
    plans.clear()
    engine.add("a", 5, temperature=0.8)
    engine.add("b", 3, temperature=0.8)
    engine.run(1)
    assert len(plans) == 1
    plan = plans[0]
    assert plan.is_contiguous() and plan.shape[1] == 4
    assert (tuple(plan.shape), plan.stride(), plan.dtype) in precompiled


@pytest.mark.tpu_test
def test_the_stock_noise_program_draws_the_host_bits_on_tpu(device):
    """The compiled stock_uniform program, which draws a random stock chunk's
    noise when VLLM_TPU_DEVICE_DECODE is on, gives on TPU the bits the host
    draws for each plan row's slot and position, over a whole padded vocabulary."""
    vocab_size = 248320
    gen = torch.Generator().manual_seed(0)
    seed_lo = torch.randint(-(2**31), 2**31, (9,), dtype=torch.int32, generator=gen)
    seed_hi = torch.randint(-(2**31), 2**31, (9,), dtype=torch.int32, generator=gen)
    # A [P, 4] plan as a stock chunk uploads it, each row's slot and position
    # in its first two columns.
    plan = torch.zeros((8, 4), dtype=torch.int32)
    plan[:, 0] = torch.tensor([8, 0, 3, 3, 7, 1, 5, 2])
    plan[:, 1] = torch.tensor([0, 1, 543, 544, 10239, 2**24 + 1, 2**31 - 1, 7])
    got = stock_uniform(
        seed_lo.to(device), seed_hi.to(device), plan.to(device), vocab_size
    )
    rows = plan[:, 0].long()
    want = counter_uniform_impl(
        seed_lo[rows], seed_hi[rows], plan[:, 1].contiguous(), vocab_size
    )
    mismatches = int((got.cpu().view(torch.int32) != want.view(torch.int32)).sum())
    assert mismatches == 0, f"{mismatches} of {want.numel()} uniforms differ"


def test_a_zero_token_step_frees_the_slots_it_preempts(monkeypatch):
    engine = SimEngine(monkeypatch, async_scheduling=False, device=True)
    engine.add("a", 5, temperature=0.8)
    engine.run(2)
    assert engine.controller.slots.slot_of("a") != NO_SLOT
    engine.sched.preempt("a")
    output = engine.step()
    assert output.total_num_scheduled_tokens == 0
    assert engine.controller.slots.slot_of("a") == NO_SLOT
