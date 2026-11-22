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
"""The device-decode controller: the flags, the configurations it refuses,
the programs it compiles, the steps it leaves to the stock path, and the
checks that turn it off."""

import contextlib
from types import SimpleNamespace

import pytest
import torch
from device_decode_test_utils import (
    FLAG,
    MAX_LOGPROBS,
    SimEngine,
    make_runner,
)
from vllm import SamplingParams

from vllm_torchtpu import envs
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform
from vllm_torchtpu.runner.device_decode import controller as controller_mod
from vllm_torchtpu.runner.device_decode.controller import (
    DeviceDecodeController,
    DeviceDecodeState,
    stock_uniform,
    stock_uniform_key,
)
from vllm_torchtpu.runner.device_decode.head import SamplerVariant, head_key
from vllm_torchtpu.runner.device_decode.inputs import DecodeShape
from vllm_torchtpu.runner.device_decode.rng import counter_uniform_impl
from vllm_torchtpu.runner.device_decode.slots import (
    MIRROR_MISMATCH,
    NO_SLOT,
    SLOT_NOT_READY,
    block_write_key,
    scalar_write_key,
    write_back_key,
)
from vllm_torchtpu.runner.thinking_budget_tracker import ThinkingBudgetTracker
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.runner.tpu_runner_async_output import DeviceDecodeInvariantError

VERIFY_FLAG = "VLLM_TPU_DEVICE_DECODE_VERIFY"


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


@pytest.mark.parametrize(("name", "value"), [(FLAG, "maybe"), (VERIFY_FLAG, "-1")])
def test_bad_flag_values_raise(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        getattr(envs, name)


def test_every_rank_gets_the_flags():
    assert {FLAG, VERIFY_FLAG} <= set(TpuPlatform.additional_env_vars)


def test_the_flags_stay_out_of_the_compile_cache_key(monkeypatch):
    """Neither flag changes a program the compile cache holds, so neither is
    one of the key's env factors."""
    from vllm_torchtpu.compilation import tpu_compiler

    for name, value in ((FLAG, "1"), (VERIFY_FLAG, "4")):
        assert name in tpu_compiler._TPU_COMPILE_ENV_IGNORED
        monkeypatch.setenv(name, value)
        assert name not in tpu_compiler._tpu_compile_env_factors()


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
    assert not controller.enabled and name in controller.disabled_reason
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


def test_async_scheduling_compiles_the_int32_token_substitution(monkeypatch):
    calls = []
    for async_scheduling in (False, True):
        runner = make_runner(monkeypatch, async_scheduling=async_scheduling)
        runner._precompile_substitute_placeholder_token = calls.append
        monkeypatch.setenv(FLAG, "1")
        DeviceDecodeController.maybe_create(runner).initialize_and_precompile()
    assert calls == [torch.int32]


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
    """The compiled stock-path noise program, the one serving runs, draws on
    TPU the bits the host draws for each plan row's slot and position, over
    a whole padded vocabulary."""
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


@pytest.mark.parametrize(
    ("params", "reason"),
    [
        ({"min_p": 0.1}, "unsupported_sampling"),
        ({"logit_bias": {3: 2.0}}, "unsupported_sampling"),
        ({"presence_penalty": 0.5}, "unsupported_sampling"),
        ({"prompt_logprobs": 1}, "prompt_logprobs"),
        ({"min_tokens": 3, "stop_token_ids": [5]}, "min_tokens"),
    ],
    ids=["min_p", "logit_bias", "penalty", "prompt_logprobs", "min_tokens"],
)
def test_sampling_the_head_cannot_apply_runs_on_the_stock_path(
    monkeypatch, params, reason
):
    engine = SimEngine(monkeypatch, async_scheduling=False, device=True)
    engine.add("a", 5, temperature=0.8)
    engine.run(3)
    assert engine.controller.counters()["device_steps"] == 2
    engine.add("b", 4, temperature=0.8, max_tokens=4, **params)
    engine.run(3)
    counters = engine.controller.counters()
    assert counters["device_steps"] == 2 and counters[reason] == 2, counters
    # Without the request the batch decodes on the device again.
    engine.run_until_done()
    assert engine.controller.counters()["device_steps"] > 2


def _track_thinking_budgets(runner, start: int, end: int) -> None:
    """The runner's thinking_token_budget tracker, as __init__ builds it,
    for a reasoning config with one-token start and end markers."""
    runner._thinking_budget = ThinkingBudgetTracker(
        SimpleNamespace(
            enabled=True,
            reasoning_start_token_ids=[start],
            reasoning_end_token_ids=[end],
            natural_reasoning_end_token_ids=[end],
        ),
        max_num_reqs=runner.max_num_reqs,
        device=runner.device,
        upload=lambda key, host: runner._upload_unless_unchanged(
            key, host, non_blocking=False
        ),
    )


@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
def test_a_thinking_budget_request_runs_on_the_stock_path(
    monkeypatch, async_scheduling
):
    """The stock sampler enforces thinking_token_budget and the head does
    not: while a request with a budget is in the batch, its steps run on the
    stock path, so the budget ends its thinking as on the stock path. Once it
    finishes, the batch decodes on the device again."""
    start, end = 90, 91
    device = SimEngine(monkeypatch, async_scheduling=async_scheduling, device=True)
    stock = SimEngine(monkeypatch, async_scheduling=async_scheduling, device=True)
    monkeypatch.setattr(stock.controller, "try_execute", lambda *_args: False)
    for engine in (device, stock):
        _track_thinking_budgets(engine.runner, start, end)
        engine.add("p", 4, temperature=0.8)
        # The prompt opens a thinking section; the budget ends it two tokens in.
        params = SamplingParams(temperature=0.8, max_tokens=5, thinking_token_budget=2)
        engine.sched.add("t", [3, 4, start], params)
        engine.tokens["t"] = [[]]
        engine.logprobs["t"] = [[]]
        engine.run_until_done()
    assert device.tokens == stock.tokens
    assert device.tokens["t"][0][2] == end
    counters = device.controller.counters()
    assert counters["unsupported_sampling"] == 4, counters
    assert counters["device_steps"] == 7, counters


def test_a_step_without_compiled_programs_runs_on_the_stock_path(monkeypatch):
    engine = SimEngine(monkeypatch, async_scheduling=False, device=True)
    engine.add("a", 5, temperature=0.8)
    engine.run(2)
    registry = set(engine.controller.registry)
    engine.controller.registry.clear()
    engine.run(1)
    assert engine.controller.counters()["unregistered_shape"] == 1
    engine.controller.registry.update(registry)
    engine.run(1)
    assert engine.controller.counters()["device_steps"] == 2


def test_a_caller_in_inference_mode_still_decodes_on_the_device(monkeypatch):
    """Warmup ran outside inference mode and serving runs inside it: the
    steps are not refused as unregistered shapes, and sample what the stock
    path samples."""
    device = SimEngine(monkeypatch, async_scheduling=True, device=True)
    stock = SimEngine(monkeypatch, async_scheduling=True, device=True)
    monkeypatch.setattr(stock.controller, "try_execute", lambda *_args: False)
    with torch.inference_mode():
        for engine in (device, stock):
            engine.add("a", 5, temperature=0.8, logprobs=0)
            engine.add("b", 3, temperature=0.0)
            engine.run_until_done()
    assert device.tokens == stock.tokens
    assert device.logprobs == stock.logprobs
    counters = device.controller.counters()
    assert "unregistered_shape" not in counters, counters
    assert counters["device_steps"] >= 8, counters


def _decode_ready_step(monkeypatch):
    """An engine whose request "a" decoded on the device once, and its next
    pure-decode step after the state update."""
    engine = SimEngine(monkeypatch, async_scheduling=False, device=True)
    engine.add("a", 5, temperature=0.8)
    engine.run(2)
    assert engine.controller.counters()["device_steps"] == 1
    return engine, engine.sched.schedule()


def test_a_row_with_tokens_left_to_compute_runs_on_the_stock_path(monkeypatch):
    """A device step keeps every row's token; a row whose token the stock
    path would discard (it has tokens to compute after this one) sends the
    step to the stock path before the forward."""
    engine, output = _decode_ready_step(monkeypatch)
    runner, controller = engine.runner, engine.controller
    runner._update_states(output)
    controller.on_states_updated(output)
    assert controller.step_ineligible_reason(output, 1) is None
    runner.input_batch.num_tokens_no_spec[runner.input_batch.req_id_to_index["a"]] += 1
    assert controller.step_ineligible_reason(output, 1) == SLOT_NOT_READY


def test_a_device_step_that_would_drop_its_token_stops_the_engine(monkeypatch):
    """The step's slot has already advanced, so the step fails closed."""
    engine, output = _decode_ready_step(monkeypatch)
    runner = engine.runner
    with torch.compiler.set_stance("force_eager"):
        assert TPUModelRunner.execute_model(runner, output) is None
        assert isinstance(runner.execute_model_state, DeviceDecodeState)
        runner.requests["a"].output_token_ids.append(0)
        with pytest.raises(DeviceDecodeInvariantError, match="tokens to compute"):
            TPUModelRunner.sample_tokens(runner, None)


def test_a_device_step_reads_no_stale_stock_host_inputs(monkeypatch):
    """Device steps leave the stock path's host query_start_loc as the last
    stock step wrote it. Nothing on a device step reads it (the default
    layout plan reads only the token paddings), so overwriting it before
    every step changes nothing."""
    device = SimEngine(monkeypatch, async_scheduling=True, device=True)
    stock = SimEngine(monkeypatch, async_scheduling=True, device=True)
    monkeypatch.setattr(stock.controller, "try_execute", lambda *_args: False)
    for engine in (device, stock):
        engine.add("a", 5, temperature=0.8, max_tokens=14)
        engine.add("b", 9, temperature=0.8)
    for _ in range(3):
        for engine in (device, stock):
            engine.step()
    for engine in (device, stock):
        engine.add("c", 4, temperature=0.8, max_tokens=8)
    for _ in range(60):
        if not (device.sched.reqs or stock.sched.reqs):
            break
        device.runner.query_start_loc_np[:] = -7
        device.step()
        stock.step()
    else:
        raise AssertionError("requests left after 60 steps")
    assert device.tokens == stock.tokens
    assert device.controller.counters()["device_steps"] >= 8


@pytest.mark.parametrize("tensor_parallel_size", [1, 2])
def test_a_slot_that_keeps_disagreeing_turns_the_path_off(
    monkeypatch, tensor_parallel_size
):
    engine = SimEngine(monkeypatch, async_scheduling=False, device=True)
    engine.runner.parallel_config.tensor_parallel_size = tensor_parallel_size
    engine.add("a", 5, temperature=0.8)
    engine.run(2)
    slots = engine.controller.slots

    def drifted(runner, num_reqs):
        slots.last_mismatch_req_id = "a"
        return MIRROR_MISMATCH

    monkeypatch.setattr(slots, "mismatch_reason", drifted)
    engine.run(2)
    assert engine.controller.enabled
    if tensor_parallel_size > 1:
        with pytest.raises(DeviceDecodeInvariantError, match="3 steps in a row"):
            engine.step()
        return
    engine.run(1)
    assert not engine.controller.enabled
    assert "3 steps in a row" in engine.controller.disabled_reason
    assert engine.controller.counters()[MIRROR_MISMATCH] == 3


def test_a_zero_token_step_frees_the_slots_it_preempts(monkeypatch):
    engine = SimEngine(monkeypatch, async_scheduling=False, device=True)
    engine.add("a", 5, temperature=0.8)
    engine.run(2)
    assert engine.controller.slots.slot_of("a") != NO_SLOT
    engine.sched.preempt("a")
    output = engine.step()
    assert output.total_num_scheduled_tokens == 0
    assert engine.controller.slots.slot_of("a") == NO_SLOT


def _verify_engine(monkeypatch, *, async_scheduling=False):
    monkeypatch.setenv(VERIFY_FLAG, "1")
    engine = SimEngine(monkeypatch, async_scheduling=async_scheduling, device=True)
    reference = SimEngine(monkeypatch, async_scheduling=async_scheduling, device=True)
    monkeypatch.setattr(reference.controller, "try_execute", lambda *_args: False)
    for e in (engine, reference):
        e.add("a", 5, temperature=0.8, logprobs=2)
        e.add("b", 9, temperature=0.0)
        e.run(3)
    return engine, reference


@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
def test_verify_mode_checks_every_device_step(monkeypatch, async_scheduling):
    engine, reference = _verify_engine(monkeypatch, async_scheduling=async_scheduling)
    for e in (engine, reference):
        e.run_until_done()
    assert engine.tokens == reference.tokens
    assert engine.logprobs == reference.logprobs
    counters = engine.controller.counters()
    assert counters["verify_steps"] == counters["device_steps"] > 0
    assert counters.get("verify_mismatches", 0) == 0
    assert engine.controller.enabled


@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
def test_verify_mode_after_a_two_chunk_stock_step(monkeypatch, async_scheduling):
    """With 12 attention rows, 14 requests prefill and decode in two stock
    chunks, the first padded to 16 rows. Two requests then finish and the
    other 12 decode on the device: verify mode must look each request's
    previous token up where the stock substitution would, in the padded
    chunks, not in the trimmed host copy."""
    monkeypatch.setenv(VERIFY_FLAG, "1")
    engine = SimEngine(
        monkeypatch, async_scheduling=async_scheduling, device=True, target_rows=12
    )
    reference = SimEngine(
        monkeypatch, async_scheduling=async_scheduling, device=True, target_rows=12
    )
    monkeypatch.setattr(reference.controller, "try_execute", lambda *_args: False)
    for e in (engine, reference):
        for i in range(14):
            e.add(f"r{i:02d}", 3, temperature=0.8, max_tokens=2 if i < 2 else 6)
        e.run_until_done()
    assert engine.tokens == reference.tokens
    counters = engine.controller.counters()
    assert counters["multi_chunk"] >= 1 and counters["device_steps"] >= 1, counters
    assert counters["verify_steps"] == counters["device_steps"], counters
    assert counters.get("verify_mismatches", 0) == 0, counters
    assert engine.controller.enabled, engine.controller.disabled_reason


def test_verify_mode_copies_padded_tensors(monkeypatch):
    """Verify mode copies whole bucketed tensors to the host and trims them
    there: a device slice to the live request count would compile a program
    per count."""
    engine, _ = _verify_engine(monkeypatch)
    controller = engine.controller
    copies: list[tuple[int, tuple[int, ...]]] = []
    live: list[int] = []
    tensor_cpu = torch.Tensor.cpu

    def cpu(self, *args, **kwargs):
        if live:
            copies.append((live[-1], tuple(self.shape)))
        return tensor_cpu(self, *args, **kwargs)

    def recording(method):
        def run(*args, **kwargs):
            live.append(engine.runner.input_batch.num_reqs)
            try:
                return method(*args, **kwargs)
            finally:
                live.pop()

        return run

    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    for name in ("_input_mismatch", "_verify_head"):
        monkeypatch.setattr(controller, name, recording(getattr(controller, name)))
    engine.run_until_done()
    assert controller.counters().get("verify_mismatches", 0) == 0
    assert copies
    # Every copy keeps a bucketed leading dimension (8 rows or more), above
    # the live request count.
    assert all(shape[0] >= 8 > num_reqs for num_reqs, shape in copies), copies


def test_verify_mode_catches_a_corrupt_slot_before_the_forward(monkeypatch):
    engine, reference = _verify_engine(monkeypatch)
    slots = engine.controller.slots
    slots.last_token[slots.slot_of("a")] += 1
    for e in (engine, reference):
        e.run_until_done()
    # The step ran on the stock path, which samples what the slots would have.
    assert engine.tokens == reference.tokens
    assert engine.logprobs == reference.logprobs
    assert not engine.controller.enabled
    assert "input token" in engine.controller.disabled_reason
    assert engine.controller.counters()["verify_mismatches"] == 1


def test_verify_mode_returns_the_stock_tokens_after_a_wrong_sample(monkeypatch):
    engine, reference = _verify_engine(monkeypatch)
    decode_head = controller_mod.decode_head

    def wrong_first_token(*args, **kwargs):
        tokens, packed = decode_head(*args, **kwargs)
        packed[0, 0] = (packed[0, 0] + 1) % engine.runner.vocab_size
        return tokens, packed

    monkeypatch.setattr(controller_mod, "decode_head", wrong_first_token)
    for e in (engine, reference):
        e.run_until_done()
    assert engine.tokens == reference.tokens
    assert engine.logprobs == reference.logprobs
    assert not engine.controller.enabled
    assert "sampled token" in engine.controller.disabled_reason


def test_verify_mode_stops_a_tensor_parallel_engine(monkeypatch):
    engine, _ = _verify_engine(monkeypatch)
    engine.runner.parallel_config.tensor_parallel_size = 2
    slots = engine.controller.slots
    slots.last_token[slots.slot_of("a")] += 1
    with pytest.raises(DeviceDecodeInvariantError, match="input token"):
        engine.step()
