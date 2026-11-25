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
"""Device decode on an M-RoPE model: every slot keeps its request's position
delta, a device step builds the positions the stock `_calc_mrope_positions`
stages, bit for bit, and a step runs on the device only when every token in
it is a generated one."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from device_decode_test_utils import (
    BLOCK,
    FLAG,
    HIDDEN,
    MAX_MODEL_LEN,
    MAX_NUM_REQS,
    VOCAB,
    FakeModel,
    SimEngine,
    SimScheduler,
    enable_device_decode,
    make_runner,
)
from transformers import Qwen3_5MoeTextConfig
from vllm import SamplingParams
from vllm.transformers_utils.config import mrope_num_dims, uses_mrope
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.worker.gpu_input_batch import CachedRequestState

from vllm_torchtpu.runner.device_decode import controller as controller_mod
from vllm_torchtpu.runner.device_decode import inputs as inputs_mod
from vllm_torchtpu.runner.device_decode import slots as slots_mod
from vllm_torchtpu.runner.device_decode.controller import DeviceDecodeController
from vllm_torchtpu.runner.device_decode.inputs import (
    DecodeShape,
    WalkGeometry,
    prepare_decode_inputs,
)
from vllm_torchtpu.runner.device_decode.slots import (
    DecodeSlots,
    SlotConfig,
    StockChunkPlan,
    num_reqs_padding,
)
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

VERIFY_FLAG = "VLLM_TPU_DEVICE_DECODE_VERIFY"

# A tiny Qwen3.5 text config with an M-RoPE section. vLLM's `uses_mrope`
# counts it as M-RoPE, so the runner stages three rows of positions, as it
# does for a multimodal Qwen3.5 checkpoint started text-only.
TINY_CONFIG = Qwen3_5MoeTextConfig(
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=32,
    rope_parameters={
        "rope_type": "default",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.25,
        "mrope_section": [2, 1, 1],
        "mrope_interleaved": True,
    },
)
# Prompt tokens the fake model lays out as an image patch and as a video
# frame. Text tokens stay below them.
IMAGE = VOCAB - 1
FRAME = VOCAB - 2
# The width of an image's grid, and how far a frame moves time ahead.
GRID_WIDTH = 2
FRAME_STRIDE = 9
MAX_POSITION = 2 * MAX_MODEL_LEN

# delta 0, as Qwen3.5 gives every text-only prompt.
TEXT = [5, 6, 7, 8, 9]
# 6 patches on a grid 2 wide take 3 positions: delta -3.
IMAGE_PROMPT = [11, 12, 13] + [IMAGE] * 6 + [14, 15]
# 2 frames, each 9 positions ahead: delta +18.
VIDEO_PROMPT = [21, 22, FRAME, 23, 24, 25, 26, FRAME, 27]
DELTAS = {"text": 0, "image": -3, "video": 2 * FRAME_STRIDE}


class MRopeModel(FakeModel):
    """The fake model of an M-RoPE model. Its hidden states depend on the
    input token and on all three position rows, and it records the positions
    of every forward. Its M-RoPE layout puts a run of IMAGE tokens on a grid
    GRID_WIDTH wide at one time step, which gives the request a negative
    delta, as an image does in Qwen2-VL. A FRAME token moves time
    FRAME_STRIDE positions ahead, which gives a positive delta, as video
    timestamps can in Qwen2.5-VL."""

    supports_mrope = True

    def __init__(self, seed: int = 0):
        super().__init__(seed)
        generator = torch.Generator().manual_seed(seed + 1)
        self.section_embedding = torch.randn(
            3, MAX_POSITION, HIDDEN, generator=generator
        )
        self.positions: list[torch.Tensor] = []

    def __call__(self, input_ids, positions, inputs_embeds=None):
        self.positions.append(positions.clone())
        hidden = self.token_embedding[input_ids.long()]
        for table, section in zip(self.section_embedding, positions):
            hidden = hidden + table[section.long()]
        return hidden

    def get_mrope_input_positions(self, input_tokens, mm_features):
        rows: list[tuple[int, int, int]] = []
        start = 0
        i = 0
        while i < len(input_tokens):
            if input_tokens[i] == IMAGE:
                end = i
                while end < len(input_tokens) and input_tokens[end] == IMAGE:
                    end += 1
                for k in range(end - i):
                    rows.append(
                        (start, start + k // GRID_WIDTH, start + k % GRID_WIDTH)
                    )
                start = max(max(row) for row in rows[i:end]) + 1
                i = end
                continue
            if input_tokens[i] == FRAME:
                start += FRAME_STRIDE
            rows.append((start, start, start))
            start += 1
            i += 1
        positions = torch.tensor(rows, dtype=torch.long).T.contiguous()
        return positions, int(positions.max()) + 1 - len(input_tokens)


def make_mrope_runner(monkeypatch, **kwargs):
    """The device-decode test runner of TINY_CONFIG: `uses_mrope` as vLLM
    derives it, the int32 [3, max_num_tokens + 1] positions buffer of
    `TPUModelRunner.__init__`, and the M-RoPE fake model."""
    runner = make_runner(monkeypatch, **kwargs)
    runner.model = MRopeModel()
    runner.uses_mrope = runner.model_config.uses_mrope = uses_mrope(TINY_CONFIG)
    buffer = torch.zeros((3, runner.max_num_tokens + 1), dtype=torch.int32)
    runner.mrope_positions = SimpleNamespace(cpu=buffer, np=buffer.numpy())
    return runner


def prompts_for(kinds):
    """{req_id: prompt}: one request per kind, ids unlike the batch order."""
    by_kind = {"text": TEXT, "image": IMAGE_PROMPT, "video": VIDEO_PROMPT}
    return {f"{kind}-{(i * 7) % 16:02d}": by_kind[kind] for i, kind in enumerate(kinds)}


def add_decoding_requests(runner, prompts, num_outputs):
    """Adds each prompt's request at a decode step: its prompt and its first
    `num_outputs` tokens are in, and the step feeds the last of them. The
    runner sets the M-RoPE positions and delta as `_update_states` does."""
    for i, (req_id, prompt) in enumerate(prompts.items()):
        seq_len = len(prompt) + num_outputs
        num_blocks = -(-seq_len // BLOCK)
        state = CachedRequestState(
            req_id=req_id,
            prompt_token_ids=list(prompt),
            mm_features=[],
            sampling_params=SamplingParams(temperature=0.8),
            generator=None,
            block_ids=(list(range(1 + 8 * i, 1 + 8 * i + num_blocks)),),
            num_computed_tokens=seq_len - 1,
            output_token_ids=[(31 + 3 * i + k) % 80 + 1 for k in range(num_outputs)],
        )
        TPUModelRunner._init_mrope_positions(runner, state)
        runner.requests[req_id] = state
        runner.input_batch.add_request(state)


def admitted(req_ids):
    output = SchedulerOutput.make_empty()
    output.scheduled_new_reqs = [
        NewRequestData(
            req_id=req_id,
            prompt_token_ids=None,
            mm_features=[],
            sampling_params=None,
            pooling_params=None,
            block_ids=(),
            num_computed_tokens=0,
            lora_request=None,
        )
        for req_id in req_ids
    ]
    return output


def decode_ready_slots(runner):
    """Slots for the batch's requests, as the stock step that sampled each
    request's last token leaves them."""
    slots = DecodeSlots(SlotConfig.from_runner(runner), runner.device)
    req_ids = list(runner.input_batch.req_ids)
    slots.sync_from_runner(runner, admitted(req_ids))
    slots.apply_staged_writes(runner)
    num_rows = num_reqs_padding(runner, len(req_ids))
    pad = num_rows - len(req_ids)
    states = [runner.requests[r] for r in req_ids]
    plan = StockChunkPlan(
        num_rows_padded=num_rows,
        slots=np.array(
            [slots.slot_of(r) for r in req_ids] + [slots.config.null_slot] * pad,
            dtype=np.int32,
        ),
        positions=np.array(
            [s.num_computed_tokens - 1 for s in states] + [0] * pad, dtype=np.int32
        ),
        produced=np.array([True] * len(req_ids) + [False] * pad),
        new_num_computed=np.array(
            [s.num_computed_tokens for s in states] + [0] * pad, dtype=np.int32
        ),
    )
    sampled = torch.tensor([s.output_token_ids[-1] for s in states] + [0] * pad)
    slots.write_back_stock_chunk(runner, plan, sampled.reshape(num_rows, 1))
    return slots


def decode_output(req_ids):
    return SimpleNamespace(
        total_num_scheduled_tokens=len(req_ids),
        num_scheduled_tokens={req_id: 1 for req_id in req_ids},
        scheduled_spec_decode_tokens={},
    )


def test_the_slot_config_follows_the_runner_positions(monkeypatch):
    """The slots of an M-RoPE runner build as many position rows as the
    runner stages: three, as vLLM counts for the tiny config."""
    runner = make_mrope_runner(monkeypatch)
    assert runner.uses_mrope and mrope_num_dims(TINY_CONFIG) == 3
    assert SlotConfig.from_runner(runner).mrope_rows == 3
    assert SlotConfig.from_runner(make_runner(monkeypatch)).mrope_rows == 0


@pytest.mark.parametrize(
    ("max_num_reqs", "widths"),
    [(8, (5, 1)), (16, (10,)), (256, (128, 1, 1, 1)), (4, ())],
)
def test_device_bytes_count_the_mrope_delta(max_num_reqs, widths):
    """With M-RoPE every slot holds one more 4-byte value, the delta, and
    `SlotConfig.device_bytes`, which the HBM reserve counts, includes it."""
    config = SlotConfig(
        max_num_reqs=max_num_reqs,
        group_widths=widths,
        vocab_size=VOCAB,
        model_seed=0,
        mrope_rows=3,
    )
    slots = DecodeSlots(config, torch.device("cpu"))
    tensors = [v for v in vars(slots).values() if isinstance(v, torch.Tensor)]
    tensors += list(slots.block_tables)
    assert slots.mrope_delta is not None
    assert config.device_bytes == sum(t.nbytes for t in tensors)
    assert config.device_bytes == 4 * (max_num_reqs + 1) * (9 + sum(widths))
    flat = SlotConfig(
        max_num_reqs=max_num_reqs, group_widths=widths, vocab_size=VOCAB, model_seed=0
    )
    assert config.device_bytes - flat.device_bytes == 4 * (max_num_reqs + 1)
    assert DecodeSlots(flat, torch.device("cpu")).mrope_delta is None


def test_slots_take_each_request_delta(monkeypatch):
    """A new slot's staged row carries its request's delta, so one scalar
    upload, one column wider, writes it with the other fields."""
    runner = make_mrope_runner(monkeypatch)
    add_decoding_requests(runner, prompts_for(["text", "image", "video"] * 3), 2)
    uploads = []
    original = slots_mod._write_scalar_rows_impl

    def spy(staged, *args):
        uploads.append(tuple(staged.shape))
        return original(staged, *args)

    monkeypatch.setattr(slots_mod, "_write_scalar_rows_impl", spy)
    slots = decode_ready_slots(runner)
    # 9 rows pad to the 16-row bucket, 9 columns each.
    assert uploads == [(16 * 9,)]
    for req_id, state in runner.requests.items():
        kind = req_id.split("-")[0]
        assert state.mrope_position_delta == DELTAS[kind], req_id
        assert int(slots.mrope_delta[slots.slot_of(req_id)]) == DELTAS[kind]
    assert int(slots.mrope_delta[slots.config.null_slot]) == 0


@pytest.mark.parametrize("num_outputs", [1, 4])
@pytest.mark.parametrize(
    ("kinds", "target_rows"),
    [
        (["text"] * 3, MAX_NUM_REQS),
        (["image", "text", "video"], MAX_NUM_REQS),
        (["video", "image", "text"] * 5, MAX_NUM_REQS),
        # A full batch on 2 attention rows: the padded columns of the
        # 16-token bucket map to the last live slot.
        (["image", "video"], 2),
    ],
    ids=["text-only", "multimodal-delta", "multimodal-delta-15", "full-batch"],
)
def test_device_positions_are_the_stock_positions(
    monkeypatch, kinds, target_rows, num_outputs
):
    """The positions a device step builds for an M-RoPE model are those the
    stock `_prepare_inputs` stages from `_calc_mrope_positions`: every live
    column holds the token's index plus its request's delta in all three
    rows, and the padded columns are zero, also those that map to a live
    slot. Same dtype and shape, bit for bit."""
    runner = make_mrope_runner(monkeypatch, target_rows=target_rows)
    prompts = prompts_for(kinds)
    add_decoding_requests(runner, prompts, num_outputs)
    slots = decode_ready_slots(runner)
    req_ids = list(runner.input_batch.req_ids)
    num_reqs = len(req_ids)
    TPUModelRunner._prepare_inputs(runner, decode_output(req_ids), 0, num_reqs)
    stock = runner.position_ids.clone()
    stock_input_ids = runner.input_ids.clone()
    shape = DecodeShape.for_batch(runner, num_reqs)
    inputs = prepare_decode_inputs(
        slots,
        torch.from_numpy(slots.idx_mapping_host(req_ids)),
        shape,
        WalkGeometry.from_runner(runner),
        None,
    )
    got = inputs.positions
    assert got.dtype == stock.dtype == torch.int32
    assert got.shape == stock.shape == (3, shape.num_tokens_padded)
    assert got.is_contiguous()
    assert torch.equal(got, stock)
    assert torch.equal(inputs.input_ids, stock_input_ids)
    expected = [
        runner.requests[r].num_computed_tokens + DELTAS[r.split("-")[0]]
        for r in req_ids
    ]
    for section in range(3):
        assert got[section, :num_reqs].tolist() == expected
    assert not got[:, num_reqs:].any()


def test_the_mrope_prepare_program_traces_as_one_graph(monkeypatch):
    """No graph break in the M-RoPE variant of the prepare program, and the
    traced program computes what the eager one does. Precompiling the
    prepare programs runs this variant."""
    runner = make_mrope_runner(monkeypatch)
    add_decoding_requests(runner, prompts_for(["image", "video", "text"] * 3), 3)
    slots = decode_ready_slots(runner)
    req_ids = list(runner.input_batch.req_ids)
    shape = DecodeShape.for_batch(runner, len(req_ids))
    walk = WalkGeometry.from_runner(runner)
    args = (
        torch.from_numpy(slots.idx_mapping_host(req_ids)),
        slots.num_computed,
        slots.last_token,
        None,
        None,
        tuple(slots.block_tables[gid] for gid in walk.gids),
        shape.num_tokens_padded,
        shape.num_reqs_padded,
        shape.target_rows,
        slots.config.null_slot,
        walk.widths,
        walk.table_geom,
        walk.mamba_geom,
        walk.ckpt_window,
        walk.single_table,
        slots.mrope_delta,
        3,
    )
    torch._dynamo.reset()
    traced = torch.compile(
        inputs_mod._prepare_decode_impl, backend="eager", fullgraph=True, dynamic=False
    )
    eager_outs = inputs_mod._prepare_decode_impl(*args)
    traced_outs = traced(*args)
    assert len(eager_outs) == len(traced_outs)
    for got, want in zip(traced_outs, eager_outs):
        assert torch.equal(got, want)
    assert eager_outs[1].shape == (3, shape.num_tokens_padded)
    calls = []
    original = inputs_mod._prepare_decode_impl

    def spy(*spy_args):
        calls.append(spy_args[15:])
        return original(*spy_args)

    monkeypatch.setattr(inputs_mod, "_prepare_decode_impl", spy)
    inputs_mod.precompile_prepare(runner, slots, walk, [shape])
    assert len(calls) == 1
    assert calls[0][0] is slots.mrope_delta and calls[0][1] == 3


class MRopeEngine(SimEngine):
    """A SimEngine over `make_mrope_runner`, given explicit prompts."""

    def __init__(self, monkeypatch, *, async_scheduling: bool = False):
        self.runner = make_mrope_runner(monkeypatch, async_scheduling=async_scheduling)
        self.sched = SimScheduler()
        self.controller = enable_device_decode(monkeypatch, self.runner)
        self.tokens: dict[str, list[list[int]]] = {}
        self.logprobs: dict[str, list[list[tuple]]] = {}

    def add_prompt(self, req_id: str, prompt: list[int], **params) -> None:
        params.setdefault("max_tokens", 10)
        self.sched.add(req_id, list(prompt), SamplingParams(**params))
        self.tokens.setdefault(req_id, []).append([])
        self.logprobs.setdefault(req_id, []).append([])


def mrope_engines(monkeypatch, *, async_scheduling=False):
    """A device engine, and a reference that takes the stock path every step
    with the keyed noise."""
    device = MRopeEngine(monkeypatch, async_scheduling=async_scheduling)
    stock = MRopeEngine(monkeypatch, async_scheduling=async_scheduling)
    monkeypatch.setattr(stock.controller, "try_execute", lambda *_args: False)
    return device, stock


def test_text_mrope_models_pass_the_startup_gate(monkeypatch):
    """M-RoPE alone no longer keeps every step on the stock path, so an
    M-RoPE model that takes no multimodal inputs decodes on the device. A
    model that accepts multimodal inputs still stays on the stock path."""
    runner = make_mrope_runner(monkeypatch)
    monkeypatch.setenv(FLAG, "1")
    controller = DeviceDecodeController.maybe_create(runner)
    assert controller.static_gate_failures() == []
    runner.supports_mm_inputs = True
    assert controller.static_gate_failures() == ["multimodal"]


@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
def test_device_steps_feed_the_backbone_the_stock_positions(
    monkeypatch, async_scheduling
):
    """Through execute_model and sample_tokens, every forward of the device
    engine gets the positions the reference's gets, bit for bit, so both
    sample the same tokens. Text-only and multimodal-delta requests decode
    on the device together; one is preempted and resumes, and one is
    resubmitted under its id with a prompt of another delta."""
    device, stock = mrope_engines(monkeypatch, async_scheduling=async_scheduling)
    engines = (device, stock)
    batches = []
    try_execute = device.controller.try_execute

    def recording(scheduler_output, target_num_chunks):
        ran = try_execute(scheduler_output, target_num_chunks)
        if ran:
            runner = device.runner
            req_ids = runner.input_batch.req_ids
            batches.append({runner.requests[r].mrope_position_delta for r in req_ids})
        return ran

    monkeypatch.setattr(device.controller, "try_execute", recording)

    def run(steps):
        for _ in range(steps):
            for engine in engines:
                engine.step()

    for engine in engines:
        engine.add_prompt("t", TEXT, temperature=0.8, max_tokens=14)
        engine.add_prompt("i", IMAGE_PROMPT, temperature=0.8, max_tokens=14)
        engine.add_prompt("v", VIDEO_PROMPT, temperature=0.0, max_tokens=12)
    run(4)
    slots = device.controller.slots
    assert int(slots.mrope_delta[slots.slot_of("i")]) == DELTAS["image"]
    for engine in engines:
        engine.sched.preempt("v")
    run(3)
    for engine in engines:
        engine.sched.abort("i")
        engine.add_prompt("i", VIDEO_PROMPT, temperature=0.8, max_tokens=8)
    run(2)
    assert int(slots.mrope_delta[slots.slot_of("i")]) == DELTAS["video"]
    for _ in range(60):
        if not any(engine.sched.reqs for engine in engines):
            break
        run(1)
    else:
        raise AssertionError("requests left after 60 steps")

    assert device.tokens == stock.tokens
    got, want = device.runner.model.positions, stock.runner.model.positions
    assert len(got) == len(want)
    for step, (a, b) in enumerate(zip(got, want)):
        assert a.dtype == b.dtype == torch.int32, step
        assert a.shape == b.shape and a.shape[0] == 3, step
        assert torch.equal(a, b), step
    counters = device.controller.counters()
    assert counters["device_steps"] >= 10, counters
    assert "mrope_prompt_token" not in counters, counters
    assert device.controller.enabled
    # Device steps ran with every kind of delta in the batch.
    assert {0, DELTAS["image"], DELTAS["video"]} <= set().union(*batches)
    assert any(len(deltas) == 3 for deltas in batches), batches


def _decode_ready_step(engine, prompt):
    """`engine` after one device step of a request, and its next pure-decode
    step after the state update."""
    engine.add_prompt("a", prompt, temperature=0.8)
    engine.run(2)
    assert engine.controller.counters()["device_steps"] == 1
    output = engine.sched.schedule()
    engine.runner._update_states(output)
    engine.controller.on_states_updated(output)
    assert not isinstance(engine.controller._plan_step(output, 1), str)
    return output


def test_a_prompt_token_keeps_the_step_on_the_stock_path(monkeypatch):
    """Only generated tokens qualify. A row whose token is still a prompt
    token sends the step to the stock path, which reads that token's
    position from the request's precomputed table. A decode-ready slot never
    has such a row, so the test makes one: the request's prompt runs one
    token past the step."""
    engine = MRopeEngine(monkeypatch)
    output = _decode_ready_step(engine, IMAGE_PROMPT)
    batch = engine.runner.input_batch
    row = batch.req_id_to_index["a"]
    batch.num_prompt_tokens[row] = batch.num_computed_tokens_cpu[row] + 1
    assert engine.controller.try_execute(output, 1) is False
    assert engine.controller.counters()["mrope_prompt_token"] == 1
    # 1D positions do not depend on the prompt: the same row decodes on the
    # device in a model without M-RoPE.
    flat = SimEngine(monkeypatch, async_scheduling=False, device=True)
    flat.sched.add("a", IMAGE_PROMPT, SamplingParams(temperature=0.8))
    flat.tokens["a"], flat.logprobs["a"] = [[]], [[]]
    flat.run(2)
    output = flat.sched.schedule()
    flat.runner._update_states(output)
    flat.controller.on_states_updated(output)
    batch = flat.runner.input_batch
    row = batch.req_id_to_index["a"]
    batch.num_prompt_tokens[row] = batch.num_computed_tokens_cpu[row] + 1
    assert not isinstance(flat.controller._plan_step(output, 1), str)


def test_the_hbm_reserve_counts_the_mrope_delta(monkeypatch):
    """For an M-RoPE model the reserve counts one more int32 per slot, and
    still holds the slots `initialize_and_precompile` builds."""
    monkeypatch.setattr(controller_mod, "_PROGRAM_SCRATCH_BYTES", 3 << 30)
    layers = {f"layer.{i}": object() for i in range(3)}
    monkeypatch.setattr(
        controller_mod, "get_layers_from_vllm_config", lambda *_args: layers
    )
    runner = make_mrope_runner(monkeypatch)
    monkeypatch.setenv(FLAG, "1")
    runner._device_decode = DeviceDecodeController.maybe_create(runner)
    width = runner.max_num_blocks_per_req
    slot_bytes = 4 * (MAX_NUM_REQS + 1) * (9 + 3 * width)
    assert runner.post_profile_hbm_reserves().device_decode == slot_bytes + (3 << 30)
    controller = enable_device_decode(monkeypatch, runner)
    assert controller.slots.config.mrope_rows == 3
    reserved = controller._reserve_slot_config()
    assert reserved.device_bytes >= controller.slots.config.device_bytes


def test_verify_mode_checks_the_mrope_positions(monkeypatch):
    """VLLM_TPU_DEVICE_DECODE_VERIFY compares a device step's positions with
    those `_calc_mrope_positions` stages, and they agree. A slot whose delta
    is off is caught before the forward: the step runs on the stock path and
    device steps turn off."""
    monkeypatch.setenv(VERIFY_FLAG, "1")
    device, stock = mrope_engines(monkeypatch)
    for engine in (device, stock):
        engine.add_prompt("t", TEXT, temperature=0.8)
        engine.add_prompt("i", IMAGE_PROMPT, temperature=0.8)
        engine.add_prompt("v", VIDEO_PROMPT, temperature=0.8)
        engine.run(3)
    counters = device.controller.counters()
    assert counters["verify_steps"] == counters["device_steps"] == 2, counters
    assert counters.get("verify_mismatches", 0) == 0, counters
    slots = device.controller.slots
    slots.mrope_delta[slots.slot_of("v")] += 1
    for engine in (device, stock):
        engine.run_until_done()
    assert device.tokens == stock.tokens
    assert not device.controller.enabled
    assert "M-RoPE position" in device.controller.disabled_reason
    assert device.controller.counters()["verify_input_mismatch"] == 1
