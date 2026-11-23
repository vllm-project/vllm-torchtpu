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
"""Device decode slots: lifecycle against the real runner state updates,
staged writes, the stock write-back and the per-step checks."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm import SamplingParams
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.worker import gpu_model_runner
from vllm.v1.worker.block_table import SlotMappingMode
from vllm.v1.worker.gpu_input_batch import InputBatch

from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.device_decode import slots as slots_mod
from vllm_torchtpu.runner.device_decode.slots import (
    MIRROR_MISMATCH,
    NO_SLOT,
    SLOT_NOT_READY,
    DecodeSlots,
    SlotConfig,
    StockChunkPlan,
    block_write_key,
    derive_request_seed,
    num_reqs_padding,
    scalar_write_key,
    write_back_key,
)
from vllm_torchtpu.runner.kernel_block_table import install_kernel_block_tables
from vllm_torchtpu.runner.min_tokens import MinTokensTracker
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

VOCAB = 1000
BLOCK = 16
PROMPT = 40


def make_runner(
    monkeypatch,
    *,
    max_num_reqs=8,
    widths=(8,),
    kernel_block_sizes=None,
    compact_mamba=False,
    model_seed=0,
):
    """A TPUModelRunner stand-in over a real InputBatch whose `_update_states`
    is the real TPU + vLLM one."""
    monkeypatch.setattr(
        gpu_model_runner, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True)
    )
    runner = object.__new__(TPUModelRunner)
    runner.device = torch.device("cpu")
    runner.max_num_reqs = max_num_reqs
    runner.num_reqs_paddings = tpu_runner._get_req_paddings(
        tpu_runner.MIN_NUM_SEQS, max_num_reqs
    )
    runner.vocab_size = VOCAB
    runner.model_config = SimpleNamespace(seed=model_seed)
    runner.input_batch = InputBatch(
        max_num_reqs=max_num_reqs,
        max_model_len=512,
        max_num_batched_tokens=512,
        device=torch.device("cpu"),
        vocab_size=VOCAB,
        block_sizes=[BLOCK] * len(widths),
        max_num_blocks_per_req=list(widths),
        slot_mapping_modes=[SlotMappingMode.NONE] * len(widths),
    )
    if kernel_block_sizes is not None:
        install_kernel_block_tables(runner.input_batch.block_table, kernel_block_sizes)
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[None] * len(widths))
    runner.requests = {}
    runner.num_prompt_logprobs = {}
    runner.speculative_config = None
    runner.scheduler_config = SimpleNamespace(async_scheduling=False)
    runner.use_async_scheduling = False
    runner.use_async_spec_decode = False
    runner.prev_num_draft_tokens = SimpleNamespace(
        np=np.zeros(max_num_reqs, dtype=np.int32)
    )
    runner._min_tokens = MinTokensTracker()
    runner._pre_async_results = None
    runner._mamba_state_pos = {}
    runner._mamba_offset_seeded = set()
    runner.kv_cache_raw_tensors = []
    runner.late_interaction_runner = SimpleNamespace(
        on_requests_finished=lambda _ids: None,
        register_request=lambda _req_id, _params: None,
    )
    runner.encoder_cache = {}
    runner.is_pooling_model = False
    runner.uses_mrope = False
    runner.uses_xdrope_dim = 0
    runner._may_reorder_batch = lambda _output: None
    runner._unfenced_h2d_sources = []
    runner._free_h2d_sources = {}
    runner._registered_h2d_sources = set()
    runner._has_mamba_state = False
    runner._mamba_slot_by_req_id = {}
    runner._free_mamba_slots = []
    if compact_mamba:
        runner._mamba_slot_stride = 1
        TPUModelRunner._init_mamba_slot_pool(runner, 4 * max_num_reqs)
    return runner


class Scheduler:
    """Feeds scheduler outputs through the runner's real `_update_states`,
    tracking what the scheduler would: each request's computed tokens and
    block ids. Requests prefill their whole prompt in one step and then
    decode one token per step; no tokens are sampled, so `num_tokens` stays
    the prompt length."""

    def __init__(self, runner, num_groups=1):
        self.runner = runner
        self.num_groups = num_groups
        self.computed: dict[str, int] = {}
        self.next_block = 1

    def blocks(self, count):
        ids = list(range(self.next_block, self.next_block + count))
        self.next_block += count
        return tuple(list(ids) for _ in range(self.num_groups))

    def step(
        self,
        *,
        new=None,
        run=(),
        finish=(),
        preempt=None,
        resume=(),
        grow=None,
        num_computed=0,
    ):
        """`new`: req_id -> SamplingParams; `run`: decoding requests; `grow`:
        req_id -> blocks appended this step; `resume`: re-prefilled from 0."""
        new = new or {}
        grow = grow or {}
        output = SchedulerOutput.make_empty()
        output.finished_req_ids = set(finish)
        output.preempted_req_ids = None if preempt is None else set(preempt)
        output.scheduled_new_reqs = [
            NewRequestData(
                req_id=req_id,
                prompt_token_ids=list(range(PROMPT)),
                mm_features=[],
                sampling_params=params,
                pooling_params=None,
                block_ids=self.blocks(3),
                num_computed_tokens=num_computed,
                lora_request=None,
            )
            for req_id, params in new.items()
        ]
        cached = list(run) + list(resume)
        for req_id in resume:
            self.computed[req_id] = 0
        output.scheduled_cached_reqs = CachedRequestData(
            req_ids=cached,
            resumed_req_ids=set(resume),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[
                self.blocks(3)
                if req_id in resume
                else (self.blocks(grow[req_id]) if req_id in grow else None)
                for req_id in cached
            ],
            num_computed_tokens=[self.computed[req_id] for req_id in cached],
            num_output_tokens=[0] * len(cached),
        )
        scheduled = {req_id: PROMPT - num_computed for req_id in new}
        scheduled.update({req_id: 1 for req_id in run})
        scheduled.update({req_id: PROMPT for req_id in resume})
        output.num_scheduled_tokens = scheduled
        output.total_num_scheduled_tokens = sum(scheduled.values())
        for req_id in new:
            self.computed[req_id] = num_computed
        self.runner._update_states(output)
        for req_id, num_tokens in scheduled.items():
            self.computed[req_id] += num_tokens
        return output


def float_bits(value):
    return int(np.float32(value).view(np.int32))


def make_slots(runner):
    return DecodeSlots(SlotConfig.from_runner(runner), runner.device)


def sync(slots, runner, output):
    slots.sync_from_runner(runner, output)
    slots.apply_staged_writes(runner)


def produce(slots, runner, output):
    """The stock write-back of `output`'s single chunk."""
    num_reqs = runner.input_batch.num_reqs
    num_rows = num_reqs_padding(runner, num_reqs)
    plan = slots.plan_stock_chunk(runner, output, 0, num_reqs, num_rows)
    tokens = torch.arange(500, 500 + num_rows, dtype=torch.int64)
    slots.write_back_stock_chunk(runner, plan, tokens.reshape(num_rows, 1))


def device_state(slots):
    return [
        t.clone()
        for t in (
            slots.num_computed,
            slots.last_token,
            slots.temperature,
            slots.top_k,
            slots.top_p,
            slots.seed_lo,
            slots.seed_hi,
            slots.mamba_slot,
            *slots.block_tables,
        )
    ]


def null_row(slots):
    return [t[slots.config.null_slot].clone() for t in device_state(slots)]


class Spy:
    """Counts the H2D uploads and write programs DecodeSlots issues."""

    def __init__(self, monkeypatch, runner):
        self.runner = runner
        self.programs = []
        for name in (
            "_write_scalar_rows_impl",
            "_write_block_rows_impl",
            "_write_back_impl",
        ):
            original = getattr(slots_mod, name)

            def spy(*args, _name=name, _original=original):
                self.programs.append((_name, args[0].shape))
                return _original(*args)

            monkeypatch.setattr(slots_mod, name, spy)
        self.uploads_before = len(runner._unfenced_h2d_sources)

    @property
    def uploads(self):
        return len(self.runner._unfenced_h2d_sources) - self.uploads_before


def test_request_seed_is_a_stable_int32_pair():
    # Known answers pin the derivation every TP rank must share, including
    # halves whose top bit is set.
    assert derive_request_seed(0, 0, "req-0") == (1811414896, 1471700483)
    assert derive_request_seed(0, 0, "req-1") == (-999913485, -1371959553)
    seed = derive_request_seed(0, 0, "req-0")
    assert derive_request_seed(0, 1, "req-0") != seed
    assert derive_request_seed(1, 0, "req-0") != seed


def test_slots_start_at_the_null_values():
    config = SlotConfig(
        max_num_reqs=8, group_widths=(5, 1), vocab_size=VOCAB, model_seed=0
    )
    slots = DecodeSlots(config, torch.device("cpu"))
    assert (config.num_slots, config.null_slot) == (9, 8)
    for name, dtype, fill in [
        ("num_computed", torch.int32, 0),
        ("last_token", torch.int32, 0),
        ("temperature", torch.float32, 1.0),
        ("top_k", torch.int32, 0),
        ("top_p", torch.float32, 1.0),
        ("seed_lo", torch.int32, 0),
        ("seed_hi", torch.int32, 0),
        ("mamba_slot", torch.int32, 0),
    ]:
        tensor = getattr(slots, name)
        assert tensor.dtype == dtype and tensor.shape == (9,), name
        assert torch.all(tensor == fill), name
    assert [tuple(t.shape) for t in slots.block_tables] == [(9, 5), (9, 1)]
    assert all(t.dtype == torch.int32 and not t.any() for t in slots.block_tables)


@pytest.mark.parametrize(
    ("max_num_reqs", "widths"),
    [(8, (5, 1)), (16, (10,)), (256, (128, 1, 1, 1)), (4, ())],
)
def test_device_bytes_are_what_the_slot_tensors_take(max_num_reqs, widths):
    """The HBM reserve counts the slot tensors as `SlotConfig.device_bytes`,
    so it must equal the bytes of every tensor `DecodeSlots` allocates."""
    config = SlotConfig(
        max_num_reqs=max_num_reqs, group_widths=widths, vocab_size=VOCAB, model_seed=0
    )
    slots = DecodeSlots(config, torch.device("cpu"))
    tensors = [v for v in vars(slots).values() if isinstance(v, torch.Tensor)]
    tensors += list(slots.block_tables)
    assert config.device_bytes == sum(t.nbytes for t in tensors)
    assert config.device_bytes == 4 * (max_num_reqs + 1) * (8 + sum(widths))


def test_allocation_is_sorted_and_identical_on_every_rank(monkeypatch):
    """Slots and seeds follow sorted request ids, not the scheduler's or the
    batch's order, so every TP rank fed the same outputs ends up equal."""
    ranks = []
    for _ in range(2):
        runner = make_runner(monkeypatch)
        sched = Scheduler(runner)
        slots = make_slots(runner)
        output = sched.step(new={r: SamplingParams() for r in ["c", "a", "b"]})
        sync(slots, runner, output)
        ranks.append((runner, slots))
    (runner, slots), (_, twin) = ranks
    assert runner.input_batch.req_ids == ["c", "a", "b"]
    assert [slots.slot_of(r) for r in "abc"] == [0, 1, 2]
    assert slots.slot_of("d") == NO_SLOT
    for mine, theirs in zip(device_state(slots), device_state(twin)):
        assert torch.equal(mine, theirs)
    for i, req_id in enumerate("abc"):
        assert (int(slots.seed_lo[i]), int(slots.seed_hi[i])) == (
            derive_request_seed(0, i, req_id)
        )


def test_new_slots_copy_the_input_batch_rows(monkeypatch):
    runner = make_runner(monkeypatch)
    sched = Scheduler(runner)
    slots = make_slots(runner)
    params = {
        "greedy": SamplingParams(temperature=0.0),
        "topk": SamplingParams(temperature=0.7, top_k=5, top_p=0.9),
        "vocab": SamplingParams(temperature=1.3, top_k=VOCAB),
    }
    # A prefix-cache hit: the request starts with 16 computed tokens.
    sync(slots, runner, sched.step(new=params, num_computed=16))
    batch = runner.input_batch
    for req_id in params:
        row, slot = batch.req_id_to_index[req_id], slots.slot_of(req_id)
        for name in ("temperature", "top_p"):
            got = getattr(slots, name)[slot].item()
            want = getattr(batch, f"{name}_cpu")[row]
            assert float_bits(got) == float_bits(want), name
        assert slots.num_computed[slot] == 16
        num_blocks = batch.block_table[0].num_blocks_per_row[row]
        expected = torch.zeros(8, dtype=torch.int32)
        expected[:num_blocks] = batch.block_table[0].get_cpu_tensor()[row, :num_blocks]
        assert torch.equal(slots.block_tables[0][slot], expected)
        assert not slots.decode_ready(slot)
    # The stock sampler's top-k: a value of at least the vocabulary is off.
    top_k = {r: int(slots.top_k[slots.slot_of(r)]) for r in params}
    assert top_k == {"greedy": 0, "topk": 5, "vocab": 0}
    assert slots.temperature[slots.slot_of("greedy")] == 0.0


def test_slots_are_freed_on_finish_preemption_and_resume(monkeypatch):
    runner = make_runner(monkeypatch)
    sched = Scheduler(runner)
    slots = make_slots(runner)
    sync(slots, runner, sched.step(new={r: SamplingParams() for r in "abc"}))
    sync(slots, runner, sched.step(run="bc", finish={"a"}))
    assert slots.slot_of("a") == NO_SLOT
    # The free list is LIFO: the next request takes the slot just freed.
    sync(slots, runner, sched.step(new={"d": SamplingParams()}, run="bc"))
    assert slots.slot_of("d") == 0
    sync(slots, runner, sched.step(run="cd", preempt={"b"}))
    assert slots.slot_of("b") == NO_SLOT
    seeds_before = (slots.seed_lo.clone(), slots.seed_hi.clone())
    output = sched.step(run="cd", resume=["b"])
    sync(slots, runner, output)
    slot = slots.slot_of("b")
    assert slot == 1 and not slots.decode_ready(slot)
    # A new allocation: a new seed, and the block rows of the new block ids.
    assert int(slots.seed_lo[slot]) != int(seeds_before[0][slot])
    row = runner.input_batch.req_id_to_index["b"]
    assert torch.equal(
        slots.block_tables[0][slot, :3],
        runner.input_batch.block_table[0].get_cpu_tensor()[row, :3],
    )


def test_resent_and_resubmitted_requests_get_a_fresh_slot(monkeypatch):
    runner = make_runner(monkeypatch)
    sched = Scheduler(runner)
    slots = make_slots(runner)
    output = sched.step(new={"a": SamplingParams(), "b": SamplingParams()})
    sync(slots, runner, output)
    produce(slots, runner, output)
    assert slots.decode_ready(slots.slot_of("a"))
    # A streaming input update re-sends "a"; an abort resubmits "b".
    output = sched.step(
        new={"a": SamplingParams(), "b": SamplingParams()}, finish={"b"}
    )
    sync(slots, runner, output)
    for i, req_id in [(2, "a"), (3, "b")]:
        slot = slots.slot_of(req_id)
        assert not slots.decode_ready(slot)
        assert (int(slots.seed_lo[slot]), int(slots.seed_hi[slot])) == (
            derive_request_seed(0, i, req_id)
        )


def test_requests_the_runner_dropped_are_freed(monkeypatch):
    runner = make_runner(monkeypatch)
    sched = Scheduler(runner)
    slots = make_slots(runner)
    sync(slots, runner, sched.step(new={"a": SamplingParams(), "b": SamplingParams()}))
    del runner.requests["a"]
    sync(slots, runner, sched.step(run="b"))
    assert slots.slot_of("a") == NO_SLOT and slots.slot_of("b") == 1


def test_steady_decode_uploads_nothing(monkeypatch):
    runner = make_runner(monkeypatch)
    sched = Scheduler(runner)
    slots = make_slots(runner)
    output = sched.step(new={r: SamplingParams() for r in "ab"})
    sync(slots, runner, output)
    produce(slots, runner, output)
    spy = Spy(monkeypatch, runner)
    # Prefill wrote 40 of the 48 tokens of 3 blocks: 7 decode steps fit.
    for _ in range(7):
        sync(slots, runner, sched.step(run="ab"))
    assert spy.uploads == 0 and spy.programs == []


def test_block_growth_rewrites_only_the_grown_row(monkeypatch):
    runner = make_runner(monkeypatch)
    sched = Scheduler(runner)
    slots = make_slots(runner)
    output = sched.step(new={r: SamplingParams() for r in "ab"})
    sync(slots, runner, output)
    produce(slots, runner, output)
    spy = Spy(monkeypatch, runner)
    sync(slots, runner, sched.step(run="ab", grow={"b": 1}))
    assert spy.uploads == 1
    assert spy.programs == [("_write_block_rows_impl", (8 * (1 + 8),))]
    batch = runner.input_batch
    row, slot = batch.req_id_to_index["b"], slots.slot_of("b")
    expected = torch.zeros(8, dtype=torch.int32)
    expected[:4] = batch.block_table[0].get_cpu_tensor()[row, :4]
    assert torch.equal(slots.block_tables[0][slot], expected)
    assert slots.mismatch_reason(runner, 2) is None


def test_split_group_rows_hold_kernel_block_ids(monkeypatch):
    runner = make_runner(monkeypatch, widths=(4, 4), kernel_block_sizes=[8, BLOCK])
    sched = Scheduler(runner, num_groups=2)
    slots = make_slots(runner)
    assert slots.config.group_widths == (8, 4)
    sync(slots, runner, sched.step(new={"a": SamplingParams()}))
    slot = slots.slot_of("a")
    # Manager blocks 1, 2, 3 are kernel blocks 2..7 of the split group.
    assert slots.block_tables[0][slot].tolist() == [2, 3, 4, 5, 6, 7, 0, 0]
    assert slots.block_tables[1][slot].tolist() == [1, 2, 3, 0]


def test_writes_pad_to_the_request_bucket_with_null_rows(monkeypatch):
    runner = make_runner(monkeypatch, max_num_reqs=32)
    sched = Scheduler(runner)
    slots = make_slots(runner)
    before = null_row(slots)
    spy = Spy(monkeypatch, runner)
    sync(
        slots, runner, sched.step(new={f"r{i:02d}": SamplingParams() for i in range(9)})
    )
    # 9 rows of each kind pad to the 16-row bucket.
    assert spy.programs == [
        ("_write_scalar_rows_impl", (16 * 8,)),
        ("_write_block_rows_impl", (16 * (1 + 8),)),
    ]
    assert spy.uploads == 2
    for got, want in zip(null_row(slots), before):
        assert torch.equal(got, want)


def test_write_back_sets_tokens_of_produced_rows_only(monkeypatch):
    runner = make_runner(monkeypatch)
    sched = Scheduler(runner)
    slots = make_slots(runner)
    sync(slots, runner, sched.step(new={r: SamplingParams() for r in "abc"}))
    null_before = null_row(slots)
    a, b, c = (slots.slot_of(r) for r in "abc")
    null = slots.config.null_slot
    plan = StockChunkPlan(
        num_rows_padded=8,
        slots=np.array([a, b, c] + [null] * 5, dtype=np.int32),
        positions=np.array([39, 39, 19] + [0] * 5, dtype=np.int32),
        produced=np.array([True, True, False] + [False] * 5),
        new_num_computed=np.array([40, 40, 20] + [0] * 5, dtype=np.int32),
    )
    sampled = torch.tensor([[11], [12], [13], [14], [15], [16], [17], [18]])
    slots.write_back_stock_chunk(runner, plan, sampled)
    assert slots.last_token[[a, b, c]].tolist() == [11, 12, 0]
    assert slots.num_computed[[a, b, c]].tolist() == [40, 40, 20]
    assert [slots.decode_ready(s) for s in (a, b, c)] == [True, True, False]
    for got, want in zip(null_row(slots), null_before):
        assert torch.equal(got, want)
    # The host mirror followed: the next step's computed tokens match.
    runner.input_batch.num_computed_tokens_cpu[:3] = [40, 40, 20]
    assert slots.mismatch_reason(runner, 2) is None


def test_a_step_without_a_token_leaves_the_slot_not_ready(monkeypatch):
    """A failed KV load rolls a decode-ready request back; its recompute
    chunk produces no token, so `last_token` is stale until one does."""
    runner = make_runner(monkeypatch)
    sched = Scheduler(runner)
    slots = make_slots(runner)
    sync(slots, runner, sched.step(new={r: SamplingParams() for r in "ab"}))
    a, b = (slots.slot_of(r) for r in "ab")
    null = slots.config.null_slot

    def write_back(produced, new_num_computed, token):
        plan = StockChunkPlan(
            num_rows_padded=8,
            slots=np.array([a, b] + [null] * 6, dtype=np.int32),
            positions=np.array([n - 1 for n in new_num_computed] + [0] * 6, np.int32),
            produced=np.array(produced + [False] * 6),
            new_num_computed=np.array(new_num_computed + [0] * 6, dtype=np.int32),
        )
        sampled = torch.full((8, 1), token, dtype=torch.int64)
        slots.write_back_stock_chunk(runner, plan, sampled)

    write_back([True, True], [40, 40], 7)
    assert slots.decode_ready(a) and slots.decode_ready(b)
    # Row a recomputes some of its tokens and produces nothing.
    write_back([False, True], [38, 41], 9)
    assert not slots.decode_ready(a) and slots.decode_ready(b)
    assert slots.last_token[[a, b]].tolist() == [7, 9]
    runner.input_batch.num_computed_tokens_cpu[:2] = [38, 41]
    assert slots.mismatch_reason(runner, 2) == SLOT_NOT_READY
    assert slots.last_mismatch_req_id == "a"
    # The step that finishes the recompute produces a token again.
    write_back([True, True], [41, 42], 11)
    assert slots.decode_ready(a)
    assert slots.last_token[[a, b]].tolist() == [11, 11]


def test_plan_follows_the_stock_discard_rule(monkeypatch):
    runner = make_runner(monkeypatch)
    sched = Scheduler(runner)
    slots = make_slots(runner)
    # "a" prefills all 40 prompt tokens; "b" only 24 of them.
    output = sched.step(new={"a": SamplingParams(), "b": SamplingParams()})
    output.num_scheduled_tokens["b"] = 24
    sync(slots, runner, output)
    plan = slots.plan_stock_chunk(runner, output, 0, 2, 8)
    rows = [runner.input_batch.req_id_to_index[r] for r in "ab"]
    assert plan.slots[rows].tolist() == [slots.slot_of("a"), slots.slot_of("b")]
    assert plan.positions[rows].tolist() == [39, 23]
    assert plan.produced[rows].tolist() == [True, False]
    assert plan.new_num_computed[rows].tolist() == [40, 24]
    assert plan.slots[2:].tolist() == [slots.config.null_slot] * 6
    assert not plan.produced[2:].any() and not plan.new_num_computed[2:].any()
    on_device = plan.to_device(runner)
    assert on_device.shape == (8, 4) and on_device.dtype == torch.int32
    assert on_device[rows].tolist() == [
        [slots.slot_of("a"), 39, 1, 40],
        [slots.slot_of("b"), 23, 0, 24],
    ]


def _decoding(monkeypatch, **kwargs):
    """Two requests prefilled on the stock path, at their first decode step."""
    runner = make_runner(monkeypatch, **kwargs)
    sched = Scheduler(runner, num_groups=len(runner.kv_cache_config.kv_cache_groups))
    slots = make_slots(runner)
    output = sched.step(new={r: SamplingParams() for r in "ab"})
    sync(slots, runner, output)
    produce(slots, runner, output)
    output = sched.step(run="ab")
    sync(slots, runner, output)
    return runner, sched, slots, output


@pytest.mark.parametrize(
    "case, reason",
    [
        ("clean", None),
        ("no_slot", SLOT_NOT_READY),
        ("not_ready", SLOT_NOT_READY),
        ("computed_drift", MIRROR_MISMATCH),
        ("stale_blocks", MIRROR_MISMATCH),
    ],
)
def test_mismatch_reason(monkeypatch, case, reason):
    runner, _, slots, _ = _decoding(monkeypatch)
    batch = runner.input_batch
    row = batch.req_id_to_index["b"]
    if case == "no_slot":
        slots._free_slot("b")
    elif case == "not_ready":
        slots._ready[slots.slot_of("b")] = False
    elif case == "computed_drift":
        # A KV load failure discarded tokens behind the slot's back.
        batch.num_computed_tokens_cpu[row] -= 1
    elif case == "stale_blocks":
        batch.block_table[0].append_row([77], row)
    assert slots.mismatch_reason(runner, 2) == reason
    assert slots.last_mismatch_req_id == (None if reason is None else "b")


def test_unassigned_compact_mamba_slot_is_a_mismatch(monkeypatch):
    runner, sched, slots, output = _decoding(monkeypatch, compact_mamba=True)
    # No stock prepare ran `_build_mamba_state_indices`, so no slot was
    # assigned and the step stays on the stock path, which assigns them.
    assert not slots.mamba_slot.any()
    assert slots.mismatch_reason(runner, 2) == MIRROR_MISMATCH
    runner._build_mamba_state_indices(0, 2, 8)
    produce(slots, runner, output)
    sync(slots, runner, sched.step(run="ab"))
    for req_id in "ab":
        assert (
            slots.mamba_slot[slots.slot_of(req_id)]
            == (runner._mamba_slot_by_req_id[req_id])
        )
    assert slots.mismatch_reason(runner, 2) is None
    # Assigned once, a mamba slot is never staged again.
    spy = Spy(monkeypatch, runner)
    sync(slots, runner, sched.step(run="ab"))
    assert spy.programs == []


def test_note_device_step_advances_only_the_mirror(monkeypatch):
    runner, sched, slots, _ = _decoding(monkeypatch)
    before = device_state(slots)
    idx = slots.idx_mapping_host(runner.input_batch.req_ids[:2])
    slots.note_device_step(idx[:2])
    for got, want in zip(device_state(slots), before):
        assert torch.equal(got, want)
    sched.step(run="ab")
    assert slots.mismatch_reason(runner, 2) is None


def test_idx_mapping_pads_with_the_null_slot(monkeypatch):
    runner, _, slots, _ = _decoding(monkeypatch)
    req_ids = runner.input_batch.req_ids[:2]
    idx = slots.idx_mapping_host(req_ids)
    assert idx.dtype == np.int32 and idx.shape == (8,)
    assert idx.tolist() == [slots.slot_of(r) for r in req_ids] + [8] * 6
    with pytest.raises(KeyError, match="'z'"):
        slots.idx_mapping_host(["a", "z"])


def test_precompile_runs_every_bucket_without_changing_state(monkeypatch):
    runner, _, slots, _ = _decoding(monkeypatch, max_num_reqs=32, widths=(8, 4))
    before = device_state(slots)
    spy = Spy(monkeypatch, runner)
    keys = slots.precompile(runner)
    assert keys == [
        key
        for rows in (8, 16, 32)
        for key in (
            scalar_write_key(rows),
            block_write_key(0, rows),
            block_write_key(1, rows),
            write_back_key(rows),
        )
    ]
    assert len(spy.programs) == len(keys)
    for got, want in zip(device_state(slots), before):
        assert torch.equal(got, want)


def test_num_reqs_padding_matches_the_stock_padding(monkeypatch):
    runner = make_runner(monkeypatch, max_num_reqs=48)
    for num_reqs in range(1, 49):
        assert num_reqs_padding(runner, num_reqs) == (
            tpu_runner._get_padded_num_reqs_with_upper_limit(num_reqs, 48)
        )


@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_write_programs_trace_as_one_graph(monkeypatch, backend):
    """No graph break in the write programs (backend-independent), and the
    traced programs write what the eager ones do. Under aot_eager, AOT
    autograd functionalizes their in-place `index_put_` updates, as it does
    the head's; the slots still end as the eager programs leave them."""
    runner, _, slots, _ = _decoding(monkeypatch)
    a, null = slots.slot_of("a"), slots.config.null_slot
    # Each backend traces the programs afresh.
    torch._dynamo.reset()

    def compiled(fn):
        return torch.compile(fn, backend=backend, fullgraph=True, dynamic=False)

    scalar_rows = np.tile(slots._scalars[null], (8, 1))
    scalar_rows[0] = slots._scalars[a]
    scalar_rows[0, slots_mod._COL_TOP_K] = 7
    scalar_rows[0, slots_mod._COL_TEMPERATURE] = float_bits(0.5)
    compiled(slots_mod._write_scalar_rows_impl)(
        torch.from_numpy(scalar_rows.reshape(-1)),
        slots.num_computed,
        slots.top_k,
        slots.seed_lo,
        slots.seed_hi,
        slots.mamba_slot,
        slots.temperature,
        slots.top_p,
    )
    block_rows = torch.zeros((8, 9), dtype=torch.int32)
    block_rows[:, 0] = null
    block_rows[0, :4] = torch.tensor([a, 9, 8, 7])
    compiled(slots_mod._write_block_rows_impl)(
        block_rows.reshape(-1), slots.block_tables[0]
    )
    plan = torch.tensor([[a, 40, 1, 41]] + [[null, 0, 0, 0]] * 7, dtype=torch.int32)
    compiled(slots_mod._write_back_impl)(
        plan,
        torch.full((8, 1), 9, dtype=torch.int64),
        slots.num_computed,
        slots.last_token,
        null,
    )
    assert (int(slots.top_k[a]), float(slots.temperature[a])) == (7, 0.5)
    assert slots.block_tables[0][a].tolist() == [9, 8, 7, 0, 0, 0, 0, 0]
    assert (int(slots.num_computed[a]), int(slots.last_token[a])) == (41, 9)
    assert int(slots.num_computed[null]) == 0 and int(slots.last_token[null]) == 0
    assert (
        float(slots.temperature[null]) == 1.0 and not slots.block_tables[0][null].any()
    )
