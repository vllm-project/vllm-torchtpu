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
"""device_decode.head: given the same uniforms, the head samples the stock TPU
sampler's tokens and logprobs, advances only the batch's slots, and packs one
int32 buffer that unpacks to the stock dtypes."""

import contextlib
import functools
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.outputs import LogprobsTensors

from tests.runner.precompile_sweep_checks import ops_outside_aten
from vllm_torchtpu.layers.adapter.sample.top_k_top_p import (
    CANDIDATE_BLOCK,
    topk_candidate_blocks,
)
from vllm_torchtpu.runner.device_decode import head
from vllm_torchtpu.runner.device_decode.head import (
    NO_LOGPROBS,
    PackedLayout,
    SamplerVariant,
    decode_head,
    decode_head_impl,
    head_key,
    precompile_head,
)
from vllm_torchtpu.runner.device_decode.rng import counter_uniform_impl
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

MAX_NUM_REQS = 16
NULL_SLOT = MAX_NUM_REQS
VOCAB = 8 * CANDIDATE_BLOCK
# Rows that all set 1 <= top_k <= 2 sample from 4 candidate blocks.
CANDIDATE_BLOCKS = 4
MAX_LOGPROBS = 5
SLOT_FIELDS = (
    "num_computed",
    "last_token",
    "temperature",
    "top_k",
    "top_p",
    "seed_lo",
    "seed_hi",
)
UNSET = VOCAB  # vLLM's InputBatch stores vocab_size as an unset top_k.
CPU = torch.device("cpu")

# Per-row (temperature, top_k, top_p) of InputBatch, tiled over the batch
# (temperature 0.0 is a greedy request), and the (all_greedy,
# topk_candidates, apply_topk_topp) the stock sampler picks for it.
SCENARIOS = {
    "greedy": ([(0.0, UNSET, 1.0)], (True, 0, True)),
    "random": (
        [(0.7, UNSET, 1.0), (0.0, UNSET, 1.0), (1.0, UNSET, 1.0), (1.3, UNSET, 1.0)],
        (False, 0, False),
    ),
    "top_k_top_p": (
        [(0.7, 5, 1.0), (0.0, UNSET, 0.9), (1.0, 3, 0.8), (1.3, UNSET, 1.0)],
        (False, 0, True),
    ),
    "candidates": (
        [(0.7, 1, 1.0), (1.0, 2, 0.9), (0.0, 2, 1.0), (1.3, 1, 0.7)],
        (False, CANDIDATE_BLOCKS, True),
    ),
}
# InputBatch.max_num_logprobs -> SamplerVariant.num_logprobs
LOGPROBS = {None: NO_LOGPROBS, 0: 0, 3: MAX_LOGPROBS}


def _runner(
    scenario: str,
    num_reqs: int,
    max_num_logprobs: int | None,
    *,
    max_num_reqs: int = MAX_NUM_REQS,
    device: torch.device = CPU,
):
    """A runner stand-in whose stock sampling helpers are the real ones; on
    TPU they run the stock compiled sampler and logprobs programs."""
    rows, _ = SCENARIOS[scenario]
    temperature, top_k, top_p = zip(*(rows[i % len(rows)] for i in range(num_reqs)))
    runner = SimpleNamespace(
        vocab_size=VOCAB,
        max_num_reqs=max_num_reqs,
        device=device,
        model_config=SimpleNamespace(max_logprobs=MAX_LOGPROBS),
        input_batch=SimpleNamespace(
            temperature_cpu_tensor=torch.tensor(temperature, dtype=torch.float32),
            top_k_cpu_tensor=torch.tensor(top_k, dtype=torch.int32),
            top_p_cpu_tensor=torch.tensor(top_p, dtype=torch.float32),
            all_greedy=all(t == 0.0 for t in temperature),
            no_top_k=all(not 0 < k < VOCAB for k in top_k),
            no_top_p=all(p >= 1.0 for p in top_p),
            max_num_logprobs=max_num_logprobs,
        ),
        _topk_candidate_blocks=lambda: topk_candidate_blocks(CANDIDATE_BLOCKS, VOCAB),
    )
    for fn in (
        TPUModelRunner._build_padded_sampling_params,
        TPUModelRunner._sampler_topk_candidates,
        TPUModelRunner._sampler_applies_topk_topp,
        TPUModelRunner._sample_chunk_logits,
        TPUModelRunner._greedy_sample,
    ):
        setattr(runner, fn.__name__, fn.__get__(runner))
    if device.type == "tpu":
        runner.sample_from_logits_func = TPUModelRunner.sample_from_logits.__get__(
            runner
        )
        runner.gather_logprobs = TPUModelRunner.gather_logprobs.__get__(runner)
    else:
        runner.sample_from_logits_func = functools.partial(
            TPUModelRunner.sample_from_logits.__wrapped__, runner
        )
        runner.gather_logprobs = functools.partial(
            TPUModelRunner.gather_logprobs.__wrapped__, runner
        )
    return runner


def _slots(runner, num_reqs: int, gen: torch.Generator):
    """CPU slot tensors, every slot filled; batch row i sits at a scattered
    slot holding the row's InputBatch sampling params, and the null slot
    (runner.max_num_reqs) holds its neutral values. Returns (slots,
    idx_mapping)."""
    null_slot = runner.max_num_reqs
    num_slots = null_slot + 1

    def words():
        return torch.randint(-(2**31), 2**31, (num_slots,), generator=gen).int()

    slots = {
        "num_computed": torch.randint(0, 2**20, (num_slots,), generator=gen).int(),
        "last_token": torch.randint(0, VOCAB, (num_slots,), generator=gen).int(),
        "temperature": torch.rand(num_slots, generator=gen) + 0.5,
        "top_k": torch.randint(1, 3, (num_slots,), generator=gen).int(),
        "top_p": torch.full((num_slots,), 0.5),
        "seed_lo": words(),
        "seed_hi": words(),
    }
    for field, null in zip(SLOT_FIELDS, (0, 0, 1.0, 0, 1.0, 0, 0)):
        slots[field][null_slot] = null
    order = torch.randperm(null_slot, generator=gen)[:num_reqs]
    idx_mapping = torch.full((null_slot,), null_slot, dtype=torch.int32)
    idx_mapping[:num_reqs] = order.int()
    batch = runner.input_batch
    top_k = batch.top_k_cpu_tensor[:num_reqs]
    slots["temperature"][order] = batch.temperature_cpu_tensor[:num_reqs]
    slots["top_k"][order] = torch.where(top_k >= VOCAB, 0, top_k)
    slots["top_p"][order] = batch.top_p_cpu_tensor[:num_reqs]
    return slots, idx_mapping


def _stock_tokens(runner, logits, u, num_reqs, monkeypatch):
    """The stock chunk sampler's tokens, with `u` as its uniforms."""

    def rand_like(like, *, dtype, generator):
        assert like is logits and dtype == torch.float32
        return u

    with monkeypatch.context() as patch:
        patch.setattr(torch, "rand_like", rand_like)
        tokens, _ = runner._sample_chunk_logits(
            logits,
            cur_start_idx=0,
            cur_end_idx=num_reqs,
            all_greedy=runner.input_batch.all_greedy,
            grammar_output=None,
            sampling_generator=None,
        )
    return tokens


def _bits(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array).view(np.int32)


def _assert_same_floats(got: np.ndarray, want: np.ndarray) -> None:
    ulps = np.abs(_bits(got).astype(np.int64) - _bits(want).astype(np.int64))
    np.testing.assert_array_equal(
        _bits(got), _bits(want), err_msg=f"max {ulps.max()} ulp"
    )


def _check_head_matches_stock(
    runner, num_reqs, num_rows, monkeypatch, *, seed, check_sampler_inputs
):
    device, null_slot = runner.device, runner.max_num_reqs
    gen = torch.Generator().manual_seed(seed)
    variant = SamplerVariant.for_batch(runner, num_reqs)
    logits = (torch.randn(num_rows, VOCAB, generator=gen) * 3).to(torch.bfloat16)
    logits = logits.to(device)
    before, idx_mapping = _slots(runner, num_reqs, gen)
    slots = {field: t.clone().to(device) for field, t in before.items()}
    rows = idx_mapping[:num_rows].long()
    # Each row's keyed noise, drawn on the host; the head draws its own.
    u = counter_uniform_impl(
        before["seed_lo"][rows],
        before["seed_hi"][rows],
        before["num_computed"][rows],
        VOCAB,
    ).to(device)
    fed = {}
    if check_sampler_inputs:
        sample = head.sample_from_logits_impl

        def spy(logits, temperatures, u, top_k, top_p, **kwargs):
            fed.update(temperatures=temperatures, u=u, top_k=top_k, top_p=top_p)
            return sample(logits, temperatures, u, top_k, top_p, **kwargs)

        monkeypatch.setattr(head, "sample_from_logits_impl", spy)

    sampled, packed = decode_head(
        logits,
        idx_mapping.to(device),
        *(slots[field] for field in SLOT_FIELDS),
        variant=variant,
        null_slot=null_slot,
    )

    if check_sampler_inputs:
        # The sampler sees the padded params stock builds, in its dtypes
        # (temperature in the logits dtype); padded rows read the null slot.
        temperatures, top_k, top_p = runner._build_padded_sampling_params(
            0, num_reqs, logits
        )
        assert fed["temperatures"].dtype == logits.dtype
        assert torch.equal(fed["temperatures"], temperatures)
        assert torch.equal(fed["top_k"], top_k)
        assert torch.equal(fed["top_p"], top_p)
        if not variant.all_greedy:
            assert torch.equal(fed["u"], u)
        # No output aliases another output or an input.
        inputs = [logits, *slots.values()]
        pointers = {t.untyped_storage().data_ptr() for t in inputs}
        outputs = {t.untyped_storage().data_ptr() for t in (sampled, packed)}
        assert len(outputs) == 2 and not outputs & pointers

    layout = PackedLayout(num_rows, variant.num_logprobs)
    sampled, packed = sampled.cpu(), packed.cpu()
    assert packed.dtype == torch.int32 and packed.shape == (num_rows, layout.width)
    assert sampled.dtype == torch.int32 and sampled.shape == (num_rows,)
    assert torch.equal(sampled, packed[:, 0])
    tokens, echo, logprobs = layout.unpack(packed, num_reqs)
    stock = _stock_tokens(runner, logits, u, num_reqs, monkeypatch)
    assert tokens.dtype == torch.int64 and tokens.shape == (num_reqs, 1)
    assert torch.equal(tokens, stock[:num_reqs].cpu())
    live = idx_mapping[:num_reqs].long()
    assert echo.dtype == np.int32
    np.testing.assert_array_equal(echo, before["num_computed"][live].numpy())

    if variant.num_logprobs == NO_LOGPROBS:
        assert logprobs is None
    else:
        lp = runner.gather_logprobs(logits, stock)
        want = LogprobsTensors(
            lp.logprob_token_ids[:num_reqs],
            lp.logprobs[:num_reqs],
            lp.selected_token_ranks[:num_reqs],
        ).tolists()
        cols = layout.logprob_cols
        assert logprobs.logprob_token_ids.dtype == np.int32
        assert logprobs.logprobs.dtype == np.float32
        assert logprobs.sampled_token_ranks.dtype == np.int32
        assert logprobs.cu_num_generated_tokens is None
        np.testing.assert_array_equal(
            logprobs.logprob_token_ids, want.logprob_token_ids[:, :cols]
        )
        _assert_same_floats(logprobs.logprobs, want.logprobs[:, :cols])
        np.testing.assert_array_equal(
            logprobs.sampled_token_ranks, want.sampled_token_ranks
        )

    # post_update: only the batch's slots advance; the null slot and the
    # sampling params never change.
    after = {field: t.cpu() for field, t in slots.items()}
    expect_computed = before["num_computed"].clone()
    expect_computed[live] += 1
    expect_last = before["last_token"].clone()
    expect_last[live] = tokens.view(-1).int()
    assert torch.equal(after["num_computed"], expect_computed)
    assert torch.equal(after["last_token"], expect_last)
    for field in SLOT_FIELDS[2:]:
        assert torch.equal(after[field], before[field]), field


@pytest.mark.parametrize("max_num_logprobs", list(LOGPROBS))
@pytest.mark.parametrize("scenario", list(SCENARIOS))
@pytest.mark.parametrize("num_reqs", [5, MAX_NUM_REQS])
def test_head_matches_the_stock_sampler(
    num_reqs, scenario, max_num_logprobs, monkeypatch
):
    runner = _runner(scenario, num_reqs, max_num_logprobs)
    assert SamplerVariant.for_batch(runner, num_reqs) == SamplerVariant(
        *SCENARIOS[scenario][1], LOGPROBS[max_num_logprobs]
    )
    _check_head_matches_stock(
        runner,
        num_reqs,
        8 if num_reqs <= 8 else 16,
        monkeypatch,
        seed=num_reqs * 100 + len(scenario),
        check_sampler_inputs=True,
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize("num_reqs,num_rows", [(5, 8), (200, 256)])
def test_head_matches_the_stock_sampler_on_tpu(device, num_reqs, num_rows, monkeypatch):
    """Every head variant against the stock compiled sampler and logprobs
    programs on TPU: tokens, ranks and logprob bits equal."""
    # Each (rows, variant) is its own dynamic=False head program: 12 per case,
    # past Dynamo's default limit of 8 per function, which fullgraph=True
    # turns into FailOnRecompileLimitHit. The runner raises the limits in
    # TPUModelRunner.__init__, which this stand-in skips. config.patch, not
    # monkeypatch.setattr: under pytest 9.2 monkeypatch's undo deletes torch
    # config entries for the rest of the session (pytest-dev/pytest#15287).
    try:
        with torch._dynamo.config.patch(
            cache_size_limit=1024, accumulated_recompile_limit=1024
        ):
            for scenario in SCENARIOS:
                for max_num_logprobs in LOGPROBS:
                    runner = _runner(
                        scenario,
                        num_reqs,
                        max_num_logprobs,
                        max_num_reqs=256,
                        device=device,
                    )
                    _check_head_matches_stock(
                        runner,
                        num_reqs,
                        num_rows,
                        monkeypatch,
                        seed=num_rows + len(scenario),
                        check_sampler_inputs=False,
                    )
    finally:
        # Drop the entries compiled past the default limit, so a later test
        # that compiles the head or the stock sampler starts afresh.
        torch._dynamo.reset()


def test_variant_follows_the_stock_decisions():
    for scenario, (_, flags) in SCENARIOS.items():
        for max_num_logprobs, num_logprobs in LOGPROBS.items():
            runner = _runner(scenario, 6, max_num_logprobs)
            variant = SamplerVariant.for_batch(runner, 6)
            assert variant == SamplerVariant(*flags, num_logprobs)
            assert variant in SamplerVariant.all_variants(runner)
    with pytest.raises(NotImplementedError, match="full logprobs"):
        SamplerVariant.for_batch(_runner("random", 6, -1), 6)


def test_all_variants_cover_each_sampler_graph_and_logprob_width():
    runner = _runner("random", 6, None)
    samplers = {(False, 0, True), (False, 0, False), (False, 4, True), (True, 0, True)}
    variants = SamplerVariant.all_variants(runner)
    assert len(variants) == len(set(variants)) == 12
    assert {(v.all_greedy, v.topk_candidates, v.apply_topk_topp) for v in variants} == (
        samplers
    )
    assert {v.num_logprobs for v in variants} == {NO_LOGPROBS, 0, MAX_LOGPROBS}
    runner.model_config.max_logprobs = 0
    runner._topk_candidate_blocks = lambda: 0
    assert len(SamplerVariant.all_variants(runner)) == 3 * 2


def test_packed_layout_widths():
    for num_logprobs, cols, width in ((NO_LOGPROBS, 0, 2), (0, 1, 5), (5, 6, 15)):
        layout = PackedLayout(8, num_logprobs)
        assert (layout.logprob_cols, layout.width) == (cols, width)


def test_unpack_keeps_every_logprob_bit():
    special = np.array(
        [-0.0, -np.inf, np.inf, 1e-45, -1.5, 0.0], dtype=np.float32
    ).view(np.int32)
    special[5] = 0x7FC12345  # a NaN with a payload
    layout = PackedLayout(4, 2)
    packed = torch.zeros((4, layout.width), dtype=torch.int32)
    packed[:, 0] = torch.tensor([7, 8, 9, 10])
    packed[:, 1] = torch.tensor([100, 200, 300, 400])
    packed[:, 2:5] = torch.tensor([[7, 1, 2], [8, 3, 4], [9, 5, 6], [10, 0, 0]])
    packed[:2, 5:8] = torch.from_numpy(special.reshape(2, 3))
    packed[:, 8] = torch.tensor([1, 2, 3, 4])

    tokens, echo, logprobs = layout.unpack(packed, 3)

    assert tokens.tolist() == [[7], [8], [9]]
    assert echo.tolist() == [100, 200, 300]
    assert logprobs.logprob_token_ids.tolist() == [[7, 1, 2], [8, 3, 4], [9, 5, 6]]
    np.testing.assert_array_equal(_bits(logprobs.logprobs[:2]), special.reshape(2, 3))
    assert logprobs.sampled_token_ranks.tolist() == [1, 2, 3]
    # The host arrays own their memory.
    packed.zero_()
    assert echo.tolist() == [100, 200, 300]


def test_unpack_rejects_a_buffer_of_another_layout():
    layout = PackedLayout(8, 0)
    with pytest.raises(ValueError, match="expected int32"):
        layout.unpack(torch.zeros((8, 2), dtype=torch.int32), 4)
    with pytest.raises(ValueError, match="expected int32"):
        layout.unpack(torch.zeros((8, 5), dtype=torch.int64), 4)
    with pytest.raises(ValueError, match="do not fit"):
        layout.unpack(torch.zeros((8, 5), dtype=torch.int32), 9)


def _uniform_batch(num_slots, gen, **fields):
    slots = {
        "num_computed": torch.zeros(num_slots, dtype=torch.int32),
        "last_token": torch.zeros(num_slots, dtype=torch.int32),
        "temperature": torch.ones(num_slots),
        "top_k": torch.zeros(num_slots, dtype=torch.int32),
        "top_p": torch.ones(num_slots),
        "seed_lo": torch.randint(-(2**31), 2**31, (num_slots,), generator=gen).int(),
        "seed_hi": torch.randint(-(2**31), 2**31, (num_slots,), generator=gen).int(),
    }
    for field, values in fields.items():
        slots[field][: len(values)] = values
    return slots


def test_draws_follow_the_tempered_softmax():
    """A chi-square test of 20480 draws from one row of logits, one per
    (seed, position) key, against softmax(logits / temperature)."""
    gen = torch.Generator().manual_seed(1)
    rows, vocab, steps, temperature = 512, 16, 40, 0.7
    row = torch.linspace(-2.0, 2.0, vocab)
    slots = _uniform_batch(rows + 1, gen, temperature=torch.full((rows,), temperature))
    idx_mapping = torch.arange(rows, dtype=torch.int32)
    logits = row.expand(rows, vocab).contiguous()
    counts = torch.zeros(vocab, dtype=torch.float64)
    variant = SamplerVariant(False, 0, False, NO_LOGPROBS)
    for _ in range(steps):
        sampled, _ = decode_head(
            logits,
            idx_mapping,
            *(slots[field] for field in SLOT_FIELDS),
            variant=variant,
            null_slot=rows,
        )
        counts += torch.bincount(sampled.long(), minlength=vocab).double()
    # Each step sampled at the next position.
    assert torch.equal(slots["num_computed"][:rows], torch.full((rows,), steps).int())

    scaled = (row / torch.tensor(temperature)).double()
    expected = torch.softmax(scaled, dim=0) * rows * steps
    chi2 = ((counts - expected) ** 2 / expected).sum()
    df = torch.tensor((vocab - 1) / 2, dtype=torch.float64)
    p_value = torch.special.gammaincc(df, chi2 / 2).item()
    assert p_value > 0.01, (p_value, counts.tolist(), expected.tolist())


def test_draws_stay_in_the_top_k_and_top_p_sets():
    gen = torch.Generator().manual_seed(2)
    rows, vocab, steps = 512, 2 * CANDIDATE_BLOCK, 20
    half = rows // 2
    # Token i has logit -0.05 i: the top-p 0.5 set is tokens 0..13 (mass
    # 0.503; tokens 0..12 hold 0.478).
    logits = (-0.05 * torch.arange(vocab, dtype=torch.float32)).expand(rows, vocab)
    slots = _uniform_batch(
        rows + 1,
        gen,
        top_k=torch.full((half,), 3, dtype=torch.int32),
        top_p=torch.cat([torch.ones(half), torch.full((half,), 0.5)]),
    )
    idx_mapping = torch.arange(rows, dtype=torch.int32)
    variant = SamplerVariant(False, 0, True, NO_LOGPROBS)
    draws = []
    for _ in range(steps):
        sampled, _ = decode_head(
            logits.contiguous(),
            idx_mapping,
            *(slots[field] for field in SLOT_FIELDS),
            variant=variant,
            null_slot=rows,
        )
        draws.append(sampled)
    draws = torch.stack(draws)
    assert set(draws[:, :half].unique().tolist()) == {0, 1, 2}
    assert set(draws[:, half:].unique().tolist()) == set(range(14))


def _variants():
    samplers = [(True, 0, True), (False, 0, False), (False, 0, True)]
    samplers.append((False, CANDIDATE_BLOCKS, True))
    return [
        SamplerVariant(*s, k) for s in samplers for k in (NO_LOGPROBS, 0, MAX_LOGPROBS)
    ]


def _variant_id(v: SamplerVariant) -> str:
    sampler = (
        "greedy" if v.all_greedy else f"c{v.topk_candidates}-t{v.apply_topk_topp:d}"
    )
    return f"{sampler}-k{v.num_logprobs}"


def _compile_and_compare(variant, backend):
    """Run the head compiled with `backend` and eagerly on equal inputs;
    outputs and slots must end equal."""
    gen = torch.Generator().manual_seed(3)
    runner = _runner("candidates", 5, None)
    logits = (torch.randn(8, VOCAB, generator=gen) * 3).to(torch.bfloat16)
    slots, idx_mapping = _slots(runner, 5, gen)
    compiled_slots = {field: t.clone() for field, t in slots.items()}
    args = (
        variant.all_greedy,
        variant.topk_candidates,
        variant.apply_topk_topp,
        variant.num_logprobs,
        NULL_SLOT,
    )
    want = decode_head_impl(logits, idx_mapping, *slots.values(), *args)
    torch._dynamo.reset()
    try:
        got = torch.compile(
            decode_head_impl, backend=backend, fullgraph=True, dynamic=False
        )(logits, idx_mapping, *compiled_slots.values(), *args)
    finally:
        # Drop this backend's entries so later tests compile afresh.
        torch._dynamo.reset()
    for g, w in zip(got, want):
        assert torch.equal(g, w)
    for field in SLOT_FIELDS:
        assert torch.equal(compiled_slots[field], slots[field]), field


@pytest.mark.parametrize("variant", _variants(), ids=_variant_id)
def test_head_traces_to_one_graph(variant):
    """Each variant is one graph, with no host sync (fullgraph) and nothing
    that may communicate, computing what the eager head computes."""
    graphs = []

    def record(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    _compile_and_compare(variant, record)
    assert len(graphs) == 1
    assert ops_outside_aten(graphs[0]) == []


@pytest.mark.parametrize(
    "variant",
    [SamplerVariant(True, 0, True, MAX_LOGPROBS), SamplerVariant(False, 0, False, 0)],
    ids=_variant_id,
)
def test_functionalized_head_keeps_its_slot_updates(variant):
    """AOT autograd functionalizes the in-place slot updates, as a TPU compile
    does; the slots still end as the eager head leaves them."""
    _compile_and_compare(variant, "aot_eager")


def test_precompile_head_compiles_every_variant_and_returns_its_keys(monkeypatch):
    synced, calls, timed = [], [], []
    monkeypatch.setattr(head, "synchronize_tensors", synced.append)
    real_decode_head = head.decode_head

    def spy(logits, idx_mapping, *slot_tensors, variant, null_slot):
        calls.append((logits.shape, idx_mapping.clone(), slot_tensors, null_slot))
        return real_decode_head(
            logits, idx_mapping, *slot_tensors, variant=variant, null_slot=null_slot
        )

    monkeypatch.setattr(head, "decode_head", spy)
    runner = _runner("random", 4, None)
    runner._dummy_logits = lambda n: torch.zeros((n, VOCAB), dtype=torch.bfloat16)
    runner._precompile_timed = lambda name: timed.append(name) or (
        contextlib.nullcontext()
    )
    variants = SamplerVariant.all_variants(runner)

    keys = precompile_head(runner, variants, [8, 16])

    assert keys == [head_key(n, v) for n in (8, 16) for v in variants]
    assert all(key[3] is False for key in keys)
    assert timed == ["device_decode_head"]
    assert len(synced) == len(calls) == len(keys)
    for (shape, idx_mapping, slot_tensors, null_slot), key in zip(calls, keys):
        assert shape == (key[1], VOCAB)
        assert null_slot == NULL_SLOT
        assert torch.equal(idx_mapping, torch.full((MAX_NUM_REQS,), NULL_SLOT).int())
        assert [t.dtype for t in slot_tensors] == [
            torch.int32,
            torch.int32,
            torch.float32,
            torch.int32,
            torch.float32,
            torch.int32,
            torch.int32,
        ]
        assert all(t.shape == (MAX_NUM_REQS + 1,) for t in slot_tensors)
        assert all(t.is_contiguous() for t in slot_tensors)
