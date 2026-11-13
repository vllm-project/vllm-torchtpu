# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_VERIFY_SHARD_CANDIDATES: the shard-local verify must give the
fused verify's tokens and row offsets bit for bit.

TP ranks are simulated on CPU: the full logits are split into equal
contiguous vocab slices, one thread per rank runs the per-rank program, and a
barrier-based all-gather stands in for the TP collective. The reference is the
fused verify (`_verify_fused_one_draw_impl` / `_verify_fused_impl`) with top-k
candidates and candidates-only target rows on the full logits.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.layers.adapter.sample import shard_candidates
from vllm_torchtpu.layers.adapter.sample.top_k_top_p import CANDIDATE_BLOCK
from vllm_torchtpu.runner.tpu_runner import (
    _split_verify_uniforms,
    _verify_fused_impl,
    _verify_fused_one_draw_impl,
    _verify_shard_impl,
    _verify_shard_one_draw_impl,
    _verify_uniforms_count,
)

K = 3


class FakeTpGroup:
    """All-gather across `size` threads, concatenating in rank order."""

    def __init__(self, size: int):
        self.size = size
        self.barrier = threading.Barrier(size)
        self.slots: list[torch.Tensor | None] = [None] * size
        self.calls = [0] * size

    def all_gather(self, rank: int):
        def gather(x: torch.Tensor, dim: int) -> torch.Tensor:
            self.calls[rank] += 1
            self.slots[rank] = x
            self.barrier.wait()
            out = torch.cat([t for t in self.slots if t is not None], dim=dim)
            self.barrier.wait()
            return out

        return gather


def run_ranks(fn, tp: int, full_logits: torch.Tensor, args: tuple, num_reqs):
    """fn(local_logits, *args, rank, tp, num_reqs, all_gather) on every rank."""
    width = full_logits.shape[-1] // tp
    group = FakeTpGroup(tp)

    def one(rank):
        local = full_logits[:, rank * width : (rank + 1) * width].contiguous()
        return fn(local, *args, rank, tp, num_reqs, group.all_gather(rank))

    with ThreadPoolExecutor(max_workers=tp) as pool:
        outs = list(pool.map(one, range(tp)))
    assert len(set(group.calls)) == 1, group.calls
    return outs


def make_case(seed: int, tp: int, num_vocab_blocks: int, num_blocks: int, kind: str):
    """A verify chunk: B padded requests (some padding rows), K drafts each,
    logits rows [B * (K + 1), V] in bf16 on a coarse grid (many ties)."""
    g = torch.Generator().manual_seed(seed)
    vocab = num_vocab_blocks * CANDIDATE_BLOCK
    num_reqs_padded = 4
    num_active = int(torch.randint(1, num_reqs_padded + 1, (1,), generator=g))
    rows = num_reqs_padded * (K + 1)
    if kind == "grid":
        # Integers / 4 in a narrow range: ties within and across blocks.
        logits = torch.randint(-6, 6, (rows, vocab), generator=g).float() / 4
    elif kind == "block_ties":
        # Every block max equal: block picks resolve by the topk tie rule,
        # greedy argmax by the first block / first index.
        logits = torch.randint(-40, 0, (rows, vocab), generator=g).float() / 8
        lane = torch.randint(0, CANDIDATE_BLOCK, (rows, num_vocab_blocks), generator=g)
        cols = torch.arange(num_vocab_blocks).unsqueeze(0) * CANDIDATE_BLOCK + lane
        logits.scatter_(1, cols, 1.0)
    elif kind == "peaked":
        logits = torch.randn((rows, vocab), generator=g) * 2
        hot = torch.randint(0, vocab, (rows, 3), generator=g)
        logits.scatter_(1, hot, 12.0 + torch.rand((rows, 3), generator=g) * 2)
    else:
        raise ValueError(kind)
    logits = logits.to(torch.bfloat16)

    temps = torch.ones(num_reqs_padded, 1, dtype=torch.bfloat16)
    top_k = torch.zeros(num_reqs_padded, 1, dtype=torch.int32)
    top_p = torch.ones(num_reqs_padded, 1, dtype=torch.float32)
    temp_choices = torch.tensor([0.0, 0.6, 1.0, 1.3])
    p_choices = torch.tensor([1.0, 0.9, 0.5])
    for r in range(num_active):
        temps[r, 0] = temp_choices[int(torch.randint(0, 4, (1,), generator=g))]
        top_k[r, 0] = int(torch.randint(1, num_blocks // 2 + 1, (1,), generator=g))
        top_p[r, 0] = p_choices[int(torch.randint(0, 3, (1,), generator=g))]

    bonus_idx = torch.arange(num_reqs_padded, dtype=torch.int32) * (K + 1) + K
    target_idx = (
        (torch.arange(num_reqs_padded).unsqueeze(1) * (K + 1) + torch.arange(K))
        .reshape(-1)
        .to(torch.int32)
    )
    lengths = torch.zeros(num_reqs_padded, dtype=torch.int32)
    lengths[:num_active] = torch.randint(0, K + 1, (num_active,), generator=g).to(
        torch.int32
    )
    segment_ids = torch.arange(num_reqs_padded).repeat_interleave(K)
    group_indices = torch.arange(K).repeat(num_reqs_padded).to(torch.int32)
    # Drafts: the target argmax (accepts), a random token, or a placeholder.
    target_rows = logits.index_select(0, target_idx.long())
    argmax = torch.argmax(target_rows, dim=-1)
    rnd = torch.randint(0, vocab, argmax.shape, generator=g)
    pick = torch.randint(0, 5, argmax.shape, generator=g)
    draft = torch.where(pick < 3, argmax, torch.where(pick < 4, rnd, -1))
    draft = draft.to(torch.int32)
    accept_shape = tuple(draft.shape)
    u = torch.rand(
        _verify_uniforms_count(num_reqs_padded, vocab, accept_shape, len(target_idx)),
        generator=g,
    )
    return dict(
        logits=logits,
        bonus_idx=bonus_idx,
        target_idx=target_idx,
        temps=temps,
        top_k=top_k,
        top_p=top_p,
        u=u,
        draft=draft,
        lengths=lengths,
        segment_ids=segment_ids,
        group_indices=group_indices,
        num_active=num_active,
    )


RATES = torch.tensor([0.9, 0.5, 0.2])


def reference(c, num_blocks, synthetic, one_draw, num_reqs):
    common = (
        c["draft"],
        c["lengths"],
        c["segment_ids"],
        c["group_indices"],
        RATES if synthetic else None,
        K,
        synthetic,
        num_blocks,
        True,
        num_reqs,
    )
    head = (
        c["logits"],
        c["bonus_idx"],
        c["target_idx"],
        c["temps"],
        c["top_k"],
        c["top_p"],
    )
    if one_draw:
        return _verify_fused_one_draw_impl(*head, c["u"], *common)
    vocab = c["logits"].shape[-1]
    split = _split_verify_uniforms(
        c["u"],
        len(c["bonus_idx"]),
        vocab,
        tuple(c["draft"].shape),
        len(c["target_idx"]),
    )
    return _verify_fused_impl(*head, *split, *common)


def shard(c, tp, num_blocks, synthetic, one_draw, num_reqs):
    vocab = c["logits"].shape[-1]
    head = (c["bonus_idx"], c["target_idx"], c["temps"], c["top_k"], c["top_p"])
    common = (
        c["draft"],
        c["lengths"],
        c["segment_ids"],
        c["group_indices"],
        RATES if synthetic else None,
        K,
        synthetic,
        num_blocks,
    )
    if one_draw:
        fn = _verify_shard_one_draw_impl
        uniforms = (c["u"],)
    else:
        fn = _verify_shard_impl
        uniforms = _split_verify_uniforms(
            c["u"],
            len(c["bonus_idx"]),
            vocab,
            tuple(c["draft"].shape),
            len(c["target_idx"]),
        )
    return run_ranks(fn, tp, c["logits"], (*head, *uniforms, *common), num_reqs)


def as_tuple(x):
    return x if isinstance(x, tuple) else (x,)


def assert_same(ref, outs):
    ref = as_tuple(ref)
    for out in outs:  # every rank holds the same (replicated) result
        out = as_tuple(out)
        assert len(out) == len(ref)
        for a, b in zip(out, ref):
            assert a.dtype == b.dtype and a.shape == b.shape
            assert torch.equal(a, b), (a, b)


# (tp, vocab blocks): tp8 x 20 blocks gives a 320-wide shard, so every
# other 128-block straddles two ranks (as 248,320 / 8 = 31,040 does).
GEOMETRIES = [(8, 20, 8), (4, 20, 8), (8, 40, 16), (2, 24, 16)]


@pytest.mark.parametrize("tp,vocab_blocks,num_blocks", GEOMETRIES)
@pytest.mark.parametrize("kind", ["grid", "block_ties", "peaked"])
def test_shard_verify_matches_fused_one_draw(tp, vocab_blocks, num_blocks, kind):
    for seed in range(6):
        c = make_case(seed, tp, vocab_blocks, num_blocks, kind)
        synthetic = seed % 3 == 2
        num_reqs = torch.tensor(c["num_active"]) if seed % 2 else None
        ref = reference(c, num_blocks, synthetic, True, num_reqs)
        outs = shard(c, tp, num_blocks, synthetic, True, num_reqs)
        assert_same(ref, outs)


@pytest.mark.parametrize("kind", ["grid", "peaked"])
def test_shard_verify_matches_fused_three_draws(kind):
    tp, vocab_blocks, num_blocks = 8, 20, 8
    for seed in range(4):
        c = make_case(100 + seed, tp, vocab_blocks, num_blocks, kind)
        num_reqs = torch.tensor(c["num_active"])
        ref = reference(c, num_blocks, False, False, num_reqs)
        outs = shard(c, tp, num_blocks, False, False, num_reqs)
        assert_same(ref, outs)


def test_cases_exercise_ties_and_mixed_rows():
    """The fixtures hit what the exactness rests on: greedy and sampled rows
    in one chunk, padded rows, block-max ties at the candidate boundary and
    row-max ties that straddle ranks."""
    seen_greedy = seen_sampled = seen_pad = boundary_tie = rank_tie = False
    for seed in range(6):
        c = make_case(seed, 8, 20, 8, "block_ties")
        t = c["temps"][: c["num_active"], 0]
        seen_greedy |= bool((t == 0).any())
        seen_sampled |= bool((t > 0).any())
        seen_pad |= c["num_active"] < 4
        bm = c["logits"].float().reshape(c["logits"].shape[0], 20, 128).amax(-1)
        top = bm.sort(dim=-1, descending=True).values
        boundary_tie |= bool((top[:, 7] == top[:, 8]).any())
        width = 2560 // 8
        rmax = c["logits"].float().amax(-1, keepdim=True)
        owners = (c["logits"].float() == rmax).nonzero()[:, 1] // width
        rank_tie |= len(set(owners.tolist())) > 1
    assert seen_greedy and seen_sampled and seen_pad and boundary_tie and rank_tie


def _mismatches(monkeypatch_target, replacement, monkeypatch, greedy_bonus=False):
    monkeypatch.setattr(shard_candidates, monkeypatch_target, replacement)
    bad = 0
    for seed in range(6):
        c = make_case(seed, 8, 20, 8, "block_ties")
        if greedy_bonus:
            # Greedy requests without drafts: each output is the bonus token,
            # the first argmax of a row whose max ties across ranks.
            c["temps"][: c["num_active"]] = 0
            c["lengths"].zero_()
        ref = reference(c, 8, False, True, None)
        outs = shard(c, 8, 8, False, True, None)
        bad += int(not torch.equal(ref, outs[0]))
    return bad


def test_mutation_last_rank_tie_rule_is_caught(monkeypatch):
    """Resolving the cross-rank argmax tie to the LAST rank holding the max
    (instead of the first, i.e. the lowest id) must change some bonus token."""

    def last_rank(values, start, tp_size, all_gather):
        idx = torch.argmax(values, dim=-1, keepdim=True)
        best = values.gather(1, idx).to(torch.float32)
        pair = torch.cat([best, (idx + start).to(torch.float32)], dim=-1)
        every = all_gather(pair, -1).view(values.shape[0], tp_size, 2)
        flipped = every.flip(1)
        rank = torch.argmax(flipped[:, :, 0], dim=-1, keepdim=True)
        return flipped[:, :, 1].gather(1, rank).view(-1).to(torch.int64)

    assert _mismatches("_first_argmax_over_ranks", last_rank, monkeypatch, True) > 0


def test_mutation_block_pick_tie_rule_is_caught(monkeypatch):
    """Picking candidate blocks with another tie rule (ties to the highest
    block id) must change some token: the block picks must be the reference's."""

    def high_ids_first(block_max, num_blocks):
        n = block_max.shape[-1]
        flipped = block_max.flip(-1)
        ids = torch.sort(flipped, dim=-1, descending=True, stable=True).indices
        return (n - 1 - ids[..., :num_blocks]).to(torch.int64)

    assert _mismatches("select_candidate_blocks", high_ids_first, monkeypatch) > 0


@pytest.mark.parametrize(
    "change,expected",
    [
        (None, True),
        ("top_k_max", True),
        ("unarmed", False),
        ("all_greedy", False),
        ("greedy_without_top_k", False),  # the candidates rule covers every row
        ("top_k_above_max", False),
        ("no_top_k", False),
        ("logprobs", False),
        ("structured", False),
    ],
)
def test_step_gate_matches_the_candidates_rule(change, expected):
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    temps = torch.tensor([0.7, 0.0, 0.7, 0.0])  # the last slot is unused
    top_k = torch.tensor([20, 20, 20, 0], dtype=torch.int32)
    if change == "top_k_max":
        top_k[2] = 32
    elif change == "greedy_without_top_k":
        top_k[1] = 0
    elif change == "top_k_above_max":
        top_k[2] = 33
    elif change == "no_top_k":
        top_k[2] = 248320
    runner = SimpleNamespace(
        _shard_candidates=None if change == "unarmed" else (0, 8),
        _topk_candidate_blocks=lambda: 64,
        input_batch=SimpleNamespace(
            all_greedy=change == "all_greedy",
            max_num_logprobs=5 if change == "logprobs" else None,
            num_reqs=3,
            temperature_cpu_tensor=temps,
            top_k_cpu_tensor=top_k,
        ),
    )
    scheduler_output = SimpleNamespace(
        has_structured_output_requests=change == "structured"
    )
    assert (
        TPUModelRunner._samples_shard_candidates(runner, scheduler_output) is expected
    )


def _arming_runner(monkeypatch, change):
    from vllm_torchtpu import envs
    from vllm_torchtpu.runner import tpu_runner

    width, tp, rank = 31040, 8, 3
    vocab = width * tp
    head = SimpleNamespace(
        tp_size=1 if change == "tp1" else tp,
        tp_rank=rank,
        num_embeddings_per_partition=width,
        org_vocab_size=vocab,
        num_embeddings=vocab + (128 if change == "added_vocab" else 0),
        bias=None,
        shard_indices=SimpleNamespace(
            org_vocab_start_index=rank * width + (64 if change == "offset" else 0),
            org_vocab_end_index=(rank + 1) * width,
        ),
    )
    processor = SimpleNamespace(
        logits_as_input=False,
        scale=0.5 if change == "scale" else 1.0,
        soft_cap=None,
        head_dtype=None,
    )
    owner = SimpleNamespace(lm_head=head, logits_processor=processor)
    monkeypatch.setattr(
        tpu_runner.greedy_top_tokens, "resolve_head_owner", lambda m, dp: owner
    )
    for name, value in {
        "VLLM_TPU_VERIFY_FUSED": change != "no_fused",
        "VLLM_TPU_SAMPLER_CANDIDATES_ONLY": change != "no_candidates_only",
    }.items():
        monkeypatch.setattr(envs, name, value, raising=False)
    return SimpleNamespace(
        _candidate_bounds=torch.tensor([0, 1])
        if change == "candidate_sampling"
        else None,
        _candidate_head=None,
        _shard_candidates=None,
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
        lora_config=None,
        model=None,
        vocab_size=vocab,
        _hidden_states_dtype=torch.bfloat16,
        _topk_candidate_blocks=lambda: 0 if change == "no_topk" else 64,
        _dp_lockstep_enabled=lambda: False,
    )


@pytest.mark.parametrize(
    "change",
    [
        None,
        "tp1",
        "offset",
        "added_vocab",
        "scale",
        "no_fused",
        "no_candidates_only",
        "no_topk",
        "candidate_sampling",
    ],
)
def test_arming_preconditions(monkeypatch, change):
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    runner = _arming_runner(monkeypatch, change)
    TPUModelRunner._init_shard_candidates(runner)
    if change is None:
        assert runner._shard_candidates == (3, 8)
        assert runner._candidate_head is not None
    else:
        assert runner._shard_candidates is None
        assert runner._candidate_head is None


@pytest.mark.parametrize("one_draw", [True, False])
def test_runner_dispatch_passes_the_fused_arguments(monkeypatch, one_draw):
    """`_sample_spec_verify_fused(shard=True)` hands the shard program the
    fused program's operands (same draws at the full vocab width, TP layout in
    place of candidates_only); running it on every rank gives the fused
    tokens."""
    from vllm_torchtpu import envs
    from vllm_torchtpu.runner import tpu_runner
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    tp, vocab_blocks, num_blocks = 8, 20, 8
    c = make_case(7, tp, vocab_blocks, num_blocks, "grid")
    vocab = c["logits"].shape[-1]
    calls = {}

    def recorder(name):
        def fn(*args):
            calls[name] = args
            return torch.zeros(1)

        return fn

    for name in (
        "_verify_fused_one_draw_compiled",
        "_verify_fused_compiled",
        "_verify_shard_one_draw_compiled",
        "_verify_shard_compiled",
    ):
        monkeypatch.setattr(tpu_runner, name, recorder(name))
    for name, value in {
        "VLLM_TPU_VERIFY_ONE_DRAW": one_draw,
        "VLLM_TPU_SAMPLER_CANDIDATES_ONLY": True,
        "VLLM_TPU_VERIFY_FUSED_READ_OFFSETS": False,
        "VLLM_TPU_VERIFY_RO_INGRAPH": False,
    }.items():
        monkeypatch.setattr(envs, name, value, raising=False)
    rank = 5
    runner = SimpleNamespace(
        _padded_sampling_params=lambda *a, **k: (c["temps"], c["top_k"], c["top_p"]),
        _sampler_topk_candidates=lambda *a: num_blocks,
        _sampler_candidates_only=TPUModelRunner._sampler_candidates_only,
        rejection_sampler=SimpleNamespace(
            synthetic_conditional_rates=None, synthetic_mode=False
        ),
        speculative_config=SimpleNamespace(num_speculative_tokens=K),
        vocab_size=vocab,
        _shard_candidates=(rank, tp),
        _fused_read_offsets=(None, c["num_active"]),
        mamba_slot_read_offsets=None,
        _verify_fused_checks_left=0,
        _verify_one_draw_logged=True,
    )
    md = SimpleNamespace(
        draft_token_ids=c["draft"],
        draft_lengths=c["lengths"],
        bonus_logits_indices=c["bonus_idx"],
        target_logits_indices=c["target_idx"],
        segment_ids=c["segment_ids"],
        group_indices=c["group_indices"],
    )
    width = vocab // tp
    local = c["logits"][:, rank * width : (rank + 1) * width].contiguous()
    TPUModelRunner._sample_spec_verify_fused(
        runner, c["logits"], md, 0, c["num_active"], torch.Generator().manual_seed(3)
    )
    TPUModelRunner._sample_spec_verify_fused(
        runner,
        local,
        md,
        0,
        c["num_active"],
        torch.Generator().manual_seed(3),
        shard=True,
    )
    if one_draw:
        fused = calls["_verify_fused_one_draw_compiled"]
        sharded = calls["_verify_shard_one_draw_compiled"]
        fused_impl, shard_impl = (
            _verify_fused_one_draw_impl,
            _verify_shard_one_draw_impl,
        )
    else:
        fused = calls["_verify_fused_compiled"]
        sharded = calls["_verify_shard_compiled"]
        fused_impl, shard_impl = _verify_fused_impl, _verify_shard_impl
    assert sharded[0] is local
    assert sharded[-2:] == (rank, tp)
    ref = fused_impl(*fused)
    outs = run_ranks(shard_impl, tp, c["logits"], sharded[1:-2], None)
    assert_same(ref, outs)
