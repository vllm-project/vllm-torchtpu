# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_VERIFY_RO_INGRAPH, alone and with VLLM_TPU_VERIFY_FUSED_READ_OFFSETS.

The in-graph read-offset scatter must give the same tokens, accepted counts
and read offsets as the path it replaces (the eager fold, or the separate
scatter program), bit for bit, across verify buckets, widths and bucket
switches. Its warmup must compile every key serving reaches: the slot-group
layout comes from the real spec-decode warmup, and every verify bucket is
compiled at every width. A step at a key the warmup did not compile keeps
the other path (one WARNING), never compiling mid-serve.

Everything runs on CPU: the compiled entry points are swapped for their
eager impls, and `_on_tpu` is forced true.
"""

import contextlib
import logging
import types

import pytest
import torch

import vllm_torchtpu.runner.tpu_runner as R
from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter.sample.top_k_top_p import CANDIDATE_BLOCK
from vllm_torchtpu.runner import compile_watch

K = 3
BLOCKS = 4
VOCAB = BLOCKS * CANDIDATE_BLOCK
NUM_BLOCKS = 96
TOKENS_PADDINGS = [4, 8, 16, 32, 64]
REQS_PADDINGS = [4, 8, 16]
MAX_REQS = 16


def _sig(args):
    """Full signature of a compiled call: every tensor's shape and dtype,
    lists of tensors with their aliasing, static values as is."""
    out = []
    for a in args:
        if isinstance(a, torch.Tensor):
            out.append((tuple(a.shape), a.dtype))
        elif isinstance(a, list):
            first: dict[int, int] = {}
            out.append(
                (
                    tuple((tuple(t.shape), t.dtype) for t in a),
                    tuple(first.setdefault(id(t), i) for i, t in enumerate(a)),
                )
            )
        else:
            out.append(a)
    return tuple(out)


class Calls:
    def __init__(self):
        self.phase = "warm"
        self.ro = {"warm": [], "serve": []}
        self.plain = {"warm": [], "serve": []}


@pytest.fixture
def calls(monkeypatch):
    c = Calls()

    def ro(*a):
        c.ro[c.phase].append(_sig(a))
        return R._verify_one_draw_ro_impl(*a)

    def ro_lb(*a):
        c.ro[c.phase].append(_sig(a))
        return R._verify_one_draw_ro_lb_impl(*a)

    def plain(*a):
        c.plain[c.phase].append(_sig(a))
        return R._verify_fused_one_draw_impl(*a)

    monkeypatch.setattr(R, "_verify_one_draw_ro_compiled", ro)
    monkeypatch.setattr(R, "_verify_one_draw_ro_lb_compiled", ro_lb)
    monkeypatch.setattr(R, "_verify_fused_one_draw_compiled", plain)
    monkeypatch.setattr(R, "synchronize_tensors", lambda *a, **k: None)
    monkeypatch.setattr(R, "_on_tpu", lambda t: True)
    return c


def _set_flags(monkeypatch, ro_ingraph, fold_ro, lb):
    monkeypatch.setattr(envs, "VLLM_TPU_VERIFY_ONE_DRAW", True, raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_VERIFY_FUSED", True, raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_VERIFY_RO_INGRAPH", ro_ingraph, raising=False)
    monkeypatch.setattr(
        envs, "VLLM_TPU_VERIFY_FUSED_READ_OFFSETS", fold_ro, raising=False
    )
    monkeypatch.setattr(envs, "VLLM_TPU_FUSED_READ_OFFSET_UPDATE", True, raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_SAMPLER_CANDIDATES_ONLY", True, raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_SAMPLER_LOGITS_BLOCKS", lb, raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_SAMPLER_TOPP_NARROW", False, raising=False)


def _fake_runner(read_offsets, num_cand):
    f = types.SimpleNamespace()
    f.rejection_sampler = types.SimpleNamespace(
        synthetic_conditional_rates=None, synthetic_mode=False
    )
    f.speculative_config = types.SimpleNamespace(num_speculative_tokens=K)
    f.vocab_size = VOCAB
    f.device = "cpu"
    f.mamba_slot_read_offsets = read_offsets
    f.num_reqs_max_model_len = MAX_REQS
    f.num_reqs_most_model_len = 8
    f.max_num_reqs = MAX_REQS
    f.num_tokens_paddings = TOKENS_PADDINGS
    f.num_reqs_paddings = REQS_PADDINGS
    f._verify_fused_checks_left = 0
    f._verify_one_draw_stream_checked = True
    f._verify_one_draw_logged = True
    f._fused_read_offsets_applied = False
    f._topk_candidate_blocks = lambda: BLOCKS
    f._num_reqs_tensor = lambda n: torch.tensor(n)
    f._sampler_candidates_only = R.TPUModelRunner._sampler_candidates_only
    f._sampler_logits_blocks = R.TPUModelRunner._sampler_logits_blocks
    f._sampler_topk_candidates = lambda s, e, g: num_cand
    f._precompile_timed = lambda name: contextlib.nullcontext()
    for name in (
        "_ro_ingraph_warmed_for",
        "_precompile_verify_fused",
        "_precompile_verify_ro_ingraph",
        "_sample_spec_verify_fused",
        "_spec_verify_bucket_reachable",
    ):
        setattr(f, name, types.MethodType(getattr(R.TPUModelRunner, name), f))
    return f


def _warm_verify_buckets(f):
    """`_precompile_rejection_sampler`'s bucket loop, verify part only."""
    for num_tokens in TOKENS_PADDINGS:
        t32 = torch.zeros(num_tokens, dtype=torch.int32)
        for num_reqs in REQS_PADDINGS:
            if not f._spec_verify_bucket_reachable(num_tokens, num_reqs):
                continue
            f._precompile_verify_fused(
                torch.zeros(num_tokens, VOCAB, dtype=torch.bfloat16),
                torch.zeros(num_reqs, dtype=torch.int32),
                t32,
                t32.clone(),
                torch.zeros(num_reqs, dtype=torch.int32),
                torch.zeros(num_tokens, dtype=torch.int64),
                t32.clone(),
                None,
                None,
            )
            if num_reqs >= min(num_tokens, MAX_REQS):
                break


def _step(seed, n, R_pad, T):
    """One verify chunk: n real requests padded to R_pad, logits length T."""
    g = torch.Generator().manual_seed(seed)
    d = torch.randint(0, K + 1, (n,), generator=g)
    while int((d + 1).sum()) > T:
        d[int(d.argmax())] -= 1
    logits = (torch.randn(T, VOCAB, generator=g) * 4).to(torch.bfloat16)
    target = torch.zeros(T, dtype=torch.int32)
    seg = torch.zeros(T, dtype=torch.int64)
    grp = torch.zeros(T, dtype=torch.int32)
    draft = torch.full((T,), -1, dtype=torch.int32)
    bonus = torch.zeros(R_pad, dtype=torch.int32)
    lengths = torch.zeros(R_pad, dtype=torch.int32)
    row = pos = 0
    for r in range(n):
        for j in range(int(d[r])):
            target[pos], seg[pos], grp[pos] = row, r, j
            draft[pos] = (
                int(logits[row].float().argmax())
                if torch.rand(1, generator=g).item() < 0.6
                else int(torch.randint(0, VOCAB, (1,), generator=g))
            )
            pos += 1
            row += 1
        bonus[r] = row
        row += 1
    lengths[:n] = d.to(torch.int32)
    temps = torch.ones(R_pad, 1, dtype=torch.bfloat16)
    top_k = torch.zeros(R_pad, 1, dtype=torch.int32)
    top_p = torch.ones(R_pad, 1, dtype=torch.float32)
    temps[:n, 0] = torch.tensor(
        [[0.6, 1.0, 1.3][int(x)] for x in torch.randint(0, 3, (n,), generator=g)]
    ).to(torch.bfloat16)
    top_k[:n, 0] = torch.randint(1, BLOCKS // 2 + 1, (n,), generator=g).to(
        torch.int32
    )
    md = types.SimpleNamespace(
        bonus_logits_indices=bonus,
        target_logits_indices=target,
        draft_token_ids=draft,
        draft_lengths=lengths,
        segment_ids=seg,
        group_indices=grp,
    )
    return logits, md, (temps, top_k, top_p)


def _groups(layout, n, width, slot_base):
    """Slot-group tensors as the metadata build hands them to the verify:
    'per_group' = 3 x [width]; 'stacked' = [2, width] + [1, width]. Real
    requests get distinct blocks, padded rows the null block 0."""

    def row(k):
        t = torch.zeros(width, dtype=torch.int32)
        t[:n] = slot_base + k * MAX_REQS + torch.arange(n, dtype=torch.int32) + 1
        return t

    if layout == "per_group":
        return [row(0), row(1), row(2)]
    return [torch.stack([row(0), row(1)]), row(2).unsqueeze(0)]


def _run(f, logits, md, params, groups, n, seed):
    f._padded_sampling_params = lambda s, e, p, dt, dev: params
    f._fused_read_offsets = (groups, n)
    f._fused_read_offsets_applied = False
    out = f._sample_spec_verify_fused(
        logits, md, 0, n, torch.Generator().manual_seed(1000 + seed)
    )
    if not f._fused_read_offsets_applied:
        # The caller's separate post-verify scatter.
        R._update_read_offsets_impl(
            f.mamba_slot_read_offsets, groups, out, torch.tensor(n)
        )
    return out


# (n real requests, padded reqs, logits length, width): bucket switches
# up and down, both widths, rows above and below the width.
SCHEDULE = [
    (3, 4, 16, 8),
    (7, 8, 32, 8),
    (12, 16, 64, 16),
    (2, 4, 8, 16),
    (16, 16, 64, 16),
    (5, 8, 16, 8),
    (1, 4, 4, 8),
    (8, 8, 32, 16),
    (13, 16, 32, 16),
    (4, 4, 16, 16),
]


def _arm(monkeypatch, calls, ro_ingraph, fold_ro, lb, num_cand, layout):
    _set_flags(monkeypatch, ro_ingraph, fold_ro, lb)
    calls.phase = "warm"
    f = _fake_runner(torch.zeros(NUM_BLOCKS * 2, dtype=torch.int32), num_cand)
    _warm_verify_buckets(f)
    # `_warmup_spec_decode`: one real verify at one width records the layout.
    logits, md, params = _step(99, 1, 4, 4)
    _run(f, logits, md, params, _groups(layout, 1, 8, 0), 1, 99)
    f._precompile_verify_ro_ingraph()
    f.mamba_slot_read_offsets.zero_()
    calls.phase = "serve"
    outs = []
    for i, (n, r_pad, t, w) in enumerate(SCHEDULE):
        logits, md, params = _step(i, n, r_pad, t)
        out = _run(f, logits, md, params, _groups(layout, n, w, 4 * i), n, i)
        outs.append((out.clone(), f.mamba_slot_read_offsets.clone()))
    return f, outs


@pytest.mark.parametrize("layout", ["per_group", "stacked"])
@pytest.mark.parametrize("lb,num_cand", [(0, 0), (0, BLOCKS), (2, BLOCKS)])
@pytest.mark.parametrize("fold_ro", [True, False])
def test_on_equals_off_bitwise_and_every_step_warm(
    monkeypatch, calls, layout, lb, num_cand, fold_ro
):
    _, off = _arm(monkeypatch, calls, False, fold_ro, lb, num_cand, layout)
    assert not calls.ro["warm"] and not calls.ro["serve"]
    calls.ro = {"warm": [], "serve": []}
    f, on = _arm(monkeypatch, calls, True, fold_ro, lb, num_cand, layout)
    for (t_off, ro_off), (t_on, ro_on) in zip(off, on):
        assert torch.equal(t_on, t_off)
        assert torch.equal(
            (t_on != R.INVALID_TOKEN_ID).sum(1), (t_off != R.INVALID_TOKEN_ID).sum(1)
        )
        assert torch.equal(ro_on, ro_off)
    # Read offsets actually moved (the comparison is not of two zero buffers).
    assert int(on[-1][1].count_nonzero()) > 0
    # Every served step took the in-graph path, at a signature the warmup
    # compiled exactly (all shapes, dtypes, statics and group aliasing).
    assert len(calls.ro["serve"]) == len(SCHEDULE)
    assert set(calls.ro["serve"]) <= set(calls.ro["warm"])
    assert not f.__dict__.get("_ro_ingraph_missed")


def test_unwarmed_key_keeps_fold_path(monkeypatch, calls, caplog):
    _set_flags(monkeypatch, True, True, 0)
    f = _fake_runner(torch.zeros(NUM_BLOCKS * 2, dtype=torch.int32), 0)
    f._ro_ingraph_warmed = set()  # warmup ran and compiled nothing
    calls.phase = "serve"
    logits, md, params = _step(3, 3, 4, 16)
    ref = f.mamba_slot_read_offsets.clone()
    groups = _groups("per_group", 3, 8, 0)
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            out = _run(f, logits, md, params, groups, 3, 3)
    assert not calls.ro["serve"] and len(calls.plain["serve"]) == 3
    assert f._fused_read_offsets_applied
    R._update_read_offsets_impl(ref, groups, out, torch.tensor(3))
    assert torch.equal(ref, f.mamba_slot_read_offsets)
    warns = [r for r in caplog.records if "was not warmed" in r.getMessage()]
    assert len(warns) == 1


def test_no_layout_seen_leaves_ingraph_off(monkeypatch, calls, caplog):
    _set_flags(monkeypatch, True, True, 0)
    f = _fake_runner(torch.zeros(NUM_BLOCKS * 2, dtype=torch.int32), 0)
    _warm_verify_buckets(f)
    with caplog.at_level(logging.WARNING):
        f._precompile_verify_ro_ingraph()
    assert f._ro_ingraph_warmed == set() and not calls.ro["warm"]
    assert any("stays off" in r.getMessage() for r in caplog.records)


def test_flag_off_touches_nothing(monkeypatch, calls):
    _set_flags(monkeypatch, False, True, 0)
    f = _fake_runner(torch.zeros(NUM_BLOCKS * 2, dtype=torch.int32), 0)
    f._ro_ingraph_warmed_for = lambda *a: pytest.fail("RO key check with flag off")
    _warm_verify_buckets(f)
    f._precompile_verify_ro_ingraph()
    logits, md, params = _step(5, 2, 4, 8)
    _run(f, logits, md, params, _groups("per_group", 2, 8, 0), 2, 5)
    assert "_ro_ingraph_buckets" not in f.__dict__
    assert "_ro_ingraph_warmed" not in f.__dict__
    assert not calls.ro["warm"] and not calls.ro["serve"]


def test_warmup_compiles_every_width_and_bucket(monkeypatch, calls):
    """Count: verify buckets x top-k variants x widths (one layout)."""
    _set_flags(monkeypatch, True, True, 2)
    f = _fake_runner(torch.zeros(NUM_BLOCKS * 2, dtype=torch.int32), BLOCKS)
    _warm_verify_buckets(f)
    buckets = len(set(f._ro_ingraph_buckets))
    logits, md, params = _step(99, 1, 4, 4)
    _run(f, logits, md, params, _groups("stacked", 1, 8, 0), 1, 99)
    f._precompile_verify_ro_ingraph()
    assert len(calls.ro["warm"]) == len(f._ro_ingraph_warmed) == buckets * 2 * 2
    widths = {s[1][0][0][0][-1] for s in calls.ro["warm"]}
    assert widths == {8, 16}


def test_structure_records_aliasing():
    a = torch.zeros(4, dtype=torch.int32)
    b = torch.zeros(2, 4, dtype=torch.int32)
    assert R._ro_ingraph_structure([a, b, a]) == (
        torch.int32,
        ((), (2,), ()),
        (0, 1, 0),
    )


def test_compile_watch_names_list_args():
    assert compile_watch._arg_shape(torch.zeros(2, 3)) == "(2, 3):float32"
    groups = [torch.zeros(16, dtype=torch.int32)] * 37
    assert compile_watch._arg_shape(groups) == "[37x(16,)..(16,):int32]"
    assert compile_watch._arg_shape([1, 2]) is None
    assert compile_watch._arg_shape(3) is None
