# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""q38 DEC-STEP-3: VLLM_TPU_GLUE_FOLD (tpu_runner._glue_*), the async spec
step's integer glue as four programs.

Bitwise: every folded output (read offsets, rejected counts, the draft source,
the substituted ids, the extracted drafts, seq_lens / positions) equals the
stock path's at 96 / 192 / 288 token rows, k2 and k3, both substitution fast
paths, one and two mamba groups. RNG: the fold draws nothing and the verify
draw code is the stock code (pinned source hash). Programs: each fold traces to
one graph with no break; the stock regions' eager op counts are printed
(Q38GLUE lines)."""

import functools
import hashlib
import inspect
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vllm_torchtpu import envs
from vllm_torchtpu.compilation import tpu_compiler
from vllm_torchtpu.runner import tpu_runner as tr
from vllm_torchtpu.runner import tpu_runner_async_output as tao
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

INV = tao.INVALID_TOKEN_ID
VOCAB = 151000
_SUBST_BODY = getattr(
    tr._substitute_placeholder_token, "_torchdynamo_orig_callable", None
) or tr._substitute_placeholder_token

# (token rows, k): seats = rows / (1 + k)
SHAPES = [(96, 2), (192, 2), (288, 2), (96, 3), (192, 3), (288, 3)]


class _Runner:
    """The runner methods the glue touches, on CPU."""

    _update_mamba_slot_read_offsets = TPUModelRunner._update_mamba_slot_read_offsets
    _num_reqs_tensor = TPUModelRunner._num_reqs_tensor
    _assemble_async_spec_substitution = TPUModelRunner._assemble_async_spec_substitution
    _drafts_per_chunk = TPUModelRunner.__dict__["_drafts_per_chunk"]
    _apply_async_token_substitution = TPUModelRunner._apply_async_token_substitution
    _warmed_next_tokens = TPUModelRunner._warmed_next_tokens
    _substitute_next_lens = TPUModelRunner._substitute_next_lens

    def __init__(self, k, seats, slots, groups_n, bucketed, seed):
        g = torch.Generator().manual_seed(seed)
        self.device = torch.device("cpu")
        self.scheduler_config = SimpleNamespace(async_scheduling=True)
        self.speculative_config = SimpleNamespace(num_speculative_tokens=k)
        self._is_async_drafter = True
        self.drafter = SimpleNamespace(bucketed_drafts=bucketed)
        self.max_num_reqs = seats + 8
        self.num_reqs_paddings = [8, 16, 32, 64, 96, 128]
        self._num_reqs_device = {}
        self._fast_token_substitution = True
        self.mamba_slot_read_offsets = torch.randint(0, k + 1, (slots,), generator=g).to(
            torch.int32
        )
        self.input_ids = torch.full((seats * (1 + k),), -7, dtype=torch.int32)
        self._pre_async_results = None
        self.groups_n = groups_n


def _fixture(rows, k, *, n_off, nt_dtype, dr_dtype, groups_n, bucketed, seed, pad_rows):
    """One spec step: verify output, drafts, metadata, mamba groups, and the next
    step's substitution / subtract inputs."""
    g = torch.Generator().manual_seed(seed)
    seats = rows // (1 + k)
    n = seats - n_off
    P = seats  # padded request rows of the chunk
    slots = 4 * P + 16
    r = _Runner(k, seats, slots, groups_n, bucketed, seed)
    dl = torch.zeros(P, dtype=torch.int32)
    dl_n = torch.randint(0, k + 1, (n,), generator=g).to(torch.int32)
    dl_n[: max(1, n // 2)] = k  # most requests carry k drafts
    dl[:n] = dl_n
    nt = torch.full((P, k + 1), INV, dtype=torch.int64)
    for i in range(P):
        if i < n:
            a = int(torch.randint(0, int(dl[i]) + 1, (1,), generator=g))
        elif pad_rows == "invalid":
            continue
        else:
            a = int(torch.randint(0, k + 1, (1,), generator=g))
        nt[i, : a + 1] = torch.randint(0, VOCAB, (a + 1,), generator=g)
    nt = nt.to(nt_dtype)
    perm = torch.randperm(slots - 1, generator=g)[: P * groups_n] + 1
    groups = []
    for j in range(groups_n):
        s = perm[j * P : (j + 1) * P].to(torch.int32).clone()
        s[n:] = 0  # padded positions -> the null slot
        groups.append(s)
    rows_d = P if bucketed else n
    drafts_t = torch.randint(0, VOCAB, (rows_d, k), generator=g).to(dr_dtype)
    drafts = [drafts_t] if bucketed else drafts_t
    md = SimpleNamespace(
        draft_lengths=dl,
        draft_lengths_cpu=dl_n.numpy().astype(np.int32),
    )
    # next step: the same requests, maybe reordered (fast path 2) or in place (fast path 1)
    N = seats * (1 + k)
    if n_off == 0:
        order = list(range(n))
    else:
        order = torch.randperm(n, generator=g).tolist()
    cur, pre = [], []
    for i, p in enumerate(order):
        for j in range(1 + k):
            cur.append(i * (1 + k) + j)
            pre.append(p * (1 + k) + j)
    cur_np = np.array(cur, dtype=np.int32)
    pre_np = np.array(pre, dtype=np.int32)
    T = n * k
    tli = torch.tensor(
        [i * (1 + k) + j for i in range(n) for j in range(k)], dtype=torch.int32
    )
    fli = torch.arange(N, dtype=torch.int32)
    pad_t = 16 - (T % 16) if T % 16 else 0
    tli = torch.cat([tli, torch.zeros(pad_t, dtype=torch.int32)])
    md2 = SimpleNamespace(
        final_logits_indices=fli, target_logits_indices=tli, draft_token_ids=None
    )
    seq_lens = torch.randint(1000, 90000, (P,), generator=g).to(torch.int32)
    positions = torch.randint(1000, 90000, (N,), generator=g).to(torch.int32)
    seq_idx = torch.full((P,), -1, dtype=torch.int32)
    pos_idx = torch.full((N,), -1, dtype=torch.int32)
    for i, p in enumerate(order):
        seq_idx[i] = p
        pos_idx[i * (1 + k) : (i + 1) * (1 + k)] = p
    return SimpleNamespace(
        r=r, k=k, n=n, P=P, N=N, nt=nt, drafts=drafts, md=md, groups=groups,
        cur_np=cur_np, pre_np=pre_np, md2=md2, seq_lens=seq_lens,
        positions=positions, seq_idx=seq_idx, pos_idx=pos_idx,
    )


def _step(fx, fold: bool):
    """sample_tokens(i)'s verify tail + assembly + pad, then execute_model(i+1)'s
    subtract + substitution + extraction, as the runner's call sites run them."""
    r = fx.r
    state = SimpleNamespace(
        num_reqs_list=[fx.n], spec_decode_metadata_list=[fx.md], logits_list=[None]
    )
    # --- sample_tokens(i): verify chunk tail
    mamba_indices = list(fx.groups)
    if envs.VLLM_TPU_GLUE_FOLD and tr._glue_post_verify(
        r, mamba_indices, fx.nt, fx.n, fx.md, state
    ):
        mamba_indices = None
    r._update_mamba_slot_read_offsets(mamba_indices, fx.nt, fx.n)
    # --- sample_tokens(i): async assembly + the bound pad
    chunks, idx, rej, ndraft = r._assemble_async_spec_substitution(
        fx.drafts, [fx.nt], state
    )
    ntt = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)
    bound = r.max_num_reqs * (1 + fx.k)
    if ntt.shape[0] < bound:
        ntt = torch.nn.functional.pad(ntt, (0, bound - ntt.shape[0]))
    r._pre_async_results = SimpleNamespace(
        next_tokens_tpu=ntt, spec_decode_num_rejected_tokens=rej
    )
    # --- execute_model(i+1): _prepare_inputs' subtract
    sub = tao.subtract_num_rejected_tokens
    if envs.VLLM_TPU_GLUE_FOLD:
        sub = functools.partial(tr._glue_subtract, r)
    seq_lens, positions = sub(
        fx.seq_lens.clone(), fx.positions.clone(), rej, fx.seq_idx, fx.pos_idx
    )
    # --- execute_model(i+1): substitution + extraction
    md2 = SimpleNamespace(**vars(fx.md2))
    if envs.VLLM_TPU_GLUE_FOLD:
        r._glue_pre_md = md2
    ids = r._apply_async_token_substitution(r.input_ids, fx.cur_np, fx.pre_np)
    if envs.VLLM_TPU_GLUE_FOLD:
        r.__dict__.pop("_glue_pre_md", None)
    if len(fx.cur_np) > 0 and not (
        envs.VLLM_TPU_GLUE_FOLD and r.__dict__.pop("_glue_pre_done", False)
    ):
        md2.draft_token_ids = tao.extract_draft_token_ids(
            ids, md2.final_logits_indices, md2.target_logits_indices
        )
    return dict(
        read_offsets=r.mamba_slot_read_offsets.clone(),
        next_tokens_tpu=ntt.clone(),
        rejected=rej.clone(),
        next_token_indices=dict(idx),
        num_draft_per_req=dict(ndraft),
        seq_lens=seq_lens.clone(),
        positions=positions.clone(),
        ids=ids.clone(),
        input_buffer=r.input_ids.clone(),
        draft_token_ids=md2.draft_token_ids.clone(),
    )


def _same(a, b):
    for key in a:
        x, y = a[key], b[key]
        if isinstance(x, torch.Tensor):
            assert x.dtype == y.dtype and x.shape == y.shape, (key, x.dtype, y.dtype, x.shape, y.shape)
            assert torch.equal(x, y), key
        else:
            assert x == y, key


def _run_pair(monkeypatch, fx_kwargs):
    monkeypatch.delenv("VLLM_TPU_GLUE_FOLD", raising=False)
    monkeypatch.setenv("VLLM_TPU_WARMED_SUBSTITUTE_SOURCE", "1")
    with patch.object(tr, "_substitute_placeholder_token", _SUBST_BODY):
        off = _step(_fixture(**fx_kwargs), fold=False)
        monkeypatch.setenv("VLLM_TPU_GLUE_FOLD", "1")
        fx = _fixture(**fx_kwargs)
        with patch.object(tao, "assemble_spec_next_tokens", wraps=tao.assemble_spec_next_tokens) as asm_spy, \
             patch.object(tao, "compute_num_rejected", wraps=tao.compute_num_rejected) as rej_spy, \
             patch.object(tr, "_substitute_placeholder_token", wraps=_SUBST_BODY) as sub_spy, \
             patch.object(tao, "extract_draft_token_ids", wraps=tao.extract_draft_token_ids) as ext_spy:
            on = _step(fx, fold=True)
        stats = fx.r._glue_stats
    return off, on, stats, (asm_spy.call_count, rej_spy.call_count, sub_spy.call_count, ext_spy.call_count)


@pytest.mark.parametrize("rows,k", SHAPES)
@pytest.mark.parametrize("n_off", [0, 3])
@pytest.mark.parametrize("groups_n", [1, 2])
def test_flag_on_matches_stock_bitwise(monkeypatch, rows, k, n_off, groups_n):
    kw = dict(
        rows=rows, k=k, n_off=n_off, nt_dtype=torch.int32, dr_dtype=torch.int32,
        groups_n=groups_n, bucketed=False, seed=rows * 10 + k + n_off + groups_n,
        pad_rows="random",
    )
    off, on, stats, spies = _run_pair(monkeypatch, kw)
    _same(off, on)
    # engaged: each fold ran once and the stock glue it replaces did not
    assert stats["post"] == 1 and stats["asm"] == 1 and stats["sub"] == 1
    assert stats["pre"] + stats["pre_id"] == 1
    assert (stats["pre_id"] == 1) == (n_off == 0)  # full batch in order -> fast path 1
    assert spies == (0, 0, 0, 0)


@pytest.mark.parametrize("nt_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("dr_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("bucketed", [False, True])
@pytest.mark.parametrize("pad_rows", ["random", "invalid"])
def test_dtypes_bucketing_and_padding_rows(monkeypatch, nt_dtype, dr_dtype, bucketed, pad_rows):
    kw = dict(
        rows=288, k=2, n_off=5, nt_dtype=nt_dtype, dr_dtype=dr_dtype, groups_n=2,
        bucketed=bucketed, seed=7, pad_rows=pad_rows,
    )
    off, on, stats, spies = _run_pair(monkeypatch, kw)
    _same(off, on)
    assert stats["post"] == stats["asm"] == stats["sub"] == stats["pre"] == 1


def test_warm_source_off_matches(monkeypatch):
    """WARMED_SUBSTITUTE_SOURCE off: the fold passes the source unchanged."""
    kw = dict(
        rows=192, k=3, n_off=4, nt_dtype=torch.int32, dr_dtype=torch.int32,
        groups_n=1, bucketed=False, seed=3, pad_rows="random",
    )
    monkeypatch.delenv("VLLM_TPU_GLUE_FOLD", raising=False)
    monkeypatch.setenv("VLLM_TPU_WARMED_SUBSTITUTE_SOURCE", "0")
    with patch.object(tr, "_substitute_placeholder_token", _SUBST_BODY):
        off = _step(_fixture(**kw), fold=False)
        monkeypatch.setenv("VLLM_TPU_GLUE_FOLD", "1")
        on = _step(_fixture(**kw), fold=True)
    _same(off, on)


@pytest.mark.parametrize("rows,k", SHAPES)
def test_post_fold_equals_tpu_body_and_cpu_fallback(rows, k):
    """The post fold's read offsets equal both stock variants: the compiled body
    (_update_read_offsets_impl, the TPU path) and the runner's eager fallback."""
    fx = _fixture(rows=rows, k=k, n_off=2, nt_dtype=torch.int32, dr_dtype=torch.int32,
                  groups_n=2, bucketed=False, seed=11 + rows, pad_rows="random")
    base = fx.r.mamba_slot_read_offsets.clone()
    a, b, c = base.clone(), base.clone(), base.clone()
    nr = torch.tensor(fx.n, dtype=torch.int64)
    dl = fx.md.draft_lengths[: fx.n]
    rej, bonus = tr._glue_post_verify_impl(a, list(fx.groups), fx.nt, nr, dl)
    tr._update_read_offsets_impl(b, list(fx.groups), fx.nt, nr)
    fx.r.mamba_slot_read_offsets = c  # CPU: the runner's eager fallback
    fx.r._update_mamba_slot_read_offsets(list(fx.groups), fx.nt, fx.n)
    assert torch.equal(a, b) and torch.equal(a, c)
    assert torch.equal(rej, tao.compute_num_rejected(fx.nt, dl)) and rej.dtype == torch.int32
    src = tao.assemble_spec_next_tokens(fx.nt, fx.drafts, fx.n).view(fx.n, k + 1)
    assert torch.equal(bonus.to(fx.drafts.dtype).view(-1), src[:, 0])


class _OpLog(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func.overloadpacket.__name__))
        return func(*args, **(kwargs or {}))


_RANDOM_OPS = {"uniform", "uniform_", "rand", "rand_like", "randn", "normal", "normal_",
               "bernoulli", "bernoulli_", "random_", "randint", "exponential_", "multinomial"}


def test_fold_draws_no_random_numbers(monkeypatch):
    """RNG order: the fold draws nothing (no random op; generator states
    unchanged), so a step's uniform draws and the next step's are the stock
    stream."""
    kw = dict(rows=288, k=2, n_off=3, nt_dtype=torch.int32, dr_dtype=torch.int32,
              groups_n=1, bucketed=False, seed=5, pad_rows="random")
    gen = torch.Generator().manual_seed(1234)
    draws = []

    def run(fold):
        fx = _fixture(**kw)
        gen.manual_seed(1234)
        cpu_state = torch.random.get_rng_state()
        d1 = [torch.rand((32, 32), generator=gen), torch.rand((32, 2), generator=gen),
              torch.rand((64, 32), generator=gen)]
        log = _OpLog()
        with log:
            out = _step(fx, fold=fold)
        d2 = [torch.rand((32, 32), generator=gen), torch.rand((32, 2), generator=gen)]
        assert torch.equal(torch.random.get_rng_state(), cpu_state)
        draws.append((d1, d2, gen.get_state().clone(), log.ops))
        return out

    monkeypatch.setenv("VLLM_TPU_WARMED_SUBSTITUTE_SOURCE", "1")
    with patch.object(tr, "_substitute_placeholder_token", _SUBST_BODY):
        monkeypatch.delenv("VLLM_TPU_GLUE_FOLD", raising=False)
        off = run(False)
        monkeypatch.setenv("VLLM_TPU_GLUE_FOLD", "1")
        on = run(True)
    _same(off, on)
    (a1, a2, sa, _), (b1, b2, sb, ops_on) = draws
    assert all(torch.equal(x, y) for x, y in zip(a1 + a2, b1 + b2))
    assert torch.equal(sa, sb)
    assert not (_RANDOM_OPS & set(ops_on)), sorted(_RANDOM_OPS & set(ops_on))


# The verify's uniform draws (bonus, accept, recover: shapes, dtype, order) are
# the stock code: these functions are byte-identical to the stack tree's.
_VERIFY_SOURCE_SHA = {
    "_sample_spec_verify_candidates": "c938085442c3c645",
    "_sample_spec_verify_fused": "49563fc60dc52c2c",
    "_sample_spec_verify_chunk": "083aae28630ea72e",
}


@pytest.mark.parametrize("name", sorted(_VERIFY_SOURCE_SHA))
def test_verify_draw_code_is_the_stock_code(name):
    src = inspect.getsource(getattr(TPUModelRunner, name))
    assert hashlib.sha256(src.encode()).hexdigest()[:16] == _VERIFY_SOURCE_SHA[name]
    assert "_glue" not in src


def test_key_off_touches_nothing(monkeypatch):
    """Key unset: no fold function runs and the runner gets no glue attribute."""
    monkeypatch.delenv("VLLM_TPU_GLUE_FOLD", raising=False)
    monkeypatch.setenv("VLLM_TPU_WARMED_SUBSTITUTE_SOURCE", "1")
    kw = dict(rows=96, k=2, n_off=2, nt_dtype=torch.int32, dr_dtype=torch.int32,
              groups_n=1, bucketed=False, seed=9, pad_rows="random")
    fx = _fixture(**kw)
    names = ["_glue_post_verify", "_glue_assemble", "_glue_subtract", "_glue_pre",
             "_glue_post_verify_impl", "_glue_assemble_impl", "_glue_pre_impl"]
    with patch.object(tr, "_substitute_placeholder_token", _SUBST_BODY):
        spies = [patch.object(tr, n, wraps=getattr(tr, n)) for n in names]
        mocks = [s.start() for s in spies]
        try:
            # the runner's real call sites, not the harness: _assemble_async_spec_substitution
            # and _apply_async_token_substitution carry inserted gates
            _step(fx, fold=False)
        finally:
            for s in spies:
                s.stop()
    assert all(m.call_count == 0 for m in mocks), [n for n, m in zip(names, mocks) if m.call_count]
    assert not [a for a in vars(fx.r) if a.startswith("_glue")]


def test_key_registered_default_off_and_compile_ignored(monkeypatch):
    """Registered, default off, and compile-ignored: the keys pick which
    programs run (each keyed by its own graph), so the compile hash and the
    env factors are the same with or without them (no cold cache for a flag-off
    or flag-on run of this tree)."""
    monkeypatch.delenv("VLLM_TPU_GLUE_FOLD", raising=False)
    monkeypatch.delenv("VLLM_TPU_GLUE_FOLD_CHECK", raising=False)
    assert envs.VLLM_TPU_GLUE_FOLD is False and envs.VLLM_TPU_GLUE_FOLD_CHECK == 0
    for key in ("VLLM_TPU_GLUE_FOLD", "VLLM_TPU_GLUE_FOLD_CHECK"):
        assert key in envs.environment_variables
        assert key in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    off = tpu_compiler._tpu_compile_env_factors()
    monkeypatch.setenv("VLLM_TPU_GLUE_FOLD", "1")
    monkeypatch.setenv("VLLM_TPU_GLUE_FOLD_CHECK", "100")
    assert envs.VLLM_TPU_GLUE_FOLD is True and envs.VLLM_TPU_GLUE_FOLD_CHECK == 100
    on = tpu_compiler._tpu_compile_env_factors()
    assert off == on and "VLLM_TPU_GLUE_FOLD" not in on


def test_engages_only_on_a_single_chunk_async_spec_step(monkeypatch):
    monkeypatch.setenv("VLLM_TPU_GLUE_FOLD", "1")
    fx = _fixture(rows=96, k=2, n_off=1, nt_dtype=torch.int32, dr_dtype=torch.int32,
                  groups_n=1, bucketed=False, seed=2, pad_rows="random")
    two = SimpleNamespace(num_reqs_list=[fx.n, 1], spec_decode_metadata_list=[fx.md, fx.md])
    assert not tr._glue_post_verify(fx.r, list(fx.groups), fx.nt, fx.n, fx.md, two)
    one = SimpleNamespace(num_reqs_list=[fx.n], spec_decode_metadata_list=[fx.md])
    assert not tr._glue_post_verify(fx.r, list(fx.groups), fx.nt, fx.n, None, one)
    fx.r.scheduler_config.async_scheduling = False
    assert not tr._glue_post_verify(fx.r, list(fx.groups), fx.nt, fx.n, fx.md, one)
    fx.r.scheduler_config.async_scheduling = True
    fx.r._is_async_drafter = False
    assert not tr._glue_post_verify(fx.r, list(fx.groups), fx.nt, fx.n, fx.md, one)
    # a stale parked result is never used for another chunk
    fx.r._is_async_drafter = True
    assert tr._glue_post_verify(fx.r, list(fx.groups), fx.nt, fx.n, fx.md, one)
    other = fx.nt.clone()
    assert tr._glue_assemble(fx.r, fx.drafts, [other], one) is None


def test_check_mode_reports_zero_and_catches_a_fault(monkeypatch):
    """VLLM_TPU_GLUE_FOLD_CHECK=N: one log line after N folded steps with 0
    mismatches; a perturbed fold (positive control) is counted."""
    monkeypatch.setenv("VLLM_TPU_GLUE_FOLD", "1")
    monkeypatch.setenv("VLLM_TPU_GLUE_FOLD_CHECK", "2")
    monkeypatch.setenv("VLLM_TPU_WARMED_SUBSTITUTE_SOURCE", "1")
    kw = dict(rows=192, k=2, n_off=3, nt_dtype=torch.int32, dr_dtype=torch.int32,
              groups_n=1, bucketed=False, seed=4, pad_rows="random")

    def lines(fault):
        fx = _fixture(**kw)
        impl = tr._glue_assemble_impl
        bad = (lambda b, d, n: impl(b, d, n) + 1) if fault else impl
        with patch.object(tr, "_substitute_placeholder_token", _SUBST_BODY), \
             patch.object(tr, "_glue_assemble_impl", bad), \
             patch.object(tr.logger, "info") as info:
            for _ in range(3):
                fx.r._pre_async_results = None
                _step(fx, fold=True)
        return [c.args[0] % c.args[1:] for c in info.call_args_list if "GLUE_FOLD check" in c.args[0]]

    clean = lines(False)
    assert len(clean) == 1 and "post=0/2" in clean[0] and "asm=0/1" in clean[0], clean
    faulted = lines(True)
    assert len(faulted) == 1 and "asm=0/" not in faulted[0], faulted


def test_compiled_folds_strip_dynamic_marks():
    """vLLM marks the forward's input_ids / positions dynamic on the tensor
    object and the folds take positions (sub) and input_ids (pre): every
    compiled fold is wrapped in _static_shape_args, which hands a stamped
    tensor to the TPU backend as an unstamped view."""
    from vllm_torchtpu.spec_decode.eagle3 import _static_shape_args

    code = _static_shape_args(lambda: None).__code__
    fns = {
        "post": tr._glue_post_verify_compiled,
        "asm": tr._glue_assemble_compiled,
        "pre": tr._glue_pre_compiled,
        "pre_id": tr._glue_pre_identity_compiled,
        "sub": tr._glue_subtract_fn(),
    }
    assert {k for k, f in fns.items() if getattr(f, "__code__", None) is not code} == set()
    seen = []
    w = _static_shape_args(lambda t: seen.append(getattr(t, "_dynamo_dynamic_indices", None)))
    t = torch.zeros(4)
    t._dynamo_dynamic_indices = {0}
    w(t)
    assert seen == [None]


def test_call_sites_in_source():
    """The runner's call sites are the ones _step mirrors."""
    st = inspect.getsource(TPUModelRunner.sample_tokens)
    a = st.index("if envs.VLLM_TPU_GLUE_FOLD and _glue_post_verify(")
    b = st.index("mamba_indices = None", a)
    c = st.index("self._update_mamba_slot_read_offsets(\n                        mamba_indices, next_tokens, num_reqs", b)
    assert a < b < c
    asm = inspect.getsource(TPUModelRunner._assemble_async_spec_substitution)
    assert asm.index("folded = _glue_assemble(self, drafts, next_tokens_per_chunk, state)") < asm.index(
        "drafts_per_chunk = self._drafts_per_chunk(drafts, state.num_reqs_list)"
    )
    prep = inspect.getsource(TPUModelRunner._prepare_inputs)
    assert prep.index("subtract_num_rejected_tokens = functools.partial(_glue_subtract, self)") < prep.index(
        "seq_lens, self.position_ids = subtract_num_rejected_tokens("
    )
    ex = inspect.getsource(TPUModelRunner.execute_model)
    i = ex.index("self._glue_pre_md = (")
    j = ex.index("input_ids = self._apply_async_token_substitution(", i)
    k = ex.index('self.__dict__.pop("_glue_pre_md", None)', j)
    m = ex.index('envs.VLLM_TPU_GLUE_FOLD and self.__dict__.pop("_glue_pre_done", False)', k)
    n = ex.index("spec_decode_metadata.draft_token_ids = extract_draft_token_ids(", m)
    assert i < j < k < m < n
    sub = inspect.getsource(TPUModelRunner._apply_async_token_substitution)
    assert sub.count("return _glue_pre(") == 2


# ---- program counts: each fold is one traced graph; the stock regions' eager ops


def _graphs(fn, *args):
    torch._dynamo.reset()
    seen = []

    def backend(gm, example_inputs):
        seen.append(sum(1 for nd in gm.graph.nodes if nd.op in ("call_function", "call_method")))
        return gm.forward

    out = torch.compile(fn, backend=backend, fullgraph=True, dynamic=False)(*args)
    return seen, out


def _eager_ops(fn, *args):
    log = _OpLog()
    with log:
        out = fn(*args)
    return [o for o in log.ops if o not in ("detach", "alias", "view", "_unsafe_view", "lift_fresh")], out


@pytest.mark.parametrize("rows,k", [(96, 2), (192, 2), (288, 2), (160, 3)])
def test_program_counts(rows, k, capsys):
    fx = _fixture(rows=rows, k=k, n_off=3, nt_dtype=torch.int32, dr_dtype=torch.int32,
                  groups_n=1, bucketed=False, seed=21, pad_rows="random")
    n, P = fx.n, fx.P
    nr = torch.tensor(n, dtype=torch.int64)
    dl = fx.md.draft_lengths[:n]
    bound = fx.r.max_num_reqs * (1 + k)
    # fold side: one graph each, no break (fullgraph), same outputs as the bodies
    g_post, (rej, bonus) = _graphs(tr._glue_post_verify_impl, fx.r.mamba_slot_read_offsets.clone(), list(fx.groups), fx.nt, nr, dl)
    g_asm, src = _graphs(tr._glue_assemble_impl, bonus, fx.drafts, bound)
    g_sub, _ = _graphs(tao.subtract_num_rejected_tokens, fx.seq_lens, fx.positions, rej, fx.seq_idx, fx.pos_idx)
    cur = torch.from_numpy(np.concatenate([fx.cur_np, np.setdiff1d(np.arange(fx.N), fx.cur_np)]).astype(np.int32))
    pre = torch.from_numpy(np.pad(fx.pre_np, (0, fx.N - len(fx.pre_np)), constant_values=-1).astype(np.int32))
    warm = bound if src.dtype != torch.int64 else 0
    g_pre, _ = _graphs(tr._glue_pre_impl, fx.r.input_ids.clone(), cur, pre, src, fx.md2.final_logits_indices, fx.md2.target_logits_indices, warm)
    assert len(g_post) == len(g_asm) == len(g_sub) == len(g_pre) == 1
    # stock side: compiled programs and the eager ops between them
    ops_asm, _ = _eager_ops(lambda: (
        torch.nn.functional.pad(tao.assemble_spec_next_tokens(fx.nt, fx.drafts, n), (0, bound - n * (1 + k))),
        torch.cat([tao.compute_num_rejected(fx.nt, dl)]),
    ))
    ops_sub, _ = _eager_ops(tao.subtract_num_rejected_tokens, fx.seq_lens, fx.positions, rej, fx.seq_idx, fx.pos_idx)
    runner = SimpleNamespace(num_reqs_paddings=fx.r.num_reqs_paddings, _is_async_drafter=True,
                             speculative_config=fx.r.speculative_config, max_num_reqs=fx.r.max_num_reqs)
    runner._substitute_next_lens = lambda: TPUModelRunner._substitute_next_lens(runner)
    ops_warm, _ = _eager_ops(TPUModelRunner._warmed_next_tokens, runner, src)
    ops_ext, _ = _eager_ops(tao.extract_draft_token_ids, fx.r.input_ids, fx.md2.final_logits_indices, fx.md2.target_logits_indices)
    stock_compiled = 2  # _update_read_offsets_compiled, _substitute_placeholder_token
    stock_eager = len(ops_asm) + len(ops_sub) + len(ops_warm) + len(ops_ext)
    stock_runs = 1 + 1 + (1 if ops_warm else 0) + 1  # eager runs between programs
    print(
        f"\nQ38GLUE PROGRAMS rows={n * (1 + k)}/{fx.N} k={k} reqs={n}: stock compiled {stock_compiled} + eager aten "
        f"{stock_eager} (asm {len(ops_asm)} sub {len(ops_sub)} warm {len(ops_warm)} extract {len(ops_ext)}) in "
        f"{stock_runs} eager runs -> program bounds [{stock_compiled + stock_runs}, {stock_compiled + stock_eager}]; "
        f"fold compiled 4 = graphs post {len(g_post)} asm {len(g_asm)} sub {len(g_sub)} pre {len(g_pre)}, "
        f"0 breaks, nodes {g_post[0]}/{g_asm[0]}/{g_sub[0]}/{g_pre[0]}, 0 eager aten"
    )
    assert stock_eager > 0
