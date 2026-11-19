# SPDX-License-Identifier: Apache-2.0
"""q38 L4 EAGER-FIX on CPU: E-a (VLLM_TPU_SPEC_TAIL_REJ_PAD), E-b
(VLLM_TPU_VPS_WARM_DRAWS), E-c (VLLM_TPU_SAMPLING_COPY_FULL) and the
eager-shape log (VLLM_TPU_EAGER_SHAPE_LOG).

The runner methods run unchanged on small fake runners (only the attributes
they read); the compiled TPU programs are replaced by their eager bodies or
recorders, since the "tpu" backend cannot run on CPU. What CPU cannot show:
torch_tpu's eager-program cache itself. The fused-path test shows the shape
*key* the runner uses to decide fused vs eager, which is what E-a changes.
Run with ``--noconftest``.
"""

import contextlib
import os
import subprocess
import sys
import types

import numpy as np
import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.runner import eager_shape_log
from vllm_torchtpu.runner import sampling_copy_full as scf
from vllm_torchtpu.runner import tpu_runner as r
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.runner.tpu_runner_async_output import (
    subtract_num_rejected_tokens,
    subtract_substitute_extract,
)

K = 3
INV = r.INVALID_TOKEN_ID


def _flag(monkeypatch, name, value):
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, str(value))


def bits(t):
    return (t.dtype, tuple(t.shape), t.contiguous().view(-1).view(torch.uint8).tolist())


def _tail_inputs(g, num_reqs, max_reqs):
    rows = r._get_padded_num_reqs_with_upper_limit(num_reqs, max_reqs)
    nt = torch.randint(0, 1000, (rows, K + 1), generator=g, dtype=torch.int32)
    for i in range(rows):
        nt[i, 1 + int(torch.randint(0, K + 1, (1,), generator=g)) :] = INV
    drafts = torch.randint(0, 1000, (num_reqs, K), generator=g, dtype=torch.int32)
    dl = torch.randint(0, K + 1, (rows,), generator=g, dtype=torch.int32)
    return nt, drafts, dl


# ---- E-a ------------------------------------------------------------------------


@pytest.mark.parametrize("max_reqs", [13, 32])
def test_spec_tail_rejected_padding_is_bitwise_on_live_rows(max_reqs):
    g = torch.Generator().manual_seed(1)
    bound = max_reqs * (1 + K)
    for num_reqs in range(1, max_reqs + 1):
        nt, drafts, dl = _tail_inputs(g, num_reqs, max_reqs)
        src0, rej0 = r._spec_tail(nt, drafts, dl, num_reqs, num_reqs, bound)
        src1, rej1 = r._spec_tail(nt, drafts, dl, num_reqs, num_reqs, bound, max_reqs)
        assert bits(src0) == bits(src1)
        assert rej0.shape == (num_reqs,) and rej1.shape == (max_reqs,)
        assert bits(rej1[:num_reqs]) == bits(rej0)
        assert torch.count_nonzero(rej1[num_reqs:]) == 0


def test_consumers_read_the_same_values_from_padded_rejected():
    g = torch.Generator().manual_seed(2)
    M = 16
    for prev in range(1, M + 1):
        for cur in (1, prev, M):
            rej = torch.randint(0, K + 1, (prev,), generator=g, dtype=torch.int32)
            pad = torch.nn.functional.pad(rej, (0, M - prev))
            T = 4 * cur
            seq_idx = torch.randint(-1, prev, (cur,), generator=g, dtype=torch.int32)
            pos_idx = torch.randint(-1, prev, (T,), generator=g, dtype=torch.int32)
            seq = torch.randint(10, 100, (cur,), generator=g, dtype=torch.int32)
            pos = torch.randint(10, 100, (T,), generator=g, dtype=torch.int64)
            a = subtract_num_rejected_tokens(seq, pos, rej, seq_idx, pos_idx)
            b = subtract_num_rejected_tokens(seq, pos, pad, seq_idx, pos_idx)
            assert all(bits(x) == bits(y) for x, y in zip(a, b))
            ids = torch.randint(0, 1000, (T,), generator=g, dtype=torch.int64)
            cur_idx = torch.randperm(T, generator=g).to(torch.int32)
            pre_idx = torch.where(torch.rand(T, generator=g) > 0.5,
                                  torch.randint(0, 4 * M, (T,), generator=g), -1).to(torch.int32)
            nxt = torch.randint(0, 1000, (4 * M,), generator=g, dtype=torch.int32)
            li = torch.arange(T, dtype=torch.int32)
            ti = torch.arange(max(T - 1, 1), dtype=torch.int32) % max(T - 1, 1)
            fa = subtract_substitute_extract(seq, pos, rej, seq_idx, pos_idx, ids, cur_idx,
                                             pre_idx, nxt, li, ti)
            fb = subtract_substitute_extract(seq, pos, pad, seq_idx, pos_idx, ids, cur_idx,
                                             pre_idx, nxt, li, ti)
            assert all(bits(x) == bits(y) for x, y in zip(fa, fb))


class _Fake:
    """A bare object the runner methods can use as `self`."""


def _postfwd_runner(M):
    f = _Fake()
    f.device = torch.device("cpu")
    f.max_num_reqs = M
    f._postfwd_subst_indices = types.MethodType(TPUModelRunner._postfwd_subst_indices, f)
    f._spec_tail_rej_extra = types.MethodType(TPUModelRunner._spec_tail_rej_extra, f)

    def subst_indices(start_index, num_reqs, nsched):
        cur = (np.arange(num_reqs, dtype=np.int32) * (1 + K))
        return cur, np.arange(num_reqs, dtype=np.int32) * (1 + K)

    f._prepare_async_token_substitution_indices = subst_indices
    return f


def _postfwd_step(f, g, cur, prev, M):
    """One `_postfwd_fused_prepare` call for `cur` live requests after a step
    with `prev`; the rejected count comes from the real `_spec_tail`."""
    bound = M * (1 + K)
    nt, drafts, dl = _tail_inputs(g, prev, M)
    src, rej = r._spec_tail(nt, drafts, dl, prev, prev, bound, *f._spec_tail_rej_extra())
    T = r._get_padded_num_reqs_with_upper_limit(cur, M) * (1 + K)
    f._pre_async_results = types.SimpleNamespace(next_tokens_tpu=src)
    f.input_ids = torch.randint(0, 1000, (T,), generator=g, dtype=torch.int64)
    f.position_ids = torch.randint(10, 100, (T,), generator=g, dtype=torch.int64)
    seq_lens = torch.randint(10, 100, (cur,), generator=g, dtype=torch.int32)
    seq_idx = torch.randint(-1, prev, (cur,), generator=g, dtype=torch.int32)
    pos_idx = torch.randint(-1, prev, (T,), generator=g, dtype=torch.int32)
    md = types.SimpleNamespace(
        final_logits_indices=torch.arange(cur * (1 + K), dtype=torch.int32),
        target_logits_indices=torch.arange(cur * K, dtype=torch.int32),
    )
    exp_seq, exp_pos = subtract_num_rejected_tokens(seq_lens, f.position_ids, rej, seq_idx, pos_idx)
    out = TPUModelRunner._postfwd_fused_prepare(
        f, seq_lens, (rej, seq_idx, pos_idx), 0, cur, np.full(cur, 1 + K), md
    )
    assert bits(out) == bits(exp_seq) and bits(f.position_ids) == bits(exp_pos)


@pytest.mark.parametrize("pad", [None, 1])
def test_served_steps_stay_on_the_fused_path_after_warmup(pad, monkeypatch):
    """E-a's effect on the fused-vs-eager decision: warm-up sweeps the live
    request count once; serving then sees every (current, previous) pair. With
    the padding no served step falls to the eager subtract path; without it
    the previous count is part of the key and served steps fall back."""
    _flag(monkeypatch, "VLLM_TPU_SPEC_TAIL_REJ_PAD", pad)
    monkeypatch.setattr(r, "_postfwd_subst_fused", subtract_substitute_extract)
    M = 8
    g = torch.Generator().manual_seed(3)
    f = _postfwd_runner(M)
    f.warmup_done = False
    for cur in range(1, M + 1):  # warm-up sweep (previous = current - 1 or 1)
        _postfwd_step(f, g, cur, max(cur - 1, 1), M)
    warm = len(f._postfwd_warm_keys)
    f.warmup_done = True
    f._postfwd_stats = [0, 0]
    for cur in range(1, M + 1):
        for prev in range(1, M + 1):
            _postfwd_step(f, g, cur, prev, M)
    fused, separate = f._postfwd_stats
    assert fused + separate == M * M
    if pad:
        assert separate == 0 and warm == M
    else:
        assert separate > 0


def _tail_runner(M, monkeypatch, calls):
    monkeypatch.setenv("VLLM_TPU_SPEC_TAIL_COMPILED", "1")
    f = _Fake()
    f.device = torch.device("cpu")
    f.max_num_reqs = M
    f.speculative_config = types.SimpleNamespace(num_speculative_tokens=K)
    f.scheduler_config = types.SimpleNamespace(async_scheduling=True)
    f.drafter = types.SimpleNamespace()
    f._drafts_per_chunk = TPUModelRunner._drafts_per_chunk
    f._spec_tail_rej_extra = types.MethodType(TPUModelRunner._spec_tail_rej_extra, f)
    f._precompile_timed = lambda name: contextlib.nullcontext()

    def recorder(*args):
        calls.append(tuple(a for a in args if not isinstance(a, torch.Tensor)))
        return r._spec_tail(*args)

    monkeypatch.setattr(r, "_spec_tail_compiled", recorder)
    monkeypatch.setattr(r, "synchronize_tensors", lambda *a, **k: None)
    return f


@pytest.mark.parametrize("pad", [None, 1])
def test_serving_tail_call_matches_a_precompiled_variant(pad, monkeypatch):
    """The served `_spec_tail` call passes exactly the int arguments one of
    `_precompile_spec_tail`'s calls passed, so serving never compiles a tail."""
    _flag(monkeypatch, "VLLM_TPU_SPEC_TAIL_REJ_PAD", pad)
    M = 13
    pre, served = [], []
    f = _tail_runner(M, monkeypatch, pre)
    TPUModelRunner._precompile_spec_tail(f)
    assert len(pre) == M
    monkeypatch.setattr(r, "_spec_tail_compiled",
                        lambda *a: served.append(tuple(x for x in a if not isinstance(x, torch.Tensor)))
                        or r._spec_tail(*a))
    g = torch.Generator().manual_seed(4)
    for num_reqs in range(1, M + 1):
        nt, drafts, dl = _tail_inputs(g, num_reqs, M)
        md = types.SimpleNamespace(draft_lengths=dl, draft_lengths_cpu=dl.numpy())
        state = types.SimpleNamespace(num_reqs_list=[num_reqs], spec_decode_metadata_list=[md])
        out = TPUModelRunner._assemble_async_spec_substitution(f, drafts, [nt], state)
        assert out[2].shape == ((M,) if pad else (num_reqs,))
    assert set(served) <= set(pre), (sorted(set(served) - set(pre)))
    assert all(len(c) == (4 if pad else 3) for c in pre)  # rows, num_reqs, bound(, M)
    assert all(c[-1] == M for c in pre) if pad else True


@pytest.mark.parametrize("pad", [None, 1])
def test_bootstrap_and_multichunk_rejected_shapes(pad, monkeypatch):
    _flag(monkeypatch, "VLLM_TPU_SPEC_TAIL_REJ_PAD", pad)
    M = 13
    f = _Fake()
    f.max_num_reqs = M
    f.drafter = types.SimpleNamespace()
    f._drafts_per_chunk = TPUModelRunner._drafts_per_chunk
    f._pad_rejected = types.MethodType(TPUModelRunner._pad_rejected, f)
    sel = [torch.arange(3, dtype=torch.int64).view(-1, 1), torch.arange(2, dtype=torch.int64).view(-1, 1)]
    drafts = torch.zeros((5, K), dtype=torch.int64)
    out = TPUModelRunner._assemble_async_prefill_bootstrap(f, drafts, sel, [3, 2])
    rej = out[2]
    assert rej.dtype == torch.int32 and torch.count_nonzero(rej) == 0
    assert rej.shape == ((M,) if pad else (5,))
    x = torch.tensor([1, 2, 3], dtype=torch.int32)
    y = f._pad_rejected(x)
    assert bits(y[:3]) == bits(x)
    assert y.shape == ((M,) if pad else (3,)) and torch.count_nonzero(y[3:]) == 0


# ---- E-c ------------------------------------------------------------------------


def _vllm_copy_slice(src, dst, n):
    from vllm.v1.utils import copy_slice

    return copy_slice(src, dst, n)


@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_full_copy_slice_rows_and_view_identical(dtype):
    g = torch.Generator().manual_seed(5)
    M = 16
    full = scf.make_copy_slice(_vllm_copy_slice, full=True)
    for n in range(0, M + 1):
        src = (torch.rand(M, generator=g) * 100).to(dtype)
        d0 = (torch.rand(M, generator=g) * 100).to(dtype)
        d1 = d0.clone()
        v0 = _vllm_copy_slice(src, d0, n)
        v1 = full(src, d1, n)
        assert bits(v0) == bits(v1)
        assert v1.shape == (n,) and v1.storage_offset() == 0
        assert v1.untyped_storage().data_ptr() == d1.untyped_storage().data_ptr()
        assert bits(d1[:n]) == bits(d0[:n])
        assert bits(d1) == bits(src)  # the whole buffer was written (no view write)
    # 2-D buffers (the allowed-token mask) keep the stock slice copy.
    s2, a2 = torch.rand(4, 8, generator=g), torch.zeros(4, 8)
    b2 = a2.clone()
    _vllm_copy_slice(s2, a2, 2)
    full(s2, b2, 2)
    assert bits(a2) == bits(b2) and torch.count_nonzero(b2[2:]) == 0


def test_input_batch_uses_the_patched_copy(monkeypatch):
    from vllm.v1.worker import gpu_input_batch

    orig = gpu_input_batch.copy_slice
    monkeypatch.setattr(gpu_input_batch, "copy_slice", orig)
    _flag(monkeypatch, "VLLM_TPU_SAMPLING_COPY_FULL", 1)
    assert scf.install()
    patched = gpu_input_batch.copy_slice
    assert patched is not orig and getattr(patched, scf._MARK)
    # The builder resolves copy_slice from the module globals at call time.
    mk = gpu_input_batch.InputBatch._make_sampling_metadata
    assert mk.__globals__ is vars(gpu_input_batch)
    assert "copy_slice" in mk.__code__.co_names
    assert scf.install() and gpu_input_batch.copy_slice is patched  # idempotent


def test_no_tpu_code_reads_the_sampling_metadata_device_tensors():
    """E-c's premise (C): nothing under src/vllm_torchtpu reads
    input_batch.sampling_metadata or the device temperature / top_p / top_k
    tensors (only their *_cpu twins)."""
    import re

    root = os.path.dirname(r.__file__.rsplit("/runner/", 1)[0] + "/")
    pat = re.compile(r"sampling_metadata|input_batch\.(temperature|top_p|top_k)\b(?!_cpu)")
    hits = []
    for dp, _dn, fns in os.walk(root):
        for fn in fns:
            if fn.endswith(".py") and fn != "sampling_copy_full.py":
                with open(os.path.join(dp, fn)) as fh:
                    for i, line in enumerate(fh, 1):
                        if pat.search(line) and not line.lstrip().startswith("#"):
                            hits.append(f"{fn}:{i}:{line.strip()}")
    assert hits == [], hits


def test_eager_shape_log_is_silent_in_warmup_and_logs_new_layouts_once(monkeypatch):
    lines = []
    monkeypatch.setattr(eager_shape_log.logger, "info", lambda fmt, *a: lines.append(fmt % a))
    eager_shape_log._reset_for_tests(limit=2)
    try:
        eager_shape_log.note("vps_draws", ((8, 64),))
        assert lines == []
        eager_shape_log.mark_serving()
        eager_shape_log.note("vps_draws", ((8, 64),))  # seen in warm-up
        eager_shape_log.note("vps_draws", ((16, 64),))
        eager_shape_log.note("vps_draws", ((16, 64),))
        eager_shape_log.note("copy_slice", ("torch.int32", (16,), 5))
        eager_shape_log.note("copy_slice", ("torch.int32", (16,), 6))  # over the limit
        assert len(lines) == 2
        assert "site=vps_draws key=((16, 64),) site_new=1" in lines[0]
        assert "site=copy_slice" in lines[1]
    finally:
        eager_shape_log._reset_for_tests(limit=0)


# ---- E-b ------------------------------------------------------------------------


def _vps_runner(monkeypatch, cand=True):
    f = _Fake()
    f.device = torch.device("cpu")
    vocab, tp = 512, 2
    f._vps_head = types.SimpleNamespace(full_vocab=vocab, tp_size=tp, shard_offset=0,
                                        shard_width=vocab // tp)
    f.rejection_sampler = types.SimpleNamespace(synthetic_conditional_rates=None,
                                                synthetic_mode=False)
    f.speculative_config = types.SimpleNamespace(num_speculative_tokens=K)
    f._vps_cand_draws = cand
    f._vps_fb_path = True
    f._vps_fb_max = 8
    f._vps_tiegap = 0
    f._vps_fb_stats = None
    f._vps_fb_chunks = 0
    f._vps_checks_left = 0
    f._upload_unless_unchanged = lambda name, t: t
    f._vps_verify_fb = types.MethodType(TPUModelRunner._vps_verify_fb, f)
    # q38 L4 T-1(i): the VPS_FALLBACK line helper and the (idle) async-copy state.
    f._log_vps_fallback = types.MethodType(TPUModelRunner._log_vps_fallback, f)
    f._vps_fb_async = r._l4_vpshost.AsyncCounts()
    f._warm_vps_draws = types.MethodType(TPUModelRunner._warm_vps_draws, f)
    gen = torch.Generator().manual_seed(7)
    f._get_sampling_generator = lambda: gen

    def params(cs, ce, n, dt, dev):
        return (torch.ones(n, 1), torch.full((n, 1), 20, dtype=torch.int32), torch.ones(n, 1))

    f._padded_sampling_params = params
    monkeypatch.setattr(r, "_vps_verify_fb_compiled",
                        lambda local, off, *a: (torch.zeros(4), torch.zeros(3, dtype=torch.int32)))
    monkeypatch.setattr(r, "_vps_verify_compiled", lambda *a: torch.zeros(4))
    monkeypatch.setattr(r, "synchronize_tensors", lambda *a, **k: None)
    return f, gen


def _record_rand(monkeypatch):
    rec = []
    real = torch.rand

    def rand(*a, **k):
        rec.append((tuple(a[0]) if a else None, k.get("generator")))
        return real(*a, **k)

    monkeypatch.setattr(torch, "rand", rand)
    return rec


@pytest.mark.parametrize("rounds", [0, 1, 4])
@pytest.mark.parametrize("cand", [True, False])
def test_warm_draws_match_serving_draws_and_leave_the_stream(rounds, cand, monkeypatch):
    _flag(monkeypatch, "VLLM_TPU_VPS_WARM_DRAWS", rounds or None)
    f, gen = _vps_runner(monkeypatch, cand)
    R, Ld, Lt = 8, 24, 24
    bidx = torch.arange(R, dtype=torch.int32)
    tidx = torch.arange(Lt, dtype=torch.int32)
    drafts = torch.zeros(Ld, dtype=torch.int32)
    dlen = torch.full((R,), K, dtype=torch.int32)
    seg = torch.arange(Ld, dtype=torch.int32) // K
    grp = torch.arange(Ld, dtype=torch.int32)
    state0 = gen.get_state()
    rec = _record_rand(monkeypatch)
    dummy = torch.zeros((R * (1 + K), 4))
    TPUModelRunner._precompile_vps_verify(f, dummy, bidx, tidx, drafts, dlen, seg, grp,
                                          torch.zeros(Ld), served_only=True)
    warm = list(rec)
    assert torch.equal(gen.get_state(), state0)  # the sampling stream is untouched
    assert all(g_ is not gen for _s, g_ in warm)
    served_rows = r.vps_wiring.fallback_slots(0, 0, 8)
    rec.clear()
    md = types.SimpleNamespace(bonus_logits_indices=bidx, target_logits_indices=tidx,
                               draft_token_ids=drafts, draft_lengths=dlen, segment_ids=seg,
                               group_indices=grp)
    TPUModelRunner._sample_spec_verify_vps(f, torch.zeros((R * (1 + K), 256)), md, 0, R,
                                           gen, fallback_rows=served_rows)
    serve = [s for s, _g in rec]
    assert len(serve) in (3, 4)
    if rounds == 0:
        assert warm == []
    else:
        assert [s for s, _g in warm] == serve * rounds
        assert len({id(g_) for _s, g_ in warm}) == 1


# ---- flags off / on in fresh processes ----------------------------------------------

_PROBE = r"""
import sys
import vllm_torchtpu
from vllm.v1.worker import gpu_input_batch
from vllm.v1 import utils as vu
from vllm_torchtpu.runner import tpu_runner as r
print("ESL", r._eager_shape_log is None)
print("SCF", "vllm_torchtpu.runner.sampling_copy_full" in sys.modules)
print("COPYSLICE_STOCK", gpu_input_batch.copy_slice is vu.copy_slice)
print("ESL_MOD", "vllm_torchtpu.runner.eager_shape_log" in sys.modules)
"""


def _probe(env_on):
    keep = ("VLLM_TPU_SAMPLING_COPY_FULL", "VLLM_TPU_EAGER_SHAPE_LOG",
            "VLLM_TPU_SPEC_TAIL_REJ_PAD", "VLLM_TPU_VPS_WARM_DRAWS")
    env = {k: v for k, v in os.environ.items() if k not in keep}
    env.update(env_on)
    out = subprocess.run([sys.executable, "-c", _PROBE], env=env, capture_output=True,
                         text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-4000:]
    keys = {"ESL", "SCF", "COPYSLICE_STOCK", "ESL_MOD"}
    pairs = [ln.split(" ", 1) for ln in out.stdout.splitlines() if " " in ln]
    return {k: v for k, v in pairs if k in keys}


def test_flags_off_identity_and_on_wiring_in_fresh_processes():
    assert _probe({}) == {"ESL": "True", "SCF": "False", "COPYSLICE_STOCK": "True",
                          "ESL_MOD": "False"}
    on = _probe({"VLLM_TPU_SAMPLING_COPY_FULL": "1", "VLLM_TPU_EAGER_SHAPE_LOG": "50"})
    assert on == {"ESL": "False", "SCF": "True", "COPYSLICE_STOCK": "False", "ESL_MOD": "True"}


def test_flags_off_spec_tail_call_is_unchanged(monkeypatch):
    _flag(monkeypatch, "VLLM_TPU_SPEC_TAIL_REJ_PAD", None)
    f = _Fake()
    f.max_num_reqs = 13
    assert TPUModelRunner._spec_tail_rej_extra(f) == ()
    _flag(monkeypatch, "VLLM_TPU_SPEC_TAIL_REJ_PAD", 1)
    assert TPUModelRunner._spec_tail_rej_extra(f) == (13,)
    assert envs.VLLM_TPU_SPEC_TAIL_REJ_PAD is True


# ---- E-d ------------------------------------------------------------------------

TOKEN_BUCKETS = [16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 512, 1024, 2048]


def _verify_runner(monkeypatch, fd, vps=True, max_num_reqs=64, cap=32):
    monkeypatch.setenv("VLLM_TPU_SPEC_VERIFY_PRECOMPILE_MAX_SEQS", str(cap))
    monkeypatch.setenv("VLLM_TPU_SPEC_VERIFY_PRECOMPILE_SERVED_ONLY", "1")
    monkeypatch.setenv("VLLM_TPU_VERIFY_FUSED", "1")
    _flag(monkeypatch, "VLLM_TPU_SPEC_VERIFY_PRECOMPILE_FULL_DRAFT_MAX_SEQS", fd)
    f = _Fake()
    f.speculative_config = types.SimpleNamespace(num_speculative_tokens=K)
    f.rejection_sampler = types.SimpleNamespace(synthetic_mode=False)
    f.max_num_reqs = max_num_reqs
    f.num_reqs_paddings = r._get_req_paddings(min_req_size=r.MIN_NUM_SEQS,
                                              max_req_size=max_num_reqs)
    f.num_tokens_paddings = list(TOKEN_BUCKETS)
    f.vocab_size = 8
    f._hidden_states_dtype = torch.bfloat16
    f.device = torch.device("cpu")
    f._vps_head = types.SimpleNamespace() if vps else None
    f._is_async_drafter = True
    f._precompile_timed = lambda name: contextlib.nullcontext()
    f._spec_verify_bucket_reachable = types.MethodType(
        TPUModelRunner._spec_verify_bucket_reachable, f)
    f._spec_verify_full_draft_pairs = types.MethodType(
        TPUModelRunner._spec_verify_full_draft_pairs, f)
    rec, fused = [], []
    f._precompile_vps_verify = (
        lambda dummy, bidx, tidx, *a, served_only=False: rec.append((dummy.shape[0], bidx.shape[0])))
    f._precompile_verify_fused = lambda dummy, bidx, *a: fused.append((dummy.shape[0], bidx.shape[0]))
    monkeypatch.setattr(r, "synchronize_tensors", lambda *a, **k: None)
    TPUModelRunner._precompile_rejection_sampler(f)
    return rec, fused


def _full_draft_pair(n, M=64):
    return (r._get_padded_token_len(TOKEN_BUCKETS, min(n * (K + 1), TOKEN_BUCKETS[-1])),
            r._get_padded_num_reqs_with_upper_limit(n, M))


@pytest.mark.parametrize(
    ("cap", "fd", "extra"),
    [(32, None, set()), (32, 32, set()), (32, 41, {(160, 64), (192, 64)}),
     (32, 48, {(160, 64), (192, 64)}),
     (32, 64, {(160, 64), (192, 64), (224, 64), (256, 64)}),
     # cap 36: the sweep already reaches (160, 64), so only (192, 64) is new
     (36, 48, {(192, 64)})],
)
def test_full_draft_verify_pairs_above_the_cap(cap, fd, extra, monkeypatch):
    base, _ = _verify_runner(monkeypatch, None, cap=cap)
    rec, _ = _verify_runner(monkeypatch, fd, cap=cap)
    assert len(rec) == len(set(rec))  # no program warmed twice
    assert set(rec) - set(base) == extra
    if cap == 32:
        assert all(nr <= 32 and nt <= 128 for nt, nr in base)
    covered = set(rec)
    for n in range(1, (fd or cap) + 1):
        assert _full_draft_pair(n) in covered, n


def test_full_draft_pairs_need_served_only_vps(monkeypatch):
    rec, fused = _verify_runner(monkeypatch, 64, vps=False)
    assert rec == []  # no VPS head: nothing extra, the fused-verify sweep unchanged
    base_rec, base_fused = _verify_runner(monkeypatch, None, vps=False)
    assert fused == base_fused and base_rec == []
