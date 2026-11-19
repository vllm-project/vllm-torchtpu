# SPDX-License-Identifier: Apache-2.0
"""q38 L1-G1 GDN-COMMIT1: the runner side (CPU, torch CPU tensors).

  H1 placement (source order in runner/tpu_runner.py): the commit runs in sample_tokens after every chunk's read-offset
     update and after AsyncTPUCopyState.from_device_chunks (the sampled-token D2H issue); the full flush runs at the top
     of execute_model (before the seed-copy flush and the read-offset reset) and in forward_model before the megakernel;
     the precompile right after bind_and_warmup; every hook is gated by `self._g1_on` (= VLLM_TPU_MK_GDN_COMMIT1, set in
     __init__), so with the flag off the runner runs the a23cf54b5 code paths.
  H2 the torch glue end to end (stash -> commit_pending / flush_pending / precompile): the per-rank jax ops run the real
     commit / scatter kernels (Pallas interpret through a fake `torch_tpu._internal.pallas.jax_op`; torch.compile is
     the identity), two calls pending in one step go through ONE program (commit2), the pool afterwards == the JAX
     reference bytes; without read offsets the staging is written in full; the precompile writes nothing; the head
     group kinds of the shared segments stage / commit / precompile next to the whole step.
  H3 the id chain with the runner's own functions: `_update_read_offsets_impl` (the post-verify scatter of
     num_accepted - 1 from the rejection output) -> commit block `wb[l, s, read_offsets[st[l, s]]]` == the next step's
     read block from `gdn_blocks` == the source `_spec_seed_rows_impl` picks at a crossing, for every live request and
     every accept count 0..3; pad requests write and read nothing. Mutant: a commit on the pre-verify offsets misses.
Run: JAX_PLATFORMS=cpu PYTHONPATH=src:. pytest -p q38_cpu_device tests/q38_l1g1/test_g1_runner_hooks.py
"""

from __future__ import annotations

import os
import re
import sys
import types
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.megakernel_runtime import qwen38_gdn_commit as gc
from vllm_torchtpu.megakernel_runtime import qwen38_gdn_state as gs
from vllm_torchtpu.megakernel_runtime.qwen38_scatter import scatter_gdn_ckpts_fast

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUNNER = os.path.join(ROOT, "src", "vllm_torchtpu", "runner", "tpu_runner.py")
T = 4


# ------------------------------------------------------------------------------------------------ H1 placement


def _method(src: str, name: str) -> str:
    m = re.search(rf"\n    def {name}\(", src)
    assert m, name
    nxt = re.search(r"\n    def \w+\(", src[m.end():])
    return src[m.start(): m.end() + (nxt.start() if nxt else len(src))]


def test_hook_placement_in_the_runner():
    src = open(RUNNER, encoding="utf-8").read()
    st = _method(src, "sample_tokens")
    i_commit = st.index("_g1.commit_pending(self)")
    i_d2h = st.index("AsyncTPUCopyState.from_device_chunks(")
    upd = [m.start() for m in re.finditer(r"self\._update_mamba_slot_read_offsets\(", st)]
    assert len(upd) == 3 and max(upd) < i_d2h < i_commit, (upd, i_d2h, i_commit)
    assert st.count("_g1.commit_pending(self)") == 1 and st.count("_g1.") == 1
    gate = st.rindex('if getattr(self, "_g1_on", False):', 0, i_commit)
    assert st[gate:i_commit].count("\n") <= 5
    ex = _method(src, "execute_model")
    i_flush = ex.index('_g1.flush_pending(self, why="execute_model")')
    assert i_flush < ex.index("self._prepare_inputs(") < ex.index("self._flush_mamba_state_seed_copies()")
    assert i_flush < ex.index("self._reset_read_offsets_for_new_requests(")
    fm = _method(src, "forward_model")
    assert fm.index('_g1.flush_pending(self, why="forward")') < fm.index("self._megakernel.try_forward(")
    i_bind = src.index("            self._megakernel.bind_and_warmup(self)\n")
    assert src.index("_g1.precompile(self)") > i_bind and src[i_bind:src.index("_g1.precompile(self)")].count("\n") <= 5
    # every hook is gated, and the gate is the env read once in __init__
    hooks = [m.start() for m in re.finditer(r"from vllm_torchtpu\.megakernel_runtime import qwen38_gdn_commit as _g1", src)]
    assert len(hooks) == 4
    for h in hooks:
        before = src[:h].splitlines()[-4:]  # the gate line, up to two comment lines, the import line's indent
        assert any(line.strip() == 'if getattr(self, "_g1_on", False):' for line in before), before
    assert src.count("self._g1_on: bool = bool(envs.VLLM_TPU_MK_GDN_COMMIT1)") == 1


# ------------------------------------------------------------------------------------------------ H2 torch glue

KB = 128


def _t2j(x: torch.Tensor):
    if x.dtype == torch.float8_e4m3fn:
        return jax.lax.bitcast_convert_type(jnp.asarray(x.view(torch.uint8).numpy()), jnp.float8_e4m3fn)
    if x.dtype == torch.bfloat16:
        return jax.lax.bitcast_convert_type(jnp.asarray(x.view(torch.int16).numpy()), jnp.bfloat16)
    return jnp.asarray(x.numpy())


def _j2t(x) -> torch.Tensor:
    if x.dtype == jnp.float8_e4m3fn:
        return torch.from_numpy(np.asarray(jax.lax.bitcast_convert_type(x, jnp.uint8))).view(torch.float8_e4m3fn)
    if x.dtype == jnp.bfloat16:
        return torch.from_numpy(np.asarray(jax.lax.bitcast_convert_type(x, jnp.int16))).view(torch.bfloat16)
    return torch.from_numpy(np.array(x))


class _FakeOp:
    def __init__(self, name, fn, donate_argnums):
        self.name, self.fn, self.donate = name, fn, donate_argnums
        self.calls = 0

    def register_fake(self, f):
        self.fake = f

    def __call__(self, *args):
        self.calls += 1
        outs = self.fn(*(_t2j(a) for a in args))
        return tuple(_j2t(o) for o in outs)


@pytest.fixture
def fake_tpu(monkeypatch):
    ops = []
    pallas = types.ModuleType("torch_tpu._internal.pallas")

    def jax_op(name, fn, donate_argnums=()):
        op = _FakeOp(name, fn, donate_argnums)
        ops.append(op)
        return op

    pallas.jax_op = jax_op
    tt = types.ModuleType("torch_tpu")
    internal = types.ModuleType("torch_tpu._internal")
    internal.pallas = pallas
    tt._internal = internal
    monkeypatch.setitem(sys.modules, "torch_tpu", tt)
    monkeypatch.setitem(sys.modules, "torch_tpu._internal", internal)
    monkeypatch.setitem(sys.modules, "torch_tpu._internal.pallas", pallas)
    monkeypatch.setattr(torch, "compile", lambda *a, **k: (lambda f: f))
    monkeypatch.setattr(gc, "INTERPRET", True)
    gc._ops.cache_clear()
    yield ops
    gc._ops.cache_clear()


def _world(NL=2, NS=2, nb_extra=8, seed=3):
    rng = np.random.default_rng(seed)
    nmb = 1 + 2 * NL * NS * T + nb_extra
    shape = (nmb * (512 // KB), KB, 1, 4, 256)
    raw = rng.integers(0, 256, size=shape, dtype=np.uint8)
    layout = gs.Qwen38GdnPoolLayout(pool_shape=shape, pool_dtype="float8_e4m3fn")
    pool_t = torch.from_numpy(raw.copy()).view(torch.float8_e4m3fn)
    whole = SimpleNamespace(pool=pool_t, kind=SimpleNamespace(layout=layout, kw={"write": True}),
                            gdn_ids=list(range(NL)), buckets=(NS * T,))
    offsets = torch.zeros(128, dtype=torch.int32)  # block-indexed; the state ids below live in 60..67
    runner = SimpleNamespace(_megakernel=SimpleNamespace(whole_module=whole), mamba_slot_read_offsets=offsets)
    return rng, raw, layout, whole, runner


def _staging(rng, NL, NS, ids, st):
    conv = rng.standard_normal((NL, NS, 8, 1280)).astype(np.float32)
    rec = rng.standard_normal((NL, NS * T * 6, 128, 128)).astype(np.float32)
    rec_j = jnp.asarray(rec).astype(jnp.bfloat16)
    return dict(conv=torch.from_numpy(conv), rec=_j2t(rec_j), wb=torch.from_numpy(ids), st=torch.from_numpy(st),
                conv_j=jnp.asarray(conv), rec_j=rec_j)


def _pool_u8(t: torch.Tensor) -> np.ndarray:
    return t.view(torch.uint8).numpy().copy()


def test_torch_glue_commit_two_calls_then_flush(monkeypatch, fake_tpu):
    monkeypatch.setenv(gc.ENV, "1")
    NL, NS = 2, 2
    rng, raw, layout, whole, runner = _world(NL, NS)
    ids = (1 + np.arange(2 * NL * NS * T, dtype=np.int32)).reshape(2, NL, NS, T)
    ids[1, :, 1] = 0  # the second call's second request is a pad
    sts = (60 + np.arange(2 * NL * NS, dtype=np.int32)).reshape(2, NL, NS)
    acc = np.array([[[0, 3], [1, 2]], [[3, 0], [2, 1]]], np.int32)
    offs = runner.mamba_slot_read_offsets
    for c in range(2):
        offs[torch.from_numpy(sts[c].reshape(-1)).long()] = torch.from_numpy(acc[c].reshape(-1))
    stg = [_staging(rng, NL, NS, ids[c], sts[c]) for c in range(2)]
    for c in range(2):  # the op's result with the 4 staging outputs (two whole-step calls: a multi-chunk step)
        outs = gc.stash(whole.kind, ["res", "pool", stg[c]["conv"], stg[c]["rec"], stg[c]["wb"], stg[c]["st"]], NS * T)
        assert outs == ["res", "pool"]
    assert len(gc.pending(runner)) == 2
    ref = jax.lax.bitcast_convert_type(jnp.asarray(raw), jnp.float8_e4m3fn)
    for c in range(2):
        ref = gc.commit_gdn_ckpts_fast(ref, stg[c]["conv_j"], stg[c]["rec_j"], jnp.asarray(ids[c]), jnp.asarray(acc[c]),
                                       layout=layout, interpret=True)
    n0 = gc.Counters.committed
    assert gc.commit_pending(runner) == 2 and gc.pending(runner) == [] and gc.Counters.committed == n0 + 2
    got = _pool_u8(whole.pool)
    want = np.asarray(jax.lax.bitcast_convert_type(ref, jnp.uint8))
    assert np.array_equal(got, want), "torch commit != the JAX commit reference"
    assert (got != raw).any(), "control: the commit wrote something"
    ops = {o.name.split("::")[1]: o for o in fake_tpu}
    assert ops["q38_g1_gdn_commit2"].calls == 1 and ops["q38_g1_gdn_commit1"].calls == 0, "two pending calls -> one program"
    assert all(o.donate == (0,) for o in fake_tpu)
    # mutant: committing on the pre-verify (all-zero) offsets writes other blocks
    _, raw2, _, whole2, runner2 = _world(NL, NS)
    for c in range(2):
        gc.stash(whole2.kind, ["r", "p", stg[c]["conv"], stg[c]["rec"], stg[c]["wb"], stg[c]["st"]], NS * T)
    gc.commit_pending(runner2)
    assert not np.array_equal(_pool_u8(whole2.pool), want), "stale offsets must not reproduce the committed bytes"
    # flush: full scatter of a pending call
    gc.stash(whole.kind, ["r", "p", stg[0]["conv"], stg[0]["rec"], stg[0]["wb"], stg[0]["st"]], NS * T)
    before = jax.lax.bitcast_convert_type(jnp.asarray(_pool_u8(whole.pool)), jnp.float8_e4m3fn)
    full = scatter_gdn_ckpts_fast(before, stg[0]["conv_j"], stg[0]["rec_j"], jnp.asarray(ids[0]), layout=layout,
                                  interpret=True)
    f0 = gc.Counters.flushed
    assert gc.flush_pending(runner, why="test") == 1 and gc.Counters.flushed == f0 + 1 and gc.pending(runner) == []
    assert np.array_equal(_pool_u8(whole.pool), np.asarray(jax.lax.bitcast_convert_type(full, jnp.uint8)))


def test_no_read_offsets_writes_in_full(monkeypatch, fake_tpu):
    monkeypatch.setenv(gc.ENV, "1")
    NL, NS = 1, 2
    rng, raw, layout, whole, runner = _world(NL, NS)
    runner.mamba_slot_read_offsets = None
    ids = (1 + np.arange(NL * NS * T, dtype=np.int32)).reshape(NL, NS, T)
    s = _staging(rng, NL, NS, ids, np.zeros((NL, NS), np.int32))
    gc.stash(whole.kind, ["r", "p", s["conv"], s["rec"], s["wb"], s["st"]], NS * T)
    assert gc.commit_pending(runner) == 1
    full = scatter_gdn_ckpts_fast(jax.lax.bitcast_convert_type(jnp.asarray(raw), jnp.float8_e4m3fn), s["conv_j"],
                                  s["rec_j"], jnp.asarray(ids), layout=layout, interpret=True)
    assert np.array_equal(_pool_u8(whole.pool), np.asarray(jax.lax.bitcast_convert_type(full, jnp.uint8)))


def test_precompile_writes_nothing_and_uses_the_staging_shapes(monkeypatch, fake_tpu):
    monkeypatch.setenv(gc.ENV, "1")
    rng, raw, layout, whole, runner = _world(2, 2)
    whole.buckets = (4, 8)
    seen = []
    orig = gc.commit_from_offsets

    def spy(pool, conv, rec, wb, st, offsets, **k):
        seen.append((tuple(conv.shape), tuple(rec.shape), str(rec.dtype), tuple(wb.shape), tuple(st.shape)))
        return orig(pool, conv, rec, wb, st, offsets, **k)

    monkeypatch.setattr(gc, "commit_from_offsets", spy)
    gc.precompile(runner)
    assert np.array_equal(_pool_u8(whole.pool), raw), "precompile on null ids must not write"
    from vllm_torchtpu.megakernel.ops import REC

    want = [((2, b // T, 8, 1280), (2, b * 6, 128, 128), str(jnp.dtype(REC)), (2, b // T, T), (2, b // T)) for b in (4, 8)]
    assert seen == want, seen
    sh = gc.staging_shapes(layout, 2, 64)
    assert sh["rec"][0] == (2, 64 * 6, 128, 128) and gc.staging_bytes(layout, 2, 64) == (
        2 * 16 * 8 * 1280 * 4 + 2 * 64 * 6 * 128 * 128 * jnp.dtype(REC).itemsize + 2 * 16 * T * 4 + 2 * 16 * 4)
    # the served whole step at b64 (33 GDN layers, bf16 state): 416.6 MiB per rank (the build note's number)
    if jnp.dtype(REC) == jnp.dtype(jnp.bfloat16):
        assert round(gc.staging_bytes(layout, 33, 64) / 2 ** 20, 1) == 416.6


def test_head_kinds_stage_commit_and_precompile(monkeypatch, fake_tpu):
    """The head group kinds (driver segments, (L, head=True)) stage, commit and precompile like the whole step; body
    segment kinds and non-writing (shadow) kinds are never targets; head staging shapes follow the driver's buckets
    (chunked above KERNEL_ROWS = 64)."""
    monkeypatch.setenv(gc.ENV, "1")
    NL, NS = 3, 2  # 1 + 2 * 3 * 2 * 4 + 8 = 57 pool blocks: room for both calls' ids (<= 48)
    rng, raw, layout, whole, runner = _world(NL, NS)
    head = SimpleNamespace(layout=layout, write=True, L=3)
    body = SimpleNamespace(layout=layout, write=True, L=4)
    shadow_head = SimpleNamespace(layout=layout, write=False, L=3)
    runner._megakernel.segments = SimpleNamespace(pool=whole.pool, kinds={(3, True): head, (4, False): body})
    runner._megakernel.buckets = (8, 16, 96)
    assert [k for k, _ in gc._targets(runner)] == [whole.kind, head]
    assert gc.stash(shadow_head, ["r", "p", "c", "x", "w", "s"], 8) == ["r", "p", "c", "x", "w", "s"]
    ids = (1 + np.arange(2 * 3 * NS * T, dtype=np.int32)).reshape(2, 3, NS, T)
    sts = (60 + np.arange(2 * 3 * NS, dtype=np.int32)).reshape(2, 3, NS)
    offs = runner.mamba_slot_read_offsets
    acc = rng.integers(0, T, (2, 3, NS)).astype(np.int32)
    for c in range(2):
        offs[torch.from_numpy(sts[c].reshape(-1)).long()] = torch.from_numpy(acc[c].reshape(-1))
    stg = [_staging(rng, 3, NS, ids[c], sts[c]) for c in range(2)]
    gc.stash(head, ["r", "p", stg[0]["conv"], stg[0]["rec"], stg[0]["wb"], stg[0]["st"]], NS * T)
    stg_w = _staging(rng, 3, NS, ids[1], sts[1])
    gc.stash(whole.kind, ["r", "p", stg_w["conv"], stg_w["rec"], stg_w["wb"], stg_w["st"]], NS * T)
    assert len(gc.pending(runner)) == 2
    ref = jax.lax.bitcast_convert_type(jnp.asarray(raw), jnp.float8_e4m3fn)
    ref = gc.commit_gdn_ckpts_fast(ref, stg_w["conv_j"], stg_w["rec_j"], jnp.asarray(ids[1]), jnp.asarray(acc[1]),
                                   layout=layout, interpret=True)
    ref = gc.commit_gdn_ckpts_fast(ref, stg[0]["conv_j"], stg[0]["rec_j"], jnp.asarray(ids[0]), jnp.asarray(acc[0]),
                                   layout=layout, interpret=True)
    assert gc.commit_pending(runner) == 2 and gc.pending(runner) == []
    assert np.array_equal(_pool_u8(whole.pool), np.asarray(jax.lax.bitcast_convert_type(ref, jnp.uint8)))
    plan = [(k is head, nl, b) for k, _, nl, b in gc._plan(runner)]
    assert plan == [(False, NL, b) for b in whole.buckets] + [(True, 3, b) for b in (8, 16, 32, 64)], plan


def test_flag_off_stash_and_hooks_are_inert(monkeypatch, fake_tpu):
    monkeypatch.delenv(gc.ENV, raising=False)
    rng, raw, layout, whole, runner = _world(1, 2)
    outs = ["r", "p", "c", "x", "w", "s"]
    assert gc.stash(whole.kind, list(outs), 8) == outs and gc.pending(runner) == []
    assert gc.commit_pending(runner) == 0 and gc.flush_pending(runner) == 0
    assert np.array_equal(_pool_u8(whole.pool), raw) and fake_tpu == []


# ------------------------------------------------------------------------------------------------ H3 id chain


def test_commit_block_is_the_next_read_and_the_seed_source():
    import vllm_torchtpu.runner.tpu_runner as rm

    from vllm_torchtpu.megakernel_runtime.qwen38_segment_call import gdn_blocks

    rng = np.random.default_rng(8)
    NL, NS, width, nb = 3, 8, 8, 400
    num_reqs = 6  # requests 6, 7 are pads
    perm = rng.permutation(np.arange(10, 200))[: NL * width].astype(np.int64)  # state blocks: distinct pool-wide
    sts = [perm[ln * width: (ln + 1) * width] for ln in range(NL)]
    cks = [np.stack([200 + 40 * ln + 4 * s + np.arange(T) for s in range(width)]).astype(np.int32) for ln in range(NL)]
    acc_true = np.array([0, 1, 2, 3, 3, 0, 0, 0], np.int64)  # every accept count among the live requests
    K = T - 1
    nt = np.full((width, K + 1), rm.INVALID_TOKEN_ID, np.int64)
    for s in range(num_reqs):
        nt[s, : acc_true[s] + 1] = 1000 + s
    offs = torch.full((nb,), 2, dtype=torch.int32)  # pre-verify junk
    pre = offs.clone()
    rm._update_read_offsets_impl(offs, [torch.from_numpy(s) for s in sts], torch.from_numpy(nt),
                                 torch.tensor(num_reqs))
    qsl = np.concatenate([[0], np.cumsum([T] * num_reqs + [0] * (width - num_reqs))]).astype(np.int32)
    seq = np.full(width, 300, np.int32)

    def meta(off):
        out = []
        for ln in range(NL):
            out += [jnp.asarray(sts[ln].astype(np.int32)), jnp.asarray(off), jnp.asarray(cks[ln]), jnp.asarray(qsl),
                    jnp.asarray(seq)]
        return out

    off_np = offs.numpy().astype(np.int32)
    _, wb, _ = (np.asarray(x) for x in gdn_blocks(meta(pre.numpy()), NS, True))  # this step's write ids
    rb_next, _, _ = (np.asarray(x) for x in gdn_blocks(meta(off_np), NS, True))  # the next step's read ids
    st = np.stack([s.astype(np.int32) for s in sts])
    src, dst, a = (np.asarray(x) for x in gc.commit_plan(jnp.asarray(wb), jnp.take(jnp.asarray(off_np), jnp.asarray(st))))
    dst = dst.reshape(NL, NS)
    for ln in range(NL):
        for s in range(NS):
            if s < num_reqs:
                assert a.reshape(NL, NS)[ln, s] == acc_true[s]
                assert dst[ln, s] == rb_next[ln, s] == cks[ln][s, acc_true[s]] != 0, (ln, s)
            else:
                assert dst[ln, s] == 0 and rb_next[ln, s] == 0, (ln, s)
    # a crossing of every live request: the seed copy's source is the committed block
    for ln in range(NL):
        rows = [[int(sts[ln][s]), 0, 1, 500 + s, *[int(x) for x in cks[ln][s]]] for s in range(num_reqs)]
        src_t, dst_t = rm._spec_seed_rows_impl(offs, torch.tensor(rows, dtype=torch.long), 1)
        assert src_t.tolist() == [int(dst[ln, s]) for s in range(num_reqs)], ln
    # mutant: the commit on the pre-verify offsets misses the next read
    _, bad, _ = (np.asarray(x) for x in gc.commit_plan(jnp.asarray(wb), jnp.take(jnp.asarray(pre.numpy()),
                                                                                jnp.asarray(st))))
    assert (bad.reshape(NL, NS)[:, :num_reqs] != rb_next[:, :num_reqs]).any()
