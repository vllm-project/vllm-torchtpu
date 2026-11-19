# SPDX-License-Identifier: Apache-2.0
"""q38 X-K-LIVE plumbing (VLLM_TPU_MK_LIVE_ROWS; L1 GDN half, shared with L3's MoE half), CPU.

  - `live_rows_from_qsl` is the `live` of `gdn_blocks` (the request has a query) repeated over the request's T rows, for
    runner-shaped query starts (longer than NS + 1 with padded entries repeating the last location, shorter, pads in the
    middle);
  - the planner declares the `live_rows` kernel input (SMEM int32 [B], last) only with `ExecSpec.live_rows`;
  - the real `build_wholestep_call` closures (kernel / QSA inputs / writers stubbed, as in the D07-b1 test) set
    `ExecSpec.live_rows` from the flag and hand the kernel `live_rows_from_qsl(qsl)` under `call` (layer 0's qsl) and
    `call_shared` (the shared qsl); with the flag unset the kernel gets no such input and the call's operands, specs and
    outputs are the flag-off ones;
  - the flag is off by default.
Run: JAX_PLATFORMS=cpu PYTHONPATH=src pytest tests/megakernel_runtime/test_l1live_plumbing.py
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tests.megakernel_runtime.test_l1d07b1_gdn_meta_shared import (  # noqa: E402  (sets XLA_FLAGS / vllm.envs stubs first)
    B,
    EXTRA,
    FLAG as META_FLAG,
    H,
    N_ATT,
    N_GDN,
    NAMES,
    POOL,
    R,
    _metadata_objects,
    _module,
    _per_layer_operands,
)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.megakernel.live_rows import ENV, LIVE_ROWS, live_rows_from_qsl  # noqa: E402
from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wsc  # noqa: E402
from vllm_torchtpu.megakernel_runtime.qwen38_segment_call import T_VERIFY, gdn_blocks  # noqa: E402

T = T_VERIFY


@pytest.mark.parametrize("NS,rows", [(4, 48), (4, 3), (8, 9), (16, 16)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mask_is_gdn_blocks_live(NS, rows, seed):
    """mask[s T + t] == request s is live in `gdn_blocks` (its write ids are non-zero; every id here is >= 1)."""
    rng = np.random.default_rng(seed * 31 + NS + rows)
    q_len = np.where(rng.random(rows) < 0.6, rng.integers(1, T + 1, rows), 0).astype(np.int32)
    q_len[min(1, rows - 1)] = 0  # a pad request in the middle
    qsl = np.concatenate([[0], np.cumsum(q_len)]).astype(np.int32)
    seq = (q_len + rng.integers(0, 40, rows)).astype(np.int32)
    st = rng.integers(0, 64, rows).astype(np.int32)
    ck = rng.integers(1, 97, (rows, T)).astype(np.int32)  # never the null block
    off = rng.integers(0, T, 97).astype(np.int32)
    got = np.asarray(live_rows_from_qsl(jnp.asarray(qsl), NS, T))
    _, wb, _ = gdn_blocks([st, off, ck, qsl, seq], NS, True)
    live = np.asarray(wb)[0, :, 0] != 0
    assert got.dtype == np.int32 and got.shape == (NS * T,)
    np.testing.assert_array_equal(got.reshape(NS, T), np.repeat(live[:, None].astype(np.int32), T, axis=1))
    if rows >= NS:
        assert live.any() and not live.all()  # not vacuous: both kinds present


def test_planner_declares_live_rows_only_with_the_exec_flag():
    from vllm_torchtpu.megakernel import kernel
    from vllm_torchtpu.megakernel.exec_spec import ExecSpec
    from vllm_torchtpu.megakernel.models import qwen38
    from vllm_torchtpu.megakernel.ops import I32

    spec = qwen38.segment_spec(qwen38.TINY, 2, L=qwen38.SEGMENT_LAYERS, pool_gdn=True)
    off = kernel.extra_inputs(spec, ExecSpec(R=2, B=16, T=4, ctx=128))
    on = kernel.extra_inputs(spec, ExecSpec(R=2, B=16, T=4, ctx=128, live_rows=True))
    assert LIVE_ROWS not in [x.name for x in off]
    assert [x.name for x in on] == [x.name for x in off] + [LIVE_ROWS]
    x = on[-1]
    assert (x.memory_space, tuple(x.shape), x.dtype) == ("smem", (16,), I32)


# ------------------------------------------------------------------- the whole-step call, stubbed

def _stub(monkeypatch, seen):
    """The D07-b1 test's stubs, except that the kernel stub records what reaches it and lists `live_rows` among its
    inputs iff the ExecSpec it was built for says so (as the planner does)."""
    from vllm_torchtpu.megakernel import kernel

    def mfss(mesh, spec, ex, extra_outs=()):
        seen["ex"] = ex
        names = EXTRA + ((LIVE_ROWS,) if ex.live_rows else ())

        def fn(pos, x, *rest):
            xs = dict(zip(names, rest[: len(names)]))
            seen["xs"] = xs
            w = rest[len(names):]
            bump = sum(jnp.sum(xs[k].astype(jnp.float32)) for k in NAMES) + jnp.sum(w[0])
            M = x.shape[0] // R
            rb, hi = xs["gdn_rb"], xs["gdn_hi"]
            z = jnp.zeros((R * N_ATT, M, 4), jnp.float32)
            if LIVE_ROWS in xs:  # carried into the residual so the jitted value can be read back
                bump = bump + 0.0 * jnp.sum(xs[LIVE_ROWS])
                seen["live"] = xs[LIVE_ROWS]
            return (x + bump, jnp.broadcast_to(rb[None], (R,) + rb.shape),
                    jnp.broadcast_to(hi[None], (R,) + hi.shape), z, z, z)

        fn.extra_names = names
        return fn, (P(), P("tp")) + (P(),) * len(names), None, 1

    def inputs(pool, tables, slots, trig, positions, starts, context, **kw):
        v = sum(jnp.sum(t) for tt in tables for t in tt) + jnp.sum(positions) + jnp.sum(starts) + jnp.sum(context)
        v = v + jnp.sum(trig).astype(jnp.int32)
        return {n: jnp.full((1,), v + j, jnp.int32) for j, n in enumerate(NAMES)}

    def scatter(pool, conv_ck, rec_ck, wb, layout=None, interpret=False):
        flat = jnp.concatenate([conv_ck.reshape(-1), rec_ck.reshape(-1), wb.reshape(-1)]).astype(pool.dtype)
        return pool.at[: flat.shape[0]].set(flat)

    def writes(pool, rows, norms, tables, qsa_layers, positions, trig, starts, context, **kw):
        return pool

    monkeypatch.setattr(wsc, "wholestep_spec", lambda *a, **k: None)
    monkeypatch.setattr(wsc, "make_flat_segment_step", mfss)
    monkeypatch.setattr(kernel, "make_step", lambda mesh, spec, ex: (
        None, {"plan": SimpleNamespace(w_names=["w0"]), "wshape": {"w0": ((4, H), jnp.float32)}}))
    monkeypatch.setattr(wsc, "qsa_step_inputs", inputs)
    monkeypatch.setattr(wsc, "scatter_gdn_ckpts_fast", scatter)
    monkeypatch.setattr(wsc, "wholestep_qsa_writes", writes)


def _build(mesh):
    from vllm_torchtpu.megakernel_runtime.qwen38_wholestep_cache import QSALayerCache

    layers = [QSALayerCache(compressed_slot=o % 2, main_slot=(o + 1) % 2) for o in range(N_ATT)]
    return wsc.build_wholestep_call(mesh, SimpleNamespace(att_every=4), B, None, layers, L=8, budget=64,
                                    max_groups=1024, norm_eps=1e-6, rotary_dim=64, interpret=True)


def _tail(rng):
    tabs = [rng.integers(0, 50, (48 * n,)).astype(np.int32) for _ in range(N_ATT) for n in (8, 32, 32)]
    positions = rng.integers(0, 300, (B,)).astype(np.int32)
    starts = np.arange(B + 1, dtype=np.int32)
    context = rng.integers(0, 300, (B,)).astype(np.int32)
    trig = rng.standard_normal((16, 128)).astype(np.float32)
    norms = [rng.standard_normal((n,)).astype(np.float32) for _ in range(N_ATT) for n in (256, 128)]
    return [*tabs, positions, starts, context, trig, *norms]


@pytest.mark.parametrize("shared", [False, True])
def test_call_hands_the_kernel_the_mask_only_with_the_flag(monkeypatch, shared):
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    if shared:
        monkeypatch.setenv(META_FLAG, "1")
    else:
        monkeypatch.delenv(META_FLAG, raising=False)
    md, layers, gdn_ids = _metadata_objects(seed=7)
    if shared:
        meta = [np.asarray(t.numpy()) for t in _module(layers, gdn_ids)._gdn_meta_shared(md)]
    else:
        meta = [np.asarray(t.numpy()) for t in _per_layer_operands(md, layers, gdn_ids)]
    rng = np.random.default_rng(3)
    hidden = rng.standard_normal((R * B, H)).astype(np.float32)
    w0 = rng.standard_normal((R * 4, H)).astype(np.float32)
    pool = np.zeros((R * POOL,), np.int32)
    tail = _tail(rng)
    outs = {}
    for flag in ("0", "1"):
        seen = {}
        _stub(monkeypatch, seen)
        if flag == "1":
            monkeypatch.setenv(ENV, "1")
        else:
            monkeypatch.delenv(ENV, raising=False)
        call, specs, osp, don = _build(mesh)
        assert call.__name__ == ("call_shared" if shared else "call")
        assert bool(seen["ex"].live_rows) is (flag == "1") and int(seen["ex"].live_rows) == int(flag == "1")
        res, p = jax.jit(call)(hidden, w0, pool, *meta, *tail)
        outs[flag] = (np.asarray(res), np.asarray(p), specs, osp, don)
        if flag == "1":  # under jit the stub saw a tracer: shape / dtype here, the values in the eager test below
            qsl = meta[1] if shared else meta[3]
            want = np.asarray(live_rows_from_qsl(jnp.asarray(qsl), B // T, T))
            assert tuple(seen["live"].shape) == (B,) and seen["live"].dtype == jnp.int32
            assert want.any() and not want.all()  # the step has live and pad requests
        else:
            assert "live" not in seen and LIVE_ROWS not in seen["xs"]
    # The call's operand list, specs, outputs and donation do not move with the flag; neither do the outputs here.
    assert outs["1"][2:] == outs["0"][2:]
    np.testing.assert_array_equal(outs["1"][0], outs["0"][0])
    np.testing.assert_array_equal(outs["1"][1], outs["0"][1])


def test_mask_reaches_the_kernel_with_the_calls_qsl(monkeypatch):
    """Eager (no jit): the array the kernel stub receives under the flag equals live_rows_from_qsl of layer 0's qsl."""
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    monkeypatch.delenv(META_FLAG, raising=False)
    monkeypatch.setenv(ENV, "1")
    seen = {}
    _stub(monkeypatch, seen)
    call, *_ = _build(mesh)
    md, layers, gdn_ids = _metadata_objects(seed=8)
    meta = [np.asarray(t.numpy()) for t in _per_layer_operands(md, layers, gdn_ids)]
    rng = np.random.default_rng(4)
    hidden = rng.standard_normal((R * B, H)).astype(np.float32)
    w0 = rng.standard_normal((R * 4, H)).astype(np.float32)
    pool = np.zeros((R * POOL,), np.int32)
    with jax.disable_jit():
        call(hidden, w0, pool, *meta, *_tail(rng))
    want = np.asarray(live_rows_from_qsl(jnp.asarray(meta[3]), B // T, T))
    np.testing.assert_array_equal(np.asarray(seen["live"]), want)
    assert want.any() and not want.all()
    # every GDN layer's qsl is the same step tensor (the call reads layer 0's)
    for k in range(N_GDN):
        np.testing.assert_array_equal(meta[5 * k + 3], meta[3])


def test_flag_off_by_default_and_registered(monkeypatch):
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation.tpu_compiler import _TPU_COMPILE_ENV_IGNORED

    monkeypatch.delenv(ENV, raising=False)
    assert ENV in envs.environment_variables and ENV not in _TPU_COMPILE_ENV_IGNORED
    assert envs.environment_variables[ENV]() is False
    monkeypatch.delenv("VLLM_TPU_MK_LIVE_GDN_COMPUTE", raising=False)
    assert wsc._live_rows_on() == 0
    monkeypatch.setenv("VLLM_TPU_MK_LIVE_GDN_COMPUTE", "1")
    assert wsc._live_rows_on() == 0  # the sub-mode does nothing without the mask flag
    monkeypatch.setenv(ENV, "1")
    assert wsc._live_rows_on() == 2
    monkeypatch.delenv("VLLM_TPU_MK_LIVE_GDN_COMPUTE")
    assert wsc._live_rows_on() == 1
    assert "VLLM_TPU_MK_LIVE_GDN_COMPUTE" in envs.environment_variables
    assert "VLLM_TPU_MK_LIVE_GDN_COMPUTE" not in _TPU_COMPILE_ENV_IGNORED


@pytest.mark.parametrize("B,want", [(4, 1), (8, 1), (16, 1), (32, 2), (48, 2), (64, 2)])
def test_gdn_compute_mode_refused_on_the_whole_state_path(monkeypatch, B, want):
    """Mode 2 builds only above 16 rows (L1 ruling; l1-live-c-aot-i2: 222 KiB over scoped VMEM at 16 rows); at <= 16 rows
    the call builds mode 1. Mode 1 alone is unchanged at every size."""
    monkeypatch.setenv(ENV, "1")
    monkeypatch.setenv("VLLM_TPU_MK_LIVE_GDN_COMPUTE", "1")
    assert wsc._live_rows_on(B) == want
    monkeypatch.delenv("VLLM_TPU_MK_LIVE_GDN_COMPUTE")
    assert wsc._live_rows_on(B) == 1


def test_build_refuses_gdn_compute_at_16_rows(monkeypatch):
    """The real build_wholestep_call (stubbed kernel, B = 16) with both envs set hands the kernel ExecSpec.live_rows == 1."""
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    monkeypatch.delenv(META_FLAG, raising=False)
    monkeypatch.setenv(ENV, "1")
    monkeypatch.setenv("VLLM_TPU_MK_LIVE_GDN_COMPUTE", "1")
    seen = {}
    _stub(monkeypatch, seen)
    _build(mesh)
    assert B == 16 and int(seen["ex"].live_rows) == 1
