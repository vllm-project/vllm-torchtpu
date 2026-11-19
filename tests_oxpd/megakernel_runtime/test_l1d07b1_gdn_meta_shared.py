# SPDX-License-Identifier: Apache-2.0
"""q38 L1-D07B1: VLLM_TPU_MK_GDN_META_DEDUP, the whole step's GDN block ids in one vectorised pass (CPU).

The runner gives every GDN layer its own metadata object (one KV cache group per GDN layer: the GCD group-size rule with
the MTP draft in the target's buckets), but the slot read offsets, query start locations and sequence lengths in them
are the step's own tensors, the same objects for every layer; only the state and checkpoint ids differ. With the flag
the whole-step call takes `off, qsl, seq` once plus the per-layer state / checkpoint ids and derives every layer's read
/ write / has-initial ids in one pass (`gdn_blocks_shared`) instead of one chain of small XLA ops per layer.

  - the shared pass is bit-identical to the per-layer `gdn_blocks` (ids, dtypes, shapes), incl. padded / dead
    requests, requests without an initial state, out-of-pool ids under the clamp, write=False, ragged inputs;
  - the real `build_wholestep_call` closures (`call` vs `call_shared`, kernel / QSA-input / writer stubbed so it runs
    on CPU) give bit-identical outputs from the module's two operand lists, built from the same metadata objects by
    the module's own `_gdn_meta_shared`; the operand specs differ only in the GDN metadata run;
  - the host check refuses a layer whose read offsets / query starts / seq lens are not the first layer's tensors;
  - the flag is off by default and the module refuses a flag change between bind and build.
Run: JAX_PLATFORMS=cpu PYTHONPATH=src pytest tests/megakernel_runtime/test_l1d07b1_gdn_meta_shared.py
"""

from __future__ import annotations

import os
import sys
import types
from types import SimpleNamespace

try:
    import vllm.envs  # noqa: F401
except ImportError:
    _vllm = types.ModuleType("vllm")
    _envs = types.ModuleType("vllm.envs")
    _envs.VLLM_KV_CACHE_LAYOUT = "NHD"
    _vllm.envs = _envs
    sys.modules.setdefault("vllm", _vllm)
    sys.modules.setdefault("vllm.envs", _envs)

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from jax.sharding import Mesh  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wsc  # noqa: E402
from vllm_torchtpu.megakernel_runtime.qwen38_segment_call import (  # noqa: E402
    GDN_SHARED,
    T_VERIFY,
    gdn_blocks,
    gdn_blocks_shared,
)

FLAG = "VLLM_TPU_MK_GDN_META_DEDUP"
NB = 97  # pool blocks (read-offset table length)


def _meta(NL: int, NS: int, rows: int, seed: int, ragged: bool = False):
    """Runner-shaped GDN metadata for NL layers: per-layer state / checkpoint ids (distinct per layer), one shared
    read-offset table, query start locations and sequence lengths. Live requests have 1..4 query tokens, some with
    no context (no initial state); requests past the live ones (and the runner rows past NS) carry junk ids, so the
    live mask is visible; checkpoint ids include negative and out-of-pool values (the clamp)."""
    rng = np.random.default_rng(seed)
    n_live = max(1, min(NS, rows) - 1)
    q_len = np.zeros(rows, np.int32)
    q_len[:n_live] = rng.integers(1, T_VERIFY + 1, n_live)
    qsl = np.concatenate([[0], np.cumsum(q_len)]).astype(np.int32)
    ctx = rng.choice(np.array([0, 5, 300], np.int32), rows)
    ctx[0] = 0  # one live request without an initial state
    ctx[1 % n_live] = 7
    seq = np.where(q_len > 0, q_len + ctx, rng.integers(1, 50, rows)).astype(np.int32)
    off = rng.integers(0, T_VERIFY, NB).astype(np.int32)
    sts, cks = [], []
    for ln in range(NL):
        r = rows + (2 if ragged and ln == 0 else 0)
        sts.append(rng.integers(0, NB, r).astype(np.int32))
        rc = rows + (3 if ragged and ln == 1 else 0)
        cks.append(rng.integers(-3, NB + 40, (rc, T_VERIFY + (1 if ragged and ln == 2 else 0))).astype(np.int32))
    return off, qsl, seq, sts, cks


def _full(off, qsl, seq, sts, cks):
    """The per-layer operand list (`[st, off, ck, qsl, seq]` per layer), as the flag-off module passes it."""
    out = []
    for st, ck in zip(sts, cks):
        out += [st, off, ck, qsl, seq]
    return out


@pytest.mark.parametrize("NS,rows", [(4, 48), (16, 48), (8, 6)])
@pytest.mark.parametrize("write", [True, False])
@pytest.mark.parametrize("clamp", [None, NB - 30])
@pytest.mark.parametrize("ragged", [False, True])
def test_shared_blocks_bit_identical_to_per_layer(NS, rows, write, clamp, ragged):
    NL = 33
    off, qsl, seq, sts, cks = _meta(NL, NS, rows, seed=NS * 7 + rows + int(write) + int(ragged) * 3, ragged=ragged)
    full = _full(off, qsl, seq, sts, cks)
    ref = jax.jit(lambda *m: gdn_blocks(m, NS, write, clamp))(*full)
    got = jax.jit(lambda o, q, s, *ids: gdn_blocks_shared(o, q, s, ids[:NL], ids[NL:], NS, write, clamp))(
        off, qsl, seq, *sts, *cks
    )
    for name, r, g in zip(("rb", "wb", "hi"), ref, got):
        assert g.shape == r.shape and g.dtype == r.dtype, (name, g.shape, r.shape, g.dtype, r.dtype)
        np.testing.assert_array_equal(np.asarray(g), np.asarray(r), err_msg=name)
    rb, wb, hi = (np.asarray(x) for x in ref)
    # Not vacuous: per-layer ids differ, live and dead requests both present, has-initial both ways.
    assert (rb != 0).any() and not (rb == rb[:1]).all()
    assert (wb != 0).any() == write
    assert hi.any() and not hi.all()


# --------------------------------------------------------------------------------- the whole-step call, stubbed

R, B, L, N_ATT, H = 2, 16, 8, 2, 8
N_GDN = L - N_ATT
POOL = 512
NAMES = ("qsa_pool", "qsa_ck", "qsa_cktab", "qsa_nvis", "qsa_tail", "qsa_mtab", "qsa_off", "qsa_scales", "qsa_rope",
         "qsa_ctx", "qsa_ring", "qsa_rope_ck")
EXTRA = NAMES + ("gdn_pool", "gdn_rb", "gdn_hi")


def _stub(monkeypatch):
    """Stub the kernel, the QSA step inputs and the writers so `build_wholestep_call` runs on CPU. The stubs carry
    what reaches them into the outputs: the residual adds every QSA input and the weight, the scatter writes the
    kernel's view of rb / hi (returned as the checkpoints) and the scatter's wb into the pool, the QSA writer adds the
    tables and step inputs."""
    from vllm_torchtpu.megakernel import kernel

    def fn(pos, x, *rest):
        xs = dict(zip(EXTRA, rest[: len(EXTRA)]))
        w = rest[len(EXTRA):]
        bump = sum(jnp.sum(xs[k].astype(jnp.float32)) for k in NAMES) + jnp.sum(w[0])
        M = x.shape[0] // R
        rb, hi = xs["gdn_rb"], xs["gdn_hi"]
        z = jnp.zeros((R * N_ATT, M, 4), jnp.float32)
        return (x + bump, jnp.broadcast_to(rb[None], (R,) + rb.shape), jnp.broadcast_to(hi[None], (R,) + hi.shape),
                z, z, z)

    fn.extra_names = EXTRA

    def mfss(mesh, spec, ex, extra_outs=()):
        return fn, (P(), P("tp")) + (P(),) * len(EXTRA), None, 1

    def inputs(pool, tables, slots, trig, positions, starts, context, **kw):
        v = sum(jnp.sum(t) for tt in tables for t in tt) + jnp.sum(positions) + jnp.sum(starts) + jnp.sum(context)
        v = v + jnp.sum(trig).astype(jnp.int32)
        return {n: jnp.full((1,), v + j, jnp.int32) for j, n in enumerate(NAMES)}

    def scatter(pool, conv_ck, rec_ck, wb, layout=None, interpret=False):
        flat = jnp.concatenate([conv_ck.reshape(-1), rec_ck.reshape(-1), wb.reshape(-1)]).astype(pool.dtype)
        return pool.at[: flat.shape[0]].set(flat)

    def writes(pool, rows, norms, tables, qsa_layers, positions, trig, starts, context, **kw):
        v = sum(jnp.sum(t) for tt in tables for t in tt) + jnp.sum(positions) + jnp.sum(starts) + jnp.sum(context)
        v = v + sum(jnp.sum(a).astype(jnp.int32) for r in rows for a in r)
        return pool.at[-1].set(v.astype(pool.dtype))

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
    return wsc.build_wholestep_call(mesh, SimpleNamespace(att_every=4), B, None, layers, L=L, budget=64,
                                    max_groups=1024, norm_eps=1e-6, rotary_dim=64, interpret=True)


def _metadata_objects(seed: int = 5):
    """The forward context's GDN metadata as the runner builds it (torch): one object per GDN layer with its own
    state / checkpoint ids and the step's shared read offsets / query starts / seq lens (the same tensor objects)."""
    import torch

    rows = 48
    off, qsl, seq, sts, cks = _meta(N_GDN, B // T_VERIFY, rows, seed)
    shared = dict(mamba_slot_read_offsets=torch.from_numpy(off), query_start_loc=torch.from_numpy(qsl),
                  seq_lens=torch.from_numpy(seq))
    gdn_ids = [i for i in range(L) if i % 4 != 3]
    layers = [SimpleNamespace(linear_attn=SimpleNamespace(prefix=f"model.layers.{i}.linear_attn")) for i in range(L)]
    md = {
        layers[i].linear_attn.prefix: SimpleNamespace(
            mamba_state_indices=torch.from_numpy(sts[k]).to(torch.int64),  # the module converts to int32
            mamba_ckpt_indices=torch.from_numpy(cks[k]),
            **shared,
        )
        for k, i in enumerate(gdn_ids)
    }
    return md, layers, gdn_ids


def _module(layers, gdn_ids, shared=True):
    from vllm_torchtpu.megakernel_runtime.qwen38_wholestep import Qwen38WholeStep

    m = SimpleNamespace(layers=layers, gdn_ids=gdn_ids, gdn_shared=shared,
                        GDN_SHARED_FIELDS=Qwen38WholeStep.GDN_SHARED_FIELDS)
    m._gdn_meta_shared = lambda md: Qwen38WholeStep._gdn_meta_shared(m, md)
    m.check_gdn_meta_shared = lambda: Qwen38WholeStep.check_gdn_meta_shared(m)
    return m


def _per_layer_operands(md, layers, gdn_ids):
    """The flag-off module's operand list (qwen38_wholestep.Qwen38WholeStep.forward's loop)."""
    import torch

    meta = []
    for i in gdn_ids:
        m = md[layers[i].linear_attn.prefix]
        meta += [m.mamba_state_indices.to(torch.int32), m.mamba_slot_read_offsets.to(torch.int32),
                 m.mamba_ckpt_indices.to(torch.int32), m.query_start_loc.to(torch.int32), m.seq_lens.to(torch.int32)]
    return meta


def test_whole_call_shared_matches_per_layer(monkeypatch):
    """`call_shared` on the module's shared operand list == `call` on its per-layer list, bit for bit."""
    _stub(monkeypatch)
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    monkeypatch.delenv(FLAG, raising=False)
    call0, specs0, outs0, don0 = _build(mesh)
    monkeypatch.setenv(FLAG, "1")
    call1, specs1, outs1, don1 = _build(mesh)
    assert call0.__name__ == "call" and call1.__name__ == "call_shared"
    n_w, k = 1, 2 + 1
    assert len(specs0) == k + 5 * N_GDN + 3 * N_ATT + 4 + 2 * N_ATT
    assert specs1[:k] == specs0[:k] and specs1[k + GDN_SHARED + 2 * N_GDN:] == specs0[k + 5 * N_GDN:]
    assert specs1[k: k + GDN_SHARED + 2 * N_GDN] == (P(),) * (GDN_SHARED + 2 * N_GDN)
    assert len(specs1) == len(specs0) - 5 * N_GDN + GDN_SHARED + 2 * N_GDN
    assert don0 == don1 == (1 + n_w,) and outs0 == outs1

    md, layers, gdn_ids = _metadata_objects()
    per_layer = [np.asarray(t.numpy()) for t in _per_layer_operands(md, layers, gdn_ids)]
    shared = [np.asarray(t.numpy()) for t in _module(layers, gdn_ids)._gdn_meta_shared(md)]
    rng = np.random.default_rng(11)
    hidden = rng.standard_normal((R * B, H)).astype(np.float32)
    w0 = rng.standard_normal((R * 4, H)).astype(np.float32)
    pool = np.zeros((R * POOL,), np.int32)
    tabs = [rng.integers(0, 50, (48 * n,)).astype(np.int32) for _ in range(N_ATT) for n in (8, 32, 32)]
    positions = rng.integers(0, 300, (B,)).astype(np.int32)
    starts = np.arange(B + 1, dtype=np.int32)
    context = rng.integers(0, 300, (B,)).astype(np.int32)
    trig = rng.standard_normal((16, 128)).astype(np.float32)
    norms = [rng.standard_normal((n,)).astype(np.float32) for _ in range(N_ATT) for n in (256, 128)]
    tail = [*tabs, positions, starts, context, trig, *norms]
    res0, pool0 = jax.jit(call0)(hidden, w0, pool, *per_layer, *tail)
    res1, pool1 = jax.jit(call1)(hidden, w0, pool, *shared, *tail)
    np.testing.assert_array_equal(np.asarray(res1), np.asarray(res0))
    np.testing.assert_array_equal(np.asarray(pool1), np.asarray(pool0))
    # Not vacuous: the pool holds the per-layer rb / hi / wb of `gdn_blocks` on the per-layer operands.
    NS = B // T_VERIFY
    rb, wb, hi = (np.asarray(x) for x in jax.jit(lambda *m: gdn_blocks(m, NS, True))(*per_layer))
    p0 = np.asarray(pool0)[:POOL]
    n = N_GDN * NS
    np.testing.assert_array_equal(p0[:n].reshape(N_GDN, NS), rb)
    np.testing.assert_array_equal(p0[n: 2 * n].reshape(N_GDN, NS), hi.astype(np.int32))
    np.testing.assert_array_equal(p0[2 * n: 2 * n + n * T_VERIFY].reshape(N_GDN, NS, T_VERIFY), wb)
    assert (rb != 0).any() and not (rb == rb[:1]).all() and hi.any() and not hi.all()


def test_module_shared_operand_order():
    """`_gdn_meta_shared`: off / qsl / seq of the first layer, then the state ids, then the checkpoint ids, int32."""
    import torch

    md, layers, gdn_ids = _metadata_objects(seed=9)
    meta = _module(layers, gdn_ids)._gdn_meta_shared(md)
    m0 = md[layers[gdn_ids[0]].linear_attn.prefix]
    assert len(meta) == GDN_SHARED + 2 * len(gdn_ids)
    assert all(t.dtype == torch.int32 for t in meta)
    for t, f in zip(meta[:GDN_SHARED], ("mamba_slot_read_offsets", "query_start_loc", "seq_lens")):
        assert torch.equal(t, getattr(m0, f).to(torch.int32)), f
    for k, i in enumerate(gdn_ids):
        m = md[layers[i].linear_attn.prefix]
        assert torch.equal(meta[GDN_SHARED + k], m.mamba_state_indices.to(torch.int32)), i
        assert torch.equal(meta[GDN_SHARED + len(gdn_ids) + k], m.mamba_ckpt_indices.to(torch.int32)), i


def _forward_context(monkeypatch, md):
    try:
        import vllm.forward_context as fc
    except Exception:  # noqa: BLE001  (no vLLM, or one that cannot import here): a bare module
        fc = types.ModuleType("vllm.forward_context")
        monkeypatch.setitem(sys.modules, "vllm.forward_context", fc)
    monkeypatch.setattr(fc, "get_forward_context", lambda: SimpleNamespace(attn_metadata=md), raising=False)


@pytest.mark.parametrize("field", ["mamba_slot_read_offsets", "query_start_loc", "seq_lens"])
def test_host_check_refuses_a_layer_with_its_own_step_tensor(monkeypatch, field):
    md, layers, gdn_ids = _metadata_objects(seed=13)
    _forward_context(monkeypatch, md)
    _module(layers, gdn_ids).check_gdn_meta_shared()  # shared: passes
    p = layers[gdn_ids[3]].linear_attn.prefix
    md[p] = SimpleNamespace(**{**vars(md[p]), field: getattr(md[p], field).clone()})  # equal values, own tensor
    with pytest.raises(RuntimeError, match=f"has its own {field}"):
        _module(layers, gdn_ids).check_gdn_meta_shared()
    _module(layers, gdn_ids, shared=False).check_gdn_meta_shared()  # flag off: no check


def test_flag_default_off_and_fixed_between_bind_and_build(monkeypatch):
    from vllm_torchtpu.megakernel_runtime import qwen38_wholestep as ws

    monkeypatch.delenv(FLAG, raising=False)
    assert wsc.gdn_meta_shared() is False
    kw = dict(budget=512, max_groups=8192, norm_eps=1e-6, rotary_dim=64, write=True)
    kind = ws._WholeKind(SimpleNamespace(size=R), None, None, [None] * N_ATT, tag="l1d07b1_off", **kw)
    assert kind.gdn_shared is False
    monkeypatch.setenv(FLAG, "1")
    assert wsc.gdn_meta_shared() is True
    with pytest.raises(AssertionError, match="changed after bind"):
        kind.callable_for(B)
    kind_on = ws._WholeKind(SimpleNamespace(size=R), None, None, [None] * N_ATT, tag="l1d07b1_on", **kw)
    assert kind_on.gdn_shared is True
