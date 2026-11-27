# Copyright 2026 Google LLC
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
"""Model-free unit tests for `cache.py` (`CachePipe`, `PallasWriter`, flat step)."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import numpy as np
import pytest
from jax._src import dispatch
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

import vllm_torchtpu.megakernel.core as mk
from tests.megakernel.core.megakernel_fakes import TINY_TILES
from tests.megakernel.core.megakernel_synth import synth_hybrid_spec
from vllm_torchtpu.megakernel.core import (
    _builtins,
    cache_common,
    cache_flat,
    cache_hlo,
    fused_io_hook,
    model_def,
    planner,
    pool,
)
from vllm_torchtpu.megakernel.core import kernel as mk_kernel
from vllm_torchtpu.megakernel.core.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.core.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.core.modules.gdn import GDNModule
from vllm_torchtpu.megakernel.core.ops import F32
from vllm_torchtpu.megakernel.core.pool.config import CacheExec, cache_exec
from vllm_torchtpu.megakernel.core.reference import spec_reference_step
from vllm_torchtpu.megakernel.core.spec import ModelSpec
from vllm_torchtpu.megakernel.core.weights.spec_inputs import (
    pack_spec_inputs,
    random_spec_inputs,
)

R = 2


def _make_paged_ex(B: int, T: int, R: int = R, *, gpr: Any = False, **cx: Any):
    tiles = TINY_TILES.with_modules({"gqa": {"kv_blk": 128, "nkv": 2}})
    return ExecSpec(
        B=B,
        T=T,
        R=R,
        ctx=256,
        stack_e=True,
        tiles=tiles,
        interpret=True,
        cache=CacheExec(paged_kv=True, num_pages=64, pool_gather=gpr, **cx),
    )


def _f32_conv(spec: Any) -> Any:
    """`spec` with an f32-conv pool backend (same callbacks)."""
    lay_f32 = dataclasses.replace(spec.cache_backend.layout, conv_dtype=F32)
    pb_f32 = pool.make_pool_backend(lay_f32, rec_chunk=GDNModule.chunk_rows)
    return dataclasses.replace(spec, cache_backend=pb_f32)


class _Case:
    """Packed inputs and fresh sharded pools for one `(spec, ex)` step."""

    def __init__(self, spec: Any, ex: ExecSpec, seed: int = 0):
        self.spec, self.ex = spec, ex
        self.lay = spec.cache_backend.layout
        self.mesh = Mesh(np.array(jax.devices("cpu")[: ex.R]), ("tp",))
        self.rep = NamedSharding(self.mesh, P())
        self.tp = NamedSharding(self.mesh, P("tp"))
        key = jax.random.PRNGKey(seed)
        self.pos, self.hid, self.w_log, self.st_log = random_spec_inputs(key, spec, ex)
        self.pos_mk, self.hid_mk, self.w_mk, _ = pack_spec_inputs(
            self.pos, self.hid, self.w_log, self.st_log, spec, ex, mesh=self.mesh
        )
        self.pt, self.ckpt, self.roff, _ = pool.allocate_test_layout(self.lay, ex)
        self.raw = pool.states_to_pool(
            self.lay, ex, self.st_log, self.pt, self.ckpt, read_offsets=self.roff
        )
        self.w_flat = [self.w_mk[k] for k in planner.plan(spec, ex).w_names]

    def pools(self, six_d: bool = False) -> tuple[jax.Array, ...]:
        return tuple(
            jax.device_put(
                np.asarray(p).reshape(self.ex.R, -1, *p.shape[1:])
                if six_d
                else np.asarray(p).copy(),
                self.tp,
            )
            for p in self.raw
        )

    def meta(self, pt: Any = None, ckpt: Any = None, roff: Any = None) -> list[Any]:
        return [
            jax.device_put(self.pt if pt is None else pt, self.rep),
            jax.device_put(self.ckpt if ckpt is None else ckpt, self.rep),
            jax.device_put(self.roff if roff is None else roff, self.rep),
        ]

    def flat_args(self, pools: Any, *meta: Any) -> list[Any]:
        return [self.pos_mk, self.hid_mk, *self.w_flat, *pools, *meta]


@pytest.mark.parametrize(
    ("T", "B", "gpr", "conv_f32"),
    [
        (1, 4, False, False),
        (1, 4, False, True),
        (4, 4, False, False),
        (4, 4, "eligible", False),
    ],
    ids=["decode_bf16", "decode_f32_conv", "verify_split", "verify_gpr"],
)
def test_flat_pooled_step_matches_reference(T: int, B: int, gpr: Any, conv_f32: bool):
    """`make_flat_pooled_step` matches the oracle (hidden + pools)."""
    spec = synth_hybrid_spec(L=2, att_every=2, R=R)
    spec = _f32_conv(spec) if conv_f32 else spec
    ex = _make_paged_ex(B=B, T=T, gpr=gpr)
    c = _Case(spec, ex, seed=20 + T)

    built = cache_common.build_pooled(
        c.mesh,
        spec,
        ex,
        donate_pools=False,
        fused_pool_io=None,
        max_b_low=0,
        min_b_hi=0,
    )
    assert built.use_gpr is bool(gpr)
    flat_fn, _, _, donated = cache_flat.make_flat_pooled_step(
        c.mesh, spec, ex, donate_pools=False
    )
    assert donated == ()
    flat_outs = flat_fn(*c.flat_args(c.pools(), *c.meta()))
    out_hid_p, out_raw_p = flat_outs[0], flat_outs[1:]

    ref_hid, ref_st = spec_reference_step(
        spec,
        spec.config,
        ex,
        c.pos,
        c.hid,
        c.w_log,
        c.st_log,
        paged_kv=True,
        pool_io=not conv_f32,
    )
    out_hid_r = out_hid_p.reshape(R, ex.M, spec.config.H)[:, :B]
    np.testing.assert_allclose(
        out_hid_r[0].astype(F32), ref_hid.astype(F32), rtol=3e-2, atol=3e-2
    )
    st_p = pool.pool_to_states(c.lay, ex, out_raw_p, c.pt, c.ckpt)
    tol = dict(rtol=3e-2, atol=3e-2)
    np.testing.assert_allclose(
        st_p["rec"].astype(F32),
        ref_st["rec_ckpt" if T > 1 else "rec"].astype(F32),
        **tol,
    )
    taps, qkv = c.lay.conv_taps, st_p["conv"].shape[-1]
    if T == 1:
        ref_conv = (
            ref_st["conv"][:, :, : taps * B].reshape(R, -1, taps, B, qkv).swapaxes(2, 3)
        )
        np.testing.assert_allclose(st_p["conv"][:, :, :, 0], ref_conv, **tol)
    else:
        for t in range(T):
            ref_conv_t = ref_st["conv_ckpt"][:, :, :, 1 + t : 1 + t + taps]
            np.testing.assert_allclose(st_p["conv"][:, :, :, t], ref_conv_t, **tol)
    for b in range(B):
        s, p_b = b // T, int(c.pos[b])
        for k in ("kc", "vc"):
            got = np.asarray(st_p[k][:, :, s, p_b]).astype(np.float32)
            want = np.asarray(ref_st[k][:, :, s, p_b]).astype(np.float32)
            np.testing.assert_allclose(got, want, rtol=0.15, atol=0.15)


def _compiled(fn: Any, *args: Any) -> str:
    return fn.lower(*args).compile().as_text()


def test_check_pooled_step_hlo_on_compiled_steps():
    """`check_pooled_step_hlo` passes donated compiled steps and catches copies."""
    spec = synth_hybrid_spec(L=2, att_every=2, R=R)
    c = _Case(spec, _make_paged_ex(B=4, T=1), seed=9)
    lay = spec.cache_backend.layout
    for donate in (True, False):
        flat_fn, *_ = cache_flat.make_flat_pooled_step(
            c.mesh, spec, c.ex, donate_pools=donate
        )
        hlo = _compiled(flat_fn, *c.flat_args(c.pools(), *c.meta()))
        ok, msg = cache_hlo.check_pooled_step_hlo(hlo, layout=lay)
        assert ok is donate, msg
        assert donate or "full-pool" in msg
        assert ("input_output_alias" in hlo) is donate
    lowered = flat_fn.lower(*c.flat_args(c.pools(), *c.meta()))
    ok_mlir, msg_mlir = cache_hlo.check_pooled_step_hlo(lowered.as_text(), layout=lay)
    assert not ok_mlir and "compile().as_text()" in msg_mlir


def test_flat_pooled_step_keeps_6d_pools_and_donation():
    """6D `(R, num_pages, ...)` pools come back 6D and stay donated."""
    spec = synth_hybrid_spec(L=2, att_every=2, R=R)
    c = _Case(spec, _make_paged_ex(B=4, T=1), seed=3)
    flat_fn, _, _, donated = cache_flat.make_flat_pooled_step(c.mesh, spec, c.ex)
    assert donated
    outs_5d = flat_fn(*c.flat_args(c.pools(), *c.meta()))
    pools_6d = c.pools(six_d=True)
    shapes_6d = [p.shape for p in pools_6d]
    hlo = _compiled(flat_fn, *c.flat_args(pools_6d, *c.meta()))
    assert "input_output_alias" in hlo
    ok, msg = cache_hlo.check_pooled_step_hlo(hlo, layout=spec.cache_backend.layout)
    assert ok, msg
    outs_6d = flat_fn(*c.flat_args(pools_6d, *c.meta()))
    assert [o.shape for o in outs_6d[1:]] == shapes_6d
    assert all(p.is_deleted() for p in pools_6d)
    np.testing.assert_array_equal(outs_6d[0], outs_5d[0])
    for got, want in zip(outs_6d[1:], outs_5d[1:], strict=True):
        np.testing.assert_array_equal(
            np.asarray(got).reshape(want.shape).view(np.uint8),
            np.asarray(want).view(np.uint8),
        )


def test_flat_pooled_step_admits_only_replicated_metadata():
    """The flat step rejects rank-folded / narrow metadata instead of misreading it."""
    spec = synth_hybrid_spec(L=2, att_every=2, R=R)
    c = _Case(spec, _make_paged_ex(B=4, T=1), seed=4)
    flat_fn, *_ = cache_flat.make_flat_pooled_step(
        c.mesh, spec, c.ex, donate_pools=False
    )
    pt_ok = np.asarray(c.pt)
    flat_fn(*c.flat_args(c.pools(), *c.meta(pt=pt_ok[None])))  # (1, NS, K)
    bad_meta = [
        ("page_table", dict(pt=np.concatenate([pt_ok] * R))),  # (R * NS, K)
        ("page_table", dict(pt=np.stack([pt_ok] * R))),  # (R, NS, K)
        ("page_table", dict(pt=pt_ok[:, :1])),  # K < max_pages_per_seq
        ("ckpt_indices", dict(ckpt=np.stack([np.asarray(c.ckpt)] * R))),  # (R, ...)
        ("read_offsets", dict(roff=np.stack([np.asarray(c.roff)] * R))),  # (R, ...)
    ]
    for match, kw in bad_meta:
        with pytest.raises(ValueError, match=match):
            flat_fn(*c.flat_args(c.pools(), *c.meta(**kw)))


def _recording_backend(spec: Any, seen: list[Any]) -> Any:
    """`spec` whose scatter callback records the `active_seqs` it receives."""
    pb = spec.cache_backend

    def scatter(ex: Any, raw: Any, pos: Any, pt: Any, out: Any, ck: Any, act: Any):
        seen.append(act)
        return pb.scatter_states(ex, raw, pos, pt, out, ck, act)

    return dataclasses.replace(
        spec, cache_backend=dataclasses.replace(pb, scatter_states=scatter)
    )


@pytest.mark.parametrize(("T", "gpr"), [(1, False), (4, "eligible")])
def test_pooled_steps_use_cache_exec_act_mode(T: int, gpr: Any):
    """`CacheExec.act_mode` reaches the pool kernels; `"ref"` fails closed."""
    seen: list[Any] = []
    spec = _recording_backend(synth_hybrid_spec(L=2, att_every=2, R=R), seen)
    ex = _make_paged_ex(B=4, T=T, gpr=gpr, act_mode="all")
    c = _Case(spec, ex, seed=5)
    flat_fn, *_ = cache_flat.make_flat_pooled_step(c.mesh, spec, ex, donate_pools=False)
    flat_fn.lower(*c.flat_args(c.pools(), *c.meta()))
    assert seen == ["all"]
    ex_ref = _make_paged_ex(B=4, T=T, gpr=gpr, act_mode="ref")
    with pytest.raises(MegakernelConfigError, match="act_mode='ref'"):
        cache_flat.make_flat_pooled_step(c.mesh, spec, ex_ref)


def test_flat_pooled_step_rejects_ref_before_building(monkeypatch):
    """`act_mode="ref"` raises before any kernel build, KV-only specs included
    (they used to be remapped to `"all"`)."""

    def _no_build(ms: Any, ex_k: ExecSpec) -> Any:
        raise AssertionError(f"built a kernel for act_mode={cache_exec(ex_k).act_mode}")

    monkeypatch.setattr(mk_kernel, "build", _no_build)
    mesh = Mesh(np.array(jax.devices("cpu")[:R]), ("tp",))
    ex_ref = _make_paged_ex(B=4, T=1, act_mode="ref")
    for att_every in (1, 2):  # KV-only, hybrid
        spec = synth_hybrid_spec(L=2, att_every=att_every, R=R)
        assert spec.cache_backend.has_recurrent_pool is (att_every > 1)
        with pytest.raises(MegakernelConfigError, match="act_mode='ref'"):
            cache_flat.make_flat_pooled_step(mesh, spec, ex_ref)


def _expect_step_error(c: _Case, step: Any, meta: list[Any], match: str) -> None:
    """`step` raises `match` on `meta`, then still runs on `c`'s metadata."""
    try:
        with pytest.raises((ValueError, jax.errors.JaxRuntimeError), match=match):
            jax.block_until_ready(step(*c.flat_args(c.pools(), *meta)))
    finally:
        # Reset even if the assertion failed. The failed metadata callback's
        # error stays in JAX's effect tokens (private `dispatch` state), and
        # every later computation with effects would re-raise it. A kernel
        # interpreted while the callback failed leaves TPU interpret mode's
        # global shared memory uncleaned, which later kernels would trip on.
        dispatch.runtime_tokens.clear()
        pltpu.reset_tpu_interpret_mode_state()
    jax.block_until_ready(step(*c.flat_args(c.pools(), *c.meta())))


def test_flat_pooled_step_checks_ckpt_bounds_in_gpr_path():
    """With `check_ckpt_bounds`, the GPR flat step (which skips the gather
    kernel) validates its read offsets and admits only `ckpt_indices` with
    `K >= T`."""
    T = 4
    spec = synth_hybrid_spec(L=2, att_every=2, R=R, VH=4)
    ex = _make_paged_ex(B=4, T=T, gpr="eligible", check_ckpt_bounds=True)
    c = _Case(spec, ex, seed=6)
    flat_fn, *_ = cache_flat.make_flat_pooled_step(
        c.mesh, spec, ex, fused_pool_io=False, donate_pools=True
    )
    jax.block_until_ready(flat_fn(*c.flat_args(c.pools(), *c.meta())))
    roff_k = np.full_like(np.asarray(c.roff), T)  # == K
    _expect_step_error(c, flat_fn, c.meta(roff=roff_k), "read_offset")
    narrow = c.meta(ckpt=np.asarray(c.ckpt)[..., :1])  # trace-time shape check
    with pytest.raises(ValueError, match="ckpt_indices last dim"):
        flat_fn(*c.flat_args(c.pools(), *narrow))


def test_flat_pooled_step_all_mode_checks_every_sequence():
    """Under `act_mode="all"` the metadata check covers a sequence whose
    checkpoint column 0 is 0 (inactive under `"from_ckpt"`, active here)."""
    T = 4
    spec = synth_hybrid_spec(L=2, att_every=2, R=R, VH=4)
    ex = _make_paged_ex(
        B=4, T=T, gpr="eligible", act_mode="all", check_ckpt_bounds=True
    )
    c = _Case(spec, ex, seed=6)
    flat_fn, *_ = cache_flat.make_flat_pooled_step(c.mesh, spec, ex, donate_pools=True)
    jax.block_until_ready(flat_fn(*c.flat_args(c.pools(), *c.meta())))
    ckpt_bad = np.asarray(c.ckpt).copy()
    ckpt_bad[:, 0, 0] = 0  # seq 0 writes checkpoint slot 0 (the reserved pages)
    _expect_step_error(c, flat_fn, c.meta(ckpt=ckpt_bad), "distinct positive")


@pytest.fixture
def fused_io_registered(monkeypatch):
    """Mark a fused pool I/O implementation as registered, without loading one.

    For tests that only resolve the pool I/O mode and never run a fused kernel.
    """
    monkeypatch.setattr(_builtins, "_LOADED", set(_builtins.BUILTINS))
    monkeypatch.setattr(fused_io_hook, "_IMPL", object())


@pytest.mark.usefixtures("fused_io_registered")
def test_pooled_steps_validate_only_the_resolved_exec_spec(monkeypatch):
    """A valid `fused_pool_io=False` override is not rejected for the fused mode."""
    # f32 conv, 2048 conv channels per rank: the T=6 fused conv window (256 rows)
    # exceeds one 128-row conv page, while split pool I/O fits.
    spec = _f32_conv(synth_hybrid_spec(L=2, att_every=2, R=R, KH=8, VH=16))
    lay = spec.cache_backend.layout
    assert lay.fused_conv_rows(6, R) > lay.conv_page_rows
    ex = _make_paged_ex(B=6, T=6, fused_pool_io=True)
    mesh = Mesh(np.array(jax.devices("cpu")[:R]), ("tp",))
    with pytest.raises(MegakernelConfigError, match="conv window"):
        cache_flat.make_flat_pooled_step(mesh, spec, ex)

    class _Built(Exception):
        pass

    built: list[ExecSpec] = []

    def _build(ms: Any, ex_k: ExecSpec) -> Any:
        # This geometry is too wide for a CPU kernel build; stop once the
        # pool I/O mode resolved and validated.
        built.append(ex_k)
        raise _Built

    monkeypatch.setattr(mk_kernel, "build", _build)
    with pytest.raises(_Built):
        cache_flat.make_flat_pooled_step(mesh, spec, ex, fused_pool_io=False)
    assert [cache_exec(e).fused_pool_io for e in built] == [False]


@pytest.mark.usefixtures("fused_io_registered")
def test_should_fuse_pool_io_requires_the_fused_conv_window_to_fit():
    """`"auto"` falls back to split when the fused conv window misses one page."""
    spec = _f32_conv(synth_hybrid_spec(L=2, att_every=2, R=R, KH=8, VH=16))
    ex_fit, ex_big = _make_paged_ex(B=4, T=1), _make_paged_ex(B=6, T=6)
    assert cache_common.should_fuse_pool_io(spec, ex_fit, fused_pool_io="auto") is True
    assert cache_common.should_fuse_pool_io(spec, ex_big, fused_pool_io="auto") is False
    with pytest.raises(MegakernelConfigError, match="conv window"):
        cache_common.should_fuse_pool_io(spec, ex_big, fused_pool_io=True)
    with pytest.raises(ValueError, match="'auto'"):
        cache_common.should_fuse_pool_io(spec, ex_fit, fused_pool_io="sometimes")


class _TinyHybridModel(mk.ModelDef):
    """Unregistered tiny hybrid `ModelDef` for `resolve_model_spec`."""

    name = "cache_pipes_tiny_hybrid"
    supported_tp_sizes = (2,)
    default_hf_config = {
        "n_vocab": 256,
        "n_layers": 2,
        "d_model": 256,
        "n_heads": 4,
        "n_kv_heads": 2,
        "d_head": 256,
        "theta": 10000.0,
        "norm_eps": 1e-6,
        "ssm_k_heads": 2,
        "ssm_v_heads": 4,
        "ssm_k_dim": 128,
        "ssm_v_dim": 128,
        "ssm_conv_kernel": 4,
        "n_experts": 128,
        "top_k": 2,
        "d_ff": 256,
    }
    geometry = mk.FromHF(
        hidden="d_model",
        layers="n_layers",
        vocab="n_vocab",
        eps="norm_eps",
        rope_theta="theta",
        rope_partial=0.25,
    )
    attn = mk.GQA.bind(
        prefix="self_attn",
        heads="n_heads",
        kv_heads="n_kv_heads",
        head_dim="d_head",
        qk_norm=True,
        output_gate=True,
    )
    linear = mk.GDN.bind(
        prefix="linear_attn",
        k_heads="ssm_k_heads",
        v_heads="ssm_v_heads",
        k_dim="ssm_k_dim",
        v_dim="ssm_v_dim",
        conv_kernel="ssm_conv_kernel",
    )
    moe = mk.MoE.bind(
        prefix="mlp",
        experts="n_experts",
        top_k="top_k",
        inter="d_ff",
        shared_inter="d_ff",
        shared_gate=True,
    )
    schedule = mk.Schedule(
        mixer=mk.Every(n=2, last=attn, rest=linear),
        ffn=moe,
        pre_norm=mk.Norm.bind("input_layernorm", zero_centered=True),
        post_norm=mk.Norm.bind("post_attention_layernorm", zero_centered=True),
    )
    embed = mk.Embedding("model.embed_tokens")
    final_norm = mk.Norm.bind("model.norm", zero_centered=True)
    lm_head = mk.LMHead("lm_head")


def test_resolve_model_spec_dispatch(monkeypatch):
    """Specs pass through; a `ModelDef` and a geometry derived from one resolve
    to that `ModelDef`'s spec; model names and other inputs raise TypeError."""
    spec = synth_hybrid_spec(L=2, att_every=2, R=R)
    assert cache_common.resolve_model_spec(spec) is spec
    m_def = _TinyHybridModel
    g_default = m_def.derive_geometry()
    g_l4 = m_def.derive_geometry(dict(m_def.default_hf_config, n_layers=4))
    assert model_def.find_model(g_l4) is m_def and (g_default.L, g_l4.L) == (2, 4)
    for target, geom in ((m_def, g_default), (g_l4, g_l4)):
        got = cache_common.resolve_model_spec(target)
        want = m_def.to_model_spec(geom)
        assert isinstance(got, ModelSpec) and got.name == want.name
        assert got.config == geom
        assert got.cache_backend.layout == want.cache_backend.layout
    # A registered name resolves to a `ModelDef` but is still rejected.
    monkeypatch.setitem(model_def._MODEL_REGISTRY, m_def.name, m_def)
    assert model_def.find_model(m_def.name) is m_def
    for bad in (object(), spec.config, "synth", m_def.name):
        with pytest.raises(TypeError, match="ModelSpec"):
            cache_common.resolve_model_spec(bad)


def test_should_fuse_pool_io_without_registered_hook(monkeypatch):
    """`should_fuse_pool_io` checks `fused_io_available` for `auto` and `True`."""
    spec = synth_hybrid_spec(L=2, att_every=2, R=2)
    ex = _make_paged_ex(B=8, T=1, R=2)
    monkeypatch.setattr(_builtins, "_LOADED", set(_builtins.BUILTINS))
    monkeypatch.setattr(fused_io_hook, "_IMPL", None)
    assert not fused_io_hook.fused_io_available()
    assert cache_common.should_fuse_pool_io(spec, ex, fused_pool_io="auto") is False
    assert cache_common.should_fuse_pool_io(spec, ex, fused_pool_io=False) is False
    with pytest.raises(MegakernelConfigError, match="fused pool I/O"):
        cache_common.should_fuse_pool_io(spec, ex, fused_pool_io=True)
