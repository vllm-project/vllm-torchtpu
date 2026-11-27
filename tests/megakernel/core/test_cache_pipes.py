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
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import dispatch
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

import vllm_torchtpu.megakernel.core as mk
from tests.megakernel.core.megakernel_fakes import TINY_TILES, make_fake_model_spec
from tests.megakernel.core.megakernel_synth import synth_hybrid_spec
from vllm_torchtpu.megakernel.core import (
    _builtins,
    cache,
    cache_common,
    cache_experimental,
    decls,
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
        self.mesh = Mesh(np.array(jax.devices("cpu")[: ex.R]), ("tp",))
        self.rep = NamedSharding(self.mesh, P())
        self.tp = NamedSharding(self.mesh, P("tp"))
        key = jax.random.PRNGKey(seed)
        self.pos, self.hid, self.w_log, self.st_log = random_spec_inputs(key, spec, ex)
        self.pos_mk, self.hid_mk, self.w_mk, _ = pack_spec_inputs(
            self.pos, self.hid, self.w_log, self.st_log, spec, ex, mesh=self.mesh
        )
        self.pt, self.ckpt, self.roff, _ = cache.default_pool_layout(spec, ex)
        self.raw = cache.states_to_pool(
            spec, ex, self.st_log, self.pt, self.ckpt, read_offsets=self.roff
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
def test_pallas_writer_and_flat_pooled_step_match_reference(
    T: int, B: int, gpr: Any, conv_f32: bool
):
    """`PallasWriter` and `make_flat_pooled_step` match the oracle (hidden + pools)."""
    spec = synth_hybrid_spec(L=2, att_every=2, R=R)
    spec = _f32_conv(spec) if conv_f32 else spec
    ex = _make_paged_ex(B=B, T=T, gpr=gpr)
    c = _Case(spec, ex, seed=20 + T)

    pooled_step = cache.PallasWriter().make_step(c.mesh, spec, ex)
    assert pooled_step.info["use_gpr"] is bool(gpr)
    out_hid_p, out_raw_p = pooled_step(c.pos_mk, c.hid_mk, c.w_mk, c.pools(), *c.meta())

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
    st_p = cache.pool_to_states(spec, ex, out_raw_p, c.pt, c.ckpt)
    tol = dict(rtol=3e-2, atol=3e-2)
    np.testing.assert_allclose(
        st_p["rec"].astype(F32),
        ref_st["rec_ckpt" if T > 1 else "rec"].astype(F32),
        **tol,
    )
    taps, qkv = spec.cache_backend.layout.conv_taps, st_p["conv"].shape[-1]
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

    # The flat step shares the per-rank body: bit-identical hidden and pools.
    flat_fn, _, _, donated = cache.make_flat_pooled_step(
        c.mesh, spec, ex, donate_pools=False
    )
    assert donated == ()
    flat_outs = flat_fn(*c.flat_args(c.pools(), *c.meta()))
    np.testing.assert_array_equal(flat_outs[0], out_hid_p)
    for got, want in zip(flat_outs[1:], out_raw_p, strict=True):
        np.testing.assert_array_equal(
            np.asarray(got).view(np.uint8), np.asarray(want).view(np.uint8)
        )


def _compiled(fn: Any, *args: Any) -> str:
    return fn.lower(*args).compile().as_text()


def test_check_pooled_step_hlo_on_compiled_steps():
    """`check_pooled_step_hlo` passes donated compiled steps and catches copies."""
    spec = synth_hybrid_spec(L=2, att_every=2, R=R)
    c = _Case(spec, _make_paged_ex(B=4, T=1), seed=9)
    lay = spec.cache_backend.layout
    for donate in (True, False):
        pooled = cache.make_pooled_step(c.mesh, spec, c.ex, donate_pools=donate)
        hlo_p = _compiled(pooled, c.pos_mk, c.hid_mk, c.w_mk, c.pools(), *c.meta())
        flat_fn, *_ = cache.make_flat_pooled_step(
            c.mesh, spec, c.ex, donate_pools=donate
        )
        hlo_f = _compiled(flat_fn, *c.flat_args(c.pools(), *c.meta()))
        for hlo in (hlo_p, hlo_f):
            ok, msg = cache.check_pooled_step_hlo(hlo, layout=lay)
            # Without donation XLA copies each pool before the in-place scatter.
            assert ok is donate, msg
            assert donate or "full-pool" in msg
            assert ("input_output_alias" in hlo) is donate
    lowered = flat_fn.lower(*c.flat_args(c.pools(), *c.meta()))
    ok_mlir, msg_mlir = cache.check_pooled_step_hlo(lowered.as_text(), layout=lay)
    assert not ok_mlir and "compile().as_text()" in msg_mlir


def test_flat_pooled_step_keeps_6d_pools_and_donation():
    """6D `(R, num_pages, ...)` pools come back 6D and stay donated."""
    spec = synth_hybrid_spec(L=2, att_every=2, R=R)
    c = _Case(spec, _make_paged_ex(B=4, T=1), seed=3)
    flat_fn, _, _, donated = cache.make_flat_pooled_step(c.mesh, spec, c.ex)
    assert donated
    outs_5d = flat_fn(*c.flat_args(c.pools(), *c.meta()))
    pools_6d = c.pools(six_d=True)
    shapes_6d = [p.shape for p in pools_6d]
    hlo = _compiled(flat_fn, *c.flat_args(pools_6d, *c.meta()))
    assert "input_output_alias" in hlo
    ok, msg = cache.check_pooled_step_hlo(hlo, layout=spec.cache_backend.layout)
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
    flat_fn, *_ = cache.make_flat_pooled_step(c.mesh, spec, c.ex, donate_pools=False)
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
    """`CacheExec.act_mode` reaches the pool kernels of the flat step and of a
    pooled step called without `active_seqs`; `"ref"` without a mask fails
    closed."""
    seen: list[Any] = []
    spec = _recording_backend(synth_hybrid_spec(L=2, att_every=2, R=R), seen)
    ex = _make_paged_ex(B=4, T=T, gpr=gpr, act_mode="all")
    c = _Case(spec, ex, seed=5)
    flat_fn, *_ = cache.make_flat_pooled_step(c.mesh, spec, ex, donate_pools=False)
    flat_fn.lower(*c.flat_args(c.pools(), *c.meta()))
    step = cache.make_pooled_step(c.mesh, spec, ex)
    step.lower(c.pos_mk, c.hid_mk, c.w_mk, c.pools(), *c.meta())
    assert seen == ["all", "all"]
    ex_ref = _make_paged_ex(B=4, T=T, gpr=gpr, act_mode="ref")
    with pytest.raises(MegakernelConfigError, match="act_mode='ref'"):
        cache.make_flat_pooled_step(c.mesh, spec, ex_ref)
    step_ref = cache.make_pooled_step(c.mesh, spec, ex_ref)
    with pytest.raises(MegakernelConfigError, match="act_mode='ref'"):
        step_ref.lower(c.pos_mk, c.hid_mk, c.w_mk, c.pools(), *c.meta())


def test_pooled_step_states_dict_and_mode_strings():
    """The `states`-dict form and string `active_seqs` (`ModeTag` through
    `jax.jit` + `shard_map`) run the requested mode; other strings raise."""
    seen: list[Any] = []
    spec = _recording_backend(synth_hybrid_spec(L=2, att_every=2, R=R), seen)
    c = _Case(spec, _make_paged_ex(B=4, T=1), seed=8)
    step = cache.make_pooled_step(c.mesh, spec, c.ex)
    args = (c.pos_mk, c.hid_mk, c.w_mk)
    hid_ref, raw_ref = step(*args, c.pools(), *c.meta())  # default "from_ckpt"
    pt, ckpt, roff = c.meta()
    pb = spec.cache_backend
    for mode in ("from_ckpt", "all"):  # every sequence is active: same outputs
        states = {
            pool.POOL_STATE_NAME: c.pools(),
            pb.page_table_state_name: pt,
            "ckpt_indices": ckpt,
            "read_offsets": roff,
            "active_seqs": mode,
            "untouched": "kept",
        }
        hid, new_states = step(*args, states)
        assert new_states.keys() == states.keys()
        assert new_states["untouched"] == "kept"
        hid_p, raw_p = step(*args, c.pools(), pt, ckpt, roff, mode)
        for got_hid, got_raw in (
            (hid, new_states[pool.POOL_STATE_NAME]),
            (hid_p, raw_p),
        ):
            np.testing.assert_array_equal(got_hid, hid_ref)
            for got, want in zip(got_raw, raw_ref, strict=True):
                np.testing.assert_array_equal(
                    np.asarray(got).view(np.uint8), np.asarray(want).view(np.uint8)
                )
    # One trace per mode (the dict and positional forms share it).
    assert seen == ["from_ckpt", "all"]
    for states_form in (False, True):
        with pytest.raises(ValueError, match="Unsupported active_seqs mode"):
            if states_form:
                step(*args, {**states, "active_seqs": "ref"})
            else:
                step(*args, c.pools(), pt, ckpt, roff, "ref")


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
            cache.make_flat_pooled_step(mesh, spec, ex_ref)


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
    flat_fn, *_ = cache.make_flat_pooled_step(
        c.mesh, spec, ex, fused_pool_io=False, donate_pools=True
    )
    jax.block_until_ready(flat_fn(*c.flat_args(c.pools(), *c.meta())))
    roff_k = np.full_like(np.asarray(c.roff), T)  # == K
    _expect_step_error(c, flat_fn, c.meta(roff=roff_k), "read_offset")
    narrow = c.meta(ckpt=np.asarray(c.ckpt)[..., :1])  # trace-time shape check
    with pytest.raises(ValueError, match="ckpt_indices last dim"):
        flat_fn(*c.flat_args(c.pools(), *narrow))


def test_flat_pooled_step_checks_ckpt_bounds_in_fused_path():
    """With `check_ckpt_bounds`, the fused flat step (which skips the gather and
    scatter kernels) validates its checkpoint write slots."""
    # Fused GDN I/O needs `2 * kh <= vh` per rank.
    spec = synth_hybrid_spec(L=2, att_every=2, R=R, VH=4)
    ex = _make_paged_ex(B=4, T=1, check_ckpt_bounds=True)
    c = _Case(spec, ex, seed=6)
    flat_fn, *_ = cache.make_flat_pooled_step(
        c.mesh, spec, ex, fused_pool_io=True, donate_pools=True
    )
    jax.block_until_ready(flat_fn(*c.flat_args(c.pools(), *c.meta())))
    ckpt_bad = np.asarray(c.ckpt).copy()
    ckpt_bad[..., -1] = 10_000  # checkpoint write slot past the pool
    _expect_step_error(c, flat_fn, c.meta(ckpt=ckpt_bad), "checkpoint write slot")


def test_flat_pooled_step_all_mode_checks_every_sequence():
    """Under `act_mode="all"` the metadata check covers a sequence whose
    checkpoint column 0 is 0 (inactive under `"from_ckpt"`, active here)."""
    T = 4
    spec = synth_hybrid_spec(L=2, att_every=2, R=R, VH=4)
    ex = _make_paged_ex(
        B=4, T=T, gpr="eligible", act_mode="all", check_ckpt_bounds=True
    )
    c = _Case(spec, ex, seed=6)
    flat_fn, *_ = cache.make_flat_pooled_step(c.mesh, spec, ex, donate_pools=True)
    jax.block_until_ready(flat_fn(*c.flat_args(c.pools(), *c.meta())))
    ckpt_bad = np.asarray(c.ckpt).copy()
    ckpt_bad[:, 0, 0] = 0  # seq 0 writes checkpoint slot 0 (the reserved pages)
    _expect_step_error(c, flat_fn, c.meta(ckpt=ckpt_bad), "distinct positive")


def test_pooled_step_checks_rank_folded_metadata_per_rank():
    """`check_ckpt_bounds` slices `(R * NS, K)` metadata per rank like the step."""
    spec = synth_hybrid_spec(L=2, att_every=2, R=R)
    ex = _make_paged_ex(B=4, T=1, check_ckpt_bounds=True)
    c = _Case(spec, ex, seed=7)
    step = cache.make_pooled_step(c.mesh, spec, ex)
    ckpt_2d, roff_1d = np.asarray(c.ckpt)[0], np.asarray(c.roff)[0]
    hid_ref, raw_ref = step(
        c.pos_mk, c.hid_mk, c.w_mk, c.pools(), c.pt, ckpt_2d, roff_1d
    )
    folded = np.concatenate([ckpt_2d] * R)  # (R * NS, K), sharded over "tp"
    act = np.ones((ex.NS,), np.int32)
    hid, raw = step(c.pos_mk, c.hid_mk, c.w_mk, c.pools(), c.pt, folded, roff_1d, act)
    np.testing.assert_array_equal(hid, hid_ref)
    for got, want in zip(raw, raw_ref, strict=True):
        np.testing.assert_array_equal(
            np.asarray(got).view(np.uint8), np.asarray(want).view(np.uint8)
        )
    bad = folded.copy()
    bad[ex.NS :, 0] = 10_000  # only rank 1's rows are out of bounds
    with pytest.raises(ValueError, match="exceeds pool capacity"):
        step(c.pos_mk, c.hid_mk, c.w_mk, c.pools(), c.pt, bad, roff_1d)


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
        cache.make_flat_pooled_step(mesh, spec, ex)

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
        cache.make_flat_pooled_step(mesh, spec, ex, fused_pool_io=False)
    with pytest.raises(_Built):
        cache.PallasWriter(fused_pool_io=False).make_step(mesh, spec, ex)
    assert [cache_exec(e).fused_pool_io for e in built] == [False, False]


@pytest.mark.usefixtures("fused_io_registered")
def test_should_fuse_pool_io_requires_the_fused_conv_window_to_fit():
    """`"auto"` falls back to split when the fused conv window misses one page."""
    spec = _f32_conv(synth_hybrid_spec(L=2, att_every=2, R=R, KH=8, VH=16))
    ex_fit, ex_big = _make_paged_ex(B=4, T=1), _make_paged_ex(B=6, T=6)
    assert cache.should_fuse_pool_io(spec, ex_fit, fused_pool_io="auto") is True
    assert cache.should_fuse_pool_io(spec, ex_big, fused_pool_io="auto") is False
    with pytest.raises(MegakernelConfigError, match="conv window"):
        cache.should_fuse_pool_io(spec, ex_big, fused_pool_io=True)
    with pytest.raises(ValueError, match="'auto'"):
        cache.should_fuse_pool_io(spec, ex_fit, fused_pool_io="sometimes")


def test_cache_layout_helpers_and_direct_hbm_pipe_on_synth_spec():
    """`cache` pool helpers and `DirectHBM` work directly on `synth_hybrid_spec`."""
    spec = synth_hybrid_spec(L=2, att_every=2, R=R)
    ex = _make_paged_ex(B=4, T=1)
    pt, ckpt, roff, min_pages = cache.default_pool_layout(spec, ex)
    assert pt.shape == (4, 2) and ckpt.shape == (1, 4, 1) and roff.shape == (1, 4)
    assert min_pages <= 64

    _, _, _, st_log = random_spec_inputs(jax.random.PRNGKey(0), spec, ex)
    raw = cache.states_to_pool(spec, ex, st_log, pt, ckpt, read_offsets=roff)
    unpacked = cache.pool_to_states(spec, ex, raw, pt, ckpt)
    np.testing.assert_allclose(
        unpacked["rec"].reshape(st_log["rec"].shape).astype(F32),
        st_log["rec"].astype(F32),
        rtol=0,
        atol=0,
    )
    raw_r0 = tuple(jnp.asarray(r)[: r.shape[0] // R] for r in raw)
    lay = spec.cache_backend.layout
    got = cache.gather_gdn_states(spec, ex, raw_r0, ckpt, roff)
    want = pool.gather_gdn_states(lay, ex, raw_r0, ckpt, roff)
    for g_arr, w_arr in zip(got, want, strict=True):
        np.testing.assert_array_equal(g_arr, w_arr)

    mesh = Mesh(np.array(jax.devices("cpu")[:R]), ("tp",))
    step, info = cache.DirectHBM().make_step(mesh, spec, ex)
    assert callable(step) and not cache_exec(info["exec"]).paged_kv


def test_resolve_pool_layout_from_standalone_model_spec():
    """`cache._resolve_pool_layout` falls back to `ms.cache_backend.layout`."""
    play = pool.derive_pool_layout(
        [decls.KVCacheDecl(hd=256, kvh=1, page_size=128)], R=2
    )
    pb = pool.make_pool_backend(play)
    ms = dataclasses.replace(
        make_fake_model_spec(L=1, att_every=1),
        name="unregistered_standalone_spec",
        cache_backend=pb,
    )
    ex = ExecSpec(B=4, T=1, R=2, ctx=256)
    assert cache._resolve_pool_layout(ms, ex) is play
    pt, ckpt, roff, min_pages = cache.default_pool_layout(ms, ex)
    assert pt.shape == (4, 2) and ckpt.shape == (1, 4, 1) and roff.shape == (1, 4)
    assert min_pages >= 8
    with pytest.raises(MegakernelConfigError, match="no cache_backend"):
        cache._resolve_pool_layout(dataclasses.replace(ms, cache_backend=None), ex)


def test_resolve_pool_layout_prefers_the_specs_own_cache_backend(monkeypatch):
    """A registered model's spec with a replaced `cache_backend` uses that layout."""
    spec = synth_hybrid_spec(L=2, att_every=2, R=R)
    ex = _make_paged_ex(B=4, T=1)
    registry_lay = spec.cache_backend.layout

    class _RegisteredModel:  # stands in for the `ModelDef` registered as spec.name
        @staticmethod
        def pool_layout(cfg: Any, *, R: int) -> pool.PoolLayout:
            return registry_lay

    monkeypatch.setitem(model_def._MODEL_REGISTRY, spec.name, _RegisteredModel)
    spec_f32 = _f32_conv(spec)
    lay_f32 = spec_f32.cache_backend.layout
    assert cache._resolve_pool_layout(spec_f32, ex) is lay_f32
    pt, ckpt, roff, _ = cache.default_pool_layout(spec_f32, ex)
    _, _, _, st_log = random_spec_inputs(jax.random.PRNGKey(1), spec_f32, ex)
    raw = cache.states_to_pool(spec_f32, ex, st_log, pt, ckpt, read_offsets=roff)
    want = pool.states_to_pool(lay_f32, ex, st_log, pt, ckpt, read_offsets=roff)
    for got_p, want_p in zip(raw, want, strict=True):
        np.testing.assert_array_equal(
            np.asarray(got_p).view(np.uint8), np.asarray(want_p).view(np.uint8)
        )


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


def test_should_fuse_pool_io_with_registered_hook():
    """With the fused implementation registered, `auto` and `True` fuse."""
    spec = synth_hybrid_spec(L=2, att_every=2, R=2)
    ex = _make_paged_ex(B=8, T=1, R=2)
    assert fused_io_hook.fused_io_available()
    assert cache.should_fuse_pool_io(spec, ex, fused_pool_io="auto") is True
    assert cache.should_fuse_pool_io(spec, ex, fused_pool_io=True) is True


def test_should_fuse_pool_io_without_registered_hook(monkeypatch):
    """`should_fuse_pool_io` checks `fused_io_available` for `auto` and `True`."""
    spec = synth_hybrid_spec(L=2, att_every=2, R=2)
    ex = _make_paged_ex(B=8, T=1, R=2)
    monkeypatch.setattr(_builtins, "_LOADED", set(_builtins.BUILTINS))
    monkeypatch.setattr(fused_io_hook, "_IMPL", None)
    assert not fused_io_hook.fused_io_available()
    assert cache.should_fuse_pool_io(spec, ex, fused_pool_io="auto") is False
    assert cache.should_fuse_pool_io(spec, ex, fused_pool_io=False) is False
    with pytest.raises(MegakernelConfigError, match="fused pool I/O"):
        cache.should_fuse_pool_io(spec, ex, fused_pool_io=True)


def test_cache_experimental_wrappers_build_fused_steps():
    """The experimental wrappers build fused steps and keep the donation guard."""
    spec = synth_hybrid_spec(L=2, att_every=2, R=R, VH=4)
    ex = _make_paged_ex(B=4, T=1)
    mesh = Mesh(np.array(jax.devices("cpu")[:R]), ("tp",))
    step = cache_experimental.make_fused_pooled_step(mesh, spec, ex)
    assert step.info["use_fused"] is True
    with pytest.raises(MegakernelConfigError, match="donate_pools=True"):
        cache_experimental.make_fused_pooled_step(mesh, spec, ex, donate_pools=False)
    _, _, _, donated = cache_experimental.make_flat_fused_pooled_step(mesh, spec, ex)
    assert donated
    with pytest.raises(MegakernelConfigError, match="donate_pools=True"):
        cache_experimental.make_flat_fused_pooled_step(
            mesh, spec, ex, donate_pools=False
        )
