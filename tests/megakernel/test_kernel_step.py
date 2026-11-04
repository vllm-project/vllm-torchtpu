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
"""Model-free unit tests for `kernel.build`, `kernel.make_step`, and helpers."""

from __future__ import annotations

import contextlib
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh

from tests.megakernel.megakernel_fakes import (
    TINY_TILES,
    FakeGeometry,
    make_fake_model_spec,
)
from tests.megakernel.megakernel_synth import synth_hybrid_spec
from vllm_torchtpu.megakernel import kernel, planner
from vllm_torchtpu.megakernel.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.exec_spec import DebugOptions, ExecSpec
from vllm_torchtpu.megakernel.ops import BF16, F32
from vllm_torchtpu.megakernel.reference import spec_reference_step
from vllm_torchtpu.megakernel.spec import (
    LayerType,
    ModelSpec,
    Module,
    SmallField,
    WeightDecl,
)
from vllm_torchtpu.megakernel.weights.spec_inputs import (
    pack_spec_inputs,
    random_spec_inputs,
    random_spec_logical_weights,
)
from vllm_torchtpu.megakernel.weights.spec_pack import pack_spec_logical_weights


def test_kernel_helpers_fold_unfold_and_unpack_local_weight():
    """`fold`, `unfold`, and `unpack_local_weight` handle rank and tile shapes."""
    x = jnp.arange(2 * 3 * 4, dtype=F32).reshape(2, 3, 4)
    folded = kernel.fold(x)
    assert folded.shape == (6, 4)
    np.testing.assert_array_equal(kernel.unfold(folded, 2), x)

    assert not kernel.is_tupled_weight(((4, 8), F32))
    assert kernel.target_weight_shape(((4, 8), F32)) == (4, 8)
    tup_entry = (jax.ShapeDtypeStruct((4, 8), F32), jax.ShapeDtypeStruct((4, 8), F32))
    assert kernel.is_tupled_weight(tup_entry)
    assert kernel.target_weight_shape(tup_entry) == (4, 8)

    w = jnp.arange(2 * 3 * 4 * 8, dtype=BF16).reshape(2, 3, 4, 8)
    np.testing.assert_array_equal(
        kernel.unpack_local_weight(w, (2, 3, 4, 8), name="w"), w
    )
    np.testing.assert_array_equal(
        kernel.unpack_local_weight(w[None], (2, 3, 4, 8), name="w"), w
    )
    w_merged = w.reshape(2, 12, 8)
    np.testing.assert_array_equal(
        kernel.unpack_local_weight(
            w_merged, (2, 3, 4, 8), name="w", allow_merged_tiles=True
        ),
        w,
    )
    np.testing.assert_array_equal(
        kernel.unpack_local_weight(
            w_merged[None], (2, 3, 4, 8), name="w", allow_merged_tiles=True
        ),
        w,
    )
    with pytest.raises(MegakernelConfigError, match="local shard has shape"):
        kernel.unpack_local_weight(w_merged, (2, 3, 4, 8), name="w")


@pytest.mark.parametrize(("T", "B"), [(1, 4), (4, 4)], ids=["decode_t1", "verify_t4"])
def test_kernel_make_step_matches_spec_reference_step(T: int, B: int):
    """`kernel.make_step` matches `spec_reference_step` on `synth_hybrid_spec`."""
    R = 2
    spec = synth_hybrid_spec(L=2, att_every=2, R=R)
    tiles = TINY_TILES.with_modules({"gqa": {"kv_blk": 128, "nkv": 2}})
    ex = ExecSpec(B=B, T=T, R=R, ctx=256, stack_e=True, tiles=tiles, interpret=True)
    mesh = Mesh(np.array(jax.devices("cpu")[:R]), ("tp",))

    pos, hid, w_log, st_log = random_spec_inputs(jax.random.PRNGKey(10 + T), spec, ex)
    pos_mk, hid_mk, w_mk, st_mk = pack_spec_inputs(
        pos, hid, w_log, st_log, spec, ex, mesh=mesh
    )

    step, info = kernel.make_step(mesh, spec, ex)
    assert info["exec"].B == B and info["exec"].T == T
    assert step.lower(pos_mk, hid_mk, w_mk, st_mk) is not None

    out_hid_mk, out_st_mk = step(pos_mk, hid_mk, w_mk, st_mk)
    jax.block_until_ready(out_hid_mk)

    out_hid_r = out_hid_mk.reshape(R, ex.M, spec.config.H)[:, :B]
    np.testing.assert_array_equal(out_hid_r[0], out_hid_r[1])

    ref_hid, ref_st = spec_reference_step(
        spec, spec.config, ex, pos, hid, w_log, st_log
    )
    np.testing.assert_allclose(
        out_hid_r[0].astype(F32), ref_hid.astype(F32), rtol=2e-2, atol=2e-2
    )

    kc_mk = kernel.unfold(out_st_mk["kc"], R)
    vc_mk = kernel.unfold(out_st_mk["vc"], R)
    np.testing.assert_allclose(
        kc_mk.astype(F32), ref_st["kc"].astype(F32), rtol=2e-2, atol=2e-2
    )
    np.testing.assert_allclose(
        vc_mk.astype(F32), ref_st["vc"].astype(F32), rtol=2e-2, atol=2e-2
    )
    rec_key = "rec_ckpt" if T > 1 else "rec"
    rec_mk = kernel.unfold(out_st_mk[rec_key], R)
    np.testing.assert_allclose(
        rec_mk.astype(F32), ref_st[rec_key].astype(F32), rtol=2e-2, atol=2e-2
    )


def test_kernel_build_ablations_and_scopes():
    """`kernel.build` supports `skip` ablations and `scopes` levels."""
    R = 2
    spec = synth_hybrid_spec(L=2, att_every=2, R=R)
    tiles = TINY_TILES.with_modules({"gqa": {"kv_blk": 128, "nkv": 2}})
    ex = ExecSpec(
        B=4,
        T=1,
        R=R,
        ctx=256,
        stack_e=True,
        tiles=tiles,
        interpret=True,
        debug=DebugOptions(skip=frozenset({"experts", "kv"}), scopes=2),
    )
    call, info = kernel.build(spec, ex)
    assert callable(call)
    assert info["exec"].skip == frozenset({"experts", "kv"})


def test_kernel_per_module_profiling_scopes(monkeypatch):
    """`kernel.build` wraps module hooks in `<module>/<phase>` at `scopes >= 1`
    and keeps finer `ctx.sc` scopes at `scopes >= 2` / `>= 3`."""

    class _ScopedModule(Module):
        name = "scoped_mod"
        reduction = "allreduce"
        input_norm = "in_norm"

        def weights(self, g: FakeGeometry, ex: ExecSpec) -> list[WeightDecl]:
            return [WeightDecl("w", K=g.H, N=g.H, dtype=BF16, stream_ring="static")]

        def smalls(self, g: FakeGeometry, ex: ExecSpec) -> list[SmallField]:
            return [SmallField("in_norm", g.H)]

        def shared_widths(self, g: FakeGeometry, ex: ExecSpec) -> dict[str, int]:
            return {"acc": g.H}

        def emit(self, ctx, lyr, slot):
            del lyr, slot
            with (
                ctx.sc("inner_l2"),
                ctx.sc("inner_l3", 3),
                ctx.stream.segment(lazy=False),
            ):
                return ctx.stream.dmatmul("w", ctx.scr["xb"], ctx.g.H)

    g = FakeGeometry(L=2, H=256)
    ms = ModelSpec(
        "scoped_test", config=g, layer_type=LayerType("lt", (_ScopedModule(),))
    )
    tiles = dataclasses.replace(TINY_TILES, bank=64 << 10, nbd=2)
    mesh = Mesh(jax.devices("cpu")[:1], ("tp",))

    def _trace_scopes(scopes_val: int | bool, *, interpret: bool = True):
        seen: list[str] = []
        traced_vals: list[tuple[str, object]] = []
        orig_ns = jax.named_scope

        @contextlib.contextmanager
        def _spy_ns(name: str):
            seen.append(name)
            with orig_ns(name):
                yield

        monkeypatch.setattr(jax, "named_scope", _spy_ns)
        monkeypatch.setattr(
            pltpu,
            "trace_value",
            lambda label, val: traced_vals.append((label, val)),
        )
        ex = ExecSpec(
            R=1,
            B=4,
            T=1,
            ctx=128,
            tiles=tiles,
            interpret=interpret,
            debug=DebugOptions(scopes=scopes_val),
        )
        p = planner.plan(ms, ex)
        wsh = planner.weight_shapes(ms, ex, p)
        pos = jnp.zeros((ex.B,), dtype=jnp.int32)
        hid = jnp.zeros((ex.R * ex.M, g.H), dtype=BF16)
        weights = {k: jnp.zeros((ex.R, *sh), dtype=dt) for k, (sh, dt) in wsh.items()}
        step, _ = kernel.make_step(mesh, ms, ex)
        if interpret:
            step.lower(pos, hid, weights, {})
        else:
            with pytest.raises(ValueError, match="Only interpret mode"):
                step.lower(pos, hid, weights, {})
        return seen, traced_vals

    s_off, tv_off = _trace_scopes(False, interpret=False)
    assert s_off == [] and tv_off == []

    for s_val in (1, True):
        s_l1, tv_l1 = _trace_scopes(s_val, interpret=False)
        assert set(s_l1) == {
            "scoped_mod/prologue",
            "scoped_mod/init_buffers",
            "scoped_mod/on_layer_start",
            "scoped_mod/emit",
            "scoped_mod/epilogue",
        }
        assert [lbl for lbl, _ in tv_l1] == ["layer"]

    s_l2, _ = _trace_scopes(2)
    assert {"scoped_mod/emit", "inner_l2", "norm_in", "smalls_sync"} <= set(s_l2)
    assert "inner_l3" not in set(s_l2)
    assert "dwait" not in set(s_l2)

    s_l3, _ = _trace_scopes(3)
    assert {"scoped_mod/emit", "inner_l2", "inner_l3", "dwait"} <= set(s_l3)


def test_pack_spec_logical_weights_slices_2d_smalls_along_layer_axis():
    """`pack_spec_logical_weights` slices 2-D `(L, W)` smalls by layer, not column."""
    ms = make_fake_model_spec(L=2, att_every=2, H=256)
    ex = ExecSpec(B=4, T=1, R=2, ctx=256, stack_e=True, tiles=TINY_TILES)
    w_log = random_spec_logical_weights(jax.random.PRNGKey(0), ms, ex)
    assert w_log["in_norm"].shape == (2, 256)
    w_log["in_norm"] = jnp.stack(
        [jnp.full((256,), 1.25, dtype=F32), jnp.full((256,), 2.75, dtype=F32)], axis=0
    )
    packed = pack_spec_logical_weights(ms, ex, w_log)
    o_in, w_in = planner.plan(ms, ex).smalls_layout["in_norm"]
    assert w_in == 256
    np.testing.assert_allclose(packed["smalls"][:, 0, 0, o_in : o_in + w_in], 1.25)
    np.testing.assert_allclose(packed["smalls"][:, 1, 0, o_in : o_in + w_in], 2.75)
