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
"""On-TPU (`interpret=False`) tests for `kernel.make_step` on `synth_hybrid_spec`."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tests.megakernel.conftest import host_device, tpu_mesh
from tests.megakernel.core.megakernel_synth import TINY_TILES, synth_hybrid_spec
from vllm_torchtpu.megakernel.core import kernel, pool
from vllm_torchtpu.megakernel.core.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.core.geometry import ModelGeometry
from vllm_torchtpu.megakernel.core.ops import F32
from vllm_torchtpu.megakernel.core.pool.config import (
    POOL_STATE_NAME,
    CacheExec,
    resolved_max_pages_per_seq,
)
from vllm_torchtpu.megakernel.core.reference import spec_reference_step
from vllm_torchtpu.megakernel.core.weights.spec_inputs import (
    pack_spec_inputs,
    random_spec_inputs,
)


def _synth_geom(R: int, *, L: int = 4, att_every: int = 2) -> Any:
    return ModelGeometry(
        L=L,
        att_every=att_every,
        H=256,
        NH=max(2, 2 * R),
        NKV=min(2, R),
        HD=256,
        ROT=64,
        KH=max(2, R),
        VH=max(4, 2 * R),
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=256,
        IS=256,
        moe_tp=False,
        e_slots=0,
        theta=10_000_000.0,
        zero_centered_norm=True,
    )


def _base_tiles():
    return replace(TINY_TILES, bank=512 << 10).with_modules(
        {"gqa": {"kv_blk": 128, "nkv": 2}}
    )


def _pack_pooled_raw_ref_states(
    st_mk: dict[str, Any],
    st_log: dict[str, Any],
    spec: Any,
    ex: ExecSpec,
    mesh: Mesh,
    *,
    pool_gather: bool,
) -> dict[str, Any]:
    assert spec.cache_backend is not None
    lay = spec.cache_backend.layout
    pt, ckpt, roff, min_pages = pool.allocate_test_layout(lay, ex)
    assert ex.cache is not None and ex.cache.num_pages >= min_pages
    tp = NamedSharding(mesh, P("tp"))
    raw_host = pool.states_to_pool(lay, ex, st_log, pt, ckpt, roff)
    out_states = dict(st_mk)
    out_states[POOL_STATE_NAME] = tuple(
        jax.device_put(np.asarray(r), tp) for r in raw_host
    )
    pt_sliced = pt[:, : resolved_max_pages_per_seq(ex)]
    out_states["page_table"] = jax.device_put(jnp.tile(pt_sliced, (ex.R, 1)), tp)
    if pool_gather:
        out_states.pop("conv", None)
        out_states.pop("rec", None)
        out_states["ckpt_indices"] = jax.device_put(jnp.tile(ckpt, (ex.R, 1, 1)), tp)
        out_states["read_offsets"] = jax.device_put(jnp.tile(roff, (ex.R, 1)), tp)
    return out_states


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    ("R", "T", "B"),
    [(1, 1, 4), (2, 1, 4), (2, 4, 8)],
    ids=["r1_decode_t1", "r2_decode_t1", "r2_verify_t4"],
)
def test_kernel_make_step_matches_spec_reference_step_on_tpu(
    R: int, T: int, B: int
) -> None:
    """Compiled (`interpret=False`) `kernel.make_step` matches `spec_reference_step`
    on `synth_hybrid_spec` across single-rank and multi-rank decode/verify steps."""
    mesh = tpu_mesh(R)
    g = _synth_geom(R)
    spec = synth_hybrid_spec(g, R=R)
    ex = ExecSpec(
        B=B,
        T=T,
        R=R,
        ctx=256,
        stack_e=True,
        tiles=_base_tiles(),
        interpret=False,
    )
    host_dev = host_device(mesh.devices.flat[0])
    with jax.default_device(host_dev):
        ex_ref = replace(ex, interpret=True)
        pos, hid, w_log, st_log = random_spec_inputs(
            jax.random.key(100 + 10 * R + T), spec, ex_ref
        )
        ref_hid, ref_st = spec_reference_step(spec, g, ex_ref, pos, hid, w_log, st_log)

    pos_mk, hid_mk, w_mk, st_mk = pack_spec_inputs(
        pos, hid, w_log, st_log, spec, ex, mesh=mesh
    )
    step, info = kernel.make_step(mesh, spec, ex)
    assert not info["exec"].interpret
    out_hid_mk, out_st_mk = step(pos_mk, hid_mk, w_mk, st_mk)
    jax.block_until_ready(out_hid_mk)

    out_hid_r = np.asarray(kernel.unfold(out_hid_mk, R)[:, :B], dtype=np.float32)
    ref_hid_np = np.asarray(ref_hid, dtype=np.float32)
    assert float(np.max(np.abs(ref_hid_np))) >= 0.05
    if R > 1:
        np.testing.assert_array_equal(out_hid_r[0], out_hid_r[1])
    np.testing.assert_allclose(out_hid_r[0], ref_hid_np, rtol=2.5e-2, atol=2.5e-2)

    kc_mk = np.asarray(kernel.unfold(out_st_mk["kc"], R), dtype=np.float32)
    vc_mk = np.asarray(kernel.unfold(out_st_mk["vc"], R), dtype=np.float32)
    np.testing.assert_allclose(
        kc_mk, np.asarray(ref_st["kc"], dtype=np.float32), rtol=2e-2, atol=2e-2
    )
    np.testing.assert_allclose(
        vc_mk, np.asarray(ref_st["vc"], dtype=np.float32), rtol=2e-2, atol=2e-2
    )

    conv_key = "conv_ckpt" if T > 1 else "conv"
    rec_key = "rec_ckpt" if T > 1 else "rec"
    conv_mk = np.asarray(kernel.unfold(out_st_mk[conv_key], R), dtype=np.float32)
    rec_mk = np.asarray(kernel.unfold(out_st_mk[rec_key], R), dtype=np.float32)
    if T == 1:
        conv_valid: Any = (slice(None), slice(None), slice(0, 3 * B), slice(None))
    else:
        conv_valid = (slice(None), slice(None), slice(None), slice(0, 3 + T))
    np.testing.assert_allclose(
        conv_mk[conv_valid],
        np.asarray(ref_st[conv_key], dtype=np.float32)[conv_valid],
        rtol=2e-2,
        atol=2e-2,
    )
    np.testing.assert_allclose(
        rec_mk, np.asarray(ref_st[rec_key], dtype=np.float32), rtol=2e-2, atol=2e-2
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    ("R", "T", "B", "pool_gather"),
    [
        (1, 1, 4, False),
        (2, 1, 4, False),
        (2, 4, 8, False),
        (2, 4, 8, True),
    ],
    ids=["r1_t1_raw_kv", "r2_t1_raw_kv", "r2_t4_raw_kv", "r2_t4_raw_kv_gpr"],
)
def test_kernel_make_step_pooled_raw_ref_matches_reference_on_tpu(
    R: int, T: int, B: int, pool_gather: bool
) -> None:
    """Compiled (`interpret=False`) `kernel.make_step` over pooled `raw_pool` refs
    (both `pool_gather=False` and `pool_gather=True`) matches `spec_reference_step`."""
    mesh = tpu_mesh(R)
    g = _synth_geom(R)
    spec = synth_hybrid_spec(g, R=R)
    ex = ExecSpec(
        B=B,
        T=T,
        R=R,
        ctx=256,
        stack_e=True,
        tiles=_base_tiles(),
        interpret=False,
        cache=CacheExec(paged_kv=True, num_pages=128, pool_gather=pool_gather),
    )
    host_dev = host_device(mesh.devices.flat[0])
    with jax.default_device(host_dev):
        ex_ref = replace(
            ex, interpret=True, cache=CacheExec(paged_kv=True, num_pages=128)
        )
        pos, hid, w_log, st_log = random_spec_inputs(
            jax.random.key(200 + 10 * R + T + int(pool_gather)), spec, ex_ref
        )
        ref_hid, ref_st = spec_reference_step(
            spec, g, ex_ref, pos, hid, w_log, st_log, paged_kv=True, pool_io=True
        )
        pos_mk, hid_mk, w_mk, st_mk_base = pack_spec_inputs(
            pos, hid, w_log, st_log, spec, ex_ref
        )

    tp = NamedSharding(mesh, P("tp"))
    rep = NamedSharding(mesh, P())
    pos_tpu = jax.device_put(pos_mk, rep)
    hid_tpu = jax.device_put(hid_mk, tp)
    w_tpu = {
        k: tuple(jax.device_put(x, tp) for x in v)
        if isinstance(v, tuple)
        else jax.device_put(v, tp)
        for k, v in w_mk.items()
    }
    st_base_tpu = {
        k: jax.device_put(v, tp)
        for k, v in st_mk_base.items()
        if not (pool_gather and k in ("conv", "rec"))
    }
    st_pooled = _pack_pooled_raw_ref_states(
        st_base_tpu, st_log, spec, ex, mesh, pool_gather=pool_gather
    )

    step, info = kernel.make_step(mesh, spec, ex)
    assert POOL_STATE_NAME in info["plan"].s_names
    out_hid, out_st = step(pos_tpu, hid_tpu, w_tpu, st_pooled)
    jax.block_until_ready(out_hid)

    out_hid_r = np.asarray(kernel.unfold(out_hid, R)[:, :B], dtype=np.float32)
    if R > 1:
        np.testing.assert_array_equal(out_hid_r[0], out_hid_r[1])
    np.testing.assert_allclose(
        out_hid_r[0], np.asarray(ref_hid, dtype=np.float32), rtol=3e-2, atol=3e-2
    )

    # Verify newly emitted FP8 KV rows (`knew`, `vnew`) match reference scatter rows.
    knew_r = np.asarray(kernel.unfold(out_st["knew"], R)[:, :, :B, :].astype(F32))
    vnew_r = np.asarray(kernel.unfold(out_st["vnew"], R)[:, :, :B, :].astype(F32))
    pos_np = np.asarray(pos, dtype=np.int32)
    seq_idx = np.arange(B) if T == 1 else (np.arange(B) // T)
    ref_kc = np.asarray(ref_st["kc"].astype(F32))
    ref_vc = np.asarray(ref_st["vc"].astype(F32))
    ref_knew = np.stack(
        [ref_kc[:, :, seq_idx[b], pos_np[b], :] for b in range(B)], axis=2
    )
    ref_vnew = np.stack(
        [ref_vc[:, :, seq_idx[b], pos_np[b], :] for b in range(B)], axis=2
    )
    # FP8 e4m3fn has 3 mantissa bits (1 ULP = 12.5% relative step); >=99% match tightly
    # and midpoint ties differ by at most 1 FP8 ULP.
    np.testing.assert_allclose(knew_r, ref_knew, rtol=0.15, atol=0.15)
    np.testing.assert_allclose(vnew_r, ref_vnew, rtol=0.15, atol=0.15)
    assert float(np.mean(np.isclose(knew_r, ref_knew, rtol=5e-2, atol=5e-2))) >= 0.99
    assert float(np.mean(np.isclose(vnew_r, ref_vnew, rtol=5e-2, atol=5e-2))) >= 0.99

    conv_key = "conv_ckpt" if T > 1 else "conv"
    rec_key = "rec_ckpt" if T > 1 else "rec"
    conv_mk = np.asarray(kernel.unfold(out_st[conv_key], R), dtype=np.float32)
    rec_mk = np.asarray(kernel.unfold(out_st[rec_key], R), dtype=np.float32)
    if T == 1:
        conv_valid: Any = (slice(None), slice(None), slice(0, 3 * B), slice(None))
    else:
        conv_valid = (slice(None), slice(None), slice(None), slice(0, 3 + T))
    np.testing.assert_allclose(
        conv_mk[conv_valid],
        np.asarray(ref_st[conv_key], dtype=np.float32)[conv_valid],
        rtol=2.5e-2,
        atol=2.5e-2,
    )
    np.testing.assert_allclose(
        rec_mk, np.asarray(ref_st[rec_key], dtype=np.float32), rtol=2.5e-2, atol=2.5e-2
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "paged_kv",
    [False, True],
    ids=["direct_hbm", "pooled_raw_ref"],
)
def test_kernel_make_step_tiles_inplace_on_tpu(paged_kv: bool) -> None:
    """`tiles.inplace=True` (`R=1`, `T=1`, `donate_states=True`, `interpret=False`)
    aliases `rw` GDN `conv`/`rec` states in-place and matches `tiles.inplace=False`
    bit-for-bit as well as `spec_reference_step` on both direct and pooled paths."""
    R, T, B = 1, 1, 4
    mesh = tpu_mesh(R)
    g = _synth_geom(R)
    spec = synth_hybrid_spec(g, R=R)
    cache_cfg = CacheExec(paged_kv=True, num_pages=128) if paged_kv else CacheExec()
    tiles_oop = replace(_base_tiles(), inplace=False)
    tiles_inp = replace(_base_tiles(), inplace=True)
    ex_oop = ExecSpec(
        B=B,
        T=T,
        R=R,
        ctx=256,
        stack_e=True,
        tiles=tiles_oop,
        interpret=False,
        cache=cache_cfg,
    )
    ex_inp = replace(ex_oop, tiles=tiles_inp)

    host_dev = host_device(mesh.devices.flat[0])
    with jax.default_device(host_dev):
        ex_ref = replace(ex_oop, interpret=True)
        pos, hid, w_log, st_log = random_spec_inputs(
            jax.random.key(300 + int(paged_kv)), spec, ex_ref
        )
        ref_hid, ref_st = spec_reference_step(
            spec,
            g,
            ex_ref,
            pos,
            hid,
            w_log,
            st_log,
            paged_kv=paged_kv,
            pool_io=paged_kv,
        )
        pos_mk, hid_mk, w_mk, st_mk_base = pack_spec_inputs(
            pos, hid, w_log, st_log, spec, ex_ref
        )

    tp = NamedSharding(mesh, P("tp"))
    rep = NamedSharding(mesh, P())
    pos_tpu = jax.device_put(pos_mk, rep)
    hid_tpu = jax.device_put(hid_mk, tp)
    w_tpu = {
        k: tuple(jax.device_put(x, tp) for x in v)
        if isinstance(v, tuple)
        else jax.device_put(v, tp)
        for k, v in w_mk.items()
    }

    def _fresh_states() -> dict[str, Any]:
        st_tpu = {
            k: jax.device_put(np.asarray(v).copy(), tp) for k, v in st_mk_base.items()
        }
        if paged_kv:
            st_tpu = _pack_pooled_raw_ref_states(
                st_tpu, st_log, spec, ex_oop, mesh, pool_gather=False
            )
        return st_tpu

    step_oop, info_oop = kernel.make_step(mesh, spec, ex_oop, donate_states=False)
    step_inp, info_inp = kernel.make_step(mesh, spec, ex_inp, donate_states=True)
    assert info_oop["input_output_aliases"] == {}
    assert len(info_inp["input_output_aliases"]) == 2

    lowered_oop = step_oop.lower(pos_tpu, hid_tpu, w_tpu, _fresh_states())
    lowered_inp = step_inp.lower(pos_tpu, hid_tpu, w_tpu, _fresh_states())
    mlir_oop = lowered_oop.as_text()
    mlir_inp = lowered_inp.as_text()
    assert "#stablehlo.output_operand_alias<" not in mlir_oop
    for in_idx, out_idx in info_inp["input_output_aliases"].items():
        expected_alias = (
            f"#stablehlo.output_operand_alias<output_tuple_indices = [{out_idx}], "
            f"operand_index = {in_idx}, operand_tuple_indices = []>"
        )
        assert expected_alias in mlir_inp, expected_alias

    hlo_oop = lowered_oop.compile().as_text()
    hlo_inp = lowered_inp.compile().as_text()
    assert hlo_oop is not None and hlo_inp is not None
    assert "input_output_alias" not in hlo_oop
    assert "output_to_operand_aliasing" not in hlo_oop
    assert "input_output_alias" in hlo_inp
    assert "output_to_operand_aliasing" in hlo_inp

    hid_oop, st_oop = step_oop(pos_tpu, hid_tpu, w_tpu, _fresh_states())
    hid_inp, st_inp = step_inp(pos_tpu, hid_tpu, w_tpu, _fresh_states())
    jax.block_until_ready((hid_oop, hid_inp))

    np.testing.assert_array_equal(
        np.asarray(hid_inp).view(np.uint16),
        np.asarray(hid_oop).view(np.uint16),
    )
    for k in st_oop:
        np.testing.assert_array_equal(
            np.asarray(st_inp[k]).view(np.uint8),
            np.asarray(st_oop[k]).view(np.uint8),
            err_msg=f"tiles.inplace mismatch on state {k!r}",
        )

    out_hid_b = np.asarray(kernel.unfold(hid_inp, R)[0, :B], dtype=np.float32)
    np.testing.assert_allclose(
        out_hid_b, np.asarray(ref_hid, dtype=np.float32), rtol=3e-2, atol=3e-2
    )
    conv_inp = np.asarray(kernel.unfold(st_inp["conv"], R), dtype=np.float32)
    rec_inp = np.asarray(kernel.unfold(st_inp["rec"], R), dtype=np.float32)
    np.testing.assert_allclose(
        conv_inp[:, :, : 3 * B, :],
        np.asarray(ref_st["conv"], dtype=np.float32)[:, :, : 3 * B, :],
        rtol=2.5e-2,
        atol=2.5e-2,
    )
    np.testing.assert_allclose(
        rec_inp, np.asarray(ref_st["rec"], dtype=np.float32), rtol=2.5e-2, atol=2.5e-2
    )
