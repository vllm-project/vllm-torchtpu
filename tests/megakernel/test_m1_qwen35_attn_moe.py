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
"""M1d end-to-end verification for `kernel.build/make_step` and `models.qwen35`."""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import pathlib

import jax
import jax.numpy as jnp
import pytest
from conftest import with_tile_fields
from jax.experimental import topologies
from jax.sharding import Mesh

from vllm_torchtpu.megakernel import kernel
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.models import qwen35
from vllm_torchtpu.megakernel.models.qwen35_profile import TINY_TILES, baseline_tiles
from vllm_torchtpu.megakernel.ops import F32


def _can_compile_tpu7x() -> bool:
    try:
        topologies.get_topology_desc(platform="tpu", topology_name="tpu7x", **{"1d": 8})
        return True
    except Exception:
        return False


def test_m1_recorded_full_ep_build_invariants():
    """Verify recorded `tk`, `nt`, `T_TOTAL`, `TE`, `NSS`, and VMEM budget."""
    g = dataclasses.replace(qwen35.FULL_EP, att_every=1)
    ex = ExecSpec(B=4, T=1, R=8, ctx=4096, tiles=baseline_tiles(4, R=8))
    _, info = qwen35.build(g, ex)
    p = info["plan"]

    assert p.tk == {
        "att_in": 512,
        "att_out": 512,
        "router": 2048,
        "sh13": 4096,
        "sh2": 128,
        "e13": 512,
        "e2": 256,
    }
    assert p.nt == {
        "att_in": 8,
        "att_out": 2,
        "router": 2,
        "sh13": 1,
        "sh2": 1,
        "e13": 8,
        "e2": 4,
    }
    assert info["T_TOTAL"] == 840
    assert p.block_params["moe"].TE == 12
    assert p.block_params["moe"].NSS == 3
    # GDN-free M1 omits `recb`/`sbuf`/`convb` -> 48,033,792 < frozen's 49,178,624.
    # (M4b: 48,031,744; +2,048 B of always-allocated M5 GQA scratch `rcs`. The
    # `rope` table and the QV staging slabs are allocated only under `gqa_qv`.)
    assert info["vmem_scratch_bytes"] == 48_033_792 < 49_178_624
    assert info["vmem_scratch_bytes"] <= ex.resolved_tiles.vmem_mib * (1 << 20)
    assert "recb" not in info["vmem_shapes"]
    assert "sbuf" not in info["vmem_shapes"]
    assert "convb" not in info["vmem_shapes"]


@pytest.mark.parametrize(
    "R,B,T,NKV,moe_tp",
    [
        (1, 2, 1, 1, False),
        (1, 4, 2, 1, True),
        (2, 2, 1, 1, False),
        pytest.param(2, 4, 2, 1, False, marks=pytest.mark.slow),
    ],
)
def test_m1_qwen35_pack_and_step_vs_logical_reference(
    R: int,
    B: int,
    T: int,
    NKV: int,
    moe_tp: bool,
):
    """Verify `pack_inputs` + `make_step` matches the `reference_step` oracle."""
    devs = jax.devices("cpu")
    assert len(devs) >= R, f"Requires {R} CPU devices, found {len(devs)}"

    g = qwen35.Qwen35Geometry(
        L=2,
        att_every=1,
        H=256,
        NH=2 * R,
        NKV=NKV,
        HD=128,
        ROT=32,
        KH=2 * R,
        VH=2 * R,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=256,
        IS=256,
        moe_tp=moe_tp,
    )
    tiles = dataclasses.replace(TINY_TILES, bank=256 << 10).with_blocks(
        {"gqa": {"split": True}}
    )
    ex = ExecSpec(B=B, T=T, R=R, ctx=256, tiles=tiles, interpret=True)

    key = jax.random.PRNGKey(100 + R * 10 + T)
    pos, hid_l, w_l, st_l = qwen35.random_logical_inputs(key, g, ex)
    ref_hid, ref_st = qwen35.reference_step(g, ex, pos, hid_l, w_l, st_l)

    mesh = Mesh(devs[:R], ("tp",))
    pos_mk, hid_mk, w_mk, st_mk = qwen35.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex, mesh=mesh
    )
    step, _ = qwen35.make_step(mesh, g, ex)
    out_hid, out_st = step(pos_mk, hid_mk, w_mk, st_mk)

    out_hid_r = kernel.unfold(out_hid, R)
    for r in range(R):
        assert jnp.allclose(
            out_hid_r[r, :B].astype(F32),
            ref_hid.astype(F32),
            atol=2e-2,
            rtol=2e-2,
        )

    kc_out = kernel.unfold(out_st["kc"], R)
    vc_out = kernel.unfold(out_st["vc"], R)
    assert jnp.allclose(
        kc_out.astype(F32), ref_st["kc"].astype(F32), atol=1e-2, rtol=1e-2
    )
    assert jnp.allclose(
        vc_out.astype(F32), ref_st["vc"].astype(F32), atol=1e-2, rtol=1e-2
    )


@pytest.mark.slow
def test_m1_qwen35_r8_multi_rank_interpret():
    """Verify 2-layer, 8-rank EP + `G=4` split GQA + `HIER=1` AR in `interpret=True`."""
    devs = jax.devices("cpu")
    assert len(devs) >= 8, f"Requires 8 CPU devices, found {len(devs)}"

    g = qwen35.Qwen35Geometry(
        L=2,
        att_every=1,
        H=1024,
        NH=8,
        NKV=2,
        HD=128,
        ROT=32,
        KH=8,
        VH=8,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=128,
        IS=1024,
        moe_tp=False,
    )
    tiles = dataclasses.replace(TINY_TILES, bank=256 << 10, hier=1, mb8=2).with_blocks(
        {"gqa": {"split": True}}
    )
    ex = ExecSpec(B=2, T=1, R=8, ctx=256, tiles=tiles, interpret=True)

    key = jax.random.PRNGKey(808)
    pos, hid_l, w_l, st_l = qwen35.random_logical_inputs(key, g, ex)
    ref_hid, ref_st = qwen35.reference_step(g, ex, pos, hid_l, w_l, st_l)

    mesh = Mesh(devs[:8], ("tp",))
    pos_mk, hid_mk, w_mk, st_mk = qwen35.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex, mesh=mesh
    )
    step, _ = qwen35.make_step(mesh, g, ex)
    out_hid, out_st = step(pos_mk, hid_mk, w_mk, st_mk)

    out_hid_r = kernel.unfold(out_hid, 8)
    for r in range(8):
        assert jnp.allclose(
            out_hid_r[r, : ex.B].astype(F32),
            ref_hid.astype(F32),
            atol=3e-2,
            rtol=3e-2,
        )
    kc_out = kernel.unfold(out_st["kc"], 8)
    assert jnp.allclose(
        kc_out.astype(F32), ref_st["kc"].astype(F32), atol=1e-2, rtol=1e-2
    )


@pytest.mark.skipif(
    not _can_compile_tpu7x(), reason="TPU7x AOT topology compiler not available"
)
def test_m1_tpu7x_aot_compile_full_ep():
    """Verify `qwen35.make_step` lowers and AOT-compiles for `tpu7x` (`1d=8`)."""
    topo = topologies.get_topology_desc(
        platform="tpu", topology_name="tpu7x", **{"1d": 8}
    )
    mesh = Mesh(topo.devices, ("tp",))
    g = dataclasses.replace(qwen35.FULL_EP, L=2, att_every=1)
    ex = ExecSpec(B=4, T=1, R=8, ctx=2048, tiles=baseline_tiles(4, R=8))

    step, info = qwen35.make_step(mesh, g, ex, donate_states=True)
    wsh = info["wshape"]
    ssh = info["sshape"]

    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    rep = NamedSharding(mesh, P())
    tp = NamedSharding(mesh, P("tp"))

    pos = jax.ShapeDtypeStruct((ex.B,), jnp.int32, sharding=rep)
    hid = jax.ShapeDtypeStruct((ex.R * ex.M, g.H), jnp.bfloat16, sharding=tp)
    weights = {
        k: tuple(
            jax.ShapeDtypeStruct((ex.R, *s.shape), s.dtype, sharding=tp) for s in v
        )
        if isinstance(v, tuple) and hasattr(v[0], "shape")
        else jax.ShapeDtypeStruct((ex.R, *v[0]), v[1], sharding=tp)
        for k, v in wsh.items()
    }
    states = {
        k: jax.ShapeDtypeStruct((ex.R * v[0][0], *v[0][1:]), v[1], sharding=tp)
        for k, v in ssh.items()
    }

    compiled = step.lower(pos, hid, weights, states).compile()
    hlo_text = compiled.as_text()
    assert "HloModule" in hlo_text
    assert "qwen3_5_decode_megakernel" in hlo_text


@pytest.mark.skipif(
    not os.environ.get("MK_FROZEN_QWEN35_DIR"),
    reason="Set MK_FROZEN_QWEN35_DIR to frozen qwen35 path for bit-exact check",
)
def test_m1_bit_exact_vs_frozen_qwen35():
    """Opt-in bit-exact comparison against frozen `qwen35_megakernel` (`c196d2500`)."""
    import sys

    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    frozen_dir = pathlib.Path(os.environ["MK_FROZEN_QWEN35_DIR"])
    if str(frozen_dir) not in sys.path:
        sys.path.insert(0, str(frozen_dir))
    spec_f = importlib.util.spec_from_file_location(
        "_frozen_qwen35_mk", frozen_dir / "megakernel.py"
    )
    assert spec_f is not None and spec_f.loader is not None
    frozen_mk = importlib.util.module_from_spec(spec_f)
    sys.modules[spec_f.name] = frozen_mk
    spec_f.loader.exec_module(frozen_mk)

    devs = jax.devices("cpu")
    mesh = Mesh(devs[:1], ("tp",))
    tp = NamedSharding(mesh, P("tp"))

    g = qwen35.Qwen35Geometry(
        L=2,
        att_every=1,
        H=256,
        NH=2,
        NKV=1,
        HD=128,
        ROT=32,
        KH=2,
        VH=2,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=256,
        IS=256,
        moe_tp=False,
    )
    tiles_lib = dataclasses.replace(TINY_TILES, bank=256 << 10)
    ex = ExecSpec(B=2, T=1, R=1, ctx=256, tiles=tiles_lib, interpret=True)
    pos, hid_lib, w_lib, st_lib = qwen35.random_inputs(
        jax.random.PRNGKey(0), g, ex, mesh=mesh
    )
    step_lib, info_lib = qwen35.make_step(mesh, g, ex)
    out_hid_lib, out_st_lib = step_lib(pos, hid_lib, w_lib, st_lib)

    tiles_f = frozen_mk.Tiles(
        bank=256 << 10, nbd=10, nbe=6, ebank=256 << 10, vmem_mib=32, kv_blk=256, nkv=3
    )
    c_f = frozen_mk.MKConfig(
        R=1,
        B=2,
        T=1,
        L=2,
        att_every=1,
        H=256,
        NH=2,
        NKV=1,
        HD=128,
        ROT=32,
        KH=2,
        VH=2,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=256,
        IS=256,
        moe_tp=False,
        ctx=256,
        stack_e=False,
    )
    f_wsh = frozen_mk.weight_shapes(c_f, tiles_f)
    f_ssh = frozen_mk.state_shapes(c_f)
    f_lay, f_S = frozen_mk.smalls_layout(c_f)
    l_lay = info_lib["plan"].smalls_layout

    sm_f = jnp.zeros((ex.R, g.L, 1, f_S), dtype=F32)
    for name, (lo, w) in l_lay.items():
        fo, _ = f_lay[name]
        sm_f = sm_f.at[:, :, 0, fo : fo + w].set(w_lib["smalls"][:, :, 0, lo : lo + w])

    w_f = {
        "lin_in": jax.device_put(
            jnp.zeros((ex.R, *f_wsh["lin_in"][0]), dtype=f_wsh["lin_in"][1]), tp
        ),
        "lin_out": jax.device_put(
            jnp.zeros((ex.R, *f_wsh["lin_out"][0]), dtype=f_wsh["lin_out"][1]), tp
        ),
        "att_in": w_lib["att_in"],
        "att_out": w_lib["att_out"],
        "router": w_lib["router"],
        "sh13": w_lib["sh13"],
        "sh2": w_lib["sh2"],
        "e13": w_lib["e13"],
        "e2": w_lib["e2"],
        "es": w_lib["es"],
        "smalls": jax.device_put(sm_f, tp),
        "final_norm": w_lib["final_norm"],
    }
    st_f = {
        "conv": jax.device_put(
            jnp.zeros(
                (ex.R * f_ssh["conv"][0][0], *f_ssh["conv"][0][1:]),
                dtype=f_ssh["conv"][1],
            ),
            tp,
        ),
        "rec": jax.device_put(
            jnp.zeros(
                (ex.R * f_ssh["rec"][0][0], *f_ssh["rec"][0][1:]), dtype=f_ssh["rec"][1]
            ),
            tp,
        ),
        "kc": st_lib["kc"],
        "vc": st_lib["vc"],
    }
    step_f, _ = frozen_mk.make_step(mesh, c_f, tiles_f, interpret=True)
    out_hid_f, out_st_f = jax.jit(step_f)(pos, hid_lib, w_f, st_f)

    assert jnp.array_equal(out_hid_lib, out_hid_f)
    assert jnp.array_equal(out_st_lib["kc"], out_st_f["kc"])
    assert jnp.array_equal(out_st_lib["vc"], out_st_f["vc"])


_FLAG_GEOM = dict(
    L=2,
    att_every=1,
    H=512,
    NH=2,
    NKV=1,
    HD=128,
    ROT=32,
    KH=2,
    VH=2,
    KD=128,
    VD=128,
    E=128,
    TOPK=10,
    I=256,
    IS=256,
    moe_tp=False,
)
# bank=96 KiB fits one [128, (2*nh+2)*HD = 768] fp8 att_in tile -> tk=128, so
# att_in has n_t = H / 128 = 4 tiles with N = 768 > 512 (paired U=2 when
# n_t > dmm_unroll). ebank=64 KiB -> e13 [512, 512] T13=4 and e2 [256, 512] T2=2,
# so T13 == 2 * T2 and nbe=6 == TE (PIPE_EP preconditions).
_FLAG_TILES_BASE = dataclasses.replace(
    TINY_TILES, bank=96 << 10, nbd=6, ebank=64 << 10, nbe=6
).with_blocks({"gqa": {"kv_blk": 128, "nkv": 3}})


@pytest.fixture(scope="module")
def m5_flag_base():
    """Frozen-parity baseline run shared by the per-flag parity tests."""
    mesh1 = Mesh(jax.devices("cpu")[:1], ("tp",))
    g = qwen35.Qwen35Geometry(**_FLAG_GEOM)
    ex = ExecSpec(B=2, T=1, R=1, ctx=256, tiles=_FLAG_TILES_BASE, interpret=True)
    pos, hid_l, w_l, st_l = qwen35.random_logical_inputs(jax.random.PRNGKey(501), g, ex)
    args = qwen35.pack_inputs(pos, hid_l, w_l, st_l, g, ex, mesh=mesh1)
    step, info = qwen35.make_step(mesh1, g, ex)
    p = info["plan"]
    assert info["T_TOTAL"] > _FLAG_TILES_BASE.nbd  # dynamic dense streaming
    assert p.nt["att_in"] == 4
    out_hid, out_st = step(*args)
    return mesh1, g, ex, args, out_hid, out_st


@pytest.mark.parametrize(
    "flags",
    [
        {"moe.topk_lanes": True},
        {"gqa.pair_kv": True},
        {"moe.pipe_ep": True},
        {"moe.moe_unroll": True},
        {"rel_fetch": True},
        {"moe.r2_direct_store": True},
        {"moe.pipe_ep": True, "moe.r2_direct_store": True},
    ],
    ids=lambda d: "+".join(sorted(k.split(".")[-1] for k in d)),
)
def test_m5_flags_cpu_bit_exact_vs_default(m5_flag_base, flags):
    """These opt-in M5 flags reproduce the default bitwise in CPU interpret mode.

    CPU bit-exactness is necessary, not sufficient: on TPU `moe_unroll` (and
    `pipe_ep`, which implies it) changes MoE numerics vs frozen (the TPU bisect
    `test_m5_tpu_v7x8_frozen_parity_bisect`), so their numerics class is `approx`.
    """
    mesh1, g, ex, args, out_hid, out_st = m5_flag_base
    ex_opt = dataclasses.replace(ex, tiles=with_tile_fields(_FLAG_TILES_BASE, flags))
    step_opt, info_opt = qwen35.make_step(mesh1, g, ex_opt)
    mp = info_opt["plan"].block_params["moe"]
    assert mp.PIPE_EP is bool(flags.get("moe.pipe_ep", False))
    assert mp.topk_lanes is bool(flags.get("moe.topk_lanes", False))
    assert info_opt["plan"].block_params["gqa"].PAIR_KV is bool(
        flags.get("gqa.pair_kv", False)
    )
    out_hid_opt, out_st_opt = step_opt(*args)
    assert jnp.array_equal(out_hid_opt, out_hid), flags
    assert jnp.array_equal(out_st_opt["kc"], out_st["kc"]), flags
    assert jnp.array_equal(out_st_opt["vc"], out_st["vc"]), flags


def test_m5_dense_acc_pair_takes_paired_path(m5_flag_base):
    """`dense_acc_pair` with `n_t > dmm_unroll` runs U=2 paired accumulation.

    The default (`dense_acc_pair=False`, `dmm_unroll=0`) runs the sequential
    `dmatmul` loop for every kind, so the two runs take different code paths;
    pairing reassociates the f32 sums (numerics class `approx`).
    """
    mesh1, g, ex, args, out_hid, _ = m5_flag_base
    dmu = 2
    tiles_dmm = dataclasses.replace(
        _FLAG_TILES_BASE, dense_acc_pair=True, dmm_unroll=dmu
    )
    ex_dmm = dataclasses.replace(ex, tiles=tiles_dmm)
    step_dmm, info_dmm = qwen35.make_step(mesh1, g, ex_dmm)
    p = info_dmm["plan"]
    n_att = p.nt["att_in"]
    N_att = (2 * (g.NH // ex.R) + 2) * g.HD
    # U = 2 preconditions in `TwoRing.dmatmul_pair` for att_in
    assert n_att > min(dmu, _FLAG_TILES_BASE.nbd) and n_att % 2 == 0 and N_att > 512
    # ... and the default never enters dmatmul_pair (n_t <= 0 is impossible)
    assert _FLAG_TILES_BASE.dmm_unroll == 0 and not _FLAG_TILES_BASE.dense_acc_pair
    out_hid_dmm, _ = step_dmm(*args)
    assert jnp.allclose(
        out_hid_dmm[: ex.B].astype(F32),
        out_hid[: ex.B].astype(F32),
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.parametrize("R_test", [2, 4])
def test_m5_hier3_butterfly_all_reduce(R_test: int):
    """HIER=3 butterfly all-reduce at power-of-2 R != 8 vs the flat ring."""
    devs = jax.devices("cpu")
    if len(devs) < R_test:
        pytest.skip(f"Need {R_test} CPU devices, have {len(devs)}")
    mesh_r = Mesh(devs[:R_test], ("tp",))
    g_r = qwen35.Qwen35Geometry(
        **dict(
            _FLAG_GEOM,
            H=256,
            L=1,
            NH=2 * R_test,
            KH=2 * R_test,
            VH=2 * R_test,
            IS=128 * R_test,  # ISR = IS / R must be a multiple of 128
            TOPK=2,
        )
    )
    tiles_h0 = dataclasses.replace(TINY_TILES, bank=256 << 10, hier=0).with_blocks(
        {"gqa": {"split": False}}
    )
    tiles_h3 = dataclasses.replace(tiles_h0, hier=3)
    ex_h0 = ExecSpec(B=2, T=1, R=R_test, ctx=256, tiles=tiles_h0, interpret=True)
    ex_h3 = dataclasses.replace(ex_h0, tiles=tiles_h3)
    pos_r, hid_lr, w_lr, st_lr = qwen35.random_logical_inputs(
        jax.random.PRNGKey(510 + R_test), g_r, ex_h0
    )
    args_r = qwen35.pack_inputs(pos_r, hid_lr, w_lr, st_lr, g_r, ex_h0, mesh=mesh_r)
    step_h0, info_h0 = qwen35.make_step(mesh_r, g_r, ex_h0)
    step_h3, info_h3 = qwen35.make_step(mesh_r, g_r, ex_h3)
    assert info_h0["core_plan"].comm.HIER == 0
    assert info_h3["core_plan"].comm.HIER == 3
    out_h0, _ = step_h0(*args_r)
    out_h3, _ = step_h3(*args_r)
    assert jnp.allclose(out_h3.astype(F32), out_h0.astype(F32), atol=1e-2, rtol=1e-2)


def test_gqa_rope_fp32_ref_vs_kernel_parity():
    """Verify `GQATunables.rope_fp32=True` matches `reference_step` in the same mode."""
    mesh1 = Mesh(jax.devices("cpu")[:1], ("tp",))
    g = qwen35.Qwen35Geometry(**dict(_FLAG_GEOM, H=256))
    tiles_def = dataclasses.replace(TINY_TILES, bank=256 << 10).with_blocks(
        {"gqa": {"kv_blk": 128, "nkv": 3, "rope_fp32": False}}
    )
    tiles_fp32 = tiles_def.with_blocks({"gqa": {"rope_fp32": True}})
    ex_def = ExecSpec(B=2, T=1, R=1, ctx=256, tiles=tiles_def, interpret=True)
    ex_fp32 = dataclasses.replace(ex_def, tiles=tiles_fp32)
    pos, hid_l, w_l, st_l = qwen35.random_logical_inputs(
        jax.random.PRNGKey(509), g, ex_fp32
    )
    ref_hid_def, ref_st_def = qwen35.reference_step(g, ex_def, pos, hid_l, w_l, st_l)
    ref_hid, ref_st = qwen35.reference_step(g, ex_fp32, pos, hid_l, w_l, st_l)
    assert not jnp.array_equal(ref_st["kc"], ref_st_def["kc"])
    args = qwen35.pack_inputs(pos, hid_l, w_l, st_l, g, ex_fp32, mesh=mesh1)
    step_def, _ = qwen35.make_step(mesh1, g, ex_def)
    step, info = qwen35.make_step(mesh1, g, ex_fp32)
    assert info["plan"].block_params["gqa"].rope_fp32 is True
    _, out_st_def = step_def(*args)
    out_hid, out_st = step(*args)
    assert not jnp.array_equal(out_st["kc"], out_st_def["kc"])
    assert jnp.allclose(
        out_hid[: ex_fp32.B].astype(F32),
        ref_hid.astype(F32),
        atol=1e-2,
        rtol=1e-2,
    )
    assert jnp.array_equal(out_st["kc"][0], ref_st["kc"][0, 0])
    assert jnp.allclose(
        out_st["kc"].astype(F32), ref_st["kc"][0].astype(F32), atol=1e-2, rtol=1e-2
    )
    assert jnp.allclose(
        out_st["vc"].astype(F32), ref_st["vc"][0].astype(F32), atol=1e-2, rtol=1e-2
    )
