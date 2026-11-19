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
"""Qwen3.5 pool integration, HLO, and core AST guard tests."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu import megakernel as mk
from vllm_torchtpu.megakernel.models import qwen35
from vllm_torchtpu.megakernel.models.qwen35_profile import TINY_TILES

NUM_PAGES = 128


@pytest.fixture(autouse=True)
def _use_cpu_default_device():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _exec(R: int, B: int, T: int, ctx: int = 256) -> mk.ExecSpec:
    tiles = replace(TINY_TILES, bank=512 << 10).with_blocks(
        {"gqa": {"kv_blk": min(TINY_TILES.block("gqa").kv_blk, ctx)}}
    )
    return mk.ExecSpec(
        R=R,
        B=B,
        T=T,
        ctx=ctx,
        stack_e=True,
        interpret=True,
        tiles=tiles,
        cache=mk.CacheExec(paged_kv=True, num_pages=NUM_PAGES),
    )


def _bytes(x: jax.Array) -> np.ndarray:
    return np.asarray(x).view(np.uint8)


def _gpr_spec_and_exec() -> tuple[mk.ModelSpec, mk.ExecSpec]:
    g = qwen35.Qwen35Geometry(
        L=4, H=256, I=256, IS=256, E=128, TOPK=2, NH=4, NKV=2, HD=256, ROT=64,
        KH=2, VH=4, KD=128, VD=128, att_every=4,
    )  # fmt: skip
    return qwen35.qwen35_spec(g), mk.with_cache(_exec(R=2, B=8, T=4), pool_gather=True)


def test_gqa_pool_layout_must_match_the_pool_backend_layout():
    spec, ex = _gpr_spec_and_exec()
    lay = spec.cache_backend.layout
    mk.planner.plan(spec, ex)
    wrong = replace(spec, cache_backend=mk.make_pool_backend(replace(lay, kvh=2)))
    with pytest.raises(mk.MegakernelConfigError, match="pool backend layout"):
        mk.planner.plan(wrong, mk.with_cache(ex, pool_gather=False))


def test_gdn_in_kernel_pool_io_takes_slot_geometry_from_layout():
    spec, ex = _gpr_spec_and_exec()
    assert mk.cache_exec(ex).GPR
    gp = mk.planner.plan(spec, ex).block_params["gdn"]
    lay = spec.cache_backend.layout
    assert gp.pool_layout is lay
    assert (
        lay.rec_pages_per_slot,
        lay.pages_per_slot,
        lay.slots_per_pool,
        lay.rec_u32_rows,
    ) == (gp.REC_PAGES, gp.GDN_PAGES, gp.G_PER_ATT, gp.SSM_U32_CAP)


def test_gdn_in_kernel_pool_io_rejects_mismatched_layout():
    spec, ex = _gpr_spec_and_exec()
    lay = spec.cache_backend.layout
    wrong = replace(
        spec, cache_backend=mk.make_pool_backend(replace(lay, slots_per_pool=1))
    )
    with pytest.raises(mk.MegakernelConfigError, match="slots_per_pool=1"):
        mk.planner.plan(wrong, ex)
    with pytest.raises(mk.MegakernelConfigError, match="pool backend"):
        mk.planner.plan(replace(spec, cache_backend=None), ex)


def test_qwen35_pool_layout_matches_geometry():
    g = qwen35.Qwen35Geometry()
    lay = mk.qwen35_pool_layout(g)
    assert (lay.num_pools, lay.slots_per_pool) == (g.NATT, g.att_every - 1)
    assert lay.num_recurrent_layers == g.NGDN
    for R in (1, 2, 8):
        gdn = mk.GDNBlock.lin_in_layout(g, R)
        assert (lay.kh_per_rank(R), lay.vh_per_rank(R)) == (gdn.kh, gdn.vh)
        assert lay.qkv_per_rank(R) == gdn.QKV
    assert mk.qwen35_pool_backend(g).layout == lay


def test_check_pooled_step_hlo_uses_layout_page_shape():
    """Full-pool copies are matched on the layout's page shape."""
    lay = mk.PoolLayout(hd=128, kvh=2)  # pages (4, 32, 4, 128)
    head = "tpu_custom_call tpu_custom_call scatter_pool_states\n"
    copy_lay = head + "%c = f8e4m3fn[64,4,32,4,128] copy(%p)"
    copy_default = head + "%c = f8e4m3fn[64,2,64,4,128] copy(%p)"
    staging = head + "%c = f8e4m3fn[5,4,32,4,128] copy(%p)"
    assert not mk.check_pooled_step_hlo(copy_lay, layout=lay)[0]
    assert mk.check_pooled_step_hlo(copy_default, layout=lay)[0]
    assert mk.check_pooled_step_hlo(staging, layout=lay)[0]
    assert not mk.check_pooled_step_hlo(copy_default)[0]


def test_full_pooled_step_att_every_2():
    """One `PallasWriter` pooled step on a 1-slot-per-pool Qwen3.5 geometry."""
    devs = jax.devices("cpu")
    if len(devs) < 2:
        pytest.skip(f"Need at least 2 CPU devices, have {len(devs)}")
    mesh = Mesh(np.array(devs[:2]), ("tp",))
    tp = NamedSharding(mesh, P("tp"))
    g = qwen35.Qwen35Geometry(
        L=4,
        H=256,
        I=256,
        IS=256,
        E=128,
        TOPK=2,
        NH=4,
        NKV=2,
        HD=256,
        ROT=64,
        KH=2,
        VH=4,
        KD=128,
        VD=128,
        att_every=2,
    )
    ex = mk.with_cache(_exec(R=2, B=2, T=1), num_pages=96)
    lay = mk.qwen35_pool_layout(g)
    assert (lay.num_pools, lay.slots_per_pool) == (2, 1)
    pos, hid_l, w_l, st_l = qwen35.random_logical_inputs(jax.random.key(5), g, ex)
    st_l = dict(st_l, conv=st_l["conv"].astype(jnp.bfloat16).astype(jnp.float32))
    _, hid_mk, w_mk, st_mk_paged = qwen35.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex, mesh=mesh, build_default_pool=True
    )
    pt, ckpt, roff, _ = mk.default_pool_layout(g, ex)
    raw_pools = tuple(
        jax.device_put(r, tp) for r in mk.states_to_pool(g, ex, st_l, pt, ckpt, roff)
    )
    mk_paged_step, _ = qwen35.make_step(mesh, g, ex, donate_states=False)
    ref_hid, mk_st = mk_paged_step(pos, hid_mk, w_mk, st_mk_paged)

    spec = qwen35.qwen35_spec(g)
    assert spec.cache_backend.layout == lay
    st_before = mk.pool_to_states(g, ex, raw_pools, pt, ckpt)
    step, _ = mk.PallasWriter().make_step(mesh, spec, ex, donate=False)
    out_hid, raw_out = step(pos, hid_mk, w_mk, raw_pools, pt, ckpt, roff)
    np.testing.assert_array_equal(
        np.asarray(out_hid.astype(jnp.float32)), np.asarray(ref_hid.astype(jnp.float32))
    )

    st_after = mk.pool_to_states(g, ex, raw_out, pt, ckpt)
    for key, new_key in (("kc", "knew"), ("vc", "vnew")):
        exp = np.array(_bytes(st_before[key]))
        new_kv = mk.unfold(mk_st[new_key], ex.R)
        for b in range(ex.B):
            exp[:, :, b, int(pos[b]), :] = _bytes(new_kv[:, :, b])
        np.testing.assert_array_equal(_bytes(st_after[key]), exp)
    np.testing.assert_array_equal(
        np.asarray(st_after["rec"]), np.asarray(mk.unfold(mk_st["rec"], ex.R))
    )
    QKV = lay.qkv_per_rank(ex.R)
    mk_conv = (
        mk.unfold(mk_st["conv"], ex.R)[:, :, : 3 * ex.B, :]
        .reshape(ex.R, g.NGDN, 3, ex.B, QKV)
        .swapaxes(2, 3)
        .astype(jnp.bfloat16)
        .astype(jnp.float32)
    )
    np.testing.assert_array_equal(
        np.asarray(st_after["conv"][:, :, :, 0]), np.asarray(mk_conv)
    )


# ---------------------------------------------------------------------------
# AST guards: the pool state name and the slot stride live in one place.
# ---------------------------------------------------------------------------
_MK_ROOT = Path(mk.__file__).resolve().parent


def _guarded_files() -> list[Path]:
    files = [_MK_ROOT / "kernel.py", *sorted(_MK_ROOT.glob("cache*.py"))]
    files += sorted((_MK_ROOT / "blocks").rglob("*.py"))
    files += sorted((_MK_ROOT / "experimental").glob("*.py"))
    files += sorted((_MK_ROOT / "models").glob("*.py"))
    files += sorted((_MK_ROOT / "pool").glob("*.py"))
    rel_paths = {p.relative_to(_MK_ROOT).as_posix() for p in files}
    expected_files = {
        "kernel.py",
        "cache.py",
        "cache_common.py",
        "cache_pooled.py",
        "cache_pooled_local.py",
        "cache_flat.py",
        "cache_experimental.py",
        "blocks/gqa/block.py",
        "blocks/gqa/kv.py",
        "blocks/gdn/block.py",
        "blocks/gdn/states.py",
        "blocks/moe/block.py",
        "experimental/fused_pool_io.py",
        "models/qwen35.py",
        "models/qwen35_pool.py",
        "pool/gather.py",
        "pool/scatter.py",
        "pool/scatter_ckpt.py",
        "pool/scatter_kv.py",
    }
    assert expected_files <= rel_paths, expected_files - rel_paths
    return files


def test_no_raw_pool_state_literal_in_core():
    """`"raw"` is defined once (`POOL_STATE_NAME`), never repeated elsewhere."""
    assert mk.POOL_STATE_NAME == "raw"
    for path in _guarded_files():
        if path.relative_to(_MK_ROOT).as_posix() == "pool/config.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == "raw":
                raise AssertionError(
                    f"{path.relative_to(_MK_ROOT)}:{node.lineno} uses the pool "
                    "state literal 'raw'; use POOL_STATE_NAME or "
                    "KernelContext.raw_refs"
                )


def test_no_slot_stride_literal_in_core():
    """No `* 5` / `5 *` page-stride literal: strides come from `PoolLayout`."""
    for path in _guarded_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)):
                continue
            for side in (node.left, node.right):
                if isinstance(side, ast.Constant) and side.value == 5:
                    raise AssertionError(
                        f"{path.relative_to(_MK_ROOT)}:{node.lineno} multiplies "
                        "by the literal 5; use PoolLayout.pages_per_slot"
                    )
