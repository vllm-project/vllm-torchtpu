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
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tests.megakernel.megakernel_fakes import (
    TINY_TILES,
    SynthHybrid,
    synth_hybrid_spec,
)
from vllm_torchtpu.megakernel import cache, planner, pool
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.modules.gdn import GDNModule
from vllm_torchtpu.megakernel.ops import F32
from vllm_torchtpu.megakernel.pool.config import CacheExec, cache_exec
from vllm_torchtpu.megakernel.reference import spec_reference_step
from vllm_torchtpu.megakernel.weights.eplb import (
    default_eplb_cores,
    encode_eplb_ecopy,
)
from vllm_torchtpu.megakernel.weights.spec_inputs import (
    pack_spec_inputs,
    random_spec_inputs,
)


def _make_paged_ex(B: int, T: int, R: int = 2, *, gpr: Any = False) -> ExecSpec:
    tiles = TINY_TILES.with_modules({"gqa": {"kv_blk": 128, "nkv": 2}})
    return ExecSpec(
        B=B,
        T=T,
        R=R,
        ctx=256,
        stack_e=True,
        tiles=tiles,
        interpret=True,
        cache=CacheExec(paged_kv=True, num_pages=64, pool_gather=gpr),
    )


def test_cache_layout_helpers_and_direct_hbm_pipe_on_synth_spec():
    """`cache` pool helpers and `DirectHBM` work directly on `synth_hybrid_spec`."""
    R = 2
    spec = synth_hybrid_spec(L=2, att_every=2, R=R)
    ex = _make_paged_ex(B=4, T=1, R=R)
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

    mesh = Mesh(np.array(jax.devices("cpu")[:R]), ("tp",))
    step, info = cache.DirectHBM().make_step(mesh, spec, ex)
    assert callable(step) and not cache_exec(info["exec"]).paged_kv


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
    """`PallasWriter` (`make_pooled_step`) and `make_flat_pooled_step` match oracle."""
    R = 2
    spec = synth_hybrid_spec(L=2, att_every=2, R=R)
    if conv_f32:
        assert spec.cache_backend is not None
        lay_f32 = dataclasses.replace(spec.cache_backend.layout, conv_dtype=F32)
        pb_f32 = pool.make_pool_backend(lay_f32, rec_chunk=GDNModule.chunk_rows)
        spec = dataclasses.replace(spec, cache_backend=pb_f32)
    ex = _make_paged_ex(B=B, T=T, R=R, gpr=gpr)
    mesh = Mesh(np.array(jax.devices("cpu")[:R]), ("tp",))
    rep, tp = NamedSharding(mesh, P()), NamedSharding(mesh, P("tp"))

    pos, hid, w_log, st_log = random_spec_inputs(jax.random.PRNGKey(20 + T), spec, ex)
    pos_mk, hid_mk, w_mk, _ = pack_spec_inputs(
        pos, hid, w_log, st_log, spec, ex, mesh=mesh
    )
    pt, ckpt, roff, _ = cache.default_pool_layout(spec, ex)
    raw = cache.states_to_pool(spec, ex, st_log, pt, ckpt, read_offsets=roff)
    raw_sh = tuple(jax.device_put(p, tp) for p in raw)
    pt_sh = jax.device_put(pt, rep)
    ckpt_sh = jax.device_put(ckpt, rep)
    roff_sh = jax.device_put(roff, rep)

    pooled_step = cache.PallasWriter().make_step(mesh, spec, ex)
    out_hid_p, out_raw_p = pooled_step(
        pos_mk, hid_mk, w_mk, raw_sh, pt_sh, ckpt_sh, roff_sh
    )
    jax.block_until_ready(out_hid_p)

    ref_hid, ref_st = spec_reference_step(
        spec,
        spec.config,
        ex,
        pos,
        hid,
        w_log,
        st_log,
        paged_kv=True,
        pool_io=not conv_f32,
    )
    out_hid_r = out_hid_p.reshape(R, ex.M, spec.config.H)[:, :B]
    np.testing.assert_allclose(
        out_hid_r[0].astype(F32), ref_hid.astype(F32), rtol=3e-2, atol=3e-2
    )
    unpacked_p = cache.pool_to_states(spec, ex, out_raw_p, pt, ckpt)
    rec_key = "rec_ckpt" if T > 1 else "rec"
    np.testing.assert_allclose(
        unpacked_p["rec"].astype(F32),
        ref_st[rec_key].astype(F32),
        rtol=3e-2,
        atol=3e-2,
    )

    if T == 1 and not conv_f32:
        raw_sh2 = tuple(jax.device_put(p, tp) for p in raw)
        flat_fn, _, _, donated = cache.make_flat_pooled_step(
            mesh, spec, ex, donate_pools=False
        )
        assert donated == ()
        w_keys = planner.plan(spec, ex).w_names
        w_flat = [w_mk[k] for k in w_keys]
        flat_outs = flat_fn(pos_mk, hid_mk, *w_flat, *raw_sh2, pt_sh, ckpt_sh, roff_sh)
        jax.block_until_ready(flat_outs[0])
        np.testing.assert_array_equal(flat_outs[0], out_hid_p)


def test_check_pooled_step_hlo_and_eplb_helpers():
    """`check_pooled_step_hlo` inspects lowered HLO and EPLB helpers validate maps."""
    R = 2
    spec = synth_hybrid_spec(L=2, att_every=2, R=R)
    ex = _make_paged_ex(B=4, T=1, R=R)
    mesh = Mesh(np.array(jax.devices("cpu")[:R]), ("tp",))
    rep, tp = NamedSharding(mesh, P()), NamedSharding(mesh, P("tp"))
    pos, hid, w_log, st_log = random_spec_inputs(jax.random.PRNGKey(9), spec, ex)
    pos_mk, hid_mk, w_mk, _ = pack_spec_inputs(
        pos, hid, w_log, st_log, spec, ex, mesh=mesh
    )
    pt, ckpt, roff, _ = cache.default_pool_layout(spec, ex)
    raw = cache.states_to_pool(spec, ex, st_log, pt, ckpt, read_offsets=roff)
    raw_sh = tuple(jax.device_put(p, tp) for p in raw)
    pooled_step = cache.make_pooled_step(mesh, spec, ex)
    hlo = pooled_step.lower(
        pos_mk,
        hid_mk,
        w_mk,
        raw_sh,
        jax.device_put(pt, rep),
        jax.device_put(ckpt, rep),
        jax.device_put(roff, rep),
    ).as_text()
    ok, msg = cache.check_pooled_step_hlo(hlo, layout=spec.cache_backend.layout)
    assert ok, msg

    bad_hlo = (
        hlo + "\n%bad = f8e4m3fn[1,64,2,64,4,128] copy(f8e4m3fn[1,64,2,64,4,128] %p)"
    )
    ok_bad, msg_bad = cache.check_pooled_step_hlo(
        bad_hlo, layout=spec.cache_backend.layout
    )
    assert not ok_bad and "full-pool" in msg_bad.lower()

    g_eplb = SynthHybrid(L=2, E=16, moe_tp=False, e_slots=10)
    e_cores = default_eplb_cores(g_eplb, R=2)
    assert len(e_cores) == 2 and len(e_cores[0]) == 2 and len(e_cores[0][0]) == 10
    ecopy = encode_eplb_ecopy(g_eplb, 2, e_cores)
    assert ecopy.shape == (2, 16)
