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
"""Qwen3.5 baseline vs the frozen kernel, named profiles, safety guards (CPU)."""

from __future__ import annotations

import dataclasses
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

import vllm_torchtpu.megakernel as mk
from tests.megakernel.frozen_utils import load_frozen_qwen35
from vllm_torchtpu.megakernel.models import qwen35
from vllm_torchtpu.megakernel.models.qwen35_candidate30 import (
    candidate30_tiles,
)
from vllm_torchtpu.megakernel.models.qwen35_profile import TINY_TILES, baseline_tiles


def _tile_value(t: mk.Tiles, name: str) -> object:
    """Field `name` of `t`, looked up in core fields, then each block's tunables."""
    if name in {f.name for f in dataclasses.fields(t)}:
        return getattr(t, name)
    owners = [
        b.name
        for b in (mk.GQABlock, mk.GDNBlock, mk.MoEBlock)
        if name in getattr(b.Tunables, "NUMERICS", {})
    ]
    assert len(owners) == 1, (name, owners)
    return getattr(t.block(owners[0]), name)


def test_qwen35_baseline_tiles_match_frozen_kernel_schedule():
    """`qwen35-baseline` tiles reproduce `frozen_qwen35.megakernel.default_tiles`.

    The only deviation is `nbe=6` for `B > 48` (numerics-neutral ring depth kept
    since M2; the frozen TPU benches apply the same override).
    """
    frozen_mk, _ = load_frozen_qwen35()
    frozen_fields = {f.name for f in dataclasses.fields(frozen_mk.Tiles)}
    for B in range(1, 65):
        c = frozen_mk.MKConfig(R=8, B=B, T=1, L=4, att_every=4, ctx=4096)
        tf = frozen_mk.default_tiles(c)
        if B > 48:
            tf = dataclasses.replace(tf, nbe=6)
        tl = mk.get_profile("qwen35-baseline").tiles_for(B, R=8)
        for name in frozen_fields:
            assert _tile_value(tl, name) == getattr(tf, name), (B, name)


def test_profile_registry_and_default_profile():
    assert {"default", "qwen35-baseline", "qwen35-candidate30"} <= set(
        mk.list_profiles()
    )
    d = mk.get_profile(None)
    assert d is mk.DEFAULT_PROFILE and mk.get_profile("") is d
    assert d.numerics == "exact" and d.fused_pool_io is False
    assert d.pool_gather is False and d.tiles is None and d.model is None
    for B in (1, 4, 8, 16, 24, 32, 48, 64):
        assert d.tiles_for(B, R=8) == mk.default_tiles(B, R=8)
    # Models without a declared default profile fall back to `default`.
    assert mk.model_default_profile(None) is d
    assert mk.model_default_profile("no-such-model") is d
    with pytest.raises(KeyError, match="unknown megakernel profile"):
        mk.get_profile("nope")
    with pytest.raises(ValueError, match="already registered"):
        mk.register_profile(mk.Profile(name="default"))
    with pytest.raises(ValueError, match="numerics"):
        mk.Profile(name="x", numerics="bitwise")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fused_pool_io"):
        mk.Profile(name="x", fused_pool_io="sometimes")


def test_qwen35_declares_baseline_as_its_default_profile():
    name = qwen35.qwen35_spec(qwen35.FULL).name
    b = mk.model_default_profile(name)
    assert b is mk.get_profile("qwen35-baseline")
    assert b.model == name and b.model_default and b.numerics == "exact"
    assert b.fused_pool_io is False and b.pool_gather is False
    assert b.tiles is baseline_tiles
    # Only one default per model; overridden copies are never the default.
    with pytest.raises(ValueError, match="already has default profile"):
        mk.register_profile(dataclasses.replace(b, name="other-default"))
    assert not b.with_overrides(pool_gather=None).model_default
    with pytest.raises(ValueError, match="model default"):
        mk.Profile(name="x", model_default=True)


def test_candidate30_profile_carries_m5_tuning():
    p = mk.get_profile("qwen35-candidate30")
    assert p.model == qwen35.qwen35_spec(qwen35.FULL).name
    assert p.numerics == "approx"
    assert p.pool_gather is None  # GPR auto
    assert p.fused_pool_io is False  # experimental: explicit opt-in only
    t4 = p.tiles_for(4, R=8)
    assert (t4.block("gqa").kv_sub, t4.nbe, t4.hier, t4.block("moe").ecopy_opt) == (
        1024,
        20,
        1,
        1,
    )
    for flag in (
        "pipe_ep",
        "r2_direct_store",
        "route_vec",
        "dense_acc_pair",
        "kv_ering",
        "pair_kv",
        "gq_pack",
        "topk_lanes",
        "gdn_head_major",
        "gqa_qv",
        "moe_unroll",
        "rel_fetch",
    ):
        assert _tile_value(t4, flag) is True, flag
    assert t4.dmm_unroll == 8
    assert p.tiles_for(4, R=8, moe_tp=True).nbe == 10
    t64 = p.tiles_for(64, R=8)
    assert (t64.nbd, t64.nbe, t64.hier) == (7, 12, 2)
    t56 = p.tiles_for(56, R=8)
    assert (t56.nbd, t56.nbe, t56.hier) == (8, 12, 2)
    t16 = p.tiles_for(16, R=8)
    assert (t16.block("gqa").nkv, t16.rsag, t16.hier, t16.nbe) == (4, True, 2, 12)
    # HIER3 remap for power-of-2 R != 8 (B <= 32), flat ring otherwise
    assert candidate30_tiles(16, R=4).hier == 3
    assert candidate30_tiles(64, R=4).hier == 0
    assert candidate30_tiles(16, R=6).hier == 0


@pytest.mark.parametrize("moe_tp", [False, True])
@pytest.mark.parametrize("R", [8, 4])
def test_candidate30_conv_stage_only_from_b8(R: int, moe_tp: bool):
    """Fused-path conv staging is on from B >= 8 only, as in candidate30's kernel."""
    for B in (4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64):
        t = candidate30_tiles(B, moe_tp=moe_tp, R=R)
        assert t.block("gdn").conv_stage is (B >= 8), (B, R, moe_tp)


def test_profile_exec_for_respects_explicit_tiles():
    p = mk.get_profile("qwen35-candidate30")
    base = mk.ExecSpec(R=8, cache=mk.CacheExec(paged_kv=True))
    ex = p.exec_for(base, B=16, T=4, g=qwen35.FULL_EP)
    assert (ex.B, ex.T) == (16, 4)
    assert ex.tiles == candidate30_tiles(16, R=8, moe_tp=False)
    assert mk.cache_exec(ex).pool_gather is None
    base_t = replace(base, tiles=TINY_TILES)
    assert p.exec_for(base_t, B=8, T=1).tiles is TINY_TILES
    # The library default leaves tiles unset: the kernel builder fits them.
    d = mk.get_profile("default").exec_for(base, B=8, T=1)
    assert d.tiles is None and d.resolved_tiles == mk.default_tiles(8)
    assert mk.cache_exec(d).pool_gather is False
    q = mk.get_profile("qwen35-baseline").exec_for(base, B=8, T=1)
    assert q.tiles == baseline_tiles(8, R=8)


def _tiny_pool_setup(R: int = 1, B: int = 2, T: int = 1):
    g = qwen35.Qwen35Geometry(
        L=4,
        H=256,
        I=256,
        IS=256,
        E=128,
        TOPK=2,
        NH=4,
        NKV=1,
        HD=256,
        ROT=64,
        KH=2,
        VH=4,
        KD=128,
        VD=128,
        att_every=4,
        moe_tp=False,
    )
    tiles = replace(TINY_TILES, bank=512 << 10)
    ex = mk.ExecSpec(
        R=R,
        B=B,
        T=T,
        ctx=256,
        stack_e=False,
        interpret=True,
        tiles=tiles,
        cache=mk.CacheExec(paged_kv=True, num_pages=64),
    )
    mesh = Mesh(np.array(jax.devices("cpu")[:R]), ("tp",))
    return g, ex, mesh


def test_fused_pool_io_requires_pool_donation():
    g, ex, mesh = _tiny_pool_setup()
    spec = qwen35.qwen35_spec(g)
    with pytest.raises(mk.MegakernelConfigError, match="requires donate_pools=True"):
        mk.make_pooled_step(mesh, spec, ex, donate_pools=False, fused_pool_io=True)
    with pytest.raises(mk.MegakernelConfigError, match="requires donate_pools=True"):
        mk.make_flat_pooled_step(mesh, spec, ex, donate_pools=False, fused_pool_io=True)
    with pytest.raises(mk.MegakernelConfigError, match="requires donate_pools=True"):
        mk.PallasWriter(fused_pool_io=True).make_step(mesh, spec, ex, donate=False)
    # "auto" selecting fused at this B is held to the same rule
    with pytest.raises(mk.MegakernelConfigError, match="requires donate_pools=True"):
        mk.make_pooled_step(mesh, spec, ex, donate_pools=False, fused_pool_io="auto")
    # Split pool I/O does not need donation
    mk.make_pooled_step(mesh, spec, ex, donate_pools=False, fused_pool_io=False)


def test_ckpt_bounds_host_check_is_debug_opt_in():
    g, ex, mesh = _tiny_pool_setup()
    spec = qwen35.qwen35_spec(g)
    pt, ckpt, roff, _ = mk.default_pool_layout(g, ex)
    bad_ckpt = ckpt + mk.cache_exec(ex).num_pages  # every slot maps past the pool
    dummy = (jnp.zeros((1,), jnp.int32),) * 3
    ex_dbg = mk.with_cache(ex, check_ckpt_bounds=True)
    step_dbg = mk.make_pooled_step(mesh, spec, ex_dbg)
    raw = tuple(
        jnp.zeros(
            (mk.cache_exec(ex).num_pages, 2, g.HD // 4, 4, mk.cache_exec(ex).page_size),
            jnp.float8_e4m3fn,
        )
        for _ in range(g.NATT)
    )
    with pytest.raises(ValueError, match="maps to pages"):
        step_dbg(*dummy, raw, pt, bad_ckpt, roff)
