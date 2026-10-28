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
"""B1 tests: per-bucket tuning lives in tile rules, enforced numerics classes."""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tests.megakernel.megakernel_fakes import EXACT_FLIPS, tile_field, with_tile_fields
from vllm_torchtpu import megakernel as mk
from vllm_torchtpu.megakernel import profile as profile_mod
from vllm_torchtpu.megakernel.cache import should_use_pool_gather
from vllm_torchtpu.megakernel.models import qwen35
from vllm_torchtpu.megakernel.models.qwen35_profile import TINY_TILES, baseline_tiles


@pytest.fixture(autouse=True)
def _use_cpu_default_device():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def test_per_bucket_thresholds_come_from_tiles():
    """`ExecSpec.WIDE` and the MoE grouped-release threshold read `Tiles`."""
    assert mk.ExecSpec(B=16).WIDE is False and mk.ExecSpec(B=17).WIDE is True
    t = dataclasses.replace(mk.default_tiles(8), wide_rows=4)
    assert mk.ExecSpec(B=8, tiles=t).WIDE is True
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
        att_every=4,
    )
    spec = qwen35.qwen35_spec(g)
    for B, rows, want in ((40, 40, True), (48, 40, False), (8, 4, False)):
        tiles = dataclasses.replace(TINY_TILES, bank=512 << 10).with_modules(
            {"moe": {"moe_group_rows": rows}}
        )
        ex = mk.ExecSpec(B=B, R=2, ctx=256, tiles=tiles, interpret=True)
        plan = mk.planner.plan(spec, ex)
        assert plan.module_params["moe"].group_rel is want, (B, rows)


# ---------------------------------------------------------------------------
# Numerics classes
# ---------------------------------------------------------------------------


def test_profile_numerics_are_derived_and_enforced():
    base = mk.get_profile("qwen35-baseline")
    assert base.tiles_for(20, R=8) == baseline_tiles(20, R=8)
    assert profile_mod.approx_features(mk.DEFAULT_PROFILE) == ()
    assert profile_mod.approx_features(base) == ()  # the Qwen3.5 reference
    # Qwen3.5 profiles are measured against `qwen35-baseline`, not `default`.
    c30 = mk.get_profile("qwen35-candidate30")
    feats = set(profile_mod.approx_features(c30))
    # Exactly its approx-class knobs; exact-class flags it turns on
    # (`pair_kv`, `kv_ering`, `gqa_qv`, `rel_fetch`, `nkv`, ...) are not listed.
    assert feats == {
        "tiles.dense_acc_pair",
        "tiles.dmm_unroll",
        "tiles.gdn.conv_stage",
        "tiles.gdn.gdn_head_major",
        "tiles.gqa.gq_pack",
        "tiles.gqa.kv_sub",
        "tiles.hier",
        "tiles.moe.ecopy_opt",
        "tiles.moe.moe_unroll",
        "tiles.moe.pipe_ep",
        "tiles.moe.r2_direct_store",
        "tiles.moe.route_vec",
        "tiles.moe.topk_lanes",
        "tiles.nbe",
    }
    fused = mk.get_profile("qwen35-candidate30-fused")
    assert fused.numerics == "approx" and fused.fused_pool_io == "auto"
    assert set(profile_mod.approx_features(fused)) == feats | {"fused_pool_io"}

    # A profile declared exact that turns on an approx feature is rejected.
    def _kv_sub_rule(B: int, *, moe_tp: bool = False, R: int = 8) -> mk.Tiles:
        return mk.default_tiles(B, R=R).with_modules({"gqa": {"kv_sub": 1024}})

    with pytest.raises(ValueError, match="tiles.gqa.kv_sub"):
        mk.Profile(name="bad-exact", tiles=_kv_sub_rule)
    with pytest.raises(ValueError, match="fused_pool_io"):
        mk.Profile(name="bad-exact-fused", fused_pool_io="auto")

    # Exact-class changes keep an exact profile exact.
    def _sched_rule(B: int, *, moe_tp: bool = False, R: int = 8) -> mk.Tiles:
        return mk.default_tiles(B, R=R).with_modules(
            {"gqa": {"pair_kv": True, "nkv": 4}}
        )

    assert mk.Profile(name="sched-only", tiles=_sched_rule).numerics == "exact"


def test_model_profile_without_tiles_inherits_the_model_default_rule(monkeypatch):
    """`Profile(model=m, tiles=None)` runs (and is measured against) the tile
    rule of `m`'s default profile, not the library default rule."""
    monkeypatch.setattr(profile_mod, "_REGISTRY", dict(profile_mod._REGISTRY))
    monkeypatch.setattr(
        profile_mod, "_MODEL_DEFAULTS", dict(profile_mod._MODEL_DEFAULTS)
    )

    def _ref_rule(B: int, *, moe_tp: bool = False, R: int = 8) -> mk.Tiles:
        return mk.default_tiles(B, R=R).with_modules({"gqa": {"kv_sub": 1024}})

    ref = mk.register_profile(
        mk.Profile(
            name="tile-rule-ref",
            model="tile_rule_model",
            tiles=_ref_rule,
            model_default=True,
        )
    )
    # Same pool-I/O mode, no tiles: exact relative to the model's default.
    p = mk.Profile(name="tile-rule-user", model="tile_rule_model", buckets=(8,))
    assert p.tile_rule() is _ref_rule and p.numerics == "exact"
    assert profile_mod.approx_features(p) == ()
    assert p.tiles_for(8, R=2) == ref.tiles_for(8, R=2)
    ex = p.exec_for(mk.ExecSpec(R=2), B=8, T=1)
    assert ex.tiles == _ref_rule(8, R=2)
    # Without a model (or a model with no default profile): the library rule,
    # left for the kernel builder's VMEM fit.
    for q in (
        mk.Profile(name="tile-rule-any"),
        mk.Profile(name="tile-rule-other", model="no_default_model"),
    ):
        assert q.tile_rule() is None
        assert q.exec_for(mk.ExecSpec(R=2), B=8, T=1).tiles is None


def test_overrides_relabel_and_never_silently_stay_exact():
    d = mk.DEFAULT_PROFILE
    assert d.with_overrides(fused_pool_io=False, pool_gather=False) is d
    f = d.with_overrides(fused_pool_io="auto")
    assert (f.name, f.numerics, f.fused_pool_io) == (
        "default+overrides",
        "approx",
        "auto",
    )
    g = d.with_overrides(pool_gather=None)  # auto GPR (WIDE only) is exact-class
    assert (g.name, g.numerics) == ("default+overrides", "exact")
    e = d.with_overrides(pool_gather=True)  # GPR on every eligible bucket: approx
    assert (e.name, e.numerics) == ("default+overrides", "approx")
    assert profile_mod.approx_features(e) == ("pool_gather_eligible",)
    c = mk.get_profile("qwen35-candidate30").with_overrides(fused_pool_io="auto")
    assert (c.name, c.numerics) == ("qwen35-candidate30+overrides", "approx")
    assert c.with_overrides(fused_pool_io=True).name == "qwen35-candidate30+overrides"


def test_candidate30_profiles_omit_internal_baseline_provenance():
    assert not hasattr(qwen35.qwen35_candidate30, "CANDIDATE30_BASELINE")
    for name in ("qwen35-baseline", "qwen35-candidate30", "qwen35-candidate30-fused"):
        p = mk.get_profile(name)
        assert p.baseline is None
        assert "hand-written" not in p.description.lower()
        assert "frozen" not in p.description.lower()


# ---------------------------------------------------------------------------
# CPU bit-exactness of exact-class tile fields and exact profiles
# ---------------------------------------------------------------------------


def _cpu_mesh(R: int) -> Mesh:
    devs = jax.devices("cpu")
    if len(devs) < R:
        pytest.skip(f"Need at least {R} CPU devices, have {len(devs)}")
    return Mesh(np.array(devs[:R]), ("tp",))


def _geom(**kw: Any) -> qwen35.Qwen35Geometry:
    base = dict(
        L=4, H=256, I=256, IS=256, E=128, TOPK=2, NH=4, NKV=2, HD=256, ROT=64,
        KH=2, VH=4, KD=128, VD=128, att_every=4,
    )  # fmt: skip
    return qwen35.Qwen35Geometry(**{**base, **kw})


def _tiny(t: mk.Tiles) -> mk.Tiles:
    return dataclasses.replace(
        t,
        bank=512 << 10,
        nbd=10,
        nbe=max(6, min(t.nbe, 12)),
        ebank=256 << 10,
        vmem_mib=32,
    ).with_modules(
        {"gqa": {"kv_blk": 256, "kv_sub": 128 if t.module("gqa").kv_sub else 0}}
    )


_OUT_CACHE: dict[tuple, list] = {}
# `ModelSpec.name` of the CPU test geometry and its default profile.
_QWEN35 = qwen35.qwen35_spec(qwen35.TINY).name
_QWEN35_DEFAULT = mk.model_default_profile(_QWEN35).name


def _flip_exec(
    prof: mk.Profile | str, *, T: int, B: int, geom_kw: dict | None = None, **tile_kw
) -> tuple[Any, mk.ExecSpec]:
    """Geometry + ExecSpec of one tiny paged bucket under `prof` (+ tile overrides)."""
    g = _geom(**(geom_kw or {}))
    base = mk.ExecSpec(
        R=2, B=B, T=T, ctx=256, stack_e=False, interpret=True,
        cache=mk.CacheExec(paged_kv=True, num_pages=512 if B <= 8 else 2048),
    )  # fmt: skip
    ex = mk.get_profile(prof).exec_for(base, B=B, T=T, g=g)
    ex = dataclasses.replace(
        ex, tiles=with_tile_fields(_tiny(ex.resolved_tiles), tile_kw)
    )
    return g, ex


def _pooled_outputs(prof: mk.Profile | str, *, T: int, B: int, **kw: Any) -> list:
    """Bytes of hidden + pools of one tiny paged pooled step under `prof` (cached)."""
    key = (
        prof if isinstance(prof, str) else id(prof),
        T,
        B,
        repr(sorted(kw.items())),
    )
    if key not in _OUT_CACHE:
        _OUT_CACHE[key] = _pooled_outputs_uncached(prof, T=T, B=B, **kw)
    return _OUT_CACHE[key]


def _pooled_outputs_uncached(
    prof: mk.Profile | str, *, T: int, B: int, **kw: Any
) -> list:
    mesh = _cpu_mesh(2)
    g, ex = _flip_exec(prof, T=T, B=B, **kw)
    spec = qwen35.qwen35_spec(g)
    pos, hid_l, w_l, st_l = mk.random_logical_inputs(jax.random.key(31 + T + B), g, ex)
    st_l = dict(st_l, conv=st_l["conv"].astype(jnp.bfloat16).astype(jnp.float32))
    _, hid_mk, w_mk, _ = mk.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex, mesh=mesh, build_default_pool=True
    )
    pt, ckpt, roff, _ = mk.default_pool_layout(g, ex)
    tp = NamedSharding(mesh, P("tp"))
    pools = tuple(
        jax.device_put(np.asarray(r).copy(), tp)
        for r in mk.states_to_pool(g, ex, st_l, pt, ckpt, roff)
    )
    step, _ = mk.PallasWriter(fused_pool_io=False).make_step(
        mesh, spec, ex, donate=False
    )
    out, raw = step(pos, hid_mk, w_mk, pools, pt, ckpt, roff)
    return [np.asarray(x).view(np.uint8) for x in (out, *raw)]


def _assert_same(a: list, b: list) -> None:
    assert len(a) == len(b)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def _plan_effect(field: str, g: Any, ex: mk.ExecSpec) -> Any:
    """The plan / scratch parameter an exact tile field controls (None: budget only)."""
    _, info = qwen35.build(g, ex)
    bp, sp = info["plan"].module_params, info["core_plan"].stream
    return {
        "nbd": sp.NBD,
        "gqa.nkv": bp["gqa"].NKV,
        "gdn.gdn_bufs": bp["gdn"].NRB,
        "lazy": (bp["gqa"].LAZY_ATT, bp["moe"].LAZY_MOE),
        "gqa.pair_kv": bp["gqa"].PAIR_KV,
        "gqa.kv_ering": bp["gqa"].KVE,
        "gqa.gqa_qv": bp["gqa"].QV,
        "rel_fetch": sp.rel_fetch,
        "moe.moe_group_rows": bp["moe"].group_rel,
    }.get(field)


# WIDE re-checks of flags already flipped at (T=2, B=4): local tiers skip them.
_SLOW_FLIPS = {"pair_kv-wide", "kv_ering-wide", "gqa_qv-wide"}


@pytest.mark.parametrize(
    "field,value,tb,extra",
    [
        pytest.param(
            *c[1:], id=c[0], marks=pytest.mark.slow if c[0] in _SLOW_FLIPS else ()
        )
        for c in EXACT_FLIPS
    ],
)
def test_exact_tile_fields_reproduce_model_default(
    field: str, value: Any, tb: tuple[int, int], extra: dict
):
    """Each `exact` tile field, moved alone off the model's default profile
    (`qwen35-baseline`), keeps hidden + pools bit-identical.

    The flip must also reach the plan (asserted below), so a field that is
    silently gated off cannot pass vacuously.
    """
    T, B = tb
    g0, ex0 = _flip_exec(_QWEN35_DEFAULT, T=T, B=B, **extra)
    g1, ex1 = _flip_exec(_QWEN35_DEFAULT, T=T, B=B, **extra, **{field: value})
    assert tile_field(ex1.resolved_tiles, field) != tile_field(
        ex0.resolved_tiles, field
    )
    before, after = _plan_effect(field, g0, ex0), _plan_effect(field, g1, ex1)
    if before is not None:
        assert before != after, (field, before, after)
    if "moe.pipe_ep" in extra:
        _, info = qwen35.build(g1, ex1)
        assert info["plan"].module_params["moe"].PIPE_EP
    ref = _pooled_outputs(_QWEN35_DEFAULT, T=T, B=B, **extra)
    _assert_same(
        _pooled_outputs(_QWEN35_DEFAULT, T=T, B=B, **extra, **{field: value}), ref
    )


def test_exact_kv_scatter_vec_ns_non_paged():
    """`kv_scatter_vec_ns` only switches the host KV update between equivalent forms."""
    R, B = 2, 4
    mesh = _cpu_mesh(R)
    g = _geom()
    outs = []
    for ns in (16, 0):
        tiles = dataclasses.replace(TINY_TILES, bank=512 << 10).with_modules(
            {"gqa": {"kv_scatter_vec_ns": ns}}
        )
        ex = mk.ExecSpec(B=B, T=1, R=R, ctx=256, tiles=tiles, interpret=True)
        pos, hid_l, w_l, st_l = mk.random_logical_inputs(jax.random.key(5), g, ex)
        pos_mk, hid_mk, w_mk, st_mk = mk.pack_inputs(
            pos, hid_l, w_l, st_l, g, ex, mesh=mesh
        )
        step, _ = qwen35.make_step(mesh, g, ex)
        out_hid, out_st = step(pos_mk, hid_mk, w_mk, st_mk)
        outs.append(
            [
                np.asarray(x).view(np.uint8)
                for x in (out_hid, out_st["kc"], out_st["vc"])
            ]
        )
    _assert_same(outs[1], outs[0])


def _exact_sched_rule(B: int, *, moe_tp: bool = False, R: int = 8) -> mk.Tiles:
    """Test-only tile rule: the Qwen3.5 baseline plus exact-class schedule changes."""
    return dataclasses.replace(
        baseline_tiles(B, moe_tp=moe_tp, R=R), rel_fetch=True
    ).with_modules(
        {
            "gqa": {"pair_kv": True, "kv_ering": True, "gqa_qv": True, "nkv": 4},
            "moe": {"moe_group_rows": 0},
        }
    )


_EXACT_SCHED_PROFILE = mk.Profile(
    name="test-exact-sched",
    model=_QWEN35,
    numerics="exact",
    tiles=_exact_sched_rule,
    description="Test-only: default numerics, exact-class schedule flags on.",
)


def test_every_exact_profile_reproduces_its_reference(
    monkeypatch: pytest.MonkeyPatch,
):
    """Every registered `exact` profile matches its reference bit-exactly (CPU).

    The reference is the default profile of the profile's model
    (`model_default_profile`); model default profiles are their own reference.
    A test-only exact Qwen3.5 profile that turns on exact-class schedule flags
    is registered so the check is not vacuous.
    """
    monkeypatch.setitem(
        profile_mod._REGISTRY, _EXACT_SCHED_PROFILE.name, _EXACT_SCHED_PROFILE
    )
    exact = [n for n in mk.list_profiles() if mk.get_profile(n).numerics == "exact"]
    assert {"default", _QWEN35_DEFAULT, _EXACT_SCHED_PROFILE.name} <= set(exact)
    checked = 0
    for name in exact:
        ref_name = mk.model_default_profile(mk.get_profile(name).model).name
        if ref_name == name:
            continue
        assert ref_name == _QWEN35_DEFAULT, name  # only Qwen3.5 profiles here
        for T, B in ((1, 8), (2, 4)):
            g0, ex0 = _flip_exec(ref_name, T=T, B=B)
            g1, ex1 = _flip_exec(name, T=T, B=B)
            assert ex0.resolved_tiles != ex1.resolved_tiles, name
            _assert_same(
                _pooled_outputs(name, T=T, B=B), _pooled_outputs(ref_name, T=T, B=B)
            )
            checked += 1
    assert checked


# ---------------------------------------------------------------------------
# pool_gather semantics and config errors
# ---------------------------------------------------------------------------


_GPR_ELIGIBLE = mk.get_profile(_QWEN35_DEFAULT).with_overrides(pool_gather=True)
_GPR_AUTO = mk.get_profile(_QWEN35_DEFAULT).with_overrides(pool_gather=None)


@pytest.mark.parametrize(
    "prof,B",
    [
        (_GPR_ELIGIBLE, 8),
        (_GPR_ELIGIBLE, 16),
        # The golden harness also pins GPR at B=24, there with an explicit
        # `pool_gather=True` (def_t4_b24_gpr).
        pytest.param(_GPR_AUTO, 24, marks=pytest.mark.slow),
    ],
)
def test_gpr_matches_split_bitwise_t4(prof: mk.Profile, B: int):
    """In-kernel GDN gather (GPR) vs split gather at T=4, Qwen3.5 baseline (CPU).

    Bitwise with baseline tiles on non-WIDE (B=8, 16) and WIDE (B=24) buckets.
    With qwen35-candidate30 tiles GPR is bitwise only on WIDE buckets (pools
    differ at B=8, hidden and pools at B=16), which is why
    `pool_gather=True` ("every eligible bucket") is classed approx and auto
    GPR (WIDE only) exact in `pool.POOL_IO_NUMERICS`.
    """
    g, ex = _flip_exec(prof, T=4, B=B)
    spec = qwen35.qwen35_spec(g)
    assert should_use_pool_gather(spec, spec.cache_backend, ex, use_fused=False)
    _assert_same(
        _pooled_outputs(prof, T=4, B=B), _pooled_outputs(_QWEN35_DEFAULT, T=4, B=B)
    )


def test_profile_pool_gather_true_means_where_eligible():
    g = _geom()
    spec = qwen35.qwen35_spec(g)
    pb = spec.cache_backend
    prof = mk.DEFAULT_PROFILE.with_overrides(pool_gather=True)
    base = mk.ExecSpec(
        R=2, ctx=256, interpret=True, cache=mk.CacheExec(paged_kv=True, num_pages=512)
    )
    ex1 = prof.exec_for(base, B=4, T=1, g=g)
    assert (
        mk.cache_exec(ex1).pool_gather == "eligible" and mk.cache_exec(ex1).GPR is False
    )
    assert should_use_pool_gather(spec, pb, ex1, use_fused=False) is False  # no raise
    ex4 = prof.exec_for(base, B=24, T=4, g=g)
    assert should_use_pool_gather(spec, pb, ex4, use_fused=False) is True
    # An explicit (strict) request on an ineligible bucket is a config error.
    with pytest.raises(mk.MegakernelConfigError, match="pool_gather=True"):
        should_use_pool_gather(
            spec, pb, mk.with_cache(ex1, pool_gather=True), use_fused=False
        )
    with pytest.raises(ValueError, match="pool_gather"):
        mk.Profile(name="bad-pg", pool_gather="eligible")  # type: ignore[arg-type]


def test_exec_for_explicit_base_settings_win():
    g = _geom()
    gpr = mk.DEFAULT_PROFILE.with_overrides(pool_gather=None)
    base = mk.ExecSpec(R=2, ctx=256, interpret=True, cache=mk.CacheExec(paged_kv=True))
    assert mk.cache_exec(gpr.exec_for(base, B=24, T=4, g=g)).pool_gather is None
    for explicit in (True, "eligible"):
        b2 = mk.with_cache(base, pool_gather=explicit)
        ex_b2 = mk.DEFAULT_PROFILE.exec_for(b2, B=24, T=4, g=g)
        assert mk.cache_exec(ex_b2).pool_gather == explicit
    t = TINY_TILES
    assert gpr.exec_for(dataclasses.replace(base, tiles=t), B=8, T=1, g=g).tiles is t


@pytest.mark.parametrize("field,match", [("vmem_mib", "VMEM"), ("smem_mib", "SMEM")])
def test_scratch_budget_overflow_is_config_error(field: str, match: str):
    """VMEM / SMEM budget overflow is a configuration error, not a layout error."""
    tiles = dataclasses.replace(_tiny(mk.default_tiles(4, R=2)), **{field: 0})
    ex = mk.ExecSpec(
        R=2,
        B=4,
        T=1,
        ctx=256,
        interpret=True,
        tiles=tiles,
        cache=mk.CacheExec(paged_kv=True, num_pages=512),
    )
    with pytest.raises(mk.MegakernelConfigError, match=match):
        qwen35.build(_geom(), ex)
    assert not issubclass(mk.MegakernelConfigError, ValueError)


def test_builtin_profiles_resolve_in_a_fresh_interpreter():
    """`get_profile` works before any profile module was imported directly."""
    code = (
        "from vllm_torchtpu import megakernel as mk; "
        "print(mk.get_profile('qwen35-candidate30-fused').name); "
        "from vllm_torchtpu.megakernel import models; "
        "print(models.qwen35_candidate30.PROFILE.name)"
    )
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.split() == ["qwen35-candidate30-fused", "qwen35-candidate30"]


def test_builtin_profile_registration_is_idempotent():
    c30 = mk.get_profile("qwen35-candidate30")
    assert mk.register_profile(c30) is c30
    assert mk.register_profile(dataclasses.replace(c30)) is c30  # equal: no-op
    with pytest.raises(ValueError, match="already registered"):
        mk.register_profile(dataclasses.replace(c30, description="different"))


def test_exec_spec_pool_gather_values_and_explicit_split():
    with pytest.raises(ValueError, match="pool_gather must be one of"):
        mk.ExecSpec(cache=mk.CacheExec(pool_gather="sometimes"))  # type: ignore[arg-type]
    g = _geom()
    spec = qwen35.qwen35_spec(g)
    gpr = mk.DEFAULT_PROFILE.with_overrides(pool_gather=None)
    base = mk.ExecSpec(
        R=2, ctx=256, interpret=True, cache=mk.CacheExec(paged_kv=True, num_pages=2048)
    )
    ex_auto = gpr.exec_for(base, B=24, T=4, g=g)
    assert should_use_pool_gather(spec, spec.cache_backend, ex_auto, use_fused=False)
    # An explicit "split" wins over a GPR profile; the default False does not.
    ex_split = gpr.exec_for(mk.with_cache(base, pool_gather="split"), B=24, T=4, g=g)
    assert (
        mk.cache_exec(ex_split).pool_gather == "split"
        and mk.cache_exec(ex_split).GPR is False
    )
    assert not should_use_pool_gather(
        spec, spec.cache_backend, ex_split, use_fused=False
    )


def test_gq_pack_without_gqa_qv_is_config_error():
    t = _tiny(mk.default_tiles(8, R=2)).with_modules({"gqa": {"gq_pack": True}})
    ex = mk.ExecSpec(
        R=2,
        B=8,
        T=4,
        ctx=256,
        interpret=True,
        tiles=t,
        cache=mk.CacheExec(paged_kv=True, num_pages=512),
    )
    with pytest.raises(mk.MegakernelConfigError, match="gq_pack requires gqa_qv"):
        mk.planner.plan(qwen35.qwen35_spec(_geom()), ex)
