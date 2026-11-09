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
"""Unit tests for `profile` (named execution profiles, numerics classes, registry)."""

from __future__ import annotations

import ast
import dataclasses
import sys
import types
from pathlib import Path

import pytest

from vllm_torchtpu.megakernel import module_options, profile
from vllm_torchtpu.megakernel.exec_spec import MAX_BATCH_ROWS, ExecSpec, Tiles
from vllm_torchtpu.megakernel.pool.config import CacheExec, cache_exec, with_cache
from vllm_torchtpu.megakernel.profile import (
    DEFAULT_PROFILE,
    Baseline,
    Profile,
    approx_features,
    get_profile,
    list_profiles,
    model_default_profile,
    register_profile,
)
from vllm_torchtpu.megakernel.profiles.default import default_tiles


@dataclasses.dataclass(frozen=True)
class _TestTunables:
    """Test-only module tunables (registered under `_TEST_MODULE`)."""

    width: int = 128
    fast: bool = False
    NUMERICS = {"width": "approx", "fast": "exact"}


_TEST_MODULE = "test_profile_module"


@pytest.fixture(autouse=True)
def _isolated_profile_state(monkeypatch: pytest.MonkeyPatch):
    """Register `_TestTunables` and isolate profile registry state per test."""
    monkeypatch.setitem(module_options._TUNABLES, _TEST_MODULE, _TestTunables)
    monkeypatch.setattr(profile, "_REGISTRY", {"default": DEFAULT_PROFILE})
    monkeypatch.setattr(profile, "_MODEL_DEFAULTS", {})


def test_profile_imports_only_exec_spec_pool_and_profiles():
    """`profile.py` imports only stdlib, `exec_spec`, `pool`, and `profiles`."""
    path = Path(profile.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] in sys.stdlib_module_names
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level == 0:
                assert mod.split(".")[0] in sys.stdlib_module_names
            else:
                assert node.level == 1
                roots = [mod.split(".")[0]] if mod else [a.name for a in node.names]
                assert set(roots) <= {"exec_spec", "pool", "profiles"}, roots


def test_default_profile_and_baseline_metadata():
    d = DEFAULT_PROFILE
    assert d.name == "default"
    assert d.model is None
    assert d.numerics == "exact"
    assert d.tiles is None
    assert d.buckets is None and d.verify_buckets is None
    assert d.fused_pool_io is False
    assert d.pool_gather is False
    assert d.baseline is None
    assert d.model_default is False
    assert approx_features(d) == ()
    assert d.tile_rule() is None
    for B in (1, 4, 8, 16, 24, 32, 48, 64):
        assert d.tiles_for(B, R=8) == default_tiles(B, R=8)

    b = Baseline(ref="v1", commit="abc123", job="j-1", config="tp8", bench="suite_a")
    p = Profile(
        name="with-baseline",
        buckets=(1, 4, 8),
        verify_buckets=(16, 32),
        baseline=b,
        description="Reference provenance test.",
    )
    assert p.baseline == b
    assert p.buckets == (1, 4, 8)
    assert p.verify_buckets == (16, 32)
    assert Baseline(ref="minimal") == Baseline(
        ref="minimal", commit="", job="", config="", bench=""
    )


def test_profile_validation_and_numerics_enforcement():
    for bad_name in ("", "  ", " leading", "trailing "):
        with pytest.raises(ValueError, match="invalid profile name"):
            Profile(name=bad_name)
    with pytest.raises(ValueError, match="numerics must be 'exact' or 'approx'"):
        Profile(name="bad-num", numerics="bitwise")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fused_pool_io must be a bool or 'auto'"):
        Profile(name="bad-fused", fused_pool_io="sometimes")
    with pytest.raises(ValueError, match="fused_pool_io must be a bool or 'auto'"):
        Profile(name="bad-fused-int", fused_pool_io=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="pool_gather must be False, None or True"):
        Profile(name="bad-pg", pool_gather="eligible")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="pool_gather must be False, None or True"):
        Profile(name="bad-pg-int", pool_gather=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="model default"):
        Profile(name="no-model-default", model_default=True)
    with pytest.raises(ValueError, match="model default"):
        Profile(
            name="approx-model-default",
            model="test_model",
            numerics="approx",
            model_default=True,
        )

    def _approx_module_rule(B: int, *, moe_tp: bool = False, R: int = 8) -> Tiles:
        return default_tiles(B, moe_tp=moe_tp, R=R).with_modules(
            {_TEST_MODULE: {"width": 256}}
        )

    def _approx_core_rule(B: int, *, moe_tp: bool = False, R: int = 8) -> Tiles:
        return dataclasses.replace(
            default_tiles(B, moe_tp=moe_tp, R=R), dense_acc_pair=True
        )

    with pytest.raises(ValueError, match=f"tiles.{_TEST_MODULE}.width"):
        Profile(name="bad-exact-module", tiles=_approx_module_rule)
    with pytest.raises(ValueError, match=r"tiles\.dense_acc_pair"):
        Profile(name="bad-exact-core", tiles=_approx_core_rule)

    def _approx_high_b_rule(B: int, *, moe_tp: bool = False, R: int = 8) -> Tiles:
        t = default_tiles(B, moe_tp=moe_tp, R=R)
        return dataclasses.replace(t, dense_acc_pair=True) if B == MAX_BATCH_ROWS else t

    with pytest.raises(ValueError, match=r"tiles\.dense_acc_pair"):
        Profile(name="bad-exact-b128", tiles=_approx_high_b_rule)
    with pytest.raises(ValueError, match="fused_pool_io"):
        Profile(name="bad-exact-fused", fused_pool_io="auto")
    with pytest.raises(ValueError, match="pool_gather_eligible"):
        Profile(name="bad-exact-pg", pool_gather=True)

    # Exact-class changes keep an exact profile exact.
    def _exact_sched_rule(B: int, *, moe_tp: bool = False, R: int = 8) -> Tiles:
        return dataclasses.replace(
            default_tiles(B, moe_tp=moe_tp, R=R), nbd=10, rel_fetch=True
        ).with_modules({_TEST_MODULE: {"fast": True}})

    exact_prof = Profile(name="sched-only", tiles=_exact_sched_rule, pool_gather=None)
    assert exact_prof.numerics == "exact"
    assert approx_features(exact_prof) == ()

    # Turning off an approx pool-I/O mode enabled on the model default is also detected.
    register_profile(
        Profile(
            name="m-approx-ref",
            model="m-io",
            numerics="exact",
            model_default=True,
            pool_gather=True,
            fused_pool_io="auto",
        )
    )
    with pytest.raises(ValueError, match="fused_pool_io.*pool_gather_eligible"):
        Profile(
            name="m-off-exact",
            model="m-io",
            numerics="exact",
            pool_gather=False,
            fused_pool_io=False,
        )


def test_approx_features_against_default_and_model_reference():
    def _ref_rule(B: int, *, moe_tp: bool = False, R: int = 8) -> Tiles:
        return default_tiles(B, moe_tp=moe_tp, R=R).with_modules(
            {_TEST_MODULE: {"width": 256}}
        )

    ref = register_profile(
        Profile(
            name="model-ref",
            model="test_model",
            tiles=_ref_rule,
            model_default=True,
        )
    )
    # A model default profile is its own reference, even with approx knobs set.
    assert approx_features(ref) == ()

    def _tuned_rule(B: int, *, moe_tp: bool = False, R: int = 8) -> Tiles:
        base = _ref_rule(B, moe_tp=moe_tp, R=R).with_modules(
            {_TEST_MODULE: {"fast": True}}
        )
        if B == 64 and R == 16 and moe_tp:
            return dataclasses.replace(base, dense_acc_pair=True, dmm_unroll=8)
        return base

    tuned = Profile(
        name="model-tuned",
        model="test_model",
        numerics="approx",
        tiles=_tuned_rule,
        fused_pool_io="auto",
        pool_gather=True,
    )
    # Measured against `model-ref`: `width=256` matches the reference and `fast`
    # is exact-class, so only the delta approx features are reported.
    assert approx_features(tuned) == (
        "fused_pool_io",
        "pool_gather_eligible",
        "tiles.dense_acc_pair",
        "tiles.dmm_unroll",
    )


def test_with_overrides_relabels_and_derives_numerics():
    d = DEFAULT_PROFILE
    assert d.with_overrides(fused_pool_io=False, pool_gather=False) is d

    f = d.with_overrides(fused_pool_io="auto")
    assert (f.name, f.numerics, f.fused_pool_io) == (
        "default+overrides",
        "approx",
        "auto",
    )
    # Chained overrides do not duplicate the `+overrides` suffix.
    f2 = f.with_overrides(fused_pool_io=True)
    assert (f2.name, f2.numerics, f2.fused_pool_io) == (
        "default+overrides",
        "approx",
        True,
    )
    # An originally `approx` profile never becomes `exact` via `with_overrides`.
    f_back = f.with_overrides(fused_pool_io=False)
    assert f_back.numerics == "approx"

    g = d.with_overrides(pool_gather=None)  # auto GPR (WIDE only) is exact-class
    assert (g.name, g.numerics) == ("default+overrides", "exact")

    e = d.with_overrides(pool_gather=True)  # GPR on every eligible bucket: approx
    assert (e.name, e.numerics) == ("default+overrides", "approx")
    assert approx_features(e) == ("pool_gather_eligible",)

    ref = register_profile(Profile(name="m-ref", model="m", model_default=True))
    over = ref.with_overrides(pool_gather=None)
    assert over.name == "m-ref+overrides"
    assert over.numerics == "exact"
    assert over.model_default is False

    for kw in (
        {"name": "default"},
        {"name": "custom"},
        {"numerics": "exact"},
        {"numerics": "approx"},
        {"model_default": False},
        {"model_default": True},
    ):
        with pytest.raises(ValueError, match="derives.*automatically"):
            d.with_overrides(**kw)


def test_model_profile_without_tiles_inherits_model_default_rule():
    def _ref_rule(B: int, *, moe_tp: bool = False, R: int = 8) -> Tiles:
        return dataclasses.replace(
            default_tiles(B, moe_tp=moe_tp, R=R),
            nbd=10 if moe_tp else 12,
        ).with_modules({_TEST_MODULE: {"width": 256}})

    ref = register_profile(
        Profile(
            name="tile-rule-ref",
            model="tile_rule_model",
            tiles=_ref_rule,
            model_default=True,
        )
    )
    p = Profile(name="tile-rule-user", model="tile_rule_model", buckets=(8,))
    assert p.tile_rule() is _ref_rule and p.numerics == "exact"
    assert approx_features(p) == ()
    assert p.tiles_for(8, R=2) == ref.tiles_for(8, R=2)
    ex = p.exec_for(ExecSpec(R=2), B=8, T=1)
    assert ex.tiles == _ref_rule(8, R=2)
    ex_tp = p.exec_for(ExecSpec(R=2), B=8, T=1, g=types.SimpleNamespace(moe_tp=True))
    assert ex_tp.tiles == _ref_rule(8, R=2, moe_tp=True)

    # Without a model (or for a model with no default profile), `tile_rule()`
    # is None and `exec_for` leaves `tiles` unset for build-time VMEM fitting.
    for q in (
        Profile(name="tile-rule-any"),
        Profile(name="tile-rule-other", model="no_default_model"),
    ):
        assert q.tile_rule() is None
        assert q.exec_for(ExecSpec(R=2), B=8, T=1).tiles is None


def test_exec_for_specializes_bucket_and_respects_explicit_base_settings():
    def _custom_rule(B: int, *, moe_tp: bool = False, R: int = 8) -> Tiles:
        return dataclasses.replace(
            default_tiles(B, moe_tp=moe_tp, R=R), nbd=10 if moe_tp else 12
        )

    prof_auto = Profile(name="custom-auto", tiles=_custom_rule, pool_gather=None)
    prof_elig = Profile(
        name="custom-elig",
        numerics="approx",
        tiles=_custom_rule,
        pool_gather=True,
    )
    base = ExecSpec(R=4, ctx=256, cache=CacheExec(paged_kv=True, num_pages=512))
    g_ep = types.SimpleNamespace(moe_tp=False)
    g_tp = types.SimpleNamespace(moe_tp=True)

    ex_auto = prof_auto.exec_for(base, B=16, T=4, g=g_ep)
    assert (ex_auto.B, ex_auto.T) == (16, 4)
    assert ex_auto.tiles == _custom_rule(16, moe_tp=False, R=4)
    assert cache_exec(ex_auto).pool_gather is None

    ex_elig = prof_elig.exec_for(base, B=24, T=2, g=g_tp)
    assert (ex_elig.B, ex_elig.T) == (24, 2)
    assert ex_elig.tiles == _custom_rule(24, moe_tp=True, R=4)
    assert cache_exec(ex_elig).pool_gather == "eligible"

    # Explicit `pool_gather` on `base` (including `"split"`) wins over profile.
    for explicit in ("split", None, "eligible", True):
        b_pg = with_cache(base, pool_gather=explicit)
        assert cache_exec(prof_elig.exec_for(b_pg, B=24, T=4)).pool_gather == explicit

    # Explicit `tiles` on `base` wins over profile's tile rule.
    explicit_tiles = dataclasses.replace(default_tiles(8, R=4), nbd=6)
    b_tiles = dataclasses.replace(base, tiles=explicit_tiles)
    assert prof_auto.exec_for(b_tiles, B=16, T=1).tiles is explicit_tiles

    # Default profile with `pool_gather=False` preserves `base.cache is None`.
    plain = ExecSpec(R=2)
    ex_plain = DEFAULT_PROFILE.exec_for(plain, B=8, T=1)
    assert ex_plain.cache is None and ex_plain.tiles is None


def test_registry_lookup_idempotency_and_replacement():
    assert list_profiles() == ("default",)
    d = get_profile(None)
    assert d is DEFAULT_PROFILE
    assert get_profile("") is d
    assert get_profile("  default  ") is d
    assert get_profile(d) is d
    assert model_default_profile(None) is d
    assert model_default_profile("unknown_model") is d

    with pytest.raises(KeyError, match="unknown megakernel profile"):
        get_profile("no-such-profile")
    with pytest.raises(ValueError, match="already registered"):
        register_profile(Profile(name="default", description="different"))

    p = register_profile(Profile(name="m-base", model="m", model_default=True))
    assert get_profile("m-base") is p
    assert model_default_profile("m") is p
    assert list_profiles() == ("default", "m-base")

    # Re-registering an equal profile is idempotent.
    assert register_profile(p) is p
    assert register_profile(dataclasses.replace(p)) is p

    # Conflicting registration or second model default fails unless `replace=True`.
    with pytest.raises(ValueError, match="already registered"):
        register_profile(dataclasses.replace(p, description="changed"))
    with pytest.raises(ValueError, match="already has default profile"):
        register_profile(Profile(name="m-base-2", model="m", model_default=True))

    p_alt = register_profile(
        Profile(name="m-base-2", model="m", model_default=True), replace=True
    )
    assert model_default_profile("m") is p_alt
    assert "m-base" not in list_profiles()

    # Overwriting a model default with `model_default=False` clears `_MODEL_DEFAULTS`.
    p_non_def = register_profile(
        Profile(
            name="m-base-2",
            model="m",
            numerics="approx",
            pool_gather=True,
            model_default=False,
        ),
        replace=True,
    )
    assert get_profile("m-base-2") is p_non_def
    assert model_default_profile("m") is DEFAULT_PROFILE


def test_registration_order_cannot_bypass_exact_enforcement():
    def _model_ref_rule(B: int, *, moe_tp: bool = False, R: int = 8) -> Tiles:
        return dataclasses.replace(
            default_tiles(B, moe_tp=moe_tp, R=R), dense_acc_pair=True
        )

    # Constructed before `ord_model` registers its default: matches library
    # `DEFAULT_PROFILE` at construction time, but diverges from `_model_ref_rule`.
    early_bad = Profile(
        name="ord-early-bad",
        model="ord_model",
        numerics="exact",
        tiles=default_tiles,
    )
    # `tiles=None` inherits the model default once registered, so it stays exact.
    early_inherit = register_profile(
        Profile(name="ord-early-inherit", model="ord_model", numerics="exact")
    )

    m_def = Profile(
        name="ord-default",
        model="ord_model",
        tiles=_model_ref_rule,
        model_default=True,
    )

    # Case 1: `early_bad` registered first -> registering `m_def` fails atomically.
    register_profile(early_bad)
    with pytest.raises(ValueError, match=r"tiles\.dense_acc_pair"):
        register_profile(m_def)
    assert model_default_profile("ord_model") is DEFAULT_PROFILE
    assert "ord-default" not in list_profiles()

    # Case 2: `m_def` registered first (after replacing `early_bad` with approx) ->
    # registering `early_bad` or resolving it via `get_profile` fails.
    register_profile(
        dataclasses.replace(early_bad, numerics="approx"),
        replace=True,
    )
    register_profile(m_def)
    assert model_default_profile("ord_model") is m_def
    assert approx_features(early_inherit) == ()
    assert early_inherit.tile_rule() is _model_ref_rule

    with pytest.raises(ValueError, match=r"tiles\.dense_acc_pair"):
        register_profile(early_bad, replace=True)
    with pytest.raises(ValueError, match=r"tiles\.dense_acc_pair"):
        get_profile(early_bad)

    # Demoting `m_def` when another registered `exact` profile uses `_model_ref_rule`
    # also fails atomically instead of leaving an invalid `exact` profile registered.
    register_profile(
        Profile(
            name="ord-explicit-ref",
            model="ord_model",
            numerics="exact",
            tiles=_model_ref_rule,
        )
    )
    with pytest.raises(ValueError, match=r"tiles\.dense_acc_pair"):
        register_profile(
            Profile(
                name="ord-default",
                model="ord_model",
                numerics="approx",
                pool_gather=True,
                model_default=False,
            ),
            replace=True,
        )
    assert model_default_profile("ord_model") is m_def
