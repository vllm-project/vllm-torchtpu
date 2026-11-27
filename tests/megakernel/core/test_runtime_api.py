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
"""Unit and integration tests for the framework-free `megakernel.core.runtime`."""

from __future__ import annotations

import ast
import dataclasses
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import vllm_torchtpu.megakernel.core as mk
from tests.megakernel.core.megakernel_synth import TINY_TILES, synth_hybrid_spec
from vllm_torchtpu.megakernel.core import (
    _builtins,
    cache_flat,
    fused_io_hook,
    model_def,
    planner,
    profile,
)
from vllm_torchtpu.megakernel.core import cache as mk_cache
from vllm_torchtpu.megakernel.core import runtime as mk_rt
from vllm_torchtpu.megakernel.core.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.core.geometry import ModelGeometry
from vllm_torchtpu.megakernel.core.model_def import (
    Every,
    FromHF,
    FromHFLayerTypes,
    LMHead,
    ModelDef,
    Schedule,
    find_model,
    register_model,
)
from vllm_torchtpu.megakernel.core.modules import GDN, GQA, Embedding, MoE, Norm
from vllm_torchtpu.megakernel.core.pool.config import CacheExec, cache_exec, with_cache
from vllm_torchtpu.megakernel.core.profile import (
    DEFAULT_PROFILE,
    Profile,
    register_profile,
)
from vllm_torchtpu.megakernel.core.runtime import (
    Bucket,
    BucketPolicy,
    JaxStepMetaOps,
    SentinelVariantPlan,
    WarmupStepPlan,
    align_cache_exec,
    bucket_exec_spec,
    build_warmup_plan,
    check_compile_hlo_mesh,
    check_post_mesh_eligibility,
    check_startup_eligibility,
    explicit_profile_settings,
    ir_required_buckets,
    pack_step_meta,
    parse_runtime_options,
    pool_schedule_info,
    resolve_driver_profile,
    verify_compiled_step_ir,
    verify_ir_coverage,
    verify_pooled_step_hlo,
    verify_sentinel_step,
)
from vllm_torchtpu.megakernel.core.weights.spec_inputs import (
    pack_spec_inputs,
    random_spec_inputs,
)


@pytest.fixture()
def isolated_model_registry(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(model_def, "_MODEL_REGISTRY", dict(model_def._MODEL_REGISTRY))
    monkeypatch.setattr(model_def, "_ARCH_REGISTRY", dict(model_def._ARCH_REGISTRY))
    monkeypatch.setattr(
        model_def, "_MODEL_TYPE_REGISTRY", dict(model_def._MODEL_TYPE_REGISTRY)
    )
    monkeypatch.setattr(profile, "_REGISTRY", dict(profile._REGISTRY))
    monkeypatch.setattr(profile, "_MODEL_DEFAULTS", dict(profile._MODEL_DEFAULTS))


def test_runtime_package_ast_and_fresh_import_free_of_vllm_and_torch():
    """`megakernel/core/runtime/*.py` never imports `vllm`/`torch`/`os` and <= 400
    lines."""
    runtime_dir = Path(mk_rt.__file__).parent
    py_files = sorted(runtime_dir.glob("*.py"))
    assert "checks.py" in {p.name for p in py_files}

    for py_path in py_files:
        src = py_path.read_text(encoding="utf-8")
        lines = src.splitlines()
        assert len(lines) <= 400, f"{py_path.name} has {len(lines)} lines (> 400)"
        assert "qwen" not in src.lower(), f"{py_path.name} mentions Qwen"
        tree = ast.parse(src, filename=str(py_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    assert top not in ("torch", "vllm", "os"), (
                        f"{py_path.name} imports {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                top = node.module.split(".")[0]
                assert top not in ("torch", "vllm", "os"), (
                    f"{py_path.name} imports from {node.module}"
                )

    code = (
        "import sys; "
        "sys.modules['torch'] = None; "
        "sys.modules['vllm'] = None; "
        "import vllm_torchtpu.megakernel.core.runtime as rt; "
        "assert hasattr(rt, 'BucketPolicy') and hasattr(rt, 'pack_step_meta')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"


def test_runtime_config_and_gpr_bucket_exec_spec():
    """Test `parse_runtime_options`, `resolve_driver_profile` with `BucketPolicy`,
    and `bucket_exec_spec` GPR rules (bucket parsing and selection:
    `test_bucket_policy.py`)."""
    prof = resolve_driver_profile(
        DEFAULT_PROFILE,
        fused_pool_io=False,
        fused_pool_io_max_b=16,
        fused_pool_io_min_b_hi=128,
        pool_gather=False,
    )
    assert prof.fused_pool_io is False
    assert prof.fused_pool_io_max_b == 16
    assert prof.fused_pool_io_min_b_hi == 128
    assert prof.pool_gather is False

    pol = BucketPolicy.resolve(
        prof,
        buckets=(8, 16),
        verify_buckets=(16, 32),
        speculative_tokens=0,
        buckets_explicit=True,
        verify_buckets_explicit=False,
    )
    assert pol.buckets == (8, 16)
    assert pol.verify_buckets == ()
    assert list(pol.iter_buckets()) == [(1, 8), (1, 16)]
    assert (pol.max_bucket, pol.max_verify_bucket) == (16, 0)
    assert pol.select(5, 1) == 8

    cfg = parse_runtime_options(
        {
            "buckets": [4, 8],
            "verify_buckets": [16],
            "fused_pool_io": "auto",
            "pool_gather": True,
            "mode": None,
        }
    )
    assert "enabled" not in cfg and cfg["mode"] == ""
    assert cfg["buckets"] == (4, 8)
    assert cfg["verify_buckets"] == (16,)
    owned = explicit_profile_settings(cfg)
    assert owned["buckets"] == (4, 8)
    assert owned["verify_buckets"] == (16,)
    assert owned["fused_pool_io"] == "auto"
    assert owned["pool_gather"] is True
    # `enabled` and the other runtime keys belong to the adapter's config
    # parser; a typo or a non-option key raises.
    for bad in ({"enabled": True}, {"profiel": "default"}, {"ab": True}):
        with pytest.raises(ValueError, match="unknown runtime option"):
            parse_runtime_options(bad)

    # A `Profile(pool_gather=True)` normalizes True -> "eligible" via
    # `Profile.exec_for` and never re-applies a raw True on T=1.
    g = ModelGeometry(
        L=4,
        att_every=4,
        H=256,
        NH=4,
        NKV=1,
        HD=256,
        ROT=64,
        KH=2,
        VH=4,
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
    ms = synth_hybrid_spec(g, R=1)
    base_ex = ExecSpec(
        R=1,
        B=4,
        T=1,
        ctx=256,
        interpret=True,
        stack_e=False,
        tiles=dataclasses.replace(TINY_TILES, bank=512 << 10),
        cache=CacheExec(paged_kv=True, num_pages=96, max_pages_per_seq=2),
    )
    gpr_prof = Profile(name="test-gpr-true", numerics="approx", pool_gather=True)
    assert gpr_prof.exec_for(base_ex, B=4, T=1, g=g).cache.pool_gather == "eligible"
    ex_t1, use_fused_t1, use_gpr_t1 = bucket_exec_spec(
        ms,
        base_ex,
        gpr_prof,
        bucket_t=4,
        tokens_per_seq=1,
        fused_pool_io=False,
        fused_pool_io_max_b=gpr_prof.fused_pool_io_max_b,
        fused_pool_io_min_b_hi=gpr_prof.fused_pool_io_min_b_hi,
    )
    assert ex_t1.cache.pool_gather is False
    assert (use_fused_t1, use_gpr_t1) == (False, False)
    # planner.plan on T=1 must succeed (would raise if pool_gather=True)
    planner.plan(ms, ex_t1)

    ex_t4, use_fused_t4, use_gpr_t4 = bucket_exec_spec(
        ms,
        base_ex,
        gpr_prof,
        bucket_t=4,
        tokens_per_seq=4,
        fused_pool_io=False,
        fused_pool_io_max_b=gpr_prof.fused_pool_io_max_b,
        fused_pool_io_min_b_hi=gpr_prof.fused_pool_io_min_b_hi,
    )
    assert ex_t4.cache.pool_gather is True
    assert (use_fused_t4, use_gpr_t4) == (False, True)
    planner.plan(ms, ex_t4)

    # An explicit base_exec `pool_gather="split"` wins over a profile that
    # gathers on that bucket without it: `gpr_prof` (True) at `(T=4, B=4)` and
    # an auto (`None`) profile at a WIDE `(T=4, B=24)` bucket.
    base_split = with_cache(base_ex, pool_gather="split")
    auto_prof = Profile(name="test-gpr-auto", pool_gather=None)
    for prof_g, b_t in ((gpr_prof, 4), (auto_prof, 24)):
        kw: dict[str, Any] = dict(
            bucket_t=b_t,
            tokens_per_seq=4,
            fused_pool_io=False,
            fused_pool_io_max_b=prof_g.fused_pool_io_max_b,
            fused_pool_io_min_b_hi=prof_g.fused_pool_io_min_b_hi,
        )
        assert bucket_exec_spec(ms, base_ex, prof_g, **kw)[2], prof_g.name
        ex_split, _, use_gpr_split = bucket_exec_spec(ms, base_split, prof_g, **kw)
        assert ex_split.cache.pool_gather is False, prof_g.name
        assert not use_gpr_split, prof_g.name


def test_bucket_exec_spec_defaults_to_the_profile_fused_policy(
    monkeypatch: pytest.MonkeyPatch,
):
    """Without `fused_pool_io*` arguments, `bucket_exec_spec` takes the policy
    and thresholds from the profile: an `"auto"` profile fuses a bucket at or
    below its `fused_pool_io_max_b` (which turns GPR off) and splits one above
    it, where its `pool_gather=True` gathers."""
    # Mark a fused pool I/O implementation as registered, without loading one:
    # only the pool I/O mode is resolved, no fused kernel runs.
    monkeypatch.setattr(_builtins, "_LOADED", set(_builtins.BUILTINS))
    monkeypatch.setattr(fused_io_hook, "_IMPL", object())
    ms = synth_hybrid_spec(L=4, att_every=4, R=1, KH=2, VH=4)
    base_ex = ExecSpec(
        R=1,
        B=4,
        T=1,
        ctx=256,
        interpret=True,
        stack_e=False,
        tiles=dataclasses.replace(TINY_TILES, bank=512 << 10),
        cache=CacheExec(paged_kv=True, num_pages=96, max_pages_per_seq=2),
    )
    prof = Profile(
        name="test-fused-auto",
        numerics="approx",
        fused_pool_io="auto",
        fused_pool_io_max_b=8,
        fused_pool_io_min_b_hi=56,
        pool_gather=True,
    )
    # B=16 fuses under the library default threshold (24), not under the
    # profile's (8).
    for bucket_t, want in ((8, (True, False)), (16, (False, True))):
        ex, use_fused, use_gpr = bucket_exec_spec(
            ms, base_ex, prof, bucket_t=bucket_t, tokens_per_seq=4
        )
        assert (use_fused, use_gpr) == want, bucket_t
        assert (cache_exec(ex).fused_pool_io, cache_exec(ex).pool_gather) == want


def test_parse_runtime_options_profile_defaults_and_value_errors(
    isolated_model_registry,
):
    """A named profile fills the profile-owned keys left unset (its
    `pool_gather=None` as `"auto"`), explicit values win, and each value
    parser rejects bad input with a `Megakernel:` error."""
    register_profile(
        Profile(
            name="test-rt-opts",
            numerics="approx",
            buckets=(8, 24),
            verify_buckets=(8, 16),
            fused_pool_io="auto",
            fused_pool_io_max_b=12,
            fused_pool_io_min_b_hi=40,
            pool_gather=None,
        )
    )
    cfg = parse_runtime_options({"profile": " test-rt-opts "})
    assert cfg["profile"] == "test-rt-opts"
    assert (cfg["buckets"], cfg["verify_buckets"]) == ((8, 24), (8, 16))
    assert not cfg["buckets_explicit"] and not cfg["verify_buckets_explicit"]
    assert cfg["fused_pool_io"] == "auto"
    assert (cfg["fused_pool_io_max_b"], cfg["fused_pool_io_min_b_hi"]) == (12, 40)
    assert cfg["pool_gather"] == "auto" and cfg["explicit"] == frozenset()
    assert explicit_profile_settings(cfg) == dict.fromkeys(
        ("buckets", "verify_buckets", "fused_pool_io", "fused_pool_io_max_b")
        + ("fused_pool_io_min_b_hi", "pool_gather")
    )
    over = {
        "profile": "test-rt-opts",
        "buckets": "16",
        "verify_buckets": [4],
        "fused_pool_io": "0",
        "fused_pool_io_max_b": "4",
        "pool_gather": "1",
    }
    cfg_ov = parse_runtime_options(over)
    assert (cfg_ov["buckets"], cfg_ov["verify_buckets"]) == ((16,), (4,))
    assert cfg_ov["fused_pool_io"] is False and cfg_ov["pool_gather"] is True
    assert (cfg_ov["fused_pool_io_max_b"], cfg_ov["fused_pool_io_min_b_hi"]) == (4, 40)
    assert cfg_ov["explicit"] == {"fused_pool_io", "fused_pool_io_max_b", "pool_gather"}
    with pytest.raises(ValueError, match="Megakernel: unknown megakernel profile"):
        parse_runtime_options({"profile": "test-rt-missing"})

    for mode, want in (("AUTO", ""), (" tp ", "tp"), ("ep", "ep"), ("", "")):
        assert parse_runtime_options({"mode": mode})["mode"] == want
    for key, val, want in (
        ("compile_hlo_check", "yes", True),
        ("compile_hlo_check", "off", False),
        ("compile_hlo_check", 1, True),
        ("fused_pool_io", "fused", True),
        ("fused_pool_io", "split", False),
        ("fused_pool_io", " Auto ", "auto"),
        ("pool_gather", "gpr", True),
        ("pool_gather", "3stage", False),
        ("pool_gather", "auto", "auto"),
        ("fused_pool_io_min_b_hi", " 48 ", 48),
    ):
        assert parse_runtime_options({key: val})[key] == want, (key, val)
    for key, val, msg in (
        ("mode", "dp", "invalid mode='dp'"),
        ("compile_hlo_check", "maybe", "invalid boolean value 'maybe'"),
        ("fused_pool_io", "sometimes", "invalid fused_pool_io='sometimes'"),
        ("pool_gather", "gather", "invalid pool_gather='gather'"),
        ("fused_pool_io_max_b", "abc", "invalid fused_pool_io_max_b='abc'"),
        ("fused_pool_io_min_b_hi", "4.5", "invalid fused_pool_io_min_b_hi='4.5'"),
        # JSON values: a bool or float is no more an integer than "4.5" is.
        ("fused_pool_io_min_b_hi", 4.5, "invalid fused_pool_io_min_b_hi=4.5"),
        ("fused_pool_io_max_b", True, "invalid fused_pool_io_max_b=True"),
    ):
        with pytest.raises(ValueError, match=f"^Megakernel: {msg}"):
            parse_runtime_options({key: val})
    cfg_int = parse_runtime_options({"fused_pool_io_max_b": 12})
    assert cfg_int["fused_pool_io_max_b"] == 12


def test_resolve_driver_profile_defaults_overrides_and_errors(
    isolated_model_registry,
):
    """`profile=None` runs the model's default profile (else `default`);
    unknown names and a profile for another model raise; `pool_gather="auto"`
    maps to the profile's auto (`None`)."""
    mdl_default = register_profile(
        Profile(name="test-rt-mdl", model="test-rt-model", model_default=True)
    )
    assert resolve_driver_profile(None, model_name="test-rt-model") is mdl_default
    assert resolve_driver_profile(None, model_name="no-such-model") is DEFAULT_PROFILE
    assert resolve_driver_profile("", model_name="test-rt-model") is mdl_default
    with pytest.raises(ValueError, match="Megakernel: unknown megakernel profile"):
        resolve_driver_profile("test-rt-missing", model_name="test-rt-model")
    with pytest.raises(ValueError, match="targets model 'test-rt-model', not 'other'"):
        resolve_driver_profile("test-rt-mdl", model_name="other")
    for val, want in (("auto", None), ("0", False), ("gpr", True), (None, False)):
        prof = resolve_driver_profile(DEFAULT_PROFILE, pool_gather=val)
        assert prof.pool_gather is want, val
    prof = resolve_driver_profile(
        "test-rt-mdl",
        model_name="test-rt-model",
        fused_pool_io="auto",
        fused_pool_io_max_b=6,
        fused_pool_io_min_b_hi=50,
    )
    assert prof.fused_pool_io == "auto"
    assert (prof.fused_pool_io_max_b, prof.fused_pool_io_min_b_hi) == (6, 50)


def test_parse_runtime_options_keeps_verify_buckets_of_any_width(
    isolated_model_registry,
):
    """`verify_buckets` (set, or from the profile) are only range-checked here:
    `BucketPolicy.resolve` checks them against the program's verify widths,
    so a program verifying `T=6` can take `6, 12` buckets."""
    register_profile(
        Profile(name="test-rt-verify6", numerics="approx", verify_buckets=(6, 12))
    )
    for opts in ({"verify_buckets": "6:12"}, {"profile": "test-rt-verify6"}):
        cfg = parse_runtime_options(opts)
        assert cfg["verify_buckets"] == (6, 12), opts
        prof = resolve_driver_profile(cfg["profile"])
        owned = explicit_profile_settings(cfg)
        kw: dict[str, Any] = dict(
            speculative_tokens=5,
            buckets_explicit=cfg["buckets_explicit"],
            verify_buckets_explicit=cfg["verify_buckets_explicit"],
        )
        pol = BucketPolicy.resolve(
            prof, owned["buckets"], owned["verify_buckets"], verify_tokens=(6,), **kw
        )
        assert pol.verify_buckets == (6, 12), opts
    with pytest.raises(ValueError, match="invalid verify_buckets"):
        parse_runtime_options({"verify_buckets": "0:8"})
    # Set buckets the program's widths don't divide still fail, in `resolve`.
    with pytest.raises(ValueError, match=r"verify_buckets \[6\] not divisible"):
        BucketPolicy.resolve(
            DEFAULT_PROFILE, None, (6, 8), verify_buckets_explicit=True
        )


def test_pack_step_meta_jax():
    """Test `pack_step_meta` over `JaxStepMetaOps` (2-D M-RoPE positions use
    row 0)."""
    ops = JaxStepMetaOps()
    pos_t4 = jnp.asarray([1, 2, 3, 4, 6, 7, 8, 9], dtype=jnp.int32)
    pos_mrope = jnp.stack([pos_t4, pos_t4 + 100, pos_t4 + 200], axis=0)
    ids = jnp.asarray([11, 12, 13, 14, 21, 22, 23, 24], dtype=jnp.int32)
    bt_1d = jnp.asarray([100, 101, 200, 201], dtype=jnp.int32)
    s0 = jnp.asarray([1, 0], dtype=jnp.int32)  # seq 1 inactive (slot 0)
    c0 = jnp.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=jnp.int32)
    r0 = jnp.asarray([0, -1, 0, 0, 0, 0], dtype=jnp.int32)
    meta = pack_step_meta(
        ids,
        pos_mrope,
        bt_1d,
        ((s0, c0, r0), (s0, c0, r0), (s0, c0, r0)),
        bucket_t=8,
        tokens_per_seq=4,
        max_pages_per_seq=2,
        has_roff=True,
        use_ckpt_2d=True,
        ops=ops,
    )
    assert meta.ids_b.shape == (8,)
    assert meta.pos_b.shape == (8,)
    np.testing.assert_array_equal(np.asarray(meta.pos_b), [1, 2, 3, 4, 0, 0, 0, 0])
    assert meta.pt_b.shape == (2, 2)
    assert meta.ckpt_b.shape == (3, 2, 4)
    assert meta.roff_b.shape == (3, 2)
    np.testing.assert_array_equal(
        np.asarray(meta.ids_b), np.array([11, 12, 13, 14, 0, 0, 0, 0], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        np.asarray(meta.roff_b[:, 0]), np.array([-1, -1, -1], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        np.asarray(meta.roff_b[:, 1]), np.array([0, 0, 0], dtype=np.int32)
    )

    with pytest.raises(ValueError, match="must be divisible by tokens_per_seq"):
        pack_step_meta(
            ids,
            pos_t4,
            bt_1d,
            ((s0, c0, r0), (s0, c0, r0), (s0, c0, r0)),
            bucket_t=6,
            tokens_per_seq=4,
            max_pages_per_seq=2,
            has_roff=False,
            use_ckpt_2d=False,
            ops=ops,
        )


def test_pack_step_meta_t1_read_column_and_1d_table_fallback():
    """`T=1` with a 2-D `c_raw` and `has_roff`: column 0 reads `c[s, roff[s]]`
    (`roff` clipped to the window), column 1 writes the base slot, and only
    negative read offsets are kept. A 1-D table that `s0.shape[0]` doesn't
    divide is reshaped by `max_pages_per_seq`."""
    ops = JaxStepMetaOps()
    s0 = jnp.asarray([1, 5, 7, 0], dtype=jnp.int32)
    c0 = jnp.asarray(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [0, 0, 0, 0]], dtype=jnp.int32
    )
    r0 = jnp.zeros((8,), dtype=jnp.int32).at[1].set(2).at[5].set(9).at[7].set(-1)
    bt_1d = jnp.asarray([10, 11, 12, 20, 21, 22, 30, 31, 32], dtype=jnp.int32)
    kw: dict[str, Any] = dict(
        bucket_t=4, tokens_per_seq=1, max_pages_per_seq=3, has_roff=True, ops=ops
    )
    ids, pos = jnp.arange(1, 5, dtype=jnp.int32), jnp.arange(10, 14, dtype=jnp.int32)
    meta = pack_step_meta(ids, pos, bt_1d, [(s0, c0, r0)], use_ckpt_2d=True, **kw)
    np.testing.assert_array_equal(
        np.asarray(meta.ckpt_b[0]), [[3, 1], [8, 5], [9, 7], [0, 0]]
    )
    np.testing.assert_array_equal(np.asarray(meta.roff_b[0]), [0, 0, -1, 0])
    np.testing.assert_array_equal(
        np.asarray(meta.pt_b), [[10, 11, 12], [20, 21, 22], [30, 31, 32], [0, 0, 0]]
    )
    np.testing.assert_array_equal(np.asarray(meta.ids_b), [1, 2, 3, 0])
    np.testing.assert_array_equal(np.asarray(meta.pos_b), [10, 11, 12, 0])
    # Without the 2-D window the read column is the base slot.
    meta_1d = pack_step_meta(ids, pos, bt_1d, [(s0, c0, r0)], use_ckpt_2d=False, **kw)
    np.testing.assert_array_equal(np.asarray(meta_1d.ckpt_b[0, :, 0]), [1, 5, 7, 0])
    with pytest.raises(ValueError, match="is not divisible by"):
        pack_step_meta(ids, pos, bt_1d[:7], [(s0, c0, r0)], use_ckpt_2d=True, **kw)


def test_pack_step_meta_validate_rejects_duplicate_write_slots():
    """`validate=True` rejects an active `T>1` row without `T` distinct positive
    write slots (concrete inputs only); the check doesn't run otherwise."""
    s_raw = jnp.asarray([1, 2], dtype=jnp.int32)
    c_dup = jnp.asarray([[3, 3], [4, 5]], dtype=jnp.int32)
    r_raw = jnp.zeros((4,), dtype=jnp.int32)

    def pack(c_raw: Any, *, validate: bool) -> Any:
        return pack_step_meta(
            jnp.arange(4, dtype=jnp.int32),
            jnp.arange(4, dtype=jnp.int32),
            jnp.asarray([[7], [8]], dtype=jnp.int32),
            [(s_raw, c_raw, r_raw)],
            bucket_t=4,
            tokens_per_seq=2,
            max_pages_per_seq=1,
            has_roff=True,
            use_ckpt_2d=True,
            validate=validate,
        )

    with pytest.raises(ValueError, match="2 distinct positive checkpoint slots"):
        pack(c_dup, validate=True)
    pack(jnp.asarray([[3, 6], [4, 5]], dtype=jnp.int32), validate=True)
    np.testing.assert_array_equal(
        np.asarray(pack(c_dup, validate=False).ckpt_b[0]), [[3, 3], [4, 5]]
    )
    traced = jax.jit(lambda c: pack(c, validate=True).ckpt_b)(c_dup)
    np.testing.assert_array_equal(np.asarray(traced[0]), [[3, 3], [4, 5]])


def test_jax_step_meta_ops_implements_the_protocol():
    """`JaxStepMetaOps` provides every `StepMetaOps` method."""
    assert isinstance(JaxStepMetaOps(), mk_rt.StepMetaOps)


def test_align_cache_exec_aligns_ctx_to_kv_blk():
    """`align_cache_exec` sets `ctx` to the largest `kv_blk` multiple the page
    table holds (never below `base_exec.ctx`) and binds the paged KV cache."""
    g = ModelGeometry(
        L=4,
        att_every=4,
        H=256,
        NH=4,
        NKV=1,
        HD=256,
        ROT=64,
        KH=2,
        VH=4,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=256,
        IS=256,
        moe_tp=False,
        theta=10_000_000.0,
    )
    tiles = dataclasses.replace(TINY_TILES, bank=512 << 10)
    base = ExecSpec(R=1, B=4, T=1, ctx=256, interpret=True, stack_e=False)
    for kv_blk, mpps, want_ctx in ((128, 3, 384), (256, 3, 256), (256, 5, 512)):
        ex0 = dataclasses.replace(
            base, tiles=tiles.with_modules({"gqa": {"kv_blk": kv_blk}})
        )
        ex = align_cache_exec(
            ex0, DEFAULT_PROFILE, g, num_pages=96, page_size=128, max_pages_per_seq=mpps
        )
        assert ex.ctx == want_ctx, (kv_blk, mpps)
        cx = ex.cache
        assert (cx.paged_kv, cx.num_pages, cx.page_size) == (True, 96, 128)
        assert cx.max_pages_per_seq == mpps
    # A page table shorter than `base_exec.ctx` keeps `ctx`; GQA's
    # `derive_params` then rejects it (`max_pages_per_seq < ctx // page_size`).
    ex_short = align_cache_exec(
        dataclasses.replace(base, tiles=tiles.with_modules({"gqa": {"kv_blk": 128}})),
        DEFAULT_PROFILE,
        g,
        num_pages=96,
        page_size=128,
        max_pages_per_seq=1,
    )
    assert ex_short.ctx == 256


_SYNTH_ROUTING_DEFAULT_HF_CONFIG: dict[str, Any] = {
    "architectures": ["SynthMoeForCausalLM"],
    "model_type": "synth_moe",
    "vocab_size": 256,
    "hidden_size": 4096,
    "num_hidden_layers": 60,
    "num_attention_heads": 32,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "partial_rotary_factor": 0.25,
    "rope_theta": 10_000_000.0,
    "rms_norm_eps": 1e-6,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 64,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "moe_intermediate_size": 1024,
    "shared_expert_intermediate_size": 1024,
    "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 15,
}


class _SynthRoutingModel(ModelDef):
    """Synthetic hybrid `ModelDef` registering 7 HF architecture / model_type keys."""

    name = "synth_routing"
    hf_architectures = (
        "SynthMoeForConditionalGeneration",
        "SynthMoeForCausalLM",
        "SynthForConditionalGeneration",
        "SynthForCausalLM",
    )
    hf_model_type = ("synth_moe", "synth_moe_text", "synth_family")
    supported_tp_sizes = (8,)
    default_hf_config = _SYNTH_ROUTING_DEFAULT_HF_CONFIG
    geometry = FromHF(
        hidden="hidden_size",
        layers="num_hidden_layers",
        vocab="vocab_size",
        eps="rms_norm_eps",
        rope_theta="rope_theta",
        rope_partial=("partial_rotary_factor", 0.25),
    )
    attn = GQA.bind(
        prefix="self_attn",
        heads="num_attention_heads",
        kv_heads="num_key_value_heads",
        head_dim="head_dim",
        qk_norm=True,
        output_gate=True,
    )
    linear = GDN.bind(
        prefix="linear_attn",
        k_heads="linear_num_key_heads",
        v_heads="linear_num_value_heads",
        k_dim="linear_key_head_dim",
        v_dim="linear_value_head_dim",
        conv_kernel="linear_conv_kernel_dim",
    )
    moe = MoE.bind(
        prefix="mlp",
        experts="num_experts",
        top_k="num_experts_per_tok",
        inter="moe_intermediate_size",
        shared_inter="shared_expert_intermediate_size",
        shared_gate=True,
        moe_tp=False,
    )
    schedule = Schedule(
        mixer=FromHFLayerTypes(
            "layer_types",
            {"full_attention": attn, "linear_attention": linear},
        ),
        ffn=moe,
        pre_norm=Norm.bind("input_layernorm", zero_centered=True),
        post_norm=Norm.bind("post_attention_layernorm", zero_centered=True),
    )
    embed = Embedding("model.embed_tokens")
    final_norm = Norm.bind("model.norm", zero_centered=True)
    lm_head = LMHead("lm_head")


@pytest.mark.parametrize(
    ("arch", "model_type", "num_experts", "exp_found", "exp_eligible", "exp_reason"),
    [
        # 1-2. Canonical MoE model_types
        ("UnknownArch", "synth_moe", 512, True, True, "ok"),
        ("UnknownArch", "synth_moe_text", 512, True, True, "ok"),
        # 3. Shared family model_type (MoE vs dense disambiguated by validate_hf_config)
        ("UnknownArch", "synth_family", 512, True, True, "ok"),
        ("UnknownArch", "synth_family", 0, True, False, "non_moe_checkpoint"),
        # 4-5. Canonical MoE HF architectures
        ("SynthMoeForConditionalGeneration", "unknown", 512, True, True, "ok"),
        ("SynthMoeForCausalLM", "unknown", 512, True, True, "ok"),
        # 6-7. Shared family HF architectures (MoE vs dense)
        ("SynthForConditionalGeneration", "unknown", 512, True, True, "ok"),
        (
            "SynthForConditionalGeneration",
            "unknown",
            0,
            True,
            False,
            "non_moe_checkpoint",
        ),
        ("SynthForCausalLM", "unknown", 512, True, True, "ok"),
        ("SynthForCausalLM", "unknown", None, True, False, "non_moe_checkpoint"),
        # Unsupported model keys return None from find_model
        ("OtherForCausalLM", "other_arch", 512, False, False, ""),
        ("AltForCausalLM", "alt_arch", 512, False, False, ""),
    ],
)
def test_multi_key_routing_and_dense_rejection(
    isolated_model_registry,
    arch: str,
    model_type: str,
    num_experts: int | None,
    exp_found: bool,
    exp_eligible: bool,
    exp_reason: str,
):
    """Verify all 7 routing keys resolve to `_SynthRoutingModel` and reject dense."""
    register_model(
        name=_SynthRoutingModel.name,
        hf_architectures=_SynthRoutingModel.hf_architectures,
        hf_model_type=_SynthRoutingModel.hf_model_type,
        replace=True,
    )(_SynthRoutingModel)
    cfg = dict(_SynthRoutingModel.default_hf_config)
    cfg["architectures"] = [arch]
    cfg["model_type"] = model_type
    if num_experts is None:
        cfg.pop("num_experts", None)
    else:
        cfg["num_experts"] = num_experts

    found = find_model(cfg)
    if not exp_found:
        assert found is None
        return
    assert found is _SynthRoutingModel
    el = found.validate_hf_config(cfg)
    assert el.eligible is exp_eligible
    assert el.reason == exp_reason


def test_schedule_driven_validate_hf_config(isolated_model_registry):
    """`ModelDef.validate_hf_config` resolves `Schedule.ffn` across schedule forms."""
    valid_cfg = dict(_SynthRoutingModel.default_hf_config)
    assert _SynthRoutingModel.validate_hf_config(valid_cfg).eligible

    dense_cfg = {k: v for k, v in valid_cfg.items() if k != "num_experts"}
    res_dense = _SynthRoutingModel.validate_hf_config(dense_cfg)
    assert not res_dense.eligible and res_dense.reason == "non_moe_checkpoint"
    assert "_SynthRoutingModel supports MoE checkpoints only" in res_dense.message

    # Non-numeric num_experts (e.g. string "512") returns non_moe_checkpoint cleanly
    str_experts_cfg = {**valid_cfg, "num_experts": "512"}
    res_str = _SynthRoutingModel.validate_hf_config(str_experts_cfg)
    assert not res_str.eligible and res_str.reason == "non_moe_checkpoint"

    # Missing hidden_size still checks present geometry keys (e.g. head_dim)
    no_hidden_bad_hd = {"num_experts": 512, "head_dim": 64}
    res_hd = _SynthRoutingModel.validate_hf_config(no_hidden_bad_hd)
    assert not res_hd.eligible and res_hd.reason == "geometry_mismatch"
    assert "HD=64 (expected 256)" in res_hd.message

    custom_moe = MoE.bind(
        prefix="mlp",
        experts="routed_expert_count",
        top_k="num_experts_per_tok",
        inter="moe_intermediate_size",
        shared_inter="shared_expert_intermediate_size",
        shared_gate=True,
    )
    other_moe = MoE.bind(
        prefix="mlp",
        experts="alt_expert_count",
        top_k="num_experts_per_tok",
        inter="moe_intermediate_size",
        shared_inter="shared_expert_intermediate_size",
        shared_gate=True,
    )
    attn = GQA.bind(
        prefix="self_attn",
        heads="num_attention_heads",
        kv_heads="num_key_value_heads",
        head_dim="head_dim",
    )
    gdn = GDN.bind(
        prefix="linear_attn",
        k_heads="linear_num_key_heads",
        v_heads="linear_num_value_heads",
        k_dim="linear_key_head_dim",
        v_dim="linear_value_head_dim",
        conv_kernel="linear_conv_kernel_dim",
    )
    dummy_non_moe = Norm.bind("mlp_norm", zero_centered=True)

    base_synth_cfg = {
        **dense_cfg,
        "architectures": [],
        "num_hidden_layers": 4,
        "routed_expert_count": 128,
        "alt_expert_count": 64,
        "layer_types": ["linear_attention", "full_attention"] * 2,
        "ffn_types": ["dense", "moe", "dense", "moe"],
    }
    geom_spec = FromHF(
        hidden="hidden_size",
        layers="num_hidden_layers",
        eps="rms_norm_eps",
        rope_theta="rope_theta",
    )

    # 1. Schedule.ffn wrapped in Every(...)
    @register_model(name="synth_every_ffn", hf_model_type="synth_every_ffn")
    class SynthEveryFFN(ModelDef):
        default_hf_config = {**base_synth_cfg, "model_type": "synth_every_ffn"}
        geometry = geom_spec
        schedule = Schedule(
            mixer=Every(n=2, last=attn, rest=gdn),
            ffn=Every(n=2, last=custom_moe, rest=dummy_non_moe),
            pre_norm=Norm.bind("input_layernorm", zero_centered=True),
            post_norm=Norm.bind("post_attention_layernorm", zero_centered=True),
        )
        final_norm = Norm.bind("model.norm", zero_centered=True)

    assert SynthEveryFFN.validate_hf_config(SynthEveryFFN.default_hf_config).eligible
    assert not SynthEveryFFN.validate_hf_config(
        {**SynthEveryFFN.default_hf_config, "routed_expert_count": 0}
    ).eligible

    # 2. Schedule.ffn wrapped in FromHFLayerTypes(...)
    class SynthLayerTypesFFN(ModelDef):
        default_hf_config = {**base_synth_cfg, "model_type": "synth_lh_ffn"}
        geometry = geom_spec
        schedule = Schedule(
            mixer=Every(n=2, last=attn, rest=gdn),
            ffn=FromHFLayerTypes(
                "ffn_types",
                {"dense": dummy_non_moe, "moe": custom_moe},
            ),
            pre_norm=Norm.bind("input_layernorm", zero_centered=True),
            post_norm=Norm.bind("post_attention_layernorm", zero_centered=True),
        )
        final_norm = Norm.bind("model.norm", zero_centered=True)

    assert SynthLayerTypesFFN.validate_hf_config(
        SynthLayerTypesFFN.default_hf_config
    ).eligible
    assert not SynthLayerTypesFFN.validate_hf_config(
        {**SynthLayerTypesFFN.default_hf_config, "routed_expert_count": None}
    ).eligible

    # 3. Schedule.ffn as a per-layer sequence + Schedule.overrides with a second MoE key
    @register_model(name="synth_seq_ov_ffn", hf_model_type="synth_seq_ov_ffn")
    class SynthSeqOverridesFFN(ModelDef):
        default_hf_config = {**base_synth_cfg, "model_type": "synth_seq_ov_ffn"}
        geometry = geom_spec
        schedule = Schedule(
            mixer=Every(n=2, last=attn, rest=gdn),
            ffn=[dummy_non_moe, custom_moe, dummy_non_moe, dummy_non_moe],
            pre_norm=Norm.bind("input_layernorm", zero_centered=True),
            post_norm=Norm.bind("post_attention_layernorm", zero_centered=True),
            overrides={3: {"ffn": other_moe}},
        )
        final_norm = Norm.bind("model.norm", zero_centered=True)

    assert SynthSeqOverridesFFN.validate_hf_config(
        SynthSeqOverridesFFN.default_hf_config
    ).eligible
    assert not SynthSeqOverridesFFN.validate_hf_config(
        {**SynthSeqOverridesFFN.default_hf_config, "routed_expert_count": 0}
    ).eligible
    assert not SynthSeqOverridesFFN.validate_hf_config(
        {**SynthSeqOverridesFFN.default_hf_config, "alt_expert_count": 0}
    ).eligible


def test_checks_hlo_ir_warmup_and_sentinel():
    """Test `check_compile_hlo_mesh`, `verify_compiled_step_ir`, and IR coverage."""
    cpu_mesh = jax.sharding.Mesh(np.asarray(jax.devices("cpu")[:2]), ("tp",))
    check_compile_hlo_mesh(cpu_mesh, True)
    fake_mesh = types.SimpleNamespace(devices=np.asarray([object(), object()]))
    with pytest.raises(mk.MegakernelConfigError, match="requires a mesh of real"):
        check_compile_hlo_mesh(fake_mesh, True)

    good_ir = """
    module @test {
      func.func public @main(
        %arg0: tensor<64x2x64x4x128xf8E4M3FN> {tf.aliasing_output = 1 : i32},
        %arg1: tensor<64x2x64x4x128xf8E4M3FN> {tf.aliasing_output = 2 : i32}
      ) -> (
        tensor<8x1024xbf16>,
        tensor<64x2x64x4x128xf8E4M3FN>,
        tensor<64x2x64x4x128xf8E4M3FN>
      ) {
        %0 = stablehlo.custom_call @tpu_custom_call(%arg0, %arg1)
          : (tensor<64x2x64x4x128xf8E4M3FN>, tensor<64x2x64x4x128xf8E4M3FN>)
          -> tensor<8x1024xbf16>
        return
      }
    }
    """
    ok_ir, msg_ir = verify_compiled_step_ir(
        good_ir, num_pages=64, page_size=128, kvh=1, hd=256, num_pools=2
    )
    assert ok_ir, msg_ir

    bad_ir = good_ir.replace("{tf.aliasing_output = 2 : i32}", "")
    ok_bad, msg_bad = verify_compiled_step_ir(
        bad_ir, num_pages=64, page_size=128, kvh=1, hd=256, num_pools=2
    )
    assert not ok_bad and "expected >= 2 donated pool arguments" in msg_bad

    assert ir_required_buckets(
        [(1, 8), (1, 16), (4, 16)],
        {(1, 8): True, (1, 16): False, (4, 16): True},
        is_eager=False,
    ) == [(1, 8), (1, 16), (4, 16)]
    assert ir_required_buckets([(1, 8)], {(1, 8): True}, is_eager=True) == [(1, 8)]
    assert ir_required_buckets([(1, 8)], {(1, 8): False}, is_eager=True) == []

    # verify_ir_coverage tests
    verify_ir_coverage([(1, 8)], {(1, 8): True}, {(1, 8)}, is_eager=False)
    verify_ir_coverage([], {}, set(), is_eager=False)
    with pytest.raises(RuntimeError, match="boom"):
        verify_ir_coverage(
            [(1, 8)], {(1, 8): True}, {(1, 8)}, is_eager=False, ir_error="boom"
        )
    with pytest.raises(RuntimeError, match="did not run for buckets"):
        verify_ir_coverage(
            [(1, 8), (1, 16)],
            {(1, 8): True, (1, 16): True},
            {(1, 8)},
            is_eager=False,
        )

    # pool_schedule_info raises a named layout error without a cache_backend
    with pytest.raises(mk.MegakernelLayoutError, match="has no pool backend layout"):
        pool_schedule_info(types.SimpleNamespace(name="kv", cache_backend=None))


@pytest.mark.parametrize(("slots", "att_every"), [(0, 1), (1, 2), (3, 4)])
def test_pool_schedule_info_att_every_counts_recurrent_layers(slots, att_every):
    """`att_every` is `slots_per_pool + 1`; a KV-only pool (`slots_per_pool=0`)
    still has one leading `ckpt_indices` row (`recurrent_slots_per_pool=1`)."""
    lay = mk.pool.PoolLayout(
        num_pools=2,
        slots_per_pool=slots,
        hd=256,
        kvh=1,
        kh=2 if slots else 0,
        vh=4 if slots else 0,
        KD=128,
        VD=128,
    )
    spec = types.SimpleNamespace(name="s", cache_backend=mk.pool.make_pool_backend(lay))
    info = pool_schedule_info(spec)
    assert (info.slots_per_pool, info.att_every) == (slots, att_every)
    assert info.recurrent_slots_per_pool == max(1, slots)


_POOL_TY = "tensor<64x2x64x4x128xf8E4M3FN>"
_MLIR = """
module @m {{
  func.func public @main(
    %arg0: tensor<8x1024xbf16> {{mhlo.sharding = "{{devices=[2,1]<=[2]}}"}},
    %arg1: {ty} {attr1},
    %arg2: {ty} {attr2}
  ) -> (tensor<8x1024xbf16>, {ty}, {ty}) {{
    %0 = stablehlo.custom_call @tpu_custom_call(%arg1, %arg2) : ({ty}, {ty}) -> {ty}
    {extra}
    return
  }}
}}
"""
_HLO = """HloModule m, input_output_alias={{ {aliases} }}
ENTRY %main {{
  %k = () custom-call(), custom_call_target="tpu_custom_call"
  {extra}
  ROOT %t = () tuple()
}}
"""
_DONOR = "{jax.buffer_donor = true}"
_SHARDED_ALIAS = (
    '{{mhlo.sharding = "{{devices=[2,1,1,1,1]<=[2]}}", tf.aliasing_output = {i} : i32}}'
)
_SDY_ALIAS = (
    '{{sdy.sharding = #sdy.sharding<@mesh, [{{"tp"}}, {{}}, {{}}, {{}}, {{}}]>, '
    "tf.aliasing_output = {i} : i32}}"
)


def _mlir(attr: str, extra: str = "") -> str:
    a1, a2 = (attr.format(i=i) if "{i}" in attr else attr for i in (1, 2))
    return _MLIR.format(ty=_POOL_TY, attr1=a1, attr2=a2, extra=extra)


def _hlo(n_alias: int, extra: str = "") -> str:
    aliases = ", ".join(f"{{{i}}}: ({i + 1}, {{}}, may-alias)" for i in range(n_alias))
    return _HLO.format(aliases=aliases, extra=extra)


@pytest.mark.parametrize(
    ("ir_text", "hlo_text", "want"),
    [
        (_mlir(_DONOR), None, "ok"),
        # Sharded arguments: the attribute dict nests braces before the alias.
        (_mlir(_SHARDED_ALIAS), None, "ok"),
        (_mlir(_SDY_ALIAS), None, "ok"),
        (_mlir('{mhlo.sharding = "{replicated}"}'), None, "donated pool arguments"),
        (_hlo(2), None, "ok"),
        (_hlo(1), None, "entries in HloModule input_output_alias"),
        (_hlo(0).replace("input_output_alias={  }", ""), None, "missing input_output"),
        (_mlir(_DONOR), _hlo(2), "ok"),
        (_mlir(_DONOR), _hlo(1), "entries in HloModule input_output_alias"),
        # Full-pool copies (MLIR and HLO forms) fail; one-page staging copies pass.
        (
            _mlir(_DONOR, f"%c = stablehlo.copy %arg1 : {_POOL_TY}"),
            None,
            "full-pool copy in compiled IR",
        ),
        (
            _mlir(_DONOR, "%c = stablehlo.copy %s : tensor<1x2x64x4x128xf8E4M3FN>"),
            None,
            "ok",
        ),
        (
            _hlo(2, "%cp = f8e4m3fn[64,2,64,4,128] copy(%p0)"),
            None,
            "full-pool copy in compiled HLO",
        ),
        (_hlo(2, "%cp = f8e4m3fn[1,2,64,4,128] copy(%p0)"), None, "ok"),
        (
            _mlir(_DONOR),
            _hlo(2, "%cp = f8e4m3fn[64,2,64,4,128] copy(%p0)"),
            "full-pool copy in compiled HLO",
        ),
    ],
    ids=[
        *("mlir_donor", "mlir_sharded_alias", "mlir_sdy_alias", "mlir_no_donor"),
        *("hlo_alias", "hlo_short_alias", "hlo_no_alias", "mlir_hlo"),
        *("mlir_hlo_short", "mlir_full_copy", "mlir_page_copy"),
        *("hlo_full_copy", "hlo_page_copy", "mlir_hlo_full_copy"),
    ],
)
def test_verify_compiled_step_ir_forms(ir_text, hlo_text, want):
    """`verify_compiled_step_ir` on StableHLO MLIR (`@main` donors, sharded
    argument attributes) and HloModule text (`input_output_alias`), and its
    full-pool copy rejection in both forms."""
    ok, msg = verify_compiled_step_ir(
        ir_text,
        num_pages=64,
        page_size=128,
        kvh=1,
        hd=256,
        num_pools=2,
        hlo_text=hlo_text,
    )
    if want == "ok":
        assert (ok, msg) == (True, "ok")
    else:
        assert not ok and want in msg, msg


@pytest.mark.parametrize(
    ("bucket_b", "paged", "want_fused"),
    [(4, True, True), (4, False, True), (8, True, False)],
    ids=["auto_fused", "auto_fused_unpaged", "split"],
)
def test_verify_pooled_step_hlo_uses_the_flat_steps_fused_decision(
    monkeypatch: pytest.MonkeyPatch, bucket_b: int, paged: bool, want_fused: bool
):
    """`verify_pooled_step_hlo` lowers the flat step from `build_global_avals`
    and checks its HLO with the fused-I/O decision `make_flat_pooled_step` made,
    for a bucket `"auto"` fuses (`B <= 4`) and one it splits. The flat step pages
    an unpaged `ex` itself, so a fused unpaged `ex` is still a fused step."""
    R, T = 2, 1
    spec = synth_hybrid_spec(L=2, att_every=2, R=R, VH=4)  # fused GDN: 2*kh <= vh
    base_ex = ExecSpec(
        B=bucket_b,
        T=T,
        R=R,
        ctx=256,
        stack_e=True,
        tiles=TINY_TILES.with_modules({"gqa": {"kv_blk": 128, "nkv": 2}}),
        interpret=True,
        cache=CacheExec(paged_kv=paged, num_pages=128),
    )
    prof = resolve_driver_profile(DEFAULT_PROFILE, model_name=spec.name)
    ex, use_fused, _ = bucket_exec_spec(
        spec,
        base_ex,
        prof,
        bucket_t=bucket_b,
        tokens_per_seq=T,
        fused_pool_io="auto",
        fused_pool_io_max_b=4,
        fused_pool_io_min_b_hi=56,
    )
    # "auto" fuses only with a registered fused pool I/O implementation.
    assert use_fused is (want_fused and fused_io_hook.fused_io_available())

    flat_fused: list[bool] = []
    real_build = cache_flat.build_pooled

    def spy_build(*a: Any, **kw: Any) -> Any:
        b = real_build(*a, **kw)
        flat_fused.append(cache_exec(b.ex_paged).fused_pool_io)
        return b

    checked_fused: list[bool] = []
    real_check = mk_cache.check_pooled_step_hlo

    def spy_check(*a: Any, **kw: Any) -> Any:
        checked_fused.append(kw["fused_pool_io"])
        return real_check(*a, **kw)

    monkeypatch.setattr(cache_flat, "build_pooled", spy_build)
    monkeypatch.setattr(mk_cache, "check_pooled_step_hlo", spy_check)

    # Per-rank weight and pool descriptors (build_global_avals reads shapes).
    mesh = jax.sharding.Mesh(np.asarray(jax.devices("cpu")[:R]), ("tp",))
    pos, hid, w_log, st_log = random_spec_inputs(jax.random.PRNGKey(3), spec, ex)
    _, _, w_mk, _ = pack_spec_inputs(pos, hid, w_log, st_log, spec, ex, mesh=mesh)
    pt, ckpt, roff, _ = mk_cache.default_pool_layout(spec, ex)
    raw = mk_cache.states_to_pool(spec, ex, st_log, pt, ckpt, read_offsets=roff)

    def rank_sds(a: Any) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((a.shape[0] // R, *a.shape[1:]), a.dtype)

    w_flat = [
        rank_sds(t)
        for k in planner.plan(spec, ex).w_names
        for t in (w_mk[k] if isinstance(w_mk[k], tuple) else (w_mk[k],))
    ]
    pools = [rank_sds(p) for p in raw]
    avals = mk_rt.build_global_avals(
        mesh, spec, ex, w_flat, pools, max_pages_per_seq=pt.shape[1]
    )
    assert [a.shape for a in avals[2 : 2 + len(w_flat)]] == [
        (R * w.shape[0], *w.shape[1:]) for w in w_flat
    ]
    assert avals[-3].shape == (ex.NS, pt.shape[1])

    ok, msg = verify_pooled_step_hlo(
        mesh, spec, ex, w_flat, pools, max_pages_per_seq=pt.shape[1]
    )
    assert ok, msg
    assert flat_fused == checked_fused == [use_fused]


def test_verify_sentinel_step_pass_and_failures():
    """`verify_sentinel_step` passes when every pool's KV page and recurrent
    slot changed and the output is finite; it compares bytes, so an unchanged
    NaN pattern counts as unchanged."""
    kv0 = [np.zeros((4,), np.uint8), np.zeros((4,), np.uint8)]
    kv1 = [np.ones((4,), np.uint8), np.full((4,), 2, np.uint8)]
    out = np.ones((2, 8), np.float32)
    kw: dict[str, Any] = dict(
        tokens_per_seq=1, bucket_t=2, variant_key=(1, False, False)
    )
    verify_sentinel_step(out, kv0, kv1, kv0, kv1, **kw)
    with pytest.raises(RuntimeError, match=r"pool\[1\].*kv_ok=False, gdn_ok=True"):
        verify_sentinel_step(out, kv0, [kv1[0], kv0[1]], kv0, kv1, **kw)
    with pytest.raises(RuntimeError, match=r"kv_ok=True, gdn_ok=False"):
        verify_sentinel_step(out, kv0, kv1, kv0, kv0, **kw)
    bad_out = out.copy()
    bad_out[0, 0] = np.inf
    with pytest.raises(RuntimeError, match=r"out_fin=15"):
        verify_sentinel_step(bad_out, kv0, kv1, kv0, kv1, **kw)
    # Non-finite output fails even with no pools to compare.
    with pytest.raises(RuntimeError, match=r"out_fin=15"):
        verify_sentinel_step(bad_out, [], [], [], [], **kw)
    nan_pool = [np.full((4,), np.nan, np.float32)] * 2
    with pytest.raises(RuntimeError, match=r"kv_ok=False"):
        verify_sentinel_step(
            out, nan_pool, [p.copy() for p in nan_pool], kv0, kv1, **kw
        )
    with pytest.raises(ValueError, match="differ in length"):
        verify_sentinel_step(out, kv0, kv1[:1], kv0, kv1, **kw)


def test_no_kw_rejects_legacy_and_unknown_kwargs():
    """Runtime API functions reject legacy or unknown keyword arguments."""
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        check_startup_eligibility(  # type: ignore[call-arg]
            num_layers=60,
            expected_num_layers=60,
            tp_size=8,
            supported_tp_sizes=(8,),
            unified_kv_layout=True,
        )
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        check_startup_eligibility(  # type: ignore[call-arg]
            num_layers=60,
            expected_num_layers=60,
            tp_size=8,
            supported_tp_sizes=(8,),
            supports_mm_inputs=False,
        )
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        check_startup_eligibility(  # type: ignore[call-arg]
            num_layers=60,
            expected_num_layers=60,
            tp_size=8,
            supported_tp_sizes=(8,),
            unknown_flag=True,
        )
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        check_post_mesh_eligibility(  # type: ignore[call-arg]
            tp_size=8,
            num_layers=60,
        )
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        resolve_driver_profile(DEFAULT_PROFILE, unknown_flag=True)  # type: ignore[call-arg]

    g = ModelGeometry(
        L=4,
        att_every=4,
        H=256,
        NH=4,
        NKV=1,
        HD=256,
        ROT=64,
        KH=2,
        VH=4,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=256,
        IS=256,
        theta=10_000_000.0,
        zero_centered_norm=True,
    )
    ms = synth_hybrid_spec(g, R=1)
    base_ex = ExecSpec(R=1, B=4, T=1, ctx=256, interpret=True, stack_e=False)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        bucket_exec_spec(  # type: ignore[call-arg]
            ms,
            base_ex,
            DEFAULT_PROFILE,
            bucket_t=4,
            tokens_per_seq=1,
            pool_gather=True,
        )
    with pytest.raises(TypeError, match="missing 1 required keyword-only argument"):
        align_cache_exec(  # type: ignore[call-arg]
            base_ex, DEFAULT_PROFILE, g, num_pages=64, max_pages_per_seq=2
        )
    with pytest.raises(TypeError, match="missing .* required keyword-only argument"):
        verify_compiled_step_ir(  # type: ignore[call-arg]
            "module {}", num_pages=64, kvh=1
        )
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        build_warmup_plan(  # type: ignore[call-arg]
            ms, ((1, 4),), {}, {}, num_pages=64, max_pages_per_seq=2, unknown=1
        )
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        verify_sentinel_step(  # type: ignore[call-arg]
            np.zeros((2, 4)),
            [],
            [],
            [],
            [],
            tokens_per_seq=1,
            bucket_t=2,
            variant_key=(1, False, False),
            unknown=1,
        )


def test_build_warmup_plan_shapes_and_sentinels():
    """`build_warmup_plan` warms each bucket over the shapes the driver runs
    (`Bucket.runner_shapes`) and plans non-overlapping sentinel pages."""
    g = ModelGeometry(
        L=4,
        att_every=4,
        H=256,
        NH=4,
        NKV=1,
        HD=256,
        ROT=64,
        KH=2,
        VH=4,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=256,
        IS=256,
        theta=10_000_000.0,
        zero_centered_norm=True,
    )
    ms = synth_hybrid_spec(g, R=1)
    # `build_warmup_plan` takes `(T, B)` `bucket_keys` directly. For the
    # default keys the expected list pins every field: two runner shapes
    # `(tok_pad, req_pad, blk_pad, c_cols)` per bucket and, on each bucket's
    # first shape, the sentinel plan of its `(T, fused, gather)` variant (its
    # own KV page, GDN slot base and first page, sub-step `T`). `T=8` verify
    # keys produce plans for `(8, b)` with `spv = spp * 8`.
    pol_def = BucketPolicy((8, 16), (16, 32))
    default_keys = list(pol_def.iter_buckets())
    fio_def = {(1, 8): True, (1, 16): False, (4, 16): True, (4, 32): False}
    gpr_def = {(1, 8): False, (1, 16): True, (4, 16): False, (4, 32): False}
    steps_def, warmed_def, n_save_def = build_warmup_plan(
        ms,
        default_keys,
        fio_def,
        gpr_def,
        num_pages=256,
        max_pages_per_seq=4,
        ckpt_window=4,
        runner_tok_pads=(16, 32),
        req_block_pairs=((8, 4), (16, 4)),
        is_eager=False,
    )
    expected_default_steps = [
        WarmupStepPlan(
            1, 8, 16, 8, 4, 4, SentinelVariantPlan((1, True, False), 1, 3, 15, 1, True)
        ),
        WarmupStepPlan(1, 8, 16, 16, 4, 4, None),
        WarmupStepPlan(
            1,
            16,
            16,
            8,
            4,
            4,
            SentinelVariantPlan((1, False, True), 2, 15, 75, 1, True),
        ),
        WarmupStepPlan(1, 16, 16, 16, 4, 4, None),
        WarmupStepPlan(
            4,
            16,
            16,
            8,
            4,
            4,
            SentinelVariantPlan((4, True, False), 3, 27, 135, 4, True),
        ),
        WarmupStepPlan(4, 16, 16, 16, 4, 4, None),
        WarmupStepPlan(
            4,
            32,
            32,
            8,
            4,
            4,
            SentinelVariantPlan((4, False, False), 4, 39, 195, 4, True),
        ),
        WarmupStepPlan(4, 32, 32, 16, 4, 4, None),
    ]
    assert steps_def == expected_default_steps
    assert warmed_def == default_keys and n_save_def == 255

    keys_t8 = [(1, 8), (8, 16), (8, 32)]
    fio_t8 = {(1, 8): True, (8, 16): True, (8, 32): False}
    gpr_t8 = {(1, 8): False, (8, 16): False, (8, 32): False}
    steps_t8, warmed_t8, n_save_t8 = build_warmup_plan(
        ms,
        keys_t8,
        fio_t8,
        gpr_t8,
        num_pages=512,
        max_pages_per_seq=4,
        ckpt_window=4,
        is_eager=False,
    )
    assert warmed_t8 == keys_t8
    assert [(s.tokens_per_seq, s.bucket_t, s.req_pad, s.c_cols) for s in steps_t8] == [
        (1, 8, 8, 4),
        (8, 16, 2, 8),
        (8, 32, 4, 8),
    ]
    assert [s.sentinel for s in steps_t8] == [
        SentinelVariantPlan((1, True, False), 1, 3, 15, 1, True),
        SentinelVariantPlan((8, True, False), 2, 27, 135, 8, True),
        SentinelVariantPlan((8, False, False), 3, 51, 255, 8, True),
    ]
    assert n_save_t8 == 375
    pads, pairs = (16, 32), ((8, 4), (16, 4))
    for T, b in default_keys:
        shapes = Bucket(T, b).runner_shapes(
            pads, pairs, ckpt_window=4, max_pages_per_seq=4
        )
        got = [s for s in steps_def if (s.tokens_per_seq, s.bucket_t) == (T, b)]
        want = [(x.tok_pad, x.req_pad, x.blk_pad, x.ckpt_cols) for x in shapes]
        assert [(s.tok_pad, s.req_pad, s.blk_pad, s.c_cols) for s in got] == want
    for s in steps_t8:
        (x,) = Bucket(s.tokens_per_seq, s.bucket_t).runner_shapes(
            (), (), ckpt_window=4, max_pages_per_seq=4
        )
        assert (s.tok_pad, s.req_pad, s.blk_pad, s.c_cols) == (
            x.tok_pad,
            x.req_pad,
            x.blk_pad,
            x.ckpt_cols,
        )
