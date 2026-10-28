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
"""Unit and integration tests for the framework-free `megakernel.runtime` package."""

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

import vllm_torchtpu.megakernel as mk
from vllm_torchtpu.megakernel import model_def, planner, profile
from vllm_torchtpu.megakernel import runtime as mk_rt
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.model_def import (
    Every,
    FromHF,
    FromHFLayerTypes,
    LMHead,
    ModelDef,
    Schedule,
    find_model,
    register_model,
)
from vllm_torchtpu.megakernel.models import qwen35
from vllm_torchtpu.megakernel.models.qwen35_profile import TINY_TILES
from vllm_torchtpu.megakernel.modules import GDN, GQA, Embedding, MoE, Norm
from vllm_torchtpu.megakernel.pool.config import CacheExec, with_cache
from vllm_torchtpu.megakernel.profile import DEFAULT_PROFILE, Profile
from vllm_torchtpu.megakernel.runtime import (
    BatchShape,
    BucketPolicy,
    JaxStepMetaOps,
    MKABCalibrator,
    StepMixTracker,
    align_cache_exec,
    bucket_exec_spec,
    build_warmup_plan,
    check_compile_hlo_mesh,
    check_mrope_uniform,
    check_post_mesh_eligibility,
    check_qsl_uniform,
    check_startup_eligibility,
    check_step_eligibility,
    explicit_profile_settings,
    inspect_batch_shape,
    ir_required_buckets,
    pack_step_meta,
    parse_mk_buckets,
    parse_mk_verify_buckets,
    parse_runtime_config,
    pool_schedule_info,
    resolve_driver_profile,
    resolve_step_positions,
    select_bucket,
    verify_compiled_step_ir,
    verify_ir_coverage,
    verify_pooled_step_hlo,
    verify_sentinel_step,
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
    """`megakernel/runtime/*.py` never imports `vllm`/`torch`/`os` and <= 400 lines."""
    runtime_dir = Path(mk_rt.__file__).parent
    py_files = sorted(runtime_dir.glob("*.py"))
    assert len(py_files) == 6

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
        "import vllm_torchtpu.megakernel.runtime as rt; "
        "assert hasattr(rt, 'BucketPolicy') and hasattr(rt, 'pack_step_meta')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"


def test_bucket_policy_config_parsing_and_gpr_bucket_exec_spec():
    """Test `BucketPolicy`, `parse_runtime_config`, and `bucket_exec_spec` GPR rules."""
    assert parse_mk_buckets("8, 32, 8, 16") == (8, 16, 32)
    assert parse_mk_buckets("none") == ()
    assert parse_mk_buckets(x for x in (16, 8, 16)) == (8, 16)
    assert parse_mk_buckets({32, 8, 16}) == (8, 16, 32)
    assert parse_mk_verify_buckets("16, 32") == (16, 32)
    assert parse_mk_verify_buckets(x for x in (32, 16)) == (16, 32)
    assert parse_mk_verify_buckets({32, 16}) == (16, 32)
    assert select_bucket(5, (8, 16, 32)) == 8
    assert select_bucket(32, (8, 16, 32)) == 32
    with pytest.raises(ValueError, match="exceeds max bucket"):
        select_bucket(33, (8, 16, 32))

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

    pol_pad = BucketPolicy((8, 16), (16, 32), token_paddings=(16, 32, 64))
    assert pol_pad.out_rows(8, 16) == 16
    assert pol_pad.out_rows(32, 32) == 32
    assert pol_pad.select(16, 4) == 16

    cfg = parse_runtime_config(
        {
            "enabled": True,
            "buckets": [4, 8],
            "verify_buckets": [16],
            "fused_pool_io": "auto",
            "pool_gather": True,
        }
    )
    assert cfg["enabled"] is True
    assert cfg["buckets"] == (4, 8)
    assert cfg["verify_buckets"] == (16,)
    owned = explicit_profile_settings(cfg)
    assert owned["buckets"] == (4, 8)
    assert owned["verify_buckets"] == (16,)
    assert owned["fused_pool_io"] == "auto"
    assert owned["pool_gather"] is True

    # BLOCKER 2 regression: Profile with pool_gather=True must normalize
    # True -> "eligible" via Profile.exec_for and never re-apply raw True on T=1.
    g = qwen35.Qwen35Geometry(
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
    )
    ms = qwen35.qwen35_spec(g)
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

    # Explicit base_exec pool_gather="split" is preserved when profile pool_gather=None
    base_split = with_cache(base_ex, pool_gather="split")
    ex_split_t4, _, use_gpr_split = bucket_exec_spec(
        ms,
        base_split,
        DEFAULT_PROFILE,
        bucket_t=4,
        tokens_per_seq=4,
        fused_pool_io=False,
        fused_pool_io_max_b=DEFAULT_PROFILE.fused_pool_io_max_b,
        fused_pool_io_min_b_hi=DEFAULT_PROFILE.fused_pool_io_min_b_hi,
    )
    assert ex_split_t4.cache.pool_gather is False
    assert not use_gpr_split


def test_pack_step_meta_and_resolve_step_positions_jax():
    """Test `pack_step_meta` and `resolve_step_positions` over `JaxStepMetaOps`."""
    ops = JaxStepMetaOps()
    seq_lens = jnp.asarray([5, 10], dtype=jnp.int32)
    pos_t1 = resolve_step_positions(
        None, seq_lens, num_reqs=2, tokens_per_seq=1, like=seq_lens, ops=ops
    )
    np.testing.assert_array_equal(np.asarray(pos_t1), np.array([4, 9], dtype=np.int32))

    pos_t4 = resolve_step_positions(
        None, seq_lens, num_reqs=2, tokens_per_seq=4, like=seq_lens, ops=ops
    )
    np.testing.assert_array_equal(
        np.asarray(pos_t4), np.array([2, 3, 4, 5, 7, 8, 9, 10], dtype=np.int32)
    )

    # Short positions triggers fallback to seq_lens; M-RoPE 2D positions are preserved
    pos_short = jnp.asarray([4], dtype=jnp.int32)
    pos_from_short = resolve_step_positions(
        pos_short, seq_lens, num_reqs=2, tokens_per_seq=1, like=seq_lens, ops=ops
    )
    np.testing.assert_array_equal(
        np.asarray(pos_from_short), np.array([4, 9], dtype=np.int32)
    )
    pos_mrope = jnp.stack([pos_t4, pos_t4, pos_t4], axis=0)
    pos_from_mrope = resolve_step_positions(
        pos_mrope, seq_lens, num_reqs=2, tokens_per_seq=4, like=seq_lens, ops=ops
    )
    assert pos_from_mrope.shape == (8,)
    np.testing.assert_array_equal(np.asarray(pos_from_mrope), np.asarray(pos_t4))

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


@pytest.mark.parametrize(
    ("overrides", "expected_reason", "expected_level", "expected_substr"),
    [
        (
            {"num_layers": 30, "expected_num_layers": 60},
            "layer_count_mismatch",
            "debug",
            "",
        ),
        (
            {"unified_cache_layout": False},
            "unaligned_cache_layout",
            "warning",
            "unified KV and recurrent cache layout",
        ),
        (
            {"multimodal_inputs": True},
            "multimodal_inputs",
            "warning",
            "multimodal inputs unsupported",
        ),
        (
            {"lora_enabled": True},
            "lora_enabled",
            "warning",
            "does not support LoRA",
        ),
        (
            {"pipeline_parallel": True},
            "pipeline_parallel",
            "warning",
            "single-stage pipeline",
        ),
        (
            {"context_parallel": True},
            "context_parallel",
            "warning",
            "context parallelism",
        ),
        (
            {"dp_lockstep": True},
            "dp_lockstep",
            "warning",
            "data-parallel lockstep",
        ),
        (
            {"aux_hidden_states": True},
            "aux_hidden_states",
            "warning",
            "auxiliary hidden-state",
        ),
        (
            {"tp_size": 4, "supported_tp_sizes": (8,)},
            "unsupported_tp_size",
            "warning",
            "8 TPU devices in mesh, got 4",
        ),
        (
            {"tp_size": 4, "supported_tp_sizes": (2, 8)},
            "unsupported_tp_size",
            "warning",
            "TPU mesh size in (2, 8), got 4",
        ),
        (
            {"tp_size": 4, "supported_tp_sizes": ()},
            "unsupported_tp_size",
            "warning",
            "does not declare any supported TPU mesh sizes",
        ),
    ],
)
def test_startup_eligibility_table_driven(
    overrides: dict,
    expected_reason: str,
    expected_level: str,
    expected_substr: str,
):
    """Table-driven test covering `check_startup_eligibility` and post-mesh checks."""
    base_kwargs = dict(
        num_layers=60,
        expected_num_layers=60,
        unified_cache_layout=True,
        multimodal_inputs=False,
        lora_enabled=False,
        pipeline_parallel=False,
        context_parallel=False,
        dp_lockstep=False,
        aux_hidden_states=False,
        tp_size=8,
        supported_tp_sizes=(8,),
    )
    res_ok = check_startup_eligibility(**base_kwargs)
    assert res_ok.eligible and res_ok.reason == "ok" and res_ok.code == "ok"

    kw = {**base_kwargs, **overrides}
    res = check_startup_eligibility(**kw)
    assert not res.eligible
    assert res.reason == expected_reason
    assert res.code == expected_reason
    assert res.level == expected_level
    assert expected_substr in res.message

    post_ok = check_post_mesh_eligibility(
        tp_size=8,
        mesh_size=8,
        device_kind="TPU7x",
        bucket_policy=BucketPolicy((8,), ()),
    )
    assert post_ok.eligible and post_ok.reason == "ok"
    res_mesh = check_post_mesh_eligibility(tp_size=4, mesh_size=2)
    assert not res_mesh.eligible and res_mesh.reason == "mesh_size_mismatch"
    assert "mesh has 2 devices, expected 4" in res_mesh.message
    res_chip = check_post_mesh_eligibility(tp_size=8, mesh_size=8, device_kind="TPU v5")
    assert not res_chip.eligible and res_chip.reason == "unsupported_tpu_chip"
    res_bkts = check_post_mesh_eligibility(
        tp_size=8, mesh_size=8, bucket_policy=BucketPolicy((), ())
    )
    assert not res_bkts.eligible and res_bkts.reason == "empty_buckets"
    assert res_bkts.level == "info"


def test_step_eligibility_batch_shape_fallbacks_and_tracker():
    """Table-driven test for `inspect_batch_shape` and `check_step_eligibility`."""
    assert check_qsl_uniform([0, 1, 2, 3], 3) == BatchShape(True, 3, 1, "ok")
    assert check_qsl_uniform([0, 4, 8], 2) == BatchShape(True, 2, 4, "ok")
    assert check_qsl_uniform([0, 1, 5], 2) == BatchShape(False, 2, 0, "stock_mixed")
    assert check_qsl_uniform([0, 8], 1) == BatchShape(False, 1, 0, "stock_q8")
    assert check_qsl_uniform([0], 2) == BatchShape(False, 0, 0, "stock_n0")
    assert check_qsl_uniform([0, 0], 1) == BatchShape(False, 1, 0, "stock_q0")

    mrope_ok = np.array([[1, 2], [1, 2], [1, 2]], dtype=np.int32)
    mrope_bad = np.array([[1, 2], [1, 3], [1, 2]], dtype=np.int32)
    assert check_mrope_uniform(mrope_ok, 2) is True
    assert check_mrope_uniform(mrope_bad, 2) is False

    # Runner path vs fallback path in inspect_batch_shape
    bs_runner = inspect_batch_shape(
        num_reqs=2,
        has_roff=True,
        decode_and_total_reqs=(2, 2),
        qsl=np.array([0, 4, 8], dtype=np.int32),
    )
    assert bs_runner == BatchShape(True, 2, 4, "ok")

    # num_reqs=0 is not None -> fails closed with stock_n0 (no fallback_num_seqs)
    bs_n0 = inspect_batch_shape(
        num_reqs=0,
        has_roff=False,
        decode_and_total_reqs=None,
        qsl=np.array([0, 1, 2, 3], dtype=np.int32),
        fallback_decode_and_total_reqs=(3, 3),
        fallback_num_seqs=3,
        fallback_qsl=np.array([0, 1, 2, 3], dtype=np.int32),
    )
    assert bs_n0 == BatchShape(False, 0, 0, "stock_n0")

    # Fallback path when num_reqs is None and runner qsl/dist are None
    bs_fb = inspect_batch_shape(
        num_reqs=None,
        has_roff=False,
        decode_and_total_reqs=None,
        qsl=None,
        fallback_decode_and_total_reqs=(3, 3),
        fallback_num_seqs=3,
        fallback_qsl=np.array([0, 1, 2, 3], dtype=np.int32),
    )
    assert bs_fb == BatchShape(True, 3, 1, "ok")

    # Missing both qsl and fallback_qsl fails closed with stock_no_qsl
    bs_no_qsl = inspect_batch_shape(
        num_reqs=3,
        has_roff=False,
        decode_and_total_reqs=(3, 3),
        qsl=None,
    )
    assert bs_no_qsl == BatchShape(False, 0, 0, "stock_no_qsl")

    # Primary qsl without roff rejects T=4 (stock_prefill); fallback_qsl allows T=4
    bs_no_roff_q4 = inspect_batch_shape(
        num_reqs=2,
        has_roff=False,
        qsl=np.array([0, 4, 8], dtype=np.int32),
    )
    assert bs_no_roff_q4 == BatchShape(False, 2, 0, "stock_prefill")
    bs_fb_q4 = inspect_batch_shape(
        num_reqs=None,
        has_roff=False,
        qsl=None,
        fallback_num_seqs=2,
        fallback_qsl=np.array([0, 4, 8], dtype=np.int32),
    )
    assert bs_fb_q4 == BatchShape(True, 2, 4, "ok")

    pol = BucketPolicy((8, 16), (16, 32), token_paddings=(16, 32))
    steps = {(1, 8): object(), (1, 16): object(), (4, 16): object(), (4, 32): object()}
    out_rows = {(1, 8): 16, (1, 16): 16, (4, 16): 16, (4, 32): 32}

    cases = [
        (
            dict(
                batch_shape=bs_runner,
                input_rows=16,
                mrope_ok=False,
            ),
            False,
            "stock_mrope",
            0,
        ),
        (
            dict(
                batch_shape=BatchShape(False, 2, 0, "stock_mixed"),
                input_rows=16,
            ),
            False,
            "stock_mixed",
            0,
        ),
        (
            dict(
                batch_shape=BatchShape(True, 2, 2, "ok"),
                input_rows=16,
            ),
            False,
            "stock_q2",
            0,
        ),
        (
            dict(
                batch_shape=BatchShape(True, 20, 1, "ok"),
                input_rows=32,
            ),
            False,
            "stock_t1_n20",
            0,
        ),
        (
            dict(
                batch_shape=BatchShape(True, 10, 4, "ok"),
                input_rows=64,
            ),
            False,
            "stock_t4_n10",
            0,
        ),
        (
            dict(
                batch_shape=BatchShape(True, 4, 1, "ok"),
                input_rows=64,
            ),
            False,
            "stock_t1_rows8",
            8,
        ),
        (
            dict(
                batch_shape=BatchShape(True, 4, 1, "ok"),
                input_rows=16,
            ),
            True,
            "mk_t1_b8",
            8,
        ),
        (
            dict(
                batch_shape=bs_runner,
                input_rows=16,
            ),
            True,
            "mk_t4_b16",
            16,
        ),
    ]
    for kw, exp_el, exp_reason, exp_bkt in cases:
        bs_case = kw["batch_shape"]
        dec = check_step_eligibility(
            bs_case,
            pol,
            steps,
            out_rows,
            input_rows=kw["input_rows"],
            mrope_ok=kw.get("mrope_ok", True),
        )
        assert dec.eligible is exp_el, kw
        assert dec.reason == exp_reason, kw
        assert dec.bucket_t == exp_bkt, kw

    # Missing compiled bucket key returns stock_t1_nobkt (bare int key 8 never matched)
    dec_nobkt = check_step_eligibility(
        BatchShape(True, 4, 1, "ok"),
        pol,
        {8: object()},
        out_rows,
        input_rows=16,
    )
    assert not dec_nobkt.eligible and dec_nobkt.reason == "stock_t1_nobkt"

    tracker = StepMixTracker(period_s=0.0)
    run_ok, snap = tracker.note_step("mk_t1_b8", True)
    assert run_ok is True
    assert snap is not None and snap[1] == 1 and snap[4] == {"mk_t1_b8": 1}

    ab = MKABCalibrator(enabled=True, period_s=60.0)
    run0, k0, d0 = ab.choose("mk_t1_b8", True)
    run1, k1, d1 = ab.choose("mk_t1_b8", True)
    assert (run0, k0, d0) == (True, "mk_t1_b8_ab_mk", "mk")
    assert (run1, k1, d1) == (False, "mk_t1_b8_ab_stock", "stock")
    ab.record(1.2, padded_tokens=8)


@pytest.mark.parametrize(
    ("arch", "model_type", "num_experts", "exp_found", "exp_eligible", "exp_reason"),
    [
        # 1-2. Canonical MoE model_types
        ("UnknownArch", "qwen3_5_moe", 512, True, True, "ok"),
        ("UnknownArch", "qwen3_5_moe_text", 512, True, True, "ok"),
        # 3. Shared family model_type (MoE vs dense disambiguated by validate_hf_config)
        ("UnknownArch", "qwen3_5", 512, True, True, "ok"),
        ("UnknownArch", "qwen3_5", 0, True, False, "non_moe_checkpoint"),
        # 4-5. Canonical MoE HF architectures
        ("Qwen3_5MoeForConditionalGeneration", "unknown", 512, True, True, "ok"),
        ("Qwen3_5MoeForCausalLM", "unknown", 512, True, True, "ok"),
        # 6-7. Shared family HF architectures (MoE vs dense)
        ("Qwen3_5ForConditionalGeneration", "unknown", 512, True, True, "ok"),
        (
            "Qwen3_5ForConditionalGeneration",
            "unknown",
            0,
            True,
            False,
            "non_moe_checkpoint",
        ),
        ("Qwen3_5ForCausalLM", "unknown", 512, True, True, "ok"),
        ("Qwen3_5ForCausalLM", "unknown", None, True, False, "non_moe_checkpoint"),
        # Unsupported model keys return None from find_model
        ("LlamaForCausalLM", "llama", 512, False, False, ""),
        ("Qwen2_5ForCausalLM", "qwen2_5", 512, False, False, ""),
    ],
)
def test_qwen35_seven_routing_keys_and_dense_rejection(
    arch: str,
    model_type: str,
    num_experts: int | None,
    exp_found: bool,
    exp_eligible: bool,
    exp_reason: str,
):
    """Verify all 7 `Qwen35` routing keys resolve to `Qwen35` and reject dense."""
    cfg = dict(qwen35.Qwen35.default_hf_config)
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
    assert found is qwen35.Qwen35
    el = found.validate_hf_config(cfg)
    assert el.eligible is exp_eligible
    assert el.reason == exp_reason


def test_schedule_driven_validate_hf_config(isolated_model_registry):
    """`ModelDef.validate_hf_config` resolves `Schedule.ffn` across schedule forms."""
    valid_qwen = dict(qwen35.Qwen35.default_hf_config)
    assert qwen35.Qwen35.validate_hf_config(valid_qwen).eligible

    dense_qwen = {k: v for k, v in valid_qwen.items() if k != "num_experts"}
    res_dense = qwen35.Qwen35.validate_hf_config(dense_qwen)
    assert not res_dense.eligible and res_dense.reason == "non_moe_checkpoint"
    assert "Qwen35 supports MoE checkpoints only" in res_dense.message

    # Non-numeric num_experts (e.g. string "512") returns non_moe_checkpoint cleanly
    str_experts_qwen = {**valid_qwen, "num_experts": "512"}
    res_str = qwen35.Qwen35.validate_hf_config(str_experts_qwen)
    assert not res_str.eligible and res_str.reason == "non_moe_checkpoint"

    # Missing hidden_size still checks present geometry keys (e.g. head_dim)
    no_hidden_bad_hd = {"num_experts": 512, "head_dim": 64}
    res_hd = qwen35.Qwen35.validate_hf_config(no_hidden_bad_hd)
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
        **dense_qwen,
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

    # pool_schedule_info raises MegakernelLayoutError when cache_backend is absent
    with pytest.raises(mk.MegakernelLayoutError, match="has no pool backend layout"):
        pool_schedule_info(types.SimpleNamespace(cache_backend=None))


def _random_hf_from_model_def(
    key: jax.Array, mdef: type[ModelDef], g: mk.ModelGeometry, R: int
) -> dict[str, jax.Array]:
    """Generate canonical HF weights purely from `mdef`'s `hf_params()` declarations."""
    ms = mdef.to_model_spec(g, R=R)
    sched = mdef.schedule
    assert sched is not None
    pre_n = sched.pre_norm.resolve(g) if sched.pre_norm is not None else None
    post_n = sched.post_norm.resolve(g) if sched.post_norm is not None else None
    final_n = mdef.final_norm.resolve(g) if mdef.final_norm is not None else None
    embed_n = (
        mdef.embed.resolve(mdef.default_hf_config)
        if mdef.embed is not None and mdef.default_hf_config is not None
        else None
    )
    lm_head_m = mdef.lm_head

    mod_groups: list[tuple[Any, tuple[int, ...]]] = [
        (pre_n, (g.L,)),
        (post_n, (g.L,)),
        (final_n, ()),
        (embed_n, ()),
    ]
    for mod in ms.all_modules():
        if isinstance(mod, mk.GQAModule):
            mod_groups.append((mod, (g.NATT,)))
        elif isinstance(mod, mk.GDNModule):
            mod_groups.append((mod, (g.NGDN,)))
        else:
            mod_groups.append((mod, (g.L,)))

    hf: dict[str, jax.Array] = {}
    idx = 0
    for mod, lead in mod_groups:
        if mod is None:
            continue
        decls = (
            mod.hf_params()
            if mod in (pre_n, post_n, final_n, embed_n)
            else mod.hf_params(g=g)
        )
        for d in decls:
            if d.name in hf:
                continue
            idx += 1
            subkey = jax.random.fold_in(key, idx)
            full_shape = (*lead, *d.shape)
            if d.transform.scale_for is not None:
                hf[d.name] = jnp.full(full_shape, 0.02, dtype=d.dtype)
            elif d.dtype == jnp.float8_e4m3fn:
                w_f32 = jax.random.normal(subkey, full_shape, dtype=jnp.float32) * 0.2
                hf[d.name] = w_f32.astype(jnp.float8_e4m3fn)
            else:
                w_f32 = jax.random.normal(subkey, full_shape, dtype=jnp.float32) * 0.05
                hf[d.name] = w_f32.astype(d.dtype)
    if lm_head_m is not None and lm_head_m.tied_to is None and embed_n is not None:
        idx += 1
        subkey = jax.random.fold_in(key, idx)
        vocab = int(embed_n.vocab)
        w_f32 = jax.random.normal(subkey, (vocab, g.H), dtype=jnp.float32) * 0.05
        hf[lm_head_m.prefix] = w_f32.astype(jnp.bfloat16)
    return hf


def test_second_framework_plain_jax_adapter_on_synthetic_model_def(
    isolated_model_registry,
):
    """Plain-JAX second framework adapter on a non-Qwen `Every(n=2)` `ModelDef`."""
    synth_hf_cfg = {
        "model_type": "synth_jax_rt",
        "architectures": ["SynthJaxRtForCausalLM"],
        "vocab_size": 256,
        "hidden_size": 256,
        "num_hidden_layers": 4,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500_000.0,
        "partial_rotary_factor": 0.25,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "num_experts": 128,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 256,
        "shared_expert_intermediate_size": 256,
    }

    @register_model(
        name="synth_jax_rt",
        hf_architectures=("SynthJaxRtForCausalLM",),
        hf_model_type="synth_jax_rt",
    )
    class SynthJaxRt(ModelDef):
        display_name = "SynthJaxRt"
        supported_tp_sizes = (2,)
        default_hf_config = synth_hf_cfg
        geometry = FromHF(
            hidden="hidden_size",
            layers="num_hidden_layers",
            vocab="vocab_size",
            eps="rms_norm_eps",
            rope_theta="rope_theta",
            rope_partial=("partial_rotary_factor", 1.0),
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
        )
        schedule = Schedule(
            mixer=Every(n=2, last=attn, rest=linear),
            ffn=moe,
            pre_norm=Norm.bind("input_layernorm", zero_centered=True),
            post_norm=Norm.bind("post_attention_layernorm", zero_centered=True),
        )
        embed = Embedding.bind(
            "model.embed_tokens", vocab="vocab_size", hidden="hidden_size"
        )
        final_norm = Norm.bind("model.norm", zero_centered=True)
        lm_head = LMHead("lm_head", tied_to=None)

    # 1. Startup & HF config validation via library runtime API
    assert SynthJaxRt.validate_hf_config(synth_hf_cfg).eligible
    g = SynthJaxRt.derive_geometry(synth_hf_cfg)
    ms = SynthJaxRt.to_model_spec(g, R=2)
    pinfo = pool_schedule_info(ms)
    assert (pinfo.num_pools, pinfo.recurrent_slots_per_pool, pinfo.att_every) == (
        2,
        1,
        2,
    )

    prof = resolve_driver_profile(
        DEFAULT_PROFILE, model_name=ms.name, fused_pool_io=False, pool_gather=False
    )
    pol = BucketPolicy.resolve(
        prof,
        buckets=(2,),
        verify_buckets=(),
        buckets_explicit=True,
        verify_buckets_explicit=True,
    )
    el = check_startup_eligibility(
        num_layers=g.L,
        expected_num_layers=g.L,
        unified_cache_layout=True,
        tp_size=2,
        supported_tp_sizes=SynthJaxRt.supported_tp_sizes,
    )
    assert el.eligible
    el_post = check_post_mesh_eligibility(
        tp_size=2,
        mesh_size=2,
        device_kind="cpu",
        bucket_policy=pol,
    )
    assert el_post.eligible

    # 2. Build base ExecSpec, hf_params()-driven weights, and pools
    num_pages, page_size, max_pages_per_seq = 96, 128, 2
    base_ex = ExecSpec(
        R=2,
        B=2,
        T=1,
        ctx=256,
        interpret=True,
        stack_e=False,
        tiles=dataclasses.replace(TINY_TILES, bank=512 << 10, nbd=6).with_modules(
            {"gqa": {"kv_blk": 128}}
        ),
        cache=CacheExec(
            paged_kv=True,
            num_pages=num_pages,
            page_size=page_size,
            max_pages_per_seq=max_pages_per_seq,
        ),
    )
    base_ex = align_cache_exec(
        base_ex,
        prof,
        g,
        num_pages=num_pages,
        page_size=page_size,
        max_pages_per_seq=max_pages_per_seq,
    )
    ex_b2, use_fused, use_gpr = bucket_exec_spec(
        ms,
        base_ex,
        prof,
        bucket_t=2,
        tokens_per_seq=1,
        fused_pool_io=False,
        fused_pool_io_max_b=prof.fused_pool_io_max_b,
        fused_pool_io_min_b_hi=prof.fused_pool_io_min_b_hi,
    )
    assert (use_fused, use_gpr) == (False, False)

    mesh = jax.sharding.Mesh(np.asarray(jax.devices("cpu")[:2]), ("tp",))
    vocab_size = int(synth_hf_cfg["vocab_size"])
    hf_raw = _random_hf_from_model_def(jax.random.PRNGKey(7), SynthJaxRt, g, R=2)
    assert "embed_tokens" in hf_raw and hf_raw["embed_tokens"].shape == (
        vocab_size,
        g.H,
    )
    assert "lm_head" in hf_raw and hf_raw["lm_head"].shape == (vocab_size, g.H)

    w_l = mk.load_logical_weights(SynthJaxRt, ex_b2, hf_raw, g=g)
    w_mk = mk.pack_logical_weights(SynthJaxRt, ex_b2, w_l, g=g)
    w_rank0 = mk.load_weights(
        SynthJaxRt, ex_b2, mk.MappingWeightSource(w_l), g=g, tp_rank=0
    )

    pool_shape = pinfo.expected_pool_shape(
        num_pages, max(1, g.NKV // ex_b2.R), g.HD, page_size
    )
    assert pool_shape == (num_pages, 2, g.HD // 4, 4, page_size)
    tp_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("tp", None, None, None, None)
    )
    raw_pools_rank0 = [
        jnp.zeros(pool_shape, dtype=jnp.float8_e4m3fn) for _ in range(pinfo.num_pools)
    ]
    raw_pools = [
        jax.device_put(
            jnp.zeros((ex_b2.R * num_pages, *pool_shape[1:]), dtype=jnp.float8_e4m3fn),
            tp_sharding,
        )
        for _ in range(pinfo.num_pools)
    ]

    ok_hlo, msg_hlo = verify_pooled_step_hlo(
        mesh,
        ms,
        ex_b2,
        w_rank0["w_flat"],
        raw_pools_rank0,
        max_pages_per_seq=max_pages_per_seq,
        ckpt_cols=2,
    )
    assert ok_hlo, msg_hlo

    # 3. Warmup plan + sentinel check via `make_pooled_step` + `pack_step_meta`
    warmup_steps, warmed_bkts, n_save = build_warmup_plan(
        ms,
        pol.buckets,
        pol.verify_buckets,
        {(1, 2): False},
        {(1, 2): False},
        num_pages=num_pages,
        max_pages_per_seq=max_pages_per_seq,
        ckpt_window=2,
        is_eager=True,
    )
    assert warmed_bkts == [(1, 2)] and n_save > 0
    sent = warmup_steps[0].sentinel
    assert sent is not None

    step_fn, _ = mk.make_pooled_step(
        mesh, ms, ex_b2, donate_pools=False, fused_pool_io=False
    )
    ops = JaxStepMetaOps()
    ids_b = jnp.asarray([1, 0], dtype=jnp.int32)
    pos_b = jnp.zeros((2,), dtype=jnp.int32)
    bt_b = jnp.asarray([[sent.kv_page, 0], [0, 0]], dtype=jnp.int32)
    s_g0 = jnp.asarray([sent.gdn_base, 0], dtype=jnp.int32)
    c_g0 = jnp.asarray([[sent.gdn_base, sent.gdn_base + 1], [0, 0]], dtype=jnp.int32)
    r_neg = jnp.full((32,), -1, dtype=jnp.int32)

    bs = inspect_batch_shape(
        num_reqs=1,
        has_roff=True,
        decode_and_total_reqs=(1, 1),
        qsl=np.array([0, 1], dtype=np.int32),
    )
    dec = check_step_eligibility(bs, pol, {(1, 2): step_fn}, {(1, 2): 2}, input_rows=2)
    assert dec.eligible and dec.reason == "mk_t1_b2"

    meta = pack_step_meta(
        ids_b,
        pos_b,
        bt_b,
        ((s_g0, c_g0, r_neg),),
        bucket_t=2,
        tokens_per_seq=1,
        max_pages_per_seq=max_pages_per_seq,
        has_roff=True,
        use_ckpt_2d=True,
        ops=ops,
    )
    assert meta.ckpt_b.shape == (1, 2, 2)
    assert meta.roff_b.shape == (1, 2)
    kv_before = [np.asarray(p[sent.kv_page]).view(np.uint8).copy() for p in raw_pools]
    gdn_before = [
        np.asarray(p[sent.gdn_p0 : sent.gdn_p0 + pinfo.rec_pages_per_slot])
        .view(np.uint8)
        .copy()
        for p in raw_pools
    ]
    tp_hid_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("tp", None)
    )
    hid_sent = jax.device_put(
        jnp.ones((ex_b2.R * ex_b2.M, g.H), dtype=jnp.bfloat16) * 0.05, tp_hid_sharding
    )
    w_mk_sharded = {}
    for k, v in w_mk.items():
        if k in ("plan", "w_flat"):
            continue
        if isinstance(v, tuple):
            w_mk_sharded[k] = tuple(
                jax.device_put(
                    x,
                    jax.sharding.NamedSharding(
                        mesh, jax.sharding.PartitionSpec("tp", *([None] * (x.ndim - 1)))
                    ),
                )
                for x in v
            )
        else:
            w_mk_sharding = jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec("tp", *([None] * (v.ndim - 1)))
            )
            w_mk_sharded[k] = jax.device_put(v, w_mk_sharding)

    out_sent, pools_after = step_fn(
        meta.pos_b,
        hid_sent,
        w_mk_sharded,
        tuple(raw_pools),
        meta.pt_b,
        meta.ckpt_b,
        meta.roff_b,
    )
    kv_after = [np.asarray(p[sent.kv_page]).view(np.uint8).copy() for p in pools_after]
    gdn_after = [
        np.asarray(p[sent.gdn_p0 : sent.gdn_p0 + pinfo.rec_pages_per_slot])
        .view(np.uint8)
        .copy()
        for p in pools_after
    ]
    verify_sentinel_step(
        np.asarray(out_sent[:2], dtype=np.float32),
        kv_before,
        kv_after,
        gdn_before,
        gdn_after,
        tokens_per_seq=1,
        bucket_t=2,
        variant_key=sent.variant_key,
    )

    # 4. Real serving step with Embedding + MakePooledStep + LMHead vs reference_step
    # Use generic module state declarations (`mod.states`) and `raw_pools`.
    ex_unpaged = with_cache(ex_b2, paged_kv=False)
    st_l = {
        s.name: jnp.zeros((ex_b2.R, g.NATT, *s.shape), dtype=s.dtype)
        for mod in ms.all_modules()
        if isinstance(mod, mk.GQAModule)
        for s in mod.states(g, ex_unpaged)
        if s.name in ("kc", "vc")
    }
    pt_step = jnp.asarray([[70, 71], [72, 73]], dtype=jnp.int32)
    pools_step = tuple(raw_pools)

    step_ids = jnp.asarray([7, 13], dtype=jnp.int32)
    step_seq_lens = jnp.asarray([1, 1], dtype=jnp.int32)
    step_pos = resolve_step_positions(
        None, step_seq_lens, num_reqs=2, tokens_per_seq=1, like=step_seq_lens, ops=ops
    )
    s_live = jnp.asarray([1, 5], dtype=jnp.int32)
    c_live = jnp.asarray([[1, 2], [5, 6]], dtype=jnp.int32)
    r_live = jnp.zeros((16,), dtype=jnp.int32)
    meta_live = pack_step_meta(
        step_ids,
        step_pos,
        pt_step.reshape(-1),
        ((s_live, c_live, r_live),),
        bucket_t=2,
        tokens_per_seq=1,
        max_pages_per_seq=max_pages_per_seq,
        has_roff=True,
        use_ckpt_2d=True,
        ops=ops,
    )

    # Embed via SynthJaxRt's declared `embed` weight, run step, project via `lm_head`
    assert SynthJaxRt.embed is not None and SynthJaxRt.lm_head is not None
    embed_mod = SynthJaxRt.embed.resolve(synth_hf_cfg)
    hid_l, _ = embed_mod.reference(
        {"embed_tokens": hf_raw["embed_tokens"]}, meta_live.ids_b
    )
    hid_pad = jnp.pad(hid_l, ((0, ex_b2.M - 2), (0, 0)))
    hid_sharded = jax.device_put(jnp.tile(hid_pad, (ex_b2.R, 1)), tp_hid_sharding)

    out_mk, pools_after_live = step_fn(
        meta_live.pos_b,
        hid_sharded,
        w_mk_sharded,
        pools_step,
        meta_live.pt_b,
        meta_live.ckpt_b,
        meta_live.roff_b,
    )
    out_ref, st_after_0 = SynthJaxRt.reference_step(
        g, ex_b2, meta_live.pos_b, hid_l, w_l, st_l, paged_kv=True, pool_io=True
    )
    np.testing.assert_allclose(
        np.asarray(out_mk[:2], dtype=np.float32),
        np.asarray(out_ref, dtype=np.float32),
        atol=8e-2,
        rtol=8e-2,
    )

    # Step 2 reads back the KV and GDN state written into `pools_after_live` by step 1
    st_l2 = {
        "kc": st_after_0["kc"],
        "vc": st_after_0["vc"],
        "conv": st_after_0["conv"][:, :, : 3 * ex_b2.B].reshape(
            ex_b2.R, g.NGDN, ex_b2.NS, 3, -1
        ),
        "rec": st_after_0["rec"].reshape(
            ex_b2.R, g.NGDN, ex_b2.NS, g.VH // ex_b2.R, g.KD, g.VD
        ),
    }
    step_ids_2 = jnp.asarray([19, 23], dtype=jnp.int32)
    step_seq_lens_2 = jnp.asarray([2, 2], dtype=jnp.int32)
    step_pos_2 = resolve_step_positions(
        None,
        step_seq_lens_2,
        num_reqs=2,
        tokens_per_seq=1,
        like=step_seq_lens_2,
        ops=ops,
    )
    meta_live_2 = pack_step_meta(
        step_ids_2,
        step_pos_2,
        pt_step.reshape(-1),
        ((s_live, c_live, r_live),),
        bucket_t=2,
        tokens_per_seq=1,
        max_pages_per_seq=max_pages_per_seq,
        has_roff=True,
        use_ckpt_2d=True,
        ops=ops,
    )
    hid_l2, _ = embed_mod.reference(
        {"embed_tokens": hf_raw["embed_tokens"]}, meta_live_2.ids_b
    )
    hid_pad2 = jnp.pad(hid_l2, ((0, ex_b2.M - 2), (0, 0)))
    hid_sharded2 = jax.device_put(jnp.tile(hid_pad2, (ex_b2.R, 1)), tp_hid_sharding)
    out_mk2, _ = step_fn(
        meta_live_2.pos_b,
        hid_sharded2,
        w_mk_sharded,
        pools_after_live,
        meta_live_2.pt_b,
        meta_live_2.ckpt_b,
        meta_live_2.roff_b,
    )
    out_ref2, _ = SynthJaxRt.reference_step(
        g, ex_b2, meta_live_2.pos_b, hid_l2, w_l, st_l2, paged_kv=True, pool_io=True
    )
    np.testing.assert_allclose(
        np.asarray(out_mk2[:2], dtype=np.float32),
        np.asarray(out_ref2, dtype=np.float32),
        atol=8e-2,
        rtol=8e-2,
    )
    w_lm = hf_raw["lm_head"].astype(jnp.float32)
    logits_mk = jnp.matmul(out_mk2[:2].astype(jnp.float32), w_lm.T)
    logits_ref = jnp.matmul(out_ref2.astype(jnp.float32), w_lm.T)
    assert logits_mk.shape == (2, vocab_size)
    np.testing.assert_allclose(
        np.asarray(logits_mk, dtype=np.float32),
        np.asarray(logits_ref, dtype=np.float32),
        atol=1e-1,
        rtol=1e-1,
    )


def _base_inspect_step_detail_ref(
    num_reqs: int | None,
    has_roff: bool,
    comb_wt: tuple[int, int] | None,
    qsl_host: list[int] | None,
    dist_dt: tuple[int, int] | None,
    fb_nseq: int | None,
    qsl_md: list[int] | None,
) -> BatchShape:
    """Direct port of base (`ab8db9c5a`) `MegakernelDriver._inspect_step_detail`."""
    if has_roff and comb_wt is not None:
        num_windowed, dist_reqs = int(comb_wt[0]), int(comb_wt[1])
        if num_reqs is None:
            num_reqs = dist_reqs
        if dist_reqs <= 0 or num_reqs <= 0:
            return BatchShape(False, 0, 0, "stock_n0")
        if num_windowed != dist_reqs:
            return BatchShape(
                False,
                num_reqs,
                0,
                "stock_prefill" if num_windowed == 0 else "stock_mixed",
            )
    if qsl_host is not None and num_reqs is not None:
        bs = check_qsl_uniform(qsl_host, num_reqs)
        if bs.eligible and not has_roff and bs.tokens_per_seq != 1:
            return BatchShape(False, bs.num_reqs, 0, "stock_prefill")
        return bs

    if dist_dt is not None:
        d_end, total_reqs = int(dist_dt[0]), int(dist_dt[1])
        if total_reqs <= 0:
            return BatchShape(False, 0, 0, "stock_n0")
        if d_end != total_reqs:
            return BatchShape(
                False,
                total_reqs,
                0,
                "stock_prefill" if d_end == 0 else "stock_mixed",
            )
        if num_reqs is None:
            num_reqs = total_reqs
    if num_reqs is None and fb_nseq is not None:
        num_reqs = int(fb_nseq)
    if qsl_md is not None and num_reqs is not None:
        return check_qsl_uniform(qsl_md, num_reqs)
    return BatchShape(False, 0, 0, "stock_no_qsl")


def test_inspect_batch_shape_differential_vs_base_reference():
    """Exhaustively verify `inspect_batch_shape` matches base `_inspect_step_detail`."""
    num_reqs_vals = (None, -1, 0, 1, 2)
    has_roff_vals = (False, True)
    comb_vals = (None, (0, 0), (1, 0), (0, 2), (1, 2), (2, 2))
    qsl_vals = (None, [0], [0, 0], [0, 1, 2], [0, 4, 8], [0, 1, 5], [0, 8, 16])
    dist_vals = (None, (0, 0), (1, 2), (2, 2))
    fb_nseq_vals = (None, 0, 1, 2)
    qsl_md_vals = (None, [0], [0, 1, 2], [0, 4, 8], [0, 1, 5])

    total_checked = 0
    for nr in num_reqs_vals:
        for hr in has_roff_vals:
            for cw in comb_vals:
                for qh in qsl_vals:
                    for dt in dist_vals:
                        for fn in fb_nseq_vals:
                            for qm in qsl_md_vals:
                                total_checked += 1
                                got = inspect_batch_shape(
                                    num_reqs=nr,
                                    has_roff=hr,
                                    decode_and_total_reqs=cw,
                                    qsl=qh,
                                    fallback_decode_and_total_reqs=dt,
                                    fallback_num_seqs=fn,
                                    fallback_qsl=qm,
                                )
                                exp = _base_inspect_step_detail_ref(
                                    nr, hr, cw, qh, dt, fn, qm
                                )
                                assert got == exp, (
                                    f"mismatch for {(nr, hr, cw, qh, dt, fn, qm)}: "
                                    f"got={got} vs exp={exp}"
                                )
    assert total_checked == 5 * 2 * 6 * 7 * 4 * 4 * 5


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
        inspect_batch_shape(qsl_host=[0, 1])  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        inspect_batch_shape(comb_windowed_and_total=(1, 1))  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        resolve_driver_profile(DEFAULT_PROFILE, unknown_flag=True)  # type: ignore[call-arg]

    g = qwen35.Qwen35Geometry(
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
    )
    ms = qwen35.qwen35_spec(g)
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
            ms, (4,), (), {}, {}, num_pages=64, max_pages_per_seq=2, unknown=1
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
