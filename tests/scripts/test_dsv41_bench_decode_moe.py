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
"""100% line & branch coverage unit test suite for .

Targets:
1. scripts/dsv41_bench/bench_decode_moe.py (full module: 533 lines, 100%)
2. src/vllm_torchtpu/kernels/experimental/moe_dense_expert.py (diffs)
3. src/vllm_torchtpu/layers/core/fused_moe_gmm.py (diffs: 100%)
"""

from __future__ import annotations

import argparse
import functools
import os
import pathlib
import runpy
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "true")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_BENCH_DIR = _REPO_ROOT / "scripts" / "dsv41_bench"
_BENCH_FILE = _BENCH_DIR / "bench_decode_moe.py"

if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.dsv41_bench.bench_decode_moe as bdm  # noqa: E402
from scripts.dsv41_bench import common  # noqa: E402
from vllm_torchtpu.kernels.experimental import moe_dense_expert as kde  # noqa: E402
from vllm_torchtpu.layers.core import fused_moe_gmm as fmg  # noqa: E402


@pytest.fixture(autouse=True)
def _patch_tpu_device_to_cpu(monkeypatch):
    """Safely map TPU tensor calls to CPU during testing."""
    orig_to = torch.Tensor.to

    def mock_to(self, *args, **kwargs):
        if args and args[0] == "tpu":
            args = ("cpu", *args[1:])
        elif kwargs.get("device") == "tpu":
            kwargs["device"] = "cpu"
        return orig_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", mock_to)

    orig_tensor = torch.tensor

    def mock_tensor(*args, **kwargs):
        if kwargs.get("device") == "tpu":
            kwargs["device"] = "cpu"
        return orig_tensor(*args, **kwargs)

    monkeypatch.setattr(torch, "tensor", mock_tensor)

    for fn_name in (
        "arange",
        "zeros",
        "ones",
        "randn",
        "empty",
        "full",
        "randint",
    ):
        orig_fn = getattr(torch, fn_name)

        def make_mock(f):
            def mock_fn(*args, **kwargs):
                if kwargs.get("device") == "tpu":
                    kwargs["device"] = "cpu"
                return f(*args, **kwargs)

            return mock_fn

        monkeypatch.setattr(torch, fn_name, make_mock(orig_fn))


# ==============================================================================
# 1. Tests for scripts/dsv41_bench/bench_decode_moe.py
# ==============================================================================


class TestBenchDecodeMoe:
    """Full-module coverage tests for scripts/dsv41_bench/bench_decode_moe.py."""

    def test_bench_ms(self, monkeypatch):
        """Test _bench_ms helper returns the median of samples."""
        mock_ctx = MagicMock()
        monkeypatch.setattr(
            common,
            "time_steps",
            lambda ctx, fn, warmup=3, iters=10: ([1.23456, 1.23456], None),
        )
        res = bdm._bench_ms(mock_ctx, lambda: None, warmup=2, iters=4)
        assert res == 1.2346

    def test_run_mhc_sweep(self, monkeypatch):
        """Test _run_mhc_sweep across all t_full configurations."""
        ctx = common.WorkerContext(
            rank=0,
            world_size=8,
            tp_size=8,
            cpu_group=None,
            result_dir=pathlib.Path("/tmp"),
        )
        args = argparse.Namespace(seed=42, subop="all")

        # Mock TP group
        mock_tp_group = SimpleNamespace(rank_in_group=0)
        monkeypatch.setattr("vllm.distributed.get_tp_group", lambda: mock_tp_group)

        # Mock bench_mhc helpers to return deterministic tensors on CPU
        def mock_full_inputs(torch_mod, t_full, geo, seed):
            return {
                "x": torch.zeros((t_full, geo.hidden_size), dtype=torch.bfloat16),
                "residual": torch.zeros(
                    (t_full, geo.hc_mult * geo.hidden_size), dtype=torch.bfloat16
                ),
                "post_mix": torch.zeros((t_full, geo.hc_mult, 1)),
                "comb": torch.zeros((t_full, geo.hc_mult, geo.hc_mult)),
                "pre_mix": torch.zeros((t_full, geo.hc_mult)),
                "fn": torch.zeros((128, geo.hc_mult * geo.hidden_size)),
                "hc_scale": torch.zeros(3),
                "hc_base": torch.zeros(geo.hc_mult * (geo.hc_mult + 2)),
                "norm_weight": torch.zeros(geo.hidden_size, dtype=torch.bfloat16),
                "mixes": torch.zeros((t_full, geo.hc_mult * (geo.hc_mult + 2))),
                "sqrsum": torch.zeros((t_full, 1)),
            }

        def mock_slice_rows(d, lo, hi):
            return {
                k: v[lo:hi] if isinstance(v, torch.Tensor) and v.shape[0] > hi else v
                for k, v in d.items()
            }

        def mock_to_tpu(d):
            return d

        def mock_cpu(d):
            return d

        def mock_make_fn(subop, geo):
            def _fn(inp):
                t = inp["x"].shape[0]
                return {
                    "residual": inp["x"],
                    "value": torch.zeros((t, geo.hidden_size), dtype=torch.bfloat16),
                }

            return _fn

        monkeypatch.setattr(bdm.bench_mhc, "_full_inputs", mock_full_inputs)
        monkeypatch.setattr(bdm.bench_mhc, "_slice_rows", mock_slice_rows)
        monkeypatch.setattr(bdm.bench_mhc, "_to_tpu", mock_to_tpu)
        monkeypatch.setattr(bdm.bench_mhc, "_cpu", mock_cpu)
        monkeypatch.setattr(bdm.bench_mhc, "_make_fn", mock_make_fn)
        monkeypatch.setattr(bdm.bench_mhc, "_maybe_compile", lambda fn, flag: fn)
        monkeypatch.setattr(
            "vllm.distributed.tensor_model_parallel_all_reduce",
            lambda z: z,
        )
        monkeypatch.setattr(
            "vllm.distributed.tensor_model_parallel_reduce_scatter",
            lambda z, dim: z,
        )
        monkeypatch.setattr(
            "vllm.distributed.tensor_model_parallel_all_gather",
            lambda z, dim: z,
        )

        def mock_bench_ms(ctx, fn):
            fn()
            return 0.1234

        monkeypatch.setattr(bdm, "_bench_ms", mock_bench_ms)

        with common.model_wrapper_context():
            results = bdm._run_mhc_sweep(ctx, args)
        assert isinstance(results, dict)
        for t in (8, 16, 32, 64, 128, 256, 512):
            key = f"T={t}"
            assert key in results
            assert results[key]["T_full"] == t
            assert results[key]["T_sp_local"] == t // 8
            assert results[key]["mhc_all_tp_ms"] == 0.1234
            assert results[key]["total_2x_per_layer_tp_ms"] == 0.4936

    def test_run_moe_sweep_and_nested_ops(self, monkeypatch):
        """Test _run_moe_sweep and execute all nested jax_op functions."""
        ctx = common.WorkerContext(
            rank=0,
            world_size=8,
            tp_size=8,
            cpu_group=None,
            result_dir=pathlib.Path("/tmp"),
        )
        args = argparse.Namespace(seed=42, subop="all")

        # Mock ep_mesh
        mock_mesh = MagicMock()
        monkeypatch.setattr(
            "vllm_torchtpu.distributed.ep_mesh.build_ep_mesh",
            lambda: mock_mesh,
        )

        # Mock collectives
        monkeypatch.setattr(
            "vllm.distributed.tensor_model_parallel_all_reduce",
            lambda x: x,
        )
        monkeypatch.setattr(
            "vllm.distributed.tensor_model_parallel_reduce_scatter",
            lambda x, dim=0: x[: max(1, x.shape[0] // 8)],
        )
        monkeypatch.setattr(
            "vllm.distributed.tensor_model_parallel_all_gather",
            lambda x, dim=0: x.repeat(8, 1) if x.ndim == 2 else x,
        )

        def mock_bench_ms(ctx, fn):
            fn()
            return 0.05

        monkeypatch.setattr(bdm, "_bench_ms", mock_bench_ms)

        # Mock shard_map and inner ops before running _run_moe_sweep
        def mock_shard_map(
            fn, mesh=None, in_specs=None, out_specs=None, check_vma=False
        ):
            return lambda *a: fn(*a)

        monkeypatch.setattr(jax, "shard_map", mock_shard_map)
        monkeypatch.setattr(
            "vllm_torchtpu.layers.core.fused_moe_gmm.fused_moe_func",
            lambda *a, **k: a[0],
        )
        monkeypatch.setattr(
            "vllm_torchtpu.kernels.experimental.moe_dense_expert.dense_expert_moe",
            lambda *a, **k: a[0],
        )
        monkeypatch.setattr(
            "vllm_torchtpu.kernels.fused_moe.v2.fused_ep_moe_v2",
            lambda *a, **k: a[0],
        )

        captured_ops: list[Any] = []

        def mock_jax_op(
            name, fn, mesh=None, input_partition_specs=None, output_partition_specs=None
        ):
            mock_op = MagicMock()
            mock_op._name = name
            mock_op._fn = fn
            mock_op._fake_fn = None

            def fake_register(fake_fn):
                mock_op._fake_fn = fake_fn

            mock_op.register_fake = fake_register

            def op_call(*args, **kwargs):
                return args[0].clone()

            mock_op.side_effect = op_call
            captured_ops.append(mock_op)
            return mock_op

        monkeypatch.setattr("torch_tpu._internal.pallas.jax_op", mock_jax_op)

        with common.model_wrapper_context():
            results = bdm._run_moe_sweep(ctx, args)
        assert isinstance(results, dict)
        for t in (4, 8, 16, 32, 64):
            key = f"T={t}"
            assert key in results
            assert results[key]["T"] == t
            assert "dense_expert_tp_ms" in results[key]
            if t >= 8:
                assert "fused_ep_v2_w4a8_tp_ms" in results[key]
                assert "gmm_sc1_onehot_sp_ms" in results[key]
                assert "dense_expert_sp_ms" in results[key]
                assert "fused_ep_v2_w4a8_sp_ms" in results[key]

        # Verify all registered fake functions were called
        for op in captured_ops:
            assert op._fake_fn is not None
            dummy_in = torch.zeros((2, 2))
            fake_out = op._fake_fn(dummy_in)
            assert fake_out.shape == dummy_in.shape

        # Now exercise the inner _fn implementations of each captured op
        dummy_jax_args = (
            jnp.zeros((4, 512), dtype=jnp.bfloat16),
            jnp.zeros((6, 256, 512), dtype=jnp.uint8),
            jnp.zeros((6, 256, 512), dtype=jnp.uint8),
            jnp.zeros((6, 1, 1, 512), dtype=jnp.float32),
            jnp.zeros((6, 1, 1, 512), dtype=jnp.float32),
            jnp.zeros((4, 6), dtype=jnp.float32),
            jnp.zeros((4, 6), dtype=jnp.int32),
            jnp.zeros((), dtype=jnp.int32),
        )

        # Call GMM op _fn
        gmm_fn = captured_ops[0]._fn
        res_gmm = gmm_fn(*dummy_jax_args)
        assert res_gmm.shape == dummy_jax_args[0].shape

        # Call Dense Expert op _fn
        dense_fn = captured_ops[3]._fn
        res_dense = dense_fn(*dummy_jax_args)
        assert res_dense.shape == dummy_jax_args[0].shape

        # Call V2 op _fn (captured_ops[4..5])
        v2_fn = captured_ops[4]._fn
        dummy_v2_args = (
            jnp.zeros((4, 512), dtype=jnp.bfloat16),
            jnp.zeros((6, 512, 256), dtype=jnp.uint8),
            jnp.zeros((6, 256, 256), dtype=jnp.uint8),
            jnp.zeros((6, 1, 512), dtype=jnp.float32),
            jnp.zeros((6, 1, 512), dtype=jnp.float32),
            jnp.zeros((4, 48), dtype=jnp.float32),
            jnp.zeros((1, 1), dtype=jnp.int32),
        )
        res_v2 = v2_fn(*dummy_v2_args)
        assert res_v2.shape == dummy_v2_args[0].shape

    def test_worker_and_main_branches(self, monkeypatch, capsys):
        """Test worker() with subops, ranks, main(), and __main__."""
        ctx_rank0 = common.WorkerContext(
            rank=0,
            world_size=8,
            tp_size=8,
            cpu_group=None,
            result_dir=pathlib.Path("/tmp"),
        )
        ctx_rank1 = common.WorkerContext(
            rank=1,
            world_size=8,
            tp_size=8,
            cpu_group=None,
            result_dir=pathlib.Path("/tmp"),
        )

        monkeypatch.setattr(bdm, "_run_mhc_sweep", lambda ctx, a: {"mhc": 1})
        monkeypatch.setattr(bdm, "_run_moe_sweep", lambda ctx, a: {"moe": 2})
        monkeypatch.setattr(
            common, "hbm_stats", lambda: {"free_gib": 16.0, "total_gib": 32.0}
        )

        # 1. worker with subop="all" on rank 0 (prints _BENCH_SUMMARY)
        args_all = argparse.Namespace(subop="all", seed=42)
        res_all = bdm.worker(ctx_rank0, args_all)
        assert res_all["mhc_decode"] == {"mhc": 1}
        assert res_all["moe_decode"] == {"moe": 2}
        captured = capsys.readouterr()
        assert "_BENCH_SUMMARY:" in captured.out

        # 2. worker with subop="mhc_only" on rank 1 (no print)
        args_mhc = argparse.Namespace(subop="mhc_only", seed=42)
        res_mhc = bdm.worker(ctx_rank1, args_mhc)
        assert res_mhc["mhc_decode"] == {"mhc": 1}
        assert res_mhc["moe_decode"] == {}
        captured = capsys.readouterr()
        assert "_BENCH_SUMMARY:" not in captured.out

        # 3. worker with subop="moe_only" on rank 0
        args_moe = argparse.Namespace(subop="moe_only", seed=42)
        res_moe = bdm.worker(ctx_rank0, args_moe)
        assert res_moe["mhc_decode"] == {}
        assert res_moe["moe_decode"] == {"moe": 2}

        # 4. test main()
        called_run_main = {}

        def mock_run_main(name, parser, worker_fn):
            called_run_main["name"] = name
            called_run_main["worker_fn"] = worker_fn

        monkeypatch.setattr(common, "run_main", mock_run_main)
        bdm.main()
        assert called_run_main["name"] == "decode_moe_mhc"
        assert called_run_main["worker_fn"] == bdm.worker

        # 5. test __main__ entrypoint via runpy
        main_invoked = []
        monkeypatch.setattr(
            common,
            "run_main",
            lambda name, parser, fn: main_invoked.append(name),
        )
        runpy.run_path(str(_BENCH_FILE), run_name="__main__")
        assert main_invoked == ["decode_moe_mhc"]


# ==============================================================================
# 2. Tests for src/vllm_torchtpu/kernels/experimental/moe_dense_expert.py diffs
# ==============================================================================


class TestMoeDenseExpertDiff:
    """Coverage tests for additions in moe_dense_expert.py."""

    def test_apply_activation_supported_and_error(self):
        """Test _apply_activation across all supported activations and exception."""
        g = jnp.ones((2, 2), dtype=jnp.float32)
        u = jnp.full((2, 2), 2.0, dtype=jnp.float32)

        out_silu = kde._apply_activation(g, u, "silu")
        assert jnp.allclose(out_silu, jax.nn.silu(g) * u)

        out_clamp = kde._apply_activation(g, u, "silu_and_mul_with_clamp")
        assert out_clamp.shape == (2, 2)

        out_swiglu = kde._apply_activation(g, u, "swigluoai")
        assert out_swiglu.shape == (2, 2)

        with pytest.raises(NotImplementedError, match="Unsupported activation"):
            kde._apply_activation(g, u, "invalid_act")

    def test_packing_factor_for_types(self):
        """Test _packing_factor_for across valid and invalid combinations."""
        fp8_t = jnp.zeros((1, 1, 1), dtype=jnp.float8_e4m3fn)
        u8_t = jnp.zeros((1, 1, 1), dtype=jnp.uint8)

        # FP8 weights with None or FP8 rhs_quant_dtype -> 1
        assert kde._packing_factor_for(fp8_t, fp8_t, None) == 1
        assert kde._packing_factor_for(fp8_t, fp8_t, jnp.float8_e4m3fn) == 1
        assert kde._packing_factor_for(u8_t, fp8_t, jnp.float8_e4m3fn) is None

        # UINT8 weights with FP4 rhs_quant_dtype -> 2
        assert kde._packing_factor_for(u8_t, u8_t, jnp.float4_e2m1fn) == 2
        assert kde._packing_factor_for(fp8_t, u8_t, jnp.float4_e2m1fn) is None

        # Unsupported rhs_quant_dtype
        assert kde._packing_factor_for(u8_t, u8_t, jnp.bfloat16) is None

    def test_can_use_dense_expert_predicates(self):
        """Test can_use_dense_expert across all rejection criteria and valid path."""

        def _make_tensors(
            m=16,
            h=512,
            i=512,
            e=4,
            pack=1,
            hidden_dtype=jnp.bfloat16,
            scale_dtype=jnp.float32,
        ):
            w_dtype = jnp.float8_e4m3fn if pack == 1 else jnp.uint8
            return (
                jax.ShapeDtypeStruct((m, h), hidden_dtype),
                jax.ShapeDtypeStruct((e, h // pack, 2 * i), w_dtype),
                jax.ShapeDtypeStruct((e, i // pack, h), w_dtype),
                jax.ShapeDtypeStruct((e, 1, 1, 2 * i), scale_dtype),
                jax.ShapeDtypeStruct((e, 1, 1, h), scale_dtype),
            )

        # Baseline valid
        x, w1, w2, s1, s2 = _make_tensors()
        assert kde.can_use_dense_expert(
            x,
            w1,
            w2,
            s1,
            s2,
            None,
            None,
            activation="silu",
            rhs_quant_dtype=None,
            skip_padded_tokens=False,
        )

        # Rejection: unsupported activation
        assert not kde.can_use_dense_expert(
            x,
            w1,
            w2,
            s1,
            s2,
            None,
            None,
            activation="relu",
            rhs_quant_dtype=None,
            skip_padded_tokens=False,
        )

        # Rejection: non-None bias
        assert not kde.can_use_dense_expert(
            x,
            w1,
            w2,
            s1,
            s2,
            x,
            None,
            activation="silu",
            rhs_quant_dtype=None,
            skip_padded_tokens=False,
        )
        assert not kde.can_use_dense_expert(
            x,
            w1,
            w2,
            s1,
            s2,
            None,
            x,
            activation="silu",
            rhs_quant_dtype=None,
            skip_padded_tokens=False,
        )

        # Rejection: None scale
        assert not kde.can_use_dense_expert(
            x,
            w1,
            w2,
            None,
            s2,
            None,
            None,
            activation="silu",
            rhs_quant_dtype=None,
            skip_padded_tokens=False,
        )
        assert not kde.can_use_dense_expert(
            x,
            w1,
            w2,
            s1,
            None,
            None,
            None,
            activation="silu",
            rhs_quant_dtype=None,
            skip_padded_tokens=False,
        )

        # Rejection: invalid ndim
        assert not kde.can_use_dense_expert(
            jax.ShapeDtypeStruct((16,), jnp.bfloat16),
            w1,
            w2,
            s1,
            s2,
            None,
            None,
            activation="silu",
            rhs_quant_dtype=None,
            skip_padded_tokens=False,
        )
        assert not kde.can_use_dense_expert(
            x,
            jax.ShapeDtypeStruct((4, 512), jnp.float8_e4m3fn),
            w2,
            s1,
            s2,
            None,
            None,
            activation="silu",
            rhs_quant_dtype=None,
            skip_padded_tokens=False,
        )
        assert not kde.can_use_dense_expert(
            x,
            w1,
            jax.ShapeDtypeStruct((4, 512), jnp.float8_e4m3fn),
            s1,
            s2,
            None,
            None,
            activation="silu",
            rhs_quant_dtype=None,
            skip_padded_tokens=False,
        )

        # Rejection: invalid packing factor
        assert not kde.can_use_dense_expert(
            x,
            w1,
            w2,
            s1,
            s2,
            None,
            None,
            activation="silu",
            rhs_quant_dtype=jnp.float4_e2m1fn,
            skip_padded_tokens=False,
        )

        # Rejection: min dimension <= 0
        x_zero, w1_zero, w2_zero, s1_zero, s2_zero = _make_tensors(m=0)
        assert not kde.can_use_dense_expert(
            x_zero,
            w1_zero,
            w2_zero,
            s1_zero,
            s2_zero,
            None,
            None,
            activation="silu",
            rhs_quant_dtype=None,
            skip_padded_tokens=False,
        )

        # Rejection: unaligned m with pack=1
        x_unaligned, _, _, _, _ = _make_tensors(m=15)
        assert not kde.can_use_dense_expert(
            x_unaligned,
            w1,
            w2,
            s1,
            s2,
            None,
            None,
            activation="silu",
            rhs_quant_dtype=None,
            skip_padded_tokens=False,
        )

        # Rejection: unaligned h or i
        x_bad_h, w1_bad_h, w2_bad_h, s1_bad_h, s2_bad_h = _make_tensors(h=600)
        assert not kde.can_use_dense_expert(
            x_bad_h,
            w1_bad_h,
            w2_bad_h,
            s1_bad_h,
            s2_bad_h,
            None,
            None,
            activation="silu",
            rhs_quant_dtype=None,
            skip_padded_tokens=False,
        )

        # Rejection: shape mismatch out != h or w1 shape mismatch
        bad_w1 = jax.ShapeDtypeStruct((4, 512, 512), jnp.float8_e4m3fn)
        assert not kde.can_use_dense_expert(
            x,
            bad_w1,
            w2,
            s1,
            s2,
            None,
            None,
            activation="silu",
            rhs_quant_dtype=None,
            skip_padded_tokens=False,
        )

        # Rejection: non-bf16 hidden
        x_f32, _, _, _, _ = _make_tensors(hidden_dtype=jnp.float32)
        assert not kde.can_use_dense_expert(
            x_f32,
            w1,
            w2,
            s1,
            s2,
            None,
            None,
            activation="silu",
            rhs_quant_dtype=None,
            skip_padded_tokens=False,
        )

        # Rejection: invalid scale shapes
        bad_s1 = jax.ShapeDtypeStruct((4, 2, 2, 1024), jnp.float32)
        assert not kde.can_use_dense_expert(
            x,
            w1,
            w2,
            bad_s1,
            s2,
            None,
            None,
            activation="silu",
            rhs_quant_dtype=None,
            skip_padded_tokens=False,
        )

        # Rejection: non-float32 scale
        _, _, _, s1_bf16, s2_bf16 = _make_tensors(scale_dtype=jnp.bfloat16)
        assert not kde.can_use_dense_expert(
            x,
            w1,
            w2,
            s1_bf16,
            s2_bf16,
            None,
            None,
            activation="silu",
            rhs_quant_dtype=None,
            skip_padded_tokens=False,
        )

    def test_dense_expert_moe_single_buffer_execution(self, monkeypatch):
        """Cover lines 261-262 and branch 266->273 with num_buffers == 1 and padding."""
        m, h, i, e = 4, 1024, 512, 4
        rng = np.random.default_rng(20261006)
        x = jnp.asarray(rng.normal(size=(m, h)), jnp.bfloat16)
        w1_f32 = jnp.asarray(rng.normal(scale=0.1, size=(e, h, 2 * i)), jnp.float32)
        w2_f32 = jnp.asarray(rng.normal(scale=0.1, size=(e, i, h)), jnp.float32)

        w1, s1 = fmg.quantize_to_native_fp4_kmajor(w1_f32, block=512, pack=True)
        w2, s2 = fmg.quantize_to_native_fp4_kmajor(w2_f32, block=512, pack=True)
        route = [0, 1, 2, -1]
        ids = jnp.asarray(np.tile(route, (m, 1)), jnp.int32)
        weights = jnp.asarray(np.tile([0.25, 0.25, 0.5, 0.0], (m, 1)), jnp.bfloat16)

        monkeypatch.setattr(
            kde.pl,
            "pallas_call",
            functools.partial(kde.pl.pallas_call, interpret=True),
        )
        monkeypatch.setattr(
            kde.pltpu,
            "get_tpu_info",
            lambda: SimpleNamespace(vmem_capacity_bytes=64 * 1024**2),
        )
        # Force num_buffers = 1
        monkeypatch.setattr(kde, "_select_expert_buffers", lambda *a, **k: 1)

        kde.dense_expert_moe.clear_cache()
        try:
            out = kde.dense_expert_moe(
                x,
                w1,
                w2,
                s1,
                s2,
                weights,
                ids,
                activation="silu_and_mul_with_clamp",
                rhs_quant_dtype=jnp.float4_e2m1fn,
            )
            assert out.shape == (m, h)
            assert np.isfinite(np.asarray(out)).all()
        finally:
            kde.dense_expert_moe.clear_cache()


# ==============================================================================
# 3. Tests for src/vllm_torchtpu/layers/core/fused_moe_gmm.py diffs
# ==============================================================================


class TestFusedMoeGmmDenseExpertIntegration:
    """Coverage tests for diffs in fused_moe_gmm.py (lines 636-650)."""

    def test_fused_moe_func_calls_dense_expert_with_parameters(self, monkeypatch):
        """Test fused_moe_func routes to dense_expert_moe with arguments."""
        captured = {}

        def mock_dense_expert_moe(
            hidden,
            w1,
            w2,
            w1_scale,
            w2_scale,
            topk_weights,
            topk_ids,
            *,
            activation="silu",
            rhs_quant_dtype=None,
        ):
            captured["activation"] = activation
            captured["rhs_quant_dtype"] = rhs_quant_dtype
            captured["shape"] = hidden.shape
            return hidden

        monkeypatch.setattr(fmg, "_DENSE_EXPERT_THRESHOLD", 64)
        monkeypatch.setattr(fmg, "dense_expert_moe", mock_dense_expert_moe)

        h, i, e = 512, 512, 4
        m = 8
        w1_u8 = jax.ShapeDtypeStruct((e, h // 2, 2 * i), jnp.uint8)
        w2_u8 = jax.ShapeDtypeStruct((e, i // 2, h), jnp.uint8)
        s1_b = jax.ShapeDtypeStruct((e, h // 512, 1, 2 * i), jnp.float32)
        s2_b = jax.ShapeDtypeStruct((e, i // 512, 1, h), jnp.float32)

        fmg.fused_moe_func.clear_cache()
        try:
            jax.eval_shape(
                functools.partial(
                    fmg.fused_moe_func,
                    topk=4,
                    activation="silu_and_mul_with_clamp",
                    rhs_quant_dtype=jnp.float4_e2m1fn,
                ),
                jax.ShapeDtypeStruct((m, h), jnp.bfloat16),
                w1_u8,
                w2_u8,
                s1_b,
                s2_b,
                None,
                None,
                jax.ShapeDtypeStruct((m, 4), jnp.bfloat16),
                jax.ShapeDtypeStruct((m, 4), jnp.int32),
            )
        finally:
            fmg.fused_moe_func.clear_cache()

        assert captured == {
            "activation": "silu_and_mul_with_clamp",
            "rhs_quant_dtype": jnp.float4_e2m1fn,
            "shape": (8, 512),
        }
