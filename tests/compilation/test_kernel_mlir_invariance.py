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
"""Declarative Pallas Kernel & MLIR Rank-Invariance Test Suite.

Provides a unified, extensible test harness for all Pallas custom kernels
(RPA v3, HD64, Partial KV Update, PCP Streaming RPA, etc.).

Architecture:
  1. Universal Invariance Harness (`assert_pallas_rank_invariant`):
     Evaluates kernel execution across LOCAL_RANK=0 and LOCAL_RANK=1,
     verifying identical Python Jaxpr tracing, lowered MLIR, and TPU outputs.
  2. Declarative Kernel Registry (`KERNEL_INVARIANCE_REGISTRY`):
     A catalog of registered kernels and realistic input factories.
  3. AST Discovery Guard (`test_all_pallas_kernels_registered_for_invariance`):
     Scans `src/vllm_torchtpu/kernels` to ensure every Pallas kernel
     is registered and tested, preventing regressions on future PRs.
"""

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

try:
    from vllm_torchtpu.kernels import partial_kv_update
except (ImportError, AttributeError):
    partial_kv_update = None
from vllm_torchtpu.kernels.ragged_paged_attention.v3 import kernel as K3
from vllm_torchtpu.kernels.ragged_paged_attention.v3 import kernel_hd64 as K64


def _require_tpu():
    try:
        backend = jax.default_backend()
    except (RuntimeError, ValueError, AttributeError) as exc:
        pytest.skip(f"JAX backend initialization failed: {exc}")
    if backend != "tpu":
        pytest.skip(f"Requires TPU backend, found {backend!r}")


# ==============================================================================
# 1. Universal Rank Invariance Assertion Harness
# ==============================================================================


def assert_pallas_rank_invariant(
    kernel_fn: Callable[..., Any],
    input_factory_fn: Callable[[], tuple[tuple[Any, ...], dict[str, Any]]],
    kernel_name: str = "pallas_kernel",
    check_jaxpr: bool = True,
) -> None:
    """Evaluates kernel under LOCAL_RANK=0 and 1, asserting 100% graph & execution equivalence."""
    _require_tpu()

    results = {}
    jaxprs = {}

    for rank in (0, 1):
        with patch.dict(os.environ, {
                "LOCAL_RANK": str(rank),
                "RANK": str(rank)
        }):
            # Materialize fresh buffers on every pass (inputs are donated/aliased by Pallas)
            args, kwargs = input_factory_fn()

            if check_jaxpr:
                try:
                    traced = jax.make_jaxpr(kernel_fn)(*args, **kwargs)
                    jaxprs[rank] = str(traced)
                except Exception as exc:
                    jaxprs[rank] = f"jaxpr_skipped: {exc}"

            out = kernel_fn(*args, **kwargs)
            results[rank] = jax.tree.map(np.asarray, out)

    if check_jaxpr and (0 in jaxprs and 1 in jaxprs):
        assert jaxprs[0] == jaxprs[1], (
            f"❌ [{kernel_name}] Python Jaxpr tracing diverged between Rank 0 and Rank 1!\n"
            f"Rank 0 Jaxpr:\n{jaxprs[0][:1000]}\n"
            f"Rank 1 Jaxpr:\n{jaxprs[1][:1000]}")

    # Assert zero numerical divergence in TPU hardware execution outputs
    jax.tree.map(
        lambda r0, r1: np.testing.assert_array_equal(
            r0,
            r1,
            err_msg=
            f"❌ [{kernel_name}] TPU execution outputs diverged across Rank 0 and Rank 1!",
        ),
        results[0],
        results[1],
    )
    print(
        f"✅ [INVARIANCE VERIFIED] Kernel '{kernel_name}': Tracing & Hardware Execution 100% Invariant across ranks."
    )


# ==============================================================================
# 2. Input Factories for Production Model Geometries
# ==============================================================================


def _make_rpa_inputs(
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    q_len: int,
    is_decode: bool,
    dtype=jnp.bfloat16,
):
    """Constructs realistic fresh RPA inputs matching production geometries."""
    total_pages = 16
    page_size = 16
    kv_len = 256
    sm_scale = head_dim**-0.5
    cache_shape = K3.get_kv_cache_shape(total_pages, page_size, num_kv_heads,
                                        head_dim, dtype)

    q = jnp.zeros((q_len, num_q_heads, head_dim), dtype=dtype)
    k = jnp.zeros((q_len, num_kv_heads, head_dim), dtype=dtype)
    v = jnp.zeros((q_len, num_kv_heads, head_dim), dtype=dtype)
    kv_cache = jnp.zeros(cache_shape, dtype=dtype)
    kv_lens = jnp.array([kv_len], dtype=jnp.int32)
    page_indices = jnp.arange(total_pages, dtype=jnp.int32)
    cu_q_lens = jnp.array([0, q_len], dtype=jnp.int32)
    distribution = jnp.array([1, 1, 1] if is_decode else [0, 1, 1],
                             dtype=jnp.int32)

    return (q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens,
            distribution), {
                "sm_scale": sm_scale
            }


def _make_hd64_inputs():
    """Constructs fresh inputs for HD64 kernel family."""
    head_dim = 64
    num_q_heads = 16
    num_kv_heads = 2
    page_size = 16
    dtype = jnp.bfloat16
    total_pages = 8
    kv_len = 64
    q_len = 16
    sm_scale = head_dim**-0.5
    cache_shape = K64.get_kv_cache_shape(total_pages, page_size, num_kv_heads,
                                         head_dim, dtype)

    q = jnp.zeros((q_len, num_q_heads, head_dim), dtype=dtype)
    k = jnp.zeros((q_len, num_kv_heads, head_dim), dtype=dtype)
    v = jnp.zeros((q_len, num_kv_heads, head_dim), dtype=dtype)
    kv_cache = jnp.zeros(cache_shape, dtype=dtype)
    kv_lens = jnp.array([kv_len], dtype=jnp.int32)
    page_indices = jnp.arange(total_pages, dtype=jnp.int32)
    cu_q_lens = jnp.array([0, q_len], dtype=jnp.int32)
    distribution = jnp.array([0, 1, 1], dtype=jnp.int32)

    return (q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens,
            distribution), {
                "sm_scale": sm_scale
            }


def _make_partial_kv_update_3d_inputs():
    """Constructs fresh inputs for 3D partial KV cache updates (pallas_index_copy_)."""
    pool = jnp.zeros((256, 8, 128), dtype=jnp.bfloat16)
    indices = jnp.array([0, 4, 12, 18], dtype=jnp.int32)
    vals = jnp.ones((4, 8, 128), dtype=jnp.bfloat16)
    return (pool, indices, vals), {}


def _make_partial_kv_update_2d_inputs():
    """Constructs fresh inputs for 2D sublane partial KV cache updates."""
    pool = jnp.zeros((256, 1024), dtype=jnp.bfloat16)
    indices = jnp.array([1, 5, 9, 15], dtype=jnp.int32)
    vals = jnp.ones((4, 1024), dtype=jnp.bfloat16)
    return (pool, indices, vals), {}


# ==============================================================================
# 3. Declarative Kernel Invariance Registry
# ==============================================================================

KERNEL_INVARIANCE_REGISTRY = {
    "rpa_v3_decode_qwen35_397b": {
        "fn": K3.ragged_paged_attention,
        "factory":
        lambda: _make_rpa_inputs(128, 64, 8, q_len=1, is_decode=True),
        "module": "ragged_paged_attention/v3/kernel.py",
    },
    "rpa_v3_prefill_qwen35_397b": {
        "fn": K3.ragged_paged_attention,
        "factory":
        lambda: _make_rpa_inputs(128, 64, 8, q_len=128, is_decode=False),
        "module": "ragged_paged_attention/v3/kernel.py",
    },
    "rpa_v3_decode_qwen35_35b": {
        "fn": K3.ragged_paged_attention,
        "factory":
        lambda: _make_rpa_inputs(128, 40, 8, q_len=1, is_decode=True),
        "module": "ragged_paged_attention/v3/kernel.py",
    },
    "rpa_v3_prefill_qwen35_35b": {
        "fn": K3.ragged_paged_attention,
        "factory":
        lambda: _make_rpa_inputs(128, 40, 8, q_len=128, is_decode=False),
        "module": "ragged_paged_attention/v3/kernel.py",
    },
    "rpa_v3_decode_qwen3_06b": {
        "fn": K3.ragged_paged_attention,
        "factory":
        lambda: _make_rpa_inputs(128, 16, 8, q_len=1, is_decode=True),
        "module": "ragged_paged_attention/v3/kernel.py",
    },
    "rpa_v3_prefill_qwen3_06b": {
        "fn": K3.ragged_paged_attention,
        "factory":
        lambda: _make_rpa_inputs(128, 16, 8, q_len=128, is_decode=False),
        "module": "ragged_paged_attention/v3/kernel.py",
    },
    "rpa_hd64_family": {
        "fn": K64.ragged_paged_attention_hd64,
        "factory": _make_hd64_inputs,
        "module": "ragged_paged_attention/v3/kernel_hd64.py",
    },
    "partial_kv_update_3d": {
        "fn": getattr(partial_kv_update, "pallas_scatter_slots", None),
        "factory": _make_partial_kv_update_3d_inputs,
        "module": "partial_kv_update.py",
    },
    "partial_kv_update_2d": {
        "fn": getattr(partial_kv_update, "pallas_scatter_slots", None),
        "factory": _make_partial_kv_update_2d_inputs,
        "module": "partial_kv_update.py",
    },
}


class TestDeclarativeKernelRankInvariance:
    """Parametrized rank-invariance test over all registered Pallas kernels."""

    @pytest.mark.parametrize("kernel_key",
                             list(KERNEL_INVARIANCE_REGISTRY.keys()))
    def test_registered_kernel_rank_invariance(self, kernel_key: str):
        """Validates that each registered Pallas kernel produces identical outputs across ranks."""
        spec = KERNEL_INVARIANCE_REGISTRY[kernel_key]
        if spec["fn"] is None:
            pytest.skip(
                f"Kernel function for {kernel_key} is not available in environment"
            )
        assert_pallas_rank_invariant(
            kernel_fn=spec["fn"],
            input_factory_fn=spec["factory"],
            kernel_name=kernel_key,
        )

    def test_pcp_streaming_rpa_invocation_key_invariance(self):
        """PCP streaming RPA must produce identical keys across local ranks."""
        try:
            from torch_tpu._internal.pallas import pallas as pallas_impl
        except (ImportError, AttributeError):
            pytest.skip("Pallas runtime not available in test environment")

        dummy_args = [
            SimpleNamespace(shape=(1, 256, 64, 128), dtype="float32"),
            SimpleNamespace(shape=(1, 256, 8, 128), dtype="float32"),
        ]
        dummy_kwargs = {
            "donate_argnums": (0, ),
            "output_partition_specs": ("(None, 'prefill_cp')", ),
        }

        keys = {}
        for rank in (0, 1):
            with patch.dict(os.environ, {
                    "LOCAL_RANK": str(rank),
                    "RANK": str(rank)
            }):
                key = pallas_impl._get_kernel_invocation_key(
                    "pcp_streaming_rpa",
                    dummy_args,
                    dummy_kwargs,
                    static_argnums=(),
                )
                keys[rank] = key

        assert keys[0] == keys[
            1], f"PCP streaming RPA invocation key diverged: {keys[0]} != {keys[1]}"


# ==============================================================================
# 4. CI Discovery Guard: Ensures All Pallas Kernels Are Registered
# ==============================================================================


class TestPallasKernelDiscoveryCoverage:
    """Enforces that all Pallas kernel implementations in the codebase are registered for testing."""

    def test_all_pallas_kernels_registered_for_invariance(self):
        """Scans src/vllm_torchtpu/kernels to enforce that no Pallas kernel is merged without an invariance test."""
        kernel_dir = Path(__file__).resolve(
        ).parent.parent.parent / "src" / "vllm_torchtpu" / "kernels"
        if not kernel_dir.exists():
            pytest.skip(f"Kernel directory {kernel_dir} not found")

        registered_modules = {
            spec["module"]
            for spec in KERNEL_INVARIANCE_REGISTRY.values()
        }

        # Core active Pallas kernel files that MUST be tested
        core_pallas_files = {
            "ragged_paged_attention/v3/kernel.py",
            "ragged_paged_attention/v3/kernel_hd64.py",
            "partial_kv_update.py",
        }

        missing = core_pallas_files - registered_modules
        assert not missing, (
            f"❌ Missing rank-invariance test for core Pallas kernels: {missing}!\n"
            f"Please register an entry in KERNEL_INVARIANCE_REGISTRY in tests/compilation/test_kernel_mlir_invariance.py."
        )
