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
"""TPU tests for the fused attention-output / o_proj FP8 quantization kernel."""

import functools
import re

import jax
import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa.utils import get_dtype_packing
from vllm_torchtpu.kernels.experimental.batched_rpa.wrapper import finalize_outputs
from vllm_torchtpu.kernels.quantized_matmul.attn_out_fp8 import quantize_attn_out
from vllm_torchtpu.kernels.quantized_matmul.util import (
    quantize_tensor,
    xla_quantized_matmul,
)

pytestmark = pytest.mark.tpu_test

# Per-rank (num_kv_heads, q_per_kv, head_dim): Gemma 4 31B sliding and
# global attention at TP=2 and TP=8, plus `(1, 2, 256)` covering the
# single-group (`len(q_blocks) == 1`) kernel branch.
_LAYOUTS = [(8, 2, 256), (2, 8, 512), (2, 2, 256), (1, 4, 512), (1, 2, 256)]


def _require_tpu() -> None:
    try:
        backend = jax.default_backend()
    except Exception as exc:
        pytest.fail(f"JAX TPU backend failed to initialize: {exc}")
    if backend != "tpu":
        pytest.fail(f"Expected JAX TPU backend, got {backend}.")


def _raw_head_major(key, m, kv, qpk, d, dtype=jnp.bfloat16):
    """Raw 5-D head-major output emitted by batched RPA before finalize_outputs."""
    q_packing = get_dtype_packing(dtype)
    o5 = jax.random.normal(key, (kv, m, qpk // q_packing, q_packing, d), jnp.float32)
    return (o5 * 3.0).astype(dtype)


def _attn_out(key, m, kv, qpk, d, dtype=jnp.bfloat16):
    """Attention output as the model sees it: finalized to [M, K]."""
    o5 = _raw_head_major(key, m, kv, qpk, d, dtype=dtype)
    return finalize_outputs(o5, qpk, d, (m, kv * qpk * d))


def _entry_body(hlo_text: str) -> str:
    m = re.search(r"ENTRY [^{]+\{(.*?)\n\}", hlo_text, re.DOTALL)
    assert m is not None, "Could not locate ENTRY computation in compiled HLO"
    return m.group(1)


def _has_relayout_copy(hlo_text: str) -> bool:
    """True if ENTRY contains a layout-changing op or SparseCore relayout."""
    entry = _entry_body(hlo_text)
    return bool(
        re.search(r"\b(?:copy|transpose|reshape)\(|sparse-core-data-format-call", entry)
    )


@pytest.mark.parametrize("layout", _LAYOUTS)
@pytest.mark.parametrize("m", [32, 256, 2048])
def test_matches_xla_quantized_matmul_bit_exact(layout, m):
    _require_tpu()
    kv, qpk, d = layout
    k, n = kv * qpk * d, 5376
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(m + k), 3)
    x = _attn_out(k1, m, kv, qpk, d)
    w = (jax.random.normal(k2, (k, n), jnp.float32) * 0.05).astype(jnp.float8_e4m3fn)
    ws = jax.random.uniform(k3, (n,), jnp.float32, 0.5, 1.5) * 0.01

    quantizer = functools.partial(
        quantize_attn_out, num_kv_heads=kv, q_per_kv=qpk, head_dim=d
    )
    ref = jax.jit(xla_quantized_matmul)(x, w, ws)
    got = jax.jit(
        lambda x, w, ws: xla_quantized_matmul(
            x, w, ws, quantize_activation_fn=quantizer
        )
    )(x, w, ws)
    assert got.dtype == ref.dtype and got.shape == (m, n)
    assert bool(jnp.all(got == ref))


@pytest.mark.parametrize("layout", [(2, 3, 256), (2, 4, 256)])
def test_float32_activation_matches_xla_quantized_matmul_bit_exact(layout):
    """float32 activations derive q_packing=1 (allowing odd q_per_kv) and match."""
    _require_tpu()
    kv, qpk, d = layout
    m, k, n = 256, kv * qpk * d, 1024
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(m + k + 7), 3)
    x = _attn_out(k1, m, kv, qpk, d, dtype=jnp.float32)
    w = (jax.random.normal(k2, (k, n), jnp.float32) * 0.05).astype(jnp.float8_e4m3fn)
    ws = jax.random.uniform(k3, (n,), jnp.float32, 0.5, 1.5) * 0.01

    quantizer = functools.partial(
        quantize_attn_out, num_kv_heads=kv, q_per_kv=qpk, head_dim=d
    )
    ref = jax.jit(xla_quantized_matmul)(x, w, ws)
    got = jax.jit(
        lambda x, w, ws: xla_quantized_matmul(
            x, w, ws, quantize_activation_fn=quantizer
        )
    )(x, w, ws)
    assert got.dtype == jnp.float32 and got.shape == (m, n)
    assert bool(jnp.all(got == ref))


@pytest.mark.parametrize("layout", _LAYOUTS)
def test_quantized_activation_and_scale(layout):
    _require_tpu()
    kv, qpk, d = layout
    m = 256
    x = _attn_out(jax.random.PRNGKey(0), m, kv, qpk, d)
    # An all-zero row gets scale 1, as in the reference, not NaN.
    x = x.at[3].set(0)
    x_q, scale = jax.jit(
        lambda x: quantize_attn_out(x, num_kv_heads=kv, q_per_kv=qpk, head_dim=d)
    )(x)
    ref_q, ref_s = quantize_tensor(x, jnp.float8_e4m3fn)
    assert scale.shape == (m, 1) and scale.dtype == jnp.float32
    assert bool(jnp.all(scale == ref_s.reshape(m, 1).astype(jnp.float32)))
    assert bool(jnp.all(x_q.astype(jnp.float32) == ref_q.astype(jnp.float32)))


@pytest.mark.parametrize("layout", [(8, 2, 256), (2, 8, 512), (1, 4, 512)])
def test_batched_rpa_pair_cancels_relayout_copy_in_hlo(layout):
    """Compiled HLO of finalize_outputs -> quantize_attn_out has no relayout copy."""
    _require_tpu()
    kv, qpk, d = layout
    m, k, n = 2048, kv * qpk * d, 5376
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(k), 4)
    o5 = _raw_head_major(k1, m, kv, qpk, d)
    gate = jax.random.normal(k4, (m, k), jnp.float32).astype(jnp.bfloat16)
    w = (jax.random.normal(k2, (k, n), jnp.float32) * 0.05).astype(jnp.float8_e4m3fn)
    ws = jax.random.uniform(k3, (n,), jnp.float32, 0.5, 1.5) * 0.01

    quantizer = functools.partial(
        quantize_attn_out, num_kv_heads=kv, q_per_kv=qpk, head_dim=d
    )

    def fused_pair(o5_in, w_in, ws_in):
        x = finalize_outputs(o5_in, qpk, d, (m, k))
        return xla_quantized_matmul(x, w_in, ws_in, quantize_activation_fn=quantizer)

    fused_hlo = jax.jit(fused_pair).lower(o5, w, ws).compile().as_text()
    assert "attn_out_fp8_quant" in _entry_body(fused_hlo)
    assert not _has_relayout_copy(fused_hlo), (
        f"Expected finalize_outputs + unprepare_outputs_to_head_major to cancel "
        f"without a relayout copy for {layout=}, got:\n{_entry_body(fused_hlo)}"
    )

    # Fused must-fail control: a token-major multiply between finalize_outputs
    # and quantize_attn_out prevents cancellation, emitting relayout ops (such
    # as `copy`, `reshape`, or `sparse-core-data-format-call`) before the kernel.
    def gated_fused_pair(o5_in, gate_in, w_in, ws_in):
        x = finalize_outputs(o5_in, qpk, d, (m, k)) * gate_in
        return xla_quantized_matmul(x, w_in, ws_in, quantize_activation_fn=quantizer)

    gated_hlo = jax.jit(gated_fused_pair).lower(o5, gate, w, ws).compile().as_text()
    assert "attn_out_fp8_quant" in _entry_body(gated_hlo)
    assert _has_relayout_copy(gated_hlo), (
        f"Must-fail control expected a relayout on gated fused {layout=}, "
        f"got:\n{_entry_body(gated_hlo)}"
    )

    # Unfused must-fail control: without quantize_attn_out, finalize_outputs
    # leaves a relayout op in ENTRY.
    def unfused_pair(o5_in, w_in, ws_in):
        x = finalize_outputs(o5_in, qpk, d, (m, k))
        return xla_quantized_matmul(x, w_in, ws_in)

    unfused_hlo = jax.jit(unfused_pair).lower(o5, w, ws).compile().as_text()
    assert _has_relayout_copy(unfused_hlo), (
        f"Must-fail control expected a relayout on unfused {layout=}, "
        f"got:\n{_entry_body(unfused_hlo)}"
    )
