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
"""Torch custom-op bridges for batch-invariant matmul and RMS normalization."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import torch

_OPS: dict[tuple[str, bool, bool], Callable] = {}

_TORCH_TO_JAX = {torch.bfloat16: "bfloat16", torch.float32: "float32"}


def _get_op(out_dtype: str, highest: bool, rhs_nk: bool) -> Callable:
    key = (out_dtype, highest, rhs_nk)
    op = _OPS.get(key)
    if op is not None:
        return op
    from torch_tpu._internal import pallas

    from vllm_torchtpu.kernels.batch_invariant.matmul import det_matmul

    dtype = jnp.dtype(out_dtype)

    # jax_op wants an annotated signature with array operands only.
    def _det_matmul(x: jax.Array, w: jax.Array) -> jax.Array:
        return det_matmul(x, w, out_dtype=dtype, highest=highest, rhs_nk=rhs_nk)

    op = pallas.jax_op(
        f"batch_inv::det_matmul_{out_dtype}_h{int(highest)}_nk{int(rhs_nk)}",
        _det_matmul,
    )
    out_torch = {v: k for k, v in _TORCH_TO_JAX.items()}[out_dtype]

    def _fake(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        n = w.shape[0] if rhs_nk else w.shape[1]
        return torch.empty((x.shape[0], n), dtype=out_torch, device=x.device)

    op.register_fake(_fake)
    _OPS[key] = op
    return op


_RMS_OPS: dict[float, Callable] = {}
_RMS_SCALED_OPS: dict[tuple[float, float, str], Callable] = {}
_O_NORM_GATE_OPS: dict[float, Callable] = {}
_CENTERED_RMS_OPS: dict[tuple[float, str], Callable] = {}
_INPUT_SCALED_RMS_OPS: dict[tuple[float, str], Callable] = {}
_RMS_RES_GATE_OPS: dict[tuple[float, str], Callable] = {}
_INPUT_SCALED_RMS_RES_GATE_OPS: dict[tuple[float, str], Callable] = {}


def _fmt_float(val: float) -> str:
    return f"{val:.6e}".replace(".", "p").replace("-", "m").replace("+", "")


def _get_rms_op(eps: float) -> Callable:
    op = _RMS_OPS.get(eps)
    if op is not None:
        return op
    from torch_tpu._internal import pallas

    from vllm_torchtpu.kernels.batch_invariant.rms_norm import det_rms_normalize

    def _det_rms(x: jax.Array) -> jax.Array:
        return det_rms_normalize(x, eps=eps)

    name = f"batch_inv::det_rms_{_fmt_float(eps)}"
    op = pallas.jax_op(name, _det_rms)

    def _fake(x: torch.Tensor) -> torch.Tensor:
        return torch.empty(x.shape, dtype=torch.float32, device=x.device)

    op.register_fake(_fake)
    _RMS_OPS[eps] = op
    return op


def _get_rms_scaled_op(eps: float, scale: float, out_dtype: str) -> Callable:
    key = (eps, scale, out_dtype)
    op = _RMS_SCALED_OPS.get(key)
    if op is not None:
        return op
    from torch_tpu._internal import pallas

    from vllm_torchtpu.kernels.batch_invariant.rms_norm import det_rms_norm_scaled

    dtype = jnp.dtype(out_dtype)

    def _det_rms_s(x: jax.Array) -> jax.Array:
        return det_rms_norm_scaled(x, eps=eps, scale=scale, out_dtype=dtype)

    name = f"batch_inv::det_rms_s_{_fmt_float(eps)}_{_fmt_float(scale)}_{out_dtype}"
    op = pallas.jax_op(name, _det_rms_s)
    out_torch = {v: k for k, v in _TORCH_TO_JAX.items()}[out_dtype]

    def _fake(x: torch.Tensor) -> torch.Tensor:
        return torch.empty(x.shape, dtype=out_torch, device=x.device)

    op.register_fake(_fake)
    _RMS_SCALED_OPS[key] = op
    return op


def _get_o_norm_gate_op(eps: float) -> Callable:
    op = _O_NORM_GATE_OPS.get(eps)
    if op is not None:
        return op
    from torch_tpu._internal import pallas

    from vllm_torchtpu.kernels.batch_invariant.rms_norm import det_o_norm_sigmoid_gate

    def _det_og(x: jax.Array, gate: jax.Array) -> jax.Array:
        return det_o_norm_sigmoid_gate(x, gate, eps=eps)

    name = f"batch_inv::det_onorm_gate_{_fmt_float(eps)}"
    op = pallas.jax_op(name, _det_og)

    def _fake(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return torch.empty(x.shape, dtype=x.dtype, device=x.device)

    op.register_fake(_fake)
    _O_NORM_GATE_OPS[eps] = op
    return op


def _get_centered_rms_op(eps: float, out_dtype: str) -> Callable:
    key = (eps, out_dtype)
    op = _CENTERED_RMS_OPS.get(key)
    if op is not None:
        return op
    from torch_tpu._internal import pallas

    from vllm_torchtpu.kernels.batch_invariant.rms_norm import det_centered_rms_norm

    dtype = jnp.dtype(out_dtype)

    def _det_c(x: jax.Array, weight: jax.Array) -> jax.Array:
        return det_centered_rms_norm(x, weight, eps=eps, out_dtype=dtype)

    name = f"batch_inv::det_c_rms_{_fmt_float(eps)}_{out_dtype}"
    op = pallas.jax_op(name, _det_c)
    out_torch = {v: k for k, v in _TORCH_TO_JAX.items()}[out_dtype]

    def _fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.empty(x.shape, dtype=out_torch, device=x.device)

    op.register_fake(_fake)
    _CENTERED_RMS_OPS[key] = op
    return op


def _get_input_scaled_rms_op(eps: float, out_dtype: str) -> Callable:
    key = (eps, out_dtype)
    op = _INPUT_SCALED_RMS_OPS.get(key)
    if op is not None:
        return op
    from torch_tpu._internal import pallas

    from vllm_torchtpu.kernels.batch_invariant.rms_norm import det_input_scaled_rms_norm

    dtype = jnp.dtype(out_dtype)

    def _det_is(x: jax.Array, weight: jax.Array) -> jax.Array:
        return det_input_scaled_rms_norm(x, weight, eps=eps, out_dtype=dtype)

    name = f"batch_inv::det_is_rms_{_fmt_float(eps)}_{out_dtype}"
    op = pallas.jax_op(name, _det_is)
    out_torch = {v: k for k, v in _TORCH_TO_JAX.items()}[out_dtype]

    def _fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.empty(x.shape, dtype=out_torch, device=x.device)

    op.register_fake(_fake)
    _INPUT_SCALED_RMS_OPS[key] = op
    return op


def _get_rms_res_gate_op(eps: float, out_dtype: str) -> Callable:
    key = (eps, out_dtype)
    op = _RMS_RES_GATE_OPS.get(key)
    if op is not None:
        return op
    from torch_tpu._internal import pallas

    from vllm_torchtpu.kernels.batch_invariant.rms_norm import (
        det_rms_norm_residual_gate,
    )

    dtype = jnp.dtype(out_dtype)

    def _det_rg(
        residual: jax.Array, y: jax.Array, alpha: jax.Array, beta: jax.Array
    ) -> jax.Array:
        return det_rms_norm_residual_gate(
            residual, y, alpha, beta, eps=eps, out_dtype=dtype
        )

    name = f"batch_inv::det_rms_rg_{_fmt_float(eps)}_{out_dtype}"
    op = pallas.jax_op(name, _det_rg)
    out_torch = {v: k for k, v in _TORCH_TO_JAX.items()}[out_dtype]

    def _fake(
        residual: torch.Tensor,
        y: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        return torch.empty(y.shape, dtype=out_torch, device=y.device)

    op.register_fake(_fake)
    _RMS_RES_GATE_OPS[key] = op
    return op


def _get_input_scaled_rms_res_gate_op(eps: float, out_dtype: str) -> Callable:
    key = (eps, out_dtype)
    op = _INPUT_SCALED_RMS_RES_GATE_OPS.get(key)
    if op is not None:
        return op
    from torch_tpu._internal import pallas

    from vllm_torchtpu.kernels.batch_invariant.rms_norm import (
        det_input_scaled_rms_norm_residual_gate,
    )

    dtype = jnp.dtype(out_dtype)

    def _det_is_rg(
        residual: jax.Array,
        y: jax.Array,
        weight: jax.Array,
        alpha: jax.Array,
        beta: jax.Array,
    ) -> jax.Array:
        return det_input_scaled_rms_norm_residual_gate(
            residual, y, weight, alpha, beta, eps=eps, out_dtype=dtype
        )

    name = f"batch_inv::det_is_rms_rg_{_fmt_float(eps)}_{out_dtype}"
    op = pallas.jax_op(name, _det_is_rg)
    out_torch = {v: k for k, v in _TORCH_TO_JAX.items()}[out_dtype]

    def _fake(
        residual: torch.Tensor,
        y: torch.Tensor,
        weight: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        return torch.empty(y.shape, dtype=out_torch, device=y.device)

    op.register_fake(_fake)
    _INPUT_SCALED_RMS_RES_GATE_OPS[key] = op
    return op


_QKV_ROPE_OPS: dict[tuple[float, float, int, int, int, bool, bool], Callable] = {}


def _get_qkv_rope_op(
    eps: float,
    qk_scale: float,
    q_size: int,
    kv_size: int,
    head_dim: int,
    rhs_nk: bool,
    use_rope: bool,
) -> Callable:
    key = (
        float(eps),
        float(qk_scale),
        int(q_size),
        int(kv_size),
        int(head_dim),
        bool(rhs_nk),
        bool(use_rope),
    )
    op = _QKV_ROPE_OPS.get(key)
    if op is not None:
        return op
    from torch_tpu._internal import pallas

    from vllm_torchtpu.kernels.batch_invariant.matmul import det_matmul
    from vllm_torchtpu.kernels.batch_invariant.rms_norm import (
        det_rms_norm_scaled as jax_rms_scaled,
    )

    n_qh = q_size // head_dim
    n_kvh = kv_size // head_dim
    hd_half = head_dim // 2
    name = (
        f"batch_inv::det_qkv_r{int(use_rope)}_{_fmt_float(eps)}_{_fmt_float(qk_scale)}"
        f"_q{q_size}_kv{kv_size}_hd{head_dim}_nk{int(rhs_nk)}"
    )

    if use_rope:

        def _qkv_rope(
            h: jax.Array,
            w: jax.Array,
            cs_cache: jax.Array,
            pos: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            m = h.shape[0]
            qkv = det_matmul(h, w, out_dtype=jnp.bfloat16, rhs_nk=rhs_nk)
            q_raw = qkv[:, :q_size].reshape(m * n_qh, head_dim)
            k_raw = qkv[:, q_size : q_size + kv_size].reshape(m * n_kvh, head_dim)
            v = qkv[:, q_size + kv_size : q_size + 2 * kv_size]
            q = jax_rms_scaled(
                q_raw, eps=eps, scale=qk_scale, out_dtype=jnp.bfloat16
            ).reshape(m, n_qh, head_dim)
            k = jax_rms_scaled(
                k_raw, eps=eps, scale=1.0, out_dtype=jnp.bfloat16
            ).reshape(m, n_kvh, head_dim)
            cs = cs_cache[pos.reshape(-1)]
            cos = cs[:, None, :hd_half].astype(jnp.float32)
            sin = cs[:, None, hd_half : 2 * hd_half].astype(jnp.float32)
            qf = q.astype(jnp.float32)
            q1, q2 = qf[:, :, :hd_half], qf[:, :, hd_half:]
            q_out = (
                jnp.concatenate([q1 * cos - q2 * sin, q2 * cos + q1 * sin], axis=-1)
                .astype(jnp.bfloat16)
                .reshape(m, q_size)
            )
            kf = k.astype(jnp.float32)
            k1, k2 = kf[:, :, :hd_half], kf[:, :, hd_half:]
            k_out = (
                jnp.concatenate([k1 * cos - k2 * sin, k2 * cos + k1 * sin], axis=-1)
                .astype(jnp.bfloat16)
                .reshape(m, kv_size)
            )
            return q_out, k_out, v

        op = pallas.jax_op(name, _qkv_rope)

        def _fake_rope(
            h: torch.Tensor,
            w: torch.Tensor,
            cs_cache: torch.Tensor,
            pos: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            m = h.shape[0]
            return (
                h.new_empty((m, q_size), dtype=torch.bfloat16),
                h.new_empty((m, kv_size), dtype=torch.bfloat16),
                h.new_empty((m, kv_size), dtype=torch.bfloat16),
            )

        op.register_fake(_fake_rope)
    else:

        def _qkv_nope(
            h: jax.Array,
            w: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            m = h.shape[0]
            qkv = det_matmul(h, w, out_dtype=jnp.bfloat16, rhs_nk=rhs_nk)
            q_raw = qkv[:, :q_size].reshape(m * n_qh, head_dim)
            k_raw = qkv[:, q_size : q_size + kv_size].reshape(m * n_kvh, head_dim)
            v = qkv[:, q_size + kv_size : q_size + 2 * kv_size]
            q_out = jax_rms_scaled(
                q_raw, eps=eps, scale=qk_scale, out_dtype=jnp.bfloat16
            ).reshape(m, q_size)
            k_out = jax_rms_scaled(
                k_raw, eps=eps, scale=1.0, out_dtype=jnp.bfloat16
            ).reshape(m, kv_size)
            return q_out, k_out, v

        op = pallas.jax_op(name, _qkv_nope)

        def _fake_nope(
            h: torch.Tensor,
            w: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            m = h.shape[0]
            return (
                h.new_empty((m, q_size), dtype=torch.bfloat16),
                h.new_empty((m, kv_size), dtype=torch.bfloat16),
                h.new_empty((m, kv_size), dtype=torch.bfloat16),
            )

        op.register_fake(_fake_nope)

    _QKV_ROPE_OPS[key] = op
    return op


def det_qkv_qk_norm_rope(
    h: torch.Tensor,
    w: torch.Tensor,
    cos_sin_cache: torch.Tensor | None,
    positions: torch.Tensor | None,
    eps: float,
    qk_scale: float,
    q_size: int,
    kv_size: int,
    head_dim: int,
    *,
    rhs_nk: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused QKV matmul, Q/K RMSNorm, and optional NeoX RoPE."""
    h2 = h.reshape(-1, h.shape[-1]).contiguous()
    use_rope = cos_sin_cache is not None and positions is not None
    op = _get_qkv_rope_op(eps, qk_scale, q_size, kv_size, head_dim, rhs_nk, use_rope)
    if use_rope:
        assert cos_sin_cache is not None and positions is not None
        pos_i32 = (
            positions if positions.dtype == torch.int32 else positions.to(torch.int32)
        )
        return op(h2, w, cos_sin_cache, pos_i32.reshape(-1).contiguous())
    return op(h2, w)


def prepare_det_ops(
    rms_eps: tuple[float, ...] = (),
    qk_scales: tuple[float, ...] = (1.0,),
    qkv_shapes: tuple[tuple[int, int, int], ...] = (),
) -> None:
    """Register batch-invariant jax_ops ahead of torch.compile tracing."""
    for rhs_nk in (False, True):
        _get_op("bfloat16", False, rhs_nk)
    scales = tuple(dict.fromkeys((1.0, *qk_scales)))
    for eps in rms_eps:
        e = float(eps)
        _get_rms_op(e)
        _get_o_norm_gate_op(e)
        for dt in ("bfloat16", "float32"):
            _get_centered_rms_op(e, dt)
            _get_input_scaled_rms_op(e, dt)
            _get_rms_res_gate_op(e, dt)
            _get_input_scaled_rms_res_gate_op(e, dt)
            for s in scales:
                _get_rms_scaled_op(e, float(s), dt)
        for s in qk_scales:
            for q_sz, kv_sz, hd in qkv_shapes:
                for rhs_nk in (False, True):
                    for use_rope in (False, True):
                        _get_qkv_rope_op(
                            e,
                            float(s),
                            int(q_sz),
                            int(kv_sz),
                            int(hd),
                            rhs_nk,
                            use_rope,
                        )


def det_rms_normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
    """fp32 `x * rsqrt(mean(x^2, -1) + eps)` whose rows do not depend on the batch."""
    shape = x.shape
    out = _get_rms_op(float(eps))(x.reshape(-1, shape[-1]).contiguous())
    return out.reshape(shape)


def det_rms_norm_scaled(
    x: torch.Tensor,
    eps: float,
    *,
    scale: float = 1.0,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Fused RMS norm with optional scalar multiplier and output cast."""
    target = out_dtype or x.dtype
    shape = x.shape
    op = _get_rms_scaled_op(float(eps), float(scale), _TORCH_TO_JAX[target])
    return op(x.reshape(-1, shape[-1]).contiguous()).reshape(shape)


def det_o_norm_sigmoid_gate(
    x: torch.Tensor, gate: torch.Tensor, eps: float, head_dim: int
) -> torch.Tensor:
    """Fused per head `o_norm(x) * sigmoid(gate)`."""
    shape = x.shape
    op = _get_o_norm_gate_op(float(eps))
    out = op(
        x.reshape(-1, head_dim).contiguous(),
        gate.reshape(-1, head_dim).contiguous(),
    )
    return out.reshape(shape)


def det_centered_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Fused centered RMSNorm `x * rsqrt(mean(x^2) + eps) * (1 + weight)`."""
    target = out_dtype or x.dtype
    shape = x.shape
    op = _get_centered_rms_op(float(eps), _TORCH_TO_JAX[target])
    return op(x.reshape(-1, shape[-1]).contiguous(), weight).reshape(shape)


def det_input_scaled_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Fused InputScaledRMSNorm."""
    target = out_dtype or x.dtype
    shape = x.shape
    op = _get_input_scaled_rms_op(float(eps), _TORCH_TO_JAX[target])
    return op(x.reshape(-1, shape[-1]).contiguous(), weight).reshape(shape)


def det_rms_norm_residual_gate(
    residual: torch.Tensor,
    y: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    eps: float,
    *,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Fused `alpha * residual + beta * rms_norm(y)`."""
    shape = y.shape
    op = _get_rms_res_gate_op(float(eps), _TORCH_TO_JAX[out_dtype])
    return op(
        residual.reshape(-1, shape[-1]).contiguous(),
        y.reshape(-1, shape[-1]).contiguous(),
        alpha,
        beta,
    ).reshape(shape)


def det_input_scaled_rms_norm_residual_gate(
    residual: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    eps: float,
    *,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Fused `alpha * residual + beta * input_scaled_rms_norm(y, weight)`."""
    shape = y.shape
    op = _get_input_scaled_rms_res_gate_op(float(eps), _TORCH_TO_JAX[out_dtype])
    return op(
        residual.reshape(-1, shape[-1]).contiguous(),
        y.reshape(-1, shape[-1]).contiguous(),
        weight,
        alpha,
        beta,
    ).reshape(shape)


def det_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    out_dtype: torch.dtype | None = None,
    highest: bool = False,
    rhs_nk: bool = False,
) -> torch.Tensor:
    """`x @ w` (w [K, N], or [N, K] if rhs_nk) whose rows do not depend on the batch."""
    out_dtype = out_dtype or x.dtype
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    out = _get_op(_TORCH_TO_JAX[out_dtype], highest, rhs_nk)(x2, w)
    n = w.shape[0] if rhs_nk else w.shape[1]
    return out.reshape(*lead, n)
