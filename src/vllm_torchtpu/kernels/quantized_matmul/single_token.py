# SPDX-License-Identifier: Apache-2.0
"""Per-channel FP8 projection using a single-token W8A16 MXU.

The FP8 quantization and BF16 output boundaries match xla_quantized_matmul.
MXU accumulation has a different reduction order and is not bitwise equal
to the XLA GEMV. The linear adapter selects only the validated shape and dtypes.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.quantized_matmul.tuned_block_sizes import (
    SINGLE_TOKEN_MXU_TILES,
)
from vllm_torchtpu.kernels.quantized_matmul.util import quantize_tensor


def _kernel(x, weight, scale, out, *, mxu_rows):
    quantized, input_scale = quantize_tensor(x[...], weight.dtype)
    # Expand already quantized values exactly; never requantize from BF16.
    lhs = jnp.broadcast_to(quantized.astype(jnp.bfloat16), (mxu_rows, x.shape[-1]))
    accumulated = jax.lax.dot_general(
        lhs,
        weight[...],
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )[:1]
    out[...] = ((accumulated * input_scale) * scale[...][None, :]).astype(out.dtype)


def single_token_mxu(x: jax.Array, weight: jax.Array, scale: jax.Array) -> jax.Array:
    """Project BF16 [1,K] using FP8 [K,N] and FP32 per-output scales.

    Only measured TPU7 K=1024,N=32768 is admitted. Weights retain the frozen
    K-major layout and FP8 storage; no model-load conversion is introduced.
    """
    info = pltpu.get_tpu_info()
    assert x.ndim == weight.ndim == 2
    k, n = weight.shape
    assert x.shape == (1, k) and scale.shape == (n,)
    assert x.dtype == jnp.bfloat16
    assert weight.dtype == jnp.float8_e4m3fn and scale.dtype == jnp.float32
    key = (info.generation, k, n)
    assert key in SINGLE_TOKEN_MXU_TILES, f"Unmeasured single-token shape: {key}"
    assert info.is_matmul_supported(jnp.bfloat16, weight.dtype)
    tile = SINGLE_TOKEN_MXU_TILES[key]
    # Four weight tiles of compiler headroom beyond the double-buffer minimum,
    # bounded by the actual device capacity. This reproduces the measured 48 MiB
    # budget for an 8 MiB tile without declaring a hardware capacity constant.
    scratch_budget = min(info.vmem_capacity_bytes, 6 * k * tile * weight.dtype.itemsize)
    return pl.pallas_call(
        functools.partial(_kernel, mxu_rows=info.num_sublanes),
        out_shape=jax.ShapeDtypeStruct((1, n), x.dtype),
        grid=(n // tile,),
        in_specs=[
            pl.BlockSpec((1, k), lambda i: (0, 0)),
            pl.BlockSpec((k, tile), lambda i: (0, i)),
            pl.BlockSpec((tile,), lambda i: (i,)),
        ],
        out_specs=pl.BlockSpec((1, tile), lambda i: (0, i)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",), vmem_limit_bytes=scratch_budget
        ),
        name="single_token_quantized_mxu",
    )(x, weight, scale)
