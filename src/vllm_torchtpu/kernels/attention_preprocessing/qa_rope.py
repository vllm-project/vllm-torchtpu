# SPDX-License-Identifier: Apache-2.0
"""Q-A/KV projection, both RMSNorms and SWA RoPE for integration into the model.

Replaces ``_fused_wqa_wkv_gemm`` -> ``_split_qkv_and_norm`` -> the SWA
``_rotate`` of ``models/vllm/deepseek_v41/v2/attention.py``:

* rows are FP8-quantized like ``quantized_matmul.util.quantize_tensor``,
  projected, scaled and rounded at the linear layer's BF16 output boundary;
* RMSNorm follows vLLM's native ``ir.ops.rms_norm``: FP32 normalization, a
  cast to the weight dtype, then the weight product (one more rounding for
  BF16 weights);
* the KV tail lanes get interleaved RoPE from the per-step ``[T, rotary_dim]``
  rows of the model's packed ``cos_sin_cache`` (``rope_gather.gather_rows``),
  read as an ordinary tiled input.

The FP8 weight is copied into VMEM once, in ``weight_chunk``-column chunks with
one semaphore each (``emit_qr=False`` copies only the KV columns, for DSpark
context KV); the first token tile waits for and multiplies each chunk as it
lands. Token tiles and outputs are double-buffered.
No cache is written; the caller publishes the rotated SWA KV.
"""

import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.attention_preprocessing import rope_gather
from vllm_torchtpu.kernels.attention_preprocessing.serial import (
    _LANE,
    _lane_gather,
    _quantize_rows,
)
from vllm_torchtpu.kernels.attention_preprocessing.tuned_block_sizes import (
    QA_ROPE_TUNING,
    QARopeConfig,
)

_ROW_ALIGN = 16  # BF16 activation tiles are (16, 128)


def native_rms_norm(values, gamma, eps):
    """``ir.ops.rms_norm`` on FP32 ``values``: round to gamma's dtype, then scale.

    For BF16 gamma the FP32 product of two BF16 values is exact, so the final
    cast reproduces torch's single-rounded BF16 multiply.
    """
    inv_rms = jax.lax.rsqrt(jnp.mean(values * values, axis=-1, keepdims=True) + eps)
    normalized = (values * inv_rms).astype(gamma.dtype).astype(jnp.float32)
    return (normalized * gamma.astype(jnp.float32)).astype(jnp.bfloat16)


def _qa_rope_kernel(
    x_ref,
    weight_hbm,
    scale_ref,
    *refs,
    rank,
    column0,
    chunk,
    eps,
    emit_qr,
    rotate_kv,
):
    refs = list(refs)
    gamma_qr_ref: Any = refs.pop(0) if emit_qr else None
    gamma_kv_ref = refs.pop(0)
    rope_rows_ref: Any = refs.pop(0) if rotate_kv else None
    qr_out_ref: Any = refs.pop(0) if emit_qr else None
    kv_out_ref, weight_ref, weight_sems, projected_ref = refs
    tile = pl.program_id(0)
    width = weight_ref.shape[1]
    chunks = width // chunk

    def weight_copy(c):
        return pltpu.make_async_copy(
            weight_hbm.at[:, pl.ds(column0 + c * chunk, chunk)],
            weight_ref.at[:, pl.ds(c * chunk, chunk)],
            weight_sems.at[c],
        )

    # The weight streams in column chunks on its own DMA priority, so the
    # pipelined x / rope_rows / output copies never queue behind it.
    @pl.when(tile == 0)
    def _():
        for c in range(chunks):
            weight_copy(c).start(priority=1)

    encoded, row_scale = _quantize_rows(x_ref[...])

    def project(columns, product):
        projected = product * row_scale * scale_ref[columns][None, :]
        # Preserve the linear layer's BF16 output boundary before normalization.
        projected_ref[:, columns] = projected.astype(jnp.bfloat16).astype(jnp.float32)

    # Only the first tile waits, chunk by chunk, so its matmul starts after
    # 1.25 MiB instead of the whole weight. Later tiles run one full-width dot
    # (control flow between chunks would drain the MXU per chunk).
    @pl.when(tile == 0)
    def _():
        for c in range(chunks):
            columns = pl.ds(c * chunk, chunk)
            weight_copy(c).wait()
            project(
                columns,
                jnp.dot(
                    encoded, weight_ref[:, columns], preferred_element_type=jnp.float32
                ),
            )

    @pl.when(tile != 0)
    def _():
        project(
            pl.ds(0, width),
            jnp.dot(encoded, weight_ref[...], preferred_element_type=jnp.float32),
        )

    kv_start = rank if emit_qr else 0
    if emit_qr:
        qr_out_ref[...] = native_rms_norm(
            projected_ref[:, :rank], gamma_qr_ref[...][None, :], eps
        )
    kv = native_rms_norm(projected_ref[:, kv_start:], gamma_kv_ref[...][None, :], eps)
    if not rotate_kv:
        kv_out_ref[...] = kv
        return
    if kv_out_ref.shape[-1] > _LANE:
        kv_out_ref[:, :-_LANE] = kv[:, :-_LANE]
    cos, sin = rope_gather.coefficients(rope_rows_ref[...])
    tail = kv[:, -_LANE:].astype(jnp.float32)
    partner = _lane_gather(tail, jnp.arange(_LANE) ^ 1)
    kv_out_ref[:, -_LANE:] = (tail * cos + partner * sin).astype(jnp.bfloat16)


def _largest_divisor(n, cap, multiple):
    for d in range(min(n, cap) // multiple * multiple, 0, -multiple):
        if n % d == 0:
            return d
    return None


def default_config(tokens: int) -> QARopeConfig:
    """Shape-derived tile; ``tokens`` must already be a multiple of 16."""
    tb = tokens if tokens <= 256 else _largest_divisor(tokens, 256, _ROW_ALIGN)
    assert tb is not None  # 16 always divides
    return QARopeConfig(token_block=tb)


def select_config(tokens: int, hidden: int, width: int) -> QARopeConfig:
    from vllm_torchtpu.kernels.quantized_matmul.tuned_block_sizes import (
        get_tpu_version,
    )

    tuned = QA_ROPE_TUNING.get((get_tpu_version(), tokens, hidden, width))
    return tuned or default_config(tokens)


def check_config(config: QARopeConfig, tokens: int, width: int) -> None:
    tb, chunk = config.token_block, config.weight_chunk
    problems = []
    if tb <= 0 or tb % _ROW_ALIGN or tokens % tb:
        problems.append(f"token_block {tb} must be a multiple of 16 dividing {tokens}")
    if chunk <= 0 or chunk % _LANE or width % chunk:
        problems.append(
            f"weight_chunk {chunk} must be a lane multiple dividing {width}"
        )
    if problems:
        raise ValueError("Invalid QARopeConfig: " + "; ".join(problems))


def vmem_bytes(config: QARopeConfig, hidden: int, width: int, rotary_dim: int):
    """Upper estimate of the call's VMEM (lane padding included)."""
    tb = config.token_block
    weight = hidden * width  # FP8, single copy
    x_tiles = 2 * tb * hidden * 2
    projected = tb * width * 4
    outputs = 2 * tb * width * 2
    rope_rows = 2 * tb * _LANE * 4  # double-buffered input tile, lane-padded
    return weight + x_tiles + projected + outputs + rope_rows + 4 * width * 4


def fused_qa_kv_norm_rope(
    x: jax.Array,
    weight: jax.Array,
    weight_scale: jax.Array,
    gamma_qr: jax.Array,
    gamma_kv: jax.Array,
    rope_rows: jax.Array | None,
    *,
    eps: float = 1e-6,
    emit_qr: bool = True,
    rotate_kv: bool = True,
    config: QARopeConfig | None = None,
) -> tuple[jax.Array | None, jax.Array]:
    """Return ``(qr, kv)``: BF16 ``[T, rank]`` (None if not emitted), ``[T, 512]``.

    ``weight`` is FP8 ``[hidden, rank + head_dim]`` in the canonical (k, n)
    layout, qr columns first, with one FP32 scale per column. ``rope_rows`` is
    the per-step ``[T, rotary_dim]`` FP32 ``rope_gather.gather_rows`` of the
    model's ``cos_sin_cache`` (unused, may be None, when ``rotate_kv=False``).
    Token counts that are not a multiple of 16 are zero-padded internally.
    """
    tokens, hidden = x.shape
    rank, head_dim = gamma_qr.shape[0], gamma_kv.shape[0]
    _check_inputs(x, weight, weight_scale, gamma_qr, gamma_kv, eps)
    if rotate_kv:
        rope_gather.check_rope_rows(rope_rows, tokens, head_dim)
    padded = -(-tokens // _ROW_ALIGN) * _ROW_ALIGN
    if padded != tokens:
        x = jnp.pad(x, ((0, padded - tokens), (0, 0)))
        if rotate_kv:
            rope_rows = jnp.pad(rope_rows, ((0, padded - tokens), (0, 0)))
    column0 = 0 if emit_qr else rank
    width = weight.shape[1] - column0
    if config is None:
        # Keyed on the projected width: kv-only mode tunes separately.
        config = select_config(padded, hidden, width)
    tb = config.token_block
    check_config(config, padded, width)
    scale = weight_scale.reshape(-1)[column0:]

    once = pl.Buffered(buffer_count=1)
    rows = lambda t: (t, 0)  # noqa: E731
    fixed = lambda t: (0,)  # noqa: E731
    in_specs = [
        pl.BlockSpec((tb, hidden), rows),
        pl.BlockSpec(memory_space=pltpu.HBM),
        pl.BlockSpec((width,), fixed, once),
    ]
    operands = [x, weight, scale]
    out_specs, out_shape = [], []
    if emit_qr:
        in_specs.append(pl.BlockSpec((rank,), fixed, once))
        operands.append(gamma_qr)
        out_specs.append(pl.BlockSpec((tb, rank), rows))
        out_shape.append(jax.ShapeDtypeStruct((padded, rank), jnp.bfloat16))
    in_specs.append(pl.BlockSpec((head_dim,), fixed, once))
    operands.append(gamma_kv)
    if rotate_kv:
        in_specs.append(pl.BlockSpec((tb, rope_rows.shape[1]), rows))
        operands.append(rope_rows)
    out_specs.append(pl.BlockSpec((tb, head_dim), rows))
    out_shape.append(jax.ShapeDtypeStruct((padded, head_dim), jnp.bfloat16))

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        grid=(padded // tb,),
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=(
            pltpu.VMEM((hidden, width), weight.dtype),
            pltpu.SemaphoreType.DMA((width // config.weight_chunk,)),
            pltpu.VMEM((tb, width), jnp.float32),
        ),
    )
    outputs = pl.pallas_call(
        functools.partial(
            _qa_rope_kernel,
            rank=rank,
            column0=column0,
            chunk=config.weight_chunk,
            eps=eps,
            emit_qr=emit_qr,
            rotate_kv=rotate_kv,
        ),
        grid_spec=grid_spec,
        out_shape=tuple(out_shape),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=config.vmem_limit_bytes,
        ),
        name=f"qa_rope_t{tb}"
        + ("" if emit_qr else "_kv")
        + ("" if rotate_kv else "_norope"),
    )(*operands)
    qr, kv = outputs if emit_qr else (None, outputs[0])
    if padded != tokens:
        qr = None if qr is None else qr[:tokens]
        kv = kv[:tokens]
    return qr, kv


def _check_inputs(x, weight, scale, gamma_qr, gamma_kv, eps):
    if x.ndim != 2 or weight.ndim != 2 or x.shape[0] == 0:
        raise ValueError("Use non-empty rank-2 activations and weights")
    if x.dtype != jnp.bfloat16 or weight.dtype != jnp.float8_e4m3fn:
        raise ValueError("Use BF16 activations and FP8 E4M3FN weights")
    if x.shape[1] != weight.shape[0]:
        raise ValueError("Weight input dimension must match the activation width")
    if scale.shape not in ((weight.shape[1],), (1, weight.shape[1])):
        raise ValueError("Use one weight scale per output column")
    if scale.dtype != jnp.float32:
        raise ValueError("Weight scales must be FP32")
    if gamma_qr.ndim != 1 or gamma_kv.ndim != 1:
        raise ValueError("RMSNorm weights must be vectors")
    if (
        gamma_qr.dtype not in (jnp.bfloat16, jnp.float32)
        or gamma_kv.dtype != gamma_qr.dtype
    ):
        raise ValueError("RMSNorm weights must share a BF16 or FP32 dtype")
    rank, head_dim = gamma_qr.shape[0], gamma_kv.shape[0]
    if weight.shape[1] != rank + head_dim:
        raise ValueError("Projection width must equal qr width plus KV width")
    if rank % _LANE or head_dim % _LANE:
        raise ValueError("qr and KV widths must be multiples of 128 lanes")
    if eps <= 0:
        raise ValueError("RMSNorm epsilon must be positive")


def supported(x, weight, weight_scale, gamma_qr, gamma_kv, rope_rows):
    """Whether ``fused_qa_kv_norm_rope`` accepts these concrete lowering inputs."""
    try:
        _check_inputs(x, weight, weight_scale, gamma_qr, gamma_kv, 1e-6)
        rope_gather.check_rope_rows(rope_rows, x.shape[0], gamma_kv.shape[0])
    except ValueError:
        return False
    return True
