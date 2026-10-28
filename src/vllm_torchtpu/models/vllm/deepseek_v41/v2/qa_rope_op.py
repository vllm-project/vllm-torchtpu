"""Torch bridges for the fused Q-A/KV projection, RMSNorm and SWA RoPE.

``get_qa_rope_op`` replaces ``_fused_wqa_wkv_gemm`` -> ``_split_qkv_and_norm``
(-> the SWA ``_rotate``) with one kernel. Its ``rope_rows`` come from
``get_rope_rows_op``: the step's ``cos_sin_cache`` rows, gathered once and
shared by every layer of a stage (``DeepseekV41Attention._rope_rows``).

``pallas.jax_op`` requires every parameter and the return of the bridged
function to be annotated arrays, so each mode gets its own explicit signature
(no ``*args``, no keyword parameters, no ``Any`` return).
"""

import functools
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.attention_preprocessing import qa_rope, rope_gather
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import get_pallas_op_mesh

_ops: dict[tuple[float, bool, bool], Any] = {}
_rows_op: Any = None


def _gather_rows(cos_sin_cache: jax.Array, positions: jax.Array) -> jax.Array:
    return rope_gather.gather_rows(cos_sin_cache, positions)


def get_rope_rows_op() -> Any:
    """Op ``(cos_sin_cache, positions) -> [T, rotary_dim]`` FP32 rows, clamped."""
    global _rows_op
    if _rows_op is None:
        _rows_op = pallas.jax_op(
            "pallas::dsv41_rope_rows",
            _gather_rows,
            mesh=get_pallas_op_mesh(),
            input_partition_specs=cast(Any, (P(), P())),
        )
        _rows_op.register_fake(
            lambda cache, pos: cache.new_empty((pos.shape[0], cache.shape[1]))
        )
    return _rows_op


def _run(x, w, scale, gamma_qr, gamma_kv, rows, *, eps, emit_qr):
    rank, head_dim = gamma_qr.shape[0], gamma_kv.shape[0]
    if x.shape[0] == 0:  # empty dummy batches
        kv = jnp.zeros((0, head_dim), jnp.bfloat16)
        return (jnp.zeros((0, rank), jnp.bfloat16), kv) if emit_qr else kv
    qr, kv = qa_rope.fused_qa_kv_norm_rope(
        x,
        w,
        scale,
        gamma_qr,
        gamma_kv,
        rows,
        eps=eps,
        emit_qr=emit_qr,
        rotate_kv=rows is not None,
    )
    return (qr, kv) if emit_qr else kv


def _bridge(eps: float, emit_qr: bool, rotate_kv: bool):
    """The JAX function for one mode, with a fully annotated signature."""
    run = functools.partial(_run, eps=eps, emit_qr=emit_qr)
    if emit_qr and rotate_kv:

        def qr_kv_rotated(
            x: jax.Array,
            w: jax.Array,
            scale: jax.Array,
            gamma_qr: jax.Array,
            gamma_kv: jax.Array,
            rope_rows: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            return run(x, w, scale, gamma_qr, gamma_kv, rope_rows)

        return qr_kv_rotated
    if emit_qr:

        def qr_kv(
            x: jax.Array,
            w: jax.Array,
            scale: jax.Array,
            gamma_qr: jax.Array,
            gamma_kv: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            return run(x, w, scale, gamma_qr, gamma_kv, None)

        return qr_kv
    if rotate_kv:

        def kv_rotated(
            x: jax.Array,
            w: jax.Array,
            scale: jax.Array,
            gamma_qr: jax.Array,
            gamma_kv: jax.Array,
            rope_rows: jax.Array,
        ) -> jax.Array:
            return run(x, w, scale, gamma_qr, gamma_kv, rope_rows)

        return kv_rotated

    def kv_only(
        x: jax.Array,
        w: jax.Array,
        scale: jax.Array,
        gamma_qr: jax.Array,
        gamma_kv: jax.Array,
    ) -> jax.Array:
        return run(x, w, scale, gamma_qr, gamma_kv, None)

    return kv_only


def get_qa_rope_op(eps: float, *, emit_qr: bool, rotate_kv: bool) -> Any:
    """Op ``(x, w, scale, gamma_qr, gamma_kv[, rope_rows]) -> (qr, kv) | kv``."""
    key = (float(eps), emit_qr, rotate_kv)
    if key in _ops:
        return _ops[key]
    op = pallas.jax_op(
        f"pallas::dsv41_qa_kv_norm_rope_{len(_ops)}",
        _bridge(float(eps), emit_qr, rotate_kv),
        mesh=get_pallas_op_mesh(),
        input_partition_specs=cast(Any, (P(),) * (6 if rotate_kv else 5)),
    )

    def fake(x, w, scale, gamma_qr, gamma_kv, *_):
        kv = x.new_empty((x.shape[0], gamma_kv.shape[0]))
        return (x.new_empty((x.shape[0], gamma_qr.shape[0])), kv) if emit_qr else kv

    op.register_fake(fake)
    _ops[key] = op
    return op
