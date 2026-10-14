# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused attention over shared BF16 sliding-window pieces."""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.logger import init_logger

from . import tuned_window_attention

logger = init_logger(__name__)


def supports_window_attention(q: jax.Array, records: jax.Array) -> bool:
    """Static envelope for the measured shared-window prefill route."""
    if (
        jax.default_backend() != "tpu"
        or str(pltpu.get_tpu_info().chip_version) != "7x"
        or q.dtype != jnp.bfloat16
        or q.shape[0] < 128
        or q.shape[0] % 64
    ):
        return False
    key = (q.shape[1], q.shape[2], records.shape[1])
    if key not in tuned_window_attention.QUERY_BLOCKS:
        logger.warning_once("No measured SWA tile for TPU7x workload %s", key)
        return False
    return True


def _kernel(q_ref, kv_ref, mask_ref, m_out, l_out, n_out, *, scale):
    queries, heads, dim = q_ref.shape
    pieces, _, keys, width = kv_ref.shape
    mask = mask_ref[0]
    used = mask.astype(jnp.int32).sum(axis=0, keepdims=True).T > 0
    split_kv = jnp.concatenate([kv_ref[p, 0] for p in range(pieces)], axis=1)
    # Restore the feature order on the small, shared KV tile; queries and
    # attention outputs remain in their native layout throughout.
    order = jnp.concatenate(
        [
            start + 2 * jnp.arange(width) + parity
            for start in range(0, dim, 2 * width)
            for parity in (0, 1)
        ]
    )
    permutation = (order[:, None] == jnp.arange(dim)[None]).astype(q_ref.dtype)
    kv = jnp.dot(
        jnp.where(used, split_kv, 0), permutation, preferred_element_type=jnp.float32
    ).astype(q_ref.dtype)
    scores = jnp.dot(
        q_ref[...].reshape(queries * heads, dim),
        kv.T,
        preferred_element_type=jnp.float32,
    )
    scores = scores.reshape(queries, heads, keys) * scale
    scores = jnp.where(mask[:, None], scores, -jnp.inf)
    maximum = jnp.maximum(-1e30, scores.max(-1))
    probabilities = jnp.exp(scores - maximum[..., None])
    m_out[...] = maximum
    l_out[...] = probabilities.sum(-1)
    numerator = jnp.dot(
        probabilities.reshape(queries * heads, keys).astype(q_ref.dtype),
        kv,
        preferred_element_type=jnp.float32,
    )
    n_out[...] = numerator.reshape(queries, heads, dim).astype(n_out.dtype)


def attend_windows(
    q: jax.Array,
    kv: jax.Array,
    mask: jax.Array,
    scale: float,
    *,
    query_block: int,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute an initial SWA partial with natural query and output layouts."""
    tokens, heads, dim = q.shape
    pieces, groups, keys, width = kv.shape
    queries = mask.shape[1]
    assert q.dtype == kv.dtype == jnp.bfloat16
    assert mask.dtype == jnp.bool_ and mask.shape == (groups, queries, keys)
    assert dim == pieces * width and width == 128
    assert pieces % 2 == 0 and query_block > 0
    assert tokens == groups * queries and queries % query_block == 0
    pad = -keys % 128
    kv = jnp.pad(kv, ((0, 0), (0, 0), (0, pad), (0, 0)))
    mask = jnp.pad(mask, ((0, 0), (0, 0), (0, pad)))
    keys += pad
    tiles = queries // query_block
    q_spec = pl.BlockSpec((query_block, heads, dim), lambda g, t: (g * tiles + t, 0, 0))
    stats_spec = pl.BlockSpec((query_block, heads), lambda g, t: (g * tiles + t, 0))
    return pl.pallas_call(
        functools.partial(_kernel, scale=scale),
        grid=(groups, tiles),
        in_specs=(
            q_spec,
            pl.BlockSpec((pieces, 1, keys, width), lambda g, t: (0, g, 0, 0)),
            pl.BlockSpec((1, query_block, keys), lambda g, t: (g, t, 0)),
        ),
        out_specs=(stats_spec, stats_spec, q_spec),
        out_shape=(
            jax.ShapeDtypeStruct((tokens, heads), jnp.float32),
            jax.ShapeDtypeStruct((tokens, heads), jnp.float32),
            jax.ShapeDtypeStruct(q.shape, q.dtype),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary")
        ),
        interpret=interpret,
        name="swa_window_attention",
    )(q, kv, mask)
