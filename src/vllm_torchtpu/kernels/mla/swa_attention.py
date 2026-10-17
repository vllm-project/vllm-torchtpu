# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SWA with native Q/O layout and pipelined DMA and MXU execution.

KV is gathered in the cache's even/odd word planes. Prefill restores feature
order on the shared KV tile; decode permutes the smaller Q/O tiles inside
VMEM. Neither route transposes Q or O in HBM. The grid
pipeline double buffers Q, KV, masks and O. Within a shared window, launch
QK(i+1) before softmax(i), and retain PV(i) in the MRB while processing the
next query tile. QK, the two pending PVs and layout permutations use
disjoint MRB slots. Permutations use MXU operations because interleaving
512-feature vector tiles otherwise creates large padded intermediates.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .tuned_block_sizes import SWA_DECODE_GROUPS, SWA_PIPELINE_GROUPS, SWA_QUERY_ROWS


def supports_swa(q: jax.Array, records: jax.Array) -> bool:
    """The BF16 512-feature v7x envelope of the asynchronous MXU kernel."""
    return (
        q.dtype == jnp.bfloat16
        and q.shape[1] in (8, 16, 32, 64)
        and q.shape[2] == 512
        and records.shape[1] in (128, 256)
        and jax.default_backend() == "tpu"
        and str(pltpu.get_tpu_info().chip_version) == "7x"
    )


def _kernel(
    q_ref, kv_ref, mask_ref, sink_ref, out_ref, *, scale, query_tile, interpret
):
    total_queries, heads, dim = q_ref.shape
    groups, queries, keys = mask_ref.shape
    rows = pl.cdiv(query_tile * heads, 16) * 16
    stride = rows // 4
    tiles = total_queries // query_tile

    def permute(x, *, inverse):
        order = jnp.concatenate(
            [
                start + 2 * jnp.arange(128) + parity
                for start in (0, 256)
                for parity in (0, 1)
            ]
        )
        matrix = (order[:, None] == jnp.arange(512)[None, :]).astype(x.dtype)
        if not inverse:
            matrix = matrix.T
        if interpret:
            return jnp.dot(x, matrix, preferred_element_type=jnp.float32).astype(
                x.dtype
            )
        for k in range(2):
            for mxu in range(2):
                pltpu.matmul_push_rhs(
                    matrix[k * 256 : (k + 1) * 256, mxu * 256 : (mxu + 1) * 256], 0, mxu
                )
                pltpu.matmul_acc_lhs(4 * stride, x[:, k * 256 : (k + 1) * 256], mxu, 0)
        return jnp.concatenate(
            [
                pltpu.matmul_pop(4 * stride, (x.shape[0], 256), jnp.float32, mxu)
                for mxu in range(2)
            ],
            axis=1,
        ).astype(x.dtype)

    def load_kv(group):
        mask = mask_ref[group]
        used = mask.astype(jnp.int32).sum(0)[:, None] > 0
        kv = jnp.concatenate([kv_ref[part, group] for part in range(4)], axis=1)
        kv = jnp.where(used, kv, 0)
        if queries != 1:
            kv = permute(kv, inverse=True)
        return jnp.pad(jnp.where(used, kv, 0), ((0, 256 - keys), (0, 0)))

    shared_kv = load_kv(0) if queries != 1 else None

    def get_kv(i):
        if queries == 1:
            return load_kv(i)
        assert shared_kv is not None
        return shared_kv

    def query(i):
        q = q_ref[pl.ds(i * query_tile, query_tile), :, :].reshape(-1, dim)
        q = jnp.pad(q, ((0, rows - query_tile * heads), (0, 0)))
        return permute(q, inverse=False) if queries == 1 else q

    def launch_qk(i):
        q = query(i)
        kv = get_kv(i)
        for k in range(2):
            ks = slice(k * 256, (k + 1) * 256)
            pltpu.matmul_push_rhs(kv[:, ks], 0, 0, transpose=True)
            pltpu.matmul_acc_lhs((i % 2) * stride, q[:, ks], 0, 0)

    def pop_qk(i):
        if interpret:
            return jnp.dot(query(i), get_kv(i).T, preferred_element_type=jnp.float32)
        return pltpu.matmul_pop((i % 2) * stride, (rows, 256), jnp.float32, 0)

    def launch_pv(i, p):
        kv = get_kv(i)
        for mxu in range(2):
            pltpu.matmul_push_rhs(kv[:, mxu * 256 : (mxu + 1) * 256], 0, mxu)
            pltpu.matmul_acc_lhs((2 + i % 2) * stride, p, mxu, 0)

    def pop_pv(i, p):
        if interpret:
            return jnp.dot(p, get_kv(i), preferred_element_type=jnp.float32)
        return jnp.concatenate(
            [
                pltpu.matmul_pop((2 + i % 2) * stride, (rows, 256), jnp.float32, mxu)
                for mxu in range(2)
            ],
            axis=1,
        )

    def store(i, p, denominator):
        value = pop_pv(i, p)[: query_tile * heads].reshape(query_tile, heads, dim)
        value = (value / denominator[..., None]).astype(out_ref.dtype)
        if queries == 1:
            value = value.reshape(query_tile * heads, dim)
            value = jnp.pad(value, ((0, rows - query_tile * heads), (0, 0)))
            value = permute(value, inverse=True)[: query_tile * heads]
        out_ref[pl.ds(i * query_tile, query_tile), :, :] = value.reshape(
            query_tile, heads, dim
        )

    if not interpret:
        launch_qk(0)
    previous_p = jnp.zeros((rows, 256), q_ref.dtype)
    previous_den = jnp.ones((query_tile, heads), jnp.float32)
    for i in range(tiles):
        scores = pop_qk(i)[: query_tile * heads, :keys].reshape(query_tile, heads, keys)
        if not interpret and i + 1 < tiles:
            launch_qk(i + 1)
        scores = jnp.where(
            mask_ref[i, :, :][:, None, :]
            if queries == 1
            else mask_ref[0, pl.ds(i * query_tile, query_tile), :][:, None, :],
            scores * scale,
            -jnp.inf,
        )
        # Include the sink in the max to keep the all-masked case finite.
        maximum = jnp.maximum(scores.max(-1), sink_ref[...][None, :])
        probabilities = jnp.exp(scores - maximum[..., None])
        denominator = probabilities.sum(-1) + jnp.exp(sink_ref[...][None, :] - maximum)
        p = jnp.pad(
            probabilities.reshape(query_tile * heads, keys),
            ((0, rows - query_tile * heads), (0, 256 - keys)),
        ).astype(q_ref.dtype)
        if not interpret:
            launch_pv(i, p)
        if i:
            store(i - 1, previous_p, previous_den)
        previous_p, previous_den = p, denominator
    store(tiles - 1, previous_p, previous_den)


def attend_swa(
    q: jax.Array,
    kv: jax.Array,
    mask: jax.Array,
    sink: jax.Array,
    scale: float,
    *,
    interpret: bool = False,
) -> jax.Array:
    """Attend shared windows in cache piece order, returning native BF16 Q layout.

    Args:
        q: BF16 [tokens, heads, 512].
        kv: BF16 [4, groups, keys, 128], even/odd pieces per 256 features.
        mask: Boolean [groups, queries, keys], including padding validity.
        sink: FP32 attention sink logits [heads].
        scale: QK scaling factor.
        interpret: Run synchronous math on CPU for numerical tests.
    """
    tokens, heads, dim = q.shape
    pieces, groups, keys, width = kv.shape
    queries = mask.shape[1]
    assert q.dtype == kv.dtype == jnp.bfloat16
    assert dim == 512 and pieces == 4 and width == 128
    assert heads in (8, 16, 32, 64) and keys in (128, 256)
    assert mask.dtype == jnp.bool_ and mask.shape == (groups, queries, keys)
    assert tokens == groups * queries and sink.shape == (heads,)
    query_tile = min(queries, SWA_QUERY_ROWS // heads)
    assert queries % query_tile == 0
    # Four attention regions plus a 256-row permutation fit in the v7x MRB.
    group_batch = math.gcd(groups, SWA_DECODE_GROUPS) if queries == 1 else 1
    windows = groups // group_batch
    pipeline_groups = math.gcd(windows, SWA_PIPELINE_GROUPS)

    def index(outer, inner):
        return outer * pipeline_groups + inner

    buffered = pl.Buffered(buffer_count=2)
    q_spec = pl.BlockSpec(
        (group_batch * queries, heads, dim), lambda o, i: (index(o, i), 0, 0), buffered
    )
    return pl.pallas_call(
        functools.partial(
            _kernel, scale=scale, query_tile=query_tile, interpret=interpret
        ),
        grid=(windows // pipeline_groups, pipeline_groups),
        in_specs=(
            q_spec,
            pl.BlockSpec(
                (4, group_batch, keys, 128),
                lambda o, i: (0, index(o, i), 0, 0),
                buffered,
            ),
            pl.BlockSpec(
                (group_batch, queries, keys), lambda o, i: (index(o, i), 0, 0), buffered
            ),
            pl.BlockSpec((heads,), lambda o, i: (0,)),
        ),
        out_specs=q_spec,
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary")
        ),
        interpret=interpret,
        name="swa_native_layout",
    )(q, kv, mask, sink.astype(jnp.float32))
