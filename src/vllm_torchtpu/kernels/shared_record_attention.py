# SPDX-License-Identifier: Apache-2.0
"""Two-request software pipeline. Distinct QK and two PV accumulator lifetimes.
The source schedules QK1 before softmax0 and PV1 before output packing0.
Compiler scheduling may hoist PV pops; final LLO must establish actual overlap.
Each request has identical mathematics to onepass; no acceptance-path change.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _kernel(records, q, kv, sink, out):
    aq = [
        jax.empty_ref(
            jax.ShapeDtypeStruct((320, 256), jnp.float32), memory_space=pltpu.ACC(m)
        )
        for m in range(2)
    ]
    av = [
        [
            jax.empty_ref(
                jax.ShapeDtypeStruct((320, 256), jnp.float32), memory_space=pltpu.ACC(m)
            )
            for r in range(2)
        ]
        for m in range(2)
    ]

    def values(r, m):
        return jnp.pad(
            jnp.concatenate((kv[2 * m, r], kv[2 * m + 1, r]), axis=1), ((0, 64), (0, 0))
        )

    def qk(r):
        for m in range(2):
            qv = jnp.concatenate((q[2 * m, r], q[2 * m + 1, r]), axis=1)
            pltpu.matmul_push_rhs(values(r, m), 0, m, transpose=True)
            pltpu.matmul_acc_lhs(aq[m], qv, load_staged_rhs=0)

    def scores(r):
        z = (pltpu.matmul_pop(aq[0]) + pltpu.matmul_pop(aq[1])) * (512**-0.5)
        return jnp.where(records[r, 0:1, :] != 0, z, -jnp.inf)

    def softmax_pv(r, s):
        maximum = jnp.maximum(jnp.max(s, axis=1, keepdims=True), -1e30)
        prob = jnp.exp(s - maximum)
        for m in range(2):
            pltpu.matmul_push_rhs(values(r, m), 0, m)
            pltpu.matmul_acc_lhs(av[m][r], prob.astype(jnp.bfloat16), load_staged_rhs=0)
        return jnp.sum(prob, axis=1, keepdims=True) + jnp.exp(sink[...] - maximum)

    qk(0)
    s0 = scores(0)
    qk(1)
    d0 = softmax_pv(0, s0)
    s1 = scores(1)
    d1 = softmax_pv(1, s1)
    for r, denominator in enumerate((d0, d1)):
        for m in range(2):
            value = pltpu.matmul_pop(av[m][r])
            halves = [
                (value[:, pl.ds(h * 128, 128)] / denominator)
                .astype(jnp.bfloat16)
                .astype(jnp.float32)
                for h in range(2)
            ]
            packed = pltpu.pack_elementwise(halves, packed_dtype=jnp.bfloat16)
            out[r, :, pl.ds(m * 256, 256)] = pltpu.bitcast(packed.T, jnp.bfloat16).T


def shared_record_attention(
    records: jax.Array, pieces: jax.Array, gathered: jax.Array, sink: jax.Array
) -> jax.Array:
    requests = records.shape[0]
    assert requests > 0 and requests % 2 == 0
    assert records.shape == (requests, 1, 256) and records.dtype == jnp.int32
    assert pieces.shape == (4, requests, 320, 128)
    assert gathered.shape == (4, requests, 192, 128)
    assert pieces.dtype == gathered.dtype == jnp.bfloat16
    assert sink.shape == (320, 1)

    def spec(shape, fn):
        return pl.BlockSpec(
            shape, fn, pipeline_mode=pl.Buffered(2), memory_space=pltpu.VMEM
        )

    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((requests, 320, 512), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(requests // 2,),
            in_specs=(
                spec((2, 1, 256), lambda i, *_: (i, jnp.int32(0), jnp.int32(0))),
                spec(
                    (4, 2, 320, 128),
                    lambda i, *_: (jnp.int32(0), i, jnp.int32(0), jnp.int32(0)),
                ),
                spec(
                    (4, 2, 192, 128),
                    lambda i, *_: (jnp.int32(0), i, jnp.int32(0), jnp.int32(0)),
                ),
                pl.BlockSpec(
                    (320, 1),
                    lambda *_: (jnp.int32(0), jnp.int32(0)),
                    memory_space=pltpu.VMEM,
                ),
            ),
            out_specs=spec(
                (2, 320, 512), lambda i, *_: (i, jnp.int32(0), jnp.int32(0))
            ),
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        name="dspark_swa_onepass_pair",
    )(records, pieces, gathered, sink)
