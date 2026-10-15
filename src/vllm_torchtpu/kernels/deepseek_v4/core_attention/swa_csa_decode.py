# SPDX-License-Identifier: Apache-2.0
"""Experimental BF16 decode composition of the existing SWA and CSA kernels.

Contract: one query per active sequence, dense decode prefix, H64/D512/topk512,
separate SWA and CSA caches. No prefill or aliased cache support. The caller
must validate the decode metadata before entering the timed compiled region.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from . import csa_gather_compact as csa_gather
from . import mla_swa, sparse_mla


def _kernel(
    kv_lens,
    pages,
    cu,
    bounds,
    csa_lens,
    sinks,
    q,
    new_kv,
    cache,
    nope,
    rope,
    output,
    updated_cache,
    bkv,
    bq,
    bo,
    sems,
    csa_nope,
    csa_rope,
    csa_sems,
    *,
    sm_scale,
    logical_page_size,
    sliding_window,
    batch_size,
    csa_batch_size,
):
    def fetch(batch, slot, wait):
        seq = batch * batch_size
        token = cu[jnp.minimum(seq, cu.shape[0] - 2)]
        size = jnp.clip(bounds[1] - seq, 0, csa_batch_size)
        for src, dst, sem in (
            (nope, csa_nope, csa_sems.at[slot, 0]),
            (rope, csa_rope, csa_sems.at[slot, 1]),
        ):
            copy = pltpu.make_async_copy(
                src.at[pl.ds(token, size)],
                dst.at[slot, pl.ds(0, size)],
                sem,
            )
            if wait:
                copy.wait()
            else:
                copy.start()

    def finalize(batch, slot, queries, swa_acc, swa_l, swa_m, out):
        seq = batch * batch_size
        token = cu[jnp.minimum(seq, cu.shape[0] - 2)]
        prev = None
        prev_out = None
        for i in range(csa_batch_size):
            length = csa_lens[token + i]
            kv = sparse_mla.decode_gathered_kv(
                csa_nope.at[slot, i].bitcast(jnp.uint8)[...],
                csa_rope.at[slot, i].bitcast(jnp.bfloat16)[...],
                length,
            )
            query = queries[i]
            if prev_out is not None:
                query = jnp.where(prev_out == jnp.inf, prev_out, query)
            p, e, denominator = sparse_mla.flash_attention_step1_qk_softmax(
                query,
                kv,
                length,
                swa_m[i, :, None],
                swa_l[i, :, None],
                sinks[...][:, None],
                sm_scale=sm_scale,
            )
            if prev is not None:
                prev_out = sparse_mla.flash_attention_step2_pv(
                    *prev,
                    q_dtype=q.dtype,
                )
                out[i - 1] = prev_out
            prev = (p, kv, e, swa_acc[i], denominator)
        assert prev is not None
        out[csa_batch_size - 1] = sparse_mla.flash_attention_step2_pv(
            *prev,
            q_dtype=q.dtype,
        )

    mla_swa._mla_batched_decode_kernel(
        kv_lens,
        pages,
        cu,
        bounds,
        sinks,
        q,
        new_kv,
        cache,
        None,
        None,
        None,
        output,
        updated_cache,
        None,
        None,
        bkv,
        bq,
        bo,
        None,
        None,
        sems,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        logical_page_size=logical_page_size,
        unnormalized_output=True,
        bkv_p=2,
        seq_batch_sz=batch_size,
        fetch_extra=fetch,
        finalize=finalize,
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "logical_page_size",
        "sliding_window",
        "use_decode_metadata",
    ),
)
def fused_decode(
    q,
    new_kv,
    sw,
    sl,
    sp,
    cu,
    distribution,
    nope,
    rope,
    topk,
    pages,
    sink,
    *,
    sm_scale=512**-0.5,
    logical_page_size=128,
    sliding_window=128,
    use_decode_metadata=False,
):
    """Full SWA update + CSA gather + fused BF16 attention; no silent fallback."""
    tokens, heads, dim = q.shape
    if (
        q.dtype != jnp.bfloat16
        or (heads, dim) != (64, 512)
        or topk.shape != (tokens, 512)
        or tokens not in (1, 2, 4, 8, 16, 32, 64)
        or logical_page_size != 128
        or sliding_window != 128
        or sw.dtype != jnp.uint8
        or sw.shape[2:] != (4, 128)
        or nope.dtype != jnp.int32
        or rope.dtype != jnp.int32
    ):
        raise NotImplementedError("Experimental fused decode envelope mismatch")
    indices, lengths = sparse_mla.prepare_gather_indices(
        topk, pages, cu, nope, use_decode_metadata=use_decode_metadata
    )
    gathered_nope, gathered_rope = csa_gather.csa_gather(
        nope,
        rope,
        indices,
        num_valid_indices=cu[distribution[0]] * 512,
        top_k=512,
    )
    gathered_nope = gathered_nope.reshape(tokens, 512, 128)
    gathered_rope = gathered_rope.reshape(tokens, 128, 128)
    sl, sp = mla_swa.narrow_page_indices(
        sl,
        sp,
        cu,
        sliding_window=sliding_window,
        logical_page_size=logical_page_size,
        max_q_len=tokens,
        max_kv_pages_per_block=2,
    )
    new_kv = mla_swa.prepare_kv_inputs(new_kv).reshape(tokens, 8, 128)
    # Avoid padding measured small batches to 32; preserve other callers.
    batch_size = min(tokens, 32) if tokens in (1, 4, 8, 16) else 32
    csa_batch_size = min(tokens, batch_size)
    bkv_size = 2 * logical_page_size
    scratch = [
        pltpu.VMEM((2, batch_size, bkv_size, 8, 128), jnp.uint8),
        pltpu.VMEM((2, batch_size, heads, dim), q.dtype),
        pltpu.VMEM((2, batch_size, heads, dim), q.dtype),
        pltpu.SemaphoreType.DMA((2, batch_size + 4)),
        pltpu.VMEM((2, csa_batch_size, 512, 128), jnp.int32),
        pltpu.VMEM((2, csa_batch_size, 128, 128), jnp.int32),
        pltpu.SemaphoreType.DMA((2, 2)),
    ]
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    name = "SWA_CSA_BF16_FUSED_DECODE"
    return jax.named_scope(name)(
        pl.pallas_call(
            functools.partial(
                _kernel,
                sm_scale=sm_scale,
                logical_page_size=logical_page_size,
                sliding_window=sliding_window,
                batch_size=batch_size,
                csa_batch_size=csa_batch_size,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=5,
                in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM), *([hbm] * 5)],
                out_specs=[hbm, hbm],
                grid=(jnp.where(distribution[0] > 0, 1, 0),),
                scratch_shapes=scratch,
            ),
            out_shape=[
                jax.ShapeDtypeStruct(q.shape, q.dtype),
                jax.ShapeDtypeStruct(sw.shape, sw.dtype),
            ],
            input_output_aliases={8: 1},
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("arbitrary",),
                vmem_limit_bytes=mla_swa.DEFAULT_VMEM_LIMIT_BYTES,
                disable_bounds_checks=True,
            ),
            name=name,
        )
    )(
        sl,
        sp,
        cu,
        jnp.stack([jnp.int32(0), distribution[0]]),
        lengths,
        sink,
        q,
        new_kv,
        sw,
        gathered_nope,
        gathered_rope,
    )
