"""Scale-one CSA E4M3 attention with paged C512 gather and SWA-output reuse.

SWA remains BF16. Short-prefix specialization is guarded by actual per-row
valid lengths; it is never selected from an assumed single-request history.
"""
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

from .csa_c512_gather import gather_c512
from .csa_fp8_offset import plain_kernel
from .sparse_mla import gather_page_ids, prepare_swa_inputs


def consume(q, kv, lens, sink, acc, swa_l, swa_m, *, offset, sm_scale):
    n = kv.shape[0]
    batch = next(b for b in range(min(n, 32), 0, -1) if n % b == 0)
    assert q.dtype == jnp.float8_e4m3fn and q.shape[1:] == (64, 512)
    assert acc.shape == q.shape and acc.dtype == jnp.bfloat16
    assert offset >= 0 and offset + n <= q.shape[0]
    acc, swa_l, swa_m = prepare_swa_inputs(acc, swa_l, swa_m)
    args = [sink, q, kv, None, acc, swa_l, swa_m, q, None, None, None, None, None]
    hbm = pl.BlockSpec(memory_space=tpu.HBM)
    specs = [pl.BlockSpec(memory_space=tpu.VMEM), hbm, hbm, None,
             hbm, hbm, hbm, hbm, None, None, None, None, None]
    qb = tpu.VMEM((2, batch, 64, 512), jnp.bfloat16)
    stats = tpu.VMEM((2, batch, 128), jnp.float32)
    scratch = [tpu.VMEM((2, batch, *kv.shape[1:]), kv.dtype), None, None,
               qb, stats, stats, qb, tpu.SemaphoreType.DMA((7, 2)),
               tpu.VMEM((2, batch, 64, 512), jnp.float8_e4m3fn),
               None, None, None, tpu.SemaphoreType.DMA((4, 2))]
    scalars = [lens, jnp.asarray([0, n, n], jnp.int32),
               jnp.zeros(2, jnp.int32), jnp.full(2, -1, jnp.int32)]
    fn = functools.partial(plain_kernel, sm_scale=sm_scale, batch_size=batch,
                           schedule='free', input_offset=offset,
                           output_offset=offset, chunk_tokens=n)
    # Four scalar leaves + sink, q8, kv, acc. Reuse the sole acc operand.
    return pl.pallas_call(
        fn, out_shape=jax.ShapeDtypeStruct(acc.shape, jnp.bfloat16),
        grid_spec=tpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4, in_specs=specs, out_specs=hbm,
            grid=(n // batch,), scratch_shapes=scratch),
        compiler_params=tpu.CompilerParams(
            dimension_semantics=('arbitrary',), vmem_limit_bytes=100 * 1024 * 1024,
            disable_bounds_checks=True),
        input_output_aliases={7: 0}, name='CSA_FP8_OFFSET_REUSE', debug=False,
    )(*scalars, *args)


def pipeline(q, cache, ids, lens, sink, acc, swa_l, swa_m, *, sm_scale):
    n = q.shape[0]
    chunk = min(512, n)
    count = pl.cdiv(n, chunk)
    compact_widths = [min(512, 128 * pl.cdiv(pl.cdiv(min((i + 1) * chunk, n), 4), 128))
                      for i in range(count)]
    widths_per_row = jnp.concatenate([
        jnp.full((min(chunk, n - i * chunk),), width, jnp.int32)
        for i, width in enumerate(compact_widths)])

    def run(compact, operands):
        q, cache, ids, lens, sink, destination, swa_l, swa_m = operands
        widths = compact_widths if compact else [512] * count
        window = count if compact else min(2, count)
        gathered = {}

        def gather(i, token=None):
            start, end = i * chunk, min((i + 1) * chunk, n)
            ix, c = ids[start:end, :widths[i]], cache
            if token is not None:
                token, c, ix = jax.lax.optimization_barrier((token, c, ix))
            gathered[i] = gather_c512(c, ix.reshape(-1)).reshape(end - start, widths[i], 128)
            return token

        for i in range(window):
            gather(i)
        for i in range(count):
            start, end = i * chunk, min((i + 1) * chunk, n)
            destination = consume(q, gathered.pop(i), lens[start:end], sink,
                                  destination, swa_l, swa_m, offset=start, sm_scale=sm_scale)
            if i + window < count:
                destination = gather(i + window, destination)
        return destination

    # The indexer emits valid entries as a prefix, also required by main's
    # sparse kernel. This predicate proves every omitted column is masked.
    compact_ok = jnp.all(lens <= widths_per_row)
    return jax.lax.cond(compact_ok, functools.partial(run, True),
                        functools.partial(run, False),
                        (q, cache, ids, lens, sink, acc, swa_l, swa_m))


@functools.partial(jax.jit, static_argnames=('sm_scale',))
def sparse_csa_fp8(q, cache, topk, pages, cu_q_lens, distribution,
                    sink, acc, swa_l, swa_m, *, sm_scale):
    assert q.shape[1:] == (64, 512) and topk.shape == (q.shape[0], 512)
    assert cache.dtype == jnp.uint8 and cache.shape[2:] == (4, 128)
    n, max_seqs = q.shape[0], cu_q_lens.shape[0] - 1
    if n == 0:
        return acc
    page_size = cache.shape[1]
    valid_tokens = cu_q_lens[distribution[2]]
    topk = jnp.where(jnp.arange(n)[:, None] < valid_tokens, topk, -1)
    seq_ids = jnp.repeat(jnp.arange(max_seqs), cu_q_lens[1:] - cu_q_lens[:-1],
                         total_repeat_length=n)
    lens = jnp.sum(topk != -1, axis=-1).astype(jnp.int32)
    physical_pages = gather_page_ids(pages.reshape(-1), topk // page_size,
                                     seq_ids, max_seqs)
    flat = jnp.arange(n * 512, dtype=jnp.int32).reshape(n, 512)
    # Preserve main's distribution of masked reads; the consumer zeros their
    # bytes before QK/PV so uninitialized cache NaNs cannot enter either GEMM.
    physical_pages = jnp.where(topk == -1, (flat * 104729) % cache.shape[0], physical_pages)
    offsets = jnp.where(topk == -1, (flat * 15485863) % page_size, topk % page_size)
    ids = physical_pages * page_size + offsets
    return pipeline(q, cache, ids, lens, sink, acc, swa_l, swa_m, sm_scale=sm_scale)
