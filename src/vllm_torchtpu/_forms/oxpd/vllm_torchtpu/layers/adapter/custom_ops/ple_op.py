# SPDX-License-Identifier: Apache-2.0
"""Torch boundaries for encoded PLE embedding lookup."""

import functools

import jax
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.qwen4_exp.ple import (
    decode_embeddings,
    resident_lookup_raw,
    short_conv,
)
from vllm_torchtpu.kernels.qwen4_exp.ple_fused import pooled_fused_ple, pooled_fused_ple_kq
from vllm_torchtpu.kernels.qwen4_exp.pool import (
    pooled_short_conv,
    pooled_short_conv_speculative,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)
# q38 PORT-BL MOE: (tokens, chunk, chunk_tokens, min_rows) already recorded.
_LOOKUP_RECORDED: set[tuple[int, int, int, int]] = set()


def _record_lookup(tokens: int, chunk: int, chunk_tokens: int, min_rows: int) -> None:
    """Trace-time engagement record of a non-default PLE lookup (log only, once per
    traced token count): how many sequential gathers the lookup program runs."""
    key = (tokens, chunk, chunk_tokens, min_rows)
    if key in _LOOKUP_RECORDED:
        return
    _LOOKUP_RECORDED.add(key)
    width = min(chunk, -(-tokens // 8) * 8) if tokens else 0
    logger.info(
        "q38 PORT-BL MOE PLE lookup traced for %d tokens: %d gather(s) of %d tokens "
        "(VLLM_TPU_PLE_LOOKUP_CHUNK_TOKENS=%d, VLLM_TPU_PLE_LOOKUP_CHUNK_MIN_ROWS=%d)",
        tokens,
        -(-tokens // width) if width else 0,
        width,
        chunk_tokens,
        min_rows,
    )


@functools.cache
def build_raw_lookup_op(
    local_rows: int, head_dim: int, chunk_tokens: int = 8, min_rows: int = 0
):
    """The shard's first row is the int32[1] operand ``vocab_start``, not part
    of the op: a rank-specific op name or constant would make every TP rank
    trace a different graph (TPU_PRECOMPILE_PEER_REPLAY could not share it).
    ``chunk_tokens``: tokens per gather (VLLM_TPU_PLE_LOOKUP_CHUNK_TOKENS).
    ``min_rows`` (q38 PORT-BL MOE, VLLM_TPU_PLE_LOOKUP_CHUNK_MIN_ROWS): when set,
    lookups of fewer than ``min_rows`` tokens keep the default 8-token gathers
    and only larger ones take ``chunk_tokens``, so the megakernel (D14) buckets,
    at most 64 rows, keep today's program. Decided at JAX trace time on the
    lookup's static token count: no torch-side row test. 0 (default): every
    lookup takes ``chunk_tokens``.
    """
    gated = bool(min_rows) and chunk_tokens != 8

    def fn(table: jax.Array, ids: jax.Array, vocab_start: jax.Array) -> jax.Array:
        chunk = 8 if gated and ids.shape[0] < min_rows else chunk_tokens
        if chunk_tokens != 8:
            _record_lookup(int(ids.shape[0]), chunk, chunk_tokens, min_rows)
        return resident_lookup_raw(
            table,
            ids,
            vocab_start,
            local_rows=local_rows,
            head_dim=head_dim,
            chunk_tokens=chunk,
        )

    suffix = "" if chunk_tokens == 8 else f"_c{chunk_tokens}"
    if gated:  # q38 PORT-BL MOE: its own op name; min_rows unset builds today's op
        suffix += f"_m{min_rows}"
    op = pallas.jax_op(f"qwen4_exp::ple_raw_lookup_{local_rows}_{head_dim}{suffix}", fn)

    @op.register_fake
    def fake(table, ids, vocab_start):
        dtype = torch.int16 if table.dtype == torch.int16 else torch.int8
        return table.new_empty((ids.shape[0], ids.shape[1] * head_dim), dtype=dtype)

    return op


@functools.cache
def build_decode_op():
    def fn(raw_embeddings: jax.Array, scale: jax.Array) -> jax.Array:
        return decode_embeddings(raw_embeddings, scale)

    op = pallas.jax_op("qwen4_exp::ple_decode", fn)

    @op.register_fake
    def fake(raw_embeddings, scale):
        return torch.empty_like(raw_embeddings, dtype=torch.bfloat16)

    return op


@functools.cache
def build_short_conv_op(dilation: int):
    def fn(
        x: jax.Array,
        weight: jax.Array,
        history: jax.Array,
        starts: jax.Array,
        slots: jax.Array,
        initial: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        output, updated = short_conv(
            x, weight, history, starts, slots, initial, dilation=dilation
        )
        return updated, output

    op = pallas.jax_op(f"qwen4_exp::ple_short_conv_{dilation}", fn, donate_argnums=(2,))

    @op.register_fake
    def fake(x, weight, history, starts, slots, initial):
        return torch.empty_like(history), torch.empty_like(x)

    def run(x, weight, history, starts, slots, initial):
        updated, output = op(x, weight, history, starts, slots, initial)
        history.copy_(updated)
        return output

    return run


@functools.cache
def build_pooled_short_conv_op(dilation, state_len, past_matmul=False):
    """``past_matmul``: history taps by one-hot matmul (VLLM_TPU_PLE_CONV_ONEHOT)."""
    fn = functools.partial(
        pooled_short_conv, dilation=dilation, state_len=state_len, past_matmul=past_matmul
    )
    suffix = "_1hm" if past_matmul == "mutant" else ("_1h" if past_matmul else "")
    op = pallas.jax_op(
        f"qwen4_exp::pooled_short_conv_{dilation}_{state_len}{suffix}",
        fn,
        donate_argnums=(2,),
    )

    @op.register_fake
    def fake(x, weight, pool, starts, ids, initial):
        return torch.empty_like(pool), torch.empty_like(x)

    def run(x, weight, pool, starts, ids, initial):
        updated, output = op(x, weight, pool, starts, ids, initial)
        pool.copy_(updated)
        return output

    return run


@functools.cache
def build_pooled_speculative_short_conv_op(dilation, state_len, bounded=0, min_rows=0, fold=False):
    """Pooled PLE history with per-verify-position checkpoint blocks.

    ``bounded`` (VLLM_TPU_PLE_DECODE_BOUNDED, q38 L1 D07-b2): the op for
    uniform-window buckets only (see pooled_short_conv_speculative); its own
    op name, so the default op and its traced graph are unchanged.
    ``min_rows`` (VLLM_TPU_PLE_DECODE_BOUNDED_MIN_ROWS, q38 L1-GLUEFIX): with
    ``bounded``, buckets of fewer tokens trace the default computation; its own
    op name too (0 builds exactly the D07-b2 op).
    ``fold`` (VLLM_TPU_PLE_D14_FOLD, q38 L1-PLE; with ``bounded`` only): the
    bounded op runs the conv and its checkpoint writes as one Pallas program;
    its own op name too (False builds exactly the D07-b2 op).
    """
    fn = functools.partial(
        pooled_short_conv_speculative, dilation=dilation, state_len=state_len
    )
    suffix = ""
    if bounded:
        fn = functools.partial(fn, bounded=int(bounded))
        suffix = f"_bnd{int(bounded)}"
        if min_rows:  # q38 L1-GLUEFIX
            fn = functools.partial(fn, bounded_min_rows=int(min_rows))
            suffix += f"_m{int(min_rows)}"
        if fold:  # q38 L1-PLE
            fn = functools.partial(fn, fold=True)
            suffix += "_fold"
    op = pallas.jax_op(
        f"qwen4_exp::pooled_spec_short_conv_{dilation}_{state_len}{suffix}",
        fn,
        donate_argnums=(2,),
    )

    @op.register_fake
    def fake(x, weight, pool, starts, ids, initial, read_ids, ckpt_ids, windowed):
        return torch.empty_like(pool), torch.empty_like(x)

    def run(x, weight, pool, starts, ids, initial, read_ids, ckpt_ids, windowed):
        updated, output = op(
            x, weight, pool, starts, ids, initial, read_ids, ckpt_ids, windowed
        )
        pool.copy_(updated)
        return output

    return run


@functools.cache
def build_pooled_fused_ple_op(
    streams, hidden, eps, dilation, state_len, token_tile, tap_mode, norm_mode="sum",
    precision="torch", weights_kn=False,
):
    """Prefill PLE after the lookup on the unified pool (VLLM_TPU_PLE_FUSED).

    Arguments: hidden [T,C], decoded embeddings [T,E], key_proj / value_proj
    weights, norm_KEY / norm_QUERY / norm_conv weights (key first: the served step's first-use order, see
    pooled_fused_ple_kq), conv weight [C,taps] or [C,1,taps],
    pool, query_start_loc, state ids, has_initial_states, valid tokens [1].
    Returns the masked addend [T,C]; the pool's history pages are updated.
    """
    fn = functools.partial(
        pooled_fused_ple_kq,
        streams=streams,
        hidden=hidden,
        eps=eps,
        dilation=dilation,
        state_len=state_len,
        token_tile=token_tile,
        tap_mode=tap_mode,
        norm_mode=norm_mode,
        precision=precision,
        weights_kn=weights_kn,
    )
    op = pallas.jax_op(
        f"qwen4_exp::pooled_fused_ple_{streams}x{hidden}_{dilation}_{state_len}"
        f"_m{token_tile}_{tap_mode}{'' if norm_mode == 'sum' else '_n' + norm_mode}"
        f"{'_xp' if precision == 'xla' else ''}{'_kn' if weights_kn else ''}",
        fn,
        donate_argnums=(8,),
    )

    @op.register_fake
    def fake(h, e, key_w, value_w, wk, wq, wc, conv_w, pool, starts, ids, initial, valid):
        return torch.empty_like(pool), torch.empty_like(h)

    def run(h, e, key_w, value_w, wk, wq, wc, conv_w, pool, starts, ids, initial, valid):
        updated, output = op(
            h, e, key_w, value_w, wk, wq, wc, conv_w, pool, starts, ids, initial, valid
        )
        pool.copy_(updated)
        return output

    return run
