# SPDX-License-Identifier: Apache-2.0
"""q38 PSTEP-QCHAIN (VLLM_TPU_QSA_QCHAIN, default off): repack of the QSA prefill serial chain (top-K end -> first walk
chunk, and the post-walk tail) on a disaggregated prefill stage. Every edit is scheduling or layout only, or integer-exact;
the class is BITWISE expected and is claimed only on a device bitwise row.

Edits (VLLM_TPU_QSA_QCHAIN_EDITS, digits, default "1234"; read only with the master key on):
  E1  qsa_rpa.qsa_attention (ZERO_PREFIX_FAST route): the decode prefix's row slices (q / gate / blocks / table [:capacity])
      become their own buffers in front of the routing conditional (one optimization barrier), so q / gate / blocks /
      table are dead after the conditional and its identity branch hands them through without copies (the measured
      2 x ~50 us bf16[6400,24,256] copies plus the s32[6400,512] copy).
  E2  qsa_union_dense.union_walk_prefill: the per-step compact pool plan and copy (selection independent) are traced in
      front of the plan-dependent conditionals instead of inside them, so they no longer wait for the SparseCore bits
      plan (the ~65 us of integer glue between the plan and the list, plus the 80 us compact copy).
  E3  (see qsa_qchain_e3) the qgate relayout feeding QPREP is ordered after the layer's last top-K window.
  E4  (see qsa_qchain_e4) the walk writes its rows in place in the 2-D [T, heads * D] layout o_proj reads.
"""

import jax

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)
EDITS = ("1", "2", "3", "4")


def edits() -> frozenset:
    """The engaged edits (empty with the master key off). A malformed selector fails at trace time."""
    if not envs.VLLM_TPU_QSA_QCHAIN:
        return frozenset()
    spec = envs.VLLM_TPU_QSA_QCHAIN_EDITS
    if not spec or set(spec) - set(EDITS):
        raise ValueError(f"VLLM_TPU_QSA_QCHAIN_EDITS must be digits from 1-4, got {spec!r}")
    return frozenset(spec)


def engaged(edit: int) -> bool:
    return str(edit) in edits()


def log_engaged(edit: int, rows: int, detail: str) -> None:
    """One INFO line per traced call (the STEP kit's engagement rows count them)."""
    logger.info("q38 PSTEP-QCHAIN E%d ENGAGED: %d rows, %s (VLLM_TPU_QSA_QCHAIN)", int(edit), int(rows), detail)


def e1_pin_decode_rows(query, gate, blocks, table, capacity):
    """E1: ``(query, gate, blocks, table), (q_rows, gate_rows, block_rows, table_rows)``; the row slices ``[:capacity]``
    are materialized in front of one optimization barrier that also carries the whole arrays, so every read of the
    whole arrays by the decode prefix happens before the routing conditional. Values unchanged (the barrier is the
    identity)."""
    out = jax.lax.optimization_barrier(
        (query, gate, blocks, table, query[:capacity], gate[:capacity], blocks[:capacity], table[:capacity])
    )
    log_engaged(1, len(query), "decode-prefix rows %d pinned before the routing conditional" % int(capacity))
    return out[:4], out[4:]


def e3_q_after_selection(produce, blocks, qgate, key, value, index_qk, nq, nk, niq, positions, trig, norm_eps, rotary_dim):
    """E3: ``(q, gate)`` = ``produce``'s walk query and gate, traced again from a qgate that one optimization barrier ties
    to the layer's selection ``blocks``: the qgate relayout and the q prep (QPREP-FUSED or the XLA form) can then start
    only after the last top-K window, outside the SparseCore top-K chain. The same function on the same values (the
    barrier is the identity); the caller's first ``produce`` q / gate become dead and XLA drops them."""
    pinned, _ = jax.lax.optimization_barrier((qgate, blocks))
    q, _, _, gate, _, _ = produce(pinned, key, value, index_qk, nq, nk, niq, positions, trig, norm_eps, rotary_dim)
    log_engaged(3, len(qgate), "q / gate traced after the selection (one barrier on the top-K output)")
    return q, gate


def e4_gate_2d(qgate, heads, head_dim):
    """E4: the gate ``qgate.reshape(T, heads, 2 * D)[..., D:]`` as [T, heads * D] in o_proj's 2-D layout, built from
    column slices of the 2-D qgate (no [T, heads, 2 * D] relayout): column h * D + d is qgate column h * 2D + D + d.
    The same bytes."""
    width = 2 * head_dim
    if qgate.ndim != 2 or qgate.shape[1] != heads * width:
        raise ValueError(f"e4_gate_2d: qgate {qgate.shape} is not [T, {heads} x {width}]")
    log_engaged(4, len(qgate), "gate and walk output in o_proj's [rows, %d] layout" % (heads * head_dim))
    return jax.numpy.concatenate(
        [qgate[:, h * width + head_dim:(h + 1) * width] for h in range(heads)], axis=1
    )
