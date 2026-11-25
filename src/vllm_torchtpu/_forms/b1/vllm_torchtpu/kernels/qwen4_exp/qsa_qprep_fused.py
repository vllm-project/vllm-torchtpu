# SPDX-License-Identifier: Apache-2.0
"""q38 PSTEP-QSAGLUE QPREP-FUSED (VLLM_TPU_QSA_PREFILL_QPREP_FUSED, default off; prefill rows only; NOT bitwise).

The prefill walk query ``main_rope(gemma_norm(qgate[..., :D], nq, eps), positions, trig, rotary_dim)`` and the gate
``qgate[..., D:]`` in ONE Pallas pass over ``qgate`` (read once, q and gate written once, both row-major).

Why: XLA lowers the reference to eight serial TensorCore ops on the 6,400-row prefill step (M, T64qB j-9ab6d23e flag arm,
stage 4): the qgate split into row-major q and gate (112 us), a copy of q to layout {0,2,1} (108 us), the norm reduce (25 us)
and apply (53 us), a copy of the pass-through dims x[..., rotary_dim:] (41 us), the rope halves (15 us), the concatenate (47
us) and a copy of q back to row-major for the walk (102 us): 503 us per QSA layer for 157 MB in / 157 MB out.

Math (same elementwise sequence as ``qsa.gemma_norm`` + ``qsa.main_rope``, with ``rounding.bf16``'s integer RNE, kept in f32):
  x = q part as f32; y = bf16((x * rsqrt(mean(x * x) + eps)) * (1 + w));
  lanes d < h (h = rotary_dim / 2):   bf16(y[d] * c[d] - y[d + h] * s[d])
  lanes h <= d < 2h:                  bf16(y[d] * c[d - h] + y[d - h] * s[d - h])
  lanes d >= 2h:                      y[d]
  gate = qgate[..., D:] (bytes copied).
The rope partner lanes come from one exact 0/1 permutation matmul on the bf16 y (one nonzero product per output, f32
accumulation: exact), so no lane rotate whose direction could differ between interpret mode and Mosaic.
NB CLASS (stated): the 256-element f32 sum of squares runs in this kernel's lane order, not in XLA's {0,2,1}-layout order, and
rsqrt is Mosaic's; q may differ from the reference by a BF16 ulp where the rounding of the norm scale moves. The gate is bitwise.
"""

import collections
import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs

MIN_ROWS = 1024  # the union walk's prefill row floor (qsa_rpa); smaller steps (decode, verify, short prefill) keep XLA's form
TOKEN_BLOCK = 128
VMEM_LIMIT = 48 << 20  # 128-token blocks: 2 x (3.1 MB in + 3.1 MB out) buffers + ~20 MB of f32 temporaries (v7x: 64 MiB)
_LANES = 128

# q38 PSTEP-WARMTWIN: what this process decided at trace time (Python only; nothing enters the program).
# _DECISIONS: (rows, heads, head_dim, rotary_dim) -> None (fused) or the first failing clause of `decide`, written by
# every `applies` call (qsa.produce calls it on every traced QSA step while the key is set). _ENGAGED_ROWS: rows ->
# traced `fused_qprep` calls (each prints the trace-time ENGAGED line). runner/prefill_twins.py reads both.
_DECISIONS: dict = {}
_ENGAGED_ROWS: collections.Counter = collections.Counter()


def decide(rows: int, heads: int, head_dim: int, rotary_dim: int, token_block: int = TOKEN_BLOCK) -> str | None:
    """Pure: None when the fused prefill q prep serves this step, else the first clause that fails."""
    if rows < MIN_ROWS:
        return f"rows<{MIN_ROWS}"
    if rows % token_block != 0:
        return f"rows%{token_block}!=0"
    if heads % 8 != 0:
        return "heads%8!=0"
    if head_dim % _LANES != 0:
        return f"head_dim%{_LANES}!=0"
    if not 0 < rotary_dim <= _LANES:
        return f"rotary_not_in_1..{_LANES}"
    if rotary_dim % 2 != 0:
        return "rotary_odd"
    if envs.VLLM_TPU_QSA_BF16_REDUCE_PRECISION:  # the kernel replicates rounding.bf16's integer RNE only
        return "bf16_reduce_precision"
    return None


def applies(rows: int, heads: int, head_dim: int, rotary_dim: int, token_block: int = TOKEN_BLOCK) -> bool:
    """Static: the fused prefill q prep serves this step (else the caller keeps the reference ops)."""
    reason = decide(rows, heads, head_dim, rotary_dim, token_block)
    _DECISIONS[(int(rows), int(heads), int(head_dim), int(rotary_dim))] = reason
    return reason is None


def _partner_matrix(rotary_dim: int) -> np.ndarray:
    """[128, 128] 0/1: (y @ P)[d] = y[d + h] for d < h, y[d - h] for h <= d < 2h, 0 elsewhere (h = rotary_dim // 2)."""
    h = rotary_dim // 2
    p = np.zeros((_LANES, _LANES), np.float32)
    for d in range(h):
        p[d + h, d] = 1.0
        p[d, d + h] = 1.0
    return p


def _rne(x):
    """``rounding.bf16(x).astype(f32)`` bit for bit, kept in f32 (its integer RNE; the result's low 16 bits are zero)."""
    bits = jax.lax.bitcast_convert_type(x, jnp.uint32)
    bits = (bits + jnp.uint32(0x7FFF) + ((bits >> 16) & 1)) & jnp.uint32(0xFFFF0000)
    return jax.lax.bitcast_convert_type(bits, jnp.float32)


def _kernel(qg_ref, w_ref, c_ref, s_ref, p_ref, q_ref, g_ref, *, head_dim, rotary_dim, eps):
    # All vector math in f32 ([tb, heads, .] with heads a multiple of the f32 sublane tile); BF16 only at the stores (exact:
    # every stored value has already been rounded to BF16 by _rne).
    x = qg_ref[:, :, :head_dim].astype(jnp.float32)                       # [tb, heads, D]
    mean = jnp.mean(x * x, axis=-1, keepdims=True)
    y = _rne(x * jax.lax.rsqrt(mean + eps) * w_ref[...])                  # gemma_norm (w_ref = 1 + nq, f32)
    lo = y[:, :, :_LANES]
    tb, heads = lo.shape[0], lo.shape[1]
    # an exact 0/1 permutation of the BF16-valued lanes (one nonzero product per output, f32 accumulation)
    partner = jnp.dot(lo.reshape(tb * heads, _LANES), p_ref[...],
                      preferred_element_type=jnp.float32).reshape(tb, heads, _LANES)
    t1 = lo * c_ref[...]                                                  # c_ref / s_ref: [tb, 1, 128]
    t2 = partner * s_ref[...]
    lane = jax.lax.broadcasted_iota(jnp.int32, lo.shape, 2)
    h = rotary_dim // 2
    roped = _rne(jnp.where(lane < h, t1 - t2, t1 + t2))
    q_ref[:, :, :_LANES] = jnp.where(lane < 2 * h, roped, lo).astype(q_ref.dtype)
    if head_dim > _LANES:
        q_ref[:, :, _LANES:] = y[:, :, _LANES:].astype(q_ref.dtype)
    g_ref[...] = qg_ref[:, :, head_dim:]


def rope_tables(positions, trig, rotary_dim):
    """The rope cos / sin rows of ``qsa._rope_parts`` (same gather, same optimization barrier), laid out per lane for the
    kernel: ``c[t, 0, d] = cos[t, d mod h]`` and likewise ``s`` for ``d < rotary_dim``, 0 elsewhere; f32 [T, 1, 128]."""
    rows = jax.lax.optimization_barrier(trig[positions])
    h = rotary_dim // 2
    cs = rows[:, :rotary_dim].reshape(len(positions), 2, h).astype(jnp.float32)
    pad = ((0, 0), (0, _LANES - rotary_dim))
    c = jnp.pad(jnp.concatenate([cs[:, 0], cs[:, 0]], -1), pad)[:, None, :]
    s = jnp.pad(jnp.concatenate([cs[:, 1], cs[:, 1]], -1), pad)[:, None, :]
    return c, s


def fused_qprep(qgate, nq, positions, trig, norm_eps, rotary_dim, *, token_block=TOKEN_BLOCK, interpret=False):
    """``qgate`` bf16 [T, heads, 2 * D] -> (q bf16 [T, heads, D], gate bf16 [T, heads, D]); see the module docstring."""
    rows, heads, width = qgate.shape
    head_dim = nq.shape[0]
    if width != 2 * head_dim or not applies(rows, heads, head_dim, rotary_dim, token_block):
        raise ValueError(f"fused_qprep: unsupported shape {qgate.shape} / head_dim {head_dim} / rotary_dim {rotary_dim}")
    from vllm_torchtpu.logger import init_logger

    # q38 engagement marker (trace time, one line per traced call; the STEP kit's eng row counts it)
    init_logger(__name__).info(
        "q38 PSTEP-QSAGLUE QPREP-FUSED ENGAGED: %d rows, %d heads, head dim %d, rotary %d (VLLM_TPU_QSA_PREFILL_QPREP_FUSED)",
        rows, heads, head_dim, int(rotary_dim),
    )
    _ENGAGED_ROWS[int(rows)] += 1
    w = (1 + nq.astype(jnp.float32)).reshape(1, 1, head_dim)
    c, s = rope_tables(positions, trig, rotary_dim)
    p = jnp.asarray(_partner_matrix(rotary_dim), jnp.float32)
    tb = int(token_block)
    blk = lambda i: (i, 0, 0)  # noqa: E731
    out = jax.ShapeDtypeStruct((rows, heads, head_dim), qgate.dtype)
    return pl.pallas_call(
        functools.partial(_kernel, head_dim=head_dim, rotary_dim=int(rotary_dim), eps=float(norm_eps)),
        grid=(rows // tb,),
        in_specs=[
            pl.BlockSpec((tb, heads, width), blk),
            pl.BlockSpec((1, 1, head_dim), lambda i: (0, 0, 0)),
            pl.BlockSpec((tb, 1, _LANES), blk),
            pl.BlockSpec((tb, 1, _LANES), blk),
            pl.BlockSpec((_LANES, _LANES), lambda i: (0, 0)),
        ],
        out_specs=[pl.BlockSpec((tb, heads, head_dim), blk), pl.BlockSpec((tb, heads, head_dim), blk)],
        out_shape=[out, out],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",), vmem_limit_bytes=VMEM_LIMIT
        ),
        interpret=interpret,
        name="qsa_qprep_fused",
    )(qgate, w, c, s, p)
