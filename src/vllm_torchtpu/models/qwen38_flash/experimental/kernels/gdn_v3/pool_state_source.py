# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Provenance: model-owned copy of ``v3_state_source`` and
#   ``_conv_qk_pair_rows_perm`` from src/vllm_torchtpu/kernels/pool_adapters.py
#   (before #1620 removed them; with the conv-state dtype argument of
#   9567ad1d23). Main's pool layout has no seq-along-lane (HND) form, so that
#   branch is dropped. Owned by models/qwen38_flash from this copy on.
"""Token-range state plan for the model's GDN V3 kernel over the unified pool.

Main's GDN V3 kernel addresses pooled state by granule (#1620); this model's
own kernel copy still streams the SSM and conv regions through a
``StateSourcePlan``, built here from the pool's shape and dtype.
"""

import math

import jax.numpy as jnp

from vllm_torchtpu.kernels.pool_adapters import _pool_geometry

from . import config


def pool_token_bytes(pool) -> int:
    """Bytes of one token row of the unified pool."""
    _, payload, lanes = _pool_geometry(pool)
    return math.prod(payload) * lanes * jnp.dtype(pool.dtype).itemsize


def pooled_conv_state_bytes(*, kernel_size: int, conv_dim: int, dtype) -> int:
    """Live bytes of one slot's conv region: (kernel_size - 1) dense rows of
    conv_dim channels in ``dtype``."""
    return (kernel_size - 1) * conv_dim * jnp.dtype(dtype).itemsize


def _conv_qk_pair_rows_perm(
    taps: int, conv_dim: int, n_v: int, d_v: int, lanes: int
) -> tuple[int, ...] | None:
    """Typed-row permutation for the QK pair-blocked pooled conv layout.

    Stored order per tap interleaves Q and K row-pairs —
    ``[Q0, K0, Q1, K1, ..., V...]`` — so that one whole pool token holds
    one head-pair's Q and K rows and a full-width state can receive a
    TP-rank shard's first conv token as a single contiguous token copy.
    Returns the load-order permutation (``logical[i] = stored[perm[i]]``),
    or ``None`` when the stored order equals the logical order (head-shard
    states whose Q segment is a single typed row).
    """
    v_elems = n_v * d_v
    qk_elems = conv_dim - v_elems
    q_elems = qk_elems // 2
    if 2 * q_elems != qk_elems or q_elems % lanes:
        raise NotImplementedError(
            "QK pair-blocked conv layout requires lane-aligned equal Q/K "
            f"segments: conv_dim={conv_dim}, n_v*d_v={v_elems}, "
            f"lanes={lanes}"
        )
    if v_elems % lanes:
        raise NotImplementedError(
            f"QK pair-blocked conv layout requires lane-aligned V: {v_elems} % {lanes}"
        )
    q_rows = q_elems // lanes
    v_rows = v_elems // lanes
    rows_per_tap = conv_dim // lanes
    if q_rows <= 1:
        return None
    perm: list[int] = []
    for tap in range(taps):
        base = tap * rows_per_tap
        perm.extend(base + 2 * i for i in range(q_rows))
        perm.extend(base + 2 * i + 1 for i in range(q_rows))
        perm.extend(base + 2 * q_rows + j for j in range(v_rows))
    return tuple(perm)


def v3_state_source(
    pool,
    *,
    split: int,
    ssm_ntok: int,
    conv_tok0: int,
    conv_ntok: int,
    conv_dim: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    recurrent_state_dtype: jnp.dtype = jnp.float32,
    qk_pair_layout: bool = False,
    conv_state_dtype: jnp.dtype = jnp.bfloat16,
) -> config.StateSourcePlan:
    """Static copy-plan letting the fused GDN V3 kernel stream the mamba
    state regions directly between this pool and its double-buffered
    pipeline — the exact bytes ``gather_region``/``scatter_region`` move
    for these regions, without the external round trip.

    The SSM region is the whole kernel blocks at the start of the manager
    window (one contiguous DMA per slot), viewed in the configured storage
    dtype. FP32 uses a ``d_v``-wide typed view; BF16 preserves the raw pool
    ref's full minormost width. The Conv region is ``conv_ntok`` token rows
    inside a single kernel block, viewed in ``conv_state_dtype`` with the tail rows
    zero-padded.
    Arguments mirror the region geometry computed by the pooled GDN
    caller; ``pool`` contributes only its shape and dtype.

    The plan describes one checkpoint. With speculative decoding the
    caller names one source block per checkpoint
    (``MetadataRef.s_idx_to_ckpt_indices``), matching how vLLM's reference
    GDN kernel indexes ``ssm_state_indices[seq, ckpt]``; every checkpoint
    reuses this same intra-block geometry, so the block size stays
    independent of ``num_speculative_tokens``.
    """
    block_size, _, lanes = _pool_geometry(pool)
    tok_bytes = pool_token_bytes(pool)
    recurrent_state_dtype = jnp.dtype(recurrent_state_dtype)

    ssm_rows = n_v * d_k
    ssm_elements = ssm_rows * d_v
    # The SSM region may extend past the typed state bytes when the state does
    # not divide the pool token row; the kernel truncates loads to rows_used
    # and zero-fills the padding rows on store.
    assert ssm_ntok * tok_bytes >= (ssm_elements * recurrent_state_dtype.itemsize), (
        ssm_ntok,
        tok_bytes,
        n_v,
        d_k,
        d_v,
        recurrent_state_dtype,
    )
    if ssm_ntok % block_size == 0:
        ssm_nblocks, ssm_nrows = ssm_ntok // block_size, block_size
    else:
        assert ssm_ntok < block_size, (ssm_ntok, block_size)
        ssm_nblocks, ssm_nrows = 1, ssm_ntok
    assert lanes % d_v == 0, (lanes, d_v)
    ssm_lane_split = lanes // d_v
    ssm_out_lanes = d_v

    # load_state_region treats these rows as one contiguous element stream and
    # reshapes it to (n_v, d_k, d_v); store_state_region reverses that reshape.
    assert ssm_elements % ssm_out_lanes == 0, (ssm_elements, ssm_out_lanes)
    ssm_rows_used = ssm_elements // ssm_out_lanes

    # Reading/writing VMEM directly with a bfloat16 view_dtype emits excessive
    # vrot instructions between VREG and VMEM. View BF16 recurrent state as
    # uint32 in VMEM (2 bfloat16 elements per uint32) and bitcast in VREG,
    # which halves the number of view rows used.
    if recurrent_state_dtype == jnp.bfloat16:
        ssm_view_dtype = jnp.dtype(jnp.uint32)
        assert ssm_rows_used % 2 == 0, ssm_rows_used
        ssm_rows_used //= 2
    else:
        ssm_view_dtype = recurrent_state_dtype

    ssm = config.StateRegion(
        kb0=0,
        nblocks=ssm_nblocks,
        row0=0,
        nrows=ssm_nrows,
        view_dtype=ssm_view_dtype,
        lane_split=ssm_lane_split,
        rows_used=ssm_rows_used,
    )

    kb0, row0 = divmod(conv_tok0, block_size)
    if row0 + conv_ntok > block_size:
        raise NotImplementedError(
            "conv region partially straddles kernel blocks: "
            f"tok0={conv_tok0} ntok={conv_ntok} kernel_block={block_size}"
        )
    assert conv_tok0 + conv_ntok <= split * block_size, (
        conv_tok0,
        conv_ntok,
        split,
        block_size,
    )
    assert (kernel_size - 1) * conv_dim % lanes == 0, (kernel_size, conv_dim, lanes)
    # The kernel regroups conv rows into (kernel_size - 1, conv_dim) via
    # 128-aligned lane concat, which needs whole rows per conv row.
    assert conv_dim % lanes == 0, (conv_dim, lanes)
    conv_dtype = jnp.dtype(conv_state_dtype)
    if conv_dtype not in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float32)):
        raise ValueError(f"unsupported pooled conv state dtype: {conv_dtype}")
    conv_rows = (kernel_size - 1) * conv_dim // lanes
    assert conv_rows * conv_dtype.itemsize * lanes <= conv_ntok * tok_bytes, (
        conv_rows,
        conv_ntok,
        tok_bytes,
    )
    conv_rows_perm = None
    if qk_pair_layout:
        conv_rows_perm = _conv_qk_pair_rows_perm(
            kernel_size - 1, conv_dim, n_v, d_v, lanes
        )
    conv = config.StateRegion(
        kb0=kb0,
        nblocks=1,
        row0=row0,
        nrows=conv_ntok,
        view_dtype=conv_dtype,
        lane_split=1,
        rows_used=conv_rows,
        rows_perm=conv_rows_perm,
    )
    return config.StateSourcePlan(
        stride=split,
        conv=conv,
        recurrent=ssm,
    )
