# SPDX-License-Identifier: Apache-2.0
"""Decode-side staging and repack for a TP1 pipeline producer feeding a TP>1 decode.

Qwen3.5's prefill shape (PP8 x TP1 stages) feeds decodes of several TP sizes
(TP8 + DCP, DP x TP2). A TP1 stage's pages hold both KV heads of a token row,
``[K0 V0 K1 V1]``, at the manager block size its TP1 GDN state needs; a TP-T
decode rank holds one head, ``[K_h V_h 0 0]``, at its own (smaller) block size,
and only its slice of the GDN heads. A head's bytes are 128-byte chunks strided
through every row, far too fine for byte spans, so the transfer stays the
proven whole-page copy and the reshape happens on the decode:

1. every decode rank holds a staging carrier in the producer's TP1 geometry,
   registered as its Raiden destination under the producer's per-layer tags
   (the producer broadcasts each span to every decode unit);
2. when a load lands, the runner thread repacks the staged pages into the
   rank's own pool pages in one donated program: main KV takes the rank's head
   slot, compressed keys, rings and PLE state are re-blocked or copied, and GDN
   state takes the rank's SSM heads and conv channels;
3. the staging pages are freed.

Pool geometry (``pool_layout.pooled_cache_specs``): every owner shares one raw
carrier of ``B``-row manager pages, 1024-byte rows ``[4 sublanes, 256 lanes]``
of FP8 (``(pages * B // 128, 128, 1, 4, 256)``). Typed state reads go through
Mosaic bitcasts, which pack the 4 sublanes at a lane into one 32-bit word, low
byte first (``kernels/typed_ldst.py``, ``megakernel_runtime/qwen38_gdn_state``).
"""

import dataclasses
import math
from collections.abc import Sequence

import torch

KERNEL_ROWS = 128
ROW_SUBLANES = 4
ROW_LANES = 256


@dataclasses.dataclass(frozen=True)
class GdnGeometry:
    """One rank's view of a GDN state page."""

    ssm_tokens: int
    conv_tokens: int
    v_heads: int
    k_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_width: int  # kernel_size - 1
    ssm_itemsize: int
    conv_itemsize: int

    @property
    def conv_dim(self) -> int:
        return 2 * self.k_heads * self.head_k_dim + self.v_heads * self.head_v_dim


@dataclasses.dataclass(frozen=True)
class RankSlice:
    """What a decode rank keeps of a TP1 producer's state."""

    tp_size: int
    tp_rank: int
    num_kv_heads: int
    main_dcp: int  # VLLM_TPU_QSA_MAIN_KV_DCP pair size (1 = off)

    @property
    def kv_head(self) -> int:
        # vLLM replicates the KV heads over TP > num_kv_heads in rank blocks.
        return self.tp_rank * self.num_kv_heads // self.tp_size

    @property
    def dcp_rank(self) -> int:
        return self.tp_rank % self.main_dcp


def owner_kind(name: str) -> str:
    if name.endswith(".compressed_k"):
        return "compressed"
    if name.endswith(".raw_ring"):
        return "ring"
    if name.endswith(".linear_attn"):
        return "gdn"
    if name.endswith(".ple"):
        return "ple"
    if name.endswith(".self_attn"):
        return "main"
    raise ValueError(f"Qwen PD staging has no repack for cache owner {name}")


def bucket(n: int, floor: int = 1024) -> int:
    """Padded index length: few distinct compiled shapes."""
    return max(floor, 1 << max(0, n - 1).bit_length())


def _pad(values: list[int], size: int) -> list[int]:
    # Repeat the last pair: duplicate writes of identical values are no-ops.
    return values + [values[-1]] * (size - len(values))


def main_rows(
    num_tokens: int,
    src_pages: Sequence[int],
    src_block: int,
    dst_pages: Sequence[int],
    dst_block: int,
    rank: RankSlice,
) -> tuple[list[int], list[int]]:
    """(source row, destination row) of every token this rank keeps.

    With main-KV DCP a destination table entry covers ``D * B`` tokens and pair
    rank ``(t // B) % D`` stores token ``t`` at row ``t % B`` of the entry's page
    (``qsa_rpa.virtual_page_metadata``).
    """
    src, dst = [], []
    d = rank.main_dcp
    for t in range(num_tokens):
        if d > 1 and (t // dst_block) % d != rank.dcp_rank:
            continue
        src.append(src_pages[t // src_block] * src_block + t % src_block)
        dst.append(dst_pages[t // (d * dst_block)] * dst_block + t % dst_block)
    return src, dst


def compressed_rows(
    num_tokens: int,
    ratio: int,
    src_pages: Sequence[int],
    src_block: int,
    dst_pages: Sequence[int],
    dst_block: int,
) -> tuple[list[int], list[int]]:
    """Rows ("pairs") holding the compressed keys of the first ``num_tokens``.

    Pair ``p`` of a page holds compressed groups ``2p`` and ``2p + 1``, i.e.
    tokens ``[2 p ratio, 2 (p + 1) ratio)`` of the page's span
    (``pooled_qsa.compressed_pairs``); the content does not depend on TP.
    """
    span = 2 * ratio
    pairs = math.ceil(math.ceil(num_tokens / ratio) / 2)
    src, dst = [], []
    for q in range(pairs):
        t = q * span
        src.append(src_pages[t // src_block] * src_block + (t % src_block) // span)
        dst.append(dst_pages[t // dst_block] * dst_block + (t % dst_block) // span)
    return src, dst


def page_prefix_rows(
    src_page: int, src_block: int, dst_page: int, dst_block: int, rows: int
) -> tuple[list[int], list[int]]:
    """The first ``rows`` rows of a per-request page (ring + tags, PLE state)."""
    return (
        [src_page * src_block + r for r in range(rows)],
        [dst_page * dst_block + r for r in range(rows)],
    )


def conv_channels(src: GdnGeometry, dst: GdnGeometry, rank: RankSlice) -> list[int]:
    """Producer conv channels of the rank's ``[q | k | v]`` (segment-major)."""
    key = src.k_heads * src.head_k_dim
    k_local = dst.k_heads * dst.head_k_dim
    v_local = dst.v_heads * dst.head_v_dim
    r = rank.tp_rank
    q = list(range(r * k_local, (r + 1) * k_local))
    k = [key + c for c in q]
    v = list(range(2 * key + r * v_local, 2 * key + (r + 1) * v_local))
    return q + k + v


# ---------------------------------------------------------------------------
# The device program (JAX, wrapped as a donated torch op on TPU)
# ---------------------------------------------------------------------------


def _typed(rows, itemsize: int):
    """``[R, 4, 256]`` uint8 rows -> the element stream of a Mosaic typed view."""
    import jax
    import jax.numpy as jnp

    r = rows.shape[0]
    if itemsize == 4:
        words = jnp.transpose(rows, (0, 2, 1))  # [R, 256, 4]: one word a lane
        return jax.lax.bitcast_convert_type(words, jnp.uint32).reshape(r * ROW_LANES)
    if itemsize == 2:
        halves = jnp.transpose(rows.reshape(r, 2, 2, ROW_LANES), (0, 1, 3, 2))
        return jax.lax.bitcast_convert_type(halves, jnp.uint16).reshape(
            r * 2 * ROW_LANES
        )
    raise ValueError(f"unsupported state itemsize {itemsize}")


def _untyped(elems, rows: int, itemsize: int):
    """Inverse of ``_typed``: element stream -> ``[rows, 4, 256]`` uint8."""
    import jax
    import jax.numpy as jnp

    per_row = ROW_SUBLANES * ROW_LANES // itemsize
    elems = jnp.pad(elems, (0, rows * per_row - elems.shape[0]))
    if itemsize == 4:
        words = elems.reshape(rows, ROW_LANES)
        return jnp.transpose(
            jax.lax.bitcast_convert_type(words, jnp.uint8), (0, 2, 1)
        )
    halves = elems.reshape(rows, 2, ROW_LANES)
    raw = jax.lax.bitcast_convert_type(halves, jnp.uint8)  # [rows, 2, 256, 2]
    return jnp.transpose(raw, (0, 1, 3, 2)).reshape(rows, ROW_SUBLANES, ROW_LANES)


def make_repack(
    *,
    src_block: int,
    dst_block: int,
    src_gdn: GdnGeometry | None,
    dst_gdn: GdnGeometry | None,
    kv_head: int,
    v_head_start: int,
    channels: Sequence[int],
):
    """``fn(pool, staging, main_src, main_dst, copy_src, copy_dst, gdn_src, gdn_dst)``.

    Row indices are flat (``page * B + row``) into the staging carrier
    (``src``) and the pool (``dst``); ``gdn_*`` are page ids, one per GDN
    layer, in the same order. Returns the updated pool (donated).
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    chan = np.asarray(channels, dtype=np.int32)

    def rows_of(carrier, flat):
        return carrier[flat // KERNEL_ROWS, flat % KERNEL_ROWS, 0]

    def put(carrier, flat, values):
        return carrier.at[flat // KERNEL_ROWS, flat % KERNEL_ROWS, 0].set(values)

    def gdn_page(page, s: GdnGeometry, d: GdnGeometry):
        """One producer GDN page -> this rank's page rows ``[ssm + conv]``."""
        ssm = _typed(page[: s.ssm_tokens], s.ssm_itemsize)
        ssm = ssm[: s.v_heads * s.head_k_dim * s.head_v_dim].reshape(s.v_heads, -1)
        ssm = ssm[v_head_start : v_head_start + d.v_heads].reshape(-1)
        conv = _typed(page[s.ssm_tokens : s.ssm_tokens + s.conv_tokens], s.conv_itemsize)
        conv = conv[: s.conv_width * s.conv_dim].reshape(s.conv_width, s.conv_dim)
        conv = conv[:, jnp.asarray(chan)].reshape(-1)
        return jnp.concatenate(
            [
                _untyped(ssm, d.ssm_tokens, d.ssm_itemsize),
                _untyped(conv, d.conv_tokens, d.conv_itemsize),
            ]
        )

    def fn(pool, staging, main_src, main_dst, copy_src, copy_dst, gdn_src, gdn_dst):
        dtype = pool.dtype
        out = jax.lax.bitcast_convert_type(pool, jnp.uint8)
        stage = jax.lax.bitcast_convert_type(staging, jnp.uint8)
        # Main KV: the rank's head pair to slot 0, the other slots zero.
        main = rows_of(stage, main_src)
        pair = main[:, 2 * kv_head : 2 * kv_head + 2]
        out = put(out, main_dst, jnp.concatenate([pair, jnp.zeros_like(pair)], axis=1))
        # Compressed keys, rings + tags, PLE state: bytes unchanged.
        out = put(out, copy_dst, rows_of(stage, copy_src))
        if src_gdn is not None and dst_gdn is not None:
            s, d = src_gdn, dst_gdn
            s_rows = s.ssm_tokens + s.conv_tokens
            d_rows = d.ssm_tokens + d.conv_tokens
            src_flat = (gdn_src[:, None] * src_block + jnp.arange(s_rows)).reshape(-1)
            region = rows_of(stage, src_flat).reshape(-1, s_rows, ROW_SUBLANES, ROW_LANES)
            pages = jax.vmap(lambda page: gdn_page(page, s, d))(region)
            dst_flat = (gdn_dst[:, None] * dst_block + jnp.arange(d_rows)).reshape(-1)
            out = put(out, dst_flat, pages.reshape(-1, ROW_SUBLANES, ROW_LANES))
        return jax.lax.bitcast_convert_type(out, dtype)

    return fn
