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

import copy
import dataclasses
import math
from collections.abc import Sequence

import numpy as np
import torch

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


def without_speculation(cfg):
    """A shallow copy of ``cfg`` with speculative decoding off (a peer's plan)."""
    if getattr(cfg, "speculative_config", None) is None:
        return cfg
    out = copy.copy(cfg)
    object.__setattr__(out, "speculative_config", None)
    return out


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


# Rows per repack program, by kind. One program for a whole request gathers
# and scatters every row at once: a 190k-token request at TP2 is millions of
# 1 KiB rows, multi-GB temporaries beside a pool sized to fill HBM. Fixed
# chunks bound the footprint and compile one shape. The DMA writer keeps a few
# int32 scalars per row in SMEM (1 MiB; ~11.5 B/row measured), so a program
# stays near 40k rows: 16k main, 16k copied, 4 GDN pages (~1.6k rows each).
REPACK_CHUNK_ROWS = 1 << 14
# Rows per scatter tile. Every run of contiguous destination rows starts on a
# tile (each layer's segment is padded to a multiple), so a tile is one DMA:
# main KV runs a page (512 rows), compressed keys 64 rows per decode page.
REPACK_TILE = 64
REPACK_GDN_PAGES = 4


def bucket(n: int, floor: int = 1024) -> int:
    """Padded index length: few distinct compiled shapes."""
    return max(floor, 1 << max(0, n - 1).bit_length())


def repack_chunks(
    main_s, main_d, copy_s, copy_d, gdn_s, gdn_d,
    size=REPACK_CHUNK_ROWS, gdn_pages=REPACK_GDN_PAGES,
):
    """Padded index lists per repack program, each of one fixed shape.

    A chunk with nothing left of a kind rewrites the null page (row 0, page 0
    on both sides) in its place.
    """
    chunks = max(
        1, -(-len(main_s) // size), -(-len(copy_s) // size), -(-len(gdn_s) // gdn_pages)
    )
    for i in range(chunks):
        part = slice(i * size, (i + 1) * size)
        pages = slice(i * gdn_pages, (i + 1) * gdn_pages)
        yield (
            _pad(main_s[part], size), _pad(main_d[part], size),
            _pad(copy_s[part], size), _pad(copy_d[part], size),
            _pad(gdn_s[pages], gdn_pages), _pad(gdn_d[pages], gdn_pages),
        )


def _pad(values, size: int):
    # Repeat the last pair: duplicate writes of identical values are no-ops.
    # A rank can own no rows (a DCP rank of a request shorter than its page);
    # row 0 is the null page's on both sides, which nothing reads.
    if isinstance(values, np.ndarray):
        if values.size == 0:
            values = np.zeros(1, dtype=np.int64)
        return np.concatenate(
            (values, np.full(size - len(values), values[-1], dtype=values.dtype))
        )
    values = values or [0]
    return values + [values[-1]] * (size - len(values))


def main_rows(
    num_tokens: int,
    src_pages: Sequence[int],
    src_block: int,
    dst_pages: Sequence[int],
    dst_block: int,
    rank: RankSlice,
    start: int = 0,
    src_first: int = 0,
) -> tuple[list[int], list[int]]:
    """(source row, destination row) of every token this rank keeps.

    With main-KV DCP a destination table entry covers ``D * B`` tokens and pair
    rank ``(t // B) % D`` stores token ``t`` at row ``t % B`` of the entry's page
    (``qsa_rpa.virtual_page_metadata``). A prefix-aware load repacks only tokens
    from ``start`` on: ``src_pages`` then begins at source page ``src_first``
    (the staged suffix), ``dst_pages`` is the full destination table.
    """
    src, dst = main_rows_np(
        num_tokens, src_pages, src_block, dst_pages, dst_block, rank, start, src_first
    )
    return src.tolist(), dst.tolist()


def main_rows_np(
    num_tokens, src_pages, src_block, dst_pages, dst_block, rank, start=0, src_first=0,
    src_pcp=None,
) -> tuple[np.ndarray, np.ndarray]:
    """``main_rows`` as int64 arrays, vectorized (a 100k-token load has a
    row per token per layer: the Python loop took ~0.7 s per request on the
    decode's worker thread, stalling its steps).

    ``src_pcp = (P, I)``: a PCP producer split this owner's main KV over its P
    ranks in units of I tokens (``qsa_rpa.dcp_owner_row``); ``src_pages`` is
    then one staged page list per producer rank, a page per P-page entry, and
    ``src_first`` counts entries."""
    t = np.arange(start, num_tokens, dtype=np.int64)
    d = rank.main_dcp
    if d > 1:
        t = t[(t // dst_block) % d == rank.dcp_rank]
    dp = np.asarray(dst_pages, dtype=np.int64)
    if src_pcp is not None:
        p, interleave = src_pcp
        span = p * src_block
        offset = t % span
        owner = (offset // interleave) % p
        row = offset // (interleave * p) * interleave + offset % interleave
        by_rank = np.asarray(src_pages, dtype=np.int64)  # [P, entries staged]
        src = by_rank[owner, t // span - src_first] * src_block + row
    else:
        sp = np.asarray(src_pages, dtype=np.int64)
        src = sp[t // src_block - src_first] * src_block + t % src_block
    dst = dp[t // (d * dst_block)] * dst_block + t % dst_block
    return src, dst


def compressed_rows(
    num_tokens: int,
    ratio: int,
    src_pages: Sequence[int],
    src_block: int,
    dst_pages: Sequence[int],
    dst_block: int,
    start: int = 0,
    src_first: int = 0,
    dst_offset: int = 0,
) -> tuple[list[int], list[int]]:
    """Rows ("pairs") holding the compressed keys of the first ``num_tokens``.

    ``dst_offset`` rows shift every destination row within its page: a
    packed member's slot (``compressed_destinations``).

    Pair ``p`` of a page holds compressed groups ``2p`` and ``2p + 1``, i.e.
    tokens ``[2 p ratio, 2 (p + 1) ratio)`` of the page's span
    (``pooled_qsa.compressed_pairs``); the content does not depend on TP. A
    prefix-aware load starts at the pair of token ``start`` (a pair boundary),
    with ``src_pages`` beginning at source page ``src_first``.
    """
    src, dst = compressed_rows_np(
        num_tokens, ratio, src_pages, src_block, dst_pages, dst_block,
        start, src_first, dst_offset,
    )
    return src.tolist(), dst.tolist()


def compressed_rows_np(
    num_tokens, ratio, src_pages, src_block, dst_pages, dst_block,
    start=0, src_first=0, dst_offset=0, src_page_rows=None,
) -> tuple[np.ndarray, np.ndarray]:
    """``compressed_rows`` as int64 arrays, vectorized. ``src_page_rows``
    (default ``src_block``): rows of a source page, when one source page holds
    the keys of ``src_block`` tokens in fewer rows (merged producer keys)."""
    span = 2 * ratio
    if start % span:
        raise ValueError(f"A repack must start on a compressed pair ({span} tokens)")
    pairs = math.ceil(math.ceil(num_tokens / ratio) / 2)
    t = np.arange(start // span, pairs, dtype=np.int64) * span
    sp = np.asarray(src_pages, dtype=np.int64)
    dp = np.asarray(dst_pages, dtype=np.int64)
    page_rows = src_block if src_page_rows is None else src_page_rows
    src = sp[t // src_block - src_first] * page_rows + (t % src_block) // span
    dst = dp[t // dst_block] * dst_block + dst_offset + (t % dst_block) // span
    return src, dst


def compressed_rows_split_np(
    num_tokens, ratio, src_pages_by_rank, src_block, dst_pages, dst_block,
    start=0, src_first=0, dst_offset=0,
) -> tuple[np.ndarray, np.ndarray]:
    """``compressed_rows_np`` for keys split over ``P`` producer ranks by page
    owner: a main-KV ID spans ``P`` pages of ``src_block`` tokens and rank
    ``(t // src_block) % P`` holds page ``t // src_block``'s keys, at the start
    of that rank's staged page for the ID."""
    span = 2 * ratio
    if start % span:
        raise ValueError(f"A repack must start on a compressed pair ({span} tokens)")
    p = len(src_pages_by_rank)
    pairs = math.ceil(math.ceil(num_tokens / ratio) / 2)
    t = np.arange(start // span, pairs, dtype=np.int64) * span
    owner = (t // src_block) % p
    entry = t // (p * src_block) - src_first
    sp = np.stack([np.asarray(pages, dtype=np.int64) for pages in src_pages_by_rank])
    src = sp[owner, entry] * src_block + (t % src_block) // span
    dp = np.asarray(dst_pages, dtype=np.int64)
    dst = dp[t // dst_block] * dst_block + dst_offset + (t % dst_block) // span
    return src, dst


def align_rows(rows: np.ndarray, tile: int = REPACK_TILE) -> np.ndarray:
    """Pad a row segment to a multiple of ``tile`` by repeating its last row.

    Applied alike to a segment's source and destination rows, the repeats
    rewrite the last row with its own bytes (a no-op), and the next segment
    then starts on a tile, so its contiguous runs fill whole tiles.
    """
    pad = -len(rows) % tile
    if not pad:
        return rows
    return np.concatenate((rows, np.full(pad, rows[-1], dtype=rows.dtype)))


def compressed_destinations(
    layout, dst_owner: dict[str, int], dst_block: int, ratio: int
) -> dict[str, tuple[int, int]]:
    """Each compressed-key owner's destination group and in-page row offset.

    Unpacked, every owner has its own group at offset 0. Packed
    (``layout.compressed_slots`` > 1), member ``s`` of a group named after
    its first member stores its ``dst_block // (2 * ratio)`` pair rows at
    kernel pages ``[s * main_kernel_pages, (s + 1) * main_kernel_pages)``,
    i.e. rows ``[s * member_rows, (s + 1) * member_rows)`` of the page.
    """
    member_rows = dst_block // (2 * ratio)
    out: dict[str, tuple[int, int]] = {}
    if layout.compressed_owners:
        if member_rows * layout.compressed_slots > dst_block:
            raise ValueError("Packed compressed members overflow the page")
        for prefix, owner, slot in layout.compressed_owners:
            out[prefix + ".compressed_k"] = (dst_owner[owner], slot * member_rows)
    else:
        for name, gi in dst_owner.items():
            if owner_kind(name) == "compressed":
                out[name] = (gi, 0)
    return out


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
    write=None,
):
    """``fn(pool, staging, main_src, main_dst, copy_src, copy_dst, gdn_src, gdn_dst)``.

    Row indices are flat (``page * B + row``) into the staging carrier
    (``src``) and the pool (``dst``); ``gdn_*`` are page ids, one per GDN
    layer, in the same order. Returns the updated pool.

    ``fn.rows(staging, main_src, copy_src, gdn_src)`` returns only the rows to
    write (uint8 ``[n, 4, 256]`` each).

    ``write(pool, rows, page, logical, valid)`` stores ``rows`` at
    ``pool[page, logical % pool.shape[1]]``; by default the forward's aliased
    Pallas DMA scatter, which writes the pool in place (an XLA scatter on the
    pool reshaped to rows copied the whole pool).
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    chan = np.asarray(channels, dtype=np.int32)

    # Rows are addressed flat (page * B + row), whatever the carrier's
    # kernel-page shape (the pools hold one kernel page per manager page).
    def rows_of(rows, flat):
        return rows[flat]

    if write is None:
        import functools

        from vllm_torchtpu.kernels.qwen4_exp.pooled_qsa import (
            _write_main_cache_universal,
        )

        write = functools.partial(
            _write_main_cache_universal, tile=REPACK_TILE, overlap_edges=True
        )

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

    has_gdn = src_gdn is not None and dst_gdn is not None
    d_rows = dst_gdn.ssm_tokens + dst_gdn.conv_tokens if has_gdn else 0

    def rows(
        staging: jax.Array, main_src: jax.Array, copy_src: jax.Array, gdn_src: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        # Gather rows in the carrier's own dtype and bitcast only those: a
        # bitcast of the whole carrier materialized a copy of it per call.
        stage = staging.reshape(-1, ROW_SUBLANES, ROW_LANES)

        def gather(flat):
            return jax.lax.bitcast_convert_type(rows_of(stage, flat), jnp.uint8)

        # Main KV: the rank's head pair to slot 0, the other slots zero.
        pair = gather(main_src)[:, 2 * kv_head : 2 * kv_head + 2]
        main = jnp.concatenate([pair, jnp.zeros_like(pair)], axis=1)
        # Compressed keys, rings + tags, PLE state: bytes unchanged.
        copies = gather(copy_src)
        if has_gdn:
            s, d = src_gdn, dst_gdn
            s_rows = s.ssm_tokens + s.conv_tokens
            src_flat = (gdn_src[:, None] * src_block + jnp.arange(s_rows)).reshape(-1)
            region = gather(src_flat).reshape(-1, s_rows, ROW_SUBLANES, ROW_LANES)
            gdn = jax.vmap(lambda page: gdn_page(page, s, d))(region)
            gdn = gdn.reshape(-1, ROW_SUBLANES, ROW_LANES)
        else:
            gdn = jnp.zeros((0, ROW_SUBLANES, ROW_LANES), jnp.uint8)
        return main, copies, gdn

    def fn(
        pool: jax.Array,
        staging: jax.Array,
        main_src: jax.Array,
        main_dst: jax.Array,
        copy_src: jax.Array,
        copy_dst: jax.Array,
        gdn_src: jax.Array,
        gdn_dst: jax.Array,
    ) -> jax.Array:
        main, copies, gdn = rows(staging, main_src, copy_src, gdn_src)
        values = [main, copies]
        flat = [main_dst, copy_dst]
        if has_gdn:
            values.append(gdn)
            flat.append((gdn_dst[:, None] * dst_block + jnp.arange(d_rows)).reshape(-1))
        values = jnp.concatenate(values)
        flat = jnp.concatenate(flat).astype(jnp.int32)
        valid = jnp.ones(flat.shape, jnp.bool_)
        # The writer takes whole tiles; the tail is masked out.
        tail = -len(flat) % REPACK_TILE
        values = jnp.pad(values, ((0, tail), (0, 0), (0, 0)))
        flat = jnp.pad(flat, (0, tail))
        valid = jnp.pad(valid, (0, tail))
        kernel_rows = pool.shape[1]
        values = jax.lax.bitcast_convert_type(values, pool.dtype).reshape(
            -1, *pool.shape[2:]
        )
        return write(pool, values, flat // kernel_rows, flat, valid)

    fn.rows = rows
    return fn


# ---------------------------------------------------------------------------
# The consumer: staging carrier, allocator, manifest and the repack call
# ---------------------------------------------------------------------------


def gdn_geometry(cfg, tp: int, layout_token_bytes: int) -> GdnGeometry:
    """A rank's GDN state page at ``tp`` (``pool_layout.pooled_cache_specs``)."""
    from vllm_torchtpu.gdn_pool_layout import derive_pooled_gdn_state_layout

    from .config import gdn_cache_spec
    from .pool_layout import POOLED_GDN_CONV_STATE_DTYPE

    t = cfg.model_config.hf_text_config
    spec = gdn_cache_spec(cfg, tp)
    conv_item = torch.empty((), dtype=POOLED_GDN_CONV_STATE_DTYPE).element_size()
    ssm_item = torch.empty((), dtype=spec.dtypes[1]).element_size()
    state = derive_pooled_gdn_state_layout(
        ssm_bytes=math.prod(spec.shapes[1]) * ssm_item,
        conv_bytes=math.prod(spec.shapes[0]) * conv_item,
        token_bytes=layout_token_bytes,
    )
    return GdnGeometry(
        ssm_tokens=state.ssm_tokens,
        conv_tokens=state.conv_tokens,
        v_heads=int(t.linear_num_value_heads) // tp,
        k_heads=int(t.linear_num_key_heads) // tp,
        head_k_dim=int(t.linear_key_head_dim),
        head_v_dim=int(t.linear_value_head_dim),
        conv_width=int(t.linear_conv_kernel_dim) - 1,
        ssm_itemsize=ssm_item,
        conv_itemsize=conv_item,
    )


class StagedConsumer:
    """One TP-T decode rank's staging carrier for TP1 pipeline-producer pages."""

    def __init__(self, cfg, decode_layout, decode_groups, pool, rank: RankSlice, staging_bytes: int):
        from vllm.v1.core.kv_cache_utils import get_kv_cache_groups

        from vllm_torchtpu.distributed.kv_transfer.raiden.pool_manifest import (
            BINDING_ALIASED_RAW,
            PoolEntry,
            PoolManifest,
            RegionSpec,
        )

        from .pd import TransferExtent, build_transfer_groups
        from .pool_layout import pack_main_kv_enabled, pooled_cache_specs

        if pack_main_kv_enabled():
            raise ValueError(
                "A staged Qwen PD decode needs QWEN4_EXP_PACK_MAIN_KV=0 (the "
                "pipeline producer's layout)"
            )
        # Packed compressed keys (QWEN4_EXP_PACK_COMPRESSED_K=1) are this
        # decode's own layout: the repack writes each layer's pairs at its
        # slot of the owner group's page. The producer never packs.
        self.cfg, self.rank, self.pool = cfg, rank, pool
        self.dst_layout = decode_layout
        self.dst_groups = decode_groups
        # The producer: TP1 stages, no main-KV DCP, unpacked compressed keys,
        # and no MTP draft: a decode-only draft (this engine's
        # speculative_config) adds draft owners and speculative state blocks
        # the producer's plan does not have.
        # A PCP producer (QWEN4_EXP_PD_PRODUCER_PCP=P, the prefill leg's PCP
        # size) splits its main KV over its P ranks in pcp_interleave() units.
        import os

        from .pool_layout import pcp_interleave

        self.src_pcp = int(os.environ.get("QWEN4_EXP_PD_PRODUCER_PCP", "1"))
        # QWEN4_EXP_PD_PRODUCER_MERGED=1: the producer runs
        # QWEN4_EXP_MERGE_COMPRESSED_K=1, so its keys ride the main-KV IDs in
        # their own pools ("<layer>.self_attn.compressed_k" on the main group).
        self.src_merged = os.environ.get("QWEN4_EXP_PD_PRODUCER_MERGED", "0") == "1"
        # QWEN4_EXP_PD_PRODUCER_SPLIT_KEYS=1: the producer also runs
        # QWEN4_EXP_PCP_SPLIT_COMPRESSED_K=1 (each PCP rank sends the keys of
        # the pages it owns, "<layer>.compressed_k.pcp<r>").
        self.src_split_keys = self.src_merged and (
            os.environ.get("QWEN4_EXP_PD_PRODUCER_SPLIT_KEYS", "0") == "1"
        )
        self.src_layout, src_specs = pooled_cache_specs(
            without_speculation(cfg),
            tp=1,
            main_dcp=self.src_pcp,
            pack_compressed=False,
            merge_compressed=self.src_merged,
            split_compressed=self.src_split_keys,
            main_interleave=pcp_interleave() if self.src_pcp > 1 else None,
        )
        self._src_split = (
            set(self.src_layout.main_dcp_owners) if self.src_pcp > 1 else set()
        )
        from .config import ensure_kv_group_size_rule

        ensure_kv_group_size_rule()
        self.src_groups = build_transfer_groups(get_kv_cache_groups(cfg, dict(src_specs)))
        self.src_block, self.dst_block = self.src_layout.block_size, decode_layout.block_size
        self.ratio = int(cfg.model_config.hf_text_config.indexer_compress_ratio)
        self.pages = max(2, staging_bytes // self.src_layout.page_bytes)
        self.staging = torch.zeros(
            # The producer pools' carrier shape, one kernel page per manager
            # page: the layout fingerprint compares it.
            (self.pages, self.src_block, 1, ROW_SUBLANES, ROW_LANES),
            dtype=pool.dtype,
            device=pool.device,
        )
        self.free = torch.ones(self.pages, dtype=torch.int32)
        self.free[0] = 0  # the null page
        # uuid -> staging pages per producer cache owner (manifest order). A
        # producer group of several layers shares block IDs across them, each
        # layer's page in its own slot tensor, so each owner stages its own.
        self.reservations: dict[int, dict[str, list[int]]] = {}
        # uuid -> (decode-local prefix tokens, first staged source page per group)
        self.skips: dict[int, tuple[int, list[int]]] = {}
        from .pd_manifest import compressed_live_bytes, page_regions, pool_tag

        # The producer sends a compressed-key page's keys only; the carrier
        # declares the same region so both pool layouts agree.
        compressed_live = compressed_live_bytes(self.src_layout)
        # Merged producer keys: one block of `key_stride` bytes per main-KV ID.
        # Each lands at the start of its own staging page, so its block id is
        # the page id: raiden bounds every block id by every pool's num_blocks.
        self._key_pools: dict[str, int] = {}
        key_entries = []
        if self.src_merged:
            span = self.src_layout.merged_key_span
            key_stride = span * self.src_layout.page_bytes // self.src_layout.compressed_columns_per_page
            self.key_stride = key_stride
            for group in self.src_groups:
                for name in group.layer_names:
                    if owner_kind(name) != "main":
                        continue
                    key = name + ".compressed_k"
                    ranks = range(self.src_pcp) if self.src_split_keys else (None,)
                    for r in ranks:
                        pool_name = key if r is None else f"{key}.pcp{r}"
                        self._key_pools[pool_name] = group.group_index
                        key_entries.append(
                            PoolEntry(
                                tag=pool_tag(
                                    group.group_index, key,
                                    pcp_rank=r or 0, split=r is not None,
                                ),
                                layer_name=pool_name,
                                storage_index=0,
                                base_offset_bytes=0,
                                block_stride_bytes=self.src_layout.page_bytes,
                                num_blocks=self.pages,
                                regions=(
                                    RegionSpec("keys", 0, self.src_layout.page_bytes, key_stride, 1),
                                ),
                                dtype_tag="qwen4-whole-fp8-page-v1",
                            )
                        )
        self.manifest = PoolManifest(
            BINDING_ALIASED_RAW,
            [self.staging],
            [
                PoolEntry(
                    # A PCP producer's split main KV: one pool per producer rank,
                    # staged separately ("<owner>.pcp<r>").
                    tag=pool_tag(group.group_index, name, pcp_rank=r, split=split),
                    layer_name=f"{name}.pcp{r}" if split else name,
                    storage_index=0,
                    base_offset_bytes=0,
                    block_stride_bytes=self.src_layout.page_bytes,
                    num_blocks=self.pages,
                    regions=page_regions(name, self.src_layout.page_bytes, compressed_live),
                    dtype_tag="qwen4-whole-fp8-page-v1",
                )
                for group in self.src_groups
                for name in group.layer_names
                for split in (name in self._src_split,)
                for r in (range(self.src_pcp) if split else (0,))
            ]
            + key_entries,
        )
        self._src_owner = {n: g.group_index for g in self.src_groups for n in g.layer_names}
        self._src_owner.update(self._key_pools)
        # Staging pool name -> producer group (a split owner's per-rank pools too).
        self._pool_group = dict(self._src_owner) | {
            f"{n}.pcp{r}": self._src_owner[n]
            for n in self._src_split
            for r in range(self.src_pcp)
        }
        self._dst_owner = {n: g.group_index for g in self.dst_groups for n in g.layer_names}
        # Compressed-key owner -> (destination group, row offset in its page):
        # a packed member sits in its owner's group at slot * member rows.
        self._dst_compressed = compressed_destinations(
            decode_layout, self._dst_owner, self.dst_block, self.ratio
        )
        # Decode-only MTP: the draft layer's owners (and its packed compressed
        # slot) have no producer pages; its prompt KV starts empty (synthetic
        # acceptance does not read it).
        missing = (set(self._dst_owner) | set(self._dst_compressed)) - set(self._src_owner)
        if any(not n.startswith("mtp.") for n in missing):
            raise ValueError(f"Qwen PD staging lacks producer pages for {sorted(missing)}")
        self._dst_owner = {n: g for n, g in self._dst_owner.items() if n not in missing}
        self._dst_compressed = {
            n: v for n, v in self._dst_compressed.items() if n not in missing
        }
        # A speculative state table holds the committed state, then K
        # speculative checkpoints; the connector hands the repack only the
        # committed entry (_Stage3TransferGroup.transfer_pages).
        self._state_entry = 0
        self._gdn_names = [n for n in self._dst_owner if owner_kind(n) == "gdn"]
        self._token_extent = TransferExtent.TOKEN_PAGES
        tp = rank.tp_size
        src_gdn = gdn_geometry(cfg, 1, self.src_layout.token_bytes) if self._gdn_names else None
        dst_gdn = gdn_geometry(cfg, tp, decode_layout.token_bytes) if self._gdn_names else None
        self._fn = make_repack(
            src_block=self.src_block,
            dst_block=self.dst_block,
            src_gdn=src_gdn,
            dst_gdn=dst_gdn,
            kv_head=rank.kv_head,
            v_head_start=rank.tp_rank * (dst_gdn.v_heads if dst_gdn else 0),
            channels=conv_channels(src_gdn, dst_gdn, rank) if src_gdn else [0],
        )
        self._op = None

    # --- allocation (identical on every rank of the engine) ---

    def reserve(
        self, uuid: int, num_tokens: int, tp_group=None, *, skip_tokens: int = 0
    ) -> bool:
        """Staging pages for one load. A prefix-aware load (``skip_tokens``:
        the decode-local prefix) stages each token group from the first source
        page not wholly inside that prefix; request-state pages stage whole."""
        from .pd import skipped_source_pages

        if uuid in self.reservations:
            return True
        first = [skipped_source_pages(g, skip_tokens) for g in self.src_groups]
        group_counts = [
            g.page_count(num_tokens) - f for g, f in zip(self.src_groups, first)
        ]
        if any(count <= 0 for count in group_counts):
            raise ValueError("A prefix-aware load must leave every group a suffix")
        owners = [pool.layer_name for pool in self.manifest.pools]
        counts = [group_counts[self._pool_group[name]] for name in owners]
        free = self.free.clone()
        if tp_group is not None and getattr(tp_group, "world_size", 1) > 1:
            import torch.distributed as dist

            # Ranks free pages at their own receive terminals; allocate only
            # pages free on all of them, in the same order everywhere.
            dist.all_reduce(free, op=dist.ReduceOp.MIN, group=tp_group.cpu_group)
        available = torch.nonzero(free, as_tuple=False).flatten().tolist()
        if len(available) < sum(counts):
            return False
        pages, cursor = {}, 0
        for name, count in zip(owners, counts):
            pages[name] = available[cursor : cursor + count]
            cursor += count
        for owner_pages in pages.values():
            self.free[owner_pages] = 0
        self.reservations[uuid] = pages
        self.skips[uuid] = (skip_tokens, first)
        return True

    def destination(self, uuid: int) -> tuple[list[str], list[int], list[int]]:
        skip, _ = self.skips.get(uuid, (0, None))
        # The repack writes rows from the local prefix's end on: it must end on
        # a compressed pair and on a destination table entry (D * B tokens
        # under main-KV DCP), or those writes would land in shared prefix pages.
        entry = self.rank.main_dcp * self.dst_block
        if skip and (skip % (2 * self.ratio) or skip % entry):
            raise ValueError(
                f"A staged prefix-aware load needs a local prefix on a {entry}-token "
                f"table entry and a {2 * self.ratio}-token compressed pair, got {skip}"
            )
        pages = self.reservations[uuid]
        tags, blocks, counts = [], [], []
        for pool in self.manifest.pools:
            owner_pages = pages[pool.layer_name]
            tags.append(pool.tag)
            blocks.extend(owner_pages)
            counts.append(len(owner_pages))
        return tags, blocks, counts

    def skip_bytes(self, uuid: int) -> list[int]:
        """Per-tag destination clips, in ``destination`` order: the bytes of
        the source pages a prefix-aware load leaves out of each pool."""
        _, first = self.skips.get(uuid, (0, [0] * len(self.src_groups)))
        # The clip is in the destination's live bytes: a key pool's page
        # carries `key_stride` of them.
        return [
            first[self._pool_group[pool.layer_name]]
            * (self.key_stride if pool.layer_name in self._key_pools else pool.block_stride_bytes)
            for pool in self.manifest.pools
        ]

    def release(self, uuid: int) -> dict[str, list[int]] | None:
        self.skips.pop(uuid, None)
        pages = self.reservations.pop(uuid, None)
        if pages is not None:
            for owner_pages in pages.values():
                self.free[owner_pages] = 1
        return pages

    # --- the repack ---

    def indices(self, src_pages, dst_tables, num_tokens, skip_tokens=0, first=None):
        """Row indices of the repack. With ``skip_tokens`` only tokens from
        there on are written (the destination's earlier rows are shared prefix
        pages); ``src_pages[owner]`` are that cache owner's staging pages and
        ``first[g]`` the source page group ``g``'s staging starts at."""
        first = first or [0] * len(self.src_groups)
        main_s, main_d, copy_s, copy_d = [], [], [], []
        gdn_s, gdn_d = [], []
        for name, (gi, offset) in self._dst_compressed.items():
            if self.src_split_keys:
                # Keys of token t come from producer rank (t // B) % P, in that
                # rank's page for t's main-KV ID (B-token pages, P per ID).
                s, d = compressed_rows_split_np(
                    num_tokens, self.ratio,
                    [src_pages[f"{name}.pcp{r}"] for r in range(self.src_pcp)],
                    self.src_block, dst_tables[gi], self.dst_block, start=skip_tokens,
                    src_first=first[self._src_owner[f"{name}.pcp0"]], dst_offset=offset,
                )
                copy_s.append(s)
                copy_d.append(d)
                continue
            merged = name in self._key_pools
            s, d = compressed_rows_np(
                num_tokens, self.ratio, src_pages[name],
                # Merged: one staging page per main-KV ID, whose keys
                # (span pages of tokens) start at the page's first row.
                self.src_block * (self.src_layout.merged_key_span if merged else 1),
                dst_tables[gi], self.dst_block, start=skip_tokens,
                src_first=first[self._src_owner[name]], dst_offset=offset,
                src_page_rows=self.src_block if merged else None,
            )
            copy_s.append(s)
            copy_d.append(d)
        for name, gi in self._dst_owner.items():
            kind = owner_kind(name)
            if kind == "compressed":
                continue
            # A PCP producer's split main KV is staged per rank (below).
            src = src_pages.get(name)
            src_first = first[self._src_owner[name]]
            dst = dst_tables[gi]
            if kind == "main":
                pcp = None
                if name in self._src_split:
                    from .pool_layout import pcp_interleave

                    # One staged page list per producer rank (same length).
                    src = [src_pages[f"{name}.pcp{r}"] for r in range(self.src_pcp)]
                    pcp = (self.src_pcp, pcp_interleave())
                s, d = main_rows_np(num_tokens, src, self.src_block, dst, self.dst_block,
                                    self.rank, start=skip_tokens, src_first=src_first,
                                    src_pcp=pcp)
                main_s.append(s)
                main_d.append(d)
            elif kind in ("ring", "ple"):
                entry = dst[self._state_entry] if kind == "ple" else dst[0]
                s, d = page_prefix_rows(src[0], self.src_block, entry, self.dst_block,
                                        min(self.src_block, self.dst_block))
                copy_s.append(np.asarray(s, dtype=np.int64))
                copy_d.append(np.asarray(d, dtype=np.int64))
        for name in self._gdn_names:
            gdn_s.append(src_pages[name][0])
            gdn_d.append(dst_tables[self._dst_owner[name]][self._state_entry])

        def cat(parts):
            parts = [align_rows(p) for p in parts if len(p)]
            return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)

        return (
            cat(main_s), cat(main_d), cat(copy_s), cat(copy_d),
            np.asarray(gdn_s, dtype=np.int64), np.asarray(gdn_d, dtype=np.int64),
        )

    def repack(self, uuid: int, dst_tables, num_tokens: int) -> None:
        pages = self.reservations[uuid]
        skip, first = self.skips.get(uuid, (0, None))
        main_s, main_d, copy_s, copy_d, gdn_s, gdn_d = self.indices(
            pages, dst_tables, num_tokens, skip, first
        )
        device = self.pool.device

        if self._op is None:
            self._op = self._build_op()
        for chunk in repack_chunks(main_s, main_d, copy_s, copy_d, gdn_s, gdn_d):
            args = [
                torch.from_numpy(np.ascontiguousarray(v, dtype=np.int32)).to(device)
                for v in chunk
            ]
            self._op(self.pool, self.staging, *args)

    def _build_op(self):
        """The repack as the runner's pool row copies run it (mamba_state_copy_op).

        A donating op and its ``copy_`` writeback fuse into one aliased
        program only inside ``torch.compile``; called eagerly the pool is not
        donated, and XLA copies it to give the in-place writer a buffer (a
        second 48 GiB pool: compile HBM OOM). The op is built outside the
        compiled function, which dynamo cannot trace its construction from.
        """
        import jax
        from torch_tpu._internal import pallas

        fn = self._fn

        def repack_fn(
            pool: jax.Array,
            staging: jax.Array,
            main_src: jax.Array,
            main_dst: jax.Array,
            copy_src: jax.Array,
            copy_dst: jax.Array,
            gdn_src: jax.Array,
            gdn_dst: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            out = fn(pool, staging, main_src, main_dst, copy_src, copy_dst, gdn_src, gdn_dst)
            return out, main_src[0]

        op = pallas.jax_op("pallas::qwen4_pd_staged_repack", repack_fn, donate_argnums=(0,))
        op.register_fake(
            lambda pool, staging, main_src, *rest: (
                torch.empty_like(pool),
                torch.empty((), dtype=main_src.dtype, device=main_src.device),
            )
        )

        @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
        def run(pool, staging, main_src, main_dst, copy_src, copy_dst, gdn_src, gdn_dst):
            new_pool, marker = op(
                pool, staging, main_src, main_dst, copy_src, copy_dst, gdn_src, gdn_dst
            )
            pool.copy_(new_pool)
            return marker

        return run
