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
    layer, in the same order. Returns the updated pool.

    ``fn.rows(staging, main_src, copy_src, gdn_src)`` returns only the rows to
    write (uint8 ``[n, 4, 256]`` each), so a caller can scatter them into a
    pool in place; ``fn.gdn_rows`` is the destination row count per GDN page.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    chan = np.asarray(channels, dtype=np.int32)

    # Carriers are indexed as flat token rows, whatever their kernel-page
    # shape (the pools hold one kernel page per manager page).
    def rows_of(rows, flat):
        return rows[flat]

    def put(rows, flat, values):
        return rows.at[flat].set(values)

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
        stage = jax.lax.bitcast_convert_type(staging, jnp.uint8).reshape(
            -1, ROW_SUBLANES, ROW_LANES
        )
        # Main KV: the rank's head pair to slot 0, the other slots zero.
        pair = rows_of(stage, main_src)[:, 2 * kv_head : 2 * kv_head + 2]
        main = jnp.concatenate([pair, jnp.zeros_like(pair)], axis=1)
        # Compressed keys, rings + tags, PLE state: bytes unchanged.
        copies = rows_of(stage, copy_src)
        if has_gdn:
            s, d = src_gdn, dst_gdn
            s_rows = s.ssm_tokens + s.conv_tokens
            src_flat = (gdn_src[:, None] * src_block + jnp.arange(s_rows)).reshape(-1)
            region = rows_of(stage, src_flat).reshape(-1, s_rows, ROW_SUBLANES, ROW_LANES)
            gdn = jax.vmap(lambda page: gdn_page(page, s, d))(region)
            gdn = gdn.reshape(-1, ROW_SUBLANES, ROW_LANES)
        else:
            gdn = jnp.zeros((0, ROW_SUBLANES, ROW_LANES), jnp.uint8)
        return main, copies, gdn

    def fn(pool, staging, main_src, main_dst, copy_src, copy_dst, gdn_src, gdn_dst):
        dtype, shape = pool.dtype, pool.shape
        out = jax.lax.bitcast_convert_type(pool, jnp.uint8).reshape(-1, ROW_SUBLANES, ROW_LANES)
        main, copies, gdn = rows(staging, main_src, copy_src, gdn_src)
        out = put(out, main_dst, main)
        out = put(out, copy_dst, copies)
        if has_gdn:
            dst_flat = (gdn_dst[:, None] * dst_block + jnp.arange(d_rows)).reshape(-1)
            out = put(out, dst_flat, gdn)
        return jax.lax.bitcast_convert_type(out.reshape(shape), dtype)

    fn.rows = rows
    fn.gdn_rows = d_rows
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
        from .pool_layout import pack_compressed_k_enabled, pack_main_kv_enabled, pooled_cache_specs

        if pack_compressed_k_enabled() or pack_main_kv_enabled():
            raise ValueError(
                "A staged Qwen PD decode needs QWEN4_EXP_PACK_COMPRESSED_K=0 and "
                "QWEN4_EXP_PACK_MAIN_KV=0 (the pipeline producer's layout)"
            )
        self.cfg, self.rank, self.pool = cfg, rank, pool
        self.dst_layout = decode_layout
        self.dst_groups = decode_groups
        # The producer: TP1 stages, no main-KV DCP, the same packing flags.
        self.src_layout, src_specs = pooled_cache_specs(cfg, tp=1, main_dcp=1)
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
        self.reservations: dict[int, list[list[int]]] = {}
        self.manifest = PoolManifest(
            BINDING_ALIASED_RAW,
            [self.staging],
            [
                PoolEntry(
                    tag=f"qwen4.g{group.group_index}.{name}",
                    layer_name=name,
                    storage_index=0,
                    base_offset_bytes=0,
                    block_stride_bytes=self.src_layout.page_bytes,
                    num_blocks=self.pages,
                    regions=(
                        RegionSpec("page", 0, self.src_layout.page_bytes, self.src_layout.page_bytes, 1),
                    ),
                    dtype_tag="qwen4-whole-fp8-page-v1",
                )
                for group in self.src_groups
                for name in group.layer_names
            ],
        )
        self._src_owner = {n: g.group_index for g in self.src_groups for n in g.layer_names}
        self._dst_owner = {n: g.group_index for g in self.dst_groups for n in g.layer_names}
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

    def reserve(self, uuid: int, num_tokens: int, tp_group=None) -> bool:
        if uuid in self.reservations:
            return True
        counts = [g.page_count(num_tokens) for g in self.src_groups]
        free = self.free.clone()
        if tp_group is not None and getattr(tp_group, "world_size", 1) > 1:
            import torch.distributed as dist

            # Ranks free pages at their own receive terminals; allocate only
            # pages free on all of them, in the same order everywhere.
            dist.all_reduce(free, op=dist.ReduceOp.MIN, group=tp_group.cpu_group)
        available = torch.nonzero(free, as_tuple=False).flatten().tolist()
        if len(available) < sum(counts):
            return False
        pages, cursor = [], 0
        for count in counts:
            pages.append(available[cursor : cursor + count])
            cursor += count
        for group_pages in pages:
            self.free[group_pages] = 0
        self.reservations[uuid] = pages
        return True

    def destination(self, uuid: int) -> tuple[list[str], list[int], list[int]]:
        pages = self.reservations[uuid]
        tags, blocks, counts = [], [], []
        for pool in self.manifest.pools:
            group_pages = pages[self._src_owner[pool.layer_name]]
            tags.append(pool.tag)
            blocks.extend(group_pages)
            counts.append(len(group_pages))
        return tags, blocks, counts

    def release(self, uuid: int) -> list[list[int]] | None:
        pages = self.reservations.pop(uuid, None)
        if pages is not None:
            for group_pages in pages:
                self.free[group_pages] = 1
        return pages

    # --- the repack ---

    def indices(self, src_pages, dst_tables, num_tokens):
        main_s, main_d, copy_s, copy_d, gdn_s, gdn_d = [], [], [], [], [], []
        for name, gi in self._dst_owner.items():
            kind = owner_kind(name)
            src = src_pages[self._src_owner[name]]
            dst = dst_tables[gi]
            if kind == "main":
                s, d = main_rows(num_tokens, src, self.src_block, dst, self.dst_block, self.rank)
                main_s += s
                main_d += d
            elif kind == "compressed":
                s, d = compressed_rows(num_tokens, self.ratio, src, self.src_block, dst, self.dst_block)
                copy_s += s
                copy_d += d
            elif kind in ("ring", "ple"):
                s, d = page_prefix_rows(src[0], self.src_block, dst[0], self.dst_block,
                                        min(self.src_block, self.dst_block))
                copy_s += s
                copy_d += d
        for name in self._gdn_names:
            gdn_s.append(src_pages[self._src_owner[name]][0])
            gdn_d.append(dst_tables[self._dst_owner[name]][0])
        return main_s, main_d, copy_s, copy_d, gdn_s, gdn_d

    def repack(self, uuid: int, dst_tables, num_tokens: int) -> None:
        pages = self.reservations[uuid]
        main_s, main_d, copy_s, copy_d, gdn_s, gdn_d = self.indices(pages, dst_tables, num_tokens)
        device = self.pool.device

        def tensor(values, size):
            return torch.tensor(_pad(values, size), dtype=torch.int32).to(device)

        msize, csize = bucket(len(main_s)), bucket(len(copy_s))
        ngdn = max(1, len(gdn_s))
        if self._op is None:
            from torch_tpu._internal import pallas

            # Only the rows to write leave the op; the pool is written in
            # place below rather than copied whole through JAX.
            self._op = pallas.jax_op("pallas::qwen4_pd_staged_repack", self._fn.rows)
            d_rows = self._fn.gdn_rows

            def fake(staging, main_src, copy_src, gdn_src):
                def empty(n):
                    return staging.new_empty((n, ROW_SUBLANES, ROW_LANES), dtype=torch.uint8)

                return empty(len(main_src)), empty(len(copy_src)), empty(len(gdn_src) * d_rows)

            self._op.register_fake(fake)
        main, copies, gdn = self._op(
            self.staging, tensor(main_s, msize), tensor(copy_s, csize), tensor(gdn_s or [0], ngdn)
        )
        rows = self.pool.view(torch.uint8).view(-1, ROW_SUBLANES, ROW_LANES)
        rows[tensor(main_d, msize).long()] = main
        rows[tensor(copy_d, csize).long()] = copies
        if gdn_s:
            d_rows = self._fn.gdn_rows
            gdn_dst = [p * self.dst_block + r for p in gdn_d for r in range(d_rows)]
            rows[tensor(gdn_dst, len(gdn_dst)).long()] = gdn
