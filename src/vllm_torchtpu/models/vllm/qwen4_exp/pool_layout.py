# SPDX-License-Identifier: Apache-2.0
"""Model-owned geometry for QSA, GDN and PLE in attention-shaped pool pages.

The scheduler page fits every state type. QSA continues to use 128-token
kernel pages; compressed keys remain BF16 even when the pool carrier is FP8.
No allocation or device initialization takes place here.
"""

import math
import os
from dataclasses import dataclass, field, replace
from typing import Any

import torch
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
)

from vllm_torchtpu import envs
from vllm_torchtpu.core.mamba_tail_fold import (
    fold_blocked_by_context_parallel,
    with_prefill_checkpoint,
)
from vllm_torchtpu.gdn_pool_layout import derive_pooled_gdn_state_layout

from .config import first_qsa_layer_index, mtp_layer_prefixes, native_cache_specs
from .qsa import QSA_BLOCK_SIZE

# Explicit physical format shared by this model's page planner and GDN ops.
POOLED_GDN_CONV_STATE_DTYPE = torch.float32

PACK_COMPRESSED_K_ENV = "QWEN4_EXP_PACK_COMPRESSED_K"


def pack_compressed_k_enabled() -> bool:
    """Whether several QSA layers' compressed keys share one pool page.

    Off by default. ``QWEN4_EXP_PACK_COMPRESSED_K=1`` packs; see
    ``pooled_cache_specs``.
    """
    return envs.QWEN4_EXP_PACK_COMPRESSED_K


MERGE_COMPRESSED_K_ENV = "QWEN4_EXP_MERGE_COMPRESSED_K"

# ``compressed_owners`` slot of a merged layer; bound as ``-cols * N`` once the
# pool's scheduler block count ``N`` is known (``Qwen4ExpPoolLayout``).
MERGED_COMPRESSED_SLOT = -1


def merge_compressed_k_enabled() -> bool:
    """Whether each QSA layer's compressed keys live in its main KV carrier.

    Off by default. ``QWEN4_EXP_MERGE_COMPRESSED_K=1`` drops the
    compressed-key cache groups: a layer's keys for block ``c`` sit in a
    region appended after the carrier's ``N`` scheduler pages and are
    addressed through the layer's main KV block table; see
    ``pooled_cache_specs``.
    """
    value = os.environ.get(MERGE_COMPRESSED_K_ENV, "0").strip().lower()
    return value in ("1", "true", "yes", "on")


INROW_COMPRESSED_K_ENV = "QWEN4_EXP_INROW_COMPRESSED_K"


def inrow_compressed_k_enabled() -> bool:
    """Whether each QSA layer's compressed keys ride its own main-KV rows.

    Off by default. ``QWEN4_EXP_INROW_COMPRESSED_K=1`` (one FP8 KV head per
    device, e.g. a TP8 decode): a main-KV token row holds the head's K and V
    in two of its four bytes per lane; the other two -- half of every row --
    hold one BF16 compressed key (``pooled_qsa.inrow_compressed_slot``). The
    compressed-key groups drop out and the keys cost no pool bytes.
    """
    value = os.environ.get(INROW_COMPRESSED_K_ENV, "0").strip().lower()
    return value in ("1", "true", "yes", "on")


PCP_SPLIT_COMPRESSED_K_ENV = "QWEN4_EXP_PCP_SPLIT_COMPRESSED_K"


def pcp_split_compressed_k_enabled() -> bool:
    """Whether a PCP group splits its merged compressed keys by page owner.

    Needs ``QWEN4_EXP_MERGE_COMPRESSED_K=1``. Each rank keeps (and scores)
    only the keys of the pages it owns: one column of the key region per
    main-KV block ID instead of ``pcp`` columns.
    """
    value = os.environ.get(PCP_SPLIT_COMPRESSED_K_ENV, "0").strip().lower()
    return value in ("1", "true", "yes", "on")


PACK_MAIN_KV_ENV = "QWEN4_EXP_PACK_MAIN_KV"


def pack_main_kv_enabled() -> bool:
    """Whether two QSA layers' main KV heads share one pool page.

    Off by default. ``QWEN4_EXP_PACK_MAIN_KV=1`` pairs; see
    ``pooled_cache_specs``.
    """
    return envs.QWEN4_EXP_PACK_MAIN_KV


MAIN_KV_DCP_ENV = "VLLM_TPU_QSA_MAIN_KV_DCP"
PCP_INTERLEAVE_ENV = "QWEN4_EXP_PCP_INTERLEAVE"


def pcp_interleave() -> int:
    """Tokens per main-KV ownership unit under prefill context parallelism.

    A PCP group splits main KV by token like main-KV DCP, but a TP1 stage's
    page (3,200 rows) is too long to be one rank's share of a paired prefill
    KV tile, so units of this many tokens alternate over the ranks instead
    (``qsa_rpa.dcp_owner_row``). 128 by default: a 256-token tile at PCP2.
    """
    value = int(os.environ.get(PCP_INTERLEAVE_ENV, "128"))
    if value <= 0 or value % 128:
        raise ValueError(f"{PCP_INTERLEAVE_ENV} must be a positive multiple of 128")
    return value


def main_kv_dcp() -> int:
    """Decode-context-parallel size of the QSA main KV (1 when off).

    ``VLLM_TPU_QSA_MAIN_KV_DCP=2``: the two TP ranks sharing a KV head
    (``tp_rank // 2``) split each request's main KV by token. One main-KV
    block-table entry then covers ``2 * B`` tokens; token ``t`` lives on pair
    rank ``(t // B) % 2`` at row ``t % B`` of that entry's page, so each rank
    stores half of the span in an unchanged page. 0 and 1 mean off. This is
    not vLLM's ``decode_context_parallel_size``, which stays 1.
    """
    value = envs.VLLM_TPU_QSA_MAIN_KV_DCP
    if value not in (0, 1, 2, 4):
        raise ValueError(f"{MAIN_KV_DCP_ENV} must be 0, 1, 2 or 4, got {value}")
    return max(value, 1)


@dataclass(frozen=True)
class Qwen4ExpPoolLayout:
    block_size: int
    page_bytes: int
    token_bytes: int
    kernel_block_size: int
    gdn_state_bytes: int
    ple_state_bytes: int
    compressed_kernel_rows: int
    compressed_kernel_bytes: int
    compressed_width: int
    # Layers whose compressed keys share one pool page (1 = unpacked).
    compressed_slots: int = 1
    # (QSA attention prefix, compressed-key cache owner name, page slot);
    # empty when unpacked, where every layer owns ``<prefix>.compressed_k``.
    compressed_owners: tuple[tuple[str, str, int], ...] = field(default=())
    # (QSA attention prefix, main KV cache owner name, head-pair slot) of
    # layers sharing main KV pages; empty when every layer owns its pages.
    main_owners: tuple[tuple[str, str, int], ...] = field(default=())
    # `main_kv_dcp()`: tokens per target main-KV block-table entry are
    # ``main_dcp * block_size``; each pair rank stores ``block_size`` rows of
    # them per page. MTP draft main KV keeps ``block_size`` unless
    # VLLM_TPU_QSA_MTP_KV_DCP splits it too.
    main_dcp: int = 1
    # Tokens per ownership unit of the main-KV split (0 = one page: main-KV
    # DCP); PCP uses pcp_interleave().
    main_dcp_interleave: int = 0
    # Main KV owner groups whose block size is scaled (the draft's with
    # VLLM_TPU_QSA_MTP_KV_DCP).
    main_dcp_owners: tuple[str, ...] = field(default=())
    # QWEN4_EXP_MERGE_COMPRESSED_K: compressed keys live in a region of each
    # main KV carrier after its scheduler pages (``merged_compressed_slot``).
    merged_compressed: bool = False
    # QWEN4_EXP_PCP_SPLIT_COMPRESSED_K: merged keys split over the PCP ranks
    # by page owner (one region column per block ID per rank).
    split_compressed: bool = False
    # QWEN4_EXP_INROW_COMPRESSED_K: compressed keys live in the second BF16
    # half of the layer's own main-KV rows (``pooled_qsa.inrow_compressed_slot``).
    inrow_compressed: bool = False

    def compressed_owner(self, attention_prefix: str) -> tuple[str, int]:
        """Cache owner name and in-page slot of a QSA layer's compressed keys."""
        for prefix, owner, slot in self.compressed_owners:
            if prefix == attention_prefix:
                return owner, slot
        return attention_prefix + ".compressed_k", 0

    def main_owner(self, attention_prefix: str) -> tuple[str, int]:
        """Cache owner name and head-pair slot of a QSA layer's main KV.

        Slot -1: the layer owns whole rows of its own pages.
        """
        for prefix, owner, slot in self.main_owners:
            if prefix == attention_prefix:
                return owner, slot
        return attention_prefix, -1

    @property
    def main_kernel_pages(self):
        return self.block_size // self.kernel_block_size

    @property
    def main_block_size(self):
        """Logical tokens per target main-KV block-table entry."""
        return self.main_dcp * self.block_size

    @property
    def compressed_physical_pages(self):
        return self.page_bytes // self.compressed_kernel_bytes

    @property
    def compressed_columns_per_page(self) -> int:
        """Blocks' worth of one layer's compressed keys that fill a page."""
        return self.compressed_physical_pages // self.main_kernel_pages

    @property
    def merged_key_span(self) -> int:
        """Pool pages of tokens one main-KV block ID covers, so columns of keys
        per ID in the merged region: ``main_dcp`` (2 under PCP2), else 1."""
        if not self.merged_compressed or self.split_compressed:
            return 1
        return max(1, int(self.main_dcp))

    def compressed_region_pages(self, num_blocks: int) -> int:
        """Pages a merged carrier appends after ``num_blocks`` scheduler pages."""
        if not self.merged_compressed:
            return 0
        return -(-num_blocks * self.merged_key_span // self.compressed_columns_per_page)

    def merged_compressed_slot(self, num_blocks: int) -> int:
        """``compressed_slot`` of a merged layer in a pool of ``num_blocks``
        scheduler pages: region ``cols * N`` and the ID's span, encoded by
        ``pooled_qsa.merged_compressed_slot``."""
        from vllm_torchtpu.kernels.qwen4_exp.pooled_qsa import merged_compressed_slot

        assert self.merged_compressed
        return merged_compressed_slot(
            self.compressed_columns_per_page * num_blocks, self.merged_key_span
        )


def pooled_cache_specs(
    cfg,
    *,
    tp: int | None = None,
    main_dcp: int | None = None,
    pack_compressed: bool | None = None,
    merge_compressed: bool | None = None,
    main_interleave: int | None = None,
    split_compressed: bool | None = None,
    draft_prefix_cacheable: bool | None = None,
    inrow_compressed: bool | None = None,
):
    """Derive uniform physical pages without widening compressed-key coverage.

    Increasing a compressed page's *logical* span merely to fill the pool
    would multiply prefix-match alignment by the compression ratio. Instead,
    main and compressed pages cover the same token interval; compressed pages
    have unused tail bytes. Raw rings retain their non-cacheable lifecycle.

    With ``QWEN4_EXP_PACK_COMPRESSED_K=1`` those tail bytes hold other
    layers' compressed keys instead: consecutive QSA layers (the MTP draft's
    included) form one scheduler group named after its first member, whose
    page stores member ``s`` at kernel pages
    ``[s * main_kernel_pages, (s + 1) * main_kernel_pages)``. Each member
    still covers exactly the group's token interval, so prefix alignment,
    hashing and eviction are unchanged; only the page count per token drops.

    ``QWEN4_EXP_PACK_MAIN_KV=1`` likewise pairs main KV pages. With one FP8
    KV head per device a token row has four head slots, K and V fill two.
    Consecutive target QSA layers pair up: the group is named after the
    first, member ``s`` stores K and V at head slots ``2s`` and ``2s + 1``.
    MTP draft layers keep pages of their own (the proposer addresses its
    layer's group by name), as does an odd last target layer.

    ``VLLM_TPU_QSA_MAIN_KV_DCP=2`` (``main_kv_dcp``) gives the target main KV
    groups, paired or not, ``block_size = 2 * B`` with unchanged page bytes:
    the two ranks of a KV-head pair each store ``B`` of the entry's ``2 * B``
    tokens. vLLM then schedules in LCM blocks (``2 * B``) and hashes in GCD
    blocks (``B``); compressed keys, rings, GDN, PLE and the MTP draft's main
    KV keep ``B``, so prefix hashing and the draft are unchanged.
    ``VLLM_TPU_QSA_MTP_KV_DCP`` gives the draft's main KV groups ``2 * B`` too.
    """
    # ``tp`` / ``main_dcp`` / ``pack_compressed`` / ``draft_prefix_cacheable``
    # describe another engine (a PD peer's geometry, pd_staging); by default
    # this engine's (config.native_cache_specs).
    tp = cfg.parallel_config.tensor_parallel_size if tp is None else tp
    specs = native_cache_specs(cfg, tp, draft_prefix_cacheable=draft_prefix_cacheable)
    text = cfg.model_config.hf_text_config
    main: Any = specs[
        f"model.layers.{first_qsa_layer_index(text.layer_types)}.self_attn"
    ]
    token_bytes = main.page_size_bytes // main.block_size
    gdn = specs[
        f"model.layers.{text.layer_types.index('linear_attention')}.linear_attn"
    ]
    assert isinstance(gdn, MambaSpec)
    state = derive_pooled_gdn_state_layout(
        ssm_bytes=math.prod(gdn.shapes[1]) * gdn.dtypes[1].itemsize,
        conv_bytes=math.prod(gdn.shapes[0]) * POOLED_GDN_CONV_STATE_DTYPE.itemsize,
        token_bytes=token_bytes,
    )
    ple_bytes = max(
        (spec.page_size_bytes for name, spec in specs.items() if name.endswith(".ple")),
        default=0,
    )
    required_rows = max(state.required_tokens, math.ceil(ple_bytes / token_bytes))
    block_size = math.ceil(required_rows / QSA_BLOCK_SIZE) * QSA_BLOCK_SIZE
    page_bytes = block_size * token_bytes
    compressed_rows = QSA_BLOCK_SIZE // text.indexer_compress_ratio
    # Preserve the pool's lane axis. Narrowing 256 lanes to 128 outside the
    # kernel can relayout the entire owner; a padded query/key contraction
    # instead pays for only the active key tiles and keeps native RPA storage.
    compressed_width = math.ceil(text.head_dim / 128) * 128
    if text.indexer_head_dim > compressed_width:
        raise ValueError(
            "Qwen4Exp pooled indexer width must fit the main KV lane width"
        )
    compressed_bytes = compressed_rows * compressed_width * 2
    if page_bytes % compressed_bytes:
        raise ValueError("Qwen4Exp pool page must contain whole BF16 indexer pages")
    # A layer's compressed keys for one manager page fill only
    # ``main_kernel_pages`` kernel pages; the rest of the pool page is free.
    packed_slots = page_bytes // (block_size // QSA_BLOCK_SIZE * compressed_bytes)
    compressed_names = [name for name in specs if name.endswith(".compressed_k")]
    owners: list[tuple[str, str, int]] = []
    if pack_compressed is None:
        pack_compressed = pack_compressed_k_enabled()
    merged = merge_compressed_k_enabled() if merge_compressed is None else merge_compressed
    inrow = inrow_compressed_k_enabled() if inrow_compressed is None else inrow_compressed
    if inrow:
        problems = [
            env
            for env, on in (
                (MERGE_COMPRESSED_K_ENV, merged),
                (PACK_COMPRESSED_K_ENV, pack_compressed),
                (PACK_MAIN_KV_ENV, pack_main_kv_enabled()),
            )
            if on
        ]
        if problems:
            raise ValueError(
                f"{INROW_COMPRESSED_K_ENV}=1 replaces {' and '.join(problems)}; unset them"
            )
        if main.dtype != torch.float8_e4m3fn or main.num_kv_heads != 1:
            raise ValueError(
                f"{INROW_COMPRESSED_K_ENV}=1 keeps keys in the free half of an FP8 "
                "main-KV row, which needs one KV head per device and "
                f"--kv-cache-dtype fp8; got {main.num_kv_heads} heads of {main.dtype}"
            )
        if 2 * compressed_width * 2 > token_bytes:
            raise ValueError("a compressed key must fit half a main-KV row")
        packed_slots = 1
    if merged:
        if pack_compressed or pack_main_kv_enabled():
            raise ValueError(
                f"{MERGE_COMPRESSED_K_ENV}=1 replaces {PACK_COMPRESSED_K_ENV} and "
                f"is not combined with {PACK_MAIN_KV_ENV}"
            )
        if page_bytes % (block_size // QSA_BLOCK_SIZE * compressed_bytes):
            raise ValueError(
                "Qwen4Exp merged compressed keys need a page of whole blocks of keys"
            )
        # Each layer's keys ride its own main KV block table (one block ID per
        # token span for both); the slot is bound once the pool size is known.
        owners = [
            (prefix, prefix, MERGED_COMPRESSED_SLOT)
            for prefix in (name.removesuffix(".compressed_k") for name in compressed_names)
        ]
        packed_slots = 1
    elif pack_compressed and packed_slots > 1 and not inrow:
        # Consecutive QSA layers (the MTP draft's last) share one owner
        # group; one block table and one page lifetime cover all of them,
        # since compressed keys of every layer track the same token span.
        for start in range(0, len(compressed_names), packed_slots):
            chunk = compressed_names[start : start + packed_slots]
            owners.extend(
                (name.removesuffix(".compressed_k"), chunk[0], slot)
                for slot, name in enumerate(chunk)
            )
    else:
        packed_slots = 1
    main_owners: list[tuple[str, str, int]] = []
    if pack_main_kv_enabled():
        packing = 4 if main.dtype == torch.float8_e4m3fn else 2
        if main.num_kv_heads != 1 or packing * main.num_head_slots != 4:
            raise ValueError(
                f"{PACK_MAIN_KV_ENV}=1 pairs layers in the two padding head "
                "slots of an FP8 main KV row, which needs one KV head per "
                f"device and --kv-cache-dtype fp8; got {main.num_kv_heads} "
                f"heads of {main.dtype}"
            )
        draft = {prefix + ".self_attn" for prefix in mtp_layer_prefixes(cfg)}
        targets = [
            name for name in specs if name.endswith(".self_attn") and name not in draft
        ]
        for start in range(0, len(targets) - 1, 2):
            owner = targets[start]
            main_owners.extend(
                (name, owner, slot)
                for slot, name in enumerate(targets[start : start + 2])
            )
    # PCP: the group splits main KV by token (main-KV DCP over the PCP ranks,
    # in pcp_interleave() units); everything else stays replicated.
    # ``main_dcp`` with ``main_interleave`` describes a PCP peer (pd_staging).
    pcp = int(getattr(cfg.parallel_config, "prefill_context_parallel_size", 1) or 1)
    interleave = main_interleave or 0
    if pcp > 1 and main_dcp is None:
        if main_kv_dcp() > 1 or tp != 1:
            raise ValueError(f"PCP replaces {MAIN_KV_DCP_ENV} and runs TP1 stages")
        main_dcp, interleave = pcp, pcp_interleave()
        if block_size % interleave:
            raise ValueError(
                f"{PCP_INTERLEAVE_ENV}={interleave} must divide the {block_size}-token page"
            )
    dcp = main_kv_dcp() if main_dcp is None else main_dcp
    if merged and dcp > 1 and not interleave:
        # PCP (an interleaved main-KV split) keeps replicated keys: each main-KV ID
        # spans `pcp` pages and owns that many columns of the region.
        raise ValueError(f"{MERGE_COMPRESSED_K_ENV}=1 is not combined with {MAIN_KV_DCP_ENV}")
    draft_attention = {prefix + ".self_attn" for prefix in mtp_layer_prefixes(cfg)}
    paired_away = {name for name, owner, _ in main_owners if name != owner}
    dcp_owners: tuple[str, ...] = ()
    if dcp > 1 and interleave:
        from vllm_torchtpu.kernels.qwen4_exp.block_sparse_gqa_prefill import (
            PAIRED_KV_TILES,
        )

        if dcp * interleave not in PAIRED_KV_TILES or main.dtype != torch.float8_e4m3fn:
            raise ValueError(
                f"PCP{dcp} with {PCP_INTERLEAVE_ENV}={interleave} needs a {dcp * interleave}-token "
                f"paired prefill tile ({PAIRED_KV_TILES}) and an FP8 main KV cache"
            )
        draft = {prefix + ".self_attn" for prefix in mtp_layer_prefixes(cfg)}
        dcp_owners = tuple(
            name
            for name in specs
            if name.endswith(".self_attn") and name not in draft and name not in paired_away
        )
    elif dcp > 1:
        p = cfg.parallel_config
        if tp < 2 or tp % dcp:
            raise ValueError(
                f"{MAIN_KV_DCP_ENV}={dcp} splits a KV head's main KV over TP rank "
                f"pairs; needs an even tensor_parallel_size >= 2, got "
                f"{p.tensor_parallel_size}"
            )
        if p.decode_context_parallel_size != 1:
            raise ValueError(
                f"{MAIN_KV_DCP_ENV} replaces vLLM's decode context parallelism; "
                "keep decode_context_parallel_size=1"
            )
        from vllm_torchtpu.kernels.qwen4_exp.block_sparse_gqa_prefill import (
            PAIRED_KV_TILES,
        )

        # DCP prefill attends one table entry (dcp pages) per KV tile through
        # the paired-page kernel; fail here, not in the first prefill.
        if dcp * block_size not in PAIRED_KV_TILES or main.dtype != torch.float8_e4m3fn:
            raise ValueError(
                f"{MAIN_KV_DCP_ENV}={dcp} needs a prefill KV tile of {dcp} x "
                f"{block_size}-row pages = {dcp * block_size} in the paired-page "
                f"kernel's {PAIRED_KV_TILES} and an FP8 main KV cache; got "
                f"{main.dtype}"
            )

        # VLLM_TPU_QSA_MTP_KV_DCP: the draft's main KV splits like the target's.
        split_draft = bool(envs.VLLM_TPU_QSA_MTP_KV_DCP)
        # VLLM_TPU_QSA_COMPRESSED_DCP: compressed keys too. Their groups hold
        # the draft's keys as well (packed), so the draft must be split.
        split_compressed = bool(envs.VLLM_TPU_QSA_COMPRESSED_DCP)
        if split_compressed and not split_draft:
            raise ValueError(
                "VLLM_TPU_QSA_COMPRESSED_DCP needs VLLM_TPU_QSA_MTP_KV_DCP: the "
                "MTP draft writes compressed keys into the split groups"
            )
        dcp_owners = tuple(
            name
            for name in specs
            if name.endswith(".self_attn")
            and (split_draft or name not in draft_attention)
            and name not in paired_away
        )
        if split_compressed:
            dcp_owners += tuple(
                name for name in specs if name.endswith(".compressed_k")
            )
    if inrow:
        if interleave:
            raise ValueError(f"{INROW_COMPRESSED_K_ENV}=1 is a TP decode layout, not PCP")
        if dcp > 1 and envs.VLLM_TPU_QSA_COMPRESSED_DCP:
            raise ValueError(
                f"{INROW_COMPRESSED_K_ENV}=1 keeps keys replicated: unset "
                "VLLM_TPU_QSA_COMPRESSED_DCP"
            )
        from vllm_torchtpu.kernels.qwen4_exp.pooled_qsa import inrow_compressed_slot

        ratio = int(text.indexer_compress_ratio)
        owners, main_owners = [], []
        for name in compressed_names:
            prefix = name.removesuffix(".compressed_k")
            # A DCP-split owner's entry covers `dcp` pages of tokens; its keys
            # (dcp * B / ratio of them) must fit the B rows of one page.
            span = dcp if prefix in dcp_owners else 1
            if span * block_size // ratio > block_size:
                raise ValueError(
                    f"{INROW_COMPRESSED_K_ENV}=1: an entry of {span} x {block_size} "
                    f"tokens has more keys than its page has rows (ratio {ratio})"
                )
            owners.append((prefix, prefix, inrow_compressed_slot(span)))
            # Slot 0: the layer's K/V writes keep bytes 2-3 (its keys).
            main_owners.append((prefix, prefix, 0))
    layout = Qwen4ExpPoolLayout(
        block_size=block_size,
        page_bytes=page_bytes,
        token_bytes=token_bytes,
        kernel_block_size=QSA_BLOCK_SIZE,
        gdn_state_bytes=state.required_bytes,
        ple_state_bytes=ple_bytes,
        compressed_kernel_rows=compressed_rows,
        compressed_kernel_bytes=compressed_bytes,
        compressed_width=compressed_width,
        compressed_slots=packed_slots,
        compressed_owners=tuple(owners),
        main_owners=tuple(main_owners),
        main_dcp=dcp,
        main_dcp_interleave=interleave,
        main_dcp_owners=dcp_owners,
        merged_compressed=merged,
        split_compressed=bool(
            merged and interleave and (
                pcp_split_compressed_k_enabled() if split_compressed is None else split_compressed
            )
        ),
        inrow_compressed=inrow,
    )
    # Members other than their group's owner get no scheduler group.
    packed_away = {prefix + ".compressed_k" for prefix, owner, _ in owners} - {
        owner for _, owner, _ in owners
    }
    # Merged: owners are the main KV caches themselves, so every
    # ``.compressed_k`` spec drops out (above) and no group is planned for it.
    packed_away |= {name for name, owner, _ in main_owners if name != owner}
    pooled: dict[str, KVCacheSpec] = {}
    # Upstream uses one persistent Mamba state block per request without APC.
    # Its logical span is max_model_len, while the physical state still fits
    # in a single uniform pool page. APC uses position-indexed align blocks.
    mamba_mode = "align" if cfg.cache_config.enable_prefix_caching else "none"
    mamba_block_size = (
        block_size if mamba_mode == "align" else cfg.model_config.max_model_len
    )
    for name, spec in specs.items():
        if name in packed_away:
            continue
        if isinstance(spec, MambaSpec):
            dtypes = (
                (POOLED_GDN_CONV_STATE_DTYPE, spec.dtypes[1])
                if name.endswith(".linear_attn")
                else spec.dtypes
            )
            pooled[name] = replace(
                spec,
                block_size=mamba_block_size,
                page_size_padded=page_bytes,
                dtypes=dtypes,
                mamba_cache_mode=mamba_mode,
            )
            if (
                mamba_mode == "align"
                and envs.VLLM_TPU_MAMBA_TAIL_FOLD
                and fold_blocked_by_context_parallel(cfg, dcp) is None
                # A PCP split (also a staged decode's view of a PCP producer)
                # keeps the fold off, as the PCP producer itself does.
                and not interleave
            ):
                # The GDN kernel and the PLE layer write the state at the
                # prompt's last block boundary from inside the step, so the
                # scheduler need not split the tail off (core/mamba_tail_fold.py).
                pooled[name] = with_prefill_checkpoint(pooled[name])
        else:
            assert isinstance(spec, AttentionSpec)
            if name.endswith(".compressed_k"):
                assert isinstance(spec, MLAAttentionSpec)
                spec = replace(spec, state_content_bytes=compressed_width * 2)
            if name in dcp_owners:
                # 2B tokens per entry, B stored rows per rank: one state per
                # `dcp` tokens keeps the unpadded page at B rows (= page_bytes).
                pooled[name] = replace(
                    spec,
                    block_size=dcp * block_size,
                    tokens_per_state=dcp,
                    page_size_padded=page_bytes,
                )
                continue
            pooled[name] = replace(
                spec,
                block_size=spec.block_size
                if name.endswith(".raw_ring")
                else block_size,
                page_size_padded=page_bytes,
            )
    return layout, pooled


def check_packed_compressed_groups(layout: Qwen4ExpPoolLayout, groups) -> None:
    """Each packed compressed-key or main KV owner needs a block table of its own.

    Members address slots of their owner's pages, so two owners sharing one
    group (one block ID per token span) would overwrite each other's slots.
    """
    if layout.merged_compressed or layout.inrow_compressed:
        # Merged / in-row keys sit in the owner layer's own pages, never shared.
        compressed = set()
    else:
        compressed = {owner for _, owner, _ in layout.compressed_owners}
    for kind, owners, env in (
        (
            "compressed keys",
            compressed,
            PACK_COMPRESSED_K_ENV,
        ),
        ("main KV", {owner for _, owner, _ in layout.main_owners}, PACK_MAIN_KV_ENV),
    ):
        for group in groups:
            names = set(group.layer_names)
            if names & owners and len(names) != 1:
                raise ValueError(
                    f"Qwen4Exp packed {kind} need one scheduler group per "
                    f"owner, got {sorted(names)}; unset {env}"
                )
