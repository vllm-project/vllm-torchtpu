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
import time
from collections.abc import Sequence

import numpy as np
import torch

try:
    from . import pd_route
except ImportError:  # loaded by path (tests, the landing microbench): no package
    import importlib.util as _ilu
    import pathlib as _pl

    _spec = _ilu.spec_from_file_location(
        "qwen4_exp_pd_route", _pl.Path(__file__).with_name("pd_route.py")
    )
    pd_route = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(pd_route)

try:
    from . import pd_shared_compressed
except ImportError:  # loaded by path: no package
    import importlib.util as _ilu
    import pathlib as _pl
    import sys as _sys

    _spec = _ilu.spec_from_file_location(
        "qwen4_exp_pd_shared_compressed",
        _pl.Path(__file__).with_name("pd_shared_compressed.py"),
    )
    pd_shared_compressed = _ilu.module_from_spec(_spec)
    _sys.modules[_spec.name] = pd_shared_compressed  # its dataclasses look it up
    _spec.loader.exec_module(pd_shared_compressed)

try:
    from vllm_torchtpu.logger import init_logger

    logger = init_logger(__name__)
except ImportError:  # loaded by path without the package
    import logging

    logger = logging.getLogger(__name__)

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


def _live_page_bytes(pool) -> int:
    """A carrier page's live bytes (the units of the request-global
    destination space and of dst_skip_bytes): a trimmed compressed-key
    page's keys, else the whole stride (pd_connector._live_page_bytes)."""
    regions = getattr(pool, "regions", ()) or ()
    if len(regions) == 1 and getattr(regions[0], "num_units", 1) == 1:
        return int(regions[0].unit_bytes)
    return int(pool.block_stride_bytes)


def quarantine_steps() -> int:
    """QWEN4_EXP_PD_STAGING_QUARANTINE_STEPS (envs; read directly when this
    module is loaded by path, without the package)."""
    try:
        from vllm_torchtpu import envs
    except ImportError:
        import os

        return max(0, int(os.environ.get("QWEN4_EXP_PD_STAGING_QUARANTINE_STEPS", "3")))
    return int(envs.QWEN4_EXP_PD_STAGING_QUARANTINE_STEPS)


def draft_kv_enabled() -> bool:
    """QWEN4_EXP_PD_DRAFT_KV (q38 F2; envs; read directly when this module is
    loaded by path, without the package)."""
    try:
        from vllm_torchtpu import envs
    except ImportError:
        import os

        value = os.environ.get("QWEN4_EXP_PD_DRAFT_KV", "").strip().lower()
        return value in ("1", "true", "yes", "on")
    return bool(envs.QWEN4_EXP_PD_DRAFT_KV)


def draft_hold_enabled() -> bool:
    """QWEN4_EXP_PD_DRAFT_HOLD (q38 F2-HOLD; envs; read directly when this
    module is loaded by path, without the package)."""
    try:
        from vllm_torchtpu import envs
    except ImportError:
        import os

        value = os.environ.get("QWEN4_EXP_PD_DRAFT_HOLD", "").strip().lower()
        return value in ("1", "true", "yes", "on")
    return bool(envs.QWEN4_EXP_PD_DRAFT_HOLD)


def force_gap_fraction() -> float:
    """QWEN4_EXP_PD_DRAFT_KV_FORCE_GAP (q38 F2-3's forced-gap arm), a fraction
    in [0, 1); read directly when this module is loaded by path."""
    try:
        from vllm_torchtpu import envs

        value = float(envs.QWEN4_EXP_PD_DRAFT_KV_FORCE_GAP)
    except ImportError:
        import os

        value = float(os.environ.get("QWEN4_EXP_PD_DRAFT_KV_FORCE_GAP", "") or 0.0)
    if not 0.0 <= value < 1.0:
        raise ValueError(f"QWEN4_EXP_PD_DRAFT_KV_FORCE_GAP must be in [0, 1), got {value}")
    return value


# The producer's prefix hit in its kv_transfer_params (q38 F2 option (a)):
# draft rows below it were never computed there (its draft KV is not
# prefix-cacheable), so a decode with a shorter local hit zero-fills them.
PRODUCER_HIT_PARAM = "qwen_producer_hit_tokens"


# The tokens an F2-HOLD producer's draft tables skip (QWEN4_EXP_PD_DRAFT_HOLD):
# its draft main-KV and compressed-key pages below this are null blocks, and
# the send fills their rows from a live page. It is at most the published hit
# by construction (the hold skips only the computed prefix); a decode never
# lands a draft row below it either way (published_hit_tokens).
PRODUCER_DRAFT_SKIP_PARAM = "qwen_producer_draft_skip_tokens"


def _positive_int(params, key) -> int:
    value = params.get(key) if isinstance(params, dict) else None
    return value if type(value) is int and value > 0 else 0


def published_hit_tokens(params) -> int:
    """The producer's published prefix hit, in tokens; 0 when absent or
    invalid (an older producer, or no hit). Under F2-HOLD it is at least the
    producer's draft skip: draft rows below the skip were filled from a live
    page, so the decode must zero-fill them as it does the rows below Hp."""
    return max(
        _positive_int(params, PRODUCER_HIT_PARAM),
        _positive_int(params, PRODUCER_DRAFT_SKIP_PARAM),
    )


# Draft tokens of the producer's prefill-only MTP under QWEN4_EXP_PD_DRAFT_KV
# (config.pd_prefill_mtp_producer admits exactly this depth): its GDN and PLE
# state tables carry one speculative block, whatever the decode's K.
PRODUCER_DRAFT_TOKENS = 1


def producer_geometry(cfg):
    """The producer's page plan as this staged decode must rebuild it:
    ``(cfg_for_layout, tp, main_dcp, pack_compressed, draft_prefix_cacheable)``,
    the arguments of ``pool_layout.pooled_cache_specs``.

    The producer is assumed to be TP1 pipeline stages: ``tp=1``, no main-KV
    DCP (``main_dcp=1``) and unpacked compressed keys
    (``pack_compressed=False``; config.validate_config refuses
    QWEN4_EXP_PACK_COMPRESSED_K=1 under PP). This is the single place a
    non-TP1 producer extends: a TP2 or PCP2 producer returns its own TP,
    DCP and packing here (and the repack then needs that geometry's source
    slicing; ``RankSlice`` and ``make_repack`` assume TP1 source pages).

    The draft. Without QWEN4_EXP_PD_DRAFT_KV the producer runs no MTP, so its
    plan has no draft owners and no speculative state blocks: the layout is
    built from ``without_speculation(cfg)`` (a decode-only draft's prompt KV
    starts empty). With it, the producer's last stage runs prefill-only MTP
    at ``num_speculative_tokens=1`` (``tpu_runner._prefill_only_mtp``) and
    allocates the draft's owners like any QSA layer's: the layout is built
    from a copy of ``cfg`` whose speculative config is a shallow copy at
    ``PRODUCER_DRAFT_TOKENS``. This decode must then run the MTP draft that
    lands those pages. The producer never prefix-caches them (option (a),
    ``draft_prefix_cacheable=False``), as its own native_cache_specs does.
    """
    if not draft_kv_enabled():
        return without_speculation(cfg), 1, 1, False, True
    spec = getattr(cfg, "speculative_config", None)
    if spec is None or getattr(spec, "method", None) != "mtp":
        raise ValueError(
            "QWEN4_EXP_PD_DRAFT_KV=1 lands the producer's MTP draft KV in this "
            "decode's draft: the decode needs an MTP speculative config, got "
            f"{getattr(spec, 'method', None)!r}"
        )
    producer_spec = copy.copy(spec)
    object.__setattr__(producer_spec, "num_speculative_tokens", PRODUCER_DRAFT_TOKENS)
    out = copy.copy(cfg)
    object.__setattr__(out, "speculative_config", producer_spec)
    # Option (a): the producer's draft owners are not prefix-cacheable
    # (config.native_cache_specs on the producer, pd_prefill_mtp_producer).
    return out, 1, 1, False, False


def check_draft_kv(missing, dst_owners, main_dcp: int, main_dcp_owners) -> None:
    """QWEN4_EXP_PD_DRAFT_KV's refusals when a staged decode is built.

    The producer ships the MTP draft's caches, so the decode lands them like
    any QSA layer's. ``missing``: decode owners (and packed compressed
    members) with no producer pages; ``dst_owners``: the decode's cache
    owners; ``main_dcp`` / ``main_dcp_owners``: its layout's main-KV split.
    """
    import os

    if missing:
        raise ValueError(
            f"QWEN4_EXP_PD_DRAFT_KV=1: the producer's plan has no pages for "
            f"{sorted(missing)}; every decode owner, the MTP draft's included, "
            "must land from the producer"
        )
    if os.environ.get("QWEN4_EXP_PD_ZERO_DRAFT_KV", "0").strip() == "1":
        raise ValueError(
            "QWEN4_EXP_PD_ZERO_DRAFT_KV=1 would zero the draft KV that "
            "QWEN4_EXP_PD_DRAFT_KV=1 lands; unset one of them"
        )
    if any(name.endswith(".compressed_k") for name in main_dcp_owners):
        raise ValueError(
            "QWEN4_EXP_PD_DRAFT_KV=1 refuses VLLM_TPU_QSA_COMPRESSED_DCP=1: the "
            "staging repack lands compressed keys unsplit"
        )
    if main_dcp > 1:
        split = set(main_dcp_owners)
        unsplit = sorted(
            name
            for name in dst_owners
            if name.startswith("mtp.") and owner_kind(name) == "main" and name not in split
        )
        if unsplit:
            raise ValueError(
                f"QWEN4_EXP_PD_DRAFT_KV=1 on a main-KV DCP decode "
                f"(VLLM_TPU_QSA_MAIN_KV_DCP={main_dcp}) needs "
                f"VLLM_TPU_QSA_MTP_KV_DCP=1: the repack lands the draft's main KV "
                f"{unsplit} as one more DCP-split owner"
            )


def draft_group_problems(groups) -> list[str]:
    """q38 F2 option (a): the KV-cache groups of the producer's plan that
    break "the draft's groups are non-prefix-cacheable and hold draft owners
    only" (so the producer's hit Hp comes from the target groups alone).

    Both legs build the plan with the same helper (config.non_prefix_cacheable),
    so a break leaves them matched; it costs option (a)'s intent: the
    producer caches draft KV (its IDs), or a target group stops caching.
    Pending the CPU probe of how the pinned vLLM merges group specs."""
    problems = []
    for index, group in enumerate(groups):
        names = list(group.layer_names)
        draft = [n for n in names if n.startswith("mtp.")]
        if draft and (
            len(draft) != len(names) or getattr(group.kv_cache_spec, "prefix_cacheable", True)
        ):
            problems.append(f"group {index} {names}")
    return problems


def draft_hold_group_indices(groups) -> frozenset[int]:
    """q38 F2-HOLD: the KV-cache groups of an F2 producer's plan whose tables
    open with null blocks below each prefix hit (QWEN4_EXP_PD_DRAFT_HOLD;
    core/producer_draft_hold): the MTP draft's non-prefix-cacheable token-page
    groups (main KV, compressed keys), not its ring. Empty with the flag off,
    and on any engine whose draft is prefix-cacheable (a decode)."""
    if not (draft_hold_enabled() and draft_kv_enabled()):
        return frozenset()
    from vllm.v1.kv_cache_interface import CircularBufferSpec, FullAttentionSpec

    held = set()
    for index, group in enumerate(groups):
        names = list(group.layer_names)
        spec = group.kv_cache_spec
        # By the spec's own marker, not by layer names: on a pipeline producer
        # vLLM's scheduler config is stage 0's projection
        # (generate_scheduler_kv_cache_config copies worker 0's config) and
        # _project_kv_cache_groups_to_worker empties the names of every group
        # with no layer on that stage, the draft's (it lives on the last stage)
        # included. F2-HOLD gate H/HD (2026-11-08): name-based selection found no
        # held group there, so every held send was refused.
        if (
            getattr(type(spec), "q38_kv_bucket", None) == "q38-f2-producer-draft"
            and not getattr(spec, "prefix_cacheable", True)
            and isinstance(spec, FullAttentionSpec)
            and not isinstance(spec, CircularBufferSpec)
            and all(n.startswith("mtp.") for n in names)
        ):
            held.add(index)
    return frozenset(held)


def warn_draft_groups(groups, leg: str) -> None:
    problems = draft_group_problems(groups)
    if problems:
        logger.warning(
            "F2 option (a) on the %s: the producer plan's MTP draft groups are not "
            "non-prefix-cacheable draft-only groups (%s); the producer would cache "
            "draft KV or stop caching a target group (config.non_prefix_cacheable, "
            "pending the vLLM probe)",
            leg,
            "; ".join(problems),
        )


def main_tokens_np(num_tokens, dst_block, rank, start=0) -> np.ndarray:
    """The tokens ``main_rows_np`` / ``main_dst_np`` / ``pd_route.main_src_rows``
    list for ``rank``, in their order."""
    t = np.arange(start, num_tokens, dtype=np.int64)
    d = rank.main_dcp
    if d > 1:
        t = t[(t // dst_block) % d == rank.dcp_rank]
    return t


def pair_tokens_np(num_tokens, ratio, start=0) -> np.ndarray:
    """The first token of each compressed pair ``compressed_rows_np`` /
    ``compressed_dst_np`` / ``pd_route.compressed_src_rows`` list, in order."""
    span = 2 * ratio
    pairs = math.ceil(math.ceil(num_tokens / ratio) / 2)
    return np.arange(start // span, pairs, dtype=np.int64) * span


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
        if values.ndim == 2:
            # Routed GDN entries, [owners, staging pages]: repeat the last row
            # (an empty list rewrites the null page's rows).
            if len(values) == 0:
                values = np.zeros((1, values.shape[1]), dtype=np.int64)
            return np.concatenate(
                (values, np.repeat(values[-1:], size - len(values), axis=0))
            )
        if values.size == 0:
            values = np.zeros(1, dtype=np.int64)
        return np.concatenate(
            (values, np.full(size - len(values), values[-1], dtype=values.dtype))
        )
    values = values or [0]
    return values + [values[-1]] * (size - len(values))


# QWEN4_EXP_PD_REPACK_V2 (q38 XFER-4 F1): main/copy rows per program, by bucket.
REPACK_V2_BUCKETS = (1024, 2048, 4096, 8192, REPACK_CHUNK_ROWS)


def repack_v2_gdn_pages() -> int:
    """QWEN4_EXP_PD_REPACK_V2=N: GDN owners per GDN program (0, the default:
    ``repack_chunks``, today's programs)."""
    from vllm_torchtpu import envs

    return int(getattr(envs, "QWEN4_EXP_PD_REPACK_V2", 0) or 0)


def repack_programs_v2(
    main_s, main_d, copy_s, copy_d, gdn_s, gdn_d, gdn_pages: int,
    size=REPACK_CHUNK_ROWS, buckets=REPACK_V2_BUCKETS, tile=REPACK_TILE, prefer_fewer=True,
):
    """QWEN4_EXP_PD_REPACK_V2: one load's repack programs, fewer than
    ``repack_chunks`` makes and writing the same rows.

    ``repack_chunks`` runs max(main, copy, GDN) chunks of one fixed shape, so a
    decode with 36 GDN owners and 4 GDN pages a program lands even a warm hit
    in 9 programs, 8 of them padding 16K main and 16K copy rows with the null
    row. Here GDN owners go in programs of their own (``gdn_pages`` owners
    each, one tile of null main / copy rows), and main / copy rows in chunks
    of ``size`` rows, each padded only to the smallest bucket holding it, with
    one null GDN entry. A warm hit lands in 2 programs. A load whose main /
    copy rows already need at least as many chunks as its GDN owners would
    (a cold miss) keeps ``repack_chunks``'s programs, which carry the GDN
    pages inside those chunks (``prefer_fewer``): never more programs than
    today.

    Padding is ``_pad``'s (the last entry repeated, or the null page when a
    kind is empty), so every write is a real row, a duplicate of one, or a
    null-page row: the pool ends byte-identical to ``repack_chunks``'s.
    """
    programs = []
    n_gdn = len(gdn_s)
    if n_gdn:
        per = min(int(gdn_pages), n_gdn)
        none_main, none_copy = main_s[0:0], copy_s[0:0]
        for i in range(0, n_gdn, per):
            programs.append((
                _pad(none_main, tile), _pad(main_d[0:0], tile),
                _pad(none_copy, tile), _pad(copy_d[0:0], tile),
                _pad(gdn_s[i : i + per], per), _pad(gdn_d[i : i + per], per),
            ))
    rows = max(len(main_s), len(copy_s))
    for i in range(max(0 if programs else 1, -(-rows // size))):
        part = slice(i * size, (i + 1) * size)
        n = max(len(main_s[part]), len(copy_s[part]))
        b = next((x for x in buckets if x >= n), size)
        programs.append((
            _pad(main_s[part], b), _pad(main_d[part], b),
            _pad(copy_s[part], b), _pad(copy_d[part], b),
            _pad(gdn_s[0:0], 1), _pad(gdn_d[0:0], 1),
        ))
    if prefer_fewer:
        legacy = list(repack_chunks(main_s, main_d, copy_s, copy_d, gdn_s, gdn_d, size=size))
        if len(legacy) <= len(programs):
            return legacy
    return programs


def v2_unpack(buf, layout):
    """The six index arrays of a packed v2 program (``StagedConsumer._v2_pack``),
    by static slices: works on a numpy array or a torch tensor alike."""
    out, start = [], 0
    for shape in layout:
        n = 1
        for d in shape:
            n *= int(d)
        out.append(buf[start : start + n].reshape(tuple(int(d) for d in shape)))
        start += n
    return out


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
    num_tokens, src_pages, src_block, dst_pages, dst_block, rank, start=0, src_first=0
) -> tuple[np.ndarray, np.ndarray]:
    """``main_rows`` as int64 arrays, vectorized (a 100k-token load has a
    row per token per layer: the Python loop took ~0.7 s per request on the
    decode's worker thread, stalling its steps)."""
    t = np.arange(start, num_tokens, dtype=np.int64)
    d = rank.main_dcp
    if d > 1:
        t = t[(t // dst_block) % d == rank.dcp_rank]
    sp = np.asarray(src_pages, dtype=np.int64)
    dp = np.asarray(dst_pages, dtype=np.int64)
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
    start=0, src_first=0, dst_offset=0,
) -> tuple[np.ndarray, np.ndarray]:
    """``compressed_rows`` as int64 arrays, vectorized."""
    span = 2 * ratio
    if start % span:
        raise ValueError(f"A repack must start on a compressed pair ({span} tokens)")
    pairs = math.ceil(math.ceil(num_tokens / ratio) / 2)
    t = np.arange(start // span, pairs, dtype=np.int64) * span
    sp = np.asarray(src_pages, dtype=np.int64)
    dp = np.asarray(dst_pages, dtype=np.int64)
    src = sp[t // src_block - src_first] * src_block + (t % src_block) // span
    dst = dp[t // dst_block] * dst_block + dst_offset + (t % dst_block) // span
    return src, dst


def main_dst_np(num_tokens, dst_pages, dst_block, rank, start=0) -> np.ndarray:
    """``main_rows_np``'s destination rows alone (routed staging)."""
    t = np.arange(start, num_tokens, dtype=np.int64)
    d = rank.main_dcp
    if d > 1:
        t = t[(t // dst_block) % d == rank.dcp_rank]
    dp = np.asarray(dst_pages, dtype=np.int64)
    return dp[t // (d * dst_block)] * dst_block + t % dst_block


def compressed_dst_np(
    num_tokens, ratio, dst_pages, dst_block, start=0, dst_offset=0
) -> np.ndarray:
    """``compressed_rows_np``'s destination rows alone (routed staging)."""
    span = 2 * ratio
    if start % span:
        raise ValueError(f"A repack must start on a compressed pair ({span} tokens)")
    pairs = math.ceil(math.ceil(num_tokens / ratio) / 2)
    t = np.arange(start // span, pairs, dtype=np.int64) * span
    dp = np.asarray(dst_pages, dtype=np.int64)
    return dp[t // dst_block] * dst_block + dst_offset + (t % dst_block) // span


def route_geometry_for(cfg, src_layout, decode_layout, rank) -> "pd_route.RouteGeometry | None":
    """The routed geometry this decode rank (or a producer, with ``rank``
    describing its consumer) runs under QWEN4_EXP_PD_ROUTED_DST, or None.

    ``rank`` needs ``tp_size``, ``main_dcp``; ``decode_layout`` ``block_size``
    and ``token_bytes``. Both legs build it the same way; it is part of the
    layout fingerprint, so a disagreement fails at admission."""
    from vllm_torchtpu import envs

    parsed = pd_route.parse_route_env(envs.QWEN4_EXP_PD_ROUTED_DST)
    if parsed is None:
        return None
    tp, dcp, block = parsed
    if (tp, dcp, block) != (rank.tp_size, rank.main_dcp, decode_layout.block_size):
        raise ValueError(
            f"QWEN4_EXP_PD_ROUTED_DST={tp},{dcp},{block} disagrees with the decode "
            f"(tp={rank.tp_size}, dcp={rank.main_dcp}, block={decode_layout.block_size})"
        )
    row = ROW_SUBLANES * ROW_LANES
    if src_layout.token_bytes != row or decode_layout.token_bytes != row:
        raise ValueError("routed staging needs 1,024-byte rows on both legs")
    src_gdn = gdn_geometry(cfg, 1, src_layout.token_bytes)
    dst_gdn = gdn_geometry(cfg, tp, decode_layout.token_bytes)
    # stage_rows: pd_route.stage_rows_for (256 wherever routing was accepted
    # before; 128 for a TP4 decode's 896-row GDN extent).
    return pd_route.RouteGeometry.fitted(
        tp=tp,
        dcp=dcp,
        dst_block=block,
        src_block=src_layout.block_size,
        ratio=int(cfg.model_config.hf_text_config.indexer_compress_ratio),
        ssm_rows_src=src_gdn.ssm_tokens,
        conv_rows_src=src_gdn.conv_tokens,
        ssm_rows_dst=dst_gdn.ssm_tokens,
    )


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


# ---------------------------------------------------------------------------
# QWEN4_EXP_PD_REPACK_L2 (q38 L2 D74 "L2" rung): one descriptor-driven program per landing
# ---------------------------------------------------------------------------
# v2 lands a load as 2..39 compiled programs, each a per-row DMA gather (padding rows included), XLA head select /
# concat / GDN transposes, and a 64-row tiled writer: on the T95h bench its ready time is ~20 ns per row slot + ~20 ns
# per row DMA + ~0.8 ms per program (I from M, scratch/l2/land). L2 lands the same bytes as ONE fixed-shape Pallas
# program driven by a table of run descriptors: every contiguous run is power-of-two DMAs (1..256 rows), staging to the
# pool directly (HBM -> HBM) except main KV, which bounces through VMEM for its head select. No XLA touches the FP8
# bytes: they move by DMA and in-kernel ``pltpu.bitcast`` only.
#
# A descriptor is 4 int32: (source flat staging row, destination flat pool row, rows, kind). Row 0 of a table is the
# header (descriptor count). A table holds L2_DCAP - 1 descriptors; a larger load runs several tables through the same
# compiled program (destinations are disjoint, so order is free).
L2_DCAP = 1 << 14
L2_BLOCK = 256  # descriptors per SMEM block
L2_COPY, L2_HEAD0, L2_HEAD1, L2_FILL, L2_ZERO = 0, 1, 2, 3, 4
L2_KIND_NAMES = ("COPY", "HEAD0", "HEAD1", "FILL", "ZERO")
L2_FILL_ROWS = 64  # FILL replicates one staging row from a VMEM block of this many rows
L2_ZERO_ROWS = 16  # ZERO writes from a zero VMEM block of this many rows
L2_MAX_RUN = 256


def repack_l2_enabled() -> bool:
    """QWEN4_EXP_PD_REPACK_L2 (q38 L2 D74), read when a consumer is built."""
    from vllm_torchtpu import envs

    return bool(getattr(envs, "QWEN4_EXP_PD_REPACK_L2", False))


def l2_caps(kernel_rows: int) -> dict:
    """Largest run per kind: a DMA never crosses a pool kernel page (``pool.shape[1]`` rows)."""
    big = min(L2_MAX_RUN, int(kernel_rows))
    return {L2_COPY: big, L2_HEAD0: big, L2_HEAD1: big, L2_FILL: min(L2_FILL_ROWS, big),
            L2_ZERO: min(L2_ZERO_ROWS, big)}


def _l2_advances(kind):
    """COPY / HEAD sources advance with the destination; FILL (one replicated row) and ZERO do not."""
    kind = np.asarray(kind)
    return (kind == L2_COPY) | (kind == L2_HEAD0) | (kind == L2_HEAD1)


def l2_gdn_template(src_gdn, dst_gdn, channels):
    """One routed compact GDN owner's landing as whole-row runs, or None (not expressible: fall back to v2).

    ``make_repack``'s compact path writes ``[the staged SSM rows | _untyped(conv[:, channels])]``. The conv element
    stream (``_typed``) packs ``1024 / itemsize`` elements per 1 KiB row, row-locally, so when every output row's
    elements are one aligned row of the source stream, the channel select is whole-row copies. The served S12 decode
    (f32 conv; key 2,048, k_local 256, v_local 768, conv_dim 10,240 elements: all multiples of 256) is: SSM 1 run,
    conv 3 x (q 1 row, k 1 row, v 3 rows) and 1 zero pad row.
    Returns [(src compact row, dst row, rows, kind)] with kind L2_COPY or L2_ZERO (src -1)."""
    s, d = src_gdn, dst_gdn
    if s is None or d is None:
        return []
    if s.ssm_itemsize != d.ssm_itemsize or s.conv_itemsize != d.conv_itemsize or s.conv_width != d.conv_width:
        return None
    chan = np.asarray(channels, dtype=np.int64)
    if len(chan) != d.conv_dim:
        return None
    pr = ROW_SUBLANES * ROW_LANES // d.conv_itemsize
    total = d.conv_width * d.conv_dim
    if total % pr or total // pr > d.conv_tokens:
        return None
    j = np.arange(total, dtype=np.int64)
    elem = (j // d.conv_dim) * s.conv_dim + chan[j % d.conv_dim]
    out_rows = total // pr
    seg = elem.reshape(out_rows, pr)
    first = seg[:, 0]
    if (first % pr).any() or (seg != first[:, None] + np.arange(pr)).any():
        return None
    if (first // pr >= s.conv_tokens).any():
        return None
    src_rows = d.ssm_tokens + first // pr
    runs = [(0, 0, int(d.ssm_tokens), L2_COPY)] if d.ssm_tokens else []
    start = 0
    for q in range(1, out_rows + 1):
        if q == out_rows or src_rows[q] != src_rows[q - 1] + 1:
            runs.append((int(src_rows[start]), int(d.ssm_tokens + start), q - start, L2_COPY))
            start = q
    pad = d.conv_tokens - out_rows
    if pad:
        runs.append((-1, int(d.ssm_tokens + out_rows), int(pad), L2_ZERO))
    return runs


def _l2_split(runs, period: int, *, space: str, origin=0):
    """Split runs (rows of ``runs``: src, dst, n, kind, then any carried columns) so none crosses ``origin + k *
    period`` in ``space`` ("dst" or "src"). Order: by piece index within each input run (stable per run)."""
    runs = np.asarray(runs, dtype=np.int64)
    if runs.ndim != 2:
        runs = runs.reshape(-1, 4)
    if not len(runs):
        return runs
    out, cur = [], runs
    while len(cur):
        adv = _l2_advances(cur[:, 3])
        pos = (cur[:, 1] if space == "dst" else cur[:, 0]) - origin
        take = np.minimum(cur[:, 2], period - pos % period)
        piece = cur.copy()
        piece[:, 2] = take
        out.append(piece)
        nxt = cur.copy()
        nxt[:, 0] = np.where(adv, cur[:, 0] + take, cur[:, 0])
        nxt[:, 1] = cur[:, 1] + take
        nxt[:, 2] = cur[:, 2] - take
        cur = nxt[nxt[:, 2] > 0]
    return np.concatenate(out)


def l2_pieces(runs, kernel_rows: int):
    """Runs [k, 4] -> descriptor rows [m, 4]: none crosses a pool kernel page, none exceeds its kind's cap."""
    runs = _l2_split(np.asarray(runs, dtype=np.int64).reshape(-1, 4), int(kernel_rows), space="dst")
    out = []
    for kind, cap in l2_caps(kernel_rows).items():
        r = runs[runs[:, 3] == kind]
        if not len(r):
            continue
        reps = -(-r[:, 2] // cap)
        idx = np.repeat(np.arange(len(r)), reps)
        off = (np.arange(len(idx)) - np.repeat(np.cumsum(reps) - reps, reps)) * cap
        p = r[idx].copy()
        p[:, 0] += off * (1 if kind in (L2_COPY, L2_HEAD0, L2_HEAD1) else 0)
        p[:, 1] += off
        p[:, 2] = np.minimum(cap, r[idx, 2] - off)
        out.append(p)
    return np.concatenate(out) if out else np.zeros((0, 4), dtype=np.int64)


def _l2_rle(s, d, kinds):
    """Run-length encode (src, dst) row pairs of one group after dropping -1 padding and align_rows' repeats."""
    s, d, kinds = (np.asarray(x, dtype=np.int64) for x in (s, d, kinds))
    keep = (s >= 0) & (d >= 0)
    if len(s) > 1:
        dup = np.zeros(len(s), dtype=bool)
        dup[1:] = (s[1:] == s[:-1]) & (d[1:] == d[:-1])
        keep &= ~dup
    s, d, kinds = s[keep], d[keep], kinds[keep]
    if not len(s):
        return np.zeros((0, 4), dtype=np.int64)
    step = np.where(kinds == L2_FILL, 0, 1)
    brk = np.ones(len(s), dtype=bool)
    brk[1:] = (kinds[1:] != kinds[:-1]) | (d[1:] != d[:-1] + 1) | (s[1:] != s[:-1] + step[1:])
    starts = np.flatnonzero(brk)
    lens = np.diff(np.append(starts, len(s)))
    return np.stack([s[starts], d[starts], lens, kinds[starts]], axis=1)


def l2_gdn_runs(gdn_s, gdn_d, template, stage_rows: int, dst_block: int):
    """GDN owners' runs: ``template`` applied to each owner's staging pages (``gdn_s``: [owners, pages]) and
    destination page (``gdn_d``), split at staging page boundaries."""
    gdn_d = np.asarray(gdn_d, dtype=np.int64).reshape(-1)
    if not len(gdn_d) or not template:
        return np.zeros((0, 4), dtype=np.int64)
    gdn_s = np.asarray(gdn_s, dtype=np.int64).reshape(len(gdn_d), -1)
    t = np.asarray(template, dtype=np.int64).reshape(-1, 4)
    owner = np.repeat(np.arange(len(gdn_d)), len(t))
    tt = np.tile(t, (len(gdn_d), 1))
    zero = tt[:, 3] == L2_ZERO
    # columns: compact src row (0 for ZERO), dst row in page, n, kind, owner
    runs = np.stack([np.where(zero, 0, tt[:, 0]), tt[:, 1], tt[:, 2], tt[:, 3], owner], axis=1)
    runs = _l2_split(runs, int(stage_rows), space="src")
    c, r, n, kind, o = runs.T
    src = np.where(kind == L2_ZERO, -1, gdn_s[o, c // stage_rows] * stage_rows + c % stage_rows)
    return np.stack([src, gdn_d[o] * dst_block + r, n, kind], axis=1)


def l2_reference_runs(idx, kv_head: int, template, stage_rows: int, dst_block: int):
    """The reference builder: run-length encoding of ``StagedConsumer.indices()``'s row lists (routed compact GDN:
    ``gdn_s`` [owners, pages]). Main rows -> HEAD runs, copy rows -> COPY runs (source row 0, the staging null page
    v2 copies for draft-gap zero fill, -> FILL), GDN owners -> ``template``. The pool ends with the bytes v2 writes
    on every row v2 writes outside the null page (padding repeats and null-page writes are dropped)."""
    main_s, main_d, copy_s, copy_d, gdn_s, gdn_d = idx
    head = L2_HEAD0 if kv_head == 0 else L2_HEAD1
    main = _l2_rle(main_s, main_d, np.full(len(main_s), head))
    copy_s = np.asarray(copy_s, dtype=np.int64)
    copy = _l2_rle(copy_s, copy_d, np.where(copy_s == 0, L2_FILL, L2_COPY))
    gdn = l2_gdn_runs(gdn_s, gdn_d, template, stage_rows, dst_block)
    return np.concatenate([main, copy, gdn])


def l2_expand(desc):
    """Descriptor rows [m, 4] -> (dst rows, kind, src rows) per destination row, sorted by dst (tests, Gdesc)."""
    desc = np.asarray(desc, dtype=np.int64).reshape(-1, 4)
    n = desc[:, 2]
    i = np.repeat(np.arange(len(desc)), n)
    j = np.arange(len(i)) - np.repeat(np.cumsum(n) - n, n)
    kind = desc[i, 3]
    src = np.where(_l2_advances(kind), desc[i, 0] + j, np.where(kind == L2_ZERO, -1, desc[i, 0]))
    dst = desc[i, 1] + j
    o = np.argsort(dst, kind="stable")
    return dst[o], kind[o], src[o]


def l2_check_disjoint(desc) -> bool:
    """No destination row written twice (the kernel issues DMAs in any order)."""
    desc = np.asarray(desc, dtype=np.int64).reshape(-1, 4)
    if len(desc) < 2:
        return True
    o = np.argsort(desc[:, 1], kind="stable")
    d, n = desc[o, 1], desc[o, 2]
    return bool((d[:-1] + n[:-1] <= d[1:]).all())


def l2_tables(desc, dcap: int = L2_DCAP):
    """Descriptor rows -> int32 tables of ``dcap`` rows (row 0: [count, 0, 0, 0]), HEAD first."""
    desc = np.asarray(desc, dtype=np.int64).reshape(-1, 4)
    order = np.argsort(np.where((desc[:, 3] == L2_HEAD0) | (desc[:, 3] == L2_HEAD1), 0, 1 + desc[:, 3]),
                       kind="stable")
    desc = desc[order]
    per = dcap - 1
    tables = []
    for a in range(0, max(1, len(desc)), per):
        part = desc[a : a + per]
        t = np.zeros((dcap, 4), dtype=np.int32)
        t[0, 0] = len(part)
        t[1 : 1 + len(part)] = part
        tables.append(t)
    return tables


def l2_routed_runs(*, g, rank, src_pages, dst_tables, num_tokens: int, skip: int, draft_end: int,
                   dst_compressed: dict, dst_owner: dict, gdn_names, state_entry: int, dst_block: int, ratio: int,
                   template):
    """The production builder: one routed load's runs straight from its page lists, O(pages) (no per-row arrays).

    Mirrors ``StagedConsumer._routed_indices`` owner by owner (``pd_route.main_src_rows`` / ``compressed_src_rows`` /
    ``prefix_src_rows``, ``main_dst_np`` / ``compressed_dst_np``, the draft gap of ``_DraftGap``): a main-KV block of
    ``dst_block`` tokens is one destination run, cut at staging pages; a compressed destination page is one run of
    ``dst_block / (2 ratio)`` pairs; ring / PLE prefixes are one run per staging page; GDN owners take ``template``.
    Draft-owner (``mtp.``) rows of tokens below ``draft_end`` become FILL from staging row 0, as v2 copies them.
    Returns (runs [k, 4], draft rows landed, draft rows zeroed, (main rows, copy rows, GDN owners))."""
    S, B, span, N = int(g.stage_rows), int(dst_block), 2 * int(ratio), int(num_tokens)
    end = int(draft_end) if draft_end > skip else 0
    head = L2_HEAD0 if rank.kv_head == 0 else L2_HEAD1
    m = B // span if span and B % span == 0 else 0
    if B % S or not m or (S % m and m % S):
        raise ValueError("L2 needs staging pages, pool pages and compressed pages that nest")
    parts, landed, zeroed, n_main, n_copy = [], 0, 0, 0, 0

    def flat(pages, c):
        pg = np.asarray(pages, dtype=np.int64)
        return pg[c // S] * S + c % S

    # compressed keys: pairs [p0, P) of the load, one run per destination page, cut at staging pages and the draft gap
    if dst_compressed:
        P = math.ceil(math.ceil(N / ratio) / 2)
        p0 = skip // span
        c_skip = pd_route.skip_rows("compressed", skip, g)
        m = B // span
        for name, (gi, offset) in dst_compressed.items():
            if P <= p0:
                continue
            mtp = name.startswith("mtp.")
            pe = -(-end // span) if (mtp and end) else -1
            cuts = [np.array([p0, P]), np.arange(-(-p0 // m) * m, P, m), p0 + np.arange(0, P - p0, S)]
            if p0 < pe < P:
                cuts.append(np.array([pe]))
            cuts = np.unique(np.concatenate(cuts))
            a, ln = cuts[:-1], np.diff(cuts)
            src = flat(src_pages[name], a - c_skip)
            t = a * span
            dst = np.asarray(dst_tables[gi], dtype=np.int64)[t // B] * B + offset + (t % B) // span
            kind = np.full(len(a), L2_COPY)
            if mtp:
                below = a < pe
                kind = np.where(below, L2_FILL, L2_COPY)
                src = np.where(below, 0, src)
                zeroed += int(ln[below].sum())
                landed += int(ln[~below].sum())
            parts.append(np.stack([src, dst, ln, kind], axis=1))
            n_copy += int(ln.sum())
    dcp = int(g.dcp)
    for name, gi in dst_owner.items():
        kind_name = owner_kind(name)
        if kind_name == "compressed":
            continue
        dst = np.asarray(dst_tables[gi], dtype=np.int64)
        mtp = name.startswith("mtp.")
        if kind_name == "main":
            if rank.main_dcp != dcp:
                raise ValueError("L2: the rank's main-KV DCP differs from the route's")
            c_skip = pd_route.skip_rows("main", skip, g)
            ks = np.arange(skip // B + rank.tp_rank % dcp, -(-N // B), dcp, dtype=np.int64)
            if not len(ks):
                continue
            ln = np.minimum(B, N - ks * B)
            t0 = ks * B
            c0 = (ks // dcp) * B - c_skip
            d0 = dst[ks // dcp] * B
            if mtp and end:
                cut = np.clip(end - t0, 0, ln)  # rows below the gap end
                lo = cut > 0
                hi = cut < ln
                runs = np.concatenate([
                    np.stack([c0[lo], d0[lo], cut[lo], np.full(lo.sum(), L2_FILL), np.zeros(lo.sum(), np.int64)], 1),
                    np.stack([c0[hi] + cut[hi], d0[hi] + cut[hi], ln[hi] - cut[hi], np.full(hi.sum(), head),
                              np.ones(hi.sum(), np.int64)], 1),
                ])
                zeroed += int(cut.sum())
                landed += int((ln - cut).sum())
                n_copy += int(cut.sum())  # v2 lists gap rows with the copy rows
            else:
                runs = np.stack([c0, d0, ln, np.full(len(ks), head), np.ones(len(ks), np.int64)], 1)
                if mtp:
                    landed += int(ln.sum())
            # cut the HEAD runs at staging pages (compact space); FILL rows take staging row 0
            fill = runs[runs[:, 3] == L2_FILL]
            heads = _l2_split(runs[runs[:, 3] != L2_FILL], S, space="src")
            heads[:, 0] = flat(src_pages[name], heads[:, 0])
            fill[:, 0] = 0
            parts.append(np.concatenate([heads, fill])[:, :4])
            n_main += int(ln.sum()) - (int(np.clip(end - t0, 0, ln).sum()) if (mtp and end) else 0)
        elif kind_name in ("ring", "ple"):
            entry = int(dst[state_entry]) if kind_name == "ple" else int(dst[0])
            pr = int(g.prefix_rows)
            q = np.arange(0, pr, S, dtype=np.int64)
            parts.append(np.stack([flat(src_pages[name], q), entry * B + q, np.minimum(S, pr - q),
                                   np.full(len(q), L2_COPY)], 1))
            n_copy += pr
            if mtp:
                landed += pr
    n_gdn = getattr(g, "gdn_pages", None) or int(g.gdn_rows) // S
    if gdn_names:
        gdn_s = [list(src_pages[n][:n_gdn]) for n in gdn_names]
        gdn_d = [dst_tables[dst_owner[n]][state_entry] for n in gdn_names]
        parts.append(l2_gdn_runs(gdn_s, gdn_d, template, S, B))
    runs = np.concatenate(parts) if parts else np.zeros((0, 4), np.int64)
    return runs, landed, zeroed, (n_main, n_copy, len(gdn_names))


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


_GATHER_MARK: dict = {}


def dma_row_gather(stage, flat, *, tile: int = REPACK_TILE, interpret: bool = False, out_dtype=None):
    """``stage[flat]`` (``stage``: ``[rows, 4, 256]`` of the carrier's dtype), byte-exact.

    ``out_dtype`` (same element width, e.g. uint8) returns the rows already reinterpreted: the rows land by DMA
    in a VMEM scratch block of the carrier dtype, and the block is stored to the ``out_dtype`` output through a
    Mosaic vreg bitcast (``pltpu.bitcast``), a register reinterpretation. A uint8-typed ``stage`` (the carrier
    passed as a uint8 view) needs neither.
    On TPU an XLA FP8 -> uint8 ``bitcast_convert_type`` also canonicalised NaNs (nanprobe j-5a9f82e0 step g1),
    so the repack must never bitcast FP8 values to an integer type in XLA.

    One DMA per row, HBM to the output block: no vector unit and no XLA op ever
    holds the bytes as FP8 values. An XLA gather of the FP8-typed carrier on
    TPU canonicalised NaN encodings (0xFF -> 0x7F; the served rowdump riders
    j-48e7faaf / j-0b7ee68f, lane_xfer section 11.29). The carrier holds BF16
    compressed keys and FP32 GDN state, not FP8 values, so that flip halved
    real values. ``interpret`` runs the kernel in Pallas TPU interpret mode
    (CPU tests).
    """
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    n = int(flat.shape[0])
    if not _GATHER_MARK.get("dma"):  # engagement marker, once per process, at trace time
        _GATHER_MARK["dma"] = True
        print("Qwen PD staged repack gather: DMA row gather traced (dtype %s, backend %s)"
              % (stage.dtype, jax.default_backend()), flush=True)
    if n == 0:
        return jnp.zeros((0, *stage.shape[1:]), stage.dtype if out_dtype is None else out_dtype)
    pad = -n % tile
    idx = jnp.pad(flat.astype(jnp.int32), (0, pad))  # padding reads row 0, then dropped

    out_dtype = stage.dtype if out_dtype is None else jnp.dtype(out_dtype)
    if jnp.dtype(out_dtype).itemsize != jnp.dtype(stage.dtype).itemsize:
        raise ValueError("dma_row_gather reinterprets rows only between same-width dtypes")

    same = jnp.dtype(out_dtype) == jnp.dtype(stage.dtype)

    def kernel(idx_ref, stage_ref, out_ref, semaphore, *scratch):
        first = pl.program_id(0) * tile
        dst = out_ref if same else scratch[0]

        def row_copy(lane):
            return pltpu.make_async_copy(
                stage_ref.at[pl.ds(idx_ref[first + lane], 1)],
                dst.at[pl.ds(lane, 1)],
                semaphore,
            )

        for lane in range(tile):
            row_copy(lane).start()
        for lane in range(tile):
            row_copy(lane).wait()
        if not same:
            out_ref[...] = pltpu.bitcast(scratch[0][...], out_dtype)

    out = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=((n + pad) // tile,),
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
            out_specs=pl.BlockSpec(
                (tile, *stage.shape[1:]),
                lambda program, *_: (program,) + (0,) * (stage.ndim - 1),
            ),
            scratch_shapes=[pltpu.SemaphoreType.DMA]
            + ([] if same else [pltpu.VMEM((tile, *stage.shape[1:]), stage.dtype)]),
        ),
        out_shape=jax.ShapeDtypeStruct((n + pad, *stage.shape[1:]), out_dtype),
        name="qwen4_pd_staged_row_gather",
        **({"interpret": pltpu.InterpretParams()} if interpret else {}),
    )(idx, stage)
    return out[:n]


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
    compact_gdn: tuple[int, int] | None = None,
    gather=None,
):
    """``fn(pool, staging, main_src, main_dst, copy_src, copy_dst, gdn_src, gdn_dst)``.

    ``compact_gdn = (stage_rows, pages)`` (QWEN4_EXP_PD_ROUTED_DST): a GDN
    owner's staged state is this rank's SSM rows (its value heads, already in
    its own order) followed by the producer's whole conv rows, over ``pages``
    staging pages of ``stage_rows`` rows; ``gdn_src`` is then ``[owners,
    pages]`` staging page ids.

    Row indices are flat (``page * B + row``) into the staging carrier
    (``src``) and the pool (``dst``); ``gdn_*`` are page ids, one per GDN
    layer, in the same order. Returns the updated pool.

    ``fn.rows(staging, main_src, copy_src, gdn_src)`` returns only the rows to
    write (uint8 ``[n, 4, 256]`` each).

    ``gather(stage, flat)`` reads staging rows ``stage[flat]`` byte for byte; by
    default the DMA row gather (``dma_row_gather``) on TPU, where an XLA gather
    of the FP8-typed carrier canonicalised NaN bytes, and a plain gather
    elsewhere (bit-exact on CPU).

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

    if gather is None:
        if jax.default_backend() == "tpu":
            import functools

            gather = functools.partial(dma_row_gather, out_dtype=jnp.uint8)
            gather.__name__ = "dma_row_gather_u8"
        else:
            gather = rows_of
    print("Qwen PD staged repack gather selected: %s (jax backend %s)"
          % (getattr(gather, "__name__", "custom"), jax.default_backend()), flush=True)

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

    def gdn_page_compact(page, s: GdnGeometry, d: GdnGeometry):
        """Routed staging: ``[this rank's SSM rows | producer conv rows]``."""
        ssm = page[: d.ssm_tokens]
        conv = _typed(page[d.ssm_tokens : d.ssm_tokens + s.conv_tokens], s.conv_itemsize)
        conv = conv[: s.conv_width * s.conv_dim].reshape(s.conv_width, s.conv_dim)
        conv = conv[:, jnp.asarray(chan)].reshape(-1)
        return jnp.concatenate([ssm, _untyped(conv, d.conv_tokens, d.conv_itemsize)])

    has_gdn = src_gdn is not None and dst_gdn is not None
    d_rows = dst_gdn.ssm_tokens + dst_gdn.conv_tokens if has_gdn else 0
    if compact_gdn is not None and has_gdn:
        # The staged SSM rows are copied as-is: they must be exactly the
        # rank's value heads in its own layout (no padded row).
        if (
            src_gdn.ssm_itemsize != dst_gdn.ssm_itemsize
            or dst_gdn.ssm_tokens * ROW_SUBLANES * ROW_LANES
            != dst_gdn.v_heads * dst_gdn.head_k_dim * dst_gdn.head_v_dim * dst_gdn.ssm_itemsize
        ):
            raise ValueError("routed GDN staging needs whole-row per-rank SSM state")

    def rows(
        staging: jax.Array, main_src: jax.Array, copy_src: jax.Array, gdn_src: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        # Gather rows in the carrier's own dtype and bitcast only those: a
        # bitcast of the whole carrier materialized a copy of it per call.
        stage = staging.reshape(-1, ROW_SUBLANES, ROW_LANES)

        def gather_u8(flat):
            # The DMA gather returns uint8 rows itself (no FP8 -> uint8 XLA bitcast: it canonicalised NaNs on
            # TPU); a gather that returns the carrier dtype (the plain CPU gather, tests) is bitcast here.
            rows = gather(stage, flat)
            return rows if rows.dtype == jnp.uint8 else jax.lax.bitcast_convert_type(rows, jnp.uint8)

        # Main KV: the rank's head pair to slot 0, the other slots zero.
        pair = gather_u8(main_src)[:, 2 * kv_head : 2 * kv_head + 2]
        main = jnp.concatenate([pair, jnp.zeros_like(pair)], axis=1)
        # Compressed keys, rings + tags, PLE state: bytes unchanged.
        copies = gather_u8(copy_src)
        if has_gdn and compact_gdn is not None:
            s, d = src_gdn, dst_gdn
            stage_rows, _ = compact_gdn
            c_rows = d.ssm_tokens + s.conv_tokens
            flat = gdn_src[:, :, None] * stage_rows + jnp.arange(stage_rows)
            src_flat = flat.reshape(gdn_src.shape[0], -1)[:, :c_rows].reshape(-1)
            region = gather_u8(src_flat).reshape(-1, c_rows, ROW_SUBLANES, ROW_LANES)
            gdn = jax.vmap(lambda page: gdn_page_compact(page, s, d))(region)
            gdn = gdn.reshape(-1, ROW_SUBLANES, ROW_LANES)
        elif has_gdn:
            s, d = src_gdn, dst_gdn
            s_rows = s.ssm_tokens + s.conv_tokens
            src_flat = (gdn_src[:, None] * src_block + jnp.arange(s_rows)).reshape(-1)
            region = gather_u8(src_flat).reshape(-1, s_rows, ROW_SUBLANES, ROW_LANES)
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


def make_shared_scatter(write=None):
    """``fn(pool, gathered, src, dst)`` (QWEN4_EXP_PD_SHARED_COMPRESSED): write
    rows ``src`` of the landing gather's buffer (``[G * Q, stage_rows, 1, 4,
    256]`` of the pool dtype) to flat pool rows ``dst``, in place. No rank
    enters it as a constant: every rank runs the same program."""
    import jax
    import jax.numpy as jnp

    if write is None:
        import functools

        from vllm_torchtpu.kernels.qwen4_exp.pooled_qsa import (
            _write_main_cache_universal,
        )

        write = functools.partial(
            _write_main_cache_universal, tile=REPACK_TILE, overlap_edges=True
        )

    def fn(pool: jax.Array, gathered: jax.Array, src: jax.Array, dst: jax.Array) -> jax.Array:
        rows = gathered.reshape(-1, *pool.shape[2:])
        if rows.dtype != pool.dtype:
            rows = jax.lax.bitcast_convert_type(rows, pool.dtype)
        values = rows[src]
        flat = dst.astype(jnp.int32)
        valid = jnp.ones(flat.shape, jnp.bool_)
        tail = -len(flat) % REPACK_TILE  # the writer takes whole tiles
        values = jnp.pad(values, ((0, tail),) + ((0, 0),) * (values.ndim - 1))
        flat = jnp.pad(flat, (0, tail))
        valid = jnp.pad(valid, (0, tail))
        return write(pool, values, flat // pool.shape[1], flat, valid)

    return fn


def _l2_head_select(x, head: int, impl: str):
    """Rows ``x`` ((rows,) + row shape, the pool dtype; 4 sublanes = K0 V0 K1 V1 bytes at each lane) -> the rank's
    head pair in slot 0 and zeros after, exactly ``make_repack``'s ``concatenate([rows[:, 2h:2h+2], zeros])``.
    "u32": a value-level ``pltpu.bitcast`` packs the 4 sublanes at a lane into one 32-bit word, low byte first
    (``kernels/typed_ldst.py``; ``_main_slot_keep_mask``), so the select is ``w & 0xFFFF`` (head 0) or ``w >> 16``
    (head 1). "slice": the same bytes by a uint8 sublane slice and concatenate."""
    import jax
    import jax.numpy as jnp
    from jax.experimental.pallas import tpu as pltpu

    dtype = x.dtype
    if impl == "u32":
        w = pltpu.bitcast(x, jnp.uint32)
        w = (w & jnp.uint32(0xFFFF)) if head == 0 else jax.lax.shift_right_logical(w, jnp.uint32(16))
        return pltpu.bitcast(w, dtype)
    if impl == "slice":
        u = x if dtype == jnp.uint8 else pltpu.bitcast(x, jnp.uint8)
        pair = u[..., 2 * head : 2 * head + 2, :]
        y = jnp.concatenate([pair, jnp.zeros_like(pair)], axis=-2)
        return y if dtype == jnp.uint8 else pltpu.bitcast(y, dtype)
    raise ValueError(f"unknown L2 head select {impl!r}")


_L2_MARK: dict = {}


def make_repack_l2(*, dcap: int = L2_DCAP, block: int = L2_BLOCK, interpret: bool = False, head_impl: str = "u32"):
    """QWEN4_EXP_PD_REPACK_L2: ``fn(pool, staging, desc) -> (pool, done)``, one Pallas program for a whole landing.

    ``desc`` is an int32 ``[dcap, 4]`` table (``l2_tables``): row 0 ``[count, 0, 0, 0]``, then ``count`` descriptors
    (src flat staging row, dst flat pool row, rows, kind). The kernel streams the table through SMEM in blocks of
    ``block`` descriptors and issues each run as power-of-two DMAs (``compressed_write``'s pattern), all in flight,
    counted per size on per-size semaphores and drained at the end of the block:
      COPY   staging -> pool, HBM to HBM;
      ZERO   a zero VMEM block -> pool;
      FILL   one staging row (staging row 0 for the draft gap) replicated in a VMEM block -> pool;
      HEAD0 / HEAD1  staging -> VMEM (2 slots), head select (``_l2_head_select``), VMEM -> pool; a slot's writes
             drain before its next load, so one run's writes overlap the next run's load.
    The pool is aliased in place. ``done`` (int32 scalar) is the count, written last: a data dependency on the
    program's completion. One compiled shape for every load (``dcap`` fixed), so a decode compiles it once."""
    if dcap % block or dcap < 2:
        raise ValueError("L2 tables must be a whole number of SMEM blocks")

    def fn(pool, staging, desc):
        import functools

        import jax
        import jax.numpy as jnp
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu

        row_shape = tuple(int(x) for x in pool.shape[2:])
        kr = int(pool.shape[1])
        if kr & (kr - 1):
            raise ValueError(f"L2 needs power-of-two pool kernel pages, got {kr} rows")
        shift = kr.bit_length() - 1
        big = min(L2_MAX_RUN, kr)
        sizes = tuple(1 << b for b in range(big.bit_length() - 1, -1, -1))  # big .. 1
        ns = len(sizes)
        frows, zrows = min(L2_FILL_ROWS, big), min(L2_ZERO_ROWS, big)
        dt = pool.dtype
        if not _L2_MARK.get("trace"):  # engagement marker, once per process, at trace time
            _L2_MARK["trace"] = True
            print("Qwen PD staged repack L2: descriptor program traced (dcap %d, kernel rows %d, head %s, backend %s)"
                  % (dcap, kr, head_impl, jax.default_backend()), flush=True)
        stage = staging.reshape((-1,) + row_shape)
        flat_desc = desc.reshape(-1).astype(jnp.int32)
        blk4 = 4 * block
        c_copy, c_vmem, c_hout, c_hseq, c_fill = 0, ns, 2 * ns, 4 * ns, 4 * ns + 1
        nctr = 4 * ns + 2
        chunk = min(32, big)

        def zeros(n):
            z = jnp.zeros((n,) + row_shape, jnp.uint8)
            return z if dt == jnp.uint8 else pltpu.bitcast(z, dt)

        def kernel(desc_hbm, stage_hbm, _pool_in, pool_out, done_ref,
                   blk, ctr, csem, vsem, hin, hout, bsem, fsem, hbuf, fbuf, zbuf):
            for k in range(nctr):
                ctr[k] = 0
            zbuf[...] = zeros(zrows)
            first = pltpu.make_async_copy(desc_hbm.at[pl.ds(0, blk4)], blk, bsem)
            first.start()
            first.wait()
            n = blk[0]

            def drain(count_at, make):
                def body(_, c):
                    make().wait()
                    return c

                jax.lax.fori_loop(0, ctr[count_at], body, 0)
                ctr[count_at] = 0

            def copy_dma(s, size):
                return pltpu.make_async_copy(stage_hbm.at[pl.ds(0, size)], pool_out.at[0, pl.ds(0, size)], csem.at[s])

            def vmem_dma(s, size):
                return pltpu.make_async_copy(fbuf.at[pl.ds(0, size)], pool_out.at[0, pl.ds(0, size)], vsem.at[s])

            def hout_dma(sl, s, size):
                return pltpu.make_async_copy(
                    hbuf.at[sl, pl.ds(0, size)], pool_out.at[0, pl.ds(0, size)], hout.at[sl, s]
                )

            def issue(i, carry):
                src, dst, ln, kind = blk[4 * i], blk[4 * i + 1], blk[4 * i + 2], blk[4 * i + 3]
                page = jax.lax.shift_right_logical(dst, shift)
                off = dst & (kr - 1)

                def runs(limit, start):
                    for s, size in enumerate(sizes):
                        if size > limit:
                            continue

                        @pl.when((ln & size) != 0)
                        def _(s=s, size=size):
                            start(s, size, ln & ~(2 * size - 1))

                @pl.when(kind == L2_COPY)
                def _copy():
                    def start(s, size, done):
                        pltpu.make_async_copy(
                            stage_hbm.at[pl.ds(src + done, size)], pool_out.at[page, pl.ds(off + done, size)],
                            csem.at[s],
                        ).start()
                        ctr[c_copy + s] = ctr[c_copy + s] + 1

                    runs(big, start)

                def from_vmem(buf):
                    def start(s, size, done):
                        pltpu.make_async_copy(
                            buf.at[pl.ds(0, size)], pool_out.at[page, pl.ds(off + done, size)], vsem.at[s]
                        ).start()
                        ctr[c_vmem + s] = ctr[c_vmem + s] + 1

                    return start

                @pl.when(kind == L2_ZERO)
                def _zero():
                    runs(zrows, from_vmem(zbuf))

                @pl.when(kind == L2_FILL)
                def _fill():
                    @pl.when(ctr[c_fill] == 0)
                    def _build():
                        for r in range(frows):
                            pltpu.make_async_copy(stage_hbm.at[pl.ds(src, 1)], fbuf.at[pl.ds(r, 1)], fsem).start()
                        for r in range(frows):
                            pltpu.make_async_copy(stage_hbm.at[pl.ds(src, 1)], fbuf.at[pl.ds(r, 1)], fsem).wait()
                        ctr[c_fill] = 1

                    runs(frows, from_vmem(fbuf))

                @pl.when((kind == L2_HEAD0) | (kind == L2_HEAD1))
                def _head():
                    slot = ctr[c_hseq] & 1
                    for sl in (0, 1):

                        @pl.when(slot == sl)
                        def _(sl=sl):
                            for s, size in enumerate(sizes):
                                drain(c_hout + sl * ns + s, functools.partial(hout_dma, sl, s, size))

                            def load(s, size, done):
                                pltpu.make_async_copy(
                                    stage_hbm.at[pl.ds(src + done, size)], hbuf.at[sl, pl.ds(done, size)], hin.at[s]
                                ).start()

                            def loaded(s, size, done):
                                pltpu.make_async_copy(
                                    stage_hbm.at[pl.ds(0, size)], hbuf.at[sl, pl.ds(0, size)], hin.at[s]
                                ).wait()

                            runs(big, load)
                            runs(big, loaded)
                            nch = (ln + chunk - 1) // chunk
                            for h, kh in ((0, L2_HEAD0), (1, L2_HEAD1)):

                                @pl.when(kind == kh)
                                def _(h=h):
                                    def body(j, c):
                                        r = pl.multiple_of(j * chunk, chunk)
                                        hbuf[sl, pl.ds(r, chunk)] = _l2_head_select(
                                            hbuf[sl, pl.ds(r, chunk)], h, head_impl
                                        )
                                        return c

                                    jax.lax.fori_loop(0, nch, body, 0)

                            def store(s, size, done):
                                pltpu.make_async_copy(
                                    hbuf.at[sl, pl.ds(done, size)], pool_out.at[page, pl.ds(off + done, size)],
                                    hout.at[sl, s],
                                ).start()
                                ctr[c_hout + sl * ns + s] = ctr[c_hout + sl * ns + s] + 1

                            runs(big, store)

                    ctr[c_hseq] = ctr[c_hseq] + 1

                return carry

            def do_block(b, carry):
                @pl.when(b > 0)
                def _load():
                    cp = pltpu.make_async_copy(desc_hbm.at[pl.ds(b * blk4, blk4)], blk, bsem)
                    cp.start()
                    cp.wait()

                lo = jnp.where(b == 0, 1, 0)
                hi = jnp.minimum(block, n + 1 - b * block)
                jax.lax.fori_loop(lo, hi, issue, 0)
                for s, size in enumerate(sizes):
                    drain(c_copy + s, functools.partial(copy_dma, s, size))
                    if size <= frows:
                        drain(c_vmem + s, functools.partial(vmem_dma, s, size))
                return carry

            jax.lax.fori_loop(0, (n + block) // block, do_block, 0)
            for sl in (0, 1):
                for s, size in enumerate(sizes):
                    drain(c_hout + sl * ns + s, functools.partial(hout_dma, sl, s, size))
            done_ref[...] = jnp.full(done_ref.shape, n, jnp.int32)

        out_pool, done = pl.pallas_call(
            kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(1,),
                in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)] * 3,
                out_specs=[pl.BlockSpec(memory_space=pltpu.HBM), pl.BlockSpec((8, 128), lambda i: (0, 0))],
                scratch_shapes=[
                    pltpu.SMEM((blk4,), jnp.int32),
                    pltpu.SMEM((nctr,), jnp.int32),
                    pltpu.SemaphoreType.DMA((ns,)),
                    pltpu.SemaphoreType.DMA((ns,)),
                    pltpu.SemaphoreType.DMA((ns,)),
                    pltpu.SemaphoreType.DMA((2, ns)),
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA,
                    pltpu.VMEM((2, big) + row_shape, dt),
                    pltpu.VMEM((frows,) + row_shape, dt),
                    pltpu.VMEM((zrows,) + row_shape, dt),
                ],
            ),
            out_shape=[jax.ShapeDtypeStruct(pool.shape, dt), jax.ShapeDtypeStruct((8, 128), jnp.int32)],
            input_output_aliases={2: 0},
            name="qwen4_pd_staged_repack_l2",
            **({"interpret": pltpu.InterpretParams()} if interpret else {}),
        )(flat_desc, stage, pool)
        return out_pool, done[0, 0]

    fn.dcap = dcap
    return fn


@dataclasses.dataclass
class RepackPlan:
    """One load's landing, planned before any device work: the shared
    compressed-key gather passes (``pd_shared_compressed.GatherPass``), then
    the repack chunks (``repack_chunks``)."""

    passes: list
    chunks: list
    # (main, copy, GDN) row counts, for QWEN4_EXP_PD_LANDING_LOG (host only).
    rows: tuple = ()
    # QWEN4_EXP_PD_REPACK_L2: ``chunks`` are descriptor tables (``l2_tables``) for the one L2 program.
    l2: bool = False


class SharedLanding:
    """The landing gather's device side (QWEN4_EXP_PD_SHARED_COMPRESSED, TPU).

    Per pass: ``pack`` copies this rank's compact staging pages into the send
    buffer and the send buffer is all-gathered over the TP device group, in
    one compiled program per bucket (``GroupCoordinator.all_gather``, i.e.
    ``all_gather_into_tensor`` on the device group, the call the compiled
    forward makes; ``tpu_runner._tpu_pin_drafts_to_rank0``: eager collectives
    wedge the cores); then a donated scatter writes the gathered rows into the
    pool (``make_shared_scatter``). Every buffer is allocated here, before any
    collective (3eeddf339: a rank blocked on an allocation inside the
    collective hangs its group). ``selftest`` compiles every bucket and
    checks the gather on rank-tagged data; it is collective.
    """

    def __init__(self, staging: torch.Tensor, pool: torch.Tensor, group, geo):
        self.staging, self.pool, self.group, self.geo = staging, pool, group, geo
        q = pd_shared_compressed.GATHER_PAGE_BUCKETS[-1]
        page = tuple(staging.shape[1:])
        self.send = torch.empty((q,) + page, dtype=staging.dtype, device=staging.device)
        self.gathered = torch.empty(
            (geo.group * q,) + page, dtype=staging.dtype, device=staging.device
        )
        self._gathers: dict[int, object] = {}
        self._scatter = None

    @property
    def buffer_bytes(self) -> int:
        return (self.send.numel() + self.gathered.numel()) * self.send.element_size()

    def _gather_fn(self, bucket: int):
        fn = self._gathers.get(bucket)
        if fn is None:
            group = self.group

            @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
            def gather(staging, pack, send, gathered):
                send.copy_(staging[pack])
                rows = group.all_gather(send.view(torch.int8), dim=0)
                gathered.copy_(rows.view(gathered.dtype))
                return pack[0]

            fn = self._gathers[bucket] = gather
        return fn

    def _scatter_fn(self):
        if self._scatter is None:
            import jax
            from torch_tpu._internal import pallas

            fn = make_shared_scatter()

            # pallas.jax_op verifies that every argument is annotated (the
            # repack op below does the same); an unannotated function is
            # refused at registration (q38-xfer-tputest-s4-i1, j-56405f1a).
            def scatter_fn(
                pool: jax.Array, gathered: jax.Array, src: jax.Array, dst: jax.Array
            ) -> tuple[jax.Array, jax.Array]:
                return fn(pool, gathered, src, dst), src[0]

            op = pallas.jax_op("pallas::qwen4_pd_shared_scatter", scatter_fn, donate_argnums=(0,))
            op.register_fake(
                lambda pool, gathered, src, dst: (
                    torch.empty_like(pool),
                    torch.empty((), dtype=src.dtype, device=src.device),
                )
            )

            @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
            def run(pool, gathered, src, dst):
                new_pool, marker = op(pool, gathered, src, dst)
                pool.copy_(new_pool)
                return marker

            self._scatter = run
        return self._scatter

    def run(self, passes) -> None:
        device = self.pool.device
        g = self.geo.group

        def dev(values):
            return torch.from_numpy(np.ascontiguousarray(values, dtype=np.int32)).to(device)

        for p in passes:
            send, gathered = self.send[: p.bucket], self.gathered[: g * p.bucket]
            self._gather_fn(p.bucket)(self.staging, dev(p.pack), send, gathered)
            self._scatter_fn()(self.pool, gathered, dev(p.scatter_src), dev(p.scatter_dst))

    def warm(self) -> float:
        """Compile the gather of every bucket for the staging carrier's own
        shape; returns the seconds taken. Collective, read-only on the carrier.

        The self-test compiles the gather for its scratch source, so the first
        landing of each bucket would otherwise compile on the serving path
        (1.5-1.7 s per shared request in q38-xfer-tputest-s4-i1 run 3,
        j-de57096a, against about 6 ms for a routed load). Call it after the
        native manager has registered the carrier: it only reads page 0 (the
        null page), as every landing reads the carrier, and writes nothing but
        the landing's own send and gathered buffers."""
        import time

        g, device = self.geo.group, self.pool.device
        t0 = time.monotonic()
        for bucket in pd_shared_compressed.GATHER_PAGE_BUCKETS:
            pack = torch.zeros(bucket, dtype=torch.int32, device=device)
            self._gather_fn(bucket)(self.staging, pack, self.send[:bucket], self.gathered[: g * bucket])
        return time.monotonic() - t0

    def selftest(self, rank: int) -> list[int]:
        """Compile the gather and the scatter of every bucket and check the
        gather's bytes; returns the failing buckets. Collective.

        It never touches the staging carrier. KVCacheManager registers that
        tensor for raw transfers and refuses it once torch_tpu has re-backed
        its storage: the first version wrote rank tags into staging page 0
        through a uint8 view and restored it, and every decode rank's manager
        then rejected the carrier ("Tensor's leading (block) dimension does not
        match its base storage buffer", q38-xfer-tputest-s4-i1 run 2,
        j-c9b417d2). The gather's source here is a scratch bucket of pages
        built on the host; each scatter rewrites pool row 0 (the null page's,
        which nothing reads), as the repack op writes the pool."""
        g, device = self.geo.group, self.pool.device
        q = pd_shared_compressed.GATHER_PAGE_BUCKETS[-1]
        page = tuple(self.staging.shape[1:])
        host = torch.full((q,) + page, rank + 1, dtype=torch.uint8)
        source = host.view(self.staging.dtype).to(device)
        failures = []
        for bucket in pd_shared_compressed.GATHER_PAGE_BUCKETS:
            pack = torch.zeros(bucket, dtype=torch.int32, device=device)
            gathered = self.gathered[: g * bucket]
            self._gather_fn(bucket)(source, pack, self.send[:bucket], gathered)
            seen = gathered.view(torch.uint8)[:, 0].reshape(g * bucket, -1)[:, 0].cpu()
            want = torch.arange(1, g + 1, dtype=torch.uint8).repeat_interleave(bucket)
            if not torch.equal(seen, want):
                failures.append(bucket)
            rows = torch.zeros(g * bucket * self.geo.page_rows, dtype=torch.int32, device=device)
            self._scatter_fn()(self.pool, gathered, rows, rows)
        return failures


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


class _DraftGap:
    """One load's MTP draft rows of tokens below ``end`` (q38 F2 option (a)):
    the producer never computed them. ``split`` keeps them out of a draft
    owner's landing rows and collects their destinations, ``emit`` appends
    those as zero-fill rows (from staging row 0, the never-reserved null
    page) to the plan's copy rows. A destination is then written once in the
    plan, never landed and zeroed both. ``end`` 0: nothing changes."""

    def __init__(self, end: int):
        self.end = int(end)
        self.zero_rows: list[np.ndarray] = []
        self.landed = 0
        self.zeroed = 0

    def split(self, name, s, d, tokens):
        """``s``, ``d``: an owner's landing rows; ``tokens()``: their tokens,
        in the same order (main_tokens_np / pair_tokens_np)."""
        if not name.startswith("mtp."):
            return s, d
        if self.end <= 0:
            self.landed += len(d)
            return s, d
        t = tokens()
        if len(t) != len(d):
            raise AssertionError(f"draft gap of {name}: {len(t)} tokens for {len(d)} rows")
        below = t < self.end
        self.zero_rows.append(d[below])
        self.zeroed += int(below.sum())
        self.landed += int(len(d) - below.sum())
        return s[~below], d[~below]

    def count(self, name, rows: int) -> None:
        if name.startswith("mtp."):
            self.landed += int(rows)

    def emit(self, copy_s, copy_d) -> None:
        for d in self.zero_rows:
            copy_s.append(np.zeros(len(d), dtype=np.int64))
            copy_d.append(d)


class StagedConsumer:
    """One TP-T decode rank's staging carrier for TP1 pipeline-producer pages."""

    # QWEN4_EXP_PD_ROUTED_DST geometry (pd_route), set by __init__; None keeps
    # the broadcast path. A class default, so a consumer assembled without
    # __init__ (tests, tools) takes today's path instead of failing.
    route = None
    # QWEN4_EXP_PD_SHARED_COMPRESSED geometry and landing buffers (set by
    # __init__ / enable_shared); the same class defaults, for the same reason.
    shared = None
    landing = None
    # QWEN4_EXP_PD_DRAFT_KV (q38 F2), set by __init__; off for a consumer
    # assembled without it.
    draft_kv = False
    # QWEN4_EXP_PD_REPACK_V2, set by __init__; off (today's chunks) without it.
    _v2_gdn_pages = 0
    # QWEN4_EXP_PD_REPACK_L2 (q38 L2 D74), set by __init__; off for shells unless a test sets it.
    _repack_l2 = False
    _op_l2 = None

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
        # The producer (producer_geometry): TP1 stages, no main-KV DCP,
        # unpacked compressed keys. Without QWEN4_EXP_PD_DRAFT_KV it runs no
        # MTP draft: a decode-only draft (this engine's speculative_config)
        # adds draft owners and speculative state blocks the producer's plan
        # does not have. With it, the producer's prefill-only MTP (K = 1)
        # allocates and ships the draft's owners too.
        draft_kv = draft_kv_enabled()
        # Option (a)'s range zero-fill (draft_zero_end) runs only with it on.
        self.draft_kv = draft_kv
        layout_cfg, src_tp, src_dcp, src_pack, src_draft_cacheable = producer_geometry(cfg)
        layout_kwargs = dict(tp=src_tp, main_dcp=src_dcp, pack_compressed=src_pack)
        if not src_draft_cacheable:
            layout_kwargs["draft_prefix_cacheable"] = False
        self.src_layout, src_specs = pooled_cache_specs(layout_cfg, **layout_kwargs)
        from .config import ensure_kv_group_size_rule

        ensure_kv_group_size_rule()
        src_kv_groups = get_kv_cache_groups(cfg, dict(src_specs))
        if draft_kv:
            warn_draft_groups(src_kv_groups, "decode")
            if not cfg.cache_config.enable_prefix_caching:
                logger.warning(
                    "F2 option (a) zero-fills draft KV on every producer prefix hit "
                    "when the decode prefix cache is off (QWEN4_EXP_PD_DRAFT_KV=1 "
                    "without --enable-prefix-caching on this decode)"
                )
        self.src_groups = build_transfer_groups(src_kv_groups)
        self.src_block, self.dst_block = self.src_layout.block_size, decode_layout.block_size
        self.ratio = int(cfg.model_config.hf_text_config.indexer_compress_ratio)
        # QWEN4_EXP_PD_ROUTED_DST: the producer routes each rank only its
        # slice, into a dense compact space per rank (pd_route); staging pages
        # are then route.stage_rows rows, else one producer page each.
        self.route = route_geometry_for(cfg, self.src_layout, decode_layout, rank)
        stage_rows = self.route.stage_rows if self.route else self.src_block
        page_bytes = stage_rows * ROW_SUBLANES * ROW_LANES
        self.pages = max(2, staging_bytes // page_bytes)
        self.staging = torch.zeros(
            # The producer pools' carrier shape, one kernel page per manager
            # page: the layout fingerprint compares it (routed: per row).
            (self.pages, stage_rows, 1, ROW_SUBLANES, ROW_LANES),
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
        # uuid -> tokens of the load (QWEN4_EXP_PD_SHARED_COMPRESSED decides
        # per load whether its compressed keys are shared).
        self.tokens: dict[int, int] = {}
        # QWEN4_EXP_PD_SHARED_COMPRESSED (pd_shared_compressed): compressed
        # pairs land on one rank each and are all-gathered at landing; the
        # device side (SharedLanding) is built by enable_shared.
        self.shared = None
        self.landing = None
        if pd_shared_compressed.enabled():
            if self.route is None:
                raise ValueError(
                    "QWEN4_EXP_PD_SHARED_COMPRESSED lands into routed compact staging: "
                    "set QWEN4_EXP_PD_ROUTED_DST"
                )
            self.shared = pd_shared_compressed.geometry_for(self.route)
        from .pd_manifest import compressed_live_bytes, page_regions

        # The producer sends a compressed-key page's keys only; the carrier
        # declares the same region so both pool layouts agree. Routed staging
        # pages are dense: every byte of a page is live.
        compressed_live = None if self.route else compressed_live_bytes(self.src_layout)
        self.manifest = PoolManifest(
            BINDING_ALIASED_RAW,
            [self.staging],
            [
                PoolEntry(
                    tag=f"qwen4.g{group.group_index}.{name}",
                    layer_name=name,
                    storage_index=0,
                    base_offset_bytes=0,
                    block_stride_bytes=page_bytes,
                    num_blocks=self.pages,
                    regions=page_regions(name, page_bytes, compressed_live),
                    dtype_tag="qwen4-whole-fp8-page-v1",
                )
                for group in self.src_groups
                for name in group.layer_names
            ],
        )
        self._src_owner = {n: g.group_index for g in self.src_groups for n in g.layer_names}
        self._dst_owner = {n: g.group_index for g in self.dst_groups for n in g.layer_names}
        # Compressed-key owner -> (destination group, row offset in its page):
        # a packed member sits in its owner's group at slot * member rows.
        self._dst_compressed = compressed_destinations(
            decode_layout, self._dst_owner, self.dst_block, self.ratio
        )
        # Decode-only MTP: the draft layer's owners (and its packed compressed
        # slot) have no producer pages; its prompt KV starts empty (synthetic
        # acceptance does not read it). QWEN4_EXP_PD_DRAFT_KV: the producer
        # ships them, so no owner may be missing, and each lands like a
        # target QSA layer's (the draft's main KV DCP-split on a DCP decode).
        missing = (set(self._dst_owner) | set(self._dst_compressed)) - set(self._src_owner)
        if draft_kv:
            check_draft_kv(
                missing, self._dst_owner, rank.main_dcp,
                getattr(decode_layout, "main_dcp_owners", ()),
            )
        elif any(not n.startswith("mtp.") for n in missing):
            raise ValueError(f"Qwen PD staging lacks producer pages for {sorted(missing)}")
        # QWEN4_EXP_PD_ZERO_DRAFT_KV=1: the repack also writes zeros over the
        # draft's rows of the landed tokens (staging row 0, the never-reserved
        # null page, is zero). The scheduler skips zeroing blocks a load will
        # fill, but no load fills the draft's, so without this they hold
        # stale bytes and real (non-synthetic) MTP acceptance depends on what
        # the pages last held. A determinism instrument for identity tests;
        # synthetic acceptance never reads draft KV.
        import os

        self._zero_draft = {}
        if os.environ.get("QWEN4_EXP_PD_ZERO_DRAFT_KV", "0").strip() == "1":
            self._zero_draft = {
                n: (self._dst_compressed[n] if n in self._dst_compressed
                    else (self._dst_owner[n], 0))
                for n in sorted(missing)
            }
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
        channels = conv_channels(src_gdn, dst_gdn, rank) if src_gdn else [0]
        self._fn = make_repack(
            src_block=self.src_block,
            dst_block=self.dst_block,
            src_gdn=src_gdn,
            dst_gdn=dst_gdn,
            kv_head=rank.kv_head,
            v_head_start=rank.tp_rank * (dst_gdn.v_heads if dst_gdn else 0),
            channels=channels,
            compact_gdn=(
                (self.route.stage_rows, self.route.gdn_rows // self.route.stage_rows)
                if self.route
                else None
            ),
        )
        self._op = None
        # QWEN4_EXP_PD_REPACK_V2 (q38 XFER-4): GDN owners per GDN program; 0 = today's chunks.
        self._v2_gdn_pages = repack_v2_gdn_pages()
        # QWEN4_EXP_PD_REPACK_L2 (q38 L2 D74): one descriptor-driven program per landing (routed staging only).
        self._l2_gdn = l2_gdn_template(src_gdn, dst_gdn, channels) if self.route is not None else None
        self._repack_l2 = False
        if repack_l2_enabled():
            why = self._l2_unsupported()
            if why:
                logger.warning("QWEN4_EXP_PD_REPACK_L2 set but off on rank %d (%s): landing with %s",
                               rank.tp_rank, "; ".join(why), "v2" if self._v2_gdn_pages else "repack_chunks")
            else:
                self._repack_l2 = True
                self._l2_fn = make_repack_l2()
                logger.info("Qwen PD repack L2 rank%d: on (GDN template %d runs per owner, kernel page %d rows)",
                            rank.tp_rank, len(self._l2_gdn), int(pool.shape[1]))

    # --- allocation (identical on every rank of the engine) ---

    def _owner_counts(self, num_tokens: int, skip_tokens: int):
        if self.route is not None:
            # Compact pages per owner, identical on every rank (one destination
            # page list per tag serves the whole engine).
            owners = [pool.layer_name for pool in self.manifest.pools]
            src_pages = [g.page_count(num_tokens) for g in self.src_groups]
            shared = self._shared_load(num_tokens)
            counts = [
                self.shared.staged_pages(num_tokens, skip_tokens)
                if shared and owner_kind(name) == "compressed"
                else pd_route.staging_pages(
                    owner_kind(name), num_tokens, src_pages[self._src_owner[name]],
                    skip_tokens, self.route,
                )
                for name in owners
            ]
            return [0] * len(self.src_groups), owners, counts
        from .pd import skipped_source_pages

        first = [skipped_source_pages(g, skip_tokens) for g in self.src_groups]
        group_counts = [
            g.page_count(num_tokens) - f for g, f in zip(self.src_groups, first)
        ]
        if any(count <= 0 for count in group_counts):
            raise ValueError("A prefix-aware load must leave every group a suffix")
        owners = [pool.layer_name for pool in self.manifest.pools]
        counts = [group_counts[self._src_owner[name]] for name in owners]
        return first, owners, counts

    def pages_needed(self, uuid: int, num_tokens: int, *, skip_tokens: int = 0) -> int:
        """Staging pages ``reserve`` would take for this load (0 if it holds
        them already); raises as ``reserve`` does for an invalid prefix."""
        if uuid in self.reservations:
            return 0
        return sum(self._owner_counts(num_tokens, skip_tokens)[2])

    def reserve(
        self,
        uuid: int,
        num_tokens: int,
        tp_group=None,
        *,
        skip_tokens: int = 0,
        reduced_free=None,
        draft_zero_until: int = 0,
    ) -> bool:
        """Staging pages for one load. A prefix-aware load (``skip_tokens``:
        the decode-local prefix) stages each token group from the first source
        page not wholly inside that prefix; request-state pages stage whole.

        ``reduced_free`` (QWEN4_EXP_PD_BATCHED_ADMISSION): the pass's free map,
        already the MIN over the TP group (``stage3_admission.prepare``); used
        instead of this load's own all-reduce, and its allocated pages are
        zeroed in it as well, so the next load sees what its own all-reduce
        would have returned.

        ``draft_zero_until`` (q38 F2 option (a)): the producer's published
        prefix hit Hp; draft rows below it were never computed there, and the
        repack zero-fills those above the local prefix (``draft_zero_end``)."""
        if uuid in self.reservations:
            return True
        first, owners, counts = self._owner_counts(num_tokens, skip_tokens)
        if reduced_free is not None:
            free = reduced_free
        else:
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
            if reduced_free is not None:
                reduced_free[owner_pages] = 0
        self.reservations[uuid] = pages
        self.skips[uuid] = (skip_tokens, first)
        # Per-load token counts (shared loads' clips); created on first use so
        # a consumer assembled without __init__ (D-MK's admission tests) works.
        self.__dict__.setdefault("tokens", {})[uuid] = int(num_tokens)
        if draft_zero_until > 0:
            self.__dict__.setdefault("draft_zero", {})[uuid] = int(draft_zero_until)
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
        skip, first = self.skips.get(uuid, (0, [0] * len(self.src_groups)))
        if self.route is not None:
            shared = self._shared_load(getattr(self, "tokens", {}).get(uuid, 0))

            def rows(kind):
                if shared and kind == "compressed":
                    # Whole pages, the same pairs' share on every rank.
                    return self.shared.clip_pages(skip) * self.shared.page_rows
                return pd_route.skip_rows(kind, skip, self.route)

            return [
                rows(owner_kind(pool.layer_name)) * pd_route.ROW_BYTES
                for pool in self.manifest.pools
            ]
        return [
            first[self._src_owner[pool.layer_name]] * _live_page_bytes(pool)
            for pool in self.manifest.pools
        ]

    def release(self, uuid: int) -> dict[str, list[int]] | None:
        """Drop a load's reservation. Its pages return to the free map only
        after QWEN4_EXP_PD_STAGING_QUARANTINE_STEPS engine steps
        (``advance_quarantine``): the repack and the S4 landing gather that read
        them are only enqueued on the device when this runs, and the next
        load's bytes land through raw host-to-device copies that do not wait
        for those reads."""
        self.skips.pop(uuid, None)
        getattr(self, "tokens", {}).pop(uuid, None)
        self.__dict__.get("draft_zero", {}).pop(uuid, None)
        pages = self.reservations.pop(uuid, None)
        if pages is not None:
            held = [int(p) for owner_pages in pages.values() for p in owner_pages]
            steps = quarantine_steps()
            if steps > 0 and held:
                # Created on first use (consumers assembled without __init__).
                self.__dict__.setdefault("quarantine", []).append([steps, held])
            elif held:
                self.free[held] = 1
        return pages

    def advance_quarantine(self) -> int:
        """Once per engine step on every rank, before the step's repacks and
        admission: count every quarantined release down by one and return the
        pages whose count reaches zero to the free map. Every rank releases
        the same loads in the same named steps (L1), and reserve allocates only
        pages free on every rank either way. Returns the pages freed."""
        held = self.__dict__.get("quarantine")
        if not held:
            return 0
        freed = []
        for entry in held:
            entry[0] -= 1
            if entry[0] <= 0:
                freed.extend(entry[1])
        if freed:
            self.free[freed] = 1
            held[:] = [entry for entry in held if entry[0] > 0]
        return len(freed)

    def quarantined_pages(self) -> int:
        return sum(len(entry[1]) for entry in self.__dict__.get("quarantine", ()))

    # --- q38 F2 option (a): draft rows the producer never computed ---

    def draft_zero_end(self, uuid: int, num_tokens: int) -> int:
        """End of the MTP draft rows a load zero-fills, or 0 (flag off).

        The producer prefix-caches no draft KV, so it computed draft rows only
        from its prefix hit Hp on (``reserve(draft_zero_until=Hp)``). Above
        this decode's local prefix (``skip_tokens``, whose draft rows are its
        own) the repack zero-fills [skip, end) instead of landing the
        producer's unwritten rows. QWEN4_EXP_PD_DRAFT_KV_FORCE_GAP raises the
        end to that fraction of the load's tokens, down to a compressed pair
        (q38 F2-3's forced-gap arm)."""
        if not getattr(self, "draft_kv", False):
            return 0
        hit = self.__dict__.get("draft_zero", {}).get(uuid, 0)
        span = 2 * self.ratio
        forced = int(force_gap_fraction() * num_tokens) // span * span
        return min(max(hit, forced), int(num_tokens))

    def draft_stats(self) -> dict[str, int]:
        """Cumulative option (a) counters of this rank (F2-2 / F2-3 engagement):
        loads, loads with a zero-filled gap, gap tokens, sum of Hp - Hd, and
        the draft rows landed and zeroed by the repack."""
        return dict(self.__dict__.get("draft_counters", {}))

    def _count_draft(self, uuid: int, skip: int, zero_end: int) -> None:
        if not getattr(self, "draft_kv", False):
            return
        stats = self.__dict__.setdefault(
            "draft_counters",
            dict(loads=0, gap_loads=0, gap_tokens=0, hp_minus_hd=0, rows_landed=0, rows_zeroed=0),
        )
        hit = self.__dict__.get("draft_zero", {}).get(uuid, 0)
        landed, zeroed = getattr(self, "last_draft_rows", (0, 0))
        stats["loads"] += 1
        stats["hp_minus_hd"] += hit - skip
        if zero_end > skip:
            stats["gap_loads"] += 1
            stats["gap_tokens"] += zero_end - skip
        stats["rows_landed"] += landed
        stats["rows_zeroed"] += zeroed
        now = time.monotonic()
        if self.rank.tp_rank == 0 and now - self.__dict__.get("_draft_logged", -1e9) >= 60.0:
            self._draft_logged = now
            logger.info(
                "Qwen PD F2 draft KV rank0: last load Hp=%d Hd=%d zero-filled [%d, %d) "
                "draft rows landed %d zeroed %d | cumulative %s",
                hit, skip, skip, max(skip, zero_end), landed, zeroed, stats,
            )

    # --- the repack ---

    def indices(
        self, src_pages, dst_tables, num_tokens, skip_tokens=0, first=None, draft_zero_until=0
    ):
        """Row indices of the repack. With ``skip_tokens`` only tokens from
        there on are written (the destination's earlier rows are shared prefix
        pages); ``src_pages[owner]`` are that cache owner's staging pages and
        ``first[g]`` the source page group ``g``'s staging starts at.

        ``draft_zero_until`` (q38 F2 option (a), ``draft_zero_end``): the MTP
        draft's main-KV rows and compressed pairs of tokens in
        [skip_tokens, draft_zero_until) are zero-filled instead of landed."""
        if self.route is not None:
            return self._routed_indices(
                src_pages, dst_tables, num_tokens, skip_tokens, draft_zero_until
            )
        first = first or [0] * len(self.src_groups)
        main_s, main_d, copy_s, copy_d = [], [], [], []
        gdn_s, gdn_d = [], []
        gap = _DraftGap(draft_zero_until if draft_zero_until > skip_tokens else 0)
        for name, (gi, offset) in self._dst_compressed.items():
            s, d = compressed_rows_np(
                num_tokens, self.ratio, src_pages[name],
                self.src_block, dst_tables[gi], self.dst_block, start=skip_tokens,
                src_first=first[self._src_owner[name]], dst_offset=offset,
            )
            s, d = gap.split(name, s, d, lambda: pair_tokens_np(num_tokens, self.ratio, skip_tokens))
            copy_s.append(s)
            copy_d.append(d)
        for name, gi in self._dst_owner.items():
            kind = owner_kind(name)
            if kind == "compressed":
                continue
            src = src_pages[name]
            src_first = first[self._src_owner[name]]
            dst = dst_tables[gi]
            if kind == "main":
                s, d = main_rows_np(num_tokens, src, self.src_block, dst, self.dst_block,
                                    self.rank, start=skip_tokens, src_first=src_first)
                s, d = gap.split(
                    name, s, d,
                    lambda: main_tokens_np(num_tokens, self.dst_block, self.rank, skip_tokens),
                )
                main_s.append(s)
                main_d.append(d)
            elif kind in ("ring", "ple"):
                entry = dst[self._state_entry] if kind == "ple" else dst[0]
                s, d = page_prefix_rows(src[0], self.src_block, entry, self.dst_block,
                                        min(self.src_block, self.dst_block))
                copy_s.append(np.asarray(s, dtype=np.int64))
                copy_d.append(np.asarray(d, dtype=np.int64))
                gap.count(name, len(d))
        for name in self._gdn_names:
            gdn_s.append(src_pages[name][0])
            gdn_d.append(dst_tables[self._dst_owner[name]][self._state_entry])
        gap.emit(copy_s, copy_d)
        self.last_draft_rows = (gap.landed, gap.zeroed)
        for name, (gi, offset) in self._zero_draft.items():
            dst = dst_tables[gi]
            kind = owner_kind(name)
            if kind == "compressed":
                pairs = np.arange(skip_tokens // (2 * self.ratio),
                                  math.ceil(math.ceil(num_tokens / self.ratio) / 2))
                t = pairs * 2 * self.ratio
                d = (np.asarray(dst, dtype=np.int64)[t // self.dst_block] * self.dst_block
                     + offset + (t % self.dst_block) // (2 * self.ratio))
            elif kind == "main":  # the draft's main KV is not DCP-split
                t = np.arange(skip_tokens, num_tokens, dtype=np.int64)
                d = (np.asarray(dst, dtype=np.int64)[t // self.dst_block] * self.dst_block
                     + t % self.dst_block)
            else:  # ring: its one page's rows
                d = np.arange(self.dst_block, dtype=np.int64) + dst[0] * self.dst_block
            copy_s.append(np.zeros(len(d), dtype=np.int64))
            copy_d.append(d)

        def cat(parts):
            parts = [align_rows(p) for p in parts if len(p)]
            return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)

        return (
            cat(main_s), cat(main_d), cat(copy_s), cat(copy_d),
            np.asarray(gdn_s, dtype=np.int64), np.asarray(gdn_d, dtype=np.int64),
        )

    def _routed_indices(self, src_pages, dst_tables, num_tokens, skip_tokens, draft_zero_until=0):
        """``indices`` over routed compact staging: same destination rows in
        the same order as the broadcast layout, read from compact rows."""
        g, rank = self.route, self.rank
        main_s, main_d, copy_s, copy_d = [], [], [], []
        gdn_s, gdn_d = [], []
        gap = _DraftGap(draft_zero_until if draft_zero_until > skip_tokens else 0)

        def pairs():
            return pair_tokens_np(num_tokens, self.ratio, skip_tokens)

        # A shared load's compressed keys are written by the landing gather
        # (shared_passes), not by the repack. The gather runs first, so a
        # draft gap's zero rows (from the repack) overwrite what it wrote.
        shared = self._shared_load(num_tokens)
        for name, (gi, offset) in self._dst_compressed.items():
            d = compressed_dst_np(
                num_tokens, self.ratio, dst_tables[gi], self.dst_block, skip_tokens, offset
            )
            if shared:
                gap.split(name, d, d, pairs)
                continue
            s = pd_route.compressed_src_rows(num_tokens, skip_tokens, src_pages[name], g)
            s, d = gap.split(name, s, d, pairs)
            copy_s.append(s)
            copy_d.append(d)
        for name, gi in self._dst_owner.items():
            kind = owner_kind(name)
            if kind == "compressed":
                continue
            dst = dst_tables[gi]
            if kind == "main":
                s, d = gap.split(
                    name,
                    pd_route.main_src_rows(num_tokens, skip_tokens, rank.tp_rank, src_pages[name], g),
                    main_dst_np(num_tokens, dst, self.dst_block, rank, skip_tokens),
                    lambda: main_tokens_np(num_tokens, self.dst_block, rank, skip_tokens),
                )
                main_s.append(s)
                main_d.append(d)
            elif kind in ("ring", "ple"):
                entry = dst[self._state_entry] if kind == "ple" else dst[0]
                copy_s.append(pd_route.prefix_src_rows(src_pages[name], g))
                copy_d.append(entry * self.dst_block + np.arange(g.prefix_rows, dtype=np.int64))
                gap.count(name, g.prefix_rows)
        n_gdn = g.gdn_rows // g.stage_rows
        for name in self._gdn_names:
            gdn_s.append(list(src_pages[name][:n_gdn]))
            gdn_d.append(dst_tables[self._dst_owner[name]][self._state_entry])
        gap.emit(copy_s, copy_d)
        self.last_draft_rows = (gap.landed, gap.zeroed)

        def cat(parts):
            parts = [align_rows(p) for p in parts if len(p)]
            return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)

        return (
            cat(main_s), cat(main_d), cat(copy_s), cat(copy_d),
            np.asarray(gdn_s, dtype=np.int64).reshape(-1, n_gdn),
            np.asarray(gdn_d, dtype=np.int64),
        )

    def _shared_load(self, num_tokens: int) -> bool:
        """Whether this load's compressed keys go once per host."""
        shared = getattr(self, "shared", None)
        return shared is not None and shared.shared(num_tokens)

    def shared_passes(self, src_pages, dst_tables, num_tokens, skip_tokens=0):
        """The landing gather's passes of a shared load (else none): the same
        on every rank, since staging pages and decode tables are."""
        if not self._shared_load(num_tokens):
            return []
        tags = [
            pd_shared_compressed.SharedTag(
                tuple(int(x) for x in src_pages[name]),
                tuple(int(x) for x in dst_tables[gi]),
                offset,
            )
            for name, (gi, offset) in self._dst_compressed.items()
        ]
        return pd_shared_compressed.gather_passes(
            num_tokens, tags, self.dst_block, self.shared, skip_tokens=skip_tokens
        )

    def enable_shared(self, group, rank: int, selftest: bool = True) -> None:
        """Build the landing gather's buffers and compiled programs (TPU), and
        self-test the all-gather of every bucket. Collective: every rank of
        the TP group calls it at the same point of its startup."""
        if self.shared is None:
            return
        self.landing = SharedLanding(self.staging, self.pool, group, self.shared)
        if selftest:
            failures = self.landing.selftest(rank)
            if failures:
                raise RuntimeError(
                    f"QWEN4_EXP_PD_SHARED_COMPRESSED self-test: the TP all-gather "
                    f"returned wrong bytes for buckets {failures}"
                )

    def warm_shared(self) -> float | None:
        """``SharedLanding.warm`` when shared compressed keys are on (else None)."""
        if getattr(self, "landing", None) is None:
            return None
        return self.landing.warm()

    def plan_repack(self, uuid: int, dst_tables, num_tokens: int) -> RepackPlan:
        """Every index of one load's landing, in numpy, before any device work
        (and before any collective)."""
        if getattr(self, "_repack_l2", False):
            tables = self._l2_plan(uuid, dst_tables, num_tokens)
            if tables is not None:
                return RepackPlan(passes=[], chunks=tables, rows=self.last_plan_rows, l2=True)
        pages = self.reservations[uuid]
        skip, first = self.skips.get(uuid, (0, None))
        zero_end = self.draft_zero_end(uuid, num_tokens)
        idx = self.indices(pages, dst_tables, num_tokens, skip, first, draft_zero_until=zero_end)
        # Row counts for QWEN4_EXP_PD_LANDING_LOG's landing line (host only).
        self.last_plan_rows = (len(idx[0]), len(idx[2]), len(idx[4]))
        self._count_draft(uuid, skip, zero_end)
        passes = (
            self.shared_passes(pages, dst_tables, num_tokens, skip)
            if getattr(self, "route", None) is not None
            else []
        )
        v2 = getattr(self, "_v2_gdn_pages", 0)
        chunks = list(repack_programs_v2(*idx, v2) if v2 else repack_chunks(*idx))
        return RepackPlan(passes=passes, chunks=chunks, rows=self.last_plan_rows)

    def run_repack(self, plan: RepackPlan) -> None:
        if getattr(plan, "l2", False):
            self._run_l2(plan.chunks)
            return
        if plan.passes:
            if getattr(self, "landing", None) is None:
                raise RuntimeError("a shared load landed before enable_shared")
            self.landing.run(plan.passes)
        self._run_chunks(plan.chunks)

    def repack(self, uuid: int, dst_tables, num_tokens: int) -> None:
        self.run_repack(self.plan_repack(uuid, dst_tables, num_tokens))

    def _run_chunks(self, chunks):
        """Dispatch a load's programs; returns the last program's marker (a device scalar: the bench's ready)."""
        device = self.pool.device
        if getattr(self, "_v2_gdn_pages", 0):
            return self._run_programs_v2(chunks)

        if self._op is None:
            self._op = self._build_op()
        marker = None
        for chunk in chunks:
            args = [
                torch.from_numpy(np.ascontiguousarray(v, dtype=np.int32)).to(device)
                for v in chunk
            ]
            marker = self._op(self.pool, self.staging, *args)
        return marker

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

    # --- QWEN4_EXP_PD_REPACK_V2 (q38 XFER-4 F1 + F2) ---

    @staticmethod
    def _v2_layout(program) -> tuple:
        """A program's array shapes: its compiled run's static key."""
        return tuple(tuple(np.shape(v)) for v in program)

    @staticmethod
    def _v2_pack(program) -> np.ndarray:
        """F2: the program's six index arrays as one int32 buffer (one H2D)."""
        return np.concatenate(
            [np.ravel(np.ascontiguousarray(v, dtype=np.int32)) for v in program]
        )

    def _run_programs_v2(self, programs):
        device = self.pool.device
        ops = self.__dict__.setdefault("_ops_v2", {})
        marker = None
        for program in programs:
            layout = self._v2_layout(program)
            run = ops.get(layout)
            if run is None:
                run = ops[layout] = self._build_op_v2(layout)
            marker = run(self.pool, self.staging, torch.from_numpy(self._v2_pack(program)).to(device))
        return marker

    def _v2_jax_op(self):
        """The repack's aliased jax op for v2 programs (its own op name, so
        today's op is built exactly as before)."""
        op = self.__dict__.get("_jop_v2")
        if op is not None:
            return op
        import jax
        from torch_tpu._internal import pallas

        fn = self._fn

        # torch_tpu's jax_op requires every argument and the return annotated
        # (pallas._verify_signature), exactly as _build_op's repack_fn is.
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

        op = pallas.jax_op("pallas::qwen4_pd_staged_repack_v2", repack_fn, donate_argnums=(0,))
        op.register_fake(
            lambda pool, staging, main_src, *rest: (
                torch.empty_like(pool),
                torch.empty((), dtype=main_src.dtype, device=main_src.device),
            )
        )
        self._jop_v2 = op
        return op

    def _build_op_v2(self, layout):
        """One compiled run per program layout: it slices the packed buffer
        at static offsets and calls the aliased op (as ``_build_op``'s run)."""
        op = self._v2_jax_op()

        @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
        def run(pool, staging, buf):
            new_pool, marker = op(pool, staging, *v2_unpack(buf, layout))
            pool.copy_(new_pool)
            return marker

        return run

    def v2_null_programs(self) -> list:
        """Every program layout a v2 landing can use, with null indices only
        (staging row / page 0 to pool row / page 0, the null pages)."""
        n_gdn = len(self._gdn_names)
        cols = None
        if self.route is not None and n_gdn:
            # The null programs carry the GDN columns this tree's landing reads.
            # T10's RouteGeometry has gdn_pages = ceil(gdn_rows / stage_rows): a
            # TP2 producer's entry is 448 rows over 256-row pages, 2 columns.
            # The PP8 RouteGeometry has no gdn_pages; it requires whole-page GDN
            # extents and lands gdn_rows // stage_rows columns (the pre-T10 source).
            cols = getattr(self.route, "gdn_pages", None)
            if cols is None:
                if self.route.gdn_rows % self.route.stage_rows:
                    raise ValueError(
                        f"routed GDN entry of {self.route.gdn_rows} rows is not whole "
                        f"{self.route.stage_rows}-row staging pages and the route has no gdn_pages"
                    )
                cols = self.route.gdn_rows // self.route.stage_rows
        empty = np.zeros(0, dtype=np.int64)
        gdn_s = np.zeros((n_gdn, cols), dtype=np.int64) if cols else np.zeros(n_gdn, dtype=np.int64)
        gdn_d = np.zeros(n_gdn, dtype=np.int64)
        programs = repack_programs_v2(empty, empty, empty, empty, gdn_s, gdn_d, self._v2_gdn_pages)
        null_gdn_s, null_gdn_d = _pad(gdn_s[0:0], 1), _pad(gdn_d[0:0], 1)
        for b in REPACK_V2_BUCKETS:
            z = np.zeros(b, dtype=np.int64)
            programs.append((z, z, z, z, null_gdn_s, null_gdn_d))
        # A cold miss's programs are repack_chunks' (prefer_fewer).
        z = np.zeros(REPACK_CHUNK_ROWS, dtype=np.int64)
        programs.append((z, z, z, z, _pad(gdn_s[0:0], REPACK_GDN_PAGES), _pad(gdn_d[0:0], REPACK_GDN_PAGES)))
        seen, out = set(), []
        for program in programs:
            key = self._v2_layout(program)
            if key not in seen:
                seen.add(key)
                out.append(program)
        return out

    def warm_repack_v2(self) -> float | None:
        """Compile every v2 program layout before serving (None when v2 is off).
        Writes only the null pages, as a padded landing does."""
        if not getattr(self, "_v2_gdn_pages", 0):
            return None
        import time

        t0 = time.monotonic()
        self._run_programs_v2(self.v2_null_programs())
        return time.monotonic() - t0

    # --- QWEN4_EXP_PD_REPACK_L2 (q38 L2 D74): one descriptor-driven program per landing ---

    def _l2_unsupported(self) -> list:
        """Why this consumer cannot land with L2 (empty: it can). Checked once, at build."""
        why = []
        if self.route is None:
            why.append("needs QWEN4_EXP_PD_ROUTED_DST staging")
        if getattr(self, "shared", None) is not None:
            why.append("not with QWEN4_EXP_PD_SHARED_COMPRESSED")
        if getattr(self, "_zero_draft", None):
            why.append("not with QWEN4_EXP_PD_ZERO_DRAFT_KV")
        if getattr(self, "_l2_gdn", None) is None:
            why.append("GDN conv channels are not whole 1 KiB rows")
        kr = int(self.pool.shape[1])
        if kr & (kr - 1):
            why.append(f"pool kernel pages of {kr} rows")
        if tuple(self.pool.shape[-2:]) != (ROW_SUBLANES, ROW_LANES) or self.rank.kv_head > 1:
            why.append("needs [K0 V0 K1 V1] rows (2 KV heads)")
        return why

    def _l2_plan(self, uuid: int, dst_tables, num_tokens: int):
        """One load's L2 descriptor tables, or None (this load lands with v2; counted in ``l2_stats``)."""
        stats = self.__dict__.setdefault("l2_stats", dict(loads=0, fallbacks=0, descriptors=0, tables=0))
        skip, _ = self.skips.get(uuid, (0, None))
        zero_end = self.draft_zero_end(uuid, num_tokens)
        try:
            runs, landed, zeroed, rows = l2_routed_runs(
                g=self.route, rank=self.rank, src_pages=self.reservations[uuid], dst_tables=dst_tables,
                num_tokens=num_tokens, skip=skip, draft_end=zero_end, dst_compressed=self._dst_compressed,
                dst_owner=self._dst_owner, gdn_names=self._gdn_names, state_entry=self._state_entry,
                dst_block=self.dst_block, ratio=self.ratio, template=self._l2_gdn,
            )
            desc = l2_pieces(runs, int(self.pool.shape[1]))
            if not l2_check_disjoint(desc):
                raise ValueError("overlapping destination runs")
        except (ValueError, IndexError) as e:
            stats["fallbacks"] += 1
            if stats["fallbacks"] <= 5:
                logger.warning("Qwen PD repack L2 rank%d: load %d lands with v2 (%s)", self.rank.tp_rank, uuid, e)
            return None
        self.last_plan_rows = rows
        self.last_draft_rows = (landed, zeroed)
        self._count_draft(uuid, skip, zero_end)
        tables = l2_tables(desc)
        stats["loads"] += 1
        stats["descriptors"] += len(desc)
        stats["tables"] += len(tables)
        now = time.monotonic()
        if self.rank.tp_rank == 0 and now - self.__dict__.get("_l2_logged", -1e9) >= 60.0:
            self._l2_logged = now
            logger.info("Qwen PD repack L2 rank0: last load %d descriptors (%d rows) in %d table(s) | cumulative %s",
                        len(desc), int(desc[:, 2].sum()) if len(desc) else 0, len(tables), stats)
        return tables

    def _build_op_l2(self):
        """The L2 program as the runner's pool copies run it: an aliased jax op inside one torch.compile, its own
        code object (one compiled shape for every load; never shares ``_build_op_v2``'s recompile budget)."""
        import jax
        from torch_tpu._internal import pallas

        fn = self._l2_fn

        def repack_l2_fn(pool: jax.Array, staging: jax.Array, desc: jax.Array) -> tuple[jax.Array, jax.Array]:
            return fn(pool, staging, desc)

        op = pallas.jax_op(getattr(self, "_l2_op_name", "pallas::qwen4_pd_staged_repack_l2"), repack_l2_fn,
                           donate_argnums=(0,))
        op.register_fake(
            lambda pool, staging, desc: (
                torch.empty_like(pool),
                torch.empty((), dtype=torch.int32, device=desc.device),
            )
        )

        @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
        def run(pool, staging, desc):
            new_pool, marker = op(pool, staging, desc)
            pool.copy_(new_pool)
            return marker

        return run

    def _run_l2(self, tables):
        """Dispatch a load's L2 tables (one program each); returns the last marker (the descriptor count)."""
        if self._op_l2 is None:
            self._op_l2 = self._build_op_l2()
        device = self.pool.device
        marker = None
        for table in tables:
            marker = self._op_l2(
                self.pool, self.staging, torch.from_numpy(np.ascontiguousarray(table, dtype=np.int32)).to(device)
            )
        return marker

    @staticmethod
    def l2_null_table(dcap: int = L2_DCAP) -> np.ndarray:
        """One COPY of staging row 0 to pool row 0 (the null pages), as a null v2 program writes them."""
        return l2_tables(np.array([[0, 0, 1, L2_COPY]], dtype=np.int64), dcap)[0]

    def warm_repack_l2(self):
        """Compile the L2 program before serving (None when L2 is off). Writes only the null pages."""
        if not getattr(self, "_repack_l2", False):
            return None
        t0 = time.monotonic()
        self._run_l2([self.l2_null_table()])
        return time.monotonic() - t0
