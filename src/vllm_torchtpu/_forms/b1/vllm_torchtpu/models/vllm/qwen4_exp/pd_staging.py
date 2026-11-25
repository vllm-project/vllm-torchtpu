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
   pair (into head slots 0-1, or a paired layer's slots under
   QWEN4_EXP_PACK_MAIN_KV), compressed keys, rings and PLE state are re-blocked
   or copied, and GDN state takes the rank's SSM heads and conv channels;
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
import threading
import time
import types
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


def verify_split_keys_enabled() -> bool:
    """QWEN4_EXP_PD_VERIFY_SPLIT_KEYS (envs; read directly when this module is
    loaded by path, without the package)."""
    try:
        from vllm_torchtpu import envs
    except ImportError:
        import os

        value = os.environ.get("QWEN4_EXP_PD_VERIFY_SPLIT_KEYS", "").strip().lower()
        return value in ("1", "true", "yes", "on")
    return bool(envs.QWEN4_EXP_PD_VERIFY_SPLIT_KEYS)


# QWEN4_EXP_PD_VERIFY_SPLIT_KEYS: loads checked first, of each kind (without
# and with a local prefix), then one load in VERIFY_EVERY; split compressed
# owners per check; rows per read program (one compiled shape).
VERIFY_FIRST_LOADS = 8
VERIFY_EVERY = 64
VERIFY_OWNERS = 3
VERIFY_CHUNK_ROWS = 4096


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


def draft_kv_split_keys_enabled() -> bool:
    """QWEN4_EXP_PD_DRAFT_KV_SPLIT_KEYS (q38 BLSTACK-R; envs; read directly
    when this module is loaded by path, without the package)."""
    try:
        from vllm_torchtpu import envs
    except ImportError:
        import os

        value = os.environ.get("QWEN4_EXP_PD_DRAFT_KV_SPLIT_KEYS", "").strip().lower()
        return value in ("1", "true", "yes", "on")
    return bool(envs.QWEN4_EXP_PD_DRAFT_KV_SPLIT_KEYS)


def fast_admit() -> bool:
    """QWEN4_EXP_PD_FAST_ADMIT (envs; read directly when this module is loaded
    by path, without the package)."""
    try:
        from vllm_torchtpu import envs
    except ImportError:
        import os

        value = os.environ.get("QWEN4_EXP_PD_FAST_ADMIT", "").strip().lower()
        return value in ("1", "true", "yes", "on")
    return bool(envs.QWEN4_EXP_PD_FAST_ADMIT)


def gather_bulk_enabled() -> bool:
    """VLLM_TPU_PD_GATHER_BULK (q38 SV-GATHER): exactly ``"1"`` turns the
    landing's bulk-tile gather on; unset or any other value is off (envs; read
    directly when this module is loaded by path, without the package)."""
    try:
        from vllm_torchtpu import envs
    except ImportError:
        import os

        return os.environ.get("VLLM_TPU_PD_GATHER_BULK") == "1"
    return bool(envs.VLLM_TPU_PD_GATHER_BULK)


def staging_fold_enabled() -> bool:
    """VLLM_TPU_PD_STAGING_FOLD (q38 SV-GATHER PD-STAGING-FOLD): exactly
    ``"1"`` makes the landing write each gathered segment straight into the
    pool (``_write_u8_rows``); unset or any other value is off (envs; read
    directly when this module is loaded by path, without the package)."""
    try:
        from vllm_torchtpu import envs
    except ImportError:
        import os

        return os.environ.get("VLLM_TPU_PD_STAGING_FOLD") == "1"
    return bool(envs.VLLM_TPU_PD_STAGING_FOLD)


# q38 SV-GATHER-2 GBULK-LINE: per-rank, warm-cache-safe engagement lines of VLLM_TPU_PD_GATHER_BULK.
GBULK_BIND = "q38 PDGBULK bind rank=%d tp_rank=%d tp_size=%d pid=%d key=%r bulk=%d"
GBULK_MISMATCH = (
    "q38 PDGBULK MISMATCH rank=%d tp_rank=%d tp_size=%d pid=%d op=%s bind=%d traced=%s "
    "rows=%d bulk=%d"
)


def _gbulk_rank(rank) -> tuple[int, int, int, int]:
    """(rank, tp_rank, tp_size, pid) of the PD-GATHER-BULK lines: the torch.distributed rank when a
    process group is initialized (-1 otherwise), the consumer's RankSlice (pd_connector builds it from
    the worker's tp_rank and the TP size) and this process id."""
    import os

    world_rank = -1
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            world_rank = int(dist.get_rank())
    except Exception:  # noqa: BLE001 - a log field
        pass
    return world_rank, int(getattr(rank, "tp_rank", -1)), int(getattr(rank, "tp_size", -1)), os.getpid()


def gather_bulk_bind_line(rank, bulk: bool) -> str:
    """The PD-GATHER-BULK engagement line, printed by every StagedConsumer as it is built.

    pd_connector.register_runner builds one consumer on every decode TP worker at every engine start,
    with the compile cache warm or cold, so this line never depends on a trace (the repack programs
    trace, and print the SV-GATHER marker, only when they compile). A plain print in each process, with
    the rank in the line (never a once-per-node log). ``key`` is the raw VLLM_TPU_PD_GATHER_BULK value
    (None when unset) and ``bulk`` the decision the consumer's repack programs are built with. Log only:
    nothing here enters a program."""
    import os

    line = GBULK_BIND % (
        *_gbulk_rank(rank), os.environ.get("VLLM_TPU_PD_GATHER_BULK"), int(bool(bulk))
    )
    print(line, flush=True)
    return line


def gather_bulk_mismatch(consumer, op_name: str, before: tuple[int, int]) -> None:
    """The GBULK-LINE MISMATCH guard, run (Python only) while a repack program traces.

    Compares the DMA row gathers the trace took since ``before`` (``_gather_counts``) with the
    consumer's bind decision (``_gather_bulk``: the value its bind line printed and its op names
    carry). A trace without a DMA gather (the plain CPU gather, a custom gather) has nothing to
    compare. A bulk bind whose program took a per-row gather, or a per-row bind whose program took a
    bulk one, prints 'q38 PDGBULK MISMATCH ...' and raises: the program would not be the one its name
    and its bind line state."""
    rows, bulk = (now - was for now, was in zip(_gather_counts(), before))
    if rows + bulk == 0:
        return
    bind = bool(getattr(consumer, "_gather_bulk", False))
    traced = "mixed" if rows and bulk else ("1" if bulk else "0")
    if traced == str(int(bind)):
        return
    line = GBULK_MISMATCH % (
        *_gbulk_rank(getattr(consumer, "rank", None)), op_name, int(bind), traced, rows, bulk
    )
    print(line, flush=True)
    raise RuntimeError(line)


# q38 SV-GATHER-2 FOLD-LINE: the same per-rank, warm-cache-safe lines for VLLM_TPU_PD_STAGING_FOLD.
FOLD_BIND = "q38 PDFOLD bind rank=%d tp_rank=%d tp_size=%d pid=%d key=%r fold=%d"
FOLD_MISMATCH = (
    "q38 PDFOLD MISMATCH rank=%d tp_rank=%d tp_size=%d pid=%d op=%s bind=%d traced=%s "
    "concat=%d fold=%d"
)


def staging_fold_bind_line(rank, fold: bool) -> str:
    """The PD-STAGING-FOLD engagement line, printed by every StagedConsumer as it is built
    (gather_bulk_bind_line's pattern and rank fields): every decode TP worker, every engine start,
    compile cache warm or cold, no trace needed (the 'staging fold traced' marker prints only when
    the repack compiles). ``key`` is the raw VLLM_TPU_PD_STAGING_FOLD value (None when unset) and
    ``fold`` the decision the consumer's repack programs are built with. Log only."""
    import os

    line = FOLD_BIND % (
        *_gbulk_rank(rank), os.environ.get("VLLM_TPU_PD_STAGING_FOLD"), int(bool(fold))
    )
    print(line, flush=True)
    return line


def staging_fold_mismatch(consumer, op_name: str, before: tuple[int, int]) -> None:
    """The FOLD-LINE MISMATCH guard, run (Python only) while a repack program traces.

    Compares the landing writes the trace took since ``before`` (``_write_counts``: the default
    concatenated write, or the fold writer ``_write_u8_rows``) with the consumer's bind decision
    (``_staging_fold``: the value its bind line printed and its op names carry). A trace with
    neither (a custom ``write``) has nothing to compare. A difference, or a mix, prints
    'q38 PDFOLD MISMATCH ...' and raises."""
    concat, fold = (now - was for now, was in zip(_write_counts(), before))
    if concat + fold == 0:
        return
    bind = bool(getattr(consumer, "_staging_fold", False))
    traced = "mixed" if concat and fold else ("1" if fold else "0")
    if traced == str(int(bind)):
        return
    line = FOLD_MISMATCH % (
        *_gbulk_rank(getattr(consumer, "rank", None)), op_name, int(bind), traced, concat, fold
    )
    print(line, flush=True)
    raise RuntimeError(line)


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


def published_hit_tokens(params) -> int:
    """The producer's published prefix hit, in tokens; 0 when absent or
    invalid (an older producer, or no hit)."""
    value = params.get(PRODUCER_HIT_PARAM) if isinstance(params, dict) else None
    return value if type(value) is int and value > 0 else 0


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
    if any(
        name.endswith(".compressed_k") for name in main_dcp_owners
    ) and not draft_kv_split_keys_enabled():
        # The staged landing splits compressed keys for a decode-only draft
        # (whose owners have no producer pages); F2's landed draft keys and
        # their draft-gap zero-fill are not validated split.
        # QWEN4_EXP_PD_DRAFT_KV_SPLIT_KEYS=1 (q38 BLSTACK-R) admits them: every
        # landing planner splits each compressed owner by _split_rank, the
        # draft's included, and routes the draft gap through the same split
        # pair tokens (indices, _routed_indices, _routed_indices_shared_rows).
        raise ValueError(
            "QWEN4_EXP_PD_DRAFT_KV=1 refuses VLLM_TPU_QSA_COMPRESSED_DCP=1: split "
            "compressed keys land for a decode-only draft only"
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


def pair_tokens_np(num_tokens, ratio, start=0, dst_block=0, rank=None) -> np.ndarray:
    """The first token of each compressed pair ``compressed_rows_np`` /
    ``compressed_dst_np`` / ``pd_route.compressed_src_rows`` list, in order
    (with ``rank``: the pairs that rank owns, ``owned_pairs``)."""
    span = 2 * ratio
    pairs = math.ceil(math.ceil(num_tokens / ratio) / 2)
    return owned_pairs(np.arange(start // span, pairs, dtype=np.int64) * span, dst_block, rank)


def owned_pairs(t, dst_block, rank) -> np.ndarray:
    """The pairs (first tokens ``t``) a decode rank stores.

    Compressed keys split under VLLM_TPU_QSA_COMPRESSED_DCP (``rank`` given,
    ``rank.main_dcp`` > 1) follow the main-KV rule: pair rank ``(t // B) % D``
    writes the groups of its own ``B``-token blocks (``B = dst_block``;
    ``pooled_qsa.compressed_pairs`` with ``dcp``). ``B`` is a multiple of the
    pair span, so one rank holds a whole pair. ``rank`` None: every pair."""
    if rank is None or rank.main_dcp <= 1:
        return t
    return t[(t // dst_block) % rank.main_dcp == rank.dcp_rank]


def pair_dst_np(t, dst_pages, dst_block, span, dst_offset=0, rank=None) -> np.ndarray:
    """Destination rows of pairs ``t``: row ``(t % B) // span`` after the
    member's ``dst_offset`` in the page of ``t``'s table entry, which covers
    ``D * B`` tokens when split (``rank``, ``owned_pairs``), else ``B``
    (``pooled_qsa.compressed_pairs``)."""
    d = 1 if rank is None else max(1, rank.main_dcp)
    if d > 1 and dst_block % span:
        raise ValueError(f"split compressed keys need whole pairs in a {dst_block}-token block")
    dp = np.asarray(dst_pages, dtype=np.int64)
    return dp[t // (d * dst_block)] * dst_block + dst_offset + (t % dst_block) // span


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
# QWEN4_EXP_PD_REPACK_SHAPE replaces the three sizes (parse_repack_shape).
# Sizes count written rows. A paired main-KV row (QWEN4_EXP_PACK_MAIN_KV)
# gathers two staging rows, so a program's main gather reads 32k rows; the
# gather's scalar prefetch is its own kernel's (4 B per gathered row).
REPACK_CHUNK_ROWS = 1 << 14
# Rows per scatter tile. Every run of contiguous destination rows starts on a
# tile (each layer's segment is padded to a multiple), so a tile is one DMA:
# main KV runs a page (512 rows), compressed keys 64 rows per decode page.
REPACK_TILE = 64
REPACK_GDN_PAGES = 4
# Distinct (tokens, prefix) owner counts a QWEN4_EXP_PD_FAST_ADMIT consumer
# keeps; the memo starts over when full.
_COUNTS_MEMO_ENTRIES = 4096


def parse_repack_shape(value: str | None) -> tuple[int, int, int]:
    """Main-KV rows, copied rows and GDN pages per repack program
    (QWEN4_EXP_PD_REPACK_SHAPE: ``"main,copy,gdn"``; empty is 16384,16384,4).

    ``":"`` separates too: the recipes' comma-separated ``*_EXTRA_SERVER_ENV``
    lists cannot carry a comma inside a value. Row counts are whole scatter
    tiles, so every segment of a later chunk still starts on a tile.
    """
    value = (value or "").strip()
    if not value:
        return REPACK_CHUNK_ROWS, REPACK_CHUNK_ROWS, REPACK_GDN_PAGES
    try:
        main, copied, gdn = (int(x) for x in value.replace(":", ",").split(","))
    except ValueError:
        main = copied = gdn = 0
    if min(main, copied, gdn) <= 0 or main % REPACK_TILE or copied % REPACK_TILE:
        raise ValueError(
            "QWEN4_EXP_PD_REPACK_SHAPE must be 'main,copy,gdn' with main and copy "
            f"positive multiples of {REPACK_TILE} and gdn positive, got {value!r}"
        )
    return main, copied, gdn


def bucket(n: int, floor: int = 1024) -> int:
    """Padded index length: few distinct compiled shapes."""
    return max(floor, 1 << max(0, n - 1).bit_length())


def repack_chunks(
    main_s, main_d, copy_s, copy_d, gdn_s, gdn_d,
    size=REPACK_CHUNK_ROWS, gdn_pages=REPACK_GDN_PAGES, copy_size=None, split_gdn=False,
    null_rows=None,
):
    """Padded index lists per repack program, each of one fixed shape.

    A chunk with nothing left of a kind rewrites the null page (row 0, page 0
    on both sides) in its place. A chunk holds ``size`` main-KV rows and
    ``copy_size`` (default ``size``) copied rows. With ``split_gdn`` the first
    chunk carries every GDN page and later chunks carry ``None`` for both GDN
    lists: they run the repack built without GDN state. ``null_rows = (staging
    page rows, pool page rows)`` pads main-KV and copied rows with distinct
    null-page rows (``_spread``) in place of the last pair repeated.
    """
    copy_size = size if copy_size is None else copy_size
    if split_gdn and len(gdn_s) > gdn_pages:
        raise ValueError(
            f"split_gdn needs every GDN page in one chunk: {len(gdn_s)} > {gdn_pages}"
        )
    chunks = max(
        1,
        -(-len(main_s) // size),
        -(-len(copy_s) // copy_size),
        -(-len(gdn_s) // gdn_pages),
    )
    for i in range(chunks):
        part = slice(i * size, (i + 1) * size)
        copied = slice(i * copy_size, (i + 1) * copy_size)
        pages = slice(i * gdn_pages, (i + 1) * gdn_pages)
        if split_gdn and i:
            gdn = (None, None)
        else:
            gdn = (_pad(gdn_s[pages], gdn_pages), _pad(gdn_d[pages], gdn_pages))
        if null_rows is None:
            rows = (
                _pad(main_s[part], size), _pad(main_d[part], size),
                _pad(copy_s[copied], copy_size), _pad(copy_d[copied], copy_size),
            )
        else:
            src_rows, dst_rows = null_rows
            rows = (
                _spread(main_s[part], size, src_rows),
                _spread(main_d[part], size, dst_rows),
                _spread(copy_s[copied], copy_size, src_rows),
                _spread(copy_d[copied], copy_size, dst_rows),
            )
        yield (*rows, *gdn)


def _spread(values, size: int, rows: int):
    """Pads a row list to ``size`` with null-page rows 0, 1, ..., ``rows`` - 1 in turn.

    Flat rows below ``rows`` are page 0 on either side: the staging carrier's
    page 0 is never reserved and stays zero, and nothing reads the pool's
    page 0, so the padding copies zeros onto rows no one reads. DMAs that all
    hit one row run one after another while distinct rows overlap (a
    padding-only program at 16384,16384,4: 3.4 ms with one row, 1.5 ms with
    distinct rows, W1b probe j-78bcecf2); padded destinations also form
    whole contiguous tiles (segments start on a tile and ``rows`` is a
    multiple of it), which the writer stores in one DMA each. Paired main-KV
    sources (``[n, 2]``, ``main_sources``) pad every column with the same
    rows: each column's gather then reads distinct rows too.
    """
    values = np.asarray(values, dtype=np.int64)
    pad = np.arange(size - len(values), dtype=np.int64) % rows
    if values.ndim == 2:
        pad = np.repeat(pad[:, None], values.shape[1], axis=1)
    return np.concatenate((values, pad))


def repack_chunks_packed(
    main_s, main_d, copy_s, copy_d, gdn_s, gdn_d,
    size=REPACK_CHUNK_ROWS, gdn_pages=REPACK_GDN_PAGES, copy_size=None, split_gdn=False,
    null_rows=None,
):
    """``repack_chunks``' chunks as ``pack_index`` vectors and their shapes
    (QWEN4_EXP_PD_FAST_ADMIT), each list written once into its chunk's int32
    vector and padded in place: main-KV and copied rows as ``_spread`` pads them
    when ``null_rows`` is given (else as ``_pad``), GDN lists as ``_pad``.
    Paired main-KV sources (``[n, 2]``) keep their shape."""
    copy_size = size if copy_size is None else copy_size
    if split_gdn and len(gdn_s) > gdn_pages:
        raise ValueError(
            f"split_gdn needs every GDN page in one chunk: {len(gdn_s)} > {gdn_pages}"
        )
    chunks = max(
        1,
        -(-len(main_s) // size),
        -(-len(copy_s) // copy_size),
        -(-len(gdn_s) // gdn_pages),
    )
    src_rows, dst_rows = null_rows if null_rows is not None else (None, None)
    for i in range(chunks):
        parts = [
            (main_s, i * size, size, src_rows), (main_d, i * size, size, dst_rows),
            (copy_s, i * copy_size, copy_size, src_rows),
            (copy_d, i * copy_size, copy_size, dst_rows),
        ]
        if not (split_gdn and i):
            parts += [
                (gdn_s, i * gdn_pages, gdn_pages, None), (gdn_d, i * gdn_pages, gdn_pages, None),
            ]
        shapes = tuple((n, *np.shape(values)[1:]) for values, _, n, _ in parts)
        host = np.empty(sum(math.prod(shape) for shape in shapes), dtype=np.int32)
        pos = 0
        for (values, start, n, spread), shape in zip(parts, shapes):
            seg = host[pos : pos + math.prod(shape)].reshape(shape)
            piece = values[start : start + n]
            k = len(piece)
            seg[:k] = piece
            if spread is not None:
                pad = np.arange(n - k, dtype=np.int64) % spread  # _spread's null-page rows
                seg[k:] = pad.reshape((-1,) + (1,) * (len(shape) - 1))
            elif k:
                seg[k:] = piece[-1]  # _pad: the last row repeated
            else:
                seg[:] = 0  # _pad's null page / null row
            pos += math.prod(shape)
        yield host, shapes


def pack_index(lists) -> tuple[np.ndarray, tuple]:
    """One repack chunk's index lists as one int32 vector, and their shapes
    (``_run_chunks_packed``; ``unpack_index`` inverts it)."""
    shapes = tuple(tuple(np.shape(v)) for v in lists)
    host = np.empty(sum(math.prod(shape) for shape in shapes), dtype=np.int32)
    pos = 0
    for values, shape in zip(lists, shapes):
        n = math.prod(shape)
        host[pos : pos + n] = np.asarray(values).reshape(-1)
        pos += n
    return host, shapes


def unpack_index(index, shapes) -> list:
    """The repack's six index arguments from a ``pack_index`` vector: its
    lists in order, then (four lists: a chunk without GDN state) a
    one-element slice for each GDN argument, which that repack never reads."""
    args, pos = [], 0
    for shape in shapes:
        n = math.prod(shape)
        args.append(index[pos : pos + n].reshape(shape))
        pos += n
    if len(args) == 4:
        args += [index[:1], index[:1]]
    return args


def unpacked_chunk(chunk) -> tuple:
    """A ``repack_chunks`` tuple from a ``repack_chunks_packed`` chunk (other
    chunks unchanged): the lists as numpy views, ``None`` for a chunk's
    missing GDN lists."""
    if len(chunk) != 2:
        return chunk
    host, shapes = chunk
    lists = tuple(unpack_index(host, shapes))
    return lists if len(shapes) == 6 else lists[:4] + (None, None)


def _pad(values, size: int):
    # Repeat the last pair: duplicate writes of identical values are no-ops.
    # A rank can own no rows (a DCP rank of a request shorter than its page);
    # row 0 is the null page's on both sides, which nothing reads.
    if isinstance(values, np.ndarray):
        if values.ndim == 2:
            # Routed GDN entries, [owners, staging pages], or paired main-KV
            # sources, [rows, members]: repeat the last row (an empty list
            # rewrites the null page's rows).
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
    start=0, src_first=0, dst_offset=0, src_page_rows=None, rank=None,
) -> tuple[np.ndarray, np.ndarray]:
    """``compressed_rows`` as int64 arrays, vectorized. ``src_page_rows``
    (default ``src_block``): rows of a source page, when one source page holds
    the keys of ``src_block`` tokens in fewer rows (merged producer keys).
    ``rank``: a DCP-split destination owner (VLLM_TPU_QSA_COMPRESSED_DCP); only
    the rank's own pairs land, at their split rows (``owned_pairs``,
    ``pair_dst_np``)."""
    span = 2 * ratio
    if start % span:
        raise ValueError(f"A repack must start on a compressed pair ({span} tokens)")
    t = pair_tokens_np(num_tokens, ratio, start, dst_block, rank)
    sp = np.asarray(src_pages, dtype=np.int64)
    page_rows = src_block if src_page_rows is None else src_page_rows
    src = sp[t // src_block - src_first] * page_rows + (t % src_block) // span
    return src, pair_dst_np(t, dst_pages, dst_block, span, dst_offset, rank)


def compressed_rows_split_np(
    num_tokens, ratio, src_pages_by_rank, src_block, dst_pages, dst_block,
    start=0, src_first=0, dst_offset=0, rank=None,
) -> tuple[np.ndarray, np.ndarray]:
    """``compressed_rows_np`` for keys split over ``P`` producer ranks by page
    owner: a main-KV ID spans ``P`` pages of ``src_block`` tokens and rank
    ``(t // src_block) % P`` holds page ``t // src_block``'s keys, at the start
    of that rank's staged page for the ID. ``rank`` as in ``compressed_rows_np``."""
    span = 2 * ratio
    if start % span:
        raise ValueError(f"A repack must start on a compressed pair ({span} tokens)")
    p = len(src_pages_by_rank)
    t = pair_tokens_np(num_tokens, ratio, start, dst_block, rank)
    owner = (t // src_block) % p
    entry = t // (p * src_block) - src_first
    sp = np.stack([np.asarray(pages, dtype=np.int64) for pages in src_pages_by_rank])
    src = sp[owner, entry] * src_block + (t % src_block) // span
    return src, pair_dst_np(t, dst_pages, dst_block, span, dst_offset, rank)


def main_dst_np(num_tokens, dst_pages, dst_block, rank, start=0) -> np.ndarray:
    """``main_rows_np``'s destination rows alone (routed staging)."""
    t = np.arange(start, num_tokens, dtype=np.int64)
    d = rank.main_dcp
    if d > 1:
        t = t[(t // dst_block) % d == rank.dcp_rank]
    dp = np.asarray(dst_pages, dtype=np.int64)
    return dp[t // (d * dst_block)] * dst_block + t % dst_block


def main_sources(srcs, columns: int) -> np.ndarray:
    """One main-KV owner's staging rows as a segment of the repack's ``main_src``.

    ``srcs``: one array per member, in head-slot order, listing the same
    tokens in the same order (the owner's destination rows). ``columns`` 1
    (unpaired layout): the owner's rows, ``[n]``. ``columns`` 2
    (QWEN4_EXP_PACK_MAIN_KV): ``[n, 2]``, member ``s`` in column ``s``; an
    owner without a partner (an odd last layer, the MTP draft) takes staging
    row 0, the never-reserved null page, so its head slots 2-3 stay zero.
    """
    if len(srcs) > columns:
        raise ValueError(f"{len(srcs)} main-KV members for {columns} source columns")
    if columns == 1:
        return np.asarray(srcs[0], dtype=np.int64)
    first = np.asarray(srcs[0], dtype=np.int64)
    cols = [np.asarray(x, dtype=np.int64) for x in srcs]
    if any(len(c) != len(first) for c in cols):
        raise ValueError("paired main-KV members must list the same tokens")
    cols += [np.zeros(len(first), dtype=np.int64)] * (columns - len(cols))
    return np.stack(cols, axis=1)


def compressed_dst_np(
    num_tokens, ratio, dst_pages, dst_block, start=0, dst_offset=0, rank=None
) -> np.ndarray:
    """``compressed_rows_np``'s destination rows alone (routed staging)."""
    span = 2 * ratio
    if start % span:
        raise ValueError(f"A repack must start on a compressed pair ({span} tokens)")
    t = pair_tokens_np(num_tokens, ratio, start, dst_block, rank)
    return pair_dst_np(t, dst_pages, dst_block, span, dst_offset, rank)


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
    if getattr(src_layout, "merged_compressed", False) or int(getattr(src_layout, "main_dcp", 1)) > 1:
        # pd_route models a plain TP1 producer: one main-KV page per ID and
        # unmerged compressed keys. Merged keys (QWEN4_EXP_MERGE_COMPRESSED_K /
        # QWEN4_EXP_PD_PRODUCER_MERGED) and a PCP producer's split main KV
        # (QWEN4_EXP_PD_PRODUCER_PCP > 1) are refused rather than mis-routed.
        raise ValueError(
            "QWEN4_EXP_PD_ROUTED_DST supports a plain TP1 producer only "
            "(no merged compressed keys, no PCP split main KV)"
        )
    src_gdn = gdn_geometry(cfg, 1, src_layout.token_bytes)
    dst_gdn = gdn_geometry(cfg, tp, decode_layout.token_bytes)
    return pd_route.RouteGeometry(
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
    return np.concatenate((rows, np.repeat(rows[-1:], pad, axis=0)))


def _cat(parts, columns: int = 1) -> np.ndarray:
    """A planner's segments, each tile-aligned (``align_rows``), as one list;
    ``columns`` > 1: paired main-KV sources (``main_sources``), ``[0, columns]``
    when empty so every program keeps its shape."""
    parts = [align_rows(p) for p in parts if len(p)]
    if parts:
        return np.concatenate(parts)
    return np.zeros((0, columns) if columns > 1 else 0, dtype=np.int64)


def cat_aligned(parts, columns: int = 1, tile: int = REPACK_TILE) -> np.ndarray:
    """``_cat(parts, columns)`` in one allocation: each segment tile-aligned by
    repeating its last row (``align_rows``), ``[0, columns]`` (or ``[0]``) when
    every part is empty."""
    parts = [p for p in parts if len(p)]
    if not parts:
        return np.zeros((0, columns) if columns > 1 else 0, dtype=np.int64)
    sizes = [len(p) + (-len(p) % tile) for p in parts]
    out = np.empty((sum(sizes), *parts[0].shape[1:]), dtype=np.result_type(*parts))
    pos = 0
    for p, size in zip(parts, sizes):
        out[pos : pos + len(p)] = p
        out[pos + len(p) : pos + size] = p[-1]
        pos += size
    return out


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


def main_pairs(layout) -> dict[str, tuple[str, ...]]:
    """Each paired main-KV owner's members in head-slot order
    (QWEN4_EXP_PACK_MAIN_KV, ``layout.main_owners``); empty when unpaired.

    The owner is member 0 and names the scheduler group; member ``s`` keeps K
    and V in head slots ``2s`` and ``2s + 1`` of the owner's rows
    (``pool_layout.pooled_cache_specs``). Owners missing here (an odd last
    target layer, the MTP draft) hold one layer in whole rows.
    """
    slots: dict[str, dict[int, str]] = {}
    for prefix, owner, slot in getattr(layout, "main_owners", ()) or ():
        slots.setdefault(owner, {})[int(slot)] = prefix
    out = {}
    for owner, members in slots.items():
        if sorted(members) != list(range(len(members))) or len(members) > 2 or members[0] != owner:
            raise ValueError(f"paired main KV owner {owner} has members {members}")
        out[owner] = tuple(members[s] for s in range(len(members)))
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

# GBULK-LINE: the DMA row gathers this thread has traced, by kind ("rows": per-row DMAs, "bulk": one
# DMA per consecutive tile). Python counters only, which _build_op's repack reads while it traces.
_GATHER_TRACES = threading.local()


def _count_gather(bulk: bool) -> None:
    kind = "bulk" if bulk else "rows"
    setattr(_GATHER_TRACES, kind, getattr(_GATHER_TRACES, kind, 0) + 1)


def _gather_counts() -> tuple[int, int]:
    """(per-row, bulk) DMA row gathers traced so far on this thread (gather_bulk_mismatch)."""
    return getattr(_GATHER_TRACES, "rows", 0), getattr(_GATHER_TRACES, "bulk", 0)


def _count_write(fold: bool) -> None:
    kind = "write_fold" if fold else "write_concat"
    setattr(_GATHER_TRACES, kind, getattr(_GATHER_TRACES, kind, 0) + 1)


def _write_counts() -> tuple[int, int]:
    """(default concatenated, fold) landing writes traced so far on this thread (staging_fold_mismatch)."""
    return getattr(_GATHER_TRACES, "write_concat", 0), getattr(_GATHER_TRACES, "write_fold", 0)


def dma_row_gather(
    stage, flat, *, tile: int = REPACK_TILE, interpret: bool = False, out_dtype=None, bulk: bool = False
):
    """``stage[flat]`` (``stage``: ``[rows, 4, 256]`` of the carrier's dtype), byte-exact.

    ``bulk`` (VLLM_TPU_PD_GATHER_BULK, ``_tile_row_gather``): a tile whose ``tile`` indices are consecutive
    rows is read with one DMA of ``tile`` rows; every other tile keeps the per-row DMAs below. The same rows
    land in the same output rows either way. Off, the program is exactly the per-row one.

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
    _count_gather(bulk)  # GBULK-LINE: the gather this trace took (a Python counter, not a program op)
    if bulk:
        return _tile_row_gather(stage, idx, n, tile, out_dtype, same, interpret)

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


def gather_bulk_tiles(idx, tile: int = REPACK_TILE):
    """Per ``tile``-row tile of the padded index list ``idx``: 1 if its indices are consecutive rows
    (``idx[first] + 0, 1, ..., tile - 1``), else 0 (int32). A tile of consecutive rows is one contiguous
    run of the carrier, so one DMA of ``tile`` rows reads exactly the rows the per-row DMAs read. The
    scatter's ``bulk`` flag (``pooled_qsa._write_main_cache_universal``) is the same test on the pool side.
    """
    import jax.numpy as jnp

    tiles = idx.reshape(-1, tile)
    run = tiles[:, :1] + jnp.arange(tile, dtype=tiles.dtype)
    return jnp.all(tiles == run, axis=1).astype(jnp.int32)


def _tile_row_gather(stage, idx, n: int, tile: int, out_dtype, same: bool, interpret: bool):
    """``dma_row_gather`` with ``bulk`` (q38 SV-GATHER): one DMA per tile of consecutive rows.

    The staged landing's rows come in runs (staging rows are numbered per owner in 256-row pages, segments
    start on a tile and their padding is a run of null-page rows), so most tiles are one run; the per-row
    gather still issued one 1 KiB DMA per row, about 20.7 ns each (TRACEAN, D-EC j-b60636b2). A tile of
    consecutive rows (``gather_bulk_tiles``) is read by one DMA of ``tile`` rows; any other tile (a run
    starting or ending inside it, a repeated last row) keeps the per-row DMAs. Either way output row ``i``
    is carrier row ``idx[i]``, byte for byte, through the DMA engine only (no vector unit holds the bytes as
    FP8 values; the u8 view goes through the same ``pltpu.bitcast`` of the landed block).
    """
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    if not _GATHER_MARK.get("bulk"):  # engagement marker, once per process, at trace time
        _GATHER_MARK["bulk"] = True
        print("q38 SV-GATHER bulk gather traced: one DMA per consecutive %d-row tile (dtype %s, backend %s)"
              % (tile, stage.dtype, jax.default_backend()), flush=True)
    bulk = gather_bulk_tiles(idx, tile)

    def kernel(idx_ref, bulk_ref, stage_ref, out_ref, semaphore, *scratch):
        program = pl.program_id(0)
        first = program * tile
        dst = out_ref if same else scratch[0]

        @pl.when(bulk_ref[program] != 0)
        def tile_copy():
            pltpu.async_copy(stage_ref.at[pl.ds(idx_ref[first], tile)], dst, semaphore).wait()

        @pl.when(bulk_ref[program] == 0)
        def row_copies():
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
            num_scalar_prefetch=2,
            grid=(idx.shape[0] // tile,),
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
            out_specs=pl.BlockSpec(
                (tile, *stage.shape[1:]),
                lambda program, *_: (program,) + (0,) * (stage.ndim - 1),
            ),
            scratch_shapes=[pltpu.SemaphoreType.DMA]
            + ([] if same else [pltpu.VMEM((tile, *stage.shape[1:]), stage.dtype)]),
        ),
        out_shape=jax.ShapeDtypeStruct((idx.shape[0], *stage.shape[1:]), out_dtype),
        name="qwen4_pd_staged_row_gather_bulk",
        **({"interpret": pltpu.InterpretParams()} if interpret else {}),
    )(idx, bulk, stage)
    return out[:n]


def _write_u8_rows(pool, rows, page, logical, valid, tile: int = REPACK_TILE):
    """``pool[page, logical % pool.shape[1]] = rows`` for uint8 ``rows`` into a pool of another dtype of the same width,
    in place (q38 SV-GATHER PD-STAGING-FOLD): ``pooled_qsa._write_main_cache_universal`` with ``overlap_edges`` (one DMA per
    whole page-local tile of consecutive rows, overlapped row DMAs elsewhere, ``valid`` masking), except that each tile of
    rows lands in a VMEM block of uint8 and is reinterpreted to the pool dtype by a Mosaic vreg bitcast (``pltpu.bitcast``,
    a register reinterpretation, the landing gather's own uint8 path in reverse) into a scratch block, which is what the
    DMAs copy. So no XLA op ever holds the bytes as pool-dtype (FP8) values: XLA FP8 <-> uint8 conversions canonicalised NaN
    encodings on TPU (nanprobe j-5a9f82e0; lane_xfer 11.29). ``rows``: ``[n, *pool.shape[2:]]`` uint8, ``n`` a multiple of
    ``tile`` (the caller pads and masks), ``pool.shape[1]`` a multiple of ``tile``."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    if len(rows) % tile or pool.shape[1] % tile:
        raise ValueError("_write_u8_rows takes whole tiles of rows into whole-tile pages")
    if jnp.dtype(rows.dtype).itemsize != jnp.dtype(pool.dtype).itemsize or tuple(rows.shape[1:]) != tuple(pool.shape[2:]):
        raise ValueError("_write_u8_rows needs same-width rows of the pool's row shape")
    if not _GATHER_MARK.get("fold"):  # engagement marker, once per process, at trace time
        _GATHER_MARK["fold"] = True
        print("q38 SV-GATHER staging fold traced: segment rows written straight into the pool (rows %s -> pool %s, backend %s)"
              % (rows.dtype, pool.dtype, jax.default_backend()), flush=True)
    _count_write(True)  # FOLD-LINE: the fold writer this trace took (a Python counter, not a program op)
    offsets = logical % pool.shape[1]
    pages_by_tile = page.reshape(-1, tile)
    offsets_by_tile = offsets.reshape(-1, tile)
    valid_by_tile = valid.reshape(-1, tile)
    bulk = (
        jnp.all(valid_by_tile, axis=1)
        & jnp.all(pages_by_tile == pages_by_tile[:, :1], axis=1)
        & jnp.all(offsets_by_tile == offsets_by_tile[:, :1] + jnp.arange(tile, dtype=offsets.dtype), axis=1)
    )
    pool_dtype = pool.dtype
    same = jnp.dtype(rows.dtype) == jnp.dtype(pool_dtype)

    def kernel(pages_ref, offsets_ref, valid_ref, bulk_ref, source_ref, _old_ref, target_ref, semaphore, *scratch_refs):
        program = pl.program_id(0)
        first = program * tile
        if same:  # nothing to reinterpret: the DMAs read the landed block itself
            scratch = source_ref
        else:
            scratch = scratch_refs[0]
            scratch[...] = pltpu.bitcast(source_ref[...], pool_dtype)

        @pl.when(bulk_ref[program] != 0)
        def store_tile():
            pltpu.async_copy(scratch, target_ref.at[pages_ref[first], pl.ds(offsets_ref[first], tile)], semaphore).wait()

        def row_copy(lane, row):
            return pltpu.make_async_copy(
                scratch.at[pl.ds(lane, 1)],
                target_ref.at[pages_ref[row], pl.ds(offsets_ref[row], 1)],
                semaphore,
            )

        @pl.when(bulk_ref[program] == 0)
        def store_edges():
            for lane in range(tile):
                row = first + lane

                @pl.when(valid_ref[row] != 0)
                def start_row(*, lane=lane, row=row):
                    row_copy(lane, row).start()

            for lane in range(tile):
                row = first + lane

                @pl.when(valid_ref[row] != 0)
                def wait_row(*, lane=lane, row=row):
                    row_copy(lane, row).wait()

    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            grid=(len(rows) // tile,),
            in_specs=[
                pl.BlockSpec((tile, *rows.shape[1:]), lambda program, *_: (program,) + (0,) * (rows.ndim - 1)),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[pltpu.SemaphoreType.DMA] + ([] if same else [pltpu.VMEM((tile, *rows.shape[1:]), pool_dtype)]),
        ),
        out_shape=jax.ShapeDtypeStruct(pool.shape, pool.dtype),
        input_output_aliases={5: 0},
        name="qwen4_pd_landing_fold_scatter",
    )(
        page.astype(jnp.int32),
        offsets.astype(jnp.int32),
        valid.astype(jnp.int32),
        bulk.astype(jnp.int32),
        rows,
        pool,
    )


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
    gather_bulk: bool = False,
    staging_fold: bool = False,
):
    """``fn(pool, staging, main_src, main_dst, copy_src, copy_dst, gdn_src, gdn_dst)``.

    ``main_src`` ``[n]``: one layer per written row, the rank's head pair to
    head slots 0-1 and zeros in 2-3. ``[n, 2]`` (QWEN4_EXP_PACK_MAIN_KV,
    ``main_sources``): two layers per written row, member ``s``'s pair (from
    staging row ``main_src[:, s]``) to head slots ``2s`` and ``2s + 1``, the
    slot rule of the decode's own paired writer
    (``pooled_qsa._write_main_cache_paired``).

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
    elsewhere (bit-exact on CPU). ``gather_bulk`` (VLLM_TPU_PD_GATHER_BULK)
    makes the TPU default the DMA gather that reads each tile of consecutive
    rows with one DMA (``dma_row_gather(bulk=True)``); the bytes are the same.
    ``staging_fold`` (VLLM_TPU_PD_STAGING_FOLD) writes each gathered segment (main KV,
    copied rows, GDN) into the pool with its own call of ``_write_u8_rows``: no
    concatenation, no whole-buffer pad and no XLA uint8 -> pool-dtype bitcast of
    the landed rows (a custom ``write`` gets each segment bitcast to the pool
    dtype instead, the CPU tests' stand-in); the same rows land with the same bytes.

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

            if gather_bulk:
                gather = functools.partial(dma_row_gather, out_dtype=jnp.uint8, bulk=True)
                gather.__name__ = "dma_row_gather_u8_bulk"
            else:
                gather = functools.partial(dma_row_gather, out_dtype=jnp.uint8)
                gather.__name__ = "dma_row_gather_u8"
        else:
            gather = rows_of
    print("Qwen PD staged repack gather selected: %s (jax backend %s)"
          % (getattr(gather, "__name__", "custom"), jax.default_backend()), flush=True)

    custom_write = write is not None
    if write is None:
        import functools

        from vllm_torchtpu.kernels.qwen4_exp.pooled_qsa import (
            _write_main_cache_universal,
        )

        write = functools.partial(
            _write_main_cache_universal, tile=REPACK_TILE, overlap_edges=True
        )

    def write_segment(pool, rows, dst):
        """One landed segment (uint8 rows ``[n, 4, 256]``, flat pool rows ``dst``) into the pool: the fold path."""
        dst = dst.astype(jnp.int32)
        valid = jnp.ones(dst.shape, jnp.bool_)
        tail = -len(dst) % REPACK_TILE
        if tail:  # whole tiles; the tail rows are masked out
            rows = jnp.pad(rows, ((0, tail), (0, 0), (0, 0)))
            dst = jnp.pad(dst, (0, tail))
            valid = jnp.pad(valid, (0, tail))
        kernel_rows = pool.shape[1]
        rows = rows.reshape(-1, *pool.shape[2:])
        if custom_write:
            return write(pool, jax.lax.bitcast_convert_type(rows, pool.dtype), dst // kernel_rows, dst, valid)
        return _write_u8_rows(pool, rows, dst // kernel_rows, dst, valid, tile=REPACK_TILE)

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

        if main_src.ndim == 2:
            # Paired main KV: both members' pairs in one written row. The
            # gather reads member 0's rows, then member 1's.
            n = main_src.shape[0]
            pairs = gather_u8(main_src.T.reshape(-1))[:, 2 * kv_head : 2 * kv_head + 2]
            main = jnp.concatenate([pairs[:n], pairs[n:]], axis=1)
        else:
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
        if staging_fold:
            pool = write_segment(pool, main, main_dst)
            pool = write_segment(pool, copies, copy_dst)
            if has_gdn:
                pool = write_segment(pool, gdn, (gdn_dst[:, None] * dst_block + jnp.arange(d_rows)).reshape(-1))
            return pool
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
        if not custom_write:
            _count_write(False)  # FOLD-LINE: the default concatenated write this trace took (a Python counter)
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


@dataclasses.dataclass
class RepackPlan:
    """One load's landing, planned before any device work: the shared
    compressed-key gather passes (``pd_shared_compressed.GatherPass``), then
    the repack chunks (``repack_chunks``; with QWEN4_EXP_PD_FAST_ADMIT,
    ``repack_chunks_packed``'s vectors and shapes)."""

    passes: list
    chunks: list


# q38 KVL-INWAIT: ``last_draft_rows`` was not set by the planning ``indices`` call.
_ROWS_UNSET = object()


@dataclasses.dataclass
class PlannedRepack:
    """One load's ``RepackPlan`` made ahead of the step that names it (q38
    SV-KVGC KVL-INWAIT, ``plan_repack_ahead``), with what ``plan_repack``'s draft
    counters read at that point: the load's skip, its draft-zero end and the
    rows ``indices`` reported (``_ROWS_UNSET`` when it reported none)."""

    plan: RepackPlan
    skip: int
    zero_end: int
    draft_rows: object


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
    # The producer's PCP / merged-key geometry (QWEN4_EXP_PD_PRODUCER_PCP,
    # _MERGED, _SPLIT_KEYS), set by __init__; a plain TP1 producer for a
    # consumer assembled without it (read-only defaults).
    src_pcp = 1
    src_merged = False
    src_split_keys = False
    _key_pools: "dict[str, int]" = types.MappingProxyType({})
    _src_split: "frozenset[str]" = frozenset()
    # The repack program shape (QWEN4_EXP_PD_REPACK_SHAPE) and the GDN split
    # (repack_chunks split_gdn), set by __init__; the default fixed chunks for
    # a consumer assembled without it.
    _repack_shape = (REPACK_CHUNK_ROWS, REPACK_CHUNK_ROWS, REPACK_GDN_PAGES)
    _split_gdn = False
    # (staging, pool) rows per page, whose page-0 rows pad repack chunks
    # (repack_chunks null_rows); set by __init__, the last pair repeated for a
    # consumer assembled without it.
    _null_rows = None
    _fn_rest = None
    _op_rest = None
    # The decode layout's DCP-split owners (main_dcp_owners), set by
    # __init__; none for a consumer assembled without it.
    _dcp_owners: "frozenset[str]" = frozenset()
    # QWEN4_EXP_PACK_MAIN_KV: main-KV owner -> its members in head-slot order
    # (layout.main_owners), set by __init__; unpaired without it.
    _main_pairs: "dict[str, tuple[str, ...]]" = types.MappingProxyType({})

    def __init__(self, cfg, decode_layout, decode_groups, pool, rank: RankSlice, staging_bytes: int):
        from vllm.v1.core.kv_cache_utils import get_kv_cache_groups

        from vllm_torchtpu.distributed.kv_transfer.raiden.pool_manifest import (
            BINDING_ALIASED_RAW,
            PoolEntry,
            PoolManifest,
            RegionSpec,
        )

        from .pd import TransferExtent, build_transfer_groups
        from .pool_layout import pooled_cache_specs

        # Packed compressed keys (QWEN4_EXP_PACK_COMPRESSED_K=1) are this
        # decode's own layout: the repack writes each layer's pairs at its
        # slot of the owner group's page. Paired main KV
        # (QWEN4_EXP_PACK_MAIN_KV=1) likewise: one written row carries both
        # members' head pairs (_main_pairs). The producer packs neither (a
        # TP1 stage holds two KV heads a row, which pairing refuses).
        self.cfg, self.rank, self.pool = cfg, rank, pool
        self.dst_layout = decode_layout
        self.dst_groups = decode_groups
        # The producer (producer_geometry): TP1 stages, no main-KV DCP,
        # unpacked compressed keys. Without QWEN4_EXP_PD_DRAFT_KV it runs no
        # MTP draft: a decode-only draft (this engine's speculative_config)
        # adds draft owners and speculative state blocks the producer's plan
        # does not have. With it, the producer's prefill-only MTP (K = 1)
        # allocates and ships the draft's owners too.
        # A PCP producer (QWEN4_EXP_PD_PRODUCER_PCP=P, the prefill leg's PCP
        # size) splits its main KV over its P ranks in pcp_interleave() units.
        import os

        from .pool_layout import pcp_interleave

        draft_kv = draft_kv_enabled()
        # Option (a)'s range zero-fill (draft_zero_end) runs only with it on.
        self.draft_kv = draft_kv
        layout_cfg, src_tp, src_dcp, src_pack, src_draft_cacheable = producer_geometry(cfg)
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
        if draft_kv and (self.src_pcp > 1 or self.src_merged):
            # F2 models a plain TP1 pipeline producer's draft owners only.
            raise ValueError(
                "QWEN4_EXP_PD_DRAFT_KV=1 supports a plain TP1 pipeline producer only "
                "(not QWEN4_EXP_PD_PRODUCER_PCP > 1 or QWEN4_EXP_PD_PRODUCER_MERGED=1)"
            )
        layout_kwargs = dict(
            tp=src_tp,
            main_dcp=self.src_pcp if self.src_pcp > 1 else src_dcp,
            pack_compressed=src_pack,
            pack_main=False,
            merge_compressed=self.src_merged,
            split_compressed=self.src_split_keys,
            main_interleave=pcp_interleave() if self.src_pcp > 1 else None,
        )
        if not src_draft_cacheable:
            layout_kwargs["draft_prefix_cacheable"] = False
        self.src_layout, src_specs = pooled_cache_specs(layout_cfg, **layout_kwargs)
        self._src_split = (
            set(self.src_layout.main_dcp_owners) if self.src_pcp > 1 else set()
        )
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
        page_bytes = (
            stage_rows * ROW_SUBLANES * ROW_LANES if self.route else self.src_layout.page_bytes
        )
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
        from .pd_manifest import compressed_live_bytes, page_regions, pool_tag

        # The producer sends a compressed-key page's keys only; the carrier
        # declares the same region so both pool layouts agree. Routed staging
        # pages are dense: every byte of a page is live.
        compressed_live = None if self.route else compressed_live_bytes(self.src_layout)
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
                    block_stride_bytes=page_bytes,
                    num_blocks=self.pages,
                    regions=page_regions(name, page_bytes, compressed_live),
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
        self._main_pairs = main_pairs(decode_layout)
        # Compressed-key owner -> (destination group, row offset in its page):
        # a packed member sits in its owner's group at slot * member rows.
        self._dst_compressed = compressed_destinations(
            decode_layout, self._dst_owner, self.dst_block, self.ratio
        )
        # Main KV under VLLM_TPU_QSA_MAIN_KV_DCP (the draft's with
        # VLLM_TPU_QSA_MTP_KV_DCP) and compressed keys under
        # VLLM_TPU_QSA_COMPRESSED_DCP: each rank lands only the rows it owns.
        self._dcp_owners = frozenset(decode_layout.main_dcp_owners)
        if self.shared is not None and self._dcp_owners.intersection(self._dst_compressed):
            # The landing gather writes every pair at its unsplit row on
            # every rank (pd_shared_compressed.gather_passes).
            raise ValueError(
                "QWEN4_EXP_PD_SHARED_COMPRESSED refuses VLLM_TPU_QSA_COMPRESSED_DCP=1: "
                "its landing gather writes compressed keys unsplit"
            )
        # Decode-only MTP: the draft layer's owners (and its packed compressed
        # slot) have no producer pages; its prompt KV starts empty (synthetic
        # acceptance does not read it). QWEN4_EXP_PD_DRAFT_KV: the producer
        # ships them, so no owner may be missing, and each lands like a
        # target QSA layer's (the draft's main KV DCP-split on a DCP decode).
        # Split keys are staged per producer rank ("<owner>.pcp<r>"), as split
        # main KV is; either way the logical owner is covered.
        staged = set(self._src_owner) | {n.rsplit(".pcp", 1)[0] for n in self._key_pools}
        members = {m for pair in self._main_pairs.values() for m in pair}
        missing = (set(self._dst_owner) | set(self._dst_compressed) | members) - staged
        if draft_kv:
            check_draft_kv(
                missing, self._dst_owner, rank.main_dcp,
                getattr(decode_layout, "main_dcp_owners", ()),
            )
            if draft_kv_split_keys_enabled():
                # Engagement (QWEN4_EXP_PD_DRAFT_KV_SPLIT_KEYS): once per rank,
                # the F2 draft compressed-key owners this rank lands split.
                split_draft = sorted(
                    n for n in self._dcp_owners
                    if n.startswith("mtp.") and owner_kind(n) == "compressed"
                )
                if split_draft:
                    logger.info(
                        "Qwen PD F2 draft KV split keys: rank %d lands %s split over "
                        "DCP %d (QWEN4_EXP_PD_DRAFT_KV_SPLIT_KEYS)",
                        rank.tp_rank, split_draft, rank.main_dcp,
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
        from vllm_torchtpu import envs

        self._repack_shape = parse_repack_shape(envs.QWEN4_EXP_PD_REPACK_SHAPE)
        # A first program that holds every GDN owner lands all GDN state; the
        # load's later programs run a repack built without GDN state, which
        # gathers no padded GDN pages.
        self._split_gdn = 0 < len(self._gdn_names) <= self._repack_shape[2]
        self._null_rows = (stage_rows, self.dst_block)
        # VLLM_TPU_PD_GATHER_BULK: the repack programs gather tiles of
        # consecutive rows with one DMA each; their ops get their own names
        # (``_build_op``), which keys every graph-keyed compile cache apart.
        self._gather_bulk = gather_bulk_enabled()
        # GBULK-LINE: one line per rank at every engine start, warm compile cache included.
        gather_bulk_bind_line(rank, self._gather_bulk)
        # VLLM_TPU_PD_STAGING_FOLD: each segment straight into the pool; its ops
        # get "_fold" names (``_build_op``) for the same reason.
        self._staging_fold = staging_fold_enabled()
        # FOLD-LINE: one line per rank at every engine start, warm compile cache included.
        staging_fold_bind_line(rank, self._staging_fold)
        repack_kwargs = dict(
            gather_bulk=self._gather_bulk,
            staging_fold=self._staging_fold,
            src_block=self.src_block,
            dst_block=self.dst_block,
            kv_head=rank.kv_head,
            v_head_start=rank.tp_rank * (dst_gdn.v_heads if dst_gdn else 0),
            channels=conv_channels(src_gdn, dst_gdn, rank) if src_gdn else [0],
            compact_gdn=(
                (self.route.stage_rows, self.route.gdn_rows // self.route.stage_rows)
                if self.route
                else None
            ),
        )
        self._fn = make_repack(src_gdn=src_gdn, dst_gdn=dst_gdn, **repack_kwargs)
        self._fn_rest = None
        if self._split_gdn:
            self._fn_rest = make_repack(src_gdn=None, dst_gdn=None, **repack_kwargs)
        self._op = None
        self._op_rest = None

    # --- allocation (identical on every rank of the engine) ---

    def _pool_group_map(self) -> dict[str, int]:
        """Staging pool name -> producer group (``_pool_group``). A consumer
        assembled without ``__init__`` (admission and reuse tests) has only
        ``_src_owner``, which is the same map without split ``.pcp<r>`` pools."""
        pool_group = getattr(self, "_pool_group", None)
        return self._src_owner if pool_group is None else pool_group

    def _fast(self) -> bool:
        """QWEN4_EXP_PD_FAST_ADMIT, read once per consumer (instance dict, so a
        consumer assembled without ``__init__`` reads it on first use)."""
        fast = self.__dict__.get("_fast_admit")
        if fast is None:
            fast = self.__dict__["_fast_admit"] = fast_admit()
        return fast

    def _owner_counts(self, num_tokens: int, skip_tokens: int):
        """(first staged source page per group, owners, staging pages per owner)
        of a load, in manifest order. With QWEN4_EXP_PD_FAST_ADMIT they come
        from a per-(tokens, prefix) memo, as fresh lists."""
        if not self._fast():
            return self._owner_counts_once(num_tokens, skip_tokens)
        key = (int(num_tokens), int(skip_tokens))
        memo = self.__dict__.setdefault("_counts_memo", {})
        hit = memo.get(key)
        if hit is None:
            if len(memo) >= _COUNTS_MEMO_ENTRIES:
                memo.clear()
            hit = memo[key] = self._owner_counts_by_kind(*key)
        first, owners, counts = hit
        return list(first), list(owners), list(counts)

    def _owner_counts_by_kind(self, num_tokens: int, skip_tokens: int):
        """``_owner_counts_once`` with one page count per distinct (cache kind,
        source group pages) instead of one per owner: owners of a kind with
        the same source pages take the same count. Raises where it does, on
        the same owner (the first of its kind and pages, evaluated in order)."""
        if self.route is None:
            first, owners, counts = self._owner_counts_once(num_tokens, skip_tokens)
            return tuple(first), tuple(owners), tuple(counts)
        meta = self.__dict__.get("_owner_meta")
        if meta is None:
            names = tuple(pool.layer_name for pool in self.manifest.pools)
            meta = self.__dict__["_owner_meta"] = (
                names,
                tuple(owner_kind(n) for n in names),
                tuple(self._src_owner[n] for n in names),
            )
        owners, kinds, groups = meta
        src_pages = [g.page_count(num_tokens) for g in self.src_groups]
        shared = self._shared_load(num_tokens)
        by_kind: dict = {}
        counts = []
        for kind, gi in zip(kinds, groups):
            key = (kind, src_pages[gi])
            count = by_kind.get(key)
            if count is None:
                count = by_kind[key] = (
                    self.shared.staged_pages(num_tokens, skip_tokens)
                    if shared and kind == "compressed"
                    else pd_route.staging_pages(kind, num_tokens, key[1], skip_tokens, self.route)
                )
            counts.append(count)
        return (0,) * len(self.src_groups), owners, tuple(counts)

    def _owner_counts_once(self, num_tokens: int, skip_tokens: int):
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
        pool_group = self._pool_group_map()
        counts = [group_counts[pool_group[name]] for name in owners]
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
        if self._fast():
            pages = self._take_pages(free, owners, counts, reduced_free)
            if pages is None:
                return False
        else:
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

    def _take_pages(self, free, owners, counts, reduced_free):
        """``reserve``'s allocation under QWEN4_EXP_PD_FAST_ADMIT: the first
        ``sum(counts)`` free pages in page order, cut per owner in manifest
        order, zeroed in one write per map (CPU maps, written through numpy
        views). None when they do not fit. The same pages as the per-owner
        path."""
        need = sum(counts)
        available = np.flatnonzero(free.numpy())
        if len(available) < need:
            return None
        take = available[:need]
        flat = take.tolist()
        pages, cursor = {}, 0
        for name, count in zip(owners, counts):
            pages[name] = flat[cursor : cursor + count]
            cursor += count
        if len(pages) != len(owners):
            # A repeated owner name keeps its last slice only, as there.
            take = np.asarray(
                [p for owner_pages in pages.values() for p in owner_pages], dtype=np.int64
            )
        self.free.numpy()[take] = 0
        if reduced_free is not None:
            reduced_free.numpy()[take] = 0
        return pages

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
        # The clip is in the destination's live bytes: a merged key pool's
        # page carries `key_stride` of them (its single "keys" region's
        # unit), a trimmed compressed-key page its keys, else the stride.
        pool_group = self._pool_group_map()
        return [
            first[pool_group[pool.layer_name]] * _live_page_bytes(pool)
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
                self._mark_free(held)
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
            self._mark_free(freed)
            held[:] = [entry for entry in held if entry[0] > 0]
        return len(freed)

    def _mark_free(self, pages: list[int]) -> None:
        """Returns ``pages`` to the free map (QWEN4_EXP_PD_FAST_ADMIT: one
        numpy write through the CPU map's view instead of a list index)."""
        if self._fast():
            self.free.numpy()[np.asarray(pages, dtype=np.int64)] = 1
        else:
            self.free[pages] = 1

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

    def _split_rank(self, name: str) -> "RankSlice | None":
        """This rank for a DCP-split destination owner (``_dcp_owners``), else None."""
        return self.rank if name in self._dcp_owners else None

    def _main_members(self, name: str) -> tuple[str, ...]:
        """The producer main-KV owners a destination main owner lands, in
        head-slot order: both paired members (``_main_pairs``), else itself."""
        return self._main_pairs.get(name, (name,))

    def _main_columns(self) -> int:
        """Source columns of the repack's ``main_src`` (``main_sources``): 2
        when the layout pairs main KV, so every program of every load has one
        shape; else 1."""
        return 2 if self._main_pairs else 1

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
            split = self._split_rank(name)
            if self.src_split_keys:
                # Keys of token t come from producer rank (t // B) % P, in that
                # rank's page for t's main-KV ID (B-token pages, P per ID).
                s, d = compressed_rows_split_np(
                    num_tokens, self.ratio,
                    [src_pages[f"{name}.pcp{r}"] for r in range(self.src_pcp)],
                    self.src_block, dst_tables[gi], self.dst_block, start=skip_tokens,
                    src_first=first[self._src_owner[f"{name}.pcp0"]], dst_offset=offset,
                    rank=split,
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
                src_page_rows=self.src_block if merged else None, rank=split,
            )
            s, d = gap.split(
                name, s, d,
                lambda split=split: pair_tokens_np(
                    num_tokens, self.ratio, skip_tokens, self.dst_block, split
                ),
            )
            copy_s.append(s)
            copy_d.append(d)
        for name, gi in self._dst_owner.items():
            kind = owner_kind(name)
            if kind == "compressed":
                continue
            # A PCP producer's split main KV is staged per rank (below).
            src = src_pages.get(name)
            dst = dst_tables[gi]
            if kind == "main":
                # One destination row list for the owner; one source list per
                # member, from that member's own staging pages, in its order.
                srcs = []
                for member in self._main_members(name):
                    member_src, pcp = src_pages.get(member), None
                    if member in self._src_split:
                        from .pool_layout import pcp_interleave

                        # One staged page list per producer rank (same length).
                        member_src = [src_pages[f"{member}.pcp{r}"] for r in range(self.src_pcp)]
                        pcp = (self.src_pcp, pcp_interleave())
                    s, d = main_rows_np(
                        num_tokens, member_src, self.src_block, dst, self.dst_block, self.rank,
                        start=skip_tokens, src_first=first[self._src_owner[member]], src_pcp=pcp,
                    )
                    srcs.append(s)
                s = main_sources(srcs, self._main_columns())
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
            split = self._split_rank(name)
            if kind == "compressed":
                span = 2 * self.ratio
                d = compressed_dst_np(num_tokens, self.ratio, dst, self.dst_block,
                                      skip_tokens // span * span, offset, rank=split)
            elif kind == "main":
                # Split under VLLM_TPU_QSA_MTP_KV_DCP, else every row.
                rank = self.rank if split else dataclasses.replace(self.rank, main_dcp=1)
                d = main_dst_np(num_tokens, dst, self.dst_block, rank, skip_tokens)
            else:  # ring: its one page's rows
                d = np.arange(self.dst_block, dtype=np.int64) + dst[0] * self.dst_block
            copy_s.append(np.zeros(len(d), dtype=np.int64))
            copy_d.append(d)

        return (
            _cat(main_s, self._main_columns()), _cat(main_d), _cat(copy_s), _cat(copy_d),
            np.asarray(gdn_s, dtype=np.int64), np.asarray(gdn_d, dtype=np.int64),
        )

    def _routed_indices(self, src_pages, dst_tables, num_tokens, skip_tokens, draft_zero_until=0):
        """``indices`` over routed compact staging: same destination rows in
        the same order as the broadcast layout, read from compact rows."""
        if self._fast():
            return self._routed_indices_shared_rows(
                src_pages, dst_tables, num_tokens, skip_tokens, draft_zero_until
            )
        g, rank = self.route, self.rank
        main_s, main_d, copy_s, copy_d = [], [], [], []
        gdn_s, gdn_d = [], []
        gap = _DraftGap(draft_zero_until if draft_zero_until > skip_tokens else 0)

        # A shared load's compressed keys are written by the landing gather
        # (shared_passes), not by the repack. The gather runs first, so a
        # draft gap's zero rows (from the repack) overwrite what it wrote.
        shared = self._shared_load(num_tokens)
        for name, (gi, offset) in self._dst_compressed.items():
            # Split keys (VLLM_TPU_QSA_COMPRESSED_DCP): every rank receives every
            # pair and lands the ones it owns.
            split = self._split_rank(name)

            def pairs(split=split):
                return pair_tokens_np(num_tokens, self.ratio, skip_tokens, self.dst_block, split)

            d = compressed_dst_np(
                num_tokens, self.ratio, dst_tables[gi], self.dst_block, skip_tokens, offset,
                rank=split,
            )
            if shared:
                gap.split(name, d, d, pairs)
                continue
            s = pd_route.compressed_src_rows(
                num_tokens, skip_tokens, src_pages[name], g,
                rank=None if split is None else rank.tp_rank,
            )
            s, d = gap.split(name, s, d, pairs)
            copy_s.append(s)
            copy_d.append(d)
        for name, gi in self._dst_owner.items():
            kind = owner_kind(name)
            if kind == "compressed":
                continue
            dst = dst_tables[gi]
            if kind == "main":
                srcs = [
                    pd_route.main_src_rows(num_tokens, skip_tokens, rank.tp_rank, src_pages[m], g)
                    for m in self._main_members(name)
                ]
                s, d = gap.split(
                    name,
                    main_sources(srcs, self._main_columns()),
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

        return (
            _cat(main_s, self._main_columns()), _cat(main_d), _cat(copy_s), _cat(copy_d),
            np.asarray(gdn_s, dtype=np.int64).reshape(-1, n_gdn),
            np.asarray(gdn_d, dtype=np.int64),
        )

    def _routed_indices_shared_rows(
        self, src_pages, dst_tables, num_tokens, skip_tokens, draft_zero_until=0
    ):
        """``_routed_indices`` under QWEN4_EXP_PD_FAST_ADMIT, with the same
        arrays. A kind's rows depend on the load and the rank, not the layer:
        each layer only indexes its own page lists with them
        (``pd_route._flat``: ``pages[row // S] * S + row % S``). So the token
        enumerations and their quotients are computed once per load (per split
        flag for compressed keys, ``_split_rank``), lazily and in the
        reference's order (its first raise is the same); paired main-KV members
        share the owner's rows (``main_sources``), and the segments are
        tile-aligned in one allocation (``cat_aligned``)."""
        g, rank, block = self.route, self.rank, self.dst_block
        stage, span = g.stage_rows, 2 * self.ratio
        main_s, main_d, copy_s, copy_d = [], [], [], []
        gdn_s, gdn_d = [], []
        gap = _DraftGap(draft_zero_until if draft_zero_until > skip_tokens else 0)
        rows: dict = {}

        def pairs(split):
            # compressed_dst_np's tokens, checks and destination quotients
            # (pair_tokens_np / pair_dst_np with ``split``).
            key = ("pairs", split is None)
            if key not in rows:
                if skip_tokens % span:
                    raise ValueError(f"A repack must start on a compressed pair ({span} tokens)")
                t = pair_tokens_np(num_tokens, self.ratio, skip_tokens, block, split)
                d = 1 if split is None else max(1, split.main_dcp)
                if d > 1 and block % span:
                    raise ValueError(
                        f"split compressed keys need whole pairs in a {block}-token block"
                    )
                rows[key] = (t, t // (d * block), (t % block) // span)
            return rows[key]

        def pair_src(split):
            # pd_route.compressed_src_rows' compact rows (rank = split's TP rank).
            key = ("pair_src", split is None)
            if key not in rows:
                index = np.arange(
                    skip_tokens // span, math.ceil(math.ceil(num_tokens / self.ratio) / 2),
                    dtype=np.int64,
                )
                if split is not None and g.dcp > 1:
                    index = index[(index * span // g.dst_block) % g.dcp == split.tp_rank % g.dcp]
                r = index - pd_route.skip_rows("compressed", skip_tokens, g)
                rows[key] = (r // stage, r % stage)
            return rows[key]

        def main():
            # main_dst_np's tokens and pd_route.main_src_rows' compact rows.
            if "main" not in rows:
                t_dst = main_tokens_np(num_tokens, block, rank, skip_tokens)
                if g.dst_block == block and g.dcp == rank.main_dcp:
                    t_src = t_dst
                else:
                    t = np.arange(skip_tokens, num_tokens, dtype=np.int64)
                    t_src = t[(t // g.dst_block) % g.dcp == rank.tp_rank % g.dcp]
                r = (
                    (t_src // g.entry) * g.dst_block
                    + t_src % g.dst_block
                    - pd_route.skip_rows("main", skip_tokens, g)
                )
                rows["main"] = (
                    t_dst, r // stage, r % stage,
                    t_dst // (rank.main_dcp * block), t_dst % block,
                )
            return rows["main"]

        def pages(values):
            return np.asarray(values, dtype=np.int64)

        shared = self._shared_load(num_tokens)
        for name, (gi, offset) in self._dst_compressed.items():
            split = self._split_rank(name)
            t, q, r = pairs(split)
            d = pages(dst_tables[gi])[q] * block + offset + r
            if shared:
                gap.split(name, d, d, lambda t=t: t)
                continue
            sq, sr = pair_src(split)
            s, d = gap.split(name, pages(src_pages[name])[sq] * stage + sr, d, lambda t=t: t)
            copy_s.append(s)
            copy_d.append(d)
        columns = self._main_columns()
        prefix = None
        for name, gi in self._dst_owner.items():
            kind = owner_kind(name)
            if kind == "compressed":
                continue
            dst = dst_tables[gi]
            if kind == "main":
                t, sq, sr, dq, dr = main()
                srcs = [pages(src_pages[m])[sq] * stage + sr for m in self._main_members(name)]
                s, d = gap.split(
                    name, main_sources(srcs, columns), pages(dst)[dq] * block + dr, lambda: t
                )
                main_s.append(s)
                main_d.append(d)
            elif kind in ("ring", "ple"):
                if prefix is None:
                    ar = np.arange(g.prefix_rows, dtype=np.int64)
                    prefix = (ar, ar // stage, ar % stage)
                ar, pq, pr = prefix
                entry = dst[self._state_entry] if kind == "ple" else dst[0]
                copy_s.append(pages(src_pages[name])[pq] * stage + pr)
                copy_d.append(entry * block + ar)
                gap.count(name, g.prefix_rows)
        n_gdn = g.gdn_rows // g.stage_rows
        for name in self._gdn_names:
            gdn_s.append(list(src_pages[name][:n_gdn]))
            gdn_d.append(dst_tables[self._dst_owner[name]][self._state_entry])
        gap.emit(copy_s, copy_d)
        self.last_draft_rows = (gap.landed, gap.zeroed)
        return (
            cat_aligned(main_s, columns), cat_aligned(main_d),
            cat_aligned(copy_s), cat_aligned(copy_d),
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

    def warm_repack(self) -> float:
        """Compiles the repack programs for this consumer's chunk shape; returns
        the seconds taken.

        Runs one null chunk through each op (the GDN-free one too when loads
        split): every index is a page-0 row or page, so it reads the carrier's
        null page and rewrites the pool's null page, which nothing reads, as
        chunk padding does. Without it the first served load compiles both
        programs on every rank. No collective: the repack has none."""
        main, copied, gdn = self._repack_shape
        empty = np.zeros(0, dtype=np.int64)
        # The served main-KV source shape: [rows, 2] when main KV is paired.
        main_src = _cat([], self._main_columns())
        gdn_src = empty
        if self.route is not None:
            pages = self.route.gdn_rows // self.route.stage_rows
            gdn_src = np.zeros((0, pages), dtype=np.int64)
        null = repack_chunks(
            main_src, empty, empty, empty, gdn_src, empty, main, gdn, copy_size=copied,
            null_rows=self._null_rows,
        )
        chunk = next(null)
        chunks = [chunk] + ([chunk[:4] + (None, None)] if self._split_gdn else [])
        t0 = time.monotonic()
        self._run_chunks(chunks)
        seconds = time.monotonic() - t0
        logger.info(
            "Qwen PD staged repack rank%d: %d program(s) warmed for chunk shape "
            "%d,%d,%d in %.1f s",
            self.rank.tp_rank, len(chunks), main, copied, gdn, seconds,
        )
        return seconds

    def plan_repack(self, uuid: int, dst_tables, num_tokens: int) -> RepackPlan:
        """Every index of one load's landing, in numpy, before any device work
        (and before any collective)."""
        pages = self.reservations[uuid]
        skip, first = self.skips.get(uuid, (0, None))
        zero_end = self.draft_zero_end(uuid, num_tokens)
        idx = self.indices(pages, dst_tables, num_tokens, skip, first, draft_zero_until=zero_end)
        self._count_draft(uuid, skip, zero_end)
        passes = (
            self.shared_passes(pages, dst_tables, num_tokens, skip)
            if getattr(self, "route", None) is not None
            else []
        )
        main, copied, gdn = self._repack_shape
        chunker = repack_chunks_packed if self._fast() else repack_chunks
        chunks = chunker(
            *idx, main, gdn, copy_size=copied, split_gdn=self._split_gdn,
            null_rows=self._null_rows,
        )
        return RepackPlan(passes=passes, chunks=list(chunks))

    def plan_repack_ahead(self, uuid: int, dst_tables, num_tokens: int) -> PlannedRepack:
        """``plan_repack`` before the load's named step (q38 SV-KVGC KVL-INWAIT): the
        same calls on the same inputs (the reservation, skips, tokens and draft-zero
        end of a load are fixed once it is reserved), without the draft counters,
        which ``repack_planned`` updates when the load lands. ``last_draft_rows`` is
        left as it was. Host only, like ``plan_repack``: numpy, no device work and
        no collective."""
        state = self.__dict__
        saved = state.pop("last_draft_rows", _ROWS_UNSET)
        try:
            pages = self.reservations[uuid]
            skip, first = self.skips.get(uuid, (0, None))
            zero_end = self.draft_zero_end(uuid, num_tokens)
            idx = self.indices(pages, dst_tables, num_tokens, skip, first, draft_zero_until=zero_end)
            rows = state.get("last_draft_rows", _ROWS_UNSET)
            passes = (
                self.shared_passes(pages, dst_tables, num_tokens, skip)
                if getattr(self, "route", None) is not None
                else []
            )
            main, copied, gdn = self._repack_shape
            chunker = repack_chunks_packed if self._fast() else repack_chunks
            chunks = chunker(
                *idx, main, gdn, copy_size=copied, split_gdn=self._split_gdn,
                null_rows=self._null_rows,
            )
            plan = RepackPlan(passes=passes, chunks=list(chunks))
        finally:
            state.pop("last_draft_rows", None)
            if saved is not _ROWS_UNSET:
                state["last_draft_rows"] = saved
        return PlannedRepack(plan=plan, skip=skip, zero_end=zero_end, draft_rows=rows)

    def repack_planned(self, uuid: int, dst_tables, num_tokens: int, planned: PlannedRepack) -> None:
        """``repack`` with a plan ``plan_repack_ahead`` made (q38 SV-KVGC KVL-INWAIT):
        the draft counters as ``plan_repack`` updates them, then the same run and
        the same verify."""
        if planned.draft_rows is not _ROWS_UNSET:
            self.last_draft_rows = planned.draft_rows
        self._count_draft(uuid, planned.skip, planned.zero_end)
        self.run_repack(planned.plan)
        skip, _ = self.skips.get(uuid, (0, None))
        if self._verify_due(skip):
            for check in (self.verify_split_keys, self.verify_main_pairs):
                try:
                    check(uuid, dst_tables, num_tokens)
                except Exception:  # a diagnostic never fails the load
                    logger.exception(
                        "Qwen PD landing verify rank%d: %s failed to run",
                        self.rank.tp_rank, check.__name__,
                    )

    def run_repack(self, plan: RepackPlan) -> None:
        if plan.passes:
            if getattr(self, "landing", None) is None:
                raise RuntimeError("a shared load landed before enable_shared")
            self.landing.run(plan.passes)
        self._run_chunks(plan.chunks)

    def repack(self, uuid: int, dst_tables, num_tokens: int) -> None:
        self.run_repack(self.plan_repack(uuid, dst_tables, num_tokens))
        skip, _ = self.skips.get(uuid, (0, None))
        if self._verify_due(skip):
            for check in (self.verify_split_keys, self.verify_main_pairs):
                try:
                    check(uuid, dst_tables, num_tokens)
                except Exception:  # a diagnostic never fails the load
                    logger.exception(
                        "Qwen PD landing verify rank%d: %s failed to run",
                        self.rank.tp_rank, check.__name__,
                    )

    # --- QWEN4_EXP_PD_VERIFY_SPLIT_KEYS: read-back of split compressed keys
    # and of paired main KV (QWEN4_EXP_PACK_MAIN_KV) ---

    def _verify_due(self, skip: int) -> bool:
        """Whether this load is checked (every rank lands the same loads in
        the same order, so all ranks check the same ones)."""
        if not verify_split_keys_enabled():
            return False
        if not (self._dcp_owners.intersection(self._dst_compressed) or self._main_pairs):
            return False
        seen = self.__dict__.setdefault("_verify_seen", {"loads": 0, "plain": 0, "prefix": 0})
        kind = "prefix" if skip else "plain"
        seen["loads"] += 1
        seen[kind] += 1
        return seen[kind] <= VERIFY_FIRST_LOADS or seen["loads"] % VERIFY_EVERY == 0

    def split_key_check_rows(self, uuid: int, dst_tables, num_tokens: int):
        """``[(owner, staging rows, pool rows)]`` of one load's check, for up to
        ``VERIFY_OWNERS`` split compressed-key owners.

        Written out apart from the repack's planners: every pair of the load
        from the local prefix on, at its staging row as it lands unsplit, then
        the pairs this rank owns by the kernel's rule, owner ``(t // B) % D``,
        row ``table[t // (D * B)] * B + slot offset + (t % B) // (2 * ratio)``
        (``pooled_qsa.compressed_pairs``). Merged or split producer keys are
        not covered (no rows)."""
        if self.src_merged:
            return []
        pages = self.reservations[uuid]
        skip, first = self.skips.get(uuid, (0, None))
        names = [
            n for n in self._dst_compressed if n in self._dcp_owners and not n.startswith("mtp.")
        ]
        if len(names) > VERIFY_OWNERS:
            names = [names[0], names[len(names) // 2], names[-1]]
        span, b, d = 2 * self.ratio, self.dst_block, self.rank.main_dcp
        pairs = math.ceil(math.ceil(num_tokens / self.ratio) / 2)
        t = np.arange(skip // span, pairs, dtype=np.int64) * span
        own = (t // b) % d == self.rank.tp_rank % d
        out = []
        for name in names:
            gi, offset = self._dst_compressed[name]
            if self.route is not None:
                src = pd_route.compressed_src_rows(num_tokens, skip, pages[name], self.route)
            else:
                sp = np.asarray(pages[name], dtype=np.int64)
                f = first[self._src_owner[name]] if first else 0
                src = sp[t // self.src_block - f] * self.src_block + (t % self.src_block) // span
            table = np.asarray(dst_tables[gi], dtype=np.int64)
            dst = table[t[own] // (d * b)] * b + offset + (t[own] % b) // span
            out.append((name, src[own], dst))
        return out

    def verify_split_keys(self, uuid: int, dst_tables, num_tokens: int, read_rows=None) -> dict:
        """Compare this rank's owned compressed pairs in the pool with the same
        pairs' staging bytes after a load landed (QWEN4_EXP_PD_VERIFY_SPLIT_KEYS);
        logs digests of both sides and the mismatched pairs, returns the counts.

        ``read_rows(src, dst)`` returns uint8 ``[n, 4, 256]`` staging rows
        ``src`` and pool rows ``dst``; by default the DMA row gather on the
        device (an XLA gather or FP8 bitcast canonicalises NaN bytes). The
        control compares each pool row with the next pair's staging row: equal
        counts near zero show the rows hold distinct keys, not zeros."""
        import hashlib

        rows = self.split_key_check_rows(uuid, dst_tables, num_tokens)
        if not rows:
            return {}
        src = np.concatenate([r[1] for r in rows])
        dst = np.concatenate([r[2] for r in rows])
        n = len(src)
        if n == 0:
            stage = pool = np.zeros((0, ROW_SUBLANES, ROW_LANES), np.uint8)
        else:
            stage, pool = (read_rows or self._read_rows_device)(src, dst)
        row = ROW_SUBLANES * ROW_LANES
        flat_p, flat_s = pool.reshape(n, row), stage.reshape(n, row)
        bad = (flat_p != flat_s).any(axis=1)
        bounds = np.cumsum([0] + [len(r[1]) for r in rows])
        by_owner = {
            name: int(bad[lo:hi].sum()) for (name, _, _), lo, hi in zip(rows, bounds, bounds[1:])
        }
        result = dict(
            pairs=n,
            mismatched=int(bad.sum()),
            by_owner=by_owner,
            nonzero=int(flat_s.any(axis=1).sum()),
            control_equal=int((flat_p[:-1] == flat_s[1:]).all(axis=1).sum()) if n > 1 else 0,
            pool_digest=hashlib.sha256(pool.tobytes()).hexdigest()[:16],
            staging_digest=hashlib.sha256(stage.tobytes()).hexdigest()[:16],
        )
        total = self.__dict__.setdefault(
            "_verify_total", dict(checks=0, pairs=0, mismatched=0)
        )
        total["checks"] += 1
        total["pairs"] += n
        total["mismatched"] += result["mismatched"]
        skip, _ = self.skips.get(uuid, (0, None))
        (logger.warning if result["mismatched"] else logger.info)(
            "Qwen PD split-key verify rank%d uuid %d tokens %d skip %d: pairs %d mismatched %d "
            "%s nonzero %d control_equal %d digest pool %s staging %s | cumulative %s",
            self.rank.tp_rank, uuid, num_tokens, skip, n, result["mismatched"], by_owner,
            result["nonzero"], result["control_equal"], result["pool_digest"],
            result["staging_digest"], total,
        )
        return result

    def main_pair_check_rows(self, uuid: int, dst_tables, num_tokens: int):
        """``[(member, head slot, staging rows, pool rows)]`` of one load's
        paired main-KV check: both members of one paired owner (the next owner
        in turn after each completed check, the same on every rank), every
        token this rank stores from the local prefix on.

        Written out apart from the repack's planners: pair rank ``(t // B) %
        D`` stores token ``t`` at row ``table[t // (D * B)] * B + t % B``
        (``qsa_rpa.virtual_page_metadata``), member ``s`` at head slots ``2s``
        and ``2s + 1`` of that row (``pooled_qsa._write_main_cache_paired``);
        the staging row is the producer's whole row, whose head ``h`` pair is
        at slots ``2h`` and ``2h + 1``. A PCP producer's split main KV is not
        covered (no rows)."""
        owners = [o for o, m in self._main_pairs.items() if len(m) == 2 and o in self._dst_owner]
        if not owners or self._src_split:
            return []
        done = self.__dict__.get("_verify_main_total", {}).get("checks", 0)
        owner = owners[done % len(owners)]
        pages = self.reservations[uuid]
        skip, first = self.skips.get(uuid, (0, None))
        b, d = self.dst_block, self.rank.main_dcp
        t = np.arange(skip, num_tokens, dtype=np.int64)
        t = t[(t // b) % d == self.rank.tp_rank % d]
        table = np.asarray(dst_tables[self._dst_owner[owner]], dtype=np.int64)
        dst = table[t // (d * b)] * b + t % b
        out = []
        for slot, member in enumerate(self._main_pairs[owner]):
            if self.route is not None:
                src = pd_route.main_src_rows(
                    num_tokens, skip, self.rank.tp_rank, pages[member], self.route
                )
            else:
                sp = np.asarray(pages[member], dtype=np.int64)
                f = first[self._src_owner[member]] if first else 0
                src = sp[t // self.src_block - f] * self.src_block + t % self.src_block
            out.append((member, slot, src, dst))
        return out

    def verify_main_pairs(self, uuid: int, dst_tables, num_tokens: int, read_rows=None) -> dict:
        """Compare both members' head-slot pairs of one paired main-KV owner in
        the pool with this rank's head pair in the members' staging rows
        after a load landed (QWEN4_EXP_PD_VERIFY_SPLIT_KEYS with
        QWEN4_EXP_PACK_MAIN_KV); logs digests of both sides and the
        mismatched rows per member, returns the counts.

        ``read_rows`` as in ``verify_split_keys``. The control compares each
        member's pool slot with the other member's staging pair: equal counts
        near zero show the two slots hold different layers' bytes."""
        import hashlib

        rows = self.main_pair_check_rows(uuid, dst_tables, num_tokens)
        if not rows:
            return {}
        src = np.concatenate([r[2] for r in rows])
        dst = np.concatenate([r[3] for r in rows])
        if len(src) == 0:
            stage = pool = np.zeros((0, ROW_SUBLANES, ROW_LANES), np.uint8)
        else:
            stage, pool = (read_rows or self._read_rows_device)(src, dst)
        h = self.rank.kv_head
        bounds = np.cumsum([0] + [len(r[2]) for r in rows])
        want = [stage[lo:hi, 2 * h : 2 * h + 2] for lo, hi in zip(bounds, bounds[1:])]
        got = [
            pool[lo:hi, 2 * slot : 2 * slot + 2]
            for (_, slot, _, _), lo, hi in zip(rows, bounds, bounds[1:])
        ]
        bad = [(g != w).any(axis=(1, 2)) for g, w in zip(got, want)]
        partner = [(got[i] == want[1 - i]).all(axis=(1, 2)) for i in range(len(rows))]
        g_all, w_all = np.concatenate(got), np.concatenate(want)
        result = dict(
            owner=rows[0][0],
            rows=int(len(src)),
            mismatched=int(sum(x.sum() for x in bad)),
            by_member={name: int(x.sum()) for (name, _, _, _), x in zip(rows, bad)},
            nonzero=int(w_all.reshape(len(w_all), 2 * ROW_LANES).any(axis=1).sum()),
            partner_equal=int(sum(x.sum() for x in partner)),
            pool_digest=hashlib.sha256(g_all.tobytes()).hexdigest()[:16],
            staging_digest=hashlib.sha256(w_all.tobytes()).hexdigest()[:16],
        )
        total = self.__dict__.setdefault(
            "_verify_main_total", dict(checks=0, rows=0, mismatched=0)
        )
        total["checks"] += 1
        total["rows"] += result["rows"]
        total["mismatched"] += result["mismatched"]
        skip, _ = self.skips.get(uuid, (0, None))
        (logger.warning if result["mismatched"] else logger.info)(
            "Qwen PD packed-main verify rank%d uuid %d tokens %d skip %d owner %s: rows %d "
            "mismatched %d %s nonzero %d partner_equal %d digest pool %s staging %s "
            "| cumulative %s",
            self.rank.tp_rank, uuid, num_tokens, skip, result["owner"], result["rows"],
            result["mismatched"], result["by_member"], result["nonzero"],
            result["partner_equal"], result["pool_digest"], result["staging_digest"], total,
        )
        return result

    def _read_rows_device(self, src, dst):
        """Staging rows ``src`` and pool rows ``dst``, read back to the host
        byte for byte, in ``VERIFY_CHUNK_ROWS`` programs of one shape."""
        if getattr(self, "_verify_op", None) is None:
            self._verify_op = self._build_verify_op()
        device = self.pool.device
        stage_parts, pool_parts = [], []
        for i in range(0, len(src), VERIFY_CHUNK_ROWS):
            k = min(VERIFY_CHUNK_ROWS, len(src) - i)
            s, d = (
                torch.from_numpy(
                    np.ascontiguousarray(_pad(v[i : i + k], VERIFY_CHUNK_ROWS), dtype=np.int32)
                ).to(device)
                for v in (src, dst)
            )
            stage_rows, pool_rows = self._verify_op(self.pool, self.staging, s, d)
            stage_parts.append(stage_rows.cpu().numpy()[:k])
            pool_parts.append(pool_rows.cpu().numpy()[:k])
        return np.concatenate(stage_parts), np.concatenate(pool_parts)

    def _build_verify_op(self):
        """The read-back as one compiled read-only program (``_build_op``'s pattern)."""
        import functools

        import jax
        import jax.numpy as jnp
        from torch_tpu._internal import pallas

        gather = functools.partial(dma_row_gather, out_dtype=jnp.uint8)

        def verify_fn(
            pool: jax.Array, staging: jax.Array, src: jax.Array, dst: jax.Array
        ) -> tuple[jax.Array, jax.Array]:
            stage_rows = gather(staging.reshape(-1, ROW_SUBLANES, ROW_LANES), src)
            pool_rows = gather(pool.reshape(-1, ROW_SUBLANES, ROW_LANES), dst)
            return stage_rows, pool_rows

        op = pallas.jax_op("pallas::qwen4_pd_verify_split_keys", verify_fn)
        op.register_fake(
            lambda pool, staging, src, dst: tuple(
                torch.empty(
                    (src.shape[0], ROW_SUBLANES, ROW_LANES), dtype=torch.uint8, device=src.device
                )
                for _ in range(2)
            )
        )

        @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
        def run(pool, staging, src, dst):
            return op(pool, staging, src, dst)

        return run

    def _run_chunks(self, chunks) -> None:
        if self._fast() and self.__dict__.get("_packed_ok", True):
            self._run_chunks_packed(chunks)
            return
        device = self.pool.device

        def dev(v):
            return torch.from_numpy(np.ascontiguousarray(v, dtype=np.int32)).to(device)

        if self._op is None:
            self._op = self._build_op(self._fn, "pallas::qwen4_pd_staged_repack")
        for chunk in chunks:
            if chunk[4] is not None:
                self._op(self.pool, self.staging, *(dev(v) for v in chunk))
                continue
            # A later chunk of a split-GDN load (repack_chunks): the repack
            # built without GDN state never reads the two GDN arguments.
            if self._op_rest is None:
                self._op_rest = self._build_op(
                    self._fn_rest, "pallas::qwen4_pd_staged_repack_no_gdn"
                )
                self._no_gdn = dev(np.zeros(1))
            index = [dev(v) for v in chunk[:4]]
            self._op_rest(self.pool, self.staging, *index, self._no_gdn, self._no_gdn)

    def _run_chunks_packed(self, chunks) -> None:
        """``_run_chunks`` with one host-to-device copy per program instead of
        one per index list (QWEN4_EXP_PD_FAST_ADMIT): a chunk's lists go out
        as one int32 vector (``pack_index``), which the compiled program
        slices back into the same arguments (``unpack_index``). If a packed
        program fails to build or run, this consumer goes back to
        ``_run_chunks``' per-list uploads from that chunk on (rewriting a
        chunk's rows is a no-op: its staging pages are still held)."""
        device = self.pool.device
        ops = self.__dict__.setdefault("_packed_ops", {})
        for i, chunk in enumerate(chunks):
            if len(chunk) == 2:  # repack_chunks_packed
                host, shapes = chunk
            else:
                host, shapes = pack_index(chunk if chunk[4] is not None else chunk[:4])
            try:
                op = ops.get(shapes)
                if op is None:
                    fn, name = (
                        (self._fn, "pallas::qwen4_pd_staged_repack_packed")
                        if len(shapes) == 6
                        else (self._fn_rest, "pallas::qwen4_pd_staged_repack_no_gdn_packed")
                    )
                    op = ops[shapes] = self._build_op(fn, name, packed=shapes)
                op(self.pool, self.staging, torch.from_numpy(host).to(device))
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(
                    "Qwen PD staged repack rank%d: packed index upload failed (%s); "
                    "per-list uploads from here on",
                    self.rank.tp_rank, exc,
                )
                self._packed_ok = False
                self._run_chunks([unpacked_chunk(c) for c in chunks[i:]])
                return

    def _build_op(self, fn, name: str, packed=None):
        """The repack as the runner's pool row copies run it (mamba_state_copy_op).

        A donating op and its ``copy_`` writeback fuse into one aliased
        program only inside ``torch.compile``; called eagerly the pool is not
        donated, and XLA copies it to give the in-place writer a buffer (a
        second 48 GiB pool: compile HBM OOM). The op is built outside the
        compiled function, which dynamo cannot trace its construction from.

        ``packed``: the shapes of the index lists, which the compiled program
        then takes as one int32 vector (``_run_chunks_packed``).
        """
        import jax
        from torch_tpu._internal import pallas

        if getattr(self, "_gather_bulk", False):
            # VLLM_TPU_PD_GATHER_BULK changes the kernel inside the op, which
            # the compiled graph names only by the op's name: the bulk
            # programs get their own names, so no graph-keyed cache (torch's
            # AOTAutogradCache, the backend's bundled entry, dynamo) can hand
            # a per-row program to a bulk consumer or the reverse.
            name = name + "_gbulk"
        if getattr(self, "_staging_fold", False):
            # VLLM_TPU_PD_STAGING_FOLD changes the landing writer inside the op:
            # its own name, for the same reason as "_gbulk".
            name = name + "_fold"

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
            traced = _gather_counts()
            written = _write_counts()
            out = fn(pool, staging, main_src, main_dst, copy_src, copy_dst, gdn_src, gdn_dst)
            gather_bulk_mismatch(self, name, traced)
            staging_fold_mismatch(self, name, written)
            return out, main_src.reshape(-1)[0]

        op = pallas.jax_op(name, repack_fn, donate_argnums=(0,))
        op.register_fake(
            lambda pool, staging, main_src, *rest: (
                torch.empty_like(pool),
                torch.empty((), dtype=main_src.dtype, device=main_src.device),
            )
        )

        if packed is not None:

            @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
            def run_packed(pool, staging, index):
                new_pool, marker = op(pool, staging, *unpack_index(index, packed))
                pool.copy_(new_pool)
                return marker

            return run_packed

        @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
        def run(pool, staging, main_src, main_dst, copy_src, copy_dst, gdn_src, gdn_dst):
            new_pool, marker = op(
                pool, staging, main_src, main_dst, copy_src, copy_dst, gdn_src, gdn_dst
            )
            pool.copy_(new_pool)
            return marker

        return run
