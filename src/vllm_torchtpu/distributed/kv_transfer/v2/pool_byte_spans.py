# SPDX-License-Identifier: Apache-2.0
"""Native byte-span declarations for Stage-3 resharding.

The producer lowers the PCP kernel's token ownership directly to the byte
ranges consumed by Raiden.  GDN state is head-sharded across PCP ranks, so its
declaration similarly lowers each rank's compact local state into the
corresponding head ranges of the full destination state.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from .common import TAG_GDN_CONV, TAG_GDN_SSM

# Physical granule of the admitted raw TPU FP8 pool layout: byte ranges
# whose offsets and sizes are whole multiples of this are placement-exact
# under the tiled physical layout, so raw span lowering fails closed on
# anything finer.
_TPU_PHYSICAL_TOKEN_BYTES = 1024


@dataclasses.dataclass(frozen=True)
class PoolByteSpan:
    """One byte copy between a registered source and destination block."""

    src_block_ordinal: int
    src_offset_bytes: int
    dst_block_index: int
    dst_offset_bytes: int
    size_bytes: int
    src_stride_bytes: int = 0
    dst_stride_bytes: int = 0
    count: int = 1


@dataclasses.dataclass(frozen=True)
class PoolSpanRegistration:
    """One rank's byte declarations for one exact pool tag."""

    tag: str
    block_ids: tuple[int, ...]
    spans: tuple[PoolByteSpan, ...]
    declared_bytes: int
    # 0: page-indexed destination spans (state classes). 1: request-global
    # compact destination byte space (FA) — the controller splits at
    # destination pages (T3.4).
    dst_space_version: int = 0


def owned_token_ranges(
    *,
    num_tokens: int,
    transfer_rank: int,
    parallelism: int,
    interleave_tokens: int,
) -> list[tuple[int, int]]:
    """Return this producer rank's PCP-owned destination token ranges."""
    from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import pcp_layout

    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    if parallelism <= 0:
        raise ValueError("parallelism must be positive")
    if not 0 <= transfer_rank < parallelism:
        raise ValueError(
            f"transfer_rank {transfer_rank} is outside parallelism "
            f"{parallelism}")
    if parallelism == 1:
        return [(0, num_tokens)]
    if interleave_tokens <= 0:
        raise ValueError("interleave_tokens must be positive")
    return list(
        pcp_layout.pcp_query_chunk_ranges(num_tokens, 0, transfer_rank,
                                          parallelism, interleave_tokens))


def lower_fa_spans(
    *,
    num_tokens: int,
    transfer_rank: int,
    parallelism: int,
    interleave_tokens: int,
    page_tokens: int,
    token_bytes: int,
    block_ids: Sequence[int],
) -> PoolSpanRegistration:
    """Lower PCP ownership to source-page / global-destination byte ranges.

    T3.4: spans are destination-page-agnostic — dst offsets address the
    request-global compact live byte space (dst_space_version=1) and the
    controller, which holds the destination manifest, splits them at
    destination page boundaries at plan build. The producer only splits at
    its OWN page boundaries (source geometry it authoritatively knows).
    """
    if page_tokens <= 0:
        raise ValueError("page_tokens must be positive")
    if token_bytes <= 0:
        raise ValueError("token_bytes must be positive")

    ranges = owned_token_ranges(
        num_tokens=num_tokens,
        transfer_rank=transfer_rank,
        parallelism=parallelism,
        interleave_tokens=interleave_tokens,
    )
    spans: list[PoolByteSpan] = []
    local_cursor = 0
    for start, end in ranges:
        cursor = start
        while cursor < end:
            block_ordinal, block_token_offset = divmod(local_cursor,
                                                       page_tokens)
            source_end = min(end, cursor + page_tokens - block_token_offset)
            spans.append(
                PoolByteSpan(
                    src_block_ordinal=block_ordinal,
                    src_offset_bytes=block_token_offset * token_bytes,
                    dst_block_index=0,
                    dst_offset_bytes=cursor * token_bytes,
                    size_bytes=(source_end - cursor) * token_bytes,
                ))
            local_cursor += source_end - cursor
            cursor = source_end

    expected_blocks = (local_cursor + page_tokens - 1) // page_tokens
    if len(block_ids) != expected_blocks:
        raise ValueError(
            "block_ids do not match dense PCP rank packing: "
            f"got {len(block_ids)}, expected {expected_blocks} for "
            f"{local_cursor} owned tokens")
    return PoolSpanRegistration(
        tag="fa",
        block_ids=tuple(int(block_id) for block_id in block_ids),
        spans=tuple(spans),
        declared_bytes=local_cursor * token_bytes,
        dst_space_version=1,
    )


def _region_value(region: object, name: str) -> int | str:
    if isinstance(region, dict):
        value = region[name]
    else:
        value = getattr(region, name)
    return str(value) if name == "name" else int(value)


def lower_gdn_state_shard_spans(
    *,
    tag: str,
    block_id: int,
    transfer_rank: int,
    parallelism: int,
    regions: Sequence[object],
) -> PoolSpanRegistration:
    """Lower one rank's GDN head shard into a full destination state.

    Under the QK pair-blocked conv layout the rank-local conv row is
    ``[QK | V]`` where the QK block shares whole pool tokens (Q and K
    row-pairs interleaved) and V is whole-token aligned.  The full-width
    destination stores the rank QK blocks in rank order followed by the
    rank V blocks in rank order, which is exactly the stored order the
    pooled kernel's ``rows_perm`` maps back to logical ``[Q | K | V]``.
    Each rank therefore contributes two contiguous spans per tap; every
    span offset and size is a whole multiple of the physical pool token,
    so the raw (physical-byte) transport moves them placement-exactly.

    SSM state is head-major and whole-token aligned: one contiguous span
    at a rank-dependent destination offset.

    ``regions`` comes from the admitted source pool manifest.  The legacy
    ``gdn_conv_q``/``gdn_conv_k`` vocabulary is rejected: its half-token
    Q/K extents are not placement-exact under the raw tiled layout.
    """
    if not tag:
        raise ValueError("state pool tag must not be empty")
    if block_id < 0:
        raise ValueError("state block id must be non-negative")
    if parallelism <= 0:
        raise ValueError("parallelism must be positive")
    if not 0 <= transfer_rank < parallelism:
        raise ValueError(
            f"transfer_rank {transfer_rank} is outside parallelism "
            f"{parallelism}")

    is_conv = tag == TAG_GDN_CONV or tag.startswith(f"{TAG_GDN_CONV}.")
    is_ssm = tag == TAG_GDN_SSM or tag.startswith(f"{TAG_GDN_SSM}.")
    if not is_conv and not is_ssm:
        raise ValueError(f"unsupported GDN state tag {tag!r}")

    normalized: dict[str, tuple[int, int, int, int, int]] = {}
    for region in regions:
        name = str(_region_value(region, "name"))
        if name in normalized:
            raise ValueError(f"duplicate GDN state region {name!r} ({tag})")
        values = tuple(
            int(_region_value(region, field)) for field in (
                "offset_bytes",
                "stride_bytes",
                "unit_bytes",
                "num_units",
                "units_per_stride",
            ))
        if min(values
               ) < 0 or values[2] <= 0 or values[3] <= 0 or values[4] <= 0:
            raise ValueError(f"invalid GDN state region {name!r} ({tag})")
        normalized[name] = values

    if is_ssm:
        if set(normalized) != {"gdn_ssm"}:
            raise ValueError(f"GDN SSM layout requires exactly gdn_ssm, got "
                             f"{sorted(normalized)}")
        offset, stride, unit, heads, units_per_stride = normalized["gdn_ssm"]
        packed_head_bytes = unit * units_per_stride
        if offset != 0 or stride != packed_head_bytes:
            raise ValueError(
                "GDN SSM compact layout must be dense and head-major: "
                f"offset={offset}, stride={stride}, "
                f"head_bytes={packed_head_bytes}")
        local_live = heads * packed_head_bytes
        if local_live % _TPU_PHYSICAL_TOKEN_BYTES:
            raise ValueError(
                "raw TPU GDN SSM shard must contain whole physical tokens: "
                f"bytes={local_live}")
        if (transfer_rank * local_live) % _TPU_PHYSICAL_TOKEN_BYTES:
            raise ValueError(
                "raw TPU GDN SSM destination must be token aligned")
        return PoolSpanRegistration(
            tag=tag,
            block_ids=(int(block_id), ),
            spans=(PoolByteSpan(
                src_block_ordinal=0,
                src_offset_bytes=0,
                dst_block_index=0,
                dst_offset_bytes=transfer_rank * local_live,
                size_bytes=local_live,
            ), ),
            declared_bytes=local_live,
        )

    names = set(normalized)
    if names == {"gdn_conv_qk", "gdn_conv_v"}:
        # QK pair-blocked layout (source transfer degree >= 8): the rank's
        # QK block shares whole pool tokens, so one QK span and one V span
        # per tap are placement-exact.
        qk = normalized["gdn_conv_qk"]
        v = normalized["gdn_conv_v"]
        taps = qk[3]
        if v[3] != taps:
            raise ValueError("GDN conv qk/v regions disagree on tap count")
        segments = (("qk", qk, qk[2] * qk[4]), ("v", v, v[2] * v[4]))
    elif names == {"gdn_conv_q", "gdn_conv_k", "gdn_conv_v"}:
        # Segment-major layout (source transfer degree < 8): each per-rank
        # segment is a whole-token multiple on its own, so the classic one
        # span per segment per tap is placement-exact.  At degree >= 8 the
        # Q/K segments are sub-token and the alignment guard below fails
        # this path closed.
        q = normalized["gdn_conv_q"]
        k = normalized["gdn_conv_k"]
        v = normalized["gdn_conv_v"]
        taps = q[3]
        if k[3] != taps or v[3] != taps:
            raise ValueError("GDN conv q/k/v regions disagree on tap count")
        segments = (("q", q, q[2] * q[4]), ("k", k, k[2] * k[4]),
                    ("v", v, v[2] * v[4]))
    else:
        raise ValueError(
            "GDN conv regions must be {gdn_conv_qk, gdn_conv_v} "
            "(QK pair-blocked layout, TPU_GDN_CONV_QK_PAIR_LAYOUT) or "
            "{gdn_conv_q, gdn_conv_k, gdn_conv_v}: got "
            f"{sorted(names)}")

    local_row_bytes = sum(size for _, _, size in segments)
    expected_offset = 0
    for name, region, size in segments:
        if region[0] != expected_offset:
            raise ValueError(
                "GDN conv compact layout must be dense tap-major: "
                f"{name} offset={region[0]}, expected={expected_offset}")
        if region[1] != local_row_bytes:
            raise ValueError(
                "GDN conv regions must share the tap row stride: "
                f"{name} stride={region[1]}, row={local_row_bytes}")
        expected_offset += size
    dst_row_bytes = parallelism * local_row_bytes
    for name, value in ([(name, size) for name, _, size in segments] +
                        [("row", local_row_bytes),
                         ("dst_row", dst_row_bytes)]):
        if value % _TPU_PHYSICAL_TOKEN_BYTES:
            raise ValueError(
                "raw TPU GDN conv extents must be whole physical tokens "
                "(sub-token Q/K segments need the QK pair-blocked layout): "
                f"{name}={value}")

    spans = []
    src_offset = 0
    dst_segment_base = 0
    for _, _, size in segments:
        spans.append(
            PoolByteSpan(
                src_block_ordinal=0,
                src_offset_bytes=src_offset,
                dst_block_index=0,
                dst_offset_bytes=dst_segment_base + transfer_rank * size,
                size_bytes=size,
                src_stride_bytes=local_row_bytes,
                dst_stride_bytes=dst_row_bytes,
                count=taps,
            ))
        src_offset += size
        dst_segment_base += parallelism * size
    return PoolSpanRegistration(
        tag=tag,
        block_ids=(int(block_id), ),
        spans=tuple(spans),
        declared_bytes=taps * local_row_bytes,
    )
