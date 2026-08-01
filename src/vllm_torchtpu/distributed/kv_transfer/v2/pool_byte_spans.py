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
    """Lower one PCP rank's GDN head shard into a full destination state.

    The compact-live conv layout on each source rank is ``taps`` rows of
    ``[q_local | k_local | v_local]``.  The non-PCP destination layout is
    ``[q_rank0..N | k_rank0..N | v_rank0..N]`` in every row, so a contiguous
    whole-slot copy cannot preserve head order.  Three strided spans scatter
    the local q/k/v runs into their rank-ordered destination ranges.  SSM
    state is already head-major and therefore needs one contiguous span at a
    rank-dependent destination offset.

    ``regions`` comes from the admitted source pool manifest.  Validating its
    exact dense GDN shape here keeps the byte declaration tied to the layout
    used by the model kernel instead of relying on pinned model sizes.
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

    expected_names = {"gdn_conv_q", "gdn_conv_k", "gdn_conv_v"}
    if set(normalized) != expected_names:
        raise ValueError("GDN conv layout requires q/k/v regions, got "
                         f"{sorted(normalized)}")
    q = normalized["gdn_conv_q"]
    k = normalized["gdn_conv_k"]
    v = normalized["gdn_conv_v"]
    taps = q[3]
    if k[3] != taps or v[3] != taps:
        raise ValueError("GDN conv q/k/v regions disagree on tap count")
    q_bytes = q[2] * q[4]
    k_bytes = k[2] * k[4]
    v_bytes = v[2] * v[4]
    local_row_bytes = q_bytes + k_bytes + v_bytes
    if ((q[0], k[0], v[0]) != (0, q_bytes, q_bytes + k_bytes)
            or any(region[1] != local_row_bytes for region in (q, k, v))):
        raise ValueError(
            "GDN conv compact layout must be dense tap-major q/k/v: "
            f"offsets={(q[0], k[0], v[0])}, "
            f"strides={(q[1], k[1], v[1])}, row={local_row_bytes}")

    dst_row_bytes = parallelism * local_row_bytes
    spans = (
        PoolByteSpan(
            src_block_ordinal=0,
            src_offset_bytes=0,
            dst_block_index=0,
            dst_offset_bytes=transfer_rank * q_bytes,
            size_bytes=q_bytes,
            src_stride_bytes=local_row_bytes,
            dst_stride_bytes=dst_row_bytes,
            count=taps,
        ),
        PoolByteSpan(
            src_block_ordinal=0,
            src_offset_bytes=q_bytes,
            dst_block_index=0,
            dst_offset_bytes=parallelism * q_bytes + transfer_rank * k_bytes,
            size_bytes=k_bytes,
            src_stride_bytes=local_row_bytes,
            dst_stride_bytes=dst_row_bytes,
            count=taps,
        ),
        PoolByteSpan(
            src_block_ordinal=0,
            src_offset_bytes=q_bytes + k_bytes,
            dst_block_index=0,
            dst_offset_bytes=(parallelism * (q_bytes + k_bytes) +
                              transfer_rank * v_bytes),
            size_bytes=v_bytes,
            src_stride_bytes=local_row_bytes,
            dst_stride_bytes=dst_row_bytes,
            count=taps,
        ),
    )
    return PoolSpanRegistration(
        tag=tag,
        block_ids=(int(block_id), ),
        spans=spans,
        declared_bytes=taps * local_row_bytes,
    )
