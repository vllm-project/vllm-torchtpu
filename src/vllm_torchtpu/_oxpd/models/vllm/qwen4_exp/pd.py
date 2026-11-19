# SPDX-License-Identifier: Apache-2.0
"""Logical committed-prefix pages for Qwen's native cache groups.

These records describe scheduler ownership, not native byte offsets. Transport
admission must separately validate the materialized layout and completion fence.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheGroupSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)


class TransferExtent(Enum):
    TOKEN_PAGES = "token_pages"
    REQUEST_STATE = "request_state"


@dataclass(frozen=True)
class TransferGroup:
    group_index: int
    layer_names: tuple[str, ...]
    block_size: int
    extent: TransferExtent
    # A request-state group whose scheduler table is indexed by block column
    # (align-mode Mamba under prefix caching): one entry per ``block_size``
    # tokens, null or checkpoint entries before the last, and the running
    # state in the last entry. Only that entry is transferred.
    positional_state: bool = False

    def page_count(self, committed_tokens: int) -> int:
        if committed_tokens < 0:
            raise ValueError("The committed prefix cannot be negative")
        if committed_tokens == 0:
            return 0
        if self.extent is TransferExtent.REQUEST_STATE:
            return 1
        return (committed_tokens + self.block_size - 1) // self.block_size


def skipped_source_pages(group: TransferGroup, skip_tokens: int) -> int:
    """Source pages of ``group`` wholly inside a ``skip_tokens`` local prefix.

    Request-state pages are not prefix-decomposable and always move whole.
    """
    if skip_tokens < 0:
        raise ValueError("A local prefix cannot be negative")
    if group.extent is TransferExtent.REQUEST_STATE:
        return 0
    return skip_tokens // group.block_size


def suffix_pages(
    group: TransferGroup, pages: Sequence[int], skip_tokens: int
) -> tuple[int, ...]:
    """The destination pages a prefix-aware load fills: a token group's pages
    after a page-aligned local prefix, a request-state group's page whole."""
    pages = tuple(pages)
    if group.extent is TransferExtent.REQUEST_STATE or skip_tokens == 0:
        return pages
    if skip_tokens % group.block_size:
        raise ValueError(
            f"Group {group.group_index}'s local prefix of {skip_tokens} tokens "
            f"does not end on a {group.block_size}-token page"
        )
    skipped = skip_tokens // group.block_size
    if skipped >= len(pages):
        raise ValueError(
            f"Group {group.group_index}'s local prefix covers the whole transfer"
        )
    return pages[skipped:]


def build_transfer_groups(
    groups: Sequence[KVCacheGroupSpec],
) -> tuple[TransferGroup, ...]:
    """Preserve native group order, including one-block rings and state."""
    result = []
    seen = set()
    for index, group in enumerate(groups):
        names = tuple(group.layer_names)
        # A pipeline stage's projected plan keeps every whole-model group, in
        # order, so group indices and the scheduler's block tables line up;
        # groups whose layers all live on other stages have no local owners.
        if len(set(names)) != len(names) or seen.intersection(names):
            raise ValueError("Transfer groups must name every cache owner once")
        seen.update(names)
        specs = (
            tuple(group.kv_cache_spec.kv_cache_specs.values())
            if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
            else (group.kv_cache_spec,)
        )
        extents = set()
        positional = set()
        for spec in specs:
            if isinstance(spec, MambaSpec):
                # "none": one persistent state block per request. "align"
                # (prefix caching): one state block per block column, the
                # running state in the last; the transfer still carries one.
                # Speculative state blocks (a decode-only MTP draft) follow
                # the committed state in the table in either mode, so the
                # committed entry is the -(K + 1)th; the transfer moves only
                # it (_Stage3TransferGroup.transfer_pages) and pd_staging
                # lands the producer's state there.
                if spec.mamba_cache_mode not in ("none", "align"):
                    raise NotImplementedError(
                        "Qwen PD request-state transfer requires the none or "
                        "align Mamba cache mode"
                    )
                extents.add(TransferExtent.REQUEST_STATE)
                positional.add(spec.mamba_cache_mode == "align")
            elif isinstance(spec, CircularBufferSpec):
                extents.add(TransferExtent.REQUEST_STATE)
                positional.add(False)
            elif isinstance(spec, FullAttentionSpec):
                extents.add(TransferExtent.TOKEN_PAGES)
                positional.add(False)
            else:
                raise NotImplementedError(
                    f"Qwen PD has no transfer extent for {type(spec).__name__}"
                )
        sizes = {spec.block_size for spec in specs}
        if (
            len(extents) != 1
            or len(positional) != 1
            or len(sizes) != 1
            or min(sizes) <= 0
        ):
            raise ValueError(
                "A transfer group needs one extent, one state indexing and a "
                "positive page size"
            )
        result.append(
            TransferGroup(index, names, sizes.pop(), extents.pop(), positional.pop())
        )
    if not result:
        raise ValueError("Qwen PD requires a nonempty cache-group plan")
    return tuple(result)


def held_heads(
    groups: Sequence[TransferGroup],
    block_ids: Sequence[Sequence[int]],
    held,
) -> tuple[int, ...]:
    """The leading null entries of each held group's table, 0 for every other
    group (q38 F2-HOLD, QWEN4_EXP_PD_DRAFT_HOLD).

    A held group is one of an F2 producer's MTP draft token groups: its
    blocks below the request's prefix hit are vLLM's null block (ID 0), since
    the producer never writes draft rows there (core/producer_draft_hold).
    """
    heads = []
    for group, ids in zip(groups, block_ids):
        n = 0
        if group.group_index in held and group.extent is TransferExtent.TOKEN_PAGES:
            for page in ids:
                if page != 0:
                    break
                n += 1
        heads.append(n)
    return tuple(heads)


def handoff_block_ids(
    groups: Sequence[TransferGroup],
    block_ids: Sequence[Sequence[int]],
    *,
    committed_tokens: int,
    computed_tokens: int,
    heads: Sequence[int] | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Validate a complete CP1 prefix before publishing its pages.

    The caller must use the producer's actual computed length. In particular,
    an N-token state cannot be published as N-1 for decoder recomputation.
    Scheduler block tables are unpadded; device table padding is not accepted.
    Request-state groups carry exactly one page: a positional (align-mode)
    state table must already be reduced to its live page (the Stage-3
    scheduler does so when it records the send).

    ``heads`` (q38 F2-HOLD, ``held_heads``): per group, how many leading null
    entries its table may open with; those come off the returned pages, and
    at least one live page must remain.
    """
    if committed_tokens != computed_tokens:
        raise ValueError(
            "All Qwen states must match the published committed prefix: "
            f"computed={computed_tokens}, committed={committed_tokens}"
        )
    if len(block_ids) != len(groups):
        raise ValueError("Handoff requires one block table per cache group")
    result = []
    for index, (group, ids) in enumerate(zip(groups, block_ids)):
        if group.group_index != index:
            raise ValueError("Transfer groups must preserve scheduler group order")
        expected = group.page_count(committed_tokens)
        pages = tuple(ids)
        if len(pages) != expected:
            raise ValueError(
                f"Group {index} needs {expected} pages for {committed_tokens} "
                f"committed tokens, got {len(pages)}"
            )
        head = int(heads[index]) if heads else 0
        if head:
            if head >= len(pages) or any(page != 0 for page in pages[:head]):
                raise ValueError(
                    f"Group {index}'s held table must open with {head} null blocks "
                    f"and keep a live page, got {len(pages)} pages"
                )
            pages = pages[head:]
        if any(type(page) is not int or page <= 0 for page in pages):
            raise ValueError(f"Group {index} contains a null or invalid block ID")
        if len(set(pages)) != len(pages):
            raise ValueError(f"Group {index} repeats a physical page")
        result.append(pages)
    return tuple(result)
