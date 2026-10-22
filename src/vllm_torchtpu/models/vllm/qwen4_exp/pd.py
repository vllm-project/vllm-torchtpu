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

    def page_count(self, committed_tokens: int) -> int:
        if committed_tokens < 0:
            raise ValueError("The committed prefix cannot be negative")
        if committed_tokens == 0:
            return 0
        if self.extent is TransferExtent.REQUEST_STATE:
            return 1
        return (committed_tokens + self.block_size - 1) // self.block_size


def build_transfer_groups(
    groups: Sequence[KVCacheGroupSpec],
) -> tuple[TransferGroup, ...]:
    """Preserve native group order, including one-block rings and state."""
    result = []
    seen = set()
    for index, group in enumerate(groups):
        names = tuple(group.layer_names)
        if not names or len(set(names)) != len(names) or seen.intersection(names):
            raise ValueError("Transfer groups must name every cache owner once")
        seen.update(names)
        specs = (
            tuple(group.kv_cache_spec.kv_cache_specs.values())
            if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
            else (group.kv_cache_spec,)
        )
        extents = set()
        for spec in specs:
            if isinstance(spec, MambaSpec):
                if spec.mamba_cache_mode != "none" or spec.num_speculative_blocks:
                    raise NotImplementedError(
                        "Qwen PD request-state transfer requires APC and MTP off"
                    )
                extents.add(TransferExtent.REQUEST_STATE)
            elif isinstance(spec, CircularBufferSpec):
                extents.add(TransferExtent.REQUEST_STATE)
            elif isinstance(spec, FullAttentionSpec):
                extents.add(TransferExtent.TOKEN_PAGES)
            else:
                raise NotImplementedError(
                    f"Qwen PD has no transfer extent for {type(spec).__name__}"
                )
        sizes = {spec.block_size for spec in specs}
        if len(extents) != 1 or len(sizes) != 1 or min(sizes) <= 0:
            raise ValueError("A transfer group needs one extent and positive page size")
        result.append(TransferGroup(index, names, sizes.pop(), extents.pop()))
    if not result:
        raise ValueError("Qwen PD requires a nonempty cache-group plan")
    return tuple(result)


def handoff_block_ids(
    groups: Sequence[TransferGroup],
    block_ids: Sequence[Sequence[int]],
    *,
    committed_tokens: int,
    computed_tokens: int,
) -> tuple[tuple[int, ...], ...]:
    """Validate a complete CP1/APC-off prefix before publishing its pages.

    The caller must use the producer's actual computed length. In particular,
    an N-token state cannot be published as N-1 for decoder recomputation.
    Scheduler block tables are unpadded; device table padding is not accepted.
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
        if any(type(page) is not int or page <= 0 for page in pages):
            raise ValueError(f"Group {index} contains a null or invalid block ID")
        if len(set(pages)) != len(pages):
            raise ValueError(f"Group {index} repeats a physical page")
        result.append(pages)
    return tuple(result)
