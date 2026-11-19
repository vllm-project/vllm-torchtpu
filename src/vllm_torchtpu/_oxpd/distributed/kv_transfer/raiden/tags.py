# SPDX-License-Identifier: Apache-2.0
"""Pool-tag vocabulary shared with the Raiden controller.

These strings are wire-visible: they name pools in manifests and byte-span
registrations on both sides of a disagg pair, and the live-pair acceptance
gate builds its expected tag list from them. The values are frozen; changing
one is a coupled two-repo protocol change, not a refactor.
"""

TAG_FA = "fa"
TAG_GDN_CONV = "gdn.conv"
TAG_GDN_SSM = "gdn.ssm"
TAG_DSA_IDX = "dsa.idx"
TAG_MLA_NOPE = "mla.nope"
TAG_MLA_ROPE = "mla.rope"

# DeepSeek-V4. Its logical caches alias one array at the same byte offset and
# are told apart only by block ID, so every tag carries a ``.g{index}`` suffix
# naming its kv-cache group. Without it a tag would address another cache's
# pages.
TAG_DSV4_CSA_NOPE = "dsv4.csa.nope"
TAG_DSV4_CSA_ROPE = "dsv4.csa.rope"
TAG_DSV4_IDX = "dsv4.idx"
TAG_DSV4_HCA = "dsv4.hca"
TAG_DSV4_SWA = "dsv4.swa"
# The compressor's paged staging buffer. One tag per compressed-KV role,
# because their windows, and so their per-request block counts, differ.
TAG_DSV4_STATE_CSA = "dsv4.state.csa"
TAG_DSV4_STATE_IDX = "dsv4.state.idx"
TAG_DSV4_STATE_HCA = "dsv4.state.hca"


def dsv4_group_tag(base_tag: str, group_index: int) -> str:
    """``dsv4.swa`` + group 3 -> ``dsv4.swa.g3``."""
    return f"{base_tag}.g{int(group_index)}"


# Per-layer tags: ``<class tag>.l<layer index>``. A pipeline-parallel
# producer stage registers only its own layers, so pools pair up by layer
# rather than by manifest position; both transfer peers must build their
# manifests with the same policy.
_LAYER_SUFFIX = ".l"


def layer_tag(tag: str, layer_index: int) -> str:
    """Tag of one layer's pool under the per-layer tagging policy."""
    if layer_index < 0:
        raise ValueError(f"layer index must be non-negative: {layer_index}")
    return f"{tag}{_LAYER_SUFFIX}{layer_index}"


def split_layer_tag(tag: str) -> tuple[str, int | None]:
    """(class tag, layer index) of a tag; the index is None when absent."""
    head, sep, tail = tag.rpartition(_LAYER_SUFFIX)
    if sep and tail.isdigit():
        return head, int(tail)
    return tag, None


def class_tag(tag: str) -> str:
    """The tag with any per-layer suffix removed."""
    return split_layer_tag(tag)[0]
