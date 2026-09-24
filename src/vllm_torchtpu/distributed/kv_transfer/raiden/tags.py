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
