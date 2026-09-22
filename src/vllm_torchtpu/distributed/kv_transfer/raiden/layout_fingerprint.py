# SPDX-License-Identifier: Apache-2.0
"""Physical-layout identity used by raw Raiden FA transfers.

The Stage-3 data path copies PjRt bytes without materializing a logical torch
tensor.  Admission therefore measures the physical layout of an already
materialized FA cache and registers a canonical fingerprint with the Raiden
controller.  The controller compares fingerprints before it emits a plan.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from collections.abc import Callable, Mapping
from typing import Any

from vllm_torchtpu import envs as tpu_envs

from .pool_manifest import TAG_FA, TAG_GDN_CONV, TAG_GDN_SSM, TAG_MLA_NOPE, PoolManifest
from .tags import class_tag

EXPECTED_FA_MINOR_TO_MAJOR = (4, 3, 2, 1, 0)
EXPECTED_FA_TILES = ((4, 128), (4, 1))
FA_LAYOUT_FINGERPRINT_SCHEMA = "qwen35-fa-raw-layout-fingerprint-v1"
GLM_EXPECTED_MINOR_TO_MAJOR = (3, 2, 1, 0)
GLM_MLA_LAYOUT_FINGERPRINT_SCHEMA = "glm-mla-raw-layout-fingerprint-v1"
KIMI_K3_LAYOUT_FINGERPRINT_SCHEMA = "kimi-k3-row-layout-fingerprint-v1"


def canonical_layout_fingerprint(value: str | Mapping[str, Any]) -> str:
    """Return a stable identity for a calibrated physical-layout payload."""
    if isinstance(value, str):
        result = value.strip()
        if not result:
            raise ValueError("layout fingerprint must not be empty")
        return result
    if not isinstance(value, Mapping):
        raise TypeError("layout fingerprint must be a string or mapping")
    encoded = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def fa_page_tokens(manifest: PoolManifest) -> int:
    """Derive and validate the common page geometry of every FA pool."""
    if not isinstance(manifest, PoolManifest):
        raise TypeError("manifest must be a PoolManifest")
    page_tokens: set[int] = set()
    for pool in manifest.pools:
        if class_tag(pool.tag) != TAG_FA:
            continue
        if len(pool.regions) != 1:
            raise ValueError(
                "FA page geometry requires exactly one live region per pool"
            )
        region = pool.regions[0]
        tokens = int(region.num_units)
        if tokens <= 0:
            raise ValueError("FA page geometry must be positive")
        page_tokens.add(tokens)
    if not page_tokens:
        raise ValueError("FA page geometry requires an admitted FA pool")
    if len(page_tokens) != 1:
        raise ValueError(
            f"FA page geometry differs across pools: {sorted(page_tokens)}"
        )
    return next(iter(page_tokens))


def _default_layout_getter(tensor: Any) -> Any:
    from torch_tpu._internal.compile import tpu_torch_compile

    return tpu_torch_compile.get_device_layout_if_materialized(tensor)


def measured_fa_layout_fingerprint(
    manifest: PoolManifest,
    *,
    layout_getter: Callable[[Any], Any] | None = None,
    package_version: Callable[[str], str] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Measure the admitted FA storage layout and return its E0' identity.

    ``layout_getter`` and ``package_version`` are injectable so the canonical
    payload can be regression-tested on hosts without a TPU runtime.
    """
    if not isinstance(manifest, PoolManifest):
        raise TypeError("manifest must be a PoolManifest")
    fa_pools = [pool for pool in manifest.pools if class_tag(pool.tag) == TAG_FA]
    if not fa_pools:
        raise ValueError("layout fingerprint requires an admitted FA pool")
    fa_pool = fa_pools[0]
    try:
        tensor = manifest.storages[fa_pool.storage_index]
    except IndexError as exc:
        raise ValueError("FA pool storage index is outside the manifest") from exc

    getter = layout_getter or _default_layout_getter
    layout = getter(tensor)
    if layout is None:
        raise RuntimeError("admitted FA storage has no materialized TPU layout")
    raw_minor_to_major, raw_tiles, raw_element_bits = layout
    minor_to_major = tuple(int(dim) for dim in raw_minor_to_major)
    tiles = tuple(tuple(int(dim) for dim in tile) for tile in raw_tiles)
    element_bits = int(raw_element_bits or 8)
    if minor_to_major != EXPECTED_FA_MINOR_TO_MAJOR:
        raise RuntimeError(
            "FA layout failed E0' minor-to-major gate: "
            f"measured={minor_to_major} "
            f"expected={EXPECTED_FA_MINOR_TO_MAJOR}"
        )
    if tiles != EXPECTED_FA_TILES:
        raise RuntimeError(
            "FA layout failed E0' tile gate: "
            f"measured={tiles} expected={EXPECTED_FA_TILES}"
        )
    if element_bits != 8:
        raise RuntimeError(
            "FA layout failed E0' element-size gate: "
            f"measured={element_bits} expected=8"
        )

    version = package_version or importlib.metadata.version
    payload = {
        "schema": FA_LAYOUT_FINGERPRINT_SCHEMA,
        "torch_tpu": version("torch_tpu"),
        "libtpu": version("libtpu"),
        "minor_to_major": list(minor_to_major),
        "tiles": [list(tile) for tile in tiles],
        "element_size_in_bits": element_bits,
        # GDN conv state layout version.  Both sides of a disagg pair must
        # agree: the byte-span convention (pair-blocked whole-token QK
        # spans vs the legacy split Q/K) is baked into the transfer plan,
        # and the geometries are size-identical, so a mixed pair would not
        # otherwise fail closed.  The controller compares fingerprints
        # before emitting a plan.
        "gdn_conv_layout": (
            "qk-pair-v1" if tpu_envs.TPU_GDN_CONV_QK_PAIR_LAYOUT else "legacy-split-qk"
        ),
    }
    return canonical_layout_fingerprint(payload), payload


__all__ = [
    "EXPECTED_FA_MINOR_TO_MAJOR",
    "EXPECTED_FA_TILES",
    "FA_LAYOUT_FINGERPRINT_SCHEMA",
    "GLM_MLA_LAYOUT_FINGERPRINT_SCHEMA",
    "KIMI_K3_LAYOUT_FINGERPRINT_SCHEMA",
    "canonical_layout_fingerprint",
    "fa_page_tokens",
    "measured_fa_layout_fingerprint",
    "measured_glm_layout_fingerprint",
    "measured_kimi_k3_layout_fingerprint",
]


def measured_kimi_k3_layout_fingerprint(
    manifest: PoolManifest,
    *,
    page_tokens: int,
    state_fragment_heads: int,
    state_fragments: int,
    layout_getter: Callable[[Any], Any] | None = None,
    package_version: Callable[[str], str] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Normalize the existing packed-row fingerprint across Kimi TP sizes."""
    if state_fragment_heads <= 0:
        raise ValueError("state_fragment_heads must be positive")
    if state_fragments <= 0:
        raise ValueError("state_fragments must be positive")
    if not any(pool.tag == TAG_FA for pool in manifest.pools):
        raise ValueError("Kimi layout fingerprint requires an FA pool")
    version = package_version or importlib.metadata.version
    payload = {
        "schema": KIMI_K3_LAYOUT_FINGERPRINT_SCHEMA,
        "torch_tpu": version("torch_tpu"),
        "libtpu": version("libtpu"),
        # The planner splits at destination page boundaries, so Kimi compares
        # physical rows independent of the role's page size.
        "layouts": _measure_packed_row_layouts(
            manifest,
            page_tokens=page_tokens,
            layout_getter=layout_getter,
            include_page_shape=False,
        ),
    }

    state_layouts: dict[str, int] = {}
    for pool in manifest.pools:
        state_tag = next(
            (tag for tag in (TAG_GDN_CONV, TAG_GDN_SSM) if pool.tag.startswith(tag)),
            None,
        )
        if state_tag is None:
            continue
        live_bytes = pool.live_bytes_per_block
        if live_bytes % state_fragments:
            raise ValueError(
                "Kimi live state must divide into source-sized "
                f"fragments: tag={pool.tag}"
            )
        fragment_bytes = live_bytes // state_fragments
        known = state_layouts.setdefault(state_tag, fragment_bytes)
        if known != fragment_bytes:
            raise ValueError(f"Kimi {state_tag} pools disagree on live state")
    if set(state_layouts) != {TAG_GDN_CONV, TAG_GDN_SSM}:
        raise ValueError("Kimi layout requires gdn.conv and gdn.ssm pools")
    payload["gdn_state_layouts"] = state_layouts
    payload["gdn_conv_layout"] = "kda-rank-blocked-tap-qkv-head-dim-bf16-v1"
    payload["state_fragment_heads"] = int(state_fragment_heads)
    return canonical_layout_fingerprint(payload), payload


def measured_glm_layout_fingerprint(
    manifest: PoolManifest,
    *,
    page_tokens: int,
    layout_getter: Callable[[Any], Any] | None = None,
    package_version: Callable[[str], str] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Fingerprint of the admitted GLM cache layout, compared across peers.

    Hash per-page geometry only: both peers must agree on it, but they size
    their pools independently.
    """
    layouts = _measure_packed_row_layouts(
        manifest,
        page_tokens=page_tokens,
        layout_getter=layout_getter,
        include_page_shape=True,
    )
    version = package_version or importlib.metadata.version
    payload = {
        "schema": GLM_MLA_LAYOUT_FINGERPRINT_SCHEMA,
        "torch_tpu": version("torch_tpu"),
        "libtpu": version("libtpu"),
        "page_tokens": int(page_tokens),
        "layouts": layouts,
    }
    return canonical_layout_fingerprint(payload), payload


def _measure_packed_row_layouts(
    manifest: PoolManifest,
    *,
    page_tokens: int,
    layout_getter: Callable[[Any], Any] | None,
    include_page_shape: bool,
) -> dict[str, Any]:
    """Validate live packed-row storage for the model-specific fingerprints.

    The transfer copies raw bytes, so a prefix of rows must also be a byte
    prefix of the page. That only holds for the natural
    [blocks, rows, packing, width] physical order, so reject anything else --
    for every pool, since one odd layer would corrupt just that layer.
    """
    if not isinstance(manifest, PoolManifest):
        raise TypeError("manifest must be a PoolManifest")
    if page_tokens <= 0:
        raise ValueError("page_tokens must be positive")
    getter = layout_getter or _default_layout_getter

    # Keyed by class tag: per-layer tags describe the same physical layout
    # per class, and a producer holding a layer subset must fingerprint
    # identically to the destination holding every layer.
    per_tag: dict[str, Any] = {}
    shape_by_tag: dict[str, tuple[int, ...]] = {}
    for pool in manifest.pools:
        tag = class_tag(pool.tag)
        tensor = manifest.storages[pool.storage_index]
        shape = tuple(int(dim) for dim in getattr(tensor, "shape", ()))
        if len(shape) != 4:
            raise RuntimeError(
                f"admitted {pool.tag} storage must be rank-4: shape={shape}"
            )
        known = shape_by_tag.setdefault(tag, shape)
        if known != shape:
            raise RuntimeError(
                f"admitted {pool.tag} storages disagree on shape: {known} vs {shape}"
            )
        layout = getter(tensor)
        if layout is None:
            raise RuntimeError(
                f"admitted {pool.tag} storage has no materialized TPU layout"
            )
        raw_minor_to_major, raw_tiles, _ = layout
        minor_to_major = tuple(int(dim) for dim in raw_minor_to_major)
        tiles = tuple(tuple(int(dim) for dim in tile) for tile in raw_tiles)
        # The measured element bits are unreliable (0 for these tensors);
        # the dtype is authoritative.
        element_bits = 8 * int(tensor.element_size())
        _, rows, packing, width = shape
        page_rows_tokens = rows if tag == TAG_MLA_NOPE else rows * packing
        if page_rows_tokens != page_tokens:
            raise RuntimeError(
                f"{pool.tag} page geometry {rows}x{packing} does not match "
                f"page_tokens {page_tokens}"
            )
        if minor_to_major != GLM_EXPECTED_MINOR_TO_MAJOR:
            raise RuntimeError(
                "Packed-row transfers require the natural physical order: "
                f"tag={pool.tag}, layer={pool.layer_name}, "
                f"minor_to_major={minor_to_major}"
            )
        if tiles != ((packing, 128), (packing, 1)):
            raise RuntimeError(
                "Packed-row transfers require the packed tile shape "
                f"(({packing},128),({packing},1)): tag={pool.tag}, "
                f"layer={pool.layer_name}, tiles={tiles}"
            )
        measured = per_tag.setdefault(
            tag,
            {
                "row_bytes": packing * width * element_bits // 8,
                "minor_to_major": list(minor_to_major),
                "tiles": [list(tile) for tile in tiles],
                "element_size_in_bits": element_bits,
            },
        )
        if include_page_shape:
            measured["page_shape"] = [rows, packing, width]
    return per_tag
