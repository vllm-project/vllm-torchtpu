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

from .pool_manifest import TAG_FA, PoolManifest

EXPECTED_FA_MINOR_TO_MAJOR = (4, 3, 2, 1, 0)
EXPECTED_FA_TILES = ((4, 128), (4, 1))
FA_LAYOUT_FINGERPRINT_SCHEMA = "qwen35-fa-raw-layout-fingerprint-v1"


def canonical_layout_fingerprint(value: str | Mapping[str, Any]) -> str:
    """Return a stable identity for a calibrated physical-layout payload."""
    if isinstance(value, str):
        result = value.strip()
        if not result:
            raise ValueError("layout fingerprint must not be empty")
        return result
    if not isinstance(value, Mapping):
        raise TypeError("layout fingerprint must be a string or mapping")
    encoded = json.dumps(dict(value),
                         sort_keys=True,
                         separators=(",", ":"),
                         ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def fa_page_tokens(manifest: PoolManifest) -> int:
    """Derive and validate the common page geometry of every FA pool."""
    if not isinstance(manifest, PoolManifest):
        raise TypeError("manifest must be a PoolManifest")
    page_tokens: set[int] = set()
    for pool in manifest.pools:
        if pool.tag != TAG_FA:
            continue
        if len(pool.regions) != 1:
            raise ValueError(
                "FA page geometry requires exactly one live region per pool")
        region = pool.regions[0]
        tokens = int(region.num_units)
        if tokens <= 0:
            raise ValueError("FA page geometry must be positive")
        page_tokens.add(tokens)
    if not page_tokens:
        raise ValueError("FA page geometry requires an admitted FA pool")
    if len(page_tokens) != 1:
        raise ValueError(
            f"FA page geometry differs across pools: {sorted(page_tokens)}")
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
    fa_pools = [pool for pool in manifest.pools if pool.tag == TAG_FA]
    if not fa_pools:
        raise ValueError("layout fingerprint requires an admitted FA pool")
    fa_pool = fa_pools[0]
    try:
        tensor = manifest.storages[fa_pool.storage_index]
    except IndexError as exc:
        raise ValueError(
            "FA pool storage index is outside the manifest") from exc

    getter = layout_getter or _default_layout_getter
    layout = getter(tensor)
    if layout is None:
        raise RuntimeError(
            "admitted FA storage has no materialized TPU layout")
    raw_minor_to_major, raw_tiles, raw_element_bits = layout
    minor_to_major = tuple(int(dim) for dim in raw_minor_to_major)
    tiles = tuple(tuple(int(dim) for dim in tile) for tile in raw_tiles)
    element_bits = int(raw_element_bits or 8)
    if minor_to_major != EXPECTED_FA_MINOR_TO_MAJOR:
        raise RuntimeError("FA layout failed E0' minor-to-major gate: "
                           f"measured={minor_to_major} "
                           f"expected={EXPECTED_FA_MINOR_TO_MAJOR}")
    if tiles != EXPECTED_FA_TILES:
        raise RuntimeError("FA layout failed E0' tile gate: "
                           f"measured={tiles} expected={EXPECTED_FA_TILES}")
    if element_bits != 8:
        raise RuntimeError("FA layout failed E0' element-size gate: "
                           f"measured={element_bits} expected=8")

    version = package_version or importlib.metadata.version
    payload = {
        "schema":
        FA_LAYOUT_FINGERPRINT_SCHEMA,
        "torch_tpu":
        version("torch_tpu"),
        "libtpu":
        version("libtpu"),
        "minor_to_major":
        list(minor_to_major),
        "tiles": [list(tile) for tile in tiles],
        "element_size_in_bits":
        element_bits,
        # GDN conv state layout version.  Both sides of a disagg pair must
        # agree: the byte-span convention (pair-blocked whole-token QK
        # spans vs the legacy split Q/K) is baked into the transfer plan,
        # and the geometries are size-identical, so a mixed pair would not
        # otherwise fail closed.  The controller compares fingerprints
        # before emitting a plan.
        "gdn_conv_layout":
        ("qk-pair-v1"
         if tpu_envs.TPU_GDN_CONV_QK_PAIR_LAYOUT else "legacy-split-qk"),
    }
    return canonical_layout_fingerprint(payload), payload


__all__ = [
    "EXPECTED_FA_MINOR_TO_MAJOR",
    "EXPECTED_FA_TILES",
    "FA_LAYOUT_FINGERPRINT_SCHEMA",
    "canonical_layout_fingerprint",
    "fa_page_tokens",
    "measured_fa_layout_fingerprint",
]
