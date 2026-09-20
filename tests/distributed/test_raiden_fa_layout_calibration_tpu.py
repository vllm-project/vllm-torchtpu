# SPDX-License-Identifier: Apache-2.0
"""E0' raw-layout calibration for controller-driven PCP8 -> DP8 FA.

This standing device gate writes the same logical fp8 pattern into the live
prefill and decode shapes, observes both through raw D2H, and proves that every
decode page is a contiguous physical subrange of a prefill page.  It skips on
deviceless hosts; acceptance runs retain the printed ``RAIDEN_STAGE3_E0_PRIME``
record as the toolchain-specific fingerprint artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from vllm_torchtpu.distributed.kv_transfer.raiden import \
    layout_fingerprint as rlf
from vllm_torchtpu.distributed.kv_transfer.raiden import pool_manifest as rpm

from .tpu_test_utils import run_in_isolated_process

_TOKEN_BYTES = 1024
_NUM_BLOCKS = 16
_EXPECTED_TOKEN_PERMUTATION_SHA256 = (
    "5099414140678a1cdda205a7f97b6106d554de04b4fd60b238aa3d78e9e2a6e4")


@dataclass(frozen=True)
class _CalibrationCase:
    topology: str
    shape: tuple[int, ...]
    block_tokens: int
    block_stride_bytes: int


_PREFILL = _CalibrationCase(
    topology="pcp8_prefill",
    shape=(256, 256, 1, 4, 256),
    block_tokens=4096,
    block_stride_bytes=4_194_304,
)
_DECODE = _CalibrationCase(
    topology="dp8_decode",
    shape=(64, 256, 1, 4, 256),
    block_tokens=1024,
    block_stride_bytes=1_048_576,
)


def _require_e0_runtime() -> tuple[Any, Any, Any, Any]:
    probe = (
        "import torch, torch_tpu\n"
        "from torch_tpu._internal import sync\n"
        "x=torch.empty((1,1,1,4,256),dtype=torch.float8_e4m3fn,device='tpu')\n"
        "sync.synchronize([x],wait=True)\n")
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.skip(f"TPU tensor probe timed out: {exc}")
    if result.returncode:
        detail = result.stderr.strip().splitlines()
        pytest.skip("real TPU tensor allocation is unavailable: " + (
            detail[-1] if detail else f"probe exited {result.returncode}"))

    import torch
    import torch_tpu  # noqa: F401
    from torch_tpu._internal import sync
    from torch_tpu._internal.batch_transfer import batch_transfer_d2h_sync
    from torch_tpu._internal.compile import tpu_torch_compile

    return torch, sync, batch_transfer_d2h_sync, tpu_torch_compile


def _apply_xla_tiled_layout(
    logical: np.ndarray,
    minor_to_major: Sequence[int],
    tiles: Sequence[Sequence[int]],
) -> np.ndarray:
    """Return physical-order elements using XLA nested-tiling semantics."""
    order = tuple(int(dim) for dim in minor_to_major)
    if sorted(order) != list(range(logical.ndim)):
        raise ValueError("minor_to_major is not a rank permutation")
    physical = np.transpose(logical, tuple(reversed(order)))
    for raw_tile in tiles:
        tile = tuple(int(dim) for dim in raw_tile)
        if not tile or any(dim <= 0 for dim in tile):
            raise ValueError(f"invalid XLA tile: {tile}")
        if len(tile) > physical.ndim:
            physical = physical.reshape((1, ) * (len(tile) - physical.ndim) +
                                        physical.shape)
        suffix_start = physical.ndim - len(tile)
        pads = [(0, 0)] * physical.ndim
        for offset, tile_dim in enumerate(tile):
            axis = suffix_start + offset
            pads[axis] = (0, (-physical.shape[axis]) % tile_dim)
        if any(after for _, after in pads):
            physical = np.pad(physical, pads)
        prefix = physical.shape[:suffix_start]
        interleaved: list[int] = []
        for dim, tile_dim in zip(physical.shape[suffix_start:], tile):
            interleaved.extend((dim // tile_dim, tile_dim))
        physical = physical.reshape(prefix + tuple(interleaved))
        prefix_rank = len(prefix)
        tile_rank = len(tile)
        axes = (tuple(range(prefix_rank)) + tuple(prefix_rank + 2 * i
                                                  for i in range(tile_rank)) +
                tuple(prefix_rank + 2 * i + 1 for i in range(tile_rank)))
        physical = np.transpose(physical, axes)
    return np.ascontiguousarray(physical).reshape(-1)


def _logical_probe(shape: Sequence[int]) -> np.ndarray:
    numel = math.prod(shape)
    assert numel % _TOKEN_BYTES == 0
    tokens = np.arange(numel // _TOKEN_BYTES, dtype=np.uint32)
    token_codes = ((tokens * 73) ^ (tokens >> 3) ^ (tokens >> 11)).astype(
        np.uint8)
    columns = np.arange(_TOKEN_BYTES, dtype=np.uint16)
    column_codes = ((columns * 29) ^ (columns >> 2)).astype(np.uint8)
    logical = np.add.outer(token_codes, column_codes, dtype=np.uint8)
    # Avoid fp8 NaN encodings, which logical torch writes may canonicalize.
    np.remainder(logical, 126, out=logical)
    logical += 1
    return logical.reshape(tuple(shape))


def _manifest_for_tensor(tensor: Any,
                         case: _CalibrationCase) -> rpm.PoolManifest:
    layer_name = "model.layers.3.self_attn.attn"
    group = type(
        "Group", (), {
            "layer_names": (layer_name, ),
            "kv_cache_spec":
            type(
                "Spec", (), {
                    "block_size": case.block_tokens,
                    "num_kv_heads": 2,
                    "head_size": 256,
                })(),
        })()
    return rpm.build_qwen35_pool_manifest(
        named_kv_caches={layer_name: tensor},
        kv_cache_groups=(group, ),
        raw_tensors=(),
        gdn_geometry=rpm.GdnHeadGeometry(
            local_key_heads=1,
            local_value_heads=1,
            key_head_dim=1,
            value_head_dim=1,
        ),
    )


def _one_token_permutation() -> np.ndarray:
    logical = np.arange(_TOKEN_BYTES, dtype="<i4").reshape(1, 1, 1, 4, 256)
    physical_to_logical = _apply_xla_tiled_layout(
        logical, rlf.EXPECTED_FA_MINOR_TO_MAJOR, rlf.EXPECTED_FA_TILES)
    logical_to_physical = np.empty_like(physical_to_logical)
    logical_to_physical[physical_to_logical] = np.arange(_TOKEN_BYTES,
                                                         dtype="<i4")
    return logical_to_physical


def test_xla_nested_tiling_reference_examples():
    logical_2d = np.arange(32, dtype=np.int64).reshape(8, 4)
    assert _apply_xla_tiled_layout(logical_2d, (1, 0), ((2, 2), ))[11] == 13
    logical_1d = np.arange(2048, dtype=np.int64)
    physical_1d = _apply_xla_tiled_layout(logical_1d, (0, ),
                                          ((1024, ), (128, ), (2, 1)))
    assert physical_1d[1] == 128
    assert physical_1d[2] == 1


def _run_qwen35_fa_raw_token_range_identity_tpu():
    torch, sync, batch_transfer_d2h_sync, layout_api = _require_e0_runtime()
    raw_by_topology: dict[str, np.ndarray] = {}
    fingerprints: set[str] = set()
    fingerprint_payload: dict[str, Any] | None = None
    measurements = []

    for case in (_PREFILL, _DECODE):
        logical = _logical_probe(case.shape)
        tensor = (torch.from_numpy(logical.reshape(-1)).view(
            torch.float8_e4m3fn).reshape(case.shape).to("tpu"))
        sync.synchronize([tensor], wait=True)
        manifest = _manifest_for_tensor(tensor, case)
        assert rlf.fa_page_tokens(manifest) == case.block_tokens
        pool = next(pool for pool in manifest.pools if pool.tag == rpm.TAG_FA)
        assert pool.num_blocks == _NUM_BLOCKS
        assert pool.block_stride_bytes == case.block_stride_bytes
        assert pool.live_bytes_per_block == case.block_stride_bytes

        fingerprint, payload = rlf.measured_fa_layout_fingerprint(manifest)
        fingerprints.add(fingerprint)
        fingerprint_payload = payload
        minor_to_major, tiles, raw_bits = (
            layout_api.get_device_layout_if_materialized(tensor))
        element_bits = int(raw_bits or 8)
        host = torch.empty(tensor.nbytes, dtype=torch.uint8)
        batch_transfer_d2h_sync([tensor], [host])
        raw = host.numpy().copy()
        expected = _apply_xla_tiled_layout(logical, minor_to_major, tiles)
        assert np.array_equal(raw, expected)

        labels = np.repeat(np.arange(_NUM_BLOCKS, dtype=np.uint8),
                           case.block_stride_bytes).reshape(case.shape)
        physical_labels = _apply_xla_tiled_layout(labels, minor_to_major,
                                                  tiles)
        assert np.array_equal(
            physical_labels,
            np.repeat(np.arange(_NUM_BLOCKS, dtype=np.uint8),
                      case.block_stride_bytes),
        )
        raw_by_topology[case.topology] = raw
        measurements.append({
            "topology":
            case.topology,
            "shape":
            list(case.shape),
            "block_tokens":
            case.block_tokens,
            "block_stride_bytes":
            case.block_stride_bytes,
            "minor_to_major": [int(dim) for dim in minor_to_major],
            "tiles": [[int(dim) for dim in tile] for tile in tiles],
            "element_size_in_bits":
            element_bits,
            "block_contained":
            True,
        })

    assert len(fingerprints) == 1
    page_ratio = _PREFILL.block_tokens // _DECODE.block_tokens
    prefill = raw_by_topology[_PREFILL.topology]
    decode = raw_by_topology[_DECODE.topology]
    ranges = []
    for decode_page in range(_NUM_BLOCKS):
        prefill_page, subpage = divmod(decode_page, page_ratio)
        src = (prefill_page * _PREFILL.block_stride_bytes +
               subpage * _DECODE.block_stride_bytes)
        dst = decode_page * _DECODE.block_stride_bytes
        assert np.array_equal(
            prefill[src:src + _DECODE.block_stride_bytes],
            decode[dst:dst + _DECODE.block_stride_bytes],
        )
        ranges.append({
            "prefill_page": prefill_page,
            "decode_page": decode_page,
            "prefill_offset_bytes": src,
            "size_bytes": _DECODE.block_stride_bytes,
        })

    permutation = _one_token_permutation()
    permutation_sha = hashlib.sha256(
        permutation.astype("<i4", copy=False).tobytes()).hexdigest()
    assert permutation_sha == _EXPECTED_TOKEN_PERMUTATION_SHA256
    fingerprint = next(iter(fingerprints))
    record = {
        "schema": "qwen35-fa-raw-token-range-identity-v1",
        "torch": torch.__version__,
        "layout_fingerprint": fingerprint,
        "layout_fingerprint_payload": fingerprint_payload,
        "token_bytes": _TOKEN_BYTES,
        "token_permutation_sha256": permutation_sha,
        "max_tokens_per_physical_1024b_chunk": 1,
        "raw_subrange_identity": True,
        "page_ratio": page_ratio,
        "decode_pages_compared": len(ranges),
        "cases": measurements,
        "ranges": ranges,
    }
    print("RAIDEN_STAGE3_E0_PRIME=" +
          json.dumps(record, sort_keys=True, separators=(",", ":")))


def test_qwen35_fa_raw_token_range_identity_tpu():
    run_in_isolated_process(_run_qwen35_fa_raw_token_range_identity_tpu)
