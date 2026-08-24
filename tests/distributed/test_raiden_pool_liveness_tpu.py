# SPDX-License-Identifier: Apache-2.0
"""Device byte-liveness test: bytes in the typed KV cache buffers must be
readable through raiden's pool surface, and bytes written through raiden must
land in those buffers.

The oracle is the bytes themselves, because metadata oracles cannot
distinguish live from dead storage:

  pattern in the typed buffers -> raiden partial D2H -> read via get_block_ref
  write via get_block_ref      -> raiden partial H2D -> read via torch

Semantics note: the raiden manager pins the storages' *current* device
buffers at construction. Torch-level eager ``copy_`` swaps a tensor's buffer,
so the pattern is baked into the tensors before admission (mirroring the real
flow, where the model's cache-update kernels write in place into the buffers
that were admitted at register_runner time). Raiden H2D is the in-place
writer exercised here.

Runs on a single local TPU chip; skips when no TPU or raiden extension is
available.
"""

import ctypes
import subprocess
import sys
from collections import Counter

import pytest

from vllm_torchtpu.distributed.kv_transfer.raiden import pool_manifest as rpm

from .tpu_test_utils import run_in_isolated_process


def _can_allocate_tpu_tensor() -> tuple[bool, str]:
    probe = ("import torch, torch_tpu\n"
             "x = torch.arange(16, dtype=torch.uint8, device='tpu')\n"
             "print(str(x.device), x.cpu().tolist())\n")
    try:
        result = subprocess.run([sys.executable, "-c", probe],
                                check=False,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                timeout=60)
    except subprocess.TimeoutExpired as exc:
        return False, f"TPU tensor probe timed out: {exc}"
    if result.returncode != 0:
        return False, f"TPU tensor probe exited with code {result.returncode}"
    return True, result.stdout.strip()


def _require_tpu_and_raiden():
    ok, detail = _can_allocate_tpu_tensor()
    if not ok:
        pytest.skip(f"real TPU tensor allocation is unavailable: {detail}")
    pytest.importorskip("tpu_sync.api.torch.kv_cache_manager")


def _pattern_bytes(seed: int, nbytes: int) -> bytes:
    # Deterministic non-trivial pattern; avoids 0x00/0xff runs.
    return bytes((seed * 131 + i * 7 + (i >> 8)) % 251 for i in range(nbytes))


def _live_extents(entry) -> list[tuple[int, int]]:
    """Per-block live byte intervals from the declared regions, coalesced.

    Mirrors raiden's region expansion (pool_layout.cc): each RegionSpec is
    num_units strided runs of units_per_stride*unit_bytes contiguous bytes.
    Region-era D2H/H2D copies exactly these intervals and nothing else.
    """
    runs = []
    for region in entry.regions:
        run_bytes = region.units_per_stride * region.unit_bytes
        for stride_idx in range(region.num_units):
            start = region.offset_bytes + stride_idx * region.stride_bytes
            runs.append((start, run_bytes))
    runs.sort()
    coalesced: list[tuple[int, int]] = []
    for start, size in runs:
        if coalesced and coalesced[-1][0] + coalesced[-1][1] == start:
            coalesced[-1] = (coalesced[-1][0], coalesced[-1][1] + size)
        else:
            coalesced.append((start, size))
    return coalesced


def _extent_bytes(block: bytes, extents) -> bytes:
    return b"".join(block[off:off + size] for off, size in extents)


def _device_bytes(tensor) -> bytes:
    import torch

    return tensor.detach().cpu().contiguous().view(-1).view(
        torch.uint8).numpy().tobytes()


def _small_hybrid_materialization(device):
    """Hybrid config materialized exactly the way the runner's alias-fallback
    path does it: torch.zeros on device (torch.empty for fp8). The shapes are
    the live TP2DP4 decode shapes with NUM_GPU_BLOCKS_OVERRIDE=16; the device
    layout of these allocations is row-major (proven by stage-1 content
    correctness on live traffic), which is what pool admission relies on.
    Buffers created via CPU->TPU transfer can get transposed layouts and are
    NOT admissible."""
    import torch

    def _gdn_layer():
        return (
            torch.zeros((16, 3, 6144), dtype=torch.bfloat16, device=device),
            torch.zeros((16, 32, 128, 128), dtype=torch.float32,
                        device=device),
        )

    named = {
        "model.layers.0.linear_attn":
        _gdn_layer(),
        "model.layers.1.linear_attn":
        _gdn_layer(),
        "model.layers.3.self_attn.attn":
        torch.empty((64, 256, 1, 4, 256),
                    dtype=torch.float8_e4m3fn,
                    device=device),
    }
    groups = (
        type(
            "G", (), {
                "layer_names": ("model.layers.3.self_attn.attn", ),
                "kv_cache_spec":
                type("S", (), {
                    "block_size": 1024,
                    "num_kv_heads": 1,
                    "head_size": 256,
                })(),
            })(),
        type(
            "G", (), {
                "layer_names":
                ("model.layers.0.linear_attn", "model.layers.1.linear_attn"),
                "kv_cache_spec":
                type("S", (), {})(),
            })(),
    )
    geometry = rpm.GdnHeadGeometry(local_key_heads=8,
                                   local_value_heads=32,
                                   key_head_dim=128,
                                   value_head_dim=128)
    return named, groups, geometry


def _run_pool_bytes_live_in_both_directions():
    _require_tpu_and_raiden()
    import torch
    from tpu_sync.api.torch.kv_cache_manager import KVCacheManager

    device = torch.device("tpu")
    named, groups, geometry = _small_hybrid_materialization(device)

    manifest = rpm.build_qwen35_pool_manifest(named_kv_caches=named,
                                              kv_cache_groups=groups,
                                              raw_tensors=(),
                                              gdn_geometry=geometry)
    assert manifest.binding == rpm.BINDING_PRIVATE_TYPED
    assert len(manifest.pools) == 5  # 2×(conv+ssm) + 1 fa
    # Decode FA geometry: 16 blocks of 1024 tokens.
    fa_entry = next(e for e in manifest.pools if e.tag == rpm.TAG_FA)
    assert fa_entry.num_blocks == 16
    assert fa_entry.block_stride_bytes == 1_048_576
    assert fa_entry.live_bytes_per_block == 524_288
    rpm.verify_storage_binding(manifest, named, raw_tensors=())
    # fp8 caches are torch.empty-backed: force device-buffer materialization
    # (wrapping an unmaterialized tensor hangs the manager constructor).
    rpm.materialize_storages(manifest)

    manager = KVCacheManager(
        kv_caches=list(manifest.storages),
        node_id=0,
        local_control_port=-1,
        max_blocks=1,
        num_slots=1,
        unsafe_skip_buffer_lock=True,
    )
    summary = manager.register_pools(manifest.pool_dicts())
    assert summary["admitted"] is True
    assert summary["pools"] == 5

    # In-process storage identity: every pool's storage is one of the typed
    # caches (python object identity through the manifest).
    typed_ids = set()
    for cache in named.values():
        tensors = cache if isinstance(cache, tuple) else (cache, )
        typed_ids.update(id(t) for t in tensors)
    assert {id(s) for s in manifest.storages} == typed_ids

    # --- Byte-liveness oracles, calibrated per physical layout --------------
    # Empirical device layouts for these live shapes: gdn.ssm fp32 buffers are
    # physically linear (byte-identical to the logical view); fa fp8 buffers
    # are tiled but the permutation stays within each vLLM block; gdn.conv
    # bf16 buffers are tiled ACROSS block boundaries. The liveness contract
    # stage 2 proves:
    #   1. raiden H2D lands in the buffers torch reads (same storage), and
    #   2. raiden D2H returns exactly the physical bytes written (round trip),
    # with strict logical-byte equality asserted where the layout permits.
    #
    # Region era (raiden 97f755a): D2H/H2D moves exactly the declared live
    # regions of each block, never the whole pitch — dead pitch bytes stay
    # untouched on the device. The pool with live < stride here (fa: 512 KiB
    # live in a 1 MiB pitch) therefore keeps its initial device bytes in the
    # dead half. Two full-pitch H2D passes with different patterns make the
    # assertion initial-state-independent: between passes, exactly the live
    # bytes change, so per block
    #   multiset(pass1 bytes) + multiset(pattern2 live bytes)
    #     == multiset(pass2 bytes) + multiset(pattern1 live bytes).
    patterns = {}
    for pool_idx, entry in enumerate(manifest.pools):
        storage = manifest.storages[entry.storage_index]
        stride = entry.block_stride_bytes
        extents = _live_extents(entry)
        assert sum(size for _, size in extents) == entry.live_bytes_per_block
        before = _device_bytes(storage)

        def _write_and_h2d(pattern,
                           *,
                           pool_idx=pool_idx,
                           entry=entry,
                           stride=stride):
            for block_id in range(entry.num_blocks):
                ref = manager.get_block_ref(pool_idx, block_id)
                assert ref["block_stride_bytes"] == stride
                assert ref["tag"] == entry.tag
                chunk = pattern[block_id * stride:(block_id + 1) * stride]
                ctypes.memmove(ref["ptr"], chunk, stride)
            manager.h2d_pool_blocks(pool_idx,
                                    list(range(entry.num_blocks))).wait()
            return _device_bytes(storage)

        pattern_first = _pattern_bytes(pool_idx + 1, storage.nbytes)
        pattern_second = _pattern_bytes(pool_idx + 101, storage.nbytes)
        patterns[pool_idx] = pattern_second
        after_first = _write_and_h2d(pattern_first)
        assert after_first != before, (
            f"pool {pool_idx} ({entry.tag}): raiden H2D did not change the "
            "typed KV cache buffer — dead storage")
        after_second = _write_and_h2d(pattern_second)
        # Storage-granular multiset identity: gdn.conv bf16 tiling permutes
        # bytes ACROSS block boundaries, so per-block multisets are not
        # preserved — but the whole-storage multiset is, for every layout.
        live_first = b"".join(
            _extent_bytes(pattern_first[b * stride:(b + 1) * stride], extents)
            for b in range(entry.num_blocks))
        live_second = b"".join(
            _extent_bytes(pattern_second[b * stride:(b + 1) * stride], extents)
            for b in range(entry.num_blocks))
        assert (
            Counter(after_first) +
            Counter(live_second) == Counter(after_second) + Counter(live_first)
        ), (f"pool {pool_idx} ({entry.tag}): the bytes that changed between "
            "H2D passes are not exactly the live-region pattern bytes — "
            "raiden wrote foreign bytes")
        if entry.tag == rpm.TAG_GDN_SSM:
            for block_id in range(entry.num_blocks):
                blk = slice(block_id * stride, (block_id + 1) * stride)
                assert (
                    _extent_bytes(after_second[blk], extents) == _extent_bytes(
                        pattern_second[blk], extents)
                ), (f"pool {pool_idx} ({entry.tag}): fp32 state buffers are "
                    "physically linear, live bytes must match exactly")

    # --- Per-block H2D granularity on the fa pool (block-contained tiling) --
    pool_idx = next(i for i, e in enumerate(manifest.pools)
                    if e.tag == rpm.TAG_FA)
    entry = manifest.pools[pool_idx]
    storage = manifest.storages[entry.storage_index]
    stride = entry.block_stride_bytes
    fa_extents = _live_extents(entry)
    before = _device_bytes(storage)
    pattern_b = _pattern_bytes(97, storage.nbytes)
    ctypes.memmove(
        manager.get_block_ref(pool_idx, 2)["ptr"],
        pattern_b[2 * stride:3 * stride], stride)
    manager.h2d_pool_blocks(pool_idx, [2]).wait()
    after = _device_bytes(storage)
    assert after[:2 * stride] == before[:2 * stride], (
        "single-block fa H2D touched logical blocks below the target")
    assert after[3 * stride:] == before[3 * stride:], (
        "single-block fa H2D touched logical blocks above the target")
    blk2 = slice(2 * stride, 3 * stride)
    assert (
        Counter(before[blk2]) +
        Counter(_extent_bytes(pattern_b[blk2], fa_extents)) == Counter(
            after[blk2]) +
        Counter(_extent_bytes(patterns[pool_idx][blk2], fa_extents))), (
            "single-block fa H2D did not deliver exactly the target block's "
            "live-region bytes")

    # --- Raiden D2H round trip: exactly the live physical bytes written -----
    # Region-era D2H fills only the live regions of the mirror; the clobbered
    # dead intervals must stay clobbered (a whole-pitch copy would revive
    # them with device padding bytes).
    for pool_idx, entry in enumerate(manifest.pools):
        stride = entry.block_stride_bytes
        extents = _live_extents(entry)
        dead = []
        cursor = 0
        for off, size in extents:
            if cursor < off:
                dead.append((cursor, off - cursor))
            cursor = off + size
        if cursor < stride:
            dead.append((cursor, stride - cursor))
        expected = bytearray(patterns[pool_idx])
        if entry.tag == rpm.TAG_FA:
            expected[2 * stride:3 * stride] = pattern_b[2 * stride:3 * stride]
        # Clobber the host mirror so a stale read cannot pass.
        for block_id in range(entry.num_blocks):
            ref = manager.get_block_ref(pool_idx, block_id)
            ctypes.memmove(ref["ptr"], bytes(stride), stride)
        manager.d2h_pool_blocks(pool_idx, list(range(entry.num_blocks))).wait()
        for block_id in range(entry.num_blocks):
            ref = manager.get_block_ref(pool_idx, block_id)
            host = ctypes.string_at(ref["ptr"], stride)
            want = bytes(expected[block_id * stride:(block_id + 1) * stride])
            assert (
                _extent_bytes(host, extents) == _extent_bytes(want, extents)
            ), (f"pool {pool_idx} ({entry.tag}) block {block_id}: raiden D2H "
                "bytes do not round-trip the live bytes written through the "
                "pool surface")
            assert _extent_bytes(host, dead) == bytes(
                sum(size for _, size in dead)
            ), (f"pool {pool_idx} ({entry.tag}) block {block_id}: raiden D2H "
                "wrote into dead pitch intervals — copies are not "
                "region-granular")


def test_pool_bytes_live_in_both_directions():
    run_in_isolated_process(_run_pool_bytes_live_in_both_directions)
