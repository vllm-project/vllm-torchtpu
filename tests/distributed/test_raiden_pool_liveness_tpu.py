# SPDX-License-Identifier: Apache-2.0
"""Device byte-liveness test: bytes in the unified KV pool buffers must be
readable through raiden's pool surface, and bytes written through raiden must
land in those buffers.

The materialization is the unified (aliased-raw) binding the runner uses for
the pooled GDN architectures: one attention-shaped raw pool per layer, with
the GDN conv/ssm state regions carved out of each GDN layer's pool. The
private-typed hybrid materialization is no longer admissible: tpu-sync
0.0.1.dev20260914 and later require every registered storage to hold the
same number of blocks (the typed GDN state tensors and the paged FA cache
do not).

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
from vllm_torchtpu.distributed.kv_transfer.raiden.tags import class_tag

from .tpu_test_utils import run_in_isolated_process


def _can_allocate_tpu_tensor() -> tuple[bool, str]:
    probe = (
        "import torch, torch_tpu\n"
        "x = torch.arange(16, dtype=torch.uint8, device='tpu')\n"
        "print(str(x.device), x.cpu().tolist())\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
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
    return b"".join(block[off : off + size] for off, size in extents)


def _dead_extents(entry, extents) -> list[tuple[int, int]]:
    """The pitch intervals of one block that hold no declared live bytes."""
    dead = []
    cursor = 0
    for off, size in extents:
        if cursor < off:
            dead.append((cursor, off - cursor))
        cursor = off + size
    if cursor < entry.block_stride_bytes:
        dead.append((cursor, entry.block_stride_bytes - cursor))
    return dead


def _mirror_holds_full_pitch(entry, block_id: int) -> bool:
    """Whether the host mirror backs the whole pitch of ``block_id``.

    The mirror of a storage ends at the last block's live extent, so only
    blocks before the last one are guaranteed a full stride of host bytes;
    dead-interval probes stay within that."""
    return block_id < entry.num_blocks - 1


def _host_write(ref, block_bytes: bytes, intervals) -> None:
    for off, size in intervals:
        ctypes.memmove(ref["ptr"] + off, block_bytes[off : off + size], size)


def _host_read(ref, intervals) -> bytes:
    return b"".join(ctypes.string_at(ref["ptr"] + off, size) for off, size in intervals)


def _device_bytes(tensor) -> bytes:
    import torch

    return (
        tensor.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes()
    )


def _small_unified_materialization(device):
    """Unified-pool hybrid config: one attention-shaped raw pool per layer,
    allocated on device the way the runner does it (torch.empty for fp8).
    GDN layers expose their pool as ``[pool]`` and derive conv/ssm views
    from the pooled kernel geometry; the FA layer's cache is its pool."""
    import torch

    num_blocks, block_size, num_kv_heads, head_size = 16, 128, 2, 256
    manager_page_bytes = block_size * 2 * num_kv_heads * head_size  # 131072

    def _pool():
        return torch.empty(
            (num_blocks, block_size, 1, 4, head_size),
            dtype=torch.float8_e4m3fn,
            device=device,
        )

    gdn0, gdn1, fa = _pool(), _pool(), _pool()
    named = {
        "model.layers.0.linear_attn": [gdn0],
        "model.layers.1.linear_attn": [gdn1],
        "model.layers.3.self_attn.attn": fa,
    }
    geometry = rpm.GdnHeadGeometry(
        local_key_heads=4,
        local_value_heads=8,
        key_head_dim=64,
        value_head_dim=32,
        conv_kernel_size=3,
    )
    conv_shape = (geometry.conv_kernel_size - 1, geometry.conv_dim)
    ssm_shape = (
        geometry.local_value_heads,
        geometry.key_head_dim,
        geometry.value_head_dim,
    )
    groups = (
        type(
            "G",
            (),
            {
                "layer_names": ("model.layers.3.self_attn.attn",),
                "kv_cache_spec": type(
                    "S",
                    (),
                    {
                        "block_size": block_size,
                        "num_kv_heads": num_kv_heads,
                        "head_size": head_size,
                    },
                )(),
            },
        )(),
        type(
            "G",
            (),
            {
                "layer_names": (
                    "model.layers.0.linear_attn",
                    "model.layers.1.linear_attn",
                ),
                "kv_cache_spec": type(
                    "S",
                    (),
                    {
                        "shapes": (conv_shape, ssm_shape),
                        "dtypes": ("torch.bfloat16", "torch.float32"),
                        "page_size_bytes": manager_page_bytes,
                    },
                )(),
            },
        )(),
    )
    raw_tensors = (gdn0, gdn1, fa)
    return named, groups, geometry, raw_tensors, manager_page_bytes


def _run_pool_bytes_live_in_both_directions():
    _require_tpu_and_raiden()
    import torch
    from tpu_sync.api.torch.kv_cache_manager import KVCacheManager

    device = torch.device("tpu")
    (named, groups, geometry, raw_tensors, page_bytes) = _small_unified_materialization(
        device
    )

    manifest = rpm.build_qwen35_pool_manifest(
        named_kv_caches=named,
        kv_cache_groups=groups,
        raw_tensors=raw_tensors,
        gdn_geometry=geometry,
        mamba_group_ordinal_by_layer={
            "model.layers.0.linear_attn": 0,
            "model.layers.1.linear_attn": 0,
        },
    )
    assert manifest.binding == rpm.BINDING_ALIASED_RAW
    assert len(manifest.pools) == 5  # 2×(conv+ssm) + 1 fa
    # Every pool is addressed in manager pages of the same geometry.
    assert all(e.num_blocks == 16 for e in manifest.pools)
    assert all(e.block_stride_bytes == page_bytes for e in manifest.pools)
    fa_entry = next(e for e in manifest.pools if class_tag(e.tag) == rpm.TAG_FA)
    assert fa_entry.live_bytes_per_block == page_bytes
    rpm.verify_storage_binding(manifest, named, raw_tensors=raw_tensors)
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

    # In-process storage identity: every pool's storage is one of the raw
    # pools (python object identity through the manifest).
    assert {id(s) for s in manifest.storages} == {id(t) for t in raw_tensors}

    # --- Byte-liveness oracles, calibrated per physical layout --------------
    # Every storage is an fp8 attention-shaped pool whose device tiling stays
    # within each manager page, so per-page multisets are preserved for the
    # fa pool; the conv/ssm regions are byte spans inside the tiled page, so
    # only whole-storage multisets are asserted for them. The liveness
    # contract stage 2 proves:
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

        def _write_and_h2d(
            pattern,
            *,
            pool_idx=pool_idx,
            entry=entry,
            stride=stride,
            extents=extents,
            storage=storage,
        ):
            for block_id in range(entry.num_blocks):
                ref = manager.get_block_ref(pool_idx, block_id)
                assert ref["block_stride_bytes"] == stride
                assert ref["tag"] == entry.tag
                chunk = pattern[block_id * stride : (block_id + 1) * stride]
                # Region era: the pool surface moves the declared live
                # regions of a block, and the host mirror is only guaranteed
                # to back those bytes, so write exactly them.
                _host_write(ref, chunk, extents)
            manager.h2d_pool_blocks(pool_idx, list(range(entry.num_blocks))).wait()
            return _device_bytes(storage)

        pattern_first = _pattern_bytes(pool_idx + 1, storage.nbytes)
        pattern_second = _pattern_bytes(pool_idx + 101, storage.nbytes)
        patterns[pool_idx] = pattern_second
        after_first = _write_and_h2d(pattern_first)
        assert after_first != before, (
            f"pool {pool_idx} ({entry.tag}): raiden H2D did not change the "
            "typed KV cache buffer — dead storage"
        )
        after_second = _write_and_h2d(pattern_second)
        # Storage-granular multiset identity: gdn.conv bf16 tiling permutes
        # bytes ACROSS block boundaries, so per-block multisets are not
        # preserved — but the whole-storage multiset is, for every layout.
        live_first = b"".join(
            _extent_bytes(pattern_first[b * stride : (b + 1) * stride], extents)
            for b in range(entry.num_blocks)
        )
        live_second = b"".join(
            _extent_bytes(pattern_second[b * stride : (b + 1) * stride], extents)
            for b in range(entry.num_blocks)
        )
        assert Counter(after_first) + Counter(live_second) == Counter(
            after_second
        ) + Counter(live_first), (
            f"pool {pool_idx} ({entry.tag}): the bytes that changed between "
            "H2D passes are not exactly the live-region pattern bytes — "
            "raiden wrote foreign bytes"
        )

    # --- Per-block H2D granularity on the fa pool (block-contained tiling) --
    pool_idx = next(
        i for i, e in enumerate(manifest.pools) if class_tag(e.tag) == rpm.TAG_FA
    )
    entry = manifest.pools[pool_idx]
    storage = manifest.storages[entry.storage_index]
    stride = entry.block_stride_bytes
    fa_extents = _live_extents(entry)
    before = _device_bytes(storage)
    pattern_b = _pattern_bytes(97, storage.nbytes)
    _host_write(
        manager.get_block_ref(pool_idx, 2),
        pattern_b[2 * stride : 3 * stride],
        fa_extents,
    )
    manager.h2d_pool_blocks(pool_idx, [2]).wait()
    after = _device_bytes(storage)
    assert after[: 2 * stride] == before[: 2 * stride], (
        "single-block fa H2D touched logical blocks below the target"
    )
    assert after[3 * stride :] == before[3 * stride :], (
        "single-block fa H2D touched logical blocks above the target"
    )
    blk2 = slice(2 * stride, 3 * stride)
    assert Counter(before[blk2]) + Counter(
        _extent_bytes(pattern_b[blk2], fa_extents)
    ) == Counter(after[blk2]) + Counter(
        _extent_bytes(patterns[pool_idx][blk2], fa_extents)
    ), (
        "single-block fa H2D did not deliver exactly the target block's "
        "live-region bytes"
    )

    # --- Raiden D2H round trip: exactly the live physical bytes written -----
    # Region-era D2H fills only the live regions of the mirror; the clobbered
    # dead intervals must stay clobbered (a whole-pitch copy would revive
    # them with device padding bytes).
    for pool_idx, entry in enumerate(manifest.pools):
        stride = entry.block_stride_bytes
        extents = _live_extents(entry)
        dead = _dead_extents(entry, extents)
        expected = bytearray(patterns[pool_idx])
        if class_tag(entry.tag) == rpm.TAG_FA:
            expected[2 * stride : 3 * stride] = pattern_b[2 * stride : 3 * stride]
        # Clobber the host mirror so a stale read cannot pass: the live
        # regions of every block, plus the dead pitch of the blocks whose
        # full pitch the mirror backs.
        zeros = bytes(stride)
        for block_id in range(entry.num_blocks):
            ref = manager.get_block_ref(pool_idx, block_id)
            _host_write(ref, zeros, extents)
            if _mirror_holds_full_pitch(entry, block_id):
                _host_write(ref, zeros, dead)
        manager.d2h_pool_blocks(pool_idx, list(range(entry.num_blocks))).wait()
        for block_id in range(entry.num_blocks):
            ref = manager.get_block_ref(pool_idx, block_id)
            want = bytes(expected[block_id * stride : (block_id + 1) * stride])
            assert _host_read(ref, extents) == _extent_bytes(want, extents), (
                f"pool {pool_idx} ({entry.tag}) block {block_id}: raiden D2H "
                "bytes do not round-trip the live bytes written through the "
                "pool surface"
            )
            if _mirror_holds_full_pitch(entry, block_id):
                assert _host_read(ref, dead) == bytes(sum(size for _, size in dead)), (
                    f"pool {pool_idx} ({entry.tag}) block {block_id}: raiden "
                    "D2H wrote into dead pitch intervals — copies are not "
                    "region-granular"
                )


def test_pool_bytes_live_in_both_directions():
    run_in_isolated_process(_run_pool_bytes_live_in_both_directions)
