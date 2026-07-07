# SPDX-License-Identifier: Apache-2.0
import multiprocessing
import os
import queue
import subprocess
import sys
import time
import traceback
import types
from functools import lru_cache
from typing import Any

import portpicker
import pytest

from .tpu_connector_v2_test_utils import _load_v2_module


class _SysModulesPatch:

    @staticmethod
    def setitem(mapping, key, value):
        mapping[key] = value

    @staticmethod
    def delitem(mapping, key, *, raising=True):
        try:
            del mapping[key]
        except KeyError:
            if raising:
                raise


def _load_v2_module_in_process():
    return _load_v2_module(_SysModulesPatch(), stub_zmq=False)


def _disable_v2_pull_start(worker: Any) -> None:
    worker._request_v2_pull_start = lambda req_meta: None


def _cdiv(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _align_to(value: int, alignment: int) -> int:
    return _cdiv(value, alignment) * alignment


def _integration_model_constants(*, decode_tp_size: int = 2) -> dict[str, Any]:
    prefill_workers = 4
    if decode_tp_size <= 0:
        raise ValueError("decode_tp_size must be positive")
    fp8_bytes = 1
    bf16_bytes = 2
    fp32_bytes = 4
    kv_packing = 4
    total_kv_heads = 2
    head_dim = 256
    linear_key_head_dim = 128
    linear_value_head_dim = 128
    linear_num_key_heads = 16
    linear_num_value_heads = 64
    linear_conv_kernel_dim = 4
    conv_segments = linear_conv_kernel_dim - 1
    source_local_kv_heads = total_kv_heads
    dest_local_kv_heads = total_kv_heads // decode_tp_size
    source_local_key_heads = linear_num_key_heads // prefill_workers
    source_local_value_heads = linear_num_value_heads // prefill_workers
    dest_local_key_heads = linear_num_key_heads // decode_tp_size
    dest_local_value_heads = linear_num_value_heads // decode_tp_size
    source_conv_dim = (source_local_key_heads * 2 +
                       source_local_value_heads) * linear_key_head_dim
    dest_conv_dim = (dest_local_key_heads * 2 +
                     dest_local_value_heads) * linear_key_head_dim
    source_conv_stride_bytes = source_conv_dim * bf16_bytes
    dest_conv_stride_bytes = dest_conv_dim * bf16_bytes
    source_conv_state_bytes = conv_segments * source_conv_stride_bytes
    dest_conv_state_bytes = conv_segments * dest_conv_stride_bytes
    source_recurrent_head_bytes = (linear_key_head_dim *
                                   linear_value_head_dim * fp32_bytes)
    dest_recurrent_head_bytes = source_recurrent_head_bytes
    source_recurrent_state_bytes = (source_local_value_heads *
                                    source_recurrent_head_bytes)
    dest_recurrent_state_bytes = (dest_local_value_heads *
                                  dest_recurrent_head_bytes)
    source_gdn_used_bytes = (source_conv_state_bytes +
                             source_recurrent_state_bytes)
    source_fa_token_bytes = (_cdiv(source_local_kv_heads * 2, kv_packing) *
                             kv_packing * head_dim * fp8_bytes)
    source_fa_head_slots = (_cdiv(source_local_kv_heads * 2, kv_packing) *
                            kv_packing // 2)
    source_block_size = _align_to(
        _cdiv(source_gdn_used_bytes, source_fa_token_bytes), 16)
    source_page_bytes = source_block_size * source_fa_token_bytes
    dest_block_size = source_block_size * prefill_workers
    dest_fa_token_bytes = (_cdiv(dest_local_kv_heads * 2, kv_packing) *
                           kv_packing * head_dim * fp8_bytes)
    dest_fa_head_slots = (_cdiv(dest_local_kv_heads * 2, kv_packing) *
                          kv_packing // 2)
    dest_page_bytes = dest_block_size * dest_fa_token_bytes
    fa_head_bytes_per_token = 2 * head_dim * fp8_bytes
    source_fa_padding_bytes_per_token = (
        source_fa_token_bytes -
        source_local_kv_heads * fa_head_bytes_per_token)
    dest_fa_padding_bytes_per_token = (
        dest_fa_token_bytes - dest_local_kv_heads * fa_head_bytes_per_token)
    fa_source_block_ids = tuple(range(1, 9))
    fa_dest_block_ids = (0, 1)
    mamba_source_block_ids = (3, 4)
    mamba_dest_block_ids = (2, 3)
    fa_num_tokens = source_block_size * len(fa_source_block_ids)
    mamba_num_tokens = source_block_size * len(mamba_source_block_ids)
    max_source_block = max(fa_source_block_ids + mamba_source_block_ids)
    max_dest_block = max(fa_dest_block_ids + mamba_dest_block_ids)
    source_page_padding_bytes = source_page_bytes - source_gdn_used_bytes
    dest_page_padding_bytes = dest_page_bytes - (dest_conv_state_bytes +
                                                 dest_recurrent_state_bytes)
    source_fa_page_shape = (
        source_block_size,
        _cdiv(source_local_kv_heads * 2, kv_packing),
        kv_packing,
        head_dim,
    )
    dest_fa_page_shape = (
        dest_block_size,
        _cdiv(dest_local_kv_heads * 2, kv_packing),
        kv_packing,
        head_dim,
    )
    source_conv_shape = (conv_segments, source_conv_dim)
    dest_conv_shape = (conv_segments, dest_conv_dim)
    source_recurrent_shape = (
        source_local_value_heads,
        linear_key_head_dim,
        linear_value_head_dim,
    )
    dest_recurrent_shape = (
        dest_local_value_heads,
        linear_key_head_dim,
        linear_value_head_dim,
    )

    return {
        "prefill_workers": prefill_workers,
        "decode_tp_size": decode_tp_size,
        "kv_packing": kv_packing,
        "total_kv_heads": total_kv_heads,
        "head_dim": head_dim,
        "linear_key_head_dim": linear_key_head_dim,
        "linear_value_head_dim": linear_value_head_dim,
        "linear_num_key_heads": linear_num_key_heads,
        "linear_num_value_heads": linear_num_value_heads,
        "conv_segments": conv_segments,
        "source_local_kv_heads": source_local_kv_heads,
        "dest_local_kv_heads": dest_local_kv_heads,
        "source_local_key_heads": source_local_key_heads,
        "source_local_value_heads": source_local_value_heads,
        "dest_local_key_heads": dest_local_key_heads,
        "dest_local_value_heads": dest_local_value_heads,
        "source_conv_stride_bytes": source_conv_stride_bytes,
        "dest_conv_stride_bytes": dest_conv_stride_bytes,
        "source_conv_state_bytes": source_conv_state_bytes,
        "dest_conv_state_bytes": dest_conv_state_bytes,
        "source_recurrent_head_bytes": source_recurrent_head_bytes,
        "dest_recurrent_head_bytes": dest_recurrent_head_bytes,
        "source_recurrent_state_bytes": source_recurrent_state_bytes,
        "dest_recurrent_state_bytes": dest_recurrent_state_bytes,
        "source_gdn_used_bytes": source_gdn_used_bytes,
        "dest_gdn_used_bytes":
        (dest_conv_state_bytes + dest_recurrent_state_bytes),
        "source_page_padding_bytes": source_page_padding_bytes,
        "dest_page_padding_bytes": dest_page_padding_bytes,
        "source_block_size": source_block_size,
        "dest_block_size": dest_block_size,
        "source_fa_token_bytes": source_fa_token_bytes,
        "dest_fa_token_bytes": dest_fa_token_bytes,
        "source_fa_head_slots": source_fa_head_slots,
        "dest_fa_head_slots": dest_fa_head_slots,
        "fa_head_bytes_per_token": fa_head_bytes_per_token,
        "source_fa_padding_bytes_per_token": source_fa_padding_bytes_per_token,
        "dest_fa_padding_bytes_per_token": dest_fa_padding_bytes_per_token,
        "source_fa_page_shape": source_fa_page_shape,
        "dest_fa_page_shape": dest_fa_page_shape,
        "source_conv_shape": source_conv_shape,
        "dest_conv_shape": dest_conv_shape,
        "source_recurrent_shape": source_recurrent_shape,
        "dest_recurrent_shape": dest_recurrent_shape,
        "source_page_bytes": source_page_bytes,
        "dest_page_bytes": dest_page_bytes,
        "fa_source_block_ids": fa_source_block_ids,
        "fa_dest_block_ids": fa_dest_block_ids,
        "mamba_source_block_ids": mamba_source_block_ids,
        "mamba_dest_block_ids": mamba_dest_block_ids,
        "fa_num_tokens": fa_num_tokens,
        "mamba_num_tokens": mamba_num_tokens,
        "source_size": (max_source_block + 1) * source_page_bytes,
        "dst_size": (max_dest_block + 1) * dest_page_bytes,
    }


def _integration_source_buffer(rank: int,
                               size: int | None = None) -> bytearray:
    if size is None:
        size = _integration_model_constants()["source_size"]
    return bytearray(_integration_source_bytes(rank, size))


@lru_cache(maxsize=None)
def _integration_source_bytes(rank: int, size: int) -> bytes:
    constants = _integration_model_constants()
    if size != constants["source_size"]:
        raise ValueError(f"unexpected source buffer size: {size}")
    buffer = bytearray(size)
    _fill_fa_pages(buffer, rank, constants)
    _fill_gdn_pages(buffer, rank, constants)
    return bytes(buffer)


def _pattern_bytes(seed: int, size: int) -> bytes:
    return bytes(((seed + offset) & 0xFF) for offset in range(size))


def _fa_seed(rank: int, block_id: int, token: int, local_head: int,
             kv_part: int) -> int:
    return (rank * 17 + block_id * 31 + token * 7 + local_head * 53 +
            kv_part * 101) & 0xFF


def _gdn_conv_seed(rank: int, block_id: int, slot: int, segment_id: int,
                   local_head: int) -> int:
    return (rank * 19 + block_id * 23 + slot * 11 + segment_id * 47 +
            local_head * 5) & 0xFF


def _gdn_recurrent_seed(rank: int, block_id: int, local_value: int) -> int:
    return (rank * 13 + block_id * 29 + local_value * 7) & 0xFF


def _fill_fa_pages(buffer: bytearray, rank: int, constants: dict[str,
                                                                 Any]) -> None:
    # FA logical view per page:
    # [block_size, cdiv(num_kv_heads * 2, packing), packing, head_dim].
    # Lanes are encoded as [K0, V0, K1, V1] so one local KV head is a
    # contiguous 2 * head_dim byte segment.
    for block_id in constants["fa_source_block_ids"]:
        page_base = block_id * constants["source_page_bytes"]
        for token in range(constants["source_block_size"]):
            token_base = (page_base +
                          token * constants["source_fa_token_bytes"])
            for local_head in range(constants["source_local_kv_heads"]):
                head_base = (token_base +
                             local_head * constants["fa_head_bytes_per_token"])
                for kv_part in range(2):
                    offset = head_base + kv_part * constants["head_dim"]
                    seed = _fa_seed(rank, block_id, token, local_head, kv_part)
                    buffer[offset:offset +
                           constants["head_dim"]] = (_pattern_bytes(
                               seed, constants["head_dim"]))


def _fill_gdn_pages(buffer: bytearray, rank: int,
                    constants: dict[str, Any]) -> None:
    # state0 logical view per page: [kernel_size - 1, dim].
    # dim is Q heads, K heads, then V heads, each bf16 head occupying 256 B.
    # state1 logical view per page: [n_v, d_k, d_v] in fp32 bytes.
    conv_head_bytes = constants["linear_key_head_dim"] * 2
    for block_id in constants["mamba_source_block_ids"]:
        page_base = block_id * constants["source_page_bytes"]
        for slot in range(constants["conv_segments"]):
            slot_base = page_base + slot * constants["source_conv_stride_bytes"]
            for local_key in range(constants["source_local_key_heads"]):
                q_offset = slot_base + local_key * conv_head_bytes
                q_seed = _gdn_conv_seed(rank, block_id, slot, 0, local_key)
                buffer[q_offset:q_offset + conv_head_bytes] = (_pattern_bytes(
                    q_seed, conv_head_bytes))

                k_offset = (
                    slot_base +
                    constants["source_local_key_heads"] * conv_head_bytes +
                    local_key * conv_head_bytes)
                k_seed = _gdn_conv_seed(rank, block_id, slot, 1, local_key)
                buffer[k_offset:k_offset + conv_head_bytes] = (_pattern_bytes(
                    k_seed, conv_head_bytes))

            value_base = (
                slot_base +
                constants["source_local_key_heads"] * 2 * conv_head_bytes)
            for local_value in range(constants["source_local_value_heads"]):
                v_offset = value_base + local_value * conv_head_bytes
                v_seed = _gdn_conv_seed(rank, block_id, slot, 2, local_value)
                buffer[v_offset:v_offset + conv_head_bytes] = (_pattern_bytes(
                    v_seed, conv_head_bytes))

        recurrent_base = page_base + constants["source_conv_state_bytes"]
        for local_value in range(constants["source_local_value_heads"]):
            offset = (recurrent_base +
                      local_value * constants["source_recurrent_head_bytes"])
            seed = _gdn_recurrent_seed(rank, block_id, local_value)
            buffer[offset:offset +
                   constants["source_recurrent_head_bytes"]] = (_pattern_bytes(
                       seed, constants["source_recurrent_head_bytes"]))


def _make_tpu_group_env(*, world_size: int, local_rank_base: int,
                        topology: str) -> dict[str, str]:
    slicebuilder_addresses = ",".join(
        f"localhost:{portpicker.pick_unused_port()}"
        for _ in range(world_size))
    return {
        "WORLD_SIZE": str(world_size),
        "LOCAL_WORLD_SIZE": str(world_size),
        "LOCAL_RANK_BASE": str(local_rank_base),
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(portpicker.pick_unused_port()),
        "TORCH_TPU_SLICEBUILDER_ADDRESSES": slicebuilder_addresses,
        "TORCH_TPU_TOPOLOGY": topology,
        "TORCH_TPU_XPROF_SESSION_ID": str(time.time_ns()),
    }


def _init_worker_tpu_group(group_env: dict[str, str], *, rank: int) -> Any:
    os.environ.update(group_env)
    os.environ["RANK"] = str(rank)
    local_rank = int(group_env["LOCAL_RANK_BASE"]) + rank
    os.environ["LOCAL_RANK"] = str(local_rank)

    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    return dist


def _destroy_worker_tpu_group(dist: Any | None) -> None:
    if dist is not None and dist.is_initialized():
        dist.destroy_process_group()


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
                                timeout=30)
    except subprocess.TimeoutExpired as exc:
        return False, f"TPU tensor probe timed out: {exc}"
    if result.returncode != 0:
        if "Check failed: context.ok()" in result.stderr:
            return False, ("torch_tpu aborted during tensor allocation "
                           "(Check failed: context.ok())")
        return False, f"TPU tensor probe exited with code {result.returncode}"
    return True, result.stdout.strip()


def _require_real_tpu_tensor_support() -> None:
    ok, detail = _can_allocate_tpu_tensor()
    if not ok:
        pytest.skip(f"real TPU tensor allocation is unavailable: {detail}")


def _integration_tpu_tensor_from_bytes(data: bytes | bytearray):
    import torch
    import torch_tpu  # noqa: F401

    cpu_tensor = torch.frombuffer(bytearray(data), dtype=torch.uint8).clone()
    return cpu_tensor.to(device=torch.device("tpu"))


def _integration_zero_tpu_tensor(size: int):
    import torch
    import torch_tpu  # noqa: F401

    return torch.zeros(size, dtype=torch.uint8, device=torch.device("tpu"))


def _integration_filled_tpu_tensor(size: int, value: int):
    import torch
    import torch_tpu  # noqa: F401

    return torch.full((size, ),
                      value,
                      dtype=torch.uint8,
                      device=torch.device("tpu"))


def _integration_tensor_bytes(tensor: Any) -> bytes:
    import torch

    return tensor.detach().cpu().contiguous().view(
        torch.uint8).numpy().tobytes()


class _TPUBackingByteStore:

    def __init__(self, backing: Any) -> None:
        self.backing = backing
        self._host_bytes: bytearray | None = None
        self._dirty = False

    @property
    def host_bytes(self) -> bytearray:
        if self._host_bytes is None:
            self._host_bytes = bytearray(
                _integration_tensor_bytes(self.backing))
        return self._host_bytes

    def read_bytes(self, offset: int, length: int) -> bytes:
        self._check_range(offset, length)
        return bytes(self.host_bytes[offset:offset + length])

    def write_bytes(self, offset: int,
                    data: bytes | bytearray | memoryview) -> None:
        payload = memoryview(data).cast("B")
        self._check_range(offset, len(payload))
        self.host_bytes[offset:offset + len(payload)] = payload
        self._dirty = True

    def flush(self) -> None:
        if not self._dirty:
            return
        import torch

        cpu_tensor = torch.frombuffer(self.host_bytes,
                                      dtype=torch.uint8).clone()
        self.backing.copy_(cpu_tensor.to(device=self.backing.device))
        self._dirty = False

    def _check_range(self, offset: int, length: int) -> None:
        if offset < 0 or length < 0:
            raise ValueError("negative byte range")
        if offset > len(
                self.host_bytes) or length > len(self.host_bytes) - offset:
            raise ValueError("byte range out of bounds")


class _TPUBackingRegionBuffer:

    def __init__(self, store: _TPUBackingByteStore, offset: int,
                 nbytes: int) -> None:
        self._store = store
        self._offset = int(offset)
        self.nbytes = int(nbytes)

    @property
    def device(self) -> Any:
        return self._store.backing.device

    def numel(self) -> int:
        return self.nbytes

    def element_size(self) -> int:
        return 1

    def is_contiguous(self) -> bool:
        return True

    def read_bytes(self, offset: int, length: int) -> bytes:
        self._check_region_range(offset, length)
        return self._store.read_bytes(self._offset + offset, length)

    def write_bytes(self, offset: int,
                    data: bytes | bytearray | memoryview) -> None:
        payload = memoryview(data).cast("B")
        self._check_region_range(offset, len(payload))
        self._store.write_bytes(self._offset + offset, payload)

    def flush(self) -> None:
        self._store.flush()

    def _check_region_range(self, offset: int, length: int) -> None:
        if offset < 0 or length < 0:
            raise ValueError("negative region byte range")
        if offset > self.nbytes or length > self.nbytes - offset:
            raise ValueError("region byte range out of bounds")


def _is_tpu_tensor(tensor: Any) -> bool:
    return getattr(getattr(tensor, "device", None), "type", None) == "tpu"


def _integration_region_source_buffers(
    metadata: Any,
    p_rank: int,
    constants: dict[str, Any],
) -> dict[str, bytearray]:
    source = _integration_source_buffer(p_rank, constants["source_size"])
    return {
        layer_name: bytearray(source)
        for layer_name in metadata.kv_caches[p_rank]
    }


def _integration_source_tensors_by_region(
    metadata: Any,
    p_rank: int,
    constants: dict[str, Any],
) -> dict[str, Any]:
    return {
        region_id: _integration_tpu_tensor_from_bytes(buffer)
        for region_id, buffer in _integration_region_source_buffers(
            metadata, p_rank, constants).items()
    }


def _integration_destination_tensors_by_region(
    destination: Any,
    constants: dict[str, Any],
) -> dict[str, Any]:
    return {
        region_id: _integration_filled_tpu_tensor(constants["dst_size"], 0xA5)
        for region_id in destination.kv_caches
    }


def _contiguous_region_placements(
    region_buffers: dict[str, bytes | bytearray],
    page_bytes_by_layer: dict[str, int],
) -> dict[str, int]:
    offsets: dict[str, int] = {}
    cursor = 0
    for region_id, buffer in region_buffers.items():
        page_bytes = page_bytes_by_layer[region_id]
        if cursor % page_bytes != 0:
            raise ValueError("region start is not page aligned: "
                             f"region_id={region_id!r} cursor={cursor} "
                             f"page_bytes={page_bytes}")
        if len(buffer) % page_bytes != 0:
            raise ValueError("region size is not page aligned: "
                             f"region_id={region_id!r} size={len(buffer)} "
                             f"page_bytes={page_bytes}")
        offsets[region_id] = cursor
        cursor += len(buffer)
    return offsets


def _integration_tpu_region_views_from_buffers(
    region_buffers: dict[str, bytes | bytearray],
    page_bytes_by_layer: dict[str, int],
) -> tuple[Any, dict[str, Any], dict[str, int]]:
    offsets = _contiguous_region_placements(region_buffers,
                                            page_bytes_by_layer)
    backing_bytes = bytearray(
        sum(len(buffer) for buffer in region_buffers.values()))
    for region_id, buffer in region_buffers.items():
        offset = offsets[region_id]
        backing_bytes[offset:offset + len(buffer)] = buffer

    backing = _integration_tpu_tensor_from_bytes(backing_bytes)
    store = _TPUBackingByteStore(backing)
    region_tensors = {
        region_id: _TPUBackingRegionBuffer(store, offsets[region_id],
                                           len(buffer))
        for region_id, buffer in region_buffers.items()
    }
    return backing, region_tensors, offsets


def _integration_source_page_views(
    metadata: Any,
    p_rank: int,
    constants: dict[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, int]]:
    return _integration_tpu_region_views_from_buffers(
        _integration_region_source_buffers(metadata, p_rank, constants),
        _page_bytes_by_layer(metadata.kv_caches[p_rank]),
    )


def _integration_destination_page_views(
    destination: Any,
    constants: dict[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, int]]:
    return _integration_tpu_region_views_from_buffers(
        {
            region_id: bytearray([0xA5]) * constants["dst_size"]
            for region_id in destination.kv_caches
        },
        _page_bytes_by_layer(destination.kv_caches),
    )


def _register_engine_regions(engine: Any, tensors: dict[str, Any],
                             page_bytes_by_layer: dict[str, int]) -> None:
    region_cls = getattr(sys.modules[engine.__class__.__module__],
                         "RegisteredMemoryRegion")
    for region_id, tensor in tensors.items():
        nbytes = getattr(tensor, "nbytes", None)
        if nbytes is None:
            nbytes = tensor.numel() * tensor.element_size()
        engine.register_local_region(
            buffer=tensor,
            region=region_cls(
                region_id=region_id,
                nbytes=nbytes,
                page_bytes=page_bytes_by_layer[region_id],
            ),
        )


def _page_bytes_by_layer(regions: dict[str, Any]) -> dict[str, int]:
    return {
        layer_name: region.physical_block_stride_bytes
        for layer_name, region in regions.items()
    }


def _integration_expected_region_bytes(
    plans: dict[int, Any],
    metadata: Any,
    destination: Any,
    constants: dict[str, Any],
) -> dict[str, bytearray]:
    expected = {
        layer_name: bytearray([0xA5]) * constants["dst_size"]
        for layer_name in destination.kv_caches
    }
    source_buffers = {
        p_rank: _integration_region_source_buffers(metadata, p_rank, constants)
        for p_rank in metadata.kv_caches
    }
    source_sizes = {
        p_rank: {
            layer_name: len(buffer)
            for layer_name, buffer in buffers.items()
        }
        for p_rank, buffers in source_buffers.items()
    }
    destination_sizes = {
        layer_name: len(buffer)
        for layer_name, buffer in expected.items()
    }
    for p_rank, plan in plans.items():
        for op in plan.ops:
            for segment_idx in range(op.num_segments):
                src_layer = op.source_region_id
                dst_layer = op.destination_region_id
                src_offset = (op.src_offset_bytes +
                              segment_idx * op.src_stride_bytes)
                dst_offset = (op.dst_offset_bytes +
                              segment_idx * op.dst_stride_bytes)
                assert src_layer in metadata.kv_caches[p_rank]
                assert dst_layer in destination.kv_caches
                assert src_offset + op.segment_bytes <= source_sizes[p_rank][
                    src_layer]
                assert dst_offset + op.segment_bytes <= destination_sizes[
                    dst_layer]
                expected[dst_layer][dst_offset:dst_offset +
                                    op.segment_bytes] = (
                                        source_buffers[p_rank][src_layer]
                                        [src_offset:src_offset +
                                         op.segment_bytes])
    return expected


def _conv_head_segment(mod: Any, *, name: str, global_heads: range,
                       local_head_start: int, base_offset_bytes: int,
                       head_dim: int, stride_bytes: int, num_segments: int):
    return mod.HeadSegment(
        name=name,
        global_heads=tuple(global_heads),
        local_head_start=local_head_start,
        local_head_count=len(global_heads),
        head_bytes=head_dim * 2,
        base_offset_bytes=base_offset_bytes,
        stride_bytes=stride_bytes,
        num_segments=num_segments,
    )


def _integration_layout(mod: Any, decode_tp_rank: int, constants: dict[str,
                                                                       Any]):
    fa_layer = "model.layers.0.self_attn"
    gdn_layers = tuple(f"model.layers.{idx}.linear_attn" for idx in (1, 2))
    c = constants
    source_regions_by_rank = {}
    for p_rank in range(c["prefill_workers"]):
        key_start = p_rank * c["source_local_key_heads"]
        value_start = p_rank * c["source_local_value_heads"]
        source_regions = {
            fa_layer:
            mod.KVCacheRegion(
                layer_name=fa_layer,
                layer_type=mod.LayerType.FULL_ATTN,
                block_size=c["source_block_size"],
                block_bytes=c["source_page_bytes"],
                layout=mod.TensorLayout.TOKEN_FIRST,
                num_heads=c["source_fa_head_slots"],
                head_bytes=None,
                token_first_layout=None,
                token_stride_bytes=None,
                head_stride_bytes=None,
                live_head_bytes=None,
                block_stride_bytes=c["source_page_bytes"],
                block_id_group_index=2,
                head_segments=(),
                physical_region_id=fa_layer,
                region_base_offset_bytes=0,
            )
        }
        for group_index, layer_name in enumerate(gdn_layers):
            source_regions[f"{layer_name}.state0"] = mod.KVCacheRegion(
                layer_name=f"{layer_name}.state0",
                layer_type=mod.LayerType.MAMBA_STATE,
                block_size=None,
                block_bytes=c["source_conv_state_bytes"],
                block_stride_bytes=c["source_page_bytes"],
                layout=mod.TensorLayout.BLOCKS_FIRST,
                num_heads=(c["source_local_key_heads"] * 2 +
                           c["source_local_value_heads"]),
                head_bytes=None,
                token_first_layout=None,
                token_stride_bytes=None,
                head_stride_bytes=None,
                live_head_bytes=None,
                head_segments=(
                    _conv_head_segment(
                        mod,
                        name="q",
                        global_heads=range(
                            key_start,
                            key_start + c["source_local_key_heads"]),
                        local_head_start=0,
                        base_offset_bytes=0,
                        head_dim=c["linear_key_head_dim"],
                        stride_bytes=c["source_conv_stride_bytes"],
                        num_segments=c["conv_segments"],
                    ),
                    _conv_head_segment(
                        mod,
                        name="k",
                        global_heads=range(
                            key_start,
                            key_start + c["source_local_key_heads"]),
                        local_head_start=c["source_local_key_heads"],
                        base_offset_bytes=(c["source_local_key_heads"] *
                                           c["linear_key_head_dim"] * 2),
                        head_dim=c["linear_key_head_dim"],
                        stride_bytes=c["source_conv_stride_bytes"],
                        num_segments=c["conv_segments"],
                    ),
                    _conv_head_segment(
                        mod,
                        name="v",
                        global_heads=range(
                            value_start,
                            value_start + c["source_local_value_heads"]),
                        local_head_start=c["source_local_key_heads"] * 2,
                        base_offset_bytes=(c["source_local_key_heads"] * 2 *
                                           c["linear_key_head_dim"] * 2),
                        head_dim=c["linear_value_head_dim"],
                        stride_bytes=c["source_conv_stride_bytes"],
                        num_segments=c["conv_segments"],
                    ),
                ),
                block_id_group_index=group_index,
                physical_region_id=f"{layer_name}.state0",
                region_base_offset_bytes=0,
            )
            source_regions[f"{layer_name}.state1"] = mod.KVCacheRegion(
                layer_name=f"{layer_name}.state1",
                layer_type=mod.LayerType.MAMBA_STATE,
                block_size=None,
                block_bytes=c["source_recurrent_state_bytes"],
                block_stride_bytes=c["source_page_bytes"],
                layout=mod.TensorLayout.BLOCKS_FIRST,
                num_heads=c["source_local_value_heads"],
                head_bytes=None,
                token_first_layout=None,
                token_stride_bytes=None,
                head_stride_bytes=None,
                live_head_bytes=None,
                head_segments=(mod.HeadSegment(
                    name="ssm",
                    global_heads=tuple(
                        range(value_start,
                              value_start + c["source_local_value_heads"])),
                    local_head_start=0,
                    local_head_count=c["source_local_value_heads"],
                    head_bytes=c["source_recurrent_head_bytes"],
                    base_offset_bytes=c["source_conv_state_bytes"],
                    stride_bytes=None,
                    num_segments=1,
                ), ),
                block_id_group_index=group_index,
                physical_region_id=f"{layer_name}.state1",
                region_base_offset_bytes=0,
            )
        source_regions_by_rank[p_rank] = source_regions

    dest_key_start = decode_tp_rank * c["dest_local_key_heads"]
    dest_value_start = decode_tp_rank * c["dest_local_value_heads"]
    dest_regions = {
        fa_layer:
        mod.KVCacheRegion(
            layer_name=fa_layer,
            layer_type=mod.LayerType.FULL_ATTN,
            block_size=c["dest_block_size"],
            block_bytes=c["dest_page_bytes"],
            layout=mod.TensorLayout.TOKEN_FIRST,
            num_heads=c["dest_fa_head_slots"],
            head_bytes=None,
            token_first_layout=None,
            token_stride_bytes=None,
            head_stride_bytes=None,
            live_head_bytes=None,
            block_stride_bytes=c["dest_page_bytes"],
            block_id_group_index=2,
            head_segments=(),
            physical_region_id=fa_layer,
            region_base_offset_bytes=0,
        )
    }
    for group_index, layer_name in enumerate(gdn_layers):
        dest_regions[f"{layer_name}.state0"] = mod.KVCacheRegion(
            layer_name=f"{layer_name}.state0",
            layer_type=mod.LayerType.MAMBA_STATE,
            block_size=None,
            block_bytes=c["dest_conv_state_bytes"],
            block_stride_bytes=c["dest_page_bytes"],
            layout=mod.TensorLayout.BLOCKS_FIRST,
            num_heads=(c["dest_local_key_heads"] * 2 +
                       c["dest_local_value_heads"]),
            head_bytes=None,
            token_first_layout=None,
            token_stride_bytes=None,
            head_stride_bytes=None,
            live_head_bytes=None,
            head_segments=(
                _conv_head_segment(
                    mod,
                    name="q",
                    global_heads=range(
                        dest_key_start,
                        dest_key_start + c["dest_local_key_heads"]),
                    local_head_start=0,
                    base_offset_bytes=0,
                    head_dim=c["linear_key_head_dim"],
                    stride_bytes=c["dest_conv_stride_bytes"],
                    num_segments=c["conv_segments"],
                ),
                _conv_head_segment(
                    mod,
                    name="k",
                    global_heads=range(
                        dest_key_start,
                        dest_key_start + c["dest_local_key_heads"]),
                    local_head_start=c["dest_local_key_heads"],
                    base_offset_bytes=(c["dest_local_key_heads"] *
                                       c["linear_key_head_dim"] * 2),
                    head_dim=c["linear_key_head_dim"],
                    stride_bytes=c["dest_conv_stride_bytes"],
                    num_segments=c["conv_segments"],
                ),
                _conv_head_segment(
                    mod,
                    name="v",
                    global_heads=range(
                        dest_value_start,
                        dest_value_start + c["dest_local_value_heads"]),
                    local_head_start=c["dest_local_key_heads"] * 2,
                    base_offset_bytes=(c["dest_local_key_heads"] * 2 *
                                       c["linear_key_head_dim"] * 2),
                    head_dim=c["linear_value_head_dim"],
                    stride_bytes=c["dest_conv_stride_bytes"],
                    num_segments=c["conv_segments"],
                ),
            ),
            block_id_group_index=group_index,
            physical_region_id=f"{layer_name}.state0",
            region_base_offset_bytes=0,
        )
        dest_regions[f"{layer_name}.state1"] = mod.KVCacheRegion(
            layer_name=f"{layer_name}.state1",
            layer_type=mod.LayerType.MAMBA_STATE,
            block_size=None,
            block_bytes=c["dest_recurrent_state_bytes"],
            block_stride_bytes=c["dest_page_bytes"],
            layout=mod.TensorLayout.BLOCKS_FIRST,
            num_heads=c["dest_local_value_heads"],
            head_bytes=None,
            token_first_layout=None,
            token_stride_bytes=None,
            head_stride_bytes=None,
            live_head_bytes=None,
            head_segments=(mod.HeadSegment(
                name="ssm",
                global_heads=tuple(
                    range(dest_value_start,
                          dest_value_start + c["dest_local_value_heads"])),
                local_head_start=0,
                local_head_count=c["dest_local_value_heads"],
                head_bytes=c["dest_recurrent_head_bytes"],
                base_offset_bytes=c["dest_conv_state_bytes"],
                stride_bytes=None,
                num_segments=1,
            ), ),
            block_id_group_index=group_index,
            physical_region_id=f"{layer_name}.state1",
            region_base_offset_bytes=0,
        )

    metadata = mod.ConnectorMetadataV2(
        req_id=123,
        block_size=c["source_block_size"],
        kv_source_layout=mod.KVParallelLayout(
            full_attn_pcp_size=c["prefill_workers"],
            full_attn_tp_size=c["prefill_workers"],
            linear_attn_pcp_size=1,
            linear_attn_tp_size=c["prefill_workers"],
        ),
        kv_caches=source_regions_by_rank,
        fa_block_ids=c["fa_source_block_ids"],
        mamba_block_ids=c["mamba_source_block_ids"],
        block_ids_by_group=(
            (c["mamba_source_block_ids"][0], ),
            (c["mamba_source_block_ids"][1], ),
            c["fa_source_block_ids"],
        ),
        fa_num_tokens=c["fa_num_tokens"],
        mamba_num_tokens=c["mamba_num_tokens"],
    )
    topology = mod.TpKVTopology(
        local_layout=mod.KVParallelLayout(
            full_attn_pcp_size=1,
            full_attn_tp_size=c["decode_tp_size"],
            linear_attn_pcp_size=1,
            linear_attn_tp_size=c["decode_tp_size"],
        ),
        block_size=c["dest_block_size"],
        tp_rank=decode_tp_rank,
        total_num_kv_heads=c["total_kv_heads"],
        total_num_mamba_key_heads=c["linear_num_key_heads"],
        total_num_mamba_heads=c["linear_num_value_heads"],
    )
    destination = mod.LocalDecodeAllocation(
        rank=decode_tp_rank,
        block_size=c["dest_block_size"],
        kv_caches=dest_regions,
        fa_block_ids=c["fa_dest_block_ids"],
        mamba_block_ids=c["mamba_dest_block_ids"],
        block_ids_by_group=(
            (c["mamba_dest_block_ids"][0], ),
            (c["mamba_dest_block_ids"][1], ),
            c["fa_dest_block_ids"],
        ),
        fa_num_tokens=c["fa_num_tokens"],
        mamba_num_tokens=c["mamba_num_tokens"],
    )
    return metadata, topology, destination, c


def test_integration_layout_uses_fp8_packed_fa_pages_and_gdn_page_stride(
        monkeypatch):
    mod = _load_v2_module(monkeypatch, stub_zmq=False)
    constants = _integration_model_constants()
    metadata, _, destination, constants = _integration_layout(
        mod, decode_tp_rank=0, constants=constants)

    assert constants["source_fa_token_bytes"] == (_cdiv(
        constants["source_local_kv_heads"] * 2, constants["kv_packing"]) *
                                                  constants["kv_packing"] *
                                                  constants["head_dim"])
    assert constants["dest_fa_token_bytes"] == (
        _cdiv(constants["dest_local_kv_heads"] * 2, constants["kv_packing"]) *
        constants["kv_packing"] * constants["head_dim"])
    assert constants["source_fa_padding_bytes_per_token"] == 0
    assert constants["dest_fa_padding_bytes_per_token"] == (
        constants["dest_fa_token_bytes"] // 2)
    assert constants["source_fa_page_shape"] == (1056, 1, 4, 256)
    assert constants["dest_fa_page_shape"] == (4224, 1, 4, 256)
    assert constants["source_conv_shape"] == (3, 3072)
    assert constants["dest_conv_shape"] == (3, 6144)
    assert constants["source_recurrent_shape"] == (16, 128, 128)
    assert constants["dest_recurrent_shape"] == (32, 128, 128)
    assert constants["source_block_size"] % 16 == 0
    assert constants["dest_block_size"] % 16 == 0
    assert constants["source_page_bytes"] >= constants["source_gdn_used_bytes"]
    assert constants["dest_page_bytes"] == (constants["source_page_bytes"] *
                                            constants["prefill_workers"])
    assert constants["dest_page_bytes"] >= constants["dest_gdn_used_bytes"]
    assert constants["source_page_padding_bytes"] == (
        constants["source_page_bytes"] - constants["source_gdn_used_bytes"])
    assert constants["dest_page_padding_bytes"] == (
        constants["dest_page_bytes"] - constants["dest_gdn_used_bytes"])
    assert len(constants["fa_source_block_ids"]) == 8
    assert len(constants["mamba_source_block_ids"]) == 2

    for regions in metadata.kv_caches.values():
        for region in regions.values():
            if region.layer_type == mod.LayerType.FULL_ATTN:
                assert region.block_bytes == constants["source_page_bytes"]
            else:
                assert (region.physical_block_stride_bytes ==
                        constants["source_page_bytes"])
    for region in destination.kv_caches.values():
        if region.layer_type == mod.LayerType.FULL_ATTN:
            assert region.block_bytes == constants["dest_page_bytes"]
        else:
            assert (region.physical_block_stride_bytes ==
                    constants["dest_page_bytes"])


def test_integration_layout_supports_fa_4pcp_gdn_4tp_to_decode_1tp(
        monkeypatch):
    mod = _load_v2_module(monkeypatch, stub_zmq=False)
    constants = _integration_model_constants(decode_tp_size=1)
    metadata, topology, destination, constants = _integration_layout(
        mod, decode_tp_rank=0, constants=constants)

    assert metadata.kv_source_layout.full_attn_pcp_size == 4
    assert metadata.kv_source_layout.full_attn_tp_size == 4
    assert metadata.kv_source_layout.linear_attn_tp_size == 4
    assert topology.local_layout.full_attn_pcp_size == 1
    assert topology.local_layout.full_attn_tp_size == 1
    assert topology.local_layout.linear_attn_tp_size == 1
    assert constants["dest_local_kv_heads"] == 2
    assert constants["dest_local_key_heads"] == 16
    assert constants["dest_local_value_heads"] == 64
    assert constants["dest_fa_padding_bytes_per_token"] == 0
    assert constants["dest_fa_page_shape"] == (4224, 1, 4, 256)
    assert constants["dest_conv_shape"] == (3, 12288)
    assert constants["dest_recurrent_shape"] == (64, 128, 128)
    assert constants["dest_page_bytes"] >= constants["dest_gdn_used_bytes"]
    assert destination.block_size == constants["dest_block_size"]


def test_integration_region_page_placement_is_contiguous(monkeypatch):
    mod = _load_v2_module(monkeypatch, stub_zmq=False)
    constants = _integration_model_constants()
    metadata, _, destination, constants = _integration_layout(
        mod, decode_tp_rank=0, constants=constants)

    source_offsets = _contiguous_region_placements(
        _integration_region_source_buffers(metadata, 0, constants),
        _page_bytes_by_layer(metadata.kv_caches[0]),
    )
    dest_offsets = _contiguous_region_placements(
        {
            region_id: bytearray([0xA5]) * constants["dst_size"]
            for region_id in destination.kv_caches
        },
        _page_bytes_by_layer(destination.kv_caches),
    )

    assert sorted(source_offsets.values()) == [
        index * constants["source_size"]
        for index in range(len(source_offsets))
    ]
    assert sorted(dest_offsets.values()) == [
        index * constants["dst_size"] for index in range(len(dest_offsets))
    ]
    assert all(
        offset %
        metadata.kv_caches[0][region_id].physical_block_stride_bytes == 0
        for region_id, offset in source_offsets.items())
    assert all(
        offset %
        destination.kv_caches[region_id].physical_block_stride_bytes == 0
        for region_id, offset in dest_offsets.items())


def _prefill_worker_process(p_rank: int, group_env: dict[str, str],
                            ready_q: Any, stop_event: Any,
                            constants: dict[str, Any]) -> None:
    dist = None
    engine = None
    try:
        dist = _init_worker_tpu_group(group_env, rank=p_rank)
        mod = _load_v2_module_in_process()
        strided = sys.modules[
            "vllm_torchtpu.distributed.kv_transfer.v2.strided_transfer"]
        metadata, _, _, constants = _integration_layout(mod,
                                                        decode_tp_rank=0,
                                                        constants=constants)
        source_backing, source_tensors, source_offsets = (
            _integration_source_page_views(metadata, p_rank, constants))
        assert _is_tpu_tensor(source_backing)
        assert all(
            _is_tpu_tensor(tensor) for tensor in source_tensors.values())
        assert all(tensor.is_contiguous()
                   for tensor in source_tensors.values())
        engine = strided.StridedKVTransferEngine(
            local_dp_rank=0,
            local_tp_rank=p_rank,
            local_worker_id=f"prefill-rank{p_rank}",
            listen_host="127.0.0.1",
            listen_port=0,
            transport="zmq",
        )
        _register_engine_regions(
            engine,
            source_tensors,
            _page_bytes_by_layer(metadata.kv_caches[p_rank]),
        )
        engine.start()
        ready_q.put(
            ("prefill_ready", p_rank, engine.local_metadata().to_dict(),
             str(source_backing.device), int(source_backing.numel()),
             source_offsets))
        stop_event.wait(30.0)
        assert _is_tpu_tensor(source_backing)
        assert all(
            _is_tpu_tensor(tensor) for tensor in source_tensors.values())
        engine.stop()
        ready_q.put(("prefill_stopped", p_rank, str(source_backing.device)))
    except BaseException:
        ready_q.put(("prefill_error", p_rank, traceback.format_exc()))
    finally:
        if engine is not None:
            engine.stop()
        _destroy_worker_tpu_group(dist)


def _decode_worker_process(decode_tp_rank: int, kv_transfer_params: dict,
                           group_env: dict[str, str], result_q: Any,
                           constants: dict[str, Any]) -> None:
    dist = None
    engine = None
    try:
        dist = _init_worker_tpu_group(group_env, rank=decode_tp_rank)
        mod = _load_v2_module_in_process()
        strided = sys.modules[
            "vllm_torchtpu.distributed.kv_transfer.v2.strided_transfer"]
        metadata, topology, destination, constants = _integration_layout(
            mod, decode_tp_rank=decode_tp_rank, constants=constants)
        dst_backing, dst_tensors, dst_offsets = (
            _integration_destination_page_views(destination, constants))
        assert _is_tpu_tensor(dst_backing)
        assert all(_is_tpu_tensor(tensor) for tensor in dst_tensors.values())
        assert all(tensor.is_contiguous() for tensor in dst_tensors.values())
        engine = strided.StridedKVTransferEngine(
            local_dp_rank=0,
            local_tp_rank=decode_tp_rank,
            local_worker_id=f"decode-rank{decode_tp_rank}",
            listen_host="127.0.0.1",
            listen_port=0,
            transport="zmq",
        )
        _register_engine_regions(
            engine,
            dst_tensors,
            _page_bytes_by_layer(destination.kv_caches),
        )
        destination = mod.TPUConnectorV2Worker.apply_local_destination_region_metadata(
            destination, engine.local_regions_metadata())
        scheduler = mod.TPUConnectorV2Scheduler(
            types.SimpleNamespace(
                kv_transfer_config=types.SimpleNamespace(is_kv_producer=False),
                parallel_config=types.SimpleNamespace(data_parallel_rank=0)))
        scheduler.set_strided_decode_metadata(topology=topology,
                                              destination=destination)
        request = types.SimpleNamespace(
            request_id="req-1",
            kv_transfer_params=kv_transfer_params,
            prompt_token_ids=list(range(constants["fa_num_tokens"] + 1)),
        )
        scheduler.update_state_after_alloc(request, object(),
                                           constants["fa_num_tokens"])
        req_meta = scheduler.reqs_to_load["req-1"]
        metadata = req_meta.strided_source_metadata
        rank_ops_by_decode_rank = (
            mod.TPUConnectorV2Worker.
            _remote_rank_ops_by_decode_rank_from_req_meta(req_meta))
        local_rank_ops = rank_ops_by_decode_rank[decode_tp_rank]
        plans = {
            p_rank: mod.RankTransferPlan(p_rank=p_rank, ops=tuple(ops))
            for p_rank, ops in local_rank_ops.items()
        }
        engine.start()
        worker = mod.TPUConnectorV2Worker(object())
        worker.tp_rank = decode_tp_rank
        worker.set_strided_transfer_bridge(
            mod.TPUConnectorV2StridedBridge(engine))
        _disable_v2_pull_start(worker)
        initial_dst_devices = {
            region_id: str(tensor.device)
            for region_id, tensor in dst_tensors.items()
        }
        connector_metadata = types.SimpleNamespace(
            reqs_to_send={},
            reqs_to_load={
                "req-1":
                types.SimpleNamespace(
                    uuid=kv_transfer_params["uuid"],
                    remote_block_ids=kv_transfer_params["remote_block_ids"],
                    remote_metadata=req_meta.remote_metadata,
                )
            })
        connector_metadata.reqs_to_load[
            "req-1"].remote_rank_ops_by_decode_rank = rank_ops_by_decode_rank

        copied = worker.process_send_load(connector_metadata)
        for tensor in dst_tensors.values():
            tensor.flush()

        expected = _integration_expected_region_bytes(plans, metadata,
                                                      destination, constants)
        actual_backing = _integration_tensor_bytes(dst_backing)
        expected_backing = bytearray([0xA5]) * int(dst_backing.numel())
        for region_id, expected_region in expected.items():
            offset = dst_offsets[region_id]
            expected_backing[offset:offset +
                             len(expected_region)] = (expected_region)
        actual = {
            region_id:
            actual_backing[dst_offsets[region_id]:dst_offsets[region_id] +
                           len(expected_region)]
            for region_id, expected_region in expected.items()
        }
        expected_copied = sum(plan.total_bytes for plan in plans.values())
        result_q.put({
            "status":
            "ok",
            "rank":
            decode_tp_rank,
            "copied":
            copied,
            "expected_copied":
            expected_copied,
            "plan_ranks":
            sorted(plans),
            "matches":
            all(actual[key] == bytes(value)
                for key, value in expected.items()),
            "destination_backing_matches":
            (actual_backing == bytes(expected_backing)),
            "destination_backing_device":
            str(dst_backing.device),
            "destination_backing_nbytes":
            int(dst_backing.numel()),
            "destination_region_offsets":
            dst_offsets,
            "initial_dst_devices":
            initial_dst_devices,
            "final_dst_devices": {
                region_id: str(tensor.device)
                for region_id, tensor in dst_tensors.items()
            },
            "decode_metadata":
            engine.local_metadata().to_dict(),
            "num_destination_regions":
            len(dst_tensors),
            "destination_metadata_has_no_obsolete_region_fields":
            all(not {"rank", "base_addr", "block_id_index"} & region.keys()
                for region in engine.local_metadata().to_dict()["regions"]),
            "num_plans":
            len(plans),
        })
    except BaseException:
        result_q.put({
            "status": "error",
            "rank": decode_tp_rank,
            "traceback": traceback.format_exc(),
        })
    finally:
        if engine is not None:
            engine.stop()
        _destroy_worker_tpu_group(dist)


def _queue_get(q: Any, timeout: float = 15.0):
    try:
        return q.get(timeout=timeout)
    except queue.Empty as exc:
        raise AssertionError("timed out waiting for worker process") from exc


def _run_multiprocess_prefill_decode_integration(
    monkeypatch: Any,
    *,
    constants: dict[str, Any],
    prefill_topology: str,
    decode_topology: str,
    expected_plan_ranks: dict[int, list[int]],
) -> None:
    _require_real_tpu_tensor_support()
    assert constants["prefill_workers"] == 4

    ctx = multiprocessing.get_context("spawn")
    ready_q = ctx.Queue()
    result_q = ctx.Queue()
    stop_event = ctx.Event()
    prefill_group_env = _make_tpu_group_env(
        world_size=constants["prefill_workers"],
        local_rank_base=0,
        topology=prefill_topology)
    decode_group_env = _make_tpu_group_env(
        world_size=constants["decode_tp_size"],
        local_rank_base=constants["prefill_workers"],
        topology=decode_topology)
    prefill_processes = [
        ctx.Process(target=_prefill_worker_process,
                    args=(rank, prefill_group_env, ready_q, stop_event,
                          constants),
                    name=f"mock-prefill-{rank}")
        for rank in range(constants["prefill_workers"])
    ]
    decode_processes = []
    remote_metadata: dict[int, dict] = {}
    prefill_backing_offsets: dict[int, dict[str, int]] = {}
    prefill_final_devices: dict[int, str] = {}

    try:
        for proc in prefill_processes:
            proc.start()

        while len(remote_metadata) < constants["prefill_workers"]:
            message = _queue_get(ready_q)
            kind = message[0]
            if kind == "prefill_ready":
                _, rank, metadata, source_device, source_nbytes, offsets = (
                    message)
                assert source_device.startswith("tpu")
                assert source_nbytes == (len(metadata["regions"]) *
                                         constants["source_size"])
                assert sorted(offsets.values()) == [
                    index * constants["source_size"]
                    for index in range(len(metadata["regions"]))
                ]
                assert all(offset % constants["source_page_bytes"] == 0
                           for offset in offsets.values())
                assert len(metadata["regions"]) == 5
                remote_metadata[rank] = metadata
                prefill_backing_offsets[rank] = offsets
            elif kind == "prefill_error":
                _, rank, tb = message
                raise AssertionError(f"prefill worker {rank} failed:\n{tb}")
            else:
                raise AssertionError(
                    f"unexpected prefill message: {message!r}")

        kv_transfer_params = {
            "uuid":
            123,
            "remote_block_ids": [1, 2],
            "remote_host":
            "unused-by-strided-bridge",
            "remote_port":
            0,
            "remote_metadata": [
                remote_metadata[rank]
                for rank in range(constants["prefill_workers"])
            ],
        }
        mod = _load_v2_module(monkeypatch, stub_zmq=False)
        source_metadata, _, _, _ = _integration_layout(mod,
                                                       decode_tp_rank=0,
                                                       constants=constants)
        kv_transfer_params["strided_source_metadata"] = (
            source_metadata.to_dict())
        decode_processes = [
            ctx.Process(target=_decode_worker_process,
                        args=(rank, kv_transfer_params, decode_group_env,
                              result_q, constants),
                        name=f"mock-decode-{rank}")
            for rank in range(constants["decode_tp_size"])
        ]
        for proc in decode_processes:
            proc.start()

        results = [
            _queue_get(result_q, timeout=20.0)
            for _ in range(constants["decode_tp_size"])
        ]
        for result in sorted(results, key=lambda item: item["rank"]):
            assert result["status"] == "ok", result.get("traceback")
            assert result["copied"] == result["expected_copied"]
            assert result["matches"] is True
            assert result["destination_backing_matches"] is True
            assert result["plan_ranks"] == expected_plan_ranks[result["rank"]]
            assert result["num_plans"] == len(result["plan_ranks"])
            assert result["num_destination_regions"] == 5
            assert result[
                "destination_metadata_has_no_obsolete_region_fields"] is True
            assert result["destination_backing_device"].startswith("tpu")
            assert result["destination_backing_nbytes"] == (
                result["num_destination_regions"] * constants["dst_size"])
            assert sorted(result["destination_region_offsets"].values()) == [
                index * constants["dst_size"]
                for index in range(result["num_destination_regions"])
            ]
            assert all(
                offset % constants["dest_page_bytes"] == 0
                for offset in result["destination_region_offsets"].values())
            assert all(
                device.startswith("tpu")
                for device in result["initial_dst_devices"].values())
            assert all(
                device.startswith("tpu")
                for device in result["final_dst_devices"].values())
            assert result["decode_metadata"]["tcp_port"] > 0
            assert len(result["decode_metadata"]["regions"]) == 5
    finally:
        stop_event.set()
        for proc in decode_processes:
            proc.join(timeout=10.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5.0)
        for proc in prefill_processes:
            proc.join(timeout=10.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5.0)

    for proc in prefill_processes + decode_processes:
        assert proc.exitcode == 0, f"{proc.name} exitcode={proc.exitcode}"

    while True:
        try:
            message = ready_q.get_nowait()
        except queue.Empty:
            break
        kind = message[0]
        if kind == "prefill_stopped":
            _, rank, source_device = message
            prefill_final_devices[rank] = source_device
        elif kind == "prefill_error":
            _, rank, tb = message
            raise AssertionError(f"prefill worker {rank} failed:\n{tb}")
        else:
            raise AssertionError(f"unexpected prefill message: {message!r}")

    assert len(prefill_final_devices) == constants["prefill_workers"]
    assert len(prefill_backing_offsets) == constants["prefill_workers"]
    assert all(
        device.startswith("tpu") for device in prefill_final_devices.values())


def test_multiprocess_prefill_decode_fa_4pcp_gdn_4tp_to_decode_2tp(
        monkeypatch):
    constants = _integration_model_constants(decode_tp_size=2)
    assert constants["decode_tp_size"] == 2
    _run_multiprocess_prefill_decode_integration(
        monkeypatch,
        constants=constants,
        prefill_topology="1,2,1,2",
        decode_topology="1,1,1,2",
        expected_plan_ranks={
            0: [0, 1, 2, 3],
            1: [0, 1, 2, 3],
        },
    )


def test_multiprocess_prefill_decode_fa_4pcp_gdn_4tp_to_decode_1tp(
        monkeypatch):
    constants = _integration_model_constants(decode_tp_size=1)
    assert constants["decode_tp_size"] == 1
    _run_multiprocess_prefill_decode_integration(
        monkeypatch,
        constants=constants,
        prefill_topology="1,2,1,2",
        decode_topology="1,1,1,1",
        expected_plan_ranks={
            0: [0, 1, 2, 3],
        },
    )
