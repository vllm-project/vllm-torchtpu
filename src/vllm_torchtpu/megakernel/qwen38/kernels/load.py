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
"""Bounded checkpoint reads and native quantized device placement.

Only the final sharded model remains resident. Checkpoint reads use bounded
selected ranges and row windows; no canonical full-model or packed disk copy is made.
"""

from __future__ import annotations

import json
import math
import os
import struct
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.sharding import NamedSharding, SingleDeviceSharding
from jax.sharding import PartitionSpec as P

from . import Config
from .layout import NVFP4_WEIGHT_NAMES, WEIGHT_NAMES, Layout, make_mesh, memory_plan

__all__ = [
    "Checkpoint",
    "Layout",
    "WEIGHT_NAMES",
    "initial_states",
    "load",
    "make_mesh",
    "memory_plan",
    "rank_weights",
    "validate_checkpoint",
]

BF16 = np.dtype(ml_dtypes.bfloat16)
FP8 = np.dtype(ml_dtypes.float8_e4m3fn)
_DTYPES = {
    "BF16": BF16,
    "F32": np.dtype(np.float32),
    "F16": np.dtype(np.float16),
    "U8": np.dtype(np.uint8),
    "F8_E4M3": FP8,
    "F8_E4M3FN": FP8,
}


@dataclass(frozen=True)
class TensorInfo:
    shape: tuple[int, ...]
    dtype: np.dtype
    path: Path | None = None
    offset: int = 0


class Checkpoint:
    """Read bounded slices directly from safetensors files, including FP8/BF16."""

    def __init__(self, directory, max_read_bytes=1 << 28):
        self.root = Path(directory)
        self.max_read_bytes = max_read_bytes
        self.headers = {}
        index = self.root / "model.safetensors.index.json"
        if index.is_file():
            self.weight_map = json.loads(index.read_text())["weight_map"]
        else:
            self.weight_map = {}
            for path in sorted(self.root.glob("*.safetensors")):
                self._header(path.name)
                for name in self.headers[path.name]:
                    if name in self.weight_map:
                        raise ValueError(f"Duplicate tensor {name}")
                    self.weight_map[name] = path.name
        if not self.weight_map:
            raise FileNotFoundError(f"No local safetensors checkpoint found in {directory}")

    def _header(self, filename):
        if filename in self.headers:
            return self.headers[filename]
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Checkpoint shard names must be relative without parent traversal")
        # Hugging Face snapshot shards normally symlink into the sibling blobs cache.
        path = self.root / relative
        with path.open("rb") as stream:
            length_bytes = stream.read(8)
            if len(length_bytes) != 8:
                raise ValueError(f"Truncated safetensors header: {path}")
            length = struct.unpack("<Q", length_bytes)[0]
            if length > 64 * 1024**2:
                raise ValueError(f"Excessive safetensors header: {path}")
            header = json.loads(stream.read(length))
        info = {}
        size = path.stat().st_size
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            dtype = _DTYPES.get(tensor["dtype"])
            if dtype is None:
                # Other tensors in a multimodal shard need not use supported dtypes.
                continue
            shape = tuple(tensor["shape"])
            if any(not isinstance(d, int) or d < 0 for d in shape):
                raise ValueError(f"Invalid safetensors shape for {name}")
            lo, hi = tensor["data_offsets"]
            if lo < 0 or hi - lo != math.prod(shape) * dtype.itemsize or 8 + length + hi > size:
                raise ValueError(f"Invalid safetensors extent for {name}")
            info[name] = TensorInfo(shape, dtype, path, 8 + length + lo)
        self.headers[filename] = info
        return info

    def has(self, name):
        return name in self.weight_map

    def info(self, name):
        if name not in self.weight_map:
            raise KeyError(f"Missing checkpoint tensor: {name}")
        header = self._header(self.weight_map[name])
        if name not in header:
            raise ValueError(f"Unsupported checkpoint dtype for {name}")
        return header[name]

    def get(self, name, selection=None):
        info = self.info(name)
        selection = _selection(info.shape, selection)
        shape = _selected_shape(info.shape, selection)
        if math.prod(shape) * info.dtype.itemsize > self.max_read_bytes:
            raise ValueError(f"Read of {name}{shape} exceeds staging limit")
        if math.prod(info.shape) == 0:
            return np.empty(shape, info.dtype)
        return _read_selected(info, selection, self.max_read_bytes)


def _read_into(descriptor, destination, offset):
    """Fill an existing byte view, including short reads, without a bytes copy."""
    done = 0
    while done < len(destination):
        count = os.preadv(descriptor, [destination[done:]], offset + done)
        if count == 0:
            raise EOFError("Checkpoint tensor payload was truncated during range read")
        done += count


def _read_selected(info, selection, max_bytes):
    """Read C-order selected runs; output plus row-window scratch fits max_bytes.

    Contiguous selections read directly into their final output. Strided columns
    coalesce adjacent rows into at most16 MiB windows. This avoids per-page FUSE
    mmap faults without materializing an unbounded tensor or persistent cache.
    """
    shape = _selected_shape(info.shape, selection)
    selected_bytes = math.prod(shape) * info.dtype.itemsize
    if selected_bytes > max_bytes:
        raise ValueError("Selected tensor exceeds staging limit")
    output = np.empty(shape, info.dtype)
    if selected_bytes == 0:
        return output
    starts, lengths = [], []
    for width, item in zip(info.shape, selection, strict=True):
        if isinstance(item, int):
            starts.append(item % width)
            lengths.append(1)
        else:
            left, right, _ = item.indices(width)
            starts.append(left)
            lengths.append(max(0, right - left))
    strides = [math.prod(info.shape[axis + 1 :]) for axis in range(len(info.shape))]
    start = sum(left * stride for left, stride in zip(starts, strides, strict=True))
    flat = output.reshape(-1)
    target = memoryview(flat.view(np.uint8))
    itemsize = info.dtype.itemsize
    descriptor = os.open(info.path, os.O_RDONLY)
    try:
        # Find a contiguous selected suffix and its adjacent, uniformly spaced
        # rows. Flatten complete row axes without creating a list of offsets.
        block_axis = max(0, len(info.shape) - 1)
        while (
            block_axis > 0
            and starts[block_axis] == 0
            and lengths[block_axis] == info.shape[block_axis]
        ):
            block_axis -= 1
        if block_axis == 0:
            _read_into(descriptor, target, info.offset + start * itemsize)
            return output
        row_axis = block_axis - 1
        while row_axis > 0 and starts[row_axis] == 0 and lengths[row_axis] == info.shape[row_axis]:
            row_axis -= 1
        row_elements = math.prod(lengths[block_axis:])
        rows = math.prod(lengths[row_axis:block_axis])
        row_stride = strides[block_axis - 1] * itemsize
        row_bytes = row_elements * itemsize
        scratch_limit = min(16 << 20, max_bytes - output.nbytes)
        rows_per_window = min(rows, max(1, (scratch_limit - row_bytes) // row_stride + 1))
        scratch = None
        if rows_per_window > 1:
            scratch = np.empty((rows_per_window - 1) * row_stride + row_bytes, np.uint8)

        def outer_offsets(axis, offset):
            if axis == row_axis:
                yield offset
            else:
                for index in range(lengths[axis]):
                    yield from outer_offsets(axis + 1, offset + index * strides[axis])

        cursor = 0
        for outer in outer_offsets(0, start):
            for first in range(0, rows, rows_per_window):
                count = min(rows_per_window, rows - first)
                position = info.offset + outer * itemsize + first * row_stride
                selected_bytes = count * row_bytes
                if count == 1:
                    _read_into(descriptor, target[cursor : cursor + selected_bytes], position)
                else:
                    window_bytes = (count - 1) * row_stride + row_bytes
                    _read_into(descriptor, memoryview(scratch)[:window_bytes], position)
                    view = np.ndarray(
                        (count, row_elements),
                        info.dtype,
                        buffer=scratch,
                        strides=(row_stride, itemsize),
                    )
                    destination = flat[cursor // itemsize : (cursor + selected_bytes) // itemsize]
                    np.copyto(destination.reshape(count, row_elements), view, casting="no")
                cursor += selected_bytes
        return output
    finally:
        os.close(descriptor)


class _ArraySource:
    def __init__(self, tensors):
        self.tensors = tensors

    def has(self, name):
        return name in self.tensors

    def info(self, name):
        value = self.tensors[name]
        return TensorInfo(tuple(value.shape), np.dtype(value.dtype))

    def get(self, name, selection=None):
        value = self.tensors[name]
        return np.array(value[_selection(value.shape, selection)], copy=True)


def _selection(shape, selection):
    if selection is None:
        return (slice(None),) * len(shape)
    if not isinstance(selection, tuple):
        selection = (selection,)
    if len(selection) > len(shape):
        raise IndexError("Too many tensor dimensions selected")
    selection += (slice(None),) * (len(shape) - len(selection))
    for width, item in zip(shape, selection, strict=True):
        if isinstance(item, int):
            if not -width <= item < width:
                raise IndexError("Tensor index outside shape")
        elif not isinstance(item, slice) or item.step not in (None, 1):
            raise ValueError("Only contiguous slices and scalar indices are supported")
    return selection


def _selected_shape(shape, selection):
    return tuple(
        max(0, item.indices(width)[1] - item.indices(width)[0])
        for width, item in zip(shape, selection, strict=True)
        if isinstance(item, slice)
    )


def _prefix(source):
    for prefix in ("model.", "model.language_model."):
        if source.has(prefix + "embed_tokens.weight"):
            return prefix
    raise KeyError("Missing text embedding tensor")


def validate_checkpoint(source, config):
    """Validate names/shapes/dtypes from headers before allocating device buffers."""
    c, prefix = config, _prefix(source)

    def check(name, shape, dtype=BF16):
        info = source.info(name)
        if info.shape != tuple(shape) or info.dtype != np.dtype(dtype):
            raise ValueError(
                f"{name}: expected {tuple(shape)} {np.dtype(dtype)}, found {info.shape} {info.dtype}"
            )

    def projection(name, shape):
        check(name + ".weight", shape, FP8 if c.quantization == "nvfp4" else BF16)
        if c.quantization == "nvfp4":
            check(name + ".input_scale", (1,), np.float32)
            check(name + ".weight_scale", (1,), np.float32)

    check(prefix + "embed_tokens.weight", (c.vocab, c.dim))
    check("lm_head.weight", (c.vocab, c.dim))
    check(prefix + "norm.weight", (c.dim,))
    for layer in range(c.layers):
        p = f"{prefix}layers.{layer}."
        for norm in ("input_layernorm", "post_attention_layernorm"):
            check(p + norm + ".weight", (c.dim,))
        m = p + "mlp."
        check(m + "gate.weight", (c.experts, c.dim))
        check(m + "shared_expert_gate.weight", (1, c.dim))
        for name in ("gate_proj", "up_proj"):
            check(m + "shared_expert." + name + ".weight", (c.intermediate, c.dim))
        check(m + "shared_expert.down_proj.weight", (c.dim, c.intermediate))
        for expert in range(c.experts):
            e = m + f"experts.{expert}."
            for name, shape in (
                ("gate_proj", (c.moe_intermediate, c.dim)),
                ("up_proj", (c.moe_intermediate, c.dim)),
                ("down_proj", (c.dim, c.moe_intermediate)),
            ):
                if c.quantization == "nvfp4":
                    n, k = shape
                    check(e + name + ".weight", (n, k // 2), np.uint8)
                    check(e + name + ".weight_scale", (n, k // 16), FP8)
                    check(e + name + ".weight_scale_2", (1,), np.float32)
                    check(e + name + ".input_scale", (1,), np.float32)
                else:
                    check(e + name + ".weight", shape, FP8)
                    check(e + name + ".weight_scale_inv", tuple(d // 128 for d in shape))
        if layer in c.full_attention:
            a = p + "self_attn."
            for name, shape in (
                ("q_proj", (2 * c.heads * c.head_dim, c.dim)),
                ("k_proj", (c.kv_heads * c.head_dim, c.dim)),
                ("v_proj", (c.kv_heads * c.head_dim, c.dim)),
                ("o_proj", (c.dim, c.heads * c.head_dim)),
            ):
                projection(a + name, shape)
            for norm in ("q_norm", "k_norm"):
                check(a + norm + ".weight", (c.head_dim,))
        else:
            a = p + "linear_attn."
            for name, shape in (
                ("in_proj_qkv", (c.conv_dim, c.dim)),
                ("in_proj_z", (c.value_dim, c.dim)),
                ("in_proj_b", (c.v_heads, c.dim)),
                ("in_proj_a", (c.v_heads, c.dim)),
                ("out_proj", (c.dim, c.value_dim)),
            ):
                projection(a + name, shape)
            check(a + "conv1d.weight", (c.conv_dim, 1, c.conv_size))
            for name in ("A_log", "dt_bias"):
                check(a + name, (c.v_heads,))
            check(a + "norm.weight", (c.v_dim,))
    return prefix


def _rank_chunks(
    source, c, rank, layout, max_payload_bytes, *, include_experts=True, layer_ids=None
):
    """Yield (family, local update offset, payload), each bounded in host memory."""
    layout.validate_rank(rank)
    prefix = _prefix(source)
    h, kh, vh = c.dim, layout.local_k_heads, layout.local_v_heads
    key_start = layout.key_head_start(rank) * c.k_dim
    value_start = rank * vh * c.v_dim
    qkv_ranges = (
        (key_start, kh * c.k_dim),
        (c.key_dim + key_start, kh * c.k_dim),
        (2 * c.key_dim + value_start, vh * c.v_dim),
    )

    def get(name, selection=None):
        return source.get(name, selection)

    def scale_pair(name, weight_suffix="weight_scale"):
        values = [get(name + ".input_scale"), get(name + "." + weight_suffix)]
        pair = np.asarray([value.item() for value in values], np.float32)
        if not np.all(np.isfinite(pair) & (pair > 0)):
            raise ValueError(f"{name}: quantization scales must be finite and positive")
        return pair

    def item(name, data, ordinal=None, expert_start=None):
        data = np.ascontiguousarray(data, dtype=np.dtype(layout.weight_specs[name].dtype))
        if ordinal is None:
            return name, (0,) * data.ndim, data
        if expert_start is None:
            return name, (ordinal,) + (0,) * data.ndim, data[None]
        return name, (ordinal, expert_start) + (0,) * (data.ndim - 1), data[None]

    vlo = rank * layout.vocab_shard
    vhi = min(vlo + layout.vocab_shard, c.vocab)
    count = max(0, vhi - vlo)
    embedding = np.zeros(layout.weight_specs["emb"].shape, BF16)
    if count:
        embedding[:count] = get(prefix + "embed_tokens.weight", (slice(vlo, vhi), slice(None)))
    yield item("emb", embedding)
    del embedding
    lm = np.zeros(layout.weight_specs["lmw"].shape, BF16)
    if count:
        lm[:, :count] = get("lm_head.weight", (slice(vlo, vhi), slice(None))).T
    yield item("lmw", lm)
    del lm
    yield item("fnorm", get(prefix + "norm.weight"))
    linear_ordinals = {layer: i for i, layer in enumerate(c.linear_layers)}
    full_ordinals = {layer: i for i, layer in enumerate(c.full_layers)}
    selected_layers = set(range(c.layers)) if layer_ids is None else set(layer_ids)
    if not selected_layers <= set(range(c.layers)):
        raise ValueError("Selected layers are outside the model")
    for layer in range(c.layers):
        if layer not in selected_layers:
            continue
        p = f"{prefix}layers.{layer}."
        yield item("norm_in", get(p + "input_layernorm.weight"), layer)
        yield item("norm_post", get(p + "post_attention_layernorm.weight"), layer)
        m = p + "mlp."
        rstart = rank % layout.router_tp * (c.experts // layout.router_tp)
        yield item(
            "router",
            get(
                m + "gate.weight",
                (slice(rstart, rstart + c.experts // layout.router_tp), slice(None)),
            ).T,
            layer,
        )
        yield item("shared_gate", get(m + "shared_expert_gate.weight")[0], layer)
        start, width, padded = (
            rank * layout.shared_width,
            layout.shared_width,
            layout.shared_width_padded,
        )
        gu = np.zeros((h, padded * 2), BF16)
        for index, name in enumerate(("gate_proj", "up_proj")):
            gu[:, index * padded : index * padded + width] = get(
                m + "shared_expert." + name + ".weight", (slice(start, start + width), slice(None))
            ).T
        yield item("gu", gu, layer)
        del gu
        down = np.zeros((padded, h), BF16)
        down[:width] = get(
            m + "shared_expert.down_proj.weight", (slice(None), slice(start, start + width))
        ).T
        yield item("dw", down, layer)
        del down
        if layer in c.full_attention:
            a, ordinal = p + "self_attn.", full_ordinals[layer]
            qlo = rank * layout.local_q_heads * c.head_dim
            klo = layout.key_head_start(rank, full=True) * c.head_dim
            q = get(
                a + "q_proj.weight",
                (slice(2 * qlo, 2 * (qlo + layout.local_q_heads * c.head_dim)), slice(None)),
            ).T
            kv = [
                get(
                    a + name + ".weight",
                    (slice(klo, klo + layout.local_kv_heads * c.head_dim), slice(None)),
                ).T
                for name in ("k_proj", "v_proj")
            ]
            yield item("qwkv", np.concatenate((q, *kv), axis=1), ordinal)
            del q, kv
            yield item(
                "ow",
                get(
                    a + "o_proj.weight",
                    (slice(None), slice(qlo, qlo + layout.local_q_heads * c.head_dim)),
                ).T,
                ordinal,
            )
            for short, name in (("qn", "q_norm"), ("kn", "k_norm")):
                yield item(short, get(a + name + ".weight"), ordinal)
            if c.quantization == "nvfp4":
                params = np.zeros((8, 128), np.float32)
                for index, name in enumerate(("q_proj", "k_proj", "v_proj", "o_proj")):
                    params[0, 2 * index : 2 * index + 2] = scale_pair(a + name)
                yield item("fparams", params, ordinal)
        else:
            a, ordinal = p + "linear_attn.", linear_ordinals[layer]
            qkvz = np.zeros(
                layout.weight_specs["qkvz"].shape[1:], np.dtype(layout.weight_specs["qkvz"].dtype)
            )
            col = 0
            for lo, size in qkv_ranges:
                qkvz[:, col : col + size] = get(
                    a + "in_proj_qkv.weight", (slice(lo, lo + size), slice(None))
                ).T
                col += size
            vhead = rank * vh
            for name, lo, size in (
                ("in_proj_z", value_start, vh * c.v_dim),
                ("in_proj_b", vhead, vh),
                ("in_proj_a", vhead, vh),
            ):
                qkvz[:, col : col + size] = get(
                    a + name + ".weight", (slice(lo, lo + size), slice(None))
                ).T
                col += size
            yield item("qkvz", qkvz, ordinal)
            del qkvz
            yield item(
                "outw",
                get(
                    a + "out_proj.weight",
                    (slice(None), slice(value_start, value_start + vh * c.v_dim)),
                ).T,
                ordinal,
            )
            conv = np.concatenate(
                [
                    get(a + "conv1d.weight", (slice(lo, lo + size), 0, slice(None))).T
                    for lo, size in qkv_ranges
                ],
                axis=1,
            )
            yield item("convw", conv, ordinal)
            del conv
            yield item(
                "alog", get(a + "A_log", slice(vhead, vhead + vh)).astype(np.float32), ordinal
            )
            yield item("dtb", get(a + "dt_bias", slice(vhead, vhead + vh)), ordinal)
            yield item("onorm", get(a + "norm.weight"), ordinal)
            if c.quantization == "nvfp4":
                params = np.zeros((8, 128), np.float32)
                for index, name in enumerate(
                    ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")
                ):
                    params[0, 2 * index : 2 * index + 2] = scale_pair(a + name)
                yield item("lparams", params, ordinal)
        if not include_experts:
            continue
        first, last = layout.expert_range(rank)
        lo, hi = layout.expert_column_range(rank)
        if layout.fast:
            from .precision import requantize_kmajor

            # One expert/projection at a time; no canonical dequantized copy.
            for expert in range(first, last):
                e = m + f"experts.{expert}."
                params = np.zeros((1, 8, 128), np.float32)
                for index, name in enumerate(("gate_proj", "up_proj", "down_proj")):
                    pair = scale_pair(e + name, "weight_scale_2")
                    params[0, 0, 2 * index : 2 * index + 2] = pair
                    if index < 2:
                        selection = (slice(lo, hi), slice(None))
                        scale_selection = selection
                    else:
                        selection = (slice(None), slice(lo // 2, hi // 2))
                        scale_selection = (slice(None), slice(lo // 16, hi // 16))
                    packed = get(e + name + ".weight", selection).T[None]
                    native_scale = get(e + name + ".weight_scale", scale_selection).T[None]
                    weight, scale = requantize_kmajor(packed, native_scale, pair[1:2])
                    del packed, native_scale
                    family = "egu" if index < 2 else "edw"
                    output_offset = index * (hi - lo) if index < 2 else 0
                    for target, value in ((family, weight), (family + "_scale", scale)):
                        if value.nbytes > max_payload_bytes:
                            raise ValueError("Converted expert exceeds bounded staging budget")
                        yield target, (layer, expert - first, 0, output_offset), value[None]
                yield item("eparams", params, layer, expert - first)
            continue
        expert_families = ("egu", "egu_scale", "edw", "edw_scale")
        if c.quantization == "nvfp4":
            expert_families += ("eparams",)
        for family in expert_families:
            spec = layout.weight_specs[family]
            per_expert = math.prod(spec.shape[2:]) * np.dtype(spec.dtype).itemsize
            chunk_size = max_payload_bytes // per_expert
            if chunk_size < 1:
                raise ValueError(f"One {family} expert exceeds staging limit")
            for begin in range(first, last, chunk_size):
                end = min(begin + chunk_size, last)
                block = np.empty((end - begin, *spec.shape[2:]), np.dtype(spec.dtype))
                scale = family.endswith("scale")
                nvfp4 = c.quantization == "nvfp4"
                suffix = (".weight_scale" if nvfp4 else ".weight_scale_inv") if scale else ".weight"
                for local, expert in enumerate(range(begin, end)):
                    e = m + f"experts.{expert}."
                    if family == "eparams":
                        block[local].fill(0)
                        for index, name in enumerate(("gate_proj", "up_proj", "down_proj")):
                            block[local, 0, 2 * index : 2 * index + 2] = scale_pair(
                                e + name, "weight_scale_2"
                            )
                    elif family.startswith("egu"):
                        # NVFP4 packs/groups K only. Gate/up ETP partitions N,
                        # so their row selection and concatenation remain exact.
                        divisor = 128 if scale and not nvfp4 else 1
                        left, right = lo // divisor, hi // divisor
                        width = (hi - lo) // divisor
                        for half, name in enumerate(("gate_proj", "up_proj")):
                            block[local, :, half * width : (half + 1) * width] = get(
                                e + name + suffix, (slice(left, right), slice(None))
                            ).T
                    else:
                        # Down ETP partitions K: select complete packed bytes
                        # or complete 16-value scale groups, never unpack them.
                        divisor = (16 if scale else 2) if nvfp4 else (128 if scale else 1)
                        left, right = lo // divisor, hi // divisor
                        block[local] = get(
                            e + "down_proj" + suffix, (slice(None), slice(left, right))
                        ).T
                yield item(family, block, layer, begin - first)
                del block


def device_record(layout, name, value):
    """Pack a stored shared-expert record (``Layout.record_specs``) into its device shape."""
    sw, pad = layout.shared_width, layout.shared_width_padded
    if name not in ("gu", "dw") or sw == pad:
        return value
    if name == "dw":
        if np.any(value[..., sw:, :]):
            raise ValueError("Shared expert down rows past the shard must be zero")
        return np.ascontiguousarray(value[..., :sw, :])
    if np.any(value[..., sw:pad]) or np.any(value[..., pad + sw :]):
        raise ValueError("Shared expert gate/up columns past the shards must be zero")
    packed = np.zeros((*value.shape[:-1], layout.shared_gate_up_width), value.dtype)
    packed[..., :sw] = value[..., :sw]
    packed[..., sw : 2 * sw] = value[..., pad : pad + sw]
    return packed


def rank_weights(canonical, config, rank, layout):
    """Pack a small HF-name tensor mapping for CPU fixtures; never use for full models."""
    if config != layout.config:
        raise ValueError("Layout and model config differ")
    host_bytes = sum(
        math.prod(spec.shape) * np.dtype(spec.dtype).itemsize
        for spec in layout.weight_specs.values()
    )
    if host_bytes > 1 << 30:
        raise ValueError("rank_weights is a small-fixture helper; use load for full checkpoints")
    source = _ArraySource(canonical) if isinstance(canonical, Mapping) else canonical
    result = {
        name: np.zeros(spec.shape, np.dtype(spec.dtype))
        for name, spec in layout.weight_specs.items()
    }
    # Zero initialization supplies the complete aligned vector-table tails;
    # checkpoint slices fill only mathematical rows and head columns.
    for name, offsets, value in _rank_chunks(source, config, rank, layout, 1 << 28):
        value = device_record(layout, name, value)
        selection = tuple(
            slice(start, start + size) for start, size in zip(offsets, value.shape, strict=True)
        )
        result[name][selection] = value
    return result


def _zero_device(spec, device):
    placement = SingleDeviceSharding(device)
    return jax.jit(lambda: jnp.zeros((1, *spec.shape), spec.dtype), out_shardings=placement)()


def _global_arrays(mesh, specs, buffers):
    sharding = NamedSharding(mesh, P("tp"))
    devices = list(mesh.devices.flat)
    rank_by_device = {device: rank for rank, device in enumerate(devices)}
    result = {}
    for name, spec in specs.items():
        shape = (mesh.size, *spec.shape)
        ordered = [
            buffers[rank_by_device[d]][name]
            for d in sharding.addressable_devices_indices_map(shape)
        ]
        result[name] = jax.make_array_from_single_device_arrays(shape, sharding, ordered)
    return result


def initial_states(mesh, config, context, batch_size, layout=None):
    layout = layout or Layout(config, mesh.size, min(4, mesh.size), min(4, mesh.size))
    if layout.config != config or layout.tp != mesh.size:
        raise ValueError("State layout does not match config/mesh")
    specs = layout.state_specs(context, batch_size)
    buffers = {
        rank: {name: _zero_device(spec, device) for name, spec in specs.items()}
        for rank, device in enumerate(mesh.devices.flat)
        if device.process_index == jax.process_index()
    }
    return _global_arrays(mesh, specs, buffers)


class _DonatedSliceUpdater:
    """Compile each bounded update shape once and verify destination reuse."""

    def __init__(self, device, log=None):
        self.device = device
        self.log = log
        self.variants = {}
        placement = SingleDeviceSharding(device)

        @jax.jit(donate_argnums=(0,), out_shardings=placement)
        def update(destination, chunk, starts):
            return jax.lax.dynamic_update_slice(destination, chunk, starts)

        self.program = update

    def _compile(self, *args):
        return self.program.lower(*args).compile()

    def _qualify(self, message):
        message = f"Device {self.device.id} loader qualification: {message}"
        if self.log is not None:
            self.log(message)
        else:
            warnings.warn(message, RuntimeWarning, stacklevel=3)

    def _audit(self, executable, destination):
        try:
            analysis = executable.memory_analysis()
        except (AttributeError, NotImplementedError):
            analysis = None
        get = (
            (lambda name: analysis.get(name))
            if isinstance(analysis, Mapping)
            else (lambda name: getattr(analysis, name, None))
        )
        aliases = get("alias_size_in_bytes")
        temporary = get("temp_size_in_bytes")
        output = get("output_size_in_bytes")
        record = {
            "destination_bytes": destination.nbytes,
            "alias_bytes": aliases,
            "temporary_bytes": temporary,
            "output_bytes": output,
        }
        if aliases is None:
            self._qualify(
                "compiler alias statistics unavailable; donation is requested but not verified"
            )
        elif aliases < destination.nbytes:
            raise MemoryError(
                f"Device {self.device.id}: loader update aliases only {aliases} of "
                f"{destination.nbytes} destination bytes; refusing a possible full-buffer copy"
            )
        if temporary is None or output is None:
            self._qualify(
                "compiler temporary/output statistics unavailable; peak update HBM is unverified"
            )
        try:
            expected = executable.input_formats[0][0].layout
        except (AttributeError, NotImplementedError):
            expected = None
        actual = getattr(getattr(destination, "format", None), "layout", None)
        if expected is not None and actual is not None and expected != actual:
            raise MemoryError(
                "Loader update requires a destination layout conversion; refusing a full-buffer copy"
            )
        if expected is None or actual is None:
            self._qualify(
                "destination layout metadata unavailable; implicit input conversion cannot be ruled out"
            )
        record["input_layout"] = expected
        if aliases is not None and temporary is not None and output is not None:
            record["additional_device_bytes"] = temporary + max(0, output - aliases)
        else:
            record["additional_device_bytes"] = None
        if self.log is not None and destination.nbytes >= 1 << 30 and aliases is not None:
            self.log(
                f"Device {self.device.id} loader update: {aliases} aliased bytes, "
                f"{temporary} temporary bytes for shape {destination.shape}"
            )
        return record

    def __call__(self, destination, chunk, starts):
        key = tuple((value.shape, str(value.dtype)) for value in (destination, chunk, starts))
        if key not in self.variants:
            if len(self.variants) >= 2 * len(NVFP4_WEIGHT_NAMES):
                raise RuntimeError("Unexpected number of loader update variants")
            executable = self._compile(destination, chunk, starts)
            record = self._audit(executable, destination)
            self.variants[key] = executable, record
        executable, record = self.variants[key]
        expected = record["input_layout"]
        actual = getattr(getattr(destination, "format", None), "layout", None)
        if expected is not None and actual is not None and expected != actual:
            raise MemoryError(
                "Destination layout changed during loading; refusing an implicit copy"
            )
        extra = record["additional_device_bytes"]
        if extra is not None:
            stats = self.device.memory_stats() or {}
            if "bytes_limit" in stats and "bytes_in_use" in stats:
                available = stats["bytes_limit"] - stats["bytes_in_use"]
                if extra > available:
                    raise MemoryError(
                        f"Device {self.device.id}: compiled update needs {extra} additional bytes, "
                        f"only {available} available after resident buffers and staging"
                    )
            elif self.device.platform == "tpu" and not record.get("capacity_qualified"):
                self._qualify(
                    "allocator statistics unavailable; compiled temporary capacity is unverified"
                )
                record["capacity_qualified"] = True
        return executable(destination, chunk, starts)


def load(checkpoint, mesh, layout, *, max_staging_bytes=1 << 30, log=None):
    """Load only addressable shards into donated final buffers, one bounded update at a time."""
    if layout.tp != mesh.size or tuple(mesh.axis_names) != ("tp",):
        raise ValueError("Mesh must match layout and have the single axis 'tp'")
    if max_staging_bytes < 1 << 20:
        raise ValueError("max_staging_bytes must be at least 1 MiB")
    if (Path(checkpoint) / "preshard-manifest.json").is_file():
        from .load_preshard import load_presharded

        return load_presharded(
            checkpoint, mesh, layout, max_staging_bytes=max_staging_bytes, log=log
        )
    if (Path(checkpoint) / "preshard-source.json").exists() or (
        Path(checkpoint) / "experts"
    ).exists():
        raise ValueError("Pre-sharded checkpoint is incomplete: published manifest is missing")
    if Config.from_checkpoint(checkpoint) != layout.config:
        raise ValueError("Checkpoint config differs from the requested layout")
    if layout.expert_overlap:
        raise ValueError("Expert overlap copies load from the pre-sharded checkpoint only")
    # Budget for source slice, packed host payload, transferred device payload,
    # and temporary transpose/concatenation. Final donated arrays are separate.
    payload_limit = max_staging_bytes // 4
    source = Checkpoint(checkpoint, max_read_bytes=payload_limit)
    validate_checkpoint(source, layout.config)
    specs = layout.weight_specs
    dense_max = max(
        math.prod(spec.shape[1:] if name not in ("emb", "lmw", "fnorm") else spec.shape)
        * np.dtype(spec.dtype).itemsize
        for name, spec in layout.record_specs.items()
        if not name.startswith("e") or name == "emb"
    )
    if dense_max > payload_limit:
        raise ValueError(f"Largest dense payload ({dense_max} bytes) exceeds staging budget")
    local = [
        (rank, device)
        for rank, device in enumerate(mesh.devices.flat)
        if device.process_index == jax.process_index()
    ]
    weight_bytes = sum(
        math.prod(spec.shape) * np.dtype(spec.dtype).itemsize for spec in specs.values()
    )
    physical_weight_bytes = memory_plan(layout.config, layout, 1, 1)[
        "physical_weight_bytes_per_device_estimate"
    ]
    for _, device in local:
        stats = device.memory_stats() or {}
        if stats.get("bytes_limit") is not None:
            available = stats["bytes_limit"] - stats.get("bytes_in_use", 0)
            needed = physical_weight_bytes + max_staging_bytes
            if available < needed:
                raise MemoryError(
                    f"Device {device.id}: {needed} weight/staging bytes required, {available} available"
                )
    buffers = {}
    for rank, device in local:
        if log:
            log(f"Allocating/loading rank {rank}: {weight_bytes / 2**30:.2f} GiB weights")
        arrays = {name: _zero_device(spec, device) for name, spec in specs.items()}
        # Capacity checks must observe every resident destination allocation.
        # These device-created arrays also avoid borrowed host storage, whose
        # external ownership can prevent runtime donation despite HLO aliases.
        jax.block_until_ready(arrays)
        update = _DonatedSliceUpdater(device, log=log)

        for name, offsets, value in _rank_chunks(
            source, layout.config, rank, layout, payload_limit
        ):
            if value.nbytes > payload_limit:
                raise ValueError(f"{name} payload exceeds staging budget")
            value = device_record(layout, name, value)
            chunk = jax.device_put(value[None], device)
            starts = jax.device_put(np.asarray((0, *offsets), np.int32), device)
            arrays[name] = update(arrays[name], chunk, starts)
            arrays[name].block_until_ready()
            if (
                log
                and name == "edw_scale"
                and offsets[1] + value.shape[1] == layout.local_experts
                and ((offsets[0] + 1) % 8 == 0 or offsets[0] + 1 == layout.config.layers)
            ):
                log(f"Rank {rank}: loaded {offsets[0] + 1}/{layout.config.layers} layers")
            del chunk, starts, value
        buffers[rank] = arrays
        if log:
            verified = sum(
                record["alias_bytes"] is not None for _, record in update.variants.values()
            )
            log(
                f"Rank {rank} resident; {verified}/{len(update.variants)} compiled update variants have verified destination aliases"
            )
    return _global_arrays(mesh, specs, buffers)
