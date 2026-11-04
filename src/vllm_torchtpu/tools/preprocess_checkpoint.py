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
"""Offline checkpoint preprocessor for TPU MoE and PLE models.

Converts HuggingFace Safetensors checkpoints (such as
``Qwen/Qwen3.8-Flash-Next-FP8``) into the pre-quantized, K-major transposed,
and EP/TP-repacked raw array format consumed by
``vllm_torchtpu.model_loader.preprocessed_loader``:

1. ``convert``:
   - Reads per-expert FP8 (or BF16) ``gate_proj``, ``up_proj``, and ``down_proj``
     tensors one MoE layer at a time via byte-range ``pread``.
   - Dequantizes 2D block FP8 weights to float32, pads intermediate size to a
     multiple of 128 if needed, requantizes along the contracting dimension,
     transposes to K-major ``[E, K, N]``, and writes ``_tp1`` raw arrays under
     ``experts/array_<layer_prefix>.mlp.experts.{w13_weight,w13_weight_scale_inv,w2_weight,w2_weight_scale_inv}/``.
   - Merges split PLE n-gram embedding shards (``ngram_embedding.shard_<i>.weight``)
     into one padded contiguous ``_tp1`` raw array under
     ``misc/array_<layer_prefix>.ple.ple_embedding.ngram_embedding.weight/``
     (plus ``weight_scale`` and hash buffers under ``_tp0``).
   - Writes remaining dense weights into consolidated ``dense-*.safetensors``
     shards with an updated ``model.safetensors.index.json`` (shrinking a
     152,089-tensor index down to ~1,400 dense tensors).

2. ``repack-tp``:
   - Repacks ``_tp1`` expert arrays into ``_ep<E>tp<T>`` so every ``(ep_rank, tp_rank)``
     tile is a single contiguous byte range on disk.
   - When ``tp > 1``, uses ``paired_columns`` for ``w13_*`` and ``rows`` for
     ``w2_*`` (with 128-alignment padding when ``(I // tp) % 128 != 0``).
   - Supports ``--verify`` to check bit-exact equivalence of all repacked tiles.

3. ``pipeline``:
   - Runs ``convert`` followed by ``repack-tp`` in a single invocation.

Usage::

    # Convert Qwen/Qwen3.8-Flash-Next-FP8 to _tp1 raw arrays + dense safetensors
    python -m vllm_torchtpu.tools.preprocess_checkpoint convert \\
        --input-dir /path/to/Qwen3.8-Flash-Next-FP8 \\
        --output-dir /path/to/Qwen3.8-Flash-Next-FP8-preprocessed

    # Repack for EP=8, TP=1 (or EP=4, TP=2) with verification
    python -m vllm_torchtpu.tools.preprocess_checkpoint repack-tp \\
        --checkpoint-dir /path/to/Qwen3.8-Flash-Next-FP8-preprocessed \\
        --ep 8 --tp 1 --verify

    # Run both stages in one command
    python -m vllm_torchtpu.tools.preprocess_checkpoint pipeline \\
        --input-dir /path/to/Qwen3.8-Flash-Next-FP8 \\
        --output-dir /path/to/Qwen3.8-Flash-Next-FP8-preprocessed \\
        --ep 8 --tp 1 --verify
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from vllm_torchtpu.model_loader.preprocessed_format import (
    MANIFEST_FILENAME,
    TORCH_TO_DTYPE_TOKEN,
    RawLeafName,
    dequantize_block_tensor,
    parse_raw_leaf_name,
    quantize_block_tensor,
    repack_expert_paired_cols_for_ep_tp,
    repack_expert_rows_for_ep_tp,
    resolve_and_read_shard,
    write_raw_array,
)

# Safetensors header dtype string -> torch.dtype
_SAFETENSORS_DTYPE_TO_TORCH: dict[str, torch.dtype] = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "F8_E4M3": torch.float8_e4m3fn,
    "I8": torch.int8,
    "U8": torch.uint8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "BOOL": torch.bool,
}

_TORCH_TO_SAFETENSORS_DTYPE: dict[torch.dtype, str] = {
    v: k for k, v in _SAFETENSORS_DTYPE_TO_TORCH.items()
}

# Regex matching per-expert tensor names in HF checkpoints:
# e.g. model.language_model.layers.0.mlp.experts.511.gate_proj.weight
# or   mtp.layers.0.mlp.experts.0.down_proj.weight_scale_inv
_EXPERT_TENSOR_RE = re.compile(
    r"^(?P<prefix>.+\.mlp\.experts)\.(?P<expert_id>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj|w1|w2|w3)\."
    r"(?P<field>weight|weight_scale_inv|weight_scale)$"
)

# Regex matching PLE n-gram embedding tensors:
# e.g. model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight
_PLE_SHARD_RE = re.compile(
    r"^(?P<prefix>.+\.ple\.ple_embedding)\.ngram_embedding\.shard_(?P<shard_id>\d+)\.weight$"
)
_PLE_AUX_RE = re.compile(
    r"^(?P<prefix>.+\.ple\.ple_embedding)\."
    r"(?P<field>ngram_embedding\.weight_scale|layer_multipliers|"
    r"ngram_heads_vocab_sizes|ngram_heads_offsets)$"
)


@dataclass(frozen=True)
class TensorLocation:
    """Byte range and metadata of a single tensor inside a `.safetensors` file."""

    file_path: Path
    dtype_str: str
    shape: tuple[int, ...]
    data_start: int
    data_end: int

    @property
    def torch_dtype(self) -> torch.dtype:
        if self.dtype_str not in _SAFETENSORS_DTYPE_TO_TORCH:
            raise ValueError(f"Unsupported safetensors dtype {self.dtype_str!r}")
        return _SAFETENSORS_DTYPE_TO_TORCH[self.dtype_str]

    @property
    def nbytes(self) -> int:
        return self.data_end - self.data_start


def canonicalize_prefix(prefix: str) -> str:
    """Normalize HF VLM wrapper prefix `model.language_model.` -> `model.`."""
    if prefix.startswith("model.language_model."):
        return "model." + prefix[len("model.language_model.") :]
    return prefix


def scan_safetensors_headers(input_dir: Path) -> dict[str, TensorLocation]:
    """Scan all `.safetensors` headers in `input_dir` without loading tensor data."""
    index_path = input_dir / "model.safetensors.index.json"
    if index_path.is_file():
        with open(index_path, encoding="utf-8") as f:
            index_data = json.load(f)
        shard_files = sorted(
            {input_dir / fname for fname in index_data.get("weight_map", {}).values()}
        )
    else:
        shard_files = sorted(input_dir.glob("*.safetensors"))

    if not shard_files:
        raise FileNotFoundError(f"No .safetensors files found in {input_dir}")

    locations: dict[str, TensorLocation] = {}
    for fpath in shard_files:
        with open(fpath, "rb") as f:
            raw_len = f.read(8)
            if len(raw_len) < 8:
                raise ValueError(f"Truncated safetensors header in {fpath}")
            (header_len,) = struct.unpack("<Q", raw_len)
            header = json.loads(f.read(header_len))
        base_offset = 8 + header_len
        for name, info in header.items():
            if name == "__metadata__":
                continue
            start, end = info["data_offsets"]
            locations[name] = TensorLocation(
                file_path=fpath,
                dtype_str=info["dtype"],
                shape=tuple(int(d) for d in info["shape"]),
                data_start=base_offset + int(start),
                data_end=base_offset + int(end),
            )
    return locations


def read_safetensors_tensor(loc: TensorLocation) -> torch.Tensor:
    """Read a single tensor from its `.safetensors` byte range via `os.pread`."""
    buf = torch.empty(loc.nbytes, dtype=torch.uint8, device="cpu")
    if loc.nbytes > 0:
        mv = memoryview(buf.numpy())
        fd = os.open(loc.file_path, os.O_RDONLY)
        try:
            pos = 0
            while pos < loc.nbytes:
                chunk = os.pread(fd, loc.nbytes - pos, loc.data_start + pos)
                if not chunk:
                    raise EOFError(
                        f"Unexpected EOF reading {loc.file_path} at "
                        f"{loc.data_start + pos}"
                    )
                mv[pos : pos + len(chunk)] = chunk
                pos += len(chunk)
        finally:
            os.close(fd)
    return buf.view(loc.torch_dtype).reshape(loc.shape)


def write_safetensors_file(
    path: Path,
    tensors: dict[str, torch.Tensor],
) -> None:
    """Write a dictionary of CPU tensors to a `.safetensors` file."""
    header: dict[str, Any] = {}
    offset = 0
    prepared: list[tuple[str, torch.Tensor, int]] = []
    for name in sorted(tensors):
        t = tensors[name].detach().cpu().contiguous()
        if t.dtype not in _TORCH_TO_SAFETENSORS_DTYPE:
            raise ValueError(
                f"Cannot serialize dtype {t.dtype} for {name} to safetensors"
            )
        nbytes = t.numel() * t.element_size()
        header[name] = {
            "dtype": _TORCH_TO_SAFETENSORS_DTYPE[t.dtype],
            "shape": list(t.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        prepared.append((name, t, nbytes))
        offset += nbytes

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # Pad header to 8-byte alignment with spaces per safetensors spec.
    pad_len = (8 - (len(header_bytes) % 8)) % 8
    if pad_len:
        header_bytes += b" " * pad_len

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.tmp.{os.getpid()}"
    with open(tmp_path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for _, t, _ in prepared:
            f.write(t.view(torch.uint8).numpy().tobytes())
    os.replace(tmp_path, path)


def _dequant_expert_2d(
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    weight_block_size: tuple[int, int] | None,
) -> torch.Tensor:
    """Dequantize one expert's `[out_dim, in_dim]` weight to `float32`."""
    if scale is None:
        return weight.to(torch.float32)
    if weight_block_size is not None:
        return dequantize_block_tensor(
            weight.unsqueeze(0),
            scale.unsqueeze(0),
            axis=(1, 2),
            out_dtype=torch.float32,
        ).squeeze(0)
    w_f32 = weight.to(torch.float32)
    s_f32 = scale.to(torch.float32)
    if s_f32.ndim == 1 and s_f32.shape[0] == w_f32.shape[0]:
        s_f32 = s_f32.unsqueeze(-1)
    return w_f32 * s_f32


def convert_moe_layer_experts(
    expert_entries: dict[int, dict[tuple[str, str], TensorLocation]],
    *,
    weight_block_size: tuple[int, int] | None,
    requant_dtype: torch.dtype = torch.float8_e4m3fn,
    requant_block_size: int | None = None,
    activation: str = "silu",
    expert_chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert all experts of one MoE layer into K-major `_tp1` tensors.

    Matches `VllmFp8MoEMethodTPU.process_weights_after_loading` bit-for-bit:
    1. Dequantizes `gate_proj`, `up_proj`, `down_proj` to float32.
    2. Concatenates `w13 = [gate, up]` (`[E, 2 * I, H]`) and `w2 = down` (`[E, H, I]`).
    3. Pads `I` to a multiple of 128 (`aligned_inter = (I + 127) // 128 * 128`).
    4. Quantizes along the contracting dimension (`axis=-1`) using `quantize_tensor`.
    5. Transposes weights to K-major (`w13`: `[E, H, 2 * aligned_I]`,
       `w2`: `[E, aligned_I, H]`) and scales to 3D (`[E, num_blocks, N]`, float32).
    """
    expert_ids = sorted(expert_entries)
    num_experts = len(expert_ids)
    if expert_ids != list(range(num_experts)):
        raise ValueError(
            f"Expected contiguous expert IDs 0..{num_experts - 1}, "
            f"got {expert_ids[:5]}..{expert_ids[-5:]}"
        )

    w13_chunks: list[torch.Tensor] = []
    w13_scale_chunks: list[torch.Tensor] = []
    w2_chunks: list[torch.Tensor] = []
    w2_scale_chunks: list[torch.Tensor] = []

    for start in range(0, num_experts, expert_chunk_size):
        end = min(start + expert_chunk_size, num_experts)
        w13_fp32_list: list[torch.Tensor] = []
        w2_fp32_list: list[torch.Tensor] = []

        for eid in range(start, end):
            fields = expert_entries[eid]
            gate_loc = fields.get(("gate_proj", "weight")) or fields.get(
                ("w1", "weight")
            )
            up_loc = fields.get(("up_proj", "weight")) or fields.get(("w3", "weight"))
            down_loc = fields.get(("down_proj", "weight")) or fields.get(
                ("w2", "weight")
            )
            if gate_loc is None or up_loc is None or down_loc is None:
                raise ValueError(f"Missing gate/up/down weight for expert {eid}")

            gate_scale_loc = (
                fields.get(("gate_proj", "weight_scale_inv"))
                or fields.get(("gate_proj", "weight_scale"))
                or fields.get(("w1", "weight_scale_inv"))
                or fields.get(("w1", "weight_scale"))
            )
            up_scale_loc = (
                fields.get(("up_proj", "weight_scale_inv"))
                or fields.get(("up_proj", "weight_scale"))
                or fields.get(("w3", "weight_scale_inv"))
                or fields.get(("w3", "weight_scale"))
            )
            down_scale_loc = (
                fields.get(("down_proj", "weight_scale_inv"))
                or fields.get(("down_proj", "weight_scale"))
                or fields.get(("w2", "weight_scale_inv"))
                or fields.get(("w2", "weight_scale"))
            )

            gate_w = read_safetensors_tensor(gate_loc)
            up_w = read_safetensors_tensor(up_loc)
            down_w = read_safetensors_tensor(down_loc)

            gate_s = (
                read_safetensors_tensor(gate_scale_loc)
                if gate_scale_loc is not None
                else None
            )
            up_s = (
                read_safetensors_tensor(up_scale_loc)
                if up_scale_loc is not None
                else None
            )
            down_s = (
                read_safetensors_tensor(down_scale_loc)
                if down_scale_loc is not None
                else None
            )

            gate_fp32 = _dequant_expert_2d(gate_w, gate_s, weight_block_size)
            up_fp32 = _dequant_expert_2d(up_w, up_s, weight_block_size)
            down_fp32 = _dequant_expert_2d(down_w, down_s, weight_block_size)

            w13_fp32_list.append(torch.cat([gate_fp32, up_fp32], dim=0))
            w2_fp32_list.append(down_fp32)

        w13_fp32 = torch.stack(w13_fp32_list, dim=0)
        w2_fp32 = torch.stack(w2_fp32_list, dim=0)
        del w13_fp32_list, w2_fp32_list

        if activation == "swigluoai":
            w1 = w13_fp32[:, ::2, :]
            w3 = w13_fp32[:, 1::2, :]
            w13_fp32 = torch.cat([w1, w3], dim=1)

        half = w13_fp32.shape[1] // 2
        aligned_half = (half + 127) // 128 * 128
        if aligned_half != half:
            pad_w13 = w13_fp32.new_zeros(
                w13_fp32.shape[0], aligned_half - half, w13_fp32.shape[2]
            )
            w13_fp32 = torch.cat(
                [w13_fp32[:, :half, :], pad_w13, w13_fp32[:, half:, :], pad_w13],
                dim=1,
            ).contiguous()

        inter = w2_fp32.shape[2]
        aligned_inter = (inter + 127) // 128 * 128
        if aligned_inter != inter:
            pad_w2 = w2_fp32.new_zeros(
                w2_fp32.shape[0], w2_fp32.shape[1], aligned_inter - inter
            )
            w2_fp32 = torch.cat([w2_fp32, pad_w2], dim=2).contiguous()

        w13_q, w13_s = quantize_block_tensor(
            w13_fp32,
            quant_dtype=requant_dtype,
            axis=-1,
            block_size=requant_block_size,
        )
        w2_q, w2_s = quantize_block_tensor(
            w2_fp32,
            quant_dtype=requant_dtype,
            axis=-1,
            block_size=requant_block_size,
        )
        del w13_fp32, w2_fp32

        # Transpose to K-major: [E, out, in] -> [E, in, out]
        # Scale [E, out, num_blocks] -> [E, num_blocks, out] in float32
        w13_chunks.append(w13_q.transpose(1, 2).contiguous())
        w13_scale_chunks.append(w13_s.transpose(1, 2).contiguous().to(torch.float32))
        w2_chunks.append(w2_q.transpose(1, 2).contiguous())
        w2_scale_chunks.append(w2_s.transpose(1, 2).contiguous().to(torch.float32))

    w13_all = torch.cat(w13_chunks, dim=0) if len(w13_chunks) > 1 else w13_chunks[0]
    w13_s_all = (
        torch.cat(w13_scale_chunks, dim=0)
        if len(w13_scale_chunks) > 1
        else w13_scale_chunks[0]
    )
    w2_all = torch.cat(w2_chunks, dim=0) if len(w2_chunks) > 1 else w2_chunks[0]
    w2_s_all = (
        torch.cat(w2_scale_chunks, dim=0)
        if len(w2_scale_chunks) > 1
        else w2_scale_chunks[0]
    )
    return w13_all, w13_s_all, w2_all, w2_s_all


def merge_ple_shards(
    shard_locs: dict[int, TensorLocation],
    *,
    pad_divisor: int = 128,
) -> torch.Tensor:
    """Merge `ngram_embedding.shard_0.weight .. shard_{S-1}.weight` into one
    contiguous 2D tensor padded along dim 0 to a multiple of `pad_divisor`.
    """
    shard_ids = sorted(shard_locs)
    num_shards = len(shard_ids)
    if shard_ids != list(range(num_shards)):
        raise ValueError(
            f"Expected contiguous PLE shard indices 0..{num_shards - 1}, "
            f"got {shard_ids[:5]}..{shard_ids[-5:]}"
        )

    first_loc = shard_locs[0]
    head_dim = first_loc.shape[1]
    dtype = first_loc.torch_dtype
    org_rows = sum(loc.shape[0] for loc in shard_locs.values())
    padded_rows = (
        math.ceil(org_rows / pad_divisor) * pad_divisor if pad_divisor > 1 else org_rows
    )

    # Allocate the full table in uint8 / int16 carrier so zero-padding on F8_E4M3 works
    # cleanly without kernel ops.
    elem_size = torch.empty((), dtype=dtype, device="cpu").element_size()
    merged_u8 = torch.zeros(
        (padded_rows, head_dim * elem_size), dtype=torch.uint8, device="cpu"
    )
    row_cursor = 0
    for sid in shard_ids:
        shard_t = read_safetensors_tensor(shard_locs[sid])
        rows = shard_t.shape[0]
        merged_u8[row_cursor : row_cursor + rows].copy_(
            shard_t.view(torch.uint8).reshape(rows, head_dim * elem_size)
        )
        row_cursor += rows
    return merged_u8.view(dtype).reshape(padded_rows, head_dim)


def _extract_quant_block_size(config_data: dict[str, Any]) -> tuple[int, int] | None:
    """Extract `weight_block_size` from `config.json` if FP8 block-quantized."""
    q_cfg = config_data.get("quantization_config")
    if not isinstance(q_cfg, dict):
        text_cfg = config_data.get("text_config")
        if isinstance(text_cfg, dict):
            q_cfg = text_cfg.get("quantization_config")
    if not isinstance(q_cfg, dict):
        return None
    block_size = q_cfg.get("weight_block_size")
    if isinstance(block_size, list) and len(block_size) == 2:
        return int(block_size[0]), int(block_size[1])
    return None


def _extract_ple_divisor(config_data: dict[str, Any]) -> int:
    """Extract `make_ngram_vocab_size_divisible_by` from `config.json`."""
    text_cfg = config_data.get("text_config", config_data)
    if isinstance(text_cfg, dict):
        return int(text_cfg.get("make_ngram_vocab_size_divisible_by", 128))
    return 128


def _convert_and_write_moe_layer(
    output_dir: Path,
    canon_prefix: str,
    expert_entries: dict[int, dict[tuple[str, str], TensorLocation]],
    *,
    weight_block_size: tuple[int, int] | None,
    requant_dtype: torch.dtype,
    requant_block_size: int | None,
) -> tuple[str, tuple[int, ...], tuple[int, ...], float]:
    """Convert and write one MoE layer's `_tp1` expert arrays."""
    t_layer = time.perf_counter()
    w13, w13_s, w2, w2_s = convert_moe_layer_experts(
        expert_entries,
        weight_block_size=weight_block_size,
        requant_dtype=requant_dtype,
        requant_block_size=requant_block_size,
    )
    w13_shape = tuple(w13.shape)
    w2_shape = tuple(w2.shape)
    write_raw_array(output_dir, "experts", f"{canon_prefix}.w13_weight", w13, "tp1")
    write_raw_array(
        output_dir,
        "experts",
        f"{canon_prefix}.w13_weight_scale_inv",
        w13_s,
        "tp1",
    )
    write_raw_array(output_dir, "experts", f"{canon_prefix}.w2_weight", w2, "tp1")
    write_raw_array(
        output_dir,
        "experts",
        f"{canon_prefix}.w2_weight_scale_inv",
        w2_s,
        "tp1",
    )
    del w13, w13_s, w2, w2_s
    return canon_prefix, w13_shape, w2_shape, time.perf_counter() - t_layer


def run_convert(
    input_dir: Path,
    output_dir: Path,
    *,
    requant_dtype_name: str = "float8_e4m3fn",
    requant_block_size: int | None = None,
    layers_filter: set[int] | None = None,
    include_mtp: bool = True,
    export_misc_arrays: bool = False,
    dense_shard_max_bytes: int = 4 << 30,
    num_workers: int = 8,
) -> dict[str, Any]:
    """Run Stage 1 (`convert`): convert HF safetensors to preprocessed raw arrays."""
    t_start = time.perf_counter()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    config_path = input_dir / "config.json"
    config_data: dict[str, Any] = {}
    if config_path.is_file():
        with open(config_path, encoding="utf-8") as f:
            config_data = json.load(f)

    weight_block_size = _extract_quant_block_size(config_data)
    ple_divisor = _extract_ple_divisor(config_data)

    dtype_map = {
        "float8_e4m3fn": torch.float8_e4m3fn,
        "int8": torch.int8,
    }
    if requant_dtype_name not in dtype_map:
        raise ValueError(f"Unsupported --requant-dtype {requant_dtype_name!r}")
    requant_dtype = dtype_map[requant_dtype_name]

    output_dir.mkdir(parents=True, exist_ok=True)
    # Copy non-weight metadata files (config.json, tokenizer*, etc.)
    for item in input_dir.iterdir():
        if (
            item.is_file()
            and not item.name.endswith(".safetensors")
            and item.name != "model.safetensors.index.json"
        ):
            shutil.copy2(item, output_dir / item.name)

    locations = scan_safetensors_headers(input_dir)
    print(
        f"[convert] Scanned {len(locations)} tensors in {input_dir}",
        file=sys.stderr,
    )

    # Group tensors into:
    # 1. MoE routed experts by canonical layer prefix
    # 2. PLE n-gram shards & aux tensors by canonical PLE prefix
    # 3. Remaining dense tensors
    moe_groups: dict[str, dict[int, dict[tuple[str, str], TensorLocation]]] = {}
    ple_shard_groups: dict[str, dict[int, TensorLocation]] = {}
    ple_aux_groups: dict[str, dict[str, TensorLocation]] = {}
    dense_locations: dict[str, TensorLocation] = {}

    for name, loc in locations.items():
        if not include_mtp and name.startswith("mtp."):
            continue

        m_exp = _EXPERT_TENSOR_RE.match(name)
        if m_exp is not None:
            raw_prefix = m_exp.group("prefix")
            canon_prefix = canonicalize_prefix(raw_prefix)
            # Check optional layer filter
            layer_match = re.search(r"\.layers\.(\d+)\.", canon_prefix)
            if (
                layers_filter is not None
                and layer_match is not None
                and int(layer_match.group(1)) not in layers_filter
            ):
                continue
            eid = int(m_exp.group("expert_id"))
            proj = m_exp.group("proj")
            field = m_exp.group("field")
            moe_groups.setdefault(canon_prefix, {}).setdefault(eid, {})[
                (proj, field)
            ] = loc
            continue

        m_ple_shard = _PLE_SHARD_RE.match(name)
        if m_ple_shard is not None:
            canon_prefix = canonicalize_prefix(m_ple_shard.group("prefix"))
            sid = int(m_ple_shard.group("shard_id"))
            ple_shard_groups.setdefault(canon_prefix, {})[sid] = loc
            continue

        m_ple_aux = _PLE_AUX_RE.match(name)
        if m_ple_aux is not None:
            canon_prefix = canonicalize_prefix(m_ple_aux.group("prefix"))
            field = m_ple_aux.group("field")
            ple_aux_groups.setdefault(canon_prefix, {})[field] = loc
            # Also keep PLE aux in dense_locations so standard AutoWeightsLoader
            # or Qwen4ExpNGramEmbedding.load_weights sees them if needed, EXCEPT
            # when the shard weights themselves are moved to misc/array_*.
            continue

        # Skip PLE hash statistics that vLLM drops anyway
        if "hashstats_" in name or "token_lookup" in name:
            continue

        dense_locations[name] = loc

    # 1. Convert MoE routed experts layer by layer (optionally in parallel)
    sorted_moe_items = sorted(moe_groups.items())
    converted_moe_prefixes = [k for k, _ in sorted_moe_items]
    if sorted_moe_items:
        if num_workers <= 1 or len(sorted_moe_items) == 1:
            for idx, (canon_prefix, expert_entries) in enumerate(sorted_moe_items):
                _, w13_shape, w2_shape, dt = _convert_and_write_moe_layer(
                    output_dir,
                    canon_prefix,
                    expert_entries,
                    weight_block_size=weight_block_size,
                    requant_dtype=requant_dtype,
                    requant_block_size=requant_block_size,
                )
                print(
                    f"[convert] ({idx + 1}/{len(sorted_moe_items)}) {canon_prefix}: "
                    f"w13={w13_shape} w2={w2_shape} in {dt:.2f}s",
                    file=sys.stderr,
                )
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as pool:
                futures = [
                    pool.submit(
                        _convert_and_write_moe_layer,
                        output_dir,
                        canon_prefix,
                        expert_entries,
                        weight_block_size=weight_block_size,
                        requant_dtype=requant_dtype,
                        requant_block_size=requant_block_size,
                    )
                    for canon_prefix, expert_entries in sorted_moe_items
                ]
                for done_count, fut in enumerate(
                    concurrent.futures.as_completed(futures), start=1
                ):
                    canon_prefix, w13_shape, w2_shape, dt = fut.result()
                    print(
                        f"[convert] ({done_count}/{len(sorted_moe_items)}) "
                        f"{canon_prefix}: w13={w13_shape} w2={w2_shape} "
                        f"in {dt:.2f}s",
                        file=sys.stderr,
                    )

    # 2. Merge and write PLE n-gram tables
    converted_ple_prefixes: list[str] = []
    for canon_prefix, shard_locs in sorted(ple_shard_groups.items()):
        t_ple = time.perf_counter()
        merged_table = merge_ple_shards(shard_locs, pad_divisor=ple_divisor)
        write_raw_array(
            output_dir,
            "misc",
            f"{canon_prefix}.ngram_embedding.weight",
            merged_table,
            "tp1",
        )
        aux_fields = ple_aux_groups.get(canon_prefix, {})
        for field_name, aux_loc in sorted(aux_fields.items()):
            aux_t = read_safetensors_tensor(aux_loc)
            write_raw_array(
                output_dir,
                "misc",
                f"{canon_prefix}.{field_name}",
                aux_t,
                "tp0",
            )
        converted_ple_prefixes.append(canon_prefix)
        print(
            f"[convert] PLE {canon_prefix}: merged {len(shard_locs)} shards -> "
            f"{tuple(merged_table.shape)} ({merged_table.dtype}) "
            f"in {time.perf_counter() - t_ple:.2f}s",
            file=sys.stderr,
        )
        del merged_table

    # 3. Consolidate remaining dense weights into dense-*-of-*.safetensors
    weight_map: dict[str, str] = {}
    total_dense_bytes = 0
    if dense_locations:
        shards: list[dict[str, torch.Tensor]] = []
        current_shard: dict[str, torch.Tensor] = {}
        current_bytes = 0

        for name in sorted(dense_locations):
            loc = dense_locations[name]
            t = read_safetensors_tensor(loc)
            nbytes = t.numel() * t.element_size()
            total_dense_bytes += nbytes
            if export_misc_arrays:
                canon_name = canonicalize_prefix(name)
                if t.dtype in TORCH_TO_DTYPE_TOKEN:
                    write_raw_array(output_dir, "misc", canon_name, t, "tp0")
            if current_shard and current_bytes + nbytes > dense_shard_max_bytes:
                shards.append(current_shard)
                current_shard = {}
                current_bytes = 0
            current_shard[name] = t
            current_bytes += nbytes

        if current_shard:
            shards.append(current_shard)

        num_shards = len(shards)
        for i, shard_dict in enumerate(shards, start=1):
            shard_fname = f"dense-{i:05d}-of-{num_shards:05d}.safetensors"
            write_safetensors_file(output_dir / shard_fname, shard_dict)
            for tensor_name in shard_dict:
                weight_map[tensor_name] = shard_fname

        index_out = {
            "metadata": {"total_size": total_dense_bytes},
            "weight_map": weight_map,
        }
        with open(
            output_dir / "model.safetensors.index.json", "w", encoding="utf-8"
        ) as f:
            json.dump(index_out, f, indent=2, sort_keys=True)

    manifest = {
        "format_version": 1,
        "model_type": config_data.get("model_type", ""),
        "requant_dtype": requant_dtype_name,
        "requant_block_size": requant_block_size,
        "moe_expert_prefixes": converted_moe_prefixes,
        "ple_prefixes": converted_ple_prefixes,
        "num_dense_tensors": len(weight_map),
        "repacked_targets": ["tp1"],
    }
    with open(output_dir / MANIFEST_FILENAME, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    elapsed = time.perf_counter() - t_start
    print(
        f"[convert] Completed in {elapsed:.2f}s: "
        f"{len(converted_moe_prefixes)} MoE layers, "
        f"{len(converted_ple_prefixes)} PLE tables, "
        f"{len(weight_map)} dense tensors -> {output_dir}",
        file=sys.stderr,
    )
    return manifest


def _find_tp1_leaf(array_dir: Path) -> tuple[Path, RawLeafName] | None:
    for entry in sorted(array_dir.iterdir()):
        if not entry.is_file():
            continue
        parsed = parse_raw_leaf_name(entry.name)
        if parsed is not None and parsed.ep_size is None and parsed.tp_size == 1:
            return entry, parsed
    return None


def _repack_and_verify_expert_dir(
    array_dir: Path,
    *,
    ep: int,
    tp: int,
    pad_multiple: int = 1,
    verify: bool = True,
    dry_run: bool = False,
) -> Path | None:
    """Repack one `experts/array_*` directory from `_tp1` into `_ep{ep}tp{tp}`."""
    found = _find_tp1_leaf(array_dir)
    if found is None:
        return None
    tp1_path, meta = found
    dir_name = array_dir.name
    is_w13 = ".w13_" in dir_name
    is_scale = dir_name.endswith("_scale_inv") or dir_name.endswith("_scale")

    split_mode = "paired_columns" if is_w13 else "rows"
    eff_pad = 1 if is_scale else pad_multiple

    if dry_run:
        print(
            f"[repack-tp] (dry-run) {dir_name}: {tp1_path.name} -> "
            f"ep{ep}tp{tp} ({split_mode}, pad={eff_pad})",
            file=sys.stderr,
        )
        return None

    # Read full _tp1 tensor
    src_tensor = resolve_and_read_shard(
        array_dir, ep_size=1, ep_rank=0, tp_size=1, tp_rank=0
    )
    if split_mode == "paired_columns":
        repacked, logical_dims = repack_expert_paired_cols_for_ep_tp(
            src_tensor, ep=ep, tp=tp, pad_multiple=eff_pad
        )
    else:
        repacked, logical_dims = repack_expert_rows_for_ep_tp(
            src_tensor, ep=ep, tp=tp, pad_multiple=eff_pad
        )

    array_name = dir_name[len("array_") :]
    base_dir = array_dir.parent.parent
    out_path = write_raw_array(
        base_dir,
        "experts",
        array_name,
        repacked,
        f"ep{ep}tp{tp}",
        dtype_token=meta.dtype_token,
        logical_dims=logical_dims,
    )

    if verify:
        for ep_r in range(ep):
            for tp_r in range(tp):
                tile_read = resolve_and_read_shard(
                    array_dir,
                    ep_size=ep,
                    ep_rank=ep_r,
                    tp_size=tp,
                    tp_rank=tp_r,
                    split_mode=split_mode,
                )
                expected_tile = repacked[ep_r, tp_r]
                if not torch.equal(
                    tile_read.view(torch.uint8), expected_tile.view(torch.uint8)
                ):
                    raise ValueError(
                        f"Verification failed for {out_path} at "
                        f"ep_rank={ep_r}, tp_rank={tp_r}"
                    )
    return out_path


def run_repack_tp(
    checkpoint_dir: Path,
    *,
    ep: int = 1,
    tp: int = 1,
    pad_to_128: bool = True,
    verify: bool = False,
    dry_run: bool = False,
) -> list[Path]:
    """Run Stage 2 (`repack-tp`): repack `_tp1` expert arrays into `_ep<E>tp<T>`."""
    t_start = time.perf_counter()
    experts_root = checkpoint_dir / "experts"
    if not experts_root.is_dir():
        raise FileNotFoundError(f"No experts/ directory found in {checkpoint_dir}")

    pad_multiple = 128 if pad_to_128 else 1
    written: list[Path] = []
    array_dirs = sorted(
        d for d in experts_root.iterdir() if d.is_dir() and d.name.startswith("array_")
    )
    for array_dir in array_dirs:
        out = _repack_and_verify_expert_dir(
            array_dir,
            ep=ep,
            tp=tp,
            pad_multiple=pad_multiple,
            verify=verify,
            dry_run=dry_run,
        )
        if out is not None:
            written.append(out)

    manifest_path = checkpoint_dir / MANIFEST_FILENAME
    if not dry_run and manifest_path.is_file():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        targets = list(manifest.get("repacked_targets", ["tp1"]))
        spec = f"ep{ep}tp{tp}"
        if spec not in targets:
            targets.append(spec)
        manifest["repacked_targets"] = targets
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)

    elapsed = time.perf_counter() - t_start
    print(
        f"[repack-tp] Completed {len(written)} expert arrays for ep={ep}, tp={tp} "
        f"(verify={verify}) in {elapsed:.2f}s",
        file=sys.stderr,
    )
    return written


def _parse_layers_arg(layers_str: str | None) -> set[int] | None:
    if not layers_str:
        return None
    result: set[int] = set()
    for part in layers_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            result.update(range(int(lo_s), int(hi_s) + 1))
        else:
            result.add(int(part))
    return result


_GCS_TOKEN_LOCK = threading.Lock()
_GCS_TOKEN_CACHE: dict[str, Any] = {"token": "", "exp": 0.0}


def _get_gce_access_token() -> str:
    """Fetch a GCE Workload Identity access token from the metadata server."""
    with _GCS_TOKEN_LOCK:
        now = time.time()
        if _GCS_TOKEN_CACHE["token"] and now < float(_GCS_TOKEN_CACHE["exp"]) - 120.0:
            return str(_GCS_TOKEN_CACHE["token"])
        last_err: Exception | None = None
        for attempt in range(6):
            try:
                req = urllib.request.Request(
                    "http://metadata.google.internal/computeMetadata/v1/"
                    "instance/service-accounts/default/token",
                    headers={"Metadata-Flavor": "Google"},
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                _GCS_TOKEN_CACHE["token"] = str(data["access_token"])
                _GCS_TOKEN_CACHE["exp"] = now + float(data.get("expires_in", 1800))
                return str(_GCS_TOKEN_CACHE["token"])
            except Exception as exc:
                last_err = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Failed to fetch GCE metadata token: {last_err}")


def upload_directory_to_gcs(
    local_dir: Path,
    gcs_uri: str,
    *,
    num_workers: int = 24,
) -> int:
    """Upload all files under ``local_dir`` to ``gs://<bucket>/<prefix>``.

    Uses GCE Workload Identity + ``curl`` (or ``/tmp/cdk_gcs.py cp`` when
    available) so it works inside lean TPU server containers that do not ship
    ``gcloud`` or ``gsutil``.
    """
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got {gcs_uri!r}")
    rest = gcs_uri[5:].rstrip("/")
    bucket, _, base_prefix = rest.partition("/")
    if not bucket:
        raise ValueError(f"Invalid GCS bucket in {gcs_uri!r}")

    local_dir = Path(local_dir).resolve()
    files = sorted(p for p in local_dir.rglob("*") if p.is_file())
    total_bytes = sum(p.stat().st_size for p in files)
    t0 = time.perf_counter()
    print(
        f"[upload-gcs] Uploading {len(files)} files "
        f"({total_bytes / (1024**3):.2f} GiB) "
        f"from {local_dir} -> {gcs_uri} with {num_workers} workers...",
        file=sys.stderr,
    )

    def _upload_one(path: Path) -> int:
        rel = path.relative_to(local_dir).as_posix()
        obj_name = f"{base_prefix}/{rel}" if base_prefix else rel
        enc_name = urllib.parse.quote(obj_name, safe="")
        url = (
            f"https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o"
            f"?uploadType=media&name={enc_name}"
        )
        for attempt in range(5):
            token = _get_gce_access_token()
            res = subprocess.run(
                [
                    "curl",
                    "-sS",
                    "-o",
                    "/dev/null",
                    "-w",
                    "%{http_code}",
                    "-X",
                    "POST",
                    "-H",
                    f"Authorization: Bearer {token}",
                    "-H",
                    "Content-Type: application/octet-stream",
                    "--data-binary",
                    f"@{path}",
                    url,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            http_code = (res.stdout or "").strip()
            if http_code in ("200", "201"):
                return path.stat().st_size
            if http_code == "401":
                with _GCS_TOKEN_LOCK:
                    _GCS_TOKEN_CACHE["exp"] = 0.0
            time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(
            f"Failed uploading {path} -> gs://{bucket}/{obj_name} "
            f"(HTTP {res.stdout!r}, stderr={res.stderr!r})"
        )

    uploaded_bytes = 0
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, num_workers)) as pool:
        futures = [pool.submit(_upload_one, p) for p in files]
        for fut in concurrent.futures.as_completed(futures):
            uploaded_bytes += fut.result()
            completed += 1
            if completed % 50 == 0 or completed == len(files):
                dt = max(1e-6, time.perf_counter() - t0)
                gib = uploaded_bytes / (1024**3)
                print(
                    f"[upload-gcs] {completed}/{len(files)} files "
                    f"({gib:.2f}/{total_bytes / (1024**3):.2f} GiB, "
                    f"{gib / dt:.2f} GiB/s)",
                    file=sys.stderr,
                )

    dt = max(1e-6, time.perf_counter() - t0)
    print(
        f"[upload-gcs] Completed upload to {gcs_uri} in {dt:.1f}s "
        f"({total_bytes / (1024**3) / dt:.2f} GiB/s)",
        file=sys.stderr,
    )
    return len(files)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m vllm_torchtpu.tools.preprocess_checkpoint",
        description=(
            "Preprocess HuggingFace MoE/PLE checkpoints into contiguous TPU raw arrays."
        ),
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # --- convert ---
    p_conv = subparsers.add_parser(
        "convert",
        help=(
            "Convert HF Safetensors into _tp1 raw arrays and consolidated "
            "dense safetensors."
        ),
    )
    p_conv.add_argument(
        "--input-dir",
        "-i",
        type=Path,
        required=True,
        help="Input HuggingFace checkpoint directory.",
    )
    p_conv.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        required=True,
        help="Output preprocessed checkpoint directory.",
    )
    p_conv.add_argument(
        "--requant-dtype",
        default="float8_e4m3fn",
        choices=["float8_e4m3fn", "int8"],
        help=(
            "Target quantization dtype for MoE expert weights (default: float8_e4m3fn)."
        ),
    )
    p_conv.add_argument(
        "--requant-block-size",
        type=int,
        default=None,
        help=(
            "Contracting-dimension block size for MoE requantization "
            "(default: None = per-channel)."
        ),
    )
    p_conv.add_argument(
        "--layers",
        type=str,
        default=None,
        help=(
            "Optional comma/range list of decoder layer indices to convert "
            "(e.g. '0-3,7')."
        ),
    )
    p_conv.add_argument(
        "--skip-mtp",
        action="store_true",
        help="Skip MTP (Multi-Token Prediction) draft layers.",
    )
    p_conv.add_argument(
        "--export-misc-arrays",
        action="store_true",
        help="Also export non-expert dense weights as misc/array_*/*_tp0 raw files.",
    )
    p_conv.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel worker threads for MoE layer conversion (default: 8).",
    )
    p_conv.add_argument(
        "--gcs-output-uri",
        type=str,
        default=None,
        help="Optional gs:// bucket/prefix URI to upload the preprocessed checkpoint.",
    )
    p_conv.add_argument(
        "--gcs-workers",
        type=int,
        default=24,
        help="Number of parallel workers for GCS upload (default: 24).",
    )

    # --- repack-tp ---
    p_repack = subparsers.add_parser(
        "repack-tp",
        help="Repack _tp1 expert arrays into contiguous _ep<E>tp<T> rank tiles.",
    )
    p_repack.add_argument(
        "--checkpoint-dir",
        "-d",
        type=Path,
        required=True,
        help="Preprocessed checkpoint directory containing experts/array_*/*_tp1.",
    )
    p_repack.add_argument(
        "--ep",
        type=int,
        default=1,
        help="Expert parallelism degree E (default: 1).",
    )
    p_repack.add_argument(
        "--tp",
        type=int,
        default=1,
        help="MoE tensor parallelism degree T within each EP group (default: 1).",
    )
    p_repack.add_argument(
        "--no-pad-128",
        action="store_true",
        help="Disable 128-alignment padding of intermediate shards when tp > 1.",
    )
    p_repack.add_argument(
        "--verify",
        action="store_true",
        help="Verify bit-exactness of every repacked (ep_rank, tp_rank) tile.",
    )
    p_repack.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned repack operations without writing files.",
    )

    # --- pipeline ---
    p_pipe = subparsers.add_parser(
        "pipeline",
        help="Run convert followed by repack-tp in a single command.",
    )
    p_pipe.add_argument(
        "--input-dir",
        "-i",
        type=Path,
        required=True,
        help="Input HuggingFace checkpoint directory.",
    )
    p_pipe.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        required=True,
        help="Output preprocessed checkpoint directory.",
    )
    p_pipe.add_argument(
        "--ep",
        type=int,
        default=1,
        help="Expert parallelism degree E (default: 1).",
    )
    p_pipe.add_argument(
        "--tp",
        type=int,
        default=1,
        help="MoE tensor parallelism degree T within each EP group (default: 1).",
    )
    p_pipe.add_argument(
        "--requant-dtype",
        default="float8_e4m3fn",
        choices=["float8_e4m3fn", "int8"],
        help="Target quantization dtype for MoE expert weights.",
    )
    p_pipe.add_argument(
        "--requant-block-size",
        type=int,
        default=None,
        help="Contracting-dimension block size for MoE requantization.",
    )
    p_pipe.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Optional comma/range list of decoder layer indices to convert.",
    )
    p_pipe.add_argument(
        "--skip-mtp",
        action="store_true",
        help="Skip MTP draft layers.",
    )
    p_pipe.add_argument(
        "--export-misc-arrays",
        action="store_true",
        help="Also export non-expert dense weights as misc/array_*/*_tp0 raw files.",
    )
    p_pipe.add_argument(
        "--verify",
        action="store_true",
        help="Verify bit-exactness of repacked tiles.",
    )
    p_pipe.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel worker threads for MoE layer conversion (default: 8).",
    )
    p_pipe.add_argument(
        "--gcs-output-uri",
        type=str,
        default=None,
        help="Optional gs:// bucket/prefix URI to upload the preprocessed checkpoint.",
    )
    p_pipe.add_argument(
        "--gcs-workers",
        type=int,
        default=24,
        help="Number of parallel workers for GCS upload (default: 24).",
    )

    # --- upload-gcs ---
    p_up = subparsers.add_parser(
        "upload-gcs",
        help="Upload a local preprocessed checkpoint directory to a gs:// URI.",
    )
    p_up.add_argument(
        "--checkpoint-dir",
        "-d",
        type=Path,
        required=True,
        help="Local preprocessed checkpoint directory to upload.",
    )
    p_up.add_argument(
        "--gcs-output-uri",
        "-g",
        type=str,
        required=True,
        help="Destination gs:// bucket/prefix URI.",
    )
    p_up.add_argument(
        "--gcs-workers",
        type=int,
        default=24,
        help="Number of parallel workers for GCS upload (default: 24).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.subcommand == "convert":
        run_convert(
            args.input_dir,
            args.output_dir,
            requant_dtype_name=args.requant_dtype,
            requant_block_size=args.requant_block_size,
            layers_filter=_parse_layers_arg(args.layers),
            include_mtp=not args.skip_mtp,
            export_misc_arrays=args.export_misc_arrays,
            num_workers=args.num_workers,
        )
        if args.gcs_output_uri:
            upload_directory_to_gcs(
                args.output_dir,
                args.gcs_output_uri,
                num_workers=args.gcs_workers,
            )
        return 0

    if args.subcommand == "repack-tp":
        run_repack_tp(
            args.checkpoint_dir,
            ep=args.ep,
            tp=args.tp,
            pad_to_128=not args.no_pad_128,
            verify=args.verify,
            dry_run=args.dry_run,
        )
        return 0

    if args.subcommand == "pipeline":
        run_convert(
            args.input_dir,
            args.output_dir,
            requant_dtype_name=args.requant_dtype,
            requant_block_size=args.requant_block_size,
            layers_filter=_parse_layers_arg(args.layers),
            include_mtp=not args.skip_mtp,
            export_misc_arrays=args.export_misc_arrays,
            num_workers=args.num_workers,
        )
        if args.ep > 1 or args.tp > 1:
            run_repack_tp(
                args.output_dir,
                ep=args.ep,
                tp=args.tp,
                verify=args.verify,
            )
        if args.gcs_output_uri:
            upload_directory_to_gcs(
                args.output_dir,
                args.gcs_output_uri,
                num_workers=args.gcs_workers,
            )
        return 0

    if args.subcommand == "upload-gcs":
        upload_directory_to_gcs(
            args.checkpoint_dir,
            args.gcs_output_uri,
            num_workers=args.gcs_workers,
        )
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
