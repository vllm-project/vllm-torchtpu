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
"""Raw array directory format, TP/EP repacking, and contiguous shard reader.

Checkpoints preprocessed by ``vllm_torchtpu.tools.preprocess_checkpoint``
store each tensor in its own directory under ``experts/`` or ``misc/``:

    <base_dir>/
      vllm_preprocess_manifest.json
      experts/
        array_<name>/
          <dtype>_<d0>_<d1>_..._<tp_spec>[_<tpu_format>_ls<sublanes>]
      misc/
        array_<name>/
          <dtype>_<d0>_<d1>_..._<tp_spec>[_<tpu_format>_ls<sublanes>]

Where ``<tp_spec>`` is:
- ``tp0``: replicated across ranks (single tile).
- ``tp1``: canonical un-repacked layout (contiguous along dim 0, so EP ranks
  with ``tp_size == 1`` can read their slice directly).
- ``tp<T>``: repacked into ``T`` contiguous tensor-parallel tiles.
- ``ep<E>tp<T>``: repacked into ``E * T`` contiguous ``(ep_rank, tp_rank)``
  tiles in ``ep_rank * T + tp_rank`` major order.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

MANIFEST_FILENAME = "vllm_preprocess_manifest.json"

RepackSplitMode = Literal["rows", "columns", "paired_columns"]

# Mapping between filename dtype tokens and PyTorch dtypes.
DTYPE_TOKEN_TO_TORCH: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "f32": torch.float32,
    "f16": torch.float16,
    "f8e4m3fn": torch.float8_e4m3fn,
    "s8": torch.int8,
    "u8": torch.uint8,
    "s32": torch.int32,
    "i64": torch.int64,
    # Sub-byte packed formats: s4 packs 8 int4 values into one int32;
    # f4e2m1fn packs 2 fp4 values into one uint8/float4_e2m1fn_x2.
    "s4": torch.int32,
    "f4e2m1fn": getattr(torch, "float4_e2m1fn_x2", torch.uint8),
}

TORCH_TO_DTYPE_TOKEN: dict[torch.dtype, str] = {
    torch.bfloat16: "bf16",
    torch.float32: "f32",
    torch.float16: "f16",
    torch.float8_e4m3fn: "f8e4m3fn",
    torch.int8: "s8",
    torch.uint8: "u8",
    torch.int32: "s32",
    torch.int64: "i64",
}
if hasattr(torch, "float4_e2m1fn_x2"):
    TORCH_TO_DTYPE_TOKEN[torch.float4_e2m1fn_x2] = "f4e2m1fn"

_EP_TP_RE = re.compile(r"^ep(\d+)tp(\d+)$")
_TP_RE = re.compile(r"^tp(\d+)$")
_LS_RE = re.compile(r"^ls(\d+)$")


@dataclass(frozen=True)
class RawLeafName:
    """Parsed metadata from a raw array leaf filename."""

    dtype_token: str
    dims: tuple[int, ...]
    ep_size: int | None
    tp_size: int
    tpu_format: str | None = None
    linear_sublanes: int | None = None

    @property
    def is_replicated(self) -> bool:
        return self.ep_size is None and self.tp_size == 0

    @property
    def is_linearized(self) -> bool:
        return self.tpu_format is not None

    @property
    def tp_spec(self) -> str:
        if self.ep_size is not None:
            return f"ep{self.ep_size}tp{self.tp_size}"
        return f"tp{self.tp_size}"

    @property
    def torch_dtype(self) -> torch.dtype:
        return DTYPE_TOKEN_TO_TORCH[self.dtype_token]

    @property
    def expected_bytes(self) -> int:
        return dtype_byte_size(self.dtype_token, self.dims)


def dtype_byte_size(dtype_token: str, dims: tuple[int, ...] | list[int]) -> int:
    """Compute the raw byte size for ``dims`` of ``dtype_token``.

     Note that ``dims`` in the leaf filename always records the logical element
    dimensions (or stored dimensions for packed ``s4`` where ``int32`` holds 8
    nibbles if stored as ``s32``, whereas ``s4``/``f4e2m1fn`` store 2 elements
    per byte).
    """
    if dtype_token not in DTYPE_TOKEN_TO_TORCH:
        raise ValueError(f"Unsupported dtype token {dtype_token!r}")
    numel = math.prod(dims)
    if dtype_token in ("s4", "f4e2m1fn"):
        if numel % 2 != 0:
            raise ValueError(
                f"Sub-byte dtype {dtype_token!r} requires an even element count, "
                f"got dims={tuple(dims)}"
            )
        return numel // 2
    torch_dtype = DTYPE_TOKEN_TO_TORCH[dtype_token]
    elem_size = torch.empty((), dtype=torch_dtype, device="cpu").element_size()
    return numel * elem_size


def parse_raw_leaf_name(filename: str) -> RawLeafName | None:
    """Parse a raw array leaf filename, or return None if it does not match.

    Supported patterns:
      ``<dtype>_<d0>_<d1>_..._tp<T>``
      ``<dtype>_<d0>_<d1>_..._ep<E>tp<T>``
      ``<dtype>_<d0>_<d1>_..._<tp_spec>_<tpu_format>_ls<S>``
    """
    if filename.startswith(".") or filename.endswith(".tmp"):
        return None
    parts = filename.split("_")
    if len(parts) < 3:
        return None
    dtype_token = parts[0]
    if dtype_token not in DTYPE_TOKEN_TO_TORCH:
        return None

    tpu_format: str | None = None
    linear_sublanes: int | None = None
    ls_match = _LS_RE.match(parts[-1])
    if ls_match is not None and len(parts) >= 5:
        linear_sublanes = int(ls_match.group(1))
        tpu_format = parts[-2]
        core_parts = parts[1:-2]
    else:
        core_parts = parts[1:]

    if len(core_parts) < 2:
        return None

    spec_part = core_parts[-1]
    dim_parts = core_parts[:-1]
    if not all(p.isdigit() for p in dim_parts):
        return None
    dims = tuple(int(p) for p in dim_parts)
    if any(d <= 0 for d in dims):
        return None

    ep_tp_match = _EP_TP_RE.match(spec_part)
    if ep_tp_match is not None:
        ep_size = int(ep_tp_match.group(1))
        tp_size = int(ep_tp_match.group(2))
        if ep_size <= 0 or tp_size <= 0:
            return None
        return RawLeafName(
            dtype_token=dtype_token,
            dims=dims,
            ep_size=ep_size,
            tp_size=tp_size,
            tpu_format=tpu_format,
            linear_sublanes=linear_sublanes,
        )

    tp_match = _TP_RE.match(spec_part)
    if tp_match is not None:
        tp_size = int(tp_match.group(1))
        return RawLeafName(
            dtype_token=dtype_token,
            dims=dims,
            ep_size=None,
            tp_size=tp_size,
            tpu_format=tpu_format,
            linear_sublanes=linear_sublanes,
        )

    return None


def format_raw_leaf_name(
    dtype_token: str,
    dims: tuple[int, ...] | list[int],
    tp_spec: str,
    *,
    tpu_format: str | None = None,
    linear_sublanes: int | None = None,
) -> str:
    """Format a raw array leaf filename."""
    if dtype_token not in DTYPE_TOKEN_TO_TORCH:
        raise ValueError(f"Unsupported dtype token {dtype_token!r}")
    dims_str = "_".join(str(int(d)) for d in dims)
    base = f"{dtype_token}_{dims_str}_{tp_spec}"
    if tpu_format is not None:
        if linear_sublanes is None:
            raise ValueError("linear_sublanes is required when tpu_format is set")
        base = f"{base}_{tpu_format}_ls{linear_sublanes}"
    return base


def write_raw_array(
    base_dir: str | Path,
    category: Literal["experts", "misc"],
    array_name: str,
    tensor: torch.Tensor,
    tp_spec: str = "tp1",
    *,
    dtype_token: str | None = None,
    logical_dims: tuple[int, ...] | list[int] | None = None,
) -> Path:
    """Write ``tensor`` as raw little-endian bytes into
    ``<base_dir>/<category>/array_<array_name>/``.

    Uses an atomic write-and-rename so readers never observe a half-written file.
    """
    if dtype_token is None:
        if tensor.dtype not in TORCH_TO_DTYPE_TOKEN:
            raise ValueError(f"Unsupported tensor dtype {tensor.dtype}")
        dtype_token = TORCH_TO_DTYPE_TOKEN[tensor.dtype]
    dims = tuple(tensor.shape) if logical_dims is None else tuple(logical_dims)
    leaf_name = format_raw_leaf_name(dtype_token, dims, tp_spec)
    array_dir = Path(base_dir) / category / f"array_{array_name}"
    array_dir.mkdir(parents=True, exist_ok=True)
    out_path = array_dir / leaf_name
    tmp_path = array_dir / f".{leaf_name}.tmp.{os.getpid()}"

    cpu_contig = tensor.detach().cpu().contiguous()
    byte_view = cpu_contig.view(torch.uint8).numpy()
    expected = dtype_byte_size(dtype_token, dims)
    if byte_view.nbytes != expected:
        raise ValueError(
            f"Byte size mismatch for {array_name}: tensor has {byte_view.nbytes} "
            f"bytes, but {leaf_name} expects {expected} bytes"
        )
    byte_view.tofile(tmp_path)
    os.replace(tmp_path, out_path)
    return out_path


def dequantize_block_tensor(
    tensor_q: torch.Tensor,
    scale: torch.Tensor,
    axis: int | Sequence[int] = -1,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize a block-scaled quantized tensor (matches
    ``vllm_torchtpu.layers.core.quantization.dequantize_tensor`` without
    requiring ``torch_tpu`` to be imported).
    """
    if isinstance(axis, int):
        axis = [axis]
    orig_shape = tensor_q.shape
    if tensor_q.ndim == scale.ndim:
        blocked_shape: list[int] = []
        norm_axes = [(a + tensor_q.ndim) % tensor_q.ndim for a in axis]
        for i, dim in enumerate(orig_shape):
            if i in norm_axes:
                num_blocks = scale.shape[i]
                block_size = dim // num_blocks
                blocked_shape.extend([num_blocks, block_size])
            else:
                blocked_shape.append(dim)
        axis_normalized = sorted(norm_axes)
        expanded_axis = [1 + n + a for n, a in enumerate(axis_normalized)]
        tensor_q = tensor_q.reshape(blocked_shape)
        for ax in expanded_axis:
            scale = scale.unsqueeze(ax)
    tensor = (tensor_q.to(torch.float32) * scale).to(out_dtype)
    return tensor.reshape(orig_shape)


def quantize_block_tensor(
    tensor: torch.Tensor,
    quant_dtype: torch.dtype,
    axis: int = -1,
    block_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a float tensor along ``axis`` with optional ``block_size``
    (matches ``vllm_torchtpu.layers.core.quantization.quantize_tensor``
    bit-for-bit without requiring ``torch_tpu`` to be imported).
    """
    block_size = block_size or tensor.shape[axis]
    if tensor.shape[axis] % block_size != 0:
        raise ValueError(
            "Quantization expects the requested axis to be divisible by "
            f"block_size, got tensor.shape={tuple(tensor.shape)}, axis={axis}, "
            f"block_size={block_size}."
        )
    is_float = torch.empty((), dtype=quant_dtype).is_floating_point()
    dtype_info = torch.finfo(quant_dtype) if is_float else torch.iinfo(quant_dtype)
    axis = axis % tensor.ndim
    moved = tensor.movedim(axis, -1)
    num_blocks = moved.shape[-1] // block_size
    blocked = moved.reshape(*moved.shape[:-1], num_blocks, block_size)

    abs_max = blocked.abs().amax(dim=-1, keepdim=True)
    dtype_max_recip = torch.tensor(
        1.0 / float(dtype_info.max), dtype=torch.float32, device=tensor.device
    )
    scale = abs_max * dtype_max_recip
    scale_inv = torch.where(scale == 0, torch.zeros_like(scale), 1.0 / scale)

    blocked_q = torch.clamp(
        blocked * scale_inv, min=float(dtype_info.min), max=float(dtype_info.max)
    )
    if not is_float:
        blocked_q = torch.round(blocked_q)
    tensor_q = blocked_q.reshape_as(moved).to(quant_dtype).movedim(-1, axis)
    scale = scale.to(torch.float32).squeeze(-1).movedim(-1, axis)
    return tensor_q, scale


def repack_columns_for_tp(tensor: torch.Tensor, tp: int) -> torch.Tensor:
    """Repack a 2D ``[K, N]`` tensor into ``[tp, K, N // tp]`` contiguous tiles.

    For column-split dense weights or K-major 2D matrices where each TP rank
    owns a slice of the second dimension ``[0:K, r*(N//tp) : (r+1)*(N//tp)]``.
    """
    if tensor.ndim != 2:
        raise ValueError(
            f"repack_columns_for_tp expects a 2D tensor, got {tensor.shape}"
        )
    k_dim, n_dim = tensor.shape
    if tp <= 0 or n_dim % tp != 0:
        raise ValueError(f"Cannot split columns N={n_dim} evenly across tp={tp}")
    shard_n = n_dim // tp
    # [K, tp, shard_n] -> [tp, K, shard_n]
    return tensor.reshape(k_dim, tp, shard_n).permute(1, 0, 2).contiguous()


def repack_expert_rows_for_ep_tp(
    tensor: torch.Tensor,
    ep: int,
    tp: int,
    *,
    pad_multiple: int = 1,
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """Repack a 3D ``[E, R, C]`` expert tensor (such as K-major ``w2_weight`` or
    ``w2_weight_scale_inv``) into ``[ep, tp, E // ep, R_shard_padded, C]``.

    Returns ``(repacked_tensor, logical_3d_dims)`` where ``logical_3d_dims`` is
    ``(E, R_shard_padded * tp, C)`` (or ``(E, 1, C)`` when ``R == 1`` is
    replicated across ``tp`` ranks, as with per-channel ``w2_weight_scale_inv``).
    """
    if tensor.ndim != 3:
        raise ValueError(
            f"repack_expert_rows_for_ep_tp expects a 3D tensor, got {tensor.shape}"
        )
    num_experts, rows, cols = tensor.shape
    if ep <= 0 or num_experts % ep != 0:
        raise ValueError(f"Cannot split E={num_experts} experts evenly across ep={ep}")
    e_per_ep = num_experts // ep

    # Special case: per-channel w2_weight_scale_inv has R == 1 (1 scale block
    # along the contracting dimension I). When tp > 1, each TP rank still needs
    # a [E // ep, 1, H] scale tensor.
    if rows == 1 and tp > 1:
        reshaped = tensor.reshape(ep, 1, e_per_ep, 1, cols)
        repacked = reshaped.expand(ep, tp, e_per_ep, 1, cols).contiguous()
        return repacked, (num_experts, tp, cols)

    if tp <= 0 or rows % tp != 0:
        raise ValueError(f"Cannot split rows R={rows} evenly across tp={tp}")
    r_shard = rows // tp
    r_padded = (
        (r_shard + pad_multiple - 1) // pad_multiple * pad_multiple
        if pad_multiple > 1
        else r_shard
    )
    # [ep, E // ep, tp, r_shard, C] -> [ep, tp, E // ep, r_shard, C]
    tiles = (
        tensor.reshape(ep, e_per_ep, tp, r_shard, cols)
        .permute(0, 2, 1, 3, 4)
        .contiguous()
    )
    if r_padded != r_shard:
        pad = tiles.new_zeros(ep, tp, e_per_ep, r_padded - r_shard, cols)
        tiles = torch.cat([tiles, pad], dim=3).contiguous()
    return tiles, (num_experts, r_padded * tp, cols)


def repack_expert_paired_cols_for_ep_tp(
    tensor: torch.Tensor,
    ep: int,
    tp: int,
    *,
    pad_multiple: int = 1,
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """Repack a 3D ``[E, R, 2 * I]`` fused gate+up expert tensor (K-major
    ``w13_weight`` or ``w13_weight_scale_inv``) into
    ``[ep, tp, E // ep, R, 2 * I_shard_padded]``.

    The last dimension holds ``[gate (I), up (I)]``. Under ``tp > 1``, each
    TP rank owns ``[gate_shard (I // tp), up_shard (I // tp)]`` (each padded
    to ``pad_multiple`` when ``pad_multiple > 1``).

    Returns ``(repacked_tensor, logical_3d_dims)`` where ``logical_3d_dims`` is
    ``(E, R, 2 * I_shard_padded * tp)``.
    """
    if tensor.ndim != 3:
        raise ValueError(
            "repack_expert_paired_cols_for_ep_tp expects a 3D tensor, "
            f"got {tensor.shape}"
        )
    num_experts, rows, two_inter = tensor.shape
    if ep <= 0 or num_experts % ep != 0:
        raise ValueError(f"Cannot split E={num_experts} experts evenly across ep={ep}")
    if two_inter % 2 != 0:
        raise ValueError(
            f"paired_columns requires an even last dimension, got {two_inter}"
        )
    inter = two_inter // 2
    if tp <= 0 or inter % tp != 0:
        raise ValueError(
            f"Cannot split intermediate half I={inter} evenly across tp={tp}"
        )
    e_per_ep = num_experts // ep
    i_shard = inter // tp
    i_padded = (
        (i_shard + pad_multiple - 1) // pad_multiple * pad_multiple
        if pad_multiple > 1
        else i_shard
    )
    # Reshape [E, R, 2, tp, i_shard] -> [ep, E // ep, R, 2, tp, i_shard]
    # Permute to [ep, tp, E // ep, R, 2, i_shard]
    tiles = (
        tensor.reshape(ep, e_per_ep, rows, 2, tp, i_shard)
        .permute(0, 4, 1, 2, 3, 5)
        .contiguous()
    )
    if i_padded != i_shard:
        pad = tiles.new_zeros(ep, tp, e_per_ep, rows, 2, i_padded - i_shard)
        tiles = torch.cat([tiles, pad], dim=5).contiguous()
    tiles = tiles.reshape(ep, tp, e_per_ep, rows, 2 * i_padded).contiguous()
    return tiles, (num_experts, rows, 2 * i_padded * tp)


def _pread_into_tensor(
    path: Path,
    byte_offset: int,
    byte_length: int,
    *,
    num_threads: int = 4,
) -> torch.Tensor:
    """Read ``[byte_offset, byte_offset + byte_length)`` from ``path`` into a
    1D ``uint8`` CPU tensor using parallel ``os.pread`` calls.
    """
    buf = torch.empty(byte_length, dtype=torch.uint8, device="cpu")
    if byte_length == 0:
        return buf
    mv = memoryview(buf.numpy())
    fd = os.open(path, os.O_RDONLY)
    try:
        # For transfers <= 16 MiB or single thread, do a single read loop.
        min_chunk = 16 << 20
        if num_threads <= 1 or byte_length <= min_chunk:
            pos = 0
            while pos < byte_length:
                chunk = os.pread(fd, byte_length - pos, byte_offset + pos)
                if not chunk:
                    raise EOFError(
                        f"Unexpected EOF reading {path} at offset {byte_offset + pos}"
                    )
                mv[pos : pos + len(chunk)] = chunk
                pos += len(chunk)
            return buf

        chunk_size = max(min_chunk, (byte_length + num_threads - 1) // num_threads)
        spans = [
            (off, min(chunk_size, byte_length - off))
            for off in range(0, byte_length, chunk_size)
        ]

        def _read_span(span: tuple[int, int]) -> None:
            rel_off, length = span
            pos = 0
            while pos < length:
                data = os.pread(fd, length - pos, byte_offset + rel_off + pos)
                if not data:
                    raise EOFError(
                        f"Unexpected EOF reading {path} at offset "
                        f"{byte_offset + rel_off + pos}"
                    )
                mv[rel_off + pos : rel_off + pos + len(data)] = data
                pos += len(data)

        with ThreadPoolExecutor(max_workers=min(num_threads, len(spans))) as pool:
            list(pool.map(_read_span, spans))
    finally:
        os.close(fd)
    return buf


def find_matching_leaf(
    array_dir: str | Path,
    *,
    ep_size: int = 1,
    tp_size: int = 1,
) -> tuple[Path, RawLeafName] | None:
    """Find the best non-linearized raw leaf file in ``array_dir`` for
    ``(ep_size, tp_size)``.

    Priority order:
    1. Exact ``_ep{ep_size}tp{tp_size}`` match.
    2. ``_tp{tp_size}`` when ``ep_size == 1``.
    3. ``_tp1`` when ``tp_size == 1`` (since ``_tp1`` is contiguous along dim 0,
       any ``ep_size >= 1`` with ``tp_size == 1`` can slice dim 0 contiguously;
       similarly a row-sharded 2D table with ``tp_size >= 1`` can slice dim 0
       from ``_tp1``).
    4. ``_tp0`` (replicated across all ranks).
    """
    d = Path(array_dir)
    if not d.is_dir():
        return None

    candidates: list[tuple[Path, RawLeafName]] = []
    for entry in sorted(d.iterdir()):
        if not entry.is_file():
            continue
        parsed = parse_raw_leaf_name(entry.name)
        if parsed is not None and not parsed.is_linearized:
            candidates.append((entry, parsed))

    if not candidates:
        return None

    # 1. Exact ep<E>tp<T>
    for path, meta in candidates:
        if meta.ep_size == ep_size and meta.tp_size == tp_size:
            return path, meta

    # 2. Exact tp<T> when ep_size == 1
    if ep_size == 1:
        for path, meta in candidates:
            if meta.ep_size is None and meta.tp_size == tp_size:
                return path, meta

    # 3. _tp1 fallback (contiguous along dim 0 for EP-only or row-sharded TP)
    for path, meta in candidates:
        if meta.ep_size is None and meta.tp_size == 1:
            return path, meta

    # 4. _tp0 (replicated)
    for path, meta in candidates:
        if meta.is_replicated:
            return path, meta

    return None


def resolve_and_read_shard(
    array_dir: str | Path,
    *,
    ep_size: int = 1,
    ep_rank: int = 0,
    tp_size: int = 1,
    tp_rank: int = 0,
    split_mode: RepackSplitMode | None = None,
    row_sharded_tp: bool = False,
    num_threads: int = 4,
) -> torch.Tensor:
    """Read this rank's contiguous shard from ``array_dir``.

    Args:
        array_dir: Path to ``array_<name>`` directory.
        ep_size: Expert-parallel world size.
        ep_rank: Expert-parallel rank in ``[0, ep_size)``.
        tp_size: Tensor-parallel world size (within each EP group for experts).
        tp_rank: Tensor-parallel rank in ``[0, tp_size)``.
        split_mode: For 3D expert arrays repacked with ``_ep<E>tp<T>``,
            specifies whether ``tp_size`` split ``"rows"`` (dim 1, e.g. ``w2``)
            or ``"paired_columns"`` (dim 2, e.g. ``w13``). Inferred from the
            directory name when ``None``.
        row_sharded_tp: When ``True`` and reading a 2D ``_tp1`` array (such as
            the PLE n-gram embedding table), slices dim 0 into ``tp_size``
            contiguous row shards using ``tp_rank``.
        num_threads: Number of parallel ``os.pread`` threads for large shards.

    Returns:
        A contiguous CPU ``torch.Tensor`` holding this rank's shard.
    """
    matched = find_matching_leaf(array_dir, ep_size=ep_size, tp_size=tp_size)
    if matched is None:
        raise FileNotFoundError(
            f"No compatible raw leaf file found in {array_dir} for "
            f"ep_size={ep_size}, tp_size={tp_size}"
        )
    leaf_path, meta = matched
    file_size = leaf_path.stat().st_size
    if file_size != meta.expected_bytes:
        raise ValueError(
            f"Corrupt raw array file {leaf_path}: file_size={file_size}, "
            f"expected={meta.expected_bytes}"
        )

    if split_mode is None:
        dir_name = Path(array_dir).name
        if ".w13_" in dir_name:
            split_mode = "paired_columns"
        elif ".w2_" in dir_name:
            split_mode = "rows"

    dims = meta.dims
    if meta.is_replicated:
        num_tiles = 1
        tile_idx = 0
        shard_shape = dims
    elif meta.ep_size is not None:
        # Repacked _ep<E>tp<T> file: E * T contiguous tiles in (ep_rank, tp_rank) order.
        if meta.ep_size != ep_size or meta.tp_size != tp_size:
            raise ValueError(
                f"Leaf {leaf_path.name} has ep={meta.ep_size}, tp={meta.tp_size}, "
                f"cannot serve ep_size={ep_size}, tp_size={tp_size}"
            )
        num_tiles = ep_size * tp_size
        tile_idx = ep_rank * tp_size + tp_rank
        if len(dims) != 3:
            raise ValueError(
                f"Expected 3D dims on ep/tp leaf {leaf_path.name}, got {dims}"
            )
        e_dim, r_dim, c_dim = dims
        if split_mode == "paired_columns":
            shard_shape = (e_dim // ep_size, r_dim, c_dim // tp_size)
        elif split_mode == "rows":
            shard_shape = (e_dim // ep_size, r_dim // tp_size, c_dim)
        else:
            raise ValueError(
                f"split_mode must be 'rows' or 'paired_columns' for {leaf_path}"
            )
    elif meta.tp_size == 1:
        # Canonical _tp1 file: contiguous along dim 0!
        if ep_size > 1 and tp_size == 1:
            if dims[0] % ep_size != 0:
                raise ValueError(
                    f"Dim 0 ({dims[0]}) of {leaf_path} is not divisible by "
                    f"ep_size={ep_size}"
                )
            num_tiles = ep_size
            tile_idx = ep_rank
            shard_shape = (dims[0] // ep_size, *dims[1:])
        elif row_sharded_tp and tp_size > 1:
            if dims[0] % tp_size != 0:
                raise ValueError(
                    f"Dim 0 ({dims[0]}) of {leaf_path} is not divisible by "
                    f"tp_size={tp_size}"
                )
            num_tiles = tp_size
            tile_idx = tp_rank
            shard_shape = (dims[0] // tp_size, *dims[1:])
        elif ep_size == 1 and tp_size == 1:
            num_tiles = 1
            tile_idx = 0
            shard_shape = dims
        else:
            raise ValueError(
                f"Cannot read contiguous shard for ep_size={ep_size}, "
                f"tp_size={tp_size} from un-repacked {leaf_path.name}; "
                f"run 'repack-tp --ep {ep_size} --tp {tp_size}' first."
            )
    else:
        # _tp<T> file: T contiguous tiles.
        if meta.tp_size != tp_size or ep_size != 1:
            raise ValueError(
                f"Leaf {leaf_path.name} has tp={meta.tp_size}, cannot serve "
                f"ep_size={ep_size}, tp_size={tp_size}"
            )
        num_tiles = tp_size
        tile_idx = tp_rank
        if len(dims) == 2 and not row_sharded_tp:
            shard_shape = (dims[0], dims[1] // tp_size)
        elif len(dims) == 2 and row_sharded_tp:
            shard_shape = (dims[0] // tp_size, dims[1])
        elif len(dims) == 3 and split_mode == "paired_columns":
            shard_shape = (dims[0], dims[1], dims[2] // tp_size)
        elif len(dims) == 3 and split_mode == "rows":
            shard_shape = (dims[0], dims[1] // tp_size, dims[2])
        else:
            shard_shape = (dims[0] // tp_size, *dims[1:])

    if file_size % num_tiles != 0:
        raise ValueError(
            f"File size {file_size} of {leaf_path} is not divisible by "
            f"num_tiles={num_tiles}"
        )
    tile_bytes = file_size // num_tiles
    byte_offset = tile_idx * tile_bytes
    raw_u8 = _pread_into_tensor(
        leaf_path, byte_offset, tile_bytes, num_threads=num_threads
    )

    if meta.dtype_token == "s4":
        # 8 int4 values packed per int32 along the last dimension.
        if shard_shape[-1] % 8 != 0:
            raise ValueError(
                f"s4 last dimension must be divisible by 8, got {shard_shape}"
            )
        packed_shape = (*shard_shape[:-1], shard_shape[-1] // 8)
        return raw_u8.view(torch.int32).reshape(packed_shape)
    if meta.dtype_token == "f4e2m1fn":
        if shard_shape[-1] % 2 != 0:
            raise ValueError(
                f"f4e2m1fn last dimension must be divisible by 2, got {shard_shape}"
            )
        packed_shape = (*shard_shape[:-1], shard_shape[-1] // 2)
        return raw_u8.view(meta.torch_dtype).reshape(packed_shape)

    return raw_u8.view(meta.torch_dtype).reshape(shard_shape)
