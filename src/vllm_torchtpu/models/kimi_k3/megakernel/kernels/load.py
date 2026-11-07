# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

# Source provenance: inferact/k3-megakernel@30c9ba966b
# src/vllm_torchtpu/megakernel/kimi_k3/kernels/load.py
"""Read Kimi K3 checkpoint assets in the TP32 megakernel layout.

This module reads the tokenizer and Hugging Face checkpoint tensors and
describes the kernel's per-rank weight layout: ``Checkpoint`` slices each
family for a rank, which the pre-shard tool
(``vllm_torchtpu.models.kimi_k3.megakernel.tools preshard``) writes out and
``rank_loader`` streams onto the TPU.
"""

from __future__ import annotations

import ast
import base64
import importlib
import json
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, TypedDict

import ml_dtypes
import numpy as np

from . import Config

if TYPE_CHECKING:
    from collections.abc import Iterable

BF16 = np.dtype(ml_dtypes.bfloat16)
PREFIX = "language_model.model."
ALL = slice(None)  # a whole axis
_MATRIX_NDIM = 2
_E8M0_NAN_SCALE = 255
_LATENT_ACTIVE_RANKS = 28
_MIN_STACK_LAYERS = 2
_FULL_STACK_LAYERS = 93


class _SafetensorMeta(TypedDict):
    """Per-tensor safetensors shard header entry."""

    data_offsets: list[int]
    dtype: str
    shape: list[int]


def load_tokenizer(directory: Path | str) -> object:
    """Read the checkpoint's tiktoken ranks, regex, and reserved-token mapping.

    Parse only the literal regex list from its tokenizer source; loading a
    tokenizer never executes checkpoint Python code or imports Transformers.
    """
    tiktoken = importlib.import_module("tiktoken")

    checkpoint_dir = Path(directory)
    config = json.loads((checkpoint_dir / "tokenizer_config.json").read_text())
    source = ast.parse((checkpoint_dir / "tokenization_kimi.py").read_text())
    cls = next(
        node
        for node in source.body
        if isinstance(node, ast.ClassDef) and node.name == "TikTokenTokenizer"
    )
    assignment = next(
        node
        for node in cls.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "pat_str"
            for target in node.targets
        )
    )
    value = assignment.value
    if not (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and isinstance(value.func.value, ast.Constant)
        and value.func.value.value == "|"
        and value.func.attr == "join"
        and len(value.args) == 1
    ):
        msg = "Unsupported checkpoint tokenizer pattern declaration"
        raise ValueError(msg)
    pattern = "|".join(ast.literal_eval(value.args[0]))
    ranks = {
        base64.b64decode(bpe_bytes): int(rank)
        for bpe_bytes, rank in (
            line.split()
            for line in (checkpoint_dir / "tiktoken.model").read_bytes().splitlines()
        )
    }
    added = config["added_tokens_decoder"]
    special = {
        added.get(str(idx), {}).get("content", f"<|reserved_token_{idx}|>"): idx
        for idx in range(len(ranks), len(ranks) + 256)
    }
    return tiktoken.Encoding(
        name="kimi-k3-checkpoint",
        pat_str=pattern,
        mergeable_ranks=ranks,
        special_tokens=special,
    )


def pack_expert_bytes(value: np.ndarray) -> np.ndarray:
    """Convert checkpoint `[N, K/2]` bytes into `[K/8, N]` uint32 nibble groups."""
    if value.dtype != np.uint8 or value.ndim != _MATRIX_NDIM or value.shape[1] % 4:
        msg = "Expected whole uint32 groups of packed FP4 bytes"
        raise ValueError(msg)
    return np.ascontiguousarray(value).view("<u4").T.copy()


class Checkpoint:
    """Safetensors reader and TP32 slicer for Kimi K3 checkpoint shards."""

    # Coalesced streaming reads: a rank's expert tensors are grouped into
    # contiguous ranges (merging gaps up to COALESCE_GAP) and each range is
    # streamed on its own handle, ranges in parallel: seeks and handles shared
    # across threads are slow on network storage.
    COALESCE_GAP = 16 << 20
    CHUNK = 16 << 20
    READ_THREADS = 8
    _DTYPES: ClassVar[dict[str, np.dtype[np.generic]]] = {
        "U8": np.dtype(np.uint8),
        "BF16": BF16,
        "F32": np.dtype(np.float32),
        "F16": np.dtype(np.float16),
    }

    def __init__(self, path: Path | str) -> None:
        """Initialize the checkpoint reader and verify all safetensors shards."""
        self.path = Path(path)
        self.config = Config.from_checkpoint(path)
        if self.config != Config():
            msg = "This loader requires the full 93-layer/896-expert K3 contract"
            raise ValueError(msg)
        self.index: dict[str, str] = json.loads(
            (self.path / "model.safetensors.index.json").read_text()
        )["weight_map"]
        missing = [
            shard
            for shard in set(self.index.values())
            if not (self.path / shard).is_file()
        ]
        if missing:
            msg = f"Checkpoint is missing {len(missing)} tensor shards"
            raise FileNotFoundError(msg)

        self._layouts: dict[str, tuple[dict[str, _SafetensorMeta], int]] = {}

    def _layout(self, filename: str) -> tuple[dict[str, _SafetensorMeta], int]:
        """Return `(header dict, data start)` of one shard, cached."""
        if filename not in self._layouts:
            with (self.path / filename).open("rb") as handle:
                size = struct.unpack("<Q", handle.read(8))[0]
                self._layouts[filename] = (
                    json.loads(handle.read(size)),
                    8 + size,
                )
        return self._layouts[filename]

    def read_many(self, keys: Iterable[str]) -> dict[str, np.ndarray]:
        """Read tensors for `keys` as numpy arrays via coalesced shard reads.

        Returns `{key: array}` with the header dtype/shape applied. Used for
        the expert families.
        """
        by_file: dict[str, list[str]] = {}
        for key in keys:
            by_file.setdefault(self.index[key], []).append(key)
        out: dict[str, np.ndarray] = {}
        for filename, file_keys in by_file.items():
            header, base = self._layout(filename)
            spans = sorted(
                (
                    header[key]["data_offsets"][0],
                    header[key]["data_offsets"][1],
                    key,
                )
                for key in file_keys
            )
            ranges: list[tuple[int, int, list[str]]] = []
            for start, end, key in spans:
                if ranges and start - ranges[-1][1] <= self.COALESCE_GAP:
                    prev_start, prev_end, prev_keys = ranges[-1]
                    prev_keys.append(key)
                    ranges[-1] = (prev_start, max(prev_end, end), prev_keys)
                else:
                    ranges.append((start, end, [key]))
            path = self.path / filename

            def stream(
                bounds: tuple[int, int, list[str]],
                *,
                shard_path: Path = path,
                data_base: int = base,
            ) -> bytes:
                """Stream one coalesced byte span from `shard_path`."""
                start, end, _ = bounds
                fd = os.open(shard_path, os.O_RDONLY)
                try:
                    pieces = [
                        os.pread(
                            fd,
                            min(self.CHUNK, end - offset),
                            data_base + offset,
                        )
                        for offset in range(start, end, self.CHUNK)
                    ]
                finally:
                    os.close(fd)
                return b"".join(pieces)

            with ThreadPoolExecutor(min(self.READ_THREADS, len(ranges))) as pool:
                buffers = dict(enumerate(pool.map(stream, ranges)))
            for range_idx, (start, _end, range_keys) in enumerate(ranges):
                buffer = np.frombuffer(buffers[range_idx], np.uint8)
                for key in range_keys:
                    meta = header[key]
                    span_start, span_end = meta["data_offsets"]
                    out[key] = (
                        buffer[span_start - start : span_end - start]
                        .view(self._DTYPES[meta["dtype"]])
                        .reshape(meta["shape"])
                    )
        return out

    @staticmethod
    def _expert_parts(
        packed: np.ndarray,
        scale: np.ndarray,
        *,
        down: bool,
        part: int | None,
        parts: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Slice one expert tensor and its scale along the TP partition axis."""
        if part is None:
            return packed, scale
        if part not in range(parts) or 3072 % parts:
            msg = f"Expert TP part must be in 0..{parts - 1}"
            raise ValueError(msg)
        width = 3072 // parts
        if down:
            return (
                packed[:, part * width // 2 : (part + 1) * width // 2],
                scale[:, part * width // 32 : (part + 1) * width // 32],
            )
        return (
            packed[part * width : (part + 1) * width],
            scale[part * width : (part + 1) * width],
        )

    def experts(
        self,
        layer: int,
        first: int,
        count: int,
        *,
        part: int | None = None,
        parts: int = 4,
    ) -> tuple[
        list[np.ndarray],
        list[np.ndarray],
        list[np.ndarray],
        list[np.ndarray],
    ]:
        """Return `(gate_up, gate_up_scales, down, down_scales)` for `count` experts.

        Each element is a list over experts in kernel layout (gate/up packed
        `[K/8, N]` with E8M0 scales `[K/32, N]`, see `pack_expert_bytes`);
        all tensors of the layer's experts are fetched with `read_many`.
        """
        base = f"{PREFIX}layers.{layer}.block_sparse_moe.experts."
        keys = [
            f"{base}{first + exp_idx}.{name}.{kind}"
            for exp_idx in range(count)
            for name in ("w1", "w2", "w3")
            for kind in ("weight_packed", "weight_scale")
        ]
        tensors = self.read_many(keys)
        gate_up: list[np.ndarray] = []
        gate_up_scales: list[np.ndarray] = []
        downs: list[np.ndarray] = []
        down_scales: list[np.ndarray] = []
        for exp_idx in range(count):
            prefix = f"{base}{first + exp_idx}."
            weights: list[np.ndarray] = []
            scales: list[np.ndarray] = []
            for name in ("w1", "w3"):
                packed, scale = self._expert_parts(
                    tensors[prefix + name + ".weight_packed"],
                    tensors[prefix + name + ".weight_scale"],
                    down=False,
                    part=part,
                    parts=parts,
                )
                if scale.dtype != np.uint8 or np.any(
                    (scale == 0) | (scale == _E8M0_NAN_SCALE)
                ):
                    msg = f"{prefix}{name}: expected normal, finite E8M0 scales"
                    raise ValueError(msg)
                weights.append(pack_expert_bytes(packed))
                scales.append(scale.T.copy())
            gate_up.append(np.concatenate(weights, axis=1))
            gate_up_scales.append(np.concatenate(scales, axis=1))
            packed, scale = self._expert_parts(
                tensors[prefix + "w2.weight_packed"],
                tensors[prefix + "w2.weight_scale"],
                down=True,
                part=part,
                parts=parts,
            )
            if scale.dtype != np.uint8 or np.any(
                (scale == 0) | (scale == _E8M0_NAN_SCALE)
            ):
                msg = f"{prefix}w2: expected normal, finite E8M0 scales"
                raise ValueError(msg)
            downs.append(pack_expert_bytes(packed))
            down_scales.append(scale.T.copy())
        return gate_up, gate_up_scales, downs, down_scales

    def read(
        self,
        key: str,
        selection: slice | int | tuple[slice | int, ...] | None = None,
    ) -> np.ndarray:
        """Stream one tensor, or `tensor[selection]`, from its shard.

        A leading-axis slice (`(rows, ...full...)` or a bare slice) is a
        contiguous byte range and is streamed as such; any other selection
        reads the whole tensor and slices in memory. Both avoid small,
        non-contiguous reads, which can be slow on network-backed storage.
        """
        filename = self.index[key]
        header, base = self._layout(filename)
        meta = header[key]
        shape = list(meta["shape"])
        dtype = self._DTYPES[meta["dtype"]]
        start, end = meta["data_offsets"]
        rows = slice(None)
        rest: tuple[slice | int, ...] = ()
        if selection is not None:
            parts = selection if isinstance(selection, tuple) else (selection,)
            leading, tail = parts[0], parts[1:]
            if isinstance(leading, slice) and all(
                isinstance(item, slice) and item == slice(None) for item in tail
            ):
                rows, rest = leading, ()
            else:
                rest = parts
        row_start, row_stop, step = rows.indices(shape[0])
        if step != 1:
            msg = "Only unit-step leading slices are supported"
            raise ValueError(msg)
        row_bytes = (end - start) // shape[0] if shape else end - start
        offset = base + start + row_start * row_bytes
        length = (row_stop - row_start) * row_bytes if shape else end - start
        fd = os.open(self.path / filename, os.O_RDONLY)
        try:
            pieces = [
                os.pread(
                    fd,
                    min(self.CHUNK, length - chunk_start),
                    offset + chunk_start,
                )
                for chunk_start in range(0, length, self.CHUNK)
            ]
        finally:
            os.close(fd)
        value = np.frombuffer(b"".join(pieces), dtype).reshape(
            [row_stop - row_start, *shape[1:]] if shape else []
        )
        return value[rest] if rest else value

    def _read_layer_matrix(
        self,
        base: str,
        suffix: str,
        output: slice = ALL,
        contraction: slice = ALL,
    ) -> np.ndarray:
        """Read and transpose `base + suffix + '.weight'[output, contraction]`."""
        return self.read(base + suffix + ".weight", (output, contraction)).T.copy()

    def _read_folded_residual(
        self,
        base: str,
        stem: str,
        *,
        global_output: bool = False,
    ) -> np.ndarray:
        """Read and fold norm and projection weights for attention residuals."""
        prefix = PREFIX if global_output else base
        norm_weight = self.read(prefix + stem + "_norm.weight").astype(np.float32)
        proj_weight = self.read(prefix + stem + "_proj.weight").astype(np.float32)
        return norm_weight * proj_weight

    def _dense_global(
        self,
        name: str,
        base: str,
        rank: int,
    ) -> np.ndarray | None:
        """Slice global embeddings, output norm, or output attention residual."""
        if name in ("embedding", "lm_head"):
            key = (
                PREFIX + "embed_tokens.weight"
                if name == "embedding"
                else "language_model.lm_head.weight"
            )
            value = self.read(key, (slice(rank * 5120, (rank + 1) * 5120), slice(None)))
            return value if name == "embedding" else value.T.copy()
        if name == "final_norm":
            return self.read(PREFIX + "norm.weight")[None]
        if name == "output_res":
            return self._read_folded_residual(
                base, "output_attn_res", global_output=True
            )
        return None

    def _dense_layer_ffn(
        self,
        name: str,
        base: str,
        rank: int,
    ) -> np.ndarray | None:
        """Slice per-layer residual, norm, dense MLP, or shared-expert weights."""
        if name in ("attn_res", "mlp_res"):
            stem = "self_attention_res" if name == "attn_res" else "mlp_res"
            return self._read_folded_residual(base, stem)
        if name in ("attn_norm", "ffn_norm"):
            norm_name = (
                "input_layernorm" if name == "attn_norm" else "post_attention_layernorm"
            )
            return self.read(base + norm_name + ".weight")[None]
        if name in ("dense_gu", "shared_gu"):
            stem, real, padded = (
                ("mlp", 1056, 1152)
                if name == "dense_gu"
                else ("block_sparse_moe.shared_experts", 192, 256)
            )
            selection = slice(rank * real, (rank + 1) * real)
            return np.concatenate(
                [
                    np.pad(
                        self._read_layer_matrix(base, f"{stem}.{proj}_proj", selection),
                        ((0, 0), (0, padded - real)),
                    )
                    for proj in ("gate", "up")
                ],
                axis=1,
            )
        if name in ("dense_down", "shared_down"):
            stem, real, padded = (
                ("mlp", 1056, 1152)
                if name == "dense_down"
                else ("block_sparse_moe.shared_experts", 192, 256)
            )
            return np.pad(
                self._read_layer_matrix(
                    base,
                    f"{stem}.down_proj",
                    contraction=slice(rank * real, (rank + 1) * real),
                ),
                ((0, padded - real), (0, 0)),
            )
        return None

    def _dense_moe_or_latent(
        self,
        name: str,
        base: str,
        rank: int,
    ) -> np.ndarray | None:
        """Slice router or routed-expert latent projection weights."""
        if name == "router":
            return self._read_layer_matrix(base, "block_sparse_moe.gate")
        if name == "router_bias":
            return self.read(base + "block_sparse_moe.gate.e_score_correction_bias")[
                None
            ]
        if name == "latent_norm":
            return np.pad(
                self.read(base + "block_sparse_moe.routed_expert_norm.weight").astype(
                    np.float32
                )[None],
                ((0, 0), (0, 512)),
            )
        if name in ("latent_down", "latent_up"):
            shape = (7168, 128) if name == "latent_down" else (128, 7168)
            if rank >= _LATENT_ACTIVE_RANKS:
                return np.zeros(shape, BF16)
            selection = slice(rank * 128, (rank + 1) * 128)
            stem = "block_sparse_moe.routed_expert_"
            return (
                self._read_layer_matrix(base, stem + "down_proj", selection)
                if name == "latent_down"
                else self._read_layer_matrix(
                    base, stem + "up_proj", contraction=selection
                )
            )
        return None

    def _dense_kda_projections(
        self,
        name: str,
        base: str,
        rank: int,
    ) -> np.ndarray | None:
        """Slice KDA input/output matrix projections for `rank`."""
        rows = slice(rank * 384, (rank + 1) * 384)
        heads = slice(rank * 3, (rank + 1) * 3)
        if name == "k_gate":
            return self._read_layer_matrix(base, "self_attn.f_a_proj")
        if name == "k_projection":
            parts = [
                self._read_layer_matrix(base, f"self_attn.{proj}_proj", rows)
                for proj in ("q", "k", "v")
            ]
            parts.append(self._read_layer_matrix(base, "self_attn.g_proj", rows))
            parts.append(
                np.pad(
                    self._read_layer_matrix(base, "self_attn.b_proj", heads),
                    ((0, 0), (0, 125)),
                )
            )
            return np.concatenate(parts, axis=1)
        if name == "k_fb":
            return self._read_layer_matrix(base, "self_attn.f_b_proj", rows)
        if name in ("k_o", "m_o"):
            return self._read_layer_matrix(base, "self_attn.o_proj", contraction=rows)
        return None

    def _dense_kda_state_weights(
        self,
        name: str,
        base: str,
        rank: int,
    ) -> np.ndarray | None:
        """Slice KDA convolution, decay, bias, and output norm weights."""
        rows = slice(rank * 384, (rank + 1) * 384)
        heads = slice(rank * 3, (rank + 1) * 3)
        if name == "k_conv":
            return np.stack(
                [
                    self.read(
                        base + f"self_attn.{proj}_conv1d.weight",
                        (rows, slice(None), slice(None)),
                    )[:, 0].T.reshape(4, 3, 128)
                    for proj in ("q", "k", "v")
                ],
                axis=1,
            )
        if name == "k_a_log":
            return self.read(base + "self_attn.A_log", heads).reshape(3, 1)
        if name == "k_dt":
            return self.read(base + "self_attn.dt_bias", rows).reshape(3, 128)
        if name == "k_norm":
            return np.pad(
                self.read(base + "self_attn.o_norm.weight")[None],
                ((0, 7), (0, 0)),
            )
        return None

    def _dense_mla(
        self,
        name: str,
        base: str,
        rank: int,
    ) -> np.ndarray | None:
        """Slice MLA query/KV/gate projections and norms for `rank`."""
        rows = slice(rank * 384, (rank + 1) * 384)
        if name in ("m_qa", "m_gate"):
            suffix = "self_attn.q_a_proj" if name == "m_qa" else "self_attn.g_proj"
            out_slice = ALL if name == "m_qa" else rows
            return self._read_layer_matrix(base, suffix, out_slice)
        if name == "m_ka":
            return np.pad(
                self._read_layer_matrix(base, "self_attn.kv_a_proj_with_mqa"),
                ((0, 0), (0, 64)),
            )
        if name in ("m_qb", "m_kb"):
            if name == "m_qb":
                value = self._read_layer_matrix(
                    base,
                    "self_attn.q_b_proj",
                    slice(rank * 576, (rank + 1) * 576),
                )
                return np.pad(
                    value.reshape(1536, 3, 192), ((0, 0), (0, 0), (0, 64))
                ).reshape(1536, 768)
            return self._read_layer_matrix(
                base,
                "self_attn.kv_b_proj",
                slice(rank * 768, (rank + 1) * 768),
            )
        if name in ("m_qnorm", "m_knorm"):
            norm_prefix = "q_a" if name == "m_qnorm" else "kv_a"
            return self.read(base + f"self_attn.{norm_prefix}_layernorm.weight")[None]
        return None

    def dense(self, name: str, layer: int, rank: int) -> np.ndarray:
        """Return one layer of a parameter family with its global-rank slice."""
        base = f"{PREFIX}layers.{layer}."
        for resolver in (
            self._dense_global,
            self._dense_layer_ffn,
            self._dense_moe_or_latent,
            self._dense_kda_projections,
            self._dense_kda_state_weights,
            self._dense_mla,
        ):
            result = resolver(name, base, rank)
            if result is not None:
                return result
        raise KeyError(name)

    def dense_selected(self, name: str, layer: int, rank: int) -> np.ndarray:
        """Return one layer of a parameter family in the TP32 kernel layout.

        It differs from `dense` only for these families:

        * `k_o`/`m_o`: paired-rank output projections. Rank `r` holds the
          contraction channels of ranks `2p, 2p+1` (`p = r // 2`, 768 rows)
          for output half `r % 2` -> `[768, hidden / 2]`.
        * `latent_up`: output-sharded over 8 blocks of 1024 hidden columns
          (block `r % 8`; block 7 is zero padding past 7168), replicated
          across the four hosts, with the padded latent rows dropped ->
          `[3584, 1024]`.
        * `shared_gu`/`shared_down`: the rank's 192 shared channels without
          the 256 padding -> `[hidden, 384]` / `[192, hidden]`.
        """
        base = f"{PREFIX}layers.{layer}."
        hidden = self.config.dim
        if name in ("k_o", "m_o"):
            pair, half = divmod(rank, 2)
            value = self.read(
                base + "self_attn.o_proj.weight",
                (
                    slice(half * hidden // 2, (half + 1) * hidden // 2),
                    slice(pair * 768, (pair + 1) * 768),
                ),
            )
            return value.T.copy()
        if name == "latent_up":
            block = rank % 8
            if (block + 1) * 1024 > hidden:
                return np.zeros((self.config.latent, 1024), BF16)
            value = self.read(
                base + "block_sparse_moe.routed_expert_up_proj.weight",
                (slice(block * 1024, (block + 1) * 1024), slice(None)),
            )
            return value.T.copy()
        if name == "shared_gu":
            rows = slice(rank * 192, (rank + 1) * 192)
            return np.concatenate(
                [
                    self.read(
                        base + f"block_sparse_moe.shared_experts.{proj}_proj.weight",
                        (rows, slice(None)),
                    ).T
                    for proj in ("gate", "up")
                ],
                axis=1,
            )
        if name == "shared_down":
            return self.read(
                base + "block_sparse_moe.shared_experts.down_proj.weight",
                (slice(None), slice(rank * 192, (rank + 1) * 192)),
            ).T.copy()
        return self.dense(name, layer, rank)


def selected_local_shapes(
    layers: int = 93,
    hidden: int = 7168,
    *,
    vocab: int = 163840,
) -> dict[str, tuple[int, ...]]:
    """Return per-rank shapes of the TP32 kernel layout.

    TP4/EP8 packed routed experts, output-sharded host-local `latent_up`,
    compact shared experts, split KDA gate projection, and paired attention
    output projections.
    """
    if not _MIN_STACK_LAYERS <= layers <= _FULL_STACK_LAYERS:
        msg = "Kimi stack supports a prefix of 2..93 layers"
        raise ValueError(msg)
    if vocab % 32:
        msg = "Vocabulary must divide into 32 shards"
        raise ValueError(msg)
    mla_count = max(1, layers // 4 + (layers == _FULL_STACK_LAYERS))
    kda_count = layers - (layers // 4 + (layers == _FULL_STACK_LAYERS))
    return {
        "router": (layers - 1, hidden, 896),
        "router_bias": (layers - 1, 1, 896),
        "latent_down": (layers - 1, hidden, 128),
        "latent_up": (layers - 1, 3584, 1024),
        "latent_norm": (layers - 1, 1, 4096),
        "shared_gu": (layers - 1, hidden, 384),
        "shared_down": (layers - 1, 192, hidden),
        "expert_gu": (layers - 1, 112, 448, 1536),
        "expert_gus": (layers - 1, 112, 112, 1536),
        "expert_down": (layers - 1, 112, 96, 3584),
        "expert_ds": (layers - 1, 112, 24, 3584),
        "k_gate": (kda_count, hidden, 128),
        "k_projection": (kda_count, hidden, 1664),
        "k_fb": (kda_count, 128, 384),
        "k_o": (kda_count, 768, hidden // 2),
        "k_conv": (kda_count, 4, 3, 3, 128),
        "k_a_log": (kda_count, 3, 1),
        "k_dt": (kda_count, 3, 128),
        "k_norm": (kda_count, 8, 128),
        "m_qa": (mla_count, hidden, 1536),
        "m_ka": (mla_count, hidden, 640),
        "m_qb": (mla_count, 1536, 768),
        "m_kb": (mla_count, 512, 768),
        "m_gate": (mla_count, hidden, 384),
        "m_o": (mla_count, 768, hidden // 2),
        "m_qnorm": (mla_count, 1, 1536),
        "m_knorm": (mla_count, 1, 512),
        "attn_res": (layers, 1, hidden),
        "mlp_res": (layers, 1, hidden),
        "attn_norm": (layers, 1, hidden),
        "ffn_norm": (layers, 1, hidden),
        "embedding": (vocab // 32, hidden),
        "lm_head": (hidden, vocab // 32),
        "dense_gu": (hidden, 2304),
        "dense_down": (1152, hidden),
        "output_res": (1, hidden),
        "final_norm": (1, hidden),
    }


def _family_layers(
    config: Config,
    name: str,
    layers: int,
) -> list[int] | None:
    """Return checkpoint layer IDs along a family's leading axis (`None` if global)."""
    kda = {
        "k_gate",
        "k_projection",
        "k_fb",
        "k_o",
        "k_conv",
        "k_a_log",
        "k_dt",
        "k_norm",
    }
    mla = {
        "m_qa",
        "m_ka",
        "m_qb",
        "m_kb",
        "m_gate",
        "m_o",
        "m_qnorm",
        "m_knorm",
    }
    moe = {
        "router",
        "router_bias",
        "latent_down",
        "latent_up",
        "latent_norm",
        "shared_gu",
        "shared_down",
        "expert_gu",
        "expert_gus",
        "expert_down",
        "expert_ds",
    }
    norms = {"attn_res", "mlp_res", "attn_norm", "ffn_norm"}
    if name in kda:
        return [idx for idx in range(layers) if idx not in config.full_attention]
    if name in mla:
        return [idx for idx in range(layers) if idx in config.full_attention] or [3]
    if name in moe:
        return list(range(1, layers))
    if name in norms:
        return list(range(layers))
    return None
