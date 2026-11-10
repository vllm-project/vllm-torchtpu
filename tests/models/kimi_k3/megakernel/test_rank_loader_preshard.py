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

"""Unit tests for Kimi-K3 preshard writer and RankDir checkpoint loader."""

from __future__ import annotations

import argparse
import dataclasses
import json
from typing import TYPE_CHECKING, Any

import ml_dtypes
import numpy as np
import pytest

from vllm_torchtpu.models.kimi_k3.megakernel.kernels import (
    dspark,
    load,
    rank_loader,
)
from vllm_torchtpu.models.kimi_k3.megakernel.tools import preshard

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

_MINI_HIDDEN = 224
_MINI_VOCAB = 64
_MINI_EXPERTS_TOTAL = 32
_MINI_EXPERTS_PER_RANK = _MINI_EXPERTS_TOTAL // 8
_ATTENTION_DP_DEGREE = 2


@dataclasses.dataclass(frozen=True)
class _MiniCheckpointConfig:
    """Typed CheckpointConfig test double."""

    dim: int
    vocab: int
    experts: int
    full_attention: frozenset[int]


@dataclasses.dataclass(frozen=True)
class _LoadShim:
    """Typed LoadModuleProtocol test double."""

    PREFIX: str
    BF16: np.dtype[Any]
    Checkpoint: Callable[[str | Path], preshard.CheckpointProtocol]

    def selected_local_shapes(
        self,
        layers: int = 93,
        hidden: int = 7168,
        *,
        vocab: int = 163840,
    ) -> dict[str, tuple[int, ...]]:
        """Return miniature per-rank tensor shapes."""
        return _mini_shapes(layers, hidden, vocab=vocab)

    def family_layers(
        self,
        config: preshard.CheckpointConfig,
        name: str,
        layers: int,
    ) -> list[int] | None:
        """Delegate active layer indexing to load.family_layers."""
        return load.family_layers(config, name, layers)


@dataclasses.dataclass(frozen=True)
class _DraftConfigLoaderShim:
    """Typed DraftConfigLoader test double."""

    def from_checkpoint(self, directory: Path) -> object:
        """Return dummy draft config token."""
        _ = directory
        return object()


@dataclasses.dataclass(frozen=True)
class _DsparkModuleShim:
    """Typed DsparkModuleProtocol test double."""

    DraftConfig: preshard.DraftConfigLoader
    shards: dict[str, np.ndarray]

    def load_checkpoint_weights(self, directory: Path, config: object) -> object:
        """Return dummy draft weights token."""
        _ = (directory, config)
        return object()

    def kernel_weight_shards(
        self,
        config: object,
        weights: object,
        *,
        max_position: int = 16384,
    ) -> dict[str, np.ndarray]:
        """Return deterministic draft shards."""
        _ = (config, weights, max_position)
        return self.shards

    def save_kernel_shards(
        self, shards: dict[str, np.ndarray], directory: Path
    ) -> None:
        """Write kernel shards to disk using real dspark helper."""
        dspark.save_kernel_shards(shards, directory)


def _mini_shapes(
    layers: int = 2, hidden: int = _MINI_HIDDEN, *, vocab: int = _MINI_VOCAB
) -> dict[str, tuple[int, ...]]:
    """Return miniature per-rank tensor shapes matching selected_local_shapes keys."""
    raw = load.selected_local_shapes(layers, hidden, vocab=vocab)
    shapes = dict(raw)
    config = _MiniCheckpointConfig(
        dim=hidden,
        vocab=vocab,
        experts=_MINI_EXPERTS_TOTAL,
        full_attention=frozenset({1}),
    )
    for name, shape in list(shapes.items()):
        ids = load.family_layers(config, name, layers)
        if ids is not None:
            shapes[name] = (len(ids), *shape[1:])
    shapes["expert_gu"] = (layers - 1, _MINI_EXPERTS_PER_RANK, 4, 8)
    shapes["expert_gus"] = (layers - 1, _MINI_EXPERTS_PER_RANK, 2, 8)
    shapes["expert_down"] = (layers - 1, _MINI_EXPERTS_PER_RANK, 4, 6)
    shapes["expert_ds"] = (layers - 1, _MINI_EXPERTS_PER_RANK, 2, 6)
    return shapes


class _MiniatureCheckpoint:
    """Deterministic miniature 2-layer checkpoint for preshard and RankDir tests."""

    READ_THREADS = 1

    def __init__(self, layers: int = 2) -> None:
        """Initialize synthetic 2-layer KDA + MLA + MoE checkpoint metadata."""
        self.layers = layers
        self.shapes = _mini_shapes(layers)
        self.config: preshard.CheckpointConfig = _MiniCheckpointConfig(
            dim=_MINI_HIDDEN,
            vocab=_MINI_VOCAB,
            experts=_MINI_EXPERTS_TOTAL,
            full_attention=frozenset({1}),
        )

    def read(self, key: str, selection: object = None) -> np.ndarray:
        """Return deterministic raw checkpoint tensors for prefill_projection_values."""
        _ = selection
        seed = sum(ord(ch) for ch in key) * 997
        rng = np.random.default_rng(seed)
        bf16 = np.dtype(ml_dtypes.bfloat16)
        latent = (_MINI_HIDDEN * 3584) // 7168
        shared_total = 2 * (_MINI_HIDDEN * 3072) // 7168
        if key.endswith("routed_expert_down_proj.weight"):
            shape = (latent, _MINI_HIDDEN)
        elif key.endswith("routed_expert_up_proj.weight"):
            shape = (_MINI_HIDDEN, latent)
        elif key.endswith(("gate_proj.weight", "up_proj.weight")):
            shape = (shared_total, _MINI_HIDDEN)
        else:
            shape = (_MINI_HIDDEN, shared_total)
        return rng.integers(-8, 8, size=shape, dtype=np.int16).astype(bf16)

    def dense_selected(self, name: str, layer: int, rank: int) -> np.ndarray:
        """Return a deterministic dense tensor shard for (name, layer, rank)."""
        seed = sum(ord(ch) for ch in name) * 1009 + (layer + 1) * 97 + rank * 31
        rng = np.random.default_rng(seed)
        full_shape = self.shapes[name]
        ids = load.family_layers(self.config, name, self.layers)
        item_shape = full_shape if ids is None else full_shape[1:]
        if name in ("router", "router_bias", "k_a_log"):
            return rng.standard_normal(item_shape).astype(np.float32)
        bf16 = ml_dtypes.bfloat16
        return rng.integers(-16, 16, size=item_shape, dtype=np.int16).astype(bf16)

    def experts(
        self,
        layer: int,
        first: int,
        count: int,
        *,
        part: int | None = None,
        parts: int = 4,
    ) -> tuple[
        Sequence[np.ndarray] | np.ndarray,
        Sequence[np.ndarray] | np.ndarray,
        Sequence[np.ndarray] | np.ndarray,
        Sequence[np.ndarray] | np.ndarray,
    ]:
        """Return deterministic MXFP4 packed weight and scale arrays for a rank."""
        part_idx = 0 if part is None else part
        base = (layer + 1) * 1000 + first * 10 + part_idx * 3 + parts
        rng = np.random.default_rng(base)
        gu = rng.integers(0, 1 << 31, size=(count, 4, 8), dtype=np.uint32)
        gus = rng.integers(0, 255, size=(count, 2, 8), dtype=np.uint8)
        down = rng.integers(0, 1 << 31, size=(count, 4, 6), dtype=np.uint32)
        ds = rng.integers(0, 255, size=(count, 2, 6), dtype=np.uint8)
        return gu, gus, down, ds


def _build_load_shim(ckpt: _MiniatureCheckpoint) -> preshard.LoadModuleProtocol:
    """Build a structural LoadModuleProtocol shim for a miniature checkpoint."""
    return _LoadShim(
        PREFIX=load.PREFIX,
        BF16=np.dtype(load.BF16),
        Checkpoint=lambda _path: ckpt,
    )


def _assert_bit_exact(got: np.ndarray, expected: np.ndarray, label: str) -> None:
    """Assert two arrays have identical shape, dtype, and raw byte representation."""
    assert got.shape == expected.shape, f"Shape mismatch on {label}"
    assert got.dtype == expected.dtype, f"Dtype mismatch on {label}"
    max_abs_diff = int(
        np.max(
            np.abs(
                got.view(np.uint8).astype(np.int32)
                - expected.view(np.uint8).astype(np.int32)
            )
        )
    )
    assert max_abs_diff == 0, f"Bit mismatch on {label}"


def _verify_rank_dir_roundtrip(
    presharded_dir: Path,
    ckpt: _MiniatureCheckpoint,
    rank: int,
    layers: int,
) -> None:
    """Verify RankDir dense and expert readback matches checkpoint bit-for-bit."""
    rank_dir = rank_loader.RankDir(presharded_dir, rank=rank, layers=layers)
    for name in rank_dir.dense_names():
        ids = load.family_layers(ckpt.config, name, layers)
        if ids is None:
            expected_dense = ckpt.dense_selected(name, 0, rank)
        else:
            expected_dense = np.stack(
                [ckpt.dense_selected(name, layer, rank) for layer in ids]
            )
        _assert_bit_exact(
            rank_dir.read_dense(name), expected_dense, f"rank{rank}:{name}"
        )

    expected_experts = preshard.expert_values(ckpt, rank, 1)
    for name in rank_loader.EXPERT_FAMILIES:
        parts = [reader() for _label, _nbytes, reader in rank_dir.expert_parts(name)]
        got_exp = np.concatenate(parts, axis=0)
        _assert_bit_exact(
            got_exp, expected_experts[name][None], f"rank{rank}:expert:{name}"
        )


def test_preshard_2_rank_roundtrip_bit_exact(tmp_path: Path) -> None:
    """Verify 2-rank preshard write, RankDir readback, and legacy KDA split."""
    assert "read_dense_all" not in rank_loader.RankDir.__dict__
    layers = 2
    ckpt = _MiniatureCheckpoint(layers=layers)
    load_shim = _build_load_shim(ckpt)
    logs: list[str] = []
    for rank_idx in (0, 1):
        preshard.convert_rank(
            load_shim, ckpt, rank_idx, tmp_path, logs.append, layers=layers
        )
        _verify_rank_dir_roundtrip(tmp_path, ckpt, rank_idx, layers)

    decode_r0 = {
        f"{group}:{name}": arr
        for group, name, arr in rank_loader.iter_decode_rank(
            tmp_path, rank=0, layers=layers, attention_dp=1
        )
    }
    assert "decode:k_projection" in decode_r0
    assert "decode:k_gate" in decode_r0

    legacy_combined = np.arange(2 * 4 * 1792, dtype=np.uint16).reshape(2, 4, 1792)
    split_fams = rank_loader.split_kda_projection(
        {"k_projection": legacy_combined.copy()}
    )
    assert split_fams["k_gate"].shape == (2, 4, 128)
    assert split_fams["k_projection"].shape == (2, 4, 1664)
    _assert_bit_exact(
        split_fams["k_gate"], legacy_combined[..., 1152:1280], "legacy_k_gate"
    )


def test_preshard_32_rank_roundtrip_bit_exact(tmp_path: Path) -> None:
    """Verify full 32-rank preshard, attention_dp=2 decode, and even/odd prefill."""
    layers = 2
    ckpt = _MiniatureCheckpoint(layers=layers)
    load_shim = _build_load_shim(ckpt)
    logs: list[str] = []
    for rank_idx in range(32):
        preshard.convert_rank(
            load_shim, ckpt, rank_idx, tmp_path, logs.append, layers=layers
        )

    for sampled_rank in (0, 1, 2, 3, 15, 16, 31):
        _verify_rank_dir_roundtrip(tmp_path, ckpt, sampled_rank, layers)

    preshard.write_prefill(
        tmp_path,
        tmp_path,
        logs.append,
        layers=layers,
        load=load_shim,
    )
    prefill_cache = tmp_path / "prefill-sp-cache"

    for rank_idx in (0, 1, 2):
        decode_items = {
            f"{group}:{name}": arr
            for group, name, arr in rank_loader.iter_decode_rank(
                tmp_path,
                rank=rank_idx,
                layers=layers,
                attention_dp=_ATTENTION_DP_DEGREE,
            )
        }
        assert decode_items["decode:k_a_log"].shape[1] == _ATTENTION_DP_DEGREE
        prefill_items = {
            f"{group}:{name}": arr
            for group, name, arr in rank_loader.iter_prefill_rank(
                tmp_path, prefill_cache, rank=rank_idx, layers=layers
            )
        }
        assert "moe:latent_down" in prefill_items
        assert "moe:shared_gu" in prefill_items
        assert "kda:k_o" in prefill_items
        assert "mla:m_o" in prefill_items
        assert any(k.startswith("moe:expert_gu@") for k in prefill_items)


def test_write_prefill_and_verify_end_to_end(tmp_path: Path) -> None:
    """Verify preshard.write_prefill and preshard.verify pass and detect corruption."""
    layers = 2
    ckpt = _MiniatureCheckpoint(layers=layers)
    load_shim = _build_load_shim(ckpt)
    draft_dir = tmp_path / "draft_ckpt"
    draft_dir.mkdir()
    (draft_dir / "config.json").write_text(json.dumps({"model_type": "dspark"}))
    rng = np.random.default_rng(7)
    draft_arrays = {
        "embed": rng.integers(0, 100, size=(32, 4, 8), dtype=np.uint16),
    }
    dspark.save_kernel_shards(
        draft_arrays, preshard.draft_directory(str(tmp_path), str(draft_dir))
    )

    dspark_shim = _DsparkModuleShim(
        DraftConfig=_DraftConfigLoaderShim(),
        shards=draft_arrays,
    )

    logs: list[str] = []
    preshard.convert_rank(load_shim, ckpt, 0, tmp_path, logs.append, layers=layers)

    preshard.write_prefill(
        tmp_path,
        tmp_path,
        logs.append,
        layers=layers,
        load=load_shim,
    )
    with pytest.raises(FileExistsError):
        preshard.write_prefill(
            tmp_path,
            tmp_path,
            logs.append,
            layers=layers,
            load=load_shim,
        )

    assert (
        preshard.verify(
            tmp_path,
            tmp_path,
            draft_dir,
            logs.append,
            rank=0,
            layers_spec="1",
            load=load_shim,
            dspark=dspark_shim,
        )
        is True
    )

    dense_bin = tmp_path / "rank00" / "dense.bin"
    raw = bytearray(dense_bin.read_bytes())
    raw[0] ^= 0xFF
    dense_bin.write_bytes(bytes(raw))
    assert (
        preshard.verify(
            tmp_path,
            tmp_path,
            draft_dir,
            logs.append,
            rank=0,
            layers_spec="1",
            load=load_shim,
            dspark=dspark_shim,
        )
        is False
    )


def test_rank_loader_fail_closed_negative_controls(tmp_path: Path) -> None:
    """Verify ValueError, FileNotFoundError, and EOFError on corrupt shard files."""
    layers = 2
    ckpt = _MiniatureCheckpoint(layers=layers)
    load_shim = _build_load_shim(ckpt)
    logs: list[str] = []
    for rank_idx in (0, 1):
        preshard.convert_rank(
            load_shim, ckpt, rank_idx, tmp_path, logs.append, layers=layers
        )

    with pytest.raises(ValueError, match="unexpected pre-shard index"):
        rank_loader.RankDir(tmp_path, rank=0, layers=93)

    index_path = tmp_path / "rank00" / "index.json"
    orig_index = index_path.read_text()
    bad_index = json.loads(orig_index)
    bad_index["rank"] = 99
    index_path.write_text(json.dumps(bad_index))
    with pytest.raises(ValueError, match="unexpected pre-shard index"):
        rank_loader.RankDir(tmp_path, rank=0, layers=layers)
    index_path.write_text(orig_index)

    bad_cache = tmp_path / "bad_cache"
    bad_cache.mkdir()
    (bad_cache / "layout.json").write_text(json.dumps({"layers": 99, "families": {}}))
    with pytest.raises(ValueError, match="unexpected projection cache"):
        next(rank_loader.iter_prefill_rank(tmp_path, bad_cache, rank=0, layers=layers))

    dense_path = tmp_path / "rank00" / "dense.bin"
    orig_dense = dense_path.read_bytes()
    dense_path.write_bytes(orig_dense[:8])
    rank0 = rank_loader.RankDir(tmp_path, rank=0, layers=layers)
    with pytest.raises(EOFError, match="short read"):
        rank0.read_dense("embedding")

    dense_path.unlink()
    with pytest.raises(FileNotFoundError):
        rank0.read_dense("embedding")

    index_path.unlink()
    with pytest.raises(FileNotFoundError):
        rank_loader.RankDir(tmp_path, rank=0, layers=layers)

    bad_ckpt = _MiniatureCheckpoint(layers=layers)
    bad_ckpt.shapes["embedding"] = (999, _MINI_HIDDEN)
    with pytest.raises(ValueError):
        preshard.convert_rank(
            load_shim, bad_ckpt, 0, tmp_path, logs.append, layers=layers
        )


_TRUNCATED_SHARD_BYTES = 4
_TEST_DENSE_ELEMS = 16


def test_rank_dir_rejects_mismatched_index_and_truncated_shard_reads(
    tmp_path: Path,
) -> None:
    """Verify RankDir fails closed on missing dirs, bad index, and short reads."""
    with pytest.raises(FileNotFoundError):
        rank_loader.RankDir(tmp_path / "missing_root", rank=0, layers=4)

    rank_dir = tmp_path / "rank00"
    rank_dir.mkdir(parents=True)
    (rank_dir / "index.json").write_text(
        json.dumps({"rank": 1, "layers": 4, "families": {}})
    )
    with pytest.raises(ValueError, match="unexpected pre-shard index"):
        rank_loader.RankDir(tmp_path, rank=0, layers=4)

    (rank_dir / "index.json").write_text(
        json.dumps(
            {
                "rank": 0,
                "layers": 4,
                "families": {
                    "attn_norm": {
                        "file": "dense.bin",
                        "offset": 0,
                        "shape": [_TEST_DENSE_ELEMS],
                        "dtype": "float32",
                    }
                },
            }
        )
    )
    (rank_dir / "dense.bin").write_bytes(b"\x00" * _TRUNCATED_SHARD_BYTES)
    reader = rank_loader.RankDir(tmp_path, rank=0, layers=4)
    with pytest.raises(EOFError, match="short read"):
        reader.read_dense("attn_norm")


def test_preshard_cli_host_argument_gates_prefill_and_slices_host_ranks(
    tmp_path: Path,
) -> None:
    """Verify preshard --host CLI flag defaults, prefill host=0 gate, and arg errors."""
    parser = argparse.ArgumentParser()
    preshard.add_arguments(parser)
    parsed_default = parser.parse_args(
        ["prefill", "--checkpoint", str(tmp_path), "--output", str(tmp_path)]
    )
    assert parsed_default.host == 0

    parsed_host2 = parser.parse_args(
        [
            "k3",
            "--checkpoint",
            str(tmp_path),
            "--output",
            str(tmp_path),
            "--host",
            "2",
        ]
    )
    assert parsed_host2.host == _ATTENTION_DP_DEGREE

    with pytest.raises(SystemExit):
        parser.parse_args(["prefill", "--output", str(tmp_path)])

    ckpt = _MiniatureCheckpoint(layers=2)
    load_shim = _build_load_shim(ckpt)
    logs: list[str] = []
    host1_args = parser.parse_args(
        [
            "prefill",
            "--checkpoint",
            str(tmp_path),
            "--output",
            str(tmp_path),
            "--layers",
            "2",
            "--host",
            "1",
        ]
    )
    assert preshard.run(host1_args, logs.append, load=load_shim) == 0
    assert not (tmp_path / "prefill-sp-cache").exists()

    host0_args = parser.parse_args(
        [
            "prefill",
            "--checkpoint",
            str(tmp_path),
            "--output",
            str(tmp_path),
            "--layers",
            "2",
            "--host",
            "0",
        ]
    )
    assert preshard.run(host0_args, logs.append, load=load_shim) == 0
    assert (tmp_path / "prefill-sp-cache" / "layout.json").is_file()
