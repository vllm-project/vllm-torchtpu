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
from typing import TYPE_CHECKING

import ml_dtypes
import numpy as np
import pytest

from vllm_torchtpu.models.kimi_k3.megakernel.common.constants import (
    HEAD_DIM,
    HIDDEN_SIZE,
    NUM_LAYERS,
    TP_SIZE,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kernels import (
    Config,
    dspark,
    dspark_mla,
    load,
    rank_loader,
)
from vllm_torchtpu.models.kimi_k3.megakernel.tools import preshard

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

# Scaled 1/32 miniature geometry: dim=7168/32=224, latent=3584/32=112 (divisible by 7),
# shared_hidden=6144/32=192 (divisible by TP_SIZE=32).
_MINI_HIDDEN = HIDDEN_SIZE // TP_SIZE
_MINI_LATENT = _MINI_HIDDEN // 2
_MINI_SHARED_HIDDEN = 6 * TP_SIZE
_MINI_VOCAB = 64
_MINI_EXPERTS_TOTAL = 32
_MINI_EXPERTS_PER_RANK = _MINI_EXPERTS_TOTAL // 8
_ATTENTION_DP_DEGREE = 2


def _mini_config(*, hidden: int = _MINI_HIDDEN, vocab: int = _MINI_VOCAB) -> Config:
    """Build a scaled miniature kernels.Config for loader and preshard tests."""
    return dataclasses.replace(
        Config(),
        dim=hidden,
        latent=_MINI_LATENT,
        shared_hidden=_MINI_SHARED_HIDDEN,
        vocab=vocab,
        experts=_MINI_EXPERTS_TOTAL,
        full_attention=(1,),
    )


@dataclasses.dataclass(frozen=True)
class _LoadShim:
    """Typed LoadModuleProtocol test double."""

    PREFIX: str
    BF16: np.dtype[np.generic]
    Checkpoint: Callable[[str | Path], preshard.CheckpointProtocol[Config]]

    def selected_local_shapes(
        self,
        layers: int = NUM_LAYERS,
        hidden: int = HIDDEN_SIZE,
        *,
        vocab: int = preshard.DEFAULT_VOCAB_SIZE,
    ) -> dict[str, tuple[int, ...]]:
        """Return miniature per-rank tensor shapes."""
        return _mini_shapes(layers, hidden, vocab=vocab)

    def family_layers(
        self,
        config: Config,
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

    DraftConfig: preshard.DraftConfigLoader[object]
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
        max_position: int = preshard.DEFAULT_MAX_POSITION,
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
    config = _mini_config(hidden=hidden, vocab=vocab)
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
        self.config: Config = _mini_config()

    def read(self, key: str) -> np.ndarray:
        """Return deterministic raw checkpoint tensors for prefill_projection_values."""
        seed = sum(ord(ch) for ch in key) * 997
        rng = np.random.default_rng(seed)
        bf16 = np.dtype(ml_dtypes.bfloat16)
        if key.endswith("routed_expert_down_proj.weight"):
            shape = (self.config.latent, _MINI_HIDDEN)
        elif key.endswith("routed_expert_up_proj.weight"):
            shape = (_MINI_HIDDEN, self.config.latent)
        elif key.endswith(("gate_proj.weight", "up_proj.weight")):
            shape = (self.config.shared_hidden, _MINI_HIDDEN)
        else:
            shape = (_MINI_HIDDEN, self.config.shared_hidden)
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


def _build_load_shim(
    checkpoint: _MiniatureCheckpoint,
) -> preshard.LoadModuleProtocol[Config]:
    """Build a structural LoadModuleProtocol shim for a miniature checkpoint."""
    return _LoadShim(
        PREFIX=load.PREFIX,
        BF16=np.dtype(load.BF16),
        Checkpoint=lambda _path: checkpoint,
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
    checkpoint: _MiniatureCheckpoint,
    rank: int,
    layers: int,
) -> None:
    """Verify RankDir dense and expert readback matches checkpoint bit-for-bit."""
    rank_dir = rank_loader.RankDir(presharded_dir, rank=rank, layers=layers)
    for name in rank_dir.dense_names():
        ids = load.family_layers(checkpoint.config, name, layers)
        if ids is None:
            expected_dense = checkpoint.dense_selected(name, 0, rank)
        else:
            expected_dense = np.stack(
                [checkpoint.dense_selected(name, layer, rank) for layer in ids]
            )
        _assert_bit_exact(
            rank_dir.read_dense(name), expected_dense, f"rank{rank}:{name}"
        )

    expected_experts = preshard.expert_values(checkpoint, rank, 1)
    for name in rank_loader.EXPERT_FAMILIES:
        parts = [reader() for _label, _nbytes, reader in rank_dir.expert_parts(name)]
        got_expert = np.concatenate(parts, axis=0)
        _assert_bit_exact(
            got_expert, expected_experts[name][None], f"rank{rank}:expert:{name}"
        )


def _dense_family_array(
    checkpoint: _MiniatureCheckpoint, name: str, rank: int, layers: int
) -> np.ndarray:
    """Stack a dense family's per-layer checkpoint slices for rank."""
    ids = load.family_layers(checkpoint.config, name, layers)
    assert ids is not None
    return np.stack([checkpoint.dense_selected(name, layer, rank) for layer in ids])


def test_preshard_2_rank_roundtrip_bit_exact(tmp_path: Path) -> None:
    """Verify 2-rank preshard write, RankDir readback, and legacy KDA split."""
    assert "read_dense_all" not in rank_loader.RankDir.__dict__
    layers = 2
    checkpoint = _MiniatureCheckpoint(layers=layers)
    load_shim = _build_load_shim(checkpoint)
    logs: list[str] = []
    context = preshard.ConvertContext(output=tmp_path, log=logs.append, layers=layers)
    for rank_idx in (0, 1):
        preshard.convert_rank(load_shim, checkpoint, rank_idx, context)
        _verify_rank_dir_roundtrip(tmp_path, checkpoint, rank_idx, layers)

    decode_rank0 = {
        f"{group}:{name}": array
        for group, name, array in rank_loader.iter_decode_rank(
            tmp_path, rank=0, layers=layers, attention_dp=1
        )
    }
    assert "decode:k_projection" in decode_rank0
    assert "decode:k_gate" in decode_rank0

    # Legacy combined k_projection: 9 * HEAD_DIM Q/K/V + HEAD_DIM gate +
    # 4 * HEAD_DIM tail.
    gate_start = rank_loader.LEGACY_KDA_GATE_START
    gate_end = gate_start + HEAD_DIM
    legacy_width = gate_end + 4 * HEAD_DIM
    legacy_combined = np.arange(2 * 4 * legacy_width, dtype=np.uint16).reshape(
        2, 4, legacy_width
    )
    split_families = rank_loader.split_kda_projection(
        {"k_projection": legacy_combined.copy()}
    )
    assert split_families["k_gate"].shape == (2, 4, HEAD_DIM)
    assert split_families["k_projection"].shape == (2, 4, legacy_width - HEAD_DIM)
    _assert_bit_exact(
        split_families["k_gate"],
        legacy_combined[..., gate_start:gate_end],
        "legacy_k_gate",
    )


def test_preshard_32_rank_roundtrip_bit_exact(tmp_path: Path) -> None:
    """Verify full 32-rank preshard, attention_dp=2 decode, and even/odd prefill."""
    layers = 2
    checkpoint = _MiniatureCheckpoint(layers=layers)
    load_shim = _build_load_shim(checkpoint)
    logs: list[str] = []
    context = preshard.ConvertContext(output=tmp_path, log=logs.append, layers=layers)
    for rank_idx in range(TP_SIZE):
        preshard.convert_rank(load_shim, checkpoint, rank_idx, context)

    for sampled_rank in (0, 1, 2, 3, 15, 16, 31):
        _verify_rank_dir_roundtrip(tmp_path, checkpoint, sampled_rank, layers)

    preshard.write_prefill(
        tmp_path,
        tmp_path,
        logs.append,
        layers=layers,
        load=load_shim,
    )
    prefill_cache = tmp_path / "prefill-sp-cache"
    dp_group = TP_SIZE // _ATTENTION_DP_DEGREE

    for rank_idx in (0, 1, 2):
        decode_items = {
            f"{group}:{name}": array
            for group, name, array in rank_loader.iter_decode_rank(
                tmp_path,
                rank=rank_idx,
                layers=layers,
                attention_dp=_ATTENTION_DP_DEGREE,
            )
        }
        peer_ranks = [
            rank_idx % dp_group + shard * dp_group
            for shard in range(_ATTENTION_DP_DEGREE)
        ]
        expected_k_a_log = np.stack(
            [
                _dense_family_array(checkpoint, "k_a_log", peer_rank, layers)
                for peer_rank in peer_ranks
            ],
            axis=1,
        )
        expected_k_o = np.concatenate(
            [
                _dense_family_array(checkpoint, "k_o", peer_rank, layers)
                for peer_rank in peer_ranks
            ],
            axis=1,
        )
        expected_k_projection = np.concatenate(
            [
                _dense_family_array(checkpoint, "k_projection", peer_rank, layers)
                for peer_rank in peer_ranks
            ],
            axis=2,
        )
        assert decode_items["decode:k_a_log"].shape[1] == _ATTENTION_DP_DEGREE
        _assert_bit_exact(
            decode_items["decode:k_a_log"],
            expected_k_a_log,
            f"rank{rank_idx}:decode:k_a_log",
        )
        _assert_bit_exact(
            decode_items["decode:k_o"],
            expected_k_o,
            f"rank{rank_idx}:decode:k_o",
        )
        _assert_bit_exact(
            decode_items["decode:k_projection"],
            expected_k_projection,
            f"rank{rank_idx}:decode:k_projection",
        )

        prefill_items = {
            f"{group}:{name}": array
            for group, name, array in rank_loader.iter_prefill_rank(
                tmp_path, prefill_cache, rank=rank_idx, layers=layers
            )
        }
        assert "moe:latent_down" in prefill_items
        assert "moe:shared_gu" in prefill_items
        assert "kda:k_o" in prefill_items
        assert "mla:m_o" in prefill_items
        assert any(k.startswith("moe:expert_gu@") for k in prefill_items)
        _assert_bit_exact(
            prefill_items["kda:k_o"],
            _expected_sp_output(checkpoint, "k_o", rank_idx, layers),
            f"rank{rank_idx}:prefill:kda:k_o",
        )


def test_write_prefill_and_verify_end_to_end(tmp_path: Path) -> None:
    """Verify preshard.write_prefill and preshard.verify pass and detect corruption."""
    layers = 2
    checkpoint = _MiniatureCheckpoint(layers=layers)
    load_shim = _build_load_shim(checkpoint)
    draft_dir = tmp_path / "draft_ckpt"
    draft_dir.mkdir()
    (draft_dir / "config.json").write_text(json.dumps({"model_type": "dspark"}))
    rng = np.random.default_rng(7)
    draft_arrays = {
        "embed": rng.integers(0, 100, size=(TP_SIZE, 4, 8), dtype=np.uint16),
    }
    dspark.save_kernel_shards(
        draft_arrays, preshard.draft_directory(str(tmp_path), str(draft_dir))
    )

    dspark_shim = _DsparkModuleShim(
        DraftConfig=_DraftConfigLoaderShim(),
        shards=draft_arrays,
    )

    logs: list[str] = []
    preshard.convert_rank(
        load_shim,
        checkpoint,
        0,
        preshard.ConvertContext(output=tmp_path, log=logs.append, layers=layers),
    )

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

    verify_options = preshard.VerifyOptions(
        rank=0,
        layers_spec="1",
        load=load_shim,
        dspark=dspark_shim,
    )
    assert (
        preshard.verify(
            tmp_path,
            tmp_path,
            draft_dir,
            logs.append,
            verify_options,
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
            verify_options,
        )
        is False
    )


def test_rank_loader_fail_closed_negative_controls(tmp_path: Path) -> None:
    """Verify ValueError, FileNotFoundError, and EOFError on corrupt shard files."""
    layers = 2
    checkpoint = _MiniatureCheckpoint(layers=layers)
    load_shim = _build_load_shim(checkpoint)
    logs: list[str] = []
    context = preshard.ConvertContext(output=tmp_path, log=logs.append, layers=layers)
    for rank_idx in (0, 1):
        preshard.convert_rank(load_shim, checkpoint, rank_idx, context)

    with pytest.raises(ValueError, match="unexpected pre-shard index"):
        rank_loader.RankDir(tmp_path, rank=0, layers=NUM_LAYERS)

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

    bad_checkpoint = _MiniatureCheckpoint(layers=layers)
    bad_checkpoint.shapes["embedding"] = (999, _MINI_HIDDEN)
    with pytest.raises(ValueError, match=r"embedding"):
        preshard.convert_rank(load_shim, bad_checkpoint, 0, context)


_TRUNCATED_SHARD_BYTES = 4
_TEST_DENSE_ELEMENTS = 16


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
                        "shape": [_TEST_DENSE_ELEMENTS],
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
    """Verify preshard --host CLI defaults, prefill host=0 gate, and k3 dispatch."""
    parser = argparse.ArgumentParser()
    preshard.add_arguments(parser)
    parsed_default = parser.parse_args(
        ["prefill", "--checkpoint", str(tmp_path), "--output", str(tmp_path)]
    )
    assert parsed_default.host == 0

    host_index = 2
    parsed_host2 = parser.parse_args(
        [
            "k3",
            "--checkpoint",
            str(tmp_path),
            "--output",
            str(tmp_path),
            "--host",
            str(host_index),
        ]
    )
    assert parsed_host2.host == host_index

    with pytest.raises(SystemExit):
        parser.parse_args(["prefill", "--output", str(tmp_path)])

    checkpoint = _MiniatureCheckpoint(layers=2)
    load_shim = _build_load_shim(checkpoint)
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

    k3_dir = tmp_path / "cli_k3_host2"
    full_checkpoint = _MiniatureCheckpoint(layers=NUM_LAYERS)
    k3_host2_args = parser.parse_args(
        [
            "k3",
            "--checkpoint",
            str(tmp_path),
            "--output",
            str(k3_dir),
            "--ranks",
            "16",
            "--host",
            str(host_index),
        ]
    )
    assert (
        preshard.run(k3_host2_args, logs.append, load=_build_load_shim(full_checkpoint))
        == 0
    )
    assert not (k3_dir / "layout.json").exists()
    assert [entry.name for entry in k3_dir.iterdir() if entry.is_dir()] == ["rank16"]


def _expected_sp_output(
    checkpoint: _MiniatureCheckpoint, name: str, rank: int, layers: int
) -> np.ndarray:
    """Assemble reference sequence-parallel output projection for rank."""
    ids = load.family_layers(checkpoint.config, name, layers)
    assert ids is not None
    even_rank = rank - (rank % 2)
    even = np.stack(
        [checkpoint.dense_selected(name, layer, even_rank) for layer in ids]
    )
    odd = np.stack(
        [checkpoint.dense_selected(name, layer, even_rank + 1) for layer in ids]
    )
    half_rows = even.shape[1] // 2
    part = rank % 2
    row_slice = slice(part * half_rows, (part + 1) * half_rows)
    return np.ascontiguousarray(
        np.concatenate((even[:, row_slice], odd[:, row_slice]), axis=-1)
    )


def _expected_regrouped_experts(
    checkpoint: _MiniatureCheckpoint, rank: int
) -> dict[str, np.ndarray]:
    """Assemble reference EP16/TP2 regrouped expert families for rank."""
    group = 4 * (rank // 4)
    quarter_ranks = (group + 2 * (rank % 2), group + 2 * (rank % 2) + 1)
    part_idx = (rank // 2) % 2
    expert_slice = slice(
        part_idx * (_MINI_EXPERTS_PER_RANK // 2),
        (part_idx + 1) * (_MINI_EXPERTS_PER_RANK // 2),
    )
    quarter0 = preshard.expert_values(checkpoint, quarter_ranks[0], 1)
    quarter1 = preshard.expert_values(checkpoint, quarter_ranks[1], 1)
    expected: dict[str, np.ndarray] = {}
    for name in ("expert_gu", "expert_gus"):
        slice0 = quarter0[name][None, expert_slice]
        slice1 = quarter1[name][None, expert_slice]
        half = slice0.shape[-1] // 2
        expected[name] = np.concatenate(
            (
                slice0[..., :half],
                slice1[..., :half],
                slice0[..., half:],
                slice1[..., half:],
            ),
            axis=-1,
        )
    for name in ("expert_down", "expert_ds"):
        expected[name] = np.concatenate(
            (quarter0[name][None, expert_slice], quarter1[name][None, expert_slice]),
            axis=2,
        )
    return expected


def test_preshard_write_k3_host_slices_ranks_and_iter_prefill_bit_exact(
    tmp_path: Path,
) -> None:
    """Verify write_k3 host=2 rank slicing and bit-exact iter_prefill_rank values."""
    host2_checkpoint = _MiniatureCheckpoint(layers=NUM_LAYERS)
    host2_load_shim = _build_load_shim(host2_checkpoint)
    logs: list[str] = []

    host2_dir = tmp_path / "host2_out"
    host_index = 2
    ranks_per_host = preshard.RANKS_PER_HOST
    first_host2_rank = ranks_per_host * host_index
    preshard.write_k3(
        tmp_path,
        host2_dir,
        logs.append,
        preshard.WriteK3Options(
            in_flight=2,
            host=host_index,
            load=host2_load_shim,
        ),
    )
    assert not (host2_dir / "layout.json").exists()
    written_ranks = sorted(
        rank_path.name for rank_path in host2_dir.iterdir() if rank_path.is_dir()
    )
    expected_ranks = [
        f"rank{rank_idx:02d}"
        for rank_idx in range(
            first_host2_rank,
            ranks_per_host * (host_index + 1),
        )
    ]
    assert written_ranks == expected_ranks
    rank16_dir = rank_loader.RankDir(
        host2_dir, rank=first_host2_rank, layers=NUM_LAYERS
    )
    assert rank16_dir.index["layers"] == NUM_LAYERS
    _assert_bit_exact(
        rank16_dir.read_dense("embedding"),
        host2_checkpoint.dense_selected("embedding", 0, first_host2_rank),
        "host2:rank16:embedding",
    )

    layers = 2
    checkpoint = _MiniatureCheckpoint(layers=layers)
    load_shim = _build_load_shim(checkpoint)
    prefill_dir = tmp_path / "prefill_out"
    context = preshard.ConvertContext(
        output=prefill_dir, log=logs.append, layers=layers
    )
    for rank_idx in range(ranks_per_host):
        preshard.convert_rank(load_shim, checkpoint, rank_idx, context)
    preshard.write_prefill(
        prefill_dir,
        prefill_dir,
        logs.append,
        layers=layers,
        load=load_shim,
    )
    prefill_cache = prefill_dir / "prefill-sp-cache"
    proj_layer1 = preshard.prefill_projection_values(load_shim, checkpoint, 1)

    for test_rank in (0, 3, 4, 7):
        prefill_items = {
            f"{group}:{name}": array
            for group, name, array in rank_loader.iter_prefill_rank(
                prefill_dir, prefill_cache, rank=test_rank, layers=layers
            )
        }
        _assert_bit_exact(
            prefill_items["kda:k_o"],
            _expected_sp_output(checkpoint, "k_o", test_rank, layers),
            f"rank{test_rank}:kda:k_o",
        )
        _assert_bit_exact(
            prefill_items["mla:m_o"],
            _expected_sp_output(checkpoint, "m_o", test_rank, layers),
            f"rank{test_rank}:mla:m_o",
        )
        expected_experts = _expected_regrouped_experts(checkpoint, test_rank)
        for family_name, expected_array in expected_experts.items():
            got_array = np.concatenate(
                [
                    prefill_items[key]
                    for key in sorted(prefill_items)
                    if key.startswith(f"moe:{family_name}@")
                ],
                axis=0,
            )
            _assert_bit_exact(
                got_array, expected_array, f"rank{test_rank}:moe:{family_name}"
            )
        _assert_bit_exact(
            prefill_items["moe:latent_down"],
            proj_layer1["latent_down"][None],
            f"rank{test_rank}:moe:latent_down",
        )
        _assert_bit_exact(
            prefill_items["moe:latent_up"],
            proj_layer1["latent_up"][None],
            f"rank{test_rank}:moe:latent_up",
        )
        _assert_bit_exact(
            prefill_items["moe:shared_gu"],
            proj_layer1[f"shared_gu_{test_rank % 2}"][None],
            f"rank{test_rank}:moe:shared_gu",
        )
        _assert_bit_exact(
            prefill_items["moe:shared_down"],
            proj_layer1[f"shared_down_{test_rank % 2}"][None],
            f"rank{test_rank}:moe:shared_down",
        )


def test_preshard_verify_logs_dense_shape_mismatch(tmp_path: Path) -> None:
    """Verify preshard.verify fails closed and logs stored shape on dense mismatch."""
    layers = 2
    checkpoint = _MiniatureCheckpoint(layers=layers)
    load_shim = _build_load_shim(checkpoint)
    draft_dir = tmp_path / "draft_ckpt"
    draft_dir.mkdir()
    (draft_dir / "config.json").write_text(json.dumps({"model_type": "dspark"}))
    rng = np.random.default_rng(11)
    draft_arrays = {
        "embed": rng.integers(0, 100, size=(TP_SIZE, 4, 8), dtype=np.uint16),
    }
    dspark.save_kernel_shards(
        draft_arrays, preshard.draft_directory(str(tmp_path), str(draft_dir))
    )
    dspark_shim = _DsparkModuleShim(
        DraftConfig=_DraftConfigLoaderShim(),
        shards=draft_arrays,
    )

    setup_logs: list[str] = []
    preshard.convert_rank(
        load_shim,
        checkpoint,
        0,
        preshard.ConvertContext(output=tmp_path, log=setup_logs.append, layers=layers),
    )

    index_path = tmp_path / "rank00" / "index.json"
    original_index = index_path.read_text()
    verify_options = preshard.VerifyOptions(
        rank=0,
        layers_spec="1",
        load=load_shim,
        dspark=dspark_shim,
    )

    for family_name, corrupt_shape in (
        ("embedding", [1, _MINI_HIDDEN]),
        ("attn_norm", []),
        ("attn_norm", [1, _MINI_HIDDEN]),
    ):
        index_payload = json.loads(original_index)
        index_payload["families"][family_name]["shape"] = corrupt_shape
        index_path.write_text(json.dumps(index_payload))
        logs: list[str] = []
        ok = preshard.verify(
            tmp_path,
            tmp_path,
            draft_dir,
            logs.append,
            verify_options,
        )
        assert ok is False
        assert any(f"{family_name}: stored shape" in entry for entry in logs)
        assert any("expert_gu layer 1: identical" in entry for entry in logs)


def test_preshard_real_modules_satisfy_protocols() -> None:
    """Verify real Kimi-K3 config and kernel modules satisfy preshard protocols."""
    config: preshard.CheckpointConfig = Config()
    load_mod: preshard.LoadModuleProtocol[Config] = load
    dspark_mod: preshard.DsparkModuleProtocol[
        dspark.DraftConfig, dict[str, np.ndarray]
    ] = dspark
    mla_mod: preshard.MlaModuleProtocol[
        dspark_mla.MlaDraftConfig, dict[str, np.ndarray]
    ] = dspark_mla
    assert config.vocab == preshard.DEFAULT_VOCAB_SIZE
    assert load_mod.PREFIX == load.PREFIX
    assert dspark_mod.DraftConfig is dspark.DraftConfig
    assert mla_mod.MlaDraftConfig is dspark_mla.MlaDraftConfig
