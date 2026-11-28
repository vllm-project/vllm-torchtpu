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
"""CPU unit tests for Kimi-K3 rank_loader and offline preshard CLI."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pytest

from vllm_torchtpu.models.kimi_k3.megakernel.kernels import (
    decode_megakernel,
    rank_loader,
)
from vllm_torchtpu.models.kimi_k3.megakernel.tools import preshard

if TYPE_CHECKING:
    from pathlib import Path

TINY_DIM = 4
TINY_VOCAB = 8
TINY_EXPERTS = 8
TINY_DENSE_LAYERS = (0, 1)
TINY_LAYER_COUNT = len(TINY_DENSE_LAYERS)
TINY_PREFILL_LAYERS = 2

# Kimi-K3 TP32 topology geometry on 2x2x4 TPU v7x:
# 4 hosts x 8 devices/host (4 dual-core chips/host) = 32 TP ranks.
TP_RANKS = preshard.RANKS
ATTENTION_DP_FACTOR = 2
PEER_RANK_DP2 = TP_RANKS // ATTENTION_DP_FACTOR

# Sequence-parallel output halves: 2 halves x 384 rows per rank pair (768 total).
SP_HALVES = 2
SP_HALF_ROWS = 384
SP_FULL_ROWS = SP_HALVES * SP_HALF_ROWS
SP_COLS = 2

# Legacy KDA projection columns: q/k/v (1152) + gate (128) + g/b (512) = 1792.
KDA_QKV_COLS = 1152
KDA_GATE_COLS = 128
KDA_TAIL_COLS = 512
KDA_GATE_START = KDA_QKV_COLS
KDA_GATE_END = KDA_GATE_START + KDA_GATE_COLS
KDA_SPLIT_COLS = KDA_QKV_COLS + KDA_TAIL_COLS
LEGACY_KDA_COMBINED_COLS = KDA_SPLIT_COLS + KDA_GATE_COLS

# Prefill latent_down pads 3584 -> 4096 (+512); shared experts split 6144
# channels across 32 ranks.
LATENT_PAD_COLS = 4096 - 3584
SHARED_EXPERT_INTERMEDIATE = 6144
SHARED_RANK_CHANNELS = SHARED_EXPERT_INTERMEDIATE // TP_RANKS
PARITY_GROUPS = 2
PARITY_RANKS = TP_RANKS // PARITY_GROUPS
SHARED_PARITY_CHANNELS = PARITY_RANKS * SHARED_RANK_CHANNELS
SHARED_HIDDEN = 4
LATENT_IN = 8
LATENT_OUT = 4

# Tiny EP8/TP4 -> EP16/TP2 regrouping fixture dimensions.
EXPERT_QUARTER_COUNT = 4
EXPERT_PER_PART = EXPERT_QUARTER_COUNT // PARITY_GROUPS
EXPERT_K_DIM = 2
EXPERT_N_DIM = 4
# Keep per-rank offsets within uint8 [0, 255] for expert_gus/expert_ds:
# max element 31 + (4 - 1) * 50 = 181 <= 255.
EXPERT_RANK_UINT8_STEP = 50


@dataclass(frozen=True)
class _TinyConfig:
    """Minimal checkpoint geometry consumed by ``preshard.convert_rank``."""

    dim: int = TINY_DIM
    vocab: int = TINY_VOCAB
    experts: int = TINY_EXPERTS


class _TinyCheckpoint:
    """Deterministic in-memory checkpoint stub for host preshard tests."""

    READ_THREADS = 1

    def __init__(self, path: str) -> None:
        """Store path and construct tiny checkpoint config."""
        self.path = path
        self.config = _TinyConfig()

    def dense_selected(self, name: str, layer: int, rank: int) -> np.ndarray:
        """Return deterministic dense tensor keyed by family, layer, and rank."""
        seed = (sum(ord(ch) for ch in name) + layer * 31 + rank * 131) % 1000
        if name == "embedding":
            return np.full((TINY_VOCAB, TINY_DIM), seed, dtype=np.float32)
        if name in ("k_conv", "k_a_log", "k_dt"):
            return np.full((TINY_LAYER_COUNT,), seed + 0.5, dtype=np.float32)
        return np.full(
            (TINY_LAYER_COUNT, TINY_LAYER_COUNT), seed + 1.0, dtype=np.float32
        )

    def experts(
        self,
        layer: int,
        first: int,
        count: int,
        *,
        part: int,
        parts: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return deterministic tiny MXFP4/E8M0 expert tensors for one layer."""
        base = layer * 17 + first * 5 + part + parts
        gate_up = np.full((count, 1, 2), base, dtype=np.uint32)
        gate_up_scales = np.full((count, 1, 2), base % 251, dtype=np.uint8)
        downs = np.full((count, 1, 2), base + 3, dtype=np.uint32)
        down_scale = np.full((count, 1, 2), (base + 7) % 251, dtype=np.uint8)
        return gate_up, gate_up_scales, downs, down_scale

    def read(self, tensor_name: str) -> np.ndarray:
        """Return deterministic projection weights for ``prefill_projection_values``."""
        if tensor_name.endswith("routed_expert_down_proj.weight"):
            return np.arange(LATENT_OUT * LATENT_IN, dtype=np.float32).reshape(
                LATENT_OUT, LATENT_IN
            )
        if tensor_name.endswith("routed_expert_up_proj.weight"):
            return np.arange(LATENT_IN * LATENT_OUT, dtype=np.float32).reshape(
                LATENT_IN, LATENT_OUT
            )
        if tensor_name.endswith("shared_experts.gate_proj.weight"):
            rows = np.arange(SHARED_EXPERT_INTERMEDIATE, dtype=np.float32)[:, None]
            return np.broadcast_to(
                rows, (SHARED_EXPERT_INTERMEDIATE, SHARED_HIDDEN)
            ).copy()
        if tensor_name.endswith("shared_experts.up_proj.weight"):
            rows = (
                np.arange(SHARED_EXPERT_INTERMEDIATE, dtype=np.float32)[:, None]
                + 10000.0
            )
            return np.broadcast_to(
                rows, (SHARED_EXPERT_INTERMEDIATE, SHARED_HIDDEN)
            ).copy()
        if tensor_name.endswith("shared_experts.down_proj.weight"):
            cols = (
                np.arange(SHARED_EXPERT_INTERMEDIATE, dtype=np.float32)[None, :]
                + 20000.0
            )
            return np.broadcast_to(
                cols, (SHARED_HIDDEN, SHARED_EXPERT_INTERMEDIATE)
            ).copy()
        msg = f"unexpected tensor name: {tensor_name}"
        raise KeyError(msg)


class _TinyLoadModule:
    """Minimal ``kernels.load`` surface required by ``preshard``."""

    PREFIX = "model."
    Checkpoint = _TinyCheckpoint

    @staticmethod
    def selected_local_shapes(
        layers: int,
        dim: int = TINY_DIM,
        *,
        vocab: int = TINY_VOCAB,
    ) -> dict[str, tuple[int, ...]]:
        """Return tiny per-rank shapes covering dense, DP-attention, and experts."""
        expert_layers = layers - 1
        shapes: dict[str, tuple[int, ...]] = {
            "embedding": (vocab, dim),
            "k_gate": (TINY_LAYER_COUNT, TINY_LAYER_COUNT, TINY_LAYER_COUNT),
        }
        for name in decode_megakernel.ATTENTION_DP_WEIGHTS:
            if name in ("k_conv", "k_a_log", "k_dt"):
                shapes[name] = (TINY_LAYER_COUNT, TINY_LAYER_COUNT)
            else:
                shapes[name] = (TINY_LAYER_COUNT, TINY_LAYER_COUNT, TINY_LAYER_COUNT)
        for expert_name in preshard.EXPERT_FAMILIES:
            shapes[expert_name] = (expert_layers, 1, 1, 2)
        return shapes

    @staticmethod
    def _family_layers(
        config: _TinyConfig,
        name: str,
        layers: int,
    ) -> tuple[int, ...] | None:
        """Return layer indices for per-layer families, or None for embedding."""
        _ = (config, layers)
        return None if name == "embedding" else TINY_DENSE_LAYERS


class _TinyDsparkModule:
    """Minimal ``kernels.dspark`` surface for ``preshard.verify``."""

    @staticmethod
    def is_mla_draft(path: Path) -> bool:
        """Report non-MLA draft so ``preshard.draft_shards`` uses this stub."""
        _ = path
        return False

    class DraftConfig:
        """Tiny draft config factory."""

        @staticmethod
        def from_checkpoint(path: Path) -> Path:
            """Return checkpoint path verbatim."""
            return path

    @staticmethod
    def load_checkpoint_weights(path: Path, config: Path) -> dict[str, np.ndarray]:
        """Return deterministic draft weights."""
        _ = (path, config)
        return {"w": np.arange(TP_RANKS * 4, dtype=np.float32).reshape(TP_RANKS, 4)}

    @staticmethod
    def kernel_weight_shards(
        config: Path,
        weights: dict[str, np.ndarray],
        *,
        max_position: int,
    ) -> dict[str, np.ndarray]:
        """Return deterministic per-rank draft shards."""
        _ = (config, max_position)
        return weights


def _parse_cli(argv: list[str]) -> argparse.Namespace:
    """Parse ``preshard`` CLI arguments using production ``add_arguments``."""
    parser = argparse.ArgumentParser()
    preshard.add_arguments(parser)
    return parser.parse_args(argv)


def _write_rank_dir(
    root: Path,
    rank: int,
    layers: int,
    dense_arrays: dict[str, np.ndarray],
    expert_arrays: dict[str, np.ndarray],
) -> None:
    """Write one synthetic rank directory matching ``RankDir`` on-disk layout."""
    rank_dir = root / f"rank{rank:02d}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    families: dict[str, object] = {}
    offset = 0
    with (rank_dir / "dense.bin").open("wb") as handle:
        for name, array in dense_arrays.items():
            contiguous = np.ascontiguousarray(array)
            handle.write(contiguous.tobytes())
            families[name] = {
                "file": "dense.bin",
                "offset": offset,
                "shape": list(contiguous.shape),
                "dtype": str(contiguous.dtype),
            }
            offset += contiguous.nbytes
    for name, array in expert_arrays.items():
        contiguous = np.ascontiguousarray(array)
        filename = f"{name}.bin"
        (rank_dir / filename).write_bytes(contiguous.tobytes())
        families[name] = {
            "files": [filename],
            "layer_ranges": [[0, contiguous.shape[0]]],
            "shape": list(contiguous.shape),
            "dtype": str(contiguous.dtype),
        }
    index = {
        "format": preshard.FORMAT,
        "rank": rank,
        "layers": layers,
        "families": families,
    }
    (rank_dir / "index.json").write_text(json.dumps(index, indent=1))


def test_preshard_k3_host_slicing_and_layout_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Slice k3 ranks by --host and write layout.json on host 0 only."""
    monkeypatch.setattr(
        preshard, "_modules", lambda: (_TinyLoadModule, _TinyDsparkModule)
    )
    logs: list[str] = []

    host0_dir = tmp_path / "host0"
    args_host0 = _parse_cli(
        [
            "k3",
            "--checkpoint",
            "tiny",
            "--output",
            str(host0_dir),
            "--host",
            "0",
            "--in-flight",
            "1",
        ]
    )
    host_ranks = preshard.RANKS_PER_HOST
    assert preshard.run(args_host0, logs.append) == 0
    assert (host0_dir / "layout.json").is_file()
    assert sorted(p.name for p in host0_dir.glob("rank*")) == [
        f"rank{rank:02d}" for rank in range(host_ranks)
    ]

    host1_dir = tmp_path / "host1"
    args_host1 = _parse_cli(
        [
            "k3",
            "--checkpoint",
            "tiny",
            "--output",
            str(host1_dir),
            "--host",
            "1",
            "--in-flight",
            "1",
        ]
    )
    assert preshard.run(args_host1, logs.append) == 0
    assert not (host1_dir / "layout.json").exists()
    assert sorted(p.name for p in host1_dir.glob("rank*")) == [
        f"rank{rank:02d}" for rank in range(host_ranks, 2 * host_ranks)
    ]


@pytest.mark.parametrize(
    ("subcommand", "extra_args"),
    [
        ("k3", ()),
        ("k3", ("--ranks", "0")),
        ("prefill", ()),
    ],
)
@pytest.mark.parametrize("use_upper_bound", [False, True])
def test_preshard_invalid_host_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    subcommand: str,
    extra_args: tuple[str, ...],
    *,
    use_upper_bound: bool,
) -> None:
    """Reject out-of-range --host values on k3 (with or without --ranks) and prefill."""
    monkeypatch.setattr(
        preshard, "_modules", lambda: (_TinyLoadModule, _TinyDsparkModule)
    )
    bad_host = preshard.HOSTS if use_upper_bound else -1
    logs: list[str] = []
    args = _parse_cli(
        [
            subcommand,
            "--checkpoint",
            "tiny",
            "--output",
            str(tmp_path / f"bad_{subcommand}_{len(extra_args)}_{bad_host}"),
            "--host",
            str(bad_host),
            *extra_args,
        ]
    )
    with pytest.raises(ValueError, match="out of range"):
        preshard.run(args, logs.append)


def test_preshard_k3_decode_attention_dp_roundtrip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round-trip k3 shards through RankDir and iter_decode_rank at dp=1 and dp=2."""
    monkeypatch.setattr(
        preshard, "_modules", lambda: (_TinyLoadModule, _TinyDsparkModule)
    )
    logs: list[str] = []
    dp_dir = tmp_path / "dp_pair"
    args_dp = _parse_cli(
        [
            "k3",
            "--checkpoint",
            "tiny",
            "--output",
            str(dp_dir),
            "--ranks",
            f"0,{PEER_RANK_DP2}",
            "--in-flight",
            "1",
        ]
    )
    assert preshard.run(args_dp, logs.append) == 0

    rank0_dir = rank_loader.RankDir(dp_dir, rank=0, layers=preshard.LAYERS)
    peer_rank_dir = rank_loader.RankDir(
        dp_dir, rank=PEER_RANK_DP2, layers=preshard.LAYERS
    )
    expected_emb = _TinyCheckpoint("tiny").dense_selected("embedding", 0, 0)
    np.testing.assert_array_equal(rank0_dir.read_dense("embedding"), expected_emb)
    expert_chunks = [reader() for _, _, reader in rank0_dir.expert_parts("expert_gu")]
    assert sum(chunk.shape[0] for chunk in expert_chunks) == preshard.LAYERS - 1

    dp1_weights = {
        name: array
        for _, name, array in rank_loader.iter_decode_rank(
            dp_dir, rank=0, attention_dp=1
        )
    }
    dp2_weights = {
        name: array
        for _, name, array in rank_loader.iter_decode_rank(
            dp_dir, rank=0, attention_dp=ATTENTION_DP_FACTOR
        )
    }
    np.testing.assert_array_equal(dp1_weights["embedding"], expected_emb)
    for name, (_, axis) in decode_megakernel.ATTENTION_DP_WEIGHTS.items():
        own_arr = rank0_dir.read_dense(name)
        peer_arr = peer_rank_dir.read_dense(name)
        expected = (
            np.stack([own_arr, peer_arr], axis=1)
            if axis is None
            else np.concatenate([own_arr, peer_arr], axis=1 + axis)
        )
        np.testing.assert_array_equal(dp2_weights[name], expected)


def test_iter_prefill_rank_regroups_experts_and_sp_outputs(tmp_path: Path) -> None:
    """Regroup EP16/TP2 experts, SP outputs, and cached projections in prefill."""
    presharded = tmp_path / "shards"
    for rank in range(EXPERT_QUARTER_COUNT):
        dense: dict[str, np.ndarray] = {}
        for name in (
            *rank_loader.PREFILL_IO,
            *rank_loader.PREFILL_COMMON,
            *rank_loader.PREFILL_KDA,
            *rank_loader.PREFILL_MLA,
            *rank_loader.PREFILL_MOE_DENSE,
        ):
            if name in ("k_o", "m_o"):
                dense[name] = (
                    np.arange(SP_FULL_ROWS * SP_COLS, dtype=np.float32).reshape(
                        1, SP_FULL_ROWS, SP_COLS
                    )
                    + rank * 10000.0
                )
            else:
                dense[name] = np.full((1, 2), float(rank + 1), dtype=np.float32)
        experts = {
            name: np.arange(
                EXPERT_QUARTER_COUNT * EXPERT_K_DIM * EXPERT_N_DIM,
                dtype=preshard.EXPERT_DTYPES[name],
            ).reshape(1, EXPERT_QUARTER_COUNT, EXPERT_K_DIM, EXPERT_N_DIM)
            + rank * EXPERT_RANK_UINT8_STEP
            for name in preshard.EXPERT_FAMILIES
        }
        _write_rank_dir(presharded, rank, TINY_PREFILL_LAYERS, dense, experts)

    cache_dir = tmp_path / "prefill-sp-cache"
    cache_dir.mkdir()
    cache_specs: dict[str, dict[str, object]] = {}
    cache_arrays: dict[str, np.ndarray] = {}
    for idx, name in enumerate(
        (
            "latent_down",
            "latent_up",
            "shared_gu_0",
            "shared_gu_1",
            "shared_down_0",
            "shared_down_1",
        )
    ):
        arr = np.full((1, 2, 2), float(idx + 1), dtype=np.float32)
        (cache_dir / f"{name}.bin").write_bytes(arr.tobytes())
        cache_arrays[name] = arr
        cache_specs[name] = {
            "file": f"{name}.bin",
            "shape": list(arr.shape),
            "dtype": "float32",
        }
    (cache_dir / "layout.json").write_text(
        json.dumps(
            {"format": 1, "layers": TINY_PREFILL_LAYERS, "families": cache_specs}
        )
    )

    loaded = {
        (group, name): array
        for group, name, array in rank_loader.iter_prefill_rank(
            presharded, cache_dir, rank=1, layers=TINY_PREFILL_LAYERS
        )
    }
    np.testing.assert_array_equal(
        loaded[("moe", "latent_down")], cache_arrays["latent_down"]
    )
    np.testing.assert_array_equal(
        loaded[("moe", "shared_gu")], cache_arrays["shared_gu_1"]
    )
    np.testing.assert_array_equal(
        loaded[("moe", "shared_down")], cache_arrays["shared_down_1"]
    )

    rank0_dir = rank_loader.RankDir(presharded, rank=0, layers=TINY_PREFILL_LAYERS)
    rank1_dir = rank_loader.RankDir(presharded, rank=1, layers=TINY_PREFILL_LAYERS)
    expected_ko = np.concatenate(
        [
            rank0_dir.read_dense("k_o")[:, SP_HALF_ROWS:SP_FULL_ROWS],
            rank1_dir.read_dense("k_o")[:, SP_HALF_ROWS:SP_FULL_ROWS],
        ],
        axis=-1,
    )
    np.testing.assert_array_equal(loaded[("kda", "k_o")], expected_ko)
    assert loaded[("moe", "expert_gu@0/1")].shape == (
        1,
        EXPERT_PER_PART,
        EXPERT_K_DIM,
        2 * EXPERT_N_DIM,
    )
    assert loaded[("moe", "expert_down@0/1")].shape == (
        1,
        EXPERT_PER_PART,
        2 * EXPERT_K_DIM,
        EXPERT_N_DIM,
    )


def test_split_kda_projection_legacy_and_already_split() -> None:
    """Split legacy 1792-col KDA projections and leave already-split tensors intact."""
    combined = np.arange(2 * LEGACY_KDA_COMBINED_COLS, dtype=np.float32).reshape(
        2, LEGACY_KDA_COMBINED_COLS
    )
    legacy = rank_loader.split_kda_projection({"k_projection": combined})
    assert legacy["k_gate"].shape == (2, KDA_GATE_COLS)
    assert legacy["k_projection"].shape == (2, KDA_SPLIT_COLS)
    np.testing.assert_array_equal(
        legacy["k_gate"], combined[:, KDA_GATE_START:KDA_GATE_END]
    )
    np.testing.assert_array_equal(
        legacy["k_projection"],
        np.concatenate(
            (combined[:, :KDA_GATE_START], combined[:, KDA_GATE_END:]), axis=-1
        ),
    )

    already_split = np.zeros((2, KDA_SPLIT_COLS), dtype=np.float32)
    modern = rank_loader.split_kda_projection({"k_projection": already_split})
    assert "k_gate" not in modern
    assert modern["k_projection"] is already_split


def test_rank_loader_fail_closed_negative_controls(tmp_path: Path) -> None:
    """Fail closed on missing, mismatched, or truncated rank and cache files."""
    with pytest.raises(FileNotFoundError):
        rank_loader.RankDir(tmp_path, rank=0, layers=TINY_PREFILL_LAYERS)

    bad_rank = tmp_path / "bad_rank"
    _write_rank_dir(
        bad_rank,
        rank=0,
        layers=TINY_PREFILL_LAYERS,
        dense_arrays={"embedding": np.zeros((2, 2), dtype=np.float32)},
        expert_arrays={},
    )
    (bad_rank / "rank00" / "index.json").write_text(
        json.dumps(
            {"format": 1, "rank": 1, "layers": TINY_PREFILL_LAYERS, "families": {}}
        )
    )
    with pytest.raises(ValueError, match="unexpected pre-shard index"):
        rank_loader.RankDir(bad_rank, rank=0, layers=TINY_PREFILL_LAYERS)

    trunc_root = tmp_path / "trunc"
    _write_rank_dir(
        trunc_root,
        rank=0,
        layers=TINY_PREFILL_LAYERS,
        dense_arrays={"embedding": np.ones((4, 4), dtype=np.float32)},
        expert_arrays={},
    )
    dense_path = trunc_root / "rank00" / "dense.bin"
    dense_path.write_bytes(dense_path.read_bytes()[:4])
    trunc_dir = rank_loader.RankDir(trunc_root, rank=0, layers=TINY_PREFILL_LAYERS)
    with pytest.raises(EOFError, match="short read"):
        trunc_dir.read_dense("embedding")

    bad_cache = tmp_path / "bad_cache"
    bad_cache.mkdir()
    (bad_cache / "layout.json").write_text(json.dumps({"layers": 99, "families": {}}))
    for rank in (0, 1):
        _write_rank_dir(
            tmp_path / "ok_pair", rank, TINY_PREFILL_LAYERS, {"e": np.zeros(1)}, {}
        )
    with pytest.raises(ValueError, match="unexpected projection cache"):
        next(
            rank_loader.iter_prefill_rank(
                tmp_path / "ok_pair", bad_cache, rank=0, layers=TINY_PREFILL_LAYERS
            )
        )


def test_preshard_prefill_skips_nonzero_host_with_log(tmp_path: Path) -> None:
    """Skip prefill on non-zero valid hosts and log the skip."""
    logs: list[str] = []
    skip_out = tmp_path / "prefill_skip"
    args_skip = _parse_cli(
        ["prefill", "--checkpoint", "tiny", "--output", str(skip_out), "--host", "1"]
    )
    assert preshard.run(args_skip, logs.append) == 0
    assert not (skip_out / "prefill-sp-cache").exists()
    assert any("skipping" in line and "host 1" in line for line in logs)


def test_prefill_projection_values_padding_and_parity() -> None:
    """Pad latent_down by 512 columns and group shared channels by rank parity."""
    ckpt = _TinyCheckpoint("tiny")
    proj = preshard.prefill_projection_values(_TinyLoadModule, ckpt, layer=1)
    assert proj["latent_down"].shape == (LATENT_IN, LATENT_OUT + LATENT_PAD_COLS)
    assert np.all(proj["latent_down"][:, LATENT_OUT:] == 0)
    assert proj["latent_up"].shape == (LATENT_OUT, LATENT_IN)
    assert proj["shared_gu_0"].shape == (SHARED_HIDDEN, 2 * SHARED_PARITY_CHANNELS)
    assert proj["shared_down_1"].shape == (SHARED_PARITY_CHANNELS, SHARED_HIDDEN)
    expected_even_first = np.arange(SHARED_RANK_CHANNELS, dtype=np.float32)
    np.testing.assert_array_equal(
        proj["shared_gu_0"][0, :SHARED_RANK_CHANNELS], expected_even_first
    )
    expected_odd_first = np.arange(
        SHARED_RANK_CHANNELS, 2 * SHARED_RANK_CHANNELS, dtype=np.float32
    )
    np.testing.assert_array_equal(
        proj["shared_down_1"][:SHARED_RANK_CHANNELS, 0], expected_odd_first + 20000.0
    )


def test_preshard_verify_matches_and_detects_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return 0 on intact k3 + dspark shards and 1 after a single-byte flip."""
    monkeypatch.setattr(
        preshard, "_modules", lambda: (_TinyLoadModule, _TinyDsparkModule)
    )
    monkeypatch.setattr(preshard, "_mla_module", lambda: _TinyDsparkModule)
    logs: list[str] = []
    verify_out = tmp_path / "verify_out"
    draft_src = tmp_path / "draft12345678"
    draft_src.mkdir()
    assert (
        preshard.run(
            _parse_cli(
                [
                    "k3",
                    "--checkpoint",
                    "tiny",
                    "--output",
                    str(verify_out),
                    "--ranks",
                    "0",
                    "--in-flight",
                    "1",
                ]
            ),
            logs.append,
        )
        == 0
    )
    draft_dir = preshard.draft_directory(str(verify_out), str(draft_src))
    draft_dir.mkdir(parents=True)
    draft_weight = _TinyDsparkModule.load_checkpoint_weights(draft_src, draft_src)["w"][
        0
    ]
    (draft_dir / "rank00.bin").write_bytes(np.ascontiguousarray(draft_weight).tobytes())
    (draft_dir / "index.json").write_text(
        json.dumps(
            {
                "arrays": {
                    "w": {
                        "offset": 0,
                        "shape": list(draft_weight.shape),
                        "dtype": str(draft_weight.dtype),
                    }
                }
            }
        )
    )

    verify_args = _parse_cli(
        [
            "verify",
            "--against",
            str(verify_out),
            "--checkpoint",
            "tiny",
            "--draft",
            str(draft_src),
            "--rank",
            "0",
            "--layers",
            "1",
        ]
    )
    assert preshard.run(verify_args, logs.append) == 0

    dense_file = verify_out / "rank00" / "dense.bin"
    corrupted = bytearray(dense_file.read_bytes())
    corrupted[0] ^= 0xFF
    dense_file.write_bytes(bytes(corrupted))
    assert preshard.run(verify_args, logs.append) == 1
