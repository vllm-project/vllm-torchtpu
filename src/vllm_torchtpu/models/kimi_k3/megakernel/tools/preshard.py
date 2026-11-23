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

# Source provenance: inferact/k3-megakernel@30c9ba966b tools/megakernel/preshard.py
"""Pre-sharding subcommand helpers (``add_arguments`` and ``run``).

Exposes the ``k3``, ``prefill``, ``dspark``, and ``verify`` subcommands for
writing and validating checkpoints in the kernels' per-rank layout.

The HF checkpoint is organized by layer (every expert of a layer spread over
97 safetensors shards), so loading it directly reads nearly all 1.5 TB on
every host. ``k3`` reads it once and writes, per rank, the arrays the decode
program loads
(``vllm_torchtpu.models.kimi_k3.megakernel.kernels.rank_loader`` streams them
onto the TPU):

    DIR/layout.json
    DIR/rank00/index.json           file, offset, shape and dtype per family
    DIR/rank00/dense.bin            every non-expert family, back to back
    DIR/rank00/expert_gu_{0..3}.bin packed MXFP4 gate/up, by layer range
    DIR/rank00/expert_gus.bin       gate/up E8M0 scales
    DIR/rank00/expert_down_{0,1}.bin, expert_ds.bin

``prefill`` writes ``DIR/prefill-sp-cache/layout.json`` and six BF16 family
files for layers 1..92. Latent projections are replicated; shared-expert
channels are grouped by rank parity. It runs on ``--host 0`` without using a
TPU, keeping only one layer in memory, and refuses to overwrite an existing
cache directory.

Each ``k3`` invocation converts the eight ranks assigned to ``--host`` (or
``--ranks`` when specified) without using a TPU. ``dspark`` writes the
speculator's shards via ``kernels.dspark.save_kernel_shards`` or
``kernels.dspark_mla.save_kernel_shards`` into
``DIR/dspark-<revision[:8]>``, where ``DsparkDraft`` and ``MlaDsparkDraft``
look for them. ``verify`` rebuilds a sample of one rank (every dense family
at the given layers, the expert families at the given layers, and the
speculator's rank file) and compares it byte for byte with an existing
pre-shard.
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Final, Protocol, cast

import numpy as np

from ..common.constants import (
    DEFAULT_VOCAB_SIZE,
    EXPERT_FAMILIES,
    HIDDEN_SIZE,
    KERNEL_SHARD_FORMAT,
    NUM_LAYERS,
    REV_PREFIX_LEN,
    TP_SIZE,
)
from ..common.paths import (
    atomic_write_text,
    require_local_path,
    staged_directory,
)
from ..kernels.rank_loader import LEGACY_KDA_GATE_START, _dtype

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable, Collection, Sequence
    from pathlib import Path

    from ..kernels import Config
    from ..kernels import dspark as _default_dspark
    from ..kernels import dspark_mla as _default_dspark_mla
    from ..kernels.rank_loader import FamilySpec, RankIndex

GU_FILES = 4  # gate/up split into layer ranges
DOWN_FILES = 2
# 2x2x4 TPU v7x: 4 hosts x 8 devices/host (4 dual-core chips/host) = TP_SIZE ranks.
RANKS_PER_HOST = 8
HOSTS = TP_SIZE // RANKS_PER_HOST
_MIN_LAYERS = 2
_DEFAULT_IN_FLIGHT = 4
DEFAULT_MAX_POSITION = 16384
_DEFAULT_VERIFY_RANK = 5
_DEFAULT_VERIFY_LAYERS = "1,46,92"
# Kimi-K3's 3584-wide latent projection (7 * 512) pads by 1/7 (512 -> 4096 = 8 * 512)
# so 8-way sequence-parallel sharding divides the padded latent width evenly.
_LATENT_UNPADDED_PARTS = 7
EXPERT_DTYPES: dict[str, type[np.generic]] = {
    "expert_gu": np.uint32,
    "expert_gus": np.uint8,
    "expert_down": np.uint32,
    "expert_ds": np.uint8,
}


class CheckpointConfig(Protocol):
    """Structural protocol for Kimi-K3 checkpoint geometry metadata."""

    @property
    def dim(self) -> int:
        """Return hidden dimension."""
        ...

    @property
    def latent(self) -> int:
        """Return latent projection width."""
        ...

    @property
    def shared_hidden(self) -> int:
        """Return shared expert intermediate width."""
        ...

    @property
    def vocab(self) -> int:
        """Return vocabulary size."""
        ...

    @property
    def experts(self) -> int:
        """Return total routed expert count."""
        ...

    @property
    def full_attention(self) -> Collection[int]:
        """Return full-attention layer indices."""
        ...


class CheckpointProtocol[ConfigT: CheckpointConfig](Protocol):
    """Structural protocol for Kimi-K3 safetensors checkpoint readers."""

    READ_THREADS: Final[int]

    @property
    def config(self) -> ConfigT:
        """Return checkpoint geometry metadata."""
        ...

    def read(self, key: str) -> np.ndarray:
        """Read a checkpoint tensor by key."""
        ...

    def dense_selected(self, name: str, layer: int, rank: int) -> np.ndarray:
        """Return one layer of a dense parameter family for rank."""
        ...

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
        """Return packed MXFP4 expert weights and scales for layer."""
        ...


class LoadModuleProtocol[ConfigT: CheckpointConfig](Protocol):
    """Structural protocol for the host-side Kimi-K3 load module."""

    PREFIX: Final[str]
    BF16: Final[np.dtype[np.generic]]
    Checkpoint: Final[Callable[[str | Path], CheckpointProtocol[ConfigT]]]

    def selected_local_shapes(
        self,
        layers: int = NUM_LAYERS,
        hidden: int = HIDDEN_SIZE,
        *,
        vocab: int = DEFAULT_VOCAB_SIZE,
    ) -> dict[str, tuple[int, ...]]:
        """Return per-rank tensor shapes for the kernel layout."""
        ...

    def family_layers(
        self,
        config: ConfigT,
        name: str,
        layers: int,
    ) -> list[int] | None:
        """Return active layer indices for parameter family ``name``, or None."""
        ...


class DraftConfigLoader[DraftConfigT](Protocol):
    """Structural protocol for draft config classes with from_checkpoint."""

    def from_checkpoint(self, directory: Path) -> DraftConfigT:
        """Load draft configuration from checkpoint directory."""
        ...


class DsparkModuleProtocol[DraftConfigT, WeightsT](Protocol):
    """Structural protocol for the DSpark draft checkpoint module."""

    DraftConfig: Final[DraftConfigLoader[DraftConfigT]]

    def load_checkpoint_weights(
        self, directory: Path, config: DraftConfigT, /
    ) -> WeightsT:
        """Load DSpark checkpoint weights from directory."""
        ...

    def kernel_weight_shards(
        self,
        config: DraftConfigT,
        weights: WeightsT,
        /,
        *,
        max_position: int = DEFAULT_MAX_POSITION,
    ) -> dict[str, np.ndarray]:
        """Build per-rank kernel-layout weight shards."""
        ...

    def save_kernel_shards(
        self, shards: dict[str, np.ndarray], directory: Path
    ) -> None:
        """Write per-rank kernel shards and index.json to directory."""
        ...


class MlaModuleProtocol[DraftConfigT, WeightsT](Protocol):
    """Structural protocol for the MLA-DSpark draft checkpoint module."""

    MlaDraftConfig: Final[DraftConfigLoader[DraftConfigT]]

    def is_mla_draft(self, directory: Path) -> bool:
        """Return True when directory contains an MLA-DSpark checkpoint."""
        ...

    def load_checkpoint_weights(
        self, directory: Path, config: DraftConfigT, /
    ) -> WeightsT:
        """Load MLA-DSpark checkpoint weights from directory."""
        ...

    def kernel_weight_shards(
        self, config: DraftConfigT, weights: WeightsT, /
    ) -> dict[str, np.ndarray]:
        """Build per-rank MLA-DSpark kernel-layout weight shards."""
        ...

    def save_kernel_shards(
        self, shards: dict[str, np.ndarray], directory: Path
    ) -> None:
        """Write per-rank MLA-DSpark kernel shards to directory."""
        ...


@dataclasses.dataclass(frozen=True, slots=True)
class ConvertContext:
    """Destination directory, logger, and layer count for rank conversion."""

    output: Path
    log: Callable[[str], None]
    layers: int = NUM_LAYERS


@dataclasses.dataclass(frozen=True, slots=True)
class WriteK3Options[ConfigT: CheckpointConfig]:
    """Rank selection, concurrency, host index, and loader override for write_k3."""

    ranks_spec: str = ""
    in_flight: int = _DEFAULT_IN_FLIGHT
    host: int = 0
    load: LoadModuleProtocol[ConfigT] | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class VerifyOptions[ConfigT: CheckpointConfig, DraftConfigT, WeightsT]:
    """Sampled rank, layer list, and module overrides for verification."""

    rank: int = _DEFAULT_VERIFY_RANK
    layers_spec: str = _DEFAULT_VERIFY_LAYERS
    load: LoadModuleProtocol[ConfigT] | None = None
    dspark: DsparkModuleProtocol[DraftConfigT, WeightsT] | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class _VerifySample:
    """Target rank, sampled layer list, and logger for rank verification."""

    rank: int
    layers: list[int]
    log: Callable[[str], None]


@dataclasses.dataclass(frozen=True, slots=True)
class _DenseWriteState:
    """Per-rank destination state shared while writing dense families."""

    rank: int
    layers: int
    rank_dir: Path
    index: RankIndex


@dataclasses.dataclass(frozen=True, slots=True)
class _VerifyContext:
    """Resolved rank metadata and logger shared by dense and expert verification."""

    rank_dir: Path
    rank: int
    layers: list[int]
    total_layers: int
    legacy: bool
    families: dict[str, FamilySpec]
    shapes: dict[str, Sequence[int]]
    log: Callable[[str], None]


def _latent_pad_cols(latent: int) -> int:
    """Return zero-padding columns that expand a 7-part latent width to 8 parts."""
    if latent % _LATENT_UNPADDED_PARTS != 0:
        msg = f"latent width {latent} must be divisible by {_LATENT_UNPADDED_PARTS}"
        raise ValueError(msg)
    return latent // _LATENT_UNPADDED_PARTS


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register k3, prefill, dspark, and verify subcommands on parser."""
    sub = parser.add_subparsers(dest="target", required=True)
    model = sub.add_parser("k3", help="the Kimi-K3 checkpoint")
    model.add_argument("--checkpoint", required=True, help="the Kimi-K3 checkpoint")
    model.add_argument("--output", required=True)
    model.add_argument(
        "--ranks",
        default="",
        help="ranks to convert (default: the eight of this host)",
    )
    model.add_argument(
        "--in-flight",
        type=int,
        default=_DEFAULT_IN_FLIGHT,
        help="ranks converted concurrently per host",
    )
    model.add_argument(
        "--host",
        type=int,
        default=0,
        help=f"host index in [0, {HOSTS}) for multi-host presharding (default: 0)",
    )
    prefill = sub.add_parser("prefill", help="the Kimi-K3 prefill projection cache")
    prefill.add_argument("--checkpoint", required=True, help="the Kimi-K3 checkpoint")
    prefill.add_argument("--output", required=True)
    prefill.add_argument(
        "--layers",
        type=int,
        default=NUM_LAYERS,
        help=f"number of layers to convert (default: {NUM_LAYERS})",
    )
    prefill.add_argument(
        "--host",
        type=int,
        default=0,
        help=(
            f"host index in [0, {HOSTS}); only host 0 writes the prefill cache "
            "(non-zero hosts log and skip; default: 0)"
        ),
    )
    draft = sub.add_parser("dspark", help="the DSpark speculator")
    draft.add_argument(
        "--draft", required=True, help="the DSpark speculator checkpoint"
    )
    draft.add_argument("--output", required=True)
    draft.add_argument(
        "--max-position",
        type=int,
        default=DEFAULT_MAX_POSITION,
        help="RoPE table length (the loader regenerates it)",
    )
    verify_parser = sub.add_parser(
        "verify", help="rebuild a sample and compare it with a pre-shard"
    )
    verify_parser.add_argument(
        "--against", required=True, help="the pre-shard to compare"
    )
    verify_parser.add_argument(
        "--checkpoint", required=True, help="the Kimi-K3 checkpoint"
    )
    verify_parser.add_argument(
        "--draft", required=True, help="the DSpark speculator checkpoint"
    )
    verify_parser.add_argument("--rank", type=int, default=_DEFAULT_VERIFY_RANK)
    verify_parser.add_argument(
        "--layers", default=_DEFAULT_VERIFY_LAYERS, help="checkpoint layers to compare"
    )


def _modules() -> tuple[
    LoadModuleProtocol[Config],
    DsparkModuleProtocol[_default_dspark.DraftConfig, dict[str, np.ndarray]],
]:
    """Return the host-side checkpoint readers without loading torch_tpu."""
    load_mod = importlib.import_module(
        "vllm_torchtpu.models.kimi_k3.megakernel.kernels.load"
    )
    dspark_mod = importlib.import_module(
        "vllm_torchtpu.models.kimi_k3.megakernel.kernels.dspark"
    )
    return (
        cast("LoadModuleProtocol[Config]", load_mod),
        cast(
            "DsparkModuleProtocol[_default_dspark.DraftConfig, dict[str, np.ndarray]]",
            dspark_mod,
        ),
    )


def _layer_ranges(layers: int, files: int) -> list[tuple[int, int]]:
    """Partition layers into files contiguous (lo, hi) layer ranges."""
    edges = [round(i * layers / files) for i in range(files + 1)]
    return [(edges[i], edges[i + 1]) for i in range(files)]


def _experts_of(
    checkpoint: CheckpointProtocol[CheckpointConfig], rank: int
) -> tuple[int, int, int]:
    """Return ``(first expert, experts, quarter)`` for ``rank``.

    Describes the rank's slice of a layer's experts (8 expert groups of 4
    ranks, each rank holding a quarter of the intermediate dimension).
    """
    count = checkpoint.config.experts // 8
    return count * (rank // 4), count, rank % 4


def dense_value(
    checkpoint: CheckpointProtocol[CheckpointConfig],
    name: str,
    rank: int,
    layer: int | None,
) -> np.ndarray:
    """Return a dense family's rank slice at ``layer`` (or ``None`` if global)."""
    return checkpoint.dense_selected(name, 0 if layer is None else layer, rank)


def expert_values(
    checkpoint: CheckpointProtocol[CheckpointConfig], rank: int, layer: int
) -> dict[str, np.ndarray]:
    """Read and pack all four MXFP4 expert families for rank and layer."""
    first, count, part = _experts_of(checkpoint, rank)
    gate_up, gate_up_scales, downs, down_scale = checkpoint.experts(
        layer, first, count, part=part, parts=4
    )
    return {
        name: np.ascontiguousarray(np.asarray(value, EXPERT_DTYPES[name]))
        for name, value in zip(
            EXPERT_FAMILIES,
            (gate_up, gate_up_scales, downs, down_scale),
            strict=True,
        )
    }


def _write_dense_families[ConfigT: CheckpointConfig](
    load: LoadModuleProtocol[ConfigT],
    checkpoint: CheckpointProtocol[ConfigT],
    shapes: dict[str, tuple[int, ...]],
    state: _DenseWriteState,
) -> int:
    """Write all non-expert families into dense.bin and update index metadata."""
    rank = state.rank
    layers = state.layers
    index = state.index
    offset = 0
    with (state.rank_dir / "dense.bin").open("wb") as dense:
        for name, shape in shapes.items():
            if name in EXPERT_FAMILIES:
                continue
            ids = load.family_layers(checkpoint.config, name, layers)
            if ids is None:
                value = dense_value(checkpoint, name, rank, None)
            else:
                first = dense_value(checkpoint, name, rank, ids[0])
                value = np.empty((len(ids), *first.shape), first.dtype)
                value[0] = first
                with ThreadPoolExecutor(checkpoint.READ_THREADS) as pool:
                    for i, layer_value in enumerate(
                        pool.map(
                            lambda layer, name=name: dense_value(
                                checkpoint, name, rank, layer
                            ),
                            ids[1:],
                        ),
                        start=1,
                    ):
                        value[i] = layer_value
            if tuple(value.shape) != tuple(shape):
                msg = (
                    f"dense family {name!r} shape {tuple(value.shape)} "
                    f"!= expected {tuple(shape)}"
                )
                raise ValueError(msg)
            contiguous_value = np.ascontiguousarray(value)
            dense.write(contiguous_value.tobytes())
            index["families"][name] = {
                "file": "dense.bin",
                "offset": offset,
                "shape": list(contiguous_value.shape),
                "dtype": str(contiguous_value.dtype),
            }
            offset += contiguous_value.nbytes
    return offset


def _validate_rank(rank: int) -> int:
    """Reject rank indices outside the TP32 topology."""
    if isinstance(rank, bool) or not 0 <= rank < TP_SIZE:
        msg = f"rank {rank!r} out of range [0, {TP_SIZE})"
        raise ValueError(msg)
    return rank


def convert_rank[ConfigT: CheckpointConfig](
    load: LoadModuleProtocol[ConfigT],
    checkpoint: CheckpointProtocol[ConfigT],
    rank: int,
    context: ConvertContext,
) -> RankIndex:
    """Write one rank's pre-sharded files and return its index dictionary."""
    _validate_rank(rank)
    layers = context.layers
    if isinstance(layers, bool) or not _MIN_LAYERS <= layers <= NUM_LAYERS:
        msg = f"layers {layers!r} out of range [{_MIN_LAYERS}, {NUM_LAYERS}]"
        raise ValueError(msg)
    output = require_local_path(context.output, label="output")
    log = context.log
    shapes = load.selected_local_shapes(
        layers, checkpoint.config.dim, vocab=checkpoint.config.vocab
    )
    rank_dir = output / f"rank{rank:02d}"
    index: RankIndex = {
        "format": KERNEL_SHARD_FORMAT,
        "rank": rank,
        "layers": layers,
        "families": {},
    }
    started = time.perf_counter()

    expert_layers = layers - 1
    gu_files = min(GU_FILES, max(1, expert_layers))
    down_files = min(DOWN_FILES, max(1, expert_layers))
    ranges = {
        "expert_gu": _layer_ranges(expert_layers, gu_files),
        "expert_gus": [(0, expert_layers)],
        "expert_down": _layer_ranges(expert_layers, down_files),
        "expert_ds": [(0, expert_layers)],
    }
    files = {
        name: (
            [f"{name}_{k}.bin" for k in range(len(spans))]
            if len(spans) > 1
            else [f"{name}.bin"]
        )
        for name, spans in ranges.items()
    }
    with staged_directory(rank_dir, exist_ok=True) as staging:
        offset = _write_dense_families(
            load,
            checkpoint,
            shapes,
            _DenseWriteState(rank=rank, layers=layers, rank_dir=staging, index=index),
        )
        log(
            f"rank {rank}: dense families written ({offset / 1e9:.1f} GB) "
            f"after {time.perf_counter() - started:.0f} s"
        )
        with contextlib.ExitStack() as stack:
            handles = {
                name: [
                    stack.enter_context((staging / filename).open("wb"))
                    for filename in names
                ]
                for name, names in files.items()
            }
            for position, layer in enumerate(range(1, layers)):
                for name, array in expert_values(checkpoint, rank, layer).items():
                    if tuple(array.shape) != tuple(shapes[name][1:]):
                        msg = (
                            f"expert family {name!r} slice shape {tuple(array.shape)} "
                            f"!= expected {tuple(shapes[name][1:])}"
                        )
                        raise ValueError(msg)
                    k = next(
                        k
                        for k, (lo, hi) in enumerate(ranges[name])
                        if lo <= position < hi
                    )
                    handles[name][k].write(array.tobytes())
                if layer % 8 == 0 or layer == layers - 1:
                    log(
                        f"rank {rank}: expert layer {layer}/{layers - 1} "
                        f"after {time.perf_counter() - started:.0f} s"
                    )
        for name in EXPERT_FAMILIES:
            index["families"][name] = {
                "files": files[name],
                "layer_ranges": [list(r) for r in ranges[name]],
                "shape": list(shapes[name]),
                "dtype": np.dtype(EXPERT_DTYPES[name]).name,
            }
        (staging / "index.json").write_text(
            json.dumps(index, indent=1), encoding="utf-8"
        )
    log(f"rank {rank}: done in {time.perf_counter() - started:.0f} s")
    return index


def _validate_host(host: int) -> None:
    """Reject host indices outside the TP32 4-host topology."""
    if isinstance(host, bool) or not 0 <= host < HOSTS:
        msg = f"host {host!r} out of range [0, {HOSTS})"
        raise ValueError(msg)


def _parse_ranks_spec(ranks_spec: str, host: int) -> list[int]:
    """Parse and validate target ranks for k3 conversion."""
    if not ranks_spec:
        first_rank = RANKS_PER_HOST * host
        return list(range(first_rank, first_rank + RANKS_PER_HOST))
    ranks = [_validate_rank(int(part)) for part in ranks_spec.split(",")]
    if len(set(ranks)) != len(ranks):
        msg = f"duplicate ranks in {ranks_spec!r}"
        raise ValueError(msg)
    return ranks


def _write_k3_with_loader[ConfigT: CheckpointConfig](
    checkpoint_dir: Path | str,
    output_dir: Path | str,
    log: Callable[[str], None],
    options: WriteK3Options[ConfigT],
    resolved_load: LoadModuleProtocol[ConfigT],
) -> None:
    """Convert the Kimi-K3 checkpoint with a resolved loader module."""
    _validate_host(options.host)
    ranks = _parse_ranks_spec(options.ranks_spec, options.host)
    checkpoint_path = require_local_path(checkpoint_dir, label="checkpoint")
    output = require_local_path(output_dir, label="output")
    checkpoint = resolved_load.Checkpoint(checkpoint_path)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    context = ConvertContext(output=output, log=log)
    with ThreadPoolExecutor(max(1, options.in_flight)) as pool:
        list(
            pool.map(
                lambda rank: convert_rank(resolved_load, checkpoint, rank, context),
                ranks,
            )
        )
    if options.host == 0:
        atomic_write_text(
            output / "layout.json",
            json.dumps(
                {
                    "format": KERNEL_SHARD_FORMAT,
                    "checkpoint": str(checkpoint_path),
                    "layers": NUM_LAYERS,
                    "ranks": TP_SIZE,
                    "gate_up_files": GU_FILES,
                    "down_files": DOWN_FILES,
                },
                indent=1,
            ),
        )
    log(f"ranks {ranks} converted in {time.perf_counter() - started:.0f} s")


def write_k3[ConfigT: CheckpointConfig](
    checkpoint_dir: Path | str,
    output_dir: Path | str,
    log: Callable[[str], None],
    options: WriteK3Options[ConfigT] | None = None,
) -> None:
    """Convert the Kimi-K3 checkpoint into per-rank pre-shard directories."""
    if options is None:
        default_options: WriteK3Options[Config] = WriteK3Options()
        _write_k3_with_loader(
            checkpoint_dir, output_dir, log, default_options, _modules()[0]
        )
        return
    if options.load is None:
        forwarded_options: WriteK3Options[Config] = WriteK3Options(
            ranks_spec=options.ranks_spec,
            in_flight=options.in_flight,
            host=options.host,
        )
        _write_k3_with_loader(
            checkpoint_dir, output_dir, log, forwarded_options, _modules()[0]
        )
        return
    _write_k3_with_loader(checkpoint_dir, output_dir, log, options, options.load)


def _mla_module() -> MlaModuleProtocol[
    _default_dspark_mla.MlaDraftConfig, dict[str, np.ndarray]
]:
    """Return the DSpark MLA draft checkpoint module lazily."""
    mla_mod = importlib.import_module(
        "vllm_torchtpu.models.kimi_k3.megakernel.kernels.dspark_mla"
    )
    return cast(
        "MlaModuleProtocol[_default_dspark_mla.MlaDraftConfig, dict[str, np.ndarray]]",
        mla_mod,
    )


def prefill_projection_values[ConfigT: CheckpointConfig](
    load: LoadModuleProtocol[ConfigT],
    checkpoint: CheckpointProtocol[ConfigT],
    layer: int,
) -> dict[str, np.ndarray]:
    """Extract and transpose prefill latent and shared-expert projections for layer."""
    stem = f"{load.PREFIX}layers.{layer}.block_sparse_moe."
    routed_down_transposed = checkpoint.read(stem + "routed_expert_down_proj.weight").T
    pad_cols = _latent_pad_cols(routed_down_transposed.shape[1])
    values: dict[str, np.ndarray] = {
        "latent_down": np.ascontiguousarray(
            np.pad(routed_down_transposed, ((0, 0), (0, pad_cols)))
        ),
        "latent_up": np.ascontiguousarray(
            checkpoint.read(stem + "routed_expert_up_proj.weight").T
        ),
    }
    shared = {
        name: checkpoint.read(stem + f"shared_experts.{name}_proj.weight")
        for name in ("gate", "up", "down")
    }
    shared_channels = shared["gate"].shape[0]
    if shared_channels % TP_SIZE != 0:
        msg = f"shared expert channels {shared_channels} must be divisible by {TP_SIZE}"
        raise ValueError(msg)
    per_rank = shared_channels // TP_SIZE
    for parity in (0, 1):
        channels = np.concatenate(
            [
                np.arange(rank * per_rank, (rank + 1) * per_rank)
                for rank in range(parity, TP_SIZE, 2)
            ]
        )
        values[f"shared_gu_{parity}"] = np.ascontiguousarray(
            np.concatenate(
                (shared["gate"][channels].T, shared["up"][channels].T), axis=1
            )
        )
        values[f"shared_down_{parity}"] = np.ascontiguousarray(
            shared["down"][:, channels].T
        )
    return values


def _write_prefill_with_loader[ConfigT: CheckpointConfig](
    checkpoint_dir: Path | str,
    output_dir: Path | str,
    log: Callable[[str], None],
    layers: int,
    resolved_load: LoadModuleProtocol[ConfigT],
) -> None:
    """Write the sequence-parallel prefill projection cache with a resolved loader."""
    total_layers = layers
    if isinstance(total_layers, bool) or not _MIN_LAYERS <= total_layers <= NUM_LAYERS:
        msg = f"layers {total_layers!r} out of range [{_MIN_LAYERS}, {NUM_LAYERS}]"
        raise ValueError(msg)
    checkpoint_path = require_local_path(checkpoint_dir, label="checkpoint")
    output_root = require_local_path(output_dir, label="output")
    checkpoint = resolved_load.Checkpoint(checkpoint_path)
    hidden = checkpoint.config.dim
    latent = checkpoint.config.latent
    latent_padded = latent + _latent_pad_cols(latent)
    if checkpoint.config.shared_hidden % TP_SIZE != 0:
        msg = (
            f"shared_hidden {checkpoint.config.shared_hidden} "
            f"must be divisible by {TP_SIZE}"
        )
        raise ValueError(msg)
    shared_half = checkpoint.config.shared_hidden // 2
    output = output_root / "prefill-sp-cache"
    shapes = {
        "latent_down": (hidden, latent_padded),
        "latent_up": (latent, hidden),
        "shared_gu_0": (hidden, 2 * shared_half),
        "shared_gu_1": (hidden, 2 * shared_half),
        "shared_down_0": (shared_half, hidden),
        "shared_down_1": (shared_half, hidden),
    }
    layout = {
        "format": KERNEL_SHARD_FORMAT,
        "layers": total_layers,
        "checkpoint": str(checkpoint_path),
        "families": {
            name: {
                "file": f"{name}.bin",
                "shape": [total_layers - 1, *shape],
                "dtype": "bfloat16",
            }
            for name, shape in shapes.items()
        },
    }
    started = time.perf_counter()
    with staged_directory(output, exist_ok=False) as staging:
        with contextlib.ExitStack() as stack:
            handles = {
                name: stack.enter_context((staging / f"{name}.bin").open("xb"))
                for name in shapes
            }
            for layer in range(1, total_layers):
                layer_values = prefill_projection_values(
                    resolved_load, checkpoint, layer
                )
                for name, value in layer_values.items():
                    if value.shape != shapes[name] or value.dtype != resolved_load.BF16:
                        msg = (
                            f"prefill family {name!r} layer {layer}: "
                            f"shape {value.shape}, dtype {value.dtype}"
                        )
                        raise ValueError(msg)
                    handles[name].write(value.tobytes())
                if layer % 8 == 0 or layer == total_layers - 1:
                    log(
                        f"prefill projections: layer {layer}/{total_layers - 1} "
                        f"after {time.perf_counter() - started:.0f} s"
                    )
        (staging / "layout.json").write_text(
            json.dumps(layout, indent=2) + "\n", encoding="utf-8"
        )
    log(
        f"prefill projections written to {output} "
        f"in {time.perf_counter() - started:.0f} s"
    )


def write_prefill[ConfigT: CheckpointConfig](
    checkpoint_dir: Path | str,
    output_dir: Path | str,
    log: Callable[[str], None],
    *,
    layers: int = NUM_LAYERS,
    load: LoadModuleProtocol[ConfigT] | None = None,
) -> None:
    """Write the sequence-parallel prefill projection cache directory."""
    if load is None:
        _write_prefill_with_loader(
            checkpoint_dir, output_dir, log, layers, _modules()[0]
        )
        return
    _write_prefill_with_loader(checkpoint_dir, output_dir, log, layers, load)


def _select_draft_shards[DraftConfigT, WeightsT](
    dspark: DsparkModuleProtocol[DraftConfigT, WeightsT],
    draft: Path | str,
    max_position: int,
) -> tuple[
    dict[str, np.ndarray],
    Callable[[dict[str, np.ndarray], Path], None],
]:
    """Build draft kernel shards and return them paired with their writer."""
    draft_path = require_local_path(draft, label="draft")
    mla = _mla_module()
    if mla.is_mla_draft(draft_path):
        mla_config = mla.MlaDraftConfig.from_checkpoint(draft_path)
        shards = mla.kernel_weight_shards(
            mla_config, mla.load_checkpoint_weights(draft_path, mla_config)
        )
        return shards, mla.save_kernel_shards
    config = dspark.DraftConfig.from_checkpoint(draft_path)
    weights = dspark.load_checkpoint_weights(draft_path, config)
    return (
        dspark.kernel_weight_shards(config, weights, max_position=max_position),
        dspark.save_kernel_shards,
    )


def draft_shards[DraftConfigT, WeightsT](
    dspark: DsparkModuleProtocol[DraftConfigT, WeightsT],
    draft: Path | str,
    max_position: int,
) -> dict[str, np.ndarray]:
    """Build the draft model's kernel-layout weight shards.

    Dispatches between ``vllm_torchtpu.models.kimi_k3.megakernel.kernels.dspark_mla``
    for MLA drafts and ``vllm_torchtpu.models.kimi_k3.megakernel.kernels.dspark``.
    """
    return _select_draft_shards(dspark, draft, max_position)[0]


def draft_directory(output: Path | str, draft: Path | str) -> Path:
    """Return the output directory path for pre-sharded DSpark draft weights."""
    output_path = require_local_path(output, label="output")
    draft_path = require_local_path(draft, label="draft")
    return output_path / f"dspark-{draft_path.resolve().name[:REV_PREFIX_LEN]}"


def _write_dspark_with_module[DraftConfigT, WeightsT](
    draft: Path | str,
    output: Path | str,
    log: Callable[[str], None],
    max_position: int,
    resolved_dspark: DsparkModuleProtocol[DraftConfigT, WeightsT],
) -> None:
    """Convert a DSpark or MLA-DSpark draft checkpoint with a resolved module."""
    started = time.perf_counter()
    output_path = require_local_path(output, label="output")
    directory = draft_directory(output_path, draft)
    shards, save_kernel_shards = _select_draft_shards(
        resolved_dspark, draft, max_position
    )
    log(f"kernel layout built in {time.perf_counter() - started:.0f} s")
    save_kernel_shards(shards, directory)
    log(f"written to {directory} in {time.perf_counter() - started:.0f} s")


def write_dspark[DraftConfigT, WeightsT](
    draft: Path | str,
    output: Path | str,
    log: Callable[[str], None],
    *,
    max_position: int = DEFAULT_MAX_POSITION,
    dspark: DsparkModuleProtocol[DraftConfigT, WeightsT] | None = None,
) -> None:
    """Convert a DSpark or MLA-DSpark draft checkpoint into per-rank shards."""
    if dspark is None:
        _write_dspark_with_module(draft, output, log, max_position, _modules()[1])
        return
    _write_dspark_with_module(draft, output, log, max_position, dspark)


def _stored(
    path: Path,
    offset: int,
    count: int,
    dtype: str | np.dtype[np.generic] | type[np.generic],
) -> np.ndarray:
    """Read count elements of dtype from path starting at byte offset."""
    resolved_dtype = _dtype(dtype)
    return np.fromfile(
        path,
        dtype=np.uint8,
        count=count * resolved_dtype.itemsize,
        offset=offset,
    ).view(resolved_dtype)


def _verify_dense_samples[ConfigT: CheckpointConfig](
    load: LoadModuleProtocol[ConfigT],
    checkpoint: CheckpointProtocol[ConfigT],
    compare: Callable[[str, np.ndarray, np.ndarray], None],
    context: _VerifyContext,
) -> bool:
    """Verify sampled dense weight slices against stored rank pre-shards."""
    ok = True
    for name, spec in context.families.items():
        if name in EXPERT_FAMILIES:
            continue
        if list(spec["shape"]) != list(context.shapes[name]):
            context.log(
                f"{name}: stored shape {spec['shape']}, loader {context.shapes[name]}"
            )
            ok = False
            continue
        ids = load.family_layers(checkpoint.config, name, context.total_layers)
        per_item = int(np.prod(spec["shape"][1:] if ids else spec["shape"]))
        for layer in (
            [None]
            if ids is None
            else [x for x in context.layers if x in ids] or ids[:1]
        ):
            position = 0 if layer is None or ids is None else ids.index(layer)
            itemsize = _dtype(spec["dtype"]).itemsize
            stored = _stored(
                context.rank_dir / spec["file"],
                spec["offset"] + position * per_item * itemsize,
                per_item,
                spec["dtype"],
            )
            rebuilt = dense_value(checkpoint, name, context.rank, layer)
            if context.legacy and name == "k_projection":
                gate = dense_value(checkpoint, "k_gate", context.rank, layer)
                rebuilt = np.concatenate(
                    (
                        rebuilt[..., :LEGACY_KDA_GATE_START],
                        gate,
                        rebuilt[..., LEGACY_KDA_GATE_START:],
                    ),
                    axis=-1,
                )
            compare(f"{name} layer {layer}", np.ascontiguousarray(rebuilt), stored)
    return ok


def _verify_expert_samples(
    checkpoint: CheckpointProtocol[CheckpointConfig],
    compare: Callable[[str, np.ndarray, np.ndarray], None],
    context: _VerifyContext,
) -> bool:
    """Verify sampled expert weight slices against stored rank pre-shards."""
    ok = True
    for name in EXPERT_FAMILIES:
        spec = context.families[name]
        if list(spec["shape"]) != list(context.shapes[name]):
            context.log(
                f"{name}: stored shape {spec['shape']}, loader {context.shapes[name]}"
            )
            ok = False
    if not ok:
        return False
    for layer in (x for x in context.layers if 1 <= x < context.total_layers):
        position = layer - 1
        rebuilt_experts = expert_values(checkpoint, context.rank, layer)
        for name in EXPERT_FAMILIES:
            spec = context.families[name]
            match = next(
                (
                    (shard_idx, span[0])
                    for shard_idx, span in enumerate(spec["layer_ranges"])
                    if span[0] <= position < span[1]
                ),
                None,
            )
            if match is None:
                context.log(f"{name}: missing layer range for position {position}")
                ok = False
                continue
            shard_idx, lo = match
            per_layer = int(np.prod(spec["shape"][1:]))
            itemsize = _dtype(spec["dtype"]).itemsize
            stored = _stored(
                context.rank_dir / spec["files"][shard_idx],
                (position - lo) * per_layer * itemsize,
                per_layer,
                spec["dtype"],
            )
            compare(f"{name} layer {layer}", rebuilt_experts[name], stored)
    return ok


def _verify_k3_rank[ConfigT: CheckpointConfig](
    resolved_load: LoadModuleProtocol[ConfigT],
    rank_dir: Path,
    checkpoint_dir: Path | str,
    compare: Callable[[str, np.ndarray, np.ndarray], None],
    sample: _VerifySample,
) -> bool | None:
    """Verify K3 dense and expert shards for one rank; None on family mismatch."""
    checkpoint = resolved_load.Checkpoint(checkpoint_dir)
    index_data: RankIndex = json.loads(
        (rank_dir / "index.json").read_text(encoding="utf-8")
    )
    if index_data.get("rank") != sample.rank:
        sample.log(
            f"rank mismatch: index has {index_data.get('rank')!r}, "
            f"expected {sample.rank}"
        )
        return None
    total_layers = index_data.get("layers", NUM_LAYERS)
    families: dict[str, FamilySpec] = index_data["families"]
    shapes: dict[str, Sequence[int]] = dict(
        resolved_load.selected_local_shapes(
            total_layers, checkpoint.config.dim, vocab=checkpoint.config.vocab
        )
    )
    # Older pre-shards store k_projection with the gate columns inside (the
    # loader splits them off): compare it combined.
    legacy = "k_gate" not in families
    if legacy:
        if "k_projection" in shapes and "k_gate" in shapes:
            proj_shape = list(shapes["k_projection"])
            gate_shape = list(shapes["k_gate"])
            shapes["k_projection"] = (*proj_shape[:-1], proj_shape[-1] + gate_shape[-1])
        shapes.pop("k_gate", None)
    if set(shapes) != set(families):
        sample.log(
            f"family sets differ: only in the loader "
            f"{sorted(set(shapes) - set(families))}, only in the pre-shard "
            f"{sorted(set(families) - set(shapes))}"
        )
        return None
    verify_context = _VerifyContext(
        rank_dir=rank_dir,
        rank=sample.rank,
        layers=sample.layers,
        total_layers=total_layers,
        legacy=legacy,
        families=families,
        shapes=shapes,
        log=sample.log,
    )
    dense_ok = _verify_dense_samples(
        resolved_load,
        checkpoint,
        compare,
        verify_context,
    )
    expert_ok = _verify_expert_samples(checkpoint, compare, verify_context)
    return dense_ok and expert_ok


def verify[ConfigT: CheckpointConfig, DraftConfigT, WeightsT](
    against: Path | str,
    checkpoint_dir: Path | str,
    draft: Path | str,
    log: Callable[[str], None],
    options: VerifyOptions[ConfigT, DraftConfigT, WeightsT] | None = None,
) -> bool:
    """Compare a rebuilt sample of the configured rank with ``against``."""
    rank = _DEFAULT_VERIFY_RANK if options is None else options.rank
    _validate_rank(rank)
    against_path = require_local_path(against, label="against")
    checkpoint_path = require_local_path(checkpoint_dir, label="checkpoint")
    draft_path = require_local_path(draft, label="draft")
    layers_spec = _DEFAULT_VERIFY_LAYERS if options is None else options.layers_spec
    rank_dir = against_path / f"rank{rank:02d}"
    layers = [int(x) for x in layers_spec.split(",")]
    ok = True

    def compare(what: str, rebuilt: np.ndarray, stored: np.ndarray) -> None:
        """Compare a rebuilt shard slice byte-for-byte against stored bytes."""
        nonlocal ok
        same = (
            rebuilt.dtype == stored.dtype
            and rebuilt.size == stored.size
            and np.array_equal(
                rebuilt.reshape(-1).view(np.uint8),
                stored.reshape(-1).view(np.uint8),
            )
        )
        ok &= same
        log(
            f"{what}: {'identical' if same else 'DIFFERS'} "
            f"({rebuilt.nbytes / 1e6:.1f} MB)"
        )

    sample = _VerifySample(rank=rank, layers=layers, log=log)
    if options is None or options.load is None:
        dense_ok = _verify_k3_rank(
            _modules()[0], rank_dir, checkpoint_path, compare, sample
        )
    else:
        dense_ok = _verify_k3_rank(
            options.load, rank_dir, checkpoint_path, compare, sample
        )
    if dense_ok is None:
        return False
    ok = ok and dense_ok

    directory = draft_directory(against_path, draft_path)
    index = json.loads((directory / "index.json").read_text(encoding="utf-8"))
    if options is None or options.dspark is None:
        shards = draft_shards(_modules()[1], draft_path, DEFAULT_MAX_POSITION)
    else:
        shards = draft_shards(options.dspark, draft_path, DEFAULT_MAX_POSITION)
    arrays_spec = index.get("arrays")
    if not isinstance(arrays_spec, dict) or set(arrays_spec) != set(shards):
        log(
            f"dspark arrays mismatch: index has {sorted(arrays_spec or ())}, "
            f"expected {sorted(shards)}"
        )
        return False
    raw = np.fromfile(directory / f"rank{rank:02d}.bin", dtype=np.uint8)
    for name, spec in arrays_spec.items():
        expected = np.ascontiguousarray(shards[name][rank])
        if list(spec["shape"]) != list(expected.shape):
            log(
                f"dspark {name}: stored shape {spec['shape']}, "
                f"expected {list(expected.shape)}"
            )
            ok = False
            continue
        resolved_dtype = _dtype(spec["dtype"])
        count = int(np.prod(spec["shape"])) * resolved_dtype.itemsize
        stored = raw[spec["offset"] : spec["offset"] + count].view(resolved_dtype)
        compare(f"dspark {name}", expected, stored)
    log(f"verify rank {rank}: {'all identical' if ok else 'MISMATCH'}")
    return ok


def run[ConfigT: CheckpointConfig, DraftConfigT, WeightsT](
    args: argparse.Namespace,
    log: Callable[[str], None],
    *,
    load: LoadModuleProtocol[ConfigT] | None = None,
    dspark: DsparkModuleProtocol[DraftConfigT, WeightsT] | None = None,
) -> int:
    """Dispatch the preshard CLI subcommand and return an exit status code."""
    if args.target == "k3":
        write_k3(
            args.checkpoint,
            args.output,
            log,
            WriteK3Options(
                ranks_spec=args.ranks,
                in_flight=args.in_flight,
                host=args.host,
                load=load,
            ),
        )
    elif args.target == "prefill":
        _validate_host(args.host)
        if args.host == 0:
            write_prefill(
                args.checkpoint,
                args.output,
                log,
                layers=args.layers,
                load=load,
            )
        else:
            log(f"prefill runs on host 0 only; skipping on host {args.host}")
    elif args.target == "dspark":
        write_dspark(
            args.draft,
            args.output,
            log,
            max_position=args.max_position,
            dspark=dspark,
        )
    else:
        return (
            0
            if verify(
                args.against,
                args.checkpoint,
                args.draft,
                log,
                VerifyOptions(
                    rank=args.rank,
                    layers_spec=args.layers,
                    load=load,
                    dspark=dspark,
                ),
            )
            else 1
        )
    return 0
