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

# Source provenance: inferact/k3-megakernel@30c9ba966b tools/megakernel/preshard.py
"""Write the checkpoints in the kernels' per-rank layout (``preshard``).

    preshard k3 --output DIR [--checkpoint DIR] [--ranks 0,1,..] [--in-flight 4]
    preshard prefill --output DIR --checkpoint DIR
    preshard dspark --output DIR [--draft DIR] [--max-position 16384]
    preshard verify [--against DIR] [--rank R] [--layers 1,46,92]

The HF checkpoint is organized by layer (every expert of a layer spread over
97 safetensors shards), so loading it directly reads nearly all 1.5 TB on
every host. ``k3`` reads it once and writes, per rank, the arrays the decode
program loads (``kernels.rank_loader`` streams them onto the TPU):

    DIR/layout.json
    DIR/rank00/index.json           file, offset, shape and dtype per family
    DIR/rank00/dense.bin            every non-expert family, back to back
    DIR/rank00/expert_gu_{0..3}.bin packed MXFP4 gate/up, by layer range
    DIR/rank00/expert_gus.bin       gate/up E8M0 scales
    DIR/rank00/expert_down_{0,1}.bin, expert_ds.bin

``prefill`` writes ``DIR/prefill-sp-cache/layout.json`` and six BF16 family
files for layers 1..92. Latent projections are replicated; shared-expert
channels are grouped by rank parity. It runs on the allocation's first host
without using a TPU, keeping only one layer in memory, and refuses to
overwrite an existing cache directory.

From the login node it runs on the allocation as one process per host (each
converts the eight ranks whose devices it owns; no TPU is used). ``dspark``
writes the speculator's shards (``kernels.dspark.save_kernel_shards``) into
``DIR/dspark-<revision[:8]>``, where ``DsparkDraft`` looks for them, and
runs where it is started. ``verify`` rebuilds a sample of one rank (every
dense family at the given layers, the expert families at the given layers,
and the speculator's rank file) and compares it byte for byte with an
existing pre-shard.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

from .. import settings

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable, Sequence


class CheckpointConfig(Protocol):
    """Structural protocol for Kimi-K3 checkpoint geometry metadata."""

    dim: int
    vocab: int
    experts: int
    full_attention: frozenset[int]


class CheckpointProtocol(Protocol):
    """Structural protocol for Kimi-K3 safetensors checkpoint readers."""

    READ_THREADS: int
    config: CheckpointConfig

    def read(self, key: str, selection: object = None) -> np.ndarray:
        """Read a checkpoint tensor or slice by key."""
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


class LoadModuleProtocol(Protocol):
    """Structural protocol for the host-side Kimi-K3 load module."""

    PREFIX: str
    BF16: np.dtype[Any]
    Checkpoint: Callable[[str | Path], CheckpointProtocol]

    def selected_local_shapes(
        self,
        layers: int = 93,
        hidden: int = 7168,
        *,
        vocab: int = 163840,
    ) -> dict[str, tuple[int, ...]]:
        """Return per-rank tensor shapes for the kernel layout."""
        ...


class DraftConfigLoader(Protocol):
    """Structural protocol for draft config classes with from_checkpoint."""

    def from_checkpoint(self, directory: Path) -> object:
        """Load draft configuration from checkpoint directory."""
        ...


class DsparkModuleProtocol(Protocol):
    """Structural protocol for the DSpark draft checkpoint module."""

    DraftConfig: DraftConfigLoader

    def load_checkpoint_weights(self, directory: Path, config: object) -> object:
        """Load DSpark checkpoint weights from directory."""
        ...

    def kernel_weight_shards(
        self,
        config: object,
        weights: object,
        *,
        max_position: int = 16384,
    ) -> dict[str, np.ndarray]:
        """Build per-rank kernel-layout weight shards."""
        ...

    def save_kernel_shards(
        self, shards: dict[str, np.ndarray], directory: Path
    ) -> None:
        """Write per-rank kernel shards and index.json to directory."""
        ...


class MlaModuleProtocol(Protocol):
    """Structural protocol for the MLA-DSpark draft checkpoint module."""

    MlaDraftConfig: DraftConfigLoader

    def is_mla_draft(self, directory: Path) -> bool:
        """Return True when directory contains an MLA-DSpark checkpoint."""
        ...

    def load_checkpoint_weights(self, directory: Path, config: object) -> object:
        """Load MLA-DSpark checkpoint weights from directory."""
        ...

    def kernel_weight_shards(
        self, config: object, weights: object
    ) -> dict[str, np.ndarray]:
        """Build per-rank MLA-DSpark kernel-layout weight shards."""
        ...

    def save_kernel_shards(
        self, shards: dict[str, np.ndarray], directory: Path
    ) -> None:
        """Write per-rank MLA-DSpark kernel shards to directory."""
        ...


FORMAT = 1
GU_FILES = 4  # gate/up split into layer ranges
DOWN_FILES = 2
EXPERT_FAMILIES = ("expert_gu", "expert_gus", "expert_down", "expert_ds")
EXPERT_DTYPES: dict[str, Any] = {
    "expert_gu": np.uint32,
    "expert_gus": np.uint8,
    "expert_down": np.uint32,
    "expert_ds": np.uint8,
}
LAYERS = 93


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
        default=4,
        help="ranks converted concurrently per host",
    )
    prefill = sub.add_parser("prefill", help="the Kimi-K3 prefill projection cache")
    prefill.add_argument("--checkpoint", required=True, help="the Kimi-K3 checkpoint")
    prefill.add_argument("--output", required=True)
    draft = sub.add_parser("dspark", help="the DSpark speculator")
    draft.add_argument(
        "--draft", required=True, help="the DSpark speculator checkpoint"
    )
    draft.add_argument("--output", required=True)
    draft.add_argument(
        "--max-position",
        type=int,
        default=16384,
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
    verify_parser.add_argument("--rank", type=int, default=5)
    verify_parser.add_argument(
        "--layers", default="1,46,92", help="checkpoint layers to compare"
    )


def runs_on_allocation(args: argparse.Namespace) -> bool:
    """Return True when the subcommand runs across the Slurm allocation."""
    return args.target in ("k3", "prefill")


def _modules() -> tuple[LoadModuleProtocol, DsparkModuleProtocol]:
    """Import the host-side checkpoint readers without loading torch_tpu."""
    package = __package__ or "vllm_torchtpu.models.kimi_k3.megakernel.tools"
    parent = package.rsplit(".", 1)[0]
    return cast(
        "tuple[LoadModuleProtocol, DsparkModuleProtocol]",
        (
            importlib.import_module(f"{parent}.kernels.load"),
            importlib.import_module(f"{parent}.kernels.dspark"),
        ),
    )


def _layer_ranges(layers: int, files: int) -> list[tuple[int, int]]:
    """Partition layers into files contiguous (lo, hi) layer ranges."""
    edges = [round(i * layers / files) for i in range(files + 1)]
    return [(edges[i], edges[i + 1]) for i in range(files)]


def _experts_of(checkpoint: CheckpointProtocol, rank: int) -> tuple[int, int, int]:
    """Return ``(first expert, experts, quarter)`` for ``rank``.

    Describes the rank's slice of a layer's experts (8 expert groups of 4
    ranks, each rank holding a quarter of the intermediate dimension).
    """
    count = checkpoint.config.experts // 8
    return count * (rank // 4), count, rank % 4


def dense_value(
    checkpoint: CheckpointProtocol, name: str, rank: int, layer: int | None
) -> np.ndarray:
    """Return a dense family's rank slice at checkpoint ``layer``."""
    return checkpoint.dense_selected(name, 0 if layer is None else layer, rank)


def expert_values(
    checkpoint: CheckpointProtocol, rank: int, layer: int
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


def _write_dense_families(
    checkpoint: CheckpointProtocol,
    family_layers_fn: Callable[..., list[int] | None],
    shapes: dict[str, tuple[int, ...]],
    rank_dir: Path,
    meta: tuple[int, int, dict[str, Any]],
) -> int:
    """Write all non-expert families into dense.bin and update index metadata."""
    rank, layers, index = meta
    offset = 0
    with (rank_dir / "dense.bin").open("wb") as dense:
        for name, shape in shapes.items():
            if name in EXPERT_FAMILIES:
                continue
            ids = family_layers_fn(checkpoint.config, name, layers)
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
                raise ValueError((name, value.shape, shape))
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


def convert_rank(
    load: LoadModuleProtocol,
    checkpoint: CheckpointProtocol,
    rank: int,
    output: Path,
    log: Callable[[str], None],
    **opts: int,
) -> dict[str, Any]:
    """Write one rank's pre-sharded files and return its index dictionary."""
    layers = opts.get("layers", LAYERS)
    shapes = load.selected_local_shapes(
        layers, checkpoint.config.dim, vocab=checkpoint.config.vocab
    )
    family_layers_fn: Callable[..., list[int] | None] = (
        vars(load).get("family_layers") or vars(load)["_family_layers"]
    )
    rank_dir = output / f"rank{rank:02d}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    index: dict[str, Any] = {
        "format": FORMAT,
        "rank": rank,
        "layers": layers,
        "families": {},
    }
    started = time.perf_counter()

    offset = _write_dense_families(
        checkpoint, family_layers_fn, shapes, rank_dir, (rank, layers, index)
    )
    log(
        f"rank {rank}: dense families written ({offset / 1e9:.1f} GB) "
        f"after {time.perf_counter() - started:.0f} s"
    )

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
    with contextlib.ExitStack() as stack:
        handles = {
            name: [
                stack.enter_context((rank_dir / filename).open("wb"))
                for filename in names
            ]
            for name, names in files.items()
        }
        for position, layer in enumerate(range(1, layers)):
            for name, array in expert_values(checkpoint, rank, layer).items():
                if tuple(array.shape) != tuple(shapes[name][1:]):
                    raise ValueError((name, array.shape, shapes[name]))
                k = next(
                    k for k, (lo, hi) in enumerate(ranges[name]) if lo <= position < hi
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
    (rank_dir / "index.json").write_text(json.dumps(index, indent=1))
    log(f"rank {rank}: done in {time.perf_counter() - started:.0f} s")
    return index


def write_k3(args: argparse.Namespace, log: Callable[[str], None]) -> None:
    """Convert the Kimi-K3 checkpoint into per-rank pre-shard directories."""
    load, _ = _modules()
    host = int(settings.SLURM_NODEID)
    ranks = (
        [int(r) for r in args.ranks.split(",")]
        if args.ranks
        else list(range(8 * host, 8 * host + 8))
    )
    checkpoint = load.Checkpoint(args.checkpoint)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if host == 0:
        (output / "layout.json").write_text(
            json.dumps(
                {
                    "format": FORMAT,
                    "checkpoint": str(args.checkpoint),
                    "layers": LAYERS,
                    "ranks": 32,
                    "gate_up_files": GU_FILES,
                    "down_files": DOWN_FILES,
                },
                indent=1,
            )
        )
    started = time.perf_counter()
    with ThreadPoolExecutor(max(1, args.in_flight)) as pool:
        list(
            pool.map(
                lambda rank: convert_rank(load, checkpoint, rank, output, log),
                ranks,
            )
        )
    log(f"ranks {ranks} converted in {time.perf_counter() - started:.0f} s")


def _mla_module() -> MlaModuleProtocol:
    """Import the DSpark MLA draft module lazily on demand."""
    package = __package__ or "vllm_torchtpu.models.kimi_k3.megakernel.tools"
    parent = package.rsplit(".", 1)[0]
    return cast(
        "MlaModuleProtocol", importlib.import_module(f"{parent}.kernels.dspark_mla")
    )


def prefill_projection_values(
    load: LoadModuleProtocol, checkpoint: CheckpointProtocol, layer: int
) -> dict[str, np.ndarray]:
    """Extract and transpose prefill latent and shared-expert projections for layer."""
    stem = f"{load.PREFIX}layers.{layer}.block_sparse_moe."
    down_t = checkpoint.read(stem + "routed_expert_down_proj.weight").T
    pad_cols = down_t.shape[1] // 7
    values: dict[str, np.ndarray] = {
        "latent_down": np.ascontiguousarray(np.pad(down_t, ((0, 0), (0, pad_cols)))),
        "latent_up": np.ascontiguousarray(
            checkpoint.read(stem + "routed_expert_up_proj.weight").T
        ),
    }
    shared = {
        name: checkpoint.read(stem + f"shared_experts.{name}_proj.weight")
        for name in ("gate", "up", "down")
    }
    per_rank = shared["gate"].shape[0] // 32
    for parity in (0, 1):
        channels = np.concatenate(
            [
                np.arange(rank * per_rank, (rank + 1) * per_rank)
                for rank in range(parity, 32, 2)
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


def write_prefill(args: argparse.Namespace, log: Callable[[str], None]) -> None:
    """Write the sequence-parallel prefill projection cache directory."""
    load: LoadModuleProtocol = vars(args).get("load_module") or _modules()[0]
    checkpoint = load.Checkpoint(args.checkpoint)
    total_layers = int(vars(args).get("layers") or LAYERS)
    hidden = checkpoint.config.dim
    latent = (hidden * 3584) // 7168
    latent_padded = latent + latent // 7
    shared_half = (hidden * 3072) // 7168
    output = Path(args.output) / "prefill-sp-cache"
    output.mkdir(parents=True, exist_ok=False)
    shapes = {
        "latent_down": (hidden, latent_padded),
        "latent_up": (latent, hidden),
        "shared_gu_0": (hidden, 2 * shared_half),
        "shared_gu_1": (hidden, 2 * shared_half),
        "shared_down_0": (shared_half, hidden),
        "shared_down_1": (shared_half, hidden),
    }
    layout = {
        "format": FORMAT,
        "layers": total_layers,
        "checkpoint": str(args.checkpoint),
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
    with contextlib.ExitStack() as stack:
        handles = {
            name: stack.enter_context((output / f"{name}.bin").open("xb"))
            for name in shapes
        }
        for layer in range(1, total_layers):
            layer_values = prefill_projection_values(load, checkpoint, layer)
            for name, value in layer_values.items():
                if value.shape != shapes[name] or value.dtype != load.BF16:
                    raise ValueError((name, layer, value.shape, value.dtype))
                handles[name].write(value.tobytes())
            if layer % 8 == 0 or layer == total_layers - 1:
                log(
                    f"prefill projections: layer {layer}/{total_layers - 1} "
                    f"after {time.perf_counter() - started:.0f} s"
                )
    (output / "layout.json").write_text(json.dumps(layout, indent=2) + "\n")
    log(
        f"prefill projections written to {output} "
        f"in {time.perf_counter() - started:.0f} s"
    )


def draft_shards(
    dspark: DsparkModuleProtocol, draft: str, max_position: int
) -> dict[str, np.ndarray]:
    """Build the draft model's kernel-layout weight shards."""
    mla = _mla_module()
    if mla.is_mla_draft(Path(draft)):
        mla_config = mla.MlaDraftConfig.from_checkpoint(Path(draft))
        return mla.kernel_weight_shards(
            mla_config, mla.load_checkpoint_weights(Path(draft), mla_config)
        )
    config = dspark.DraftConfig.from_checkpoint(Path(draft))
    weights = dspark.load_checkpoint_weights(Path(draft), config)
    return dspark.kernel_weight_shards(config, weights, max_position=max_position)


def draft_directory(output: str, draft: str) -> Path:
    """Return the output directory path for pre-sharded DSpark draft weights."""
    return Path(output) / f"dspark-{Path(draft).resolve().name[:8]}"


def write_dspark(args: argparse.Namespace, log: Callable[[str], None]) -> None:
    """Convert a DSpark or MLA-DSpark draft checkpoint into per-rank shards."""
    _, dspark = _modules()
    started = time.perf_counter()
    shards = draft_shards(dspark, args.draft, args.max_position)
    log(f"kernel layout built in {time.perf_counter() - started:.0f} s")
    directory = draft_directory(args.output, args.draft)
    mla = _mla_module()
    (mla if mla.is_mla_draft(Path(args.draft)) else dspark).save_kernel_shards(
        shards, directory
    )
    log(f"written to {directory} in {time.perf_counter() - started:.0f} s")


def _stored(
    path: Path,
    offset: int,
    count: int,
    dtype: str | np.dtype[Any] | type[np.generic],
) -> np.ndarray:
    """Read count elements of dtype from path starting at byte offset."""
    resolved_dtype = np.dtype(dtype)
    return np.fromfile(
        path,
        dtype=np.uint8,
        count=count * resolved_dtype.itemsize,
        offset=offset,
    ).view(resolved_dtype)


def _verify_dense_samples(
    checkpoint: CheckpointProtocol,
    family_layers_fn: Callable[..., list[int] | None],
    compare: Callable[[str, np.ndarray, np.ndarray], None],
    meta: tuple[Path, int, list[int], bool, dict[str, Any], dict[str, Any]],
) -> bool:
    """Verify sampled dense weight slices against stored rank pre-shards."""
    rank_dir, rank, layers, legacy, families, shapes = meta
    total_layers = int(families.get("attn_norm", {}).get("shape", [LAYERS])[0])
    ok = True
    for name, spec in families.items():
        if name in EXPERT_FAMILIES:
            continue
        if list(spec["shape"]) != list(shapes[name]):
            ok = False
            continue
        ids = family_layers_fn(checkpoint.config, name, total_layers)
        per_item = int(np.prod(spec["shape"][1:] if ids else spec["shape"]))
        for layer in (
            [None] if ids is None else [x for x in layers if x in ids] or ids[:1]
        ):
            position = 0 if layer is None or ids is None else ids.index(layer)
            itemsize = np.dtype(spec["dtype"]).itemsize
            stored = _stored(
                rank_dir / spec["file"],
                spec["offset"] + position * per_item * itemsize,
                per_item,
                spec["dtype"],
            )
            rebuilt = dense_value(checkpoint, name, rank, layer)
            if legacy and name == "k_projection":
                gate = dense_value(checkpoint, "k_gate", rank, layer)
                rebuilt = np.concatenate(
                    (rebuilt[..., :1152], gate, rebuilt[..., 1152:]), axis=-1
                )
            compare(f"{name} layer {layer}", np.ascontiguousarray(rebuilt), stored)
    return ok


def _verify_expert_samples(
    checkpoint: CheckpointProtocol,
    compare: Callable[[str, np.ndarray, np.ndarray], None],
    rank_dir: Path,
    rank: int,
    meta: tuple[list[int], dict[str, Any]],
) -> None:
    """Verify sampled expert weight slices against stored rank pre-shards."""
    layers, families = meta
    total_layers = int(families.get("attn_norm", {}).get("shape", [LAYERS])[0])
    for layer in (x for x in layers if 1 <= x < total_layers):
        position = layer - 1
        rebuilt_experts = expert_values(checkpoint, rank, layer)
        for name in EXPERT_FAMILIES:
            spec = families[name]
            k, (lo, _) = next(
                (k, span)
                for k, span in enumerate(spec["layer_ranges"])
                if span[0] <= position < span[1]
            )
            per_layer = int(np.prod(spec["shape"][1:]))
            itemsize = np.dtype(spec["dtype"]).itemsize
            stored = _stored(
                rank_dir / spec["files"][k],
                (position - lo) * per_layer * itemsize,
                per_layer,
                spec["dtype"],
            )
            compare(f"{name} layer {layer}", rebuilt_experts[name], stored)


def verify(args: argparse.Namespace, log: Callable[[str], None]) -> bool:
    """Compare a rebuilt sample of ``--rank`` with ``--against``."""
    default_load, default_dspark = _modules()
    load: LoadModuleProtocol = vars(args).get("load_module") or default_load
    dspark: DsparkModuleProtocol = vars(args).get("dspark_module") or default_dspark
    family_layers_fn: Callable[..., list[int] | None] = (
        vars(load).get("family_layers") or vars(load)["_family_layers"]
    )
    checkpoint = load.Checkpoint(args.checkpoint)
    rank_dir = Path(args.against) / f"rank{args.rank:02d}"
    index_data = json.loads((rank_dir / "index.json").read_text())
    total_layers = int(index_data.get("layers", LAYERS))
    families = index_data["families"]
    shapes = dict(
        load.selected_local_shapes(
            total_layers, checkpoint.config.dim, vocab=checkpoint.config.vocab
        )
    )
    # Older pre-shards store k_projection with the gate columns inside (the
    # loader splits them off): compare it combined.
    legacy = "k_gate" not in families
    if legacy:
        shapes["k_projection"] = families["k_projection"]["shape"]
        shapes.pop("k_gate", None)
    if set(shapes) != set(families):
        log(
            f"family sets differ: only in the loader "
            f"{sorted(set(shapes) - set(families))}, only in the pre-shard "
            f"{sorted(set(families) - set(shapes))}"
        )
        return False
    layers = [int(x) for x in args.layers.split(",")]
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

    dense_ok = _verify_dense_samples(
        checkpoint,
        family_layers_fn,
        compare,
        (rank_dir, args.rank, layers, legacy, families, shapes),
    )
    ok = ok and dense_ok
    _verify_expert_samples(checkpoint, compare, rank_dir, args.rank, (layers, families))

    directory = draft_directory(args.against, args.draft)
    index = json.loads((directory / "index.json").read_text())
    shards = draft_shards(dspark, args.draft, 16384)
    raw = np.fromfile(directory / f"rank{args.rank:02d}.bin", dtype=np.uint8)
    for name, spec in index["arrays"].items():
        count = int(np.prod(spec["shape"])) * np.dtype(spec["dtype"]).itemsize
        stored = raw[spec["offset"] : spec["offset"] + count].view(spec["dtype"])
        compare(
            f"dspark {name}",
            np.ascontiguousarray(shards[name][args.rank]),
            stored,
        )
    log(f"verify rank {args.rank}: {'all identical' if ok else 'MISMATCH'}")
    return ok


def run(args: argparse.Namespace, log: Callable[[str], None]) -> int:
    """Dispatch the preshard CLI subcommand and return an exit status code."""
    if args.target == "k3":
        write_k3(args, log)
    elif args.target == "prefill":
        if int(settings.SLURM_NODEID) == 0:
            write_prefill(args, log)
    elif args.target == "dspark":
        write_dspark(args, log)
    else:
        return 0 if verify(args, log) else 1
    return 0
