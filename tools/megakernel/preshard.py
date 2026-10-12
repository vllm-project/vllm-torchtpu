# SPDX-License-Identifier: Apache-2.0
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
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

FORMAT = 1
GU_FILES = 4  # gate/up split into layer ranges
DOWN_FILES = 2
EXPERT_FAMILIES = ("expert_gu", "expert_gus", "expert_down", "expert_ds")
EXPERT_DTYPES = {
    "expert_gu": np.uint32,
    "expert_gus": np.uint8,
    "expert_down": np.uint32,
    "expert_ds": np.uint8,
}
LAYERS = 93


def add_arguments(parser) -> None:
    sub = parser.add_subparsers(dest="target", required=True)
    model = sub.add_parser("k3", help="the Kimi-K3 checkpoint")
    model.add_argument("--checkpoint", required=True, help="the Kimi-K3 checkpoint")
    model.add_argument("--output", required=True)
    model.add_argument(
        "--ranks", default="", help="ranks to convert (default: the eight of this host)"
    )
    model.add_argument(
        "--in-flight", type=int, default=4, help="ranks converted concurrently per host"
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
    verify = sub.add_parser(
        "verify", help="rebuild a sample and compare it with a pre-shard"
    )
    verify.add_argument("--against", required=True, help="the pre-shard to compare")
    verify.add_argument("--checkpoint", required=True, help="the Kimi-K3 checkpoint")
    verify.add_argument(
        "--draft", required=True, help="the DSpark speculator checkpoint"
    )
    verify.add_argument("--rank", type=int, default=5)
    verify.add_argument(
        "--layers", default="1,46,92", help="checkpoint layers to compare"
    )


def runs_on_allocation(args) -> bool:
    return args.target in ("k3", "prefill")


def _modules():
    """The checkpoint readers, imported directly: the op registry would bring
    in torch_tpu, which a host-side conversion must not touch."""
    import importlib

    return tuple(
        importlib.import_module(f"vllm_torchtpu.megakernel.kimi_k3.kernels.{m}")
        for m in ("load", "dspark")
    )


def _layer_ranges(layers: int, files: int):
    edges = [round(i * layers / files) for i in range(files + 1)]
    return [(edges[i], edges[i + 1]) for i in range(files)]


def _experts_of(checkpoint, rank: int) -> tuple[int, int, int]:
    """``(first expert, experts, quarter)``: the rank's slice of a layer's
    experts (8 expert groups of 4 ranks, each rank a quarter of the
    intermediate dimension)."""
    count = checkpoint.config.experts // 8
    return count * (rank // 4), count, rank % 4


def dense_value(checkpoint, name: str, rank: int, layer: int | None):
    """A dense family's rank slice at checkpoint ``layer`` (None: a family
    that is not per layer)."""
    return checkpoint.dense_selected(name, 0 if layer is None else layer, rank)


def expert_values(checkpoint, rank: int, layer: int) -> dict[str, np.ndarray]:
    first, count, part = _experts_of(checkpoint, rank)
    gate_up, gate_up_scales, downs, down_scale = checkpoint.experts(
        layer, first, count, part=part, parts=4
    )
    return {
        name: np.ascontiguousarray(np.asarray(value, EXPERT_DTYPES[name]))
        for name, value in zip(
            EXPERT_FAMILIES, (gate_up, gate_up_scales, downs, down_scale)
        )
    }


def convert_rank(load, checkpoint, rank: int, output: Path, log) -> dict:
    """Write one rank's files; returns its index."""
    shapes = load.selected_local_shapes(
        LAYERS, checkpoint.config.dim, vocab=checkpoint.config.vocab
    )
    rank_dir = output / f"rank{rank:02d}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    index = {"format": FORMAT, "rank": rank, "layers": LAYERS, "families": {}}
    started = time.perf_counter()

    offset = 0
    with open(rank_dir / "dense.bin", "wb") as dense:
        for name, shape in shapes.items():
            if name in EXPERT_FAMILIES:
                continue
            ids = load._family_layers(checkpoint.config, name, LAYERS)
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
            value = np.ascontiguousarray(value)
            dense.write(value.tobytes())
            index["families"][name] = {
                "file": "dense.bin",
                "offset": offset,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            offset += value.nbytes
            del value
    log(
        f"rank {rank}: dense families written ({offset / 1e9:.1f} GB) "
        f"after {time.perf_counter() - started:.0f} s"
    )

    expert_layers = LAYERS - 1
    ranges = {
        "expert_gu": _layer_ranges(expert_layers, GU_FILES),
        "expert_gus": [(0, expert_layers)],
        "expert_down": _layer_ranges(expert_layers, DOWN_FILES),
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
            name: [stack.enter_context(open(rank_dir / f, "wb")) for f in names]
            for name, names in files.items()
        }
        for position, layer in enumerate(range(1, LAYERS)):
            for name, array in expert_values(checkpoint, rank, layer).items():
                if tuple(array.shape) != tuple(shapes[name][1:]):
                    raise ValueError((name, array.shape, shapes[name]))
                k = next(
                    k for k, (lo, hi) in enumerate(ranges[name]) if lo <= position < hi
                )
                handles[name][k].write(array.tobytes())
            if layer % 8 == 0 or layer == LAYERS - 1:
                log(
                    f"rank {rank}: expert layer {layer}/{LAYERS - 1} "
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


def write_k3(args, log) -> None:
    load, _ = _modules()
    host = int(os.environ.get("SLURM_NODEID", "0"))
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
                lambda rank: convert_rank(load, checkpoint, rank, output, log), ranks
            )
        )
    log(f"ranks {ranks} converted in {time.perf_counter() - started:.0f} s")


def prefill_projection_values(load, checkpoint, layer: int) -> dict[str, np.ndarray]:
    stem = f"{load.PREFIX}layers.{layer}.block_sparse_moe."
    values = {
        "latent_down": np.ascontiguousarray(
            np.pad(
                checkpoint.read(stem + "routed_expert_down_proj.weight").T,
                ((0, 0), (0, 512)),
            )
        ),
        "latent_up": np.ascontiguousarray(
            checkpoint.read(stem + "routed_expert_up_proj.weight").T
        ),
    }
    shared = {
        name: checkpoint.read(stem + f"shared_experts.{name}_proj.weight")
        for name in ("gate", "up", "down")
    }
    for parity in (0, 1):
        channels = np.concatenate(
            [np.arange(rank * 192, (rank + 1) * 192) for rank in range(parity, 32, 2)]
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


def write_prefill(args, log) -> None:
    load, _ = _modules()
    checkpoint = load.Checkpoint(args.checkpoint)
    output = Path(args.output) / "prefill-sp-cache"
    output.mkdir(parents=True, exist_ok=False)
    shapes = {
        "latent_down": (7168, 4096),
        "latent_up": (3584, 7168),
        "shared_gu_0": (7168, 6144),
        "shared_gu_1": (7168, 6144),
        "shared_down_0": (3072, 7168),
        "shared_down_1": (3072, 7168),
    }
    layout = {
        "format": FORMAT,
        "layers": LAYERS,
        "checkpoint": str(args.checkpoint),
        "families": {
            name: {
                "file": f"{name}.bin",
                "shape": [LAYERS - 1, *shape],
                "dtype": "bfloat16",
            }
            for name, shape in shapes.items()
        },
    }
    started = time.perf_counter()
    with contextlib.ExitStack() as stack:
        handles = {
            name: stack.enter_context(open(output / f"{name}.bin", "xb"))
            for name in shapes
        }
        for layer in range(1, LAYERS):
            values = prefill_projection_values(load, checkpoint, layer)
            for name, value in values.items():
                if value.shape != shapes[name] or value.dtype != load.BF16:
                    raise ValueError((name, layer, value.shape, value.dtype))
                handles[name].write(value.tobytes())
            del values
            if layer % 8 == 0 or layer == LAYERS - 1:
                log(
                    f"prefill projections: layer {layer}/{LAYERS - 1} "
                    f"after {time.perf_counter() - started:.0f} s"
                )
    (output / "layout.json").write_text(json.dumps(layout, indent=2) + "\n")
    log(
        f"prefill projections written to {output} "
        f"in {time.perf_counter() - started:.0f} s"
    )


def draft_shards(dspark, draft: str, max_position: int):
    config = dspark.DraftConfig.from_checkpoint(Path(draft))
    weights = dspark.load_checkpoint_weights(Path(draft), config)
    return dspark.kernel_weight_shards(config, weights, max_position=max_position)


def draft_directory(output: str, draft: str) -> Path:
    return Path(output) / f"dspark-{Path(draft).resolve().name[:8]}"


def write_dspark(args, log) -> None:
    _, dspark = _modules()
    started = time.perf_counter()
    shards = draft_shards(dspark, args.draft, args.max_position)
    log(f"kernel layout built in {time.perf_counter() - started:.0f} s")
    directory = draft_directory(args.output, args.draft)
    dspark.save_kernel_shards(shards, directory)
    log(f"written to {directory} in {time.perf_counter() - started:.0f} s")


def _stored(path: Path, offset: int, count: int, dtype) -> np.ndarray:
    dtype = np.dtype(dtype)
    return np.fromfile(
        path, dtype=np.uint8, count=count * dtype.itemsize, offset=offset
    ).view(dtype)


def verify(args, log) -> bool:
    """Compare a rebuilt sample of ``--rank`` with ``--against``."""
    load, dspark = _modules()
    checkpoint = load.Checkpoint(args.checkpoint)
    rank_dir = Path(args.against) / f"rank{args.rank:02d}"
    families = json.loads((rank_dir / "index.json").read_text())["families"]
    shapes = load.selected_local_shapes(
        LAYERS, checkpoint.config.dim, vocab=checkpoint.config.vocab
    )
    # Older pre-shards store k_projection with the gate columns inside (the
    # loader splits them off): compare it combined.
    legacy = "k_gate" not in families
    if legacy:
        shapes = dict(shapes, k_projection=families["k_projection"]["shape"])
        del shapes["k_gate"]
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
        nonlocal ok
        same = (
            rebuilt.dtype == stored.dtype
            and rebuilt.size == stored.size
            and np.array_equal(
                rebuilt.reshape(-1).view(np.uint8), stored.reshape(-1).view(np.uint8)
            )
        )
        ok &= same
        log(
            f"{what}: {'identical' if same else 'DIFFERS'} "
            f"({rebuilt.nbytes / 1e6:.1f} MB)"
        )

    for name, spec in families.items():
        if name in EXPERT_FAMILIES:
            continue
        if list(spec["shape"]) != list(shapes[name]):
            log(f"{name}: stored shape {spec['shape']}, loader {shapes[name]}")
            ok = False
            continue
        ids = load._family_layers(checkpoint.config, name, LAYERS)
        per_item = int(np.prod(spec["shape"][1:] if ids else spec["shape"]))
        for layer in (
            [None] if ids is None else [x for x in layers if x in ids] or ids[:1]
        ):
            position = 0 if layer is None else ids.index(layer)
            itemsize = np.dtype(spec["dtype"]).itemsize
            stored = _stored(
                rank_dir / spec["file"],
                spec["offset"] + position * per_item * itemsize,
                per_item,
                spec["dtype"],
            )
            rebuilt = dense_value(checkpoint, name, args.rank, layer)
            if legacy and name == "k_projection":
                gate = dense_value(checkpoint, "k_gate", args.rank, layer)
                rebuilt = np.concatenate(
                    (rebuilt[..., :1152], gate, rebuilt[..., 1152:]), axis=-1
                )
            rebuilt = np.ascontiguousarray(rebuilt)
            compare(f"{name} layer {layer}", rebuilt, stored)

    for layer in (x for x in layers if 1 <= x < LAYERS):
        position = layer - 1
        rebuilt = expert_values(checkpoint, args.rank, layer)
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
            compare(f"{name} layer {layer}", rebuilt[name], stored)

    directory = draft_directory(args.against, args.draft)
    index = json.loads((directory / "index.json").read_text())
    shards = draft_shards(dspark, args.draft, 16384)
    raw = np.fromfile(directory / f"rank{args.rank:02d}.bin", dtype=np.uint8)
    for name, spec in index["arrays"].items():
        count = int(np.prod(spec["shape"])) * np.dtype(spec["dtype"]).itemsize
        stored = raw[spec["offset"] : spec["offset"] + count].view(spec["dtype"])
        compare(f"dspark {name}", np.ascontiguousarray(shards[name][args.rank]), stored)
    log(f"verify rank {args.rank}: {'all identical' if ok else 'MISMATCH'}")
    return ok


def run(args, log) -> int:
    if args.target == "k3":
        write_k3(args, log)
    elif args.target == "prefill":
        if int(os.environ.get("SLURM_NODEID", "0")) == 0:
            write_prefill(args, log)
    elif args.target == "dspark":
        write_dspark(args, log)
    else:
        return 0 if verify(args, log) else 1
    return 0
