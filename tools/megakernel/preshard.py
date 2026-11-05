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
"""Write the checkpoints in the kernels' per-rank layout (``preshard``).

    preshard k3 --output DIR [--checkpoint DIR] [--ranks 0,1,..] [--in-flight 4]
    preshard prefill --output DIR --checkpoint DIR
    preshard dspark --output DIR [--draft DIR] [--max-position 16384]
    preshard prefill-experts --source DIR --output DIR [--ranks 0-31] [--verify]
    preshard indexer --output DIR --checkpoint DIR [--ranks 0,1,..]
    preshard serve-view --ranks DIR --extras DIR --output DIR
    preshard verify [--against DIR] [--rank R] [--layers 1,46,92]
    preshard verify-prefill --against DIR --checkpoint DIR [--layers 1,46,92]
    preshard verify-indexer --against DIR --checkpoint DIR [--ranks 0,5]
                            [--layers 3,92]

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
runs where it is started. ``prefill-experts`` writes, per TP32 prefill rank,
the expert families the prefill loads (EP16/TP2: half its EP8 group's
experts over two TP4 quarters, ``rank_loader.join_quarters``) into
``DIR/prefill-experts/rankNN`` in the decode files' layer ranges, so the
prefill reads one contiguous file per part instead of regrouping two
quarters' strided halves; a pre-shard (or a view of one) with that directory
loads them (``rank_loader.prefill_expert_dir``). It reads each source quarter
file once for the two prefill ranks it feeds, runs where it is started (one
host, no TPU), and ``--verify`` instead compares the written ranks byte for
byte with the regrouping loader's arrays.
``indexer`` adds Kimi-K3.1's indexer to the rank directories ``k3`` wrote from the
same checkpoint: ``DIR/rank00/indexer.bin`` (``kimi_k3.dsa.PRESHARD_INDEXER``) and
its entries in ``index.json``, leaving everything else as it is. It runs where it is
started,
reads each tensor once and refuses a rank that already has them. ``verify``
rebuilds a sample of one rank (every dense family at the given layers, the expert
families at the given layers, the indexer families if the rank has them, and with
``--draft`` the speculator's rank file) and compares it byte for byte with an
existing pre-shard; ``verify-indexer`` does so for the indexer families alone over
several ranks; ``verify-prefill`` for the prefill projection cache at the given
layers. ``serve-view`` links rank directories from one pre-shard and the prefill
cache, prefill experts and speculator shards from another into a new directory,
for pre-shards written in separate places.

From a Hugging Face snapshot to the serving directory, on an allocation J of four
hosts, from a checkout of this tree. Each step reads only the snapshots and, after
the first, ``$OUT``; no TPU is used. Everything lands in ``$OUT``, which is the
serving directory as written (``--model-loader-extra-config`` key ``presharded``),
so no view is needed.

    cd <this tree>; export PYTHONPATH=$PWD/src VENV=<the serving venv>
    SNAP=/mnt/models/hub/models--Inferact--Kimi-K3.1-Pro/snapshots/<revision>
    DRAFT=/mnt/models/hub/models--Inferact--Kimi-K3.1-DSpark/snapshots/<revision>
    OUT=/mnt/models/<new directory>/<revision>
    MK=("$VENV/bin/python" -u -m tools.megakernel)
    ALLOC=("${MK[@]}" --jobid J --venv "$VENV" preshard)
    HOST=(srun --jobid=J --overlap -N1 -n1 --chdir="$PWD" "${MK[@]}" preshard)
    FROM=(--checkpoint "$SNAP")

    # Rank directories, one process per host (run from the login node).
    "${ALLOC[@]}" k3 "${FROM[@]}" --output "$OUT"
    # K3.1's indexer into those rank directories (K3 has none: skip).
    "${HOST[@]}" indexer "${FROM[@]}" --output "$OUT"
    # The prefill projection cache, on the first host (run from the login node).
    "${ALLOC[@]}" prefill "${FROM[@]}" --output "$OUT"
    # The speculator's shards.
    "${HOST[@]}" dspark --draft "$DRAFT" --output "$OUT"
    # The TP32 prefill's expert files, from the rank directories.
    "${HOST[@]}" prefill-experts --source "$OUT" --output "$OUT"
    # Checks: byte for byte against the snapshots and the regrouping loader.
    "${HOST[@]}" verify --against "$OUT" "${FROM[@]}" --draft "$DRAFT"
    "${HOST[@]}" verify-indexer --against "$OUT" "${FROM[@]}"
    "${HOST[@]}" verify-prefill --against "$OUT" "${FROM[@]}"
    "${HOST[@]}" prefill-experts --source "$OUT" --output "$OUT" --ranks 0,5 --verify
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
    experts = sub.add_parser(
        "prefill-experts", help="the TP32 prefill's expert families, from a pre-shard"
    )
    experts.add_argument(
        "--source", required=True, help="the TP32 pre-shard (preshard k3)"
    )
    experts.add_argument("--output", required=True)
    experts.add_argument(
        "--ranks", default="0-31", help="prefill ranks to write (a-b or a,b,..)"
    )
    experts.add_argument(
        "--in-flight", type=int, default=4, help="quarter pairs converted concurrently"
    )
    experts.add_argument(
        "--verify",
        action="store_true",
        help="compare the written ranks with the regrouping loader instead of writing",
    )
    indexer = sub.add_parser(
        "indexer", help="add the Kimi-K3.1 indexer to an existing rank cache"
    )
    indexer.add_argument("--checkpoint", required=True, help="the Kimi-K3.1 checkpoint")
    indexer.add_argument("--output", required=True, help="the rank cache (from k3)")
    indexer.add_argument("--ranks", default="", help="ranks to add (default: all)")
    view = sub.add_parser(
        "serve-view", help="link pre-shards written in separate places into one"
    )
    view.add_argument("--ranks", required=True, help="has layout.json and rankNN")
    view.add_argument(
        "--extras",
        required=True,
        help="has prefill-sp-cache, prefill-experts and dspark-*",
    )
    view.add_argument("--output", required=True, help="a new directory")
    verify_prefill_parser = sub.add_parser(
        "verify-prefill", help="compare a pre-shard's prefill projection cache"
    )
    verify_prefill_parser.add_argument(
        "--against", required=True, help="the pre-shard to compare"
    )
    verify_prefill_parser.add_argument(
        "--checkpoint", required=True, help="the checkpoint"
    )
    verify_prefill_parser.add_argument(
        "--layers", default="1,46,92", help="checkpoint layers to compare"
    )
    verify = sub.add_parser(
        "verify", help="rebuild a sample and compare it with a pre-shard"
    )
    verify.add_argument("--against", required=True, help="the pre-shard to compare")
    verify.add_argument("--checkpoint", required=True, help="the Kimi-K3 checkpoint")
    verify.add_argument(
        "--draft", default="", help="the DSpark speculator checkpoint (none: skipped)"
    )
    verify.add_argument("--rank", type=int, default=5)
    verify.add_argument(
        "--layers", default="1,46,92", help="checkpoint layers to compare"
    )
    verify_indexer = sub.add_parser(
        "verify-indexer", help="compare a sample of a rank cache's indexer files"
    )
    verify_indexer.add_argument(
        "--against", required=True, help="the pre-shard to compare"
    )
    verify_indexer.add_argument(
        "--checkpoint", required=True, help="the Kimi-K3.1 checkpoint"
    )
    verify_indexer.add_argument("--ranks", default="0,5,17,31")
    verify_indexer.add_argument(
        "--layers", default="3,47,92", help="checkpoint MLA layers to compare"
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


INDEXER_TENSORS = (
    "wq_b.weight",
    "wk.weight",
    "weights_proj.weight",
    "k_norm.weight",
    "k_norm.bias",
)


def indexer_tensor(load, checkpoint, layer: int, name: str) -> np.ndarray:
    """A tensor of ``INDEXER_TENSORS`` at checkpoint MLA ``layer``."""
    from vllm_torchtpu.megakernel.kimi_k3.dsa import INDEX_DIM, INDEX_HEADS

    config = checkpoint.config
    shape = {
        "wq_b.weight": (INDEX_HEADS * INDEX_DIM, config.q_rank),
        "wk.weight": (INDEX_DIM, config.dim),
        "weights_proj.weight": (INDEX_HEADS, config.dim),
    }.get(name, (INDEX_DIM,))
    key = f"{load.PREFIX}layers.{layer}.self_attn.indexer.{name}"
    if key not in checkpoint.index:
        raise SystemExit(f"{checkpoint.path} has no {key}: not a Kimi-K3.1 checkpoint")
    value = checkpoint.read(key)
    if value.shape != shape or value.dtype != load.BF16:
        raise ValueError((key, value.shape, value.dtype))
    return value


def indexer_values(load, checkpoint, layer: int) -> tuple[np.ndarray, ...]:
    return tuple(
        indexer_tensor(load, checkpoint, layer, name) for name in INDEXER_TENSORS
    )


def _check_indexer_target(output: Path, ranks, checkpoint: str) -> None:
    """Refuse, before writing anything, a cache that holds another checkpoint's
    weights, a rank without its cache and a rank that has an indexer already."""
    from vllm_torchtpu.megakernel.kimi_k3 import dsa

    layout = output / "layout.json"
    if not layout.is_file():
        raise SystemExit(f"{layout} is missing: write the rank cache first (k3)")
    source = json.loads(layout.read_text()).get("checkpoint", "")
    if Path(source).resolve() != Path(checkpoint).resolve():
        raise SystemExit(f"{output} holds the weights of {source}, not {checkpoint}")
    for rank in ranks:
        rank_dir = output / f"rank{rank:02d}"
        if not (rank_dir / "index.json").is_file():
            raise SystemExit(f"{rank_dir} is not complete: no index.json")
        families = json.loads((rank_dir / "index.json").read_text())["families"]
        if (rank_dir / dsa.INDEXER_FILE).exists() or set(dsa.PRESHARD_INDEXER) & set(
            families
        ):
            raise SystemExit(f"{rank_dir} already has an indexer: not overwriting it")


def write_rank_indexer(rank_dir: Path, values: list[dict]) -> None:
    """Add a rank's indexer (per MLA layer, ``dsa.rank_indexer_values``) to its
    directory: the file, then its index entries. Readers see either index."""
    from vllm_torchtpu.megakernel.kimi_k3 import dsa

    index = json.loads((rank_dir / "index.json").read_text())
    partial = rank_dir / (dsa.INDEXER_FILE + ".partial")
    offset = 0
    with open(partial, "wb") as out:
        for name in dsa.PRESHARD_INDEXER:
            value = np.stack([layer[name] for layer in values])
            out.write(value.tobytes())
            index["families"][name] = {
                "file": dsa.INDEXER_FILE,
                "offset": offset,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            offset += value.nbytes
    os.replace(partial, rank_dir / dsa.INDEXER_FILE)
    partial = rank_dir / "index.json.partial"
    partial.write_text(json.dumps(index, indent=1))
    os.replace(partial, rank_dir / "index.json")


def write_indexer(args, log) -> None:
    from vllm_torchtpu.megakernel.kimi_k3 import dsa

    load, _ = _modules()
    checkpoint = load.Checkpoint(args.checkpoint)
    output = Path(args.output)
    ranks = (
        [int(r) for r in args.ranks.split(",")]
        if args.ranks
        else list(range(dsa.INDEXER_RANKS))
    )
    _check_indexer_target(output, ranks, args.checkpoint)
    started = time.perf_counter()
    ids = load._family_layers(checkpoint.config, "m_ka", LAYERS)
    with ThreadPoolExecutor(checkpoint.READ_THREADS) as pool:
        tensors = list(
            pool.map(lambda layer: indexer_values(load, checkpoint, layer), ids)
        )
    log(f"{len(ids)} MLA layers read after {time.perf_counter() - started:.0f} s")
    for rank in ranks:
        write_rank_indexer(
            output / f"rank{rank:02d}",
            [dsa.rank_indexer_values(*layer, rank) for layer in tensors],
        )
        log(f"rank {rank}: indexer added after {time.perf_counter() - started:.0f} s")


def _mla_module():
    import importlib

    return importlib.import_module(
        "vllm_torchtpu.megakernel.kimi_k3.kernels.dspark_mla"
    )


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
    """The draft's kernel-layout shards: the MLA draft (``Inferact/Kimi-K3-
    DSpark``, ``kernels.dspark_mla``) or the Qwen3-style one (``kernels.dspark``)."""
    mla = _mla_module()
    if mla.is_mla_draft(Path(draft)):
        config = mla.MlaDraftConfig.from_checkpoint(Path(draft))
        return mla.kernel_weight_shards(
            config, mla.load_checkpoint_weights(Path(draft), config)
        )
    config = dspark.DraftConfig.from_checkpoint(Path(draft))
    weights = dspark.load_checkpoint_weights(Path(draft), config)
    return dspark.kernel_weight_shards(config, weights, max_position=max_position)


def draft_directory(output: str, draft: str) -> Path:
    return Path(output) / f"dspark-{Path(draft).resolve().name[:8]}"


def _rank_list(text: str) -> list[int]:
    ranks = set()
    for item in text.split(","):
        lo, _, hi = item.partition("-")
        ranks.update(range(int(lo), int(hi or lo) + 1))
    return sorted(ranks)


EXPERT_CHUNK_LAYERS = 2  # layers of both source quarters held per write


def write_prefill_experts(args, log) -> None:
    from vllm_torchtpu.megakernel.kimi_k3.kernels import rank_loader as rl

    source, ranks = Path(args.source), set(_rank_list(args.ranks))
    root = Path(args.output) / rl.PREFILL_EXPERTS
    # Each quarter pair feeds the two prefill ranks of its parity, one per half.
    pairs = sorted({rl.prefill_quarters(rank)[0] for rank in ranks})
    started = time.perf_counter()

    def convert(quarters) -> int:
        sources = [rl.RankDir(source, q, LAYERS) for q in quarters]
        targets = {
            rl.prefill_quarters(rank)[1]: rank
            for rank in range(32)
            if rl.prefill_quarters(rank)[0] == quarters and rank in ranks
        }
        index = {
            part: {
                "format": FORMAT,
                "rank": rank,
                "layers": LAYERS,
                "source": str(source),
                "quarters": list(quarters),
                "part": part,
                "families": {},
            }
            for part, rank in targets.items()
        }
        for part, rank in targets.items():
            (root / f"rank{rank:02d}").mkdir(parents=True, exist_ok=True)
        written = 0
        for name in EXPERT_FAMILIES:
            spec = sources[0].spec(name)
            shape, dtype = tuple(spec["shape"]), rl._dtype(spec["dtype"])
            experts, per_layer = shape[1] // 2, int(np.prod(shape[1:])) * dtype.itemsize
            for filename, (lo, hi) in zip(spec["files"], spec["layer_ranges"]):
                with contextlib.ExitStack() as stack:
                    handles = {
                        part: stack.enter_context(
                            open(root / f"rank{rank:02d}" / filename, "xb")
                        )
                        for part, rank in targets.items()
                    }
                    for a in range(lo, hi, EXPERT_CHUNK_LAYERS):
                        b = min(hi, a + EXPERT_CHUNK_LAYERS)
                        quarter_layers = [
                            rl.read_array(
                                d.path / filename,
                                (a - lo) * per_layer,
                                (b - a, *shape[1:]),
                                dtype,
                            )
                            for d in sources
                        ]
                        for part, handle in handles.items():
                            joined = rl.join_quarters(
                                name,
                                [
                                    q[:, part * experts : (part + 1) * experts]
                                    for q in quarter_layers
                                ],
                            )
                            handle.write(joined.tobytes())
                            written += joined.nbytes
            for part in targets:
                joined_shape = list(
                    rl.join_quarters(
                        name,
                        [np.empty((1, experts, *shape[2:]), np.uint8)] * len(quarters),
                    ).shape
                )
                index[part]["families"][name] = {
                    "shape": [shape[0], *joined_shape[1:]],
                    "dtype": spec["dtype"],
                    "files": spec["files"],
                    "layer_ranges": spec["layer_ranges"],
                }
        for (
            part,
            rank,
        ) in targets.items():  # last: the loader takes a rank once its index exists
            (root / f"rank{rank:02d}" / "index.json").write_text(
                json.dumps(index[part], indent=2) + "\n"
            )
        log(
            f"prefill experts of quarters {quarters} -> ranks "
            f"{sorted(targets.values())}: "
            f"{written / 2**30:.1f} GiB after {time.perf_counter() - started:.0f} s"
        )
        return written

    with ThreadPoolExecutor(max(1, args.in_flight)) as pool:
        total = sum(pool.map(convert, pairs))
    elapsed = time.perf_counter() - started
    log(
        f"prefill experts of {len(ranks)} ranks written to {root}: "
        f"{total / 2**30:.1f} GiB "
        f"in {elapsed:.0f} s ({total / elapsed / 1e9:.2f} GB/s)"
    )


def verify_prefill_experts(args, log) -> bool:
    """Every file of the given prefill ranks against the regrouping loader."""
    from vllm_torchtpu.megakernel.kimi_k3.kernels import rank_loader as rl

    source, ok = Path(args.source), True
    for rank in _rank_list(args.ranks):
        prefill = rl.RankDir(Path(args.output) / rl.PREFILL_EXPERTS, rank, LAYERS)
        own = rl.RankDir(source, rank, LAYERS)
        rl.check_prefill_layout(prefill, own)
        quarters, part, parts = rl.prefill_quarters(rank)
        quarter_dirs = [rl.RankDir(source, q, LAYERS) for q in quarters]
        for name in EXPERT_FAMILIES:
            regrouped = own.regrouped_expert_parts(name, quarter_dirs, part, parts)
            for (label, _, want), (_, _, got) in zip(
                regrouped, prefill.expert_parts(name), strict=True
            ):
                same = np.array_equal(want().view(np.uint8), got().view(np.uint8))
                ok = ok and same
                log(
                    f"verify prefill rank {rank} {label}: "
                    f"{'identical' if same else 'MISMATCH'}"
                )
    return ok


def write_dspark(args, log) -> None:
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


def _stored(path: Path, offset: int, count: int, dtype) -> np.ndarray:
    dtype = np.dtype(dtype)
    return np.fromfile(
        path, dtype=np.uint8, count=count * dtype.itemsize, offset=offset
    ).view(dtype)


def _identical(rebuilt: np.ndarray, stored: np.ndarray) -> bool:
    return (
        rebuilt.dtype == stored.dtype
        and rebuilt.size == stored.size
        and np.array_equal(
            rebuilt.reshape(-1).view(np.uint8), stored.reshape(-1).view(np.uint8)
        )
    )


def compare_indexer(
    load, checkpoint, rank_dir, rank, families, layers, compare
) -> None:
    """``compare`` each of ``rank``'s stored indexer families at the checkpoint
    MLA ``layers`` with the checkpoint's."""
    from vllm_torchtpu.megakernel.kimi_k3 import dsa

    ids = load._family_layers(checkpoint.config, "m_ka", LAYERS)
    for layer in layers:
        rebuilt = dsa.rank_indexer_values(
            *indexer_values(load, checkpoint, layer), rank
        )
        for name in dsa.PRESHARD_INDEXER:
            spec = families[name]
            per_item = int(np.prod(spec["shape"][1:]))
            stored = _stored(
                rank_dir / spec["file"],
                spec["offset"] + ids.index(layer) * per_item * load.BF16.itemsize,
                per_item,
                load.BF16,
            )
            compare(f"{name} rank {rank} layer {layer}", rebuilt[name], stored)


def verify_prefill(args, log) -> bool:
    """Rebuild the prefill projection cache at ``--layers`` and compare it byte
    for byte with ``--against``."""
    load, _ = _modules()
    checkpoint = load.Checkpoint(args.checkpoint)
    cache = Path(args.against) / "prefill-sp-cache"
    families = json.loads((cache / "layout.json").read_text())["families"]
    ok = True
    for layer in (int(item) for item in args.layers.split(",")):
        for name, value in prefill_projection_values(load, checkpoint, layer).items():
            family = families[name]
            shape = tuple(family["shape"][1:])
            if value.shape != shape:
                raise ValueError((name, layer, value.shape, shape))
            count = int(np.prod(shape))
            stored = _stored(
                cache / family["file"],
                (layer - 1) * count * value.dtype.itemsize,
                count,
                value.dtype,
            )
            same = _identical(value, stored.reshape(shape))
            log(f"prefill {name} layer {layer}: {'identical' if same else 'DIFFERENT'}")
            ok = ok and same
    return ok


def write_serve_view(args, log) -> None:
    ranks, extras = Path(args.ranks).resolve(), Path(args.extras).resolve()
    layout = json.loads((ranks / "layout.json").read_text())
    links = [ranks / "layout.json"]
    links += [ranks / f"rank{rank:02d}" for rank in range(layout["ranks"])]
    links += [extras / "prefill-sp-cache", extras / "prefill-experts"]
    drafts = sorted(extras.glob("dspark-*"))
    if not drafts:
        raise SystemExit(f"no dspark-* directory in {extras}")
    links += drafts
    missing = [str(path) for path in links if not path.exists()]
    if missing:
        raise SystemExit(f"missing from the view: {missing}")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    for path in links:
        os.symlink(path, output / path.name)
    log(f"serving view with {len(links)} links at {output}")


def verify_indexer(args, log) -> bool:
    """Compare ``--ranks``' stored indexer files at ``--layers`` with the
    checkpoint, and their sizes with their indexes."""
    from vllm_torchtpu.megakernel.kimi_k3 import dsa

    load, _ = _modules()
    checkpoint = load.Checkpoint(args.checkpoint)
    ids = load._family_layers(checkpoint.config, "m_ka", LAYERS)
    layers = [int(x) for x in args.layers.split(",")]
    if not set(layers) <= set(ids):
        raise SystemExit(f"--layers takes MLA layers ({ids}), not {layers}")
    ok = True

    def compare(what: str, rebuilt: np.ndarray, stored: np.ndarray) -> None:
        nonlocal ok
        same = _identical(rebuilt, stored)
        ok &= same
        log(f"{what}: {'identical' if same else 'DIFFERS'}")

    for rank in (int(r) for r in args.ranks.split(",")):
        rank_dir = Path(args.against) / f"rank{rank:02d}"
        families = json.loads((rank_dir / "index.json").read_text())["families"]
        if not set(dsa.PRESHARD_INDEXER) <= set(families):
            log(f"rank {rank}: no indexer families")
            ok = False
            continue
        size = (rank_dir / dsa.INDEXER_FILE).stat().st_size
        indexed = sum(
            int(np.prod(families[name]["shape"])) * load.BF16.itemsize
            for name in dsa.PRESHARD_INDEXER
        )
        if size != indexed:
            log(f"rank {rank}: {dsa.INDEXER_FILE} holds {size} bytes, not {indexed}")
            ok = False
        compare_indexer(load, checkpoint, rank_dir, rank, families, layers, compare)
    log(f"verify indexer: {'all identical' if ok else 'MISMATCH'}")
    return ok


def verify(args, log) -> bool:
    """Compare a rebuilt sample of ``--rank`` with ``--against``."""
    from vllm_torchtpu.megakernel.kimi_k3 import dsa

    load, dspark = _modules()
    checkpoint = load.Checkpoint(args.checkpoint)
    rank_dir = Path(args.against) / f"rank{args.rank:02d}"
    every_family = json.loads((rank_dir / "index.json").read_text())["families"]
    # The optional indexer families are compared on their own, below.
    families = {
        name: spec
        for name, spec in every_family.items()
        if name not in dsa.PRESHARD_INDEXER
    }
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
        same = _identical(rebuilt, stored)
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

    if set(dsa.PRESHARD_INDEXER) <= set(every_family):
        ids = load._family_layers(checkpoint.config, "m_ka", LAYERS)
        compare_indexer(
            load,
            checkpoint,
            rank_dir,
            args.rank,
            every_family,
            [x for x in layers if x in ids] or ids[:1],
            compare,
        )

    if args.draft:
        directory = draft_directory(args.against, args.draft)
        index = json.loads((directory / "index.json").read_text())
        shards = draft_shards(dspark, args.draft, 16384)
        raw = np.fromfile(directory / f"rank{args.rank:02d}.bin", dtype=np.uint8)
        for name, spec in index["arrays"].items():
            count = int(np.prod(spec["shape"])) * np.dtype(spec["dtype"]).itemsize
            stored = raw[spec["offset"] : spec["offset"] + count].view(spec["dtype"])
            compare(
                f"dspark {name}", np.ascontiguousarray(shards[name][args.rank]), stored
            )
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
    elif args.target == "prefill-experts":
        if args.verify:
            return 0 if verify_prefill_experts(args, log) else 1
        write_prefill_experts(args, log)
    elif args.target == "indexer":
        write_indexer(args, log)
    elif args.target == "serve-view":
        write_serve_view(args, log)
    elif args.target == "verify-indexer":
        return 0 if verify_indexer(args, log) else 1
    elif args.target == "verify-prefill":
        return 0 if verify_prefill(args, log) else 1
    else:
        return 0 if verify(args, log) else 1
    return 0
