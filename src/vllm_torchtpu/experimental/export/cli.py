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
"""CLI parser and entrypoint for `python -m vllm_torchtpu.experimental.export`."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from typing import Any

import portpicker

from vllm_torchtpu.experimental.export import (
    api,
    checkpoint,
    exporter,
    universal_adapter,
)
from vllm_torchtpu.experimental.export.trace import trace

# What `--export-mode` writes: the serving artifact without weights (`model`), the
# weights of an existing artifact as an Orbax checkpoint (`checkpoint`), or both
# from one load (`all`, the default).
EXPORT_MODES = ("model", "checkpoint", "all")


def _parse_int_list(raw: str | Sequence[int] | None) -> tuple[int, ...] | None:
    """Parses a comma-separated string or sequence of integers into a tuple."""
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return tuple(int(x) for x in raw)
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    return tuple(int(p) for p in parts) if parts else None


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds the `vllm serve` argument parser plus the export options.

    The exporter takes `vllm serve`'s command line 1:1, model tag included: its
    engine arguments shape the artifact, and its frontend arguments, which do not,
    are reported and ignored (see `ignored_frontend_args`).
    """
    try:
        from vllm.entrypoints.launchers.cli_args import make_arg_parser
        from vllm.utils.argparse_utils import FlexibleArgumentParser

        parser: argparse.ArgumentParser = make_arg_parser(
            FlexibleArgumentParser(
                prog="python -m vllm_torchtpu.experimental.export",
                description=(
                    "Export a vLLM model to a standardized StableHLO serving "
                    "directory. Takes the `vllm serve` command line."
                ),
            )
        )
    except Exception:
        parser = argparse.ArgumentParser(
            prog="python -m vllm_torchtpu.experimental.export",
            description=(
                "Export a vLLM model to a standardized StableHLO serving directory."
            ),
        )
        parser.add_argument(
            "--model",
            type=str,
            default="Qwen/Qwen3-0.6B",
            help="Name or path of the HuggingFace model to export.",
        )
        parser.add_argument(
            "--tokenizer",
            type=str,
            default=None,
            help="Name or path of the HuggingFace tokenizer.",
        )
        parser.add_argument(
            "-tp",
            "--tensor-parallel-size",
            type=int,
            default=1,
            help="Number of tensor parallel replicas.",
        )
        parser.add_argument(
            "-dp",
            "--data-parallel-size",
            type=int,
            default=1,
            help="Number of data parallel replicas.",
        )
        parser.add_argument(
            "-pp",
            "--pipeline-parallel-size",
            type=int,
            default=1,
            help="Number of pipeline parallel stages.",
        )
        parser.add_argument(
            "--dtype",
            type=str,
            default="auto",
            help="Data type for model weights and activations.",
        )
        parser.add_argument(
            "--max-model-len",
            type=int,
            default=None,
            help="Model context length.",
        )
        parser.add_argument(
            "--block-size",
            type=int,
            default=16,
            help="Token block size for contiguous chunks of tokens.",
        )
        parser.add_argument(
            "--max-num-seqs",
            type=int,
            default=64,
            help="Maximum number of sequences per iteration.",
        )
        parser.add_argument(
            "--load-format",
            type=str,
            default="dummy",
            help="The format of the model weights to load.",
        )

    export_group = parser.add_argument_group("Export Options")
    export_group.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Destination directory for exported StableHLO artifacts.",
    )
    export_group.add_argument(
        "--target-tpu",
        type=str,
        default=None,
        help="Target TPU generation label ('v6e', 'v5p'). Auto-detected if omitted.",
    )
    export_group.add_argument(
        "--token-buckets",
        type=str,
        default=None,
        help="Comma-separated token buckets for forward_bucket{B} (e.g. '16,32,64').",
    )
    export_group.add_argument(
        "--req-buckets",
        type=str,
        default=None,
        help="Comma-separated request buckets for logits and samplers (e.g. '8,16').",
    )
    export_group.add_argument(
        "--num-blocks",
        type=int,
        default=None,
        help=(
            "Override number of paged KV cache blocks per layer. Defaults to the "
            "blocks that fit in --kv-cache-memory-bytes, or else in HBM at "
            "--gpu-memory-utilization."
        ),
    )
    export_group.add_argument(
        "--export-mode",
        choices=EXPORT_MODES,
        default="all",
        help=(
            "'all' (default): on TPU, load the real weights once, export the "
            "artifact and save them as an Orbax checkpoint. 'model': export the "
            "artifact, traced with dummy weights, and save no weights. "
            "'checkpoint': on CPU, load the real weights through vLLM and save "
            "them as an Orbax checkpoint in the existing artifact at --output-dir."
        ),
    )
    # Internal: marks a worker process started by `main`.
    parser.add_argument("--export-worker", action="store_true", help=argparse.SUPPRESS)
    # Internal: set by the launchers on the workers they start.
    parser.add_argument(
        checkpoint.JAX_COORDINATOR_FLAG, dest="jax_coordinator", help=argparse.SUPPRESS
    )
    parser.add_argument(
        checkpoint.DATA_DIR_FLAG, dest="checkpoint_data_dir", help=argparse.SUPPRESS
    )
    return parser


# Destinations of the export options and internal flags, which are not
# `vllm serve` arguments.
_EXPORT_DESTS = frozenset(
    {
        "output_dir",
        "target_tpu",
        "token_buckets",
        "req_buckets",
        "num_blocks",
        "export_mode",
        "export_worker",
        "jax_coordinator",
        "checkpoint_data_dir",
    }
)


def ignored_frontend_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> list[str]:
    """The `vllm serve` frontend options set in `args`, which the export ignores.

    Engine arguments (the fields of vLLM's `AsyncEngineArgs`) shape the artifact;
    everything else `vllm serve` takes configures its HTTP frontend.
    """
    try:
        from vllm.engine.arg_utils import AsyncEngineArgs
    except Exception:
        return []
    engine_dests = {f.name for f in dataclasses.fields(AsyncEngineArgs)}
    ignored = []
    for action in parser._actions:
        dest = action.dest
        if (
            dest in engine_dests
            or dest in _EXPORT_DESTS
            or dest in ("help", "model_tag")
            or not action.option_strings
        ):
            continue
        if getattr(args, dest, action.default) != action.default:
            # The positive long form, not `--no-...` of a boolean option.
            names = [
                o
                for o in action.option_strings
                if o.startswith("--") and not o.startswith("--no-")
            ]
            ignored.append((names or action.option_strings)[0])
    return ignored


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses CLI arguments."""
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    # `vllm serve MODEL`: the positional model tag names the model, as in vLLM.
    if getattr(args, "model_tag", None):
        args.model = args.model_tag
    if not getattr(args, "export_worker", False):
        ignored = ignored_frontend_args(parser, args)
        if ignored:
            sys.stderr.write(
                "[export] Ignoring vllm serve frontend options, which do not shape "
                f"the artifact: {', '.join(ignored)}\n"
            )
    # Force dummy weight loading so AOT export never reads multi-GB safetensors data.
    args.load_format = "dummy"
    return args


# Speculative methods the artifact can declare: ngram drafts on the host from the
# sequence itself, so it needs no exported draft graphs.
SPECULATIVE_METHODS = ("ngram",)
_SPECULATIVE_FIELDS = frozenset(
    {"method", "num_speculative_tokens", "prompt_lookup_max", "prompt_lookup_min"}
)


def export_speculative_config(raw: Any) -> dict[str, Any] | None:
    """The `serving_config.speculative_config` for `--speculative-config`.

    Keeps vLLM's field names. Rejects methods that would need exported draft
    graphs (MTP, EAGLE, a draft model) and fields the runtime would not honor.
    """
    if raw is None:
        return None
    cfg = json.loads(raw) if isinstance(raw, str) else dict(raw)
    method = cfg.get("method")
    if method not in SPECULATIVE_METHODS:
        raise ValueError(
            f"--speculative-config method {method!r} is not supported by the "
            f"exporter; supported: {', '.join(SPECULATIVE_METHODS)}"
        )
    unknown = sorted(set(cfg) - _SPECULATIVE_FIELDS)
    if unknown:
        raise ValueError(
            f"--speculative-config fields {unknown} are not supported for method "
            f"{method!r}; supported: {sorted(_SPECULATIVE_FIELDS)}"
        )

    def positive(name: str) -> int:
        value = cfg.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(
                f"--speculative-config {name} must be a positive integer, got {value!r}"
            )
        return value

    lookup_max = positive("prompt_lookup_max")
    cfg.setdefault("prompt_lookup_min", lookup_max)
    return {
        "method": method,
        "num_speculative_tokens": positive("num_speculative_tokens"),
        "prompt_lookup_max": lookup_max,
        "prompt_lookup_min": positive("prompt_lookup_min"),
    }


def create_vllm_config_from_args(args: argparse.Namespace) -> Any:
    """Creates a `VllmConfig` from parsed CLI `EngineArgs`."""
    args.load_format = "dummy"
    return universal_adapter.create_export_vllm_config(args)


@trace("export.run")
def run_export_from_args(
    args: argparse.Namespace,
    *,
    vllm_config: Any = None,
    model: Any = None,
) -> api.TextOnlyLLMExportConfig:
    """Executes the AOT model export given parsed CLI arguments."""
    speculative_config = export_speculative_config(
        getattr(args, "speculative_config", None)
    )
    # The serving runtime drafts; vLLM must not build a drafter for the export.
    args.speculative_config = None
    if vllm_config is None and model is None:
        vllm_config = create_vllm_config_from_args(args)
    if vllm_config is not None:
        universal_adapter.check_supported_parallelism(vllm_config.parallel_config)
    # Load before building the export config: loading finalizes the KV block size.
    export_mode = getattr(args, "export_mode", "model")
    if export_mode == "all":
        # Before any JAX use: the Pallas kernels the model loads with use JAX.
        checkpoint.init_jax(
            int(os.environ.get("RANK", 0)),
            int(os.environ.get("WORLD_SIZE", 1)),
            getattr(args, "jax_coordinator", None),
        )
    loaded_on_device = model is None
    if loaded_on_device:
        model = universal_adapter.load_vllm_model(
            vllm_config, load_format="auto" if export_mode == "all" else "dummy"
        )

    token_buckets = _parse_int_list(getattr(args, "token_buckets", None))
    req_buckets = _parse_int_list(getattr(args, "req_buckets", None))
    target_tpu = getattr(args, "target_tpu", None)

    export_cfg = universal_adapter.build_export_config_from_vllm(
        vllm_config,
        model_name=getattr(args, "model", None),
        hf_checkpoint_dir=getattr(args, "model", None),
        target_tpu=target_tpu,
        token_buckets=token_buckets,
        req_buckets=req_buckets,
    )
    num_blocks_override = getattr(args, "num_blocks", None)
    if num_blocks_override is not None:
        export_cfg = dataclasses.replace(
            export_cfg, num_blocks=int(num_blocks_override)
        )
    if speculative_config is not None:
        export_cfg = dataclasses.replace(
            export_cfg, speculative_config=speculative_config
        )

    if isinstance(model, api.ExportableTextOnlyLLM):
        exportable = model
    else:
        exportable = universal_adapter.UniversalVllmExportableLLM(
            model=model,
            export_config=export_cfg,
            vllm_config=vllm_config,
        )
        cache_cfg = vllm_config.cache_config
        kv_cache_memory_bytes = getattr(cache_cfg, "kv_cache_memory_bytes", None)
        if (
            (loaded_on_device or kv_cache_memory_bytes)
            and num_blocks_override is None
            and cache_cfg.num_gpu_blocks_override is None
        ):
            exportable.fit_num_blocks_to_hbm(
                cache_cfg.gpu_memory_utilization, kv_cache_memory_bytes
            )

    weights = (
        checkpoint.host_tensors(getattr(exportable, "model", exportable))
        if export_mode == "all"
        else None
    )
    exporter.export(exportable, args.output_dir)
    if weights is not None:
        checkpoint.save_model_checkpoint(
            weights,
            args.output_dir,
            getattr(args, "checkpoint_data_dir", None)
            or os.path.join(args.output_dir, checkpoint.CHECKPOINT_DATA_DIR),
        )
    return exportable.export_config


# vllm_torchtpu.envs switch of the unified KV block pool.
UNIFIED_BLOCK_POOL_ENV = "TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL"


def _worker_env(tp_size: int) -> dict[str, str]:
    """Environment shared by the workers of a `tp_size`-device export.

    Prepared the way vLLM's single-host serving prepares its workers; a single
    device skips the distributed TPU bootstrap.
    """
    from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

    saved = dict(os.environ)
    try:
        os.environ["WORLD_SIZE"] = str(tp_size)
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(portpicker.pick_unused_port()))
        # Recurrent (GDN) layers keep per-layer `[conv, recurrent]` state rather
        # than byte regions of the attention-shaped unified block pool.
        os.environ.setdefault(UNIFIED_BLOCK_POOL_ENV, "0")
        TpuPlatform._prepare_singlehost_tpu_env(tp_size)
        return dict(os.environ)
    finally:
        os.environ.clear()
        os.environ.update(saved)


@trace("export.launch_workers")
def _launch_workers(
    argv: Sequence[str], tp_size: int, output_dir: str, *, save_checkpoint: bool = False
) -> None:
    """Runs one export worker per tensor-parallel rank and keeps rank 0's output.

    Every rank traces the same per-device programs (collectives need all of
    them); rank 0's artifact is copied to `output_dir`. With `save_checkpoint`,
    the workers also save their weights as the Orbax checkpoint of `output_dir`.
    """
    env = _worker_env(tp_size)
    coordinator = (
        f"{env.get('MASTER_ADDR', '127.0.0.1')}:{portpicker.pick_unused_port()}"
    )
    worker_flags = [checkpoint.JAX_COORDINATOR_FLAG, coordinator]
    if save_checkpoint:
        data_dir = os.path.join(
            os.path.abspath(output_dir), checkpoint.CHECKPOINT_DATA_DIR
        )
        shutil.rmtree(data_dir, ignore_errors=True)
        worker_flags += [checkpoint.DATA_DIR_FLAG, data_dir]
    parent_dir = os.path.dirname(os.path.abspath(output_dir))
    os.makedirs(parent_dir, exist_ok=True)
    rank_root = tempfile.mkdtemp(prefix=".export_ranks_", dir=parent_dir)
    procs = []
    try:
        for rank in range(tp_size):
            rank_env = dict(env, RANK=str(rank), LOCAL_RANK=str(rank))
            rank_dir = os.path.join(rank_root, f"rank{rank}")
            procs.append(
                subprocess.Popen(
                    [sys.executable, "-m", "vllm_torchtpu.experimental.export"]
                    + [
                        *argv,
                        *worker_flags,
                        "--output-dir",
                        rank_dir,
                        "--export-worker",
                    ],
                    env=rank_env,
                )
            )
        failed = [r for r, proc in enumerate(procs) if proc.wait() != 0]
        if failed:
            raise RuntimeError(f"export workers for ranks {failed} failed")
        shutil.copytree(
            os.path.join(rank_root, "rank0"), output_dir, dirs_exist_ok=True
        )
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
        shutil.rmtree(rank_root, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for `python -m vllm_torchtpu.experimental.export`.

    Starts one worker process per tensor-parallel rank (one for a single-device
    export); each worker runs `run_export_from_args`.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    launch = _launch_options(argv)
    if launch.export_mode == "checkpoint":
        # CPU workers parse the engine arguments: vLLM picks its platform when
        # imported, so only processes started with `checkpoint.CPU_ENV` run on CPU.
        if launch.export_worker:
            checkpoint.run_cpu_load_worker(parse_args(argv))
        else:
            checkpoint.export_checkpoint(
                argv, launch.tensor_parallel_size, launch.output_dir
            )
        return 0
    args = parse_args(argv)
    # Reject before starting workers, each of which would load the model first.
    universal_adapter.check_supported_parallelism(args)
    if args.export_worker:
        run_export_from_args(args)
        return 0
    tp_size = int(getattr(args, "tensor_parallel_size", 1) or 1)
    _launch_workers(
        argv, tp_size, args.output_dir, save_checkpoint=args.export_mode == "all"
    )
    return 0


def _launch_options(argv: Sequence[str]) -> argparse.Namespace:
    """The options `main` dispatches on, read without vLLM's argument parser."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--export-mode", choices=EXPORT_MODES, default="all")
    parser.add_argument("--output-dir")
    parser.add_argument("-tp", "--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--export-worker", action="store_true")
    return parser.parse_known_args(argv)[0]
