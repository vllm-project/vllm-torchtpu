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
    exporter,
    universal_adapter,
)


def _parse_int_list(raw: str | Sequence[int] | None) -> tuple[int, ...] | None:
    """Parses a comma-separated string or sequence of integers into a tuple."""
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return tuple(int(x) for x in raw)
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    return tuple(int(p) for p in parts) if parts else None


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds CLI argument parser aligned with `EngineArgs` plus `--output-dir`."""
    try:
        from vllm.engine.arg_utils import EngineArgs
        from vllm.utils.argparse_utils import FlexibleArgumentParser

        flexible_parser = FlexibleArgumentParser(
            prog="python -m vllm_torchtpu.experimental.export",
            description=(
                "Export a vLLM model to a standardized StableHLO serving directory."
            ),
        )
        parser: argparse.ArgumentParser = EngineArgs.add_cli_args(flexible_parser)
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
    # Internal: marks a worker process started by `main`.
    parser.add_argument("--export-worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses CLI arguments."""
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    # Force dummy weight loading so AOT export never reads multi-GB safetensors data.
    args.load_format = "dummy"
    return args


def create_vllm_config_from_args(args: argparse.Namespace) -> Any:
    """Creates a `VllmConfig` from parsed CLI `EngineArgs`."""
    from vllm.engine.arg_utils import EngineArgs

    engine_args = EngineArgs.from_cli_args(args)
    engine_args.load_format = "dummy"
    return engine_args.create_engine_config()


def run_export_from_args(
    args: argparse.Namespace,
    *,
    vllm_config: Any = None,
    model: Any = None,
) -> api.TextOnlyLLMExportConfig:
    """Executes the AOT model export given parsed CLI arguments."""
    if vllm_config is None and model is None:
        vllm_config = create_vllm_config_from_args(args)
    if vllm_config is not None:
        universal_adapter.check_supported_parallelism(vllm_config.parallel_config)
    # Load before building the export config: loading finalizes the KV block size.
    loaded_on_device = model is None
    if loaded_on_device:
        model = universal_adapter.load_dummy_vllm_model(vllm_config)

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
        import dataclasses

        export_cfg = dataclasses.replace(
            export_cfg, num_blocks=int(num_blocks_override)
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

    exporter.export(exportable, args.output_dir)
    return exportable.export_config


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
        TpuPlatform._prepare_singlehost_tpu_env(tp_size)
        return dict(os.environ)
    finally:
        os.environ.clear()
        os.environ.update(saved)


def _launch_workers(argv: Sequence[str], tp_size: int, output_dir: str) -> None:
    """Runs one export worker per tensor-parallel rank and keeps rank 0's output.

    Every rank traces the same per-device programs (collectives need all of
    them); rank 0's artifact is copied to `output_dir`.
    """
    env = _worker_env(tp_size)
    rank_root = tempfile.mkdtemp(
        prefix=".export_ranks_", dir=os.path.dirname(os.path.abspath(output_dir))
    )
    procs = []
    try:
        for rank in range(tp_size):
            rank_env = dict(env, RANK=str(rank), LOCAL_RANK=str(rank))
            rank_dir = os.path.join(rank_root, f"rank{rank}")
            procs.append(
                subprocess.Popen(
                    [sys.executable, "-m", "vllm_torchtpu.experimental.export"]
                    + [*argv, "--output-dir", rank_dir, "--export-worker"],
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
    args = parse_args(argv)
    # Reject before starting workers, each of which would load the model first.
    universal_adapter.check_supported_parallelism(args)
    if args.export_worker:
        run_export_from_args(args)
        return 0
    tp_size = int(getattr(args, "tensor_parallel_size", 1) or 1)
    _launch_workers(argv, tp_size, args.output_dir)
    return 0
