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
from collections.abc import Sequence
from typing import Any

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
        help="Override number of paged KV cache blocks per layer.",
    )
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

    exporter.export(exportable, args.output_dir)
    return exportable.export_config


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for `python -m vllm_torchtpu.experimental.export`."""
    args = parse_args(argv)
    run_export_from_args(args)
    return 0
