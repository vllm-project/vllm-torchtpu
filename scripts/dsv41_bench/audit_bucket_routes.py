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
"""CPU descriptor audit of kernel admission, without serving registration.

Like bench_csa2, load the real kernel modules with package startup bypassed.
The v7 hardware descriptor is explicit: no TPU program executes, and this
does not establish compiled bridge selection, numeric parity or performance.
"""

import argparse
import dataclasses
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch


def audit(
    root: Path, buckets: tuple[int, ...], requests: int, trace: bool = False
) -> dict:
    """Evaluate real eligibility and tile selectors on abstract arrays."""
    import jax
    import jax.numpy as jnp
    from jax.experimental.pallas import tpu as pltpu

    # Standalone kernel audit must not initialize the platform or model plugin.
    for name in (
        "vllm_torchtpu",
        "vllm_torchtpu.kernels",
        "vllm_torchtpu.layers",
        "vllm_torchtpu.layers.core",
    ):
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(root / "src" / name.replace(".", "/"))]
            sys.modules[name] = package

    from vllm_torchtpu.kernels.attention_preprocessing import (
        compress_rope,
        qa_rope,
        qb_rope,
    )
    from vllm_torchtpu.kernels.attention_preprocessing import (
        tuned_block_sizes as prep_tuning,
    )
    from vllm_torchtpu.kernels.deepseek_v41 import output_projection
    from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8 import prepare
    from vllm_torchtpu.kernels.deepseek_v41.mhc import shifted_post_pre_rms
    from vllm_torchtpu.kernels.experimental import decode_preparation as prep
    from vllm_torchtpu.kernels.quantized_matmul import tuned_block_sizes

    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    regions = {
        "moe_routing": prep.ROUTING_TOKENS,
        "compressor_compact": prep.COMPACT_TOKENS,
        "compressor_state": prep.COMPRESSOR_STATE_TOKENS,
        "compressor_address": prep.COMPRESSOR_ADDRESS_TOKENS,
        "swa_pack": prep.SWA_PACK_TOKENS,
        "swa_metadata": prep.SWA_BATCHES,
        "main_records": prep.MAIN_RECORD_BATCHES,
    }
    rows = []
    with (
        patch.object(jax, "default_backend", return_value="tpu"),
        patch.object(pltpu, "get_tpu_info", return_value=info),
        patch.object(tuned_block_sizes, "get_tpu_version", return_value=7),
        patch.object(output_projection, "get_tpu_version", return_value=7),
    ):
        for tokens in buckets:
            qa = qa_rope.select_config(tokens, 5120, 1792)
            qb = qb_rope.select_config(tokens, 1280, 32768, 64)
            comp = compress_rope.select_config(tokens, 5120, 2)
            qa_rope.check_config(qa, tokens, 1792)
            segment = qb_rope._segment_tokens(qb, tokens, 1280, 64, 512)
            qb_rope.check_config(qb, segment, 64)
            compress_rope.check_config(comp, tokens, 1024)
            shape = jax.ShapeDtypeStruct
            output_inputs = (
                shape((tokens, 64, 512), jnp.bfloat16),
                shape((4096, 8192), jnp.float8_e4m3fn),
                shape((8192,), jnp.float32),
                shape((8192, 5120), jnp.float8_e4m3fn),
                shape((5120,), jnp.float32),
            )
            row = {
                "tokens": tokens,
                "qa": {
                    "config": dataclasses.asdict(qa),
                    "tuned": (7, tokens, 5120, 1792) in prep_tuning.QA_ROPE_TUNING,
                    "vmem_bytes": qa_rope.vmem_bytes(qa, 5120, 1792, 64),
                },
                "qb": {
                    "config": dataclasses.asdict(qb),
                    "tuned": (7, tokens, 1280, 32768) in prep_tuning.QB_ROPE_TUNING,
                    "segment_tokens": segment,
                },
                "compressor": {
                    "config": dataclasses.asdict(comp),
                    "tuned": (7, tokens, 5120, 2) in prep_tuning.COMPRESS_ROPE_TUNING,
                    "vmem_bytes": compress_rope.vmem_bytes(comp, 5120, 2),
                },
                "fused_output_projection": output_projection.is_supported(
                    *output_inputs
                ),
                "shifted_mhc": shifted_post_pre_rms.is_supported_shape(tokens, 4, 5120),
                "sorted_fp8_index_write": prepare.supports_sorted_fp8(
                    (shape((512, 128), jnp.int32),) * 2,
                    shape((tokens,), jnp.int32),
                    shape((tokens, 128), jnp.bfloat16),
                    256,
                ),
                "metadata_fusions": {
                    name: prep.admit(
                        name,
                        tokens=(tokens, requests)
                        if isinstance(next(iter(allowed)), tuple)
                        else tokens,
                        buckets=allowed,
                    )
                    for name, allowed in regions.items()
                },
            }
            assert row["qa"]["vmem_bytes"] <= qa.vmem_limit_bytes
            assert row["compressor"]["vmem_bytes"] <= comp.vmem_limit_bytes
            if trace:
                rope = shape((tokens, 64), jnp.float32)
                x = shape((tokens, 5120), jnp.bfloat16)
                calls = {
                    "qa": (
                        qa_rope.fused_qa_kv_norm_rope,
                        (
                            x,
                            shape((5120, 1792), jnp.float8_e4m3fn),
                            shape((1792,), jnp.float32),
                            shape((1280,), jnp.bfloat16),
                            shape((512,), jnp.bfloat16),
                            rope,
                        ),
                    ),
                    "qb": (
                        qb_rope.fused_qb_rope,
                        (
                            shape((tokens, 1280), jnp.bfloat16),
                            shape((1280, 32768), jnp.float8_e4m3fn),
                            shape((32768,), jnp.float32),
                            rope,
                        ),
                    ),
                    "compressor_r2": (
                        lambda *args: compress_rope.fused_compress_rope(*args, ratio=2),
                        (
                            x,
                            shape((5120, 1024), jnp.bfloat16),
                            shape((512,), jnp.bfloat16),
                            rope,
                            shape((tokens, 1), jnp.int32),
                            shape((tokens, 1), jnp.int32),
                            (shape((8192, 128), jnp.int32),) * 2,
                            shape((tokens,), jnp.int32),
                            shape((tokens,), jnp.int32),
                        ),
                    ),
                    "output_projection": (
                        output_projection.fused_output_projection,
                        output_inputs,
                    ),
                }
                row["jaxpr"] = {}
                for name, (function, args) in calls.items():
                    graph = jax.make_jaxpr(function)(*args)
                    row["jaxpr"][name] = {
                        "equations": len(graph.jaxpr.eqns),
                        "pallas_call": "pallas_call" in str(graph),
                    }
            rows.append(row)
    return {
        "scope": "CPU descriptors and real selectors; hardware is mocked v7",
        "jax": jax.__version__,
        "requests": requests,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buckets", nargs="+", type=int, default=[288, 512, 2048])
    parser.add_argument("--requests", type=int, default=48)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        Path(__file__).resolve().parents[2],
        tuple(args.buckets),
        args.requests,
        args.trace,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
