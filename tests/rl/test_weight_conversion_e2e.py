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
"""Tier 2 TPU E2E parity test for the Distributed RL Weight Conversion Protocol.

Verifies that a vLLM TPU engine initialized with ``load_format="dummy"`` and
populated via:

1. ``collective_rpc("get_weight_mapping_manifest")``
2. Pure-PyTorch ``from_hf(dst_rule, hf_weights)``
3. ``collective_rpc("apply_converted_weights", args=(converted, dst_rule))``
   (or ``collective_rpc("load_from_hf", ...)``)

produces identical greedy tokens and matching logprobs compared to a baseline
``LLM(...)`` loaded through standard vLLM ``load_model()``.

Run via pytest (on a TPU host):
    pytest tests/rl/test_weight_conversion_e2e.py -v

Run standalone (with custom model / TP / EP):
    python tests/rl/test_weight_conversion_e2e.py \\
        --model Qwen/Qwen2.5-0.5B --tp-size 1
    python tests/rl/test_weight_conversion_e2e.py \\
        --model Qwen/Qwen3.5-35B-A3B --tp-size 4 --enable-ep
"""

from __future__ import annotations

import argparse
import gc
import glob
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch

from vllm_torchtpu.rl.weight_conversion import from_hf, validate_manifest


# Check availability without importing torch_tpu or initializing JAX/PJRT at
# collection time (see tests/conftest.py::pytest_collection_finish).
def _has_torch_tpu() -> bool:
    try:
        return importlib.util.find_spec("torch_tpu") is not None
    except ValueError:
        return False


_HAS_TORCH_TPU = _has_torch_tpu()

DEFAULT_MODEL = os.environ.get("RL_E2E_MODEL", "Qwen/Qwen2.5-0.5B")
DEFAULT_TP_SIZE = int(os.environ.get("RL_E2E_TP_SIZE", "1"))
DEFAULT_ENABLE_EP = os.environ.get("RL_E2E_ENABLE_EP", "0") == "1"
MOE_MODEL = os.environ.get("RL_E2E_MOE_MODEL", "")
MOE_TP_SIZE = int(os.environ.get("RL_E2E_MOE_TP_SIZE", "4"))

DEFAULT_PROMPTS = [
    "The capital of France is",
    "In mathematics, the Pythagorean theorem states that",
    "Write a Python function to compute factorial:",
    "Hello, my name is",
]


def load_hf_state_dict(
    model_name_or_path: str,
    *,
    revision: str | None = None,
) -> dict[str, torch.Tensor]:
    """Load raw Hugging Face checkpoint tensors on CPU from disk or HF Hub."""
    local_dir = Path(model_name_or_path)
    if not local_dir.is_dir():
        from huggingface_hub import snapshot_download

        downloaded = snapshot_download(
            repo_id=model_name_or_path,
            revision=revision,
            allow_patterns=["*.safetensors", "*.bin"],
        )
        local_dir = Path(downloaded)

    safetensor_files = sorted(glob.glob(str(local_dir / "*.safetensors")))
    state_dict: dict[str, torch.Tensor] = {}
    if safetensor_files:
        from safetensors.torch import load_file

        for path in safetensor_files:
            state_dict.update(load_file(path, device="cpu"))
        return state_dict

    bin_files = sorted(glob.glob(str(local_dir / "pytorch_model*.bin")))
    if bin_files:
        for path in bin_files:
            loaded = torch.load(path, map_location="cpu", weights_only=True)
            state_dict.update(loaded)
        return state_dict

    raise FileNotFoundError(
        f"No .safetensors or pytorch_model*.bin files found under {local_dir}"
    )


def _extract_tokens_and_logprobs(
    outputs: list[Any],
) -> list[tuple[list[int], list[float]]]:
    """Extract per-prompt generated token IDs and chosen-token logprobs."""
    extracted: list[tuple[list[int], list[float]]] = []
    for req_out in outputs:
        completion = req_out.outputs[0]
        token_ids = list(completion.token_ids)
        chosen_logprobs: list[float] = []
        if completion.logprobs:
            for tok_id, step_lp in zip(token_ids, completion.logprobs):
                if step_lp and tok_id in step_lp:
                    entry = step_lp[tok_id]
                    chosen_logprobs.append(
                        float(entry.logprob if hasattr(entry, "logprob") else entry)
                    )
        extracted.append((token_ids, chosen_logprobs))
    return extracted


def compare_generation_outputs(
    baseline_results: list[tuple[list[int], list[float]]],
    converted_results: list[tuple[list[int], list[float]]],
    *,
    max_logprob_diff: float = 0.05,
) -> None:
    """Assert exact greedy token match and bounded logprob drift."""
    assert len(baseline_results) == len(converted_results), (
        f"Output count mismatch: {len(baseline_results)} vs {len(converted_results)}"
    )
    for idx, ((base_toks, base_lps), (conv_toks, conv_lps)) in enumerate(
        zip(baseline_results, converted_results)
    ):
        assert base_toks == conv_toks, (
            f"Prompt {idx}: token mismatch:\n"
            f"  baseline : {base_toks}\n"
            f"  converted: {conv_toks}"
        )
        assert len(base_lps) == len(conv_lps), (
            f"Prompt {idx}: logprob count mismatch: {len(base_lps)} vs {len(conv_lps)}"
        )
        for step, (b_lp, c_lp) in enumerate(zip(base_lps, conv_lps)):
            diff = abs(b_lp - c_lp)
            assert diff <= max_logprob_diff, (
                f"Prompt {idx} step {step}: logprob diff {diff:.5f} exceeds "
                f"tolerance {max_logprob_diff} "
                f"(baseline={b_lp:.5f}, converted={c_lp:.5f})"
            )


def _shutdown_llm(llm: Any) -> None:
    """Explicitly shut down vLLM's engine core before releasing references."""
    engine = getattr(llm, "llm_engine", None)
    engine_core = getattr(engine, "engine_core", None)
    if engine_core is not None and hasattr(engine_core, "shutdown"):
        engine_core.shutdown()


def run_e2e_parity(
    *,
    model: str = DEFAULT_MODEL,
    tensor_parallel_size: int = DEFAULT_TP_SIZE,
    enable_expert_parallel: bool = DEFAULT_ENABLE_EP,
    max_model_len: int = 256,
    max_tokens: int = 16,
    gpu_memory_utilization: float = 0.6,
    max_logprob_diff: float = 0.05,
    use_load_from_hf: bool = False,
    prompts: list[str] | None = None,
) -> None:
    """Run baseline vs dummy+converted-weights E2E parity check on TPU."""
    from vllm import LLM, SamplingParams
    from vllm.distributed import cleanup_dist_env_and_memory

    test_prompts = prompts or DEFAULT_PROMPTS
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        logprobs=1,
    )
    common_kwargs: dict[str, Any] = {
        "model": model,
        "tensor_parallel_size": tensor_parallel_size,
        "enable_expert_parallel": enable_expert_parallel,
        "max_model_len": max_model_len,
        "max_num_seqs": max(8, len(test_prompts)),
        "max_num_batched_tokens": max(256, max_model_len),
        "gpu_memory_utilization": gpu_memory_utilization,
        "enforce_eager": False,
    }

    # 1. Baseline: standard vLLM checkpoint load_model()
    baseline_llm = LLM(**common_kwargs)
    try:
        baseline_outputs = baseline_llm.generate(
            test_prompts, sampling_params=sampling_params
        )
        baseline_results = _extract_tokens_and_logprobs(baseline_outputs)
    finally:
        _shutdown_llm(baseline_llm)
        del baseline_llm
        gc.collect()
        cleanup_dist_env_and_memory()

    # 2. Protocol path: dummy init -> get_weight_mapping_manifest ->
    #    from_hf -> apply_converted_weights (or load_from_hf)
    hf_weights = load_hf_state_dict(model)

    rl_llm = LLM(**common_kwargs, load_format="dummy")
    try:
        manifests = rl_llm.collective_rpc("get_weight_mapping_manifest")
        assert len(manifests) == tensor_parallel_size
        dst_rule = manifests[0]
        assert isinstance(dst_rule, str)
        validate_manifest(json.loads(dst_rule))
        for rank_idx, rank_manifest in enumerate(manifests[1:], start=1):
            assert rank_manifest == dst_rule, (
                f"Worker rank {rank_idx} returned a different manifest than rank 0"
            )

        if use_load_from_hf:
            rl_llm.collective_rpc("load_from_hf", args=(hf_weights, dst_rule))
        else:
            converted_weights = from_hf(dst_rule, hf_weights)
            rl_llm.collective_rpc(
                "apply_converted_weights",
                args=(converted_weights, dst_rule),
            )

        converted_outputs = rl_llm.generate(
            test_prompts, sampling_params=sampling_params
        )
        converted_results = _extract_tokens_and_logprobs(converted_outputs)
    finally:
        _shutdown_llm(rl_llm)
        del rl_llm
        gc.collect()
        cleanup_dist_env_and_memory()

    compare_generation_outputs(
        baseline_results,
        converted_results,
        max_logprob_diff=max_logprob_diff,
    )


@pytest.mark.tpu_test
@pytest.mark.spawns_tpu_workers
@pytest.mark.skipif(
    not _HAS_TORCH_TPU,
    reason="TPU hardware / torch_tpu is required for Tier 2 E2E test",
)
@pytest.mark.parametrize("use_load_from_hf", [False, True])
def test_qwen2_5_weight_conversion_e2e(use_load_from_hf: bool) -> None:
    """Verify Qwen2.5 dense parity via dummy load + weight conversion RPC."""
    run_e2e_parity(
        model=DEFAULT_MODEL,
        tensor_parallel_size=DEFAULT_TP_SIZE,
        enable_expert_parallel=False,
        use_load_from_hf=use_load_from_hf,
    )


@pytest.mark.tpu_test
@pytest.mark.multichip
@pytest.mark.spawns_tpu_workers
@pytest.mark.skipif(
    not _HAS_TORCH_TPU or not MOE_MODEL,
    reason=(
        "Requires TPU hardware / torch_tpu and RL_E2E_MOE_MODEL set "
        "(e.g. Qwen/Qwen3.5-35B-A3B)"
    ),
)
def test_qwen3_5_moe_weight_conversion_e2e() -> None:
    """Verify Qwen3.5-MoE parity via dummy load + weight conversion RPC."""
    run_e2e_parity(
        model=MOE_MODEL,
        tensor_parallel_size=MOE_TP_SIZE,
        enable_expert_parallel=DEFAULT_ENABLE_EP,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tier 2 TPU E2E parity test for the Distributed RL Weight "
            "Conversion Protocol."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="HF model ID or local checkpoint directory.",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=DEFAULT_TP_SIZE,
        help="Tensor parallel size.",
    )
    parser.add_argument(
        "--enable-ep",
        action="store_true",
        default=DEFAULT_ENABLE_EP,
        help="Enable expert parallelism for MoE models.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=256,
        help="Maximum sequence length for the engine.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16,
        help="Number of tokens to generate per prompt.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.6,
        help="HBM utilization fraction for vLLM.",
    )
    parser.add_argument(
        "--max-logprob-diff",
        type=float,
        default=0.05,
        help="Maximum allowed absolute difference in chosen-token logprobs.",
    )
    parser.add_argument(
        "--use-load-from-hf",
        action="store_true",
        help="Use worker load_from_hf() RPC instead of host-side from_hf().",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_e2e_parity(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        enable_expert_parallel=args.enable_ep,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_logprob_diff=args.max_logprob_diff,
        use_load_from_hf=args.use_load_from_hf,
    )
    print(
        f"PASS: RL weight conversion E2E parity verified for {args.model} "
        f"(tp={args.tp_size}, ep={args.enable_ep})."
    )


if __name__ == "__main__":
    main()
