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
"""End-to-end tests for vLLM multi-modal LLM generation API on TPU.

This module tests multimodal vision-language model inference using Qwen2.5-VL,
verifying that the MMEncoderManager (cudagraph_mm_encoder) compiles and executes
the vision encoder forward correctly on TPU.

Run with:
    pytest tests/entrypoints/llm/test_multi_modal_inference.py -v
"""

from __future__ import annotations

import difflib
import multiprocessing
import os
import traceback
from dataclasses import asdict

import pytest
from vllm import LLM, EngineArgs, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.multimodal.image import convert_image_mode

from vllm_torchtpu import tpu_info

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

# Known-good partial text outputs for the cherry-blossom image.
EXPECTED_TEXTS = (
    "The image depicts a tall, cylindrical tower with a lattice-like structure, "
    "surrounded by cherry blossom trees in full bloom. The cherry blossoms are in "
    "various stages of opening, with pink petals covering the branches. The sky is "
    "clear and blue, providing a vibrant backdrop to the scene. The tower appears to "
    "be a significant landmark",
    "The image depicts a stunning view of the Tokyo Skytree, a tall broadcasting tower "
    "located in the Odaiba district of Tokyo, Japan. "
    "The skytree is surrounded by cherry "
    "blossom trees in full bloom, creating a picturesque and vibrant scene. The cherry "
    "blossoms are in various stages of bloom, with some branches densely covered",
)


def _get_tensor_parallel_size() -> int:
    tpu_type = tpu_info.get_tpu_type() or ""
    if tpu_type.startswith("tpu7x") or os.environ.get("TPU_VERSION") == "tpu7x":
        return 2
    return 1


def _build_qwen2_5_vl_prompt(question: str) -> str:
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _run_mm_generation_worker(
    prompts: list[str],
    images: list,
    cudagraph_mm_encoder: bool,
    max_tokens: int,
    max_num_seqs: int,
    queue: multiprocessing.Queue,
) -> None:
    """Instantiate LLM and execute multi-modal generation in a child process."""
    try:
        os.environ["SKIP_JAX_PRECOMPILE"] = "1"
        os.environ["VLLM_XLA_CHECK_RECOMPILATION"] = "0"
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        tp_size = _get_tensor_parallel_size()

        compilation_config: dict = {
            "cudagraph_mm_encoder": cudagraph_mm_encoder,
            "cudagraph_capture_sizes": [],
        }

        engine_args = EngineArgs(
            model=MODEL_NAME,
            max_model_len=4096,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=0.5,
            max_num_seqs=max_num_seqs,
            compilation_config=compilation_config,
            mm_processor_kwargs={
                "size": {"longest_edge": 1003520, "shortest_edge": 3136},
                "fps": 1,
            },
            limit_mm_per_prompt={"image": 1},
            enforce_eager=False,
            disable_log_stats=True,
        )

        args_dict = asdict(engine_args)
        args_dict.pop("fault_tolerance_config", None)

        pass_config = args_dict.get("compilation_config", {}).get("pass_config") or {}
        args_dict["compilation_config"]["pass_config"] = {
            k: v for k, v in pass_config.items() if v is not None
        }

        llm = LLM(**args_dict)

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
        )

        if len(prompts) == 1:
            inputs = {
                "prompt": prompts[0],
                "multi_modal_data": {"image": images[0]},
            }
        else:
            inputs = [
                {
                    "prompt": p,
                    "multi_modal_data": {"image": img},
                }
                for p, img in zip(prompts, images)
            ]

        outputs = llm.generate(inputs, sampling_params=sampling_params)
        generated_texts = [o.outputs[0].text.strip() for o in outputs]
        queue.put(("SUCCESS", generated_texts))
    except Exception:
        queue.put(("ERROR", traceback.format_exc()))


def run_mm_generation_in_isolated_process(
    prompts: list[str],
    images: list,
    cudagraph_mm_encoder: bool = True,
    max_tokens: int = 64,
    max_num_seqs: int = 1,
) -> list[str]:
    """Runs generation in an isolated OS process to ensure clean device cleanup."""
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    p = ctx.Process(
        target=_run_mm_generation_worker,
        args=(prompts, images, cudagraph_mm_encoder, max_tokens, max_num_seqs, queue),
    )
    p.start()
    p.join()
    status, result = queue.get()
    if status == "ERROR":
        raise RuntimeError(f"Isolated multimodal worker failed:\n{result}")
    return result


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("cudagraph_mm_encoder", [True, False])
def test_multi_modal_inference(cudagraph_mm_encoder: bool):
    """Verifies that multi-modal inference accurately describes an image."""
    image = convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB")
    prompt = _build_qwen2_5_vl_prompt("What is the content of this image?")

    generated_texts = run_mm_generation_in_isolated_process(
        prompts=[prompt],
        images=[image],
        cudagraph_mm_encoder=cudagraph_mm_encoder,
        max_tokens=64,
        max_num_seqs=1,
    )

    generated_text = generated_texts[0]
    similarity_score = max(
        difflib.SequenceMatcher(None, generated_text, expected, autojunk=False).ratio()
        for expected in EXPECTED_TEXTS
    )

    assert similarity_score >= 0.85, (
        f"Multi-modal text similarity too low ({similarity_score:.2f}) "
        f"with cudagraph_mm_encoder={cudagraph_mm_encoder}.\n"
        f"Expected one of: {EXPECTED_TEXTS}\n"
        f"Actual: {generated_text}"
    )


@pytest.mark.timeout(1200)
def test_multi_modal_batch_inference_with_mm_encoder_manager():
    """Verifies batch multi-modal inference with MMEncoderManager."""
    image = convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB")
    questions = [
        "What is the content of this image?",
        "Describe the colors and scenery in this image.",
    ]
    prompts = [_build_qwen2_5_vl_prompt(q) for q in questions]
    images = [image, image]

    generated_texts = run_mm_generation_in_isolated_process(
        prompts=prompts,
        images=images,
        cudagraph_mm_encoder=True,
        max_tokens=32,
        max_num_seqs=2,
    )

    assert len(generated_texts) == 2
    for text in generated_texts:
        assert len(text) > 0, "Generated batch response should not be empty."
