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
"""End-to-end tests for vision-encoder video frame chunking on TPU.

`MM_ENCODER_FRAME_CHUNK_PATCH_SIZE` splits a video's vision forward on the
temporal axis (see `vllm_torchtpu.runner.mm_video_chunking`). For Qwen2.5-VL
that split is lossless: vision attention is segmented per temporal patch
(`cu_seqlens` has one entry per frame) and the vision RoPE is spatial-only,
so each frame is encoded identically however the video is chunked.

These tests hold the feature to that claim end to end: greedy generation with
chunking on must produce exactly the same token ids as with chunking off.
They also check that the output still describes the video.

Chunk budgets are derived from the reference run's video-token count instead
of hard-coded, so the tests keep exercising a multi-chunk split even if the
processor's resize policy changes the patch grid. Every chunked case asserts
the video really exceeded its budget; a budget the video fits in would take
the unchunked path and make the exactness check vacuous.

Both the eager encoder path (`cudagraph_mm_encoder=False`, via
`chunked_video_encoding`) and the compiled encoder path
(`cudagraph_mm_encoder=True`, via `MMEncoderManager._execute_local` when total
batch tokens exceed the largest compiled bucket) are tested here.

Run with:
    pytest tests/entrypoints/llm/test_mm_video_chunking_e2e.py -v
"""

from __future__ import annotations

import multiprocessing
import os
import queue as queue_lib
import traceback
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pytest
from PIL import Image
from vllm import LLM, EngineArgs, SamplingParams
from vllm.assets.video import VideoAsset

from vllm_torchtpu import tpu_info

pytestmark = pytest.mark.tpu_test

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
CHUNK_ENV = "MM_ENCODER_FRAME_CHUNK_PATCH_SIZE"

# Frames are resized up front so the grid does not depend on the source
# resolution. 336 is a multiple of patch_size * spatial_merge_size (14 * 2)
# and 336*336 sits inside Qwen2.5-VL's default per-frame video pixel range, so
# the processor keeps it: a 24x24 patch grid per frame.
FRAME_SIZE = 336
# Qwen2.5-VL merges 2x2 patches into one LLM token (spatial_merge_size=2).
PATCHES_PER_VIDEO_TOKEN = 4
# Compiled encoder token budget for chunked runs: half of the 16-frame video
# (4 temporal frames = 576 tokens = 2304 patches), so total_tokens > max_budget
# triggers MMEncoderManager._execute_local chunking.
CHUNK_TOKEN_BUDGET = 576
MAX_TOKENS = 48
# Worker results arrive over a queue; a child that dies hard never sends one.
WORKER_TIMEOUT_S = 1500

# The baby_reading clip shows a baby in glasses reading a book.
EXPECTED_KEYWORDS = ("baby", "child", "toddler", "kid", "book", "read")


def _get_tensor_parallel_size() -> int:
    tpu_type = tpu_info.get_tpu_type() or ""
    if tpu_type.startswith("tpu7x") or os.environ.get("TPU_VERSION") == "tpu7x":
        return 2
    return 1


def _build_qwen2_5_vl_video_prompt(question: str) -> str:
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _load_video(num_frames: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Loads RGB video frames resized to a fixed square resolution and its metadata.

    Args:
        num_frames: Number of video frames to load. Must be even because
            Qwen2.5-VL packs frames in temporal pairs.

    Returns:
        Tuple of resized RGB frames array and video metadata.
    """
    asset = VideoAsset(name="baby_reading", num_frames=num_frames)
    frames = asset.np_ndarrays
    video = np.stack(
        [
            np.asarray(
                Image.fromarray(frame)
                .convert("RGB")
                .resize((FRAME_SIZE, FRAME_SIZE), Image.BICUBIC)
            )
            for frame in frames
        ]
    )
    raw_metadata = getattr(asset, "video_metadata", None)
    if isinstance(raw_metadata, dict):
        metadata = dict(raw_metadata)
    elif hasattr(raw_metadata, "__dict__"):
        metadata = dict(vars(raw_metadata))
    else:
        metadata = {
            "fps": 1.0,
            "duration": float(num_frames),
            "total_num_frames": num_frames,
            "video_backend": "opencv",
            "frames_indices": list(range(num_frames)),
            "do_sample_frames": True,
        }
    return video, metadata


@dataclass(frozen=True)
class GenerationResult:
    token_ids: tuple[int, ...]
    text: str
    prompt_token_ids: tuple[int, ...]
    num_video_tokens: int

    @property
    def num_video_patches(self) -> int:
        """Returns the number of vision-encoder patches (t*h*w) for this video."""
        return self.num_video_tokens * PATCHES_PER_VIDEO_TOKEN


def _run_video_generation_worker(
    prompts: list[str],
    videos: list[np.ndarray | tuple[np.ndarray, dict[str, Any]]],
    chunk_patch_size: int,
    max_num_seqs: int,
    cudagraph_mm_encoder: bool,
    queue: multiprocessing.Queue,
) -> None:
    """Instantiates LLM and runs greedy video generation in a child process.

    Args:
        prompts: Text prompts for generation.
        videos: Video arrays corresponding to each prompt.
        chunk_patch_size: Maximum patch size per chunk, or 0 to disable.
        max_num_seqs: Maximum number of sequences to process in parallel.
        cudagraph_mm_encoder: Whether to enable `MMEncoderManager`.
        queue: Multiprocessing queue to send back results or errors.
    """
    try:
        os.environ["SKIP_JAX_PRECOMPILE"] = "1"
        os.environ["VLLM_XLA_CHECK_RECOMPILATION"] = "0"
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        # Set explicitly, including 0, so an inherited value can't turn the
        # reference run into a chunked one. Engine-core and worker processes
        # are spawned from here, after this, so they inherit it.
        os.environ[CHUNK_ENV] = str(chunk_patch_size)

        compilation_config: dict[str, object] = {
            "cudagraph_mm_encoder": cudagraph_mm_encoder,
            "cudagraph_capture_sizes": [],
        }
        if cudagraph_mm_encoder:
            compilation_config["encoder_cudagraph_token_budgets"] = [CHUNK_TOKEN_BUDGET]

        engine_args = EngineArgs(
            model=MODEL_NAME,
            max_model_len=8192,
            tensor_parallel_size=_get_tensor_parallel_size(),
            gpu_memory_utilization=0.5,
            max_num_seqs=max_num_seqs,
            compilation_config=compilation_config,
            limit_mm_per_prompt={"image": 0, "video": 1},
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
        video_pad_id = llm.get_tokenizer().convert_tokens_to_ids("<|video_pad|>")

        inputs = [
            {
                "prompt": prompt,
                "multi_modal_data": {
                    "video": (
                        video
                        if isinstance(video, tuple)
                        else (
                            video,
                            getattr(video, "video_metadata", None)
                            or {
                                "fps": 1.0,
                                "duration": float(len(video)),
                                "total_num_frames": len(video),
                                "video_backend": "opencv",
                                "frames_indices": list(range(len(video))),
                                "do_sample_frames": True,
                            },
                        )
                    )
                },
            }
            for prompt, video in zip(prompts, videos)
        ]
        outputs = llm.generate(
            inputs,
            sampling_params=SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS),
        )

        results = [
            GenerationResult(
                token_ids=tuple(out.outputs[0].token_ids),
                text=out.outputs[0].text.strip(),
                prompt_token_ids=tuple(out.prompt_token_ids),
                num_video_tokens=sum(
                    1 for tok in out.prompt_token_ids if tok == video_pad_id
                ),
            )
            for out in outputs
        ]
        queue.put(("SUCCESS", results))
    except Exception:
        queue.put(("ERROR", traceback.format_exc()))


def run_video_generation_in_isolated_process(
    prompts: list[str],
    videos: list[np.ndarray | tuple[np.ndarray, dict[str, Any]]],
    chunk_patch_size: int,
    max_num_seqs: int = 1,
    cudagraph_mm_encoder: bool = False,
) -> list[GenerationResult]:
    """Runs generation in an isolated OS process so TPU devices are released.

    Args:
        prompts: Text prompts for generation.
        videos: Video arrays corresponding to each prompt.
        chunk_patch_size: Maximum patch size per chunk, or 0 to disable.
        max_num_seqs: Maximum number of sequences to process in parallel.
        cudagraph_mm_encoder: Whether to enable `MMEncoderManager`.

    Returns:
        List of GenerationResult for each prompt/video input.

    Raises:
        RuntimeError: If the worker process times out or encounters an error.
    """
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_run_video_generation_worker,
        args=(
            prompts,
            videos,
            chunk_patch_size,
            max_num_seqs,
            cudagraph_mm_encoder,
            queue,
        ),
    )
    proc.start()
    try:
        # Drain before join: a child blocks on exit until its queued payload
        # has been read.
        status, result = queue.get(timeout=WORKER_TIMEOUT_S)
    except queue_lib.Empty:
        proc.kill()
        proc.join()
        raise RuntimeError(
            f"Video generation worker ({CHUNK_ENV}={chunk_patch_size}) sent "
            f"no result within {WORKER_TIMEOUT_S}s; exit code "
            f"{proc.exitcode}."
        ) from None
    proc.join()
    if status == "ERROR":
        raise RuntimeError(
            f"Video generation worker ({CHUNK_ENV}={chunk_patch_size}) "
            f"failed:\n{result}"
        )
    return result


def _assert_describes_video(text: str, label: str) -> None:
    """Asserts that generated text contains at least one expected keyword.

    Args:
        text: Generated output text.
        label: Descriptive label for the test case in assertion messages.
    """
    lowered = text.lower()
    assert any(keyword in lowered for keyword in EXPECTED_KEYWORDS), (
        f"{label} output does not describe the video; expected one of "
        f"{EXPECTED_KEYWORDS}.\nActual: {text}"
    )


def _first_divergence(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    """Returns the index of the first differing token between two sequences.

    Args:
        a: First sequence of token IDs.
        b: Second sequence of token IDs.

    Returns:
        Index of the first differing token, or the shorter length if one is a
        prefix of the other.
    """
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def _assert_bit_exact(
    reference: GenerationResult, chunked: GenerationResult, label: str
) -> None:
    """Asserts that chunked generation output matches reference bit-for-bit.

    Args:
        reference: Generation result from the unchunked reference run.
        chunked: Generation result from the chunked run.
        label: Descriptive label for error reporting.
    """
    # Same processor, same video -> same prompt. If this differs, the two
    # runs saw different inputs and the output comparison means nothing.
    assert chunked.prompt_token_ids == reference.prompt_token_ids, (
        f"{label}: prompt token ids differ between runs; the runs did not "
        f"see the same input."
    )
    if chunked.token_ids == reference.token_ids:
        return
    first_diff = _first_divergence(reference.token_ids, chunked.token_ids)
    pytest.fail(
        f"{label}: chunked output diverges from the unchunked "
        f"reference at generated token {first_diff}.\n"
        f"reference: {reference.text!r}\n"
        f"chunked:   {chunked.text!r}"
    )


@pytest.fixture(scope="module")
def single_video() -> tuple[np.ndarray, dict[str, Any]]:
    """Loads a 16-frame test video."""
    return _load_video(num_frames=16)


@pytest.fixture(scope="module")
def single_video_prompt() -> str:
    """Builds a prompt requesting a description of the test video."""
    return _build_qwen2_5_vl_video_prompt("Describe this video.")


@pytest.fixture(scope="module")
def reference_result(
    single_video: tuple[np.ndarray, dict[str, Any]], single_video_prompt: str
) -> GenerationResult:
    """Runs unchunked greedy video generation shared across tests in this module.

    Args:
        single_video: Test video array fixture.
        single_video_prompt: Prompt fixture.

    Returns:
        Unchunked reference GenerationResult.
    """
    [result] = run_video_generation_in_isolated_process(
        prompts=[single_video_prompt],
        videos=[single_video],
        chunk_patch_size=0,
    )
    return result


@pytest.mark.timeout(1800)
def test_video_inference_reference(reference_result: GenerationResult):
    """Tests that unchunked video inference produces a valid description.

    Guards the baseline the exactness tests compare against: if the reference
    is garbage, matching it proves nothing.
    """
    assert reference_result.num_video_tokens > 0, (
        "No <|video_pad|> tokens in the prompt; the video was not attached."
    )
    _assert_describes_video(reference_result.text, "Unchunked")


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("cudagraph_mm_encoder", [False, True])
def test_video_baseline_is_deterministic(
    cudagraph_mm_encoder: bool,
    single_video: tuple[np.ndarray, dict[str, Any]],
    single_video_prompt: str,
    reference_result: GenerationResult,
):
    """Tests that two unchunked runs agree token for token.

    Exact comparison against chunked runs is only meaningful if the baseline
    reproduces itself. If this fails, a chunked mismatch below is not evidence
    against chunking.
    """
    [rerun] = run_video_generation_in_isolated_process(
        prompts=[single_video_prompt],
        videos=[single_video],
        chunk_patch_size=0,
        cudagraph_mm_encoder=cudagraph_mm_encoder,
    )
    _assert_bit_exact(
        reference_result,
        rerun,
        f"Unchunked rerun (cudagraph_mm_encoder={cudagraph_mm_encoder})",
    )


# Budgets as a fraction of the video's total patches, so the split pattern
# holds regardless of the exact grid. With the default 16 frames (t=8):
#   half        -> 4 temporal patches per forward, 2 even chunks
#   three_eighths -> 3 per forward, a ragged [3, 3, 2] split
#   one_patch   -> budget below one frame, clamped to 1 temporal patch per
#                  forward, 8 chunks
_BUDGETS = {
    "half": lambda total: total // 2,
    "three_eighths": lambda total: 3 * total // 8,
    "one_patch": lambda total: 1,
}


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("cudagraph_mm_encoder", [False, True])
@pytest.mark.parametrize("budget_name", list(_BUDGETS))
def test_video_chunking_is_bit_exact(
    cudagraph_mm_encoder: bool,
    budget_name: str,
    single_video: tuple[np.ndarray, dict[str, Any]],
    single_video_prompt: str,
    reference_result: GenerationResult,
):
    """Tests that chunked greedy output is identical to unchunked reference."""
    total_patches = reference_result.num_video_patches
    budget = _BUDGETS[budget_name](total_patches)
    assert 0 < budget < total_patches, (
        f"budget {budget} does not split a {total_patches}-patch video; the "
        f"run would take the unchunked path and prove nothing."
    )

    [chunked] = run_video_generation_in_isolated_process(
        prompts=[single_video_prompt],
        videos=[single_video],
        chunk_patch_size=budget,
        cudagraph_mm_encoder=cudagraph_mm_encoder,
    )

    label = (
        f"{CHUNK_ENV}={budget} ({budget_name}, {total_patches} patches, "
        f"cudagraph_mm_encoder={cudagraph_mm_encoder})"
    )
    _assert_describes_video(chunked.text, label)
    _assert_bit_exact(reference_result, chunked, label)


@pytest.fixture(scope="module")
def batched_videos_and_reference() -> tuple[
    list[str], list[np.ndarray], list[GenerationResult]
]:
    """Runs unchunked batched video generation shared across batched tests."""
    prompts = [
        _build_qwen2_5_vl_video_prompt("Describe this video."),
        _build_qwen2_5_vl_video_prompt("What is the baby doing in this video?"),
    ]
    videos = [_load_video(num_frames=16), _load_video(num_frames=8)]
    reference = run_video_generation_in_isolated_process(
        prompts=prompts,
        videos=videos,
        chunk_patch_size=0,
        max_num_seqs=2,
    )
    return prompts, videos, reference


@pytest.mark.timeout(2400)
@pytest.mark.parametrize("cudagraph_mm_encoder", [False, True])
def test_batched_videos_chunking_is_bit_exact(
    cudagraph_mm_encoder: bool,
    batched_videos_and_reference: tuple[
        list[str], list[np.ndarray], list[GenerationResult]
    ],
):
    """Tests that batched videos of different lengths stay bit-exact.

    Covers the per-item split: when the scheduler hands both videos to one
    encoder call and their combined patches exceed the budget, the group is
    split back into per-item forwards and each item is chunked on its own.
    """
    prompts, videos, reference = batched_videos_and_reference
    # Half the shorter video: both videos exceed it, so both get chunked.
    budget = min(r.num_video_patches for r in reference) // 2
    assert budget > 0

    chunked = run_video_generation_in_isolated_process(
        prompts=prompts,
        videos=videos,
        chunk_patch_size=budget,
        max_num_seqs=2,
        cudagraph_mm_encoder=cudagraph_mm_encoder,
    )

    assert len(chunked) == len(reference) == 2
    for i, (ref, out) in enumerate(zip(reference, chunked)):
        label = (
            f"video {i} ({ref.num_video_patches} patches), {CHUNK_ENV}={budget}, "
            f"cudagraph_mm_encoder={cudagraph_mm_encoder}"
        )
        assert ref.num_video_patches > budget, label
        _assert_describes_video(ref.text, f"Unchunked {label}")
        _assert_bit_exact(ref, out, label)
