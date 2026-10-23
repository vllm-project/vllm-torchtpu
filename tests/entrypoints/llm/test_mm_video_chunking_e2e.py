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
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pytest
from PIL import Image
from vllm import LLM, EngineArgs, SamplingParams
from vllm.assets.video import VideoAsset

from vllm_torchtpu import tpu_info
from vllm_torchtpu.runner.mm_video_chunking import get_video_chunk_patch_size

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
# Each suite worker boots once and runs up to 7 generates (including lazy
# encoder compiles for half, 3/8, one_patch, and batched chunk shapes).
WORKER_TIMEOUT_S = 2100
# Selecting only the compiled cases (`-k True`) makes the first compiled test
# set up both `eager_encoder_suite` (via `reference_result`) and
# `compiled_encoder_suite` inside a single test timeout window.
TEST_TIMEOUT_S = 2 * WORKER_TIMEOUT_S + 300

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


# A 16-frame 336x336 video is 8*12*12 = 1152 vision tokens + ~16 prompt tokens
# + 48 output tokens (~1216 total), and the 2-video batch (16 + 8 frames) is
# 1728 vision tokens + ~32 prompt tokens (~1760 prefill tokens). Capping
# max_num_batched_tokens at 2048 avoids compiling the 4096 and 8192 token
# buckets that no step in this file reaches, while keeping max_model_len=8192
# preserves the Pallas RPA page size and tuned block sizes.
MAX_MODEL_LEN = 8192
MAX_NUM_BATCHED_TOKENS = 2048


def _batched_chunk_budget(results: list[GenerationResult]) -> int:
    """Half the shorter video's patches so every video in the batch is chunked."""
    return min(r.num_video_patches for r in results) // 2


@dataclass(frozen=True)
class CaseOutcome:
    """Value or captured traceback for one generation case inside a suite worker."""

    value: Any = None
    error: str | None = None

    def unwrap(self, label: str) -> Any:
        if self.error is not None:
            pytest.fail(f"{label} failed in suite worker:\n{self.error}")
        assert self.value is not None
        return self.value


def _capture_case(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> CaseOutcome:
    try:
        return CaseOutcome(value=fn(*args, **kwargs))
    except Exception:
        return CaseOutcome(error=traceback.format_exc())


def _set_video_chunk_patch_size(self, chunk_patch_size: int) -> int:
    os.environ[CHUNK_ENV] = str(chunk_patch_size)
    return get_video_chunk_patch_size(self.model_runner.model_config)


def _format_video_inputs(
    prompts: list[str],
    videos: list[tuple[np.ndarray, dict[str, Any]]],
) -> list[dict[str, Any]]:
    # Assign a fresh UUID per request so Scheduler.encoder_cache_manager and
    # TPUModelRunner.encoder_cache compute a distinct mm_hash on every call and
    # re-run the vision encoder under the current MM_ENCODER_FRAME_CHUNK_PATCH_SIZE.
    return [
        {
            "prompt": prompt,
            "multi_modal_data": {"video": video},
            "multi_modal_uuids": {"video": uuid.uuid4().hex},
        }
        for prompt, video in zip(prompts, videos)
    ]


def _extract_generation_results(
    outputs: list[Any],
    video_pad_id: int,
) -> list[GenerationResult]:
    return [
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


def _build_llm(
    max_num_seqs: int,
    cudagraph_mm_encoder: bool,
    initial_chunk_patch_size: int = 0,
) -> LLM:
    os.environ["SKIP_JAX_PRECOMPILE"] = "1"
    os.environ["VLLM_XLA_CHECK_RECOMPILATION"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    # Set explicitly, including 0, so an inherited value can't turn the
    # reference run into a chunked one. Engine-core and worker processes
    # are spawned from here, after this, so they inherit it.
    os.environ[CHUNK_ENV] = str(initial_chunk_patch_size)

    compilation_config: dict[str, object] = {
        "cudagraph_mm_encoder": cudagraph_mm_encoder,
        "cudagraph_capture_sizes": [],
    }
    if cudagraph_mm_encoder:
        compilation_config["encoder_cudagraph_token_budgets"] = [CHUNK_TOKEN_BUDGET]
        # EncoderCudaGraphManager otherwise derives max_batch_size (and from it
        # max_frames_per_batch and the static cu_seqlens template length) from
        # min(max_num_batched_tokens, max_model_len) // min_budget (64). Keep
        # the 8192 // 64 = 128 template shape when max_num_batched_tokens=2048.
        compilation_config["encoder_cudagraph_max_vision_items_per_batch"] = 128

    engine_args = EngineArgs(
        model=MODEL_NAME,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
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
    return LLM(**args_dict)


def _run_with_chunk_budget(
    llm: LLM,
    video_pad_id: int,
    prompts: list[str],
    videos: list[tuple[np.ndarray, dict[str, Any]]],
    chunk_patch_size: int,
) -> list[GenerationResult]:
    worker_budgets = llm.llm_engine.collective_rpc(
        _set_video_chunk_patch_size,
        args=(chunk_patch_size,),
    )
    assert worker_budgets and all(b == chunk_patch_size for b in worker_budgets), (
        f"Workers did not observe {CHUNK_ENV}={chunk_patch_size}: got {worker_budgets}"
    )
    outputs = llm.generate(
        _format_video_inputs(prompts, videos),
        sampling_params=SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS),
    )
    return _extract_generation_results(outputs, video_pad_id)


def _run_single_with_chunk_budget(
    llm: LLM,
    video_pad_id: int,
    prompt: str,
    video: tuple[np.ndarray, dict[str, Any]],
    chunk_patch_size: int,
) -> GenerationResult:
    [result] = _run_with_chunk_budget(
        llm,
        video_pad_id,
        [prompt],
        [video],
        chunk_patch_size=chunk_patch_size,
    )
    return result


@dataclass(frozen=True)
class EncoderModeSuiteResult:
    """Per-case video generation outcomes for one `cudagraph_mm_encoder` setting."""

    reference: CaseOutcome | None
    rerun: CaseOutcome
    chunked_by_budget: dict[str, CaseOutcome]
    batched_reference: CaseOutcome
    batched_chunked: CaseOutcome


def _run_encoder_mode_suite_worker(
    single_prompt: str,
    single_video: tuple[np.ndarray, dict[str, Any]],
    batched_prompts: list[str],
    batched_videos: list[tuple[np.ndarray, dict[str, Any]]],
    cudagraph_mm_encoder: bool,
    include_single_reference: bool,
    queue: multiprocessing.Queue,
) -> None:
    """Runs all chunk-budget checks for one encoder mode in a single LLM boot."""
    try:
        # Both single-video (num_reqs=1) and 2-video (num_reqs=2) runs pad to
        # MIN_NUM_SEQS=8 on TPU, so one max_num_seqs=2 engine serves both
        # without compiling any extra sequence or token buckets.
        llm = _build_llm(
            max_num_seqs=2,
            cudagraph_mm_encoder=cudagraph_mm_encoder,
            initial_chunk_patch_size=0,
        )
        video_pad_id = llm.get_tokenizer().convert_tokens_to_ids("<|video_pad|>")

        reference: CaseOutcome | None = None
        if include_single_reference:
            reference = _capture_case(
                _run_single_with_chunk_budget,
                llm,
                video_pad_id,
                single_prompt,
                single_video,
                chunk_patch_size=0,
            )

        rerun = _capture_case(
            _run_single_with_chunk_budget,
            llm,
            video_pad_id,
            single_prompt,
            single_video,
            chunk_patch_size=0,
        )

        baseline_single = (
            reference
            if reference is not None and reference.value is not None
            else rerun
        )
        chunked_by_budget: dict[str, CaseOutcome] = {}
        for budget_name, budget_fn in _BUDGETS.items():
            if baseline_single.value is None:
                chunked_by_budget[budget_name] = CaseOutcome(
                    error=baseline_single.error
                )
                continue
            budget = budget_fn(baseline_single.value.num_video_patches)
            chunked_by_budget[budget_name] = _capture_case(
                _run_single_with_chunk_budget,
                llm,
                video_pad_id,
                single_prompt,
                single_video,
                chunk_patch_size=budget,
            )

        batched_reference = _capture_case(
            _run_with_chunk_budget,
            llm,
            video_pad_id,
            batched_prompts,
            batched_videos,
            chunk_patch_size=0,
        )
        if batched_reference.value is None:
            batched_chunked = CaseOutcome(error=batched_reference.error)
        else:
            batched_budget = _batched_chunk_budget(batched_reference.value)
            batched_chunked = _capture_case(
                _run_with_chunk_budget,
                llm,
                video_pad_id,
                batched_prompts,
                batched_videos,
                chunk_patch_size=batched_budget,
            )

        queue.put(
            (
                "SUCCESS",
                EncoderModeSuiteResult(
                    reference=reference,
                    rerun=rerun,
                    chunked_by_budget=chunked_by_budget,
                    batched_reference=batched_reference,
                    batched_chunked=batched_chunked,
                ),
            )
        )
    except Exception:
        queue.put(("ERROR", traceback.format_exc()))


def _run_encoder_mode_suite_in_isolated_process(
    single_prompt: str,
    single_video: tuple[np.ndarray, dict[str, Any]],
    batched_prompts: list[str],
    batched_videos: list[tuple[np.ndarray, dict[str, Any]]],
    cudagraph_mm_encoder: bool,
    include_single_reference: bool,
) -> EncoderModeSuiteResult:
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_run_encoder_mode_suite_worker,
        args=(
            single_prompt,
            single_video,
            batched_prompts,
            batched_videos,
            cudagraph_mm_encoder,
            include_single_reference,
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
            f"Video suite worker (cudagraph_mm_encoder={cudagraph_mm_encoder}) "
            f"sent no result within {WORKER_TIMEOUT_S}s; exit code "
            f"{proc.exitcode}."
        ) from None
    proc.join()
    if status == "ERROR":
        raise RuntimeError(
            f"Video suite worker (cudagraph_mm_encoder={cudagraph_mm_encoder}) "
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
def batched_video_inputs(
    single_video: tuple[np.ndarray, dict[str, Any]],
) -> tuple[list[str], list[tuple[np.ndarray, dict[str, Any]]]]:
    """Builds prompts and videos for the 2-video batched tests."""
    prompts = [
        _build_qwen2_5_vl_video_prompt("Describe this video."),
        _build_qwen2_5_vl_video_prompt("What is the baby doing in this video?"),
    ]
    videos = [single_video, _load_video(num_frames=8)]
    return prompts, videos


@pytest.fixture(scope="module")
def eager_encoder_suite(
    single_video: tuple[np.ndarray, dict[str, Any]],
    single_video_prompt: str,
    batched_video_inputs: tuple[list[str], list[tuple[np.ndarray, dict[str, Any]]]],
) -> EncoderModeSuiteResult:
    """Runs all cudagraph_mm_encoder=False cases in one isolated LLM boot."""
    batched_prompts, batched_videos = batched_video_inputs
    return _run_encoder_mode_suite_in_isolated_process(
        single_prompt=single_video_prompt,
        single_video=single_video,
        batched_prompts=batched_prompts,
        batched_videos=batched_videos,
        cudagraph_mm_encoder=False,
        include_single_reference=True,
    )


@pytest.fixture(scope="module")
def compiled_encoder_suite(
    single_video: tuple[np.ndarray, dict[str, Any]],
    single_video_prompt: str,
    batched_video_inputs: tuple[list[str], list[tuple[np.ndarray, dict[str, Any]]]],
) -> EncoderModeSuiteResult:
    """Runs all cudagraph_mm_encoder=True cases in one isolated LLM boot."""
    batched_prompts, batched_videos = batched_video_inputs
    return _run_encoder_mode_suite_in_isolated_process(
        single_prompt=single_video_prompt,
        single_video=single_video,
        batched_prompts=batched_prompts,
        batched_videos=batched_videos,
        cudagraph_mm_encoder=True,
        include_single_reference=False,
    )


@pytest.fixture(scope="module")
def reference_result(eager_encoder_suite: EncoderModeSuiteResult) -> GenerationResult:
    """Returns unchunked greedy video generation shared across tests in this module."""
    assert eager_encoder_suite.reference is not None
    return eager_encoder_suite.reference.unwrap("Unchunked reference")


@pytest.mark.timeout(TEST_TIMEOUT_S)
def test_video_inference_reference(reference_result: GenerationResult):
    """Tests that unchunked video inference produces a valid description.

    Guards the baseline the exactness tests compare against: if the reference
    is garbage, matching it proves nothing.
    """
    assert reference_result.num_video_tokens > 0, (
        "No <|video_pad|> tokens in the prompt; the video was not attached."
    )
    _assert_describes_video(reference_result.text, "Unchunked")


@pytest.mark.timeout(TEST_TIMEOUT_S)
@pytest.mark.parametrize("cudagraph_mm_encoder", [False, True])
def test_video_baseline_is_deterministic(
    cudagraph_mm_encoder: bool,
    reference_result: GenerationResult,
    request: pytest.FixtureRequest,
):
    """Tests that two unchunked runs agree token for token.

    Exact comparison against chunked runs is only meaningful if the baseline
    reproduces itself. If this fails, a chunked mismatch below is not evidence
    against chunking.
    """
    suite: EncoderModeSuiteResult = request.getfixturevalue(
        "compiled_encoder_suite" if cudagraph_mm_encoder else "eager_encoder_suite"
    )
    label = f"Unchunked rerun (cudagraph_mm_encoder={cudagraph_mm_encoder})"
    rerun = suite.rerun.unwrap(label)
    _assert_bit_exact(
        reference_result,
        rerun,
        label,
    )


@pytest.mark.timeout(TEST_TIMEOUT_S)
@pytest.mark.parametrize("cudagraph_mm_encoder", [False, True])
@pytest.mark.parametrize("budget_name", list(_BUDGETS))
def test_video_chunking_is_bit_exact(
    cudagraph_mm_encoder: bool,
    budget_name: str,
    reference_result: GenerationResult,
    request: pytest.FixtureRequest,
):
    """Tests that chunked greedy output is identical to unchunked reference."""
    total_patches = reference_result.num_video_patches
    budget = _BUDGETS[budget_name](total_patches)
    assert 0 < budget < total_patches, (
        f"budget {budget} does not split a {total_patches}-patch video; the "
        f"run would take the unchunked path and prove nothing."
    )

    suite: EncoderModeSuiteResult = request.getfixturevalue(
        "compiled_encoder_suite" if cudagraph_mm_encoder else "eager_encoder_suite"
    )
    label = (
        f"{CHUNK_ENV}={budget} ({budget_name}, {total_patches} patches, "
        f"cudagraph_mm_encoder={cudagraph_mm_encoder})"
    )
    chunked = suite.chunked_by_budget[budget_name].unwrap(label)

    _assert_describes_video(chunked.text, label)
    _assert_bit_exact(reference_result, chunked, label)


@pytest.fixture(scope="module")
def batched_videos_and_reference(
    batched_video_inputs: tuple[list[str], list[tuple[np.ndarray, dict[str, Any]]]],
    eager_encoder_suite: EncoderModeSuiteResult,
) -> tuple[list[str], list[tuple[np.ndarray, dict[str, Any]]], list[GenerationResult]]:
    """Returns unchunked batched video generation shared across batched tests."""
    prompts, videos = batched_video_inputs
    return (
        prompts,
        videos,
        eager_encoder_suite.batched_reference.unwrap("Unchunked batched reference"),
    )


@pytest.mark.timeout(TEST_TIMEOUT_S)
@pytest.mark.parametrize("cudagraph_mm_encoder", [False, True])
def test_batched_videos_chunking_is_bit_exact(
    cudagraph_mm_encoder: bool,
    batched_videos_and_reference: tuple[
        list[str], list[tuple[np.ndarray, dict[str, Any]]], list[GenerationResult]
    ],
    request: pytest.FixtureRequest,
):
    """Tests that batched videos of different lengths stay bit-exact.

    Covers the per-item split: when the scheduler hands both videos to one
    encoder call and their combined patches exceed the budget, the group is
    split back into per-item forwards and each item is chunked on its own.
    """
    _, _, reference = batched_videos_and_reference
    total_prompt_tokens = sum(len(r.prompt_token_ids) for r in reference)
    assert total_prompt_tokens <= MAX_NUM_BATCHED_TOKENS, (
        f"Combined batched prompt tokens ({total_prompt_tokens}) exceed "
        f"MAX_NUM_BATCHED_TOKENS ({MAX_NUM_BATCHED_TOKENS}); both videos must "
        f"fit in one step to exercise the batched encoder per-item split."
    )
    # Half the shorter video: both videos exceed it, so both get chunked.
    budget = _batched_chunk_budget(reference)
    assert budget > 0

    suite: EncoderModeSuiteResult = request.getfixturevalue(
        "compiled_encoder_suite" if cudagraph_mm_encoder else "eager_encoder_suite"
    )
    chunked = suite.batched_chunked.unwrap(
        f"Batched chunked ({CHUNK_ENV}={budget}, "
        f"cudagraph_mm_encoder={cudagraph_mm_encoder})"
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
