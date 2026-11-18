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
"""TPU host-only tests for the vLLM-Omni benchmark CLI subcommands."""

import argparse
import json
import time
from pathlib import Path

import pytest
from PIL import Image
from vllm_omni.benchmarks import serve
from vllm_omni.benchmarks.duplex.omni_duplex_eval_judge import DuplexJudge
from vllm_omni.benchmarks.duplex.omni_duplex_eval_runner import GenerateSampleResult
from vllm_omni.benchmarks.patch import patch
from vllm_omni.entrypoints.cli.benchmark import omni_duplex_eval as duplex_cli
from vllm_omni.entrypoints.cli.benchmark.cli_args import preprocess_serve_args
from vllm_omni.entrypoints.cli.benchmark.main import OmniBenchmarkSubcommand
from vllm_omni.entrypoints.cli.benchmark.omni_duplex_eval import (
    OmniDuplexEvalSubcommand,
)
from vllm_omni.entrypoints.cli.benchmark.serve import OmniBenchmarkServingSubcommand
from vllm_omni.platforms import current_omni_platform

from tests_omni.omni_test_utils import PR_TIMEOUT_S
from vllm_torchtpu.omni.platform import OmniTpuPlatform

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

_EXPECTED_NUM_PROMPTS = 4
_EXPECTED_NUM_WARMUPS = 2
_EXPECTED_MEAN_E2EL_MS = 100.0
_EXPECTED_TTFT_MEAN_MS = 50.0


def test_benchmark_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Benchmark CLI (vllm bench serve --omni, omni_duplex_eval)

    - TPU path: Runs ``OmniBenchmarkSubcommand``, ``OmniBenchmarkServingSubcommand``,
      and ``OmniDuplexEvalSubcommand`` with ``OmniTpuPlatform`` active.
    - Setup check: Verifies ``OmniTpuPlatform`` is active and both ``"serve"`` and
      ``"omni-duplex-eval"`` subcommands register on ``vllm bench``.
    - Outcome check: Runs ``serve`` on a synthetic custom-image JSONL dataset and
      ``omni-duplex-eval`` ``generate`` -> ``evaluate`` -> ``summarize`` with stubbed
      network/judge calls, checking counts, backend resolution, and metrics.
    """
    assert isinstance(current_omni_platform, OmniTpuPlatform), (
        f"expected OmniTpuPlatform, got {type(current_omni_platform).__name__}"
    )

    root_parser = argparse.ArgumentParser()
    subparsers = root_parser.add_subparsers(dest="command")
    bench_parser = OmniBenchmarkSubcommand().subparser_init(subparsers)

    parsed_serve = bench_parser.parse_args(
        [
            "serve",
            "--omni",
            "--model",
            "test-model",
            "--endpoint",
            "/v1/images/generations",
        ]
    )
    assert parsed_serve.bench_type == "serve", (
        f"expected bench_type 'serve', got {parsed_serve.bench_type!r}"
    )
    assert parsed_serve.dispatch_function is OmniBenchmarkServingSubcommand.cmd, (
        "serve subcommand did not bind OmniBenchmarkServingSubcommand.cmd"
    )

    parsed_duplex = bench_parser.parse_args(
        ["omni-duplex-eval", "summarize", "--score-root", str(tmp_path)]
    )
    assert parsed_duplex.bench_type == "omni-duplex-eval", (
        f"expected bench_type 'omni-duplex-eval', got {parsed_duplex.bench_type!r}"
    )
    assert parsed_duplex.dispatch_function is OmniDuplexEvalSubcommand.cmd, (
        "omni-duplex-eval subcommand did not bind OmniDuplexEvalSubcommand.cmd"
    )

    img_path = tmp_path / "dummy.png"
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(img_path)
    dataset_path = tmp_path / "prompts.jsonl"
    dataset_path.write_text(
        "\n".join(
            json.dumps({"prompt": f"prompt {i}", "image_files": [str(img_path)]})
            for i in range(6)
        )
        + "\n",
        encoding="utf-8",
    )
    serve_parser = argparse.ArgumentParser()
    OmniBenchmarkServingSubcommand.add_cli_args(serve_parser)
    serve_args = serve_parser.parse_args(
        [
            "--model",
            "test-model",
            "--endpoint",
            "/v1/images/generations",
            "--dataset-name",
            "custom_image",
            "--dataset-path",
            str(dataset_path),
            "--skip-tokenizer-init",
            "--disable-shuffle",
            "--disable-tqdm",
            "--num-prompts",
            str(_EXPECTED_NUM_PROMPTS),
            "--num-warmups",
            str(_EXPECTED_NUM_WARMUPS),
            "--max-concurrency",
            "1",
            "--ready-check-timeout-sec",
            "0",
            "--percentile-metrics",
            "e2el",
            "--save-result",
            "--result-dir",
            str(tmp_path),
            "--result-filename",
            "serve_result.json",
        ]
    )
    preprocess_serve_args(serve_args)

    recorded_requests = []

    async def fake_request(request_func_input, session, pbar=None):
        recorded_requests.append(request_func_input)
        if pbar is not None:
            pbar.update(1)
        is_warmup = len(recorded_requests) <= _EXPECTED_NUM_WARMUPS
        return patch.MixRequestFuncOutput(
            success=True,
            prompt_len=1,
            output_tokens=1,
            image_count=1,
            start_time=time.perf_counter(),
            latency=99.0 if is_warmup else 0.1,
        )

    monkeypatch.setitem(
        patch.ASYNC_REQUEST_FUNCS, "/v1/images/generations", fake_request
    )
    serve_result = serve.main(serve_args)

    assert serve_args.backend == "/v1/images/generations", (
        f"expected '/v1/images/generations' backend, got {serve_args.backend!r}"
    )
    expected_total = _EXPECTED_NUM_WARMUPS + _EXPECTED_NUM_PROMPTS
    assert len(recorded_requests) == expected_total, (
        f"expected {expected_total} requests (warmups + measured), "
        f"got {len(recorded_requests)}"
    )
    assert serve_result["completed"] == _EXPECTED_NUM_PROMPTS, (
        f"expected {_EXPECTED_NUM_PROMPTS} completed requests, "
        f"got {serve_result['completed']}"
    )
    saved_serve = json.loads(
        (tmp_path / "serve_result.json").read_text(encoding="utf-8")
    )
    assert saved_serve["mean_e2el_ms"] == pytest.approx(_EXPECTED_MEAN_E2EL_MS), (
        f"mean_e2el_ms {saved_serve['mean_e2el_ms']} != {_EXPECTED_MEAN_E2EL_MS} "
        "(warmups should be excluded)"
    )

    manifest_path = tmp_path / "duplex_manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "id": "s0",
                    "split": "PR_correction",
                    "question_text": "Correct the answer.",
                    "answer1": "Expected answer.",
                }
            ]
        ),
        encoding="utf-8",
    )
    response_root = tmp_path / "duplex_responses"
    score_root = tmp_path / "duplex_scores"

    async def fake_generate_sample(sample, *, output_root, **kwargs):
        out_path = Path(output_root) / sample.split / f"{sample.id}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"clock": "media"}
        timed = [{"sentence": "Corrected.", "start": 0.0, "end": 0.5}]
        out_path.with_name(f"{sample.id}.meta.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )
        out_path.write_text(json.dumps(timed), encoding="utf-8")
        return GenerateSampleResult(
            output=out_path,
            request_metrics=[
                {"stage0_tokens": {"output_token_count": 2, "itls_ms": [10.0, 20.0]}}
            ],
            session_metrics={
                "stream_ttft_ms": _EXPECTED_TTFT_MEAN_MS,
                "stream_ttfp_ms": 80.0,
                "stream_rtf": 0.5,
            },
            timed_sentences=timed,
            metadata=meta,
            output_tokens=2,
        )

    monkeypatch.setattr(duplex_cli, "generate_sample", fake_generate_sample)
    monkeypatch.setattr(
        DuplexJudge,
        "chat",
        lambda self, content, **kwargs: '{"success_score": 1}',
    )

    duplex_parser = argparse.ArgumentParser()
    OmniDuplexEvalSubcommand.add_cli_args(duplex_parser)

    gen_args = duplex_parser.parse_args(
        [
            "generate",
            "--dataset",
            str(manifest_path),
            "--model",
            "test-model",
            "--response-root",
            str(response_root),
        ]
    )
    OmniDuplexEvalSubcommand.cmd(gen_args)

    metrics_path = response_root / "duplex_metrics.json"
    assert metrics_path.is_file(), f"missing duplex_metrics.json at {metrics_path}"
    duplex_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    stream_ttft = duplex_metrics["duplex_stream_ttft_ms"]["mean"]
    assert stream_ttft == pytest.approx(_EXPECTED_TTFT_MEAN_MS), (
        f"duplex stream ttft_ms mean {stream_ttft} != {_EXPECTED_TTFT_MEAN_MS}"
    )

    eval_args = duplex_parser.parse_args(
        [
            "evaluate",
            "--dataset",
            str(manifest_path),
            "--response-root",
            str(response_root),
            "--score-root",
            str(score_root),
            "--judge-model",
            "judge-model",
        ]
    )
    OmniDuplexEvalSubcommand.cmd(eval_args)

    sum_args = duplex_parser.parse_args(["summarize", "--score-root", str(score_root)])
    OmniDuplexEvalSubcommand.cmd(sum_args)

    summary = duplex_cli.summarize_scores(score_root)
    assert summary["samples"] == 1, (
        f"expected 1 summarized duplex sample, got {summary['samples']}"
    )
    assert summary["pr"]["mean_all_success"] == pytest.approx(1.0), (
        f"expected mean_all_success == 1.0, got {summary['pr']['mean_all_success']}"
    )
