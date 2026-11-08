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

"""Unit tests for Kimi-K3 megakernel CLI tools (parser, coverage, harness, k3)."""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from vllm_torchtpu.models.kimi_k3.megakernel.tools import (
    __main__ as cli_main,
)
from vllm_torchtpu.models.kimi_k3.megakernel.tools import (
    bench,
    check,
    coverage,
    harness,
    k3,
)

_STEPS_ARG = 12
_PROMPT_TOKENS_ARG = 256
_GENERATE_ARG = 16
_SPEC_TOKENS_ARG = 3
_COMPUTED_TOKENS_ARG = 4
_SCHEDULED_TOKENS_ARG = 2


def test_cli_parser_bench_and_check_subcommands() -> None:
    """Verify CLI parser builds bench, check, profile, and preshard subcommands."""
    top = cli_main.parser()
    bench_args = top.parse_args(
        [
            "--jobid",
            "100",
            "--presharded",
            "/mnt/presharded",
            "--set",
            "attention_dp=2",
            "bench",
            "decode",
            "--batches",
            "1,8",
            "--steps",
            str(_STEPS_ARG),
        ]
    )
    assert bench_args.command == "bench"
    assert bench_args.role == "decode"
    assert bench_args.batches == "1,8"
    assert bench_args.steps == _STEPS_ARG
    assert bench_args.set == ["attention_dp=2"]

    check_args = top.parse_args(
        [
            "--presharded",
            "/mnt/presharded",
            "check",
            "greedy",
            "--model",
            "/mnt/ckpt",
            "--prompt-tokens",
            str(_PROMPT_TOKENS_ARG),
            "--generate",
            str(_GENERATE_ARG),
            "--output",
            "greedy.json",
        ]
    )
    assert check_args.command == "check"
    assert check_args.check == "greedy"
    assert check_args.prompt_tokens == _PROMPT_TOKENS_ARG
    assert check_args.generate == _GENERATE_ARG


def test_coverage_line_tracer_and_report(
    tmp_path: Path,
) -> None:
    """Verify LineCoverage traces lines under ROOTS and report formats output."""
    assert all(Path(root).is_dir() for root in coverage.ROOTS)
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()
    sample_py = pkg_dir / "sample_kernel.py"
    sample_py.write_text(
        "def executed_fn(x: int) -> int:\n"
        "    if x > 0:\n"
        "        return x + 1\n"
        "    return 0\n\n"
        "def unentered_fn(y: int) -> int:\n"
        "    return y * 2\n",
        encoding="utf-8",
    )
    cov_json = tmp_path / "cov.json"
    spec = importlib.util.spec_from_file_location("sample_kernel", sample_py)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    with mock.patch.object(coverage, "ROOTS", (str(pkg_dir),)):
        tracer = coverage.LineCoverage(str(cov_json))
        tracer.start()
        try:
            res = mod.executed_fn(5)
        finally:
            tracer.stop()
        expected_return = 6
        assert res == expected_return
        assert cov_json.exists()

        buf = io.StringIO()
        with mock.patch("sys.stdout", buf):
            coverage.report(str(cov_json), detail=["sample_kernel.py"])
    out = buf.getvalue()
    assert "sample_kernel.py" in out
    assert "never entered: unentered_fn" in out


def test_harness_records_and_k3_helpers(
    tmp_path: Path,
) -> None:
    """Verify harness.Records writes JSONL on rank 0 and k3 helpers configure cache."""
    rec_path = tmp_path / "records.jsonl"
    cfg_stub = SimpleNamespace(key=lambda: "cfg123")
    with mock.patch.object(harness, "rank", return_value=0):
        records = harness.Records(str(rec_path), "bench decode", cfg_stub)
        records.add(step=1, ok=True)
    lines = rec_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["command"] == "bench decode"
    assert parsed["config"] == "cfg123"
    assert parsed["step"] == 1
    assert parsed["ok"] is True

    k3.configure_cache("bfloat16")
    assert k3.PAGE > 0
    assert k3.logical() >= k3.PAGE
    cfg = k3.engine_config(
        "decode",
        max_num_seqs=8,
        max_model_len=4096,
        max_num_batched_tokens=512,
        num_speculative_tokens=_SPEC_TOKENS_ARG,
    )
    assert cfg.role == "decode"
    assert cfg.num_speculative_tokens == _SPEC_TOKENS_ARG
    req = k3.request(
        "req-0",
        [10, 20],
        np.asarray([1, 2], dtype=np.int32),
        computed=_COMPUTED_TOKENS_ARG,
    )
    assert req.req_id == "req-0"
    assert req.computed == _COMPUTED_TOKENS_ARG
    assert req.scheduled == _SCHEDULED_TOKENS_ARG
    assert callable(bench.run)
    assert callable(check.run)


def test_cli_main_worker_dispatch() -> None:
    """Verify cli_main.main dispatches worker bench and check commands."""
    with (
        mock.patch.object(harness, "is_worker", return_value=True),
        mock.patch.object(harness, "init") as init_mock,
        mock.patch.object(harness, "rank", return_value=0),
        mock.patch.object(check, "run") as check_run_mock,
    ):
        rc = cli_main.main(
            [
                "--presharded",
                "/mnt/presharded",
                "check",
                "greedy",
                "--model",
                "/mnt/m",
                "--output",
                "out.json",
            ]
        )
    assert rc == 0
    init_mock.assert_called_once()
    check_run_mock.assert_called_once()
