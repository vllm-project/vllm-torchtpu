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
"""scripts/vllm/integration/qwen35_pd_checks.py: the Qwen3.5 P/D pair checks
that an accuracy score cannot see."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "vllm"
    / "integration"
    / "qwen35_pd_checks.py"
)
_spec = importlib.util.spec_from_file_location("qwen35_pd_checks", _SCRIPT)
assert _spec is not None and _spec.loader is not None
checks = importlib.util.module_from_spec(_spec)
sys.modules["qwen35_pd_checks"] = checks
_spec.loader.exec_module(checks)

# The lines the checks read, as the engines log them.
_PAGE_LINE = (
    "INFO TPU block_size derivation path: backend=PALLAS -> "
    "backend_supported_kernel_block_sizes={page} -> input_block_size=1280 -> ..."
)
_MISMATCH_LINE = (
    "ERROR Stage 3 submit failed: Remote Controller Server execution failed: "
    "Layout fingerprint mismatch between source and destination"
)


def _log(*pages: str, mismatches: int = 0) -> str:
    lines = [_PAGE_LINE.format(page=p) for p in pages]
    lines += [_MISMATCH_LINE] * mismatches
    return "\n".join(["INFO starting"] + lines + ["INFO Application startup complete."])


def test_a_leg_at_the_measured_page_with_no_mismatch_passes():
    status = checks.leg_status(_log("[128]", "[128]", "[128]"))
    assert status == {
        "kernel_pages": ["[128]"],
        "kernel_page_lines": 3,
        "fingerprint_mismatches": 0,
    }
    assert checks.leg_failures(status, 128) == []


@pytest.mark.parametrize(
    "log,reason",
    [
        # A leg that drifted to another page still answers GSM8K well.
        (_log("[128]", "[256]"), "kernel pages ['[128]', '[256]']"),
        (_log("[256]"), "kernel pages ['[256]']"),
        (_log(), "no worker logged its kernel page"),
        (_log("[128]", mismatches=2), "2 layout-fingerprint mismatch"),
    ],
)
def test_a_drifted_page_a_missing_page_or_a_mismatch_fails(log, reason):
    failures = checks.leg_failures(checks.leg_status(log), 128)
    assert any(reason in f for f in failures), failures


def _metrics(**series: float) -> str:
    lines = ["# HELP vllm:spec_decode_num_drafts_total drafts", "# TYPE x counter"]
    for name, value in series.items():
        metric, _, label = name.partition("__")
        labels = f'{{model_name="m",source="{label}"}}' if label else '{model_name="m"}'
        lines.append(f"vllm:{metric}{labels} {value}")
    return "\n".join(lines)


def test_mtp_acceptance_sums_every_decode_server():
    # Two API servers: 100 + 300 drafts of 3 tokens, 250 + 650 accepted.
    bodies = [
        _metrics(
            spec_decode_num_drafts_total=100,
            spec_decode_num_draft_tokens_total=300,
            spec_decode_num_accepted_tokens_total=250,
        ),
        _metrics(
            spec_decode_num_drafts_total=300,
            spec_decode_num_draft_tokens_total=900,
            spec_decode_num_accepted_tokens_total=650,
        ),
    ]
    rate, length = checks.mtp_acceptance(bodies)
    assert rate == pytest.approx(900 / 1200)
    assert length == pytest.approx(1 + 900 / 400)
    assert checks.mtp_acceptance([_metrics()]) is None


def test_decode_compute_share_reads_the_prompt_token_sources():
    body = _metrics(
        prompt_tokens_by_source_total__external_kv_transfer=9000,
        prompt_tokens_by_source_total__local_compute=100,
        prompt_tokens_by_source_total__local_cache_hit=900,
    )
    transferred, pct = checks.decode_compute_share([body])
    assert transferred == 9000
    assert pct == pytest.approx(1.0)
    assert checks.decode_compute_share([_metrics()]) is None


def test_metric_sum_matches_the_whole_name_and_the_label():
    body = _metrics(
        prefix_cache_hits_total=5,
        prefix_cache_hits_total_extra=7,
        prompt_tokens_by_source_total__local_compute=3,
        prompt_tokens_by_source_total__local_cache_hit=4,
    )
    assert checks.metric_sum(body, "vllm:prefix_cache_hits_total") == 5
    assert (
        checks.metric_sum(
            body, "vllm:prompt_tokens_by_source_total", 'source="local_cache_hit"'
        )
        == 4
    )
    assert checks.metric_sum(body, "vllm:num_preemptions_total") is None


def test_cli_leg_check_exit_codes(tmp_path, capsys):
    log = tmp_path / "serve.log"
    status = tmp_path / "leg_status.json"
    log.write_text(_log("[128]"))
    assert checks.main(["leg-status", "--log", str(log), "--out", str(status)]) == 0
    assert json.loads(status.read_text())["kernel_pages"] == ["[128]"]
    assert (
        checks.main(
            ["leg-check", "--status", str(status), "--name", "decode", "--page", "128"]
        )
        == 0
    )
    assert (
        checks.main(
            ["leg-check", "--status", str(status), "--name", "decode", "--page", "256"]
        )
        == 1
    )
    assert "LEG_CHECK_FAIL decode" in capsys.readouterr().out
    missing = str(tmp_path / "absent.json")
    assert (
        checks.main(
            ["leg-check", "--status", missing, "--name", "decode", "--page", "128"]
        )
        == 2
    )
