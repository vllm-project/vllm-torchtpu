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


def _serve(monkeypatch, bodies: dict[str, str], completions: list[dict] | None = None):
    """Route the checks' HTTP calls to canned /metrics bodies and completions."""

    def get(url: str, timeout: float = 30) -> str:
        if url not in bodies:
            raise OSError(f"no route to {url}")
        return bodies[url]

    sent = list(completions or [])

    def post(url: str, body: dict, timeout: float = 1800) -> dict:
        if not sent:
            raise OSError("proxy refused")
        return sent.pop(0)

    monkeypatch.setattr(checks, "_get", get)
    monkeypatch.setattr(checks, "_post", post)


_DECODE = ["http://d:7200/metrics", "http://d:7201/metrics"]


def _spec(drafts: float, tokens: float, accepted: float) -> str:
    return _metrics(
        spec_decode_num_drafts_total=drafts,
        spec_decode_num_draft_tokens_total=tokens,
        spec_decode_num_accepted_tokens_total=accepted,
    )


@pytest.mark.parametrize(
    "accepted,rc", [(2250, 0), (1980, 0), (1950, 1), (1500, 1)], ids=str
)
def test_cmd_mtp_holds_the_rate_floor(monkeypatch, accepted, rc):
    # 2 servers x 1000 drafts x 3 tokens; floor 0.66 = 3960 of 6000 accepted.
    _serve(monkeypatch, {u: _spec(1000, 3000, accepted) for u in _DECODE})
    assert checks.main(["mtp", "--metrics", *_DECODE, "--min-rate", "0.66"]) == rc


def test_cmd_mtp_fails_without_drafts_and_cannot_measure_a_dead_server(monkeypatch):
    _serve(monkeypatch, {u: _metrics() for u in _DECODE})
    assert checks.main(["mtp", "--metrics", *_DECODE, "--min-rate", "0.66"]) == 1
    _serve(monkeypatch, {_DECODE[0]: _spec(1000, 3000, 2250)})
    assert checks.main(["mtp", "--metrics", *_DECODE, "--min-rate", "0.66"]) == 2


def _sources(transfer: float, compute: float, hit: float) -> str:
    return _metrics(
        prompt_tokens_by_source_total__external_kv_transfer=transfer,
        prompt_tokens_by_source_total__local_compute=compute,
        prompt_tokens_by_source_total__local_cache_hit=hit,
    )


@pytest.mark.parametrize(
    "transfer,compute,rc",
    [(9000, 100, 0), (9000, 500, 0), (9000, 600, 1), (0, 100, 1)],
    ids=str,
)
def test_cmd_transfer_needs_a_transfer_and_bounds_decode_compute(
    monkeypatch, transfer, compute, rc
):
    # Per server: 9000 transferred + 500 reused; 5% of 10000 is 500.
    _serve(monkeypatch, {u: _sources(transfer, compute, 500) for u in _DECODE})
    args = ["transfer", "--decode-metrics", *_DECODE, "--max-compute-pct", "5"]
    assert checks.main(args) == rc


def test_cmd_transfer_cannot_measure_without_the_counters(monkeypatch):
    _serve(monkeypatch, {u: _metrics() for u in _DECODE})
    args = ["transfer", "--decode-metrics", *_DECODE, "--max-compute-pct", "5"]
    assert checks.main(args) == 2


_PREFILL = "http://p:8000/metrics"


def _completion(cached: int) -> dict:
    return {
        "choices": [{"text": " 4242"}],
        "usage": {
            "prompt_tokens": 8000,
            "prompt_tokens_details": {"cached_tokens": cached},
        },
    }


@pytest.mark.parametrize(
    "decode_hit,prefill_hit,last_cached,rc",
    [
        (7680, 6912, 7680, 0),
        (0, 6912, 7680, 1),  # decode never reused its own prefix cache
        (7680, 0, 7680, 1),  # prefill never hit its cache
        (7680, 6912, 0, 1),  # the repeat reported no cached tokens
    ],
    ids=["good", "no_decode_hit", "no_prefill_hit", "no_cached_tokens"],
)
def test_cmd_prefix_hit_needs_both_legs_to_reuse_the_prefix(
    monkeypatch, decode_hit, prefill_hit, last_cached, rc
):
    bodies = {u: _sources(9000, 10, decode_hit / 2) for u in _DECODE}
    bodies[_PREFILL] = _metrics(prefix_cache_hits_total=prefill_hit)
    _serve(
        monkeypatch, bodies, [_completion(0), _completion(0), _completion(last_cached)]
    )
    args = ["prefix-hit", "--proxy", "http://x:8100", "--model", "m", "--repeats", "3"]
    args += ["--prefill-metrics", _PREFILL, "--decode-metrics", *_DECODE]
    assert checks.main(args) == rc


def test_cmd_prefix_hit_fails_on_a_refused_request(monkeypatch):
    _serve(monkeypatch, {}, [_completion(0)])
    args = ["prefix-hit", "--proxy", "http://x:8100", "--model", "m", "--repeats", "2"]
    args += ["--prefill-metrics", _PREFILL, "--decode-metrics", *_DECODE]
    assert checks.main(args) == 1


_GSM8K = "gsm8k"
_STRICT = "exact_match,strict-match"


def _eval_dir(
    tmp_path, *, result=True, effective=1319, row="PASS", samples=True, strict=True
):
    out = tmp_path / "eval"
    (out / "model").mkdir(parents=True)
    if result:
        metric = _STRICT if strict else "exact_match,flexible-extract"
        stderr = metric.replace("exact_match,", "exact_match_stderr,")
        body: dict = {"results": {_GSM8K: {metric: 0.97, stderr: 0.004}}}
        if samples:
            body["n-samples"] = {_GSM8K: {"original": 1319, "effective": effective}}
        (out / "model" / "results_2026.json").write_text(json.dumps(body))
    if row:
        (out / f"eval_check_{_GSM8K}.md").write_text(
            f"| `{_GSM8K}` | {_STRICT} | 0.9700 | 0.9700 | 0.0057 | 0.0200 "
            f"| +0.0000 | {row} |\n"
        )
    return out


_MEASURED = {_GSM8K: {_STRICT: 0.97}}
_PENDING = {"_pending": "not measured yet"}


@pytest.mark.parametrize(
    "kwargs,log,baseline,reason",
    [
        ({}, "", _MEASURED, None),
        ({"row": "FAIL"}, "", _MEASURED, None),  # a regression is the gate's verdict
        ({"row": None}, "", _PENDING, None),  # pending: no gated row to print
        ({"result": False}, "", _MEASURED, "lm-eval wrote no result"),
        ({"effective": 100}, "", _MEASURED, "ran 100 of its 1319 samples"),
        # Without n-samples a short split cannot be told from the whole one.
        ({"samples": False}, "", _MEASURED, "does not say how many samples ran"),
        # A result scored on another filter is not the gate's metric.
        ({"strict": False}, "", _PENDING, "has no exact_match,strict-match"),
        (
            {},
            "WARNING API request failed, retrying\n",
            _MEASURED,
            "1 lm-eval request(s) failed",
        ),
        (
            {"row": None},
            "",
            _MEASURED,
            "the gate printed no exact_match,strict-match row",
        ),
    ],
    ids=[
        "ran",
        "ran_regressed",
        "pending",
        "no_result",
        "partial",
        "no_sample_count",
        "no_strict_match",
        "retried",
        "no_row",
    ],
)
def test_eval_failures_needs_a_whole_split_with_no_failed_request(
    tmp_path, kwargs, log, baseline, reason
):
    out = _eval_dir(tmp_path, **kwargs)
    failures = checks.eval_failures(str(out), _GSM8K, _STRICT, log, baseline)
    if reason is None:
        assert failures == []
    else:
        assert any(reason in f for f in failures), failures


def _result_dir(tmp_path, metric: str, row_metric: str | None):
    out = tmp_path / "eval"
    (out / "model").mkdir(parents=True)
    stderr = metric.replace("exact_match,", "exact_match_stderr,")
    body = {
        "results": {"task": {metric: 0.8, stderr: 0.01}},
        "n-samples": {"task": {"original": 100, "effective": 100}},
    }
    (out / "model" / "results_2026.json").write_text(json.dumps(body))
    if row_metric:
        (out / "eval_check_task.md").write_text(
            f"| `task` | {row_metric} | 0.8000 | 0.8000 | 0.0141 | 0.0200 "
            "| +0.0000 | PASS |\n"
        )
    return str(out)


def test_eval_ran_reads_the_metric_from_the_baseline(tmp_path):
    """With no --metric, a task is checked on its baseline's metric (the
    result's own while the baseline is pending), not on gsm8k's."""
    custom = "exact_match,custom-extract"
    measured = {"task": {custom: 0.8, "exact_match_stderr,custom-extract": 0.01}}
    assert (
        checks.eval_failures(
            _result_dir(tmp_path / "a", custom, custom), "task", None, "", measured
        )
        == []
    )
    assert (
        checks.eval_failures(
            _result_dir(tmp_path / "b", custom, None), "task", None, "", _PENDING
        )
        == []
    )
    # A measured strict-match baseline and a result scored on another filter.
    strict = {"task": {_STRICT: 0.8, "exact_match_stderr,strict-match": 0.01}}
    failures = checks.eval_failures(
        _result_dir(tmp_path / "c", "exact_match,flexible-extract", None),
        "task",
        None,
        "",
        strict,
    )
    assert any("has no exact_match,strict-match" in f for f in failures), failures
