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
"""Unit tests for scripts/vllm/benchmark/inferencex/sweep_summary.py.

The parsers are pure functions over JSON files, so they are pinned down here
without a TPU. The AgentX fixture mirrors the layout InferenceX's
process_agentic_result.py writes (nested request_metrics / server_metrics).
"""

import importlib.util
import json
import math
import os
import pathlib
import time

import pytest

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "vllm"
    / "benchmark"
    / "inferencex"
    / "sweep_summary.py"
)
_spec = importlib.util.spec_from_file_location("ix_sweep_summary", _SCRIPT)
summary = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(summary)


def _write_results(root, model_dir, name, results):
    d = root / model_dir
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text(json.dumps({"results": results}))
    return p


def test_lm_eval_score_mmlu_pro_uses_custom_extract(tmp_path):
    _write_results(
        tmp_path,
        "moonshotai__Kimi-K3",
        "results_a.json",
        {
            "mmlu_pro": {
                "exact_match,custom-extract": 0.835,
                "exact_match_stderr,custom-extract": 0.0098,
            },
            "mmlu_pro_law": {"exact_match,custom-extract": 0.71},
        },
    )
    assert summary.lm_eval_score(str(tmp_path), "mmlu_pro") == (
        "exact_match,custom-extract",
        0.835,
        0.0098,
    )


def test_lm_eval_score_gsm8k_prefers_flexible_extract(tmp_path):
    _write_results(
        tmp_path,
        "m",
        "results_a.json",
        {
            "gsm8k": {
                "exact_match,strict-match": 0.2,
                "exact_match_stderr,strict-match": 0.03,
                "exact_match,flexible-extract": 1.0,
                "exact_match_stderr,flexible-extract": 0.0,
            }
        },
    )
    assert summary.lm_eval_score(str(tmp_path), "gsm8k") == (
        "exact_match,flexible-extract",
        1.0,
        0.0,
    )


def test_lm_eval_score_picks_newest_file(tmp_path):
    old = _write_results(
        tmp_path,
        "m",
        "results_old.json",
        {"gsm8k": {"exact_match,flexible-extract": 0.1}},
    )
    new = _write_results(
        tmp_path,
        "m",
        "results_new.json",
        {"gsm8k": {"exact_match,flexible-extract": 0.9}},
    )
    now = time.time()
    os.utime(old, (now - 100, now - 100))
    os.utime(new, (now, now))
    _, score, stderr = summary.lm_eval_score(str(tmp_path), "gsm8k")
    assert score == 0.9 and math.isnan(stderr)


def test_lm_eval_score_errors_are_explicit(tmp_path):
    with pytest.raises(FileNotFoundError):
        summary.lm_eval_score(str(tmp_path), "gsm8k")
    _write_results(
        tmp_path, "m", "results_a.json", {"other": {"exact_match,strict-match": 1.0}}
    )
    with pytest.raises(KeyError):
        summary.lm_eval_score(str(tmp_path), "gsm8k")


def test_lm_eval_score_cli_never_fails_the_sweep(tmp_path, capsys):
    assert (
        summary.main(
            ["lm-eval-score", "--output-dir", str(tmp_path), "--task", "gsm8k"]
        )
        == 0
    )
    out, err = capsys.readouterr()
    assert out.strip() == "none nan nan"
    assert "ERROR" in err and "gsm8k" in err


@pytest.mark.parametrize(
    "task,conc,expected",
    [
        ("gsm8k", 1, (128, 128)),
        ("gsm8k", 64, (128, 128)),
        ("gsm8k", 128, (256, 256)),
        ("gsm8k", 256, (512, 512)),
        ("mmlu_pro", 1, (10, 140)),
        ("mmlu_pro", 64, (10, 140)),
        ("mmlu_pro", 128, (19, 266)),
        ("mmlu_pro", 256, (37, 518)),
    ],
)
def test_question_limit_reaches_twice_the_concurrency(task, conc, expected):
    limit, total = summary.question_limit(task, conc, 128, 10)
    assert (limit, total) == expected
    assert total >= 2 * conc


def test_question_limit_rejects_unknown_task():
    with pytest.raises(ValueError):
        summary.question_limit("arc", 1, 128, 10)


def test_gsm8k_row(tmp_path):
    _write_results(
        tmp_path,
        "m",
        "results_a.json",
        {
            "gsm8k": {
                "exact_match,strict-match": 0.97,
                "exact_match,flexible-extract": 1.0,
            }
        },
    )
    assert summary.gsm8k_row(str(tmp_path), 8) == "| 8 | 0.97 | 1.0 |"
    assert summary.gsm8k_row(str(tmp_path / "missing"), 8).startswith(
        "| 8 | (no results file)"
    )


def _aggregate(per_gpu_out, requests_ok, chips=32, hw="tpu7x-32"):
    # Shape of InferenceX's aggregate JSON (process_agentic_result.py). Its own
    # per_gpu is whatever the parallelism it was told implies; the table does not
    # read it, so the two throughput fields here only have to be a plausible pair.
    return {
        "conc": 1,
        "hw": hw,
        "num_requests_successful": requests_ok,
        "request_metrics": {
            "throughput": {
                "output": {"tokens_per_second": per_gpu_out * chips},
                "per_gpu": {"output_tput_tps": per_gpu_out},
            },
            "latency": {
                "ttft": {"p50": 1.5, "p90": 7.2},
                "tpot": {"p50": 0.0255},
                "intvty": {"p90": 37.0},
                "e2e_norm_intvty": {"p90": 0.9},
                "e2el": {"p50": 12.0},
            },
            "cache": {"theoretical_cache_hit_rate": 0.928},
        },
        "server_metrics": {"kv_cache": {"gpu_usage_pct": 41.0}},
    }


def test_agentx_table_rows_and_markers(tmp_path):
    base = "agentx_k3"
    (tmp_path / f"{base}_c1.json").write_text(json.dumps(_aggregate(1.54, 48)))
    (tmp_path / "conc_4").mkdir()
    (tmp_path / "conc_4" / "HARD_TIMEOUT").write_text("")
    table = summary.agentx_table(str(tmp_path), base, [1, 4, 8], chips=32)
    rows = table.splitlines()
    assert rows[0].startswith("| conc | out tok/$ | out tok/s total |")
    # 49.28 tok/s for an hour on 32 chips at 1.2 USD/chip-hour.
    assert rows[2].startswith("| 1 | 4620 | 49.28 | 48 | 1.5 | 7.2 | 0.0255 | 37 |")
    assert "hard timeout" in rows[3] and rows[3].startswith("| 4 |")
    assert rows[4].startswith("| 8 | (no aggregate:") and "hard timeout" not in rows[4]


def test_agentx_table_marks_a_timeout_that_still_wrote_an_aggregate(tmp_path):
    # The failure the watchdog exists for: aiperf writes the aggregate and then
    # hangs in shutdown. The numbers are real but the point ran to the cap, so
    # the row has to say so instead of looking clean.
    base = "agentx_k3"
    (tmp_path / f"{base}_c2.json").write_text(json.dumps(_aggregate(1.1, 12)))
    (tmp_path / "conc_2").mkdir()
    (tmp_path / "conc_2" / "HARD_TIMEOUT").write_text("")
    row = summary.agentx_table(str(tmp_path), base, [2], chips=32).splitlines()[2]
    assert row.startswith("| 2 (HARD_TIMEOUT) | 3300 |")


def test_agentx_table_survives_an_unreadable_aggregate(tmp_path):
    base = "agentx_k3"
    (tmp_path / f"{base}_c1.json").write_text('{"conc": 1, "request_met')
    (tmp_path / f"{base}_c2.json").write_text(json.dumps(_aggregate(2.5, 20)))
    rows = summary.agentx_table(str(tmp_path), base, [1, 2], chips=32).splitlines()
    assert rows[2].startswith("| 1 | (unreadable: JSONDecodeError)")
    assert rows[3].startswith("| 2 | 7500 |")


def test_agentx_table_costs_the_chips_the_caller_states(tmp_path):
    # The column is computed here, from --chips, not from anything InferenceX
    # writes: its own per_gpu divides by the parallelism it was told about.
    base = "agentx_k3"
    (tmp_path / f"{base}_c1.json").write_text(json.dumps(_aggregate(3.0, 9, chips=64)))
    row = summary.agentx_table(str(tmp_path), base, [1], chips=32).splitlines()[2]
    assert row.startswith("| 1 | 18000 | 192 |")  # 192 tok/s on 32 chips, not 64


def test_agentx_table_tok_per_dollar_is_blank_without_a_chip_count(tmp_path):
    base = "agentx_k3"
    (tmp_path / f"{base}_c1.json").write_text(json.dumps(_aggregate(1.0, 4)))
    row = summary.agentx_table(str(tmp_path), base, [1]).splitlines()[2]
    assert row.startswith("| 1 | - | 32 |")


def test_agentx_table_tok_per_dollar_is_blank_for_unpriced_hardware(tmp_path):
    # Blank beats normalizing by a guess: a row whose hw has no rate yet still
    # carries every measured column.
    base = "agentx_k3"
    (tmp_path / f"{base}_c1.json").write_text(
        json.dumps(_aggregate(1.0, 4, hw="gb200-72"))
    )
    row = summary.agentx_table(str(tmp_path), base, [1], chips=32).splitlines()[2]
    assert row.startswith("| 1 | - | 32 |")


def test_accelerator_rate_matches_the_family_and_nothing_else():
    assert summary.accelerator_usd_per_hour("tpu7x-32") == 1.2
    assert summary.accelerator_usd_per_hour("tpu7x") == 1.2
    assert summary.accelerator_usd_per_hour("tpu7x9-8") is None
    assert summary.accelerator_usd_per_hour("gb200-72") is None
    assert summary.accelerator_usd_per_hour(None) is None


def test_dotted_get_ignores_bools_and_missing():
    assert summary.dotted_get({"a": {"b": 2}}, "a.b") == 2
    assert summary.dotted_get({"a": {"b": True}}, "a.b") is None
    assert summary.dotted_get({"a": 1}, "a.b") is None
    assert summary.dotted_get({}, "a") is None
