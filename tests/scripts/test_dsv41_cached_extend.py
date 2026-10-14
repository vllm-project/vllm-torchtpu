# SPDX-License-Identifier: Apache-2.0
"""Validate cached-extend timing, native output capture and startup accounting."""

import copy
import datetime
import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts/dsv41_bench/server_steps"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import server_step_worker as worker  # noqa: E402
from bench_cached_extend import (  # noqa: E402
    assert_no_compilation,
    summarize_extend,
    summarize_startup,
)
from run import source_identity  # noqa: E402
from server_step_worker import SCENARIO, token_counts  # noqa: E402

pytestmark = pytest.mark.cpu_test


def test_compilation_audit_excludes_startup_and_handles_new_year():
    start = datetime.datetime(2026, 12, 31, 23, 59, 58).timestamp() * 1e9
    intervals = [[int(start), int(start + 5e9)]]
    assert_no_compilation("12-31 23:58:00 Compiling FX graph", intervals)
    with pytest.raises(ValueError, match="Compilation in formal"):
        assert_no_compilation("01-01 00:00:01 Compiling FX graph", intervals)
    with pytest.raises(ValueError, match="Cannot timestamp"):
        assert_no_compilation("Compiling FX graph", intervals)


def test_parallel_ranks_are_not_summed():
    log = "\n".join(
        f"(EngineCore_DP{rank} pid=1) init engine (warmup) took {total} s "
        f"(compilation: {compile_time} s)"
        for rank, total, compile_time in [(0, 20, 12), (1, 23, 15)]
    )
    result = summarize_startup(log)
    assert result["complete_ranks"] == 2
    assert result["max_engine_init_seconds"] == 23
    assert result["max_reported_compilation_seconds"] == 15


def test_partial_startup_and_cache_events_are_not_success():
    result = summarize_startup(
        "(Worker_DP0_EP0 pid=1) Saved compiled artifact to cache\n"
        "(Worker_DP0_EP0 pid=1) Loading compiled artifact from cache\n"
        "(Worker_DP1_EP1 pid=2) Starting torch.compile\n"
    )
    assert result["complete_ranks"] == 0
    assert result["max_engine_init_seconds"] is None
    assert result["ranks"][0]["artifact_saves"] == 1
    assert result["ranks"][0]["artifact_load_attempts"] == 1


def fixture():
    ranks = []
    for rank in range(8):
        steps = []
        start = SCENARIO.prefix_tokens - 128
        for index in range(9):
            end = min(start + 2044, SCENARIO.prefix_tokens + SCENARIO.extend_tokens)
            stage = dict(
                stage="encoder",
                phase="prefill",
                starts=[start],
                ends=[end],
                prompts=[SCENARIO.prefix_tokens + SCENARIO.extend_tokens],
                requests=1,
                rows=end - start,
                padded_rows=2048,
                forward_id=index + 1,
                **token_counts([start], [end], SCENARIO.prefix_tokens),
            )
            steps.append(
                dict(
                    start_ns=1000000000 + index * 200000000,
                    observed_ns=1100000000 + index * 200000000 + rank * 1000,
                    observation="native_output_consumption",
                    output_path="async_discard_without_wait",
                    stages=[stage],
                )
            )
            start = end
        ranks.append(
            dict(
                rank=rank,
                hostname="one",
                scenario=SCENARIO.to_dict(),
                steps=steps,
            )
        )
    return ranks


class AccountingTest(unittest.TestCase):
    def test_counter_interval_alignment(self):
        report = summarize_extend(fixture())
        middle = report["categories"]["encoder_middle"]
        self.assertEqual(middle["steps"], 7)
        self.assertEqual(middle["effective_tokens"], 7 * 8 * 2044)
        self.assertAlmostEqual(middle["total_seconds"], 1.4)
        self.assertAlmostEqual(middle["global_tokens_s"], 81760)
        self.assertFalse(report["capacity_qualified"])

    def test_reject_corruption(self):
        for mutation in [
            lambda r: r.pop(),
            lambda r: r[0]["steps"].pop(3),
            lambda r: r[0]["steps"][2]["stages"].append({"stage": "decoder"}),
            lambda r: r[0]["steps"][2]["stages"][0].update(new_tokens=2048),
            lambda r: r[0]["steps"][2]["stages"][0].update(padded_rows=4096),
            lambda r: r[0]["steps"][2].update(observed_ns=0),
            lambda r: r[0]["steps"][2].update(observation="synchronized"),
        ]:
            ranks = copy.deepcopy(fixture())
            mutation(ranks)
            with self.assertRaises(ValueError):
                summarize_extend(ranks)


class CaptureTest(unittest.TestCase):
    def test_decoder_uses_encoder_snapshot_and_removes_padding(self):
        metadata = SimpleNamespace(
            query_start_loc_cpu=(0, 384, 384),
            seq_lens_cpu=(SCENARIO.prefix_tokens, 0),
            prompt_lens_cpu=(SCENARIO.prefix_tokens, 0),
        )
        context = SimpleNamespace(attn_metadata={"attention": metadata})
        recorder = SimpleNamespace(
            forward_id=0,
            decoder_snapshot=None,
            scenario=SCENARIO,
            stages=[],
            pending={"stages": []},
        )
        modules = {
            "torch": SimpleNamespace(
                compiler=SimpleNamespace(is_compiling=lambda: False)
            ),
            "vllm.forward_context": SimpleNamespace(
                get_forward_context=lambda: context
            ),
            "vllm_torchtpu.models.vllm.deepseek_v41.v2.ced": SimpleNamespace(
                plan_decoder_replay=lambda *args: SimpleNamespace(
                    query_start_loc=(0, 128, 128)
                )
            ),
        }
        stage = SimpleNamespace(
            owner=SimpleNamespace(_ced_window=128, _ced_compile_sizes=[128, 256, 2048])
        )
        positions = SimpleNamespace(shape=(256,))

        def original(*args, **kwargs):
            return "output"

        with (
            patch.dict(sys.modules, modules),
            patch.object(worker, "_ACTIVE", recorder),
        ):
            worker._stage_wrapper(original, "encoder")(stage, None, positions)
            context.attn_metadata = {
                "attention": SimpleNamespace(
                    query_start_loc_cpu=None, seq_lens_cpu=None, prompt_lens_cpu=None
                )
            }
            worker._stage_wrapper(original, "decoder")(stage, None, positions)
        encoder, decoder = recorder.stages
        self.assertEqual(encoder["starts"], [SCENARIO.prefix_tokens - 384])
        self.assertEqual(decoder["starts"], [SCENARIO.prefix_tokens - 128])
        self.assertEqual(decoder["ends"], [SCENARIO.prefix_tokens])
        self.assertEqual(decoder["rows"], 128)
        self.assertEqual(decoder["requests"], 1)
        self.assertEqual(decoder["phase"], "prefill")
        self.assertEqual(decoder["forward_id"], encoder["forward_id"])
        self.assertNotIn("synchronized", decoder)
        self.assertIn("dispatch_ns", decoder)


class DeferredObservationTest(unittest.TestCase):
    def test_native_caller_owns_consumption(self):
        class AsyncOutput:
            def __init__(self):
                self.calls = 0
                self.copy_state = None

            def get_output(self):
                self.calls += 1
                return SimpleNamespace(sampled_token_ids=[[]], req_ids=["r"])

        output = AsyncOutput()
        runner = SimpleNamespace(
            execute_model=lambda *a: output, sample_tokens=lambda *a: None
        )
        recorder = worker.Recorder(runner, SCENARIO, 0)
        recorder.enabled = True
        with patch.dict(
            sys.modules,
            {"vllm.v1.outputs": SimpleNamespace(AsyncModelRunnerOutput=AsyncOutput)},
        ):
            result = runner.execute_model(None)
            self.assertIs(result, output)
            self.assertEqual(output.calls, 0)
            self.assertEqual(recorder.in_flight, 1)
            result.get_output()
            self.assertEqual(recorder.in_flight, 0)
            timestamp = recorder.steps[0]["observed_ns"]
            result.get_output()
            self.assertEqual(recorder.steps[0]["observed_ns"], timestamp)
            self.assertEqual(recorder.in_flight, 0)


def test_source_identity_records_checkout_changes(tmp_path):
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    source = tmp_path / "module.py"
    source.write_text("value = 1\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "-m",
            "fixture",
        ],
        check=True,
        capture_output=True,
    )
    original = source_identity(tmp_path, "vllm")
    assert original["diff_sha256"] == hashlib.sha256(b"").hexdigest()
    source.write_text("value = 2\n")
    changed = source_identity(tmp_path, "vllm")
    assert changed["commit"] == original["commit"]
    assert changed["diff_sha256"] != original["diff_sha256"]


def test_source_identity_accepts_installed_git_dependency(tmp_path, monkeypatch):
    url = {
        "url": "https://github.com/vllm-project/vllm.git",
        "vcs_info": {"vcs": "git", "commit_id": "a" * 40},
    }
    metadata = SimpleNamespace(version="0.31.1rc0", read_text=lambda _: json.dumps(url))
    monkeypatch.setattr("importlib.metadata.distribution", lambda _: metadata)
    identity = source_identity(tmp_path, "vllm")
    assert identity["commit"] == "a" * 40
    assert identity["package_version"] == "0.31.1rc0"
    assert identity["direct_url"] == url


@pytest.mark.parametrize("direct_url", [None, "{}", '{"dir_info": {}}'])
def test_source_identity_rejects_unidentified_install(
    tmp_path, monkeypatch, direct_url
):
    metadata = SimpleNamespace(version="0.31.1rc0", read_text=lambda _: direct_url)
    monkeypatch.setattr("importlib.metadata.distribution", lambda _: metadata)
    with pytest.raises(ValueError, match="provenance|commit identity"):
        source_identity(tmp_path, "vllm")


def test_source_identity_accepts_buildkite_image(tmp_path, monkeypatch):
    metadata = SimpleNamespace(version="0.0.0", read_text=lambda _: None)
    monkeypatch.setattr("importlib.metadata.distribution", lambda _: metadata)
    identity = source_identity(tmp_path, "vllm-torchtpu", build_commit="b" * 40)
    assert identity["commit"] == "b" * 40
    assert identity["provenance"] == "Buildkite CI image"
    with pytest.raises(ValueError, match="full Git commit SHA"):
        source_identity(tmp_path, "vllm-torchtpu", build_commit="HEAD")
