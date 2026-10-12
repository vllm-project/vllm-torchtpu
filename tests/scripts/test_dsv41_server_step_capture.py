# SPDX-License-Identifier: Apache-2.0
"""Regression for decoder metadata losing its init=False CPU fields."""

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
from scenarios import SCENARIOS  # noqa: E402

pytestmark = pytest.mark.cpu_test


class CaptureTest(unittest.TestCase):
    def test_decoder_uses_encoder_snapshot_and_removes_padding(self):
        metadata = SimpleNamespace(
            query_start_loc_cpu=(0, 384, 384),
            seq_lens_cpu=(65536, 0),
            prompt_lens_cpu=(65536, 0),
        )
        context = SimpleNamespace(attn_metadata={"attention": metadata})
        recorder = SimpleNamespace(
            forward_id=0,
            decoder_snapshot=None,
            scenario=SCENARIOS["swa_replay"],
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
        self.assertEqual(encoder["starts"], [65152])
        self.assertEqual(decoder["starts"], [65408])
        self.assertEqual(decoder["ends"], [65536])
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
        recorder = worker.Recorder(runner, SCENARIOS["encoder_extend"], 0)
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


if __name__ == "__main__":
    unittest.main()
