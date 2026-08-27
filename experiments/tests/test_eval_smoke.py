"""test_eval_smoke.py.

End-to-end hardware evaluation smoke test running on physical Cloud TPU v6e chips.
Boots a live vLLM OpenAI API server with Gemma 4 31B-IT, executes a single-question
MMLU-Pro evaluation query via lm-evaluation-harness, and verifies clean teardown.
"""

import os
import tempfile
import unittest
import pytest

from core.models import ExecutionContext
from runners.eval_runner import TpuEvalRunner

try:
    from tests.fixtures import find_gemma4_checkpoint
except ImportError:
    from fixtures import find_gemma4_checkpoint


@pytest.mark.hardware_smoke
class TestLiveGemma4EvalSmoke(unittest.TestCase):
    """Smoke test for live Cloud TPU vLLM OpenAI server and lm_eval harness integration."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    @unittest.skipUnless(
        os.path.exists("/dev/vfio")
        and len([f for f in os.listdir("/dev/vfio") if f.isdigit()]) >= 4,
        "Requires at least 4 physical Cloud TPU accelerator chips (/dev/vfio/0..3).",
    )
    def test_live_gemma4_eval_smoke(self):
        """Validates that Gemma 4 31B boots on TPU chips, serves lm_eval, and tears down cleanly."""
        # 1. Resolve local HuggingFace cache snapshot or fallback to HF hub repo id
        model_path = find_gemma4_checkpoint() or "google/gemma-4-31B-it"

        # 2. Build execution context (limit=1 for fast smoke testing)
        context = ExecutionContext(
            model_path=model_path,
            tensor_parallel_size=4,
            max_model_len=8192,
            tasks="mmlu_pro",
            limit=1,
            batch_size=4,
            output_dir=self.output_dir,
            engine_args={
                "kv_cache_dtype": "fp8",
                "gpu_memory_utilization": 0.85,
            },
        )

        # 3. Execute evaluation runner
        runner = TpuEvalRunner()
        result = runner.execute(context)

        # 4. Assertions
        self.assertGreater(result.duration_s, 0.0)
        self.assertIn("mmlu_pro_exact_match,custom-extract", result.metrics)
        accuracy = float(result.metrics["mmlu_pro_exact_match,custom-extract"])
        self.assertGreater(
            accuracy,
            0.0,
            f"Expected positive accuracy on MMLU-Pro smoke test, got: {accuracy}",
        )
        self.assertIn("Eval_Duration_s", result.metrics)

        # 5. Verify hardware lock cleanup
        self.assertFalse(
            os.path.exists("/tmp/libtpu_lockfile"),
            "/tmp/libtpu_lockfile was not removed during server teardown!",
        )


if __name__ == "__main__":
    unittest.main()
