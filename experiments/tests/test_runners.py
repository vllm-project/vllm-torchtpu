"""
test_runners.py
---------------
Comprehensive unit and integration test suite for polymorphic benchmark runners:
- RunnerRegistry: Dynamic lookup, custom runner registration, and error handling for unknown types
- TpuProfileRunner: Single-batch throughput benchmarking, latency aggregation, and profiler lifecycle
- TpuEvalRunner: Full OpenAI API server subprocess lifecycle, HTTP 200 OK polling, and lm_eval output parsing
"""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from core.models import BenchmarkResult, ExecutionContext
from runners.base_runner import BaseVllmRunner
from runners.eval_runner import TpuEvalRunner
from runners.profile_runner import TpuProfileRunner
from runners.registry import RunnerRegistry


class TestRunnerRegistry(unittest.TestCase):
    """Tests RunnerRegistry factory and runner dispatching."""

    def test_registry_lookups(self):
        """
        Arrange-Act-Assert: Verify standard runner instance retrieval from RunnerRegistry.
        """
        profile_runner = RunnerRegistry.get("profile")
        self.assertIsInstance(profile_runner, TpuProfileRunner)

        eval_runner = RunnerRegistry.get("eval")
        self.assertIsInstance(eval_runner, TpuEvalRunner)

    def test_unknown_runner_error(self):
        """
        Arrange-Act-Assert: Verify error handling when an unregistered runner mode is requested.
        """
        with self.assertRaises(KeyError):
            RunnerRegistry.get("non_existent_runner_xyz")

    def test_custom_runner_registration(self):
        """
        Arrange-Act-Assert: Dynamic registration of custom runner class via @RunnerRegistry.register.
        """

        @RunnerRegistry.register("custom_mock_runner")
        class CustomMockRunner(BaseVllmRunner):

            def execute(self, context: ExecutionContext) -> BenchmarkResult:
                return BenchmarkResult(duration_s=1.0,
                                       metrics={"custom_metric": 100.0})

        runner = RunnerRegistry.get("custom_mock_runner")
        self.assertIsInstance(runner, CustomMockRunner)
        res = runner.execute(ExecutionContext(model_path="/dummy"))
        self.assertEqual(res.metrics["custom_metric"], 100.0)


class TestTpuProfileRunner(unittest.TestCase):
    """Tests TpuProfileRunner token generation benchmarking and profiler tracing."""

    @patch("runners.profile_runner.LLM")
    @patch("runners.profile_runner.SamplingParams")
    def test_profile_runner_standard_execution(self, mock_sampling_params,
                                               mock_llm_cls):
        """
        Arrange-Act-Assert: Execute standard throughput profiling.
        Verifies:
        1. Mocks vLLM LLM engine initialization and output token counts.
        2. Computes total tokens (batch_size * output_len).
        3. Calculates throughput (tok/s) and latency metrics.
        """
        # Arrange: Setup mock vLLM output structures
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(token_ids=[i for i in range(64)])]
        mock_engine = MagicMock()
        mock_engine.generate.return_value = [mock_output
                                             ] * 4  # 4 sequences in batch
        mock_llm_cls.return_value = mock_engine

        runner = TpuProfileRunner()
        context = ExecutionContext(model_path="/path/to/model",
                                   tensor_parallel_size=4,
                                   batch_size=4,
                                   input_len=128,
                                   output_len=64,
                                   trace=False)

        # Act: Execute benchmark
        result = runner.execute(context)

        # Assert: Validate metric structure and values
        self.assertIsInstance(result, BenchmarkResult)
        self.assertIn("Throughput_tok_s", result.metrics)
        self.assertIn("Duration_s", result.metrics)

    @patch("runners.profile_runner.LLM")
    def test_profile_runner_with_jax_profiler_tracing(self, mock_llm_cls):
        """
        Arrange-Act-Assert: Execute throughput profiling with JAX/TPU profiler active.
        Verifies that start_profile() and stop_profile() lifecycle hooks are invoked on the engine.
        """
        # Arrange: Setup mock LLM with profiler capabilities
        mock_engine = MagicMock()
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(token_ids=[1, 2, 3])]
        mock_engine.generate.return_value = [mock_output]
        mock_llm_cls.return_value = mock_engine

        runner = TpuProfileRunner()
        context = ExecutionContext(model_path="/path/to/model",
                                   tensor_parallel_size=4,
                                   trace=True,
                                   jax_advanced_configuration={
                                       "tpu_enable_periodic_counter_sampling":
                                       True
                                   })

        # Act: Run profiling
        result = runner.execute(context)

        # Assert: Verify profiler lifecycle methods were invoked
        mock_engine.start_profile.assert_called_once()
        mock_engine.stop_profile.assert_called_once()
        self.assertIsInstance(result, BenchmarkResult)


class TestTpuEvalRunner(unittest.TestCase):
    """Tests TpuEvalRunner OpenAI server lifecycle and lm_eval execution."""

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.test_dir.name

    def tearDown(self):
        self.test_dir.cleanup()

    @patch("runners.eval_runner.subprocess.Popen")
    @patch("runners.eval_runner.urllib.request.urlopen")
    @patch("runners.eval_runner.subprocess.run")
    def test_eval_runner_full_lifecycle(self, mock_subproc_run, mock_urlopen,
                                        mock_subproc_popen):
        """
        Arrange-Act-Assert: Verify full evaluation pipeline lifecycle.
        Steps Tested:
        1. Subprocess launching of vLLM OpenAI API server.
        2. Polling /health HTTP endpoint until 200 OK status is returned.
        3. Execution of lm_eval evaluation CLI.
        4. Ingestion of evaluation JSON results and metric extraction.
        5. Guaranteed server process termination in finally block.
        """
        # Arrange: Create simulated lm_eval result JSON file
        eval_logs = os.path.join(self.output_dir, "eval_logs")
        os.makedirs(eval_logs, exist_ok=True)
        results_file = os.path.join(eval_logs, "results_2026-08-26.json")
        sample_results = {
            "results": {
                "mmlu_pro": {
                    "exact_match,custom-extract": 0.8810,
                    "exact_match_stderr,custom-extract": 0.012
                },
                "mmlu_pro_biology": {
                    "exact_match,custom-extract": 1.0000
                }
            }
        }
        with open(results_file, "w") as f:
            json.dump(sample_results, f)

        # 1. Mock server process and lm_eval process
        mock_server_proc = MagicMock()
        mock_server_proc.poll.return_value = None  # Running process
        mock_server_proc.returncode = 0

        mock_eval_proc = MagicMock()
        mock_eval_proc.stdout.readline.side_effect = [
            "Processed prompts:  50%|█████     | 300/608 [01:00, est. speed input: 4000.00 toks/s, output: 500.00 toks/s]\n",
            "Processed prompts: 100%|██████████| 608/608 [02:00, est. speed input: 6000.00 toks/s, output: 800.00 toks/s]\n",
            ""
        ]
        mock_eval_proc.wait.return_value = 0

        # Return mock_server_proc for server launch, and mock_eval_proc for lm_eval execution
        mock_subproc_popen.side_effect = [mock_server_proc, mock_eval_proc]

        # 2. Mock /health endpoint response
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        # 3. Mock lm_eval run
        mock_subproc_run.return_value = MagicMock(returncode=0)

        # Act: Execute evaluation runner
        runner = TpuEvalRunner()
        context = ExecutionContext(model_path="/tmp/mock_models/gemma4",
                                   tensor_parallel_size=4,
                                   max_model_len=8192,
                                   tasks="mmlu_pro",
                                   output_dir=self.output_dir,
                                   engine_args={
                                       "gpu_memory_utilization": 0.85,
                                       "kv_cache_dtype": "bfloat16"
                                   })

        result = runner.execute(context)

        # Assert: Validate extracted metrics
        self.assertIsInstance(result, BenchmarkResult)
        self.assertIn("mmlu_pro_exact_match,custom-extract", result.metrics)
        self.assertEqual(result.metrics["mmlu_pro_exact_match,custom-extract"],
                         "0.8810")
        self.assertEqual(
            result.metrics["mmlu_pro_biology_exact_match,custom-extract"],
            "1.0000")
        self.assertIn("Eval_Duration_s", result.metrics)

        # Verify server shutdown was invoked in finally
        mock_server_proc.terminate.assert_called_once()
        self.assertGreaterEqual(mock_subproc_run.call_count, 1)
        self.assertIn("Eval_E2E_Latency_s", result.metrics)
        self.assertIn("Eval_Input_Throughput_tok_s", result.metrics)
        self.assertEqual(result.metrics["Eval_Input_Throughput_tok_s"],
                         "6000.00")
        self.assertEqual(result.metrics["Eval_Output_Throughput_tok_s"],
                         "800.00")


if __name__ == "__main__":
    unittest.main()
