"""
test_integration.py
-------------------
End-to-end integration test validating the entire sweep pipeline lifecycle:
Synthetic model generation -> Dynamic Resolver & Layer Splicer -> Benchmark Runner Execution -> CsvResultRecorder Persistence.
"""

import csv
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_sweep_orchestrator as run_sweep  # noqa: E402
from core.models import BenchmarkResult  # noqa: E402

try:
    from tests.fixtures import create_synthetic_model  # noqa: E402
except ImportError:
    from fixtures import create_synthetic_model  # noqa: E402


class TestEndToEndPipelineIntegration(unittest.TestCase):
    """Integrates checkpoint resolution, layer splicing, runner dispatch, and metric persistence."""

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.dense_dir = os.path.join(self.test_dir.name, "dense_base")
        self.sparse_dir = os.path.join(self.test_dir.name, "sparse_base")
        self.results_dir = os.path.join(self.test_dir.name, "results")
        self.config_path = os.path.join(self.test_dir.name,
                                        "sweep_config.yaml")

        # Generate miniature 4-layer 2-shard synthetic base models
        create_synthetic_model(self.dense_dir,
                               num_layers=4,
                               num_shards=2,
                               sparsity_ratio=0.0)
        create_synthetic_model(self.sparse_dir,
                               num_layers=4,
                               num_shards=2,
                               sparsity_ratio=0.5)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_full_sparsity_sweep_pipeline(self):
        """
        Arrange-Act-Assert: Executes an end-to-end multi-point sparsity sweep pipeline.
        Pipeline Flow:
        1. Generates 4-layer synthetic dense and sparse models in temp storage.
        2. Configures a sweep evaluating 0, 2, and 4 sparse layers.
        3. Orchestrator calls SparseGptResolver to splice checkpoints on the fly.
        4. Mock runner simulates evaluation accuracy.
        5. CsvResultRecorder logs all 3 points to results.csv.
        """
        # Arrange: Define sweep YAML configuration
        config_data = {
            "mode": "eval",
            "model": self.dense_dir,
            "tensor_parallel_size": 4,
            "sweep_matrix": {
                "sparse_layers": [0, 2, 4],
                "batch_size": [4]
            },
            "model_resolver": {
                "module": "sparsegpt",
                "config": {
                    "base_model": self.dense_dir,
                    "fully_sparse_model": self.sparse_dir,
                    "sparsity_strategy": "first_to_last"
                }
            }
        }

        with open(self.config_path, "w") as f:
            yaml.dump(config_data, f)

        # Mock runner execution with dynamic simulated metrics
        def mock_execute_side_effect(context):
            sparse_count = context.sweep_metadata.get("sparse_layers", 0)
            simulated_acc = 0.88 - (sparse_count * 0.05)
            return BenchmarkResult(duration_s=12.5,
                                   metrics={
                                       "mmlu_pro_exact_match":
                                       f"{simulated_acc:.4f}",
                                       "Eval_Duration_s": "12.50"
                                   },
                                   log_path="/path/to/log.json")

        mock_runner = MagicMock()
        mock_runner.execute.side_effect = mock_execute_side_effect

        # Act: Execute full orchestrator pipeline
        with patch("runners.registry.RunnerRegistry.get",
                   return_value=mock_runner):
            orchestrator = run_sweep.SweepOrchestrator(
                config_path=self.config_path,
                result_dir=self.results_dir,
                upload_gcs=False)
            orchestrator.run()

            # Assert: Validate resulting CSV format and recorded values
            csv_path = orchestrator.csv_file
            self.assertTrue(os.path.exists(csv_path),
                            "Results CSV must be written to disk")

            with open(csv_path, "r") as f:
                reader = list(csv.DictReader(f))
                self.assertEqual(len(reader), 3,
                                 "Must record 3 sweep coordinates")
                self.assertIn("mmlu_pro_exact_match", reader[0])
                self.assertEqual(reader[0]["sparse_layers"], "0")
                self.assertEqual(reader[1]["sparse_layers"], "2")
                self.assertEqual(reader[2]["sparse_layers"], "4")


if __name__ == "__main__":
    unittest.main()
