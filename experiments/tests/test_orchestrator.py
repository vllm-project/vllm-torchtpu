"""
test_orchestrator.py
--------------------
Unit test suite for the master sweep orchestrator:
- Cartesian grid parameter expansion from sweep_matrix configurations.
- Checkpoint resumption capability (automatically skipping previously evaluated coordinates in results.csv).
- Error resilience and result persistence across sweep iterations.
"""

import csv
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_sweep_orchestrator as run_sweep  # noqa: E402
from core.models import BenchmarkResult  # noqa: E402


class TestSweepOrchestrator(unittest.TestCase):
    """Tests Cartesian grid sequencing, parameter expansion, and execution flow."""

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.test_dir.name, "test_config.yaml")
        self.results_dir = os.path.join(self.test_dir.name, "results")

    def tearDown(self):
        self.test_dir.cleanup()

    def _write_yaml(self, data):
        with open(self.config_path, "w") as f:
            yaml.dump(data, f)

    def test_grid_generation_and_execution(self):
        """
        Arrange-Act-Assert: Cartesian product expansion across sweep matrix dimensions.
        Verifies:
        1. 2 models x 2 batch sizes = 4 total grid coordinates.
        2. Orchestrator iterates over all 4 coordinates sequentially.
        3. Results CSV contains exactly 4 recorded rows with all sweep parameters.
        """
        # Arrange: Define sweep matrix
        config_data = {
            "mode": "eval",
            "model": "base_model",
            "tensor_parallel_size": 4,
            "sweep_matrix": {
                "model": ["model_A", "model_B"],
                "batch_size": [4, 8]
            },
            "engine_args": {
                "gpu_memory_utilization": 0.85,
                "kv_cache_dtype": "bfloat16"
            }
        }
        self._write_yaml(config_data)

        # Mock runner execution
        mock_runner = MagicMock()
        mock_runner.execute.return_value = BenchmarkResult(
            duration_s=10.0,
            metrics={"exact_match": "0.85"},
            log_path="dummy.json")

        # Act: Execute sweep
        with patch("runners.registry.RunnerRegistry.get",
                   return_value=mock_runner):
            orchestrator = run_sweep.SweepOrchestrator(
                config_path=self.config_path, result_dir=self.results_dir)
            orchestrator.run()

            # Assert: Runner was invoked 4 times and CSV has 4 rows
            self.assertEqual(mock_runner.execute.call_count, 4,
                             "Should execute 4 grid coordinates")
            csv_path = orchestrator.csv_file
            self.assertTrue(os.path.exists(csv_path))

            with open(csv_path, "r") as f:
                reader = list(csv.DictReader(f))
                self.assertEqual(len(reader), 4)

    def test_resume_skipping_already_evaluated_points(self):
        """
        Arrange-Act-Assert: Crash recovery and sweep resumption logic.
        Verifies:
        1. When an existing results.csv contains an already-completed coordinate, the orchestrator skips execution for that point.
        2. Only new un-evaluated coordinates are dispatched to the runner.
        """
        # Arrange: Setup 2-point sweep
        config_data = {
            "mode": "eval",
            "model": "base_model",
            "tensor_parallel_size": 4,
            "sweep_matrix": {
                "model": ["model_A", "model_B"],
                "batch_size": [4]
            }
        }
        self._write_yaml(config_data)

        exp_dir = os.path.join(self.results_dir, "test_resume_exp")
        os.makedirs(exp_dir, exist_ok=True)
        csv_file = os.path.join(exp_dir, "results.csv")

        # Pre-seed CSV with model_A already completed
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "batch_size", "exact_match"])
            writer.writerow(["model_A", "4", "0.88"])

        # Pre-seed metadata.yaml matching config
        metadata_file = os.path.join(exp_dir, "metadata.yaml")
        with open(metadata_file, "w") as f:
            yaml.dump(config_data, f)

        mock_runner = MagicMock()
        mock_runner.execute.return_value = BenchmarkResult(
            duration_s=5.0,
            metrics={"exact_match": "0.82"},
            log_path="dummy.json")

        # Act: Run orchestrator pointing to pre-seeded directory
        with patch("runners.registry.RunnerRegistry.get",
                   return_value=mock_runner):
            orchestrator = run_sweep.SweepOrchestrator(
                config_path=self.config_path,
                result_dir=self.results_dir,
                experiment_id="test_resume_exp")
            orchestrator.run()

            # Assert: Runner only invoked 1 time (for model_B)
            self.assertEqual(
                mock_runner.execute.call_count, 1,
                "model_A was completed, so runner should only run once for model_B"
            )

            with open(csv_file, "r") as f:
                reader = list(csv.DictReader(f))
                self.assertEqual(len(reader), 2,
                                 "Final CSV should contain both rows")

    def test_resume_skipping_already_evaluated_points_from_sweep_json(self):
        """Verifies that sweep.json is the primary source of truth for skipping evaluated coordinates."""
        config_data = {
            "mode": "profile",
            "sweep_matrix": {
                "model": ["model_A", "model_B"],
                "batch_size": [4]
            }
        }
        self._write_yaml(config_data)

        exp_dir = os.path.join(self.results_dir, "test_resume_json_exp")
        os.makedirs(exp_dir, exist_ok=True)
        json_file = os.path.join(exp_dir, "sweep.json")
        metadata_file = os.path.join(exp_dir, "metadata.yaml")

        # Pre-seed sweep.json with model_A already completed
        sweep_data = {
            "schema_version":
            "1.0.0",
            "sweep_id":
            "test_resume_json_exp",
            "status":
            "RUNNING",
            "metadata":
            config_data,
            "experiments": [{
                "experiment_id": "test_resume_json_exp_trial_000",
                "trial_index": 0,
                "status": "COMPLETED",
                "timestamp": "2026-08-28T12:00:00Z",
                "parameters": {
                    "model": "model_A",
                    "batch_size": 4
                },
                "metrics": {
                    "Throughput_tok_s": 50.0
                }
            }]
        }
        with open(json_file, "w") as f:
            json.dump(sweep_data, f)

        # Pre-seed metadata.yaml matching config
        with open(metadata_file, "w") as f:
            yaml.dump(config_data, f)

        mock_runner = MagicMock()
        mock_runner.execute.return_value = BenchmarkResult(
            duration_s=5.0,
            metrics={"Throughput_tok_s": 60.0},
            log_path="dummy.json")

        with patch("runners.registry.RunnerRegistry.get",
                   return_value=mock_runner):
            orchestrator = run_sweep.SweepOrchestrator(
                config_path=self.config_path,
                result_dir=self.results_dir,
                experiment_id="test_resume_json_exp")
            orchestrator.run()

            # Assert: Runner only invoked 1 time (for model_B) because model_A was in sweep.json
            self.assertEqual(
                mock_runner.execute.call_count, 1,
                "model_A was recorded in sweep.json, so only model_B should run"
            )

            with open(json_file, "r") as f:
                data = json.load(f)
                self.assertEqual(
                    len(data["experiments"]), 2,
                    "Final sweep.json should contain both trials")

    def test_metadata_file_creation_with_message(self):
        """Verifies that a new experiment creates metadata.yaml containing config and description."""
        config_data = {
            "mode": "profile",
            "model": "google/gemma-4-31B-it",
            "sweep_matrix": {
                "batch_size": [1]
            }
        }
        self._write_yaml(config_data)

        orchestrator = run_sweep.SweepOrchestrator(
            config_path=self.config_path,
            result_dir=self.results_dir,
            message="Evaluating Gemma 4 throughput")

        metadata_path = orchestrator.metadata_file
        self.assertTrue(os.path.exists(metadata_path),
                        "metadata.yaml must be saved")

        with open(metadata_path, "r") as f:
            saved = yaml.safe_load(f)

        self.assertEqual(saved["description"], "Evaluating Gemma 4 throughput")
        self.assertEqual(saved["mode"], "profile")
        self.assertEqual(saved["model"], "google/gemma-4-31B-it")
        self.assertEqual(saved["sweep_matrix"], {"batch_size": [1]})

    def test_resume_matching_config_ignores_message(self):
        """Verifies that resuming with -id ignores -m and succeeds when configs match."""
        config_data = {
            "mode": "eval",
            "model": "google/gemma-4-31B-it",
            "sweep_matrix": {
                "batch_size": [2]
            }
        }
        self._write_yaml(config_data)

        exp_dir = os.path.join(self.results_dir, "existing_exp")
        os.makedirs(exp_dir, exist_ok=True)
        metadata_file = os.path.join(exp_dir, "metadata.yaml")

        saved_data = {"description": "Original description", **config_data}
        with open(metadata_file, "w") as f:
            yaml.dump(saved_data, f)

        # Resume with different message: -m is ignored, original metadata untouched
        _ = run_sweep.SweepOrchestrator(config_path=self.config_path,
                                        result_dir=self.results_dir,
                                        experiment_id="existing_exp",
                                        message="New ignored description")

        with open(metadata_file, "r") as f:
            persisted = yaml.safe_load(f)

        self.assertEqual(persisted["description"], "Original description")

    def test_resume_mismatched_config_raises_error(self):
        """Verifies that resuming with a modified config raises ValueError."""
        original_config = {
            "mode": "eval",
            "model": "google/gemma-4-31B-it",
            "sweep_matrix": {
                "batch_size": [1]
            }
        }
        exp_dir = os.path.join(self.results_dir, "exp_to_fail")
        os.makedirs(exp_dir, exist_ok=True)
        metadata_file = os.path.join(exp_dir, "metadata.yaml")
        with open(metadata_file, "w") as f:
            yaml.dump({"description": "Base", **original_config}, f)

        # Different config
        modified_config = {
            "mode": "eval",
            "model": "google/gemma-4-31B-it",
            "sweep_matrix": {
                "batch_size": [8]
            }  # Modified!
        }
        self._write_yaml(modified_config)

        with self.assertRaises(ValueError) as ctx:
            run_sweep.SweepOrchestrator(config_path=self.config_path,
                                        result_dir=self.results_dir,
                                        experiment_id="exp_to_fail")
        self.assertIn("provided config does not match the config saved",
                      str(ctx.exception))

    def test_resume_missing_metadata_raises_error(self):
        """Verifies that explicitly resuming an experiment ID with no metadata.yaml raises FileNotFoundError."""
        config_data = {
            "mode": "eval",
            "model": "base_model",
            "sweep_matrix": {
                "batch_size": [1]
            }
        }
        self._write_yaml(config_data)

        with self.assertRaises(FileNotFoundError) as ctx:
            run_sweep.SweepOrchestrator(config_path=self.config_path,
                                        result_dir=self.results_dir,
                                        experiment_id="non_existent_exp",
                                        resume=True)
        self.assertIn("metadata file not found", str(ctx.exception))

    def test_fresh_run_with_custom_experiment_id(self):
        """Verifies that passing an experiment_id without resume=True starts a fresh run with that ID."""
        config_data = {
            "mode": "eval",
            "model": "base_model",
            "sweep_matrix": {
                "batch_size": [1]
            }
        }
        self._write_yaml(config_data)

        orchestrator = run_sweep.SweepOrchestrator(
            config_path=self.config_path,
            result_dir=self.results_dir,
            experiment_id="custom_named_exp",
            resume=False)
        self.assertEqual(orchestrator.exp_id, "custom_named_exp")
        self.assertTrue(os.path.exists(orchestrator.metadata_file))


if __name__ == "__main__":
    unittest.main()
