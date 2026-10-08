"""run_sweep_orchestrator.py.

Modular TPU Sweep Orchestrator for throughput profiling and benchmark accuracy evaluation.

Adheres to SOLID architecture principles:
- SRP: Generates parameter cartesian grids and delegates execution to runners, resolvers, and recorders.
- OCP: Extensible via RunnerRegistry and ResolverRegistry without modifying the orchestration loop.
- LSP: Polymorphic execution through IRunner and IModelResolver interfaces.
- ISP: Clean interface separation via ExecutionContext and BenchmarkResult dataclasses.
- DIP: Decoupled orchestration depending strictly on abstractions.
"""

import argparse
import csv
import itertools
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

# Ensure experiments root directory is in the Python search path
EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
if EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENTS_DIR)

from core.csv_recorder import CsvResultRecorder  # noqa: E402

# Patch transformers for heterogeneous architectures (e.g. Gemma 4)
try:
    from transformers.configuration_utils import PretrainedConfig  # noqa: E402
    PretrainedConfig.allow_global_per_layer_attribute_access = True
except Exception:
    pass
from core.env import setup_tpu_env  # noqa: E402
from core.models import BenchmarkResult, ExecutionContext  # noqa: E402
from resolvers.registry import ResolverRegistry  # noqa: E402
from runners.registry import RunnerRegistry  # noqa: E402

from core.json_recorder import CompositeResultRecorder, JsonResultRecorder  # noqa: E402  # isort: skip

# Apply TPU execution flags, cache variables, and repository paths
setup_tpu_env()

DEFAULT_RESULT_DIR = os.path.join(EXPERIMENTS_DIR, "results")


class SweepOrchestrator:
    """Main orchestrator for managing multi-dimensional parameter sweeps on Cloud TPUs.

    Coordinates parameter grid expansion, model resolution, runner execution,
    and metric persistence.
    """

    def __init__(
        self,
        config_path: str,
        result_dir: str = DEFAULT_RESULT_DIR,
        experiment_id: Optional[str] = None,
        message: Optional[str] = None,
        resume: bool = False,
    ):
        """Initializes the sweep orchestrator and prepares output directory and recorders."""
        self.config_path = config_path
        self.result_dir = result_dir
        self.experiment_id = experiment_id
        self.message = message
        self.resume = resume
        self.config = self._load_config(config_path)

        self.exp_id = (self.experiment_id if self.experiment_id else
                       datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.exp_dir = os.path.join(self.result_dir, self.exp_id)
        os.makedirs(self.exp_dir, exist_ok=True)
        self.csv_file = os.path.join(self.exp_dir, "results.csv")
        self.json_file = os.path.join(self.exp_dir, "sweep.json")
        self.metadata_file = os.path.join(self.exp_dir, "metadata.yaml")

        self._setup_metadata()

        # Initialize dual result recorders (CSV for flat logs, JSON for hierarchical schema)
        self.csv_recorder = CsvResultRecorder(self.csv_file)
        hardware_info = {
            "accelerator_type": self.config.get("accelerator_type", "TPU v6e"),
            "chip_count": int(self.config.get("tensor_parallel_size", 4)),
        }
        self.json_recorder = JsonResultRecorder(
            json_file=self.json_file,
            sweep_id=self.exp_id,
            metadata=getattr(self, "metadata_content", self.config),
            hardware=hardware_info,
        )
        self.recorder = CompositeResultRecorder(
            [self.csv_recorder, self.json_recorder])

    def _setup_metadata(self) -> None:
        """Handles metadata.yaml persistence and config verification on resume."""
        current_config = {
            k: v
            for k, v in self.config.items() if k != "description"
        }

        if self.resume and not os.path.exists(self.metadata_file):
            raise FileNotFoundError(
                f"Cannot resume experiment '{self.exp_id}': metadata file"
                f" not found at {self.metadata_file}.")

        if os.path.exists(self.metadata_file):
            saved_metadata = self._load_config(self.metadata_file) or {}
            self.metadata_content = saved_metadata
            if "config" in saved_metadata and isinstance(
                    saved_metadata["config"], dict):
                saved_config = saved_metadata["config"]
            else:
                saved_config = {
                    k: v
                    for k, v in saved_metadata.items() if k != "description"
                }

            if current_config != saved_config:
                raise ValueError(
                    f"Cannot resume experiment '{self.exp_id}': provided"
                    " config does not match the config saved in"
                    f" {self.metadata_file}.")
            print(f">>> [RESUME] Resuming experiment {self.exp_id} with"
                  " verified matching config (ignoring -m/--message).")
        else:
            # Fresh run: save a copy of the exact config under id/metadata.yaml with description
            metadata: Dict[str, Any] = {}
            metadata["description"] = self.message or ""
            for k, v in self.config.items():
                if k != "description":
                    metadata[k] = v

            with open(self.metadata_file, "w") as f:
                yaml.dump(metadata,
                          f,
                          default_flow_style=False,
                          sort_keys=False)
            self.metadata_content = metadata
            print(">>> [METADATA] Saved experiment metadata to"
                  f" {self.metadata_file}")

    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        """Loads and parses the YAML experiment configuration file."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _resolve_model(self,
                       current_config: Dict[str, Any]) -> Tuple[str, Set[str]]:
        """Resolves target model path using ResolverRegistry or falls back to static model path."""
        spec = self.config.get("model_resolver")

        # For normal inference parameter sweeps without sparsity or modifying model, we can just return the model path from the config
        if not spec:
            return current_config.get("model", self.config.get("model",
                                                               "")), set()

        # Otherwise if we need to resolve a model dynamically (e.g., sparsity, pruning, etc.), we use the resolver registry
        resolver = ResolverRegistry.get(spec.get("module", "sparsegpt"))
        return resolver.resolve(current_config, spec.get("config", {}))

    def _get_completed_coordinates(self) -> List[Dict[str, Any]]:
        """Reads existing completed trial parameters from sweep.json (primary source of truth)

        with fallback to results.csv for legacy backwards compatibility.
        """
        completed: List[Dict[str, Any]] = []

        # 1. Primary Source of Truth: sweep.json
        if os.path.exists(self.json_file):
            try:
                with open(self.json_file, "r") as f:
                    data = json.load(f)
                    for trial in data.get("experiments", []):
                        if trial.get(
                                "status"
                        ) == "COMPLETED" and "parameters" in trial:
                            completed.append(trial["parameters"])
                if completed:
                    return completed
            except Exception as e:
                print(f">>> [WARNING] Failed to parse {self.json_file}: {e}")

        # 2. Fallback for legacy experiments: results.csv
        if os.path.exists(self.csv_file):
            with open(self.csv_file, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    completed.append(row)
        return completed

    def run(self) -> str:
        """Executes the full sweep matrix across all Cartesian product coordinates."""
        mode = self.config.get("mode", "profile").lower()
        sweep = self.config.get("sweep_matrix", {})
        eval_cfg = self.config.get("evaluation", {})

        # Normalize sweep matrix values into lists
        for k, v in sweep.items():
            if not isinstance(v, list):
                sweep[k] = [v]

        sweep_keys = list(sweep.keys())
        sweep_values = list(sweep.values())
        cross_product = list(itertools.product(*sweep_values))

        # Check for previously completed coordinates from sweep.json (or csv fallback)
        completed_configs = self._get_completed_coordinates()

        print("=======================================================")
        print(
            f">>> [SWEEP ORCHESTRATOR] Initializing Sweep: {self.config_path}")
        print(
            f">>> Total Grid Points: {len(cross_product)} | Mode: {mode.upper()}"
        )
        print(f">>> Experiment Directory: {self.exp_dir}")
        print("=======================================================")

        for combo in cross_product:
            current_config = dict(zip(sweep_keys, combo))

            # Resumption check: skip coordinate if already recorded in sweep.json or CSV
            skip = False
            for completed_row in completed_configs:
                if all(
                        str(completed_row.get(k)) == str(v)
                        for k, v in current_config.items()):
                    skip = True
                    break

            if skip:
                source_file = (self.json_file if os.path.exists(self.json_file)
                               and completed_configs else self.csv_file)
                print(
                    f">>> [SKIPPED] Coordinate {current_config} already completed in {os.path.basename(source_file)}."
                )
                continue

            # 1. Resolve target model checkpoint dynamically
            target_model, consumed_keys = self._resolve_model(current_config)

            # 2. Build normalized ExecutionContext
            combined_metadata = {**current_config}
            batch_size = current_config.get(
                "batch_size",
                eval_cfg.get("batch_size",
                             self.config.get("batch_size", "auto")),
            )
            tasks = current_config.get("tasks",
                                       eval_cfg.get("tasks", "mmlu_pro"))
            limit = current_config.get("limit", eval_cfg.get("limit", None))

            # Resolve trace configuration and advanced JAX profiler options
            trace_configs = self.config.get("trace_configs", [])
            trace = bool(trace_configs)
            matching_jax_config = {}
            for tc in trace_configs:
                matches = True
                for k in ("batch_size", "input_len", "output_len"):
                    if k in tc and str(tc[k]) != str(current_config.get(k)):
                        matches = False
                        break
                if matches:
                    matching_jax_config = tc.get("jax_advanced_configuration",
                                                 {})
                    break
            if trace and not matching_jax_config and trace_configs:
                matching_jax_config = trace_configs[0].get(
                    "jax_advanced_configuration", {})

            context = ExecutionContext(
                model_path=target_model,
                tensor_parallel_size=self.config.get("tensor_parallel_size",
                                                     4),
                dtype=self.config.get("dtype", "bfloat16"),
                max_model_len=self.config.get("max_model_len", None),
                engine_args=self.config.get("engine_args", {}),
                sweep_metadata=combined_metadata,
                output_dir=self.exp_dir,
                csv_file=self.csv_file,
                batch_size=batch_size,
                input_len=current_config.get("input_len",
                                             current_config.get("inputs",
                                                                128)),
                output_len=current_config.get("output_len",
                                              current_config.get("output",
                                                                 64)),
                trace=trace,
                jax_advanced_configuration=matching_jax_config,
                tasks=tasks,
                limit=limit,
            )

            # 3. Execute benchmark via polymorphic runner (profile, eval, or both)
            runner_names = ["profile", "eval"] if mode == "both" else [mode]
            cmd_repro = (f"python3 run_sweep_orchestrator.py --config "
                         f"{self.config_path}")
            for r_name in runner_names:
                runner = RunnerRegistry.get(r_name)
                try:
                    result: BenchmarkResult = runner.execute(context)
                except Exception as e:
                    # Persist the failure so sweep.json never stays RUNNING
                    # after a crash, then propagate to the caller.
                    error_message = f"{type(e).__name__}: {e}"
                    self.json_recorder.record_failure(combined_metadata,
                                                      error_message, cmd_repro)
                    self.recorder.finalize(status="FAILED",
                                           error_message=error_message)
                    raise

                # 4. Persist trial metrics to CSV and JSON
                self.recorder.record(
                    metadata=combined_metadata,
                    metrics=result.metrics,
                    reproduction_cmd=cmd_repro,
                )

        # Finalize recorders (e.g. stamp completed_at in sweep.json)
        self.recorder.finalize(status="COMPLETED")

        print(f">>> [SWEEP COMPLETE] Results written to {self.csv_file}")
        if os.path.exists(self.json_file):
            print(
                f">>> [SWEEP COMPLETE] Sweep JSON written to {self.json_file}")
        return self.csv_file


def main():
    """CLI entry point for running parameter sweeps."""
    parser = argparse.ArgumentParser(
        description="Modular TPU Sweep Orchestrator")
    parser.add_argument("--config",
                        required=True,
                        help="Path to YAML configuration file")
    parser.add_argument(
        "--result-dir",
        default=DEFAULT_RESULT_DIR,
        help="Base directory for results",
    )
    parser.add_argument(
        "--experiment-id",
        "--id",
        "-id",
        dest="experiment_id",
        default=None,
        help="Experiment identifier to run or resume",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help=
        "Require resuming an existing experiment (fails if metadata is missing)",
    )
    parser.add_argument(
        "-m",
        "--message",
        dest="message",
        default=None,
        help="Description of the experiment we are running",
    )
    args = parser.parse_args()

    orchestrator = SweepOrchestrator(
        config_path=args.config,
        result_dir=args.result_dir,
        experiment_id=args.experiment_id,
        message=args.message,
        resume=args.resume,
    )
    orchestrator.run()


if __name__ == "__main__":
    main()
