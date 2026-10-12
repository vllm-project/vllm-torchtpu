# SPDX-License-Identifier: Apache-2.0
"""Opt-in worker extension; no instrumentation in normal serving.

Records natural serving output observations without adding device waits.
Host cadence is not a device event clock or isolated encoder/decoder duration.
"""

import functools
import hashlib
import json
import logging
import socket
import time
from pathlib import Path
from typing import Any

from scenarios import SCENARIOS, classify_rows, token_counts

_ACTIVE = None
_INSTALLED = False


def _stage_wrapper(original: Any, stage: str) -> Any:
    @functools.wraps(original)
    def call(module: Any, *args: Any, **kwargs: Any) -> Any:
        import torch
        from vllm.forward_context import get_forward_context

        if _ACTIVE is None or torch.compiler.is_compiling():
            return original(module, *args, **kwargs)
        recorder = _ACTIVE
        if stage == "encoder":
            recorder.forward_id += 1
        if stage == "encoder":
            recorder.decoder_snapshot = None
            metadata_map = get_forward_context().attn_metadata
            if not isinstance(metadata_map, dict) or not metadata_map:
                return original(module, *args, **kwargs)
            metadata = next(iter(metadata_map.values()))
            if metadata.query_start_loc_cpu is None or metadata.seq_lens_cpu is None:
                return original(module, *args, **kwargs)
            bounds = [int(n) for n in metadata.query_start_loc_cpu]
            ends = [int(n) for n in metadata.seq_lens_cpu]
            prompts = [int(n) for n in metadata.prompt_lens_cpu]
            from vllm_torchtpu.models.vllm.deepseek_v41.v2.ced import (
                plan_decoder_replay,
            )

            plan = plan_decoder_replay(
                metadata,
                module.owner._ced_window,
                max(bounds[-1], 1),
                module.owner._ced_compile_sizes,
            )
            if plan is None:
                raise RuntimeError("Decoder replay plan unavailable")
            # dataclasses.replace drops the metadata's init=False CPU fields
            # when the model constructs decoder metadata. Preserve the actual
            # host replay plan before the compiled encoder runs.
            recorder.decoder_snapshot = (list(plan.query_start_loc), ends, prompts)
        else:
            if recorder.decoder_snapshot is None:
                raise RuntimeError("Decoder has no matching encoder host snapshot")
            bounds, ends, prompts = recorder.decoder_snapshot
        active = [i for i, (a, b) in enumerate(zip(bounds, bounds[1:])) if b > a]
        starts = [ends[i] - (bounds[i + 1] - bounds[i]) for i in active]
        ends = [ends[i] for i in active]
        prompts = [prompts[i] for i in active]
        count = len(active)
        row = {
            "stage": stage,
            "phase": classify_rows(starts, ends, prompts),
            "starts": starts,
            "ends": ends,
            "prompts": prompts,
            "requests": count,
            "rows": bounds[-1],
            "padded_rows": args[1].shape[0],
            **token_counts(starts, ends, recorder.scenario.prefix_tokens),
        }
        row["forward_id"] = recorder.forward_id
        # Shape metadata only: host dispatch time is not device execution time.
        row["dispatch_ns"] = time.perf_counter_ns()
        result = original(module, *args, **kwargs)
        recorder.stages.append(row)
        if recorder.pending is not None:
            recorder.pending["stages"].append(row)
        return result

    return call


class Recorder:
    def __init__(self, runner: Any, scenario: Any, rank: int) -> None:
        self.runner = runner
        self.scenario = scenario
        self.rank = rank
        self.provenance = {}
        self.forward_id = 0
        self.decoder_snapshot = None
        self.stages = []
        self.steps = []
        self.pending = None
        self.enabled = False
        self.in_flight = 0
        self.execute = runner.execute_model
        self.sample = runner.sample_tokens
        runner.execute_model = self.execute_model
        runner.sample_tokens = self.sample_tokens

    def execute_model(self, scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
        if not self.enabled:
            return self.execute(scheduler_output, *args, **kwargs)
        if self.pending is not None:
            raise RuntimeError("Previous benchmark step did not complete sampling")
        self.pending = {"start_ns": time.perf_counter_ns(), "stages": []}
        result = self.execute(scheduler_output, *args, **kwargs)
        if result is not None:
            self.finish(result)
        return result

    def sample_tokens(self, *args: Any, **kwargs: Any) -> Any:
        result = self.sample(*args, **kwargs)
        if self.enabled and self.pending is not None:
            self.finish(result)
        return result

    def finish(self, result: Any) -> None:
        from vllm.v1.outputs import AsyncModelRunnerOutput

        row = self.pending
        self.pending = None
        self.steps.append(row)
        self.in_flight += 1

        def observed(output: Any) -> None:
            if "observed_ns" in row:
                return  # The native async result may be read more than once.
            row.update(
                observed_ns=time.perf_counter_ns(),
                observation="native_output_consumption",
                accepted_output_tokens=(
                    sum(len(tokens) for tokens in output.sampled_token_ids)
                    if output is not None
                    else 0
                ),
                output_requests=len(output.req_ids) if output is not None else 0,
                output_request_ids=list(output.req_ids) if output is not None else [],
                accepted_output_counts=(
                    [len(tokens) for tokens in output.sampled_token_ids]
                    if output is not None
                    else []
                ),
            )
            self.in_flight -= 1

        if isinstance(result, AsyncModelRunnerOutput):
            # Read host bookkeeping only; never query/wait on a device event.
            copy = result.copy_state
            row["output_path"] = (
                "async_no_copy"
                if copy is None
                else "async_discard_without_wait"
                if (
                    not copy.spec_logprobs
                    and len(result._model_runner_output.req_ids) > 0
                    and len(result.discard_sampled_tokens_req_indices)
                    == len(result._model_runner_output.req_ids)
                )
                else "async_native_copy_wait"
            )
            original_get_output = result.get_output

            def get_output() -> Any:
                # Only the existing serving caller invokes this. Never force an
                # early copy/wait to obtain benchmark counters.
                output = original_get_output()
                observed(output)
                return output

            result.get_output = get_output
        else:
            row["output_path"] = "synchronous_return"
            observed(result)


def _gather_dp(value: dict) -> list[dict]:
    import torch.distributed
    from vllm.distributed import get_dp_group

    group = get_dp_group()
    if group.world_size != 8:
        raise ValueError("Expected eight node-local DP workers")
    gathered = [None] * group.world_size
    # vLLM's DP utility RPC executes on every engine but returns engine 0 only.
    # Gather on the workers so that the returned payload contains all eight ranks.
    torch.distributed.all_gather_object(gathered, value, group=group.cpu_group)
    return gathered


class ServerStepWorker:
    def benchmark_begin(self, scenario: str) -> list[dict]:
        global _ACTIVE, _INSTALLED
        from vllm_torchtpu.models.vllm.deepseek_v41.v2.model import (
            DeepseekV41DecoderStage,
            DeepseekV41EncoderStage,
        )

        config = self.vllm_config.parallel_config
        if (
            config.tensor_parallel_size,
            config.data_parallel_size,
            config.pipeline_parallel_size,
            config.enable_expert_parallel,
        ) != (1, 8, 1, True):
            raise ValueError("Benchmark requires TP1/DP8/EP8/PP1")
        spec = self.vllm_config.speculative_config
        if spec is None or spec.method != "dspark" or spec.num_speculative_tokens != 5:
            raise ValueError("Benchmark requires DSpark K=5")
        hf = self.vllm_config.model_config.hf_config
        if hf.num_hidden_layers != 40 or hf.sliding_window != 128:
            raise ValueError("Expected the 40-layer DSv4.1 model with SWA=128")
        selected = SCENARIOS[scenario]
        if self.vllm_config.scheduler_config.max_num_seqs != selected.local_concurrency:
            raise ValueError(
                "max_num_seqs must equal the scenario's per-DP concurrency"
            )
        if (
            self.vllm_config.scheduler_config.max_num_batched_tokens
            != selected.chunk_tokens
        ):
            raise ValueError("Token budget differs from the selected scenario")
        if not _INSTALLED:
            for cls, stage in [
                (DeepseekV41EncoderStage, "encoder"),
                (DeepseekV41DecoderStage, "decoder"),
            ]:
                cls.__call__ = _stage_wrapper(cls.__call__, stage)
            _INSTALLED = True
        if not hasattr(self, "_server_step_recorder"):
            self._server_step_recorder = Recorder(
                self.model_runner, selected, config.data_parallel_rank
            )
        recorder = self._server_step_recorder
        if recorder.enabled:
            raise RuntimeError("Benchmark already active")
        from vllm_torchtpu.models.vllm.deepseek_v41.v2 import model as model_module

        source = Path(model_module.__file__)
        recorder.provenance = {
            "model_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "model": self.vllm_config.model_config.model,
            "max_model_len": self.vllm_config.model_config.max_model_len,
        }
        recorder.scenario = selected
        recorder.forward_id = 0
        recorder.decoder_snapshot = None
        recorder.stages = []
        recorder.steps = []
        recorder.pending = None
        recorder.in_flight = 0
        recorder.enabled = True
        _ACTIVE = recorder
        return _gather_dp({"rank": recorder.rank, "scenario": selected.to_dict()})

    def benchmark_end(self) -> list[dict]:
        global _ACTIVE
        recorder = self._server_step_recorder
        if recorder.pending is not None or recorder.in_flight:
            raise RuntimeError("Serving has not naturally consumed all outputs")
        recorder.enabled = False
        _ACTIVE = None
        for step in recorder.steps:
            logging.getLogger(__name__).info(
                "DSV41_SERVER_STEP %s",
                json.dumps({"rank": recorder.rank, **step}),
            )
        return _gather_dp(
            {
                "rank": recorder.rank,
                "hostname": socket.gethostname(),
                "steps": recorder.steps,
                "stages": recorder.stages,
                "scenario": recorder.scenario.to_dict(),
                "provenance": recorder.provenance,
            }
        )
