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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Scenario:
    name: str
    concurrency: int
    prefix_tokens: int = 98304
    extend_tokens: int = 0
    output_tokens: int = 1
    dp_size: int = 8
    chunk_tokens: int = 2048
    dspark_tokens: int = 5
    physical_chips: int = 4

    @property
    def local_concurrency(self) -> int:
        return self.concurrency // self.dp_size

    def to_dict(self) -> dict:
        return asdict(self)


SCENARIO = Scenario("encoder_extend", 8, extend_tokens=16384)
DECODE_SCENARIO = Scenario(
    "decode", 384, prefix_tokens=98304, output_tokens=512, chunk_tokens=2048
)
SCENARIOS = {s.name: s for s in (SCENARIO, DECODE_SCENARIO)}


def classify_rows(starts: list[int], ends: list[int], prompts: list[int]) -> str:
    if not starts:
        return "idle"
    if all(start >= prompt for start, prompt in zip(starts, prompts, strict=True)):
        return "decode"
    if all(end <= prompt for end, prompt in zip(ends, prompts, strict=True)):
        return "prefill"
    return "mixed"


def token_counts(starts: list[int], ends: list[int], prefix: int) -> dict:
    return {
        "computed_tokens": sum(b - a for a, b in zip(starts, ends, strict=True)),
        "new_tokens": sum(
            max(0, b - max(a, prefix)) for a, b in zip(starts, ends, strict=True)
        ),
        "replay_tokens": sum(
            max(0, min(b, prefix) - a) for a, b in zip(starts, ends, strict=True)
        ),
    }


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
        if scenario.name == "decode":
            drafter = runner.drafter
            prepare = drafter._prepare_dflash_inputs

            def prepare_draft(chunk: Any, *args: Any, **kwargs: Any) -> Any:
                started = time.perf_counter_ns()
                result = prepare(chunk, *args, **kwargs)
                if self.enabled and self.pending is not None:
                    row = {
                        "stage": "draft_inputs",
                        "forward_id": self.forward_id,
                        "requests": chunk.num_reqs,
                        "rows": chunk.num_reqs * drafter.block_size,
                        "padded_rows": result[0].shape[0],
                        "input_shape": list(result[0].shape),
                        "positions_shape": list(result[1].shape),
                        "dispatch_ns": started,
                        "host_return_ns": time.perf_counter_ns(),
                    }
                    self.pending["stages"].append(row)
                    self.stages.append(row)
                return result

            drafter._prepare_dflash_inputs = prepare_draft

    def execute_model(self, scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
        if not self.enabled:
            return self.execute(scheduler_output, *args, **kwargs)
        if self.pending is not None:
            raise RuntimeError("Previous benchmark step did not complete sampling")
        self.pending = {"start_ns": time.perf_counter_ns(), "stages": []}
        if self.scenario.name == "decode":
            self.pending["scheduled_draft_lengths"] = {
                req: len(tokens)
                for req, tokens in scheduler_output.scheduled_spec_decode_tokens.items()
            }
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
    def synchronized_profile_arm(self, output_dir: str):
        import torch.distributed as dist
        from vllm.distributed import get_dp_group

        from vllm_torchtpu.utils import synchronize_device

        runner = self.model_runner
        group = get_dp_group()
        rank = self.vllm_config.parallel_config.data_parallel_rank
        assert group.world_size == 8
        assert not hasattr(self, "_sync_capture")
        scenario = DECODE_SCENARIO
        original = runner._dp_coordinated_step
        state = {
            "rank": rank,
            "ticks": 0,
            "captured_steps": 0,
            "active": False,
            "completed": False,
            "rows": [],
            "profile_rank": rank,
        }
        self._sync_capture = state
        root = Path(output_dir) / "synchronized-profile"
        root.mkdir(exist_ok=True)

        def boundary(*args, **kwargs):
            result = original(*args, **kwargs)
            if state["completed"]:
                return result
            state["ticks"] += 1
            tokens = int(args[0] if args else kwargs["num_tokens"])
            n = runner.input_batch.num_reqs
            contexts = [int(v) for v in runner.input_batch.num_computed_tokens_cpu[:n]]
            ready = (
                tokens == scenario.local_concurrency * (scenario.dspark_tokens + 1)
                and n == scenario.local_concurrency
                and min(contexts, default=0) >= scenario.prefix_tokens + 44
            )
            row = {
                "tick": state["ticks"],
                "tokens": tokens,
                "requests": n,
                "min_context": min(contexts, default=0),
                "wall_ns": time.time_ns(),
            }
            if state["active"] and state["captured_steps"] == 6:
                synchronize_device()
                dist.barrier(group=group.cpu_group)
                state["stop_boundary_ns"] = time.time_ns()
                self.profile(is_start=False)
                # No rank may dispatch subsequent compute during trace export.
                dist.barrier(group=group.cpu_group)
                state["export_done_ns"] = time.time_ns()
                state["active"] = False
                state["completed"] = True
                runner._dp_coordinated_step = original
                (root / f"rank-{rank}-capture.json").write_text(
                    json.dumps(state, indent=2)
                )
                return result
            if not state["active"]:
                gathered = [None] * 8
                dist.all_gather_object(
                    gathered,
                    {"rank": rank, "ready": ready, **row},
                    group=group.cpu_group,
                )
                if not all(r["ready"] for r in gathered):
                    if state["ticks"] > 150:
                        raise RuntimeError("No common full-DP decode window found")
                    return result
                assert len({r["tick"] for r in gathered}) == 1, gathered
                synchronize_device()
                dist.barrier(group=group.cpu_group)
                self.profile(
                    is_start=True,
                    profile_prefix="steady-decode",
                    profiler_kwargs={
                        "host_tracer_level": 1,
                        "python_tracer_level": 0,
                        "device_tracer_level": 1,
                    },
                )
                dist.barrier(group=group.cpu_group)
                state["start_boundary_ns"] = time.time_ns()
                state["start_ranks"] = gathered
                state["active"] = True
            assert ready, row
            state["rows"].append(row)
            state["captured_steps"] += 1
            return result

        runner._dp_coordinated_step = boundary
        return {"rank": rank, "armed": True}

    def synchronized_profile_check(self):
        state = self._sync_capture
        assert state["completed"] and state["captured_steps"] == 6, state
        return _gather_dp(state)

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
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}")
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
