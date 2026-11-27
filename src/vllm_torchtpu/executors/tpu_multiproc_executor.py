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
"""TpuMultiprocExecutor: MultiprocExecutor with per-shape KV offload prewarm.

Splits H2D Pallas kernel prewarm into one collective_rpc call per power-of-2
shape so each shape gets its own 60-second shm_broadcast window (~4s per shape)
instead of all 14 shapes sharing a single window and timing out.

It also broadcasts the workers a copy of each SchedulerOutput without the
O(context) token-id lists they do not need as lists (see `_worker_copy`).
"""

import dataclasses
import functools
from array import array
from importlib.metadata import entry_points

import numpy as np
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.sample.logits_processor import LOGITSPROCS_GROUP

from vllm_torchtpu.executors.kv_block_override import reconcile_num_gpu_blocks_override
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# `_with_array_prompt` views an array("i") as int32.
assert array("i").itemsize == np.dtype(np.int32).itemsize


class TpuMultiprocExecutor(MultiprocExecutor):
    def _init_executor(self) -> None:
        # EngineCore does not run platform patch setup before spawning workers.
        from vllm_torchtpu import _patch_multiproc_worker_global_rank_env

        _patch_multiproc_worker_global_rank_env()
        super()._init_executor()
        logger.info(
            "New-request prompt token ids broadcast as int32 arrays: %s",
            self._can_compact_prompts(),
        )

    def execute_model(self, scheduler_output: SchedulerOutput, *args, **kwargs):
        return super().execute_model(
            self._worker_copy(scheduler_output), *args, **kwargs
        )

    def _worker_copy(self, scheduler_output: SchedulerOutput) -> SchedulerOutput:
        """Copy of `scheduler_output` without the O(context) token-id lists the
        workers do not need as lists.

        Token-id lists cost O(N) Python work on the path from scheduling to the
        forward: pickling into the broadcast here, unpickling one int object
        per id in every worker, and for prompts converting element by element
        into the worker's input-batch row. The engine core keeps its own
        `scheduler_output`, so scheduler-side consumers (KV connectors,
        `update_from_output`) still see the full lists.
        """
        new_reqs = scheduler_output.scheduled_new_reqs
        if new_reqs and self._can_compact_prompts():
            new_reqs = [_with_array_prompt(req) for req in new_reqs]
        cached = scheduler_output.scheduled_cached_reqs
        if cached.all_token_ids:
            cached = _with_output_tails(cached)
        if (
            new_reqs is scheduler_output.scheduled_new_reqs
            and cached is scheduler_output.scheduled_cached_reqs
        ):
            return scheduler_output
        return dataclasses.replace(
            scheduler_output, scheduled_new_reqs=new_reqs, scheduled_cached_reqs=cached
        )

    def _can_compact_prompts(self) -> bool:
        model_config = self.vllm_config.model_config
        return not (
            model_config.uses_mrope
            or model_config.runner_type == "pooling"
            or model_config.logits_processors
            or _logits_processor_plugins_installed()
        )

    def get_kv_cache_specs(self) -> list[dict[str, KVCacheSpec]]:
        specs = super().get_kv_cache_specs()

        if self.vllm_config.cache_config.num_gpu_blocks_override is None:
            # Compact-mamba sizing sets `num_gpu_blocks_override` on the worker's
            # cache_config during the RPC above; workers are separate processes, so
            # copy it to the engine-side config here.
            pc = self.vllm_config.parallel_config
            agreed = reconcile_num_gpu_blocks_override(
                self.collective_rpc("get_num_gpu_blocks_override"),
                workers_per_stage=pc.world_size // pc.pipeline_parallel_size,
            )
            if agreed is not None:
                self.vllm_config.cache_config.num_gpu_blocks_override = agreed

        return specs

    def initialize_from_config(self, kv_cache_configs: list[KVCacheConfig]) -> None:
        super().initialize_from_config(kv_cache_configs)
        # Per-shape H2D prewarm: each shape is its own collective_rpc call with
        # its own 60-second shm_broadcast window.  Workers that have no KV
        # offload spec return [] from get_kv_prewarm_shapes() — a no-op.
        shapes = self.collective_rpc("get_kv_prewarm_shapes")
        if not shapes or not shapes[0]:
            return
        for p in shapes[0]:
            self.collective_rpc("prewarm_kv_offload_shape", args=(p,))


@functools.cache
def _logits_processor_plugins_installed() -> bool:
    # vLLM loads these in every engine, on top of model_config.logits_processors.
    return bool(entry_points(group=LOGITSPROCS_GROUP))


def _with_array_prompt(req: NewRequestData) -> NewRequestData:
    """A new request's prompt ids as an int32 array, which the worker copies as
    raw bytes into its input batch. Requests whose ids are read as lists
    elsewhere on the worker keep their lists; the readers checked against vLLM
    0.32 are M-RoPE position init, prompt logprobs, pooling, logits processors
    (`BatchUpdate.added`) and the thinking-budget state.
    """
    ids = req.prompt_token_ids
    params = req.sampling_params
    if (
        not isinstance(ids, list)
        or req.prompt_embeds is not None
        or req.prompt_is_token_ids is not None
        or (
            params is not None
            and (
                params.prompt_logprobs is not None
                or params.thinking_token_budget is not None
            )
        )
    ):
        return req
    # The field is annotated list[int]; on the workers it holds this array.
    # array("i") converts a list about twice as fast as np.array; the view is free.
    return dataclasses.replace(
        req, prompt_token_ids=np.frombuffer(array("i", ids), dtype=np.int32)
    )


def _with_output_tails(cached: CachedRequestData) -> CachedRequestData:
    """`all_token_ids` cut to what the worker reads of it.

    The scheduler sends every token of a running request that was not scheduled
    in the previous step, such as a long prefill held back for decode steps.
    The worker only re-reads the last `num_output_tokens` of them, to restore
    the request's outputs when it re-adds it to the batch
    (`GPUModelRunner._update_states`, the only worker-side reader in vLLM
    0.32); its prompt is already cached there.
    """
    num_output = dict(zip(cached.req_ids, cached.num_output_tokens))
    tails = {
        req_id: ids[-n:] if (n := num_output.get(req_id, 0)) > 0 else []
        for req_id, ids in cached.all_token_ids.items()
    }
    return dataclasses.replace(cached, all_token_ids=tails)
