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

It also broadcasts a new request's prompt token ids as an int32 array (see
`_compact_new_prompts`).
"""

import dataclasses
from array import array

import numpy as np
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec

from vllm_torchtpu.executors.kv_block_override import reconcile_num_gpu_blocks_override
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


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
            self._compact_new_prompts(scheduler_output), *args, **kwargs
        )

    def _compact_new_prompts(
        self, scheduler_output: SchedulerOutput
    ) -> SchedulerOutput:
        """Copy of `scheduler_output` whose new requests carry their prompt
        token ids as int32 numpy arrays instead of lists.

        A list of N ids costs O(N) Python work three times on the path from
        scheduling a request to the forward that prefills it: pickling it into
        the broadcast here, unpickling one int object per id in every worker,
        and converting it element by element into the worker's input-batch
        row. An array is copied as raw bytes at each of those steps. Every id
        of the prompt travels, including the prefix-cache hit, so an agentic
        prompt of 10^5 tokens pays this in full on every turn.

        The engine core keeps its own `scheduler_output`, so scheduler-side
        consumers (KV connectors, `update_from_output`) still see lists. On the
        workers the arrays reach `InputBatch.add_request` and the cached
        request state. Configs and requests whose ids are read as lists
        elsewhere on the worker keep their lists; the readers checked against
        vLLM 0.32 are M-RoPE position init, prompt logprobs, pooling, logits
        processors (`BatchUpdate.added`) and the thinking-budget state.
        """
        if not scheduler_output.scheduled_new_reqs or not self._can_compact_prompts():
            return scheduler_output
        return dataclasses.replace(
            scheduler_output,
            scheduled_new_reqs=[
                _with_array_prompt(req) for req in scheduler_output.scheduled_new_reqs
            ],
        )

    def _can_compact_prompts(self) -> bool:
        model_config = self.vllm_config.model_config
        return not (
            model_config.uses_mrope
            or model_config.runner_type == "pooling"
            or model_config.logits_processors
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


def _with_array_prompt(req: NewRequestData) -> NewRequestData:
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
