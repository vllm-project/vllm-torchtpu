# SPDX-License-Identifier: Apache-2.0
"""TpuMultiprocExecutor: MultiprocExecutor with per-shape KV offload prewarm.

Splits H2D Pallas kernel prewarm into one collective_rpc call per power-of-2
shape so each shape gets its own 60-second shm_broadcast window (~4s per shape)
instead of all 14 shapes sharing a single window and timing out.

It also sends new requests' prompts to the workers as one buffer
(`_pack_new_request_prompts`).
"""

from array import array
from importlib.metadata import entry_points
from typing import cast

from vllm.config import VllmConfig
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.sample.logits_processor import LOGITSPROCS_GROUP

from vllm_torchtpu.executors.kv_block_override import reconcile_num_gpu_blocks_override
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def _pack_new_request_prompts(scheduler_output: SchedulerOutput) -> None:
    """Turn each new request's `prompt_token_ids` list into an `array("i")`.

    The list is the whole prompt, prefix-cached part included: the broadcast
    pickles it int by int, every worker unpickles it into new int objects, and
    `InputBatch.add_request` converts it back element by element -- milliseconds
    for a 100k-token prompt, on the step that first schedules the request. An
    `array("i")` pickles as one buffer and copies into `token_ids_cpu` as one
    block; indexing it still yields ints. The scheduler's own consumers (KV
    connector metadata) ran in `schedule()`, before this, on the list.
    """
    for new_req in scheduler_output.scheduled_new_reqs:
        if isinstance(new_req.prompt_token_ids, list):
            # Deliberately not the declared list[int]: the same ints, sent as
            # one buffer to consumers `_may_pack_prompts` vetted.
            new_req.prompt_token_ids = cast(
                list[int], array("i", new_req.prompt_token_ids)
            )


def _may_pack_prompts(vllm_config: VllmConfig) -> bool:
    """Whether no worker-side consumer can rely on the prompt being a list.

    The worker hands prompts to logits processors, configured or installed as
    plugins, and to the thinking-budget state (`BatchUpdate.added`), which may
    use list semantics: a slice of an array compared with a list is never
    equal, and array + list raises.
    """
    return (
        vllm_config.reasoning_config is None
        and not vllm_config.model_config.logits_processors
        and not entry_points(group=LOGITSPROCS_GROUP)
    )


class TpuMultiprocExecutor(MultiprocExecutor):
    def _init_executor(self) -> None:
        # EngineCore does not run platform patch setup before spawning workers.
        from vllm_torchtpu import _patch_multiproc_worker_global_rank_env

        _patch_multiproc_worker_global_rank_env()
        super()._init_executor()
        self._pack_prompts = _may_pack_prompts(self.vllm_config)
        logger.info(
            "New requests' prompts go to the workers as %s.",
            "array('i')" if self._pack_prompts else "lists",
        )

    def execute_model(self, scheduler_output: SchedulerOutput, non_block: bool = False):
        if self._pack_prompts:
            _pack_new_request_prompts(scheduler_output)
        return super().execute_model(scheduler_output, non_block)

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
