# SPDX-License-Identifier: Apache-2.0
"""TpuMultiprocExecutor: MultiprocExecutor with per-shape KV offload prewarm.

Splits H2D Pallas kernel prewarm into one collective_rpc call per power-of-2
shape so each shape gets its own 60-second shm_broadcast window (~4s per shape)
instead of all 14 shapes sharing a single window and timing out.

It also sends new requests' prompts to the workers as one buffer
(`_pack_new_request_prompts`), packed in the engine core's input thread
(`install_prompt_prepack`).
"""

import weakref
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

# Tokens converted per call when packing a prompt, so the input thread yields
# the GIL to the engine loop between slices.
_PACK_SLICE = 16384

# Request id -> its prompt packed by `install_prompt_prepack`. The vLLM Request
# holds the array, so an entry lives no longer than its request. Written by the
# engine core's input thread, read by its busy loop. None while prompts are sent
# as lists.
_prepacked: "weakref.WeakValueDictionary[str, array] | None" = None


def _pack(tokens: list[int]) -> array:
    packed = array("i")
    for start in range(0, len(tokens), _PACK_SLICE):
        packed.fromlist(tokens[start : start + _PACK_SLICE])
    return packed


def install_prompt_prepack() -> None:
    """Pack each new request's prompt in the engine core's input thread.

    Packing a whole prompt (prefix-cached part included) costs milliseconds for
    a 100k-token prompt. Done in `execute_model`, it runs on the engine loop
    between scheduling the request and sending the step to the workers, which
    then start that step late; an idle DP rank's late start delays every rank.
    `preprocess_add_request` already runs in the input thread, ahead of the
    loop, for this kind of per-request work; it has no extension point, so it
    is wrapped here. Inert unless `TpuMultiprocExecutor` packs prompts. The
    packed copy (4 bytes per token) stays on the Request until it is freed.
    """
    from vllm.v1.engine.core import EngineCore

    original = EngineCore.preprocess_add_request
    if getattr(original, "_tpu_prompt_prepack", False):
        return

    def preprocess_add_request(self, request):
        req, wave = original(self, request)
        # A resumable (streaming-input) session reuses its request id: each
        # later ADD carries only its chunk, and the scheduler extends the
        # session's prompt. Its prompt is packed when it is scheduled.
        if (
            _prepacked is not None
            and not req.resumable
            and isinstance(req.prompt_token_ids, list)
        ):
            req._tpu_packed_prompt = _pack(req.prompt_token_ids)
            _prepacked[req.request_id] = req._tpu_packed_prompt
        return req, wave

    preprocess_add_request._tpu_prompt_prepack = True  # type: ignore[attr-defined]
    EngineCore.preprocess_add_request = preprocess_add_request  # type: ignore[method-assign]


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
    prepacked = _prepacked if _prepacked is not None else {}
    for new_req in scheduler_output.scheduled_new_reqs:
        packed = prepacked.pop(new_req.req_id, None)
        if not isinstance(new_req.prompt_token_ids, list):
            continue
        prompt = new_req.prompt_token_ids
        if (
            packed is None
            or len(packed) != len(prompt)
            or (prompt and (packed[0] != prompt[0] or packed[-1] != prompt[-1]))
        ):
            # Not packed ahead (a resumable session, the in-process client
            # without engine_core patches) or the prompt changed after it was
            # packed (e.g. a KV connector truncated it): pack what is sent.
            packed = _pack(prompt)
        # Deliberately not the declared list[int]: the same ints, sent as one
        # buffer to consumers `_may_pack_prompts` vetted.
        new_req.prompt_token_ids = cast(list[int], packed)


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
        if self._pack_prompts:
            global _prepacked
            _prepacked = weakref.WeakValueDictionary()
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
