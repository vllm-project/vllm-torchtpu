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
"""Static checks for pipeline parallelism on TPU."""

from typing import Any


def _legacy_ray_executor() -> bool:
    import vllm.envs as vllm_envs

    from vllm_torchtpu import envs

    return (
        envs.TPU_MULTIHOST_BACKEND == "ray"
        and not vllm_envs.VLLM_USE_RAY_V2_EXECUTOR_BACKEND
    )


def _torch_tpu_builds_stage_meshes() -> bool:
    """Whether the installed torch_tpu builds a Pallas mesh over a pipeline
    stage's process group; its releases accept world-size meshes only."""
    from torch_tpu._internal.pallas import pallas

    return not hasattr(pallas, "_validate_pallas_mesh_world_size")


# The `layer_types` spellings vLLM counts as full-attention layers
# (`_ATTENTION_LAYER_TYPES`,
# https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/config/model.py#L111).
# transformers v5.18 renamed deepseek_sparse_attention and
# qwen_sparse_attention to indexed_attention
# (https://github.com/huggingface/transformers/pull/48974); vLLM keeps all
# three spellings, and so does this list.
_FULL_ATTENTION_LAYER_TYPES = [
    "full_attention",
    "indexed_attention",
    "deepseek_sparse_attention",
    "qwen_sparse_attention",
]

# KV transfer connectors that register each stage's own layers: the
# disaggregation connector with the stage's peer, the offloading connector
# with the offload controller.
_RAIDEN_CONNECTORS = ("TPURaidenConnector", "TPURaidenOffloadingConnector")


def _kv_connector_names(kv_transfer_config: Any) -> list[str]:
    """The connector a config names, or the children of a TPUMultiConnector."""
    name = kv_transfer_config.kv_connector
    if name != "TPUMultiConnector":
        return [name]
    extra = getattr(kv_transfer_config, "kv_connector_extra_config", None) or {}
    return [child["kv_connector"] for child in extra.get("connectors", [])]


def _non_raiden_connectors(kv_transfer_config: Any) -> list[str]:
    return [
        name
        for name in _kv_connector_names(kv_transfer_config)
        if name not in _RAIDEN_CONNECTORS
    ]


def _stage_without_full_attention(vllm_config: Any) -> str | None:
    """Describe the first stage holding no full-attention layer, or None.

    Only hybrid models declare ``layer_types``; every layer of the others
    has a KV cache.
    """
    from vllm.distributed.utils import get_pp_indices

    hf_config = vllm_config.model_config.hf_text_config
    layer_types = getattr(hf_config, "layer_types", None)
    if layer_types is None:
        return None
    stages = vllm_config.parallel_config.pipeline_parallel_size
    for stage in range(stages):
        start, end = get_pp_indices(len(layer_types), stage, stages)
        kinds = layer_types[start:end]
        if not any(kind in _FULL_ATTENTION_LAYER_TYPES for kind in kinds):
            return (
                f"stage {stage} holds layers {start}-{end - 1}, "
                f"{'/'.join(sorted(set(kinds)))} only"
            )
    return None


def validate_pipeline_parallel_config(vllm_config: Any) -> None:
    """Reject configurations the TPU pipeline-parallel path does not serve.

    Raises NotImplementedError naming the unsupported feature, and
    ValueError for a ring hand-off depth below one.
    """
    parallel_config = vllm_config.parallel_config
    if parallel_config.pipeline_parallel_size <= 1:
        return

    unsupported = None
    remedy = ""
    if parallel_config.tensor_parallel_size > 1:
        # On torch_tpu (dev20260813) a stage's TP collectives stall on the
        # first serving step, when the other stages are not launching the
        # same collectives; warmup, where every stage runs them, passes.
        unsupported = "tensor parallelism inside a stage"
        remedy = " Use --tensor-parallel-size 1."
    elif (
        parallel_config.prefill_context_parallel_size > 1
        or (
            parallel_config.data_parallel_size > 1
            and parallel_config.enable_expert_parallel
        )
    ) and not _torch_tpu_builds_stage_meshes():
        # A stage of several devices builds its meshes, and the hand-off its
        # per-column mesh, over a group smaller than the world.
        # TODO(yinlin): google-pytorch/torch_tpu#4387 builds those for a group
        # made standalone; make each stage group standalone and drop this
        # guard once the pinned torch_tpu has it.
        unsupported = (
            "pipeline stages of several devices (context parallelism, or expert "
            "parallelism across data-parallel ranks) with this torch_tpu"
        )
        remedy = (
            " They need a torch_tpu that builds Pallas meshes over a stage's"
            " process group."
        )
    elif (
        parallel_config.prefill_context_parallel_size > 1
        and vllm_config.kv_transfer_config is not None
        and _non_raiden_connectors(vllm_config.kv_transfer_config)
    ):
        # These connectors pair a stage with its peer through one pool per
        # device; a context-parallel stage spreads a request over several.
        # The raiden connectors lower each PCP worker's chunks of the
        # stage's layers.
        unsupported = "KV transfer connectors with context-parallel stages"
    elif vllm_config.scheduler_config.async_scheduling:
        # Stages before the last never see the sampled tokens the async
        # placeholder substitution reads from the device.
        unsupported = "async scheduling"
        remedy = " Pass --no-async-scheduling."
    elif (spec := vllm_config.speculative_config) is not None and (
        spec.method != "mtp"
        or spec.num_speculative_tokens != 1
        or vllm_config.kv_transfer_config is None
        or not vllm_config.kv_transfer_config.is_kv_producer
    ):
        # The last stage hosts the draft and runs it over each prefill chunk
        # so the draft layer's KV reaches a decode peer; it never verifies,
        # and the runner refuses any decode step. That is the PCP producer's
        # support boundary too.
        unsupported = (
            "speculative decoding other than prefill-only MTP with "
            "num_speculative_tokens=1 on a KV producer"
        )
        remedy = (
            " Use method=mtp with num_speculative_tokens=1 on an engine whose "
            "kv_transfer_config has kv_role kv_producer."
        )
    elif vllm_config.kv_transfer_config is not None and (
        others := _non_raiden_connectors(vllm_config.kv_transfer_config)
    ):
        # The raiden connectors transfer each stage's own layers; the
        # other connectors assume every worker holds every layer.
        unsupported = (
            "KV transfer connectors other than TPURaidenConnector and "
            f"TPURaidenOffloadingConnector ({', '.join(others)})"
        )
    elif vllm_config.kv_transfer_config is not None and (
        stage := _stage_without_full_attention(vllm_config)
    ):
        # The raiden connectors register a stage's pools through the
        # measured layout of a full-attention KV pool, so a stage made of
        # linear-attention layers alone has nothing to register.
        unsupported = (
            f"a raiden connector with a stage holding no full-attention layer ({stage})"
        )
        remedy = (
            " Choose a pipeline size or VLLM_PP_LAYER_PARTITION that "
            "gives every stage a full-attention layer."
        )
    elif _legacy_ray_executor():
        # The pipeline's hand-off pushes go through the executor hook that
        # the multiprocess executor and the Ray V2 executor share; the
        # legacy Ray executor has no such hook.
        unsupported = "the legacy Ray executor"
        remedy = " Set VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1."
    if unsupported is not None:
        raise NotImplementedError(
            f"Pipeline parallelism on TPU does not support {unsupported} yet.{remedy}"
        )
    from vllm_torchtpu import envs
    from vllm_torchtpu.distributed.pp_push import ring_handoff

    if ring_handoff() and envs.TPU_PP_RING_DEPTH < 1:
        # A stage queues at most this many forwards for the next one; with
        # none it could hand nothing on.
        raise ValueError(
            f"TPU_PP_RING_DEPTH must be at least 1, got {envs.TPU_PP_RING_DEPTH}."
        )
