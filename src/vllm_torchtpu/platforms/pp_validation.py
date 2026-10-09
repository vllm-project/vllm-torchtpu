# SPDX-License-Identifier: Apache-2.0
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


_FULL_ATTENTION_LAYER_TYPES = ["full_attention", "deepseek_sparse_attention"]


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

    Raises NotImplementedError naming the unsupported feature.
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
        and vllm_config.kv_transfer_config.kv_connector != "TPURaidenConnector"
    ):
        # These connectors pair a stage with its peer through one pool per
        # device; a context-parallel stage spreads a request over several.
        # The raiden connector lowers each PCP worker's chunks of the
        # stage's layers.
        unsupported = "KV transfer connectors with context-parallel stages"
    elif vllm_config.scheduler_config.async_scheduling:
        # Stages before the last never see the sampled tokens the async
        # placeholder substitution reads from the device.
        unsupported = "async scheduling"
        remedy = " Pass --no-async-scheduling."
    elif vllm_config.speculative_config is not None:
        unsupported = "speculative decoding"
    elif (
        vllm_config.kv_transfer_config is not None
        and vllm_config.kv_transfer_config.kv_connector != "TPURaidenConnector"
    ):
        # The raiden connector transfers each stage's own layers; the
        # other connectors assume every worker holds every layer.
        unsupported = "KV transfer connectors other than TPURaidenConnector"
    elif vllm_config.kv_transfer_config is not None and (
        stage := _stage_without_full_attention(vllm_config)
    ):
        # The raiden connector pairs a stage with its peer through the
        # measured layout of a full-attention KV pool, so a stage made of
        # linear-attention layers alone has nothing to register.
        unsupported = (
            f"TPURaidenConnector with a stage holding no full-attention layer ({stage})"
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
