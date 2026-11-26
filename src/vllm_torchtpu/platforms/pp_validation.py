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


def _skips_topk(hf_config: Any, layer: int) -> bool:
    """Whether `layer` reuses an earlier layer's top-k (IndexCache), by the
    rule vLLM v0.29.0's DeepseekV2 attention builds its layers with."""
    pattern = getattr(hf_config, "index_topk_pattern", None)
    if pattern is not None:
        return 0 <= layer < len(pattern) and pattern[layer] == "S"
    freq = getattr(hf_config, "index_topk_freq", 1)
    offset = getattr(hf_config, "index_skip_topk_offset", 2)
    return max(layer - offset + 1, 0) % freq != 0


def _stage_starting_on_shared_topk(vllm_config: Any) -> str | None:
    """Describe the first stage whose first layer reuses the top-k of a layer
    on the stage before it, or None."""
    from vllm.distributed.utils import get_pp_indices

    hf_config = vllm_config.model_config.hf_text_config
    if not hasattr(hf_config, "index_topk"):
        return None
    num_layers = hf_config.num_hidden_layers
    stages = vllm_config.parallel_config.pipeline_parallel_size
    for stage in range(1, stages):
        start, _ = get_pp_indices(num_layers, stage, stages)
        if _skips_topk(hf_config, start):
            return f"stage {stage} starts at layer {start}"
    return None


def _speculative_refusal(vllm_config: Any) -> tuple[str, str] | None:
    """Why the pipeline refuses this speculative config, or None.

    One case is served: a KV producer running MTP only for its drafter's KV,
    opted into with TPU_PP_MTP_PREFILL_KV. The drafter lives on the last stage
    and never proposes, so no step verifies drafts and no draft token crosses
    a stage. kv_both counts as a producer, and such a server may still decode
    a request itself; it decodes without speculation.
    """
    from vllm_torchtpu import envs

    spec = vllm_config.speculative_config
    transfer = vllm_config.kv_transfer_config
    if not envs.TPU_PP_MTP_PREFILL_KV:
        reason = "speculative decoding"
    elif spec.method != "mtp":
        reason = f"speculative decoding with method {spec.method!r}"
    elif spec.num_speculative_tokens != 1:
        # Drafts are discarded, so steps past the first write KV for nothing.
        reason = "MTP for prefill KV with num_speculative_tokens != 1"
    elif transfer is None or not transfer.is_kv_producer:
        # Without a producer connector nothing reads the drafter's KV.
        reason = "MTP for prefill KV without a KV producer connector"
    else:
        return None
    return reason, (
        " Under a pipeline only MTP that writes the drafter's prefill KV is "
        "served: set TPU_PP_MTP_PREFILL_KV=1 with method mtp, "
        "num_speculative_tokens 1 and a kv_producer or kv_both connector."
    )


def pp_mtp_prefill_kv_only(vllm_config: Any) -> bool:
    """Whether this config runs the drafter for its KV only (validated above)."""
    return (
        vllm_config.parallel_config.pipeline_parallel_size > 1
        and vllm_config.speculative_config is not None
    )


def validate_pipeline_parallel_config(vllm_config: Any) -> None:
    """Reject configurations the TPU pipeline-parallel path does not serve.

    Raises NotImplementedError naming the unsupported feature.
    """
    parallel_config = vllm_config.parallel_config
    if parallel_config.pipeline_parallel_size <= 1:
        return

    unsupported = None
    remedy = ""
    if (
        parallel_config.tensor_parallel_size > 1
        and not _torch_tpu_builds_stage_meshes()
    ):
        # On torch_tpu builds with world-size meshes only (dev20260813), a
        # stage's TP collectives stall on the first serving step, when the
        # other stages are not launching the same collectives. A torch_tpu that
        # builds meshes over a stage's group (0.3.0) runs them.
        unsupported = "tensor parallelism inside a stage with this torch_tpu"
        remedy = (
            " Use --tensor-parallel-size 1, or a torch_tpu that builds Pallas"
            " meshes over a stage's process group."
        )
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
    elif vllm_config.speculative_config is not None and (
        refusal := _speculative_refusal(vllm_config)
    ):
        unsupported, remedy = refusal
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
    elif stage := _stage_starting_on_shared_topk(vllm_config):
        # A layer that skips its indexer reads the top-k buffer an earlier
        # layer of the same stage wrote this step; on a stage's first layer
        # that buffer is stale.
        unsupported = f"a stage starting on a layer that reuses the top-k ({stage})"
        remedy = (
            " Choose a VLLM_PP_LAYER_PARTITION that starts every stage on a layer"
            " with its own indexer."
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
