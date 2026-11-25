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

import os
from typing import Any

from vllm_torchtpu.distributed.kv_transfer.connector_names import kv_connector_names
from vllm_torchtpu.distributed.pipeline_stage import stage_size


def _torch_tpu_builds_stage_meshes() -> bool:
    """Whether the installed torch_tpu compiles a process group's collectives
    and Pallas meshes for the group's own devices (0.3.0 and later); earlier
    releases compile collectives over the world."""
    from importlib import metadata as importlib_metadata

    from packaging.version import Version

    try:
        installed = importlib_metadata.version("torch-tpu")
    except importlib_metadata.PackageNotFoundError:
        return False
    return Version(installed).release >= (0, 3)


def _stage_devices(parallel_config: Any) -> int:
    """The devices of one pipeline stage: its workers in one engine, and in
    every data-parallel engine when expert parallelism joins the engines."""
    engines = (
        parallel_config.data_parallel_size
        if parallel_config.enable_expert_parallel
        else 1
    )
    return stage_size(parallel_config) * engines


def _stage_mixes_groups(parallel_config: Any) -> bool:
    """Whether an axis inside a stage forms process groups smaller than the
    stage, so that a stage's program combines groups of different devices."""
    sizes = (
        parallel_config.tensor_parallel_size,
        parallel_config.prefill_context_parallel_size,
    )
    return any(1 < size < _stage_devices(parallel_config) for size in sizes)


# The `layer_types` spellings vLLM counts as full-attention layers
# (`_ATTENTION_LAYER_TYPES`,
# https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/config/model.py#L112).
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


def _non_raiden_connectors(kv_transfer_config: Any) -> list[str]:
    return [
        name
        for name in kv_connector_names(kv_transfer_config, ("TPUMultiConnector",))
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


# Pipeline env vars that no longer exist, with what to do instead. Set, they
# would otherwise be ignored and the pipeline would run differently from what
# the deployment asked for.
_REMOVED_PIPELINE_ENVS = {
    "TPU_PP_HANDOFF": (
        "the buffered point-to-point hand-off is the only one. Unset it."
    ),
    "TPU_PP_RING_DEPTH": "set TPU_PP_HANDOFF_DEPTH instead.",
    "TPU_PP_LAZY_PUSH": (
        "it timed the pushes of the removed lockstep hand-off. Unset it."
    ),
    "TPU_PP_DYNAMIC_CHUNKS": (
        "set TPU_PP_LONE_PROMPT_CHUNKS and TPU_PP_EQUAL_COST_CHUNKS instead."
    ),
}


def validate_pipeline_parallel_config(vllm_config: Any) -> None:
    """Reject configurations the TPU pipeline-parallel path does not serve.

    Raises NotImplementedError naming the unsupported feature, and
    ValueError for a removed pipeline env var that is still set or a
    buffered hand-off depth below one or deeper than a device's receive
    flags allow.
    """
    parallel_config = vllm_config.parallel_config
    if parallel_config.pipeline_parallel_size <= 1:
        return
    for name, remedy in _REMOVED_PIPELINE_ENVS.items():
        if name in os.environ:
            raise ValueError(f"{name} has been removed; {remedy}")

    from vllm_torchtpu import envs

    unsupported = None
    remedy = ""
    if (parallel_config.decode_context_parallel_size or 1) > 1:
        unsupported = "decode context parallelism"
        remedy = " Pass --decode-context-parallel-size 1."
    elif envs.MOE_FUSED_EP_SORTED_GATHER:
        # ep_sorted_gather_group() forms its process group lazily over one
        # stage's EP ranks, while torch.distributed.new_group() requires every
        # rank in the world to pass the same rank list in the same order.
        unsupported = "MOE_FUSED_EP_SORTED_GATHER"
        remedy = " Unset MOE_FUSED_EP_SORTED_GATHER or set it to 0."
    elif _stage_devices(parallel_config) > 1 and not _torch_tpu_builds_stage_meshes():
        # A stage of several devices runs its collectives and builds its
        # meshes over process groups smaller than the world, and launches
        # them while the other stages launch theirs.
        # TODO(yinlin): drop this guard once the pinned torch_tpu is 0.3.0+.
        unsupported = "pipeline stages of several devices with this torch_tpu"
        remedy = " Upgrade to torch_tpu 0.3.0 or later."
    elif _stage_mixes_groups(parallel_config):
        # torch_tpu compiles a program that combines groups of different
        # devices over the world, so a stage could launch it only together
        # with every other stage.
        # TODO(yinlin): drop this guard with the torch_tpu pin that compiles
        # such a program for the stage alone.
        unsupported = (
            "pipeline stages that combine parallel groups of different sizes "
            "with this torch_tpu"
        )
        remedy = (
            " Inside a stage, use only one of tensor, prefill context, or"
            " data-parallel expert parallelism."
        )
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
    if unsupported is not None:
        raise NotImplementedError(
            f"Pipeline parallelism on TPU does not support {unsupported} yet.{remedy}"
        )

    if envs.TPU_PP_HANDOFF_DEPTH < 1:
        # A stage queues at most this many forwards for the next one; with
        # none it could hand nothing on.
        raise ValueError(
            f"TPU_PP_HANDOFF_DEPTH must be at least 1, got {envs.TPU_PP_HANDOFF_DEPTH}."
        )
    # Counts the activations and, for a model whose layers share a top-k
    # index table, that table; the hand-off checks again with the
    # tensors the model hands on.
    from vllm_torchtpu.distributed.pp_buffered import check_buffered_receives

    hf_config = vllm_config.model_config.hf_text_config
    carried = 2 if getattr(hf_config, "index_topk", None) is not None else 1
    check_buffered_receives(
        parallel_config.pipeline_parallel_size, envs.TPU_PP_HANDOFF_DEPTH, carried
    )
