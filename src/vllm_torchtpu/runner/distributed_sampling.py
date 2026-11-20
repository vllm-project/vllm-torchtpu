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
"""Runner-side config, load-time admission, and per-chunk gates for
vocabulary-parallel sampling.

Controlled via ``--additional-config '{"tpu_vocab_parallel_sampling": ...}'``
with values ``"off"`` (default), ``"greedy"``, or ``"all"``. Device-side
sampling lives in ``layers/adapter/sample/distributed_sampling.py``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Any, cast

import torch
from vllm.lora.layers import LogitsProcessorWithLoRA
from vllm.model_executor.layers.logits_processor import LogitsProcessor

from vllm_torchtpu.layers.adapter import dp_vocab_parallel
from vllm_torchtpu.layers.adapter.sample.distributed_sampling import (
    MAX_DISTRIBUTED_TOPK,
    DistributedSamplingMode,
)
from vllm_torchtpu.layers.adapter.sample.top_k_top_p import SAMPLING_EPS
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.runner.utils import VOCAB_PARALLEL_SAMPLING_KEY

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

    from vllm_torchtpu.runner.vllm_v1.worker.gpu_input_batch import (
        CachedRequestState,
        InputBatch,
    )

logger = init_logger(__name__)


class VocabParallelSamplingConfig(StrEnum):
    OFF = "off"
    GREEDY = "greedy"
    ALL = "all"


def get_vocab_parallel_sampling_config(
    vllm_config: VllmConfig | None,
) -> VocabParallelSamplingConfig:
    """Parse ``additional_config["tpu_vocab_parallel_sampling"]``."""
    if vllm_config is None:
        return VocabParallelSamplingConfig.OFF
    additional_config = cast("dict[str, Any]", vllm_config.additional_config)
    raw = additional_config.get(
        VOCAB_PARALLEL_SAMPLING_KEY, VocabParallelSamplingConfig.OFF.value
    )
    if not isinstance(raw, str):
        valid = [m.value for m in VocabParallelSamplingConfig]
        raise ValueError(
            f"additional_config[{VOCAB_PARALLEL_SAMPLING_KEY!r}] must be one "
            f"of {valid}, got {raw!r}"
        )
    try:
        return VocabParallelSamplingConfig(raw)
    except ValueError:
        valid = [m.value for m in VocabParallelSamplingConfig]
        raise ValueError(
            f"additional_config[{VOCAB_PARALLEL_SAMPLING_KEY!r}] must be one "
            f"of {valid}, got {raw!r}"
        ) from None


class _LogitsProbe(torch.nn.Module):
    """Sentinel logits_processor used to verify passthrough ``compute_logits``."""

    def __init__(self, expected_head: torch.nn.Module) -> None:
        super().__init__()
        self.expected_head = expected_head
        self.sentinel_in = object()
        self.sentinel_out = object()
        self.matched = False

    def forward(
        self,
        lm_head: torch.nn.Module,
        hidden_states: object,
        embedding_bias: torch.Tensor | None = None,
    ) -> object:
        expected_bias = getattr(self.expected_head, "bias", None)
        self.matched = (
            lm_head is self.expected_head
            and hidden_states is self.sentinel_in
            and embedding_bias is expected_bias
        )
        return self.sentinel_out


def _verify_passthrough_compute_logits(
    model: torch.nn.Module, owner: torch.nn.Module
) -> bool:
    """True when ``model.compute_logits`` is absent or passes straight through
    ``owner.logits_processor(owner.lm_head, hidden_states, lm_head.bias)``."""
    compute_logits = getattr(model, "compute_logits", None)
    if not callable(compute_logits):
        return True
    probe = _LogitsProbe(cast(torch.nn.Module, owner.lm_head))
    orig_processor = owner.logits_processor
    owner.logits_processor = probe
    try:
        out = compute_logits(probe.sentinel_in)
    except (AttributeError, TypeError):
        out = None
    finally:
        owner.logits_processor = orig_processor
    return probe.matched and out is probe.sentinel_out


def _find_passthrough_head_owner(
    model: torch.nn.Module,
) -> tuple[torch.nn.Module | None, str | None]:
    for module in model.modules():
        processor = getattr(module, "logits_processor", None)
        head = getattr(module, "lm_head", None)
        has_top_tokens = isinstance(processor, LogitsProcessorWithLoRA) or hasattr(
            type(processor), "get_top_tokens"
        )
        if has_top_tokens and hasattr(head, "shard_indices"):
            break
    else:
        return None, (
            f"{type(model).__name__} has no module with an lm_head and a "
            "supported logits_processor"
        )
    assert processor is not None and head is not None
    if processor.scale <= 0.0:
        return None, f"non-positive logit scale {processor.scale}"
    if not _verify_passthrough_compute_logits(model, module):
        return None, (
            f"{type(model).__name__}.compute_logits performs custom pre- or "
            "post-processing beyond logits_processor(lm_head, hidden_states)"
        )
    return module, None


def _ineligible_reason(
    model: torch.nn.Module,
    dp_lockstep: bool,
    *,
    speculative: bool,
) -> tuple[torch.nn.Module | None, str | None]:
    if speculative:
        return None, "speculative decoding uses the full-logits rejection sampler"
    owner, reason = _find_passthrough_head_owner(model)
    if owner is None:
        return None, reason
    processor = owner.logits_processor
    head = cast("VocabParallelEmbedding", owner.lm_head)
    if (
        getattr(head, dp_vocab_parallel.DP_VOCAB_ATTR, False)
        or getattr(head, "parallel_group", None) is not None
    ):
        return (
            None,
            "lm_head is sharded over a custom or EP group (TPU_DP_VOCAB_PARALLEL)",
        )
    if dp_lockstep:
        return None, "expert-parallel data parallelism is enabled"
    if head.tp_size <= 1:
        return None, "lm_head is not tensor-parallel (tp_size <= 1)"
    if head.shard_indices.num_added_elements > 0:
        return None, "lm_head has added vocabulary elements"
    if isinstance(processor, LogitsProcessorWithLoRA):
        return None, "LoRA-wrapped LogitsProcessor requires the full-logits LoRA delta"
    if type(processor) not in (LogitsProcessor, dp_vocab_parallel.TpuLogitsProcessor):
        return None, f"unsupported logits processor {type(processor).__name__}"
    processor = cast(LogitsProcessor, processor)
    if processor.logits_as_input:
        return None, "logits_as_input=True requires full logits"
    if processor.soft_cap is not None and processor.soft_cap <= 0.0:
        return None, f"non-positive soft_cap={processor.soft_cap}"
    vocab_start_buf = getattr(head, "org_vocab_start_index", None)
    if vocab_start_buf is None or not isinstance(vocab_start_buf, torch.Tensor):
        return None, "lm_head shard bounds are not registered as buffers"
    return owner, None


def resolve_head_owner(
    model: torch.nn.Module,
    dp_lockstep: bool,
    *,
    speculative: bool = False,
) -> torch.nn.Module | None:
    """Returns the module holding ``lm_head`` and ``logits_processor``, or
    ``None`` (with one ``warning_once``) when the config is ineligible."""
    owner, reason = _ineligible_reason(model, dp_lockstep, speculative=speculative)
    if reason is not None:
        logger.warning_once(
            "Ignoring %s=%s for non-greedy sampling: %s. Falling back to "
            "full-logits sampling.",
            VOCAB_PARALLEL_SAMPLING_KEY,
            VocabParallelSamplingConfig.ALL.value,
            reason,
        )
        return None
    return owner


def resolve_greedy_head_owner(
    model: torch.nn.Module,
    dp_lockstep: bool,
    config: VocabParallelSamplingConfig,
) -> torch.nn.Module | None:
    """Resolves the greedy vocab-parallel owner for ``additional_config``
    (``"greedy"`` or ``"all"``), returning ``None`` with one ``warning_once``
    when ineligible."""
    owner, reason = _find_passthrough_head_owner(model)
    if owner is not None:
        head = cast("VocabParallelEmbedding", owner.lm_head)
        if getattr(head, dp_vocab_parallel.DP_VOCAB_ATTR, False):
            owner, reason = None, "lm_head is sharded over DP (TPU_DP_VOCAB_PARALLEL)"
        elif dp_lockstep and head.tp_size > 1:
            owner, reason = (
                None,
                "expert-parallel data parallelism with TP > 1 is enabled",
            )
        elif getattr(head, "bias", None) is not None:
            owner, reason = None, "greedy top-tokens does not forward lm_head.bias"
    if reason is not None:
        logger.warning_once(
            "Ignoring %s=%s for greedy sampling: %s. Falling back to "
            "full-logits sampling.",
            VOCAB_PARALLEL_SAMPLING_KEY,
            config.value,
            reason,
        )
        return None
    return owner


def head_tp_and_shard_vocab(owner: torch.nn.Module) -> tuple[int, int]:
    """Returns ``(tp_size, shard_vocab)`` from ``owner.lm_head``."""
    head = cast("VocabParallelEmbedding", owner.lm_head)
    return int(head.tp_size), int(head.num_embeddings_per_partition)


def _has_structured_output(
    input_batch: InputBatch, requests: Mapping[str, CachedRequestState]
) -> bool:
    for req_id in input_batch.req_ids:
        if req_id is None:
            continue
        params = requests[req_id].sampling_params
        if params is not None and params.structured_outputs is not None:
            return True
    return False


def step_eligible(
    input_batch: InputBatch, requests: Mapping[str, CachedRequestState]
) -> bool:
    """Whether a step may use distributed non-greedy sampling."""
    if input_batch.all_greedy or input_batch.max_num_logprobs is not None:
        return False
    if _has_structured_output(input_batch, requests):
        logger.info_once(
            "%s: steps with a structured-output request gather the full logits",
            VOCAB_PARALLEL_SAMPLING_KEY,
        )
        return False
    return True


def select_chunk_mode(
    input_batch: InputBatch, start: int, end: int, vocab_size: int
) -> DistributedSamplingMode | None:
    """Picks the sampling mode for requests ``[start, end)`` from CPU metadata.

    An all-greedy chunk inside a mixed multi-chunk step uses ``TOPK_TOPP``,
    which carries the exact cross-shard argmax for greedy rows. Chunks mixing
    ``TOPK_TOPP`` and ``PURE_TEMP`` rows return ``MIXED``. ``None`` means
    full-logits sampling.
    """
    if end <= start:
        return None
    non_greedy = input_batch.temperature_cpu_tensor[start:end] > SAMPLING_EPS
    if not bool(non_greedy.any()):
        return DistributedSamplingMode.TOPK_TOPP
    top_k = input_batch.top_k_cpu_tensor[start:end][non_greedy]
    top_p = input_batch.top_p_cpu_tensor[start:end][non_greedy]
    has_top_k = (top_k >= 1) & (top_k < vocab_size)
    is_topk_topp = has_top_k & (top_k <= MAX_DISTRIBUTED_TOPK)
    is_pure_temp = ~has_top_k & (top_p >= 1.0)
    if bool(is_topk_topp.all()):
        return DistributedSamplingMode.TOPK_TOPP
    if bool(is_pure_temp.all()):
        return DistributedSamplingMode.PURE_TEMP
    if bool((is_topk_topp | is_pure_temp).all()):
        return DistributedSamplingMode.MIXED
    logger.info_once(
        "%s: slices with top_k > %d or top_p without top_k gather full logits",
        VOCAB_PARALLEL_SAMPLING_KEY,
        MAX_DISTRIBUTED_TOPK,
    )
    return None


def log_step_path(mode: DistributedSamplingMode, req_ids: Sequence[str | None]) -> None:
    """Logs once per mode when served traffic (not warmup) takes this path."""
    if all(req_id is None or req_id.startswith("__") for req_id in req_ids):
        return
    logger.info_once("%s: step took mode=%s", VOCAB_PARALLEL_SAMPLING_KEY, mode.value)
