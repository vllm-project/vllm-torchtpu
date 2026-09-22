# SPDX-License-Identifier: Apache-2.0
"""Optional model-owned runner extension, selected once before weight loading.

The ordinary runner/cache path is unchanged when selection returns None.
Adapters own input snapshots, dummy/profile geometry and cache lifecycles;
model forward receives explicit kwargs rather than consulting runner state.
"""
from dataclasses import dataclass
from typing import Any, Protocol

import torch


@dataclass(frozen=True)
class ModelInputSnapshot:
    token_store: torch.Tensor
    row_start: int
    num_requests: int
    num_tokens: int
    attention_metadata: dict[str, Any]


class ModelAdapter(Protocol):

    def bind_model(self, model):
        ...

    def begin_step(self):
        ...

    def prepare_inputs(self, snapshot: ModelInputSnapshot) -> dict[str, Any]:
        ...

    def dummy_run(self, num_tokens, num_reqs, profile_prefix=None):
        ...

    def get_kv_cache_spec(self):
        ...

    def initialize_kv_cache(self, config):
        ...


def select_model_adapter(runner) -> ModelAdapter | None:
    if runner.model_config.architecture in (
            "Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration"):
        from vllm_torchtpu.models.vllm.qwen4_exp.inputs import Qwen4ExpAdapter
        return Qwen4ExpAdapter(runner)
    return None
