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
"""MiniMax M3's switchable pieces (slots) and their variants.

``VARIANTS[slot][name]`` is a `Variant`: an implementation (a
``"module:attr"`` import path, or the value the model hands to a shared
hook) and an ``eligible(vllm_config)`` rule. `profiles.py` lists, per
profile, the variants each slot may use in order of preference; `plan.py`
picks the first eligible one. Nothing here reads the environment or
--additional-config.

A behavior that won everywhere it was measured is not a slot: it is the
code (MSA DCP tile selection, fp8-fed int4 index keys). A slot exists only
where a setup can need a different choice.
"""

import dataclasses
import importlib
from collections.abc import Callable
from typing import Any

_PKG = "vllm_torchtpu.models.minimax_m3.experimental"


def _always(vllm_config: Any) -> bool:
    return True


@dataclasses.dataclass(frozen=True)
class Variant:
    #: ``"module:attr"`` (imported by `load`) or a plain value.
    impl: Any
    eligible: Callable[[Any], bool] = _always

    def load(self) -> Any:
        if isinstance(self.impl, str) and ":" in self.impl:
            module, _, attr = self.impl.partition(":")
            return getattr(importlib.import_module(module), attr)
        return self.impl


def _parallel(vllm_config: Any) -> Any:
    return vllm_config.parallel_config


def _tp_group_is_ep_world(vllm_config: Any) -> bool:
    """The Pallas TP all-reduce needs the EP mesh to span the TP group and
    the world (`ops.tp_all_reduce`): TP > 1, DP 1, PCP 1, EP on."""
    p = _parallel(vllm_config)
    return (
        p.tensor_parallel_size > 1
        and p.data_parallel_size == 1
        and getattr(p, "prefill_context_parallel_size", 1) == 1
        and bool(p.enable_expert_parallel)
    )


def _speculative(vllm_config: Any) -> bool:
    return vllm_config.speculative_config is not None


def _fp8_kv_cache(vllm_config: Any) -> bool:
    """--kv-cache-dtype fp8: the other attention caches (the DSpark draft's)
    store fp8 next to 4-bit MSA caches."""
    return vllm_config.cache_config.cache_dtype in ("fp8", "fp8_e4m3")


VARIANTS: dict[str, dict[str, Variant]] = {
    # The target model and the checkpoint's dspark/ draft.
    "model": {"msa_sparse": Variant(f"{_PKG}.model:MiniMaxM3SparseForCausalLM")},
    "draft": {"dspark": Variant(f"{_PKG}.dspark:MiniMaxM3DSparkForCausalLM")},
    # Largest Pallas TP all-reduce view in bytes (the model installs
    # ops.tp_all_reduce.PallasTpAllReduce with it as its model_all_reduce);
    # 0 = XLA's all-reduce. 524288 admits
    # the decode buckets up to 40 rows of 6144 (served in every agentx
    # config; a 1572864 cap measured a loss).
    "tp_all_reduce": {
        "pallas_512k": Variant(524288, eligible=_tp_group_is_ep_world),
        "xla": Variant(0),
    },
    # Non-greedy spec verify as one program (spec_verify.FusedSpecVerify,
    # installed as the model's spec_verify_sampler): exact, verify
    # 3.76 -> 3.15 ms. None keeps the runner's unfused verify.
    "spec_verify": {
        "fused": Variant(f"{_PKG}.spec_verify:FusedSpecVerify", eligible=_speculative),
        "unfused": Variant(None),
    },
    # Storage of the model-owned MSA caches (K/V, index keys):
    # --kv-cache-dtype's (bf16 or fp8), or 4-bit with per-token-head scales
    # (`kernels.msa_fp4`; attention.cache_dtype(..., int4=True)). int4 holds
    # twice the context per HBM byte: served at c32 / c48 / c64 (DCP 4).
    "msa_cache": {
        "kv_cache_dtype": Variant(False),
        "int4": Variant(True, eligible=_fp8_kv_cache),
    },
}
