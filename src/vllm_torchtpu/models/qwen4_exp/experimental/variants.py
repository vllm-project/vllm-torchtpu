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
"""Experimental Qwen3.8-Flash's switchable pieces (slots) and their variants.

``VARIANTS[slot][name]`` is a `Variant`: the feature values it sets and an
``eligible(vllm_config)`` rule. `profiles.py` lists, per profile, the variants
each slot may use in order of preference; `plan.py` picks the first eligible
one. Nothing here reads the environment or --additional-config.

A behavior that won everywhere it was served (agentx 4fd8b1601e on the
infx-qwen38-flash-agentx recipe) is not a slot: it is the code. A slot exists
only where a served setup needs a different choice:

- decode_batch: the large-batch stack served at max_num_seqs >= 32 (c32 and
  c48), including main-KV decode context parallelism over rank pairs;
- c48_stack: the stack served only at c48 (no config value separates it from
  c32, so it is the agentx-c48 profile's choice);
- pool_budget: the unified KV pool budget (fixed below gpu-util 0.95);
- gdn_spec_slots: GDN speculative state blocks per request: one per verify
  position ("per_token"), or two ping-pong blocks plus accepted-token replay
  ("two_slot", half the GDN speculative state blocks; unified pool and MTP only).
"""

import dataclasses
from collections.abc import Callable, Mapping
from typing import Any


def _always(vllm_config: Any) -> bool:
    return True


@dataclasses.dataclass(frozen=True)
class Variant:
    #: Feature name -> value this variant serves (see `plan.value`).
    features: Mapping[str, Any]
    eligible: Callable[[Any], bool] = _always


def _max_num_seqs(vllm_config: Any) -> int:
    return int(vllm_config.scheduler_config.max_num_seqs)


def _large_batch(vllm_config: Any) -> bool:
    return _max_num_seqs(vllm_config) >= 32


def _below_full_hbm(vllm_config: Any) -> bool:
    return float(vllm_config.cache_config.gpu_memory_utilization) < 0.95


_LARGE_BATCH = {
    "QWEN4_EXP_PREFILL_SP_SHARED_LOCAL": True,
    "TPU_GDN_CHUNK_PAIR": True,
    "VLLM_TPU_GDN_CONV_WEIGHT_KMAJOR": True,
    "VLLM_TPU_HC_PREFILL_COMBINE_MIX": True,
    "VLLM_TPU_HC_PREFILL_WIDE_TILE": 256,
    "VLLM_TPU_QSA_COMPRESSED_PAIR_SHIFT": True,
    "VLLM_TPU_QSA_COMPRESSED_RUN_WRITE": True,
    "VLLM_TPU_QSA_DCP_HEAD_EXCHANGE": True,
    "VLLM_TPU_QSA_DCP_PREFILL_QUERY_BLOCK": 0,
    "VLLM_TPU_QSA_DCP_Q_GATHER": True,
    "VLLM_TPU_QSA_DECODE_COMPACT_SCHEDULE": True,
    "VLLM_TPU_QSA_DECODE_RPA_BKV": 512,
    "VLLM_TPU_QSA_DECODE_RPA_LANES": 4,
    "VLLM_TPU_QSA_DECODE_RPA_MAX_QUERIES": 128,
    "VLLM_TPU_QSA_INDEXER_MIXED_ONLY": True,
    "VLLM_TPU_QSA_MAIN_KV_DCP": 2,
    "VLLM_TPU_QWEN4_FUSED_PREP_FUNCTIONAL": True,
}
_SMALL_BATCH = {
    "QWEN4_EXP_PREFILL_SP_SHARED_LOCAL": False,
    "TPU_GDN_CHUNK_PAIR": False,
    "VLLM_TPU_GDN_CONV_WEIGHT_KMAJOR": False,
    "VLLM_TPU_HC_PREFILL_COMBINE_MIX": False,
    "VLLM_TPU_HC_PREFILL_WIDE_TILE": 0,
    "VLLM_TPU_QSA_COMPRESSED_PAIR_SHIFT": False,
    "VLLM_TPU_QSA_COMPRESSED_RUN_WRITE": False,
    "VLLM_TPU_QSA_DCP_HEAD_EXCHANGE": False,
    "VLLM_TPU_QSA_DCP_PREFILL_QUERY_BLOCK": 1024,
    "VLLM_TPU_QSA_DCP_Q_GATHER": False,
    "VLLM_TPU_QSA_DECODE_COMPACT_SCHEDULE": False,
    "VLLM_TPU_QSA_DECODE_RPA_BKV": 256,
    "VLLM_TPU_QSA_DECODE_RPA_LANES": 8,
    "VLLM_TPU_QSA_DECODE_RPA_MAX_QUERIES": 0,
    "VLLM_TPU_QSA_INDEXER_MIXED_ONLY": False,
    "VLLM_TPU_QSA_MAIN_KV_DCP": 0,
    "VLLM_TPU_QWEN4_FUSED_PREP_FUNCTIONAL": False,
}
_C48_ON = {
    "PLE_CPU_OFFLOAD": True,
    "VLLM_TPU_GDN_HOIST_METADATA": True,
    "VLLM_TPU_HC_PREFILL_INJECT": True,
    "VLLM_TPU_HC_PREFILL_UP_T": True,
    "VLLM_TPU_PLE_HOST_FAST_LOOKUP": True,
    "VLLM_TPU_PLE_HOST_OVERLAP": True,
    "VLLM_TPU_QSA_COMPRESSED_DCP": True,
    "VLLM_TPU_QSA_DCP_GATE_IN_COMBINE": True,
    "VLLM_TPU_QSA_DCP_QGATE_PAIR_MAJOR": True,
    "VLLM_TPU_QSA_DCP_SHARD_MERGE": True,
    "VLLM_TPU_QSA_DCP_SPLIT_LSE": True,
    "VLLM_TPU_QSA_DECODE_TOKEN_SHARD": True,
    "VLLM_TPU_QSA_FUSED_DCP_BLOCKS": True,
    "VLLM_TPU_QSA_INDEXER_HOIST_META": True,
    "VLLM_TPU_QSA_INDEXER_TILE_ALIGNED_SCORES": True,
    "VLLM_TPU_QSA_MTP_KV_DCP": True,
    "VLLM_TPU_QSA_PREFILL_BITSET_OWNED": True,
    "VLLM_TPU_QSA_PREFILL_LIVE_MASK_PADDED": True,
}
_C48_OFF = {
    "PLE_CPU_OFFLOAD": False,
    "VLLM_TPU_GDN_HOIST_METADATA": False,
    "VLLM_TPU_HC_PREFILL_INJECT": False,
    "VLLM_TPU_HC_PREFILL_UP_T": False,
    "VLLM_TPU_PLE_HOST_FAST_LOOKUP": False,
    "VLLM_TPU_PLE_HOST_OVERLAP": False,
    "VLLM_TPU_QSA_COMPRESSED_DCP": False,
    "VLLM_TPU_QSA_DCP_GATE_IN_COMBINE": False,
    "VLLM_TPU_QSA_DCP_QGATE_PAIR_MAJOR": False,
    "VLLM_TPU_QSA_DCP_SHARD_MERGE": False,
    "VLLM_TPU_QSA_DCP_SPLIT_LSE": False,
    "VLLM_TPU_QSA_DECODE_TOKEN_SHARD": False,
    "VLLM_TPU_QSA_FUSED_DCP_BLOCKS": False,
    "VLLM_TPU_QSA_INDEXER_HOIST_META": False,
    "VLLM_TPU_QSA_INDEXER_TILE_ALIGNED_SCORES": False,
    "VLLM_TPU_QSA_MTP_KV_DCP": False,
    "VLLM_TPU_QSA_PREFILL_BITSET_OWNED": False,
    "VLLM_TPU_QSA_PREFILL_LIVE_MASK_PADDED": False,
}
# Served in every agentx setup (mixed mode). Not a slot: the plan always
# applies them; `PLAIN` is the code's behavior without them (unit tests).
SERVED = {
    "QWEN4_EXP_PACK_MAIN_KV": True,
    "QWEN4_EXP_PREFILL_SP": True,
    "QWEN4_EXP_PREFILL_SP_MOE": True,
    "VLLM_TPU_QSA_DECODE_POOL_PAGE_DMA": True,
    "VLLM_TPU_QSA_DECODE_POOL_PAGE_TABLE": True,
    "VLLM_TPU_QSA_DRAFT_SHARED_SELECTION": True,
    "VLLM_TPU_QSA_INDEXER_SC_BOUNDED_COPY": True,
    "VLLM_TPU_QSA_INDEXER_TOKEN_SHARD": True,
    "VLLM_TPU_QSA_INDEXER_UPDATE_SHIFT": True,
    "VLLM_TPU_QSA_PREFILL_LEAN": True,
    "VLLM_TPU_QSA_PREFILL_SC_BOUNDED_COPY": True,
    "VLLM_TPU_QSA_PREFIX_SLICE": True,
    "VLLM_TPU_QWEN4_FUSED_PREP": True,
    "VLLM_TPU_QWEN4_NGRAM_WINDOW": "1",
}
PLAIN = {
    "QWEN4_EXP_PACK_MAIN_KV": False,
    "QWEN4_EXP_PREFILL_SP": False,
    "QWEN4_EXP_PREFILL_SP_MOE": False,
    "VLLM_TPU_QSA_DECODE_POOL_PAGE_DMA": False,
    "VLLM_TPU_QSA_DECODE_POOL_PAGE_TABLE": False,
    "VLLM_TPU_QSA_DRAFT_SHARED_SELECTION": False,
    "VLLM_TPU_QSA_INDEXER_SC_BOUNDED_COPY": False,
    "VLLM_TPU_QSA_INDEXER_TOKEN_SHARD": False,
    "VLLM_TPU_QSA_INDEXER_UPDATE_SHIFT": False,
    "VLLM_TPU_QSA_PREFILL_LEAN": False,
    "VLLM_TPU_QSA_PREFILL_SC_BOUNDED_COPY": False,
    "VLLM_TPU_QSA_PREFIX_SLICE": False,
    "VLLM_TPU_QWEN4_FUSED_PREP": False,
    "VLLM_TPU_QWEN4_NGRAM_WINDOW": "0",
}

VARIANTS: dict[str, dict[str, Variant]] = {
    "decode_batch": {
        "large": Variant(_LARGE_BATCH, eligible=_large_batch),
        "small": Variant(_SMALL_BATCH),
    },
    "c48_stack": {"on": Variant(_C48_ON), "off": Variant(_C48_OFF)},
    # QWEN4_EXP_MAX_POOL_BYTES: 61.6 GB at gpu-util 0.92, 0 (HBM sizing) at 0.95.
    "pool_budget": {
        "fixed_61g": Variant(
            {"QWEN4_EXP_MAX_POOL_BYTES": 61603840000}, eligible=_below_full_hbm
        ),
        "auto": Variant({"QWEN4_EXP_MAX_POOL_BYTES": 0}),
    },
    "gdn_spec_slots": {
        "two_slot": Variant({"QWEN4_EXP_GDN_TWO_SLOT": True}),
        "per_token": Variant({"QWEN4_EXP_GDN_TWO_SLOT": False}),
    },
}
