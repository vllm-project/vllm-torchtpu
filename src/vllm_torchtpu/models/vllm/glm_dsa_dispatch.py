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
"""Hardware and architecture auto-dispatch predicates for GLM-5 DSA on TPU v7x."""

import collections
import itertools
from collections.abc import Callable
from typing import Any

import torch

from vllm_torchtpu.compilation import shape_variants
from vllm_torchtpu.compilation.shape_variants import (
    bucket_gate_edge as bucket_gate_edge,
)


def in_shape_variants_warmup() -> bool:
    return shape_variants.in_warmup()


GLM_DSA_ARCH = "GlmMoeDsaForCausalLM"
GLM_DSA_MODEL_TYPE = "glm_moe_dsa"
SP_SHARED_GROUP_FULL_MAX_MODEL_LEN = 262_144
SP_SHARED_GROUP_MIN_FREE_HBM_PER_CORE = 24 * (1024**3)
MLA_DECODE_ROW_WIDTHS: tuple[int, ...] = (128, 256, 384)
MLA_MIXED_LIVE_WIDTHS: tuple[int, ...] = (256, 384)
DCP_MLA_SMALL_CHUNK_BATCH = 4
DCP_MLA_SMALL_CHUNK_TOKENS = 32
DCP_MLA_SEND_GROUP_ROWS = 8
DCP_MLA_SEND_SMALL_A2A_MAX_TOKENS = 8
MLA_FUSED_SMALL_A2A_MERGE_MAX_TOKENS = 4
MLA_FUSED_SMALL_A2A_PIN_MIN_TOKENS = 8
MLA_FUSED_SMALL_A2A_VMEM_MIB = 16
MLA_FUSED_SMALL_A2A_PIN_VMEM_MIB = 32
DCP_MERGE_CHUNKED_DECODE_SPLITS = 2
DCP_MERGE_CHUNKED_PREFILL_SPLITS = 8
DCP_MERGE_CHUNKED_MIN_TOKENS = 1024
DCP_MERGE_CHUNKED_LSE_AG_MAX_TOKENS = 2048
DCP_MERGE_WEIGHTED_HM_MIN_TOKENS = 2048
DCP_MLA_MERGE_PIECES = 2
DCP_MLA_MERGE_PIECES_MIN_TOKENS = 2048
INDEXER_QPREP_MIN_TOKENS = 128
STREAMIDX_DEFAULT_MIXED_KV_PAGES = 4
SEGMENT_TOKENIZE_MIN_MODEL_LEN = 65_536
SEGMENT_TOKENIZE_MIN_PROMPT_CHARS = 32_768
SEGMENT_TOKENIZE_CACHE_TOKENS = 33_554_432
DCP_TP_ZIGZAG_BANDS = 4
MLA_SPLIT_QKV_A_CHUNK = 1024
GLM_DSA_SC_ALL_GATHER_OFFLOAD_MIN_BYTES = 6 << 20

_ACTIVE_ARCH: str | None = None
_ACTIVE_MAX_MODEL_LEN: int = 0


def set_active_model_config(model_config: Any | None) -> None:
    global _ACTIVE_ARCH, _ACTIVE_MAX_MODEL_LEN
    from vllm_torchtpu import tpu_info
    from vllm_torchtpu.models.vllm.gemma4 import is_gemma4_model_config

    _ACTIVE_ARCH = extract_architecture(model_config)
    max_len = (
        getattr(model_config, "max_model_len", 0) if model_config is not None else 0
    )
    _ACTIVE_MAX_MODEL_LEN = int(max_len) if isinstance(max_len, int) else 0
    tpu_info.set_v7x_model_fastpaths(
        model_config is not None
        and (is_glm_dsa_model(model_config) or is_gemma4_model_config(model_config))
    )


def extract_architecture(model_config: Any | None) -> str | None:
    if model_config is None or isinstance(model_config, str):
        return model_config
    hf_cfg = getattr(model_config, "hf_text_config", None) or getattr(
        model_config, "hf_config", None
    )
    for cfg in (model_config, hf_cfg):
        archs = getattr(cfg, "architectures", None) if cfg is not None else None
        if archs and isinstance(archs, (list, tuple)):
            return str(archs[0])
    if getattr(hf_cfg, "model_type", None) == GLM_DSA_MODEL_TYPE:
        return GLM_DSA_ARCH
    return None


def is_glm_dsa_model(model_config: Any | None = None) -> bool:
    arch = _ACTIVE_ARCH if model_config is None else extract_architecture(model_config)
    return arch == GLM_DSA_ARCH


@torch.compiler.assume_constant_result
def is_tpu_v7x() -> bool:
    from vllm_torchtpu import tpu_info

    return tpu_info.is_tpu_v7x()


@torch.compiler.assume_constant_result
def glm_dsa_v7x_active(model_config: Any | None = None) -> bool:
    return is_glm_dsa_model(model_config) and is_tpu_v7x()


def resolve_ep_tp_size(tp_size: int, model_config: Any | None = None) -> int:
    if glm_dsa_v7x_active(model_config) and tp_size >= 4 and tp_size % 2 == 0:
        return 2
    return 1


def select_sp_shared_group(
    tp_size: int,
    max_model_len: int | None = None,
    free_hbm_per_core_bytes: int | None = None,
) -> int:
    eff_len = _ACTIVE_MAX_MODEL_LEN if max_model_len is None else int(max_model_len)
    low_hbm = (
        free_hbm_per_core_bytes is not None
        and free_hbm_per_core_bytes < SP_SHARED_GROUP_MIN_FREE_HBM_PER_CORE
    )
    if eff_len <= SP_SHARED_GROUP_FULL_MAX_MODEL_LEN and not low_hbm:
        return 1
    return next((c for c in (4, 2) if tp_size >= c and tp_size % c == 0), 1)


def resolve_dcp_mla_topk_cap(explicit_cap: int, dcp_size: int) -> int:
    if explicit_cap > 0:
        return int(explicit_cap)
    return 512 if glm_dsa_v7x_active() and dcp_size >= 8 else 0


def resolve_dcp_mla_topk_cap_chunk(explicit_chunk: int) -> int:
    if glm_dsa_v7x_active() and explicit_chunk == 256:
        return 512
    return int(explicit_chunk)


def resolve_dcp_lse_merge_a2a_max_tokens(explicit_max: int) -> int:
    if glm_dsa_v7x_active() and explicit_max == 64:
        return 32
    return int(explicit_max)


def fused_small_a2a_bucket_config(
    num_tokens: int,
) -> tuple[bool | tuple[str, ...], int]:
    if num_tokens == 4:
        return ("kv",), MLA_FUSED_SMALL_A2A_VMEM_MIB
    pin = num_tokens >= MLA_FUSED_SMALL_A2A_PIN_MIN_TOKENS
    return (
        pin,
        MLA_FUSED_SMALL_A2A_PIN_VMEM_MIB if pin else MLA_FUSED_SMALL_A2A_VMEM_MIB,
    )


_shared_jax_ops: dict[tuple[str, tuple[Any, ...]], Any] = {}
_shared_jax_op_ids: collections.defaultdict[str, Any] = collections.defaultdict(
    itertools.count
)


def get_shared_jax_op(
    kind: str, key: tuple[Any, ...], build: Callable[[str], Any]
) -> Any:
    full_key = (kind, key)
    if (op := _shared_jax_ops.get(full_key)) is None:
        op = _shared_jax_ops[full_key] = build(
            f"pallas::{kind}_{next(_shared_jax_op_ids[kind])}"
        )
    return op
