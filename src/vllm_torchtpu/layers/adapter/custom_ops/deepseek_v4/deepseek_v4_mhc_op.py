# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper for DeepSeek-V4 mHC Pallas custom ops.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.layers.deepseek_v4_mhc`.
"""

from typing import Any

from vllm_torchtpu.llm_tpu_core.layers.deepseek_v4_mhc import (
    MHCOps,
)
from vllm_torchtpu.llm_tpu_core.layers.deepseek_v4_mhc import (
    get_mhc_ops as _core_get_mhc_ops,
)
from vllm_torchtpu.llm_tpu_core.layers.deepseek_v4_mhc import (
    get_mhc_post_op as _core_get_mhc_post_op,
)
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    get_vllm_model_wrapper_context,
)


def _resolve_mesh(mesh: Any = None) -> Any:
    if mesh is not None:
        return mesh
    try:
        return get_vllm_model_wrapper_context().mesh
    except Exception:
        return None


def get_mhc_post_op(mesh: Any = None):
    return _core_get_mhc_post_op(mesh=_resolve_mesh(mesh))


def get_mhc_ops(
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    mesh: Any = None,
) -> MHCOps:
    return _core_get_mhc_ops(
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        mesh=_resolve_mesh(mesh),
    )


__all__ = [
    "MHCOps",
    "get_mhc_ops",
    "get_mhc_post_op",
]
