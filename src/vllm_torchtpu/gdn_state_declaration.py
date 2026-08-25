# SPDX-License-Identifier: Apache-2.0
"""The GDN recurrent-state declaration TPU gives vLLM, in one place.

vLLM reads it twice: `MambaBase.get_kv_cache_spec` asks the layer instance,
and the platform asks the model class to size the KV block slot, before any
layer exists. Upstream keeps the two in agreement by routing both through
`MambaState*Calculator`; TPU deviates from that baseline -- an extra conv
layout axis, PCP sharding on top of TP, pinned state dtypes -- so both
readers call in here and pass nothing but `vllm_config`.
"""

import torch

from vllm_torchtpu.gdn_pool_layout import POOLED_GDN_ARCHITECTURES


def gdn_state_declaration_applies(vllm_config) -> bool:
    """Whether this model's recurrent layers declare their state from here.

    `VllmGatedDeltaNetAttention` is registered out-of-tree for
    `QwenGatedDeltaNetAttention`, so it backs exactly these architectures.
    Other hybrid families keep their own layer and their own declaration.
    """
    return vllm_config.model_config.architecture in POOLED_GDN_ARCHITECTURES


def gdn_state_shapes(vllm_config) -> tuple[tuple[int, ...], ...]:
    """`(conv, SSM)` state shapes held by one rank."""
    from vllm.model_executor.layers.mamba.mamba_utils import \
        is_conv_state_dim_first
    from vllm.model_executor.models import ModelRegistry

    model_cls, _ = ModelRegistry.resolve_model_cls(
        vllm_config.model_config.architecture,
        model_config=vllm_config.model_config)
    conv_shape, ssm_shape = model_cls.get_mamba_state_shape_from_config(
        vllm_config)

    pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
    if pcp_size > 1:
        # The pooled op holds the whole PCP-local state in one page, so PCP
        # shards it exactly like an extra factor of TP.
        if is_conv_state_dim_first():
            raise NotImplementedError("TPU GDN PCP state sharding requires "
                                      "VLLM_SSM_CONV_STATE_LAYOUT=SD.")
        if conv_shape[-1] % pcp_size or ssm_shape[0] % pcp_size:
            raise ValueError("GDN state must be divisible by PCP size: "
                             f"conv={conv_shape}, ssm={ssm_shape}, "
                             f"pcp_size={pcp_size}")
        conv_shape = (*conv_shape[:-1], conv_shape[-1] // pcp_size)
        ssm_shape = (ssm_shape[0] // pcp_size, *ssm_shape[1:])

    # The GDN kernel reads conv state as (state_len, 1, dim).
    return (*conv_shape[:-1], 1, conv_shape[-1]), tuple(ssm_shape)


def gdn_state_dtypes(vllm_config) -> tuple[torch.dtype, ...]:
    """`(conv, SSM)` state dtypes held by one rank.

    f32 on both layouts. Off the pool this is the storage dtype; on the pool
    it is a budget, and conv is over-declared by one tile per block because
    the pooled kernel stores bf16 rows.

    TODO: Support bf16 conv state, and drop the over-declaration with it.
    TODO: Honor --mamba-ssm-cache-dtype; pinning f32 for the pool's SSM
    region also drops the flag on the per-layer path.
    """
    del vllm_config
    return (torch.float32, torch.float32)
