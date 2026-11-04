# SPDX-License-Identifier: Apache-2.0
"""Register the measured Engram tail before tracing; retain the native fallback."""

from typing import Any, cast

import jax
import torch
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental import engram_gate as kernel

_gate_op: Any = None


def initialize_engram_gate() -> None:
    global _gate_op
    if _gate_op is not None or jax.default_backend() != "tpu":
        return
    from vllm_torchtpu.kernels.quantized_matmul.tuned_block_sizes import (
        get_tpu_version,
    )

    if get_tpu_version() != 7:
        return
    from torch_tpu._internal import pallas

    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import get_pallas_op_mesh

    op = pallas.jax_op(
        "pallas::dsv41_engram_gate_residual_v1",
        kernel.gate_residual,
        mesh=get_pallas_op_mesh(),
        input_partition_specs=cast(Any, (P(),) * 5),
    )

    @op.register_fake
    def fake(hidden, kv, qw, kw, mask):
        return torch.empty_like(hidden)

    _gate_op = op


def supports_engram_gate(hidden, kv, qw, kw, mask, eps, clamp) -> bool:
    """Only the TPU geometry, dtypes and constants validated as a full region."""
    return (
        _gate_op is not None
        and eps == kernel.EPS
        and clamp == kernel.CLAMP
        and hidden.shape[1:] == (kernel.MULT, kernel.DIM)
        and kv.shape == (hidden.shape[0], (kernel.MULT + 1) * kernel.DIM)
        and qw.shape == kw.shape == (kernel.MULT, kernel.DIM)
        and all(
            tensor.device.type == "tpu"
            and tensor.device == hidden.device
            and tensor.dtype == torch.bfloat16
            for tensor in (hidden, kv, qw, kw)
        )
        and (
            mask is None
            or (
                mask.shape == (hidden.shape[0],)
                and mask.dtype == torch.float32
                and mask.device == hidden.device
            )
        )
    )


def engram_gate(hidden, kv, qw, kw, mask):
    """Call only after supports_engram_gate; a missing mask keeps all tokens."""
    assert _gate_op is not None
    if mask is None:
        mask = torch.ones(hidden.shape[0], dtype=torch.float32, device=hidden.device)
    return _gate_op(hidden, kv, qw, kw, mask)
