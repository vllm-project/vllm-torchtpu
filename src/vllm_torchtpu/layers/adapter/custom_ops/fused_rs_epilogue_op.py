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
"""Torch custom-op bridge for hierarchical reduce-scatter + residual epilogue.

Wraps `hierrs_tc/kernel_stream.py` (streaming buckets `>= 1024`, returning
token-sharded outputs) and `hierrs_tc/kernel_oneshot_rs_ag.py` (FP8 decode
buckets `128 <= tokens < 1024`, returning rank-ordered gathered outputs) over an
8-device tpu7x 2x2x1 tensor-parallel group (`distributed/tp_mesh.py`). Because
torch custom ops require fixed tensor arities, one op is registered per operand
combination used by dense decoder sublayers.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import torch
import torch.distributed as dist
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas
from vllm.distributed import parallel_state

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.hier_reduce_scatter import (
    agreed_refusal_reason,
    unsupported_tpu7x_4x2_chip_topology_reason,
)
from vllm_torchtpu.distributed.tp_mesh import TP_AXIS_NAME, build_tp_mesh
from vllm_torchtpu.kernels.collectives.hierrs_tc import (
    kernel_oneshot_rs_ag,
    kernel_stream,
)
from vllm_torchtpu.kernels.collectives.hierrs_tc.wrapper import supports_num_devices
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

MIN_STREAMING_RS_TOKENS = kernel_stream.MIN_STREAM_ROWS
# Buckets below 128 tokens have < 16 rows/rank on 8 devices and are barrier
# bound, so they keep native XLA all-reduce.
MIN_ONESHOT_RS_AG_TOKENS = kernel_oneshot_rs_ag.MIN_ROWS

# Any: torch_tpu annotates partition specs as axis-name tuples, while jax_op
# forwards PartitionSpec instances directly to NamedSharding at runtime.
_SHARDED: Any = P(TP_AXIS_NAME, None)
_REPLICATED: Any = P()
_QUANTIZED_OUT = (_SHARDED, _SHARDED, _SHARDED)
_BF16_NEXT_OUT = (_SHARDED, _SHARDED)
_ONESHOT_QUANTIZED_OUT = (_SHARDED, _REPLICATED, _REPLICATED)


@dataclasses.dataclass(frozen=True)
class FusedSublayerCollectives:
    """Custom ops fusing TP reduce-scatter with a decoder sublayer's epilogue."""

    # (partial, residual, post_norm_weight, next_norm_weight)
    #   -> (residual, x_q, x_scale) for float8_e4m3fn else (residual, x_norm)
    norm_add_next_norm: Callable
    # (partial, residual, post_norm_weight, scale, next_norm_weight)
    #   -> (residual, x_q, x_scale) for float8_e4m3fn else (residual, x_norm)
    norm_add_scale_next_norm: Callable
    # (partial, residual, post_norm_weight, scale) -> residual
    norm_add_scale: Callable
    # True when `x_q`, `x_scale`, and `norm_add_scale`'s output are already
    # gathered across all TP ranks in rank order.
    outputs_gathered: bool = False
    next_act_dtype: torch.dtype = torch.float8_e4m3fn


_STREAM_COLLECTIVES_CACHE: dict[tuple, FusedSublayerCollectives] = {}
_ONESHOT_COLLECTIVES_CACHE: dict[tuple, FusedSublayerCollectives] = {}
_RESOLVED_TP_TOPOLOGY: tuple[Any, tuple[int, ...], tuple[int, ...]] | None = None
_REFUSED_REASON: str | None = None


def _eps_op_tag(eps: float) -> str:
    return f"{eps:.8g}".replace("-", "m").replace("+", "p").replace(".", "_")


def supports_streaming_sublayer(num_tokens: Any, hidden: int) -> bool:
    """Whether `[num_tokens, hidden]` fits the streaming reduce-scatter op."""
    return kernel_stream.supports_stream_epilogue(num_tokens, hidden)


def supports_oneshot_sublayer(num_tokens: Any, hidden: int) -> bool:
    """Whether `[num_tokens, hidden]` fits the single-shot VMEM RS + AG op."""
    return (
        num_tokens >= MIN_ONESHOT_RS_AG_TOKENS
        and kernel_oneshot_rs_ag.supports_oneshot_rs_ag(num_tokens, hidden)
    )


def _unsupported_tp_topology_reason() -> str | None:
    if not dist.is_available() or not dist.is_initialized():
        return "torch.distributed is not initialized"
    tp_group = parallel_state.get_tp_group()
    if not supports_num_devices(int(tp_group.world_size)):
        return f"the kernel requires TP=8, got TP={tp_group.world_size}"
    if int(tp_group.world_size) != int(dist.get_world_size()):
        return "the TP group does not span the whole torch.distributed world"
    return unsupported_tpu7x_4x2_chip_topology_reason(require_paired_chips=True)


def _resolve_tp_mesh_and_order() -> tuple[Any, tuple[int, ...], tuple[int, ...]] | None:
    global _RESOLVED_TP_TOPOLOGY, _REFUSED_REASON
    if _RESOLVED_TP_TOPOLOGY is not None:
        return _RESOLVED_TP_TOPOLOGY
    if _REFUSED_REASON is not None:
        return None
    reason = agreed_refusal_reason(_unsupported_tp_topology_reason())
    mesh = order = None
    device_ids: tuple[int, ...] = ()
    if reason is None:
        tp_topology = build_tp_mesh()
        if tp_topology is None:
            reason = "the TP group does not span the whole torch.distributed world"
        else:
            mesh, order = tp_topology
            device_ids = tuple(int(d.id) for d in mesh.devices.flat)
            if device_ids != tuple(range(len(order))):
                reason = (
                    f"the TP mesh does not span device ids 0..{len(order) - 1}: "
                    f"{list(device_ids)}"
                )
    if reason is not None:
        _REFUSED_REASON = reason
        logger.warning_once(
            "TPU_DENSE_SP_HIER_RS=%s is set, but dense sequence parallelism "
            "keeps XLA reduce-scatter: %s.",
            envs.TPU_DENSE_SP_HIER_RS,
            reason,
        )
        return None
    assert mesh is not None and order is not None
    _RESOLVED_TP_TOPOLOGY = (mesh, order, device_ids)
    return _RESOLVED_TP_TOPOLOGY


def _shard_map_sublayer(
    mesh: Any, fn: Callable, num_weights: int, out_specs: Any
) -> Callable:
    return jax.shard_map(
        fn,
        mesh=mesh,
        in_specs=(_SHARDED, _SHARDED, *(_REPLICATED,) * num_weights),
        out_specs=out_specs,
        check_vma=False,
    )


def build_streaming_sublayer_collectives(
    eps: float,
    *,
    fp8_wire: bool,
    next_act_dtype: torch.dtype = torch.float8_e4m3fn,
) -> FusedSublayerCollectives | None:
    """Builds streaming fused RS + epilogue ops for the active TP group."""
    if next_act_dtype not in (torch.float8_e4m3fn, torch.bfloat16):
        raise ValueError(f"Unsupported next_act_dtype: {next_act_dtype}")
    resolved = _resolve_tp_mesh_and_order()
    if resolved is None:
        return None
    mesh, order, device_ids = resolved
    key = (device_ids, order, eps, fp8_wire, next_act_dtype)
    cached = _STREAM_COLLECTIVES_CACHE.get(key)
    if cached is not None:
        return cached

    jax_next_dtype = (
        jnp.float8_e4m3fn if next_act_dtype == torch.float8_e4m3fn else jnp.bfloat16
    )

    def local(partial, residual, post_norm_weight, scale, next_norm_weight):
        return kernel_stream.reduce_scatter_epilogue_stream_local(
            partial,
            residual,
            post_norm_weight=post_norm_weight,
            scale=scale,
            next_norm_weight=next_norm_weight,
            eps=eps,
            axis_name=TP_AXIS_NAME,
            order=order,
            fp8_wire=fp8_wire,
            next_act_dtype=jax_next_dtype,
        )

    def norm_add_scale(
        partial: jax.Array,
        residual: jax.Array,
        post_norm_weight: jax.Array,
        scale: jax.Array,
    ) -> jax.Array:
        return _shard_map_sublayer(
            mesh, lambda p, r, w, s: local(p, r, w, s, None)[0], 2, _SHARDED
        )(partial, residual, post_norm_weight, scale)

    def fake_residual(partial, residual: torch.Tensor, *weights):
        del partial, weights
        return torch.empty_like(residual)

    wire = "fp8" if fp8_wire else "bf16"
    eps_tag = _eps_op_tag(eps)
    if next_act_dtype == torch.float8_e4m3fn:

        def norm_add_next_norm_fp8(
            partial: jax.Array,
            residual: jax.Array,
            post_norm_weight: jax.Array,
            next_norm_weight: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            return _shard_map_sublayer(
                mesh, lambda p, r, w, n: local(p, r, w, None, n), 2, _QUANTIZED_OUT
            )(partial, residual, post_norm_weight, next_norm_weight)

        def norm_add_scale_next_norm_fp8(
            partial: jax.Array,
            residual: jax.Array,
            post_norm_weight: jax.Array,
            scale: jax.Array,
            next_norm_weight: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            return _shard_map_sublayer(mesh, local, 3, _QUANTIZED_OUT)(
                partial, residual, post_norm_weight, scale, next_norm_weight
            )

        def fake_quantized(partial, residual: torch.Tensor, *weights):
            del partial, weights
            return (
                torch.empty_like(residual),
                residual.new_empty(residual.shape, dtype=torch.uint8),
                residual.new_empty((residual.shape[0], 1), dtype=torch.float32),
            )

        op_defs: tuple[tuple[Callable, Callable, int, Any, str], ...] = (
            (
                norm_add_next_norm_fp8,
                fake_quantized,
                2,
                _QUANTIZED_OUT,
                f"norm_add_next_norm_{wire}_{eps_tag}",
            ),
            (
                norm_add_scale_next_norm_fp8,
                fake_quantized,
                3,
                _QUANTIZED_OUT,
                f"norm_add_scale_next_norm_{wire}_{eps_tag}",
            ),
            # Single-element tuple: torch_tpu iterates a bare PartitionSpec as a list.
            (
                norm_add_scale,
                fake_residual,
                2,
                (_SHARDED,),
                f"norm_add_scale_{wire}_{eps_tag}",
            ),
        )
    else:

        def norm_add_next_norm_bf16(
            partial: jax.Array,
            residual: jax.Array,
            post_norm_weight: jax.Array,
            next_norm_weight: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            return _shard_map_sublayer(
                mesh,
                lambda p, r, w, n: local(p, r, w, None, n)[:2],
                2,
                _BF16_NEXT_OUT,
            )(partial, residual, post_norm_weight, next_norm_weight)

        def norm_add_scale_next_norm_bf16(
            partial: jax.Array,
            residual: jax.Array,
            post_norm_weight: jax.Array,
            scale: jax.Array,
            next_norm_weight: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            return _shard_map_sublayer(
                mesh,
                lambda p, r, w, s, n: local(p, r, w, s, n)[:2],
                3,
                _BF16_NEXT_OUT,
            )(partial, residual, post_norm_weight, scale, next_norm_weight)

        def fake_bf16_next(partial, residual: torch.Tensor, *weights):
            del partial, weights
            return (torch.empty_like(residual), torch.empty_like(residual))

        op_defs = (
            (
                norm_add_next_norm_bf16,
                fake_bf16_next,
                2,
                _BF16_NEXT_OUT,
                f"norm_add_next_norm_{wire}_bf16act_{eps_tag}",
            ),
            (
                norm_add_scale_next_norm_bf16,
                fake_bf16_next,
                3,
                _BF16_NEXT_OUT,
                f"norm_add_scale_next_norm_{wire}_bf16act_{eps_tag}",
            ),
            (
                norm_add_scale,
                fake_residual,
                2,
                (_SHARDED,),
                f"norm_add_scale_{wire}_bf16act_{eps_tag}",
            ),
        )

    ops = []
    for fn, fake, num_weights, out_specs, op_suffix in op_defs:
        # pyrefly: ignore[no-matching-overload]
        op = pallas.jax_op(
            f"pallas::dense_sp_rs_{op_suffix}",
            fn,
            mesh=mesh,
            input_partition_specs=(_SHARDED, _SHARDED, *(_REPLICATED,) * num_weights),
            output_partition_specs=out_specs,
        )
        op.register_fake(fake)
        ops.append(op)

    built = FusedSublayerCollectives(
        *ops, outputs_gathered=False, next_act_dtype=next_act_dtype
    )
    _STREAM_COLLECTIVES_CACHE[key] = built
    return built


def build_oneshot_sublayer_collectives(
    eps: float,
) -> FusedSublayerCollectives | None:
    """Builds single-shot VMEM RS + epilogue + AG ops for `128 <= tokens < 1024`."""
    resolved = _resolve_tp_mesh_and_order()
    if resolved is None:
        return None
    mesh, order, device_ids = resolved
    key = (device_ids, order, eps)
    cached = _ONESHOT_COLLECTIVES_CACHE.get(key)
    if cached is not None:
        return cached

    def local(partial, residual, post_norm_weight, scale, next_norm_weight):
        return kernel_oneshot_rs_ag.reduce_scatter_epilogue_oneshot_local(
            partial,
            residual,
            post_norm_weight=post_norm_weight,
            scale=scale,
            next_norm_weight=next_norm_weight,
            eps=eps,
            axis_name=TP_AXIS_NAME,
            order=order,
        )

    def oneshot_norm_add_next_norm(
        partial: jax.Array,
        residual: jax.Array,
        post_norm_weight: jax.Array,
        next_norm_weight: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return _shard_map_sublayer(
            mesh,
            lambda p, r, w, n: local(p, r, w, None, n),
            2,
            _ONESHOT_QUANTIZED_OUT,
        )(partial, residual, post_norm_weight, next_norm_weight)

    def oneshot_norm_add_scale_next_norm(
        partial: jax.Array,
        residual: jax.Array,
        post_norm_weight: jax.Array,
        scale: jax.Array,
        next_norm_weight: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return _shard_map_sublayer(mesh, local, 3, _ONESHOT_QUANTIZED_OUT)(
            partial, residual, post_norm_weight, scale, next_norm_weight
        )

    def oneshot_norm_add_scale(
        partial: jax.Array,
        residual: jax.Array,
        post_norm_weight: jax.Array,
        scale: jax.Array,
    ) -> jax.Array:
        return _shard_map_sublayer(
            mesh, lambda p, r, w, s: local(p, r, w, s, None)[1], 2, _REPLICATED
        )(partial, residual, post_norm_weight, scale)

    def fake_quantized(partial: torch.Tensor, residual: torch.Tensor, *weights):
        del weights
        return (
            torch.empty_like(residual),
            partial.new_empty(partial.shape, dtype=torch.uint8),
            partial.new_empty((partial.shape[0], 1), dtype=torch.float32),
        )

    def fake_gathered_residual(partial: torch.Tensor, residual: torch.Tensor, *weights):
        del residual, weights
        return torch.empty_like(partial)

    eps_tag = _eps_op_tag(eps)
    ops = []
    for fn, fake, num_weights, out_specs, op_tag in (
        (
            oneshot_norm_add_next_norm,
            fake_quantized,
            2,
            _ONESHOT_QUANTIZED_OUT,
            "norm_add_next_norm",
        ),
        (
            oneshot_norm_add_scale_next_norm,
            fake_quantized,
            3,
            _ONESHOT_QUANTIZED_OUT,
            "norm_add_scale_next_norm",
        ),
        (
            oneshot_norm_add_scale,
            fake_gathered_residual,
            2,
            (_REPLICATED,),
            "norm_add_scale",
        ),
    ):
        # pyrefly: ignore[no-matching-overload]
        op = pallas.jax_op(
            f"pallas::dense_sp_oneshot_{op_tag}_{eps_tag}",
            fn,
            mesh=mesh,
            input_partition_specs=(_SHARDED, _SHARDED, *(_REPLICATED,) * num_weights),
            output_partition_specs=out_specs,
        )
        op.register_fake(fake)
        ops.append(op)

    built = FusedSublayerCollectives(
        *ops, outputs_gathered=True, next_act_dtype=torch.float8_e4m3fn
    )
    _ONESHOT_COLLECTIVES_CACHE[key] = built
    return built
