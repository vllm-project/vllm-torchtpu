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

Wraps `hierrs_tc/kernel_stream.py` (streaming buckets `>= 128 * TP` on 4- or
8-TensorCore tpu7x TP groups, returning token-sharded outputs) and
`hierrs_tc/kernel_oneshot_rs_ag.py` (decode buckets `256 <= tokens < 1024` on
8-TensorCore tpu7x TP groups, returning rank-ordered gathered outputs). Because
torch custom ops require fixed tensor arities, one op is registered per operand
combination used by dense decoder sublayers.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import jax
import torch
import torch.distributed as dist
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas
from vllm.distributed import parallel_state

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.hier_reduce_scatter import (
    agreed_refusal_reason,
    unsupported_chip_version_reason,
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
# At 128 tokens on TP=8, the single-shot kernel's fixed barrier/pack overhead
# slightly trails XLA all-reduce (~1.8%), while >= 256 tokens wins by 8-10%.
MIN_ONESHOT_RS_AG_TOKENS = 256
ONESHOT_SUPPORTED_TP_SIZES = kernel_oneshot_rs_ag.SUPPORTED_NUM_DEVICES

# Any: torch_tpu annotates partition specs as axis-name tuples, while jax_op
# forwards PartitionSpec instances directly to NamedSharding at runtime.
_SHARDED: Any = P(TP_AXIS_NAME, None)
_REPLICATED: Any = P()
_QUANTIZED_OUT = (_SHARDED, _SHARDED, _SHARDED)
_ONESHOT_QUANTIZED_OUT = (_SHARDED, _REPLICATED, _REPLICATED)


@dataclasses.dataclass(frozen=True)
class FusedSublayerCollectives:
    """Custom ops fusing TP reduce-scatter with a decoder sublayer's epilogue."""

    # (partial, residual, post_norm_weight, next_norm_weight)
    #   -> (residual, x_q, x_scale)
    norm_add_next_norm: Callable
    # (partial, residual, post_norm_weight, scale, next_norm_weight)
    #   -> (residual, x_q, x_scale)
    norm_add_scale_next_norm: Callable
    # (partial, residual, post_norm_weight, scale) -> residual
    norm_add_scale: Callable
    # True when `x_q`, `x_scale`, and `norm_add_scale`'s output are already
    # gathered across all TP ranks in rank order.
    outputs_gathered: bool = False


_STREAM_COLLECTIVES_CACHE: dict[tuple, FusedSublayerCollectives] = {}
_ONESHOT_COLLECTIVES_CACHE: dict[tuple, FusedSublayerCollectives] = {}
_RESOLVED_TP_TOPOLOGY: tuple[Any, tuple[int, ...], tuple[int, ...]] | None = None
_REFUSED_REASON: str | None = None


def _eps_op_tag(eps: float) -> str:
    return f"{eps:.8g}".replace("-", "m").replace("+", "p").replace(".", "_")


def supports_streaming_sublayer(num_tokens: Any, hidden: int) -> bool:
    """Whether `[num_tokens, hidden]` fits the streaming reduce-scatter op."""
    if _RESOLVED_TP_TOPOLOGY is None:
        # Before the TP group is resolved, only `hidden` is being checked.
        return kernel_stream.supports_stream_epilogue(num_tokens, hidden)
    return kernel_stream.supports_stream_epilogue(
        num_tokens, hidden, num_devices=len(_RESOLVED_TP_TOPOLOGY[1])
    )


def supports_oneshot_sublayer(num_tokens: Any, hidden: int, tp_size: int) -> bool:
    """Whether `[num_tokens, hidden]` fits the single-shot VMEM RS + AG op.

    `tp_size` is only used until the TP group is resolved.
    """
    if _RESOLVED_TP_TOPOLOGY is not None:
        tp_size = len(_RESOLVED_TP_TOPOLOGY[1])
    return (
        num_tokens >= MIN_ONESHOT_RS_AG_TOKENS
        and kernel_oneshot_rs_ag.supports_oneshot_rs_ag(
            num_tokens, hidden, num_devices=tp_size
        )
    )


def _unsupported_tp_topology_reason() -> str | None:
    if not dist.is_available() or not dist.is_initialized():
        return "torch.distributed is not initialized"
    tp_group = parallel_state.get_tp_group()
    tp_size = int(tp_group.world_size)
    if tp_size not in kernel_stream.SUPPORTED_NUM_DEVICES:
        return f"the kernel requires TP=4 or TP=8, got TP={tp_size}"
    if tp_size != int(dist.get_world_size()):
        return "the TP group does not span the whole torch.distributed world"
    if supports_num_devices(tp_size):
        return unsupported_tpu7x_4x2_chip_topology_reason(require_paired_chips=True)
    # TP=4 is two dual-core chips. Under dense DP each replica is its own
    # world, so the host-wide 4-chip check does not apply. The kernel
    # addresses peers by mesh index and `_resolve_tp_mesh_and_order` still
    # requires device ids 0..3, where ids 2c and 2c+1 are chip c's TensorCores.
    return unsupported_chip_version_reason()


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


def _register_sublayer_ops(
    mesh: Any,
    name_prefix: str,
    name_suffix: str,
    variants: tuple[tuple[Callable, Callable, int, Any, str], ...],
) -> list[Callable]:
    """Registers one torch custom op per `(fn, fake, num_weights, out_specs, tag)`."""
    ops: list[Callable] = []
    for fn, fake, num_weights, out_specs, op_tag in variants:
        # pyrefly: ignore[no-matching-overload]
        op = pallas.jax_op(
            f"pallas::{name_prefix}_{op_tag}_{name_suffix}",
            fn,
            mesh=mesh,
            input_partition_specs=(_SHARDED, _SHARDED, *(_REPLICATED,) * num_weights),
            output_partition_specs=out_specs,
        )
        op.register_fake(fake)
        ops.append(op)
    return ops


def build_streaming_sublayer_collectives(
    eps: float, *, fp8_wire: bool
) -> FusedSublayerCollectives | None:
    """Builds streaming fused RS + epilogue ops for the active TP group."""
    resolved = _resolve_tp_mesh_and_order()
    if resolved is None:
        return None
    mesh, order, device_ids = resolved
    key = (device_ids, order, eps, fp8_wire)
    cached = _STREAM_COLLECTIVES_CACHE.get(key)
    if cached is not None:
        return cached

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
        )

    def norm_add_next_norm(
        partial: jax.Array,
        residual: jax.Array,
        post_norm_weight: jax.Array,
        next_norm_weight: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return _shard_map_sublayer(
            mesh, lambda p, r, w, n: local(p, r, w, None, n), 2, _QUANTIZED_OUT
        )(partial, residual, post_norm_weight, next_norm_weight)

    def norm_add_scale_next_norm(
        partial: jax.Array,
        residual: jax.Array,
        post_norm_weight: jax.Array,
        scale: jax.Array,
        next_norm_weight: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return _shard_map_sublayer(mesh, local, 3, _QUANTIZED_OUT)(
            partial, residual, post_norm_weight, scale, next_norm_weight
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

    def fake_quantized(partial, residual: torch.Tensor, *weights):
        del partial, weights
        return (
            torch.empty_like(residual),
            residual.new_empty(residual.shape, dtype=torch.uint8),
            residual.new_empty((residual.shape[0], 1), dtype=torch.float32),
        )

    def fake_residual(partial, residual: torch.Tensor, *weights):
        del partial, weights
        return torch.empty_like(residual)

    wire = "fp8" if fp8_wire else "bf16"
    order_tag = "_".join(str(r) for r in order)
    ops = _register_sublayer_ops(
        mesh,
        "dense_sp_rs",
        f"{wire}_{_eps_op_tag(eps)}_ord_{order_tag}",
        (
            (
                norm_add_next_norm,
                fake_quantized,
                2,
                _QUANTIZED_OUT,
                "norm_add_next_norm",
            ),
            (
                norm_add_scale_next_norm,
                fake_quantized,
                3,
                _QUANTIZED_OUT,
                "norm_add_scale_next_norm",
            ),
            # Single-element tuple: torch_tpu iterates a bare PartitionSpec as a list.
            (norm_add_scale, fake_residual, 2, (_SHARDED,), "norm_add_scale"),
        ),
    )

    built = FusedSublayerCollectives(*ops, outputs_gathered=False)
    _STREAM_COLLECTIVES_CACHE[key] = built
    return built


def build_oneshot_sublayer_collectives(
    eps: float,
) -> FusedSublayerCollectives | None:
    """Builds single-shot VMEM RS + epilogue + AG ops for `256 <= tokens < 1024`."""
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

    order_tag = "_".join(str(r) for r in order)
    ops = _register_sublayer_ops(
        mesh,
        "dense_sp_oneshot",
        f"{_eps_op_tag(eps)}_ord_{order_tag}",
        (
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
        ),
    )

    built = FusedSublayerCollectives(*ops, outputs_gathered=True)
    _ONESHOT_COLLECTIVES_CACHE[key] = built
    return built
