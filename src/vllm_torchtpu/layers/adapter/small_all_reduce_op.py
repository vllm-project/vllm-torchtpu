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
"""Torch bridges for small TP all-reduce and fused row-parallel matmul + AR."""

import inspect
from typing import Any

import jax
import torch
from jax.sharding import PartitionSpec
from vllm.model_executor.layers.linear import RowParallelLinear

from vllm_torchtpu.compilation import shape_variants
from vllm_torchtpu.distributed.ep_mesh import (
    EP_AXIS_NAME,
    build_ep_mesh,
    ep_device_ids,
    ep_mesh_index,
    get_ep_group,
)
from vllm_torchtpu.distributed.mesh_utils import get_pipeline_stage
from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op
from vllm_torchtpu.kernels.experimental.small_collectives import all_reduce
from vllm_torchtpu.kernels.experimental.small_collectives.common import LANES
from vllm_torchtpu.kernels.quantized_matmul import fp8_oproj_all_reduce
from vllm_torchtpu.layers.adapter import token_padding
from vllm_torchtpu.layers.adapter.linear_common import (
    FP8_PALLAS_N_ATTR,
    FP8_PALLAS_WEIGHT,
    KEEP_VLLM_LAYOUT_ATTR,
    PREQUANT_ATTR,
)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.glm_dsa_dispatch import glm_dsa_v7x_active

logger = init_logger(__name__)

MAX_ROWS = 32
MIN_CORES = 8
RMSNORM_MIN_M = 5


def max_rows() -> int:
    return MAX_ROWS


_AR_OP = None
_AR_VALID_OP = None
_VALID_TILE_ROWS = 0
_RANK: torch.Tensor | None = None
_CORES = 0
_MESH = None
_MESH_RANKS: tuple[int, ...] = ()
_MM_AR_OP = None
_MM_AR_ONE_ROW_OP = None
_MM_AR_RN_OPS: dict[float, Any] = {}
_MM_AR_RN_ONE_ROW_OPS: dict[float, Any] = {}
_PENDING_RMSNORM_EPS: set[float] = set()


def _build_ep_op(mesh, name: str, arg_names: tuple[str, ...], body, fake):
    spec = PartitionSpec(EP_AXIS_NAME)
    specs = (spec,) * len(arg_names)

    def fn(*args: jax.Array) -> jax.Array:
        return jax.shard_map(
            body, mesh=mesh, in_specs=specs, out_specs=spec, check_vma=False
        )(*args)

    params = [
        inspect.Parameter(
            n, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=jax.Array
        )
        for n in arg_names
    ]
    fn.__signature__ = inspect.Signature(params, return_annotation=jax.Array)  # type: ignore[attr-defined]
    return sharded_jax_op(
        name,
        fn,
        mesh=mesh,
        input_partition_specs=specs,
        output_partition_specs=spec,
        fake=fake,
    )


def _fake_same(x, *_):
    return torch.empty_like(x)


def _fake_mm(x, w, *_):
    return x.new_empty((x.shape[0], w.shape[1]))


def _build_all_reduce(mesh):
    n = mesh.shape[EP_AXIS_NAME]
    return _build_ep_op(
        mesh,
        "pallas::small_all_reduce",
        ("x", "rank"),
        lambda b, r: all_reduce.small_all_reduce(
            b, num_cores=n, axis=EP_AXIS_NAME, rank=r
        ),
        _fake_same,
    )


def _build_valid_rows_all_reduce(mesh, tile_rows: int):
    n = mesh.shape[EP_AXIS_NAME]
    return _build_ep_op(
        mesh,
        "pallas::small_all_reduce_valid_rows",
        ("x", "n_valid", "rank"),
        lambda b, nv, r: all_reduce.small_all_reduce_valid_rows(
            b, nv, r, num_cores=n, axis=EP_AXIS_NAME, tile_rows=tile_rows
        ),
        _fake_same,
    )


def _build_matmul_all_reduce(mesh, waves: bool, *, one_row: bool = False):
    n = mesh.shape[EP_AXIS_NAME]
    if one_row:
        return _build_ep_op(
            mesh,
            "pallas::fp8_matmul_all_reduce_one_row",
            ("x", "w", "w_s", "n_valid", "rank"),
            lambda x, w, ws, nv, r: fp8_oproj_all_reduce.fp8_matmul_all_reduce(
                x, w, ws, r, num_cores=n, axis=EP_AXIS_NAME, waves=waves, n_valid=nv
            ),
            _fake_mm,
        )
    return _build_ep_op(
        mesh,
        "pallas::fp8_matmul_all_reduce",
        ("x", "w", "w_s", "rank"),
        lambda x, w, ws, r: fp8_oproj_all_reduce.fp8_matmul_all_reduce(
            x, w, ws, r, num_cores=n, axis=EP_AXIS_NAME, waves=waves
        ),
        _fake_mm,
    )


def _build_matmul_all_reduce_one_row(mesh, waves: bool):
    return _build_matmul_all_reduce(mesh, waves, one_row=True)


def _build_matmul_all_reduce_rmsnorm(mesh, waves: bool, eps: float, one_row: bool):
    """Fused o_proj + all-reduce + residual_add + RMSNorm over ``mesh``."""
    n = mesh.shape[EP_AXIS_NAME]
    spec = PartitionSpec(EP_AXIS_NAME)
    tag = f"{float(eps):.3e}".replace(".", "p").replace("-", "m").replace("+", "")
    eps_f = float(eps)

    if one_row:

        def body_one(x, w, w_s, n_valid, rank, residual, norm_w):
            return fp8_oproj_all_reduce.fp8_matmul_all_reduce(
                x,
                w,
                w_s,
                rank,
                num_cores=n,
                axis=EP_AXIS_NAME,
                waves=waves,
                n_valid=n_valid,
                residual=residual,
                norm_w=norm_w,
                eps=eps_f,
            )

        def fn_one(
            x: jax.Array,
            w: jax.Array,
            w_s: jax.Array,
            n_valid: jax.Array,
            rank: jax.Array,
            residual: jax.Array,
            norm_w: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            return jax.shard_map(
                body_one,
                mesh=mesh,
                in_specs=(spec,) * 7,
                out_specs=(spec, spec),
                check_vma=False,
            )(x, w, w_s, n_valid, rank, residual, norm_w)

        def fake_one(x, w, w_s, n_valid, rank, residual, norm_w):
            return x.new_empty((x.shape[0], w.shape[1])), torch.empty_like(residual)

        return sharded_jax_op(
            f"pallas::fp8_matmul_all_reduce_one_row_rmsnorm_{tag}",
            fn_one,
            mesh=mesh,
            input_partition_specs=(spec,) * 7,
            output_partition_specs=(spec, spec),
            fake=fake_one,
        )

    def body(x, w, w_s, rank, residual, norm_w):
        return fp8_oproj_all_reduce.fp8_matmul_all_reduce(
            x,
            w,
            w_s,
            rank,
            num_cores=n,
            axis=EP_AXIS_NAME,
            waves=waves,
            residual=residual,
            norm_w=norm_w,
            eps=eps_f,
        )

    def fn(
        x: jax.Array,
        w: jax.Array,
        w_s: jax.Array,
        rank: jax.Array,
        residual: jax.Array,
        norm_w: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(spec,) * 6,
            out_specs=(spec, spec),
            check_vma=False,
        )(x, w, w_s, rank, residual, norm_w)

    def fake(x, w, w_s, rank, residual, norm_w):
        return x.new_empty((x.shape[0], w.shape[1])), torch.empty_like(residual)

    return sharded_jax_op(
        f"pallas::fp8_matmul_all_reduce_rmsnorm_{tag}",
        fn,
        mesh=mesh,
        input_partition_specs=(spec,) * 6,
        output_partition_specs=(spec, spec),
        fake=fake,
    )


def prepare_rmsnorm_op(eps: float) -> None:
    """Register (and build if prebuilt) the fused o_proj AR + RMSNorm op for ``eps``."""
    eps_f = float(eps)
    _PENDING_RMSNORM_EPS.add(eps_f)
    if _MM_AR_OP is not None and _MESH is not None and eps_f not in _MM_AR_RN_OPS:
        _MM_AR_RN_OPS[eps_f] = _build_matmul_all_reduce_rmsnorm(
            _MESH, waves=True, eps=eps_f, one_row=False
        )
        if _MM_AR_ONE_ROW_OP is not None:
            _MM_AR_RN_ONE_ROW_OPS[eps_f] = _build_matmul_all_reduce_rmsnorm(
                _MESH, waves=True, eps=eps_f, one_row=True
            )


def prebuild() -> bool:
    global _AR_OP, _AR_VALID_OP, _VALID_TILE_ROWS, _RANK, _CORES, _MESH, _MESH_RANKS
    global _MM_AR_OP, _MM_AR_ONE_ROW_OP
    if _AR_OP is not None:
        return True
    if not glm_dsa_v7x_active():
        return False
    if not torch.distributed.is_initialized():
        return False
    device_ids = ep_device_ids()
    if device_ids is None or len(device_ids) < MIN_CORES:
        return False
    world_size = int(torch.distributed.get_world_size())
    _, pp_stages = get_pipeline_stage()
    if len(device_ids) * pp_stages != world_size:
        return False
    mesh = build_ep_mesh()
    if mesh is None:
        return False
    cores = int(mesh.shape[EP_AXIS_NAME])
    if cores < MIN_CORES:
        return False
    from jax.experimental.pallas import tpu as pltpu
    from vllm.distributed.parallel_state import get_tp_group

    ep_group = get_ep_group()
    assert ep_group is not None
    _MESH = mesh
    _CORES = cores
    _MESH_RANKS = tuple(int(r) for r in ep_group.ranks)
    _RANK = torch.tensor([ep_mesh_index()], dtype=torch.int32, device="tpu")
    _AR_OP = _build_all_reduce(mesh)
    info = pltpu.get_tpu_info()
    _VALID_TILE_ROWS = int(info.get_sublane_tiling(jax.numpy.bfloat16))
    _AR_VALID_OP = _build_valid_rows_all_reduce(mesh, _VALID_TILE_ROWS)
    tp_all_reduce = tuple(int(r) for r in get_tp_group().ranks) == _MESH_RANKS
    if tp_all_reduce:
        _MM_AR_OP = _build_matmul_all_reduce(mesh, waves=True)
        _MM_AR_ONE_ROW_OP = _build_matmul_all_reduce_one_row(mesh, waves=True)
        for eps_f in sorted(_PENDING_RMSNORM_EPS):
            if eps_f not in _MM_AR_RN_OPS:
                _MM_AR_RN_OPS[eps_f] = _build_matmul_all_reduce_rmsnorm(
                    mesh, waves=True, eps=eps_f, one_row=False
                )
                _MM_AR_RN_ONE_ROW_OPS[eps_f] = _build_matmul_all_reduce_rmsnorm(
                    mesh, waves=True, eps=eps_f, one_row=True
                )
    logger.info(
        "Small collectives armed | cores=%d | max_rows=%d | tp_all_reduce=%s"
        " | valid_tile_rows=%d",
        _CORES,
        max_rows(),
        tp_all_reduce,
        _VALID_TILE_ROWS,
    )
    return True


def _fits(x: torch.Tensor, width_multiple: int) -> bool:
    return (
        torch.compiler.is_compiling()
        and shape_variants.in_warmup()
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and x.shape[1] % width_multiple == 0
    )


def _padding_mask() -> torch.Tensor | None:
    if _AR_VALID_OP is None or token_padding._padding_state is None:
        return None
    return token_padding._padding_state.local_padding_mask


def retain(out: torch.Tensor) -> torch.Tensor:
    """Attach a dead-code-safe zero read from `_RANK` (and the padding mask if
    present) so every token-bucket warmup trace lifts identical buffers."""
    if _RANK is None or not shape_variants.in_warmup():
        return out
    value = _RANK[0].to(out.dtype)
    mask = _padding_mask()
    if mask is not None:
        value = value + mask[0].to(out.dtype)
    return out + (value - value)


def _valid_rows(rows: int, mask: torch.Tensor) -> torch.Tensor:
    padding = mask[:rows].to(torch.int32).sum()
    return (rows - padding).reshape(1).to(torch.int32)


def _small_all_reduce(x: torch.Tensor) -> torch.Tensor:
    assert _AR_OP is not None
    mask = _padding_mask()
    if mask is None:
        return _AR_OP(x, _RANK)
    rows, hidden = x.shape
    if all_reduce.fold_geometry(rows, hidden, _CORES, _VALID_TILE_ROWS) is not None:
        assert _AR_VALID_OP is not None
        return _AR_VALID_OP(x, _valid_rows(rows, mask), _RANK)
    return _AR_OP(retain(x), _RANK)


def all_reduce_or(x: torch.Tensor, ranks, fallback) -> torch.Tensor:
    """Sum `x` over `ranks` via small Pallas all-reduce when eligible, else
    `fallback` (riding `retain(x)` on the input so XLA fuses the zero read)."""
    if _AR_OP is None or tuple(ranks) != _MESH_RANKS:
        return fallback(x)
    if not _fits(x, _CORES * LANES):
        return fallback(x)
    if x.shape[0] <= max_rows():
        return _small_all_reduce(x)
    return fallback(retain(x))


def _fused_layer_weights(layer: torch.nn.Module):
    if not isinstance(layer, RowParallelLinear):
        return None
    if (
        FP8_PALLAS_N_ATTR not in layer.__dict__
        or FP8_PALLAS_WEIGHT in layer._parameters
        or bool(layer.__dict__.get(KEEP_VLLM_LAYOUT_ATTR, False))
        or PREQUANT_ATTR in layer.__dict__
        or layer.bias is not None
        or not layer.reduce_results
        or layer.tp_size != _CORES
    ):
        return None
    w = layer._parameters.get("weight")
    w_s = layer._parameters.get("weight_scale")
    if (
        not isinstance(w, torch.Tensor)
        or not isinstance(w_s, torch.Tensor)
        or w.dim() != 2
        or w_s.dim() != 1
    ):
        return None
    if w.dtype != torch.float8_e4m3fn or w_s.shape[0] != w.shape[1]:
        return None
    return w, w_s


def row_parallel_or(x: torch.Tensor, layer: torch.nn.Module) -> torch.Tensor:
    if _MM_AR_OP is not None and _fits(x, LANES):
        weights = _fused_layer_weights(layer)
        if weights is not None:
            w, w_s = weights
            rows, k = x.shape
            if k == w.shape[0] and fp8_oproj_all_reduce.supports(
                rows, k, w.shape[1], _CORES
            ):
                mask = _padding_mask()
                if (
                    _MM_AR_ONE_ROW_OP is not None
                    and mask is not None
                    and fp8_oproj_all_reduce.one_row_supported(w.shape[1])
                ):
                    n_valid = _valid_rows(rows, mask)
                    return _MM_AR_ONE_ROW_OP(retain(x), w, w_s, n_valid, _RANK)
                return _MM_AR_OP(retain(x), w, w_s, _RANK)
    return layer(x)[0]


def can_fuse_rmsnorm(rows: int, layer: torch.nn.Module, norm: Any) -> bool:
    """Whether ``rows`` in [RMSNORM_MIN_M, MAX_M] can fuse ``layer`` + ``norm``."""
    if (
        _MM_AR_OP is None
        or not torch.compiler.is_compiling()
        or not shape_variants.in_warmup()
        or rows < RMSNORM_MIN_M
        or rows > fp8_oproj_all_reduce.MAX_M
        or norm is None
        or not getattr(norm, "has_weight", True)
        or getattr(norm, "variance_size_override", None) is not None
        or getattr(norm, "_prequant_slot", None) is not None
    ):
        return False
    norm_w = getattr(norm, "weight", None)
    if norm_w is None or norm_w.dim() != 1:
        return False
    eps_f = float(getattr(norm, "variance_epsilon", 0.0))
    if eps_f not in _MM_AR_RN_OPS:
        return False
    weights = _fused_layer_weights(layer)
    if weights is None:
        return False
    w, _ = weights
    return norm_w.shape[0] == w.shape[1] and fp8_oproj_all_reduce.supports(
        rows, w.shape[0], w.shape[1], _CORES
    )


def row_parallel_or_rmsnorm(
    x: torch.Tensor,
    layer: torch.nn.Module,
    residual: torch.Tensor,
    norm_w: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused ``o_proj + all-reduce + residual_add + RMSNorm`` for 8 <= m <= 64."""
    eps_f = float(eps)
    rn_op = _MM_AR_RN_OPS.get(eps_f)
    weights = _fused_layer_weights(layer) if _fits(x, LANES) else None
    if rn_op is not None and weights is not None:
        w, w_s = weights
        rows, k = x.shape
        if (
            RMSNORM_MIN_M <= rows <= fp8_oproj_all_reduce.MAX_M
            and k == w.shape[0]
            and fp8_oproj_all_reduce.supports(rows, k, w.shape[1], _CORES)
            and residual.shape == (rows, w.shape[1])
            and residual.dtype == x.dtype
            and norm_w.shape == (w.shape[1],)
        ):
            mask = _padding_mask()
            rn_one_op = _MM_AR_RN_ONE_ROW_OPS.get(eps_f)
            if (
                rn_one_op is not None
                and mask is not None
                and fp8_oproj_all_reduce.one_row_supported(w.shape[1])
            ):
                n_valid = _valid_rows(rows, mask)
                return rn_one_op(retain(x), w, w_s, n_valid, _RANK, residual, norm_w)
            return rn_op(retain(x), w, w_s, _RANK, residual, norm_w)
    raise RuntimeError(
        f"row_parallel_or_rmsnorm called for ineligible shape "
        f"{tuple(x.shape)} / eps={eps}"
    )
