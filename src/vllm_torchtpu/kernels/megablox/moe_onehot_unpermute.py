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
"""Blockwise implementation of the existing MoE output one-hot combine."""

import functools
import operator

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_ROUTE_BLOCK_SIZE = 128
_VMEM_WORKSPACE_MARGIN_BYTES = 4 * 1024 * 1024


def _explicit_vmem_bytes(route_capacity: int, hidden_size: int, num_tokens: int) -> int:
    """Size of the explicitly allocated VMEM buffers used by the kernel."""
    owner_and_store = num_tokens * hidden_size * (4 + 2)
    double_buffered_routes = 2 * _ROUTE_BLOCK_SIZE * hidden_size * 2
    route_metadata = route_capacity * (4 + 4)
    return owner_and_store + double_buffered_routes + route_metadata


def can_use_blockwise_onehot_unpermute(
    route_output: jax.Array,
    token_indices_sorted: jax.Array,
    topk_weights_sorted: jax.Array,
    valid_count: jax.Array,
    *,
    num_tokens: int,
) -> bool:
    """Return whether this one-hot shape satisfies kernel safety constraints."""
    if route_output.ndim != 2 or route_output.dtype != jnp.bfloat16:
        return False
    route_capacity, hidden_size = route_output.shape
    if token_indices_sorted.shape != (route_capacity,):
        return False
    if token_indices_sorted.dtype != jnp.int32:
        return False
    if topk_weights_sorted.shape != (route_capacity,):
        return False
    if topk_weights_sorted.dtype != jnp.bfloat16:
        return False
    if valid_count.shape != (1,) or valid_count.dtype != jnp.int32:
        return False

    tpu_info = pltpu.get_tpu_info()
    if route_capacity % _ROUTE_BLOCK_SIZE:
        return False
    if hidden_size % tpu_info.mxu_column_size:
        return False
    if num_tokens % tpu_info.get_sublane_tiling(route_output.dtype):
        return False
    vmem_limit_bytes = int(tpu_info.vmem_capacity_bytes * 0.9)
    required_vmem_bytes = (
        _explicit_vmem_bytes(route_capacity, hidden_size, num_tokens)
        + _VMEM_WORKSPACE_MARGIN_BYTES
    )
    return required_vmem_bytes <= vmem_limit_bytes


def _make_route_block_selector(
    token_ids: jax.Array,
    route_weights: jax.Array,
    route_valid: jax.Array,
    *,
    num_tokens: int,
) -> jax.Array:
    """Build a token-by-route selector for one contiguous route block."""
    owner_ids = jnp.arange(num_tokens, dtype=jnp.int32)[:, None]
    token_in_range = jnp.logical_and(token_ids >= 0, token_ids < num_tokens)
    route_valid = jnp.logical_and(route_valid, token_in_range)
    safe_token_ids = jnp.where(route_valid, token_ids, -1)
    safe_route_weights = jnp.where(route_valid, route_weights, 0)
    matches = owner_ids == safe_token_ids[None, :]
    return matches.astype(route_weights.dtype) * safe_route_weights[None, :]


def _accumulate_route_block(
    route_rows: jax.Array,
    owner_ref: jax.Array,
    token_ids: jax.Array,
    route_weights: jax.Array,
    route_valid: jax.Array,
    route_row_valid: jax.Array,
    *,
    num_tokens: int,
) -> None:
    """Accumulate one contiguous BF16 route block into its token owners."""
    selector = _make_route_block_selector(
        token_ids,
        route_weights.astype(jnp.bfloat16),
        route_valid,
        num_tokens=num_tokens,
    ).astype(jnp.bfloat16)

    # Standard GMM2 initializes the valid prefix but leaves its capacity tail
    # undefined. A zero selector does not suppress a NaN tail row, so clear
    # invalid rows before the dot. The predicate is constructed directly at
    # the 2-D route shape because Mosaic cannot lay out a [128] -> [128, 1]
    # predicate reshape.
    route_rows = jnp.where(route_row_valid, route_rows, jnp.zeros_like(route_rows))

    mxu_size = pltpu.get_tpu_info().mxu_column_size
    for n_start in range(0, route_rows.shape[-1], mxu_size):
        n_end = n_start + mxu_size
        block_owner = jnp.matmul(
            selector,
            route_rows[:, n_start:n_end],
            preferred_element_type=jnp.float32,
        )
        owner_ref[:, n_start:n_end] = owner_ref[:, n_start:n_end] + block_owner


def _pipeline_body(
    route_block_ref: jax.Array,
    owner_ref: jax.Array,
    *,
    token_indices_sorted_ref: jax.Array,
    topk_weights_sorted_ref: jax.Array,
    valid_count_ref: jax.Array,
    route_capacity: int,
    num_tokens: int,
) -> None:
    """Accumulate one HBM-streamed route block into a VMEM owner buffer."""
    block_id = pl.program_id(0)
    route_start = pl.multiple_of(block_id * _ROUTE_BLOCK_SIZE, _ROUTE_BLOCK_SIZE)
    route_slice = pl.ds(route_start, _ROUTE_BLOCK_SIZE)
    valid_count = jnp.clip(valid_count_ref[0], 0, route_capacity)
    route_lanes = jnp.arange(_ROUTE_BLOCK_SIZE, dtype=jnp.int32)
    route_valid = route_start + route_lanes < valid_count
    route_rows = route_block_ref[...]
    route_row_valid = (
        route_start
        + lax.broadcasted_iota(
            jnp.int32,
            route_rows.shape,
            0,
        )
        < valid_count
    )
    _accumulate_route_block(
        route_rows,
        owner_ref,
        token_indices_sorted_ref[route_slice],
        topk_weights_sorted_ref[route_slice],
        route_valid,
        route_row_valid,
        num_tokens=num_tokens,
    )


def _kernel(
    # Scalar prefetch
    valid_count_ref: jax.Array,
    # In
    token_indices_sorted_ref: jax.Array,
    topk_weights_sorted_ref: jax.Array,
    route_hbm_ref: jax.Array,
    # Out
    owner_out_ref: jax.Array,
    # Scratch
    owner_ref: jax.Array,
    store_ref: jax.Array,
    owner_store_sem_ref: jax.Array,
    *,
    route_capacity: int,
    hidden_size: int,
    num_tokens: int,
) -> None:
    """Stream valid standard-GMM2 rows from HBM and combine token owners."""
    valid_count = jnp.clip(valid_count_ref[0], 0, route_capacity)
    # Keep one all-masked iteration for valid_count == 0 so the dynamic grid is
    # always non-empty and the result is deterministically zero.
    num_route_blocks = jnp.maximum(
        1,
        pl.cdiv(valid_count, _ROUTE_BLOCK_SIZE),
    )
    owner_ref[...] = jnp.zeros_like(owner_ref)

    route_block_spec = pl.BlockSpec(
        (_ROUTE_BLOCK_SIZE, hidden_size),
        lambda block_id: (block_id, 0),
        pipeline_mode=pl.Buffered(buffer_count=2, use_lookahead=True),
        memory_space=pltpu.VMEM,
    )
    pipeline_fn = pltpu.emit_pipeline(
        functools.partial(
            _pipeline_body,
            token_indices_sorted_ref=token_indices_sorted_ref,
            topk_weights_sorted_ref=topk_weights_sorted_ref,
            valid_count_ref=valid_count_ref,
            route_capacity=route_capacity,
            num_tokens=num_tokens,
        ),
        grid=(num_route_blocks,),
        in_specs=(route_block_spec,),
        out_specs=(),
        dimension_semantics=("arbitrary",),
    )
    pipeline_fn(route_hbm_ref, scratches=[owner_ref])

    store_ref[...] = owner_ref[...].astype(jnp.bfloat16)
    owner_store = pltpu.make_async_copy(
        store_ref,
        owner_out_ref,
        owner_store_sem_ref.at[0],
    )
    owner_store.start()
    owner_store.wait()


def _validate_inputs(
    route_output: jax.Array,
    token_indices_sorted: jax.Array,
    topk_weights_sorted: jax.Array,
    valid_count: jax.Array,
    *,
    num_tokens: int,
) -> tuple[int, int, int]:
    try:
        num_tokens = operator.index(num_tokens)
    except TypeError as exc:
        raise TypeError("num_tokens must be a static Python integer") from exc

    if not can_use_blockwise_onehot_unpermute(
        route_output,
        token_indices_sorted,
        topk_weights_sorted,
        valid_count,
        num_tokens=num_tokens,
    ):
        raise ValueError(
            "blockwise one-hot unpermute does not support the requested "
            "dtype, shape, alignment, or VMEM requirement; got "
            f"{(pltpu.get_tpu_info().generation, *route_output.shape, num_tokens)}"
        )
    route_capacity, hidden_size = route_output.shape
    return route_capacity, hidden_size, num_tokens


@jax.jit(static_argnames=("num_tokens",))
def blockwise_onehot_unpermute(
    route_output: jax.Array,
    token_indices_sorted: jax.Array,
    topk_weights_sorted: jax.Array,
    valid_count: jax.Array,
    *,
    num_tokens: int,
) -> jax.Array:
    """Combine the existing output one-hot in 128-route HBM blocks.

    The valid prefix of ``route_output`` must contain standard GMM2 results in
    expert-sorted order. Capacity-tail rows may be uninitialized.
    """
    route_capacity, hidden_size, num_tokens = _validate_inputs(
        route_output,
        token_indices_sorted,
        topk_weights_sorted,
        valid_count,
        num_tokens=num_tokens,
    )
    topk_weights_vmem = topk_weights_sorted.astype(jnp.float32)
    out_shape = jax.ShapeDtypeStruct(
        (num_tokens, hidden_size),
        jnp.bfloat16,
    )
    tpu_info = pltpu.get_tpu_info()
    vmem_limit_bytes = int(tpu_info.vmem_capacity_bytes * 0.9)

    return pl.pallas_call(
        functools.partial(
            _kernel,
            route_capacity=route_capacity,
            hidden_size=hidden_size,
            num_tokens=num_tokens,
        ),
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.VMEM),
                pl.BlockSpec(memory_space=pltpu.VMEM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM((num_tokens, hidden_size), jnp.float32),
                pltpu.VMEM((num_tokens, hidden_size), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((1,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit_bytes,
            disable_bounds_checks=True,
        ),
        name=(
            f"blockwise_onehot_unpermute-r_{route_capacity}"
            f"-h_{hidden_size}-t_{num_tokens}-rb_{_ROUTE_BLOCK_SIZE}"
        ),
        metadata={
            "onehot_unpermute.route_block_size": _ROUTE_BLOCK_SIZE,
            "onehot_unpermute.route_capacity": route_capacity,
            "onehot_unpermute.hidden_size": hidden_size,
            "onehot_unpermute.num_tokens": num_tokens,
        },
    )(
        valid_count,
        token_indices_sorted,
        topk_weights_vmem,
        route_output,
    )
