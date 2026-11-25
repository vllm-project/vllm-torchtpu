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
"""Deviceless TPU v7x lowering and static contract checks for Kimi-K3 decode.

Production geometry and mesh sources:
- Physical 2x2x4 TPU v7x slice (`collectives.py`, virtual `tpu7x_2x2x4`,
  `TP_SIZE = 32` ranks in `common/constants.py`).
- Full model depth `NUM_LAYERS = 93` (`common/constants.py`), `MLA_LAYERS = 24`
  (`layout.py`), and `decode_ops.decode_options(8)` (`alias_expert_scales=True`).
- Downscaled `LAYERS = 4` (layer 0 dense MLP + KDA, layers 1..2 KDA + MoE,
  layer 3 MLA + MoE) and `PAGES = 16` serve as AOT compile-speed fixtures.
"""

from __future__ import annotations

import base64
import dataclasses
import functools
import gc
import hashlib
import inspect
import json
import math
import multiprocessing
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict, cast

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from jax._src import source_info_util  # No public JAX API suppresses file paths in HLO.
from jax.experimental import topologies
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jaxlib.xla_client import hlo as jax_hlo

from vllm_torchtpu.models.kimi_k3.megakernel import (
    config as kernel_config,
)
from vllm_torchtpu.models.kimi_k3.megakernel import (
    decode_ops,
)
from vllm_torchtpu.models.kimi_k3.megakernel.common import (
    pool_alias as shared_pool_alias,
)
from vllm_torchtpu.models.kimi_k3.megakernel.common.constants import (
    AXIS_TP,
    CONV_HEAD_ROWS,
    CONV_TAPS,
    DEFAULT_VOCAB_SIZE,
    DP2_GROUPS,
    HEAD_DIM,
    KDA_HEADS,
    KDA_QKV_PARTS,
    MLA_LATENT_ROW_LANES,
    NUM_LAYERS,
    SUBLANE_ROWS,
    TP_SIZE,
    VREG_LANE_COUNT,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kernels import (
    decode_megakernel,
    load,
    mla_dcp,
    paged,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kernels.decode_megakernel import (
    _project_mla_value_up_heads,
)
from vllm_torchtpu.models.kimi_k3.megakernel.layout import MLA_LAYERS

if TYPE_CHECKING:
    from collections.abc import Callable

    from vllm_torchtpu.models.kimi_k3.megakernel.common.protocols import (
        PallasRefProtocol,
    )

# AOT compile-speed fixtures covering layer 0 dense MLP, KDA, MLA, and MoE.
LAYERS = 4
PAGES = 16
BATCH = 8
# reuse_row_plan activates at batch_size in (32, 64, 96, 128).
ROW_PLAN_BATCH_SIZE = 32
# Smallest KDA+MoE prefix that compiles stage_planned_rows on CPU.
ROW_PLAN_LAYERS = 2
# Decode recipe uses an MLA cache batch tile of 2 at batch 32.
ROW_PLAN_MLA_BATCH_TILE = 2
RANKS = TP_SIZE
DCP_POOL_LAYERS = MLA_LAYERS

_FLOAT32_BYTES = 4
_UINT32_BYTES = 4
_BF16_BYTES = 2
_UINT8_BYTES = 1
_MXFP4_VALUES_PER_WORD = 8
_MXFP4_VALUES_PER_SCALE = 32
_EP_SHARDS = 8
_ACTIVE_LATENT_UP_LANES = mla_dcp.HOST_LANES - 1
_FP8_POOL_PACKING = 4
_KDA_PROJECTION_PARTS = 4
_EXPECTED_MLA_WORKSPACE_FIELDS = 10
_EXPECTED_DECODE_LOCAL_CUSTOM_CALLS = 4
_EXPECTED_DECODE_PAGED_CUSTOM_CALLS = 6
_EXPECTED_COMMIT_LATENT_CUSTOM_CALLS = 4
_EXPECTED_LAZY_DATACLASS_COUNT = 7
_LAZY_RACE_WORKERS = 4
_MLA_KEY_HANDOFF_LANES = 2 * HEAD_DIM
_EXPECTED_WEIGHT_FAMILIES = 37
_GATE_UP_FACTORS = 2
_HIDDEN_HALVES = 2
_RANK_PAIR_SIZE = 2
_ROPE_PAD_FACTORS = 2
_KV_HEAD_PROJECTIONS = 2
_LAYER_NORM_PAIR = 2
_EMBEDDING_AND_LM_HEAD = 2
_DUMMY_LATENT_WIDTH = 8

# Weights stored as packed MXFP4 words, per-block scales, and FP32 norms/biases.
_UINT32_WEIGHTS = ("expert_gu", "expert_down")
_UINT8_WEIGHTS = ("expert_gus", "expert_ds")
_FLOAT32_WEIGHTS = (
    "router_bias",
    "latent_norm",
    "k_a_log",
    "k_dt",
    "attn_res",
    "mlp_res",
    "output_res",
)

# Decode options per mode: (attention_token_scatter, moe_token_scatter_output,
# moe_routed_reduce_scatter, sequence_parallel_pre_attention, aux_hidden_layers).
MODES: dict[str, tuple[bool, bool, bool, bool, tuple[int, ...]]] = {
    "plain": (False, False, False, False, ()),
    "scatter": (True, False, False, False, ()),
    "scatter_moe_out": (True, True, False, False, ()),
    "scatter_moe_reduce_scatter": (True, True, True, False, ()),
    "full_sequence_parallel": (True, True, True, True, ()),
    "aux_hidden": (False, False, False, False, (2,)),
}

# Reference canonical HLO oracle across all 8 decode regimes (7 decode_local
# modes plus B8 paged.decode_paged with alias_expert_scales=True) under jax
# 0.11.1: (batch_size, mode_key, byte_len, custom_calls, sha256_hi, sha256_lo).
# Regenerate after intentional outer-HLO or JAX pin changes by running:
#   JAX_PLATFORMS=cpu pytest -q \
#     tests/models/kimi_k3/megakernel/test_decode_lowering.py \
#     -k test_decode_canonical_hlo_sha256_oracle
# and verifying 0 B inner `kimi_decoder_stack` Mosaic body parity.
_CANONICAL_HLO_ORACLE_JAX_0_11_1: dict[str, tuple[int, str, int, int, str, str]] = {
    "batch1_plain": (
        1,
        "plain",
        1_102_290,
        _EXPECTED_DECODE_LOCAL_CUSTOM_CALLS,
        "4198288959f2e296d2f1598424eb9338",
        "d5da2b2d9ee9ea7fa005a3d8723b3997",
    ),
    "batch8_plain": (
        8,
        "plain",
        1_446_420,
        _EXPECTED_DECODE_LOCAL_CUSTOM_CALLS,
        "e503ed74cb6a945733587a3b06159a27",
        "12b3f08896fdaf68d6205a2458378028",
    ),
    "scatter": (
        8,
        "scatter",
        1_552_500,
        _EXPECTED_DECODE_LOCAL_CUSTOM_CALLS,
        "e7063d366debd9d81ba7af358c88058c",
        "be0e3c4496ce80bd203945bfc9c2eede",
    ),
    "scatter_moe_out": (
        8,
        "scatter_moe_out",
        1_572_284,
        _EXPECTED_DECODE_LOCAL_CUSTOM_CALLS,
        "7ecca1cc0a5864a9a617c6028f19353a",
        "4fd64abc0b3a1eec491183507ac20aef",
    ),
    "scatter_moe_reduce_scatter": (
        8,
        "scatter_moe_reduce_scatter",
        1_598_044,
        _EXPECTED_DECODE_LOCAL_CUSTOM_CALLS,
        "38f8500cb3c9f85f8ef33059e215fea0",
        "d346120f5c214643f33d6cf3a724e6ae",
    ),
    "full_sequence_parallel": (
        8,
        "full_sequence_parallel",
        1_610_144,
        _EXPECTED_DECODE_LOCAL_CUSTOM_CALLS,
        "30761f2fb5df3076b414c2410cf4e17d",
        "ccbbf382c3e87d73a0e8a05097e6fff6",
    ),
    "aux_hidden": (
        8,
        "aux_hidden",
        1_448_789,
        _EXPECTED_DECODE_LOCAL_CUSTOM_CALLS,
        "8ba06125b88e5f8f063b06eda0d5922f",
        "f6859bd6c419ae943c81b07f1411cdad",
    ),
    "batch8_paged_alias_expert_scales": (
        8,
        "paged_alias_expert_scales",
        1_775_589,
        _EXPECTED_DECODE_PAGED_CUSTOM_CALLS,
        "02a315ad90879cf9262791e7b17798de",
        "07c96c370ff6725833c24a18fa8006f3",
    ),
}

# Reference canonical HLO oracle across all 8 decode regimes under jax 0.11.2.
_CANONICAL_HLO_ORACLE_JAX_0_11_2: dict[str, tuple[int, str, int, int, str, str]] = {
    "batch1_plain": (
        1,
        "plain",
        1_090_121,
        _EXPECTED_DECODE_LOCAL_CUSTOM_CALLS,
        "3698df9a6e060f50d8182b135a15d692",
        "aeb2c2d9f69cad1a4f0e17d7f488f736",
    ),
    "batch8_plain": (
        8,
        "plain",
        1_436_796,
        _EXPECTED_DECODE_LOCAL_CUSTOM_CALLS,
        "64794fd510daf63de12ff82a9bbc21d7",
        "d0e2e1e6e1f8005c4db462ebbf15080b",
    ),
    "scatter": (
        8,
        "scatter",
        1_542_876,
        _EXPECTED_DECODE_LOCAL_CUSTOM_CALLS,
        "be283dc52e3f0522739746c0643a156e",
        "ec210f010c1f7753d941a24fc0157650",
    ),
    "scatter_moe_out": (
        8,
        "scatter_moe_out",
        1_562_660,
        _EXPECTED_DECODE_LOCAL_CUSTOM_CALLS,
        "0dfe3f0d933caff62b3e8d11093f79df",
        "c2ea4db0beee47d7d9293f7fb7fe5cc4",
    ),
    "scatter_moe_reduce_scatter": (
        8,
        "scatter_moe_reduce_scatter",
        1_588_420,
        _EXPECTED_DECODE_LOCAL_CUSTOM_CALLS,
        "a8a777b479bc80917cea149744b47a6c",
        "8d841ddb018a9d61fa06e19a2ae1c984",
    ),
    "full_sequence_parallel": (
        8,
        "full_sequence_parallel",
        1_607_256,
        _EXPECTED_DECODE_LOCAL_CUSTOM_CALLS,
        "f4cef6fe2f7cf8414b1a22bc18110e06",
        "2ff4d0a423fc87ffdb9dc9a056f2536a",
    ),
    "aux_hidden": (
        8,
        "aux_hidden",
        1_439_165,
        _EXPECTED_DECODE_LOCAL_CUSTOM_CALLS,
        "c66a1c1358b21bc255b433b5a201924f",
        "d77c16d92e54472043cea631272c32eb",
    ),
    "batch8_paged_alias_expert_scales": (
        8,
        "paged_alias_expert_scales",
        1_772_685,
        _EXPECTED_DECODE_PAGED_CUSTOM_CALLS,
        "cef6d2ddbdfb67e183c44867bf118d00",
        "f0f4db1bb934f9df72246057a75e7406",
    ),
}

_CANONICAL_HLO_ORACLES_BY_JAX_VERSION: dict[
    str, dict[str, tuple[int, str, int, int, str, str]]
] = {
    "0.11.1": _CANONICAL_HLO_ORACLE_JAX_0_11_1,
    "0.11.2": _CANONICAL_HLO_ORACLE_JAX_0_11_2,
}


def _active_canonical_hlo_oracle() -> dict[str, tuple[int, str, int, int, str, str]]:
    """Return the pinned canonical HLO oracle for the active JAX version."""
    oracle = _CANONICAL_HLO_ORACLES_BY_JAX_VERSION.get(jax.__version__)
    if oracle is None:
        supported = sorted(_CANONICAL_HLO_ORACLES_BY_JAX_VERSION)
        msg = (
            f"Unsupported jax version {jax.__version__!r} for canonical HLO "
            f"oracle (supported: {supported}); regenerate oracle entries "
            "after confirming 0 B inner kimi_decoder_stack Mosaic body parity."
        )
        raise AssertionError(msg)
    return oracle


def _ceil_multiple(value: int, alignment: int) -> int:
    """Round ``value`` up to a multiple of ``alignment``."""
    return -(-value // alignment) * alignment


def _derive_single_copy_weight_bytes(model_config: load.Config) -> int:
    """Derive exact per-rank single-copy weight footprint from ``load.Config``."""
    moe_layers = model_config.layers - 1
    mla_layers = len(model_config.full_attention)
    kda_layers = model_config.layers - mla_layers
    tp_moe = TP_SIZE // _EP_SHARDS
    local_experts = model_config.experts // _EP_SHARDS
    local_heads = model_config.heads // TP_SIZE
    local_head_dim = local_heads * model_config.head_dim
    half_hidden = model_config.dim // _HIDDEN_HALVES
    dense_local = _ceil_multiple(model_config.dense_hidden // TP_SIZE, VREG_LANE_COUNT)
    shared_local = model_config.shared_hidden // TP_SIZE
    latent_down_cols = _ceil_multiple(model_config.latent // TP_SIZE, VREG_LANE_COUNT)
    latent_up_cols = model_config.dim // _ACTIVE_LATENT_UP_LANES
    latent_norm_cols = TP_SIZE * latent_down_cols
    expert_local_hidden = model_config.expert_hidden // tp_moe

    moe_bytes = moe_layers * (
        model_config.dim * model_config.experts * _BF16_BYTES
        + model_config.experts * _FLOAT32_BYTES
        + model_config.dim * latent_down_cols * _BF16_BYTES
        + model_config.latent * latent_up_cols * _BF16_BYTES
        + latent_norm_cols * _FLOAT32_BYTES
        + model_config.dim * (_GATE_UP_FACTORS * shared_local) * _BF16_BYTES
        + shared_local * model_config.dim * _BF16_BYTES
        + local_experts
        * (model_config.latent // _MXFP4_VALUES_PER_WORD)
        * (_GATE_UP_FACTORS * expert_local_hidden)
        * _UINT32_BYTES
        + local_experts
        * (model_config.latent // _MXFP4_VALUES_PER_SCALE)
        * (_GATE_UP_FACTORS * expert_local_hidden)
        * _UINT8_BYTES
        + local_experts
        * (expert_local_hidden // _MXFP4_VALUES_PER_WORD)
        * model_config.latent
        * _UINT32_BYTES
        + local_experts
        * (expert_local_hidden // _MXFP4_VALUES_PER_SCALE)
        * model_config.latent
        * _UINT8_BYTES
    )
    kda_bytes = kda_layers * (
        model_config.dim * model_config.head_dim * _BF16_BYTES
        + model_config.dim
        * (_KDA_PROJECTION_PARTS * local_head_dim + model_config.head_dim)
        * _BF16_BYTES
        + model_config.head_dim * local_head_dim * _BF16_BYTES
        + (_RANK_PAIR_SIZE * local_head_dim) * half_hidden * _BF16_BYTES
        + model_config.conv_size
        * CONV_TAPS
        * local_heads
        * model_config.head_dim
        * _BF16_BYTES
        + local_heads * _FLOAT32_BYTES
        + local_heads * model_config.head_dim * _FLOAT32_BYTES
        + SUBLANE_ROWS * model_config.head_dim * _BF16_BYTES
    )
    mla_bytes = mla_layers * (
        model_config.dim * model_config.q_rank * _BF16_BYTES
        + model_config.dim
        * (model_config.kv_rank + _ROPE_PAD_FACTORS * model_config.qk_extra)
        * _BF16_BYTES
        + model_config.q_rank
        * (
            local_heads
            * (model_config.head_dim + _ROPE_PAD_FACTORS * model_config.qk_extra)
        )
        * _BF16_BYTES
        + model_config.kv_rank
        * (local_heads * _KV_HEAD_PROJECTIONS * model_config.head_dim)
        * _BF16_BYTES
        + model_config.dim * local_head_dim * _BF16_BYTES
        + (_RANK_PAIR_SIZE * local_head_dim) * half_hidden * _BF16_BYTES
        + model_config.q_rank * _BF16_BYTES
        + model_config.kv_rank * _BF16_BYTES
    )
    layer_norm_bytes = model_config.layers * (
        _LAYER_NORM_PAIR * model_config.dim * _FLOAT32_BYTES
        + _LAYER_NORM_PAIR * model_config.dim * _BF16_BYTES
    )
    boundary_bytes = (
        _EMBEDDING_AND_LM_HEAD
        * (model_config.vocab // TP_SIZE)
        * model_config.dim
        * _BF16_BYTES
        + model_config.dim * (_GATE_UP_FACTORS * dense_local) * _BF16_BYTES
        + dense_local * model_config.dim * _BF16_BYTES
        + model_config.dim * _FLOAT32_BYTES
        + model_config.dim * _BF16_BYTES
    )
    return moe_bytes + kda_bytes + mla_bytes + layer_norm_bytes + boundary_bytes


@functools.lru_cache(maxsize=1)
def _cached_tp_mesh() -> Mesh:
    """Build and cache a 32-device TPU v7x 2x2x4 mesh without TPU hardware."""
    devices = topologies.get_topology_desc(
        topology_name="tpu7x_2x2x4", platform="tpu"
    ).devices
    return Mesh(devices, (AXIS_TP,))


@pytest.fixture(scope="module")
def tp_mesh() -> Mesh:
    """Provide the 32-device TPU v7x 2x2x4 mesh fixture."""
    return _cached_tp_mesh()


def _abstract_weights(layers: int = LAYERS) -> dict[str, jax.ShapeDtypeStruct]:
    """Return per-rank weight shapes of a ``layers``-layer prefix in kernel layout."""
    weights: dict[str, jax.ShapeDtypeStruct] = {}
    for name, shape in load.selected_local_shapes(layers=layers).items():
        if name in _UINT32_WEIGHTS:
            dtype = jnp.uint32
        elif name in _UINT8_WEIGHTS:
            dtype = jnp.uint8
        elif name in _FLOAT32_WEIGHTS:
            dtype = jnp.float32
        else:
            dtype = jnp.bfloat16
        weights[name] = jax.ShapeDtypeStruct(shape, dtype)
    return weights


def _to_canonical_hlo_text(lowered: jax.stages.Lowered) -> str:
    """Serialize a lowered JAX computation into canonical HLO with backend configs."""
    compiler_ir = lowered.compiler_ir(dialect="hlo")
    assert compiler_ir is not None
    hlo_module = compiler_ir.as_hlo_module()
    options = jax_hlo.HloPrintOptions.canonical()
    options.print_backend_config = True
    options.print_metadata = False
    return hlo_module.to_string(options)


class _LocalDecodeOptions(TypedDict, total=False):
    """Typed keyword options forwarded to ``decode_megakernel.decode_local``."""

    attention_token_scatter: bool
    moe_token_scatter_output: bool
    moe_routed_reduce_scatter: bool
    sequence_parallel_pre_attention: bool
    aux_hidden_layers: tuple[int, ...]
    residual_reduction_order: Literal["reference", "row", "relaxed"]
    mla_cache_batch_tile_size: int | None
    alias_expert_scales: bool


def _abstract_local_decode_inputs(
    batch_size: int,
    layers: int = LAYERS,
) -> tuple[jax.ShapeDtypeStruct, ...]:
    """Build abstract local decode inputs and KV/recurrent state buffers."""
    return (
        jax.ShapeDtypeStruct((batch_size,), jnp.int32),
        jax.ShapeDtypeStruct((batch_size,), jnp.int32),
        jax.ShapeDtypeStruct(
            (
                batch_size,
                layers,
                CONV_TAPS,
                KDA_QKV_PARTS,
                CONV_HEAD_ROWS,
                HEAD_DIM,
            ),
            jnp.float32,
        ),
        jax.ShapeDtypeStruct(
            (batch_size, layers, KDA_HEADS, HEAD_DIM, HEAD_DIM),
            jnp.float32,
        ),
        jax.ShapeDtypeStruct(
            (PAGES, VREG_LANE_COUNT, _FP8_POOL_PACKING, MLA_LATENT_ROW_LANES),
            jnp.float8_e4m3fn,
        ),
        jax.ShapeDtypeStruct((batch_size, PAGES), jnp.int32),
    )


def _make_sharded_local_decode(
    mesh: Mesh,
    weights: dict[str, jax.ShapeDtypeStruct],
    batch_size: int,
    layers: int,
    decode_options: _LocalDecodeOptions,
) -> tuple[Callable[..., object], tuple[jax.ShapeDtypeStruct, ...]]:
    """Build abstract local decode inputs and wrap ``decode_local`` in ``shard_map``."""
    inputs = _abstract_local_decode_inputs(batch_size, layers)

    def step(*arrays: jax.Array) -> object:
        """Invoke ``decode_local`` on per-rank abstract arrays."""
        tokens, positions, conv_state, recurrent_state, pool, block_tables = arrays[:6]
        return decode_megakernel.decode_local(
            dict(zip(weights, arrays[6:], strict=True)),
            (conv_state, recurrent_state, pool),
            tokens,
            positions,
            batch_size=batch_size,
            sequence_rows=1,
            layers=layers,
            block_tables=block_tables,
            dcp_pool_map=(0,) * DCP_POOL_LAYERS,
            dcp_owner_table=tuple(range(RANKS)),
            attention_dp=1,
            **decode_options,
        )

    sharded = jax.shard_map(
        step,
        mesh=mesh,
        in_specs=(P(),) * (len(inputs) + len(weights)),
        out_specs=P(),
        check_vma=False,
    )
    return sharded, inputs


def _lower_decode_local_uncached(
    batch_size: int,
    mode: str,
) -> str:
    """Lower one ``decode_local`` mode without caching and restore global JAX config."""
    previous_tracebacks = jax.config.jax_include_full_tracebacks_in_locations
    exclude_paths = vars(source_info_util)["_exclude_paths"]
    previous_exclusions = list(exclude_paths)
    try:
        jax.config.update("jax_include_full_tracebacks_in_locations", bool(0))
        source_info_util.register_exclusion("/")
        active = kernel_config.active()
        assert (active.attention_dp, active.dcp_group) == (1, RANKS), active
        scatter, moe_out, moe_reduce_scatter, sequence_parallel, aux_layers = MODES[
            mode
        ]
        weights = _abstract_weights()
        decode_options: _LocalDecodeOptions = {
            "attention_token_scatter": scatter,
            "moe_token_scatter_output": moe_out,
            "moe_routed_reduce_scatter": moe_reduce_scatter,
            "sequence_parallel_pre_attention": sequence_parallel,
            "aux_hidden_layers": aux_layers,
        }
        sharded, inputs = _make_sharded_local_decode(
            _cached_tp_mesh(),
            weights,
            batch_size,
            LAYERS,
            decode_options,
        )
        return _to_canonical_hlo_text(
            jax.jit(sharded).lower(*inputs, *weights.values())
        )
    finally:
        jax.config.update(
            "jax_include_full_tracebacks_in_locations", previous_tracebacks
        )
        exclude_paths[:] = previous_exclusions


def _lower_paged_alias_expert_scales_uncached(batch_size: int) -> str:
    """Lower ``paged.decode_paged`` with aliased expert scales."""
    previous_tracebacks = jax.config.jax_include_full_tracebacks_in_locations
    exclude_paths = vars(source_info_util)["_exclude_paths"]
    previous_exclusions = list(exclude_paths)
    try:
        jax.config.update("jax_include_full_tracebacks_in_locations", bool(0))
        source_info_util.register_exclusion("/")
        active = kernel_config.active()
        assert (active.attention_dp, active.dcp_group) == (1, RANKS), active
        weights = _abstract_weights()
        decode_options = decode_ops.decode_options(batch_size)
        assert decode_options["alias_expert_scales"] is True
        slot_capacity = batch_size + 1

        def step(*arrays: jax.Array) -> object:
            """Invoke ``paged.decode_paged`` with aliased expert scales."""
            (
                tokens,
                positions,
                block_tables,
                slots,
                conv_slots,
                recurrent_slots,
                pool,
            ) = arrays[:7]
            return paged.decode_paged(
                dict(zip(weights, arrays[7:], strict=True)),
                conv_slots,
                recurrent_slots,
                (pool,),
                tokens,
                positions,
                block_tables,
                slots,
                batch_size=batch_size,
                layers=LAYERS,
                dcp_pool_map=(0,) * DCP_POOL_LAYERS,
                dcp_owner_table=tuple(range(RANKS)),
                **decode_options,
            )

        inputs = (
            jax.ShapeDtypeStruct((batch_size,), jnp.int32),
            jax.ShapeDtypeStruct((batch_size,), jnp.int32),
            jax.ShapeDtypeStruct((batch_size, PAGES), jnp.int32),
            jax.ShapeDtypeStruct((batch_size,), jnp.int32),
            jax.ShapeDtypeStruct(
                (
                    slot_capacity,
                    LAYERS,
                    CONV_TAPS,
                    KDA_QKV_PARTS,
                    CONV_HEAD_ROWS,
                    HEAD_DIM,
                ),
                jnp.float32,
            ),
            jax.ShapeDtypeStruct(
                (slot_capacity, LAYERS, KDA_HEADS, HEAD_DIM, HEAD_DIM),
                jnp.float32,
            ),
            jax.ShapeDtypeStruct(
                (PAGES, VREG_LANE_COUNT, _FP8_POOL_PACKING, MLA_LATENT_ROW_LANES),
                jnp.float8_e4m3fn,
            ),
        )
        sharded = jax.shard_map(
            step,
            mesh=_cached_tp_mesh(),
            in_specs=(P(),) * (len(inputs) + len(weights)),
            out_specs=P(),
            check_vma=False,
        )
        return _to_canonical_hlo_text(
            jax.jit(sharded).lower(*inputs, *weights.values())
        )
    finally:
        jax.config.update(
            "jax_include_full_tracebacks_in_locations", previous_tracebacks
        )
        exclude_paths[:] = previous_exclusions


@functools.lru_cache(maxsize=16)
def _lower_canonical_hlo(batch_size: int, mode: str) -> str:
    """Cached canonical HLO lowering across all 8 decode oracle regimes."""
    if mode == "paged_alias_expert_scales":
        return _lower_paged_alias_expert_scales_uncached(batch_size)
    return _lower_decode_local_uncached(batch_size, mode)


@pytest.mark.parametrize("mode", sorted(MODES))
def test_decode_megakernel_lowers_to_tpu_custom_call(tp_mesh: Mesh, mode: str) -> None:
    """Verify each decode mode lowers to a Mosaic custom call on TPU v7x."""
    assert tp_mesh.devices.size == RANKS
    hlo = _lower_canonical_hlo(BATCH, mode)
    assert "tpu_custom_call" in hlo


def test_decode_canonical_hlo_sha256_oracle() -> None:
    """Verify 0 B canonical HLO parity and pruned signatures across 8 regimes."""
    tp_mesh = _cached_tp_mesh()
    assert tp_mesh.devices.size == RANKS
    padded_rows_cls = decode_megakernel.PaddedBf16Rows
    assert isinstance(padded_rows_cls, type)
    assert issubclass(padded_rows_cls, shared_pool_alias.Reinterpret)
    expert_dot_fn = decode_megakernel.quantized_expert_dot
    expert_dot_params = inspect.signature(expert_dot_fn).parameters
    assert "output_size" not in expert_dot_params
    stack_params = inspect.signature(decode_megakernel.stack).parameters
    assert "gate_first_kda_projection" not in stack_params
    decode_local_params = inspect.signature(decode_megakernel.decode_local).parameters
    assert "gate_first_kda_projection" not in decode_local_params
    mla_workspace_fields = dataclasses.fields(
        decode_megakernel.MultiHeadLatentAttentionWorkspace
    )
    assert len(mla_workspace_fields) == _EXPECTED_MLA_WORKSPACE_FIELDS

    oracle = _active_canonical_hlo_oracle()
    for oracle_mode in sorted(oracle):
        (
            batch_size,
            mode_key,
            expected_bytes,
            expected_custom_calls,
            sha_hi,
            sha_lo,
        ) = oracle[oracle_mode]
        canon_text = _lower_canonical_hlo(batch_size, mode_key)
        assert "partition-id" not in canon_text, oracle_mode
        canon_bytes = canon_text.encode("utf-8")
        digest = hashlib.sha256(canon_bytes).hexdigest()
        assert len(canon_bytes) == expected_bytes, (
            f"{oracle_mode} jax={jax.__version__}: "
            f"bytes={len(canon_bytes)} sha256={digest}"
        )
        assert (
            canon_text.count('custom_call_target="tpu_custom_call"')
            == expected_custom_calls
        ), oracle_mode
        assert digest == f"{sha_hi}{sha_lo}", (
            f"{oracle_mode} jax={jax.__version__}: "
            f"bytes={len(canon_bytes)} sha256={digest}"
        )


def test_decode_vmem_plan_raises_value_error_and_rejects_unknown_kwarg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify static budget overflow raises ValueError and unknown kwargs fail fast."""
    with monkeypatch.context() as mp:
        mp.setattr(
            decode_megakernel,
            "_config",
            lambda: kernel_config.KernelConfig(vmem_limit_mb=0),
        )
        with pytest.raises(ValueError, match="exceeds the limit"):
            _lower_decode_local_uncached(1, "plain")

    empty_array = jnp.zeros((1,), dtype=jnp.int32)
    bad_kwargs: dict[str, bool] = {"unknown_decode_option": True}
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        decode_megakernel.decode_local(
            {},
            (empty_array, empty_array, empty_array),
            empty_array,
            empty_array,
            batch_size=1,
            **bad_kwargs,
        )
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        decode_megakernel.stack(
            empty_array,
            empty_array,
            {},
            (empty_array, empty_array, empty_array),
            empty_array,
            batch_size=1,
            **bad_kwargs,
        )
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        paged.decode_paged(
            {},
            jnp.zeros((2, 1), dtype=jnp.float32),
            jnp.zeros((2, 1), dtype=jnp.float32),
            (empty_array,),
            empty_array,
            empty_array,
            jnp.zeros((1, 1), dtype=jnp.int32),
            empty_array,
            batch_size=1,
            dcp_pool_map=(0,) * DCP_POOL_LAYERS,
            **bad_kwargs,
        )


def _check_fresh_import_laziness(
    result_queue: multiprocessing.Queue[tuple[bool, bool, bool, bool]],
) -> None:
    """Run inside a freshly spawned process to verify lazy registration."""
    uninit = object.__new__(decode_megakernel.PaddedBf16Rows)
    unregistered_at_import = jax.tree_util.tree_leaves(uninit) == [uninit]
    real_register = jax.tree_util.register_dataclass
    active_lock = threading.Lock()
    active_count = [0]
    overlap_detected = [False]
    registered_calls: list[type] = []

    def _tracked_register_dataclass(
        nodetype: type,
        *args: object,
        **kwargs: object,
    ) -> type:
        """Record concurrent entries while delaying the first registration."""
        with active_lock:
            active_count[0] += 1
            if active_count[0] > 1:
                overlap_detected[0] = True
            call_index = len(registered_calls)
            registered_calls.append(nodetype)
        if call_index == 0:
            threading.Event().wait(0.05)
        try:
            return real_register(nodetype, *args, **kwargs)
        finally:
            with active_lock:
                active_count[0] -= 1

    start_barrier = threading.Barrier(_LAZY_RACE_WORKERS)

    def _register_from_worker(_worker: int) -> None:
        """Synchronize worker threads at the barrier before registering."""
        start_barrier.wait(timeout=10)
        decode_megakernel.ensure_dataclasses_registered()

    jax.tree_util.register_dataclass = _tracked_register_dataclass
    try:
        with ThreadPoolExecutor(max_workers=_LAZY_RACE_WORKERS) as pool:
            list(pool.map(_register_from_worker, range(_LAZY_RACE_WORKERS)))
    finally:
        jax.tree_util.register_dataclass = real_register

    view = decode_megakernel.PaddedBf16Rows((16, VREG_LANE_COUNT))
    leaves, treedef = jax.tree_util.tree_flatten(view)
    roundtrip_ok = (
        leaves == [] and jax.tree_util.tree_unflatten(treedef, leaves) == view
    )
    result_queue.put(
        (
            unregistered_at_import,
            roundtrip_ok,
            not overlap_detected[0],
            len(registered_calls) == _EXPECTED_LAZY_DATACLASS_COUNT,
        )
    )


def test_decode_dataclasses_register_lazily_in_fresh_subprocess() -> None:
    """Verify lazy thread-safe PyTree registration and weight byte total."""
    ctx = multiprocessing.get_context("spawn")
    result_queue: multiprocessing.Queue[tuple[bool, bool, bool, bool]] = ctx.Queue()
    proc = ctx.Process(target=_check_fresh_import_laziness, args=(result_queue,))
    proc.start()
    try:
        proc.join(timeout=60)
        assert proc.exitcode == 0
        unregistered_at_import, roundtrip_ok, no_overlap, exact_count = (
            result_queue.get(timeout=5)
        )
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join()
    assert unregistered_at_import
    assert roundtrip_ok
    assert no_overlap
    assert exact_count

    model_config = load.Config()
    assert model_config.vocab == DEFAULT_VOCAB_SIZE
    expected_single_copy_bytes = _derive_single_copy_weight_bytes(model_config)
    full_shapes = load.selected_local_shapes(layers=NUM_LAYERS)
    assert len(full_shapes) == _EXPECTED_WEIGHT_FAMILIES
    total_bytes = sum(
        math.prod(shape)
        * (
            _FLOAT32_BYTES
            if name in _FLOAT32_WEIGHTS
            else (
                _UINT32_BYTES
                if name in _UINT32_WEIGHTS
                else (_UINT8_BYTES if name in _UINT8_WEIGHTS else _BF16_BYTES)
            )
        )
        for name, shape in full_shapes.items()
    )
    assert total_bytes == expected_single_copy_bytes


def test_commit_latent_rows_fused_path_and_interpret(tp_mesh: Mesh) -> None:
    """Verify fused commit_latent_rows runs in interpret mode without partition-id."""
    pool_zero = jnp.zeros(
        (2, VREG_LANE_COUNT, 2, MLA_LATENT_ROW_LANES), dtype=jnp.bfloat16
    )
    latent_rows = jnp.full((1, 1, MLA_LATENT_ROW_LANES), 3.5, dtype=jnp.bfloat16)
    block_tables = jnp.array([[1]], dtype=jnp.int32)
    positions = jnp.array([3], dtype=jnp.int32)
    cpu_mesh = Mesh(jax.devices()[:1], (AXIS_TP,))

    def _commit(
        pool_array: jax.Array,
        row_array: jax.Array,
        block_table_array: jax.Array,
        position_array: jax.Array,
    ) -> tuple[jax.Array, ...]:
        """Run fused commit_latent_rows for one MLA layer."""
        return paged.commit_latent_rows(
            (pool_array,),
            row_array,
            block_table_array,
            position_array,
            (0,),
            group=1,
            owner_table=(0,),
            global_pages=True,
        )

    sharded_cpu = jax.shard_map(
        _commit,
        mesh=cpu_mesh,
        in_specs=(P(), P(), P(), P()),
        out_specs=P(),
        check_vma=False,
    )
    with pltpu.force_tpu_interpret_mode():
        committed = sharded_cpu(pool_zero, latent_rows, block_tables, positions)
    assert len(committed) == 1
    assert bool(jnp.all(committed[0][1, 1, 1] == jnp.bfloat16(3.5)))
    assert bool(jnp.all(committed[0][0] == 0))
    assert bool(jnp.all(committed[0][1, 0] == 0))

    sharded_tpu = jax.shard_map(
        _commit,
        mesh=tp_mesh,
        in_specs=(P(), P(), P(), P()),
        out_specs=P(),
        check_vma=False,
    )
    lowered = jax.jit(sharded_tpu).lower(
        jax.ShapeDtypeStruct(pool_zero.shape, pool_zero.dtype),
        jax.ShapeDtypeStruct(latent_rows.shape, latent_rows.dtype),
        jax.ShapeDtypeStruct(block_tables.shape, block_tables.dtype),
        jax.ShapeDtypeStruct(positions.shape, positions.dtype),
    )
    assert "kimi_commit_unified_pools" in lowered.as_text()
    hlo = _to_canonical_hlo_text(lowered)
    assert (
        hlo.count('custom_call_target="tpu_custom_call"')
        == _EXPECTED_COMMIT_LATENT_CUSTOM_CALLS
    )
    assert "partition-id" not in hlo


def test_checkpoint_and_load_tokenizer_tmp_path_and_uri_rejection(
    tmp_path: Path,
) -> None:
    """Verify Checkpoint and load_tokenizer against real local files and URI guards."""
    cfg = load.Config()
    config_payload = {
        "text_config": {
            "mla_use_nope": True,
            "mla_use_output_gate": True,
            "latent_moe_use_norm": True,
            "moe_renormalize": True,
            "moe_router_activation_func": "sigmoid",
            "hidden_act": "situ",
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "num_expert_group": 1,
            "routed_scaling_factor": 1.0,
            "hidden_size": cfg.dim,
            "routed_expert_hidden_size": cfg.latent,
            "moe_intermediate_size": cfg.expert_hidden,
            "num_shared_experts": cfg.shared_hidden // cfg.expert_hidden,
            "intermediate_size": cfg.dense_hidden,
            "num_experts": cfg.experts,
            "num_experts_per_token": cfg.top_k,
            "num_hidden_layers": cfg.layers,
            "num_attention_heads": cfg.heads,
            "q_lora_rank": cfg.q_rank,
            "kv_lora_rank": cfg.kv_rank,
            "qk_rope_head_dim": cfg.qk_extra,
            "vocab_size": cfg.vocab,
            "attn_res_block_size": cfg.residual_block,
            "rms_norm_eps": cfg.eps,
            "activation_situ_beta": cfg.situ_beta,
            "activation_situ_linear_beta": cfg.situ_linear_beta,
            "linear_attn_config": {
                "use_full_rank_gate": True,
                "head_dim": cfg.head_dim,
                "short_conv_kernel_size": cfg.conv_size,
                "gate_lower_bound": cfg.gate_lower_bound,
                "full_attn_layers": [layer_id + 1 for layer_id in cfg.full_attention],
            },
        }
    }
    (tmp_path / "config.json").write_text(json.dumps(config_payload), encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"added_tokens_decoder": {"3": {"content": "<|eos|>"}}}),
        encoding="utf-8",
    )
    (tmp_path / "tokenization_kimi.py").write_text(
        'class TikTokenTokenizer:\n    pat_str = "|".join([r"[a-z]+", r"\\s+"])\n',
        encoding="utf-8",
    )
    bpe_lines = [
        f"{base64.b64encode(b'hello').decode('ascii')} 0",
        f"{base64.b64encode(b' ').decode('ascii')} 1",
        f"{base64.b64encode(b'world').decode('ascii')} 2",
    ]
    (tmp_path / "tiktoken.model").write_text(
        "\n".join(bpe_lines) + "\n", encoding="utf-8"
    )

    bf16_dtype = np.dtype(ml_dtypes.bfloat16)
    sample = np.arange(32, dtype=np.float32).reshape(4, 8).astype(bf16_dtype)
    raw_bytes = sample.tobytes()
    header_bytes = json.dumps(
        {
            "test.weight": {
                "dtype": "BF16",
                "shape": [4, 8],
                "data_offsets": [0, len(raw_bytes)],
            }
        }
    ).encode("utf-8")
    shard_name = "model-00001-of-00001.safetensors"
    (tmp_path / shard_name).write_bytes(
        struct.pack("<Q", len(header_bytes)) + header_bytes + raw_bytes
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"test.weight": shard_name}}),
        encoding="utf-8",
    )

    encoding = load.load_tokenizer(tmp_path)
    token_ids = encoding.encode("hello world")
    assert encoding.decode(token_ids) == "hello world"
    assert encoding.encode("<|eos|>", allowed_special="all") == [3]

    checkpoint = load.Checkpoint(tmp_path)
    np.testing.assert_array_equal(checkpoint.read("test.weight"), sample)
    np.testing.assert_array_equal(
        checkpoint.read("test.weight", (slice(1, 3), slice(None))),
        sample[1:3],
    )
    many = checkpoint.read_many(["test.weight"])
    np.testing.assert_array_equal(many["test.weight"], sample)
    with pytest.raises(ValueError, match="step=2"):
        checkpoint.read("test.weight", (slice(0, 4, 2), slice(None)))

    for bad_uri in ("s3://bucket/kimi-k3", "s3:/bucket/kimi-k3"):
        with pytest.raises(ValueError, match="local"):
            load.Config.from_checkpoint(bad_uri)
        with pytest.raises(ValueError, match="local"):
            load.load_tokenizer(bad_uri)
        with pytest.raises(ValueError, match="local"):
            load.Checkpoint(bad_uri)


def test_paged_and_mla_dcp_static_contracts() -> None:
    """Verify public paged and mla_dcp contracts and CPU scatter commits."""
    commit_positional_names = [
        "pools",
        "rows",
        "block_tables",
        "positions",
        "pool_map",
    ]
    decode_positional_names = [
        "weights",
        "conv_slots",
        "recurrent_slots",
        "pool",
    ]
    commit_params = list(inspect.signature(paged.commit_latent_rows).parameters)
    assert commit_params[: len(commit_positional_names)] == commit_positional_names
    decode_paged_params = list(inspect.signature(paged.decode_paged).parameters)
    assert decode_paged_params[: len(decode_positional_names)] == (
        decode_positional_names
    )

    pool_zero = jnp.zeros(
        (2, VREG_LANE_COUNT, 2, _DUMMY_LATENT_WIDTH), dtype=jnp.bfloat16
    )
    latent_rows = jnp.ones((1, 1, _DUMMY_LATENT_WIDTH), dtype=jnp.bfloat16)
    block_tables = jnp.zeros((1, 1), dtype=jnp.int32)
    positions = jnp.zeros((1,), dtype=jnp.int32)
    cpu_mesh = Mesh(jax.devices()[:1], (AXIS_TP,))
    with pltpu.force_tpu_interpret_mode():
        committed = jax.shard_map(
            lambda pool_array,
            row_array,
            block_table_array,
            position_array: paged.commit_latent_rows(
                (pool_array,),
                row_array,
                block_table_array,
                position_array,
                (0,),
                group=1,
                owner_table=(0,),
                global_pages=True,
            ),
            mesh=cpu_mesh,
            in_specs=(P(), P(), P(), P()),
            out_specs=P(),
            check_vma=False,
        )(pool_zero, latent_rows, block_tables, positions)
    assert len(committed) == 1
    assert bool(jnp.all(committed[0][0, 0, 0] == 1))

    persistent_scratch = mla_dcp.persistent_scratch(
        BATCH, force_chunks=True, sequence_rows=1
    )
    scoped_scratch = mla_dcp.scoped_scratch(BATCH, force_chunks=True, sequence_rows=1)
    assert len(persistent_scratch) == len(mla_dcp.PERSISTENT_NAMES)
    assert len(scoped_scratch) == len(mla_dcp.SCOPED_NAMES)


class FakeCheckpoint(load.Checkpoint):
    """In-memory Checkpoint double returning deterministic coordinate tensors."""

    def __init__(self) -> None:
        """Initialize Kimi-K3 config without opening safetensors shards on disk."""
        self.config = load.Config()

    def read(self, name: str, slices: tuple[slice, ...] | None = None) -> np.ndarray:
        """Synthesize deterministic float32 slice values from tensor coordinates."""
        full_o_cols = self.config.heads * self.config.head_dim
        if name.endswith("self_attn.o_proj.weight"):
            full = np.arange(self.config.dim * full_o_cols, dtype=np.float32).reshape(
                self.config.dim, full_o_cols
            )
        elif name.endswith("routed_expert_up_proj.weight"):
            full = np.ones((self.config.dim, self.config.latent), dtype=np.float32)
        else:
            msg = f"Unexpected tensor {name!r}"
            raise KeyError(msg)
        return full[slices] if slices is not None else full


def test_mla_dcp_chunk_layout_and_page_plan_across_dp_regimes() -> None:
    """Verify MLA DCP chunk sizing, page buffer plans, and packet layouts."""
    dp2_heads = DP2_GROUPS * mla_dcp.HEADS_PER_RANK
    # B16 uses 8-row chunks (3 ring slots); B24 uses 4-row chunks (2 ring slots).
    expected_chunk_b16 = (8, 3)
    expected_chunk_b24 = (4, 2)
    # Wide BF16 page plans drop from 4 default ring buffers to 3 to stay in budget.
    expected_wide_bf16_buffers = mla_dcp.BUFFERS - 1
    # B2 packs 3 heads + 1 zero pad slot into 4 slots across MAX_GROUPS=2 row groups.
    expected_b2_packet_layout = (mla_dcp.HEADS_PER_RANK + 1, mla_dcp.MAX_GROUPS)
    expected_b8_packet_layout = (mla_dcp.HEADS_PER_RANK, 1)

    assert mla_dcp.chunk_layout(16, heads=mla_dcp.HEADS_PER_RANK) == expected_chunk_b16
    assert mla_dcp.chunk_layout(24, heads=mla_dcp.HEADS_PER_RANK) == expected_chunk_b24
    with pytest.raises(ValueError, match="must divide the batch"):
        mla_dcp.chunk_layout(10, heads=mla_dcp.HEADS_PER_RANK, chunk_rows=4)
    with pytest.raises(ValueError, match="rows must divide"):
        mla_dcp.chunk_layout(12, heads=mla_dcp.HEADS_PER_RANK, chunk_rows=6)

    assert (
        mla_dcp.page_plan(4, heads=mla_dcp.HEADS_PER_RANK)["buffers"] == mla_dcp.BUFFERS
    )
    assert (
        mla_dcp.page_plan(64, heads=mla_dcp.HEADS_PER_RANK, cache_dtype=jnp.bfloat16)[
            "buffers"
        ]
        == expected_wide_bf16_buffers
    )
    assert (
        mla_dcp.page_plan(64, heads=dp2_heads, cache_dtype=jnp.float8_e4m3fn)["buffers"]
        == mla_dcp.BUFFERS
    )
    assert (
        mla_dcp.packet_layout(2, heads=mla_dcp.HEADS_PER_RANK)
        == expected_b2_packet_layout
    )
    assert (
        mla_dcp.packet_layout(8, heads=mla_dcp.HEADS_PER_RANK)
        == expected_b8_packet_layout
    )


def test_load_selected_local_shapes_validates_bounds_and_rank_slicing() -> None:
    """Verify selected_local_shapes preconditions and rank 0/13/31 shard slicing."""
    model_config = load.Config()
    kda_output_rows = (
        _RANK_PAIR_SIZE * (model_config.heads // TP_SIZE) * model_config.head_dim
    )
    half_hidden = model_config.dim // _HIDDEN_HALVES
    latent_up_cols = model_config.dim // _ACTIVE_LATENT_UP_LANES
    test_rank_mid = 13
    test_rank_last = TP_SIZE - 1

    with pytest.raises(ValueError, match=r"2\.\.93 layers"):
        load.selected_local_shapes(layers=1)
    with pytest.raises(ValueError, match=r"2\.\.93 layers"):
        load.selected_local_shapes(layers=NUM_LAYERS + 1)
    with pytest.raises(ValueError, match="32 shards"):
        load.selected_local_shapes(layers=LAYERS, vocab=model_config.vocab + 1)

    checkpoint = FakeCheckpoint()
    for rank in (0, test_rank_mid, test_rank_last):
        kda_output_shard = checkpoint.dense_selected("k_o", layer=0, rank=rank)
        assert kda_output_shard.shape == (kda_output_rows, half_hidden)

    up_rank0 = checkpoint.dense_selected("latent_up", layer=0, rank=0)
    up_rank13 = checkpoint.dense_selected("latent_up", layer=0, rank=test_rank_mid)
    up_rank31 = checkpoint.dense_selected("latent_up", layer=0, rank=test_rank_last)
    assert up_rank0.shape == (model_config.latent, latent_up_cols)
    assert up_rank13.shape == (model_config.latent, latent_up_cols)
    assert float(np.sum(up_rank0)) > 0.0
    assert float(np.sum(up_rank13)) > 0.0
    assert up_rank31.shape == (model_config.latent, latent_up_cols)
    assert np.count_nonzero(up_rank31) == 0


def test_decode_megakernel_batch32_compiles_with_int16_gather(
    tp_mesh: Mesh,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Batch-32 decode compiles through Mosaic using int16 page-offset gathers."""
    orig_take_along_axis = jnp.take_along_axis
    gather_calls: list[tuple[jnp.dtype, object, object, object]] = []

    def _spy_take_along_axis(
        arr: jax.Array,
        indices: jax.Array,
        *args: object,
        **kwargs: object,
    ) -> jax.Array:
        """Record index dtype, axis, mode, and wrap flag while forwarding to JAX."""
        gather_calls.append(
            (
                indices.dtype,
                kwargs.get("axis"),
                kwargs.get("mode"),
                kwargs.get("wrap_negative_indices"),
            )
        )
        return orig_take_along_axis(arr, indices, *args, **kwargs)

    monkeypatch.setattr(jnp, "take_along_axis", _spy_take_along_axis)
    weights = _abstract_weights(layers=ROW_PLAN_LAYERS)
    sharded, inputs = _make_sharded_local_decode(
        tp_mesh,
        weights,
        ROW_PLAN_BATCH_SIZE,
        ROW_PLAN_LAYERS,
        {
            "attention_token_scatter": True,
            "moe_token_scatter_output": True,
            "moe_routed_reduce_scatter": True,
            "sequence_parallel_pre_attention": True,
            "residual_reduction_order": "relaxed",
            "mla_cache_batch_tile_size": ROW_PLAN_MLA_BATCH_TILE,
            "alias_expert_scales": True,
        },
    )
    try:
        lowered = jax.jit(sharded).lower(*inputs, *weights.values())
        hlo = _to_canonical_hlo_text(lowered)
        lowered.compile()
    finally:
        # Release the compiled 32-device executable before later CPU tests run.
        jax.clear_caches()
        gc.collect()
    assert "partition-id" not in hlo
    assert any(
        dtype == jnp.int16
        and axis == 0
        and mode == "promise_in_bounds"
        and wrap is False
        for dtype, axis, mode, wrap in gather_calls
    ), (
        "Expected int16 axis=0 promise_in_bounds gather with "
        f"wrap_negative_indices=False, saw {gather_calls}"
    )


def test_project_mla_value_up_heads_matches_unhoisted_reference() -> None:
    """Verify shared W_UV value-up helper matches un-hoisted two-load FP32 split."""
    kv_rank = load.Config().kv_rank
    heads_per_rank = mla_dcp.HEADS_PER_RANK
    dp_groups = DP2_GROUPS
    rng = np.random.default_rng(0)
    merged = jnp.asarray(
        rng.standard_normal((BATCH, dp_groups * heads_per_rank, kv_rank)),
        dtype=jnp.float32,
    )
    up_weight = jnp.asarray(
        rng.standard_normal((kv_rank, heads_per_rank * _MLA_KEY_HANDOFF_LANES)),
        dtype=jnp.bfloat16,
    )

    for group_index in range(dp_groups):
        head_offset = group_index * heads_per_rank
        got = _project_mla_value_up_heads(
            merged,
            cast("PallasRefProtocol", up_weight),
            head_offset,
        )
        assert len(got) == heads_per_rank
        for head in range(heads_per_rank):
            value_start = head * _MLA_KEY_HANDOFF_LANES + HEAD_DIM
            latent = merged[:, head_offset + head]
            high = latent.astype(jnp.bfloat16)
            low = (latent - high.astype(jnp.float32)).astype(jnp.bfloat16)
            high_ref = 0 + jax.lax.dot_general(
                high,
                up_weight[:, value_start : value_start + HEAD_DIM],
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            low_ref = jax.lax.dot_general(
                low,
                up_weight[:, value_start : value_start + HEAD_DIM],
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            assert np.array_equal(np.asarray(got[head]), np.asarray(high_ref + low_ref))
