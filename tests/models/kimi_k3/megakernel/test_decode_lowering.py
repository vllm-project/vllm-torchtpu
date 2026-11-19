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

import ast
import dataclasses
import functools
import hashlib
import inspect
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import source_info_util
from jax._src.lib import hlo as jax_hlo
from jax.experimental import topologies
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

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
    DP2_GROUPS,
    KDA_HEADS,
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
from vllm_torchtpu.models.kimi_k3.megakernel.layout import MLA_LAYERS

# AOT compile-speed fixtures covering layer 0 dense MLP, KDA, MLA, and MoE.
LAYERS = 4
PAGES = 16
BATCH = 8
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
_KDA_STATE_GROUPS = 3
_KDA_PROJECTION_PARTS = 4
_EXPECTED_MLA_WORKSPACE_FIELDS = 10
_EXPECTED_CUSTOM_CALLS = 2
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
# 0.11.1: (batch_size, mode_key, byte_len, sha256_upper_half, sha256_lower_half).
_CANONICAL_HLO_ORACLE_JAX_0_11_1: dict[str, tuple[int, str, int, str, str]] = {
    "batch1_plain": (
        1,
        "plain",
        1_100_304,
        "147fbdb6e5274ea972ae56bc1b2cdd35",
        "ba2652fcd7877dfb793f688714de4261",
    ),
    "batch8_plain": (
        8,
        "plain",
        1_444_434,
        "5072c2938e73beb8c3bd40a3c08733ba",
        "0e6b592b569ab27f0dd989432f636f40",
    ),
    "scatter": (
        8,
        "scatter",
        1_550_514,
        "40923b718fec10b246cda1f9dae314b6",
        "03cdd0e9b5316de0c02bbb19c5f69b1f",
    ),
    "scatter_moe_out": (
        8,
        "scatter_moe_out",
        1_570_298,
        "74f094dd0d93d41f47a2e144cb1e39be",
        "286097f8e8b08371ececa22f99588af6",
    ),
    "scatter_moe_reduce_scatter": (
        8,
        "scatter_moe_reduce_scatter",
        1_596_058,
        "74887ae9222451a3a418b72a448d0b54",
        "7646e55fdf81f007c0b303a4d5ada94c",
    ),
    "full_sequence_parallel": (
        8,
        "full_sequence_parallel",
        1_608_158,
        "94712e13e01090cc5fb602ff253509c5",
        "ada95ad47473e896502dab5e385d33f3",
    ),
    "aux_hidden": (
        8,
        "aux_hidden",
        1_446_803,
        "46f05d17efb3f7c2118acae160f82410",
        "2f91c32e7fa7a73330feb1cd0b30eb11",
    ),
    "batch8_paged_alias_expert_scales": (
        8,
        "paged_alias_expert_scales",
        1_771_525,
        "5708b96509927d597e0e4ed00359528b",
        "86c53cb6ced1ffee4d9c042924a37573",
    ),
}

# Reference canonical HLO oracle across all 8 decode regimes under jax 0.11.2.
_CANONICAL_HLO_ORACLE_JAX_0_11_2: dict[str, tuple[int, str, int, str, str]] = {
    "batch1_plain": (
        1,
        "plain",
        1_088_135,
        "eba415c2ca76b5eec17169e2215cda41",
        "805884aefd3e88b1982e8beb091ca374",
    ),
    "batch8_plain": (
        8,
        "plain",
        1_434_810,
        "8a98b75d85c4da0b9836a2e536b3207e",
        "c2c806097aad549c88a9adbcbb602746",
    ),
    "scatter": (
        8,
        "scatter",
        1_540_890,
        "374dff18e9bf962139f9ea1a5cbdccc2",
        "f182ef7e3cd0520b35bef19af5f30f35",
    ),
    "scatter_moe_out": (
        8,
        "scatter_moe_out",
        1_560_674,
        "e2db06e16b6839083cca2c93f2ac6d43",
        "8065e40c5c2dac39fd9f4d994ea805e8",
    ),
    "scatter_moe_reduce_scatter": (
        8,
        "scatter_moe_reduce_scatter",
        1_586_434,
        "891bf4bc23eb4d919e05f4cd1788ecfd",
        "c86bee9c2e68360b3407b307792348f8",
    ),
    "full_sequence_parallel": (
        8,
        "full_sequence_parallel",
        1_605_270,
        "05e5cb8b595b9eb62a41f6bde85f46b4",
        "5fe6876ec2998d5cfb4fbe63a570cfad",
    ),
    "aux_hidden": (
        8,
        "aux_hidden",
        1_437_179,
        "2b870c3a0e529fb0713ce212a370867b",
        "8c5b94bd85ae9d475ac2675a1cb300e1",
    ),
    "batch8_paged_alias_expert_scales": (
        8,
        "paged_alias_expert_scales",
        1_768_621,
        "ddddd46f6f7ed314eadcfb34de41a9ba",
        "91b1c8b89793116a722027b5f75efb58",
    ),
}

CANONICAL_HLO_ORACLE: dict[str, tuple[int, str, int, str, str]] = (
    _CANONICAL_HLO_ORACLE_JAX_0_11_1
    if jax.__version__ == "0.11.1"
    else _CANONICAL_HLO_ORACLE_JAX_0_11_2
)


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
        * _KDA_STATE_GROUPS
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


def _abstract_weights() -> dict[str, jax.ShapeDtypeStruct]:
    """Return per-rank weight shapes of a LAYERS-layer prefix in kernel layout."""
    weights: dict[str, jax.ShapeDtypeStruct] = {}
    for name, shape in load.selected_local_shapes(layers=LAYERS).items():
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


def _lower_decode_local_uncached(batch_size: int, mode: str) -> str:
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

        def step(*arrays: jax.Array) -> object:
            """Invoke ``decode_local`` on per-rank abstract arrays."""
            tokens, positions, conv_state, recurrent_state, pool, block_tables = arrays[
                :6
            ]
            return decode_megakernel.decode_local(
                dict(zip(weights, arrays[6:], strict=True)),
                (conv_state, recurrent_state, pool),
                tokens,
                positions,
                batch_size=batch_size,
                sequence_rows=1,
                layers=LAYERS,
                block_tables=block_tables,
                dcp_pool_map=(0,) * DCP_POOL_LAYERS,
                dcp_owner_table=tuple(range(RANKS)),
                attention_dp=1,
                attention_token_scatter=scatter,
                moe_token_scatter_output=moe_out,
                moe_routed_reduce_scatter=moe_reduce_scatter,
                sequence_parallel_pre_attention=sequence_parallel,
                aux_hidden_layers=aux_layers,
            )

        inputs = (
            jax.ShapeDtypeStruct((batch_size,), jnp.int32),
            jax.ShapeDtypeStruct((batch_size,), jnp.int32),
            jax.ShapeDtypeStruct(
                (batch_size, LAYERS, _KDA_STATE_GROUPS, KDA_HEADS, 4, 128),
                jnp.float32,
            ),
            jax.ShapeDtypeStruct(
                (batch_size, LAYERS, KDA_HEADS, 128, 128), jnp.float32
            ),
            jax.ShapeDtypeStruct(
                (PAGES, VREG_LANE_COUNT, _FP8_POOL_PACKING, MLA_LATENT_ROW_LANES),
                jnp.float8_e4m3fn,
            ),
            jax.ShapeDtypeStruct((batch_size, PAGES), jnp.int32),
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
                (slot_capacity, LAYERS, _KDA_STATE_GROUPS, KDA_HEADS, 4, 128),
                jnp.float32,
            ),
            jax.ShapeDtypeStruct(
                (slot_capacity, LAYERS, KDA_HEADS, 128, 128), jnp.float32
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

    for oracle_mode in sorted(CANONICAL_HLO_ORACLE):
        batch_size, mode_key, expected_bytes, sha_hi, sha_lo = CANONICAL_HLO_ORACLE[
            oracle_mode
        ]
        canon_text = _lower_canonical_hlo(batch_size, mode_key)
        canon_bytes = canon_text.encode("utf-8")
        digest = hashlib.sha256(canon_bytes).hexdigest()
        assert len(canon_bytes) == expected_bytes, (
            f"{oracle_mode} jax={jax.__version__}: "
            f"bytes={len(canon_bytes)} sha256={digest}"
        )
        assert (
            canon_text.count('custom_call_target="tpu_custom_call"')
            == _EXPECTED_CUSTOM_CALLS
        ), oracle_mode
        assert digest == f"{sha_hi}{sha_lo}", (
            f"{oracle_mode} jax={jax.__version__}: "
            f"bytes={len(canon_bytes)} sha256={digest}"
        )


def test_decode_vmem_plan_raises_value_error_and_rejects_unknown_kwarg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify static budget overflow raises ValueError and unknown kwargs fail fast."""
    monkeypatch.setattr(decode_megakernel, "VMEM_LIMIT_BYTES", 1)
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


def test_decode_dataclasses_register_lazily_on_instantiation() -> None:
    """Verify lazy PyTree dataclass registration and derived per-rank byte total."""
    source_path = Path(inspect.getfile(decode_megakernel))
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            decorator_name = ast.unparse(target).rsplit(".", 1)[-1]
            assert not decorator_name.startswith("register")

    register_fn = decode_megakernel.ensure_dataclasses_registered
    assert callable(register_fn)

    padded_rows_cls = decode_megakernel.PaddedBf16Rows
    padded_view = padded_rows_cls((16, VREG_LANE_COUNT))
    leaves, treedef = jax.tree_util.tree_flatten(padded_view)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt == padded_view

    model_config = load.Config()
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
