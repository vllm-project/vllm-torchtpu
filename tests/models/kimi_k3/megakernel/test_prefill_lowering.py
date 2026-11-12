# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Unmocked TPU v7x AOT lowering and contract tests for Kimi-K3 prefill megakernel."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import pytest
from jax.experimental import topologies
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.models.kimi_k3.megakernel.kernels import load, prefill_megakernel

_TP_RANKS = 32
_HIDDEN_DIM = 7168
_MLA_HEADS = 3
_PACKED_TOKENS = 256
_TILE_TOKENS = 128
_MULTI_BLOCK_ARENA_TOKENS = 256
_KEY_DIM = 256
_VALUE_DIM = 128
_NO_ARENA_SEGMENT_FIELDS = 3
_ARENA_SEGMENT_FIELDS = 5
_PREFILL_SEQ_LEN = 512
_TEST_DEPTH = 20
_TAP_LAYER_INDEX = 18
_KDA_LAYERS = 69
_MLA_LAYERS = 24
_KDA_HEADS_PER_RANK = 3
_KDA_CONV_TAPS = 3
_MLA_LATENT_DIM = 576
_CP_POOL_BLOCKS = 64
_CP_POOL_ROWS = 2
_CP_POOL_COLS = 640
_CP_ID_SLOTS = 10
_STACK_OUTPUTS_NO_TAP = 4
_STACK_OUTPUTS_WITH_TAP = 5


def _single_device_tpu7x_mesh() -> tuple[Mesh, NamedSharding]:
    """Return a single-device TPU v7x AOT mesh and replicated sharding."""
    devices = topologies.get_topology_desc(
        topology_name="tpu7x_2x2x4", platform="tpu"
    ).devices
    mesh = Mesh(devices[:1], ("x",))
    return mesh, NamedSharding(mesh, P())


def _build_prefill_stack_inputs(
    mesh: Mesh,
    *,
    seq_len: int,
    segments: int,
    depth: int,
) -> tuple[object, ...]:
    """Build abstract TPU v7x sharded inputs for ``make_prefill_stack_sp_indexed``."""
    sharding = NamedSharding(mesh, P("k3_sp"))
    replicated = NamedSharding(mesh, P())
    shapes = load.selected_local_shapes(layers=depth)

    hidden = jax.ShapeDtypeStruct(
        (_TP_RANKS, seq_len // _TP_RANKS, _HIDDEN_DIM),
        jnp.bfloat16,
        sharding=sharding,
    )
    common = {
        name: jax.ShapeDtypeStruct(
            (_TP_RANKS, *shapes[name]),
            jnp.float32 if name.endswith("_res") else jnp.bfloat16,
            sharding=sharding,
        )
        for name in (
            "attn_res",
            "attn_norm",
            "mlp_res",
            "ffn_norm",
            "dense_gu",
            "dense_down",
        )
    }

    kda_keys = (
        "k_gate",
        "k_fb",
        "k_projection",
        "k_conv",
        "k_a_log",
        "k_dt",
        "k_norm",
        "k_o",
    )
    kda_families = {
        key: jax.ShapeDtypeStruct(
            (_TP_RANKS, depth, 384, _HIDDEN_DIM)
            if key == "k_o"
            else (_TP_RANKS, *shapes[key]),
            jnp.float32
            if key in ("k_a_log", "k_dt", "k_conv", "k_norm")
            else jnp.bfloat16,
            sharding=sharding,
        )
        for key in kda_keys
    }

    mla_families = {
        key: jax.ShapeDtypeStruct(
            (_TP_RANKS, depth, 384, _HIDDEN_DIM)
            if key == "m_o"
            else (_TP_RANKS, *shapes[key]),
            jnp.float32 if "norm" in key else jnp.bfloat16,
            sharding=sharding,
        )
        for key in prefill_megakernel.MLA_FAMILIES
    }

    moe_overrides: dict[str, tuple[tuple[int, ...], jnp.dtype]] = {
        "latent_down": ((_TP_RANKS, depth, _HIDDEN_DIM, 4096), jnp.bfloat16),
        "latent_up": ((_TP_RANKS, depth, 3584, _HIDDEN_DIM), jnp.bfloat16),
        "shared_gu": ((_TP_RANKS, depth, _HIDDEN_DIM, 8192), jnp.bfloat16),
        "shared_down": ((_TP_RANKS, depth, 3072, _HIDDEN_DIM), jnp.bfloat16),
        "expert_gu": ((_TP_RANKS, depth, 56, 3584, 3072), jnp.uint4),
        "expert_gus": ((_TP_RANKS, depth, 56, 28, 3072), jnp.uint8),
        "expert_down": ((_TP_RANKS, depth, 56, 1536, 3584), jnp.uint4),
        "expert_ds": ((_TP_RANKS, depth, 56, 12, 3584), jnp.uint8),
    }
    moe_families: dict[str, jax.ShapeDtypeStruct] = {}
    for key in prefill_megakernel.MOE_FAMILIES:
        if key in moe_overrides:
            ov_shape, ov_dtype = moe_overrides[key]
            moe_families[key] = jax.ShapeDtypeStruct(
                ov_shape, ov_dtype, sharding=sharding
            )
        else:
            dt = jnp.float32 if key in ("router_bias", "latent_norm") else jnp.bfloat16
            moe_families[key] = jax.ShapeDtypeStruct(
                (_TP_RANKS, *shapes[key]), dt, sharding=sharding
            )

    kda_state = (
        jax.ShapeDtypeStruct(
            (
                _TP_RANKS,
                _KDA_LAYERS,
                segments,
                _KDA_HEADS_PER_RANK,
                _VALUE_DIM,
                _VALUE_DIM,
            ),
            jnp.float32,
            sharding=sharding,
        ),
        jax.ShapeDtypeStruct(
            (
                _TP_RANKS,
                _KDA_LAYERS,
                segments,
                _KDA_CONV_TAPS,
                _KDA_CONV_TAPS,
                _KDA_HEADS_PER_RANK,
                _VALUE_DIM,
            ),
            jnp.bfloat16,
            sharding=sharding,
        ),
        jax.ShapeDtypeStruct(
            (
                _TP_RANKS,
                _KDA_LAYERS,
                segments,
                _KDA_HEADS_PER_RANK,
                _VALUE_DIM,
                _VALUE_DIM,
            ),
            jnp.float32,
            sharding=sharding,
        ),
        jax.ShapeDtypeStruct(
            (
                _TP_RANKS,
                _KDA_LAYERS,
                segments,
                _KDA_CONV_TAPS,
                _KDA_CONV_TAPS,
                _KDA_HEADS_PER_RANK,
                _VALUE_DIM,
            ),
            jnp.bfloat16,
            sharding=sharding,
        ),
    )
    mla_state = (
        jax.ShapeDtypeStruct(
            (_TP_RANKS, _MLA_LAYERS, seq_len, _MLA_LATENT_DIM),
            jnp.bfloat16,
            sharding=sharding,
        ),
    )
    length = jax.ShapeDtypeStruct((segments,), jnp.int32, sharding=replicated)
    segment_desc = jax.ShapeDtypeStruct(
        (_ARENA_SEGMENT_FIELDS * segments,), jnp.int32, sharding=replicated
    )
    arena = prefill_megakernel.CpPrefix(
        pools=(
            jax.ShapeDtypeStruct(
                (_TP_RANKS, _CP_POOL_BLOCKS, _CP_POOL_ROWS, _CP_POOL_COLS),
                jnp.bfloat16,
                sharding=sharding,
            ),
        ),
        ids=jax.ShapeDtypeStruct(
            (_TP_RANKS, _CP_ID_SLOTS), jnp.int32, sharding=sharding
        ),
        local=jax.ShapeDtypeStruct(
            (_TP_RANKS, 2 * segments), jnp.int32, sharding=sharding
        ),
    )
    return (
        hidden,
        common,
        kda_families,
        mla_families,
        moe_families,
        kda_state,
        mla_state,
        length,
        segment_desc,
        arena,
    )


def test_mla_flash_prefill_packed_traces_with_and_without_arena() -> None:
    """Verify mla_flash_prefill_packed lowers on TPU v7x with and without arena."""
    mesh, replicated = _single_device_tpu7x_mesh()
    query = jax.ShapeDtypeStruct(
        (_MLA_HEADS, _PACKED_TOKENS, _KEY_DIM), jnp.bfloat16, sharding=replicated
    )
    key = jax.ShapeDtypeStruct(
        (_MLA_HEADS, _PACKED_TOKENS, _KEY_DIM), jnp.bfloat16, sharding=replicated
    )
    value = jax.ShapeDtypeStruct(
        (_MLA_HEADS, _PACKED_TOKENS, _VALUE_DIM), jnp.bfloat16, sharding=replicated
    )
    length = jax.ShapeDtypeStruct((1,), jnp.int32, sharding=replicated)
    segments_no_arena = jax.ShapeDtypeStruct(
        (_NO_ARENA_SEGMENT_FIELDS,), jnp.int32, sharding=replicated
    )
    segments_arena = jax.ShapeDtypeStruct(
        (_ARENA_SEGMENT_FIELDS,), jnp.int32, sharding=replicated
    )
    arena_keys = jax.ShapeDtypeStruct(
        (1, _MLA_HEADS, _TILE_TOKENS, _KEY_DIM), jnp.bfloat16, sharding=replicated
    )
    arena_values = jax.ShapeDtypeStruct(
        (1, _MLA_HEADS, _VALUE_DIM, _TILE_TOKENS), jnp.bfloat16, sharding=replicated
    )

    with jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
        lowered_no_arena = jax.jit(
            functools.partial(
                prefill_megakernel.mla_flash_prefill_packed, count=1, arena=None
            )
        ).lower(query, key, value, length, segments_no_arena)
        assert "tpu_custom_call" in lowered_no_arena.as_text()

        lowered_arena = jax.jit(
            lambda q,
            k,
            v,
            ln,
            seg,
            ak,
            av: prefill_megakernel.mla_flash_prefill_packed(
                q, k, v, ln, seg, count=1, arena=(ak, av)
            )
        ).lower(query, key, value, length, segments_arena, arena_keys, arena_values)
        assert "tpu_custom_call" in lowered_arena.as_text()

    with pytest.raises(ValueError, match="segments must be"):
        jax.eval_shape(
            functools.partial(
                prefill_megakernel.mla_flash_prefill_packed,
                count=1,
                arena=(
                    jnp.zeros((1, _MLA_HEADS, _TILE_TOKENS, _KEY_DIM), jnp.bfloat16),
                    jnp.zeros((1, _MLA_HEADS, _VALUE_DIM, _TILE_TOKENS), jnp.bfloat16),
                ),
            ),
            jnp.zeros((_MLA_HEADS, _PACKED_TOKENS, _KEY_DIM), jnp.bfloat16),
            jnp.zeros((_MLA_HEADS, _PACKED_TOKENS, _KEY_DIM), jnp.bfloat16),
            jnp.zeros((_MLA_HEADS, _PACKED_TOKENS, _VALUE_DIM), jnp.bfloat16),
            jnp.array([_PACKED_TOKENS], jnp.int32),
            jnp.zeros((_NO_ARENA_SEGMENT_FIELDS,), jnp.int32),
        )


def test_load_kv_block_accepts_partial_keyword_arguments() -> None:
    """Verify multi-block arena lowering binds _load_kv_block keyword buffers."""
    mesh, replicated = _single_device_tpu7x_mesh()
    query = jax.ShapeDtypeStruct(
        (_MLA_HEADS, _PACKED_TOKENS, _KEY_DIM), jnp.bfloat16, sharding=replicated
    )
    key = jax.ShapeDtypeStruct(
        (_MLA_HEADS, _PACKED_TOKENS, _KEY_DIM), jnp.bfloat16, sharding=replicated
    )
    value = jax.ShapeDtypeStruct(
        (_MLA_HEADS, _PACKED_TOKENS, _VALUE_DIM), jnp.bfloat16, sharding=replicated
    )
    length = jax.ShapeDtypeStruct((1,), jnp.int32, sharding=replicated)
    segments_arena = jax.ShapeDtypeStruct(
        (_ARENA_SEGMENT_FIELDS,), jnp.int32, sharding=replicated
    )
    arena_keys = jax.ShapeDtypeStruct(
        (1, _MLA_HEADS, _MULTI_BLOCK_ARENA_TOKENS, _KEY_DIM),
        jnp.bfloat16,
        sharding=replicated,
    )
    arena_values = jax.ShapeDtypeStruct(
        (1, _MLA_HEADS, _VALUE_DIM, _MULTI_BLOCK_ARENA_TOKENS),
        jnp.bfloat16,
        sharding=replicated,
    )

    with jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
        lowered = jax.jit(
            lambda q,
            k,
            v,
            ln,
            seg,
            ak,
            av: prefill_megakernel.mla_flash_prefill_packed(
                q, k, v, ln, seg, count=1, arena=(ak, av)
            )
        ).lower(query, key, value, length, segments_arena, arena_keys, arena_values)
        hlo_text = lowered.as_text()
    assert "tpu_custom_call" in hlo_text
    assert lowered.out_info.shape == (_MLA_HEADS, _PACKED_TOKENS, _VALUE_DIM)


def test_prefill_stack_traces_with_tap_layer_none_and_layer_index() -> None:
    """Verify make_prefill_stack_sp_indexed traces with tap_layer and keyword arena."""
    devices = topologies.get_topology_desc(
        topology_name="tpu7x_2x2x4", platform="tpu"
    ).devices
    mesh = Mesh(devices, ("k3_sp",))
    args = _build_prefill_stack_inputs(
        mesh, seq_len=_PREFILL_SEQ_LEN, segments=1, depth=_TEST_DEPTH
    )
    expected_hidden_shape = (_TP_RANKS, _PREFILL_SEQ_LEN // _TP_RANKS, _HIDDEN_DIM)

    stack_none = prefill_megakernel.make_prefill_stack_sp_indexed(
        mesh,
        segments=1,
        seq_len=_PREFILL_SEQ_LEN,
        prefix_capacity=2048,
        tap_layer=None,
        depth=_TEST_DEPTH,
    )
    out_none = jax.eval_shape(stack_none, *args)
    assert len(out_none) == _STACK_OUTPUTS_NO_TAP
    assert out_none[0].shape == expected_hidden_shape

    out_kw = jax.eval_shape(
        functools.partial(stack_none, segment_desc=args[-2], arena=args[-1]),
        *args[:-2],
    )
    assert len(out_kw) == _STACK_OUTPUTS_NO_TAP
    assert out_kw[0].shape == expected_hidden_shape

    stack_tap = prefill_megakernel.make_prefill_stack_sp_indexed(
        mesh,
        segments=1,
        seq_len=_PREFILL_SEQ_LEN,
        prefix_capacity=2048,
        tap_layer=_TAP_LAYER_INDEX,
        depth=_TEST_DEPTH,
    )
    out_tap = jax.eval_shape(stack_tap, *args)
    assert len(out_tap) == _STACK_OUTPUTS_WITH_TAP
    assert out_tap[0].shape == expected_hidden_shape
    assert out_tap[4].shape == expected_hidden_shape

    for invalid_tap in (0, _TEST_DEPTH):
        with pytest.raises(ValueError, match="tap_layer"):
            prefill_megakernel.make_prefill_stack_sp_indexed(
                mesh,
                segments=1,
                seq_len=_PREFILL_SEQ_LEN,
                prefix_capacity=2048,
                tap_layer=invalid_tap,
                depth=_TEST_DEPTH,
            )
