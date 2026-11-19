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
"""Unmocked TPU v7x AOT lowering and contract tests for Kimi-K3 prefill megakernel."""

from __future__ import annotations

import functools
import inspect

import jax
import jax.numpy as jnp
import pytest
from jax.experimental import topologies
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.models.kimi_k3.megakernel.common.constants import (
    AXIS_SP,
    CONV_TAPS,
    DP2_GROUPS,
    HEAD_DIM,
    HIDDEN_SIZE,
    KDA_HEADS,
    MLA_LATENT_ROW_LANES,
    PAGE_TOKENS,
    PREFILL_WINDOWS,
    ROUTED_EXPERTS,
    STORE_TILE,
    TP_SIZE,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kernels import load, prefill_megakernel
from vllm_torchtpu.models.kimi_k3.megakernel.layout import KDA_LAYERS, MLA_LAYERS

# Production model geometry:
_MLA_HEADS_PER_RANK = KDA_HEADS
_ATTENTION_PROJECTION_WIDTH = KDA_HEADS * HEAD_DIM
_KEY_DIM = 2 * HEAD_DIM
_VALUE_DIM = HEAD_DIM
# 512 KV-LoRA + 64 RoPE = 576 compressed MLA latent dims.
_MLA_LATENT_DIM = 576
_NO_ARENA_SEGMENT_FIELDS = 3
_ARENA_SEGMENT_FIELDS = 5
_PREFILL_SEQ_LEN = PREFILL_WINDOWS[0]
_LOCAL_EXPERTS = ROUTED_EXPERTS // (TP_SIZE // DP2_GROUPS)
# Reduced layer count, tap index, and pool blocks for AOT compile-speed fixtures:
_TEST_DEPTH = 20
_TAP_LAYER_INDEX = 18
_CP_POOL_BLOCKS = 64
_CP_POOL_ROWS = 2
_CP_ID_SLOTS = 10
_PREFIX_CAPACITY = PREFILL_WINDOWS[2]
_STACK_OUTPUTS_NO_TAP = 4
_STACK_OUTPUTS_WITH_TAP = 5


def _single_device_tpu7x_mesh() -> tuple[Mesh, NamedSharding]:
    """Return a single-device TPU v7x AOT mesh (from 2x2x4) and sharding."""
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
    sharding = NamedSharding(mesh, P(AXIS_SP))
    replicated = NamedSharding(mesh, P())
    shapes = load.selected_local_shapes(layers=depth)

    hidden = jax.ShapeDtypeStruct(
        (TP_SIZE, seq_len // TP_SIZE, HIDDEN_SIZE),
        jnp.bfloat16,
        sharding=sharding,
    )
    common = {
        name: jax.ShapeDtypeStruct(
            (TP_SIZE, *shapes[name]),
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
            (TP_SIZE, depth, _ATTENTION_PROJECTION_WIDTH, HIDDEN_SIZE)
            if key == "k_o"
            else (TP_SIZE, *shapes[key]),
            jnp.float32
            if key in ("k_a_log", "k_dt", "k_conv", "k_norm")
            else jnp.bfloat16,
            sharding=sharding,
        )
        for key in kda_keys
    }

    mla_families = {
        key: jax.ShapeDtypeStruct(
            (TP_SIZE, depth, _ATTENTION_PROJECTION_WIDTH, HIDDEN_SIZE)
            if key == "m_o"
            else (TP_SIZE, *shapes[key]),
            jnp.float32 if "norm" in key else jnp.bfloat16,
            sharding=sharding,
        )
        for key in prefill_megakernel.MLA_FAMILIES
    }

    moe_overrides: dict[str, tuple[tuple[int, ...], jnp.dtype]] = {
        "latent_down": ((TP_SIZE, depth, HIDDEN_SIZE, 4096), jnp.bfloat16),
        "latent_up": ((TP_SIZE, depth, 3584, HIDDEN_SIZE), jnp.bfloat16),
        "shared_gu": ((TP_SIZE, depth, HIDDEN_SIZE, 8192), jnp.bfloat16),
        "shared_down": ((TP_SIZE, depth, 3072, HIDDEN_SIZE), jnp.bfloat16),
        "expert_gu": ((TP_SIZE, depth, _LOCAL_EXPERTS, 3584, 3072), jnp.uint4),
        "expert_gus": ((TP_SIZE, depth, _LOCAL_EXPERTS, 28, 3072), jnp.uint8),
        "expert_down": ((TP_SIZE, depth, _LOCAL_EXPERTS, 1536, 3584), jnp.uint4),
        "expert_ds": ((TP_SIZE, depth, _LOCAL_EXPERTS, 12, 3584), jnp.uint8),
    }
    moe_families: dict[str, jax.ShapeDtypeStruct] = {}
    for key in prefill_megakernel.MOE_FAMILIES:
        if key in moe_overrides:
            override_shape, override_dtype = moe_overrides[key]
            moe_families[key] = jax.ShapeDtypeStruct(
                override_shape, override_dtype, sharding=sharding
            )
        else:
            weight_dtype = (
                jnp.float32 if key in ("router_bias", "latent_norm") else jnp.bfloat16
            )
            moe_families[key] = jax.ShapeDtypeStruct(
                (TP_SIZE, *shapes[key]), weight_dtype, sharding=sharding
            )

    kda_rec_shape = (
        TP_SIZE,
        KDA_LAYERS,
        segments,
        KDA_HEADS,
        _VALUE_DIM,
        _VALUE_DIM,
    )
    kda_hist_shape = (
        TP_SIZE,
        KDA_LAYERS,
        segments,
        CONV_TAPS,
        CONV_TAPS,
        KDA_HEADS,
        _VALUE_DIM,
    )
    kda_state = (
        jax.ShapeDtypeStruct(kda_rec_shape, jnp.float32, sharding=sharding),
        jax.ShapeDtypeStruct(kda_hist_shape, jnp.bfloat16, sharding=sharding),
        jax.ShapeDtypeStruct(kda_rec_shape, jnp.float32, sharding=sharding),
        jax.ShapeDtypeStruct(kda_hist_shape, jnp.bfloat16, sharding=sharding),
    )
    mla_state = (
        jax.ShapeDtypeStruct(
            (TP_SIZE, MLA_LAYERS, seq_len, _MLA_LATENT_DIM),
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
                (TP_SIZE, _CP_POOL_BLOCKS, _CP_POOL_ROWS, MLA_LATENT_ROW_LANES),
                jnp.bfloat16,
                sharding=sharding,
            ),
        ),
        ids=jax.ShapeDtypeStruct((TP_SIZE, _CP_ID_SLOTS), jnp.int32, sharding=sharding),
        local=jax.ShapeDtypeStruct(
            (TP_SIZE, 2 * segments), jnp.int32, sharding=sharding
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


def test_prefill_stack_explicit_keyword_arguments_preserved() -> None:
    """Verify prefill builders and returned closures preserve explicit parameters."""
    variadic_kinds = {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
    for builder in (
        prefill_megakernel.make_embedding_sp,
        prefill_megakernel.make_regular_group8_sp_indexed,
        prefill_megakernel.make_final_mla_sp_indexed,
        prefill_megakernel.make_prefill_stack_sp_indexed,
        prefill_megakernel.make_prefill_logits_sp,
    ):
        sig = inspect.signature(builder)
        assert not any(
            param.kind in variadic_kinds for param in sig.parameters.values()
        )
    builder_sig = inspect.signature(prefill_megakernel.make_prefill_stack_sp_indexed)
    for expected_kw in (
        "mesh",
        "seq_len",
        "depth",
        "segments",
        "prefix_capacity",
        "mla_flash_tile",
        "unroll",
        "tap_layer",
    ):
        assert expected_kw in builder_sig.parameters

    devices = topologies.get_topology_desc(
        topology_name="tpu7x_2x2x4", platform="tpu"
    ).devices
    mesh = Mesh(devices, (AXIS_SP,))
    stack_fn = prefill_megakernel.make_prefill_stack_sp_indexed(
        mesh,
        seq_len=_PREFILL_SEQ_LEN,
        depth=_TEST_DEPTH,
        segments=1,
        prefix_capacity=_PREFIX_CAPACITY,
    )
    run_sig = inspect.signature(stack_fn)
    assert not any(
        param.kind in variadic_kinds for param in run_sig.parameters.values()
    )
    for expected_run_arg in ("segment_desc", "arena"):
        assert expected_run_arg in run_sig.parameters


def test_mla_flash_prefill_packed_traces_with_and_without_arena() -> None:
    """Verify mla_flash_prefill_packed lowers on TPU v7x with and without arena."""
    mesh, replicated = _single_device_tpu7x_mesh()
    query = jax.ShapeDtypeStruct(
        (_MLA_HEADS_PER_RANK, PAGE_TOKENS, _KEY_DIM), jnp.bfloat16, sharding=replicated
    )
    key = jax.ShapeDtypeStruct(
        (_MLA_HEADS_PER_RANK, PAGE_TOKENS, _KEY_DIM), jnp.bfloat16, sharding=replicated
    )
    value = jax.ShapeDtypeStruct(
        (_MLA_HEADS_PER_RANK, PAGE_TOKENS, _VALUE_DIM),
        jnp.bfloat16,
        sharding=replicated,
    )
    length = jax.ShapeDtypeStruct((1,), jnp.int32, sharding=replicated)
    segments_no_arena = jax.ShapeDtypeStruct(
        (_NO_ARENA_SEGMENT_FIELDS,), jnp.int32, sharding=replicated
    )
    segments_arena = jax.ShapeDtypeStruct(
        (_ARENA_SEGMENT_FIELDS,), jnp.int32, sharding=replicated
    )
    arena_keys = jax.ShapeDtypeStruct(
        (1, _MLA_HEADS_PER_RANK, STORE_TILE, _KEY_DIM),
        jnp.bfloat16,
        sharding=replicated,
    )
    arena_values = jax.ShapeDtypeStruct(
        (1, _MLA_HEADS_PER_RANK, _VALUE_DIM, STORE_TILE),
        jnp.bfloat16,
        sharding=replicated,
    )

    with jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
        lowered_no_arena = jax.jit(
            functools.partial(
                prefill_megakernel.mla_flash_prefill_packed, count=1, arena=None
            )
        ).lower(query, key, value, length, segments_no_arena)
        assert "tpu_custom_call" in lowered_no_arena.as_text()

        lowered_arena = jax.jit(
            lambda query_arg,
            key_arg,
            value_arg,
            length_arg,
            segment_arg,
            arena_keys_arg,
            arena_values_arg: prefill_megakernel.mla_flash_prefill_packed(
                query_arg,
                key_arg,
                value_arg,
                length_arg,
                segment_arg,
                count=1,
                arena=(arena_keys_arg, arena_values_arg),
            )
        ).lower(query, key, value, length, segments_arena, arena_keys, arena_values)
        assert "tpu_custom_call" in lowered_arena.as_text()

    with pytest.raises(ValueError, match="segments must be"):
        jax.eval_shape(
            functools.partial(
                prefill_megakernel.mla_flash_prefill_packed,
                count=1,
                arena=(
                    jnp.zeros(
                        (1, _MLA_HEADS_PER_RANK, STORE_TILE, _KEY_DIM), jnp.bfloat16
                    ),
                    jnp.zeros(
                        (1, _MLA_HEADS_PER_RANK, _VALUE_DIM, STORE_TILE), jnp.bfloat16
                    ),
                ),
            ),
            jnp.zeros((_MLA_HEADS_PER_RANK, PAGE_TOKENS, _KEY_DIM), jnp.bfloat16),
            jnp.zeros((_MLA_HEADS_PER_RANK, PAGE_TOKENS, _KEY_DIM), jnp.bfloat16),
            jnp.zeros((_MLA_HEADS_PER_RANK, PAGE_TOKENS, _VALUE_DIM), jnp.bfloat16),
            jnp.array([PAGE_TOKENS], jnp.int32),
            jnp.zeros((_NO_ARENA_SEGMENT_FIELDS,), jnp.int32),
        )


def test_mla_flash_prefill_packed_lowers_with_multiblock_arena() -> None:
    """Verify mla_flash_prefill_packed lowers with a multi-block arena."""
    mesh, replicated = _single_device_tpu7x_mesh()
    query = jax.ShapeDtypeStruct(
        (_MLA_HEADS_PER_RANK, PAGE_TOKENS, _KEY_DIM), jnp.bfloat16, sharding=replicated
    )
    key = jax.ShapeDtypeStruct(
        (_MLA_HEADS_PER_RANK, PAGE_TOKENS, _KEY_DIM), jnp.bfloat16, sharding=replicated
    )
    value = jax.ShapeDtypeStruct(
        (_MLA_HEADS_PER_RANK, PAGE_TOKENS, _VALUE_DIM),
        jnp.bfloat16,
        sharding=replicated,
    )
    length = jax.ShapeDtypeStruct((1,), jnp.int32, sharding=replicated)
    segments_arena = jax.ShapeDtypeStruct(
        (_ARENA_SEGMENT_FIELDS,), jnp.int32, sharding=replicated
    )
    arena_keys = jax.ShapeDtypeStruct(
        (1, _MLA_HEADS_PER_RANK, PAGE_TOKENS, _KEY_DIM),
        jnp.bfloat16,
        sharding=replicated,
    )
    arena_values = jax.ShapeDtypeStruct(
        (1, _MLA_HEADS_PER_RANK, _VALUE_DIM, PAGE_TOKENS),
        jnp.bfloat16,
        sharding=replicated,
    )

    with jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
        lowered = jax.jit(
            lambda query_arg,
            key_arg,
            value_arg,
            length_arg,
            segment_arg,
            arena_keys_arg,
            arena_values_arg: prefill_megakernel.mla_flash_prefill_packed(
                query_arg,
                key_arg,
                value_arg,
                length_arg,
                segment_arg,
                count=1,
                arena=(arena_keys_arg, arena_values_arg),
            )
        ).lower(query, key, value, length, segments_arena, arena_keys, arena_values)
        hlo_text = lowered.as_text()
    assert "tpu_custom_call" in hlo_text
    assert lowered.out_info.shape == (_MLA_HEADS_PER_RANK, PAGE_TOKENS, _VALUE_DIM)


def test_prefill_stack_traces_with_tap_layer_none_and_layer_index() -> None:
    """Verify make_prefill_stack_sp_indexed traces with tap_layer and keyword arena."""
    devices = topologies.get_topology_desc(
        topology_name="tpu7x_2x2x4", platform="tpu"
    ).devices
    mesh = Mesh(devices, (AXIS_SP,))
    args = _build_prefill_stack_inputs(
        mesh, seq_len=_PREFILL_SEQ_LEN, segments=1, depth=_TEST_DEPTH
    )
    expected_hidden_shape = (TP_SIZE, _PREFILL_SEQ_LEN // TP_SIZE, HIDDEN_SIZE)

    stack_none = prefill_megakernel.make_prefill_stack_sp_indexed(
        mesh,
        segments=1,
        seq_len=_PREFILL_SEQ_LEN,
        prefix_capacity=_PREFIX_CAPACITY,
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
        prefix_capacity=_PREFIX_CAPACITY,
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
                prefix_capacity=_PREFIX_CAPACITY,
                tap_layer=invalid_tap,
                depth=_TEST_DEPTH,
            )
