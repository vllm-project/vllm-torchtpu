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

import ast
import functools
import inspect
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
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
    KDA_QKV_PARTS,
    MLA_LATENT_ROW_LANES,
    PAGE_TOKENS,
    PREFILL_WINDOWS,
    ROUTED_EXPERTS,
    STORE_TILE,
    TP_SIZE,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kernels import (
    helpers,
    load,
    prefill_megakernel,
)
from vllm_torchtpu.models.kimi_k3.megakernel.layout import KDA_LAYERS, MLA_LAYERS

# Production model geometry:
_DEFAULT_MODEL_CONFIG = load.Config()
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
_SC_GATHER_ROWS = 4096
_SC_GATHER_WIDTH = 3584
_SC_REDUCE_GROUP_SIZE = 16
_HELPER_ATOL = 1e-5
_HELPER_RTOL = 1e-5
_E8M0_EXPONENT_BIAS = 127
_KDA_DECAY_LOWER_BOUND = -1.0
_KDA_NORM_EPS = 1e-6
_SITU_GATE_BETA = _DEFAULT_MODEL_CONFIG.situ_beta
_SITU_UP_BETA = _DEFAULT_MODEL_CONFIG.situ_linear_beta
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
        KDA_QKV_PARTS,
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


def _is_type_checking_guard(node: ast.If) -> bool:
    """Return True when an ``if`` statement is a ``TYPE_CHECKING`` guard."""
    if isinstance(node.test, ast.Name):
        return node.test.id == "TYPE_CHECKING"
    if isinstance(node.test, ast.Attribute):
        return node.test.attr == "TYPE_CHECKING"
    return False


def _format_jax_src_import(stmt: ast.stmt) -> list[str]:
    """Return formatted ``jax._src`` import lines for one AST statement."""
    if isinstance(stmt, ast.Import):
        return [
            f"L{stmt.lineno}: import {alias.name}"
            for alias in stmt.names
            if alias.name == "jax._src" or alias.name.startswith("jax._src.")
        ]
    if isinstance(stmt, ast.ImportFrom):
        module = stmt.module or ""
        if module == "jax._src" or module.startswith("jax._src."):
            names = ", ".join(alias.name for alias in stmt.names)
            return [f"L{stmt.lineno}: from {module} import {names}"]
    return []


def _child_statement_blocks(stmt: ast.stmt) -> list[list[ast.stmt]]:
    """Return child statement lists executed at runtime under ``stmt``."""
    if isinstance(stmt, ast.If):
        blocks = [stmt.orelse]
        if not _is_type_checking_guard(stmt):
            blocks.insert(0, stmt.body)
        return blocks
    blocks = [
        child
        for attr in ("body", "orelse", "finalbody")
        if isinstance(child := getattr(stmt, attr, None), list)
    ]
    if isinstance(stmt, ast.Try):
        blocks.extend(handler.body for handler in stmt.handlers)
    return blocks


def _runtime_jax_src_imports(source: str) -> list[str]:
    """Collect ``jax._src`` imports executed outside ``if TYPE_CHECKING:`` blocks."""
    stack: list[list[ast.stmt]] = [ast.parse(source).body]
    hits: list[str] = []
    while stack:
        for stmt in stack.pop():
            hits.extend(_format_jax_src_import(stmt))
            stack.extend(_child_statement_blocks(stmt))
    return hits


def test_helpers_numerics_vs_fp64_reference() -> None:
    """Verify helpers math functions match FP64 NumPy reference formulas."""
    rng = np.random.default_rng(0)

    gate_fp64 = np.concatenate(
        [
            np.array([-30.0, -4.0, -0.5, 0.0, 0.5, 4.0, 30.0], dtype=np.float64),
            rng.standard_normal(25),
        ]
    )
    value_fp64 = np.concatenate(
        [
            np.array([-50.0, -25.0, -1.0, 0.0, 1.0, 25.0, 50.0], dtype=np.float64),
            rng.standard_normal(25),
        ]
    )
    expected_act = (
        _SITU_GATE_BETA
        * np.tanh(gate_fp64 / _SITU_GATE_BETA)
        * (1.0 / (1.0 + np.exp(-gate_fp64)))
        * _SITU_UP_BETA
        * np.tanh(value_fp64 / _SITU_UP_BETA)
    )
    actual_act = helpers.expert_activation(
        jnp.asarray(gate_fp64, dtype=jnp.float32),
        jnp.asarray(value_fp64, dtype=jnp.float32),
    )
    assert actual_act.dtype == jnp.float32
    np.testing.assert_allclose(
        np.asarray(actual_act, dtype=np.float64),
        expected_act,
        rtol=_HELPER_RTOL,
        atol=_HELPER_ATOL,
    )

    query_fp64 = rng.standard_normal((4, KDA_HEADS, HEAD_DIM))
    key_fp64 = rng.standard_normal((4, KDA_HEADS, HEAD_DIM))
    expected_q = (
        query_fp64
        / np.sqrt(
            np.sum(query_fp64 * query_fp64, axis=-1, keepdims=True) + _KDA_NORM_EPS
        )
        * (HEAD_DIM**-0.5)
    )
    expected_k = key_fp64 / np.sqrt(
        np.sum(key_fp64 * key_fp64, axis=-1, keepdims=True) + _KDA_NORM_EPS
    )
    actual_q, actual_k = helpers.kda_normalize_query_key(
        jnp.asarray(query_fp64, dtype=jnp.float32),
        jnp.asarray(key_fp64, dtype=jnp.float32),
        float(HEAD_DIM),
    )
    np.testing.assert_allclose(
        np.asarray(actual_q, dtype=np.float64),
        expected_q,
        rtol=_HELPER_RTOL,
        atol=_HELPER_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(actual_k, dtype=np.float64),
        expected_k,
        rtol=_HELPER_RTOL,
        atol=_HELPER_ATOL,
    )

    rate_fp64 = np.exp(rng.uniform(-2.0, 1.0, size=(1, KDA_HEADS, 1)))
    raw_gate_fp64 = rng.standard_normal((4, KDA_HEADS, HEAD_DIM))
    bias_fp64 = rng.standard_normal((1, KDA_HEADS, HEAD_DIM))
    expected_decay = _KDA_DECAY_LOWER_BOUND / (
        1.0 + np.exp(-rate_fp64 * (raw_gate_fp64 + bias_fp64))
    )
    actual_decay = helpers.kda_decay(
        _KDA_DECAY_LOWER_BOUND,
        jnp.asarray(rate_fp64, dtype=jnp.float32),
        jnp.asarray(raw_gate_fp64, dtype=jnp.float32),
        jnp.asarray(bias_fp64, dtype=jnp.float32),
    )
    np.testing.assert_allclose(
        np.asarray(actual_decay, dtype=np.float64),
        expected_decay,
        rtol=_HELPER_RTOL,
        atol=_HELPER_ATOL,
    )

    exponents = np.array([0, 1, 63, 126, 127, 128, 200, 254], dtype=np.uint8)
    expected_scales = np.where(
        exponents == 0,
        np.float32(0.0),
        np.ldexp(
            np.ones_like(exponents, dtype=np.float64),
            exponents.astype(np.int32) - _E8M0_EXPONENT_BIAS,
        ).astype(np.float32),
    )
    actual_scales = helpers.e8m0_scale(jnp.asarray(exponents, dtype=jnp.uint8))
    assert actual_scales.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(actual_scales), expected_scales)


def test_prefill_sparse_core_pl_kernel_aot_lowering_and_no_jax_src() -> None:
    """Verify prefill_megakernel drops runtime jax._src imports and uses pl.kernel."""
    source = Path(prefill_megakernel.__file__).read_text(encoding="utf-8")
    runtime_imports = _runtime_jax_src_imports(source)
    assert not runtime_imports, (
        f"Unexpected runtime jax._src imports: {runtime_imports}"
    )
    assert "core_map_kernel" not in vars(prefill_megakernel)
    assert "_empty_out_ref" not in vars(prefill_megakernel)

    mesh, replicated = _single_device_tpu7x_mesh()
    x = jax.ShapeDtypeStruct(
        (_SC_GATHER_ROWS, _SC_GATHER_WIDTH), jnp.bfloat16, sharding=replicated
    )
    indices = jax.ShapeDtypeStruct((_SC_GATHER_ROWS,), jnp.int32, sharding=replicated)
    start = jax.ShapeDtypeStruct((1,), jnp.int32, sharding=replicated)
    end = jax.ShapeDtypeStruct((1,), jnp.int32, sharding=replicated)
    weights = jax.ShapeDtypeStruct((_SC_GATHER_ROWS,), jnp.float32, sharding=replicated)
    valid = jax.ShapeDtypeStruct((_SC_GATHER_ROWS,), jnp.bool_, sharding=replicated)

    with jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
        gather_hlo = (
            jax.jit(prefill_megakernel.ragged_gather_v2)
            .lower(x, indices, start, end)
            .as_text()
        )
        reduce_hlo = (
            jax.jit(
                functools.partial(
                    prefill_megakernel.ragged_gather_reduce,
                    reduce_group_size=_SC_REDUCE_GROUP_SIZE,
                )
            )
            .lower(x, indices, weights, valid)
            .as_text()
        )
    assert "tpu_custom_call" in gather_hlo
    assert "AllocateBuffer" not in gather_hlo
    assert "tpu_custom_call" in reduce_hlo
    assert "AllocateBuffer" not in reduce_hlo
