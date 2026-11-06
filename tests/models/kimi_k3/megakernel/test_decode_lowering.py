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
"""Deviceless TPU v7x lowering and static contract checks for Kimi-K3 decode."""

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
from jax._src import source_info_util
from jax._src.lib import hlo as jax_hlo
from jax.experimental import topologies
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.megakernel import pool_alias as shared_pool_alias
from vllm_torchtpu.models.kimi_k3.megakernel import config as kernel_config
from vllm_torchtpu.models.kimi_k3.megakernel.kernels import (
    decode_megakernel,
    load,
    mla_dcp,
    paged,
)

LAYERS = 4
FULL_LAYERS = 93
EXPECTED_FAMILIES = 37
EXPECTED_SINGLE_COPY_BYTES = 51_442_171_196
EXPECTED_SINGLE_COPY_GIB = 47.909
BYTES_PER_GIB = 1024**3
FLOAT32_BYTES = 4
BF16_BYTES = 2
UINT8_BYTES = 1
EXPECTED_MLA_WORKSPACE_FIELDS = 10
EXPECTED_CUSTOM_CALLS = 2
PAGES = 16
BATCH = 8
RANKS = 32
DCP_POOL_LAYERS = 24
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

# Pre-cleanup canonical HLO oracle across all 7 decode modes:
# (batch_size, mode_key, expected_byte_len, sha256_upper_half, sha256_lower_half).
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
}

# Pre-cleanup canonical HLO oracle across all 7 decode modes under jax 0.11.2.
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
}

CANONICAL_HLO_ORACLE: dict[str, tuple[int, str, int, str, str]] = (
    _CANONICAL_HLO_ORACLE_JAX_0_11_1
    if jax.__version__ == "0.11.1"
    else _CANONICAL_HLO_ORACLE_JAX_0_11_2
)


@functools.lru_cache(maxsize=1)
def _cached_tp_mesh() -> Mesh:
    """Build and cache a 32-device TPU v7x 2x2x4 mesh without TPU hardware."""
    devices = topologies.get_topology_desc(
        topology_name="tpu7x_2x2x4", platform="tpu"
    ).devices
    return Mesh(devices, ("tp",))


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


@functools.lru_cache(maxsize=16)
def _lower_canonical_hlo(batch_size: int, mode: str) -> str:
    """Lower one decode mode on the TPU v7x mesh and return canonical HLO text."""
    jax.config.update("jax_include_full_tracebacks_in_locations", bool(0))
    source_info_util.register_exclusion("/")
    active = kernel_config.active()
    assert (active.attention_dp, active.dcp_group) == (1, RANKS), active
    scatter, moe_out, moe_reduce_scatter, sequence_parallel, aux_layers = MODES[mode]
    weights = _abstract_weights()

    def step(*arrays: jax.Array) -> object:
        """Invoke `decode_local` on per-rank abstract arrays."""
        tokens, positions, conv_state, rec_state, pool, block_tables = arrays[:6]
        return decode_megakernel.decode_local(
            dict(zip(weights, arrays[6:], strict=True)),
            (conv_state, rec_state, pool),
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
        jax.ShapeDtypeStruct((batch_size, LAYERS, 3, 3, 4, 128), jnp.float32),
        jax.ShapeDtypeStruct((batch_size, LAYERS, 3, 128, 128), jnp.float32),
        jax.ShapeDtypeStruct((PAGES, 128, 4, 640), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((batch_size, PAGES), jnp.int32),
    )
    sharded = jax.shard_map(
        step,
        mesh=_cached_tp_mesh(),
        in_specs=(P(),) * (len(inputs) + len(weights)),
        out_specs=P(),
        check_vma=False,
    )
    lowered = jax.jit(sharded).lower(*inputs, *weights.values())
    compiler_ir = lowered.compiler_ir(dialect="hlo")
    assert compiler_ir is not None
    hlo_module = compiler_ir.as_hlo_module()
    options = jax_hlo.HloPrintOptions.canonical()
    options.print_backend_config = True
    options.print_metadata = False
    return hlo_module.to_string(options)


def test_decode_megakernel_lowers_to_tpu_custom_call() -> None:
    """Verify each decode mode lowers to a Mosaic custom call on TPU v7x."""
    tp_mesh = _cached_tp_mesh()
    assert tp_mesh.devices.size == RANKS
    for mode in sorted(MODES):
        hlo = _lower_canonical_hlo(BATCH, mode)
        assert "tpu_custom_call" in hlo


def test_decode_canonical_hlo_sha256_oracle() -> None:
    """Verify 0 B canonical HLO diff and pruned signatures across 7 decode modes."""
    tp_mesh = _cached_tp_mesh()
    assert tp_mesh.devices.size == RANKS
    module_symbols = vars(decode_megakernel)
    padded_rows_cls = module_symbols["_PaddedBf16Rows"]
    assert isinstance(padded_rows_cls, type)
    assert issubclass(padded_rows_cls, shared_pool_alias.Reinterpret)
    expert_dot_fn = module_symbols["_quantized_expert_dot"]
    expert_dot_params = inspect.signature(expert_dot_fn).parameters
    assert "output_size" not in expert_dot_params
    stack_params = inspect.signature(decode_megakernel.stack).parameters
    assert "gate_first_kda_projection" not in stack_params
    mla_ws_fields = dataclasses.fields(
        decode_megakernel.MultiHeadLatentAttentionWorkspace
    )
    assert len(mla_ws_fields) == EXPECTED_MLA_WORKSPACE_FIELDS

    for oracle_mode in sorted(CANONICAL_HLO_ORACLE):
        batch_size, mode_key, expected_bytes, sha_hi, sha_lo = CANONICAL_HLO_ORACLE[
            oracle_mode
        ]
        canon_text = _lower_canonical_hlo(batch_size, mode_key)
        canon_bytes = canon_text.encode("utf-8")
        assert len(canon_bytes) == expected_bytes, oracle_mode
        assert (
            canon_text.count('custom_call_target="tpu_custom_call"')
            == EXPECTED_CUSTOM_CALLS
        ), oracle_mode
        digest = hashlib.sha256(canon_bytes).hexdigest()
        assert digest == f"{sha_hi}{sha_lo}", oracle_mode


def test_decode_dataclasses_register_lazily_on_instantiation() -> None:
    """Verify lazy PyTree dataclass registration and per-rank byte accounting."""
    source_path = Path(inspect.getfile(decode_megakernel))
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in tree.body:
        for decorator in getattr(node, "decorator_list", ()):
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            decorator_name = ast.unparse(target).rsplit(".", 1)[-1]
            assert not decorator_name.startswith("register")

    module_symbols = vars(decode_megakernel)
    register_fn = module_symbols.get("_ensure_dataclasses_registered")
    assert callable(register_fn)

    padded_rows_cls = module_symbols["_PaddedBf16Rows"]
    padded_view = padded_rows_cls((16, 128))
    leaves, treedef = jax.tree_util.tree_flatten(padded_view)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt == padded_view

    full_shapes = load.selected_local_shapes(layers=FULL_LAYERS)
    assert len(full_shapes) == EXPECTED_FAMILIES
    total_bytes = sum(
        math.prod(shape)
        * (
            FLOAT32_BYTES
            if name in (*_FLOAT32_WEIGHTS, *_UINT32_WEIGHTS)
            else (UINT8_BYTES if name in _UINT8_WEIGHTS else BF16_BYTES)
        )
        for name, shape in full_shapes.items()
    )
    assert total_bytes == EXPECTED_SINGLE_COPY_BYTES
    assert round(total_bytes / BYTES_PER_GIB, 3) == EXPECTED_SINGLE_COPY_GIB


def test_paged_and_mla_dcp_static_contracts() -> None:
    """Verify refactored paged and mla_dcp contracts and CPU scatter commit."""
    expected_commit_pos_args = 5
    expected_decode_pos_args = 4
    commit_params = list(inspect.signature(paged.commit_latent_rows).parameters)
    assert commit_params[:expected_commit_pos_args] == [
        "pools",
        "rows",
        "block_tables",
        "positions",
        "pool_map",
    ]
    decode_paged_params = list(inspect.signature(paged.decode_paged).parameters)
    assert decode_paged_params[:expected_decode_pos_args] == [
        "weights",
        "conv_slots",
        "recurrent_slots",
        "pool",
    ]

    pool_zero = jnp.zeros((2, 128, 2, 8), dtype=jnp.bfloat16)
    latent_rows = jnp.ones((1, 1, 8), dtype=jnp.bfloat16)
    block_tables = jnp.zeros((1, 1), dtype=jnp.int32)
    positions = jnp.zeros((1,), dtype=jnp.int32)
    cpu_mesh = Mesh(jax.devices()[:1], ("tp",))
    committed = jax.shard_map(
        lambda pool_arr, row_arr, blk_arr, pos_arr: paged.commit_latent_rows(
            (pool_arr,),
            row_arr,
            blk_arr,
            pos_arr,
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

    conv_slots = jnp.zeros((2, 2), dtype=jnp.bfloat16)
    rec_slots = jnp.zeros((2, 2), dtype=jnp.bfloat16)
    slots = jnp.array([1], dtype=jnp.int32)
    step_cfg = paged.PagedKdaStepConfig(batch_size=1, dp=1, owners=4, state_row=None)
    conv, rec, state_index, group_rows, in_place = paged.prepare_paged_kda_inputs(
        conv_slots,
        rec_slots,
        slots,
        step_cfg,
    )
    assert conv.shape == (1, 2)
    assert rec.shape == (1, 2)
    assert state_index.shape == (1,)
    assert group_rows is None
    assert not in_place

    expected_group = 2
    p_scratch = mla_dcp.persistent_scratch(BATCH, force_chunks=True, sequence_rows=1)
    s_scratch = mla_dcp.scoped_scratch(BATCH, force_chunks=True, sequence_rows=1)
    assert len(p_scratch) > 0
    assert len(s_scratch) > 0
    loop_layout = mla_dcp.ChunkLoopLayout(chunk=1, count=expected_group, ring_slots=2)
    seq_spec = mla_dcp.SequenceMergeSpec(group=expected_group, slots=1, sequence_rows=1)
    key_spec = mla_dcp.KeyMergeSpec(group=expected_group, lse_rows=8)
    assert loop_layout.count == expected_group
    assert seq_spec.group == expected_group
    assert key_spec.group == expected_group
