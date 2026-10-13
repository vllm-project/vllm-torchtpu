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
"""`TPU_IDX_TOPK_SCHED_GROUPS`: a per-layer XLA scheduling group in DCP indexer
layers.

G2(L) = 2**16 + L holds the layer's cross-rank SparseCore top-k (the
`sparsecore_topk_dcp_merge` stage of `streamindex_topk_dcp`) and its q_b_proj
(the fp8_skinny kernel). The two members live in different jax ops, so each
test checks that one member carries its id in the exported
(`platforms=["tpu"]`) module, for two layers, and that nothing is annotated
without the flag or outside an indexer layer. Whether XLA then schedules
q_b_proj under the top-k is a TPU question, outside these tests. The SparseCore
kernels cannot be built without a TPU, so the top-k side uses the stub pattern
of test_dsv4_sparsecore_topk.py.
"""

import functools
import re

import jax
import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as sc_module
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    DCP_AXIS_NAME,
    config,
    streamindex_topk_dcp,
)
from vllm_torchtpu.layers.adapter import linear_common

FLAG = "TPU_IDX_TOPK_SCHED_GROUPS"
LAYERS = (2, 14)
S = jax.ShapeDtypeStruct


def _group(layer):
    return config.indexer_layer_scheduling_group(layer)


def _ids(text):
    return re.findall(r'_scheduling_group_id = "(\d+)"', text)


def _export(fn, *args):
    return jax.export.export(jax.jit(fn), platforms=["tpu"])(*args)


def _asm(exported):
    """The module without location info, the form torch_tpu fingerprints."""
    from jax.interpreters import mlir

    with mlir.make_ir_context():
        module = mlir.ir.Module.parse(exported.mlir_module())
        return str(module.operation.get_asm(enable_debug_info=False))


@pytest.fixture
def tpu7x_info(monkeypatch):
    """Lower Mosaic kernels for TPU7x on a CPU host: kernels that size
    themselves from `get_tpu_info()` otherwise refuse the CPU device kind."""
    from jax._src.pallas.mosaic import tpu_info

    kind = jax.devices()[0].device_kind
    if tpu_info.chip_version_from_device_kind(kind) is None:
        monkeypatch.setitem(
            tpu_info.registry,
            kind,
            lambda: tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_7X, 1),
        )
    tpu_info.get_tpu_info.cache_clear()
    yield
    tpu_info.get_tpu_info.cache_clear()


@pytest.fixture
def registry(monkeypatch):
    """A fresh indexer-layer record, so tests cannot see each other's layers."""
    layers = set()
    monkeypatch.setattr(config, "_indexer_layers", layers)
    return layers


def test_layer_ids_are_fixed_distinct_and_clear_of_the_counter():
    ids = [_group(layer) for layer in range(100)]
    assert len(set(ids)) == len(ids)
    assert min(ids) == config.INDEXER_LAYER_GROUP_BASE == 1 << 16
    assert _group(14) == 65550
    for name in (
        "model.layers.14.self_attn.indexer.k_cache",
        "model.layers.14.self_attn.q_b_proj",
        "pallas::streamidx_topk_dcp_model_layers_14_self_attn_indexer_k_cache",
    ):
        assert config.layer_index_of(name) == 14, name
    assert config.layer_index_of("model.norm") is None
    assert config.layer_index_of("model.xlayers.3.q_b_proj") is None
    with pytest.raises(ValueError):
        _group(-1)


def test_only_registered_layers_have_groups(registry):
    assert config.registered_layer_scheduling_group("model.layers.14.x") is None
    assert config.register_indexer_layer(
        "model.layers.14.self_attn.indexer.k_cache"
    ) == _group(14)
    assert config.registered_layer_scheduling_group(
        "model.layers.14.self_attn.q_b_proj"
    ) == _group(14)
    assert config.registered_layer_scheduling_group("model.layers.15.x") is None
    with pytest.raises(ValueError):
        config.register_indexer_layer("model.norm")


# --------------------------------------------------------------------------- #
# SparseCore top-k: the cross-rank merge -> G2, the local top-k untouched.
# --------------------------------------------------------------------------- #
DCP = 2
T, H_IDX, D_IDX, TOPK = 16, 32, 128, 2048


def _topk_group_ids(monkeypatch, group, chunk_tokens=256):
    """The id each `_sc_topk_direct` call of the decode-shaped DCP top-k gets,
    in call order, plus the exported module text."""
    seen = []
    # A trace cached by an earlier call would skip the stub and record nothing.
    jax.clear_caches()

    def _stub(scores, k_, row_lengths, *, scheduling_group_id=None, **kwargs):
        del row_lengths
        seen.append(scheduling_group_id)
        out = jnp.zeros((scores.shape[0], k_), jnp.int32)
        return (out, out) if kwargs.get("return_scores", True) else out

    monkeypatch.setattr(sc_module, "_sc_topk_direct", _stub)
    mesh = jax.sharding.AbstractMesh((DCP,), (DCP_AXIS_NAME,))
    args = (
        S((T, H_IDX, D_IDX), jnp.float8_e4m3fn),
        S((T, H_IDX), jnp.float32),
        # The cache is the one DCP-sharded input (in_specs P(dcp)); newer JAX
        # checks the spec against the argument's sharding.
        S(
            (DCP * 16, 64, 4, 256),
            jnp.uint8,
            sharding=jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec(DCP_AXIS_NAME)
            ),
        ),
        S((8,), jnp.int32),
        S((8 * 16,), jnp.int32),
        S((9,), jnp.int32),
        S((3,), jnp.int32),
    )
    fn = functools.partial(
        streamindex_topk_dcp.__wrapped__,
        mesh=mesh,
        k=TOPK,
        compression_ratio=1,
        dcp_size=DCP,
        interleave_size=4,
        # As VllmTPUSparseAttnIndexer._build_streamidx_dcp_op builds it.
        num_kv_pages_per_block=(2, 2, 2),
        num_queries_per_block=(1, 128, 128),
        chunk_tokens=chunk_tokens,
        **({} if group == "default" else dict(global_scheduling_group_id=group)),
    )
    with jax.sharding.use_abstract_mesh(mesh):
        text = _asm(_export(fn, *args))
    return seen, text


@pytest.mark.parametrize("layer", LAYERS)
def test_dcp_cross_rank_topk_gets_the_layer_group(monkeypatch, tpu7x_info, layer):
    g2 = _group(layer)
    seen, _ = _topk_group_ids(monkeypatch, g2)
    # Calls at decode (T=16, one chunk, no early-exit conditional on the DCP
    # path): the local top-k, then the cross-rank merge (partition 1).
    assert seen == [None, g2]


def test_dcp_topk_without_a_group_is_unannotated_and_unchanged(monkeypatch, tpu7x_info):
    seen, default = _topk_group_ids(monkeypatch, "default")
    assert seen == [None, None]
    _, explicit_none = _topk_group_ids(monkeypatch, None)
    assert default == explicit_none
    assert "_scheduling_group_id" not in default


def test_chunked_topk_keeps_its_own_groups(monkeypatch, tpu7x_info):
    """With > 1 chunk the counter's pipeline groups own the local stage; the
    layer's G2 marks only the merge."""
    g2 = _group(14)
    config.reset_scheduling_group_ids()
    seen, _ = _topk_group_ids(monkeypatch, g2, chunk_tokens=8)
    assert seen.count(g2) == 1 and seen[-1] == g2
    assert any(s is not None and s < config.INDEXER_LAYER_GROUP_BASE for s in seen)


# --------------------------------------------------------------------------- #
# q_b_proj: the fp8_skinny kernel -> G2.
# --------------------------------------------------------------------------- #
# GLM-5.2 q_b_proj under DCP query replication: [16, 2048] x [2048, 16384].
QB_ARGS = (
    S((16, 2048), jnp.bfloat16),
    S((2048, 16384), jnp.float8_e4m3fn),
    S((16384,), jnp.float32),
)


def _hlo(exported):
    from jax._src.lib import _jax

    comp = _jax.mlir.mlir_module_to_xla_computation(exported.mlir_module_serialized)
    return comp.as_hlo_text()


def _custom_call_lines(hlo):
    return [
        line
        for line in hlo.splitlines()
        if 'custom_call_target="tpu_custom_call"' in line
    ]


@pytest.mark.parametrize("layer", LAYERS)
def test_q_b_proj_kernel_carries_g2(layer):
    g2 = _group(layer)
    exported = _export(linear_common._fp8_linear_pallas_grouped_jax(g2), *QB_ARGS)
    assert _ids(_asm(exported)) == [str(g2)]
    # On the kernel itself, and still there once the module is HLO.
    (kernel,) = _custom_call_lines(_hlo(exported))
    assert f'_scheduling_group_id="{g2}"' in kernel


def test_q_b_proj_grouped_differs_from_the_shared_op_only_by_the_id():
    base = _asm(_export(linear_common._fp8_linear_pallas_jax, *QB_ARGS))
    assert "_scheduling_group_id" not in base
    grouped = _asm(
        _export(linear_common._fp8_linear_pallas_grouped_jax(_group(14)), *QB_ARGS)
    )

    def strip(text):
        text = re.sub(r"jit__fp8_linear_pallas_\w+", "jit_F", text)
        text = re.sub(r'_scheduling_group_id = "\d+",? ?', "", text)
        return text

    assert strip(grouped) == strip(base)


def test_q_b_proj_group_goes_to_indexer_layers_only(monkeypatch, registry):
    from vllm_torchtpu.layers.adapter.quantization import fp8 as fp8_mod

    config.register_indexer_layer("model.layers.14.self_attn.indexer.k_cache")
    qb = "model.layers.14.self_attn.q_b_proj"
    monkeypatch.delenv(FLAG, raising=False)
    assert fp8_mod._fp8_pallas_sched_group(qb) is None
    monkeypatch.setenv(FLAG, "0")
    assert fp8_mod._fp8_pallas_sched_group(qb) is None
    monkeypatch.setenv(FLAG, "1")
    assert fp8_mod._fp8_pallas_sched_group(qb) == _group(14)
    assert fp8_mod._fp8_pallas_sched_group("model.layers.15.self_attn.q_b_proj") is None
    assert fp8_mod._fp8_pallas_sched_group("model.layers.14.self_attn.o_proj") is None
    assert (
        fp8_mod._fp8_pallas_sched_group("model.layers.14.self_attn.fused_qkv_a_proj")
        is None
    )


def test_grouped_op_is_its_own_op_and_the_shared_op_is_unchanged():
    shared = linear_common.get_fp8_linear_pallas_op(False)
    assert linear_common.get_fp8_linear_pallas_op(False, None) is shared
    g2, g2b = (linear_common.get_fp8_linear_pallas_op(False, _group(n)) for n in LAYERS)
    assert len({id(shared), id(g2), id(g2b)}) == 3
    assert linear_common.get_fp8_linear_pallas_op(False, _group(LAYERS[0])) is g2
