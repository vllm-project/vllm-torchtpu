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
"""Strict verification tests for Kimi K3 megakernel Phase 1 isolation."""

from __future__ import annotations

import ast
import dataclasses
import importlib
import json
import operator
import pathlib
from typing import Any

import jax
import jax.numpy as jnp
import pytest
from jax.experimental import topologies
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.models.kimi_k3.megakernel.kernels import (
    dspark_mla,
    dspark_mla_kernel,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kernels import (
    prefill_megakernel as _pmk_init,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kernels.prefill_moe import (
    fused,
    indexed_ragged_gmm,
    moe_down_valu,
)

_Dynamic = Any
_CONST_10 = 10
_CONST_128 = 128
_CONST_16 = 16
_CONST_1792 = 1792
_CONST_2 = 2
_CONST_2048 = 2048
_CONST_256 = 256
_CONST_3 = 3
_CONST_32 = 32
_CONST_4096 = 4096
_CONST_512 = 512
_CONST_64 = 64
_CONST_7 = 7
_CONST_7168 = 7168
_CONST_8 = 8
_CONST_896 = 896
_CONST_93 = 93
_CONST_48 = 48
_CONST_51 = 51
_CONST_56 = 56
_CONST_112 = 112
_CONST_192 = 192
_CONST_448 = 448
_CONST_3072 = 3072
_CONST_3584 = 3584
_CONST_8192 = 8192


def test_k3_phase1_containment_and_live_files() -> _Dynamic:
    """Verify clean import of megakernel, live files, and no legacy imports."""
    mk = importlib.import_module("vllm_torchtpu.models.kimi_k3.megakernel")
    base_dir = pathlib.Path(mk.__file__).resolve().parent

    for live_file in [
        "kernels/prefill_moe/indexed_ragged_gmm.py",
        "kernels/prefill_moe/moe_down_valu.py",
    ]:
        target = base_dir / live_file
        assert target.exists(), f"Required live file missing: {target}"

    for py_file in base_dir.rglob("*.py"):
        if py_file.name in ("spec.py", "adapter.py"):
            continue
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "vllm_torchtpu.megakernel", (
                        f"Legacy import {alias.name} found in {py_file}"
                    )
                    assert not alias.name.startswith("vllm_torchtpu.megakernel."), (
                        f"Legacy import {alias.name} found in {py_file}"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                # The shared Mosaic reshape hook comes from main's megakernel
                # library; nothing else may come from there.
                if node.module == "vllm_torchtpu.megakernel" and [
                    a.name for a in node.names
                ] == ["pool_alias"]:
                    continue
                assert node.module != "vllm_torchtpu.megakernel", (
                    f"Legacy import {node.module} found in {py_file}"
                )
                assert not node.module.startswith("vllm_torchtpu.megakernel."), (
                    f"Legacy import {node.module} found in {py_file}"
                )


def test_load_checkpoint_config_validation(tmp_path: _Dynamic) -> _Dynamic:
    """Verify config.from_checkpoint parses valid and rejects invalid."""
    load_mod = importlib.import_module(
        "vllm_torchtpu.models.kimi_k3.megakernel.kernels.load"
    )
    config = load_mod.Config

    base_config = {
        "hidden_size": 7168,
        "routed_expert_hidden_size": 3584,
        "moe_intermediate_size": 3072,
        "num_shared_experts": 2,
        "intermediate_size": 33792,
        "num_experts": 896,
        "num_experts_per_token": 16,
        "num_hidden_layers": 93,
        "num_attention_heads": 96,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "vocab_size": 163840,
        "attn_res_block_size": 12,
        "rms_norm_eps": 1e-5,
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
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
        "linear_attn_config": {
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
            "full_attn_layers": [*list(range(4, 93, 4)), 93],
            "use_full_rank_gate": True,
        },
    }

    valid_dir = tmp_path / "valid_k3"
    valid_dir.mkdir()
    (valid_dir / "config.json").write_text(json.dumps(base_config))
    cfg = config.from_checkpoint(valid_dir)
    assert cfg.dim == _CONST_7168
    assert cfg.layers == _CONST_93
    assert cfg.experts == _CONST_896

    invalid_dir = tmp_path / "invalid_k3"
    invalid_dir.mkdir()
    bad_config = dict(base_config)
    bad_config["mla_use_nope"] = False
    (invalid_dir / "config.json").write_text(json.dumps(bad_config))
    with pytest.raises(ValueError, match="Unsupported K3 setting"):
        config.from_checkpoint(invalid_dir)


def test_prefill_moe_live_helpers_and_default_reduce_dtype() -> _Dynamic:
    """Verify the default float32 MoE reduce dtype and live overflow helpers."""
    config_mod = importlib.import_module(
        "vllm_torchtpu.models.kimi_k3.megakernel.config"
    )
    cfg = config_mod.KernelConfig()
    assert cfg.moe_reduce_dtype == "float32"


def test_pytree_dataclass_registration_and_fields() -> _Dynamic:
    """Verify test pytree dataclass registration and fields logic."""
    dm = importlib.import_module(
        "vllm_torchtpu.models.kimi_k3.megakernel.kernels.decode_megakernel"
    )
    k = importlib.import_module(
        "vllm_torchtpu.models.kimi_k3.megakernel.kernels.prefill_megakernel"
    )
    names = [f.name for f in dataclasses.fields(dm.MultiHeadLatentAttentionWorkspace)]
    assert len(names) == _CONST_10
    assert "maximum_vmem" in names
    assert "denominator_vmem" in names
    assert "accumulator_vmem" in names

    ws = dm.MultiHeadLatentAttentionWorkspace(*[jnp.zeros(1)] * 10)
    assert len(jax.tree.leaves(ws)) == _CONST_10

    w_ref = k.WeightsRef(weight=jnp.zeros(1), scale=jnp.zeros(1))
    assert len(jax.tree.leaves(w_ref)) == _CONST_2

    m_ref = k.MetadataRef(
        gm_id_to_group_id=jnp.zeros(1), gm_id_to_m_offset=jnp.zeros(1)
    )
    assert len(jax.tree.leaves(m_ref)) == _CONST_2


def test_k3_phase1_provenance_headers() -> _Dynamic:
    """Verify source provenance headers in all 11 adapted modules."""
    mk = importlib.import_module("vllm_torchtpu.models.kimi_k3.megakernel")
    base_dir = pathlib.Path(mk.__file__).resolve().parent

    modules_11 = [
        "common/vmem_pool.py",
        "common/collectives32.py",
        "kernels/decode_megakernel.py",
        "kernels/dspark_mla.py",
        "kernels/dspark_mla_kernel.py",
        "kernels/mla_dcp.py",
        "kernels/prefill_megakernel.py",
        "kernels/prefill_moe/fused.py",
        "kernels/prefill_moe/indexed_ragged_gmm.py",
        "kernels/prefill_moe/moe_down_valu.py",
    ]
    gmm_provenance_modules = {
        "kernels/prefill_megakernel.py",
        "kernels/prefill_moe/indexed_ragged_gmm.py",
        "kernels/prefill_moe/moe_down_valu.py",
    }

    for rel_path in modules_11:
        target = base_dir / rel_path
        assert target.exists(), f"Module {rel_path} does not exist"
        content = target.read_text(encoding="utf-8")
        assert "Source provenance:" in content, (
            f"Missing 'Source provenance:' header in {rel_path}"
        )
        assert "inferact/k3-megakernel@30c9ba966b" in content, (
            f"Missing inferact provenance tag in {rel_path}"
        )
        if rel_path in gmm_provenance_modules:
            assert "vllm_torchtpu/kernels/megablox/gmm_v2.py@644426d851" in content, (
                f"Missing megablox gmm_v2 provenance tag in {rel_path}"
            )


def _lower_dspark_mla_on_tpu7x(devices: _Dynamic) -> None:
    """Verify deviceless TPU v7x AOT lowering for fused_draft_local."""
    mesh_tp = Mesh(devices, ("tp",))
    c_draft = dspark_mla.MlaDraftConfig()
    r = _CONST_2
    rows = _CONST_8
    pages = _CONST_64
    block = _CONST_7
    dp = 1
    t = r * rows

    fused_draft = dspark_mla_kernel.fused_draft_local(
        c_draft, requests=r, pages=pages, block=block, rows=rows, attention_dp=dp
    )
    weights = {
        "ln1": jax.ShapeDtypeStruct((c_draft.layers, 1, c_draft.hidden), jnp.bfloat16),
        "ln2": jax.ShapeDtypeStruct((c_draft.layers, 1, c_draft.hidden), jnp.bfloat16),
        "q_norm": jax.ShapeDtypeStruct(
            (c_draft.layers, 1, c_draft.q_lora), jnp.bfloat16
        ),
        "kv_norm": jax.ShapeDtypeStruct((c_draft.layers, 1, 640), jnp.bfloat16),
        "context_norm": jax.ShapeDtypeStruct((1, c_draft.hidden), jnp.bfloat16),
        "final_norm": jax.ShapeDtypeStruct((1, c_draft.hidden), jnp.bfloat16),
        "mask_embed": jax.ShapeDtypeStruct((1, c_draft.hidden), jnp.bfloat16),
        "rot": jax.ShapeDtypeStruct((c_draft.rope, c_draft.rope), jnp.bfloat16),
        "qkv_tile": jax.ShapeDtypeStruct(
            (c_draft.layers, c_draft.hidden, _CONST_128), jnp.bfloat16
        ),
        "context_tile": jax.ShapeDtypeStruct(
            (c_draft.hidden, _CONST_128), jnp.bfloat16
        ),
        "q_b": jax.ShapeDtypeStruct(
            (c_draft.layers, c_draft.q_lora, _CONST_512),
            jnp.bfloat16,
        ),
        "w_uk": jax.ShapeDtypeStruct(
            (c_draft.layers, _CONST_2, c_draft.nope, c_draft.kv_lora), jnp.bfloat16
        ),
        "w_uv": jax.ShapeDtypeStruct(
            (c_draft.layers, _CONST_2, c_draft.kv_lora, c_draft.v_head),
            jnp.bfloat16,
        ),
        "o": jax.ShapeDtypeStruct(
            (c_draft.layers, _CONST_2 * c_draft.v_head, c_draft.hidden), jnp.bfloat16
        ),
        "gate": jax.ShapeDtypeStruct(
            (c_draft.layers, c_draft.hidden, _CONST_512), jnp.bfloat16
        ),
        "up": jax.ShapeDtypeStruct(
            (c_draft.layers, c_draft.hidden, _CONST_512), jnp.bfloat16
        ),
        "down": jax.ShapeDtypeStruct(
            (c_draft.layers, _CONST_512, c_draft.hidden), jnp.bfloat16
        ),
        "fc": jax.ShapeDtypeStruct((8960, _CONST_896), jnp.bfloat16),
        "w2t": jax.ShapeDtypeStruct(
            (_CONST_256, c_draft.vocab // _CONST_32), jnp.bfloat16
        ),
        "w1": jax.ShapeDtypeStruct((c_draft.vocab, _CONST_256), jnp.bfloat16),
        "rope": jax.ShapeDtypeStruct(
            (_CONST_512, _CONST_2 * c_draft.rope), jnp.float32
        ),
        "lm_head": jax.ShapeDtypeStruct(
            (c_draft.hidden, c_draft.vocab // _CONST_32), jnp.bfloat16
        ),
        "embed": jax.ShapeDtypeStruct(
            (c_draft.vocab // _CONST_32, c_draft.hidden), jnp.bfloat16
        ),
    }
    pools = tuple(
        jax.ShapeDtypeStruct((_CONST_16, _CONST_128, 4, 640), jnp.float8_e4m3fn)
        for _ in range(c_draft.layers)
    )
    aux = jax.ShapeDtypeStruct(
        (len(c_draft.aux_layers), t, c_draft.hidden), jnp.bfloat16
    )
    logits = jax.ShapeDtypeStruct((r, rows, c_draft.vocab // _CONST_32), jnp.float32)
    previous = jax.ShapeDtypeStruct((r, _CONST_128), jnp.int32)
    position = jax.ShapeDtypeStruct((r,), jnp.int32)
    forced_count = jax.ShapeDtypeStruct((r,), jnp.int32)
    forced_anchor = jax.ShapeDtypeStruct((r,), jnp.int32)
    tables = jax.ShapeDtypeStruct((r, pages), jnp.int32)

    part_a_sharded = jax.shard_map(
        fused_draft,
        mesh=mesh_tp,
        in_specs=P(),
        out_specs=P(),
        check_vma=False,
    )
    lowered_draft = jax.jit(part_a_sharded).lower(
        weights,
        pools,
        aux,
        logits,
        previous,
        position,
        forced_count,
        forced_anchor,
        tables,
    )
    assert "tpu_custom_call" in lowered_draft.as_text()


def _lower_moe_down_valu_on_tpu7x(devices: _Dynamic) -> None:
    """Verify deviceless TPU v7x AOT lowering for moe_down_valu.down_valu."""
    mesh_sp = Mesh(devices, ("k3_sp",))
    tokens_val = _CONST_4096
    columns_val = _CONST_1792
    tile_val = _CONST_128
    unroll_val = _CONST_8
    capacity_val = _CONST_512
    size_k = 1536
    size_group = 56
    num_quant_blocks = size_k // _CONST_32

    x_val = jax.ShapeDtypeStruct((capacity_val, size_k), jnp.bfloat16)
    rhs_val = jax.ShapeDtypeStruct((size_group, size_k // _CONST_2, 7168), jnp.uint8)
    scales_val = jax.ShapeDtypeStruct((size_group, num_quant_blocks, 7168), jnp.uint8)
    counts_val = jax.ShapeDtypeStruct((size_group,), jnp.int32)
    destinations_val = jax.ShapeDtypeStruct((capacity_val,), jnp.int32)
    probabilities_val = jax.ShapeDtypeStruct((capacity_val,), jnp.float32)
    tables_val = jax.ShapeDtypeStruct((3, 3, _CONST_32), jnp.int32)

    def _part_b_wrapper(operands: tuple[_Dynamic, ...]) -> _Dynamic:
        """Wrap down valu call for shard map."""
        return moe_down_valu.down_valu(
            *operands,
            tokens=tokens_val,
            columns=columns_val,
            tile=tile_val,
            unroll=unroll_val,
            layer=None,
            stream_metadata=False,
            metadata_cache_rows=0,
            metadata_row_capacity=capacity_val,
            ring_dtype=jnp.float32,
        )

    part_b_sharded = jax.shard_map(
        _part_b_wrapper,
        mesh=mesh_sp,
        in_specs=P(),
        out_specs=P(),
        check_vma=False,
    )
    lowered_valu = jax.jit(part_b_sharded).lower(
        (
            x_val,
            rhs_val,
            scales_val,
            counts_val,
            destinations_val,
            probabilities_val,
            tables_val,
        )
    )
    lowered_text = lowered_valu.as_text()
    assert "tpu_custom_call" in lowered_text


def test_k3_deviceless_tpu7x_aot_lowering() -> _Dynamic:
    """Verify deviceless TPU v7x AOT lowering for fused draft and down VALU."""
    assert _pmk_init is not None
    devices = topologies.get_topology_desc(
        topology_name="tpu7x_2x2x4", platform="tpu"
    ).devices

    _lower_dspark_mla_on_tpu7x(devices)
    _lower_moe_down_valu_on_tpu7x(devices)

    mk = importlib.import_module("vllm_torchtpu.models.kimi_k3.megakernel")
    base_dir = pathlib.Path(mk.__file__).resolve().parent
    valu_source = (base_dir / "kernels/prefill_moe/moe_down_valu.py").read_text(
        encoding="utf-8"
    )
    assert (
        "lambda: pltpu.make_async_copy(payload, o_ref, store_sem)" not in valu_source
    ), "Final store DMA in moe_down_valu.py must call .start() and .wait()"
    assert "for step_idx in range(state[0], stop):" not in valu_source, (
        "Traced state[0] must not be used in Python range() in moe_down_valu.py"
    )


def test_k3_bridge_compat_with_main_sharded_jax_op(
    monkeypatch: _Dynamic, tmp_path: _Dynamic
) -> None:
    """Verify bridge.register_op and export_cache against stock sharded_jax_op."""
    bridge_mod = importlib.import_module(
        "vllm_torchtpu.models.kimi_k3.megakernel.runtime_shim.bridge"
    )
    export_cache_mod = importlib.import_module(
        "vllm_torchtpu.models.kimi_k3.megakernel.runtime_shim.export_cache"
    )
    vllm_envs = importlib.import_module("vllm.envs")
    monkeypatch.setenv("VLLM_XLA_CACHE_PATH", str(tmp_path))
    monkeypatch.setattr(vllm_envs, "VLLM_XLA_CACHE_PATH", str(tmp_path), raising=False)

    class _FakeCallable:
        """Represent stock JaxCallable without output_partition_specs."""

        def __init__(self, name: str) -> None:
            """Initialize fake callable."""
            self.name = name
            self.output_partition_specs: _Dynamic = None
            self.exported = lambda *_a, **_kw: None

    class _FakeCustomOp:
        """Represent torch custom_op wrapper holding _init_fn."""

        def __init__(self, init_fn: _Dynamic) -> None:
            """Initialize fake custom op."""
            self._init_fn = init_fn

    def _stock_sharded_jax_op(
        name: str,
        fn: _Dynamic,
        *args: _Dynamic,
        **kwargs: _Dynamic,
    ) -> _Dynamic:
        """Simulate origin/main sharded_jax_op without configure_callable."""
        _ = (fn, args, kwargs)
        return _FakeCustomOp(_FakeCallable(name))

    monkeypatch.setattr(bridge_mod, "sharded_jax_op", _stock_sharded_jax_op)
    if hasattr(bridge_mod, "_SHARDED_JAX_OP_ACCEPTS_CONFIGURE"):
        monkeypatch.setattr(bridge_mod, "_SHARDED_JAX_OP_ACCEPTS_CONFIGURE", False)

    devices = jax.devices("cpu")
    mesh = Mesh(devices[:1], ("tp",))
    op = bridge_mod.register_op(
        "k3_test_stock_op",
        mesh,
        lambda x: (x,),
        (P("tp"),),
        (P("tp"),),
        (),
        "stock_export_key",
    )
    init_fn = operator.attrgetter("_init_fn")(op)
    assert init_fn.output_partition_specs == [P("tp")]
    assert getattr(init_fn, "_export_cache", None) is not None

    global_op = bridge_mod.register_global_op(
        "k3_test_stock_global_op",
        mesh,
        lambda x: (x,),
        (P("tp"),),
        (P("tp"),),
        (),
        "stock_global_export_key",
    )
    global_init_fn = operator.attrgetter("_init_fn")(global_op)
    assert global_init_fn.output_partition_specs == [P("tp")]
    assert getattr(global_init_fn, "_export_cache", None) is not None

    class _FakeLowered:
        """Represent lowered export without _out_named_shardings."""

        def __init__(self) -> None:
            """Initialize fake lowered object."""
            self.out_avals = [
                jax.core.ShapedArray((_CONST_32, _CONST_16), jnp.bfloat16)
            ]
            self._out_named_shardings = None
            self.mlir_module_serialized = b"fake_mlir"
            self.out_tree = jax.tree.structure((0,))

    cache = operator.attrgetter("_ExportCache")(export_cache_mod)(
        tmp_path / "k3_export", "fallback_key"
    )
    pkl_path = tmp_path / "k3_export" / "entry.pkl"
    operator.attrgetter("_store")(cache)(pkl_path, "fp123", _FakeLowered(), op=init_fn)
    pickle_mod = importlib.import_module("pickle")
    saved = vars(pickle_mod)["loads"](pkl_path.read_bytes())
    assert saved["outputs"][0][0] is True
    assert saved["outputs"][0][3] == P("tp")


def test_k3_harness_init_exports_master_addr(monkeypatch: _Dynamic) -> None:
    """Verify harness.init derives MASTER_ADDR and the 16 slice endpoints."""
    harness_mod = importlib.import_module(
        "vllm_torchtpu.models.kimi_k3.megakernel.tools.harness"
    )
    torch_mod = importlib.import_module("torch")
    os_mod = importlib.import_module("os")

    monkeypatch.delenv("MASTER_ADDR", raising=False)
    monkeypatch.setenv("TP32_HOST_IPS", "127.0.0.1,127.0.0.2")
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setenv("SLURM_LOCALID", "0")
    monkeypatch.setenv("SLURM_NODEID", "0")
    monkeypatch.setenv("TP32_SESSION", "k3-harness-test")
    monkeypatch.setenv("MASTER_PORT", "23457")

    captured: dict[str, str | None] = {}

    def _fake_init_pg(**kwargs: _Dynamic) -> None:
        """Record environment variables during init_process_group."""
        _ = kwargs
        captured["MASTER_ADDR"] = os_mod.environ.get("MASTER_ADDR")
        captured["MASTER_PORT"] = os_mod.environ.get("MASTER_PORT")

    monkeypatch.setattr(torch_mod.distributed, "init_process_group", _fake_init_pg)
    harness_mod.init()
    assert captured.get("MASTER_ADDR") == "127.0.0.1"
    assert captured.get("MASTER_PORT") == "23457"
    addresses = os_mod.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"].split(",")
    assert len(addresses) == _CONST_16
    assert addresses[0] == "127.0.0.1:8070"
    assert addresses[-1] == "127.0.0.2:8077"


def test_k3_settings_reads_lazily_and_requires_launcher_vars(
    monkeypatch: _Dynamic,
) -> None:
    """Verify settings reads at access time, keeps defaults, and raises when unset."""
    settings = importlib.import_module(
        "vllm_torchtpu.models.kimi_k3.megakernel.settings"
    )
    monkeypatch.delenv("TRACE_RANKS", raising=False)
    monkeypatch.delenv("TP32_HOST_IPS", raising=False)
    monkeypatch.setenv("SLURM_PROCID", "7")
    assert settings.TRACE_RANKS == "0"
    assert settings.SLURM_PROCID == _CONST_7
    monkeypatch.setenv("TRACE_RANKS", "0,8")
    assert settings.TRACE_RANKS == "0,8"
    with pytest.raises(KeyError, match="TP32_HOST_IPS"):
        _ = settings.TP32_HOST_IPS
    assert "TP32_HOST_IPS" in dir(settings)


def test_k3_phase2_zero_deleted_modules_and_kept_count() -> None:
    """Verify dead prefill_moe and dspark_common modules are pruned and unreferenced."""
    mk = importlib.import_module("vllm_torchtpu.models.kimi_k3.megakernel")
    base_dir = pathlib.Path(mk.__file__).resolve().parent

    pruned_files = (
        "kernels/dspark_common.py",
        "kernels/prefill_moe/layer.py",
        "kernels/prefill_moe/packed_layout.py",
        "kernels/prefill_moe/routing_count_kernel.py",
        "kernels/prefill_moe/routing_topk_kernel.py",
        "kernels/prefill_moe/shared_ring_kernel.py",
        "kernels/prefill_moe/streaming_ring.py",
    )
    for rel_path in pruned_files:
        assert not (base_dir / rel_path).exists(), (
            f"Dead module still present: {rel_path}"
        )

    for kept_path in (
        "draft_layers.py",
        "kernels/dspark.py",
        "kernels/prefill_moe/fused.py",
        "kernels/prefill_moe/indexed_ragged_gmm.py",
        "kernels/prefill_moe/moe_down_valu.py",
    ):
        assert (base_dir / kept_path).exists(), f"Required module missing: {kept_path}"

    py_files = sorted(base_dir.rglob("*.py"))
    assert len(py_files) in (51, 52)
    assert hasattr(indexed_ragged_gmm, "PackedRows")

    deleted_stems = {
        "dspark_common",
        "packed_layout",
        "routing_count_kernel",
        "routing_topk_kernel",
        "shared_ring_kernel",
        "streaming_ring",
    }
    for py_file in py_files:
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                mod_parts = set(mod.split(".")) if mod else set()
                imported_names = {alias.name for alias in node.names}
                assert not (mod_parts & deleted_stems), f"{py_file}: imports {mod}"
                assert not (imported_names & deleted_stems), (
                    f"{py_file}: imports {imported_names & deleted_stems}"
                )
                assert not mod.endswith("prefill_moe.layer"), (
                    f"{py_file}: imports {mod}"
                )
                if py_file.parent.name == "prefill_moe":
                    assert mod != "layer", (
                        f"{py_file}: imports deleted prefill_moe.layer"
                    )
                    assert "layer" not in imported_names or mod not in ("", "."), (
                        f"{py_file}: imports deleted prefill_moe.layer"
                    )


def test_k3_prefill_moe_8192_routed_experts_prepared_lowering() -> None:
    """Verify 8192-token routed_experts_prepared and overflow path lower on TPU v7x."""
    assert hasattr(fused, "routed_experts_prepared")
    devices = topologies.get_topology_desc(
        topology_name="tpu7x_2x2x4", platform="tpu"
    ).devices
    mesh_sp = Mesh(devices, ("k3_sp",))
    tables = jnp.asarray(
        _pmk_init.ring_tables(list(mesh_sp.devices.flat), tuple(range(_CONST_32)))
    )

    latent = jax.ShapeDtypeStruct((_CONST_8192, _CONST_2, _CONST_1792), jnp.bfloat16)
    route_cap = _CONST_8192 * _CONST_16
    sorted_tokens = jax.ShapeDtypeStruct((route_cap,), jnp.int32)
    sorted_probs = jax.ShapeDtypeStruct((route_cap,), jnp.float32)
    counts = jax.ShapeDtypeStruct((_CONST_56,), jnp.int32)
    expert_gu = jax.ShapeDtypeStruct(
        (1, _CONST_56, _CONST_448, _CONST_3072), jnp.uint32
    )
    expert_gus = jax.ShapeDtypeStruct(
        (1, _CONST_56, _CONST_112, _CONST_3072), jnp.uint8
    )
    expert_down = jax.ShapeDtypeStruct(
        (1, _CONST_56, _CONST_192, _CONST_3584), jnp.uint32
    )
    expert_ds = jax.ShapeDtypeStruct((1, _CONST_56, _CONST_48, _CONST_3584), jnp.uint8)
    slot = jax.ShapeDtypeStruct((1,), jnp.int32)
    peak = jax.ShapeDtypeStruct((), jnp.int32)

    def _prepared_wrapper(operands: tuple[_Dynamic, ...]) -> jax.Array:
        """Invoke routed_experts_prepared inside a 32-rank k3_sp shard_map."""
        (
            lat,
            tok_ids,
            probs,
            cnts,
            gu_w,
            gu_s,
            dn_w,
            dn_s,
            layer_slot,
            route_max,
        ) = operands
        return fused.routed_experts_prepared(
            lat,
            tok_ids,
            probs,
            cnts,
            gu_w,
            gu_s,
            dn_w,
            dn_s,
            reduce_tables=tables,
            expert_layer=layer_slot,
            global_route_max=route_max,
        )

    sharded_prepared = jax.shard_map(
        _prepared_wrapper,
        mesh=mesh_sp,
        in_specs=P(),
        out_specs=P(),
        check_vma=False,
    )
    lowered = jax.jit(sharded_prepared).lower(
        (
            latent,
            sorted_tokens,
            sorted_probs,
            counts,
            expert_gu,
            expert_gus,
            expert_down,
            expert_ds,
            slot,
            peak,
        )
    )
    assert "tpu_custom_call" in lowered.as_text()
    assert lowered.out_info.shape == (_CONST_256, _CONST_3584)
    assert lowered.out_info.dtype == jnp.float32


def test_pytrees_register_lazily_on_first_use(monkeypatch: _Dynamic) -> None:
    """Verify decode and prefill dataclasses avoid import-time registration."""
    dm = importlib.import_module(
        "vllm_torchtpu.models.kimi_k3.megakernel.kernels.decode_megakernel"
    )
    k = importlib.import_module(
        "vllm_torchtpu.models.kimi_k3.megakernel.kernels.prefill_megakernel"
    )
    for mod in (dm, k):
        tree = ast.parse(
            pathlib.Path(mod.__file__).read_text(encoding="utf-8"),
            filename=mod.__file__,
        )
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                decos = [ast.unparse(d) for d in node.decorator_list]
                assert all("register" not in d and "pytree" not in d for d in decos)

    recorded_decode: list[type[object]] = []
    monkeypatch.setattr(dm, "_REGISTERED_DECODE_PYTREES", set())
    monkeypatch.setattr(
        jax.tree_util,
        "register_dataclass",
        recorded_decode.append,
    )
    _ = dm.MultiHeadLatentAttentionWorkspace(*[jnp.zeros(1)] * _CONST_10)
    assert len(recorded_decode) == _CONST_7
    _ = dm.MultiHeadLatentAttentionWorkspace(*[jnp.zeros(1)] * _CONST_10)
    assert len(recorded_decode) == _CONST_7

    recorded_prefill: list[type[object]] = []
    monkeypatch.setattr(k, "_REGISTERED_PREFILL_PYTREES", set())
    monkeypatch.setattr(
        jax.tree_util,
        "register_dataclass",
        recorded_prefill.append,
    )
    _ = k.MetadataRef(gm_id_to_group_id=jnp.zeros(1), gm_id_to_m_offset=jnp.zeros(1))
    assert len(recorded_prefill) == _CONST_3
    _ = k.WeightsRef(weight=jnp.zeros(1), scale=jnp.zeros(1))
    assert len(recorded_prefill) == _CONST_3
