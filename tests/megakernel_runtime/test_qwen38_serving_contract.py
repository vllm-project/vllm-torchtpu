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
"""CPU unit tests for Qwen3.8 serving contract with stock vLLM TorchTPU."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.experimental.pallas.tpu import ChipVersion
from jax.sharding import Mesh

import vllm_torchtpu.megakernel_runtime.driver as mk_driver
from tests.megakernel_runtime.test_weight_collector import (
    make_synthetic_qwen35_backbone,
)
from vllm_torchtpu.kernels.experimental.batched_rpa.configs import KVLayout
from vllm_torchtpu.kernels.experimental.batched_rpa.wrapper import get_kv_cache_shape
from vllm_torchtpu.megakernel.errors import MegakernelConfigError, MegakernelLayoutError
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.models.qwen38_moe import (
    TINY_EP,
    Qwen38Moe,
    Qwen38MoeGeometry,
)
from vllm_torchtpu.megakernel.models.qwen38_moe_profile import (
    DEFAULT_HF_CONFIG,
)
from vllm_torchtpu.megakernel.pool.config import CacheExec
from vllm_torchtpu.megakernel.pool.layout import PoolLayout
from vllm_torchtpu.megakernel.profile import get_profile
from vllm_torchtpu.megakernel_runtime.driver import (
    Qwen38Megakernel,
    _check_host_major_mesh,
    expected_pool_shape,
    validate_pool_shape,
)
from vllm_torchtpu.megakernel_runtime.driver import maybe_create as driver_maybe_create
from vllm_torchtpu.megakernel_runtime.exported_bridge import (
    _torch_dtype_to_jax,
    export_pooled_step,
)
from vllm_torchtpu.megakernel_runtime.weight_collector import (
    pack_expert_views_layer,
    pack_router_and_shared_layer,
    pack_weights,
)
from vllm_torchtpu.platforms.tpu_block_size_utils import (
    _align_block_to_backend,
    derive_pooled_gdn_state_layout,
)

Qwen38 = Qwen38Moe
Qwen38Geometry = Qwen38MoeGeometry


def test_stock_qwen38_tp32_block_and_gdn_slot():
    """Verify stock pooled GDN layout and block alignment for Qwen3.8 TP32."""
    layout = derive_pooled_gdn_state_layout(
        ssm_bytes=4 * 128 * 128 * 2,
        conv_bytes=3 * 768 * 2,
        token_bytes=512,
    )
    assert layout.ssm_tokens == 256
    assert layout.conv_tokens == 16
    assert layout.required_tokens == 272

    aligned = _align_block_to_backend(layout.required_tokens, [128])
    assert aligned == 384


def test_stock_seq_along_lane_pool_shape_qwen38():
    """Verify stock SEQ_ALONG_LANE pool shape with page_size=384."""
    shape = get_kv_cache_shape(
        10,
        384,
        1,
        256,
        jnp.float8_e4m3fn,
        KVLayout.SEQ_ALONG_LANE,
        ChipVersion.TPU_7X,
    )
    assert shape == (10, 2, 64, 4, 384)


def test_megakernel_rejects_stock_qwen38_pool_page():
    """Verify megakernel PoolLayout and driver reject page_size=384 naming 384."""
    layout = PoolLayout(page_size=384)
    with pytest.raises((MegakernelConfigError, MegakernelLayoutError), match="384"):
        layout.validate()

    geom = Qwen38MoeGeometry()
    shape_384 = (10, 2, 64, 4, 384)
    with pytest.raises(MegakernelLayoutError, match="384"):
        validate_pool_shape(shape_384, geom, r_cnt=32, page_size=384)

    # Sanity check: expected_pool_shape produces expected tuple
    expected = expected_pool_shape(geom, r_cnt=32, num_pages=10, page_size=128)
    assert expected == (10, 2, 64, 4, 128)


# ModelOpt NVFP4 checkpoints nest `quant_algo` under `quantization`.
NVFP4_HF_CONFIG = dict(
    DEFAULT_HF_CONFIG,
    quantization_config={
        "quant_method": "modelopt",
        "quantization": {"quant_algo": "NVFP4", "group_size": 16},
    },
)


def test_qwen38_nvfp4_two_column_gate_up_scale_packing():
    """pack_expert_views_layer accepts the ModelOpt [EW, 2] w1/w3 global scale."""
    g = dataclasses.replace(TINY_EP, expert_quant="nvfp4")
    ex = ExecSpec(R=8, B=4, T=1, stack_e=False)
    ew, h, ir = g.EW(ex.R), g.H, g.IR(ex.R)
    experts = SimpleNamespace(
        w13_weight=torch.zeros((ew, 2 * ir, h // 2), dtype=torch.uint8),
        w13_weight_scale=torch.ones((ew, 2 * ir, h // 16), dtype=torch.float8_e4m3fn),
        w13_weight_scale_2=torch.tensor([[0.25, 0.5]] * ew, dtype=torch.float32),
        w2_weight=torch.zeros((ew, h, ir // 2), dtype=torch.uint8),
        w2_weight_scale=torch.ones((ew, h, ir // 16), dtype=torch.float8_e4m3fn),
        w2_weight_scale_2=torch.full((ew,), 0.125, dtype=torch.float32),
    )
    _, _, es = pack_expert_views_layer(
        SimpleNamespace(experts=experts), g, ex, 128, 128
    )
    assert float(es[0, 0, 0]) == pytest.approx(0.25)
    assert float(es[0, 0, ir]) == pytest.approx(0.5)
    assert float(es[0, 0, 2 * ir]) == pytest.approx(0.125)


def test_qwen38_derive_geometry_reads_nested_modelopt_quant_algo():
    """The nested ModelOpt quant_algo selects NVFP4 experts; no config stays FP8."""
    assert Qwen38.derive_geometry(NVFP4_HF_CONFIG).expert_quant == "nvfp4"
    assert Qwen38.derive_geometry(DEFAULT_HF_CONFIG).expert_quant == "fp8"


def test_qwen38_nvfp4_checkpoint_keeps_megakernel_enabled():
    """NVFP4 checkpoints pass HF validation and build an NVFP4 MoE spec."""
    runner = SimpleNamespace(model_config=SimpleNamespace(hf_config=NVFP4_HF_CONFIG))
    assert Qwen38Megakernel._validate_runner_hf_config(runner)
    drv = Qwen38Megakernel.__new__(Qwen38Megakernel)
    experts = SimpleNamespace(w13_weight=torch.empty((16, 0), dtype=torch.uint8))
    drv.backbone = SimpleNamespace(
        layers=[SimpleNamespace(mlp=SimpleNamespace(experts=experts))]
    )
    assert drv._build_spec(mode="ep", r_cnt=32).config.expert_quant == "nvfp4"


def _fake_mesh(pids):
    return SimpleNamespace(
        devices=np.array([SimpleNamespace(process_index=p) for p in pids], dtype=object)
    )


def test_host_major_mesh_accepts_one_process_per_device():
    """32 ranks: one process per host or one per device pass; interleaved fails."""
    assert _check_host_major_mesh(_fake_mesh([r // 8 for r in range(32)]))
    assert _check_host_major_mesh(_fake_mesh(list(range(32))))
    assert not _check_host_major_mesh(_fake_mesh([r % 4 for r in range(32)]))


def test_skipped_megakernel_releases_nvfp4_stash():
    """When no driver is created, the CPU NVFP4 stash is dropped from the model."""
    experts = torch.nn.Module()
    experts._mk_nvfp4_raw = (torch.zeros(1),)
    model = torch.nn.Sequential(experts)
    runner = SimpleNamespace(
        model=model,
        model_config=SimpleNamespace(hf_config={"architectures": ["NotAModel"]}),
    )
    assert driver_maybe_create(runner) is None
    assert not hasattr(experts, "_mk_nvfp4_raw")


def test_bridge_maps_uint8_nvfp4_weights():
    """Packed NVFP4 expert weights are uint8 and must cross the JAX bridge."""
    assert _torch_dtype_to_jax(torch.uint8) == jnp.uint8


def test_qwen38_nvfp4_driver_packs_weights_with_profile_ebank():
    """Profile-specialized exec_for configures ebank so NVFP4 weights pack."""
    g = dataclasses.replace(
        Qwen38.derive_geometry(NVFP4_HF_CONFIG, moe_tp=False),
        L=4,
        e_slots=16,
    )
    ex_dummy = ExecSpec(R=32, B=4, T=1, stack_e=False)
    backbone = make_synthetic_qwen35_backbone(g, ex_dummy)
    ew, ir, h = g.EW(32), g.IR(32), g.H
    for layer in backbone.layers:
        experts = SimpleNamespace(
            w13_weight=torch.zeros((ew, 2 * ir, h // 2), dtype=torch.uint8),
            w13_weight_scale=torch.ones(
                (ew, 2 * ir, h // 16), dtype=torch.float8_e4m3fn
            ),
            w13_weight_scale_2=torch.tensor([[0.25, 0.5]] * ew, dtype=torch.float32),
            w2_weight=torch.zeros((ew, h, ir // 2), dtype=torch.uint8),
            w2_weight_scale=torch.ones((ew, h, ir // 16), dtype=torch.float8_e4m3fn),
            w2_weight_scale_2=torch.full((ew,), 0.125, dtype=torch.float32),
        )
        layer.mlp.experts = experts

    model = SimpleNamespace(model=backbone)
    mesh = _fake_mesh(list(range(32)))
    drv = Qwen38Megakernel(
        model,
        mesh,
        tp_rank=0,
        mode="ep",
        geom=g,
        profile="qwen38-tp32-dcp8",
        buckets=(4, 8),
    )
    assert len(drv.w_flat) > 0


def test_qwen38_dcp8_warmup_sentinel_skips_kv_on_non_owner_rank():
    """DCP8 non-owner ranks skip the KV sentinel page equality check."""
    drv = Qwen38Megakernel.__new__(Qwen38Megakernel)
    drv._bound = True
    pool = torch.zeros((32, 2, 64, 4, 128), dtype=torch.float8_e4m3fn)
    drv.raw_pools = [pool]
    drv.num_pages = 32
    drv._ckpt_window = 1
    drv._zero_roff_1d = torch.zeros(1, dtype=torch.int32)
    g = Qwen38.derive_geometry(DEFAULT_HF_CONFIG, moe_tp=False)
    drv.spec = Qwen38.to_model_spec(g, R=32)
    drv.base_exec = ExecSpec(
        R=32,
        B=4,
        T=1,
        cache=CacheExec(paged_kv=True, dcp_kv_sharded=True, dcp_size=8),
    )
    drv.buckets = (4,)
    drv.verify_buckets = ()
    drv.max_pages_per_seq = 2
    drv._bucket_fused_io = {}
    drv._bucket_gpr = {}
    drv.w_flat = []
    drv._ir_required_buckets = lambda *args, **kwargs: None
    drv._step_mix = {}
    drv.ab_calibrator = SimpleNamespace(reset=lambda: None)

    def stub(*args, **kwargs):
        drv.raw_pools[0][5:20].view(torch.uint8).fill_(0xAA)
        return torch.ones((4, 8192), dtype=torch.bfloat16)

    drv._bucket_steps = {(1, 4): stub}

    drv.tp_rank = 1
    drv.warmup(eager_only=True)

    drv.tp_rank = 0
    with pytest.raises(RuntimeError, match="warmup sentinel check failed"):
        drv.warmup(eager_only=True)


def test_qwen38_tp32_reshards_shared_expert_from_64_to_128_isr():
    """TP32 shared_expert 64-wide shards gather into partner-rank 128-ISR tiles."""
    g = Qwen38.derive_geometry(NVFP4_HF_CONFIG, moe_tp=False)
    ex = ExecSpec(R=32, B=4, T=1, stack_e=False)
    local_isr = g.IS // ex.R
    assert local_isr == 64
    assert g.ISR(ex.R) == 128
    assert g.sh_rep(ex.R) == 2

    gen = torch.Generator().manual_seed(0)
    w_gu_ranks = torch.randint(
        -8, 9, (ex.R, g.H, 2 * local_isr), generator=gen, dtype=torch.int8
    ).to(torch.float8_e4m3fn)
    s_gu_ranks = (
        (torch.arange(1, ex.R + 1, dtype=torch.float32).unsqueeze(1) * 0.01)
        .expand(ex.R, 2 * local_isr)
        .contiguous()
    )
    w_down_ranks = torch.randint(
        -8, 9, (ex.R, local_isr, g.H), generator=gen, dtype=torch.int8
    ).to(torch.float8_e4m3fn)
    s_down_ranks = (
        (torch.arange(1, ex.R + 1, dtype=torch.float32).unsqueeze(1) * 0.01)
        .expand(ex.R, g.H)
        .contiguous()
    )

    gate_w = torch.zeros((g.E, g.H), dtype=torch.bfloat16)
    sh_gate_w = torch.ones((1, g.H), dtype=torch.bfloat16)

    gathered_table = {
        ((g.H, 2 * local_isr), torch.float8_e4m3fn): w_gu_ranks,
        ((2 * local_isr,), torch.float32): s_gu_ranks,
        ((local_isr, g.H), torch.float8_e4m3fn): w_down_ranks,
        ((g.H,), torch.float32): s_down_ranks,
    }

    def fake_all_gather(t: torch.Tensor) -> torch.Tensor:
        return gathered_table[(tuple(t.shape), t.dtype)]

    def make_rank_mlp(r: int) -> SimpleNamespace:
        gu_proj = SimpleNamespace(
            weight=w_gu_ranks[r],
            weight_scale=s_gu_ranks[r],
            _tpu_weight_flipped=True,
        )
        down_proj = SimpleNamespace(
            weight=w_down_ranks[r],
            weight_scale=s_down_ranks[r],
            _tpu_weight_flipped=True,
        )
        return SimpleNamespace(
            gate=SimpleNamespace(weight=gate_w),
            shared_expert_gate=SimpleNamespace(weight=sh_gate_w),
            shared_expert=SimpleNamespace(gate_up_proj=gu_proj, down_proj=down_proj),
        )

    mlp2 = make_rank_mlp(2)
    mlp3 = make_rank_mlp(3)
    rs2 = pack_router_and_shared_layer(
        mlp2,
        g,
        ex,
        tp_rank=2,
        tk_router=128,
        tk_sh13=128,
        tk_sh2=128,
        tp_all_gather_fn=fake_all_gather,
    )
    rs3 = pack_router_and_shared_layer(
        mlp3,
        g,
        ex,
        tp_rank=3,
        tk_router=128,
        tk_sh13=128,
        tk_sh2=128,
        tp_all_gather_fn=fake_all_gather,
    )

    assert rs2["sh13"].shape == (g.H // 128, 128, 256)
    assert rs2["sh13_s"].shape == (256,)
    assert rs2["sh2"].shape == (1, 128, g.H)
    assert rs2["sh2_s"].shape == (g.H,)

    assert torch.equal(rs2["sh13"].view(torch.uint8), rs3["sh13"].view(torch.uint8))
    assert torch.equal(rs2["sh13_s"], rs3["sh13_s"])
    assert torch.equal(rs2["sh2"].view(torch.uint8), rs3["sh2"].view(torch.uint8))
    assert torch.equal(rs2["sh2_s"], rs3["sh2_s"])

    ref_gate = torch.cat(
        [
            w_gu_ranks[2, :, :64].float() * s_gu_ranks[2, :64],
            w_gu_ranks[3, :, :64].float() * s_gu_ranks[3, :64],
        ],
        dim=1,
    )
    ref_up = torch.cat(
        [
            w_gu_ranks[2, :, 64:].float() * s_gu_ranks[2, 64:],
            w_gu_ranks[3, :, 64:].float() * s_gu_ranks[3, 64:],
        ],
        dim=1,
    )
    ref_sh13 = torch.cat([ref_gate, ref_up], dim=1)
    deq_sh13 = rs2["sh13"].reshape(g.H, 256).float() * rs2["sh13_s"]
    assert torch.equal(deq_sh13, ref_sh13)

    ref_sh2 = torch.cat(
        [
            w_down_ranks[2].float() * s_down_ranks[2].unsqueeze(0),
            w_down_ranks[3].float() * s_down_ranks[3].unsqueeze(0),
        ],
        dim=0,
    )
    deq_sh2 = rs2["sh2"].reshape(128, g.H).float() * rs2["sh2_s"].unsqueeze(0)
    # FP8 E4M3 has 3 mantissa bits -> normal round-to-nearest half-ULP <= |x| * 2^-4.
    e4m3_half_ulp_bound = float(ref_sh2.abs().max().item()) / 16.0
    assert float((deq_sh2 - ref_sh2).abs().max().item()) <= e4m3_half_ulp_bound

    with pytest.raises(ValueError, match="tp_all_gather_fn"):
        pack_router_and_shared_layer(
            mlp2,
            g,
            ex,
            tp_rank=2,
            tk_router=128,
            tk_sh13=128,
            tk_sh2=128,
        )


def test_qwen38_dcp8_exported_step_eliminates_spmd_partition_id():
    """DCP8 exported StableHLO threads rank as a tensor arg with zero partition_id."""
    cpu_client = jax.devices("cpu")[0].client

    class _MockTpuDevice:
        platform = "tpu"
        device_kind = "TPU7x"
        num_cores = 1
        client = cpu_client

        def __init__(self, idx: int) -> None:
            self.id = idx
            self.process_index = idx
            self.coords = (idx % 2, (idx // 2) % 2, idx // 4)
            self.core_on_chip = 0

        def __repr__(self) -> str:
            return f"TpuDevice(id={self.id})"

        def __hash__(self) -> int:
            return hash(self.id)

        def __eq__(self, other: object) -> bool:
            return isinstance(other, _MockTpuDevice) and self.id == other.id

    r_cnt = 8
    g_dcp = dataclasses.replace(TINY_EP, NKV=1)
    mesh = Mesh(
        np.array([_MockTpuDevice(i) for i in range(r_cnt)], dtype=object),
        ("tp",),
    )
    spec = Qwen38.to_model_spec(g_dcp, R=r_cnt)
    ex = ExecSpec(
        R=r_cnt,
        B=16,
        T=1,
        ctx=2048,
        stack_e=False,
        cache=CacheExec(
            num_pages=64,
            page_size=128,
            max_pages_per_seq=2,
            paged_kv=True,
            dcp_kv_sharded=True,
            dcp_size=8,
            fused_pool_io=False,
            pool_gather=False,
        ),
    )
    backbone = make_synthetic_qwen35_backbone(g_dcp, ex)
    w_flat = pack_weights(backbone, g_dcp, ex, tp_rank=0)["w_flat"]
    pools = [torch.zeros((64, 2, 64, 4, 128), dtype=torch.float8_e4m3fn)]
    exported = export_pooled_step(
        mesh,
        spec,
        ex,
        w_flat,
        pools,
        max_pages_per_seq=2,
        ckpt_cols=2,
        platforms=("tpu",),
    )
    mlir_text = exported.exported.mlir_module()
    assert "sdy.manual_computation" in mlir_text
    assert "stablehlo.partition_id" not in mlir_text
    assert exported.input_output_aliases == {2 + len(w_flat): 1}

    ex_fused = dataclasses.replace(
        ex,
        B=8,
        cache=dataclasses.replace(ex.cache, fused_pool_io=True),
    )
    exported_fused = export_pooled_step(
        mesh,
        spec,
        ex_fused,
        w_flat,
        pools,
        max_pages_per_seq=2,
        ckpt_cols=2,
        platforms=("tpu",),
    )
    mlir_fused = exported_fused.exported.mlir_module()
    assert "sdy.manual_computation" in mlir_fused
    assert "stablehlo.partition_id" not in mlir_fused
    assert exported_fused.input_output_aliases == {2 + len(w_flat): 1}


def test_qwen38_tpu_bucket_step_decouples_embed_from_pooled_megakernel():
    """TPU _build_bucket_op compiles input+embedding separately from pooled step."""
    g = dataclasses.replace(TINY_EP, NKV=1)
    drv = Qwen38Megakernel.__new__(Qwen38Megakernel)
    drv.geom = g
    drv.spec = Qwen38.to_model_spec(g, R=32)
    drv.profile = get_profile("qwen38-tp32-dcp8")
    drv.base_exec = ExecSpec(
        R=32,
        B=16,
        T=1,
        cache=CacheExec(paged_kv=True, dcp_kv_sharded=True, dcp_size=8),
    )
    drv.fused_pool_io = None
    drv.fused_pool_io_max_b = 16
    drv.fused_pool_io_min_b_hi = None
    drv.max_pages_per_seq = 2
    drv.tp_rank = 0
    drv.mode = "ep"
    drv.mesh = _fake_mesh(list(range(32)))
    drv.raw_pools = [SimpleNamespace(device=SimpleNamespace(type="tpu"))]
    drv.w_flat = []
    drv._hlo_checked_buckets = {(1, 16)}
    drv._bucket_fused_io = {}
    drv._bucket_gpr = {}
    drv._bucket_ops = {}
    drv._rank_tensor = lambda _dev: torch.arange(32, dtype=torch.int32)
    drv.embed_tokens = torch.nn.Embedding(32, g.H)

    compiled_calls: list[tuple[object, dict[str, object]]] = []
    pooled_invocations: list[dict[str, object]] = []

    def fake_compile(fn, **kwargs):
        compiled_calls.append((fn, kwargs))
        return fn

    def fake_build_op(*_args, **_kwargs):
        return ("fake_pooled_op", None, None, None)

    def fake_exec_inplace(
        op, pos_b, hid, pt_b, ckpt_b, roff_b, weights, pools, *, tp_rank_1d=None
    ):
        pooled_invocations.append(
            {
                "op": op,
                "hid_shape": tuple(hid.shape),
                "pt_shape": tuple(pt_b.shape),
                "tp_rank_1d": tp_rank_1d,
            }
        )
        return hid + 1

    with (
        patch.object(torch, "compile", side_effect=fake_compile),
        patch.object(mk_driver, "build_pooled_step_op", side_effect=fake_build_op),
        patch.object(
            mk_driver, "execute_pooled_step_inplace", side_effect=fake_exec_inplace
        ),
    ):
        step_fn = drv._build_bucket_op(16, tokens_per_seq=1)
        assert step_fn.__code__.co_name == "step_t1_b16"
        assert len(compiled_calls) == 2

        prep_fn, prep_kw = compiled_calls[0]
        pooled_fn, pooled_kw = compiled_calls[1]
        assert prep_fn.__code__.co_name == "prep_step_t1_b16"
        assert prep_kw["options"] == {"name": "prep_step_t1_b16", "serializable": False}
        assert "embed" in prep_fn.__code__.co_freevars
        assert "op" not in prep_fn.__code__.co_freevars

        assert pooled_fn.__code__.co_name == "step_t1_b16"
        assert pooled_kw["options"]["name"] == "step_t1_b16"
        assert callable(pooled_kw["options"]["debug_callback"])
        assert "embed" not in pooled_fn.__code__.co_freevars
        assert "prepare_step_inputs_eager" not in pooled_fn.__code__.co_names

        ids = torch.ones((16,), dtype=torch.int32)
        pos = torch.arange(16, dtype=torch.int32)
        bt = torch.zeros((16, 2), dtype=torch.int32)
        s = torch.arange(16, dtype=torch.int32)
        c = s.reshape(-1, 1)
        roff = torch.zeros((16,), dtype=torch.int32)
        out = step_fn(ids, pos, bt, s, c, s, c, s, c, roff, [], [])

    assert out.shape == (16, g.H)
    assert len(pooled_invocations) == 1
    assert pooled_invocations[0]["op"] == "fake_pooled_op"
    assert pooled_invocations[0]["hid_shape"] == (16, g.H)
    assert pooled_invocations[0]["pt_shape"] == (16, 2)
    assert pooled_invocations[0]["tp_rank_1d"].shape == (32,)
