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
"""Qwen3.8-2.4T megakernel plugin spec, hook, parity, and agreement tests."""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
import multiprocessing
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, TypedDict, Unpack

import jax
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax._src.pallas.mosaic.interpret import interpret_pallas_call as ipc

import vllm_torchtpu.megakernel_runtime.agreement as agreement_mod
from vllm_torchtpu.megakernel_runtime import config as config_mod
from vllm_torchtpu.megakernel_runtime import hooks, testing, testkit
from vllm_torchtpu.megakernel_runtime.hooks import StepContext
from vllm_torchtpu.megakernel_runtime.plugin import (
    Bucket,
    Decline,
    ForwardView,
    LayoutDecline,
    LayoutRequest,
    RunnerView,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from vllm_torchtpu.models.qwen38_2_4t.megakernel.adapter import (
        Qwen38MoeProgram,
        StepDecision,
    )
    from vllm_torchtpu.models.qwen38_2_4t.megakernel.shapes import (
        Qwen38Config,
    )

_TEST_DIR = Path(__file__).resolve().parent
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))

oracle = importlib.import_module("oracle_fp64")


@pytest.fixture(autouse=True)
def _mock_uninitialized_distributed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub uninitialized vLLM TP rank getters to return rank 0 during unit tests."""
    with contextlib.suppress(ImportError, AttributeError):
        ps = importlib.import_module("vllm.distributed.parallel_state")
        if vars(ps).get("_TP") is None:
            monkeypatch.setattr(ps, "get_tensor_model_parallel_rank", lambda: 0)
            dist_mod = importlib.import_module("vllm.distributed")
            monkeypatch.setattr(dist_mod, "get_tensor_model_parallel_rank", lambda: 0)


_PKG = "vllm_torchtpu.models.qwen38_2_4t.megakernel"

_EXPECTED_BLOCK_SIZE: int = 128
_EXPECTED_PAGE_SIZE: int = 128
_EXPECTED_DCP_INTERLEAVE_SIZE: int = 128
_EXPECTED_MAMBA_SLOT_TOKENS: int = 640
_EXPECTED_STREAM_ITEMS: int = 5
_CUSTOM_W13_BYTE: int = 0x33
_CUSTOM_L1_W2_BYTE: int = 0x55
_MIN_CPU_DEVICES: int = 32
_MIN_TILED_DIM: int = 2
_MAX_KV_ULP_TOLERANCE: int = 2
_EXPECTED_L8_REGIONS: int = 2


class VllmCfgOptions(TypedDict, total=False):
    """Optional settings for fake vLLM configuration."""

    arch: str
    model_type: str
    quant_method: str
    megakernel: str | bool | Mapping[str, object] | None
    block_size: int


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset environment variables and route loggers to null sink."""
    monkeypatch.delenv("VLLM_TPU_MEGAKERNEL", raising=False)
    monkeypatch.setattr(agreement_mod, "_tp_cpu_group", lambda: None)
    testkit.route_lazy_loggers(monkeypatch, _PKG)


class _Qwen38FakeAttentionBackend:
    """Deterministic attention backend for cache-slot floor derivation tests."""

    @staticmethod
    def get_name() -> str:
        """Return backend identifier."""
        return "QWEN38_TEST_BACKEND"

    @staticmethod
    def get_min_page_size(_vllm_config: object) -> int:
        """Return minimum supported page size in tokens."""
        return _EXPECTED_PAGE_SIZE

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        """Return supported kernel block granularity."""
        return [_EXPECTED_PAGE_SIZE]

    @staticmethod
    def get_kv_cache_page_size_bytes(
        block_size: int, *_args: object, **_kwargs: object
    ) -> int:
        """Return linear 1 KiB-per-token physical slot byte size."""
        bytes_per_token = 1024
        return block_size * bytes_per_token


def _vllm_cfg(
    num_layers: int = 92,
    hidden_size: int = 8192,
    **kwargs: Unpack[VllmCfgOptions],
) -> SimpleNamespace:
    """Build mock vLLM configuration for plugin selection and hook tests."""
    arch = kwargs.get("arch", "Qwen3_5MoeForCausalLM")
    model_type = kwargs.get("model_type", "qwen3_5_moe_text")
    quant_method = kwargs.get("quant_method", "nvfp4")
    megakernel = kwargs.get("megakernel", {"plugin": _PKG})
    block_size = kwargs.get("block_size", 256)

    hf_cfg = SimpleNamespace(
        architectures=[arch],
        model_type=model_type,
        num_hidden_layers=num_layers,
        hidden_size=hidden_size,
        quantization_config={"quant_method": quant_method},
    )
    cfg = testing.fake_vllm_config(
        megakernel=megakernel,
        block_size=block_size,
        kv_cache_dtype=torch.bfloat16,
        hf_config=hf_cfg,
        tp_size=32,
        dcp_size=8,
    )
    cfg.parallel_config.enable_expert_parallel = True
    return cfg


def _verify_adapter_create_92l(adapter_mod: object) -> None:
    """Verify adapter.create succeeds on valid 92-layer production geometry."""
    exp_prod_layers = 92
    exp_bucket_4 = 4
    prod_hf = SimpleNamespace(
        architectures=["Qwen3_5MoeForCausalLM"],
        model_type="qwen3_5_moe_text",
        num_hidden_layers=exp_prod_layers,
        hidden_size=8192,
        num_attention_heads=128,
        num_key_value_heads=4,
        head_dim=256,
        linear_num_key_heads=16,
        linear_num_value_heads=64,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        num_experts=512,
        num_experts_per_tok=8,
        moe_intermediate_size=2048,
        shared_expert_intermediate_size=2048,
        quantization_config={"quant_method": "nvfp4"},
    )
    r_prod = testing.fake_runner(
        model=SimpleNamespace(layers=[], norm=None),
        hf_config=prod_hf,
        tp_size=32,
        enable_expert_parallel=True,
    )
    r_prod.parallel_config.decode_context_parallel_size = 8
    create_fn = vars(adapter_mod)["create"]
    prog_cls = vars(adapter_mod)["Qwen38MoeProgram"]
    res_prod = create_fn(RunnerView(r_prod, tp_rank=0), {"buckets": "4"})
    assert isinstance(res_prod, prog_cls)
    assert int(res_prod.base_config.L) == exp_prod_layers
    assert int(res_prod.base_config.B) == exp_bucket_4


def test_qwen38_spec_is_stdlib_only_and_shorthand_resolves() -> None:
    """Verify plugin specification, explicit module selection, and scratch plan."""
    spec = testkit.check_spec(_PKG)
    assert spec.name == "qwen38_2_4t"
    assert spec.version == "1"
    assert spec.create == ".adapter:create"
    assert spec.layout_request == ".adapter:layout_request"
    assert spec.load_tensor == ".weights:load_tensor"
    assert spec.collective_domain == "tp_x_dcp"
    assert spec.chips == ("TPU7x",)
    assert spec.parallelism == frozenset({"tp", "dcp", "ep_in_tp"})
    assert spec.features == frozenset()
    assert spec.numerics == "approx"

    vcfg_92 = _vllm_cfg(megakernel={"plugin": _PKG}, num_layers=92, hidden_size=8192)
    _, res_92 = config_mod.select_plugin(
        None, vcfg_92.additional_config, vcfg_92.model_config.hf_config
    )
    assert res_92 is not None
    assert res_92[0].module == _PKG

    _, res_env = config_mod.select_plugin(
        f"plugin={_PKG}", {}, vcfg_92.model_config.hf_config
    )
    assert res_env is not None
    assert res_env[0].module == _PKG

    vcfg_60 = _vllm_cfg(megakernel=True, num_layers=60, hidden_size=4096)
    _, res_60 = config_mod.select_plugin(
        None, vcfg_60.additional_config, vcfg_60.model_config.hf_config
    )
    assert res_60 is not None
    assert res_60[0].module == "vllm_torchtpu.models.qwen3_5.megakernel"

    # Verify stage scratchpad plan reuses disjoint intervals on mini and prod shapes.
    shapes = importlib.import_module(f"{_PKG}.shapes")
    plan = importlib.import_module(f"{_PKG}.plan")
    for cfg in (shapes.mini_parity_config(batch_size=4), shapes.Qwen38Config(B=16)):
        stage_plan, slices = plan.build_stage_scratch_plan(cfg)
        assert set(slices) == {"wo_buf", "wout_buf", "wse2_buf"}
        assert slices["wo_buf"].offset_bytes == 0
        assert slices["wout_buf"].offset_bytes == 0
        assert slices["wse2_buf"].offset_bytes == 0
        assert stage_plan.peak_bytes < (
            slices["wo_buf"].size_bytes
            + slices["wout_buf"].size_bytes
            + slices["wse2_buf"].size_bytes
        )
        summary = plan.scratch_footprint(cfg)
        assert summary["stage_scratch_high_water_bytes"] == stage_plan.peak_bytes
        assert int(summary["stage_scratch_saved_bytes"]) > 0
        assert summary["fits_in_limit"] is True
        assert summary["fits_in_scalar_limit"] is True

    adapter_mod = importlib.import_module(f"{_PKG}.adapter")
    _assert_qwen38_decline_cases(adapter_mod)
    _verify_adapter_create_92l(adapter_mod)


def _assert_qwen38_decline_cases(adapter_mod: object) -> None:
    """Verify `adapter.create` declines on mismatched geometry or quantization."""
    create_fn = vars(adapter_mod)["create"]
    bad_hf = SimpleNamespace(
        architectures=["Qwen3_5MoeForCausalLM"],
        model_type="qwen3_5_moe_text",
        num_hidden_layers=92,
        hidden_size=8192,
        num_attention_heads=64,
        quantization_config={"quant_method": "nvfp4"},
    )
    r_bad_geom = testing.fake_runner(hf_config=bad_hf, tp_size=32)
    r_bad_geom.parallel_config.decode_context_parallel_size = 8
    res_bad_geom = create_fn(RunnerView(r_bad_geom, tp_rank=0), {})
    assert isinstance(res_bad_geom, Decline)
    assert res_bad_geom.code == "unsupported_geometry"

    for bad_qcfg in (None, {"quant_method": "modelopt", "quant_algo": "FP8"}):
        unquant_hf = SimpleNamespace(
            architectures=["Qwen3_5MoeForCausalLM"],
            model_type="qwen3_5_moe_text",
            num_hidden_layers=92,
            hidden_size=8192,
            quantization_config=bad_qcfg,
        )
        r_bad_q = testing.fake_runner(hf_config=unquant_hf, tp_size=32)
        r_bad_q.parallel_config.decode_context_parallel_size = 8
        res_bad_q = create_fn(RunnerView(r_bad_q, tp_rank=0), {})
        assert isinstance(res_bad_q, Decline)
        assert res_bad_q.code == "unsupported_quant"


def test_qwen38_stage_scratch_plan_and_live_layout_contract() -> None:
    """Verify typed LayoutRequest return and stage scratchpad naming contract."""
    adapter_mod = importlib.import_module(f"{_PKG}.adapter")
    plan_mod = importlib.import_module(f"{_PKG}.plan")
    shapes_mod = importlib.import_module(f"{_PKG}.shapes")

    vcfg = _vllm_cfg(megakernel={"plugin": _PKG}, block_size=256)
    req = adapter_mod.layout_request(vcfg)
    assert isinstance(req, LayoutRequest)
    assert req == LayoutRequest(
        block_size=_EXPECTED_BLOCK_SIZE,
        cp_kv_cache_interleave_size=_EXPECTED_DCP_INTERLEAVE_SIZE,
        mamba_slot_tokens=_EXPECTED_MAMBA_SLOT_TOKENS,
    )
    assert req.slot_tokens_floor == _EXPECTED_MAMBA_SLOT_TOKENS

    cfg = shapes_mod.mini_parity_config(batch_size=4)
    stage_plan, slices = plan_mod.build_stage_scratch_plan(cfg)
    assert stage_plan.peak_bytes == max(s.size_bytes for s in slices.values())
    specs = plan_mod.make_scratch_specs(cfg)
    assert "stage_scratch_u8" in specs
    assert "stage_arena_u8" not in specs


def test_qwen38_load_tensor_and_layout_request_hooks() -> None:
    """Verify layout request application, slot-floor sizing, and weight hooks."""
    shapes = importlib.import_module(f"{_PKG}.shapes")
    weights_mod = importlib.import_module(f"{_PKG}.weights")
    block_utils_mod = importlib.import_module(
        "vllm_torchtpu.platforms.tpu_block_size_utils"
    )

    # 1. Verify apply_layout_request returns LayoutRequest and sets 128-token page/DCP.
    vcfg = _vllm_cfg(megakernel={"plugin": _PKG}, block_size=256)
    req = hooks.apply_layout_request(vcfg)
    assert req == LayoutRequest(
        block_size=_EXPECTED_BLOCK_SIZE,
        cp_kv_cache_interleave_size=_EXPECTED_DCP_INTERLEAVE_SIZE,
        mamba_slot_tokens=_EXPECTED_MAMBA_SLOT_TOKENS,
    )
    assert vcfg.cache_config.block_size == _EXPECTED_BLOCK_SIZE
    assert vcfg.parallel_config.cp_kv_cache_interleave_size == (
        _EXPECTED_DCP_INTERLEAVE_SIZE
    )
    assert vcfg.parallel_config.dcp_kv_cache_interleave_size == (
        _EXPECTED_DCP_INTERLEAVE_SIZE
    )
    assert "_megakernel_layout_applied" not in vars(vcfg)

    # 2. Verify update_tpu_block_size_and_slot_config raises slot floor to 640
    # and bind() enforces the 640-token Mamba slot floor.
    cfg = shapes.mini_parity_config(batch_size=4)
    prob = oracle.make_synthetic_problem(cfg, seed=7)
    mock_model = oracle.build_mock_vllm_qwen38_rank_models(prob, cfg)
    adapter_mod = importlib.import_module(f"{_PKG}.adapter")

    r_layout = testing.fake_runner(
        model=mock_model.model,
        hf_config=SimpleNamespace(
            architectures=["Qwen3_5MoeForCausalLM"],
            model_type="qwen3_5_moe_text",
            num_hidden_layers=cfg.L,
            hidden_size=cfg.H,
            quantization_config={"quant_method": "nvfp4"},
        ),
        tp_size=32,
        block_size=128,
        max_pages_per_seq=cfg.max_pages_per_seq,
        enable_expert_parallel=True,
    )
    r_layout.parallel_config.decode_context_parallel_size = 8
    prog_layout = adapter_mod.create(
        RunnerView(r_layout, tp_rank=0), {"allow_mini_geometry": True}
    )
    with pytest.raises(LayoutDecline):
        prog_layout.bind(RunnerView(r_layout, tp_rank=0))

    block_utils_mod.update_tpu_block_size_and_slot_config(
        vcfg, _Qwen38FakeAttentionBackend, layout_request=req
    )
    assert vcfg.cache_config.block_size == _EXPECTED_MAMBA_SLOT_TOKENS
    r_layout.vllm_config = vcfg
    r_layout.cache_config = vcfg.cache_config
    bind_info = prog_layout.bind(RunnerView(r_layout, tp_rank=0))
    assert bind_info["num_pages"] == cfg.num_pages

    weights_mod.clear_observed_weights()
    layer0_exps = mock_model.model.layers[0].mlp.experts
    orig_w13 = layer0_exps.w13_weight.clone()
    orig_s13 = layer0_exps.w13_weight_scale.clone()
    custom_w13 = torch.full_like(orig_w13, _CUSTOM_W13_BYTE)
    layer1_exps = mock_model.model.layers[1].mlp.experts
    orig_l1_w2 = layer1_exps.w2_weight.clone()
    orig_l1_s2 = layer1_exps.w2_weight_scale.clone()
    custom_l1_w2 = torch.full_like(orig_l1_w2, _CUSTOM_L1_W2_BYTE)

    stream = [
        ("model.layers.0.mlp.experts.w13_weight", custom_w13),
        ("model.layers.0.mlp.experts.w13_weight_scale", orig_s13),
        ("model.layers.1.mlp.experts.down_proj", custom_l1_w2),
        ("model.layers.1.mlp.experts.down_proj_scale", orig_l1_s2),
        ("model.layers.0.input_layernorm.weight", torch.ones(cfg.H)),
    ]
    out_stream = list(hooks.load_tensor_stream(vcfg, stream))
    assert len(out_stream) == _EXPECTED_STREAM_ITEMS

    # Clear post-load attributes on rank 0 layers 0 & 1 to verify observed weight read.
    layer0_exps.w13_weight = torch.zeros_like(orig_w13)
    layer1_exps.w2_weight = torch.zeros_like(orig_l1_w2)
    extracted = weights_mod.extract_backbone_weights_torch(
        mock_model, cfg, device=torch.device("cpu")
    )
    assert int(torch.max(extracted["w_e13"][0, 0]).item()) == _CUSTOM_W13_BYTE
    assert int(torch.max(extracted["w_e2"][0, 1]).item()) == _CUSTOM_L1_W2_BYTE
    weights_mod.clear_observed_weights()


def _verify_kv_pool_updates(
    got_out: dict[str, Any],
    ref_out: dict[str, Any],
    cfg: Qwen38Config,
    prob: dict[str, Any],
) -> None:
    """Verify KV pool ULP accuracy and untouched memory preservation."""
    pos_np = np.asarray(jax.device_get(prob["pos"])[0], dtype=np.int32)
    pt_np = np.asarray(jax.device_get(prob["page_table"]), dtype=np.int32)
    written_mask = np.zeros((cfg.R, cfg.num_pages), dtype=bool)
    max_kv_ulp = 0
    for r in range(cfg.R):
        dcp_rank = r % cfg.DCP
        for att_idx in range(cfg.NATT):
            for b in range(cfg.B):
                pb = int(pos_np[b])
                gp = pb // cfg.page_size
                if gp % cfg.DCP == dcp_rank:
                    lp = gp // cfg.DCP
                    off = pb % cfg.page_size
                    p_idx = int(pt_np[r, att_idx, b, lp])
                    written_mask[r, p_idx] = True
                    ulp_k = oracle.max_fp8_ulp_diff(
                        got_out["pool"][r, p_idx, 0, :, :, off],
                        ref_out["pool"][r, p_idx, 0, :, :, off],
                    )
                    ulp_v = oracle.max_fp8_ulp_diff(
                        got_out["pool"][r, p_idx, 1, :, :, off],
                        ref_out["pool"][r, p_idx, 1, :, :, off],
                    )
                    max_kv_ulp = max(max_kv_ulp, ulp_k, ulp_v)
    assert max_kv_ulp <= _MAX_KV_ULP_TOLERANCE, (
        f"max_kv_ulp={max_kv_ulp} > {_MAX_KV_ULP_TOLERANCE}"
    )

    pool_init_u8 = np.asarray(jax.device_get(prob["pool"]).view(jnp.uint8))
    pool_got_u8 = np.asarray(jax.device_get(got_out["pool"]).view(jnp.uint8))
    untouched_diff = int(
        np.max(
            np.abs(
                pool_got_u8[~written_mask].astype(np.int32)
                - pool_init_u8[~written_mask].astype(np.int32)
            )
        )
    )
    assert untouched_diff == 0, f"untouched_diff={untouched_diff}"


def _verify_output_tensors(
    got_out: dict[str, Any],
    ref_out: dict[str, Any],
    cfg: Qwen38Config,
    prob: dict[str, Any],
) -> tuple[np.ndarray, float]:
    """Verify RMSE bounds and KV cache updates against FP64 reference."""
    got_x = np.asarray(
        jax.device_get(got_out["x"]).astype(jnp.float32), dtype=np.float64
    )
    ref_x = ref_out["x"]
    rmse_x = float(np.sqrt(np.mean((got_x - ref_x) ** 2)))
    rms_x = float(np.sqrt(np.mean(ref_x**2)))
    bound_x = oracle.layer_bf16_error_bound(rms_x, 3, 4)
    assert rmse_x <= bound_x, f"rmse_x={rmse_x} > bound_x={bound_x}"

    got_conv = np.asarray(
        jax.device_get(got_out["conv"]).astype(jnp.float32), dtype=np.float64
    )
    ref_conv = ref_out["conv"]
    rmse_conv = float(np.sqrt(np.mean((got_conv - ref_conv) ** 2)))
    bound_conv = oracle.layer_bf16_error_bound(
        float(np.sqrt(np.mean(ref_conv**2))), 1, 1
    )
    assert rmse_conv <= bound_conv, f"rmse_conv={rmse_conv} > {bound_conv}"

    got_rec = np.asarray(jax.device_get(got_out["rec"]), dtype=np.float64)
    ref_rec = ref_out["rec"]
    rmse_rec = float(np.sqrt(np.mean((got_rec - ref_rec) ** 2)))
    bound_rec = oracle.layer_bf16_error_bound(float(np.sqrt(np.mean(ref_rec**2))), 1, 1)
    assert rmse_rec <= bound_rec, f"rmse_rec={rmse_rec} > {bound_rec}"

    _verify_kv_pool_updates(got_out, ref_out, cfg, prob)
    return got_x, bound_x


def _verify_program_driver_e2e(
    prob: dict[str, Any],
    cfg: Qwen38Config,
    ref_out: dict[str, Any],
) -> None:
    """Verify ProgramDriver parity via try_forward."""
    mock_model = oracle.build_mock_vllm_qwen38_rank_models(prob, cfg)
    hf_cfg = SimpleNamespace(
        architectures=["Qwen3_5MoeForCausalLM"],
        model_type="qwen3_5_moe_text",
        num_hidden_layers=cfg.L,
        hidden_size=cfg.H,
        quantization_config={"quant_method": "nvfp4"},
    )
    qsl = np.zeros(16, dtype=np.int32)
    qsl[: cfg.B + 1] = np.arange(cfg.B + 1, dtype=np.int32)
    runner = testing.fake_runner(
        model=mock_model,
        hf_config=hf_cfg,
        tp_size=32,
        block_size=640,
        token_paddings=(cfg.B,),
        req_block_pairs=((cfg.B, cfg.max_pages_per_seq),),
        max_pages_per_seq=cfg.max_pages_per_seq,
        enable_expert_parallel=True,
        _dcp_enabled=True,
        _attn_metadata_builder_ctx=SimpleNamespace(num_reqs=cfg.B),
        query_start_loc_np=qsl,
        _cached_comb_key=None,
        _combined_request_distribution_cpu=None,
    )
    runner.parallel_config.decode_context_parallel_size = 8
    runner.vllm_config.parallel_config.decode_context_parallel_size = 8

    pt_np = np.asarray(jax.device_get(prob["page_table"]), dtype=np.int32)
    bt_t = torch.from_numpy(pt_np[0, 0].copy()).to(torch.int32)
    mamba_slots = prob["mamba_slots"]

    def _build_md(prog: Qwen38MoeProgram) -> dict[str, SimpleNamespace]:
        """Construct attention metadata for mock driver."""
        md: dict[str, SimpleNamespace] = {
            prog.attn_layer_names[0]: SimpleNamespace(block_table=bt_t),
        }
        for g_idx, m_name in enumerate(prog.mamba_layer_names):
            md[m_name] = SimpleNamespace(
                mamba_state_indices=torch.from_numpy(mamba_slots[g_idx].copy()).to(
                    torch.int32
                )
            )
        return md

    pos_np = np.asarray(jax.device_get(prob["pos"])[0], dtype=np.int32)
    ctx_step = StepContext(
        input_ids=torch.arange(cfg.B, dtype=torch.int32),
        positions=torch.from_numpy(pos_np.copy()).to(torch.int32),
    )
    e2e = testkit.run_cpu_e2e(
        _PKG,
        runner,
        [ctx_step],
        metadata=_build_md,
        options={
            "buckets": str(cfg.B),
            "allow_mini_geometry": True,
            "interpret": True,
        },
    )
    assert e2e.hits == [True]
    assert e2e.step_mix == {f"mk_t1_b{cfg.B}": 1}

    out_prog = e2e.outputs[0]
    got_prog_x = out_prog.detach().to(torch.float32).numpy().astype(np.float64)
    rmse_prog = float(np.sqrt(np.mean((got_prog_x - ref_out["x"]) ** 2)))
    bound_prog = oracle.layer_bf16_error_bound(
        float(np.sqrt(np.mean(ref_out["x"] ** 2))), 3, 4
    )
    assert rmse_prog <= bound_prog, f"rmse_prog={rmse_prog} > {bound_prog}"


def _verify_negative_controls(
    prob: dict[str, Any],
    cfg: Qwen38Config,
    got_x: np.ndarray,
    bound_x: float,
    ref_out: dict[str, Any],
) -> None:
    """Verify weight mutation sensitivity via negative controls."""
    # Negative control 1: zeroing routed expert w_e2 must breach > 5x bound_x.
    prob_no_e2 = dict(prob)
    prob_no_e2["weights"] = dict(prob["weights"])
    prob_no_e2["weights"]["w_e2"] = jnp.zeros_like(prob["weights"]["w_e2"])
    ref_no_e2 = oracle.run_fp64_reference(prob_no_e2, cfg, use_forced_routing=False)
    rmse_no_e2 = float(np.sqrt(np.mean((got_x[0] - ref_no_e2["x"]) ** 2)))
    assert rmse_no_e2 > 5.0 * bound_x, (
        f"Routed expert sensitivity too weak: rmse_no_e2={rmse_no_e2} <= 5*{bound_x}"
    )

    # Negative control 2: zeroing shared expert w_se2 must breach > 5x bound_x.
    prob_no_se2 = dict(prob)
    prob_no_se2["weights"] = dict(prob["weights"])
    prob_no_se2["weights"]["w_se2"] = jnp.zeros_like(prob["weights"]["w_se2"])
    ref_no_se2 = oracle.run_fp64_reference(prob_no_se2, cfg, use_forced_routing=False)
    rmse_no_se2 = float(np.sqrt(np.mean((got_x[0] - ref_no_se2["x"]) ** 2)))
    assert rmse_no_se2 > 5.0 * bound_x, (
        f"Shared expert sensitivity too weak: rmse_no_se2={rmse_no_se2} <= 5*{bound_x}"
    )

    _verify_program_driver_e2e(prob, cfg, ref_out)


def _execute_qwen38_cpu_e2e_test() -> None:
    """Run full parity and negative controls on 32 CPU devices."""
    assert len(jax.devices()) >= _MIN_CPU_DEVICES, (
        f"Expected >={_MIN_CPU_DEVICES} CPU devices, got {len(jax.devices())}"
    )
    shapes = importlib.import_module(f"{_PKG}.shapes")
    plan = importlib.import_module(f"{_PKG}.plan")
    kernel = importlib.import_module(f"{_PKG}.kernel")
    weights_mod = importlib.import_module(f"{_PKG}.weights")
    weights_mod.clear_observed_weights()

    cfg = shapes.mini_parity_config(batch_size=2)
    prob = oracle.make_synthetic_problem(
        cfg,
        seed=7,
        pos=(0, 2600),
        forced_top_ids=[
            [[0, 2], [4, 6]],
            [[1, 3], [5, 7]],
            [[2, 4], [6, 8]],
            [[3, 5], [7, 9]],
        ],
    )

    norm_w = plan.normalize_weights_for_pallas(prob["weights"], cfg)
    assert norm_w["wqkv"].shape[2] >= _MIN_TILED_DIM
    assert norm_w["w_inproj"].shape[2] >= _MIN_TILED_DIM
    assert norm_w["w_se13"].shape[2] >= _MIN_TILED_DIM
    assert norm_w["w_e13"].shape[3] >= _MIN_TILED_DIM
    assert norm_w["w_e2"].shape[3] >= _MIN_TILED_DIM

    ref_out = oracle.run_fp64_reference(prob, cfg, use_forced_routing=False)
    got_out = kernel.run_qwen38_dcp8_step(
        prob,
        cfg,
        use_forced_routing=False,
        return_kv_updates=True,
        interpret=True,
    )

    got_x, bound_x = _verify_output_tensors(got_out, ref_out, cfg, prob)
    _verify_negative_controls(prob, cfg, got_x, bound_x, ref_out)


def _patch_uninitialized_tp_rank() -> None:
    """Stub uninitialized vLLM TP rank getters in standalone CPU worker processes."""
    with contextlib.suppress(ImportError, AttributeError):
        ps = importlib.import_module("vllm.distributed.parallel_state")
        if vars(ps).get("_TP") is None:
            vars(ps)["get_tensor_model_parallel_rank"] = lambda: 0
            dist_mod = importlib.import_module("vllm.distributed")
            vars(dist_mod)["get_tensor_model_parallel_rank"] = lambda: 0


def _run_qwen38_cpu_parity_worker() -> None:
    """Worker function executed in spawned 32-device process."""
    _patch_uninitialized_tp_rank()
    _execute_qwen38_cpu_e2e_test()


def test_qwen38_cpu_e2e_parity_and_zero_weight_negative_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Qwen3.8 CPU 32-device parity, KV updates, and negative controls."""
    if len(jax.devices()) < _MIN_CPU_DEVICES:
        monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
        monkeypatch.setenv("VLLM_PLUGINS", "")
        monkeypatch.setenv("JAX_PLATFORMS", "cpu")
        monkeypatch.setenv("PJRT_DEVICE", "CPU")
        monkeypatch.setenv(
            "XLA_FLAGS",
            "--xla_force_host_platform_device_count=32 "
            "--xla_cpu_multi_thread_eigen=true "
            "intra_op_parallelism_threads=8",
        )
        ctx = multiprocessing.get_context("spawn")
        proc = ctx.Process(target=_run_qwen38_cpu_parity_worker)
        proc.start()
        proc.join(timeout=300)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
            pytest.fail("Child process timed out after 300s")
        assert proc.exitcode == 0, (
            f"Child process failed with exit code {proc.exitcode}"
        )
        return

    _execute_qwen38_cpu_e2e_test()


def _build_l8_problem_and_runner(
    cfg_l8: Qwen38Config,
) -> tuple[dict[str, Any], np.ndarray, dict[str, np.ndarray], Any, Any]:
    """Construct synthetic 2-region `L=8` problem, FP64 reference, and runner."""
    prob_l8 = oracle.make_synthetic_problem(
        cfg_l8,
        seed=99,
        pos=(130, 260, 130, 260),
    )
    pool_l8_raw = np.stack(
        [
            np.asarray(jax.device_get(prob_l8["pool"]).astype(jnp.float32)),
            np.asarray(jax.device_get(prob_l8["pool"]).astype(jnp.float32)) * 0.7
            + 0.05,
        ],
        axis=1,
    )
    prob_l8["pool"] = jnp.asarray(pool_l8_raw, dtype=jnp.float8_e4m3fn)
    pt_l8_np = np.asarray(jax.device_get(prob_l8["page_table"]), dtype=np.int32).copy()
    pt_l8_np[:, 1] = pt_l8_np[:, 0]
    prob_l8["page_table"] = jnp.asarray(pt_l8_np, dtype=jnp.int32)
    prob_l8["mamba_slots"][:, 0] = 4
    prob_l8["mamba_slots"][:, 1] = 5
    prob_l8["mamba_slots"][:, 2:] = -1
    ref_l8 = oracle.run_fp64_reference(prob_l8, cfg_l8, use_forced_routing=False)

    mock_l8 = oracle.build_mock_vllm_qwen38_rank_models(prob_l8, cfg_l8)
    hf_l8 = SimpleNamespace(
        architectures=["Qwen3_5MoeForCausalLM"],
        model_type="qwen3_5_moe_text",
        num_hidden_layers=cfg_l8.L,
        hidden_size=cfg_l8.H,
        quantization_config={"quant_method": "nvfp4"},
    )
    qsl_pad = np.zeros(16, dtype=np.int32)
    qsl_pad[:3] = np.arange(3, dtype=np.int32)
    runner_single = testing.fake_runner(
        model=mock_l8.model,
        hf_config=hf_l8,
        tp_size=32,
        block_size=640,
        token_paddings=(4,),
        req_block_pairs=((2, cfg_l8.max_pages_per_seq), (4, cfg_l8.max_pages_per_seq)),
        max_pages_per_seq=cfg_l8.max_pages_per_seq,
        enable_expert_parallel=True,
        _dcp_enabled=True,
        _attn_metadata_builder_ctx=SimpleNamespace(num_reqs=2),
        query_start_loc_np=qsl_pad,
        _cached_comb_key=None,
        _combined_request_distribution_cpu=None,
    )
    runner_single.parallel_config.decode_context_parallel_size = 8
    runner_single.vllm_config.parallel_config.decode_context_parallel_size = 8
    return prob_l8, pt_l8_np, ref_l8, mock_l8, runner_single


def _assert_l8_outputs_and_states(
    prog: Qwen38MoeProgram,
    dec_l8: StepDecision,
    ctx_l8: StepContext,
    fwd_l8: ForwardView,
    l8_bundle: tuple[
        Any,
        Any,
        Qwen38Config,
        dict[str, Any],
        np.ndarray,
        dict[str, Any],
        torch.Tensor,
        torch.Tensor,
    ],
) -> torch.Tensor:
    """Verify L=8 multi-region KV pools, GDN states, and decode output parity."""
    (
        mock_l8,
        kernel_mod,
        cfg_l8,
        prob_l8,
        pt_l8_np,
        ref_l8,
        page0_reg0_before,
        page0_reg1_before,
    ) = l8_bundle
    out_l8 = prog.forward(dec_l8, ctx_l8, fwd_l8)
    assert ipc.races is not None, "Interpret race detection did not run"
    assert not ipc.races.races_found, f"Interpret race detected: {ipc.races}"
    assert out_l8.shape == (2, cfg_l8.H)

    got_x_l8 = out_l8.detach().to(torch.float32).numpy().astype(np.float64)
    ref_x_l8 = ref_l8["x"][:2]
    rmse_x_l8 = float(np.sqrt(np.mean((got_x_l8 - ref_x_l8) ** 2)))
    bound_x_l8 = oracle.layer_bf16_error_bound(
        float(np.sqrt(np.mean(ref_x_l8**2))), 3, cfg_l8.L
    )
    assert rmse_x_l8 <= bound_x_l8, f"rmse_x_l8={rmse_x_l8} > {bound_x_l8}"

    for att_idx in range(cfg_l8.NATT):
        reg_pool_got = jnp.asarray(
            prog.region_pools[att_idx].view(torch.uint8).numpy()
        ).view(jnp.float8_e4m3fn)
        for b in range(2):
            pb = int(ctx_l8.positions[b].item())
            lp = pb // cfg_l8.page_size
            off = pb % cfg_l8.page_size
            p_idx = int(pt_l8_np[0, att_idx, b, lp])
            ulp_k = oracle.max_fp8_ulp_diff(
                reg_pool_got[p_idx, 0, :, :, off],
                ref_l8["pool"][0, att_idx, p_idx, 0, :, :, off],
            )
            ulp_v = oracle.max_fp8_ulp_diff(
                reg_pool_got[p_idx, 1, :, :, off],
                ref_l8["pool"][0, att_idx, p_idx, 1, :, :, off],
            )
            assert max(ulp_k, ulp_v) <= _MAX_KV_ULP_TOLERANCE

    assert prog.unified_pool is not None
    got_conv_l8, got_rec_l8 = kernel_mod.unpack_gdn_states_from_unified_pool(
        jnp.asarray(prog.unified_pool.view(torch.uint8).numpy()),
        jnp.asarray(prob_l8["mamba_slots"], dtype=jnp.int32),
        cfg_l8,
    )
    got_conv_np = np.asarray(
        jax.device_get(got_conv_l8[:, :2]).astype(jnp.float32), dtype=np.float64
    )
    ref_conv_np = ref_l8["conv"][0, :, :2]
    rmse_conv_l8 = float(np.sqrt(np.mean((got_conv_np - ref_conv_np) ** 2)))
    bound_conv_l8 = oracle.layer_bf16_error_bound(
        float(np.sqrt(np.mean(ref_conv_np**2))), 2, cfg_l8.L
    )
    assert rmse_conv_l8 <= bound_conv_l8, (
        f"rmse_conv_l8={rmse_conv_l8} > {bound_conv_l8}"
    )

    got_rec_np = np.asarray(jax.device_get(got_rec_l8[:, :2]), dtype=np.float64)
    ref_rec_np = ref_l8["rec"][0, :, :2]
    rmse_rec_l8 = float(np.sqrt(np.mean((got_rec_np - ref_rec_np) ** 2)))
    bound_rec_l8 = oracle.layer_bf16_error_bound(
        float(np.sqrt(np.mean(ref_rec_np**2))), 2, cfg_l8.L
    )
    assert rmse_rec_l8 <= bound_rec_l8, f"rmse_rec_l8={rmse_rec_l8} > {bound_rec_l8}"

    page0_reg0_after = (
        mock_l8.model.layers[3].self_attn.attn.kv_cache[0].view(torch.uint8)
    )
    page0_reg1_after = (
        mock_l8.model.layers[7].self_attn.attn.kv_cache[0].view(torch.uint8)
    )
    assert torch.equal(page0_reg0_after, page0_reg0_before)
    assert torch.equal(page0_reg1_after, page0_reg1_before)
    assert not torch.equal(
        mock_l8.model.layers[3].self_attn.attn.kv_cache[1].view(torch.uint8),
        mock_l8.model.layers[7].self_attn.attn.kv_cache[1].view(torch.uint8),
    )

    # Negative control: scaling final_norm by -4.0 must trigger _sync_jax_weights
    return out_l8


def test_qwen38_multi_region_l8_and_single_backbone() -> None:
    """Verify 2-region L=8 KV pools and single-backbone execution."""
    shapes = importlib.import_module(f"{_PKG}.shapes")
    adapter_mod = importlib.import_module(f"{_PKG}.adapter")
    kernel_mod = importlib.import_module(f"{_PKG}.kernel")
    weights_mod = importlib.import_module(f"{_PKG}.weights")
    weights_mod.clear_observed_weights()

    cfg_l8 = dataclasses.replace(
        shapes.mini_parity_config(batch_size=4),
        R=1,
        DCP=1,
        hier_group=1,
        Q_H=4,
        KV_H=1,
        KH=1,
        VH=2,
        L=8,
        E=4,
        K=2,
        h_tile=64,
    )
    prob_l8, pt_l8_np, ref_l8, mock_l8, runner_single = _build_l8_problem_and_runner(
        cfg_l8
    )

    prog = adapter_mod.Qwen38MoeProgram(
        mock_l8.model,
        tp_rank=0,
        decode_buckets=(4,),
        base_config=cfg_l8,
        interpret=pltpu.InterpretParams(
            detect_races=True,
            out_of_bounds_reads="raise",
            uninitialized_memory="nan",
        ),
    )
    prog.load(RunnerView(runner_single, tp_rank=0))
    bind_res = prog.bind(RunnerView(runner_single, tp_rank=0))
    assert prog.rank_pools is None
    assert prog.unified_pool is not None
    assert bind_res["num_regions"] == _EXPECTED_L8_REGIONS
    assert len(prog.region_pools) == _EXPECTED_L8_REGIONS

    step_fn = prog.build(Bucket(1, 4))
    bt_l8_t = torch.from_numpy(pt_l8_np[0, 0, :2].copy()).to(torch.int32)
    shared_ms = torch.tensor([4, 5], dtype=torch.int32)
    md_l8: dict[str, Any] = {}
    for a_name in prog.attn_layer_names:
        md_l8[a_name] = SimpleNamespace(block_table=bt_l8_t)
    for m_name in prog.mamba_layer_names:
        md_l8[m_name] = SimpleNamespace(mamba_state_indices=shared_ms)

    page0_reg0_before = (
        mock_l8.model.layers[3].self_attn.attn.kv_cache[0].view(torch.uint8).clone()
    )
    page0_reg1_before = (
        mock_l8.model.layers[7].self_attn.attn.kv_cache[0].view(torch.uint8).clone()
    )
    ctx_l8 = StepContext(
        input_ids=torch.arange(2, dtype=torch.int32),
        positions=torch.tensor([130, 260], dtype=torch.int32),
    )
    fwd_l8 = ForwardView(
        attn_metadata=md_l8,
        mrope_positions=None,
        steps={(1, 4): step_fn},
        num_tokens=2,
        inspect=lambda _a0, _m0: SimpleNamespace(
            eligible=True, num_reqs=2, tokens_per_seq=1
        ),
    )
    dec_l8 = prog.decide(ctx_l8, fwd_l8)
    assert dec_l8 is not None
    assert dec_l8.eligible
    out_l8 = _assert_l8_outputs_and_states(
        prog,
        dec_l8,
        ctx_l8,
        fwd_l8,
        (
            mock_l8,
            kernel_mod,
            cfg_l8,
            prob_l8,
            pt_l8_np,
            ref_l8,
            page0_reg0_before,
            page0_reg1_before,
        ),
    )
    # and scale the output by -4.0.
    out_l8_orig = out_l8.detach().to(torch.float32).clone()
    prog.set_interpret_mode(interpret=True)
    prog.w_flat[-1] = prog.w_flat[-1] * -4.0
    step_fn_bad = prog.build(Bucket(1, 4))
    fwd_l8_bad = ForwardView(
        attn_metadata=md_l8,
        mrope_positions=None,
        steps={(1, 4): step_fn_bad},
        num_tokens=2,
        inspect=lambda _a0, _m0: SimpleNamespace(
            eligible=True, num_reqs=2, tokens_per_seq=1
        ),
    )
    bad_l8 = prog.forward(dec_l8, ctx_l8, fwd_l8_bad).detach().to(torch.float32)
    rmse_bad_l8 = float(torch.sqrt(torch.mean((bad_l8 - out_l8_orig) ** 2)).item())
    assert rmse_bad_l8 > 1.0, f"Negative control too weak: rmse_bad_l8={rmse_bad_l8}"


_TWO_RANK_SHIM = """
import torch.distributed as dist
import vllm_torchtpu.megakernel_runtime.driver as _drv
from vllm_torchtpu.megakernel_runtime.plugin import TpuMonoSpec
from vllm_torchtpu.models.qwen38_2_4t.megakernel import SPEC as _REAL_SPEC
from vllm_torchtpu.models.qwen38_2_4t.megakernel.adapter import (
    create,
    layout_request,
)
from vllm_torchtpu.models.qwen38_2_4t.megakernel.weights import load_tensor

_drv._local_tp_rank = lambda: int(dist.get_rank()) if dist.is_initialized() else 0
SPEC = TpuMonoSpec(
    _REAL_SPEC.name,
    _REAL_SPEC.version,
    create=create,
    decline_codes=_REAL_SPEC.decline_codes,
    layout_request=layout_request,
    load_tensor=load_tensor,
)
"""


def test_qwen38_two_ranks_uniform_agreement(tmp_path: Path) -> None:
    """Verify uniform decline agreement across two Gloo ranks when unsupported."""
    results = testkit.assert_rank_uniform_declines(
        tmp_path, sources=[_TWO_RANK_SHIM, _TWO_RANK_SHIM]
    )
    assert results == [
        {"type": "DeclinedDriver", "code": "unsupported_tp_size"},
        {"type": "DeclinedDriver", "code": "unsupported_tp_size"},
    ]


def test_qwen38_prepare_input_tensors_fail_closed_missing_embed() -> None:
    """Verify fail-closed RuntimeError when embed_tokens or metadata is missing."""
    adapter_mod = importlib.import_module(f"{_PKG}.adapter")
    shapes_mod = importlib.import_module(f"{_PKG}.shapes")
    cfg = shapes_mod.mini_parity_config(batch_size=2)
    ctx = SimpleNamespace(inputs_embeds=None, input_ids=torch.tensor([1, 2]))
    backbone_no_embed = SimpleNamespace(
        layers=[], norm=SimpleNamespace(weight=torch.ones(cfg.H))
    )
    with pytest.raises(
        RuntimeError,
        match="Model is missing callable embed_tokens required to embed input_ids",
    ):
        adapter_mod.prepare_input_tensors(
            ctx, (2, 2), cfg, torch.device("cpu"), backbone_no_embed
        )

    ctx_no_pos = SimpleNamespace(
        inputs_embeds=torch.zeros((2, cfg.H), dtype=torch.bfloat16),
        input_ids=None,
        positions=None,
    )
    with pytest.raises(
        RuntimeError,
        match="StepContext is missing required positions tensor",
    ):
        adapter_mod.prepare_input_tensors(
            ctx_no_pos, (2, 2), cfg, torch.device("cpu"), backbone_no_embed
        )

    with pytest.raises(
        RuntimeError,
        match="Missing block_table in attention metadata",
    ):
        adapter_mod.prepare_page_table(
            {}, ["model.layers.3.self_attn.attn"], (2, 2, 4), torch.device("cpu")
        )

    with pytest.raises(
        RuntimeError,
        match="Missing mamba_state_indices in attention metadata",
    ):
        adapter_mod.prepare_mamba_slots(
            {}, ["model.layers.0.linear_attn"], 2, 2, torch.device("cpu")
        )
