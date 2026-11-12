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
import datetime
import importlib
import inspect
import logging
import multiprocessing
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import jax
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from jax import export as jax_export
from jax._src.pallas.mosaic.interpret import interpret_pallas_call as ipc
from jax.sharding import AbstractMesh, NamedSharding
from jax.sharding import PartitionSpec as P

import vllm_torchtpu.megakernel.adapter.agreement as agreement_mod
import vllm_torchtpu.megakernel.adapter.driver as driver_mod
import vllm_torchtpu.megakernel.adapter.guards as guards_mod
import vllm_torchtpu.megakernel.adapter.plugin as plugin_mod
import vllm_torchtpu.megakernel.adapter.salt as salt_mod
import vllm_torchtpu.megakernel.adapter.stepping as stepping_mod
from tests.models.qwen38_2_4t import constants as test_constants
from tests.models.qwen38_2_4t import mock_model
from tests.models.qwen38_2_4t import oracle_fp64 as oracle
from tests.models.qwen38_2_4t.constants import (
    EXPECTED_BATCH_2,
    EXPECTED_BLOCK_SIZE,
    EXPECTED_BUCKET_4,
    EXPECTED_DCP_INTERLEAVE_SIZE,
    EXPECTED_L8_OUT_ROWS,
    EXPECTED_L8_REGIONS,
    EXPECTED_MAMBA_SLOT_TOKENS,
    EXPECTED_PAGE_SIZE,
    EXPECTED_PREFILL_TOKENS,
    EXPECTED_SALT_LEN,
    EXPECTED_TWO_RANKS,
    MAX_KV_ULP_TOLERANCE,
    MIN_CPU_DEVICES,
    MIN_TILED_DIM,
    PKG,
    PROD_NUM_LAYERS,
)
from tests.models.qwen38_2_4t.mock_model import StubLayerMetadata
from vllm_torchtpu.megakernel.adapter import config as config_mod
from vllm_torchtpu.megakernel.adapter import testing, testkit
from vllm_torchtpu.megakernel.adapter.hooks import StepContext
from vllm_torchtpu.megakernel.adapter.plugin import ForwardView, RunnerView
from vllm_torchtpu.megakernel.core.runtime.contract import (
    Bucket,
    Decline,
    LayoutDecline,
    StepDecision,
)
from vllm_torchtpu.megakernel.core.runtime.stats import BatchShape
from vllm_torchtpu.models.qwen38_2_4t.megakernel import (
    constants as qwen38_constants,
)
from vllm_torchtpu.models.qwen38_2_4t.megakernel import (
    kernel,
    plan,
    shapes,
)
from vllm_torchtpu.models.qwen38_2_4t.megakernel.adapter import (
    Qwen38AdapterError,
    Qwen38LayoutRequest,
    Qwen38MoeProgram,
    _exec_single_rank_step,
    _WarmupSession,
    create,
    layout_request,
    prepare_input_tensors,
    prepare_mamba_slots,
    prepare_page_table,
)
from vllm_torchtpu.models.qwen38_2_4t.megakernel.plan import Qwen38PlanError
from vllm_torchtpu.platforms import tpu_block_size_utils as block_utils_mod

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from tests.models.qwen38_2_4t.mock_model import MockRankModelHolder
    from tests.models.qwen38_2_4t.oracle_fp64 import SyntheticProblem
    from vllm_torchtpu.models.qwen38_2_4t.megakernel.shapes import Qwen38Config


@dataclasses.dataclass
class _StubBadScratchSpec:
    """Stub scratch spec with invalid dtype string for footprint error checks."""

    memory_space: object
    shape: tuple[int, ...]
    dtype: str


@dataclasses.dataclass(frozen=True)
class _L8Scenario:
    """Typed bundle of artifacts produced for an `L=8` multi-region run."""

    cfg: Qwen38Config
    prob: SyntheticProblem
    pt_np: np.ndarray
    ref_out: dict[str, np.ndarray]
    mock_model: MockRankModelHolder
    runner: SimpleNamespace
    prog: Qwen38MoeProgram
    dec: StepDecision
    ctx: StepContext
    fwd: ForwardView
    md: dict[str, mock_model.StubLayerMetadata]
    page0_reg0_before: torch.Tensor
    page0_reg1_before: torch.Tensor


class _Qwen38FakeAttentionBackend:
    """Deterministic attention backend for cache-slot floor derivation tests."""

    @staticmethod
    def get_name() -> str:
        """Return backend identifier."""
        return "QWEN38_TEST_BACKEND"

    @staticmethod
    def get_min_page_size(_vllm_config: object) -> int:
        """Return minimum supported page size in tokens."""
        return EXPECTED_PAGE_SIZE

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        """Return supported kernel block granularity."""
        return [EXPECTED_PAGE_SIZE]

    @staticmethod
    def get_kv_cache_page_size_bytes(
        block_size: int, *_args: object, **_kwargs: object
    ) -> int:
        """Return linear 1 KiB-per-token physical slot byte size."""
        bytes_per_token = 1024
        return block_size * bytes_per_token


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear `VLLM_TPU_MEGAKERNEL` across tests."""
    monkeypatch.delenv("VLLM_TPU_MEGAKERNEL", raising=False)


@pytest.fixture(autouse=True)
def _mock_uninitialized_distributed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub uninitialized vLLM TP rank getters to return rank 0 during unit tests."""
    with contextlib.suppress(ImportError, AttributeError):
        ps = importlib.import_module("vllm.distributed.parallel_state")
        if getattr(ps, "_TP", None) is None:
            monkeypatch.setattr(ps, "get_tensor_model_parallel_rank", lambda: 0)
            dist_mod = importlib.import_module("vllm.distributed")
            monkeypatch.setattr(dist_mod, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(driver_mod, "_get_pallas_mesh_fn", lambda: None)


class _PatchedWorkerEnv:
    """Temporarily set module attributes in a spawned child process and restore."""

    def __init__(self, patches: list[tuple[object, str, object]]) -> None:
        """Initialize with patch descriptors."""
        self._patches = patches
        self._missing = object()
        self._saved: list[tuple[object, str, object]] = []

    def __enter__(self) -> None:
        """Apply all attribute patches while recording prior states."""
        for target, attr, val in self._patches:
            self._saved.append((target, attr, getattr(target, attr, self._missing)))
            setattr(target, attr, val)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Restore all attribute patches in reverse order."""
        for target, attr, prev in reversed(self._saved):
            if prev is self._missing:
                delattr(target, attr)
            else:
                setattr(target, attr, prev)


def _patched_worker_env(
    patches: list[tuple[object, str, object]],
) -> _PatchedWorkerEnv:
    """Return a context manager to temporarily set attributes and restore."""
    return _PatchedWorkerEnv(patches)


def _worker_tp_patches() -> list[tuple[object, str, object]]:
    """Build restoration-tracked patches for uninitialized distributed child workers."""
    patches: list[tuple[object, str, object]] = [
        (driver_mod, "_get_pallas_mesh_fn", lambda: None),
    ]
    with contextlib.suppress(ImportError, AttributeError):
        ps = importlib.import_module("vllm.distributed.parallel_state")
        if getattr(ps, "_TP", None) is None:
            patches.append((ps, "get_tensor_model_parallel_rank", lambda: 0))
            dist_mod = importlib.import_module("vllm.distributed")
            patches.append((dist_mod, "get_tensor_model_parallel_rank", lambda: 0))
    patches.extend(
        [
            (guards_mod, "_dp_size", lambda _p_cfg: 1),
            (guards_mod, "_ep_group_size", lambda: 1),
        ]
    )
    return patches


def _vllm_cfg(
    num_layers: int = 92,
    hidden_size: int = 8192,
    *,
    megakernel: object = None,
    block_size: int = 256,
) -> SimpleNamespace:
    """Build mock vLLM configuration for plugin selection and hook tests."""
    hf_cfg = mock_model.qwen38_hf_config(
        num_hidden_layers=num_layers,
        hidden_size=hidden_size,
        quantization_config={"quant_method": "nvfp4", "group_size": 16},
    )
    resolved_mk = {"plugin": PKG} if megakernel is None else megakernel
    add_cfg: dict[str, object] = (
        {} if resolved_mk is None else {"megakernel": resolved_mk}
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=hf_cfg,
            hf_text_config=hf_cfg,
            architecture="Qwen3_5MoeForCausalLM",
            is_hybrid=False,
            enable_sleep_mode=False,
            multimodal_config=None,
            dtype=torch.bfloat16,
            get_num_kv_heads=lambda _p: 1,
            get_head_size=lambda: 256,
        ),
        cache_config=SimpleNamespace(
            block_size=block_size,
            cache_dtype=torch.bfloat16,
            enable_prefix_caching=False,
            user_specified_block_size=False,
            mamba_block_size=640,
            mamba_cache_mode="align",
            mamba_page_size_padded=0,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=32,
            decode_context_parallel_size=8,
            enable_expert_parallel=True,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        additional_config=add_cfg,
    )


def test_qwen38_spec_uses_domain_tp_without_monkeypatch() -> None:
    """SPEC uses DOMAIN_TP so `guards.check_spec` passes without SPEC mutation."""
    runner = testing.fake_runner(
        hf_config=mock_model.qwen38_hf_config(),
        tp_size=32,
        block_size=128,
        enable_expert_parallel=True,
    )
    runner.parallel_config.decode_context_parallel_size = 8
    runner.vllm_config.parallel_config.decode_context_parallel_size = 8
    spec = testkit.check_plugin_spec(PKG)
    assert spec.collective_domain == "tp"
    assert "pool_layout_unknown" in spec.decline_codes
    assert "pool_layout_mismatch" in spec.decline_codes
    assert (
        guards_mod.check_spec(runner, spec, view=RunnerView(runner, tp_rank=0)) is None
    )


def test_qwen38_spec_is_stdlib_only_and_shorthand_resolves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify plugin specification, explicit module selection, and scratch plan."""
    spec = testkit.check_plugin_spec(PKG)
    assert spec.name == "qwen38_2_4t"
    assert spec.version == "1"
    assert spec.create == ".adapter:create"
    assert spec.layout_request == ".adapter:layout_request"
    assert spec.collective_domain == "tp"
    assert spec.chips == ("TPU7x",)
    assert spec.parallelism == frozenset({"tp", "dcp", "ep_in_tp"})
    assert spec.features == frozenset()
    assert spec.numerics == "approx"

    vcfg_92 = _vllm_cfg(num_layers=92, hidden_size=8192)
    _, res_92 = config_mod.select_plugin(
        None, vcfg_92.additional_config, vcfg_92.model_config.hf_config
    )
    assert res_92 is not None
    assert res_92[0].module == PKG

    _, res_env = config_mod.select_plugin(
        f"plugin={PKG}", {}, vcfg_92.model_config.hf_config
    )
    assert res_env is not None
    assert res_env[0].module == PKG

    monkeypatch.setitem(
        plugin_mod.SHORTHANDS,
        "qwen35",
        plugin_mod.Shorthand(
            "vllm_torchtpu.models.qwen3_5.megakernel",
            ("Qwen3_5MoeForCausalLM",),
            ("qwen3_5_moe_text",),
        ),
    )
    vcfg_60 = _vllm_cfg(num_layers=60, hidden_size=4096, megakernel=True)
    _, res_60 = config_mod.select_plugin(
        None, vcfg_60.additional_config, vcfg_60.model_config.hf_config
    )
    assert res_60 is not None
    assert res_60[0].module == "vllm_torchtpu.models.qwen3_5.megakernel"

    r_prod = testing.fake_runner(
        model=mock_model.MockBackbone(
            layers=[], norm=mock_model.MockWeightHolder(weight=torch.ones(1))
        ),
        hf_config=mock_model.qwen38_hf_config(),
        tp_size=32,
        enable_expert_parallel=True,
    )
    r_prod.parallel_config.decode_context_parallel_size = 8
    res_prod = create(RunnerView(r_prod, tp_rank=0), {"buckets": "4"})
    assert isinstance(res_prod, Qwen38MoeProgram)
    assert int(res_prod.base_config.L) == PROD_NUM_LAYERS
    assert int(res_prod.base_config.B) == EXPECTED_BUCKET_4


@pytest.mark.parametrize(
    ("hf_cfg", "options", "expected_code"),
    [
        (
            SimpleNamespace(
                architectures=["Qwen3_5MoeForCausalLM"],
                model_type="qwen3_5_moe_text",
                quantization_config={"quant_method": "nvfp4"},
            ),
            {},
            "unsupported_geometry",
        ),
        (
            mock_model.qwen38_hf_config(num_attention_heads=64),
            {},
            "unsupported_geometry",
        ),
        (
            mock_model.qwen38_hf_config(num_hidden_layers=4, hidden_size=128),
            {},
            "unsupported_geometry",
        ),
        (
            mock_model.qwen38_hf_config(quantization_config=None),
            {},
            "unsupported_quant",
        ),
        (
            mock_model.qwen38_hf_config(
                quantization_config={"quant_method": "modelopt", "quant_algo": "FP8"}
            ),
            {},
            "unsupported_quant",
        ),
    ],
)
def test_qwen38_validate_hf_geometry_fails_closed(
    hf_cfg: object,
    options: dict[str, object],
    expected_code: str,
) -> None:
    """Decline fail-closed when HF geometry fields are omitted or mismatched."""
    runner = testing.fake_runner(hf_config=hf_cfg, tp_size=32)
    runner.parallel_config.decode_context_parallel_size = 8
    res = create(RunnerView(runner, tp_rank=0), options)
    assert isinstance(res, Decline)
    assert res.code == expected_code


def test_qwen38_stage_scratch_plan_and_live_layout_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify typed Qwen38LayoutRequest return and stage scratchpad contract."""
    vcfg = _vllm_cfg(block_size=256)
    req = layout_request(vcfg)
    assert isinstance(req, Qwen38LayoutRequest)
    assert req == Qwen38LayoutRequest(
        block_size=EXPECTED_BLOCK_SIZE,
        cp_kv_cache_interleave_size=EXPECTED_DCP_INTERLEAVE_SIZE,
        mamba_slot_tokens=EXPECTED_MAMBA_SLOT_TOKENS,
    )
    assert req.slot_tokens_floor == EXPECTED_MAMBA_SLOT_TOKENS

    for cfg in (shapes.mini_parity_config(batch_size=4), shapes.Qwen38Config(B=16)):
        stage_plan, slices = plan.build_stage_scratch_plan(cfg)
        assert set(slices) == {"wo_buf", "wout_buf", "wse2_buf"}
        assert stage_plan.peak_bytes == max(s.size_bytes for s in slices.values())
        assert stage_plan.peak_bytes < sum(s.size_bytes for s in slices.values())
        summary = plan.scratch_footprint(cfg)
        assert summary["stage_scratch_high_water_bytes"] == stage_plan.peak_bytes
        assert int(summary["stage_scratch_saved_bytes"]) > 0
        assert summary["fits_in_limit"] is True
        assert summary["fits_in_scalar_limit"] is True

    cfg = shapes.mini_parity_config(batch_size=4)
    assert "stage_scratch_u8" in plan.make_scratch_specs(cfg)
    monkeypatch.setattr(
        plan,
        "make_vmem_scratch_specs",
        lambda _cfg: {
            "bad_vmem": _StubBadScratchSpec(
                memory_space=pltpu.VMEM, shape=(4,), dtype="not_a_dtype"
            )
        },
    )
    with pytest.raises(Qwen38PlanError, match="Invalid scratch spec dtype"):
        plan.scratch_footprint(cfg)


def test_qwen38_layout_request_hook() -> None:
    """Verify layout request resolution and slot-floor sizing."""
    vcfg = _vllm_cfg(block_size=256)
    _, resolved = config_mod.select_plugin(
        None, vcfg.additional_config, vcfg.model_config.hf_config
    )
    assert resolved is not None
    ref, hook_opts = resolved
    plugin, module = config_mod.import_plugin(ref)
    assert plugin.layout_request is not None
    hook_fn = config_mod.resolve_callable(
        plugin.layout_request, module, what="layout_request"
    )
    req = hook_fn(vcfg, hook_opts)
    assert req == Qwen38LayoutRequest(
        block_size=EXPECTED_BLOCK_SIZE,
        cp_kv_cache_interleave_size=EXPECTED_DCP_INTERLEAVE_SIZE,
        mamba_slot_tokens=EXPECTED_MAMBA_SLOT_TOKENS,
    )
    salt = salt_mod.compile_cache_salt(vcfg)
    assert salt is not None
    assert len(salt) == EXPECTED_SALT_LEN

    vcfg.cache_config.block_size = req.block_size
    vcfg.parallel_config.cp_kv_cache_interleave_size = req.cp_kv_cache_interleave_size
    vcfg.parallel_config.dcp_kv_cache_interleave_size = req.cp_kv_cache_interleave_size

    cfg = shapes.mini_parity_config(batch_size=4)
    prob = oracle.make_synthetic_problem(cfg, seed=7)
    model_holder = mock_model.build_mock_vllm_qwen38_rank_models(prob, cfg)
    r_layout = testing.fake_runner(
        model=model_holder.model,
        hf_config=mock_model.qwen38_hf_config(
            num_hidden_layers=cfg.L, hidden_size=cfg.H
        ),
        tp_size=32,
        block_size=128,
        max_pages_per_seq=cfg.max_pages_per_seq,
        enable_expert_parallel=True,
    )
    r_layout.parallel_config.decode_context_parallel_size = 8
    prog_layout = create(RunnerView(r_layout, tp_rank=0), {"allow_mini_geometry": True})
    assert isinstance(prog_layout, Qwen38MoeProgram)
    with pytest.raises(LayoutDecline):
        prog_layout.bind(RunnerView(r_layout, tp_rank=0))

    block_utils_mod.update_tpu_block_size_and_slot_config(
        vcfg, _Qwen38FakeAttentionBackend
    )
    assert vcfg.cache_config.block_size == EXPECTED_BLOCK_SIZE
    vcfg.cache_config.block_size = req.slot_tokens_floor
    assert vcfg.cache_config.block_size == EXPECTED_MAMBA_SLOT_TOKENS
    r_layout.vllm_config = vcfg
    r_layout.cache_config = vcfg.cache_config
    bind_info = prog_layout.bind(RunnerView(r_layout, tp_rank=0))
    assert bind_info["num_pages"] == cfg.num_pages


@pytest.mark.parametrize(
    ("case_kind", "expected_code"),
    [
        ("bad_option", None),
        ("bad_tp", "unsupported_tp_size"),
        ("bad_quant", "unsupported_quant"),
        ("empty_buckets", "empty_buckets"),
    ],
)
def test_qwen38_create_declines_and_errors(
    case_kind: str,
    expected_code: str | None,
) -> None:
    """Verify `adapter.create` declines and raises cleanly on invalid configs."""

    if case_kind == "bad_option":
        runner = testing.fake_runner(tp_size=2)
        with pytest.raises(Qwen38AdapterError, match="unknown option"):
            create(RunnerView(runner, tp_rank=0), {"bad_option": True})
    elif case_kind == "bad_tp":
        runner = testing.fake_runner(tp_size=2)
        res = create(RunnerView(runner, tp_rank=0), {})
        assert isinstance(res, Decline)
        assert res.code == expected_code
    elif case_kind == "bad_quant":
        runner = testing.fake_runner(
            hf_config=mock_model.qwen38_hf_config(
                quantization_config={"quant_method": "fp8"}
            ),
            tp_size=32,
            enable_expert_parallel=True,
        )
        res = create(RunnerView(runner, tp_rank=0), {})
        assert isinstance(res, Decline)
        assert res.code == expected_code
    else:
        runner = testing.fake_runner(
            model=mock_model.MockBackbone(
                layers=[], norm=mock_model.MockWeightHolder(weight=torch.ones(1))
            ),
            hf_config=mock_model.qwen38_hf_config(),
            tp_size=32,
            enable_expert_parallel=True,
        )
        runner.parallel_config.decode_context_parallel_size = 8
        res = create(RunnerView(runner, tp_rank=0), {"buckets": []})
        assert isinstance(res, Decline)
        assert res.code == expected_code


def test_qwen38_decide_accepts_real_batch_shape_tokens_per_req() -> None:
    """Verify `Qwen38MoeProgram.decide` reads `BatchShape.tokens_per_req`."""
    cfg = shapes.mini_parity_config(batch_size=4)
    prob = oracle.make_synthetic_problem(cfg, seed=11)
    model_holder = mock_model.build_mock_vllm_qwen38_rank_models(prob, cfg)
    prog = Qwen38MoeProgram(
        model_holder.model, tp_rank=0, decode_buckets=(4,), base_config=cfg
    )
    prog.attn_layer_names = ["model.layers.3.self_attn.attn"]
    prog.mamba_layer_names = [f"model.layers.{idx}.linear_attn" for idx in (0, 1, 2)]
    bt_t = torch.zeros((2, cfg.max_pages_per_seq), dtype=torch.int32)
    ms_t = torch.zeros((2,), dtype=torch.int32)
    md: dict[str, mock_model.StubLayerMetadata] = {
        a_name: mock_model.StubLayerMetadata(block_tables=bt_t)
        for a_name in prog.attn_layer_names
    }
    for m_name in prog.mamba_layer_names:
        md[m_name] = mock_model.StubLayerMetadata(mamba_state_indices=ms_t)

    def _noop_step(*_args: object) -> torch.Tensor:
        """Return synthetic output tensor for routing eligibility tests."""
        return torch.zeros((4, cfg.H), dtype=torch.bfloat16)

    step_ctx = StepContext(
        input_ids=torch.arange(2, dtype=torch.int32),
        positions=torch.tensor([10, 20], dtype=torch.int32),
    )
    fwd_decode = ForwardView(
        attn_metadata=md,
        mrope_positions=None,
        steps={(1, 4): _noop_step},
        num_tokens=2,
        inspect=lambda _a0, _m0: BatchShape(
            eligible=True, num_reqs=2, tokens_per_req=1
        ),
    )
    dec = prog.decide(step_ctx, fwd_decode)
    assert isinstance(dec, StepDecision)
    assert dec.eligible is True
    assert dec.reason == "mk_t1_b4"
    assert dec.num_reqs == EXPECTED_L8_OUT_ROWS
    assert dec.tokens_per_seq == 1
    assert dec.bucket_t == EXPECTED_BUCKET_4

    fwd_prefill = ForwardView(
        attn_metadata=md,
        mrope_positions=None,
        steps={(1, 4): _noop_step},
        num_tokens=4,
        inspect=lambda _a0, _m0: BatchShape(
            eligible=True, num_reqs=2, tokens_per_req=2
        ),
    )
    dec_prefill = prog.decide(step_ctx, fwd_prefill)
    assert isinstance(dec_prefill, StepDecision)
    assert dec_prefill.eligible is False
    assert dec_prefill.reason == "stock_prefill"
    assert dec_prefill.tokens_per_seq == EXPECTED_PREFILL_TOKENS


def test_qwen38_prepare_input_tensors_fail_closed_missing_embed() -> None:
    """Verify fail-closed RuntimeError when embed_tokens or metadata is missing."""
    cfg = shapes.mini_parity_config(batch_size=2)
    backbone = mock_model.MockBackbone(
        layers=[], norm=mock_model.MockWeightHolder(weight=torch.ones(cfg.H))
    )
    ctx_no_embed = StepContext(
        input_ids=torch.tensor([1, 2], dtype=torch.int32),
        positions=torch.tensor([10, 20], dtype=torch.int32),
        inputs_embeds=None,
    )
    with pytest.raises(
        RuntimeError,
        match="Model is missing callable embed_tokens required to embed input_ids",
    ):
        prepare_input_tensors(ctx_no_embed, (2, 2), cfg, torch.device("cpu"), backbone)

    ctx_no_pos = StepContext(
        input_ids=None,
        positions=cast("torch.Tensor", None),
        inputs_embeds=torch.zeros((2, cfg.H), dtype=torch.bfloat16),
    )
    with pytest.raises(
        RuntimeError, match="StepContext is missing required positions tensor"
    ):
        prepare_input_tensors(ctx_no_pos, (2, 2), cfg, torch.device("cpu"), backbone)

    with pytest.raises(
        RuntimeError, match="Missing block_tables in attention metadata"
    ):
        prepare_page_table(
            {}, ["model.layers.3.self_attn.attn"], (2, 2, 4), torch.device("cpu")
        )
    with pytest.raises(
        RuntimeError, match="Missing mamba_state_indices in attention metadata"
    ):
        prepare_mamba_slots(
            {}, ["model.layers.0.linear_attn"], 2, 2, torch.device("cpu")
        )


def _verify_kv_pool_updates(
    got_out: Mapping[str, jax.Array],
    ref_out: Mapping[str, np.ndarray],
    cfg: Qwen38Config,
    prob: SyntheticProblem,
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
    assert max_kv_ulp <= MAX_KV_ULP_TOLERANCE, (
        f"max_kv_ulp={max_kv_ulp} > {MAX_KV_ULP_TOLERANCE}"
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
    got_out: Mapping[str, jax.Array],
    ref_out: Mapping[str, np.ndarray],
    cfg: Qwen38Config,
    prob: SyntheticProblem,
) -> tuple[np.ndarray, float]:
    """Verify RMSE bounds and KV cache updates against FP64 reference."""
    got_x = np.asarray(
        jax.device_get(got_out["x"]).astype(jnp.float32), dtype=np.float64
    )
    rmse_x = float(np.sqrt(np.mean((got_x - ref_out["x"]) ** 2)))
    bound_x = oracle.layer_bf16_error_bound(
        float(np.sqrt(np.mean(ref_out["x"] ** 2))), 4, cfg.L
    )
    assert rmse_x <= bound_x, f"rmse_x={rmse_x} > bound_x={bound_x}"

    got_conv = np.asarray(
        jax.device_get(got_out["conv"]).astype(jnp.float32), dtype=np.float64
    )
    rmse_conv = float(np.sqrt(np.mean((got_conv - ref_out["conv"]) ** 2)))
    bound_conv = oracle.layer_bf16_error_bound(
        float(np.sqrt(np.mean(ref_out["conv"] ** 2))), 1, 1
    )
    assert rmse_conv <= bound_conv, f"rmse_conv={rmse_conv} > {bound_conv}"

    got_rec = np.asarray(jax.device_get(got_out["rec"]), dtype=np.float64)
    rmse_rec = float(np.sqrt(np.mean((got_rec - ref_out["rec"]) ** 2)))
    bound_rec = oracle.layer_bf16_error_bound(
        float(np.sqrt(np.mean(ref_out["rec"] ** 2))), 1, 1
    )
    assert rmse_rec <= bound_rec, f"rmse_rec={rmse_rec} > {bound_rec}"

    _verify_kv_pool_updates(got_out, ref_out, cfg, prob)
    return got_x, bound_x


def _verify_negative_controls(
    prob: SyntheticProblem,
    cfg: Qwen38Config,
    got_x: np.ndarray,
    bound_x: float,
) -> None:
    """Verify zeroing routed or shared experts exceeds error bound by > 5x."""
    for key in ("w_e2", "w_se2"):
        mutated: dict[str, object] = dict(prob)
        w_mut = dict(prob["weights"])
        w_mut[key] = jnp.zeros_like(prob["weights"][key])
        mutated["weights"] = w_mut
        ref_mut = oracle.run_fp64_reference(mutated, cfg, use_forced_routing=False)
        rmse_mut = float(np.sqrt(np.mean((got_x[0] - ref_mut["x"]) ** 2)))
        assert rmse_mut > 5.0 * bound_x, (
            f"{key} sensitivity too weak: rmse={rmse_mut} <= 5*{bound_x}"
        )


def _verify_program_driver_e2e(
    prob: SyntheticProblem,
    cfg: Qwen38Config,
    ref_out: Mapping[str, np.ndarray],
) -> None:
    """Verify ProgramDriver E2E parity directly with production `SPEC` (`DOMAIN_TP`)."""
    model_holder = mock_model.build_mock_vllm_qwen38_rank_models(prob, cfg)
    qsl = np.zeros(16, dtype=np.int32)
    qsl[: cfg.B + 1] = np.arange(cfg.B + 1, dtype=np.int32)
    runner = testing.fake_runner(
        model=model_holder,
        hf_config=mock_model.qwen38_hf_config(
            num_hidden_layers=cfg.L, hidden_size=cfg.H
        ),
        tp_size=32,
        block_size=640,
        token_paddings=(cfg.B,),
        req_block_pairs=((cfg.B, cfg.max_pages_per_seq),),
        max_pages_per_seq=cfg.max_pages_per_seq,
        enable_expert_parallel=True,
        _dcp_enabled=True,
        _attn_metadata_builder_ctx=mock_model.StubBuilderCtx(num_reqs=cfg.B),
        query_start_loc_np=qsl,
        _cached_comb_key=None,
        _combined_request_distribution_cpu=None,
    )
    runner.parallel_config.decode_context_parallel_size = 8
    runner.vllm_config.parallel_config.decode_context_parallel_size = 8

    pt_np = np.asarray(jax.device_get(prob["page_table"]), dtype=np.int32)
    bt_t = torch.from_numpy(pt_np[0, 0].copy()).to(torch.int32)
    mamba_slots = prob["mamba_slots"]

    def _build_md(prog: Qwen38MoeProgram) -> dict[str, mock_model.StubLayerMetadata]:
        """Construct attention metadata for mock driver."""
        md = {prog.attn_layer_names[0]: mock_model.StubLayerMetadata(block_tables=bt_t)}
        for g_idx, m_name in enumerate(prog.mamba_layer_names):
            md[m_name] = mock_model.StubLayerMetadata(
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
        PKG,
        runner,
        [ctx_step],
        metadata=_build_md,
        options={"buckets": str(cfg.B), "allow_mini_geometry": True, "interpret": True},
    )
    assert e2e.hits == [True]
    assert e2e.step_mix == {f"mk_t1_b{cfg.B}": 1}
    got_prog_x = e2e.outputs[0].detach().to(torch.float32).numpy().astype(np.float64)
    rmse_prog = float(np.sqrt(np.mean((got_prog_x - ref_out["x"]) ** 2)))
    bound_prog = oracle.layer_bf16_error_bound(
        float(np.sqrt(np.mean(ref_out["x"] ** 2))), 4, cfg.L
    )
    assert rmse_prog <= bound_prog, f"rmse_prog={rmse_prog} > {bound_prog}"


def _execute_qwen38_cpu_e2e_test() -> None:
    """Run full 32-device parity, negative controls, and ProgramDriver E2E check."""
    assert len(jax.devices()) >= MIN_CPU_DEVICES
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
    for k in ("wqkv", "w_inproj", "w_se13"):
        assert norm_w[k].shape[2] >= MIN_TILED_DIM
    for k in ("w_e13", "w_e2"):
        assert norm_w[k].shape[3] >= MIN_TILED_DIM

    ref_out = oracle.run_fp64_reference(prob, cfg, use_forced_routing=False)
    got_out = kernel.run_qwen38_dcp8_step(
        prob, cfg, use_forced_routing=False, return_kv_updates=True, interpret=True
    )
    got_x, bound_x = _verify_output_tensors(got_out, ref_out, cfg, prob)
    _verify_negative_controls(prob, cfg, got_x, bound_x)
    _verify_program_driver_e2e(prob, cfg, ref_out)


def _run_qwen38_cpu_parity_worker(
    err_queue: multiprocessing.Queue[BaseException | None],
) -> None:
    """Worker function in spawned 32-device process with exception propagation."""
    with _patched_worker_env(_worker_tp_patches()):
        try:
            _execute_qwen38_cpu_e2e_test()
            err_queue.put(None)
        except (
            AssertionError,
            RuntimeError,
            ValueError,
            TypeError,
            OSError,
            KeyError,
            ArithmeticError,
            AttributeError,
            ImportError,
        ) as exc:
            err_queue.put(exc)


@pytest.mark.nightly
def test_qwen38_cpu_e2e_parity_and_zero_weight_negative_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Qwen3.8 CPU 32-device parity, KV updates, and ProgramDriver E2E."""
    if len(jax.devices()) < MIN_CPU_DEVICES:
        monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
        monkeypatch.setenv("VLLM_PLUGINS", "torchtpu")
        monkeypatch.setenv("JAX_PLATFORMS", "cpu")
        monkeypatch.setenv("PJRT_DEVICE", "CPU")
        monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
        monkeypatch.setenv(
            "XLA_FLAGS",
            "--xla_force_host_platform_device_count=32 "
            "--xla_cpu_multi_thread_eigen=true "
            "intra_op_parallelism_threads=1",
        )
        ctx = multiprocessing.get_context("spawn")
        err_queue: multiprocessing.Queue[BaseException | None] = ctx.Queue()
        proc = ctx.Process(target=_run_qwen38_cpu_parity_worker, args=(err_queue,))
        proc.start()
        try:
            proc.join(timeout=600)
            if proc.is_alive():
                pytest.fail("Child process timed out after 600s")
        finally:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
        err = err_queue.get() if not err_queue.empty() else None
        if err is not None:
            raise err
        assert proc.exitcode == 0, (
            f"Child process failed with exit code {proc.exitcode}"
        )
        return

    with _patched_worker_env(_worker_tp_patches()):
        _execute_qwen38_cpu_e2e_test()


def _verify_l8_zero_copy_bind(
    model_l8: MockRankModelHolder,
    runner: SimpleNamespace,
    prog: Qwen38MoeProgram,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bind `prog` and verify zero-copy `data_ptr()` preservation across regions."""
    layer3_sa = model_l8.model.layers[3].self_attn
    layer7_sa = model_l8.model.layers[7].self_attn
    layer0_la = model_l8.model.layers[0].linear_attn
    layer1_la = model_l8.model.layers[1].linear_attn
    assert layer3_sa is not None
    assert layer7_sa is not None
    assert layer0_la is not None
    assert layer1_la is not None
    reg0_ptr = layer3_sa.attn.kv_cache.data_ptr()
    reg1_ptr = layer7_sa.attn.kv_cache.data_ptr()
    assert layer0_la.kv_cache.data_ptr() == reg0_ptr
    assert layer1_la.kv_cache.data_ptr() == reg1_ptr

    prog.load(None)
    bind_res = prog.bind(RunnerView(runner, tp_rank=0))
    assert prog.rank_pools is None
    assert prog.gdn_region_map == (0, 1, 0, 1, 0, 1)
    assert bind_res["num_regions"] == EXPECTED_L8_REGIONS
    assert len(prog.region_pools) == EXPECTED_L8_REGIONS
    assert prog.region_pools[0].data_ptr() == reg0_ptr
    assert prog.region_pools[1].data_ptr() == reg1_ptr
    assert layer3_sa.attn.kv_cache.data_ptr() == reg0_ptr
    assert layer7_sa.attn.kv_cache.data_ptr() == reg1_ptr
    return (
        layer3_sa.attn.kv_cache[0].view(torch.uint8).clone(),
        layer7_sa.attn.kv_cache[0].view(torch.uint8).clone(),
    )


def _build_l8_scenario(*, interpret: bool | pltpu.InterpretParams) -> _L8Scenario:
    """Construct a 2-region `L=8` single-backbone scenario ready for forward()."""
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
    prob_l8 = oracle.make_synthetic_problem(cfg_l8, seed=99, pos=(130, 260, 130, 260))
    pool_f32 = np.asarray(jax.device_get(prob_l8["pool"]).astype(jnp.float32))
    prob_l8["pool"] = jnp.asarray(
        np.stack([pool_f32, pool_f32 * 0.7 + 0.05], axis=1), dtype=jnp.float8_e4m3fn
    )
    pt_l8_np = np.asarray(jax.device_get(prob_l8["page_table"]), dtype=np.int32).copy()
    pt_l8_np[:, 1] = pt_l8_np[:, 0]
    prob_l8["page_table"] = jnp.asarray(pt_l8_np, dtype=jnp.int32)
    for g_idx in range(cfg_l8.NGDN):
        grp = g_idx // cfg_l8.NATT
        prob_l8["mamba_slots"][g_idx, 0] = 4 + 2 * grp
        prob_l8["mamba_slots"][g_idx, 1] = 5 + 2 * grp
    prob_l8["mamba_slots"][:, 2:] = -1
    ref_l8 = oracle.run_fp64_reference(prob_l8, cfg_l8, use_forced_routing=False)
    model_l8 = mock_model.build_mock_vllm_qwen38_rank_models(
        prob_l8, cfg_l8, gdn_region_mode="strided"
    )

    qsl_pad = np.zeros(16, dtype=np.int32)
    qsl_pad[:3] = np.arange(3, dtype=np.int32)
    runner = testing.fake_runner(
        model=model_l8.model,
        hf_config=mock_model.qwen38_hf_config(
            num_hidden_layers=cfg_l8.L, hidden_size=cfg_l8.H
        ),
        tp_size=32,
        block_size=640,
        token_paddings=(4,),
        req_block_pairs=((2, cfg_l8.max_pages_per_seq), (4, cfg_l8.max_pages_per_seq)),
        max_pages_per_seq=cfg_l8.max_pages_per_seq,
        enable_expert_parallel=True,
        _dcp_enabled=True,
        _attn_metadata_builder_ctx=mock_model.StubBuilderCtx(num_reqs=2),
        query_start_loc_np=qsl_pad,
    )
    runner.parallel_config.decode_context_parallel_size = 8
    runner.vllm_config.parallel_config.decode_context_parallel_size = 8

    prog = Qwen38MoeProgram(
        model_l8.model,
        tp_rank=0,
        decode_buckets=(4,),
        base_config=cfg_l8,
        interpret=interpret,
    )
    page0_reg0_before, page0_reg1_before = _verify_l8_zero_copy_bind(
        model_l8, runner, prog
    )

    step_fn = prog.build(Bucket(1, 4))
    bt_l8_t = torch.from_numpy(pt_l8_np[0, 0, :2].copy()).to(torch.int32)
    md: dict[str, mock_model.StubLayerMetadata] = {
        a_name: mock_model.StubLayerMetadata(block_tables=bt_l8_t)
        for a_name in prog.attn_layer_names
    }
    for g_idx, m_name in enumerate(prog.mamba_layer_names):
        grp = g_idx // cfg_l8.NATT
        ms_t = torch.tensor([4 + 2 * grp, 5 + 2 * grp], dtype=torch.int32)
        md[m_name] = mock_model.StubLayerMetadata(mamba_state_indices=ms_t)

    ctx_l8 = StepContext(
        input_ids=torch.arange(2, dtype=torch.int32),
        positions=torch.tensor([130, 260], dtype=torch.int32),
    )
    fwd_l8 = ForwardView(
        attn_metadata=md,
        mrope_positions=None,
        steps={(1, 4): step_fn},
        num_tokens=2,
        inspect=lambda _a0, _m0: BatchShape(
            eligible=True, num_reqs=2, tokens_per_req=1
        ),
    )
    dec_l8 = prog.decide(ctx_l8, fwd_l8)
    assert dec_l8 is not None
    assert dec_l8.eligible
    return _L8Scenario(
        cfg=cfg_l8,
        prob=prob_l8,
        pt_np=pt_l8_np,
        ref_out=ref_l8,
        mock_model=model_l8,
        runner=runner,
        prog=prog,
        dec=dec_l8,
        ctx=ctx_l8,
        fwd=fwd_l8,
        md=md,
        page0_reg0_before=page0_reg0_before,
        page0_reg1_before=page0_reg1_before,
    )


@pytest.mark.nightly
def test_qwen38_multi_region_l8_and_single_backbone() -> None:
    """Verify 2-region L=8 KV pools, GDN state parity, and race-free interpret."""
    sc = _build_l8_scenario(
        interpret=pltpu.InterpretParams(
            detect_races=True,
            out_of_bounds_reads="raise",
            uninitialized_memory="nan",
        )
    )
    out_l8 = sc.prog.forward(sc.dec, sc.ctx, sc.fwd)
    assert ipc.races is not None, "Interpret race detection did not run"
    assert not ipc.races.races_found, f"Interpret race detected: {ipc.races}"
    assert out_l8.shape == (EXPECTED_L8_OUT_ROWS, sc.cfg.H)

    got_x = out_l8.detach().to(torch.float32).numpy().astype(np.float64)
    ref_x = sc.ref_out["x"][:2]
    rmse_x = float(np.sqrt(np.mean((got_x - ref_x) ** 2)))
    bound_x = oracle.layer_bf16_error_bound(
        float(np.sqrt(np.mean(ref_x**2))), 3, sc.cfg.L
    )
    assert rmse_x <= bound_x, f"rmse_x={rmse_x} > {bound_x}"

    assert sc.ctx.positions is not None
    for att_idx in range(sc.cfg.NATT):
        reg_pool_got = jnp.asarray(
            sc.prog.region_pools[att_idx].view(torch.uint8).numpy()
        ).view(jnp.float8_e4m3fn)
        for b in range(EXPECTED_BATCH_2):
            pb = int(sc.ctx.positions[b].item())
            lp = pb // sc.cfg.page_size
            off = pb % sc.cfg.page_size
            p_idx = int(sc.pt_np[0, att_idx, b, lp])
            ulp_k = oracle.max_fp8_ulp_diff(
                reg_pool_got[p_idx, 0, :, :, off],
                sc.ref_out["pool"][0, att_idx, p_idx, 0, :, :, off],
            )
            ulp_v = oracle.max_fp8_ulp_diff(
                reg_pool_got[p_idx, 1, :, :, off],
                sc.ref_out["pool"][0, att_idx, p_idx, 1, :, :, off],
            )
            assert max(ulp_k, ulp_v) <= MAX_KV_ULP_TOLERANCE

    layer3_sa = sc.mock_model.model.layers[3].self_attn
    layer7_sa = sc.mock_model.model.layers[7].self_attn
    assert layer3_sa is not None
    assert layer7_sa is not None
    assert layer3_sa.attn.kv_cache.data_ptr() == sc.prog.region_pools[0].data_ptr()
    assert layer7_sa.attn.kv_cache.data_ptr() == sc.prog.region_pools[1].data_ptr()
    got_conv, got_rec = kernel.unpack_gdn_states_from_pools(
        [jnp.asarray(p.view(torch.uint8).numpy()) for p in sc.prog.region_pools],
        jnp.asarray(sc.prob["mamba_slots"], dtype=jnp.int32),
        sc.cfg,
        gdn_region_map=sc.prog.gdn_region_map,
    )
    for got_arr, ref_arr, label in (
        (
            np.asarray(
                jax.device_get(got_conv[:, :2]).astype(jnp.float32), dtype=np.float64
            ),
            sc.ref_out["conv"][0, :, :2],
            "conv",
        ),
        (
            np.asarray(jax.device_get(got_rec[:, :2]), dtype=np.float64),
            sc.ref_out["rec"][0, :, :2],
            "rec",
        ),
    ):
        rmse = float(np.sqrt(np.mean((got_arr - ref_arr) ** 2)))
        bound = oracle.layer_bf16_error_bound(
            float(np.sqrt(np.mean(ref_arr**2))), 2, sc.cfg.L
        )
        assert rmse <= bound, f"rmse_{label}={rmse} > {bound}"

    layer3_sa = sc.mock_model.model.layers[3].self_attn
    layer7_sa = sc.mock_model.model.layers[7].self_attn
    assert layer3_sa is not None
    assert layer7_sa is not None
    assert torch.equal(
        layer3_sa.attn.kv_cache[0].view(torch.uint8), sc.page0_reg0_before
    )
    assert torch.equal(
        layer7_sa.attn.kv_cache[0].view(torch.uint8), sc.page0_reg1_before
    )
    assert not torch.equal(
        layer3_sa.attn.kv_cache[1].view(torch.uint8),
        layer7_sa.attn.kv_cache[1].view(torch.uint8),
    )


@pytest.mark.nightly
def test_qwen38_l8_weight_mutation_negative_control() -> None:
    """Verify `_sync_jax_weights` invalidates cache when `final_norm` changes."""
    sc = _build_l8_scenario(interpret=True)
    base_out = (
        sc.prog.forward(sc.dec, sc.ctx, sc.fwd).detach().to(torch.float32).clone()
    )
    sc.prog.w_flat[-1] = sc.prog.w_flat[-1] * -4.0
    step_fn_bad = sc.prog.build(Bucket(1, 4))
    fwd_bad = ForwardView(
        attn_metadata=sc.md,
        mrope_positions=None,
        steps={(1, 4): step_fn_bad},
        num_tokens=2,
        inspect=lambda _a0, _m0: BatchShape(
            eligible=True, num_reqs=2, tokens_per_req=1
        ),
    )
    bad_out = sc.prog.forward(sc.dec, sc.ctx, fwd_bad).detach().to(torch.float32)
    rmse_bad = float(torch.sqrt(torch.mean((bad_out - base_out) ** 2)).item())
    assert rmse_bad > 1.0, f"Negative control too weak: rmse_bad={rmse_bad}"


def _run_qwen38_two_rank_worker(
    rank: int,
    world_size: int,
    init_method: str,
    res_queue: multiprocessing.Queue[tuple[int, dict[str, object]]],
) -> None:
    """Execute one Gloo rank for uniform startup decline agreement."""
    patches = _worker_tp_patches()
    for mod in (driver_mod, agreement_mod, stepping_mod):
        lazy = getattr(mod, "logger", None)
        if lazy is not None:
            patches.append((lazy, "_logger", logging.getLogger(mod.__name__)))
    with _patched_worker_env(patches):
        dist.init_process_group(
            "gloo",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=60),
        )
        try:
            with _patched_worker_env(
                [
                    (agreement_mod, "_tp_cpu_group", lambda: dist.group.WORLD),
                    (driver_mod, "_local_tp_rank", lambda: rank),
                ]
            ):
                runner = testing.fake_runner(tp_size=2, megakernel={"plugin": PKG})
                drv = driver_mod.ProgramDriver.maybe_create(runner)
                payload: dict[str, object] = {
                    "type": type(drv).__name__,
                    "code": (
                        drv.decline.code
                        if isinstance(drv, agreement_mod.DeclinedDriver)
                        else None
                    ),
                }
                res_queue.put((rank, payload))
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


@pytest.mark.nightly
def test_qwen38_two_ranks_uniform_agreement(tmp_path: Path) -> None:
    """Verify uniform decline agreement across two Gloo ranks without SPEC overrides."""
    init_method = f"file://{tmp_path / 'pg_init'}"
    ctx = multiprocessing.get_context("spawn")
    res_queue: multiprocessing.Queue[tuple[int, dict[str, object]]] = ctx.Queue()
    mp.spawn(
        _run_qwen38_two_rank_worker,
        args=(EXPECTED_TWO_RANKS, init_method, res_queue),
        nprocs=EXPECTED_TWO_RANKS,
        join=True,
    )
    results = dict(res_queue.get(timeout=5) for _ in range(EXPECTED_TWO_RANKS))
    assert results == {
        0: {"type": "DeclinedDriver", "code": "unsupported_tp_size"},
        1: {"type": "DeclinedDriver", "code": "unsupported_tp_size"},
    }


def test_qwen38_constants_and_producer_block_tables_contract() -> None:
    """Validate qwen38 constants.py, block_tables contract, and _WarmupSession."""
    assert hasattr(qwen38_constants, "PROD_NUM_LAYERS")
    assert hasattr(qwen38_constants, "NVFP4_GROUP_SIZE")
    assert hasattr(qwen38_constants, "H_TILED_4D_KEYS")
    assert hasattr(qwen38_constants, "WEIGHT_KEYS")
    assert hasattr(qwen38_constants, "PAGE_TOKENS")

    assert hasattr(test_constants, "GLOBAL_POOL_NDIM")

    assert hasattr(StubLayerMetadata, "block_tables")
    assert not hasattr(StubLayerMetadata, "block_table")

    assert not inspect.isgeneratorfunction(_WarmupSession)


def _assert_unnamed_layer_declines(
    prob_e2e: SyntheticProblem,
    cfg_fast: Qwen38Config,
    qsl_fast: np.ndarray,
) -> None:
    """Verify `bind` raises `LayoutDecline` when a layer lacks explicit naming."""
    unnamed_model = mock_model.build_mock_vllm_qwen38_rank_models(prob_e2e, cfg_fast)
    unnamed_sa = unnamed_model.model.layers[3].self_attn
    assert unnamed_sa is not None
    unnamed_sa.prefix = ""
    unnamed_sa.attn.layer_name = ""
    bad_runner = testing.fake_runner(
        model=unnamed_model.model,
        hf_config=mock_model.qwen38_hf_config(
            num_hidden_layers=cfg_fast.L, hidden_size=cfg_fast.H
        ),
        tp_size=32,
        block_size=640,
        token_paddings=(2,),
        req_block_pairs=((2, cfg_fast.max_pages_per_seq),),
        max_pages_per_seq=cfg_fast.max_pages_per_seq,
        enable_expert_parallel=True,
        _dcp_enabled=True,
        _attn_metadata_builder_ctx=mock_model.StubBuilderCtx(num_reqs=2),
        query_start_loc_np=qsl_fast,
    )
    bad_prog = Qwen38MoeProgram(
        unnamed_model.model,
        tp_rank=0,
        decode_buckets=(2,),
        base_config=cfg_fast,
        interpret=True,
    )
    bad_prog.load(None)
    with pytest.raises(LayoutDecline):
        bad_prog.bind(RunnerView(bad_runner, tp_rank=0))


def test_qwen38_pr_fast_single_backbone_forward_parity() -> None:
    """Fast per-PR CPU forward parity, offline VMEM query, and `-1` DMA bounds."""
    expected_v7x_vmem_bytes = 62 * 1024 * 1024
    assert plan.vmem_limit_bytes() == expected_v7x_vmem_bytes

    cfg_fast = dataclasses.replace(
        shapes.mini_parity_config(batch_size=2),
        R=1,
        DCP=1,
        hier_group=1,
        Q_H=4,
        KV_H=1,
        KH=1,
        VH=2,
        L=4,
        E=4,
        K=2,
        h_tile=64,
    )
    prob = oracle.make_synthetic_problem(cfg_fast, seed=21, pos=(5, 0))
    pt_np = np.asarray(jax.device_get(prob["page_table"]), dtype=np.int32).copy()
    pt_np[0, 0, 0, 0] = 2
    pt_np[0, 0, 1, :] = -1
    prob["page_table"] = jnp.asarray(pt_np, dtype=jnp.int32)
    forced_np = np.asarray(
        jax.device_get(prob["forced_top_ids"]), dtype=np.int32
    ).copy()
    forced_np[:, 1, 1] = -1
    prob["forced_top_ids"] = jnp.asarray(forced_np, dtype=jnp.int32)

    pool_page0_before = np.asarray(
        jax.device_get(prob["pool"][0, 0]).view(jnp.uint8)
    ).copy()
    step_out = kernel.run_qwen38_dcp8_step(
        prob,
        cfg_fast,
        use_forced_routing=True,
        return_kv_updates=True,
        interpret=True,
    )
    pool_page0_after = np.asarray(
        jax.device_get(step_out["pool"][0, 0]).view(jnp.uint8)
    )
    np.testing.assert_array_equal(pool_page0_after, pool_page0_before)

    prob_e2e = oracle.make_synthetic_problem(cfg_fast, seed=22, pos=(5, 9))
    ref_e2e = oracle.run_fp64_reference(prob_e2e, cfg_fast, use_forced_routing=False)
    mock_fast = mock_model.build_mock_vllm_qwen38_rank_models(prob_e2e, cfg_fast)
    qsl_fast = np.array([0, 1, 2, 0], dtype=np.int32)
    _assert_unnamed_layer_declines(prob_e2e, cfg_fast, qsl_fast)

    runner = testing.fake_runner(
        model=mock_fast.model,
        hf_config=mock_model.qwen38_hf_config(
            num_hidden_layers=cfg_fast.L, hidden_size=cfg_fast.H
        ),
        tp_size=32,
        block_size=640,
        token_paddings=(2,),
        req_block_pairs=((2, cfg_fast.max_pages_per_seq),),
        max_pages_per_seq=cfg_fast.max_pages_per_seq,
        enable_expert_parallel=True,
        _dcp_enabled=True,
        _attn_metadata_builder_ctx=mock_model.StubBuilderCtx(num_reqs=2),
        query_start_loc_np=qsl_fast,
    )
    fast_sa = mock_fast.model.layers[3].self_attn
    assert fast_sa is not None
    ptr_before = fast_sa.attn.kv_cache.data_ptr()
    prog = Qwen38MoeProgram(
        mock_fast.model,
        tp_rank=0,
        decode_buckets=(2,),
        base_config=cfg_fast,
        interpret=True,
    )
    prog.load(None)
    prog.bind(RunnerView(runner, tp_rank=0))
    assert prog.region_pools[0].data_ptr() == ptr_before
    assert fast_sa.attn.kv_cache.data_ptr() == ptr_before
    step_fn = prog.build(Bucket(1, 2))
    pt_e2e_np = np.asarray(jax.device_get(prob_e2e["page_table"]), dtype=np.int32)
    bt_t = torch.from_numpy(pt_e2e_np[0, 0].copy()).to(torch.int32)
    md_fast: dict[str, mock_model.StubLayerMetadata] = {
        a_name: mock_model.StubLayerMetadata(block_tables=bt_t)
        for a_name in prog.attn_layer_names
    }
    for g_idx, m_name in enumerate(prog.mamba_layer_names):
        ms_t = torch.from_numpy(prob_e2e["mamba_slots"][g_idx].copy()).to(torch.int32)
        md_fast[m_name] = mock_model.StubLayerMetadata(mamba_state_indices=ms_t)
    ctx_fast = StepContext(
        input_ids=torch.arange(2, dtype=torch.int32),
        positions=torch.tensor([5, 9], dtype=torch.int32),
    )
    fwd_fast = ForwardView(
        attn_metadata=md_fast,
        mrope_positions=None,
        steps={(1, 2): step_fn},
        num_tokens=2,
        inspect=lambda _a0, _m0: BatchShape(
            eligible=True, num_reqs=2, tokens_per_req=1
        ),
    )
    dec_fast = prog.decide(ctx_fast, fwd_fast)
    assert dec_fast.eligible
    out_fast = prog.forward(dec_fast, ctx_fast, fwd_fast)
    assert fast_sa.attn.kv_cache.data_ptr() == ptr_before
    got_x = out_fast.detach().to(torch.float32).numpy().astype(np.float64)
    ref_x = ref_e2e["x"]
    rmse_x = float(np.sqrt(np.mean((got_x - ref_x) ** 2)))
    bound_x = oracle.layer_bf16_error_bound(
        float(np.sqrt(np.mean(ref_x**2))), 3, cfg_fast.L
    )
    assert rmse_x <= bound_x, f"rmse_x={rmse_x} > {bound_x}"


@dataclasses.dataclass(frozen=True, slots=True)
class _AbstractTpu7xDevice:
    """Hashable abstract TPU7x device descriptor for offline Mosaic lowering."""

    device_kind: str = "TPU7x"
    num_cores: int = 1
    platform: str = "tpu"

    def _repr(self) -> str:
        """Return compact device kind label for JAX AbstractMesh formatting."""
        return self.device_kind


def test_qwen38_tpu7x_offline_mosaic_lowering_and_no_all_gather() -> None:
    """Verify compiled (`interpret=False`) TPU7x Mosaic lowering.

    Covers both R=1 and TP32xDCP8 configurations.
    """
    cfg_32 = shapes.mini_parity_config(batch_size=8)
    prob_32 = oracle.make_synthetic_problem(cfg_32, seed=1, pos=tuple(range(8)))
    norm_w32 = plan.normalize_weights_for_pallas(prob_32["weights"], cfg_32)
    per_rank_w32 = {k: v[0] for k, v in norm_w32.items()}
    pool_in32, _ = kernel.adapt_pool_for_kernel(prob_32["pool"], cfg_32)
    reg_pool_shape = (pool_in32.shape[0], *pool_in32.shape[2:])
    amesh_32 = AbstractMesh((32,), ("tp",), abstract_device=_AbstractTpu7xDevice())
    with jax.sharding.use_abstract_mesh(amesh_32):
        tp_sharding = NamedSharding(amesh_32, P("tp"))
        pool_sds32 = tuple(
            jax.ShapeDtypeStruct(reg_pool_shape, pool_in32.dtype, sharding=tp_sharding)
            for _ in range(cfg_32.NATT)
        )
        args_sds32 = (
            jax.ShapeDtypeStruct(
                prob_32["x"].shape, prob_32["x"].dtype, sharding=tp_sharding
            ),
            jax.ShapeDtypeStruct(
                prob_32["pos"].shape, prob_32["pos"].dtype, sharding=tp_sharding
            ),
            jax.ShapeDtypeStruct(
                prob_32["page_table"].shape,
                prob_32["page_table"].dtype,
                sharding=tp_sharding,
            ),
            pool_sds32,
            jax.ShapeDtypeStruct(
                prob_32["conv"].shape, prob_32["conv"].dtype, sharding=tp_sharding
            ),
            jax.ShapeDtypeStruct(
                prob_32["rec"].shape, prob_32["rec"].dtype, sharding=tp_sharding
            ),
            {
                k: jax.ShapeDtypeStruct(v.shape, v.dtype, sharding=tp_sharding)
                for k, v in norm_w32.items()
            },
        )
        per_rank_fn32 = kernel.make_pallas_step_fn(
            cfg_32,
            per_rank_w32,
            has_forced_routing=False,
            interpret=False,
            axis_name="tp",
        )
        w_pspec = {k: P("tp") for k in per_rank_w32}
        pool_pspec = tuple(P("tp") for _ in range(cfg_32.NATT))

        def _mapped_step(
            *args: object,
        ) -> tuple[jax.Array, tuple[jax.Array, ...], jax.Array, jax.Array]:
            """Squeeze and expand leading rank axis around per-rank Pallas call."""
            squeezed = [
                jax.tree.map(lambda z: jnp.squeeze(z, axis=0), arg) for arg in args
            ]
            xo, po, co, ro = per_rank_fn32(*squeezed)
            return (
                jnp.expand_dims(xo, axis=0),
                tuple(jnp.expand_dims(p, axis=0) for p in po),
                jnp.expand_dims(co, axis=0),
                jnp.expand_dims(ro, axis=0),
            )

        sharded_32 = jax.jit(
            jax.shard_map(
                _mapped_step,
                mesh=amesh_32,
                in_specs=(
                    P("tp"),
                    P("tp"),
                    P("tp"),
                    pool_pspec,
                    P("tp"),
                    P("tp"),
                    w_pspec,
                ),
                out_specs=(P("tp"), pool_pspec, P("tp"), P("tp")),
                check_vma=False,
            )
        )
        exported_32 = jax_export.export(sharded_32, platforms=["tpu"])(*args_sds32)
        hlo_32 = exported_32.mlir_module()
        assert "tpu_custom_call" in hlo_32
        plan.assert_no_all_gather(hlo_32, context="TP32xDCP8-TPU7x")


def test_qwen38_multi_region_donated_pools_alias_in_place() -> None:
    """Verify `_exec_single_rank_step` aliases all `NATT` donated pools in place."""
    cfg_2reg = dataclasses.replace(
        shapes.mini_parity_config(batch_size=2),
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
    prob = oracle.make_synthetic_problem(cfg_2reg, seed=3, pos=(1, 2))
    norm_w = plan.normalize_weights_for_pallas(prob["weights"], cfg_2reg)
    per_rank_w = {k: v[0] for k, v in norm_w.items()}
    rank_step = kernel.make_pallas_step_fn(
        cfg_2reg,
        per_rank_w,
        has_forced_routing=False,
        interpret=False,
        axis_name=None,
    )
    gdn_map = (0, 1, 0, 1, 0, 1)

    def _step_2reg(
        *args: object,
    ) -> tuple[jax.Array, ...]:
        """Execute `_exec_single_rank_step` across 2 donated region pools."""
        jax_args = cast("tuple[jax.Array, ...]", args[:-1])
        w_arg = cast("dict[str, jax.Array]", args[-1])
        pools_out, x_res = _exec_single_rank_step(
            jax_args, cfg_2reg, w_arg, rank_step, gdn_region_map=gdn_map
        )
        return (*pools_out, x_res)

    amesh_1 = AbstractMesh((1,), ("tp",), abstract_device=_AbstractTpu7xDevice())
    pool_u8_shape = (
        cfg_2reg.num_pages,
        2,
        cfg_2reg.K_R,
        cfg_2reg.D,
        cfg_2reg.page_size,
    )
    w_sds = {k: jax.ShapeDtypeStruct(v.shape, v.dtype) for k, v in per_rank_w.items()}
    with jax.sharding.use_abstract_mesh(amesh_1):
        sds_args = (
            jax.ShapeDtypeStruct((cfg_2reg.B, cfg_2reg.H), jnp.bfloat16),
            jax.ShapeDtypeStruct((cfg_2reg.B,), jnp.int32),
            jax.ShapeDtypeStruct(
                (cfg_2reg.NATT, cfg_2reg.B, cfg_2reg.max_pages_per_seq), jnp.int32
            ),
            jax.ShapeDtypeStruct((cfg_2reg.NGDN, cfg_2reg.B), jnp.int32),
            jax.ShapeDtypeStruct(pool_u8_shape, jnp.uint8),
            jax.ShapeDtypeStruct(pool_u8_shape, jnp.uint8),
            w_sds,
        )
        exported = jax_export.export(
            jax.jit(_step_2reg, donate_argnums=(4, 5)),
            platforms=["tpu"],
        )(*sds_args)
        mlir = exported.mlir_module()
        assert "tf.aliasing_output = 0 : i32" in mlir
        assert "tf.aliasing_output = 1 : i32" in mlir


@pytest.mark.tpu_test
def test_qwen38_tpu_single_device_megakernel_parity() -> None:
    """Execute compiled (`interpret=False`) Pallas megakernel on TPU.

    Checks numerical parity against FP64 oracle.
    """
    pytest.importorskip("torch_tpu")
    tpu_devices = jax.devices("tpu")
    mesh = jax.sharding.Mesh(np.asarray(tpu_devices[:1]), ("tp",))
    cfg_tpu = dataclasses.replace(
        shapes.mini_parity_config(batch_size=8),
        R=1,
        DCP=1,
        hier_group=1,
        Q_H=8,
        KV_H=8,
        KH=8,
        VH=8,
        L=4,
        E=128,
        K=8,
        I_E=256,
        SE_I=256,
        h_tile=128,
    )
    prob = oracle.make_synthetic_problem(
        cfg_tpu, seed=31, pos=(0, 3, 7, 11, 15, 19, 23, 27)
    )
    ref_out = oracle.run_fp64_reference(prob, cfg_tpu, use_forced_routing=False)
    got_out = kernel.run_qwen38_dcp8_step(
        prob,
        cfg_tpu,
        use_forced_routing=False,
        return_kv_updates=True,
        mesh=mesh,
        interpret=False,
    )
    got_x = np.asarray(
        jax.device_get(got_out["x"])[0].astype(jnp.float32), dtype=np.float64
    )
    rmse_x = float(np.sqrt(np.mean((got_x - ref_out["x"]) ** 2)))
    bound_x = oracle.layer_bf16_error_bound(
        float(np.sqrt(np.mean(ref_out["x"] ** 2))), 3, cfg_tpu.L
    )
    assert rmse_x <= bound_x, f"TPU compiled rmse_x={rmse_x} > {bound_x}"

    prob_zero = dict(prob)
    w_zero = dict(prob["weights"])
    w_zero["wo"] = jnp.zeros_like(w_zero["wo"])
    prob_zero["weights"] = w_zero
    got_zero = kernel.run_qwen38_dcp8_step(
        prob_zero,
        cfg_tpu,
        use_forced_routing=False,
        return_kv_updates=False,
        mesh=mesh,
        interpret=False,
    )
    got_zero_x = np.asarray(
        jax.device_get(got_zero["x"])[0].astype(jnp.float32), dtype=np.float64
    )
    rmse_diff = float(np.sqrt(np.mean((got_x - got_zero_x) ** 2)))
    assert rmse_diff > 5.0 * bound_x
