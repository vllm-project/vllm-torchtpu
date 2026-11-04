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
"""Certification tests for the Kimi K3 TpuMonoSpec and MonoProgram adapter."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import torch

from vllm_torchtpu.megakernel.runtime.contract import (
    DOMAIN_KERNEL_MESH,
    PAR_DP,
    LayoutDecline,
)
from vllm_torchtpu.megakernel_runtime import config as config_mod
from vllm_torchtpu.megakernel_runtime import hooks as hooks_mod
from vllm_torchtpu.megakernel_runtime.plugin import (
    Bucket,
    ForwardView,
    RunnerView,
)
from vllm_torchtpu.megakernel_runtime.testing import fake_runner
from vllm_torchtpu.megakernel_runtime.testkit import (
    assert_rank_uniform_declines,
    check_spec,
    route_lazy_loggers,
    run_cpu_e2e,
    run_two_ranks,
)
from vllm_torchtpu.models.kimi_k3.megakernel import SPEC, adapter
from vllm_torchtpu.models.kimi_k3.megakernel.adapter import create
from vllm_torchtpu.models.kimi_k3.megakernel.kernels.load import pack_expert_bytes

if TYPE_CHECKING:
    from pathlib import Path

PLUGIN_MODULE = "vllm_torchtpu.models.kimi_k3.megakernel"
EXPECTED_BF16_BLOCK_SIZE = 256
EXPECTED_FP8_BLOCK_SIZE = 512
EXPECTED_STREAMED_WEIGHTS = 2
EXPECTED_RANK_COUNT = 2
MIN_WARMED_BUCKETS = 2
EXPECTED_CAPTURED_SHARDS = 3


def _kimi_hf_config(
    *,
    num_hidden_layers: int = 93,
    hidden_size: int = 7168,
    arch: str = "KimiK3ForConditionalGeneration",
) -> SimpleNamespace:
    """Build a synthetic Kimi K3 HuggingFace config namespace."""
    return SimpleNamespace(
        architectures=[arch],
        model_type="kimi_k3",
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
    )


def test_k3_mono_spec_check_spec() -> None:
    """Verify `check_spec` contract and architecture shorthand resolution."""
    spec = check_spec(PLUGIN_MODULE, lazy_create=True)
    assert spec.name == "kimi_k3"
    assert spec.collective_domain == DOMAIN_KERNEL_MESH
    assert spec.chips == ("TPU7x",)
    assert spec.numerics == "approx"
    assert spec.layout_request == ".adapter:layout_request"
    assert spec.load_tensor == ".adapter:load_tensor"

    hf_k3 = SimpleNamespace(
        architectures=["KimiK3ForConditionalGeneration"],
        text_config=SimpleNamespace(hidden_size=7168, num_hidden_layers=93),
    )
    assert config_mod.shorthand_for_arch(hf_k3) == "kimi_k3"


def test_k3_hooks_load_tensor_and_layout_request() -> None:
    """Verify `layout_request`, `1/32` `load_tensor`, and `bind` layout checks."""
    adapter.clear_observed_weights()
    raw_packed = torch.arange(32 * 8, dtype=torch.uint8).reshape(32, 8)
    raw_scales = torch.arange(32 * 4, dtype=torch.uint8).reshape(32, 4)

    vllm_cfg = SimpleNamespace(
        additional_config={
            "megakernel": {
                "enabled": True,
                "plugin": "kimi_k3",
                "options": {"load_tensors": True, "kernel_rank": 0},
            }
        },
        cache_config=SimpleNamespace(
            block_size=16,
            cache_dtype="bfloat16",
        ),
        parallel_config=SimpleNamespace(
            cp_kv_cache_interleave_size=1,
        ),
        model_config=SimpleNamespace(hf_config=_kimi_hf_config()),
    )

    hooks_mod.apply_layout_request(vllm_cfg)
    assert vllm_cfg.cache_config.block_size == EXPECTED_BF16_BLOCK_SIZE
    assert (
        vllm_cfg.parallel_config.cp_kv_cache_interleave_size == EXPECTED_BF16_BLOCK_SIZE
    )
    assert (
        vllm_cfg.parallel_config.dcp_kv_cache_interleave_size
        == EXPECTED_BF16_BLOCK_SIZE
    )

    fp8_cfg = SimpleNamespace(
        additional_config={"megakernel": {"enabled": True, "plugin": "kimi_k3"}},
        cache_config=SimpleNamespace(
            block_size=16,
            cache_dtype="fp8_e4m3",
        ),
        parallel_config=SimpleNamespace(
            cp_kv_cache_interleave_size=1,
        ),
        model_config=SimpleNamespace(hf_config=_kimi_hf_config()),
    )
    hooks_mod.apply_layout_request(fp8_cfg)
    assert fp8_cfg.cache_config.block_size == EXPECTED_FP8_BLOCK_SIZE
    assert (
        fp8_cfg.parallel_config.cp_kv_cache_interleave_size == EXPECTED_FP8_BLOCK_SIZE
    )
    assert (
        fp8_cfg.parallel_config.dcp_kv_cache_interleave_size == EXPECTED_FP8_BLOCK_SIZE
    )

    weights = [
        ("model.layers.1.mlp.experts.w13_weight_packed", raw_packed),
        ("model.layers.1.mlp.experts.w13_weight_scale", raw_scales),
    ]
    streamed = list(hooks_mod.load_tensor_stream(vllm_cfg, weights))
    assert len(streamed) == EXPECTED_STREAMED_WEIGHTS

    snap = adapter.observed_weights_snapshot()
    packed_entry = snap["model.layers.1.mlp.experts.w13_weight_packed"]
    scale_entry = snap["model.layers.1.mlp.experts.w13_weight_scale"]
    assert packed_entry.kind == "mxfp4_packed"
    assert packed_entry.packed_u32 is not None
    assert packed_entry.packed_u32.shape == (2, 1)
    assert packed_entry.packed_u32.dtype == np.uint32
    assert np.array_equal(
        packed_entry.packed_u32, pack_expert_bytes(raw_packed[:1].numpy())
    )
    assert scale_entry.kind == "e8m0_scale"
    assert scale_entry.raw_u8 is not None
    assert scale_entry.raw_u8.shape == (4, 1)
    assert np.array_equal(scale_entry.raw_u8, raw_scales[:1].numpy().T)

    # Verify bind() succeeds after apply_layout_request without manual interleave setup
    runner_ok = fake_runner(
        hf_config=_kimi_hf_config(),
        block_size=16,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            cp_kv_cache_interleave_size=1,
        ),
    )
    runner_ok.vllm_config.additional_config = {
        "megakernel": {"enabled": True, "plugin": "kimi_k3"}
    }
    runner_ok.vllm_config.parallel_config = SimpleNamespace(
        tensor_parallel_size=1,
        cp_kv_cache_interleave_size=1,
    )
    hooks_mod.apply_layout_request(runner_ok.vllm_config)
    prog = create(RunnerView(runner_ok, tp_rank=0), {})
    assert not isinstance(prog, adapter.Decline)
    assert adapter.observed_weights_snapshot() == {}
    assert len(prog.observed_weights) == EXPECTED_STREAMED_WEIGHTS
    prog.load("cpu_mesh")
    assert adapter.observed_weights_snapshot() == {}
    bound = prog.bind(RunnerView(runner_ok, tp_rank=0))
    assert bound["block_size"] == EXPECTED_BF16_BLOCK_SIZE

    runner_bad_block = fake_runner(
        hf_config=_kimi_hf_config(),
        block_size=128,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            cp_kv_cache_interleave_size=128,
        ),
    )
    runner_bad_block.vllm_config.parallel_config = SimpleNamespace(
        tensor_parallel_size=1,
        cp_kv_cache_interleave_size=128,
    )
    prog_bad_block = create(RunnerView(runner_bad_block, tp_rank=0), {})
    assert not isinstance(prog_bad_block, adapter.Decline)
    with pytest.raises(LayoutDecline, match="unaligned_cache_layout"):
        prog_bad_block.bind(RunnerView(runner_bad_block, tp_rank=0))

    runner_bad_interleave = fake_runner(
        hf_config=_kimi_hf_config(),
        block_size=EXPECTED_BF16_BLOCK_SIZE,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            cp_kv_cache_interleave_size=128,
        ),
    )
    runner_bad_interleave.vllm_config.parallel_config = SimpleNamespace(
        tensor_parallel_size=1,
        cp_kv_cache_interleave_size=128,
    )
    prog_bad_interleave = create(RunnerView(runner_bad_interleave, tp_rank=0), {})
    assert not isinstance(prog_bad_interleave, adapter.Decline)
    with pytest.raises(LayoutDecline, match="unaligned_cache_layout"):
        prog_bad_interleave.bind(RunnerView(runner_bad_interleave, tp_rank=0))


def test_k3_streaming_load_tensor_releases_host_buffers() -> None:
    """Verify `1/32` ascending-id shard slicing and `WEIGHT_STORE` cleanup."""
    adapter.clear_observed_weights()
    raw_packed = torch.arange(128 * 16, dtype=torch.uint8).reshape(128, 16)
    raw_scales = (
        torch.arange(128 * 4, dtype=torch.int32).remainder(250).add(1).to(torch.uint8)
    ).reshape(128, 4)
    raw_w2 = torch.arange(8 * 32, dtype=torch.uint8).reshape(8, 32)
    opts = {"load_tensors": True, "kernel_rank": 5}

    adapter.load_tensor(
        "model.layers.1.mlp.experts.w13_weight_packed", raw_packed, opts
    )
    adapter.load_tensor("model.layers.1.mlp.experts.w13_weight_scale", raw_scales, opts)
    adapter.load_tensor(
        "model.layers.1.block_sparse_moe.experts.0.w1.weight_packed", raw_w2, opts
    )
    adapter.load_tensor(
        "model.layers.1.block_sparse_moe.experts.115.w2.weight_packed", raw_w2, opts
    )

    snap = adapter.observed_weights_snapshot()
    assert "model.layers.1.block_sparse_moe.experts.0.w1.weight_packed" not in snap
    w13_entry = snap["model.layers.1.mlp.experts.w13_weight_packed"]
    scale_entry = snap["model.layers.1.mlp.experts.w13_weight_scale"]
    w2_entry = snap["model.layers.1.block_sparse_moe.experts.115.w2.weight_packed"]

    assert w13_entry.packed_u32 is not None
    assert w13_entry.packed_u32.shape == (4, 4)
    assert np.array_equal(
        w13_entry.packed_u32, pack_expert_bytes(raw_packed[20:24].numpy())
    )
    assert scale_entry.raw_u8 is not None
    assert scale_entry.raw_u8.shape == (4, 4)
    assert np.array_equal(scale_entry.raw_u8, raw_scales[20:24].numpy().T)
    assert w2_entry.packed_u32 is not None
    assert w2_entry.packed_u32.shape == (2, 8)
    assert np.array_equal(
        w2_entry.packed_u32, pack_expert_bytes(raw_w2[:, 8:16].numpy())
    )

    cpu_runner = fake_runner(
        hf_config=_kimi_hf_config(),
        block_size=EXPECTED_BF16_BLOCK_SIZE,
    )
    prog = create(RunnerView(cpu_runner, tp_rank=0), {})
    assert not isinstance(prog, adapter.Decline)
    assert adapter.observed_weights_snapshot() == {}
    assert len(prog.observed_weights) == EXPECTED_CAPTURED_SHARDS

    adapter.load_tensor("model.layers.2.mlp.experts.w13_weight_scale", raw_scales, opts)
    assert len(adapter.observed_weights_snapshot()) == 1
    prog.load("cpu_mesh")
    assert adapter.observed_weights_snapshot() == {}

    adapter.load_tensor(
        "model.layers.3.mlp.experts.w13_weight_packed", raw_packed, opts
    )
    assert len(adapter.observed_weights_snapshot()) == 1
    tpu_runner = fake_runner(
        hf_config=_kimi_hf_config(),
        tp_size=32,
        device=SimpleNamespace(type="tpu"),
        model=object(),
    )
    dec_tpu = create(RunnerView(tpu_runner, tp_rank=0), {})
    assert isinstance(dec_tpu, adapter.Decline)
    assert dec_tpu.code == "runner_lifecycle_unwired"
    assert adapter.observed_weights_snapshot() == {}

    adapter.load_tensor(
        "model.layers.4.mlp.experts.w13_weight_packed", raw_packed, opts
    )
    skew_runner = fake_runner(
        hf_config=_kimi_hf_config(num_hidden_layers=90),
        block_size=EXPECTED_BF16_BLOCK_SIZE,
    )
    dec_skew = create(RunnerView(skew_runner, tp_rank=1), {})
    assert isinstance(dec_skew, adapter.Decline)
    assert dec_skew.code == "invalid_geometry"
    assert adapter.observed_weights_snapshot() == {}


def test_k3_spec_declares_par_dp_and_accepts_attention_dp4() -> None:
    """Verify `PAR_DP`, `attention_dp=4`, and CPU-only step callable guard."""
    assert PAR_DP in SPEC.parallelism
    assert "runner_lifecycle_unwired" in SPEC.decline_codes
    assert "decoder_unwired" not in SPEC.decline_codes

    tpu_dp4 = fake_runner(
        hf_config=_kimi_hf_config(),
        tp_size=32,
        device=SimpleNamespace(type="tpu"),
        model=object(),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=32, decode_context_parallel_size=8
        ),
    )
    tpu_dp4.vllm_config.parallel_config = SimpleNamespace(
        tensor_parallel_size=32, decode_context_parallel_size=8
    )
    dec_dp4 = create(RunnerView(tpu_dp4, tp_rank=0), {"attention_dp": 4})
    assert isinstance(dec_dp4, adapter.Decline)
    assert dec_dp4.code == "runner_lifecycle_unwired"

    dec_bad_dp = create(RunnerView(tpu_dp4, tp_rank=0), {"attention_dp": 8})
    assert isinstance(dec_bad_dp, adapter.Decline)
    assert dec_bad_dp.code == "unsupported_attention_dp"

    tpu_bad_dcp = fake_runner(
        hf_config=_kimi_hf_config(),
        tp_size=32,
        device=SimpleNamespace(type="tpu"),
        model=object(),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=32, decode_context_parallel_size=16
        ),
    )
    tpu_bad_dcp.vllm_config.parallel_config = SimpleNamespace(
        tensor_parallel_size=32, decode_context_parallel_size=16
    )
    dec_bad_dcp = create(RunnerView(tpu_bad_dcp, tp_rank=0), {"attention_dp": 4})
    assert isinstance(dec_bad_dcp, adapter.Decline)
    assert dec_bad_dcp.code == "unsupported_dcp_size"

    cpu_runner = fake_runner(
        hf_config=_kimi_hf_config(),
        block_size=EXPECTED_BF16_BLOCK_SIZE,
    )
    prog = create(RunnerView(cpu_runner, tp_rank=0), {"synthetic_hidden_size": 64})
    assert not isinstance(prog, adapter.Decline)
    step_fn = prog.build(Bucket(1, 2, 2))
    meta_ids = torch.ones((2,), dtype=torch.int32, device="meta")
    meta_pos = torch.zeros((2,), dtype=torch.int32, device="meta")
    with pytest.raises(RuntimeError, match="CPU-only"):
        step_fn(meta_ids, meta_pos)


def test_k3_mono_program_cpu_e2e(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify CPU end-to-end decode, T=8 verify, SlotMap lifecycle, and declines."""
    route_lazy_loggers(monkeypatch)
    monkeypatch.delenv("VLLM_TPU_MEGAKERNEL", raising=False)
    monkeypatch.setattr(
        "vllm_torchtpu.megakernel_runtime.driver._local_tp_rank", lambda: 0
    )

    runner = fake_runner(
        hf_config=_kimi_hf_config(),
        block_size=EXPECTED_BF16_BLOCK_SIZE,
        token_paddings=(),
        req_block_pairs=((4, 2),),
        speculative_tokens=7,
    )
    opts = {
        "buckets": [2, 4],
        "verify_buckets": [8],
        "synthetic_hidden_size": 128,
    }

    decode_md = {
        "model.layers.0.linear_attn": SimpleNamespace(num_seqs=2),
        "model.layers.3.self_attn.attn": SimpleNamespace(
            num_seqs=2,
            query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
            request_distribution=torch.tensor([2, 2, 2], dtype=torch.int32),
        ),
    }
    ids_decode = torch.tensor([11, 22], dtype=torch.int32)
    pos_decode = torch.tensor([5, 9], dtype=torch.int32)

    res = run_cpu_e2e(
        PLUGIN_MODULE,
        runner,
        [(ids_decode, pos_decode)],
        metadata=decode_md,
        options=opts,
    )
    assert res.hits == [True]
    assert res.outputs[0].shape == (2, 128)
    res.program.admit_requests(["req-a", "req-b"])
    assert "req-a" in res.program.slot_map
    assert "req-b" in res.program.slot_map

    # Test release on the same program instance
    res.program.admit_requests(["req-a"], finished_req_ids=["req-b"])
    assert "req-a" in res.program.slot_map
    assert "req-b" not in res.program.slot_map

    # T=8 verify step: 1 seq, 8 tokens, query_start_loc=[0, 8] -> output (8, 128)
    verify_md = {
        "model.layers.3.self_attn.attn": SimpleNamespace(
            num_seqs=1,
            query_start_loc=torch.tensor([0, 8], dtype=torch.int32),
            request_distribution=torch.tensor([1, 1, 1], dtype=torch.int32),
        ),
    }
    ids_verify = torch.arange(8, dtype=torch.int32) + 100
    pos_verify = torch.arange(8, dtype=torch.int32) + 10
    res_verify = run_cpu_e2e(
        PLUGIN_MODULE,
        runner,
        [(ids_verify, pos_verify)],
        metadata=verify_md,
        options=opts,
    )
    assert res_verify.hits == [True]
    assert res_verify.outputs[0].shape == (8, 128)

    bad_tpu_runner = fake_runner(
        hf_config=_kimi_hf_config(),
        tp_size=4,
        device=SimpleNamespace(type="tpu"),
    )
    dec_tp = create(RunnerView(bad_tpu_runner, tp_rank=0), {})
    assert isinstance(dec_tp, adapter.Decline)
    assert dec_tp.code == "unsupported_tp_size"

    bad_layout_runner = fake_runner(
        hf_config=_kimi_hf_config(),
        block_size=16,
    )
    with pytest.raises(AssertionError, match="unaligned_cache_layout"):
        run_cpu_e2e(
            PLUGIN_MODULE,
            bad_layout_runner,
            [(ids_decode, pos_decode)],
            metadata=decode_md,
            options=opts,
        )


def test_k3_tpu_device_declines_runner_lifecycle_unwired() -> None:
    """Verify TPU device declines cleanly with `runner_lifecycle_unwired`."""
    tpu_runner = fake_runner(
        hf_config=_kimi_hf_config(),
        tp_size=32,
        device=SimpleNamespace(type="tpu"),
        model=object(),
    )
    dec = create(RunnerView(tpu_runner, tp_rank=0), {})
    assert isinstance(dec, adapter.Decline)
    assert dec.code == "runner_lifecycle_unwired"


def _init_cpu_worker_env() -> object:
    """Configure CPU JAX/distributed globals for two-rank workers and return SPEC."""
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["PJRT_DEVICE"] = "CPU"
    import torch.distributed as dist

    import vllm_torchtpu.megakernel_runtime.agreement as agreement_mod

    sys.modules["torch_tpu"] = None  # type: ignore[assignment]
    sys.modules["vllm.distributed"] = None  # type: ignore[assignment]
    agreement_mod.get_world_group = None  # type: ignore[attr-defined]
    agreement_mod._world_cpu_group = lambda: dist.group.WORLD
    from vllm_torchtpu.models.kimi_k3.megakernel import SPEC

    if os.environ.get("K3_SKEW_RANK1_GEOM") == "1" and dist.get_rank() == 1:
        import vllm_torchtpu.models.kimi_k3.megakernel.adapter as adapter_mod

        orig_create = adapter_mod.create

        def _skew_create(view: Any, options: Any) -> Any:
            """Inject mismatched layer count on rank 1 to test cross-rank decline."""
            view._runner.model_config.hf_config = SimpleNamespace(
                model_type="kimi_k3",
                architectures=["KimiK3ForConditionalGeneration"],
                num_hidden_layers=90,
                hidden_size=7168,
            )
            return orig_create(view, options)

        adapter_mod.create = _skew_create
    return SPEC


def test_k3_mono_program_two_ranks_kernel_mesh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify 2-rank `DOMAIN_KERNEL_MESH` startup and rank-uniform declines."""
    opts = {
        "buckets": [2],
        "verify_buckets": [8],
        "synthetic_hidden_size": 64,
    }
    runner_kw = {
        "block_size": EXPECTED_BF16_BLOCK_SIZE,
        "token_paddings": [8, 16],
        "req_block_pairs": [[2, 2]],
    }
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("PJRT_DEVICE", "CPU")
    shim_src = (
        "from tests.models.kimi_k3.megakernel.test_k3_mono_spec import (\n"
        "    _init_cpu_worker_env,\n"
        ")\n"
        "__package__ = 'vllm_torchtpu.models.kimi_k3.megakernel'\n"
        "SPEC = _init_cpu_worker_env()\n"
    )

    serve_dir = tmp_path / "serve"
    serve_dir.mkdir()
    outcomes = run_two_ranks(
        serve_dir,
        plugin="k3_rank_test_plugin",
        sources=[shim_src, shim_src],
        options=opts,
        runner_kwargs=runner_kw,
    )
    assert len(outcomes) == EXPECTED_RANK_COUNT
    for res in outcomes:
        assert res["type"] == "ProgramDriver", outcomes
        assert res.get("warmed", 0) >= MIN_WARMED_BUCKETS, outcomes

    decline_dir = tmp_path / "decline"
    decline_dir.mkdir()
    monkeypatch.setenv("K3_SKEW_RANK1_GEOM", "1")
    declined = assert_rank_uniform_declines(
        decline_dir,
        plugin="k3_rank_test_plugin",
        sources=[shim_src, shim_src],
        options=opts,
        runner_kwargs=runner_kw,
    )
    codes = {r.get("code") for r in declined}
    assert "invalid_geometry" in codes
    assert "peer_declined:invalid_geometry" in codes


def test_k3_stacked_3d_expert_slicing_and_boundary_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify 3D stacked expert ETP slicing, rank bounds, and divisibility guards."""
    adapter.clear_observed_weights()
    stacked_3d = (
        torch.arange(8 * 8 * 16, dtype=torch.int32)
        .remainder(251)
        .to(dtype=torch.uint8)
        .reshape(8, 8, 16)
    )

    adapter.load_tensor(
        "model.layers.0.mlp.experts.w13_weight_packed",
        stacked_3d,
        {"load_tensors": True, "kernel_rank": 0, "num_experts": 8},
    )
    adapter.load_tensor(
        "model.layers.0.mlp.experts.w2_weight_packed",
        stacked_3d,
        {"load_tensors": True, "kernel_rank": 31, "num_experts": 8},
    )
    adapter.load_tensor(
        "model.layers.0.mlp.shared_experts.w13_weight_packed",
        stacked_3d,
        {"load_tensors": True, "kernel_rank": 0, "num_experts": 8},
    )
    snap = adapter.observed_weights_snapshot()
    assert "model.layers.0.mlp.shared_experts.w13_weight_packed" not in snap
    w13_r0 = snap["model.layers.0.mlp.experts.w13_weight_packed"]
    w2_r31 = snap["model.layers.0.mlp.experts.w2_weight_packed"]
    assert w13_r0.packed_u32 is not None
    assert np.array_equal(
        w13_r0.packed_u32,
        pack_expert_bytes(stacked_3d[0, 0:2, :].numpy()),
    )
    assert w2_r31.packed_u32 is not None
    assert np.array_equal(
        w2_r31.packed_u32,
        pack_expert_bytes(stacked_3d[7, :, 12:16].numpy()),
    )
    adapter.clear_observed_weights()

    with pytest.raises(ValueError, match="kernel_rank position must be in 0..31"):
        adapter.load_tensor(
            "model.layers.0.mlp.experts.w13_weight_packed",
            stacked_3d,
            {"load_tensors": True, "kernel_rank": 32},
        )

    with pytest.raises(ValueError, match="kernel_rank must be an integer"):
        adapter.load_tensor(
            "model.layers.0.mlp.experts.w13_weight_packed",
            stacked_3d,
            {"load_tensors": True, "kernel_rank": [1]},
        )

    # Without kernel_rank when global JAX devices != 32, _default_kernel_position raises
    monkeypatch.setattr(adapter.jax, "devices", lambda: [SimpleNamespace(id=0)])
    with pytest.raises(ValueError, match="requires 32 global JAX devices"):
        adapter.load_tensor(
            "model.layers.0.mlp.experts.w13_weight_packed",
            stacked_3d,
            {"load_tensors": True},
        )

    # Indivisible EP or ETP dimensions raise ValueError naming tensor and divisor
    bad_ep_3d = torch.zeros((7, 8, 16), dtype=torch.uint8)
    with pytest.raises(ValueError, match="not divisible by EP=8"):
        adapter.load_tensor(
            "model.layers.0.mlp.experts.w13_weight_packed",
            bad_ep_3d,
            {"load_tensors": True, "kernel_rank": 0},
        )

    bad_etp_3d = torch.zeros((8, 6, 16), dtype=torch.uint8)
    with pytest.raises(ValueError, match="not divisible by ETP=4"):
        adapter.load_tensor(
            "model.layers.0.mlp.experts.w13_weight_packed",
            bad_etp_3d,
            {"load_tensors": True, "kernel_rank": 0},
        )

    bad_etp_2d = torch.zeros((32, 6), dtype=torch.uint8)
    with pytest.raises(ValueError, match="not divisible by ETP=4"):
        adapter.load_tensor(
            "model.layers.0.mlp.experts.w2_weight_packed",
            bad_etp_2d,
            {"load_tensors": True, "kernel_rank": 0},
        )


def test_k3_forward_pads_sub_bucket_tokens_and_dp_slot_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify sub-bucket padding, ineligible steps, and DP SlotMap allocation."""
    route_lazy_loggers(monkeypatch)
    monkeypatch.delenv("VLLM_TPU_MEGAKERNEL", raising=False)
    monkeypatch.setattr(
        "vllm_torchtpu.megakernel_runtime.driver._local_tp_rank", lambda: 0
    )

    runner = fake_runner(
        hf_config=_kimi_hf_config(),
        block_size=EXPECTED_BF16_BLOCK_SIZE,
        token_paddings=(),
        req_block_pairs=((4, 2),),
    )
    opts = {
        "buckets": [2, 4],
        "verify_buckets": [],
        "synthetic_hidden_size": 64,
        "max_slots": 64,
        "attention_dp": 4,
    }

    pad_md = {
        "model.layers.3.self_attn.attn": SimpleNamespace(
            num_seqs=3,
            query_start_loc=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
            request_distribution=torch.tensor([3, 3, 3], dtype=torch.int32),
        ),
    }
    ids_3 = torch.tensor([10, 20, 30], dtype=torch.int32)
    pos_3 = torch.tensor([1, 2, 3], dtype=torch.int32)
    res_pad = run_cpu_e2e(
        PLUGIN_MODULE,
        runner,
        [(ids_3, pos_3)],
        metadata=pad_md,
        options=opts,
    )
    assert res_pad.hits == [True]
    assert res_pad.outputs[0].shape == (3, 64)
    prog = res_pad.program

    # With attention_dp=4 and max_slots=64, each group has 16 slots.
    # Allocating 24 requests across the 4 DP groups succeeds (6 per group).
    req_batch = [f"req-{i}" for i in range(24)]
    slots = prog.admit_requests(req_batch)
    assert len(set(slots)) == len(req_batch)
    for req_id in req_batch:
        assert req_id in prog.slot_map

    # Releasing 4 requests and admitting 4 new requests reclaims their DP-group slots.
    prog.admit_requests(
        [f"req-new-{i}" for i in range(4)],
        finished_req_ids=req_batch[:4],
    )
    for gone in req_batch[:4]:
        assert gone not in prog.slot_map
    for added in [f"req-new-{i}" for i in range(4)]:
        assert added in prog.slot_map

    ineligible_md = {
        "model.layers.3.self_attn.attn": SimpleNamespace(
            num_seqs=8,
            query_start_loc=torch.arange(9, dtype=torch.int32),
            request_distribution=torch.tensor([8, 8, 8], dtype=torch.int32),
        ),
    }
    ctx_8 = SimpleNamespace(
        input_ids=torch.ones((8,), dtype=torch.int32),
        positions=torch.arange(8, dtype=torch.int32),
    )
    fwd_ineligible = ForwardView(
        attn_metadata=ineligible_md,
        mrope_positions=None,
        steps={(1, 2): lambda x, p: x, (1, 4): lambda x, p: x},
        num_tokens=8,
        inspect=res_pad.driver._inspect_step_detail,
    )
    dec_ineligible = prog.decide(ctx_8, fwd_ineligible)
    assert dec_ineligible is not None
    assert not dec_ineligible.eligible


@pytest.mark.parametrize(
    ("raw_verify", "expected"),
    [
        ("8,16,72", (8, 16, 72)),
        ([8, 16, 72], (8, 16, 72)),
        ("none", ()),
        ([], ()),
    ],
)
def test_k3_verify_buckets_validation_valid(
    raw_verify: object, expected: tuple[int, ...]
) -> None:
    """Verify valid string and list `verify_buckets` forms resolve multiples of 8."""
    cpu_runner = fake_runner(
        hf_config=_kimi_hf_config(),
        block_size=EXPECTED_BF16_BLOCK_SIZE,
    )
    prog = create(
        RunnerView(cpu_runner, tp_rank=0),
        {"buckets": [2], "verify_buckets": raw_verify},
    )
    assert not isinstance(prog, adapter.Decline)
    assert prog.verify_buckets == expected


@pytest.mark.parametrize(
    "bad_verify",
    ["12", [12], "0", [0], "8,12", [8, 12], "not_an_int", 123],
)
def test_k3_verify_buckets_validation_invalid(bad_verify: object) -> None:
    """Verify invalid `verify_buckets` forms raise `ValueError`."""
    cpu_runner = fake_runner(
        hf_config=_kimi_hf_config(),
        block_size=EXPECTED_BF16_BLOCK_SIZE,
    )
    with pytest.raises(ValueError, match="verify_buckets"):
        create(
            RunnerView(cpu_runner, tp_rank=0),
            {"buckets": [2], "verify_buckets": bad_verify},
        )
