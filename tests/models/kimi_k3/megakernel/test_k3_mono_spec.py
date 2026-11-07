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

import json
from collections.abc import Iterator
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest import mock

import numpy as np
import pytest
import torch

from vllm_torchtpu.megakernel.errors import MegakernelConfigError
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
from vllm_torchtpu.models.kimi_k3.megakernel import SPEC, adapter, weight_stream
from vllm_torchtpu.models.kimi_k3.megakernel.adapter import create
from vllm_torchtpu.models.kimi_k3.megakernel.kernels import Config
from vllm_torchtpu.models.kimi_k3.megakernel.kernels.load import (
    BF16,
    PREFIX,
    Checkpoint,
    pack_expert_bytes,
    selected_local_shapes,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kernels.load import (
    _family_layers as canonical_family_layers,
)

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
        "__package__ = 'vllm_torchtpu.models.kimi_k3.megakernel'\n"
        "import os, sys, types\n"
        "os.environ['JAX_PLATFORMS'] = 'cpu'\n"
        "os.environ['PJRT_DEVICE'] = 'CPU'\n"
        "import torch.distributed as dist\n"
        "import vllm_torchtpu.megakernel_runtime.agreement as _a\n"
        "sys.modules['torch_tpu'] = None\n"
        "_a._world_cpu_group = lambda: dist.group.WORLD\n"
        "from vllm_torchtpu.models.kimi_k3.megakernel import SPEC\n"
        "if os.environ.get('K3_SKEW_RANK1_GEOM') == '1' and dist.get_rank() == 1:\n"
        "    import vllm_torchtpu.models.kimi_k3.megakernel.adapter as _adapter\n"
        "    _orig_create = _adapter.create\n"
        "    def _skew_create(view, options):\n"
        "        view._runner.model_config.hf_config = types.SimpleNamespace(\n"
        "            model_type='kimi_k3',\n"
        "            architectures=['KimiK3ForConditionalGeneration'],\n"
        "            num_hidden_layers=90,\n"
        "            hidden_size=7168,\n"
        "        )\n"
        "        return _orig_create(view, options)\n"
        "    _adapter.create = _skew_create\n"
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


SYNTH_NUM_LAYERS = 2
SYNTH_TP_RANKS = 32


def _family_layers(cfg: Config, fam: str, total_layers: int) -> list[int] | None:
    """Return layer indices contributing to stacked `fam` array, or None."""
    return canonical_family_layers(cfg, fam, total_layers)


def _max_bit_diff(got_t: torch.Tensor, ref_np: np.ndarray) -> float:
    """Compute maximum absolute difference between device tensor and NumPy reference."""
    g = got_t.squeeze(0).cpu().contiguous()
    if g.dtype == torch.bfloat16:
        g_arr = g.view(torch.uint16).numpy().view(BF16)
    else:
        g_arr = g.numpy()
    assert g_arr.shape == ref_np.shape, (g_arr.shape, ref_np.shape)
    g_raw = np.ascontiguousarray(g_arr).view(np.uint8)
    ref_raw = np.ascontiguousarray(ref_np).view(np.uint8)
    diff = np.abs(g_raw.astype(np.int32) - ref_raw.astype(np.int32))
    return float(np.max(diff))


def _tensor_view(
    mm: np.memmap, shape: tuple[int, ...], dt_name: str, a: int, b: int
) -> torch.Tensor:
    """Slice `mm[a:b]` as a zero-copy torch tensor with target shape/dtype."""
    raw = mm[a:b]
    if dt_name == "BF16":
        arr = raw.view(np.uint16).reshape(shape)
        return torch.from_numpy(arr).view(torch.bfloat16)
    if dt_name == "F32":
        arr = raw.view(np.float32).reshape(shape)
        return torch.from_numpy(arr)
    arr = raw.view(np.uint8).reshape(shape)
    return torch.from_numpy(arr)


def _write_synthetic_k3_checkpoint(
    ckpt_dir: Path,
) -> tuple[list[tuple[str, tuple[int, ...], str, int, int]], int]:
    """Write a minimal 2-layer Kimi K3 safetensors checkpoint with valid E8M0 scales."""
    cfg = Config()
    entries = _build_synthetic_checkpoint_entries(cfg)
    _write_synthetic_k3_config(ckpt_dir, cfg)
    spans, data_base = _write_safetensors_file(ckpt_dir / "model.safetensors", entries)
    index_json = {
        "metadata": {"total_size": spans[-1][-1] if spans else 0},
        "weight_map": {name: "model.safetensors" for name, *_ in entries},
    }
    with open(ckpt_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index_json, f)
    return spans, data_base


def _write_safetensors_file(
    path: Path, entries: list[tuple[str, tuple[int, ...], str]]
) -> tuple[list[tuple[str, tuple[int, ...], str, int, int]], int]:
    """Serialize entry metadata and contiguous payload into a valid safetensors file."""
    import json
    hdr: dict[str, object] = {}
    cur = 0
    spans: list[tuple[str, tuple[int, ...], str, int, int]] = []
    dtype_map = {"BF16": ("BF16", 2), "U8": ("U8", 1), "F32": ("F32", 4)}
    for name, sh, dt in entries:
        st_dt, sz = dtype_map[dt]
        nbytes = int(np.prod(sh)) * sz
        hdr[name] = {
            "dtype": st_dt,
            "shape": list(sh),
            "data_offsets": [cur, cur + nbytes],
        }
        spans.append((name, sh, dt, cur, cur + nbytes))
        cur += nbytes
    hdr_bytes = json.dumps(hdr).encode("utf-8")
    hdr_len = len(hdr_bytes)
    with open(path, "wb") as f:
        f.write(int(hdr_len).to_bytes(8, "little"))
        f.write(hdr_bytes)
        for name, sh, dt, a, b in spans:
            sz = b - a
            if "weight_scale" in name or "e8m0" in name:
                f.write(np.full(sz, 128, dtype=np.uint8).tobytes())
            elif dt == "BF16":
                f.write(np.arange(sz // 2, dtype=np.uint16).tobytes())
            elif dt == "F32":
                f.write(np.zeros(sz // 4, dtype=np.float32).tobytes())
            else:
                f.write(np.zeros(sz, dtype=np.uint8).tobytes())
    return spans, 8 + hdr_len


def _build_synthetic_checkpoint_entries(
    cfg: Config,
) -> list[tuple[str, tuple[int, ...], str]]:
    """Assemble expected checkpoint parameter tensor metadata for a 2-layer model."""
    h, lat, eh = cfg.dim, cfg.latent, cfg.expert_hidden
    entries: list[tuple[str, tuple[int, ...], str]] = [
        (f"{PREFIX}embed_tokens.weight", (cfg.vocab, h), "BF16"),
    ]
    for l_id in range(SYNTH_NUM_LAYERS):
        p = f"{PREFIX}layers.{l_id}."
        entries.extend([
            (f"{p}input_layernorm.weight", (h,), "BF16"),
            (f"{p}post_attention_layernorm.weight", (h,), "BF16"),
            (f"{p}self_attention_res_norm.weight", (h,), "BF16"),
            (f"{p}self_attention_res_proj.weight", (1, h), "F32"),
            (f"{p}mlp_res_norm.weight", (h,), "BF16"),
            (f"{p}mlp_res_proj.weight", (1, h), "F32"),
        ])
        if l_id == 0:
            entries.extend([
                (f"{p}mlp.gate_proj.weight", (cfg.dense_hidden, h), "BF16"),
                (f"{p}mlp.up_proj.weight", (cfg.dense_hidden, h), "BF16"),
                (f"{p}mlp.down_proj.weight", (h, cfg.dense_hidden), "BF16"),
            ])
        else:
            mp = f"{p}block_sparse_moe."
            entries.extend([
                (f"{mp}gate.weight", (cfg.experts, h), "BF16"),
                (f"{mp}gate.e_score_correction_bias", (cfg.experts,), "BF16"),
                (f"{mp}routed_expert_norm.weight", (lat,), "BF16"),
                (f"{mp}routed_expert_down_proj.weight", (lat, h), "BF16"),
                (f"{mp}routed_expert_up_proj.weight", (h, lat), "BF16"),
                (
                    f"{mp}shared_experts.gate_proj.weight",
                    (cfg.shared_hidden, h),
                    "BF16",
                ),
                (
                    f"{mp}shared_experts.up_proj.weight",
                    (cfg.shared_hidden, h),
                    "BF16",
                ),
                (
                    f"{mp}shared_experts.down_proj.weight",
                    (h, cfg.shared_hidden),
                    "BF16",
                ),
            ])
            for eid in range(cfg.experts):
                ep = f"{mp}experts.{eid}."
                entries.extend([
                    (f"{ep}w1.weight_packed", (eh, lat // 2), "U8"),
                    (f"{ep}w1.weight_scale", (eh, lat // 32), "U8"),
                    (f"{ep}w3.weight_packed", (eh, lat // 2), "U8"),
                    (f"{ep}w3.weight_scale", (eh, lat // 32), "U8"),
                    (f"{ep}w2.weight_packed", (lat, eh // 2), "U8"),
                    (f"{ep}w2.weight_scale", (lat, eh // 32), "U8"),
                ])
        entries.extend([
            (f"{p}self_attn.q_proj.weight", (cfg.heads * cfg.head_dim, h), "BF16"),
            (f"{p}self_attn.k_proj.weight", (cfg.heads * cfg.head_dim, h), "BF16"),
            (f"{p}self_attn.v_proj.weight", (cfg.heads * cfg.head_dim, h), "BF16"),
            (f"{p}self_attn.g_proj.weight", (cfg.heads * cfg.head_dim, h), "BF16"),
            (f"{p}self_attn.b_proj.weight", (cfg.heads, h), "BF16"),
            (
                f"{p}self_attn.q_conv1d.weight",
                (cfg.heads * cfg.head_dim, 1, cfg.conv_size),
                "BF16",
            ),
            (
                f"{p}self_attn.k_conv1d.weight",
                (cfg.heads * cfg.head_dim, 1, cfg.conv_size),
                "BF16",
            ),
            (
                f"{p}self_attn.v_conv1d.weight",
                (cfg.heads * cfg.head_dim, 1, cfg.conv_size),
                "BF16",
            ),
            (f"{p}self_attn.A_log", (cfg.heads,), "BF16"),
            (f"{p}self_attn.dt_bias", (cfg.heads * cfg.head_dim,), "BF16"),
            (f"{p}self_attn.f_a_proj.weight", (128, h), "BF16"),
            (f"{p}self_attn.f_b_proj.weight", (cfg.heads * cfg.head_dim, 128), "BF16"),
            (f"{p}self_attn.o_norm.weight", (cfg.head_dim,), "BF16"),
            (f"{p}self_attn.o_proj.weight", (h, cfg.heads * cfg.head_dim), "BF16"),
        ])
    mla_p = f"{PREFIX}layers.3.self_attn."
    entries.extend([
        (f"{mla_p}q_a_proj.weight", (cfg.q_rank, h), "BF16"),
        (f"{mla_p}kv_a_proj_with_mqa.weight", (cfg.kv_rank + cfg.qk_extra, h), "BF16"),
        (f"{mla_p}q_b_proj.weight", (cfg.heads * 192, cfg.q_rank), "BF16"),
        (f"{mla_p}kv_b_proj.weight", (cfg.heads * 256, cfg.kv_rank), "BF16"),
        (f"{mla_p}g_proj.weight", (cfg.heads * cfg.head_dim, h), "BF16"),
        (f"{mla_p}o_proj.weight", (h, cfg.heads * cfg.head_dim), "BF16"),
        (f"{mla_p}q_a_layernorm.weight", (cfg.q_rank,), "BF16"),
        (f"{mla_p}kv_a_layernorm.weight", (cfg.kv_rank,), "BF16"),
        (f"{PREFIX}output_attn_res_norm.weight", (h,), "BF16"),
        (f"{PREFIX}output_attn_res_proj.weight", (1, h), "F32"),
        (f"{PREFIX}norm.weight", (h,), "BF16"),
        ("language_model.lm_head.weight", (cfg.vocab, h), "BF16"),
    ])
    return entries


def _write_synthetic_k3_config(ckpt_dir: Path, cfg: Config) -> None:
    """Write synthetic Kimi K3 text_config to `config.json`."""
    import json
    hf_json = {
        "text_config": {
            "hidden_size": cfg.dim,
            "routed_expert_hidden_size": cfg.latent,
            "moe_intermediate_size": cfg.expert_hidden,
            "num_shared_experts": cfg.shared_hidden // cfg.expert_hidden,
            "intermediate_size": cfg.dense_hidden,
            "num_experts": cfg.experts,
            "num_experts_per_token": cfg.top_k,
            "num_hidden_layers": cfg.layers,
            "num_attention_heads": cfg.heads,
            "q_lora_rank": cfg.q_rank,
            "kv_lora_rank": cfg.kv_rank,
            "qk_rope_head_dim": cfg.qk_extra,
            "vocab_size": cfg.vocab,
            "attn_res_block_size": cfg.residual_block,
            "rms_norm_eps": cfg.eps,
            "activation_situ_beta": cfg.situ_beta,
            "activation_situ_linear_beta": cfg.situ_linear_beta,
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
                "use_full_rank_gate": True,
                "head_dim": cfg.head_dim,
                "short_conv_kernel_size": cfg.conv_size,
                "gate_lower_bound": cfg.gate_lower_bound,
                "full_attn_layers": [i + 1 for i in cfg.full_attention],
            },
        }
    }
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(hf_json, f)


def _verify_rank_parity(
    ckpt: Checkpoint,
    cfg: Config,
    rank: int,
    dev_tensors: dict[str, torch.Tensor],
    expected_2l: dict[str, tuple[int, ...]],
) -> float:
    """Verify bit-exact parity between streamed tensors and reference loader."""
    worst = 0.0
    first = (cfg.experts // 8) * (rank // 4)
    count = cfg.experts // 8
    part = rank % 4
    gu, gus, dn, ds = ckpt.experts(1, first, count, part=part, parts=4)
    exp_refs = {
        "expert_gu": np.asarray(gu, np.uint32)[None, ...],
        "expert_gus": np.asarray(gus, np.uint8)[None, ...],
        "expert_down": np.asarray(dn, np.uint32)[None, ...],
        "expert_ds": np.asarray(ds, np.uint8)[None, ...],
    }
    for fam, exp_sh in expected_2l.items():
        got = dev_tensors[fam]
        assert tuple(got.shape) == (1, *exp_sh), (rank, fam, got.shape, exp_sh)
        if fam in exp_refs:
            worst = max(worst, _max_bit_diff(got, exp_refs[fam]))
            continue
        l_ids = _family_layers(cfg, fam, SYNTH_NUM_LAYERS)
        if l_ids is None:
            ref = ckpt.dense_selected(fam, 0, rank)
        else:
            ref = np.stack([ckpt.dense_selected(fam, l_id, rank) for l_id in l_ids], 0)
        worst = max(worst, _max_bit_diff(got, ref))
    return worst


def _assert_tpu_bridge_upload(cfg: Config) -> None:
    """Verify TPU device mode delegates directly to `bridge.upload`."""
    fake_bridge = SimpleNamespace(
        upload=mock.Mock(
            side_effect=lambda arr: torch.zeros((1, *arr.shape), dtype=torch.uint8)
        )
    )
    pkg = weight_stream.__package__
    shim_mods = {
        f"{pkg}.runtime_shim": SimpleNamespace(__path__=[]),
        f"{pkg}.runtime_shim.bridge": fake_bridge,
    }
    with mock.patch.dict("sys.modules", shim_mods):
        tpu_streamer = adapter.RankWeightStreamer(0, num_layers=2, device="tpu")
        res = tpu_streamer.consume(
            "model.norm.weight", torch.ones((cfg.dim,), dtype=torch.bfloat16)
        )
        assert res is hooks_mod.WEIGHT_CONSUMED
        assert fake_bridge.upload.call_count == 1
        assert "final_norm" in tpu_streamer.device_tensors


def test_single_copy_weight_consumed_stream(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify 32-rank bit-exact single-copy Kimi K3 streaming and host eviction."""
    cfg = Config()
    spans, data_base = _write_synthetic_k3_checkpoint(tmp_path)
    ckpt = Checkpoint(tmp_path)
    mm = np.memmap(
        tmp_path / "model.safetensors", mode="r+", dtype=np.uint8, offset=data_base
    )
    expected_2l = selected_local_shapes(SYNTH_NUM_LAYERS, cfg.dim, vocab=cfg.vocab)

    with mock.patch.object(
        hooks_mod,
        "_plugin_hook",
        return_value=(adapter.load_tensor, {"load_tensors": True, "kernel_rank": 0}),
    ):
        passthrough = list(
            hooks_mod.load_tensor_stream(
                SimpleNamespace(),
                [
                    (spans[0][0], _tensor_view(mm, *spans[0][1:])),
                    (spans[1][0], _tensor_view(mm, *spans[1][1:])),
                ],
            )
        )
    expected_passthrough_len = 2
    assert len(passthrough) == expected_passthrough_len
    adapter.clear_observed_weights()

    _assert_tpu_bridge_upload(cfg)

    max_diff = 0.0
    max_peak_staging = 0
    elem_bytes: dict[str, int] = {}
    for rank in range(SYNTH_TP_RANKS):
        opts: dict[str, object] = {
            "load_tensors": True,
            "consume_weights": True,
            "kernel_rank": rank,
            "num_layers": SYNTH_NUM_LAYERS,
            "device": "cpu",
        }

        def _raw_iter() -> Iterator[tuple[str, torch.Tensor]]:
            """Yield synthetic checkpoint tensors in shard order."""
            for key, shape, dt, a, b in spans:
                yield key, _tensor_view(mm, shape, dt, a, b)

        with mock.patch.object(
            hooks_mod, "_plugin_hook", return_value=(adapter.load_tensor, dict(opts))
        ):
            yielded = list(hooks_mod.load_tensor_stream(SimpleNamespace(), _raw_iter()))
        assert yielded == []

        stats = adapter.streamed_staging_stats(rank)
        assert stats is not None
        assert stats.peak_in_flight_layers == 1
        assert stats.active_host_staging_bytes == 0
        max_peak_staging = max(max_peak_staging, stats.peak_host_staging_bytes)

        dev_tensors = adapter.streamed_weights_snapshot(rank)
        if rank == 0:
            elem_bytes = {k: v.element_size() for k, v in dev_tensors.items()}
            fake = fake_runner(
                hf_config=_kimi_hf_config(),
                block_size=EXPECTED_BF16_BLOCK_SIZE,
                token_paddings=(1, 4),
            )
            view = RunnerView(fake, tp_rank=0)
            prog = create(view, {**opts, "buckets": (1, 4)})
            assert not isinstance(prog, LayoutDecline)
            assert hasattr(prog, "streamed_weights")
            assert set(prog.streamed_weights) == set(expected_2l)
            assert adapter.streamed_weights_snapshot(0) == {}
        rank_diff = _verify_rank_parity(ckpt, cfg, rank, dev_tensors, expected_2l)
        max_diff = max(max_diff, rank_diff)
        adapter.clear_streamed_weights(rank)

    assert max_diff == 0.0
    shapes_93 = adapter.RankWeightStreamer(
        0, num_layers=cfg.layers, device="cpu"
    ).expected_shapes
    total_93_bytes = sum(
        int(np.prod((1, *sh))) * elem_bytes[fam] for fam, sh in shapes_93.items()
    )
    gib = float(1 << 30)
    hbm_gib_per_tc = total_93_bytes / gib
    staging_gib_per_tc = max_peak_staging / gib
    assert hbm_gib_per_tc > 0
    assert staging_gib_per_tc > 0


def test_consume_weights_declined_raises_error() -> None:
    """Verify consume_weights=True raises MegakernelConfigError when declined."""
    fake = fake_runner(
        hf_config=_kimi_hf_config(),
        block_size=EXPECTED_BF16_BLOCK_SIZE,
        token_paddings=(1, 4),
    )
    view = RunnerView(fake, tp_rank=0)
    opts = {
        "load_tensors": True,
        "consume_weights": True,
        "kernel_rank": 0,
        "buckets": (),
        "verify_buckets": (),
    }
    with pytest.raises(MegakernelConfigError, match="cannot decline megakernel"):
        create(view, opts)


def test_consume_weights_incomplete_stream_raises_error() -> None:
    """Verify create raises MegakernelConfigError when stream has missing tensors."""
    adapter.clear_streamed_weights(0)
    fake = fake_runner(
        hf_config=_kimi_hf_config(),
        block_size=EXPECTED_BF16_BLOCK_SIZE,
        token_paddings=(1, 4),
    )
    view = RunnerView(fake, tp_rank=0)
    opts = {
        "load_tensors": True,
        "consume_weights": True,
        "kernel_rank": 0,
        "num_layers": 2,
        "buckets": (1, 4),
    }
    with pytest.raises(
        MegakernelConfigError, match="streaming weight verification failed|no weights"
    ):
        create(view, opts)


def test_consume_weights_without_load_tensors_raises_value_error() -> None:
    """Verify consume_weights=True without load_tensors=True raises ValueError."""
    with pytest.raises(
        ValueError, match="consume_weights=True requires load_tensors=True"
    ):
        adapter.load_tensor(
            "model.norm.weight",
            torch.ones((64,), dtype=torch.bfloat16),
            {"consume_weights": True, "load_tensors": False},
        )


def test_resolve_kernel_position_rank_keys() -> None:
    """Verify rank position resolution across explicit, device_position, and default."""
    assert adapter._resolve_kernel_position({"kernel_rank": 7}) == 7
    assert adapter._resolve_kernel_position({"device_position": 12}) == 12
    assert adapter._resolve_kernel_position({}, default_pos=3) == 3
    with pytest.raises(ValueError, match="position must be in 0..31"):
        adapter._resolve_kernel_position({"kernel_rank": 32})
