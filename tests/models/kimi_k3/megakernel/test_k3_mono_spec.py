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

import contextlib
import json
import re
from types import SimpleNamespace
from typing import TYPE_CHECKING, Protocol, cast
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.experimental import topologies
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.megakernel.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.runtime.contract import (
    DOMAIN_KERNEL_MESH,
    PAR_DP,
    LayoutDecline,
)
from vllm_torchtpu.megakernel.runtime.stats import BatchShape
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
from vllm_torchtpu.models.kimi_k3.megakernel import (
    SPEC,
    adapter,
    decode,
    draft_mla,
    layout,
    pools,
    weight_stream,
)
from vllm_torchtpu.models.kimi_k3.megakernel.adapter import create
from vllm_torchtpu.models.kimi_k3.megakernel.kernels import (
    Config,
    decode_megakernel,
    dspark_mla,
    dspark_mla_kernel,
    paged,
)
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
    from collections.abc import Callable, Iterator
    from pathlib import Path

_ROUTE_LAZY_LOGGERS = cast("Callable[[object], None]", route_lazy_loggers)

PLUGIN_MODULE = "vllm_torchtpu.models.kimi_k3.megakernel"
EXPECTED_BF16_BLOCK_SIZE = 256
EXPECTED_FP8_BLOCK_SIZE = 512
EXPECTED_STREAMED_WEIGHTS = 2
EXPECTED_RANK_COUNT = 2
EXPECTED_ATTENTION_DP4 = 4
EXPECTED_POOL_BLOCKS = 4
MIN_WARMED_BUCKETS = 2
EXPECTED_CAPTURED_SHARDS = 3
EXPECTED_POS_7 = 7
EXPECTED_POS_12 = 12
EXPECTED_POS_3 = 3


class _MonkeyPatchLike(Protocol):
    """Minimal subset of `pytest.MonkeyPatch` used by unit tests."""

    def setattr(
        self,
        target: object,
        name: object,
        value: object = ...,
        *,
        raising: bool = ...,
    ) -> None:
        """Patch attribute on `target`."""

    def setenv(self, name: str, value: str, *, prepend: str | None = None) -> None:
        """Set environment variable `name`."""

    def delenv(self, name: str, *, raising: bool = True) -> None:
        """Delete environment variable `name`."""


@contextlib.contextmanager
def _assert_raises(exc_type: type[BaseException], match: str) -> Iterator[None]:
    """Assert that the block raises `exc_type` matching regex `match`."""
    caught: BaseException | None = None
    try:
        yield
    except exc_type as exc:
        caught = exc
    if caught is None:
        msg = f"Expected {exc_type.__name__} matching {match!r}"
        raise AssertionError(msg)
    assert re.search(match, str(caught)) is not None, (
        f"Expected {match!r} in {caught!r}"
    )


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
    _assert_bind_layout_checks()


def _assert_bind_layout_checks() -> None:
    """Verify `bind()` succeeds after `layout_request` and rejects misalignment."""
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
    with _assert_raises(LayoutDecline, "unaligned_cache_layout"):
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
    with _assert_raises(LayoutDecline, "unaligned_cache_layout"):
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
    assert isinstance(dec_tpu, adapter.KimiK3MonoProgram)
    assert dec_tpu.is_tpu is True
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
    assert isinstance(dec_dp4, adapter.KimiK3MonoProgram)
    assert dec_dp4.attention_dp == EXPECTED_ATTENTION_DP4

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
    with _assert_raises(RuntimeError, "CPU-only"):
        step_fn(meta_ids, meta_pos)


def test_k3_mono_program_cpu_e2e(monkeypatch: _MonkeyPatchLike) -> None:
    """Verify CPU end-to-end decode, T=8 verify, SlotMap lifecycle, and declines."""
    _ROUTE_LAZY_LOGGERS(monkeypatch)
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
        "verify_commit_hook_wired": True,
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
    with _assert_raises(AssertionError, "unaligned_cache_layout"):
        run_cpu_e2e(
            PLUGIN_MODULE,
            bad_layout_runner,
            [(ids_decode, pos_decode)],
            metadata=decode_md,
            options=opts,
        )


def test_k3_tpu_device_declines_runner_lifecycle_unwired() -> None:
    """Verify TPU device creates KimiK3MonoProgram with model and declines without."""
    tpu_runner = fake_runner(
        hf_config=_kimi_hf_config(),
        tp_size=32,
        device=SimpleNamespace(type="tpu"),
        model=object(),
    )
    prog = create(RunnerView(tpu_runner, tp_rank=0), {})
    assert isinstance(prog, adapter.KimiK3MonoProgram)
    assert prog.is_tpu is True

    no_model_runner = fake_runner(
        hf_config=_kimi_hf_config(),
        tp_size=32,
        device=SimpleNamespace(type="tpu"),
        model=None,
    )
    dec = create(RunnerView(no_model_runner, tp_rank=0), {})
    assert isinstance(dec, adapter.Decline)
    assert dec.code == "backbone_not_found"


def test_k3_mono_program_two_ranks_kernel_mesh(
    tmp_path: Path, monkeypatch: _MonkeyPatchLike
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
    monkeypatch: _MonkeyPatchLike,
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

    with _assert_raises(ValueError, r"kernel_rank position must be in 0\.\.31"):
        adapter.load_tensor(
            "model.layers.0.mlp.experts.w13_weight_packed",
            stacked_3d,
            {"load_tensors": True, "kernel_rank": 32},
        )

    with _assert_raises(ValueError, "kernel_rank must be an integer"):
        adapter.load_tensor(
            "model.layers.0.mlp.experts.w13_weight_packed",
            stacked_3d,
            {"load_tensors": True, "kernel_rank": [1]},
        )

    # Without kernel_rank when global JAX devices != 32, _default_kernel_position raises
    monkeypatch.setattr(adapter.jax, "devices", lambda: [SimpleNamespace(id=0)])
    with _assert_raises(ValueError, "requires 32 global JAX devices"):
        adapter.load_tensor(
            "model.layers.0.mlp.experts.w13_weight_packed",
            stacked_3d,
            {"load_tensors": True},
        )

    # Indivisible EP or ETP dimensions raise ValueError naming tensor and divisor
    bad_ep_3d = torch.zeros((7, 8, 16), dtype=torch.uint8)
    with _assert_raises(ValueError, "not divisible by EP=8"):
        adapter.load_tensor(
            "model.layers.0.mlp.experts.w13_weight_packed",
            bad_ep_3d,
            {"load_tensors": True, "kernel_rank": 0},
        )

    bad_etp_3d = torch.zeros((8, 6, 16), dtype=torch.uint8)
    with _assert_raises(ValueError, "not divisible by ETP=4"):
        adapter.load_tensor(
            "model.layers.0.mlp.experts.w13_weight_packed",
            bad_etp_3d,
            {"load_tensors": True, "kernel_rank": 0},
        )

    bad_etp_2d = torch.zeros((32, 6), dtype=torch.uint8)
    with _assert_raises(ValueError, "not divisible by ETP=4"):
        adapter.load_tensor(
            "model.layers.0.mlp.experts.w2_weight_packed",
            bad_etp_2d,
            {"load_tensors": True, "kernel_rank": 0},
        )


def test_k3_forward_pads_sub_bucket_tokens_and_dp_slot_allocation(
    monkeypatch: _MonkeyPatchLike,
) -> None:
    """Verify sub-bucket padding, ineligible steps, and DP SlotMap allocation."""
    _ROUTE_LAZY_LOGGERS(monkeypatch)
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
        steps={(1, 2): lambda x, _p: x, (1, 4): lambda x, _p: x},
        num_tokens=8,
        inspect=lambda _m, _k: BatchShape(
            eligible=True, num_reqs=8, tokens_per_seq=1, reason="ok"
        ),
    )
    dec_ineligible = prog.decide(ctx_8, fwd_ineligible)
    assert dec_ineligible is not None
    assert not dec_ineligible.eligible


def test_k3_verify_buckets_validation_valid() -> None:
    """Verify valid string and list `verify_buckets` forms resolve multiples of 8."""
    cpu_runner = fake_runner(
        hf_config=_kimi_hf_config(),
        block_size=EXPECTED_BF16_BLOCK_SIZE,
    )
    for raw_verify, expected in (
        ("8,16,72", (8, 16, 72)),
        ([8, 16, 72], (8, 16, 72)),
        ("none", ()),
        ([], ()),
    ):
        prog = create(
            RunnerView(cpu_runner, tp_rank=0),
            {"buckets": [2], "verify_buckets": raw_verify},
        )
        assert not isinstance(prog, adapter.Decline)
        assert prog.verify_buckets == expected


def test_k3_verify_buckets_validation_invalid() -> None:
    """Verify invalid `verify_buckets` forms raise `ValueError`."""
    cpu_runner = fake_runner(
        hf_config=_kimi_hf_config(),
        block_size=EXPECTED_BF16_BLOCK_SIZE,
    )
    for bad_verify in ("12", [12], "0", [0], "8,12", [8, 12], "not_an_int", 123):
        with _assert_raises(ValueError, "verify_buckets"):
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
    with (ckpt_dir / "model.safetensors.index.json").open("w", encoding="utf-8") as f:
        json.dump(index_json, f)
    return spans, data_base


def _write_safetensors_file(
    path: Path, entries: list[tuple[str, tuple[int, ...], str]]
) -> tuple[list[tuple[str, tuple[int, ...], str, int, int]], int]:
    """Serialize entry metadata and contiguous payload into a valid safetensors file."""
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
    data_base = 8 + hdr_len
    with path.open("wb") as f:
        f.write(int(hdr_len).to_bytes(8, "little"))
        f.write(hdr_bytes)
        f.truncate(data_base + cur)
    mm = np.memmap(path, mode="r+", dtype=np.uint8, offset=data_base)
    for name, _sh, dt, a, b in spans:
        sz = b - a
        if "weight_scale" in name or "e8m0" in name:
            mm[a:b] = 128
        elif dt == "BF16":
            sample_elems = min(sz // 2, 16384)
            mm[a : a + sample_elems * 2].view(np.uint16)[:] = np.arange(
                sample_elems, dtype=np.uint16
            )
    mm.flush()
    return spans, data_base


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
        entries.extend(
            [
                (f"{p}input_layernorm.weight", (h,), "BF16"),
                (f"{p}post_attention_layernorm.weight", (h,), "BF16"),
                (f"{p}self_attention_res_norm.weight", (h,), "BF16"),
                (f"{p}self_attention_res_proj.weight", (1, h), "F32"),
                (f"{p}mlp_res_norm.weight", (h,), "BF16"),
                (f"{p}mlp_res_proj.weight", (1, h), "F32"),
            ]
        )
        if l_id == 0:
            entries.extend(
                [
                    (f"{p}mlp.gate_proj.weight", (cfg.dense_hidden, h), "BF16"),
                    (f"{p}mlp.up_proj.weight", (cfg.dense_hidden, h), "BF16"),
                    (f"{p}mlp.down_proj.weight", (h, cfg.dense_hidden), "BF16"),
                ]
            )
        else:
            mp = f"{p}block_sparse_moe."
            entries.extend(
                [
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
                ]
            )
            for eid in range(cfg.experts):
                ep = f"{mp}experts.{eid}."
                entries.extend(
                    [
                        (f"{ep}w1.weight_packed", (eh, lat // 2), "U8"),
                        (f"{ep}w1.weight_scale", (eh, lat // 32), "U8"),
                        (f"{ep}w3.weight_packed", (eh, lat // 2), "U8"),
                        (f"{ep}w3.weight_scale", (eh, lat // 32), "U8"),
                        (f"{ep}w2.weight_packed", (lat, eh // 2), "U8"),
                        (f"{ep}w2.weight_scale", (lat, eh // 32), "U8"),
                    ]
                )
        entries.extend(
            [
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
                (
                    f"{p}self_attn.f_b_proj.weight",
                    (cfg.heads * cfg.head_dim, 128),
                    "BF16",
                ),
                (f"{p}self_attn.o_norm.weight", (cfg.head_dim,), "BF16"),
                (f"{p}self_attn.o_proj.weight", (h, cfg.heads * cfg.head_dim), "BF16"),
            ]
        )
    mla_p = f"{PREFIX}layers.3.self_attn."
    entries.extend(
        [
            (f"{mla_p}q_a_proj.weight", (cfg.q_rank, h), "BF16"),
            (
                f"{mla_p}kv_a_proj_with_mqa.weight",
                (cfg.kv_rank + cfg.qk_extra, h),
                "BF16",
            ),
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
        ]
    )
    return entries


def _write_synthetic_k3_config(ckpt_dir: Path, cfg: Config) -> None:
    """Write synthetic Kimi K3 text_config to `config.json`."""
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
    with (ckpt_dir / "config.json").open("w", encoding="utf-8") as f:
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


def test_single_copy_weight_consumed_stream(tmp_path: Path) -> None:
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
    for rank in (0, 15, 31):
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
    with _assert_raises(MegakernelConfigError, "cannot decline megakernel"):
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
    with _assert_raises(
        MegakernelConfigError, r"streaming weight verification failed|no weights"
    ):
        create(view, opts)


def test_consume_weights_without_load_tensors_raises_value_error() -> None:
    """Verify consume_weights=True without load_tensors=True raises ValueError."""
    with _assert_raises(ValueError, "consume_weights=True requires load_tensors=True"):
        adapter.load_tensor(
            "model.norm.weight",
            torch.ones((64,), dtype=torch.bfloat16),
            {"consume_weights": True, "load_tensors": False},
        )


def test_resolve_kernel_position_rank_keys() -> None:
    """Verify rank position resolution across explicit, device_position, and default."""
    assert adapter.resolve_kernel_position({"kernel_rank": 7}) == EXPECTED_POS_7
    assert adapter.resolve_kernel_position({"device_position": 12}) == EXPECTED_POS_12
    assert adapter.resolve_kernel_position({}, default_pos=3) == EXPECTED_POS_3
    with _assert_raises(ValueError, r"position must be in 0\.\.31"):
        adapter.resolve_kernel_position({"kernel_rank": 32})


AOT_SYNTHETIC_LAYERS = 4
AOT_SYNTHETIC_KDA_LAYERS = 3
AOT_DECODE_BATCH = 4
AOT_SCATTER_BATCH = 8
AOT_NUM_SLOTS = 8
AOT_NUM_BLOCKS = 16
AOT_PAGE_WIDTH = 8
K3_HIDDEN_DIM = 7168
KDA_SSM_ROW_COUNT = 77
KDA_CONV_ROW_COUNT = 4
U32_WEIGHT_KEYS = frozenset({"expert_gu", "expert_down"})
U8_WEIGHT_KEYS = frozenset({"expert_gus", "expert_ds"})
F32_WEIGHT_KEYS = frozenset(
    {
        "router_bias",
        "latent_norm",
        "k_a_log",
        "k_dt",
        "attn_res",
        "mlp_res",
        "output_res",
    }
)


def _synthetic_decode_weight_structs(
    layers: int = AOT_SYNTHETIC_LAYERS,
    attention_dp: int = 1,
) -> dict[str, jax.ShapeDtypeStruct]:
    """Build per-rank ShapeDtypeStruct decoder weights for deviceless AOT lowering."""
    shapes = selected_local_shapes(layers=layers)
    out: dict[str, jax.ShapeDtypeStruct] = {}
    for name, base_shp in shapes.items():
        shp = base_shp
        if attention_dp > 1 and name in decode_megakernel.ATTENTION_DP_WEIGHTS:
            _kind, axis = decode_megakernel.ATTENTION_DP_WEIGHTS[name]
            if axis is None:
                shp = (shp[0], attention_dp, *shp[1:])
            else:
                cat_ax = 1 + axis
                shp = (
                    *shp[:cat_ax],
                    shp[cat_ax] * attention_dp,
                    *shp[cat_ax + 1 :],
                )
        if name in U32_WEIGHT_KEYS:
            dtype = jnp.uint32
        elif name in U8_WEIGHT_KEYS:
            dtype = jnp.uint8
        elif name in F32_WEIGHT_KEYS:
            dtype = jnp.float32
        else:
            dtype = jnp.bfloat16
        out[name] = jax.ShapeDtypeStruct(shp, dtype)
    return out


def _write_cpu_kda_block(
    raw_pool: torch.Tensor,
    block_id: int,
    conv_active: torch.Tensor,
    recurrent_state: torch.Tensor,
) -> None:
    """Write 77 FP32 recurrent rows and 4 BF16 conv rows into a CPU pool block."""
    raw_bytes = raw_pool[block_id].view(torch.uint8)
    ssm_flat = raw_bytes[:KDA_SSM_ROW_COUNT].view(torch.float32).reshape(-1, 128)
    ssm_flat[: 3 * 128].copy_(recurrent_state.reshape(3 * 128, 128))
    conv_flat = (
        raw_bytes[KDA_SSM_ROW_COUNT : KDA_SSM_ROW_COUNT + KDA_CONV_ROW_COUNT]
        .view(torch.bfloat16)
        .reshape(-1)
    )
    conv_flat[: 3 * 3 * 3 * 128].copy_(conv_active.reshape(-1))


def _assert_return_hidden_aot_lowering(mesh_tp: Mesh) -> None:
    """Assert deviceless TPU v7x return_hidden AOT lowering for local and paged."""
    weight_structs = _synthetic_decode_weight_structs(AOT_SYNTHETIC_LAYERS)
    pool_map = tuple(range(pools.MLA_LAYERS))
    pool_structs = tuple(
        jax.ShapeDtypeStruct((AOT_NUM_BLOCKS, 128, 2, 640), jnp.bfloat16)
        for _ in range(pools.MLA_LAYERS)
    )
    conv_b4 = jax.ShapeDtypeStruct(
        (AOT_DECODE_BATCH, AOT_SYNTHETIC_KDA_LAYERS, 3, 3, layout.CONV_HEAD_ROWS, 128),
        jnp.bfloat16,
    )
    rec_b4 = jax.ShapeDtypeStruct(
        (AOT_DECODE_BATCH, AOT_SYNTHETIC_KDA_LAYERS, 3, 128, 128),
        jnp.float32,
    )
    toks_b4 = jax.ShapeDtypeStruct((AOT_DECODE_BATCH,), jnp.int32)

    def _local_b4_hidden(
        w: dict[str, jax.Array],
        c: jax.Array,
        r: jax.Array,
        ps: tuple[jax.Array, ...],
        ids_arr: jax.Array,
    ) -> tuple[jax.Array, ...]:
        """Lower B=4 non-scattered decode_local with return_hidden=True."""
        out = decode_megakernel.decode_local(
            w,
            (c, r, *ps),
            ids_arr,
            ids_arr,
            batch_size=AOT_DECODE_BATCH,
            layers=AOT_SYNTHETIC_LAYERS,
            block_tables=jnp.zeros((AOT_DECODE_BATCH, AOT_PAGE_WIDTH), dtype=jnp.int32),
            dcp_pool_map=pool_map,
            return_hidden=True,
        )
        return (out[0],)

    lowered_local = jax.jit(
        jax.shard_map(
            _local_b4_hidden,
            mesh=mesh_tp,
            in_specs=P(),
            out_specs=P(),
            check_vma=False,
        )
    ).lower(weight_structs, conv_b4, rec_b4, pool_structs, toks_b4)
    assert lowered_local.out_info[0].shape == (AOT_DECODE_BATCH, K3_HIDDEN_DIM)
    assert lowered_local.out_info[0].dtype == jnp.bfloat16
    assert "tpu_custom_call" in lowered_local.as_text()

    conv_slots = jax.ShapeDtypeStruct(
        (AOT_NUM_SLOTS, AOT_SYNTHETIC_KDA_LAYERS, 3, 3, layout.CONV_HEAD_ROWS, 128),
        jnp.bfloat16,
    )
    rec_slots = jax.ShapeDtypeStruct(
        (AOT_NUM_SLOTS, AOT_SYNTHETIC_KDA_LAYERS, 3, 128, 128),
        jnp.float32,
    )
    toks_b8 = jax.ShapeDtypeStruct((AOT_SCATTER_BATCH,), jnp.int32)
    tables_b8 = jax.ShapeDtypeStruct((AOT_SCATTER_BATCH, AOT_PAGE_WIDTH), jnp.int32)

    def _paged_b8_hidden(
        w: dict[str, jax.Array],
        cs: jax.Array,
        rs: jax.Array,
        ps: tuple[jax.Array, ...],
        meta: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array, tuple[jax.Array, ...]]:
        """Lower B=8 scattered decode_paged with return_hidden=True."""
        ids_arr, pos_arr, tbl_arr, slt_arr = meta
        paged_out = paged.decode_paged(
            w,
            cs,
            rs,
            ps,
            ids_arr,
            pos_arr,
            tbl_arr,
            slt_arr,
            batch_size=AOT_SCATTER_BATCH,
            layers=AOT_SYNTHETIC_LAYERS,
            dcp_pool_map=pool_map,
            attention_token_scatter=True,
            moe_token_scatter_output=True,
            moe_routed_reduce_scatter=True,
            sequence_parallel_pre_attention=True,
            return_hidden=True,
        )
        return paged_out[0], paged_out[-3], paged_out[-2], paged_out[-1]

    lowered_paged = jax.jit(
        jax.shard_map(
            _paged_b8_hidden,
            mesh=mesh_tp,
            in_specs=P(),
            out_specs=P(),
            check_vma=False,
        )
    ).lower(
        weight_structs,
        conv_slots,
        rec_slots,
        pool_structs,
        (toks_b8, toks_b8, tables_b8, toks_b8),
    )
    assert lowered_paged.out_info[0].shape == (AOT_SCATTER_BATCH, K3_HIDDEN_DIM)
    assert lowered_paged.out_info[0].dtype == jnp.bfloat16
    assert "tpu_custom_call" in lowered_paged.as_text()


def _assert_cpu_kda_importer_roundtrip() -> tuple[
    dict[str, torch.Tensor], torch.Tensor
]:
    """Verify CPU KdaImporter 77 FP32 + 4 BF16 row unpack and null-block zeroing."""
    raw_pools = [
        torch.zeros((EXPECTED_POOL_BLOCKS, 128, 2, 640), dtype=torch.bfloat16)
        for _ in range(pools.MLA_LAYERS)
    ]
    raw_pools[0][0].fill_(7.0)
    caches: dict[str, torch.Tensor] = {}
    for idx in range(layout.NUM_LAYERS):
        pool_idx = (
            pools.mla_state_index(idx)
            if pools.is_mla_layer(idx)
            else pools.kda_state_index(idx) // 3
        )
        caches[f"model.layers.{idx}.self_attn"] = raw_pools[pool_idx]

    expected_rec = torch.arange(
        pools.KDA_LAYERS * 3 * 128 * 128, dtype=torch.float32
    ).reshape(pools.KDA_LAYERS, 3, 128, 128) * (1.0 / 65536.0)
    expected_conv = (
        torch.arange(pools.KDA_LAYERS * 3 * 3 * 3 * 128, dtype=torch.float32)
        .remainder(127.0)
        .to(torch.bfloat16)
        .reshape(pools.KDA_LAYERS, 3, 3, 3, 128)
    )
    kda_blocks = torch.zeros((pools.KDA_LAYERS,), dtype=torch.int32)
    for k_idx in range(pools.KDA_LAYERS):
        blk = 0 if k_idx == 0 else (k_idx % 3) + 1
        kda_blocks[k_idx] = blk
        if blk != 0:
            _write_cpu_kda_block(
                raw_pools[k_idx // 3], blk, expected_conv[k_idx], expected_rec[k_idx]
            )

    cpu_mesh = Mesh(jax.devices("cpu")[:1], ("tp",))
    importer = pools.KdaImporter(
        mesh=cpu_mesh, attention_dp=1, num_slots=EXPECTED_POOL_BLOCKS
    )
    assert importer.bind(caches, "bfloat16") == EXPECTED_POOL_BLOCKS
    unpacked_conv, unpacked_rec = pools.read_kda_states(
        importer.raw_pools, importer.kda_raw, kda_blocks
    )
    assert float(torch.max(torch.abs(unpacked_rec[0])).item()) == 0.0
    assert float(torch.max(torch.abs(unpacked_conv[0])).item()) == 0.0
    assert (
        float(torch.max(torch.abs(unpacked_rec[1:] - expected_rec[1:])).item()) == 0.0
    )
    conv_diff = torch.max(
        torch.abs(unpacked_conv[1:, :, :, :3, :].float() - expected_conv[1:].float())
    ).item()
    assert conv_diff == 0.0
    assert (
        float(torch.max(torch.abs(unpacked_conv[:, :, :, 3, :].float())).item()) == 0.0
    )

    conv_bank = torch.zeros(
        (1, EXPECTED_POOL_BLOCKS, pools.KDA_LAYERS, 3, 3, layout.CONV_HEAD_ROWS, 128),
        dtype=torch.bfloat16,
    )
    rec_bank = torch.zeros(
        (1, EXPECTED_POOL_BLOCKS, pools.KDA_LAYERS, 3, 128, 128), dtype=torch.float32
    )
    conv_bank, rec_bank = importer.import_state(conv_bank, rec_bank, kda_blocks, slot=2)
    assert (
        float(torch.max(torch.abs(rec_bank[0, 2, 1:] - expected_rec[1:])).item()) == 0.0
    )
    assert float(torch.max(torch.abs(rec_bank[0, 1])).item()) == 0.0

    importer_dp2 = pools.KdaImporter(mesh=cpu_mesh, attention_dp=2, num_slots=9)
    assert importer_dp2.bind(caches, "bfloat16") == EXPECTED_POOL_BLOCKS
    conv_dp2 = torch.zeros(
        (1, 10, pools.KDA_LAYERS, 3, 3, layout.CONV_HEAD_ROWS, 128),
        dtype=torch.bfloat16,
    )
    rec_dp2 = torch.zeros((1, 10, pools.KDA_LAYERS, 3, 128, 128), dtype=torch.float32)
    conv_dp2, rec_dp2 = importer_dp2.import_state(conv_dp2, rec_dp2, kda_blocks, slot=5)
    assert float(torch.max(torch.abs(rec_dp2[0, 0])).item()) == 0.0
    assert float(torch.max(torch.abs(conv_dp2[0, 0].float())).item()) == 0.0
    return caches, kda_blocks


NON_SP_LOGITS_BF16_ATOL = 5e-2
SYNTHETIC_VOCAB_SHARD = 64


def _assert_return_hidden_numeric_parity() -> None:
    """Verify numeric parity between `return_hidden=True` and `logits` projection."""
    cpu_dev = jax.devices("cpu")[0]
    with jax.default_device(cpu_dev):
        rng = np.random.default_rng(42)
        prefix_b4 = jnp.asarray(
            rng.standard_normal((AOT_DECODE_BATCH, K3_HIDDEN_DIM), dtype=np.float32)
            * 0.125
        )
        blocks_b4 = jnp.asarray(
            rng.standard_normal((AOT_DECODE_BATCH, 2, K3_HIDDEN_DIM), dtype=np.float32)
            * 0.125
        )
        weights = {
            "output_res": jnp.asarray(
                rng.standard_normal((1, 2, K3_HIDDEN_DIM), dtype=np.float32) * 0.125
            ),
            "final_norm": jnp.ones((K3_HIDDEN_DIM,), dtype=jnp.bfloat16),
            "lm_head": jnp.asarray(
                rng.standard_normal(
                    (K3_HIDDEN_DIM, SYNTHETIC_VOCAB_SHARD), dtype=np.float32
                )
                * 0.0625,
                dtype=jnp.bfloat16,
            ),
        }
        hid_b4 = decode_megakernel.finish_decoder_outputs(
            prefix_b4,
            blocks_b4,
            weights,
            return_hidden=True,
            options=(AOT_SYNTHETIC_LAYERS, 1e-5, False, False),
        )
        logits_b4 = decode_megakernel.finish_decoder_outputs(
            prefix_b4,
            blocks_b4,
            weights,
            return_hidden=False,
            options=(AOT_SYNTHETIC_LAYERS, 1e-5, False, False),
        )
        proj_b4 = jnp.dot(
            hid_b4, weights["lm_head"], preferred_element_type=jnp.float32
        )
        assert hid_b4.dtype == jnp.bfloat16
        assert float(jnp.max(jnp.abs(proj_b4 - logits_b4))) < NON_SP_LOGITS_BF16_ATOL

        prefix_b8 = jnp.asarray(
            rng.standard_normal((AOT_SCATTER_BATCH, K3_HIDDEN_DIM), dtype=np.float32)
            * 0.125,
            dtype=jnp.bfloat16,
        )
        blocks_b8 = jnp.zeros((AOT_SCATTER_BATCH, 2, K3_HIDDEN_DIM), dtype=jnp.bfloat16)
        for batched_bounds in (False, True):
            hid_b8 = decode_megakernel.finish_decoder_outputs(
                prefix_b8,
                blocks_b8,
                weights,
                return_hidden=True,
                options=(AOT_SYNTHETIC_LAYERS, 1e-5, True, batched_bounds),
            )
            logits_b8 = decode_megakernel.finish_decoder_outputs(
                prefix_b8,
                blocks_b8,
                weights,
                return_hidden=False,
                options=(AOT_SYNTHETIC_LAYERS, 1e-5, True, batched_bounds),
            )
            proj_b8 = jnp.dot(
                hid_b8, weights["lm_head"], preferred_element_type=jnp.float32
            )
            max_diff = float(jnp.max(jnp.abs(proj_b8 - logits_b8)))
            assert hid_b8.dtype == jnp.bfloat16
            if batched_bounds:
                assert max_diff == 0.0
            else:
                scan_gemv_fp32_atol = K3_HIDDEN_DIM * float(np.finfo(np.float32).eps)
                assert max_diff < scan_gemv_fp32_atol


def test_kimi_k3_mono_program_return_hidden_and_kda_slot() -> None:
    """Verify TPU v7x return_hidden lowering, KdaImporter CPU unpack, and bind/build."""
    assert pools.SSM_ROWS == KDA_SSM_ROW_COUNT
    assert pools.CONV_ROWS == KDA_CONV_ROW_COUNT

    devices = topologies.get_topology_desc(
        topology_name="tpu7x_2x2x4", platform="tpu"
    ).devices
    mesh_tp = Mesh(devices, ("tp",))
    _assert_return_hidden_aot_lowering(mesh_tp)
    _assert_return_hidden_numeric_parity()
    caches, kda_blocks = _assert_cpu_kda_importer_roundtrip()

    tpu_runner = fake_runner(
        hf_config=_kimi_hf_config(),
        tp_size=32,
        block_size=EXPECTED_BF16_BLOCK_SIZE,
        device=SimpleNamespace(type="tpu"),
        model=object(),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=32,
            decode_context_parallel_size=16,
            cp_kv_cache_interleave_size=EXPECTED_BF16_BLOCK_SIZE,
        ),
    )
    tpu_runner.vllm_config.parallel_config = SimpleNamespace(
        tensor_parallel_size=32,
        decode_context_parallel_size=16,
        cp_kv_cache_interleave_size=EXPECTED_BF16_BLOCK_SIZE,
    )
    tpu_runner.vllm_config.compilation_config = SimpleNamespace(
        static_forward_context=caches
    )
    prog_tpu = create(
        RunnerView(tpu_runner, tp_rank=0),
        {"attention_dp": 2, "max_slots": 9, "buckets": [4, 8]},
    )
    assert isinstance(prog_tpu, adapter.KimiK3MonoProgram)
    assert prog_tpu.is_tpu is True
    prog_tpu.load(mesh_tp)
    bind_info = prog_tpu.bind(RunnerView(tpu_runner, tp_rank=0))
    assert bind_info["num_blocks"] == EXPECTED_POOL_BLOCKS
    assert prog_tpu.kda_importer is not None

    slots_g0 = prog_tpu.admit_requests(["dp0-a", "dp0-b"], group=0)
    slots_g1 = prog_tpu.admit_requests(["dp1-a"], group=1)
    assert prog_tpu.slot_map.group("dp0-a") == 0
    assert prog_tpu.slot_map.group("dp1-a") == 1
    assert set(slots_g0).isdisjoint(set(slots_g1))
    prog_tpu.import_kda_state("dp0-a", kda_blocks)
    prog_tpu.admit_requests(["dp0-b"], finished_req_ids=["dp0-a"], group=0)
    assert "dp0-a" not in prog_tpu.slot_map

    tpu_step = prog_tpu.build(Bucket(1, 8, 16))
    assert callable(tpu_step)


VERIFY_ROWS = 8
VERIFY_BATCHED_ROWS = 16
DRAFT_MLA_LAYER_COUNT = 5
DRAFT_VECTOR_WIDTH = 128
DRAFT_LATENT_WIDTH = 640
ACCEPTED_REQ0 = 3
ACCEPTED_REQ1 = 5
EXPECTED_REQ1_SOURCE_ROW = 13


def _synthetic_mla_draft_weight_structs(
    cfg: dspark_mla.MlaDraftConfig,
) -> dict[str, jax.ShapeDtypeStruct]:
    """Build per-rank ShapeDtypeStruct MLA DSpark draft weights for AOT lowering."""
    lay = cfg.layers
    hid = cfg.hidden
    bf16 = jnp.bfloat16
    spec_map: dict[str, tuple[tuple[int, ...], jnp.dtype]] = {
        "ln1": ((lay, 1, hid), bf16),
        "ln2": ((lay, 1, hid), bf16),
        "q_norm": ((lay, 1, cfg.q_lora), bf16),
        "kv_norm": ((lay, 1, DRAFT_LATENT_WIDTH), bf16),
        "context_norm": ((1, hid), bf16),
        "final_norm": ((1, hid), bf16),
        "mask_embed": ((1, hid), bf16),
        "rot": ((cfg.rope, cfg.rope), bf16),
        "qkv_tile": ((lay, hid, DRAFT_VECTOR_WIDTH), bf16),
        "context_tile": ((hid, DRAFT_VECTOR_WIDTH), bf16),
        "q_b": ((lay, cfg.q_lora, 512), bf16),
        "w_uk": ((lay, 2, cfg.nope, cfg.kv_lora), bf16),
        "w_uv": ((lay, 2, cfg.kv_lora, cfg.v_head), bf16),
        "o": ((lay, 2 * cfg.v_head, hid), bf16),
        "gate": ((lay, hid, 512), bf16),
        "up": ((lay, hid, 512), bf16),
        "down": ((lay, 512, hid), bf16),
        "fc": ((dspark_mla.FC_IN, dspark_mla.CHUNK), bf16),
        "w2t": ((cfg.markov_rank, cfg.vocab // dspark_mla.RANKS), bf16),
        "w1": ((cfg.vocab, cfg.markov_rank), bf16),
        "rope": ((512, 2 * cfg.rope), jnp.float32),
        "lm_head": ((hid, cfg.vocab // dspark_mla.RANKS), bf16),
        "embed": ((cfg.vocab // dspark_mla.RANKS, hid), bf16),
    }
    return {
        name: jax.ShapeDtypeStruct(*spec_map[name])
        for name in dspark_mla_kernel.WEIGHT_NAMES
    }


def _assert_verify8_and_draft_aot_lowering(mesh_tp: Mesh) -> None:
    """Assert deviceless TPU v7x AOT lowering for fused_draft_local and verify8."""
    cfg = dspark_mla.MlaDraftConfig()
    draft_w = _synthetic_mla_draft_weight_structs(cfg)
    draft_pools = tuple(
        jax.ShapeDtypeStruct(
            (AOT_NUM_BLOCKS, 128, 4, DRAFT_LATENT_WIDTH), jnp.float8_e4m3fn
        )
        for _ in range(cfg.layers)
    )
    aux_s = jax.ShapeDtypeStruct(
        (len(cfg.aux_layers), VERIFY_ROWS, cfg.hidden), jnp.bfloat16
    )
    logits_s = jax.ShapeDtypeStruct(
        (1, VERIFY_ROWS, cfg.vocab // dspark_mla.RANKS), jnp.float32
    )
    prev_s = jax.ShapeDtypeStruct((1, DRAFT_VECTOR_WIDTH), jnp.int32)
    pos_s = jax.ShapeDtypeStruct((1,), jnp.int32)
    tbl_s = jax.ShapeDtypeStruct((1, AOT_PAGE_WIDTH), jnp.int32)

    fused_fn = dspark_mla_kernel.fused_draft_local(
        cfg, requests=1, pages=AOT_PAGE_WIDTH, rows=VERIFY_ROWS
    )
    lowered_draft = jax.jit(
        jax.shard_map(
            fused_fn, mesh=mesh_tp, in_specs=P(), out_specs=P(), check_vma=False
        )
    ).lower(draft_w, draft_pools, aux_s, logits_s, prev_s, pos_s, pos_s, pos_s, tbl_s)
    assert lowered_draft.out_info[0].shape == (1, DRAFT_VECTOR_WIDTH)
    assert lowered_draft.out_info[1].shape == (
        DRAFT_MLA_LAYER_COUNT,
        VERIFY_ROWS,
        DRAFT_LATENT_WIDTH,
    )
    assert lowered_draft.out_info[1].dtype == jnp.bfloat16
    assert "tpu_custom_call" in lowered_draft.as_text()

    dec_w = _synthetic_decode_weight_structs(AOT_SYNTHETIC_LAYERS)
    pool_map = tuple(range(pools.MLA_LAYERS))
    pool_s = tuple(
        jax.ShapeDtypeStruct((AOT_NUM_BLOCKS, 128, 2, DRAFT_LATENT_WIDTH), jnp.bfloat16)
        for _ in range(pools.MLA_LAYERS)
    )
    conv_s = jax.ShapeDtypeStruct(
        (AOT_NUM_SLOTS, AOT_SYNTHETIC_KDA_LAYERS, 3, 3, layout.CONV_HEAD_ROWS, 128),
        jnp.bfloat16,
    )
    rec_s = jax.ShapeDtypeStruct(
        (AOT_NUM_SLOTS, AOT_SYNTHETIC_KDA_LAYERS, 3, 128, 128), jnp.float32
    )
    toks_v8 = jax.ShapeDtypeStruct((VERIFY_ROWS,), jnp.int32)
    tbl_v8 = jax.ShapeDtypeStruct((VERIFY_ROWS, AOT_PAGE_WIDTH), jnp.int32)

    def _verify8_and_commit(
        w: dict[str, jax.Array],
        cs: jax.Array,
        rs: jax.Array,
        ps: tuple[jax.Array, ...],
        meta: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Lower T=8 verify8 decode_paged + _finish_verify KDA selected commit."""
        tok_arr, tbl_arr, acc_arr = meta
        paged_out = paged.decode_paged(
            w,
            cs,
            rs,
            ps,
            tok_arr,
            tok_arr,
            tbl_arr,
            tok_arr,
            batch_size=VERIFY_ROWS,
            layers=AOT_SYNTHETIC_LAYERS,
            dcp_pool_map=pool_map,
            sequence_rows=VERIFY_ROWS,
            defer_kda_commit=True,
            attention_token_scatter=True,
            moe_token_scatter_output=True,
            moe_routed_reduce_scatter=True,
            sequence_parallel_pre_attention=True,
            return_hidden=True,
        )
        hid, new_c, new_r = paged_out[0], paged_out[-3], paged_out[-2]
        srcs, tgts = adapter.select_verify_kda_rows(
            tok_arr, acc_arr, requests=1, attention_dp=1, bank_slots=AOT_NUM_SLOTS
        )
        c_out, r_out = adapter.commit_verify_kda_states(
            cs,
            rs,
            new_c,
            new_r,
            (tok_arr, acc_arr),
        )
        _ = (srcs, tgts)
        return hid, c_out, r_out

    lowered_v8 = jax.jit(
        jax.shard_map(
            _verify8_and_commit,
            mesh=mesh_tp,
            in_specs=P(),
            out_specs=P(),
            check_vma=False,
        )
    ).lower(dec_w, conv_s, rec_s, pool_s, (toks_v8, tbl_v8, pos_s))
    assert lowered_v8.out_info[0].shape == (VERIFY_ROWS, K3_HIDDEN_DIM)
    assert "kda_selected_state_commit" in lowered_v8.as_text()


def test_kimi_k3_mono_program_verify8_spec_decode() -> None:
    """Verify speculative_config verify8 wiring, draft pools, and _finish_verify."""
    assert decode.KimiDecoder.WIDTH == VERIFY_ROWS
    assert draft_mla.MlaDsparkDraft.parts(4, dp=2, rows_per=VERIFY_ROWS) == [
        (0, 1, 2, 3)
    ]
    st_cpu = draft_mla.make_draft_state(anchor=11, device=torch.device("cpu"))
    assert st_cpu.vector.shape == (1, DRAFT_VECTOR_WIDTH)
    assert st_cpu.vector.device.type == "cpu"

    devices = topologies.get_topology_desc(
        topology_name="tpu7x_2x2x4", platform="tpu"
    ).devices
    mesh_tp = Mesh(devices, ("tp",))
    _assert_verify8_and_draft_aot_lowering(mesh_tp)

    caches, _ = _assert_cpu_kda_importer_roundtrip()
    for d_idx in range(DRAFT_MLA_LAYER_COUNT):
        layer_num = pools.DRAFT_LAYER0 + d_idx
        caches[f"model.layers.{layer_num}.self_attn"] = torch.zeros(
            (EXPECTED_POOL_BLOCKS, 128, 2, DRAFT_LATENT_WIDTH), dtype=torch.bfloat16
        )

    cpu_runner = fake_runner(
        hf_config=_kimi_hf_config(),
        tp_size=2,
        block_size=EXPECTED_BF16_BLOCK_SIZE,
        token_paddings=(8, 16),
    )
    cpu_runner.vllm_config.speculative_config = SimpleNamespace(
        num_speculative_tokens=7
    )
    prog = create(
        RunnerView(cpu_runner, tp_rank=0),
        {
            "buckets": [4, 8],
            "verify_buckets": [8, 16],
            "max_slots": 8,
            "caches": caches,
        },
    )
    assert isinstance(prog, adapter.KimiK3MonoProgram)
    assert prog.speculative_enabled is True
    assert prog.policy.verify_tokens == (VERIFY_ROWS,)

    bind_info = prog.bind(RunnerView(cpu_runner, tp_rank=0))
    assert bind_info["draft_pools"] == DRAFT_MLA_LAYER_COUNT
    assert len(prog.draft_pools) == DRAFT_MLA_LAYER_COUNT
    assert prog.draft_aux_layers == dspark_mla.MlaDraftConfig().aux_layers

    slots = prog.admit_requests(["v-req-0", "v-req-1"])
    step_v8 = prog.build(Bucket(VERIFY_ROWS, VERIFY_ROWS, VERIFY_BATCHED_ROWS))
    out_v8 = step_v8(
        torch.arange(VERIFY_ROWS, dtype=torch.int32),
        torch.arange(VERIFY_ROWS, dtype=torch.int32),
    )
    assert out_v8.shape == (VERIFY_BATCHED_ROWS, K3_HIDDEN_DIM)

    step_vb = prog.build(Bucket(VERIFY_ROWS, VERIFY_BATCHED_ROWS, VERIFY_BATCHED_ROWS))
    out_vb = step_vb(
        torch.arange(VERIFY_BATCHED_ROWS, dtype=torch.int32),
        torch.arange(VERIFY_BATCHED_ROWS, dtype=torch.int32),
    )
    assert out_vb.shape == (VERIFY_BATCHED_ROWS, K3_HIDDEN_DIM)
    assert prog.recurrent_slots is not None
    assert float(torch.max(torch.abs(prog.recurrent_slots)).item()) == 0.0

    deferred_rec = prog.last_verify_recurrent
    assert deferred_rec is not None
    accepted = torch.tensor([ACCEPTED_REQ0, ACCEPTED_REQ1], dtype=torch.int32)
    _, rec_after = prog.finish_verify(accepted)
    diff0 = torch.max(
        torch.abs(rec_after[0, slots[0]] - deferred_rec[0, ACCEPTED_REQ0])
    ).item()
    diff1 = torch.max(
        torch.abs(rec_after[0, slots[1]] - deferred_rec[0, EXPECTED_REQ1_SOURCE_ROW])
    ).item()
    assert diff0 == 0.0
    assert diff1 == 0.0
    assert float(torch.max(torch.abs(rec_after[0, 0])).item()) == 0.0
    assert prog.last_verify_conv is None
    assert prog.last_verify_recurrent is None


EXPECTED_GLOBAL_POOL_BLOCKS = 128
EXPECTED_DP2_BANK_SLOTS = 10
EXPECTED_VERIFY_REQ_COUNT = 2
EXPECTED_TABLE_WIDTH = 2


def _lower_mono_program_tpu_bodies(
    prog: adapter.KimiK3MonoProgram, mesh_tp: Mesh
) -> None:
    """Lower `KimiK3MonoProgram` TPU decode, verify8, and finish bodies on `tp=32`."""
    local_w = _synthetic_decode_weight_structs(
        AOT_SYNTHETIC_LAYERS, attention_dp=prog.attention_dp
    )
    prog.decoder_weights = {
        name: torch.empty((1,), dtype=torch.bfloat16) for name in local_w
    }
    w_names = sorted(local_w.keys())
    global_w = [
        jax.ShapeDtypeStruct((adapter.TP_SIZE, *local_w[n].shape), local_w[n].dtype)
        for n in w_names
    ]
    conv_g = jax.ShapeDtypeStruct(
        (
            adapter.TP_SIZE,
            EXPECTED_DP2_BANK_SLOTS,
            AOT_SYNTHETIC_KDA_LAYERS,
            3,
            3,
            layout.CONV_HEAD_ROWS,
            128,
        ),
        jnp.bfloat16,
    )
    rec_g = jax.ShapeDtypeStruct(
        (
            adapter.TP_SIZE,
            EXPECTED_DP2_BANK_SLOTS,
            AOT_SYNTHETIC_KDA_LAYERS,
            3,
            128,
            128,
        ),
        jnp.float32,
    )
    pools_g = [
        jax.ShapeDtypeStruct(
            (adapter.TP_SIZE * EXPECTED_POOL_BLOCKS, 128, 2, DRAFT_LATENT_WIDTH),
            jnp.bfloat16,
        )
        for _ in range(len(prog.raw_pools))
    ]

    dec_body, dec_in, dec_out, dec_don, _ = prog.make_tpu_decode_body(
        Bucket(1, AOT_SCATTER_BATCH, VERIFY_BATCHED_ROWS),
        layers=AOT_SYNTHETIC_LAYERS,
    )
    toks_b8 = jax.ShapeDtypeStruct((AOT_SCATTER_BATCH,), jnp.int32)
    tbls_b8 = jax.ShapeDtypeStruct((AOT_SCATTER_BATCH, AOT_PAGE_WIDTH), jnp.int32)
    lowered_dec = jax.jit(
        jax.shard_map(
            dec_body,
            mesh=mesh_tp,
            in_specs=dec_in,
            out_specs=dec_out,
            check_vma=False,
        ),
        donate_argnums=dec_don,
    ).lower(*global_w, conv_g, rec_g, *pools_g, toks_b8, toks_b8, tbls_b8, toks_b8)
    assert lowered_dec.out_info[0].shape == (VERIFY_BATCHED_ROWS, K3_HIDDEN_DIM)
    assert lowered_dec.out_info[3].shape == (
        EXPECTED_GLOBAL_POOL_BLOCKS,
        128,
        2,
        DRAFT_LATENT_WIDTH,
    )

    ver_body, ver_in, ver_out, ver_don, _ = prog.make_tpu_decode_body(
        Bucket(VERIFY_ROWS, VERIFY_BATCHED_ROWS, VERIFY_BATCHED_ROWS),
        layers=AOT_SYNTHETIC_LAYERS,
    )
    toks_v16 = jax.ShapeDtypeStruct((VERIFY_BATCHED_ROWS,), jnp.int32)
    tbls_v16 = jax.ShapeDtypeStruct((VERIFY_BATCHED_ROWS, AOT_PAGE_WIDTH), jnp.int32)
    lowered_ver = jax.jit(
        jax.shard_map(
            ver_body,
            mesh=mesh_tp,
            in_specs=ver_in,
            out_specs=ver_out,
            check_vma=False,
        ),
        donate_argnums=ver_don,
    ).lower(*global_w, conv_g, rec_g, *pools_g, toks_v16, toks_v16, tbls_v16, toks_v16)
    assert lowered_ver.out_info[0].shape == (VERIFY_BATCHED_ROWS, K3_HIDDEN_DIM)
    new_conv_g = jax.ShapeDtypeStruct(
        lowered_ver.out_info[1].shape, lowered_ver.out_info[1].dtype
    )
    new_rec_g = jax.ShapeDtypeStruct(
        lowered_ver.out_info[2].shape, lowered_ver.out_info[2].dtype
    )

    fin_body, fin_in, fin_out, fin_don, _ = prog.make_tpu_finish_body(
        requests=EXPECTED_VERIFY_REQ_COUNT,
        bank_slots=EXPECTED_DP2_BANK_SLOTS,
    )
    acc_s = jax.ShapeDtypeStruct((EXPECTED_VERIFY_REQ_COUNT,), jnp.int32)
    lowered_fin = jax.jit(
        jax.shard_map(
            fin_body,
            mesh=mesh_tp,
            in_specs=fin_in,
            out_specs=fin_out,
            check_vma=False,
        ),
        donate_argnums=fin_don,
    ).lower(conv_g, rec_g, new_conv_g, new_rec_g, toks_v16, acc_s)
    assert "kda_selected_state_commit" in lowered_fin.as_text()


def _make_fake_tpu_bridge(
    recorded_calls: list[tuple[str, tuple[object, ...]]],
) -> type[object]:
    """Create a host-tensor bridge double simulating TPU custom-op donation."""

    class _FakeBridge:
        """Capture registered TPU custom ops and simulate buffer donation."""

        @staticmethod
        def register_op(
            name: str,
            *_specs: object,
            **_kwargs: object,
        ) -> object:
            """Return a callable simulating custom-op execution on host tensors."""

            def _call(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
                """Record call operands and return fresh post-step buffers."""
                recorded_calls.append((name, args))
                if "finish_verify" in name:
                    c_in, r_in, new_c, new_r, _slt, acc = args
                    c_next = c_in.clone()
                    r_next = r_in.clone()
                    if c_in.device.type == "cpu":
                        c_next[0, 1] = new_c[0, int(acc[0].item())]
                        r_next[0, 1] = new_r[0, int(acc[0].item())]
                    return c_next, r_next
                c_in, r_in = args[1], args[2]
                pools_in, toks = args[3:-4], args[-4]
                rows = int(toks.shape[0])
                out_r = int(name.rsplit("_r", 1)[1])
                hid = torch.full((out_r, K3_HIDDEN_DIM), 2.0, dtype=torch.bfloat16)
                fresh_pools = [p.clone().add_(1.0) for p in pools_in]
                if "_t8_" in name:
                    ver_c = (
                        torch.arange(rows, dtype=torch.float32)
                        .view(1, rows, 1, 1, 1, 1, 1)
                        .expand(1, rows, *c_in.shape[2:])
                        .to(torch.bfloat16)
                        .clone()
                    )
                    ver_r = (
                        torch.arange(rows, dtype=torch.float32)
                        .view(1, rows, 1, 1, 1, 1)
                        .expand(1, rows, *r_in.shape[2:])
                        .clone()
                    )
                    return (hid, ver_c, ver_r, *fresh_pools)
                return (
                    hid,
                    c_in.clone().add_(1.0),
                    r_in.clone().add_(1.0),
                    *fresh_pools,
                )

            return _call

    return _FakeBridge


def _assert_tpu_meta_finish_verify(
    prog: adapter.KimiK3MonoProgram,
    recorded_calls: list[tuple[str, tuple[object, ...]]],
) -> None:
    """Verify non-CPU `finish_verify` dispatches the compiled TPU custom op."""
    assert prog.conv_slots is not None
    prog.conv_slots = prog.conv_slots.to(device=torch.device("meta"))
    assert prog.recurrent_slots is not None
    prog.recurrent_slots = prog.recurrent_slots.to(device=torch.device("meta"))
    assert prog.last_verify_conv is not None
    prog.last_verify_conv = prog.last_verify_conv.to(device=torch.device("meta"))
    assert prog.last_verify_recurrent is not None
    prog.last_verify_recurrent = prog.last_verify_recurrent.to(
        device=torch.device("meta")
    )
    prog.clear_tpu_op_cache()
    prog.finish_verify(torch.tensor([ACCEPTED_REQ0, ACCEPTED_REQ1], dtype=torch.int32))
    assert recorded_calls[-1][0] == "pallas::k3_mono_finish_verify_r2_dp2_s10"
    assert prog.last_verify_conv is None
    assert prog.last_verify_recurrent is None


def test_kimi_k3_mono_program_tpu_step_and_verify8_execution(
    monkeypatch: _MonkeyPatchLike,
) -> None:
    """Verify TPU decode/verify8 step execution, pool set_ adoption, and DP tables."""
    caches, _ = _assert_cpu_kda_importer_roundtrip()
    par_cfg = SimpleNamespace(
        tensor_parallel_size=32,
        decode_context_parallel_size=16,
        cp_kv_cache_interleave_size=EXPECTED_BF16_BLOCK_SIZE,
    )
    tpu_runner = fake_runner(
        hf_config=_kimi_hf_config(),
        tp_size=32,
        block_size=EXPECTED_BF16_BLOCK_SIZE,
        device=SimpleNamespace(type="tpu"),
        model=object(),
        speculative_tokens=7,
        token_paddings=(8, 16),
        parallel_config=par_cfg,
    )
    tpu_runner.vllm_config.parallel_config = par_cfg

    prog_default = create(
        RunnerView(tpu_runner, tp_rank=0),
        {"attention_dp": 2, "max_slots": 9, "buckets": [4, 8], "caches": caches},
    )
    assert isinstance(prog_default, adapter.KimiK3MonoProgram)
    assert prog_default.speculative_enabled is False
    assert prog_default.policy.verify_tokens == ()

    prog = create(
        RunnerView(tpu_runner, tp_rank=0),
        {
            "attention_dp": 2,
            "max_slots": 9,
            "buckets": [4, 8],
            "verify_buckets": [8, 16],
            "enable_verify_bucket": True,
            "caches": caches,
        },
    )
    assert isinstance(prog, adapter.KimiK3MonoProgram)
    assert prog.speculative_enabled is True
    assert prog.policy.verify_tokens == (VERIFY_ROWS,)

    devices = topologies.get_topology_desc(
        topology_name="tpu7x_2x2x4", platform="tpu"
    ).devices
    mesh_tp = Mesh(devices, ("tp",))
    prog.load(mesh_tp)
    prog.bind(RunnerView(tpu_runner, tp_rank=0))
    _lower_mono_program_tpu_bodies(prog, mesh_tp)

    orig_pool_ids = [id(p) for p in prog.raw_pools]
    prog.decoder_weights = {"w0": torch.ones((1, 4), dtype=torch.bfloat16)}
    recorded_calls: list[tuple[str, tuple[object, ...]]] = []
    monkeypatch.setattr(adapter, "_bridge", _make_fake_tpu_bridge(recorded_calls))

    s_g0 = prog.admit_requests(["req-dp0"], group=0)[0]
    s_g1 = prog.admit_requests(["req-dp1"], group=1)[0]
    prog.set_active_batch(
        [s_g0, s_g1],
        torch.tensor([[11, 12], [21, 22]], dtype=torch.int32),
    )

    dec_step = prog.build(Bucket(1, AOT_SCATTER_BATCH, VERIFY_BATCHED_ROWS))
    out_dec = dec_step(
        torch.arange(AOT_SCATTER_BATCH, dtype=torch.int32),
        torch.arange(AOT_SCATTER_BATCH, dtype=torch.int32),
    )
    assert out_dec.shape == (VERIFY_BATCHED_ROWS, K3_HIDDEN_DIM)
    assert [id(p) for p in prog.raw_pools] == orig_pool_ids
    assert float(prog.raw_pools[0][1, 0, 0, 0].item()) == 1.0

    ver_step = prog.build(Bucket(VERIFY_ROWS, VERIFY_BATCHED_ROWS, VERIFY_BATCHED_ROWS))
    out_ver = ver_step(
        torch.arange(VERIFY_BATCHED_ROWS, dtype=torch.int32),
        torch.arange(VERIFY_BATCHED_ROWS, dtype=torch.int32),
    )
    assert out_ver.shape == (VERIFY_BATCHED_ROWS, K3_HIDDEN_DIM)
    assert [id(p) for p in prog.raw_pools] == orig_pool_ids
    ver_name, ver_args = recorded_calls[-1]
    assert ver_name == "pallas::k3_mono_decode_t8_b16_r16"
    ver_tables = cast("torch.Tensor", ver_args[-2])
    assert isinstance(ver_tables, torch.Tensor)
    assert tuple(ver_tables.shape) == (VERIFY_BATCHED_ROWS, EXPECTED_TABLE_WIDTH)
    ver_rows = ver_tables.tolist()
    assert ver_rows[:VERIFY_ROWS] == [[11, 12]] * VERIFY_ROWS
    assert ver_rows[VERIFY_ROWS:] == [[21, 22]] * VERIFY_ROWS

    with _assert_raises(RuntimeError, "uncommitted deferred verify KDA state"):
        dec_step(
            torch.arange(AOT_SCATTER_BATCH, dtype=torch.int32),
            torch.arange(AOT_SCATTER_BATCH, dtype=torch.int32),
        )
    _assert_tpu_meta_finish_verify(prog, recorded_calls)


def test_mono_program_tpu_step_adopts_donated_pools_in_place(
    monkeypatch: _MonkeyPatchLike,
) -> None:
    """Verify TPU decode step rebinds donated pool storages in place."""
    caches, _ = _assert_cpu_kda_importer_roundtrip()
    par_cfg = SimpleNamespace(
        tensor_parallel_size=32,
        decode_context_parallel_size=16,
        cp_kv_cache_interleave_size=EXPECTED_BF16_BLOCK_SIZE,
    )
    tpu_runner = fake_runner(
        hf_config=_kimi_hf_config(),
        tp_size=32,
        block_size=EXPECTED_BF16_BLOCK_SIZE,
        device=SimpleNamespace(type="tpu"),
        model=object(),
        parallel_config=par_cfg,
    )
    tpu_runner.vllm_config.parallel_config = par_cfg
    prog = create(
        RunnerView(tpu_runner, tp_rank=0),
        {"attention_dp": 2, "max_slots": 9, "buckets": [4, 8], "caches": caches},
    )
    assert isinstance(prog, adapter.KimiK3MonoProgram)
    devices = topologies.get_topology_desc(
        topology_name="tpu7x_2x2x4", platform="tpu"
    ).devices
    mesh_tp = Mesh(devices, ("tp",))
    prog.load(mesh_tp)
    prog.bind(RunnerView(tpu_runner, tp_rank=0))

    orig_pool_ids = [id(pool) for pool in prog.raw_pools]
    prog.decoder_weights = {"w0": torch.ones((1, 4), dtype=torch.bfloat16)}
    recorded_calls: list[tuple[str, tuple[object, ...]]] = []
    monkeypatch.setattr(adapter, "_bridge", _make_fake_tpu_bridge(recorded_calls))

    slot_zero = prog.admit_requests(["req-dp0"], group=0)[0]
    slot_one = prog.admit_requests(["req-dp1"], group=1)[0]
    prog.set_active_batch(
        [slot_zero, slot_one],
        torch.tensor([[11, 12], [21, 22]], dtype=torch.int32),
    )
    dec_step = prog.build(Bucket(1, AOT_SCATTER_BATCH, VERIFY_BATCHED_ROWS))
    out_dec = dec_step(
        torch.arange(AOT_SCATTER_BATCH, dtype=torch.int32),
        torch.arange(AOT_SCATTER_BATCH, dtype=torch.int32),
    )
    assert out_dec.shape == (VERIFY_BATCHED_ROWS, K3_HIDDEN_DIM)
    assert [id(pool) for pool in prog.raw_pools] == orig_pool_ids
    assert float(prog.raw_pools[0][1, 0, 0, 0].item()) == 1.0

    finish_ctx = SimpleNamespace(
        input_ids=torch.zeros((1,), dtype=torch.int32),
        positions=torch.zeros((1,), dtype=torch.int32),
    )
    finish_fwd = ForwardView(
        attn_metadata={
            "mla": SimpleNamespace(
                req_ids=["req-dp1"],
                finished_req_ids=["req-dp0"],
                block_tables=torch.tensor([[21, 22]], dtype=torch.int32),
            )
        },
        mrope_positions=None,
        steps={(1, 4): lambda ids, pos: ids + pos},
        num_tokens=1,
        inspect=lambda _m, _k: BatchShape(
            eligible=True, num_reqs=1, tokens_per_seq=1, reason="ok"
        ),
    )
    prog.decide(finish_ctx, finish_fwd)
    assert "req-dp0" not in prog.slot_map
    assert "req-dp1" in prog.slot_map
    assert prog.slot_map.allocate("req-dp1", group=1) == slot_one


def test_mono_program_verify8_declines_without_accepted_counts_hook() -> None:
    """Verify decide() declines verify8 steps unless the commit hook is wired."""
    caches, _ = _assert_cpu_kda_importer_roundtrip()
    par_cfg = SimpleNamespace(
        tensor_parallel_size=32,
        decode_context_parallel_size=16,
        cp_kv_cache_interleave_size=EXPECTED_BF16_BLOCK_SIZE,
    )
    tpu_runner = fake_runner(
        hf_config=_kimi_hf_config(),
        tp_size=32,
        block_size=EXPECTED_BF16_BLOCK_SIZE,
        device=SimpleNamespace(type="tpu"),
        model=object(),
        speculative_tokens=7,
        token_paddings=(8, 16),
        parallel_config=par_cfg,
    )
    tpu_runner.vllm_config.parallel_config = par_cfg
    prog = create(
        RunnerView(tpu_runner, tp_rank=0),
        {
            "attention_dp": 2,
            "max_slots": 9,
            "buckets": [4, 8],
            "verify_buckets": [8, 16],
            "enable_verify_bucket": True,
            "caches": caches,
        },
    )
    assert isinstance(prog, adapter.KimiK3MonoProgram)
    prog.admit_requests(["req-v0"], group=0)

    fwd_verify = ForwardView(
        attn_metadata={
            "mla": SimpleNamespace(
                req_ids=["req-v0"],
                block_tables=torch.tensor([[3, 4]], dtype=torch.int32),
                seq_lens=torch.tensor([VERIFY_ROWS], dtype=torch.int32),
                query_start_loc=torch.tensor([0, VERIFY_ROWS], dtype=torch.int32),
            )
        },
        mrope_positions=None,
        steps={(VERIFY_ROWS, VERIFY_ROWS): lambda ids, pos: ids + pos},
        num_tokens=VERIFY_ROWS,
        inspect=lambda _m, _k: BatchShape(
            eligible=True, num_reqs=1, tokens_per_seq=VERIFY_ROWS, reason="ok"
        ),
    )
    ctx_unwired = SimpleNamespace(
        input_ids=torch.zeros((VERIFY_ROWS,), dtype=torch.int32),
        positions=torch.arange(VERIFY_ROWS, dtype=torch.int32),
    )
    decision_unwired = prog.decide(ctx_unwired, fwd_verify)
    assert decision_unwired is not None
    assert decision_unwired.eligible is False
    assert decision_unwired.reason == "verify_commit_hook_unwired"

    ctx_wired = SimpleNamespace(
        input_ids=torch.zeros((VERIFY_ROWS,), dtype=torch.int32),
        positions=torch.arange(VERIFY_ROWS, dtype=torch.int32),
        commit_verify=prog.finish_verify,
    )
    decision_wired = prog.decide(ctx_wired, fwd_verify)
    assert decision_wired is not None
    assert decision_wired.eligible is True
    assert decision_wired.tokens_per_seq == VERIFY_ROWS


def test_mono_program_verify8_expands_block_tables_and_dp_slots() -> None:
    """Verify verify8 block_tables expansion and DP=4 finish_verify slot commits."""
    caches, _ = _assert_cpu_kda_importer_roundtrip()
    par_cfg = SimpleNamespace(
        tensor_parallel_size=32,
        decode_context_parallel_size=16,
        cp_kv_cache_interleave_size=EXPECTED_BF16_BLOCK_SIZE,
    )
    cpu_runner = fake_runner(
        hf_config=_kimi_hf_config(),
        tp_size=32,
        block_size=EXPECTED_BF16_BLOCK_SIZE,
        device=SimpleNamespace(type="cpu"),
        model=object(),
        speculative_tokens=7,
        token_paddings=(8, 16, 32),
        parallel_config=par_cfg,
    )
    cpu_runner.vllm_config.parallel_config = par_cfg
    prog = create(
        RunnerView(cpu_runner, tp_rank=0),
        {
            "attention_dp": 4,
            "max_slots": 60,
            "buckets": [4, 8],
            "verify_buckets": [32],
            "enable_verify_bucket": True,
            "caches": caches,
        },
    )
    assert isinstance(prog, adapter.KimiK3MonoProgram)
    prog.load("cpu_mesh")
    prog.bind(RunnerView(cpu_runner, tp_rank=0))

    req_names = ["req-g0", "req-g1", "req-g2", "req-g3"]
    for group_idx, req_name in enumerate(req_names):
        prog.dp_groups[req_name] = group_idx

    raw_tables = torch.tensor(
        [[10, 11], [20, 21], [30, 31], [40, 41]], dtype=torch.int32
    )
    qsl = torch.tensor(
        [0, VERIFY_ROWS, 2 * VERIFY_ROWS, 3 * VERIFY_ROWS, 4 * VERIFY_ROWS],
        dtype=torch.int32,
    )
    ctx_wired = SimpleNamespace(
        input_ids=torch.zeros((4 * VERIFY_ROWS,), dtype=torch.int32),
        positions=torch.arange(4 * VERIFY_ROWS, dtype=torch.int32),
        commit_verify=prog.finish_verify,
    )
    fwd_verify = ForwardView(
        attn_metadata={
            "mla": SimpleNamespace(
                req_ids=req_names,
                block_tables=raw_tables,
                seq_lens=torch.full((4,), VERIFY_ROWS, dtype=torch.int32),
                query_start_loc=qsl,
            )
        },
        mrope_positions=None,
        steps={(VERIFY_ROWS, 4 * VERIFY_ROWS): lambda ids, pos: ids + pos},
        num_tokens=4 * VERIFY_ROWS,
        inspect=lambda _m, _k: BatchShape(
            eligible=True, num_reqs=4, tokens_per_seq=VERIFY_ROWS, reason="ok"
        ),
    )
    decision = prog.decide(ctx_wired, fwd_verify)
    assert decision is not None
    assert decision.eligible is True
    assert prog.active_slots == [1, 16, 31, 46]

    tables_padded, slot_vec = prog.prepare_tpu_step_inputs(
        4 * VERIFY_ROWS,
        VERIFY_ROWS,
        torch.device("cpu"),
    )
    assert slot_vec[::VERIFY_ROWS].tolist() == [1, 16, 31, 46]
    assert tables_padded.shape == (4 * VERIFY_ROWS, 2)
    for req_idx in range(4):
        start_row = req_idx * VERIFY_ROWS
        assert torch.equal(
            tables_padded[start_row : start_row + VERIFY_ROWS],
            raw_tables[req_idx].unsqueeze(0).expand(VERIFY_ROWS, 2),
        )

    assert prog.conv_slots is not None
    assert prog.recurrent_slots is not None
    total_rows = 4 * VERIFY_ROWS
    row_values = torch.arange(1, total_rows + 1, dtype=torch.float32)
    prog.last_verify_conv = (
        row_values.view(1, total_rows, 1, 1, 1, 1, 1)
        .expand(1, total_rows, *prog.conv_slots.shape[2:])
        .to(prog.conv_slots.dtype)
        .clone()
    )
    prog.last_verify_recurrent = (
        row_values.view(1, total_rows, 1, 1, 1, 1)
        .expand(1, total_rows, *prog.recurrent_slots.shape[2:])
        .to(prog.recurrent_slots.dtype)
        .clone()
    )
    accepted = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    prog.finish_verify(accepted)
    for group_idx, bank_slot in enumerate([1, 17, 33, 49]):
        expected_val = float(
            group_idx * VERIFY_ROWS + int(accepted[group_idx].item()) + 1
        )
        got_conv = float(prog.conv_slots[0, bank_slot, 0, 0, 0, 0, 0].item())
        got_rec = float(prog.recurrent_slots[0, bank_slot, 0, 0, 0, 0].item())
        assert got_conv == expected_val
        assert got_rec == expected_val
    for null_slot in [0, 16, 32, 48]:
        assert float(prog.conv_slots[0, null_slot].abs().max().item()) == 0.0
        assert float(prog.recurrent_slots[0, null_slot].abs().max().item()) == 0.0
