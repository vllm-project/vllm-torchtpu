# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""CPU Pallas-interpret numerical parity, torch.testing.assert_close, static-ring
tile reduction (`-180` DMAs at L=45), and MoE `rb_hoist` verification for
`VLLM_TPU_MK_HC_MOE_FUSED`.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import ml_dtypes
import numpy as np
import pytest
import torch

import vllm_torchtpu.envs as envs
from tests.megakernel import test_moe_rowbucket as tmrb
from tests.megakernel.test_wholestep import _run
from vllm_torchtpu.megakernel import vmem
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.models import qwen38
from vllm_torchtpu.megakernel.wholestep.spec import wholestep_spec
from vllm_torchtpu.megakernel_runtime.qwen38_wholestep_call import (
    indexed_options,
    wholestep_tiles,
)


def _to_torch(arr) -> torch.Tensor:
    a = np.asarray(arr)
    if a.dtype == ml_dtypes.bfloat16:
        return torch.from_numpy(a.astype(np.float32))
    return torch.from_numpy(np.array(a, copy=True))


def test_hc_moe_fused_default_off_and_compile_hash():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("VLLM_TPU_MK_HC_MOE_FUSED", None)
        assert envs.VLLM_TPU_MK_HC_MOE_FUSED is False
        os.environ.pop("VLLM_TPU_MK_HC_FP8", None)
        assert envs.VLLM_TPU_MK_HC_FP8 is False

    from types import SimpleNamespace
    from vllm_torchtpu.compilation.tpu_compiler import compute_tpu_compilation_hash

    cfg = SimpleNamespace(
        cache_config=SimpleNamespace(
            gpu_memory_utilization=0.9,
            num_gpu_blocks=1024,
            num_gpu_blocks_override=None,
            enable_prefix_caching=True,
            mamba_cache_mode="align",
        ),
        model_config=SimpleNamespace(max_model_len=32768),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192, max_num_seqs=32),
        parallel_config=SimpleNamespace(data_parallel_size=1),
        speculative_config=None,
    )
    with patch(
        "vllm_torchtpu.compilation.tpu_compiler._iter_runtime_cache_key_files",
        return_value=[],
    ):
        with patch.dict(os.environ, {"VLLM_TPU_MK_HC_MOE_FUSED": "0"}):
            h0 = compute_tpu_compilation_hash(cfg)
        with patch.dict(os.environ, {"VLLM_TPU_MK_HC_MOE_FUSED": "1"}):
            h1 = compute_tpu_compilation_hash(cfg)
    assert h0 != h1


def test_l45_paired_fp8_hc_saves_180_static_stream_tiles_and_zero_extra_acc_vmem():
    """At full Qwen3.8 L=45 under MK_HC_FP8=1, paired [384, 5120] FP8 hup streams
    cut static dense ring tiles from 4 -> 2 per HC block across 90 HC sublayers
    (-180 stream tiles & semaphore waits per step) with 0 extra acc VMEM bytes."""
    g = qwen38.FULL
    R, L, B = 8, 45, 64
    with patch.dict(
        os.environ,
        {
            "VLLM_TPU_MK_HC_FP8": "1",
            "VLLM_TPU_MK_ROW_BUCKET": "1",
            "VLLM_TPU_MK_HC_MOE_FUSED": "0",
        },
    ):
        ex_base = ExecSpec(
            R=R, B=B, T=4, ctx=4096, stack_e=True, tiles=wholestep_tiles(B, 32768)
        )
        spec_base = wholestep_spec(
            g,
            R,
            L=L,
            first="qsa",
            pool_gdn=True,
            qsa=dict(budget=512),
            step=True,
            indexed=dict(
                max_groups=32768,
                pages_per_seq=256,
                carrier="fp8",
                n_layers=12,
                **indexed_options(),
            ),
            hc_fp8=True,
            hc_fused=False,
        )
        cp_base, _ = vmem.build_scratch(spec_base, ex_base)

    with patch.dict(
        os.environ,
        {
            "VLLM_TPU_MK_HC_FP8": "1",
            "VLLM_TPU_MK_ROW_BUCKET": "1",
            "VLLM_TPU_MK_HC_MOE_FUSED": "1",
        },
    ):
        ex_opt = ExecSpec(
            R=R, B=B, T=4, ctx=4096, stack_e=True, tiles=wholestep_tiles(B, 32768)
        )
        spec_opt = wholestep_spec(
            g,
            R,
            L=L,
            first="qsa",
            pool_gdn=True,
            qsa=dict(budget=512),
            step=True,
            indexed=dict(
                max_groups=32768,
                pages_per_seq=256,
                carrier="fp8",
                n_layers=12,
                **indexed_options(),
            ),
            hc_fp8=True,
            hc_fused=True,
        )
        cp_opt, _ = vmem.build_scratch(spec_opt, ex_opt)

    assert cp_base.stream.T_TOTAL - cp_opt.stream.T_TOTAL == 180
    assert cp_opt.vmem_bytes <= cp_base.vmem_bytes
    assert cp_opt.vmem_shapes["acc"] == cp_base.vmem_shapes["acc"]
    assert cp_opt.block("moe").RB_HOIST is True


@pytest.mark.parametrize("hc_fp8", [False, True])
def test_wholestep_hc_fused_bit_exact_cpu_interpret(hc_fp8):
    res_base, _, _, _, _ = _run(identical_experts=True, hc_fp8=hc_fp8, hc_fused=False)
    res_opt, _, _, _, _ = _run(identical_experts=True, hc_fp8=hc_fp8, hc_fused=True)
    torch.testing.assert_close(_to_torch(res_opt), _to_torch(res_base), rtol=0, atol=0)
    np.testing.assert_array_equal(np.asarray(res_opt), np.asarray(res_base))


@pytest.mark.parametrize("B,hot", [(32, False), (48, True)])
def test_moe_rb_hoist_parity(B, hot):
    """MoETunables.rb_hoist=True matches row_bucket=True across normal & hot-expert chunked routing."""
    out_base, _, _ = tmrb._run(B, row_bucket=True, hot=hot, direct0=True)
    orig_with_blocks = tmrb.TINY_TILES.__class__.with_blocks

    def patched_with_blocks(self, updates):
        u = dict(updates)
        if "moe" in u:
            u["moe"] = dict(u["moe"], rb_hoist=True)
        return orig_with_blocks(self, u)

    with patch.object(tmrb.TINY_TILES.__class__, "with_blocks", patched_with_blocks):
        out_opt, _, _ = tmrb._run(B, row_bucket=True, hot=hot, direct0=True)

    torch.testing.assert_close(_to_torch(out_opt), _to_torch(out_base), rtol=0, atol=0)
