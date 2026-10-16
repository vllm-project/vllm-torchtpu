# SPDX-License-Identifier: Apache-2.0
"""Real compact PLE loading and compiled lookup without a full model."""

import pytest
import torch
from test_qwen4_exp import _set_tp
from torch_tpu._internal import compile as tpu_compile
from torch_tpu._internal import sync
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.models.qwen4_exp.common.ngram_embedding import Qwen4ExpPLEEmbeddingMethod
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

from vllm_torchtpu.models.vllm.qwen4_exp.ple_storage import ResidentPLETable


@pytest.mark.tpu_test
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
def test_loaded_compact_table_compiles_and_preserves_checkpoint_rows(monkeypatch, fp8):
    _set_tp(monkeypatch, 0, 1)
    cfg = VllmConfig(device_config=DeviceConfig(device="cpu"))
    resolve_kv_cache_layout(cfg, [["LBNHC", "LBHNC"]])
    dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    method = Qwen4ExpPLEEmbeddingMethod.from_quant_config(
        None, "ple", "float8_e4m3fn" if fp8 else None
    )
    with set_current_vllm_config(cfg), set_default_torch_dtype(torch.bfloat16):
        table = ResidentPLETable(
            64, 160, embedding_method=method, padding_size=64, prefix="ple"
        )
    checkpoint = (
        ((torch.arange(64 * 160, device="cpu") % 17 - 8).float() / 4)
        .reshape(64, 160)
        .to(dtype)
    )
    for start in (0, 22, 44):
        table.weight_loader(table.weight, checkpoint[start : start + 22], start)
    if fp8:
        table.get_parameter("weight_scale").data.fill_(0.25)
    table.to("tpu")
    assert table.weight.device.type == "cpu"
    table.finalize_weights()
    assert table.weight.numel() == 0
    assert table.table.numel() == checkpoint.numel()
    run = torch.compile(
        table.local_rows,
        backend=tpu_compile.TpuBackend(),
        fullgraph=True,
        dynamic=False,
    )
    for ids in (
        torch.tensor([[0, 21, 22, 63], [-1, 64, 31, 32]], dtype=torch.int32),
        torch.tensor([[63, 22, 21, 0], [32, 31, 64, -1]], dtype=torch.int32),
    ):
        actual = run(ids.to("tpu"))
        sync.synchronize([actual], wait=True)
        expected = checkpoint[ids.clamp(0, 63)].to(torch.bfloat16)
        if fp8:
            expected = expected * torch.tensor(0.25, dtype=torch.bfloat16)
        expected[(ids < 0) | (ids >= 64)] = 0
        actual_cpu = actual.cpu()
        assert bool(torch.isfinite(actual_cpu).all())
        torch.testing.assert_close(actual_cpu, expected, rtol=0, atol=0)
