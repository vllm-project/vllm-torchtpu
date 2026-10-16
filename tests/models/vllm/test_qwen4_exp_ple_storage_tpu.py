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


@pytest.mark.tpu_test
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
def test_native_ple_history_matches_common_layer_on_tpu(monkeypatch, fp8):
    """Real hash/lookup/projection/gate/convolution with local TP1 group fixtures."""
    from test_qwen4_exp import _PLE, _text_config

    from vllm_torchtpu.models.vllm.qwen4_exp.parallel import PLEParallel
    from vllm_torchtpu.models.vllm.qwen4_exp.ple import PLEInputs, Qwen4ExpPLELayer
    from vllm_torchtpu.models.vllm.qwen4_exp.pooled_ple import Qwen4ExpPooledPLE

    _set_tp(monkeypatch, 0, 1)

    class Group:
        rank_in_group = 0
        world_size = 1

        def all_reduce(self, rows):
            return rows

    group = Group()
    cfg = VllmConfig(device_config=DeviceConfig(device="cpu"))
    resolve_kv_cache_layout(cfg, [["LBNHC", "LBHNC"]])
    config = _text_config(
        **dict(
            _PLE,
            hidden_size=128,
            hc_count=2,
            ple_embed_dim=640,
            ple_embedding_dtype="float8_e4m3fn" if fp8 else None,
        )
    )
    with set_current_vllm_config(cfg), set_default_torch_dtype(torch.bfloat16):
        reference = Qwen4ExpPLELayer(config, 0, None, "ple")
        adapted = Qwen4ExpPooledPLE(
            config,
            0,
            None,
            "ple",
            parallel=PLEParallel(group, group, embedding_across_dp=False),
            table_group=group,
        )
    gen = torch.Generator(device="cpu").manual_seed(73)
    with torch.no_grad():
        for name, param in reference.named_parameters():
            param.copy_((torch.randn(param.shape, generator=gen) * 0.1).to(param.dtype))
        if fp8:
            reference.ple_embedding.ngram_embedding.get_parameter("weight_scale").fill_(
                0.25
            )
    weights = dict(reference.state_dict())
    checkpoint = weights.pop("ple_embedding.ngram_embedding.weight")
    for i, rows in enumerate(checkpoint.split(22)):
        weights[f"ple_embedding.ngram_embedding.shard_{i}.weight"] = rows
    adapted.load_checkpoint(iter(weights.items()))
    adapted.to("tpu")
    adapted.finalize_weights()
    history = torch.zeros(4, reference.conv_state_len, 256, dtype=torch.bfloat16)
    history[1].fill_(4)
    device_history = history.to("tpu")
    adapted.bind_history(device_history)
    run = torch.compile(
        adapted.forward_with_history,
        backend=tpu_compile.TpuBackend(),
        fullgraph=True,
        dynamic=False,
    )
    contexts = torch.full((3, 2), config.eos_token_id, dtype=torch.int64)
    with torch.no_grad():
        for ids, starts, slots, initial in (
            ([3, 4, 5, 6, 7, 0], [0, 3, 5, 5], [1, 2, -1], [False, False, False]),
            ([8, 9, 0], [0, 1, 2, 2], [2, 1, -1], [True, True, False]),
        ):
            ids, starts, slots = map(
                lambda v: torch.tensor(v, dtype=torch.int32), (ids, starts, slots)
            )
            initial = torch.tensor(initial)
            hidden = (torch.randn(len(ids), 256, generator=gen) * 0.1).to(
                torch.bfloat16
            )
            windows = torch.where(
                initial[:, None, None], history[slots.clamp_min(0)], 0
            )
            expected, updated = reference(
                hidden, PLEInputs(ids, starts, contexts), windows
            )
            actual = run(
                hidden.to("tpu"),
                PLEInputs(ids.to("tpu"), starts.to("tpu"), contexts.to("tpu")),
                slots.to("tpu"),
                initial.to("tpu"),
                starts[-1:].to("tpu"),
            )
            sync.synchronize([actual, device_history], wait=True)
            actual_cpu = actual.cpu()
            valid = int(starts[-1])
            assert bool(torch.isfinite(actual_cpu).all())
            # Same BF16 output boundaries; TPU matrix reductions may differ from CPU.
            torch.testing.assert_close(
                actual_cpu[:valid], expected[:valid], rtol=0.02, atol=0.002
            )
            assert torch.count_nonzero(actual_cpu[valid:]) == 0
            for i, slot in enumerate(slots.tolist()):
                if slot > 0:
                    history[slot] = updated[i]
            torch.testing.assert_close(
                device_history.cpu(), history, rtol=0.02, atol=0.002
            )
