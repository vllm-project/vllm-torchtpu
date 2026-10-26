# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.kernels import fused_greedy_head as kernel
from vllm_torchtpu.layers.adapter.sample.fused_greedy_head import FusedGreedyHead
from vllm_torchtpu.models.vllm.deepseek_v41.v2.model import DeepseekV41ForCausalLM
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def _runner(weight):
    model = DeepseekV41ForCausalLM.__new__(DeepseekV41ForCausalLM)
    torch.nn.Module.__init__(model)
    model.lm_head = torch.nn.Module()
    model.lm_head.weight = torch.nn.Parameter(weight, requires_grad=False)
    return SimpleNamespace(
        model=model,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1
        ),
        speculative_config=SimpleNamespace(method="dspark", num_speculative_tokens=5),
        max_num_reqs=48,
        lora_config=None,
        input_batch=SimpleNamespace(all_greedy=True, max_num_logprobs=None),
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize("backend", [None, "auto", "pallas", "logits"])
def test_greedy_head_backend_selection(monkeypatch, backend):
    monkeypatch.delenv("TPU_GREEDY_HEAD_BACKEND", raising=False)
    if backend is not None:
        monkeypatch.setenv("TPU_GREEDY_HEAD_BACKEND", backend)
    selected = envs.environment_variables["TPU_GREEDY_HEAD_BACKEND"]()
    assert selected == (backend or "auto")
    monkeypatch.setattr(envs, "TPU_GREEDY_HEAD_BACKEND", selected)
    monkeypatch.setattr(kernel, "get_device_name", lambda: "TPU v7")
    runner = _runner(torch.empty(129280, 5120, dtype=torch.bfloat16, device="meta"))
    TPUModelRunner._init_fused_greedy_head(runner)
    head = runner._fused_greedy_head
    if selected == "logits":
        assert head is None
    else:
        assert head.weight is runner.model.lm_head.weight
        assert head.rows == 288


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "unsupported",
    [
        "model",
        "tp",
        "pp",
        "lora",
        "no_spec",
        "method",
        "rows",
        "shape",
        "dtype",
        "device",
    ],
)
def test_auto_greedy_head_falls_back_and_forced_mode_rejects(monkeypatch, unsupported):
    weight = torch.empty(129280, 5120, dtype=torch.bfloat16, device="meta")
    if unsupported == "shape":
        weight = torch.empty(128, 5120, dtype=torch.bfloat16, device="meta")
    elif unsupported == "dtype":
        weight = weight.float()
    runner = _runner(weight)
    monkeypatch.setattr(
        kernel,
        "get_device_name",
        lambda: "TPU v6" if unsupported == "device" else "TPU v7",
    )
    if unsupported == "model":
        runner.model = torch.nn.Module()
    elif unsupported in ("tp", "pp"):
        setattr(
            runner.parallel_config,
            "tensor_parallel_size" if unsupported == "tp" else "pipeline_parallel_size",
            2,
        )
    elif unsupported == "lora":
        runner.lora_config = object()
    elif unsupported == "no_spec":
        runner.speculative_config = None
    elif unsupported == "method":
        runner.speculative_config.method = "eagle"
    elif unsupported == "rows":
        runner.max_num_reqs = 1
    monkeypatch.setattr(envs, "TPU_GREEDY_HEAD_BACKEND", "auto")
    TPUModelRunner._init_fused_greedy_head(runner)
    assert runner._fused_greedy_head is None
    monkeypatch.setattr(envs, "TPU_GREEDY_HEAD_BACKEND", "pallas")
    with pytest.raises(AssertionError, match="Forced fused greedy head"):
        TPUModelRunner._init_fused_greedy_head(runner)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "fallback", [None, "rows", "metadata", "sampling", "grammar", "logprobs"]
)
def test_greedy_head_request_dispatch(monkeypatch, fallback):
    monkeypatch.setattr(envs, "TPU_GREEDY_HEAD_BACKEND", "auto")
    monkeypatch.setattr(kernel, "get_device_name", lambda: "TPU v7")
    runner = _runner(torch.empty(129280, 5120, dtype=torch.bfloat16, device="meta"))
    TPUModelRunner._init_fused_greedy_head(runner)
    scheduler = SimpleNamespace(has_structured_output_requests=fallback == "grammar")
    runner.input_batch.all_greedy = fallback != "sampling"
    runner.input_batch.max_num_logprobs = 1 if fallback == "logprobs" else None
    enabled = TPUModelRunner._can_use_fused_greedy_head(
        runner,
        scheduler,
        None if fallback == "metadata" else object(),
        1 if fallback == "rows" else 288,
    )
    assert enabled is (fallback is None)


def test_fused_greedy_head_cpu_reference(device, monkeypatch):
    """Check full-vocabulary CPU64 rounding, including a cross-tile tie."""
    generator = torch.Generator().manual_seed(9126)
    hidden = torch.randn(288, 5120, generator=generator).to(torch.bfloat16)
    weight = torch.randn(129280, 5120, generator=generator).to(torch.bfloat16)
    # A deliberately dominant identical pair straddles the tile boundary.
    hidden[0].zero_()
    hidden[0, 0] = 1
    weight[:, 0] = 0
    weight[1279:1281, 0] = 32
    reference = (hidden[:4].double() @ weight.double().T).to(torch.bfloat16)
    expected = reference.argmax(dim=-1)
    device_weight = weight.to(device)
    monkeypatch.delenv("TPU_GREEDY_HEAD_BACKEND", raising=False)
    monkeypatch.setattr(
        envs,
        "TPU_GREEDY_HEAD_BACKEND",
        envs.environment_variables["TPU_GREEDY_HEAD_BACKEND"](),
    )
    runner = _runner(device_weight)
    TPUModelRunner._init_fused_greedy_head(runner)
    head = runner._fused_greedy_head
    assert isinstance(head, FusedGreedyHead)
    assert head.weight is runner.model.lm_head.weight
    compiled = torch.compile(head, backend="tpu", fullgraph=True, dynamic=False)
    actual = compiled(hidden.to(device)).cpu()
    torch.testing.assert_close(actual[:4].long(), expected, rtol=0, atol=0)
    assert actual[0].item() == 1279

    # The compiled head must read the shared weight, not a load-time snapshot.
    device_weight[1279, 0] = 0
    updated = compiled(hidden.to(device)).cpu()
    assert updated[0].item() == 1280
