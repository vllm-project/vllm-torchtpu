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
"""CPU/fake coverage of the PLE gate env selection and adapter."""

import copy
import gc
import weakref

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from vllm.models.qwen4_exp.common.hyperconnection import GroupedGemmaRMSNorm

from tests.models.vllm.test_qwen4_exp import (
    _PLE,
    _set_tp,
    _text_config,
)
from tests.models.vllm.test_qwen4_exp import (
    _cpu_model_tests as _cpu_model_tests,
)
from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter.custom_ops import ple_op
from vllm_torchtpu.models.vllm.qwen4_exp import ple

TOKENS = [0, 1, 7, 8, 9, 128, 129, 1024]
STREAMS, HIDDEN = 4, 2560
ENV = "TPU_PLE_FUSED_GATE"


@pytest.fixture(autouse=True)
def no_tpu_initialization():
    assert not torch.tpu.is_initialized()
    yield
    assert not torch.tpu.is_initialized()


@pytest.mark.parametrize(
    "value,enabled",
    [
        (None, False),
        ("", False),
        ("0", False),
        ("OFF", False),
        ("1", True),
        ("true", True),
    ],
)
def test_fused_gate_env_default_and_override(monkeypatch, value, enabled):
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    assert envs.TPU_PLE_FUSED_GATE is enabled


def test_fused_gate_invalid_env_fails_at_construction(monkeypatch):
    monkeypatch.setenv(ENV, "invalid")
    with pytest.raises(ValueError, match=ENV):
        _Gate()


@pytest.fixture
def cpu_gate(monkeypatch):
    """Replace the CPU error implementation with a scoped observation fixture."""
    monkeypatch.setenv(ENV, "1")
    ple_op.build_normalized_gate_dot_op(STREAMS)
    name = f"qwen4_exp::ple_gate_dot_s{STREAMS}"
    assert torch._C._dispatch_has_kernel_for_dispatch_key(name, "CPU")
    calls = []

    def backend(key, query, key_inv, query_inv, key_weight, query_weight):
        calls.append((key, query, key_inv, query_inv, key_weight, query_weight))
        return key_inv.to(torch.bfloat16) + key_weight[0] + query_weight[0]

    library = torch.library.Library("qwen4_exp", "IMPL", "CPU")
    library.impl(f"ple_gate_dot_s{STREAMS}", backend, allow_override=True)
    lifetime = weakref.ref(library)
    try:
        yield calls
    finally:
        del library
        gc.collect()
        assert lifetime() is None
        assert torch._C._dispatch_has_kernel_for_dispatch_key(name, "CPU")
        raw = torch.empty((0, STREAMS * HIDDEN), dtype=torch.bfloat16)
        inverse = torch.empty((0, STREAMS))
        weight = torch.zeros(STREAMS * HIDDEN, dtype=torch.bfloat16)
        with pytest.raises(NotImplementedError, match="requires TPU tensors"):
            ple_op.build_normalized_gate_dot_op(STREAMS)(
                raw, raw, inverse, inverse, weight, weight
            )


class _Gate(torch.nn.Module):
    def __init__(self, streams=STREAMS, hidden=HIDDEN):
        super().__init__()
        self.norm_key, self.norm_query = (
            GroupedGemmaRMSNorm(streams * hidden, 1e-6, hidden, torch.bfloat16)
            for _ in range(2)
        )
        self.run = ple_op.build_normalized_gate_dot_dispatch(streams, hidden)

    def forward(self, key, query):
        return self.run(key, query, self.norm_key, self.norm_query)


@pytest.mark.parametrize("tokens", TOKENS)
def test_adapter_opt_in_all_lengths_and_current_weights(cpu_gate, tokens):
    generator = torch.Generator().manual_seed(2416)
    key = torch.randn((tokens, STREAMS * HIDDEN), generator=generator).bfloat16()
    query = torch.randn(key.shape, generator=generator).bfloat16()
    module = _Gate()
    for norm in (module.norm_key, module.norm_query):
        norm.weight = torch.nn.Parameter(
            torch.randn((STREAMS * HIDDEN,), generator=generator).bfloat16(),
            requires_grad=False,
        )
    actual = module(key, query)
    assert len(cpu_gate) == 1
    passed = cpu_gate[0]
    for operand, inverse in zip((key, query), passed[2:4]):
        grouped = operand.double().reshape(tokens, STREAMS, HIDDEN)
        expected = torch.rsqrt(grouped.square().mean(-1) + 1e-6).float()
        torch.testing.assert_close(inverse, expected, rtol=2e-7, atol=2e-7)
    assert passed[0] is key and passed[1] is query
    assert passed[4] is module.norm_key.weight
    assert passed[5] is module.norm_query.weight
    observed = passed[2].bfloat16() + passed[4][0] + passed[5][0]
    torch.testing.assert_close(actual, observed.unsqueeze(-1), rtol=0, atol=0)
    for norm in (module.norm_key, module.norm_query):
        norm.weight = torch.nn.Parameter(torch.zeros_like(norm.weight))
    module(key, query)
    assert cpu_gate[-1][4] is module.norm_key.weight
    assert cpu_gate[-1][5] is module.norm_query.weight


def test_real_adapter_dynamic_dynamo_retains_one_graph(cpu_gate):
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    compiled = torch.compile(_Gate(), backend=backend, fullgraph=True)
    try:
        for tokens in (7, 8, 9, 128, 129, 1024):
            raw = torch.ones((tokens, STREAMS * HIDDEN), dtype=torch.bfloat16)
            torch._dynamo.mark_dynamic(raw, 0, min=2, max=1024)
            result = compiled(raw, raw)
            assert result.shape == (tokens, STREAMS, 1)
        assert len(graphs) == 1
        targets = [str(node.target) for node in graphs[0].graph.nodes]
        assert sum("ple_gate_dot_s4" in target for target in targets) == 1
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize("tokens", [0, 1, 8])
def test_real_adapter_export_fake_shapes(monkeypatch, tokens):
    monkeypatch.setenv(ENV, "1")
    with FakeTensorMode():
        key = torch.empty((tokens, STREAMS * HIDDEN), dtype=torch.bfloat16)
        dynamic = None
        if tokens > 1:
            count = torch.export.Dim("tokens", min=2, max=1024)
            dynamic = ({0: count}, {0: count})
        exported = torch.export.export(
            _Gate(), (key, key.clone()), dynamic_shapes=dynamic, strict=True
        )
        targets = [str(node.target) for node in exported.graph.nodes]
        assert sum("ple_gate_dot_s4" in target for target in targets) == 1
        result = exported.module()(key, key.clone())
        assert result.shape == (tokens, STREAMS, 1)
        assert result.dtype == torch.bfloat16


def test_opt_in_real_cpu_operands_fail_without_tpu_initialization(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    raw = torch.ones((1, 2 * 129), dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="requires TPU tensors"):
        _Gate(streams=2, hidden=129)(raw, raw)


@pytest.mark.parametrize("value", [None, "0"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("streams,hidden", [(4, 2560), (2, 128), (3, 129)])
def test_model_disabled_gate_is_original(monkeypatch, value, dtype, streams, hidden):
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    _set_tp(monkeypatch, 0, 1)

    def forbidden(*args):
        raise AssertionError("Disabled fusion must not construct the native gate op")

    monkeypatch.setattr(ple_op, "build_normalized_gate_dot_op", forbidden)
    config = _text_config(**{**_PLE, "hidden_size": hidden, "hc_count": streams})
    layer = ple.Qwen4ExpPLELayer(config, 0, None, "ple").to(dtype=dtype)
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.uniform_(-0.1, 0.1)
    embeddings = torch.randn((7, config.ple_embed_dim), dtype=dtype)
    hidden_states = torch.randn((7, streams * hidden), dtype=dtype)
    actual = layer.gated_values(hidden_states, embeddings)
    key, _ = layer.key_proj(embeddings)
    value, _ = layer.value_proj(embeddings)
    normalized_key = layer.norm_key(key).reshape(7, streams, hidden)
    normalized_query = layer.norm_query(hidden_states).reshape(7, streams, hidden)
    dot = (normalized_key * normalized_query).float().sum(-1, keepdim=True).to(dtype)
    gate = dot / hidden**0.5
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    expected = (torch.sigmoid(gate) * value.unsqueeze(-2)).flatten(-2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("enabled", [False, True])
def test_gate_copy_uses_current_independent_norm_parameters(
    monkeypatch, cpu_gate, enabled
):
    monkeypatch.setenv(ENV, "1" if enabled else "0")
    _set_tp(monkeypatch, 0, 1)
    config = _text_config(**{**_PLE, "hidden_size": 128, "hc_count": STREAMS})
    layer = ple.Qwen4ExpPLELayer(config, 0, None, "ple")
    keys = set(layer.state_dict())
    assert {key for key in keys if key.startswith("norm_")} == {
        "norm_key.weight",
        "norm_query.weight",
        "norm_conv.weight",
    }
    assert not any(key.startswith("gate_dot.") for key in keys)
    original = _Gate(hidden=128)
    with torch.no_grad():
        for parameter in original.parameters():
            parameter.uniform_(-0.1, 0.1)
    cloned = copy.deepcopy(original)
    assert (
        set(original.state_dict())
        == set(cloned.state_dict())
        == {"norm_key.weight", "norm_query.weight"}
    )
    raw = torch.randn((7, STREAMS * 128), dtype=torch.bfloat16)
    before = original(raw, raw).clone()
    torch.testing.assert_close(cloned(raw, raw), before, rtol=0, atol=0)
    cloned.norm_key.weight = torch.nn.Parameter(
        torch.full_like(cloned.norm_key.weight, 0.5)
    )
    with torch.no_grad():
        cloned.norm_query.weight.add_(0.5)
    cloned.norm_query.variance_epsilon = 0.01
    assert cloned.norm_key.weight is not original.norm_key.weight
    assert cloned.norm_query.weight.data_ptr() != original.norm_query.weight.data_ptr()
    copied = cloned(raw, raw)
    assert not torch.equal(copied, before)
    if enabled:
        assert cpu_gate[-1][4] is cloned.norm_key.weight
        assert cpu_gate[-1][5] is cloned.norm_query.weight
        expected_inverse = torch.rsqrt(
            raw.float().reshape(7, STREAMS, 128).square().mean(-1) + 0.01
        )
        torch.testing.assert_close(cpu_gate[-1][3], expected_inverse, rtol=0, atol=0)
    torch.testing.assert_close(original(raw, raw), before, rtol=0, atol=0)


@pytest.mark.parametrize(
    "error",
    [
        ValueError("Unsupported TPU device kind: fixture"),
        ValueError("unexpected runtime error"),
        RuntimeError("kernel failure"),
    ],
)
def test_opt_in_backend_failures_propagate(monkeypatch, error):
    monkeypatch.setenv(ENV, "1")

    def fail(*args):
        raise error

    monkeypatch.setattr(ple_op, "build_normalized_gate_dot_op", lambda streams: fail)
    module = _Gate(streams=2, hidden=129)
    raw = torch.ones((1, 2 * 129), dtype=torch.bfloat16)
    with pytest.raises(type(error)) as caught:
        module(raw, raw)
    assert caught.value is error
