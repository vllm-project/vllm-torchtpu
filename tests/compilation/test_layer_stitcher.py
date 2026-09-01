# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Transformer Layer FX Template Stitching."""

import torch
from torch import fx, nn

from vllm_torchtpu.compilation.layer_stitcher import (inspect_model_layers,
                                                      layer_signature,
                                                      stitch_monolithic_graph)


class MockAttention(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 sliding_window: int | None = None) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.sliding_window = sliding_window

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = torch.matmul(q, k.transpose(-1, -2)) * 0.125
        attn = torch.softmax(scores, dim=-1)
        return self.o_proj(torch.matmul(attn, v))


class MockMLP(nn.Module):

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.relu(self.gate_proj(x)) * self.up_proj(x))


class MockDecoderLayer(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        layer_idx: int = 0,
        sliding_window: int | None = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = MockAttention(hidden_size,
                                       sliding_window=sliding_window)
        self.mlp = MockMLP(hidden_size, intermediate_size)
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        normed = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(normed)
        normed_mlp = self.post_attention_layernorm(attn_out)
        mlp_out = self.mlp(normed_mlp)
        return mlp_out + hidden_states


class MockTransformerModel(nn.Module):

    def __init__(
        self,
        num_layers: int = 4,
        hidden_size: int = 16,
        intermediate_size: int = 32,
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(64, hidden_size)
        self.layers = nn.ModuleList([
            MockDecoderLayer(
                hidden_size,
                intermediate_size,
                layer_idx=i,
            ) for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.norm(hidden_states)


def test_layer_signature_isomorphism():
    """Verify that structurally identical layers produce identical signatures."""
    layer0 = MockDecoderLayer(16, 32, layer_idx=0)
    layer1 = MockDecoderLayer(16, 32, layer_idx=1)

    sig0 = layer_signature(layer0)
    sig1 = layer_signature(layer1)

    assert sig0 == sig1, "Identical layers with different layer_idx must produce the same signature"


def test_layer_signature_heterogeneity():
    """Verify that layers with different structural attributes produce distinct signatures."""
    layer_std = MockDecoderLayer(16, 32, layer_idx=0, sliding_window=None)
    layer_sliding = MockDecoderLayer(16, 32, layer_idx=1, sliding_window=2048)

    sig_std = layer_signature(layer_std)
    sig_sliding = layer_signature(layer_sliding)

    assert sig_std != sig_sliding, "Layers with different sliding_window must produce distinct signatures"


def test_inspect_model_layers():
    """Verify layer grouping by structural signature."""
    model = MockTransformerModel(num_layers=4, hidden_size=16)
    inspection = inspect_model_layers(model)

    assert inspection is not None
    layers, attr_name, sig_map = inspection
    assert len(layers) == 4
    assert attr_name == "layers"
    assert len(sig_map) == 1
    assert list(sig_map.values())[0] == [0, 1, 2, 3]


def test_stitch_monolithic_graph_expansion():
    """Verify expanding a 1-layer template to an N-layer monolithic graph."""
    num_layers = 4
    hidden_size = 16
    model = MockTransformerModel(num_layers=num_layers,
                                 hidden_size=hidden_size)

    input_ids = torch.randint(0, 64, (1, 8), dtype=torch.long)

    # Capture 1-layer template
    original_layers = list(model.layers)
    model.layers = nn.ModuleList([original_layers[0]])
    template_gm = fx.symbolic_trace(model)
    model.layers = nn.ModuleList(original_layers)

    # Stitch the graph to 4 layers
    stitched_gm = stitch_monolithic_graph(
        template_gm,
        num_total_layers=num_layers,
        layers_attr="layers",
        template_layer_idx=0,
        all_layers=original_layers,
    )

    # Check that modules exist for all 4 layers
    module_targets = {
        node.target
        for node in stitched_gm.graph.nodes
        if node.op == "call_module" and isinstance(node.target, str)
    }

    for i in range(num_layers):
        expected_target = f"layers.{i}.self_attn.q_proj"
        assert expected_target in module_targets, f"Expected {expected_target} in stitched graph"

    # Run numerical check: compare stitched graph output with eager model forward
    with torch.no_grad():
        expected_out = model(input_ids)
        stitched_out = stitched_gm(input_ids)

    torch.testing.assert_close(
        stitched_out,
        expected_out,
        rtol=1e-4,
        atol=1e-4,
        msg="Stitched graph output must match eager model output",
    )


def test_wrap_callable_with_stitching():
    """Verify wrap_callable_with_stitching end-to-end with fast-path execution."""
    from vllm_torchtpu.compilation.layer_stitcher import \
        wrap_callable_with_stitching

    num_layers = 4
    hidden_size = 16
    model = MockTransformerModel(num_layers=num_layers,
                                 hidden_size=hidden_size)

    input_ids = torch.randint(0, 64, (1, 8), dtype=torch.long)

    # Wrap model forward
    original_callable = model.forward
    wrapped_fn = wrap_callable_with_stitching(model, original_callable)

    with torch.no_grad():
        expected_out = model(input_ids)
        actual_out = wrapped_fn(input_ids)

    torch.testing.assert_close(
        actual_out,
        expected_out,
        rtol=1e-4,
        atol=1e-4,
        msg="Stitched wrapper execution output must match eager model output",
    )


def test_tracing_count_and_speedup_proof():
    """Formally prove that only 1 layer is traced, and N-1 layers have 0 trace passes."""
    from vllm_torchtpu.compilation.layer_stitcher import \
        wrap_callable_with_stitching

    num_layers = 16
    model = MockTransformerModel(num_layers=num_layers,
                                 hidden_size=32,
                                 intermediate_size=64)
    input_ids = torch.randint(0, 64, (1, 16), dtype=torch.long)

    # Track forward calls
    layer_call_counts = [0] * num_layers
    for i, layer in enumerate(model.layers):
        layer.register_forward_hook(
            lambda m, inp, out, idx=i: layer_call_counts.__setitem__(
                idx, layer_call_counts[idx] + 1))

    # Stitched run
    wrapped = wrap_callable_with_stitching(model, model.forward)
    wrapped(input_ids)

    # Formal assertions:
    # 1. Layer 0 template is traced exactly once.
    assert layer_call_counts[
        0] == 1, "Layer 0 template must be traced exactly once"
    # 2. Layers 1..15 are NEVER traced by Python (count is exactly 0).
    assert all(c == 0 for c in layer_call_counts[1:]), (
        f"Layers 1..{num_layers-1} must have 0 traces, got: {layer_call_counts[1:]}"
    )


def test_stitch_dynamo_monolithic_graph():
    """Verify Dynamo 1-layer template capture and N-layer stitching with 100% numerical parity."""
    from vllm_torchtpu.compilation.layer_stitcher import \
        stitch_dynamo_monolithic_graph

    num_layers = 4
    hidden_size = 32
    model = MockTransformerModel(num_layers=num_layers,
                                 hidden_size=hidden_size)
    input_ids = torch.randint(0, 64, (1, 8), dtype=torch.long)
    with torch.no_grad():
        expected_out = model(input_ids)

    # 1. Truncate to 1 layer
    orig_layers = list(model.layers)
    model.layers = nn.ModuleList([orig_layers[0]])

    captured_gm = None
    captured_inputs = None

    def my_backend(gm, sample_inputs):
        nonlocal captured_gm, captured_inputs
        captured_gm = gm
        captured_inputs = list(sample_inputs)
        return gm

    torch.compile(model.forward,
                  fullgraph=True,
                  dynamic=False,
                  backend=my_backend)(input_ids)
    model.layers = nn.ModuleList(orig_layers)

    assert captured_gm is not None
    assert captured_inputs is not None

    # 2. Stitch to 4 layers
    stitched_gm, full_inputs, inputs_builder = stitch_dynamo_monolithic_graph(
        captured_gm,
        captured_inputs,
        model,
        num_total_layers=num_layers,
        layers_attr="layers",
    )

    with torch.no_grad():
        actual_out = stitched_gm(*inputs_builder([input_ids]))
        if isinstance(actual_out, (tuple, list)):
            actual_out = actual_out[0]

    torch.testing.assert_close(
        actual_out,
        expected_out,
        rtol=1e-4,
        atol=1e-4,
        msg=
        "Stitched Dynamo GraphModule execution output must match eager model output",
    )
