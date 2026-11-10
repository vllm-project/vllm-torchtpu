# SPDX-License-Identifier: Apache-2.0
"""Bridge registration validates signatures before any TPU compilation."""

import jax
import numpy as np
import pytest


@pytest.mark.cpu_test
@pytest.mark.parametrize("capacity", [8, 16])
@pytest.mark.parametrize("mode", ["auto", "off"])
@pytest.mark.parametrize("region", ["compressor_state", "compressor_addresses"])
def test_compressor_admission_fullgraph(monkeypatch, caplog, capacity, mode, region):
    """Include the AUTO fallback diagnostic, not just the layout comparison."""
    from types import SimpleNamespace

    import torch

    from vllm_torchtpu.kernels.experimental import decode_preparation as prep
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention import (
        DeepseekV41Attention,
    )

    monkeypatch.setattr(prep, "_mode", prep.FusionMode(mode))
    monkeypatch.setattr(prep, "_mode_used", False)
    monkeypatch.setattr(prep, "_reported", set())
    torch._dynamo.reset()
    owner = SimpleNamespace(window_size=128)

    class Cache:
        def __init__(self, shape, stride, offset=0):
            self.shape = shape
            self.page_stride = stride
            self.row_offset = offset

    def forward(x, raw):
        # Match cache properties that construct a fresh view during forward.
        state = Cache(raw.shape, capacity)
        main = Cache((1, 128), 128)
        index = Cache((1, 256), 256)
        admitted = DeepseekV41Attention._compressor_layout_admitted(
            owner, region, state, main, index
        )
        return x + (1 if admitted else 2)

    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    compiled = torch.compile(forward, backend=backend, fullgraph=True, dynamic=False)
    raw = torch.zeros((1, capacity))
    with caplog.at_level("WARNING"):
        for tokens in (96, 288, 2048):
            x = torch.zeros(tokens)
            torch._dynamo.mark_dynamic(x, 0)
            expected = x + (1 if capacity == 16 and mode == "auto" else 2)
            torch.testing.assert_close(compiled(x, raw), expected)
    assert len(graphs) == 1  # Token count must remain dynamic.
    warnings = [r for r in caplog.records if "falls back" in r.message]
    assert len(warnings) == (1 if capacity == 8 and mode == "auto" else 0)
    if warnings:
        assert "state_capacity=8" in warnings[0].message


def test_csa2_bridge_registration(monkeypatch):
    from types import SimpleNamespace

    from vllm_torchtpu.models.vllm import vllm_model_wrapper_context as context
    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import csa2_adapter

    mesh = jax.sharding.Mesh(np.array(jax.devices("cpu")[:1]), ("data",))
    monkeypatch.setattr(
        context,
        "get_vllm_model_wrapper_context",
        lambda: SimpleNamespace(mesh=mesh),
    )
    monkeypatch.setattr(context, "get_pallas_op_mesh", lambda: mesh)
    csa2_adapter.initialize_csa2_ops()
    assert "prepare_publish_swa" in csa2_adapter._ops
    assert "prepare_publish_swa_banked" in csa2_adapter._ops


def test_swa_bridge_preserves_dynamic_token_dimension(monkeypatch):
    from types import SimpleNamespace

    import torch

    from vllm_torchtpu.models.vllm import vllm_model_wrapper_context as context
    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import csa2_adapter

    mesh = jax.sharding.Mesh(np.array(jax.devices("cpu")[:1]), ("data",))
    monkeypatch.setattr(
        context, "get_vllm_model_wrapper_context", lambda: SimpleNamespace(mesh=mesh)
    )
    monkeypatch.setattr(context, "get_pallas_op_mesh", lambda: mesh)
    csa2_adapter.initialize_csa2_ops()

    class Cache:
        def __init__(self, raw):
            self.raw = raw
            self.raw_hi = None
            # Second SWA page of a block-outer 512 KiB block.
            self.page_stride = 512
            self.row_offset = 64

    class Bridge(torch.nn.Module):
        def forward(self, raw, table, starts, positions, minimum, values):
            cache = Cache(raw)
            return csa2_adapter.prepare_publish_swa(
                table, starts, positions, minimum, cache, values
            )

    arrays = (
        torch.zeros((512 * 128, 128), dtype=torch.int32),
        torch.ones((48, 1539), dtype=torch.int32),
        torch.arange(49, dtype=torch.int32) * 6,
        torch.arange(288, dtype=torch.int32),
        torch.zeros(48, dtype=torch.int32),
        torch.zeros((288, 512), dtype=torch.bfloat16),
    )
    tokens = torch.export.Dim("tokens", min=2, max=2048)
    exported = torch.export.export(
        Bridge(),
        arrays,
        dynamic_shapes=(None, None, None, {0: tokens}, None, {0: tokens}),
        strict=True,
    )
    assert any("prepare_publish_swa" in str(n.target) for n in exported.graph.nodes)
