# SPDX-License-Identifier: Apache-2.0
"""Bridge registration validates signatures before any TPU compilation."""

import jax
import numpy as np


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

    class Bridge(torch.nn.Module):
        def forward(self, raw, table, starts, positions, minimum, values):
            cache = Cache(raw)
            return csa2_adapter.prepare_publish_swa(
                table, starts, positions, minimum, cache, values
            )

    arrays = (
        torch.zeros((512, 64, 512), dtype=torch.bfloat16),
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
