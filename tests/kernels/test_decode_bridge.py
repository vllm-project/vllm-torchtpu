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
