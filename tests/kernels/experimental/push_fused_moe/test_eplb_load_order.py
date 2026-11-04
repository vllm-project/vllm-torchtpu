# SPDX-License-Identifier: Apache-2.0
"""CPU contract for push EPLB load accumulation after mesh relabeling."""

import ast
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
KERNELS = ROOT / "src/vllm_torchtpu/kernels"


@pytest.mark.parametrize("ranks", [None, (0, 1, 2), (2, 0, 1)])
def test_push_load_accumulation_uses_ep_rank_order(ranks):
    source = ast.parse((KERNELS / "fused_moe/v2/layer.py").read_text())
    helper = next(
        node
        for node in source.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_unpermute_load_to_ep_rank_order"
    )
    push = ast.parse((KERNELS / "experimental/push_fused_moe/layer.py").read_text())
    assignment = next(
        node
        for node in ast.walk(push)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "updated_load" for t in node.targets
        )
    )
    ns = {"jnp": jnp}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), "helper", "exec"), ns)
    counts = np.array([10, 11, 20, 21, 30, 31], dtype=np.int32)
    ns.update(
        rows_by_dest=jnp.asarray(counts[None, :]),
        expert_load_l=jnp.ones(6, dtype=jnp.int32),
        g_local=2,
        mesh_ep_ranks=ranks,
    )
    code = compile(ast.Module(body=[assignment], type_ignores=[]), "push", "exec")
    exec(code, ns)
    expected = counts if ranks != (2, 0, 1) else counts[[2, 3, 4, 5, 0, 1]]
    np.testing.assert_array_equal(ns["updated_load"], expected + 1)
    ns["expert_load_l"] = None
    exec(code, ns)
    assert ns["updated_load"] is None
