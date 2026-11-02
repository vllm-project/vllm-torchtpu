# SPDX-License-Identifier: Apache-2.0
"""Keep the unaccepted Engram candidate outside the production forward path."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "src/vllm_torchtpu/models/vllm/deepseek_v41/common/engram.py"
)
BRIDGE = SOURCE.parents[4] / "kernels/engram/vllm_adapter.py"
pytestmark = pytest.mark.cpu_test


def test_fused_bridge_has_annotations_for_torch_tpu_schema_inference():
    tree = ast.parse(BRIDGE.read_text())
    bridge = next(n for n in tree.body if getattr(n, "name", None) == "_engram_jax")
    assert all(arg.annotation is not None for arg in bridge.args.args)
    assert bridge.returns is not None
    assert len(bridge.args.args) == 9


def test_unaccepted_fusion_does_not_change_the_production_forward():
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if getattr(n, "name", None) == "Engram")
    assert not any(
        isinstance(n, ast.ImportFrom)
        and n.module == "vllm_torchtpu.kernels.engram.vllm_adapter"
        for n in tree.body
    )
    assert not any(
        getattr(n, "name", None) in {"_uses_fused_offload", "_fused_forward"}
        for n in cls.body
    )
    forward = next(n for n in cls.body if getattr(n, "name", None) == "forward")
    assert isinstance(forward.body[0], ast.If)
    assert isinstance(forward.body[0].test, ast.Attribute)
    assert forward.body[0].test.attr == "use_sequence_parallel"


def test_standalone_jax_bridge_selects_only_the_target_token_bucket():
    tree = ast.parse(BRIDGE.read_text())
    bridge = next(n for n in tree.body if getattr(n, "name", None) == "_engram_jax")
    bridge = ast.FunctionDef(
        name=bridge.name,
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=x.arg) for x in bridge.args.args],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=bridge.body,
        decorator_list=[],
    )
    calls = []
    namespace = dict(
        fused_engram=lambda *args: calls.append("fused") or args[1],
        dequantize_rows=lambda *args: args[0],
        xla_quantized_matmul=lambda *args: args[0],
        gated_residual=lambda *args: calls.append("unfused") or args[0],
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[bridge], type_ignores=[])),
            str(BRIDGE),
            "exec",
        ),
        namespace,
    )
    for tokens in (0, 128, 288, 2048):
        hidden = torch.empty((tokens, 4, 5120), device="meta")
        packed = torch.empty((tokens * 1584,), device="meta")
        result = namespace["_engram_jax"](
            packed,
            hidden,
            SimpleNamespace(shape=(6144, 25600)),
            None,
            None,
            None,
            None,
            1e-20,
            1e-6,
        )
        assert result is hidden
    assert calls == ["unfused", "fused", "unfused"]
