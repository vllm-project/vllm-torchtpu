# SPDX-License-Identifier: Apache-2.0
"""Packed ACC Engram integration must keep each layer's own storage contract."""

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


def methods(**namespace):
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if getattr(n, "name", None) == "Engram")
    names = {"_uses_fused_offload", "_fused_forward"}
    body = [n for n in cls.body if getattr(n, "name", None) in names]
    namespace.update(torch=torch)
    exec(
        compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), "exec"), namespace
    )
    return SimpleNamespace(**namespace)


def owner(index):
    return SimpleNamespace(
        _has_fused_offload_bucket=True,
        use_sequence_parallel=False,
        uses_hbm_lookup=False,
        embed_tokens=SimpleNamespace(
            host_offload=True,
            dp_size=1,
            tp_size=1,
            block_size=32,
            n_hash_cols=24,
        ),
        _host_raw=torch.empty((1, 24, 256), dtype=torch.uint8),
        _active_raw=torch.full((288, 1584), index, dtype=torch.int32),
        wkv=SimpleNamespace(
            weight=torch.empty((6144, 25600), device="meta", dtype=torch.float8_e4m3fn),
            weight_scale=torch.empty((25600,), device="meta", dtype=torch.float32),
        ),
        q_weight=torch.tensor([index]),
        k_weight=torch.tensor([index + 1]),
        eps=1e-20,
        clamp_value=1e-6,
    )


def hidden(tokens=288):
    return SimpleNamespace(
        device=SimpleNamespace(type="tpu"),
        dtype=torch.bfloat16,
        shape=(tokens, 4, 5120),
    )


def test_fused_bridge_has_annotations_for_torch_tpu_schema_inference():
    tree = ast.parse(BRIDGE.read_text())
    bridge = next(n for n in tree.body if getattr(n, "name", None) == "_engram_jax")
    assert all(arg.annotation is not None for arg in bridge.args.args)
    assert bridge.returns is not None
    assert len(bridge.args.args) == 9


def test_both_layers_bind_their_own_staged_rows_and_weights():
    calls = []
    module = methods(fused_engram_torch=lambda *args: calls.append(args) or args[1])
    for layer_id in (1, 14):
        layer, h = owner(layer_id), hidden()
        assert module._uses_fused_offload(layer, h)
        assert module._fused_forward(layer, h, torch.ones(288, dtype=torch.bool)) is h
        call = calls[-1]
        assert call[0].data_ptr() == layer._active_raw.data_ptr()
        assert call[2] is layer.wkv.weight
        assert call[3] is layer.wkv.weight_scale
        assert call[4] is layer.q_weight
        assert call[5] is layer.k_weight


@pytest.mark.parametrize("tokens", [0, 1, 128, 256, 288, 2048])
def test_torch_eligibility_does_not_specialize_dynamic_tokens(tokens):
    assert methods()._uses_fused_offload(owner(1), hidden(tokens))


@pytest.mark.parametrize(
    "buckets,enabled", [([128, 2048], False), ([288, 2048], True), (None, False)]
)
def test_staging_derives_eligibility_from_static_compilation_buckets(buckets, enabled):
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if getattr(n, "name", None) == "Engram")
    method = next(n for n in cls.body if getattr(n, "name", None) == "_init_staging")
    body = []
    for node in method.body:
        if isinstance(node, ast.If):
            break
        body.append(node)
    config = SimpleNamespace(compilation_config=SimpleNamespace(compile_sizes=buckets))
    layer = SimpleNamespace()
    namespace = dict(self=layer, get_current_vllm_config_or_none=lambda: config)
    exec(
        compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), "exec"), namespace
    )
    assert layer._has_fused_offload_bucket is enabled


def test_prefill_only_configuration_never_enters_the_jax_fusion_bridge():
    layer = owner(1)
    layer._has_fused_offload_bucket = False
    for tokens in (128, 288, 2048):
        assert not methods()._uses_fused_offload(layer, hidden(tokens))


@pytest.mark.parametrize("name", ["uses_hbm_lookup", "use_sequence_parallel"])
def test_other_storage_or_sequence_parallel_paths_remain_original(name):
    layer = owner(1)
    setattr(layer, name, True)
    assert not methods()._uses_fused_offload(layer, hidden())


@pytest.mark.parametrize(
    "name,value",
    [("host_offload", False), ("dp_size", 8), ("tp_size", 2), ("block_size", 64)],
)
def test_other_table_contracts_keep_the_original_path(name, value):
    layer = owner(1)
    setattr(layer.embed_tokens, name, value)
    assert not methods()._uses_fused_offload(layer, hidden())


def test_missing_mask_uses_an_all_true_mask():
    calls = []
    module = methods(fused_engram_torch=lambda *args: calls.append(args))
    h = torch.zeros((288, 4, 5120), dtype=torch.bfloat16)
    module._fused_forward(owner(1), h, None)
    assert calls[0][6].dtype == torch.bool
    assert calls[0][6].all()


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
        fused_engram_acc=lambda *args: calls.append("fused") or args[1],
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
        h = torch.empty((tokens, 4, 5120), device="meta")
        packed = torch.empty((tokens * 1584,), device="meta")
        result = namespace["_engram_jax"](
            packed,
            h,
            SimpleNamespace(shape=(6144, 25600)),
            None,
            None,
            None,
            None,
            1e-20,
            1e-6,
        )
        assert result is h
    assert calls == ["unfused", "fused", "unfused"]
