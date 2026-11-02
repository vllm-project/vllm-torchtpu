# SPDX-License-Identifier: Apache-2.0
"""The common Engram path must bind each layer's own staged rows and weights."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "src/vllm_torchtpu/models/vllm/deepseek_v41/common/engram.py"
)
pytestmark = pytest.mark.cpu_test


def test_fused_bridge_has_annotations_for_torch_tpu_schema_inference():
    path = SOURCE.parents[4] / "kernels/engram/vllm_adapter.py"
    tree = ast.parse(path.read_text())
    bridge = next(n for n in tree.body if getattr(n, "name", None) == "_engram_jax")
    assert all(arg.annotation is not None for arg in bridge.args.args)
    assert bridge.returns is not None
    assert len(bridge.args.args) == 9


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


def test_both_engram_layers_bind_their_own_inputs():
    calls = []

    def fused(*args):
        calls.append(args)
        return args[1]

    module = methods(fused_engram_torch=fused)
    mask = torch.arange(288) % 3 != 0
    for layer_id in (1, 14):
        layer = owner(layer_id)
        h = hidden()
        assert module._uses_fused_offload(layer, h)
        assert module._fused_forward(layer, h, mask) is h
        call = calls[-1]
        assert call[0].data_ptr() == layer._active_raw.data_ptr()
        assert call[2] is layer.wkv.weight
        assert call[3] is layer.wkv.weight_scale
        assert call[4] is layer.q_weight
        assert call[5] is layer.k_weight
        assert call[6] is mask
    assert calls[0][0][0] == 1
    assert calls[1][0][0] == 14


@pytest.mark.parametrize("tokens", [1, 128, 256, 2048])
def test_other_buckets_keep_the_original_path(tokens):
    assert not methods()._uses_fused_offload(owner(1), hidden(tokens))


@pytest.mark.parametrize(
    "name,value",
    [("host_offload", False), ("dp_size", 8), ("tp_size", 2), ("block_size", 64)],
)
def test_sharded_or_different_embedding_contract_keeps_the_original_path(name, value):
    layer = owner(1)
    setattr(layer.embed_tokens, name, value)
    assert not methods()._uses_fused_offload(layer, hidden())


def test_missing_mask_uses_an_all_true_mask():
    calls = []
    module = methods(fused_engram_torch=lambda *args: calls.append(args))
    h = torch.zeros((288, 4, 5120), dtype=torch.bfloat16)
    module._fused_forward(owner(1), h, None)
    assert calls[0][6].dtype == torch.bool
    assert calls[0][6].shape == (288,)
    assert calls[0][6].all()
