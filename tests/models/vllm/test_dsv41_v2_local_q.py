"""token-local q projection CPU contracts."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_torchtpu.layers.adapter.linear_common import KEEP_VLLM_LAYOUT_ATTR
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention as v2_attn

TP = 2


def _fake_all_gather(tensor, dim):
    """Two ranks: rank 1's shard is rank 0's with every byte/value flipped."""
    other = tensor ^ 0xFF if tensor.dtype == torch.uint8 else -tensor
    return torch.cat((tensor, other), dim=dim)


def _stub(monkeypatch, *, enabled=True, tp=TP, compress_ratio=4, groups=1):
    monkeypatch.setattr(v2_attn, "get_tensor_model_parallel_world_size", lambda: tp)
    monkeypatch.setattr(v2_attn, "tensor_model_parallel_all_gather", _fake_all_gather)
    attn = object.__new__(v2_attn.DeepseekV41Attention)
    torch.nn.Module.__init__(attn)
    attn._fp8_kv = True
    attn.compress_ratio = compress_ratio
    attn.n_local_groups = groups
    attn._local_q_attn = False
    attn._token_parallel_attn = enabled
    gen = torch.Generator().manual_seed(0)
    wq = torch.randn(16, 8, generator=gen).to(torch.float8_e4m3fn)  # (k, n_local)
    wo = torch.randn(32, 4, generator=gen).to(torch.float8_e4m3fn)  # (D_g, R)
    attn.wq_b = SimpleNamespace(weight=wq, weight_scale=torch.rand(8, generator=gen))
    attn.wo_a = SimpleNamespace(weight=wo, weight_scale=torch.rand(1, 4, generator=gen))
    return attn


def _flip(t):
    return (t.view(torch.uint8) ^ 0xFF).view(t.dtype)


def test_init_local_q_weights_gathers_rank_major_columns(monkeypatch):
    attn = _stub(monkeypatch)
    attn._init_local_q_weights()
    assert attn._local_q_attn is True
    wq, wo = attn.wq_b.weight, attn.wo_a.weight
    assert attn._wq_b_full.dtype == torch.float8_e4m3fn
    assert torch.equal(
        attn._wq_b_full.view(torch.uint8),
        torch.cat((wq, _flip(wq)), 1).view(torch.uint8),
    )
    assert torch.equal(
        attn._wq_b_full_scale,
        torch.cat((attn.wq_b.weight_scale, -attn.wq_b.weight_scale)),
    )
    assert torch.equal(
        attn._wo_a_full.view(torch.uint8),
        torch.cat((wo, _flip(wo)), 1).view(torch.uint8),
    )
    scale = attn.wo_a.weight_scale.reshape(-1)
    assert torch.equal(attn._wo_a_full_scale, torch.cat((scale, -scale)))
    # Not persistent: never written to / expected from checkpoints.
    assert "_wq_b_full" not in attn.state_dict()


def test_init_local_q_weights_keep_vllm_layout_disables(monkeypatch):
    attn = _stub(monkeypatch)
    setattr(attn.wq_b, KEEP_VLLM_LAYOUT_ATTR, True)  # stored [n_out, n_in]
    attn._init_local_q_weights()
    assert attn._local_q_attn is False


@pytest.mark.parametrize(
    "kwargs,extra",
    [
        ({"enabled": False}, {}),
        ({"tp": 1}, {}),
        ({"compress_ratio": 0}, {}),
        ({"compress_ratio": 4}, {"_fp8_kv": False}),
        ({"compress_ratio": 4}, {"_ced_decoder": True}),
        ({"groups": 2}, {}),
    ],
)
def test_init_local_q_weights_disabled(monkeypatch, kwargs, extra):
    attn = _stub(monkeypatch, **kwargs)
    for k, v in extra.items():
        setattr(attn, k, v)
    attn._init_local_q_weights()
    assert attn._local_q_attn is False
    assert not hasattr(attn, "_wq_b_full")


def test_init_local_q_weights_requires_per_channel_scales(monkeypatch):
    attn = _stub(monkeypatch)
    attn.wq_b.weight_scale = torch.ones(1, 8)
    attn._init_local_q_weights()
    assert attn._local_q_attn is False
    attn = _stub(monkeypatch)
    attn.wo_a = SimpleNamespace(weight=attn.wo_a.weight)
    attn._init_local_q_weights()
    assert attn._local_q_attn is False


@pytest.mark.parametrize("local", [False, True])
def test_forward_routes_projected_output_to_wo_b(monkeypatch, local):
    attn = _stub(monkeypatch)
    attn._local_q_attn = local
    attn._sp_sharded_input = True
    attn.padded_heads = attn.n_local_heads = 2
    attn.head_dim = 4
    attn.q_lora_rank = 3
    attn.indexer = None
    tokens = 4
    qr = torch.arange(2 * 3, dtype=torch.float32).reshape(2, 3)
    kv = torch.zeros(2, 4)
    gathered_widths = []
    monkeypatch.setattr(
        attn, "_run_parallel_input_projections", lambda h: (None, None, None)
    )
    monkeypatch.setattr(attn, "_split_qkv_and_norm", lambda x: (qr, None, kv))

    def fake_sp_gather(t):
        gathered_widths.append(t.shape[-1])
        return torch.cat((t, t))

    monkeypatch.setattr(v2_attn, "sp_all_gather", fake_sp_gather)
    seen: dict[str, Any] = {}
    projected = torch.full((tokens, 5), 7.0)

    def prepare(*args, qr_local=None):
        seen["qr_local"] = qr_local
        seen["qr"] = args[1]
        seen["kv"] = args[2]
        return projected if qr_local is not None else None

    monkeypatch.setattr(attn, "_prepare_and_attn_fn", prepare, raising=False)
    attn.wo_b = lambda x: (x * 2, None)
    monkeypatch.setattr(attn, "_o_proj", lambda o, positions: "o_proj")
    out = attn.forward(torch.zeros(tokens, dtype=torch.long), torch.zeros(2, 8))
    if local:
        assert seen["qr_local"] is qr
        assert seen["qr"] is qr
        assert seen["kv"].shape == (tokens, attn.head_dim)
        assert gathered_widths == [attn.head_dim]
        assert torch.equal(out, projected * 2)
    else:
        assert seen["qr_local"] is None
        assert seen["qr"].shape == (tokens, attn.q_lora_rank)
        assert gathered_widths == [attn.q_lora_rank + attn.head_dim]
        assert out == "o_proj"
