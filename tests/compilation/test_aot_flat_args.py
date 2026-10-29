# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn


class _Layer(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.lin = nn.Linear(h, h, bias=False)
        self.register_buffer("scale", torch.ones(h))
        self.kv = torch.zeros(2, h)

    def forward(self, x):
        return self.lin(x) * self.scale + self.kv.sum(0)


class _Model(nn.Module):
    def __init__(self, h=8, n=6):
        super().__init__()
        self.layers = nn.ModuleList(_Layer(h) for _ in range(n))

    def forward(self, x, pos):
        for layer in self.layers:
            x = layer(x)
        self.layers[0].kv.add_(0)
        return x + pos.unsqueeze(-1) * x.shape[0]


def _inputs(n):
    x, pos = torch.randn(n, 8), torch.arange(float(n))
    torch._dynamo.mark_dynamic(x, 0)
    torch._dynamo.mark_dynamic(pos, 0)
    return x, pos


def test_flat_args_cache_matches_and_canary_catches_stale(monkeypatch):
    """VLLM_TPU_AOT_FLAT_ARGS: after the recorded calls the forward calls the
    compiled graph with the cached flat list (same outputs, sizes and keyword
    arguments filled per call), and a canary call switches it off when a
    lifted tensor attribute is replaced."""
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation import aot_flat_args

    monkeypatch.setattr(envs, "VLLM_TPU_AOT_FLAT_ARGS", True, raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_AOT_FLAT_ARGS_CHECK", 0, raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_AOT_FLAT_ARGS_CANARY", 4, raising=False)
    aot_flat_args.patch_aot_flat_args()
    model = _Model()
    compiled = torch.compile(
        _Model.forward, fullgraph=True, backend="eager"
    ).aot_compile(((model, *_inputs(4)), {}))
    compiled._guard_check_enabled = False
    fwd = compiled.fn
    assert isinstance(fwd, aot_flat_args.FlatArgsForward)
    for i in range(40):
        x, pos = _inputs((4, 8, 12)[i % 3])
        assert torch.equal(compiled(model, x, pos=pos), _Model.forward(model, x, pos))
    (layout,) = fwd.layouts.values()
    assert layout.ready and fwd.calls > 0 and not fwd.off
    model.layers[2].kv = torch.ones(2, 8)
    for _ in range(4):
        x, pos = _inputs(4)
        compiled(model, x, pos=pos)
    assert fwd.off
