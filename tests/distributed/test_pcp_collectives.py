# SPDX-License-Identifier: Apache-2.0

import builtins

import torch

from vllm_torchtpu.distributed import pcp


class _FakePcpGroup:

    world_size = 2

    def all_gather(self, tensor: torch.Tensor, dim: int = 0):
        assert tensor.is_contiguous()
        return torch.cat([tensor, tensor], dim=dim)


def test_all_gather_equal_tokens_materializes_contiguous_input(monkeypatch):
    monkeypatch.setattr(pcp, "get_pcp_group", lambda: _FakePcpGroup())
    tensor = torch.arange(12, dtype=torch.float32).reshape(3,
                                                           4).transpose(0, 1)

    gathered = pcp.all_gather_equal_tokens(tensor, dim=0)

    assert torch.equal(gathered, torch.cat([tensor, tensor], dim=0))


def test_get_pcp_group_returns_none_when_native_symbol_is_missing(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "vllm.distributed.parallel_state":
            raise ImportError("native PCP group is unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert pcp.get_pcp_group() is None
