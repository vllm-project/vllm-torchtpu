# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the GLM 5.3 token-dimension TP collectives."""

from __future__ import annotations

import types

import torch

from vllm_torchtpu.layers.adapter.custom_ops import glm53_collectives_tpu


def test_eager_collectives_delegate_to_tp_group(monkeypatch):
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    fake_tp = types.SimpleNamespace(
        rank_in_group=1,
        world_size=2,
        all_gather=lambda t, dim=0: torch.cat([t, t + 100], dim=dim),
        reduce_scatter=lambda t, dim=0: t[3:] * 2,
    )
    monkeypatch.setattr(glm53_collectives_tpu, "get_tp_group", lambda: fake_tp)
    ag_tok = glm53_collectives_tpu.token_all_gather(x)
    assert ag_tok.shape == (6, 4)
    torch.testing.assert_close(ag_tok[:3], x)
    torch.testing.assert_close(ag_tok[3:], x + 100)

    ag_voc = glm53_collectives_tpu.vocab_all_gather(x)
    assert ag_voc.shape == (3, 8)
    torch.testing.assert_close(ag_voc[:, :4], x)
    torch.testing.assert_close(ag_voc[:, 4:], x + 100)

    rs_tok = glm53_collectives_tpu.token_reduce_scatter(ag_tok)
    assert rs_tok.shape == (3, 4)
    torch.testing.assert_close(rs_tok, (x + 100) * 2)


def test_all_reduce_fallback_formulations(monkeypatch):
    shards = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
    ]
    full_tok = torch.cat(shards, dim=0)
    full_voc = torch.cat(shards, dim=-1)

    for rank in (0, 1):
        fake_tp = types.SimpleNamespace(rank_in_group=rank, world_size=2)
        monkeypatch.setattr(
            glm53_collectives_tpu, "get_tp_group", lambda tp=fake_tp: tp
        )
        monkeypatch.setattr(
            glm53_collectives_tpu,
            "tensor_model_parallel_all_reduce",
            lambda t: full_tok if t.shape[0] == 4 else full_voc,
        )
        got_tok = glm53_collectives_tpu._gather_by_all_reduce(shards[rank])
        torch.testing.assert_close(got_tok, full_tok)
        got_voc = glm53_collectives_tpu._vocab_gather_by_all_reduce(shards[rank])
        torch.testing.assert_close(got_voc, full_voc)

        monkeypatch.setattr(
            glm53_collectives_tpu,
            "tensor_model_parallel_all_reduce",
            lambda t: t * 2,
        )
        got_rs = glm53_collectives_tpu._scatter_by_all_reduce(full_tok)
        torch.testing.assert_close(got_rs, shards[rank] * 2)


def test_spmd_collectives_use_per_output_partition_spec_tuple(monkeypatch):
    """Regression guard: torch_tpu's jax_op requires a 1-tuple of PartitionSpec
    for single-output ops so multi-character axis names ('glm53_sp') are not
    iterated character-by-character."""
    import sys

    from jax.sharding import PartitionSpec as P

    captured: list[dict[str, object]] = []

    class _FakeOp:
        def __init__(self) -> None:
            self.fake_fn = None

        def register_fake(self, fn):
            self.fake_fn = fn

    def fake_jax_op(name, fn, **kwargs):
        op = _FakeOp()
        captured.append({"name": name, "fn": fn, "op": op, **kwargs})
        return op

    fake_pallas = types.ModuleType("torch_tpu._internal.pallas")
    fake_pallas.jax_op = fake_jax_op  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch_tpu", types.ModuleType("torch_tpu"))
    monkeypatch.setitem(
        sys.modules, "torch_tpu._internal", types.ModuleType("torch_tpu._internal")
    )
    monkeypatch.setitem(sys.modules, "torch_tpu._internal.pallas", fake_pallas)
    fake_mesh = types.SimpleNamespace(devices=types.SimpleNamespace(size=8))
    monkeypatch.setattr(glm53_collectives_tpu, "_mesh", lambda: fake_mesh)
    glm53_collectives_tpu._build.cache_clear()
    glm53_collectives_tpu._build_vocab_gather.cache_clear()

    try:
        ag_op = glm53_collectives_tpu._build(True, 2)
        rs_op = glm53_collectives_tpu._build(False, 2)
        voc_op = glm53_collectives_tpu._build_vocab_gather(2)

        assert len(captured) == 3
        for entry in captured:
            out_specs = entry["output_partition_specs"]
            in_specs = entry["input_partition_specs"]
            assert isinstance(in_specs, tuple) and len(in_specs) == 1
            assert isinstance(out_specs, tuple) and len(out_specs) == 1
            assert isinstance(out_specs[0], P)

        assert ag_op.fake_fn(torch.empty(256, 4096)).shape == (2048, 4096)
        assert rs_op.fake_fn(torch.empty(2048, 4096)).shape == (256, 4096)
        assert voc_op.fake_fn(torch.empty(16, 19200)).shape == (16, 153600)
    finally:
        glm53_collectives_tpu._build.cache_clear()
        glm53_collectives_tpu._build_vocab_gather.cache_clear()
