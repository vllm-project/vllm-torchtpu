# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_PLE_CONV_ONEHOT wiring in Qwen4ExpPooledPLE (CPU; device ops are recording doubles).
The kernel is tested in tests/kernels/test_qwen4_exp_ple_onehot.py."""

from types import SimpleNamespace

import pytest
import torch

from tests.models.vllm.test_qwen4_exp import (
    _PLE,
    HC,
    HIDDEN,
    _randomize_ple,
    _set_tp,
    _text_config,
)
from vllm_torchtpu.models.vllm.qwen4_exp.ple import PLEInputs


class _Group:
    rank_in_group = 0
    world_size = 1

    def all_reduce(self, rows):
        return rows


def _adapter(monkeypatch, env):
    from vllm.utils.torch_utils import set_default_torch_dtype

    from vllm_torchtpu.models.vllm.qwen4_exp import ple_storage, pooled_ple
    from vllm_torchtpu.models.vllm.qwen4_exp.parallel import PLEParallel

    for name in (
        "VLLM_TPU_PLE_CONV_ONEHOT",
        "Q38_PROBE_PLE_ONEHOT_MUTANT",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    _set_tp(monkeypatch, 0, 1)
    group = _Group()

    def raw_lookup(rows, width, chunk_tokens=8):
        def run(table, ids, vocab_start):
            result = table.reshape(rows, width)[ids.clamp(0, rows - 1)].clone()
            result[(ids < 0) | (ids >= rows)] = 0
            return result.flatten(-2)

        return run

    monkeypatch.setattr(ple_storage, "build_raw_lookup_op", raw_lookup)
    monkeypatch.setattr(
        ple_storage, "build_decode_op", lambda: lambda raw, scale: raw.view(torch.bfloat16)
    )
    log = SimpleNamespace(conv=[], spec=0, fused=[], built=[])

    def conv_builder(dilation, state_len, past_matmul=False):
        log.built.append(("conv", dilation, state_len, past_matmul))

        def run(x, weight, pool, starts, ids, initial):
            log.conv.append(x.shape[0])
            return torch.zeros_like(x)

        return run

    def spec_builder(dilation, state_len):
        def run(x, *args):
            log.spec += 1
            return torch.zeros_like(x)

        return run

    monkeypatch.setattr(pooled_ple, "build_pooled_short_conv_op", conv_builder)
    monkeypatch.setattr(pooled_ple, "build_pooled_speculative_short_conv_op", spec_builder)
    with set_default_torch_dtype(torch.bfloat16):
        reference = pooled_ple.Qwen4ExpPLELayer(_text_config(**_PLE), 0, None, "ple")
        _randomize_ple(reference)
        adapted = pooled_ple.Qwen4ExpPooledPLE(
            _text_config(**_PLE), 0, None, "ple", parallel=PLEParallel(group, group,
            embedding_across_dp=False), table_group=group,
        )
    weights = dict(reference.state_dict())
    table = weights.pop("ple_embedding.ngram_embedding.weight")
    for i, rows in enumerate(table.split(22)):
        weights[f"ple_embedding.ngram_embedding.shard_{i}.weight"] = rows
    adapted.load_checkpoint(iter(weights.items()))
    adapted.finalize_weights()
    pool = torch.zeros(5, 16, dtype=torch.bfloat16)
    adapted.bind_pool(pool)
    return adapted, pool, log


def _step(adapted, tokens, speculative=None):
    ids = torch.arange(tokens, dtype=torch.int64) % 7 + 4
    starts = torch.tensor([0, tokens // 2, tokens, tokens], dtype=torch.int32)
    slots = torch.tensor([2, 3, -1], dtype=torch.int32)
    initial = torch.tensor([True, False, False])
    contexts = torch.full((3, 2), _PLE["eos_token_id"], dtype=torch.int64)
    hidden = torch.randn(tokens, HC * HIDDEN).to(torch.bfloat16)
    valid = starts[-1:]
    with torch.no_grad():
        out = adapted.forward_with_history(
            hidden, PLEInputs(ids, starts, contexts), slots, initial, valid, None, speculative
        )
    return out, dict(hidden=hidden, starts=starts, slots=slots, initial=initial, valid=valid)


def test_onehot_flag_reaches_the_pooled_conv_builder(monkeypatch):
    _, _, log = _adapter(monkeypatch, {"VLLM_TPU_PLE_CONV_ONEHOT": "1"})
    assert [b[3] for b in log.built if b[0] == "conv"] == [True]


def test_probe_mutant_flag_reaches_the_pooled_conv_builder(monkeypatch):
    _, _, log = _adapter(monkeypatch, {"Q38_PROBE_PLE_ONEHOT_MUTANT": "1"})
    assert [b[3] for b in log.built if b[0] == "conv"] == ["mutant"]


def test_default_conv_builder_has_no_onehot(monkeypatch):
    _, _, log = _adapter(monkeypatch, {})
    assert [b[3] for b in log.built if b[0] == "conv"] == [False]


def test_pooled_short_conv_op_builds_through_jax_op():
    """The one-hot conv op builds through torch_tpu's real pallas.jax_op (VM venv)."""
    from vllm_torchtpu.layers.adapter.custom_ops import ple_op

    assert callable(ple_op.build_pooled_short_conv_op(3, 9, True))
    assert callable(ple_op.build_pooled_short_conv_op(3, 9, "mutant"))
