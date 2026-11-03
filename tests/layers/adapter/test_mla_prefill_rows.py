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
"""TPU_MLA_PREFILL_SEQUENCE_ROWS: the bucket gate, the refusals and which MoE
layers take the rows block. No TPU."""

from types import SimpleNamespace

import pytest
import torch

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.adapter import dcp_token_parallel_op as tp_op
from vllm_torchtpu.layers.adapter import mla_prefill_rows as rows
from vllm_torchtpu.layers.adapter import moe_prefill_sp as sp

_ON = dict(
    TPU_MLA_PREFILL_SEQUENCE_ROWS=True,
    TPU_MLA_PREFILL_ROWS_MIN_TOKENS=2048,
    TPU_MOE_PREFILL_SEQUENCE_PARALLEL=True,
    TPU_DCP_PREFILL_TOKEN_PARALLEL=True,
)


def _set_envs(monkeypatch, **values):
    for k, v in (_ON | values).items():
        monkeypatch.setattr(envs, k, v)


@pytest.mark.parametrize(
    "flags,tag,tokens,want",
    [
        ({}, "backbone", 8192, True),
        ({}, "backbone", 2048, True),
        ({}, "backbone", 1024, False),  # the splits take it, the rows block not
        ({}, "backbone", 512, False),
        ({}, "eagle_head", 8192, False),
        (dict(TPU_MLA_PREFILL_SEQUENCE_ROWS=False), "backbone", 8192, False),
        (dict(TPU_MLA_PREFILL_ROWS_MIN_TOKENS=8192), "backbone", 4096, False),
    ],
)
def test_applies_to_split_buckets_from_the_threshold(
    monkeypatch, flags, tag, tokens, want
):
    _set_envs(monkeypatch, **flags)
    monkeypatch.setattr(sp.vllm_compilation_backends, "model_tag", tag)
    monkeypatch.setattr(tp_op.vllm_compilation_backends, "model_tag", tag)
    assert rows.applies(tokens) is want


def _config(tp=16, dcp=16, max_num_seqs=64):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp, decode_context_parallel_size=dcp
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
    )


def test_the_served_config_is_admitted(monkeypatch):
    _set_envs(monkeypatch)
    rows.check_config(_config())


@pytest.mark.parametrize(
    "config,flags,match",
    [
        ({}, dict(TPU_MOE_PREFILL_SEQUENCE_PARALLEL=False), "SEQUENCE_PARALLEL"),
        ({}, dict(TPU_DCP_PREFILL_TOKEN_PARALLEL=False), "TOKEN_PARALLEL"),
        (dict(dcp=8), {}, "decode_context_parallel_size"),
        # 128 DCP rows need rank 0 to hold 128 rows from the smallest bucket.
        (dict(max_num_seqs=128), {}, "MIN_TOKENS"),
        (
            {},
            dict(TPU_MLA_PREFILL_SEQUENCE_ROWS=False, TPU_MLA_PREFILL_ROWS_DENSE=True),
            "needed by TPU_MLA_PREFILL_ROWS_DENSE",
        ),
    ],
)
def test_unserved_configs_are_refused(monkeypatch, config, flags, match):
    _set_envs(monkeypatch, **flags)
    with pytest.raises(NotImplementedError, match=match):
        rows.check_config(_config(**config))


def test_raising_the_threshold_admits_more_dcp_rows(monkeypatch):
    _set_envs(monkeypatch, TPU_MLA_PREFILL_ROWS_MIN_TOKENS=4096)
    rows.check_config(_config(max_num_seqs=128))


@pytest.mark.parametrize(
    "skip_topk,want",
    [
        ([False, True, True, False, True], [True] * 5),
        # Layers reusing the top-k rows of a layer before the first MoE layer
        # keep today's block until the next indexer layer.
        ([True, True, False, True], [False, False, True, True]),
        ([], []),
    ],
)
def test_rows_layers_start_at_an_indexer_layer(skip_topk, want):
    assert rows.rows_layers(skip_topk) == want


def _armable_wrapper(**overrides):
    """An MLA wrapper as `_check_layer` reads it, every requirement met."""
    attn = SimpleNamespace(
        sparse_mla_tp_rows_op=object(),
        W_UK_T_dcp_qrep_scale=None,
        W_UV=torch.zeros(4, 8, 16, dtype=torch.bfloat16),
        kv_cache_quantized_dtype=torch.float8_e4m3fn,
    )
    o_proj = SimpleNamespace(
        weight=torch.zeros(64, 32, dtype=torch.float8_e4m3fn),
        weight_scale=torch.ones(32),
        bias=None,
    )
    setattr(o_proj, rows.WEIGHT_FLIPPED_ATTR, True)
    mla = SimpleNamespace(
        mla_attn=attn,
        q_lora_rank=16,
        kimi_preprocess=None,
        gate_is_fused=False,
        g_proj=None,
        is_sparse=True,
        dcp_q_replicate=True,
        o_proj=o_proj,
        skip_topk=True,
        indexer=None,
        prefix="model.layers.3.self_attn",
    )
    for k, v in overrides.items():
        setattr(mla, k, v)
    return mla


def test_the_decode_only_mla_options_are_accepted():
    """TPU_INDEXER_QB_OVERLAP / TPU_MLA_DECODE_PROLOGUE act only on the
    buckets the splits leave, so a layer that has them still takes the rows
    block (and the rest of the check still bites)."""
    rows._check_layer(_armable_wrapper(qb_overlap=True, decode_prologue=True))
    with pytest.raises(NotImplementedError, match="q_lora"):
        rows._check_layer(_armable_wrapper(qb_overlap=True, g_proj=object()))


def test_rows_topk_calls_the_dcp_indexer_op_without_the_q_b_operands():
    """The DCP indexer op serves the q_b overlap and the plain top-k
    (`op(*args, qb=...)` vs `op(*args)`); the rows block must take the
    plain one, a bare top-k tensor."""
    calls = []
    s, rows_per_rank, topk = 4, 8, 2

    def dcp_op(*args, **kwargs):
        calls.append(kwargs)
        return torch.zeros(s, topk, dtype=torch.int32)

    def tp_rows_op(*args, **kwargs):
        return torch.ones(rows_per_rank, topk, dtype=torch.int32)

    tp_rows_op.layout = SimpleNamespace(split_rows=s)
    indexer = SimpleNamespace(topk_op_dcp=dcp_op, topk_op_tp_rows=tp_rows_op)
    seq_lens = torch.tensor([20], dtype=torch.int32)
    qsl = torch.tensor([0, 12], dtype=torch.int32)
    dist = torch.tensor([0, 0, 1], dtype=torch.int32)
    out = tp_op.indexer_rows_aligned(
        indexer,
        torch.zeros(rows_per_rank, 8, dtype=torch.uint8),
        torch.zeros(rows_per_rank, 2),
        torch.zeros(s, 8, dtype=torch.uint8),
        torch.zeros(s, 2),
        torch.zeros(1),
        torch.zeros(12, 9, dtype=torch.uint8),
        seq_lens,
        torch.zeros(1, 4, dtype=torch.int32),
        qsl,
        dist,
    )
    assert calls == [{}]
    assert out.shape == (s + rows_per_rank, topk)
