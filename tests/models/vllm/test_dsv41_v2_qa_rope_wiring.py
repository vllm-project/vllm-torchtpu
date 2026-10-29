# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the fused Q-A/KV, Q-B and RoPE-row ops in V4.1 v2."""

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import jax
import pytest
import torch

from vllm_torchtpu.kernels.attention_preprocessing import rope_gather
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention as v2_attn
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import dspark as v2_dspark
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import preprocess_ops

RANK, HEAD_DIM = 8, 4


def _attn(monkeypatch):
    calls = SimpleNamespace(qa=[], rows=0)

    def qa_rope(emit_qr, eps):
        def op(x, *args):
            calls.qa.append((emit_qr, eps, len(args)))
            kv = x[:, :HEAD_DIM] + 1
            return (x[:, :RANK], kv) if emit_qr else kv

        return op

    def rope_rows():
        def op(cache, positions):
            calls.rows += 1
            return cache[positions]

        return op

    monkeypatch.setattr(preprocess_ops, "qa_rope", qa_rope)
    monkeypatch.setattr(preprocess_ops, "rope_rows", rope_rows)
    attn = object.__new__(v2_attn.DeepseekV41Attention)
    torch.nn.Module.__init__(attn)
    attn._fused_preprocess = True
    attn.q_lora_rank, attn.head_dim = RANK, HEAD_DIM
    attn.padded_heads = attn.n_local_heads = 2
    attn.compressor = attn.indexer = None
    attn.fused_wqa_wkv = SimpleNamespace(weight=None, weight_scale=None)
    attn.q_norm = SimpleNamespace(weight=None, variance_epsilon=1e-5)
    attn.kv_norm = SimpleNamespace(weight=None, variance_epsilon=1e-5)
    attn._tpu_rope_cache = torch.arange(32.0).reshape(16, 2)
    attn._fused_wqa_wkv_gemm = MagicMock(side_effect=AssertionError("unfused"))
    attn._rotate = MagicMock(side_effect=AssertionError("rotated twice"))
    return attn, calls


def test_forward_runs_one_fused_qa_kv_op(monkeypatch):
    attn, calls = _attn(monkeypatch)
    attn._prepare_and_attn_fn = MagicMock()
    attn._o_proj = lambda o, pos: o
    x = torch.randn(3, 16, dtype=torch.bfloat16)
    positions = torch.tensor([0, 1, 2], dtype=torch.int32)
    v2_attn.DeepseekV41Attention.forward(attn, positions, x)
    assert calls.qa == [(True, 1e-5, 5)]  # rope_rows appended
    args, kwargs = attn._prepare_and_attn_fn.call_args
    assert not kwargs
    assert torch.equal(args[1], x[:, :RANK])
    assert torch.equal(args[2], x[:, :HEAD_DIM] + 1)
    assert args[3] is None and args[4] is None  # no qr scale, no kv_score


def test_rope_rows_gathered_once_per_stage(monkeypatch):
    other, _ = _attn(monkeypatch)
    attn, calls = _attn(monkeypatch)  # the last patch counts into ``calls``
    other._tpu_rope_cache = attn._tpu_rope_cache  # layers share one table
    positions = torch.tensor([3, 5], dtype=torch.int32)
    with v2_attn.stage_attention_cache_scope():
        first = attn._rope_rows(positions)
        assert other._rope_rows(positions) is first
        assert calls.rows == 1
        attn._rope_rows(positions.clone())  # a new step's positions
    assert calls.rows == 2
    assert torch.equal(first, attn._tpu_rope_cache[positions])
    attn._rope_rows(positions)  # no stage scope: no caching
    assert calls.rows == 3
    with pytest.raises(AssertionError):
        attn._rope_rows(positions.long())


def test_dspark_context_kv_uses_kv_only_mode(monkeypatch):
    attn, calls = _attn(monkeypatch)
    inserted = []
    monkeypatch.setattr(
        v2_dspark,
        "_insert_context_kv",
        lambda a, kv, pos, md: inserted.append(kv),
    )
    model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=16),
        layers=[SimpleNamespace(attn=attn)],
    )
    x = torch.randn(2, 16, dtype=torch.bfloat16)
    v2_dspark.DSparkDeepseekV41Model.precompute_and_store_context_kv(
        model, x, torch.tensor([4, 5], dtype=torch.int32), [None]
    )
    assert calls.qa == [(False, 1e-5, 5)]
    assert torch.equal(inserted[0], x[:, :HEAD_DIM] + 1)


def test_unfused_context_kv_rotates_once():
    attn = SimpleNamespace(
        _fused_preprocess=False,
        _fused_wqa_wkv_gemm=lambda x: x * 2,
        _split_qkv_and_norm=lambda qr_kv: (None, None, qr_kv + 1),
        _rotate=lambda kv, positions: kv + positions[:, None],
    )
    x = torch.ones(2, 4)
    positions = torch.tensor([10, 20])
    kv = v2_dspark._context_kv(attn, x, positions)
    assert torch.equal(kv, x * 2 + 1 + positions[:, None])


def test_insert_context_kv_writes_rotated_rows(monkeypatch):
    written = []
    monkeypatch.setattr(
        v2_dspark, "_insert", lambda cache, slots, kv, valid: written.append(kv)
    )
    attn = SimpleNamespace(
        swa_cache_layer=SimpleNamespace(kv_cache=torch.zeros(2, 4, 1, 8)),
        _rotate=MagicMock(side_effect=AssertionError("rotated twice")),
    )
    kv = torch.ones(2, 8)
    v2_dspark._insert_context_kv(attn, kv, torch.tensor([0, 1]), torch.tensor([0, 1]))
    assert written[0] is kv


def _fp8_linear(k, n):
    layer = SimpleNamespace(
        weight=torch.zeros(k, n, dtype=torch.float8_e4m3fn),
        weight_scale=torch.ones(n),
    )
    setattr(layer, WEIGHT_FLIPPED_ATTR, True)
    return layer


def _loaded_attn():
    attn = object.__new__(v2_attn.DeepseekV41Attention)
    torch.nn.Module.__init__(attn)
    attn.prefix = "model.layers.3.attn"
    attn.hidden_size, attn.q_lora_rank, attn.head_dim = 16, 128, 512
    attn.n_local_heads = 1
    attn.compress_ratio = 2
    attn._config = {"qk_rope_head_dim": 64}
    attn._tpu_rope_cache = torch.zeros(8, 64)
    attn.fused_wqa_wkv = _fp8_linear(16, 640)
    attn.wq_b = _fp8_linear(128, 512)
    norm = SimpleNamespace(weight=torch.ones(128), variance_epsilon=1e-6)
    attn.q_norm = norm
    attn.kv_norm = SimpleNamespace(weight=torch.ones(512), variance_epsilon=1e-6)
    projection = SimpleNamespace(weight=torch.zeros(16, 1024, dtype=torch.bfloat16))
    setattr(projection, WEIGHT_FLIPPED_ATTR, True)
    attn.compressor = SimpleNamespace(
        fused_wkv_wgate=projection, norm=SimpleNamespace(weight=torch.ones(512))
    )
    return attn


def test_fused_weight_layout_is_checked_at_load():
    attn = _loaded_attn()
    attn._check_fused_weights()
    attn.fused_wqa_wkv.weight = torch.zeros(16, 640, dtype=torch.bfloat16)
    attn.compressor.fused_wkv_wgate.weight = torch.zeros(16, 512)
    with pytest.raises(NotImplementedError, match="fused_wqa_wkv.*compressor"):
        attn._check_fused_weights()


def test_bridges_have_fully_annotated_signatures():
    """torch_tpu's ``pallas.jax_op`` rejects ``*args``, keyword parameters and
    unannotated arguments; the mocked ops above cannot see that."""
    functions = [
        rope_gather.gather_rows,
        preprocess_ops._qa_bridge(True, 1e-6),
        preprocess_ops._qa_bridge(False, 1e-6),
        preprocess_ops._qb,
        *(
            preprocess_ops._compress_bridge(ratio, banks, 1e-6)
            for ratio, banks in ((1, 0), (2, 1), (2, 2))
        ),
    ]
    for fn in functions:
        params = inspect.signature(fn).parameters.values()
        assert all(p.kind is p.POSITIONAL_OR_KEYWORD for p in params), fn
        assert all(p.default is p.empty for p in params), fn
        assert all(p.annotation is jax.Array for p in params), fn
    counts = [len(inspect.signature(fn).parameters) for fn in functions]
    assert counts == [2, 6, 6, 4, 6, 9, 10]
