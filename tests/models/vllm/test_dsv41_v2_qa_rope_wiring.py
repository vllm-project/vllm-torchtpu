# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for wiring the fused Q-A/KV norm + RoPE op into V4.1 v2."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention as v2_attn
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import dspark as v2_dspark

RANK, HEAD_DIM = 8, 4


def _attn(monkeypatch, *, sp=False):
    calls = SimpleNamespace(qa=[], rows=0)

    def get_qa_rope_op(eps, *, emit_qr, rotate_kv):
        def op(x, *args):
            calls.qa.append((emit_qr, rotate_kv, len(args)))
            kv = x[:, :HEAD_DIM] + 1
            return (x[:, :RANK], kv) if emit_qr else kv

        return op

    def get_rope_rows_op():
        def op(cache, positions):
            calls.rows += 1
            return cache[positions]

        return op

    monkeypatch.setattr(v2_attn, "get_qa_rope_op", get_qa_rope_op)
    monkeypatch.setattr(v2_attn, "get_rope_rows_op", get_rope_rows_op)
    attn = object.__new__(v2_attn.DeepseekV41Attention)
    torch.nn.Module.__init__(attn)
    attn._qa_rope_enabled = True
    attn._sp_sharded_input = sp
    attn.q_lora_rank, attn.head_dim = RANK, HEAD_DIM
    attn.padded_heads = attn.n_local_heads = 2
    attn.compressor = attn.indexer = None
    attn._ced_global_from_encoder = False
    attn.fused_wqa_wkv = SimpleNamespace(weight=None, weight_scale=None)
    attn.q_norm = SimpleNamespace(weight=None, variance_epsilon=1e-6)
    attn.kv_norm = SimpleNamespace(weight=None, variance_epsilon=1e-6)
    attn._tpu_rope_cache = torch.arange(32.0).reshape(16, 2)
    attn._fused_wqa_wkv_gemm = MagicMock(side_effect=AssertionError("unfused"))
    return attn, calls


def test_forward_uses_fused_op_and_skips_kv_rotate(monkeypatch):
    attn, calls = _attn(monkeypatch)
    attn._prepare_and_attn_fn = MagicMock()
    attn._o_proj = lambda o, pos: o
    x = torch.randn(3, 16, dtype=torch.bfloat16)
    positions = torch.tensor([0, 1, 2], dtype=torch.int32)
    v2_attn.DeepseekV41Attention.forward(attn, positions, x)
    attn._fused_wqa_wkv_gemm.assert_not_called()
    assert calls.qa == [(True, True, 5)]  # rope_rows appended
    args, kwargs = attn._prepare_and_attn_fn.call_args
    assert kwargs == {"kv_rotated": True}
    assert torch.equal(args[1], x[:, :RANK]) and args[3] is None
    assert torch.equal(args[2], x[:, :HEAD_DIM] + 1)


def test_unqualified_input_keeps_unfused_path(monkeypatch):
    attn, calls = _attn(monkeypatch)
    x = torch.randn(3, 16, dtype=torch.float32)
    assert not v2_attn._uses_qa_rope(attn, x)
    attn._qa_rope_enabled = False
    assert not v2_attn._uses_qa_rope(attn, x.bfloat16())
    assert not v2_attn._uses_qa_rope(SimpleNamespace(), x.bfloat16())


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


def test_sp_shard_leaves_kv_unrotated(monkeypatch):
    attn, calls = _attn(monkeypatch, sp=True)
    x = torch.randn(2, 16, dtype=torch.bfloat16)
    qr, kv, rotated = attn._qa_kv_norm_rope(x, torch.tensor([0, 1], dtype=torch.int32))
    assert not rotated and calls.rows == 0
    assert calls.qa == [(True, False, 4)]


def test_dspark_context_kv_uses_kv_only_mode(monkeypatch):
    attn, calls = _attn(monkeypatch)
    inserted = []
    monkeypatch.setattr(
        v2_dspark,
        "_insert_context_kv",
        lambda a, kv, pos, md, *, rotated: inserted.append((kv, rotated)),
    )
    model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=16),
        layers=[SimpleNamespace(attn=attn)],
    )
    x = torch.randn(2, 16, dtype=torch.bfloat16)
    v2_dspark.DSparkDeepseekV41Model.precompute_and_store_context_kv(
        model, x, torch.tensor([4, 5], dtype=torch.int32), [None]
    )
    assert calls.qa == [(False, True, 5)]
    kv, rotated = inserted[0]
    assert rotated and torch.equal(kv, x[:, :HEAD_DIM] + 1)


def test_insert_context_kv_skips_rotate_when_rotated(monkeypatch):
    written = []
    monkeypatch.setattr(
        v2_dspark, "_insert", lambda cache, slots, kv, valid: written.append(kv)
    )
    attn = SimpleNamespace(
        swa_cache_layer=SimpleNamespace(kv_cache=torch.zeros(2, 4, 1, 8)),
        _rotate=MagicMock(side_effect=AssertionError("rotated twice")),
    )
    kv = torch.ones(2, 8)
    v2_dspark._insert_context_kv(
        attn, kv, torch.tensor([0, 1]), torch.tensor([0, 1]), rotated=True
    )
    assert written[0] is kv


def test_supported_requires_fused_weight():
    attn = object.__new__(v2_attn.DeepseekV41Attention)
    torch.nn.Module.__init__(attn)
    assert not attn._qa_rope_supported()


def test_bridges_have_fully_annotated_signatures():
    """torch_tpu's ``pallas.jax_op`` rejects ``*args``, keyword parameters and
    unannotated arguments; the mocked ops above cannot see that."""
    import inspect

    import jax

    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import qa_rope_op

    functions = [qa_rope_op._gather_rows] + [
        qa_rope_op._bridge(1e-6, emit_qr, rotate_kv)
        for emit_qr in (True, False)
        for rotate_kv in (True, False)
    ]
    for fn in functions:
        params = inspect.signature(fn).parameters.values()
        assert all(p.kind is p.POSITIONAL_OR_KEYWORD for p in params), fn
        assert all(p.default is p.empty for p in params), fn
        assert all(p.annotation is jax.Array for p in params), fn
    rotated = qa_rope_op._bridge(1e-6, True, True)
    assert len(inspect.signature(rotated).parameters) == 6
    assert len(inspect.signature(qa_rope_op._bridge(1e-6, True, False)).parameters) == 5
