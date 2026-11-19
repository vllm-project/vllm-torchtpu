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
"""W_UK absorbed in the decode q_b window gather (torch wiring, any device).

The kernel claim -- the raw q bitwise, the absorbed q within a bf16 ulp of the
replicated absorption -- is in tests/kernels/test_fp8_window_all_gather.py.
This file checks the plumbing: which buckets absorb, that the gather is handed
the very `W_UK_T_dcp_qrep` the attention backend would read (so a bucket's
trace lifts the same buffer either way), and that the backend then attends with
the absorbed q as handed over while every other input stays the same.
"""

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

# pytest prepends this directory to sys.path (no __init__.py here).
from test_mla_rope_neox_permute import KV_LORA, NOPE, QK, V_DIM, _inputs, _wrapper

from vllm_torchtpu.kernels.quantized_matmul import fp8_window_all_gather as wag
from vllm_torchtpu.layers.adapter.attention import PallasMLAttentionBackendImpl
from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op as mla_op

HEADS, GROUP = 4, 4  # 16 group heads


def _no_rope(positions, q, k):
    return q, k


def _absorb(q, w_uk):
    """Reference latent q `[T, G * KV_LORA]`: per head bf16(q_nope . W_UK), f32."""
    t = q.shape[0]
    q_nope = q.reshape(t, -1, QK)[..., :NOPE].float()
    return torch.einsum("thd,hdl->thl", q_nope, w_uk.float()).bfloat16().reshape(t, -1)


def _absorb_wrapper(monkeypatch, *, full_layer=False):
    w = _wrapper(
        heads=HEADS, group_size=GROUP, rank_in_group=1, layout="fp8", rope=_no_rope
    )
    gen = torch.Generator().manual_seed(5)
    w_uk = (torch.randn(HEADS * GROUP, NOPE, KV_LORA, generator=gen) / 14).bfloat16()
    w.mla_attn = MagicMock()
    w.mla_attn.W_UK_T_dcp_qrep = w_uk
    w.mla_attn.W_O_dcp_qrep = None
    w.mla_attn.sparse_mla_tp_op = None
    project = lambda q_c: w.q_b_proj(q_c)[0]  # noqa: E731
    w.local_q_fused_op = MagicMock(side_effect=lambda q_c, *_: project(q_c))
    w.local_q_absorb_op = MagicMock(
        side_effect=lambda q_c, *a: (project(q_c), _absorb(project(q_c), a[-1]))
    )
    w.local_q_window = w.local_q_lane = None
    w.local_q_absorb = True
    if full_layer:
        w.skip_topk = False
        w.qb_overlap = True
        w.indexer_rope_emb = None

        def indexer(hidden, qr, positions, rope, q_b_layer=None, w_uk=None):
            topk = torch.zeros(qr.shape[0], 8, dtype=torch.int32)
            if w_uk is None:
                return topk, q_b_layer(qr)[0]
            q = q_b_layer(qr)[0]
            return topk, q, _absorb(q, w_uk)

        w.indexer = MagicMock(side_effect=indexer)
    weights = (w.q_b_proj.weight, w.q_b_proj.weight_scale)
    monkeypatch.setattr(mla_op, "_qb_overlap_operands", lambda layer: weights)
    monkeypatch.setattr(wag, "supports", lambda m, *_: m <= 16)
    monkeypatch.setattr(
        mla_op, "vllm_compilation_backends", SimpleNamespace(model_tag="backbone")
    )
    return w


def _forward(w, num_tokens):
    hidden, positions = _inputs(num_tokens)
    w.mla_attn.return_value = torch.zeros(num_tokens, HEADS * V_DIM)
    mla_op.VllmMultiHeadLatentAttentionWrapper.forward(
        w, positions=positions, hidden_states=hidden
    )
    return w.mla_attn.call_args


@pytest.mark.parametrize("full_layer", [False, True], ids=["index_share", "full"])
@pytest.mark.parametrize("num_tokens", [4, 8, 16])
def test_small_buckets_attend_with_the_gathered_absorbed_q(
    monkeypatch, full_layer, num_tokens
):
    absorbs = num_tokens <= wag.ABSORB_MAX_M
    on = _absorb_wrapper(monkeypatch, full_layer=full_layer)
    off = _absorb_wrapper(monkeypatch, full_layer=full_layer)
    off.local_q_absorb = False
    got, ref = _forward(on, num_tokens), _forward(off, num_tokens)
    w_uk = on.mla_attn.W_UK_T_dcp_qrep
    if full_layer:
        assert (on.indexer.call_args.kwargs.get("w_uk") is w_uk) == absorbs
        assert off.indexer.call_args.kwargs.get("w_uk") is None
    else:
        # The gather is handed the backend's own W_UK, not a copy of it.
        assert on.local_q_absorb_op.called == absorbs
        assert on.local_q_fused_op.called != absorbs
        if absorbs:
            assert on.local_q_absorb_op.call_args.args[-1] is w_uk
    assert got.kwargs["q_dcp_absorbed"] == absorbs
    assert ref.kwargs["q_dcp_absorbed"] is False
    head, *rest = got.kwargs["q_dcp_replicated"]
    ref_head, *ref_rest = ref.kwargs["q_dcp_replicated"]
    # Everything but the q head is what the plain gather hands over, bitwise.
    for a, b in zip(rest, ref_rest, strict=True):
        assert torch.equal(a, b)
    for a, b in zip(got.args[:3], ref.args[:3], strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    if absorbs:
        assert head.shape == (num_tokens, HEADS * GROUP, KV_LORA)
        # Within a bf16 ulp of what the backend computes from the plain q.
        backend = torch.einsum("thd,hdl->thl", ref_head.float(), w_uk.float())
        torch.testing.assert_close(head.float(), backend, rtol=8e-3, atol=1e-2)
    else:
        assert torch.equal(head, ref_head)


def test_llama4_scaling_keeps_the_plain_gather(monkeypatch):
    w = _absorb_wrapper(monkeypatch)
    assert w._local_q_absorbs(4, None)
    assert not w._local_q_absorbs(4, torch.ones(4, 1, 1))
    assert not w._local_q_absorbs(wag.ABSORB_MAX_M + 1, None)


def test_the_mtp_drafter_never_absorbs(monkeypatch):
    """The drafter is traced once for every bucket: no token-count rule."""
    w = _absorb_wrapper(monkeypatch)
    monkeypatch.setattr(
        mla_op, "vllm_compilation_backends", SimpleNamespace(model_tag="eagle_head")
    )
    assert not w._local_q_absorbs(4, None)


def _armed(monkeypatch, *, scale=None, dtype=torch.bfloat16, qb_overlap=False):
    w = _wrapper(heads=4, group_size=16, rank_in_group=0, rope=_no_rope)
    w.mla_attn = SimpleNamespace(
        W_UK_T_dcp_qrep=torch.empty((64, NOPE, KV_LORA), dtype=dtype, device="meta"),
        W_UK_T_dcp_qrep_scale=scale,
    )
    w.qb_overlap = qb_overlap
    w.indexer = SimpleNamespace(indexer_op=SimpleNamespace(topk_op_dcp_qb_uk=None))
    # GLM-5 q_b under query replication: [2048, 64 * (192 + 64)] fp8.
    qb = torch.empty((2048, 16384), dtype=torch.float8_e4m3fn, device="meta")
    monkeypatch.setattr(mla_op, "_qb_overlap_operands", lambda layer: (qb, qb))
    w.arm_local_q_absorb()
    return w.local_q_absorb


def test_arms_on_a_bf16_group_w_uk_with_whole_head_windows(monkeypatch):
    assert _armed(monkeypatch)


@pytest.mark.parametrize(
    "kw",
    [
        dict(scale=torch.ones(1)),
        dict(dtype=torch.float8_e4m3fn),
        dict(qb_overlap=True),  # a full layer whose indexer cannot absorb
    ],
    ids=["fp8_w_uk_scale", "fp8_w_uk", "no_indexer_op"],
)
def test_stays_unarmed_where_the_gather_cannot_absorb(monkeypatch, kw):
    assert not _armed(monkeypatch, **kw)


def test_indexer_hands_the_op_w_uk_with_the_q_b_operands(monkeypatch):
    n_head, head_dim, t = 2, 128, 4
    idx = mla_op.VllmTPUIndexer.__new__(mla_op.VllmTPUIndexer)
    torch.nn.Module.__init__(idx)
    idx.n_head, idx.head_dim, idx.rope_dim, idx.softmax_scale = (
        n_head,
        head_dim,
        64,
        1.0,
    )
    idx.wq_b = lambda x: (torch.ones(x.shape[0], n_head * head_dim), None)
    idx.wk_weights_proj = lambda x: (torch.ones(x.shape[0], head_dim + n_head), None)
    idx.k_norm = torch.nn.Identity()
    idx.indexer_op = MagicMock(return_value=("topk", "q", "ql"))
    layer = SimpleNamespace(qrep_active=True)
    weights = (torch.ones(1), torch.ones(1))
    monkeypatch.setattr(mla_op, "_qb_overlap_operands", lambda layer: weights)
    qr, w_uk = torch.randn(t, 128).bfloat16(), torch.ones(1)
    out = idx.forward(
        torch.randn(t, 64), qr, torch.arange(t), _no_rope, q_b_layer=layer, w_uk=w_uk
    )
    assert out == ("topk", "q", "ql")
    kwargs = idx.indexer_op.call_args.kwargs
    assert kwargs["w_uk"] is w_uk
    assert torch.equal(kwargs["qb_overlap"][0], qr)
    assert kwargs["qb_overlap"][1:] == weights


def test_indexer_op_without_metadata_returns_neither_q(monkeypatch):
    op = mla_op.VllmTPUSparseAttnIndexer.__new__(mla_op.VllmTPUSparseAttnIndexer)
    torch.nn.Module.__init__(op)
    op.k_cache = SimpleNamespace(prefix="layers.0.attn")
    op.topk_indices_buffer = torch.zeros(4, 8, dtype=torch.int32)
    monkeypatch.setattr(
        mla_op, "get_forward_context", lambda: SimpleNamespace(attn_metadata={})
    )
    k = torch.zeros(4, 128).bfloat16()
    q = torch.zeros(4, 2, 128).to(torch.float8_e4m3fn)
    qb = (torch.zeros(4, 8), torch.zeros(8, 8), torch.zeros(8))
    rows, q_out, ql = op.forward_oot(
        None, q, k, torch.zeros(4, 2), qb_overlap=qb, w_uk=torch.zeros(1)
    )
    assert rows is op.topk_indices_buffer and q_out is None and ql is None


def test_local_q_absorb_op_is_its_own_two_output_op(monkeypatch):
    import inspect

    from vllm_torchtpu.distributed import dcp
    from vllm_torchtpu.distributed import sharded_jax_op as sjo
    from vllm_torchtpu.layers.adapter import cp_mla_attention as cp_mla

    class _Mesh:
        shape = {dcp.DCP_AXIS_NAME: 16}
        axis_names = (dcp.DCP_AXIS_NAME,)

    built = {}

    def checked_op(name, fn, *, fake, output_partition_specs, **_):
        signature = inspect.signature(inspect.unwrap(fn), eval_str=True)
        sjo.pallas_impl._verify_signature(signature)
        built[name] = (fake, output_partition_specs)
        return object()

    monkeypatch.setattr(dcp, "get_or_create_dcp_mesh", lambda: _Mesh())
    monkeypatch.setattr(sjo, "sharded_jax_op", checked_op)
    monkeypatch.setattr(cp_mla, "_LOCAL_Q_ALL_GATHER_OPS", {})
    plain, absorb = cp_mla.local_q_all_gather_op(), cp_mla.local_q_all_gather_op(True)
    assert plain is not absorb
    fake, specs = built["pallas::fp8_window_all_gather_absorb_16"]
    assert len(specs) == 2
    x = torch.empty(4, 2048)
    q, ql = fake(x, torch.empty(2048, 16384), None, None, None, torch.empty(64, 1, 512))
    assert q.shape == (4, 16384) and ql.shape == (4, 64 * 512)


def test_backend_refuses_an_absorbed_q_off_the_query_replicated_path():
    impl = PallasMLAttentionBackendImpl.__new__(PallasMLAttentionBackendImpl)
    impl.dcp_size = 1
    q = (torch.zeros(4, 4, KV_LORA), torch.zeros(4, 4, 64))
    with pytest.raises(ValueError, match="q_dcp_absorbed"):
        impl.forward(
            SimpleNamespace(),
            q,
            torch.zeros(4, KV_LORA),
            torch.zeros(4, 1, 64),
            (torch.zeros(1), torch.zeros(1)),
            None,
            topk_indices=torch.zeros(4, 8, dtype=torch.int32),
            q_dcp_absorbed=True,
        )


def test_reference_absorption_is_per_head():
    """`_absorb` really reads each head's own nope columns (a positive control
    for the comparisons above)."""
    q = torch.zeros(1, 2 * QK).bfloat16()
    q[0, QK : QK + NOPE] = 1.0  # head 1's nope only
    w_uk = torch.ones(2, NOPE, KV_LORA).bfloat16()
    out = _absorb(q, w_uk).view(1, 2, KV_LORA)
    assert out[0, 0].abs().max() == 0 and out[0, 1, 0] == NOPE
    assert math.isclose(float(out.float().sum()), NOPE * KV_LORA)
