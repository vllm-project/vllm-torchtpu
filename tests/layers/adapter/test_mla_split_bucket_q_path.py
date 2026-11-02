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
"""TPU_INDEXER_QB_OVERLAP and TPU_MLA_DECODE_PROLOGUE next to the prefill
splits (torch, any device): a bucket either split takes projects and ropes q
exactly as with both flags off, and the other buckets keep both."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

# pytest prepends this directory to sys.path (no __init__.py here).
from test_indexer_qb_overlap import _FakeIndexer  # noqa: E402
from test_mla_rope_neox_permute import QK, V_DIM, _inputs, _wrapper  # noqa: E402

from vllm_torchtpu.layers.adapter import dcp_token_parallel_op
from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op as mla_op
from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
    VllmMultiHeadLatentAttentionWrapper,
)


@pytest.fixture(autouse=True)
def _one_thread():
    """Two forwards are compared bit for bit; a multi-threaded CPU GEMM may
    reduce in a different order call to call."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def _flagged(flags_on: bool):
    w = _wrapper(heads=4, group_size=4, rank_in_group=1, layout="fp8")
    w.permute_rope_to_half_split()
    w.indexer = _FakeIndexer()
    w.indexer_rope_emb = None
    w.skip_topk = False
    w.qb_overlap = flags_on
    w.decode_prologue = flags_on
    return w


def _forward(w, hidden, positions, split: str | None):
    """Forward on a bucket `split` takes (None: neither); returns what the
    attention layer was handed."""
    w.mla_attn = MagicMock(
        return_value=torch.zeros((hidden.shape[0], w.num_heads * V_DIM))
    )
    # A token-parallel op that does not replicate q, so no q-rows path.
    w.mla_attn.sparse_mla_tp_op = SimpleNamespace(q_replicated=False)
    prefill_group = None
    if split == "sequence_parallel":
        prefill_group = SimpleNamespace(project_reduce_scatter=lambda x, o: x)
    with patch.object(
        dcp_token_parallel_op, "applies", lambda n: split == "token_parallel"
    ):
        VllmMultiHeadLatentAttentionWrapper.forward(
            w, positions=positions, hidden_states=hidden, prefill_group=prefill_group
        )
    (q_nope, q_pe), kv_c, k_pe = w.mla_attn.call_args.args[:3]
    kwargs = w.mla_attn.call_args.kwargs
    return dict(
        q_nope=q_nope,
        q_pe=q_pe,
        kv_c=kv_c,
        k_pe=k_pe,
        rep=kwargs["q_dcp_replicated"],
        topk=kwargs["topk_indices"],
    )


@pytest.mark.parametrize("split", ["token_parallel", "sequence_parallel"])
def test_split_buckets_keep_the_plain_q_path(split):
    hidden, positions = _inputs(16, seed=4)
    on, off = _flagged(True), _flagged(False)
    r_on = _forward(on, hidden, positions, split)
    r_off = _forward(off, hidden, positions, split)
    # No q_b layer offered to the indexer, no rope table handed to attention.
    assert on.indexer.calls == [None]
    assert len(r_on["rep"]) == 2
    for name in ("q_nope", "q_pe", "kv_c", "k_pe", "topk"):
        assert torch.equal(r_on[name], r_off[name]), name
    for a, b in zip(r_on["rep"], r_off["rep"]):
        assert torch.equal(a, b)


def test_other_buckets_take_both_flags():
    hidden, positions = _inputs(16, seed=4)
    w = _flagged(True)
    out = _forward(w, hidden, positions, None)
    assert w.indexer.calls == [w.q_b_proj]
    assert len(out["rep"]) == 3  # q_nope, the raw q head, the rope table


def test_overlap_reads_the_rope_cache_before_the_indexer():
    """The split path's q rope reads the rope cache the indexer shares before
    the indexer runs; the overlap path must lift it first too, or the warmup's
    traces name it after different modules and the bucket hand-over fails."""
    hidden, positions = _inputs(16, seed=4)
    w = _flagged(True)
    events = []
    real = w.indexer.forward

    def indexer(*args, **kwargs):
        events.append(("indexer", None))
        return real(*args, **kwargs)

    def retain(out, *buffers):
        events.extend(("retain", b) for b in buffers)
        return out

    w.indexer.forward = indexer
    with patch.object(mla_op, "_retain_buffers", retain):
        _forward(w, hidden, positions, None)
    cache = w.rotary_emb.cos_sin_cache
    first = next(i for i, e in enumerate(events) if e[1] is cache)
    assert first < events.index(("indexer", None))


def test_decode_buckets_gather_the_local_q_window():
    """TPU_MLA_LOCAL_Q_GATHER: a decode bucket projects this rank's column
    window of q_b and all-gathers the heads; attention gets the q the whole
    replicated projection gives. Here the window op slices the replicated
    output and the gather rebuilds it from every rank's window, so only the
    wrapper's plumbing (operands, window, head order) is under test."""
    hidden, positions = _inputs(16, seed=4)
    ref = _forward(_flagged(False), hidden, positions, None)
    w = _flagged(False)
    width = w.num_heads * QK
    seen = {}

    def window_op(q_c, w_q, w_s, window):
        seen.update(q_c=q_c, operands=(w_q, w_s), window=window)
        r = int(window)
        return w.q_b_proj(q_c)[0][:, r * width : (r + 1) * width]

    def gather(heads):
        full = w.q_b_proj(seen["q_c"])[0]
        windows = full.split(width, dim=-1)
        r = int(seen["window"])
        assert torch.equal(heads.reshape(heads.shape[0], -1), windows[r])
        return full.view(heads.shape[0], -1, QK)

    w.local_q_op = window_op
    w.local_q_window = torch.tensor([1], dtype=torch.int32)
    lifted = []

    def retain(out, *buffers):
        lifted.extend(buffers)
        return out

    wrapper_cls = mla_op.VllmMultiHeadLatentAttentionWrapper
    with (
        patch.object(mla_op, "_qb_overlap_operands", lambda layer: ("w", "s")),
        patch.object(wrapper_cls, "_local_q_serves", lambda self: True),
        patch.object(mla_op, "all_gather_heads", gather),
        patch.object(mla_op.vllm_compilation_backends, "model_tag", "backbone"),
        patch.object(mla_op, "_retain_buffers", retain),
    ):
        out = _forward(w, hidden, positions, None)
        assert seen["operands"] == ("w", "s") and seen["window"] is w.local_q_window
        seen.clear()
        _forward(w, hidden, positions, "token_parallel")
        assert not seen  # a split bucket keeps the replicated q_b
    # Both buckets lift the window operand (the warmup's hand-over rule).
    assert sum(t is w.local_q_window for t in lifted) == 2
    for name in ("q_nope", "q_pe", "kv_c", "k_pe", "topk"):
        assert torch.equal(out[name], ref[name]), name
    for a, b in zip(out["rep"], ref["rep"]):
        assert torch.equal(a, b)


def test_decode_buckets_run_the_fused_window_all_gather():
    """TPU_MLA_LOCAL_Q_FUSED_AG: the window and the gather are one op, handed
    q_c, the q_b operands, the window (rank) and the lane; its output is q. Both
    buckets lift the window and the lane operands."""
    hidden, positions = _inputs(16, seed=4)
    ref = _forward(_flagged(False), hidden, positions, None)
    w = _flagged(False)
    seen = {}

    def fused_op(q_c, w_q, w_s, window, lane):
        seen.update(operands=(w_q, w_s), window=window, lane=lane)
        return w.q_b_proj(q_c)[0]

    def unfused(*args):
        raise AssertionError("the fused op serves the bucket")

    w.local_q_op = unfused
    w.local_q_fused_op = fused_op
    w.local_q_window = torch.tensor([1], dtype=torch.int32)
    w.local_q_lane = torch.tensor([0], dtype=torch.int32)
    lifted = []

    def retain(out, *buffers):
        lifted.extend(buffers)
        return out

    wrapper_cls = mla_op.VllmMultiHeadLatentAttentionWrapper
    with (
        patch.object(mla_op, "_qb_overlap_operands", lambda layer: ("w", "s")),
        patch.object(wrapper_cls, "_local_q_serves", lambda self: True),
        patch.object(mla_op.vllm_compilation_backends, "model_tag", "backbone"),
        patch.object(mla_op, "_retain_buffers", retain),
    ):
        out = _forward(w, hidden, positions, None)
        assert seen["operands"] == ("w", "s")
        assert seen["window"] is w.local_q_window and seen["lane"] is w.local_q_lane
        seen.clear()
        _forward(w, hidden, positions, "token_parallel")
        assert not seen  # a split bucket keeps the replicated q_b
    for t in (w.local_q_window, w.local_q_lane):
        assert sum(x is t for x in lifted) == 2
    for name in ("q_nope", "q_pe", "kv_c", "k_pe", "topk"):
        assert torch.equal(out[name], ref[name]), name
