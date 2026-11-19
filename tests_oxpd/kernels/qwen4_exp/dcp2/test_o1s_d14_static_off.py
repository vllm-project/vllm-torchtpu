# SPDX-License-Identifier: Apache-2.0
"""q38 L2 O1SD14: the Qwen3.8 whole-step graph (D14) never takes O1s, and decides it statically.

SAB-D14 (L5, decode pod msrlr): with VLLM_TPU_QSA_DECODE_REQUEST_SHARD=1 and _MIN_ROWS=48 the whole step
attached, then at its first call (16 rows) shape_variants refused bucket 48 ("Token bucket 48 cannot be served
by the FX graph vLLM traced for this model: it assumes s72 <= 47") and the driver ran stock from then on.
vLLM traces Qwen4ExpWholeStepModel once, at its first call, and compiles that one graph for every whole-step
bucket outside shape_variants' warmup; layer 47's attention (Qwen4ExpBackbone.forward_wholestep, the one QSA
layer outside the kernel) evaluated Qwen4ExpQSA._request_shard_applies on the symbolic token count inside it.
forward_wholestep now passes attend(..., request_shard=False), a python constant read before any shape. Every
other caller keeps the default, so the stock backbone's O1s routing is unchanged.

The D14 graph here is the real Qwen4ExpWholeStepModel.forward and Qwen4ExpBackbone.forward_wholestep around the
real bound QSA layer (test_o1s_request_shard._bound_qsa: the real bind_pool and jax_ops through their fakes),
traced by Dynamo with the token dimension dynamic (as support_torch_compile marks it) and split by vLLM's own
split_graph; the megakernel calls (whole step, segment kernel) are stubs, opaque to the trace as on TPU. The
refusal is checked with shape_variants.runnable_for, the call that raised on the pod, outside any warmup.
"""

import re

import pytest

import test_o1s_request_shard as T

WHOLE_BUCKETS = (8, 16, 32, 48, 64)  # the whole-step program's compile sizes (wholerows 64)
PLAIN = ["pooled_qsa_request_plain_dcp", "qsa_dcp_combine"]
SHARD = ["pooled_qsa_request_attend_dcp", "pooled_qsa_request_select_dcp", "qsa_dcp_combine"]
# The pod's text (M, msrlr) read "it assumes s72 <= 47"; the symbol is numbered per trace.
SAB_D14 = re.compile(
    r"Token bucket (\d+) cannot be served by the FX graph vLLM traced for this model: it assumes "
    r"(s\d+ <= 47|s\d+ < 48)\."
)


class _NoShape:
    """Stands in for ``hidden`` / ``metadata``: reading any attribute (``.shape``, ``.context_lens``) fails."""

    def __getattr__(self, name):
        raise AssertionError(f"read .{name}")


def _module(tp=8, mql=4, min_rows=48):
    import types

    return types.SimpleNamespace(
        _request_shard_dcp_op=object(), tp_size=tp, max_query_len=mql, request_shard_min_rows=min_rows
    )


def test_static_off_decides_before_any_shape(monkeypatch):
    """request_shard=False returns False without reading hidden or metadata and without asking the predicate;
    the default reads both shapes and asks it (control)."""
    pytest.importorskip("torch_tpu")
    import torch

    from vllm_torchtpu.models.vllm.qwen4_exp import qsa as qsa_mod

    asked = []

    def spy(*args):
        asked.append(args)
        return True

    monkeypatch.setattr(qsa_mod, "request_shard_applies", spy)
    applies = qsa_mod.Qwen4ExpQSA._request_shard_applies
    assert applies(_module(), _NoShape(), _NoShape(), False) is False
    assert applies(_module(), _NoShape(), _NoShape(), request_shard=False) is False
    assert asked == []
    with pytest.raises(AssertionError, match=r"read \.shape"):
        applies(_module(), _NoShape(), _NoShape())  # control: the default reads the token count
    meta = type("M", (), {"context_lens": torch.zeros(64, dtype=torch.int32)})()
    assert applies(_module(), torch.zeros(16, 8), meta) is True
    assert applies(_module(), torch.zeros(16, 8), meta, True) is True
    assert asked == [(16, 64, 8, 4, 48), (16, 64, 8, 4, 48)]


@pytest.mark.parametrize("min_rows", [0, 48])
def test_default_routing_unchanged_8_to_256(min_rows):
    """With the flag unset (the default and request_shard=True) the routing is the O1s predicate, unchanged, at
    every row count from 8 to 256 (the stock backbone keeps O1s above 64 rows); request_shard=False is never."""
    pytest.importorskip("torch_tpu")
    import types

    import torch

    from vllm_torchtpu.layers.adapter.custom_ops.qsa_op import request_shard_applies
    from vllm_torchtpu.models.vllm.qwen4_exp.qsa import Qwen4ExpQSA

    seen = set()
    for t in (8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 96, 128, 160, 192, 224, 256):
        for slots in (16, 64):
            for mql in (1, 4):
                for tp in (1, 2, 8):
                    m = _module(tp, mql, min_rows)
                    meta = types.SimpleNamespace(context_lens=torch.zeros(slots, dtype=torch.int32))
                    hidden = torch.zeros(t, 8)
                    want = tp > 1 and t % tp == 0 and t <= slots * mql and t >= min_rows
                    assert request_shard_applies(t, slots, tp, mql, min_rows) == want
                    assert Qwen4ExpQSA._request_shard_applies(m, hidden, meta) == want, (t, slots, mql, tp)
                    assert Qwen4ExpQSA._request_shard_applies(m, hidden, meta, True) == want
                    assert Qwen4ExpQSA._request_shard_applies(m, hidden, meta, request_shard=False) is False
                    seen.add(want)
    assert seen == {True, False}
    # The served stock shapes above the whole-step rows: B24 / B64 verify steps (MTP k = 3) at TP 8, 64 slots.
    for t in (96, 256):
        meta = types.SimpleNamespace(context_lens=torch.zeros(64, dtype=torch.int32))
        assert Qwen4ExpQSA._request_shard_applies(_module(min_rows=min_rows), torch.zeros(t, 8), meta)


@pytest.mark.parametrize("min_rows", [0, 48])
def test_stock_attend_keeps_o1s_above_64_rows(monkeypatch, min_rows):
    """The stock graph (attend with its default) still traces the request-sharded selection at 96 and 256 rows."""
    pytest.importorskip("torch_tpu")
    m = T._bound_qsa(monkeypatch, True)
    m.request_shard_min_rows = min_rows
    graphs = T._traced_subgraphs(m, (96, 256))
    assert graphs, "no trace captured"
    assert [T._qsa_ops(g) for g in graphs] == [SHARD] * len(graphs), [T._qsa_ops(g) for g in graphs]


# ---------------------------------------------------------------------------------------------------------------
# The whole-step graph: real Qwen4ExpWholeStepModel.forward / forward_wholestep, real bound QSA as layer 47.
# ---------------------------------------------------------------------------------------------------------------


def _d14(qsa):
    """A module whose forward IS Qwen4ExpWholeStepModel.forward over a backbone whose forward_wholestep IS
    Qwen4ExpBackbone.forward_wholestep: one GDN layer (run by the stub whole step) and the bound QSA layer."""
    import torch
    from torch import nn

    from vllm_torchtpu.models.vllm.qwen4_exp.model import Qwen4ExpBackbone, Qwen4ExpWholeStepModel

    class Mix(nn.Module):
        def mix(self, hidden):
            return hidden, hidden

    class Layer(nn.Module):
        def __init__(self, kind, attn=None):
            super().__init__()
            self.layer_type = kind
            self.attn_hyper_connection = Mix()
            if attn is not None:
                self.self_attn = attn

    class Mixer(nn.Module):
        def mix(self, hidden):
            return (hidden,)

    class Backbone(nn.Module):
        forward_wholestep = Qwen4ExpBackbone.forward_wholestep

        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(32, T.HID_, dtype=torch.bfloat16)
            self.layers = nn.ModuleList([Layer("linear_attention"), Layer("full_attention", qsa)])
            self.stage = [0, 1]
            self.hc_count, self.host_ple_index, self.emit_mtp_hidden = 1, None, False
            self.hyper_connection_mixer = Mixer()

    class Whole(nn.Module):  # the whole-step megakernel call (layers 3..46 on TPU)
        def forward(self, hidden, positions, query_start_loc, context_lens):
            return hidden * 2

    class SegmentCall(nn.Module):  # the segment kernel after layer 47's attention core
        head_groups = ()

        def forward(self, s, hidden, attention):
            return hidden + attention.flatten(1).sum(-1, keepdim=True).to(hidden.dtype)

    class D14(nn.Module):
        forward = Qwen4ExpWholeStepModel.forward

        def __init__(self):
            super().__init__()
            self.backbone, self.segments, self.whole = Backbone(), SegmentCall(), Whole()

    return D14()


def _d14_graphs(module, buckets=(16,)):
    """vLLM's piecewise subgraph of every Dynamo trace the D14 module needs over ``buckets`` (the first call is the
    16-row step, as on the pod), never run."""
    import torch
    from vllm.compilation.backends import split_graph

    graphs = []

    def backend(gm, example_inputs):
        _, items = split_graph(gm, [])
        assert len(items) == 1, [i.submod_name for i in items]
        graphs.append(items[0].graph)

        def run(*args):
            raise T._Captured()

        return run

    torch._dynamo.reset()
    compiled = torch.compile(module, backend=backend, fullgraph=True, dynamic=False)
    starts = torch.zeros(65, dtype=torch.int32)
    context = torch.zeros(64, dtype=torch.int32)  # 64 request slots
    for tokens in buckets:
        ids = torch.zeros(tokens, dtype=torch.int32)
        positions = torch.zeros(tokens, dtype=torch.int64)
        valid = torch.ones(tokens, dtype=torch.int32)
        for x in (ids, positions, valid):  # support_torch_compile's dynamic_arg_dims
            torch._dynamo.maybe_mark_dynamic(x, 0)
        try:
            with torch.no_grad():
                compiled(ids, positions, None, query_start_loc=starts, ngram_context=None, state_slots=None,
                         context_lens=context, valid_tokens=valid, main_block_table=None,
                         compressed_block_table=None)
        except Exception as exc:  # the runnable's sentinel, possibly wrapped by Dynamo
            if not isinstance(exc, T._Captured) and not isinstance(exc.__cause__, T._Captured):
                raise
    torch._dynamo.reset()
    return graphs


def _refusals(graph, buckets=WHOLE_BUCKETS):
    """bucket -> ShapeSpecializationError text from shape_variants.runnable_for outside any warmup (the
    whole-step program's compile path on the pod), for every bucket the graph cannot serve."""
    from vllm_torchtpu.compilation import shape_variants

    assert shape_variants._session is None, "must run outside shape_variants.warmup()"
    out = {}
    for b in buckets:
        try:
            got = shape_variants.runnable_for(None, f"artifact_compile_range_{b}_{b}_subgraph_0", graph, b)
        except shape_variants.ShapeSpecializationError as exc:
            out[b] = str(exc)
        else:
            assert got is None, (b, got)
    return out


def _guards(graph):
    from vllm_torchtpu.compilation import shape_variants

    env = shape_variants.trace_shape_env(graph)
    assert env is not None, "the subgraph carries no ShapeEnv"
    return sorted(str(g.expr) for g in env.guards)


def test_control_token_count_branch_is_refused_outside_warmup(monkeypatch):
    """Positive control, the SAB-D14 failure: the QSA layer's default attend with MIN_ROWS 48, traced once at 16
    rows, refuses buckets 48 and 64 outside the warmup with the pod's message (only the symbol name differs)."""
    pytest.importorskip("torch_tpu")
    m = T._bound_qsa(monkeypatch, True)
    m.request_shard_min_rows = 48
    (graph,) = T._traced_subgraphs(m, (16,))
    assert T._qsa_ops(graph) == PLAIN  # gated off at 16 rows, but through a guard on the token count
    refused = _refusals(graph)
    print("CONTROL refusals:", refused, "guards:", _guards(graph))
    assert sorted(refused) == [48, 64], refused
    assert SAB_D14.search(refused[48]) and SAB_D14.search(refused[48]).group(1) == "48", refused[48]


@pytest.mark.parametrize("min_rows", [48, 0])
def test_whole_step_graph_takes_no_o1s_and_serves_every_bucket(monkeypatch, min_rows):
    """The D14 graph traced once at 16 rows: layer 47 takes the plain DCP step (no O1s), runnable_for refuses no
    whole-step bucket, and the trace's guards equal those of the same graph over a module without the request
    shard op (where the predicate returns before any shape): the O1s decision left no guard behind."""
    pytest.importorskip("torch_tpu")
    m = T._bound_qsa(monkeypatch, True)
    m.request_shard_min_rows = min_rows
    graphs = _d14_graphs(_d14(m), (16,))
    assert len(graphs) == 1, len(graphs)
    (graph,) = graphs
    assert T._qsa_ops(graph) == PLAIN, T._qsa_ops(graph)
    refused = _refusals(graph)
    guards = _guards(graph)
    print(f"D14 min_rows={min_rows} ops={T._qsa_ops(graph)} refusals={refused} guards={guards}")
    assert refused == {}, refused

    ref = T._bound_qsa(monkeypatch, True)
    ref.request_shard_min_rows = min_rows
    ref._request_shard_dcp_op = None  # _request_shard_applies returns at its first line
    (ref_graph,) = _d14_graphs(_d14(ref), (16,))
    assert T._qsa_ops(ref_graph) == PLAIN
    assert guards == _guards(ref_graph), (guards, _guards(ref_graph))


def test_whole_step_graph_is_one_trace_for_every_bucket(monkeypatch):
    """Called at every whole-step bucket in turn (MIN_ROWS 48), the D14 graph is traced once: nothing in it
    depends on the token count, so the one trace vLLM compiles serves 8..64."""
    pytest.importorskip("torch_tpu")
    m = T._bound_qsa(monkeypatch, True)
    m.request_shard_min_rows = 48
    graphs = _d14_graphs(_d14(m), (16, 8, 32, 48, 64))
    assert len(graphs) == 1, f"the D14 graph was traced {len(graphs)} times"
    assert T._qsa_ops(graphs[0]) == PLAIN
