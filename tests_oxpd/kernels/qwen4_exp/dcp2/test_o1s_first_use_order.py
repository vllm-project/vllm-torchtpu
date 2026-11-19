# SPDX-License-Identifier: Apache-2.0
"""DS-O1S shape_variants safety check (DECODE-LEAD 12:0xZ 11-11): FIRST_USE order of the QSA decode module's vLLM
subgraph inputs, per Dynamo trace, over the served decode buckets either side of 128 tokens (where the selection
changes from one window to a two-window scan inside the JAX op) and a prefill-shaped bucket.

shape_variants hands a compiled bucket between traces on a (dtype, shape) signature and binds by position, so every
trace of the module must lift its tensors in the same order. This prints each trace's placeholder (name, dtype,
shape) list and asserts that the name / dtype / static-shape order is identical across all traces, with O1s on
(request-sharded selection: a token-count branch) and off (O1). Reuses test_o1s_request_shard's real-module helpers
(_bound_qsa: the real bind_pool and jax_ops through their fakes; _traced_subgraphs: Dynamo with the token dim dynamic,
vLLM's split_graph)."""
import pytest

import test_o1s_request_shard as T

BUCKETS = (128, 192, 256, 512)


def _sig(graph):
    import torch

    out = []
    for node in graph.graph.nodes:
        if node.op != "placeholder":
            continue
        v = node.meta.get("example_value", node.meta.get("val"))
        if isinstance(v, torch.Tensor):
            shape = tuple(int(d) if isinstance(d, int) else "s" for d in v.shape)
            out.append((node.name, str(v.dtype), shape))
    return out


@pytest.mark.parametrize("flag", [True, False], ids=["o1s", "o1"])
def test_first_use_order_same_across_decode_traces(monkeypatch, flag):
    pytest.importorskip("torch_tpu")
    from vllm_torchtpu.compilation import shape_variants

    graphs = T._traced_subgraphs(T._bound_qsa(monkeypatch, flag), BUCKETS)
    sigs = [_sig(g) for g in graphs]
    ops = [T._qsa_ops(g) for g in graphs]
    for i, (s, o) in enumerate(zip(sigs, ops)):
        print(f"FIRST_USE flag={'o1s' if flag else 'o1'} trace {i} of {len(sigs)} ops={o}")
        for k, x in enumerate(s):
            print(f"  {k:2d} {x}")
    assert sigs, "no trace captured"
    # Name, dtype and static shape in the same order in every trace (the hand-over binds by position).
    assert all(s == sigs[0] for s in sigs), sigs
    # The production hand-over adopts the first trace's bucket in every other trace.
    tracked = shape_variants.BucketExecutables()
    key = "artifact_compile_range_128_128_subgraph_0"
    runnable = object()
    tracked.put("cache_dir", key, graphs[0], runnable)
    for g in graphs[1:]:
        assert tracked.get("cache_dir", key, g) is runnable
    if not flag:
        assert len(graphs) == 1, f"O1 traced {len(graphs)} times: its graph depends on the token count"
    else:
        assert any("request_shard_index" in n for n, _, _ in sigs[0]) and any("dcp_rank" in n for n, _, _ in sigs[0])
        assert len(graphs) >= 2, "O1s: expected a request-shard trace and a plain trace"
