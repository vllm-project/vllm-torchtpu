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
"""The expert routing counter (tools/expert_placement/routing_counter) on CPU.

What runs compiled: moe_tp_decode's real ``moe_tp_decode_forward``, wrapped by
the counter, under ``torch.compile(fullgraph=True)`` with an AOTAutograd
backend set up like torch_tpu's (``keep_inference_input_mutations=False``),
so the counter's in-place update is a program output that the runtime
wrapper writes back, as on TPU. The op itself (a Pallas kernel on TPU) is a
torch stand-in that routes like moe_tp_decode's routing. Reference ids come
from the Pallas router kernel the op calls (interpret mode) and from the
decode kernel's in-kernel routing, never from the counter's own path.
"""

import inspect
import json
import os
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd
from vllm import forward_context as fc

from vllm_torchtpu.kernels import router_topk as k_router_topk
from vllm_torchtpu.kernels.experimental.moe_tp_decode import kernel as tpd_kernel
from vllm_torchtpu.layers.adapter import (
    moe_routing,
    moe_tp_decode,
    router_topk,
    token_padding,
)
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata

TOOLS = pathlib.Path(__file__).resolve().parents[2] / "tools" / "expert_placement"
COUNTER_DIR = TOOLS / "routing_counter"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(COUNTER_DIR))

import aggregate_counts  # noqa: E402
import ep_routing_counter as epc  # noqa: E402

E, TOPK, H = 256, 8, 32
MAX_TOKENS = 4096
MAX_SEQS = 32
TARGET_LAYERS = (3, 4, 5)
DRAFT_LAYER = 6
MAX_DECODE_Q = 4  # 1 + 3 speculative tokens

_ORIG_TPD = moe_tp_decode.moe_tp_decode_forward
_ORIG_ROUTE = moe_routing.route
GRAPHS: list[torch.fx.GraphModule] = []


def _fw_compiler(gm, example_inputs):
    del example_inputs
    GRAPHS.append(gm)
    return make_boxed_func(gm.forward)


# As torch_tpu's backend calls aot_autograd.
BACKEND = aot_autograd(fw_compiler=_fw_compiler, keep_inference_input_mutations=False)


def stub_op(
    x, scores, bias, pad_mask, w13, w2, s13, s2, sw13, ss13, sw2, ss2, rank, e0
):
    """Stand-in for the moe_tp_decode op: routes like its routing, then a
    data-dependent combine. Returns (out, weights, ids)."""
    del w13, w2, s13, s2, sw13, ss13, sw2, ss2, rank, e0
    w, ids = router_topk.rowmax_select_bias(scores, bias, TOPK)
    total = w.sum(dim=-1, keepdim=True)
    w = (w / total.clamp(min=1e-20)).to(x.dtype)
    w = torch.where(pad_mask[: x.shape[0], None], torch.zeros_like(w), w)
    out = x * w.float().sum(dim=-1, keepdim=True).to(x.dtype) + ids[:, :1].to(x.dtype)
    return out, w, ids


STATICS = moe_tp_decode._OpStatics(
    topk=TOPK,
    routed_scale=1.0,
    activation="silu",
    use_sparse_core=False,
    onehot_threshold=0,
    router_bias=True,
)


class FakeRoutedExperts(torch.nn.Module):
    """What apply_monolithic hands moe_tp_decode_forward once arming has
    registered its buffers (sigmoid scores, correction bias, top-8 of 256)."""

    def __init__(self, layer_name: str, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.layer_name = layer_name
        self.global_num_experts = E
        self.scoring_func = "sigmoid"
        self.moe_config = SimpleNamespace(experts_per_token=TOPK)
        self.quant_method = SimpleNamespace(_moe_tp_decode_op=stub_op)
        self.e_score_correction_bias = torch.nn.Parameter(
            torch.randn(E, generator=g) * 0.05, requires_grad=False
        )
        for name in ("w13_weight", "w2_weight"):
            setattr(
                self, name, torch.nn.Parameter(torch.zeros(1, 1), requires_grad=False)
            )
        for name in ("w13_weight_scale_inv", "w2_weight_scale_inv"):
            setattr(self, name, torch.nn.Parameter(torch.ones(1), requires_grad=False))
        bias = self.e_score_correction_bias.detach().to(torch.float32).contiguous()
        tensors = (
            torch.zeros(1, 1),
            torch.ones(1),
            torch.zeros(1, 1),
            torch.ones(1),
            bias,
            torch.zeros(1, 1, dtype=torch.int32),
            torch.zeros(1, 1, dtype=torch.int32),
        )
        for name, t in zip(moe_tp_decode._BUFFERS, tensors):
            self.register_buffer(name, t, persistent=False)


class FakeMoE(torch.nn.Module):
    """Holds the experts as ``routed_experts``, as vLLM's MoE runner does."""

    def __init__(self, layer: int, seed: int):
        super().__init__()
        self.routed_experts = FakeRoutedExperts(
            f"model.layers.{layer}.mlp.experts", seed
        )


class Model(torch.nn.Module):
    def __init__(self, layer_ids, seed0):
        super().__init__()
        self.layers = torch.nn.ModuleList(FakeMoE(i, seed0 + i) for i in layer_ids)

    def experts(self):
        return [m.routed_experts for m in self.layers]


@pytest.fixture
def env(monkeypatch):
    """Target (layers 3-5) and draft (layer 6) models instrumented, the
    padding mask, and a forward context whose metadata tensors are updated
    in place between steps (as the runner's persistent device tensors are)."""
    monkeypatch.setattr(epc, "_STATE", None)
    monkeypatch.setattr(moe_tp_decode, "_OPS", {(None, STATICS, False): stub_op})
    monkeypatch.setattr(
        moe_tp_decode,
        "moe_tp_decode_forward",
        epc.make_counted_tp_decode_forward(_ORIG_TPD),
    )
    state = token_padding.TokenPaddingState.create(MAX_TOKENS, torch.device("cpu"))
    monkeypatch.setattr(token_padding, "_padding_state", state)
    target, draft = Model(TARGET_LAYERS, 100), Model((DRAFT_LAYER,), 200)
    n = epc.instrument_models(
        [(target, "target"), (draft, "draft")], max_decode_query_len=MAX_DECODE_Q
    )
    assert n == len(TARGET_LAYERS) + 1
    rd = torch.zeros(3, dtype=torch.int32)
    qsl = torch.zeros(MAX_SEQS + 1, dtype=torch.int32)
    md = AttentionMetadata(
        input_positions=None, query_start_loc=qsl, request_distribution=rd
    )
    ctx = fc.ForwardContext(
        no_compile_layers={},
        attn_metadata={"model.layers.0.self_attn.attn": md},
        slot_mapping={},
    )
    monkeypatch.setattr(fc, "_forward_context", ctx)
    torch._dynamo.reset()
    GRAPHS.clear()
    yield SimpleNamespace(target=target, draft=draft, rd=rd, qsl=qsl)
    torch._dynamo.reset()


def _model_fn(layers):
    """Several armed MoE layers; moe_tp_decode_forward is looked up at call
    time, as apply_monolithic does."""

    def fn(x, logits):
        f = moe_tp_decode.moe_tp_decode_forward
        return [f(lay.quant_method, lay, x, logits[i]) for i, lay in enumerate(layers)]

    return fn


def _compiled(layers, **kw):
    return torch.compile(_model_fn(layers), backend=BACKEND, fullgraph=True, **kw)


def _set_step(env, query_lens, bucket):
    """Writes one forward's metadata in place and marks its padding. As the
    runner does with reorder_batch_threshold=1: request_distribution [0] and
    [1] count the leading 1-token decodes only, [2] all requests."""
    q = np.zeros(MAX_SEQS + 1, dtype=np.int64)
    q[1 : len(query_lens) + 1] = np.cumsum(query_lens)
    q[len(query_lens) + 1 :] = q[len(query_lens)]
    env.qsl.copy_(torch.from_numpy(q).to(torch.int32))
    n1 = next((i for i, v in enumerate(query_lens) if v != 1), len(query_lens))
    env.rd.copy_(torch.tensor([n1, n1, len(query_lens)], dtype=torch.int32))
    n_valid = int(q[len(query_lens)])
    token_padding.mark_padding(n_valid, bucket)
    return q, n_valid


def _inputs(bucket, n_layers, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(bucket, H, generator=g).to(torch.bfloat16)
    return x, [torch.randn(bucket, E, generator=g) * 2.0 for _ in range(n_layers)]


def _ref_ids(layer, logits):
    """Top-k ids from the Pallas router kernel the op calls (interpret)."""
    scores = jax.nn.sigmoid(jnp.asarray(logits.numpy(), jnp.float32))
    bias = jnp.asarray(layer.e_score_correction_bias.detach().numpy(), jnp.float32)
    _, ids = k_router_topk.select_bias(scores, bias[None, :], TOPK, interpret=True)
    return np.asarray(ids, dtype=np.int64)


def _expected_delta(ids, q, num_reqs, n_valid):
    """[NUM_CLS, E] classes and [NUM_STEP] counters of one step by the rule:
    a request decodes iff its query length <= MAX_DECODE_Q."""
    t = ids.shape[0]
    qlen = np.diff(q[: num_reqs + 1])
    dec_req = qlen <= MAX_DECODE_Q
    is_pf = bool((~dec_req).any())
    req_of_row = np.searchsorted(q[: num_reqs + 1], np.arange(t), side="right") - 1
    h = np.zeros((epc.NUM_CLS, E), dtype=np.int64)
    for r in range(t):
        if r >= n_valid:
            c = epc.CLS_PREFILL_PAD if is_pf else epc.CLS_DECODE_PAD
        elif not is_pf:
            c = epc.CLS_DECODE
        else:
            c = epc.CLS_PREFILL_DECODE if dec_req[req_of_row[r]] else epc.CLS_PREFILL
        np.add.at(h[c], ids[r], 1)
    steps = np.zeros(epc.NUM_STEP, dtype=np.int64)
    steps[epc.STEP_PREFILL if is_pf else epc.STEP_DECODE] = int(n_valid > 0)
    valid_end = int(q[num_reqs])
    steps[epc.STEP_PAD_MISMATCH] = int(
        np.sum((np.arange(t) >= n_valid) != (np.arange(t) >= valid_end))
    )
    return h, steps


def _counts():
    return epc._STATE.counts.detach().numpy().astype(np.int64).copy()


def _block(counts, slot):
    st = epc._STATE
    row = counts[slot * st.row_len : (slot + 1) * st.row_len]
    return row[: epc.NUM_CLS * E].reshape(epc.NUM_CLS, E), row[epc.NUM_CLS * E :]


def assert_step_counted(before, after, layers, logits, q, nreq, n_valid):
    for i, layer in enumerate(layers):
        exp_h, exp_s = _expected_delta(_ref_ids(layer, logits[i]), q, nreq, n_valid)
        b_h, b_s = _block(before, layer._ep_counter_slot)
        a_h, a_s = _block(after, layer._ep_counter_slot)
        np.testing.assert_array_equal(a_h - b_h, exp_h, err_msg=layer.layer_name)
        np.testing.assert_array_equal(a_s - b_s, exp_s, err_msg=layer.layer_name)


DECODE_128 = ([4] * 32, 128)  # 32 requests x (1 + 3 speculative tokens)
DECODE_PAD = ([4] * 30, 128)  # 120 valid rows, 8 padded
MIXED_128 = ([4] * 20 + [40], 128)  # 20 decodes + a 40-token prefill chunk
PREFILL_4096 = ([4] * 5 + [1000, 2000, 1000], 4096)  # 20 decode rows, 76 padded


def _run_steps(env, fn, layers, steps, seed0=0):
    outs = []
    for k, (lens, bucket) in enumerate(steps):
        q, n_valid = _set_step(env, lens, bucket)
        x, logits = _inputs(bucket, len(layers), seed0 + k)
        before = _counts()
        fn(x, logits)
        outs.append((before, _counts(), logits, q, len(lens), n_valid))
    return outs


def _check(outs, layers):
    for before, after, logits, q, nreq, n_valid in outs:
        assert_step_counted(before, after, layers, logits, q, nreq, n_valid)


def test_instrumentation_reads_the_op_statics(env):
    st = epc._STATE
    assert [s[0] for s in st.slots] == [*TARGET_LAYERS, DRAFT_LAYER]
    assert [s[2] for s in st.slots] == ["target"] * 3 + ["draft"]
    assert st.topk == TOPK and st.num_experts == E and not st.slot_order
    for layer in env.target.experts():
        assert layer._ep_counter_topk == TOPK and layer._ep_counter_router_bias


def test_update_is_a_program_output(env):
    """No in-graph copy_ into the buffer: the update is a graph output that
    AOTAutograd's runtime wrapper applies (keep_inference_input_mutations
    False, as on TPU)."""
    layers = env.target.experts()
    _run_steps(env, _compiled(layers, dynamic=False), layers, [DECODE_128])
    (gm,) = GRAPHS
    targets = [str(n.target) for n in gm.graph.nodes if n.op == "call_function"]
    assert not any("copy_" in t for t in targets), targets
    assert any("index_add" in t or "scatter_add" in t for t in targets), targets


def test_classes_are_runtime_data(env):
    """One compiled T=128 program serves decode, padded-decode and mixed
    steps: the classes follow the metadata written in place."""
    layers = env.target.experts()
    steps = [DECODE_128, DECODE_PAD, MIXED_128, DECODE_128]
    outs = _run_steps(env, _compiled(layers, dynamic=False), layers, steps)
    assert len(GRAPHS) == 1, f"recompiled: {len(GRAPHS)} graphs"
    _check(outs, layers)
    h, s = _block(_counts(), layers[1]._ep_counter_slot)
    assert h[epc.CLS_DECODE_PAD].sum() == 8 * TOPK
    assert h[epc.CLS_PREFILL].sum() == 40 * TOPK
    assert h[epc.CLS_PREFILL_DECODE].sum() == 80 * TOPK
    assert h[epc.CLS_PREFILL_PAD].sum() == 8 * TOPK
    assert list(s) == [3, 1, 0, 0]


def test_routing_outputs_bitwise_identical(env):
    """The op's outputs (combine, weights, ids) are bit for bit the same with
    and without the counter, compiled the same way."""
    layers = env.target.experts()
    counted = _compiled(layers, dynamic=False)

    def plain_fn(x, logits):
        return [
            _ORIG_TPD(lay.quant_method, lay, x, logits[i])
            for i, lay in enumerate(layers)
        ]

    plain = torch.compile(plain_fn, backend=BACKEND, fullgraph=True, dynamic=False)
    _set_step(env, *DECODE_128)
    x, logits = _inputs(DECODE_128[1], len(layers), 7)
    c0 = _counts()
    a = counted(x, logits)
    c1 = _counts()
    b = plain(x, logits)
    assert np.array_equal(_counts(), c1) and c1.sum() > c0.sum()
    for (oa, wa, ia), (ob, wb, ib) in zip(a, b):
        assert torch.equal(ia, ib) and torch.equal(wa, wb) and torch.equal(oa, ob)
        assert oa.dtype == ob.dtype and ia.dtype == ib.dtype


@pytest.mark.parametrize("router_bias", [True, False])
def test_counted_ids_match_the_decode_kernel_routing(env, monkeypatch, router_bias):
    """The ids counted equal the decode kernel's in-kernel picks (token
    counts the fused kernel serves route inside it) and the router kernel's,
    with TPU_MOE_ROUTER_TOPK_BIAS on and off."""
    layer = env.target.experts()[0]
    monkeypatch.setattr(layer, "_ep_counter_router_bias", router_bias)
    for rows in (16, 32, 64):
        _, logits = _inputs(rows, 1, rows)
        ids = epc.armed_topk_ids(layer, logits[0]).numpy()
        scores = jax.nn.sigmoid(jnp.asarray(logits[0].numpy(), jnp.float32))
        bias = jnp.asarray(layer.e_score_correction_bias.detach().numpy(), jnp.float32)
        w = np.asarray(tpd_kernel._route(scores, bias[None, :], rows, topk=TOPK))
        picked = [set(np.flatnonzero(w[r])) for r in range(rows)]
        assert all(len(p) == TOPK for p in picked)
        assert [set(ids[r]) for r in range(rows)] == picked
        ref = _ref_ids(layer, logits[0])
        np.testing.assert_array_equal(np.sort(ids, 1), np.sort(ref, 1))


def test_pad_mismatch_counter(env):
    layers = env.target.experts()[:1]
    fn = _compiled(layers, dynamic=False)
    _set_step(env, [4] * 30, 128)  # metadata: 120 valid rows
    token_padding.mark_padding(116, 128)  # mask: 116 valid rows
    fn(*_inputs(128, 1, 3))
    _, s = _block(_counts(), layers[0]._ep_counter_slot)
    assert s[epc.STEP_PAD_MISMATCH] == 4
    # The other way: rows the mask calls valid that no request holds.
    token_padding.mark_padding(124, 128)
    before = _counts()
    fn(*_inputs(128, 1, 4))
    h, s = _block(_counts() - before, layers[0]._ep_counter_slot)
    assert s[epc.STEP_PAD_MISMATCH] == 4
    assert h[epc.CLS_UNKNOWN].sum() == 4 * TOPK
    assert h[epc.CLS_DECODE].sum() == 120 * TOPK


def test_out_of_range_ids_add_nothing(env):
    layer = env.target.experts()[0]
    _set_step(env, [1] * 4, 4)
    ids = torch.tensor([[0, E - 1], [-1, 5], [E, 5], [7, 7]])
    before = _counts()
    epc.accumulate(layer, ids)
    h, s = _block(_counts() - before, layer._ep_counter_slot)
    assert h.sum() == 6 and h[epc.CLS_DECODE, 5] == 2 and h[epc.CLS_DECODE, 7] == 2
    assert h[epc.CLS_DECODE, 0] == 1 and h[epc.CLS_DECODE, E - 1] == 1
    assert list(s) == [1, 0, 0, 0]


def test_all_padded_forward_counts_no_step(env):
    """Warm-up forwards (every row padded) add padded hits, not steps."""
    layers = env.target.experts()[:1]
    fn = _compiled(layers, dynamic=False)
    _set_step(env, [1] * 32, 128)
    token_padding.mark_padding(0, 128)
    fn(*_inputs(128, 1, 4))
    h, s = _block(_counts(), layers[0]._ep_counter_slot)
    assert h[epc.CLS_DECODE_PAD].sum() == 128 * TOPK and h[epc.CLS_DECODE].sum() == 0
    assert s[epc.STEP_DECODE] == 0 and s[epc.STEP_PREFILL] == 0


def test_one_token_decodes_and_short_prefill_limit(env):
    """1-token decodes mixed with verify windows are a decode step. Known
    limit: a prefill chunk of <= 1 + num_speculative_tokens rows looks like a
    verify window in the metadata and counts as decode."""
    layers = env.target.experts()[:1]
    steps = [([1] * 4 + [4] * 28, 128), ([4] * 31 + [3], 128)]
    outs = _run_steps(env, _compiled(layers, dynamic=False), layers, steps)
    _check(outs, layers)
    _, s = _block(_counts(), layers[0]._ep_counter_slot)
    assert list(s) == [2, 0, 0, 0]


def test_one_dynamic_trace_serves_every_bucket(env):
    """vLLM traces once with a dynamic token dimension: one trace serves
    T=128 decode, T=4096 prefill and padded steps, and accepts every bucket."""
    layers = env.target.experts()
    compiled = _compiled(layers, dynamic=False)
    outs = []
    for k, (lens, bucket) in enumerate(
        [DECODE_128, PREFILL_4096, DECODE_PAD, MIXED_128]
    ):
        q, n_valid = _set_step(env, lens, bucket)
        x, logits = _inputs(bucket, len(layers), 90 + k)
        for t in (x, *logits):
            torch._dynamo.mark_dynamic(t, 0)
        before = _counts()
        compiled(x, logits)
        outs.append((before, _counts(), logits, q, len(lens), n_valid))
    assert len(GRAPHS) == 1, f"{len(GRAPHS)} traces for 4 buckets"
    _check(outs, layers)
    from vllm_torchtpu.compilation import shape_variants

    shape_env = shape_variants.trace_shape_env(GRAPHS[0])
    assert shape_env is not None
    for size in (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
        assert shape_variants.unsupported_reason(shape_env, size) is None, size


_ORIG_CLASSIFY = epc.classify_rows


def _break_out_of_place(layer, topk_ids):
    """Negative control: an out-of-place add whose result is dropped (what a
    compiled graph that never writes back looks like)."""
    st = epc._STATE
    idx = (layer._ep_counter_slot * st.row_len + topk_ids.to(torch.int64)).reshape(-1)
    st.counts.index_add(0, idx, torch.ones_like(idx, dtype=torch.int32))


def _break_one_token_rule(num_tokens, device, meta, pad, max_decode_q=1):
    """Negative control: only 1-token requests decode (misses verify windows)."""
    del max_decode_q
    return _ORIG_CLASSIFY(num_tokens, device, meta, pad, 1)


@pytest.mark.parametrize(
    "attr,broken",
    [
        ("accumulate", _break_out_of_place),
        ("classify_rows", _break_one_token_rule),
    ],
    ids=["not-written-back", "one-token-rule"],
)
def test_negative_controls_fail(env, monkeypatch, attr, broken):
    monkeypatch.setattr(epc, attr, broken)
    layers = env.target.experts()
    outs = _run_steps(
        env, _compiled(layers, dynamic=False), layers, [DECODE_128, MIXED_128]
    )
    with pytest.raises(AssertionError):
        _check(outs, layers)


def test_armed_layers_return_before_route():
    """moe_tp_decode routes inside its op: an armed layer returns before
    moe_routing.route, so the counter must wrap moe_tp_decode_forward."""
    from vllm_torchtpu.layers.adapter.quantization import fp8 as fp8_mod

    src = inspect.getsource(fp8_mod.VllmFp8MoEMethodTPU.apply_monolithic)
    assert src.index("return moe_tp_decode_forward(") < src.index("moe_routing.route(")


def test_route_wrapper_counts_unarmed_layers(env):
    layer = env.target.experts()[1]
    layer.custom_routing_function = None
    layer.use_grouped_topk = False
    layer.routed_scaling_factor = 1.0
    layer.renormalize = True
    counted = epc.make_counted_route(_ORIG_ROUTE)
    q, n_valid = _set_step(env, *MIXED_128)
    x, logits = _inputs(128, 1, 13)
    before = _counts()
    _, ids = counted(layer, x, logits[0])
    np.testing.assert_array_equal(
        np.sort(ids.numpy(), 1), np.sort(_ref_ids(layer, logits[0]), 1)
    )
    assert_step_counted(
        before, _counts(), [layer], logits, q, len(MIXED_128[0]), n_valid
    )


# --------------------------------------------------------------------------
# Dumps -> aggregate_counts.py.
# --------------------------------------------------------------------------


def _dump(directory):
    return epc.Dumper(output_dir=str(directory)).dump()


def test_dump_and_aggregate(env, tmp_path):
    layers = env.target.experts()
    fn = _compiled(layers, dynamic=False)
    _run_steps(env, fn, layers, [DECODE_128, MIXED_128])
    early = _dump(tmp_path / "early")
    _run_steps(env, fn, layers, [PREFILL_4096, DECODE_PAD], seed0=10)
    draft = env.draft.experts()
    _run_steps(env, _compiled(draft, dynamic=False), draft, [([1] * 32, 32)])
    path = _dump(tmp_path / "late")
    assert os.path.basename(path) == f"ep_routing_counts_rank0_pid{os.getpid()}.npz"
    # A second process with the same routing (ranks share the tokens).
    pid2 = f"ep_routing_counts_rank1_pid{os.getpid() + 1}.npz"
    for d, p in (("early", early), ("late", path)):
        with np.load(p) as z:
            payload = dict(z)
        payload["rank"], payload["pid"] = np.int64(1), np.int64(os.getpid() + 1)
        np.savez(tmp_path / d / pid2, **payload)

    out = tmp_path / "counts.npz"
    assert aggregate_counts.main([str(tmp_path / "late"), "-o", str(out)]) == 0
    st = epc._STATE
    snap = epc.snapshot_arrays(_counts(), st)
    with np.load(out) as z:
        assert list(z["layers"]) == list(TARGET_LAYERS)  # draft layer left out
        assert int(z["workers"]) == 2 and bool(z["ranks_identical"])
        assert int(z["topk"]) == TOPK and int(z["num_experts"]) == E
        for i, layer in enumerate(TARGET_LAYERS):
            by = snap["by_class"][i]
            np.testing.assert_array_equal(z["decode"][layer], 2 * by[epc.CLS_DECODE])
            np.testing.assert_array_equal(
                z["prefill"][layer],
                2 * (by[epc.CLS_PREFILL] + by[epc.CLS_PREFILL_DECODE]),
            )
        assert z["decode"][3].sum() == 2 * (128 + 120) * TOPK
        assert list(z["steps"][4]) == [4, 4, 0, 0]
        checks = json.loads(str(z["checks_json"]))
        assert checks["unknown_hits"] == 0 and checks["hits_multiple_of_topk"]

    window = tmp_path / "window.npz"
    assert (
        aggregate_counts.main(
            [
                str(tmp_path / "late"),
                "-o",
                str(window),
                "--subtract",
                str(tmp_path / "early"),
            ]
        )
        == 0
    )
    with np.load(window) as z:
        assert z["decode"][3].sum() == 2 * 120 * TOPK  # DECODE_PAD only
        assert list(z["steps"][3]) == [2, 2, 0, 0]
    with pytest.raises(ValueError, match="same processes"):
        aggregate_counts.main(
            [
                str(tmp_path / "late"),
                "-o",
                str(window),
                "--subtract",
                str(tmp_path / "x"),
            ]
        )


def test_placed_layer_counts_equal_the_unplaced_layer(env, monkeypatch, tmp_path):
    """A layer placed by TPU_EXPERT_PLACEMENT routes over router rows that
    the loader reordered (virtual_order); the counter maps its counts back,
    so they equal the counts of the same tokens on the unplaced layer."""
    from vllm_torchtpu import expert_placement as ep

    perm = np.random.default_rng(0).permutation(E).tolist()
    raw = {
        "experts": E,
        "shards": 16,
        "layers": {"3": [perm[r * 16 : (r + 1) * 16] for r in range(16)]},
    }
    table = ep.parse(raw, "test")
    monkeypatch.setattr(ep, "placement", lambda: table)
    monkeypatch.setattr(ep, "_LOCAL_IDS", {3: frozenset(table.cores[3][0])})
    layer, other = env.target.experts()[:2]
    order = epc._placement_order(layer.layer_name)
    assert order == ep.virtual_order(table, 3)
    assert epc._placement_order(other.layer_name) is None  # layer 4: linear
    _set_step(env, *DECODE_128)
    x, logits = _inputs(128, 1, 21)
    expected = np.zeros(E, dtype=np.int64)
    np.add.at(expected, _ref_ids(layer, logits[0]).reshape(-1), 1)
    # The placed layer, as the loader leaves it: router rows (so the logits'
    # columns) and the correction bias in virtual order.
    idx = torch.tensor(order)
    layer._moe_tp_decode_bias = layer._moe_tp_decode_bias.index_select(0, idx)
    layer.e_score_correction_bias.data = layer.e_score_correction_bias.data[idx]
    st = epc._STATE
    monkeypatch.setattr(st, "slot_order", {layer._ep_counter_slot: order})
    before = _counts()
    _compiled([layer], dynamic=False)(x, [logits[0][:, idx]])
    delta = (
        epc.snapshot_arrays(_counts(), st)["by_class"]
        - epc.snapshot_arrays(before, st)["by_class"]
    )
    np.testing.assert_array_equal(
        delta[layer._ep_counter_slot, epc.CLS_DECODE], expected
    )
    with np.load(_dump(tmp_path)) as z:
        assert list(z["slot_placed"]) == [True, False, False, False]


def test_dumps_between_steps_after_the_interval(env, tmp_path):
    d = epc.Dumper(output_dir=str(tmp_path), interval_s=3600)
    layers = env.target.experts()[:1]
    fn = _compiled(layers, dynamic=False)
    _set_step(env, *DECODE_128)
    x, logits = _inputs(128, 1, 5)
    d.step(fn, x, logits)
    assert not list(tmp_path.iterdir())  # interval not reached: no dump
    d.interval_s = 0.0
    d.step(fn, x, logits)
    (path,) = tmp_path.iterdir()
    with np.load(path) as z:
        assert z["counts_by_class"][0, epc.CLS_DECODE].sum() == 2 * 128 * TOPK


def test_data_parallel_is_refused(monkeypatch):
    """With data parallelism the rows a layer routes are not the rows the
    metadata describes: load_model leaves everything uncounted."""

    class Runner:
        parallel_config = SimpleNamespace(
            data_parallel_size=2, prefill_context_parallel_size=1
        )
        model = Model(TARGET_LAYERS, 0)

        def load_model(self):
            return "loaded"

        def execute_model(self):
            return "ran"

    monkeypatch.setattr(epc, "_STATE", None)
    assert epc.install_runner_patch(SimpleNamespace(TPUModelRunner=Runner))
    assert Runner().load_model() == "loaded"
    assert epc._STATE is None
    assert not hasattr(Runner.execute_model, "_ep_counter_orig")


def _run_python(code, enable, tmp_path):
    """A fresh interpreter with the counter directory first on PYTHONPATH."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(COUNTER_DIR), env.get("PYTHONPATH", "")])
    env["EP_ROUTING_COUNTER"] = "1" if enable else "0"
    env.pop("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS", None)
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=600,
        check=True,
    )
    return (out.stdout + out.stderr).splitlines()


def test_sitecustomize_chains_and_stays_light(tmp_path):
    """The counter's sitecustomize runs the one it shadows (vllm_torchtpu's
    sets torch_tpu's import-time defaults) and imports no torch or vLLM in a
    process that never loads them."""
    lines = _run_python(
        "import os, sys; print('MAT=' + os.environ.get("
        "'TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS', 'unset'));"
        " print('HEAVY=' + str(any(m in sys.modules for m in ('torch', 'vllm'))))",
        True,
        tmp_path,
    )
    assert "MAT=false" in lines and "HEAVY=False" in lines, lines


def test_sitecustomize_wraps_targets_on_import(tmp_path):
    runner = (
        "import vllm_torchtpu.runner.tpu_runner as m;"
        " load = m.TPUModelRunner.load_model;"
        " print('RUNNER=' + str(hasattr(load, '_ep_counter_orig')))"
    )
    on = _run_python(runner, True, tmp_path)
    assert "RUNNER=True" in on, on
    off = _run_python(runner, False, tmp_path)
    assert "RUNNER=False" in off, off
