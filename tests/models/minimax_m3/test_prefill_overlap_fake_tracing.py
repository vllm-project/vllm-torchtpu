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
"""The MiniMax M3 overlap step (`models/minimax_m3/overlap.py`) under
``torch.compile(fullgraph=True)`` with the token dim dynamic, as Dynamo runs
it in the runner's warmup: the token halves are symbolic (s0 // 2), the half
all-reduces trace as torch functional collectives (``_c10d_functional.
all_reduce`` + ``wait_tensor``, which torch_tpu lowers as XLA all-reduces in
the step's computation, not inside a jax_op), the padding mask of half B is
sliced at the symbolic row offset, and the trace leaves a guard on the token
count (so `compilation.shape_variants` splits the buckets at the threshold).
A one-rank gloo group stands in for the TP group.
"""

import types

import pytest
import torch
import torch.distributed as dist

from tests.models.minimax_m3 import test_prefill_overlap as cpu
from vllm_torchtpu.layers.adapter import token_padding
from vllm_torchtpu.models.minimax_m3 import model as mm
from vllm_torchtpu.models.minimax_m3 import overlap

pytest.importorskip("torch_tpu")


@pytest.fixture(scope="module")
def group():
    if not dist.is_initialized():
        dist.init_process_group(
            "gloo", init_method="tcp://127.0.0.1:29533", rank=0, world_size=1
        )
    yield dist.group.WORLD


@pytest.fixture
def tp(monkeypatch, group):
    tp_group = types.SimpleNamespace(device_group=group)
    monkeypatch.setattr(overlap, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(overlap.shape_variants, "aot_cache_tag", lambda: "trace1")


def _compile(fn, *args, dynamic=(0,)):
    graphs = []

    def backend(gm, example_inputs):
        gm.trace_shape_env = torch._guards.detect_fake_mode(example_inputs).shape_env
        graphs.append(gm)
        return gm.forward

    for i, a in enumerate(args):
        if i in dynamic:
            torch._dynamo.mark_dynamic(a, 0)
    torch._dynamo.reset()
    out = torch.compile(fn, backend=backend, fullgraph=True, dynamic=False)(*args)
    assert len(graphs) == 1
    return out, graphs[0]


def _count(gm, name):
    return sum(
        1
        for n in gm.graph.nodes
        if n.op == "call_function" and str(n.target).startswith(name)
    )


def test_xla_route_traces_functional_all_reduce_on_symbolic_halves(tp):
    def fn(ids, x):
        a, b = overlap.split(x, ids.shape[0] // 2)
        ra = overlap.all_reduce(a, "xla")
        rb = overlap.all_reduce(b * 2, "xla")
        return torch.cat([overlap.wait(ra), overlap.wait(rb)])

    ids = torch.zeros(64, dtype=torch.long)
    x = torch.randn(64, 16)
    out, gm = _compile(fn, ids, x, dynamic=(0, 1))
    torch.testing.assert_close(out, torch.cat([x[:32], x[32:] * 2]))
    assert _count(gm, "_c10d_functional.all_reduce") == 2
    assert _count(gm, "_c10d_functional.wait_tensor") == 2


def test_padding_row_offset_slices_at_the_symbolic_half(tp):
    state = token_padding.TokenPaddingState.create(128, "cpu")
    state.update(40, 64)
    token_padding.set_padding_state(state)
    try:

        def fn(ids):
            half = ids.shape[0] // 2
            with token_padding.row_offset(half):
                return state.get_local_padding_mask(half)

        ids = torch.zeros(64, dtype=torch.long)
        out, _ = _compile(fn, ids)
        assert out.tolist() == [False] * 8 + [True] * 24
    finally:
        token_padding.set_padding_state(None)


def test_model_overlap_step_compiles_fullgraph(tp, monkeypatch):
    """The whole overlap forward (stand-in layers, `[T, R, H]` activations,
    real functional all-reduces over the one-rank group) traces into one
    graph with a dynamic token dim and matches the eager plain path."""
    monkeypatch.setattr(cpu, "_ar", lambda x: x)  # the group does the sum
    cfg = overlap.OverlapConfig(32, "xla", True)
    layers = cpu._layers(cfg, n=2)
    fake = cpu._fake_model(layers, cfg, aux=(1,))
    fake.tp_size = 8  # the step gate needs TP > 1
    x = cpu._input()
    ids = torch.zeros(cpu.T, dtype=torch.long)
    positions = torch.zeros(cpu.T, dtype=torch.long)

    def fn(ids, x):
        return mm.MiniMaxM3Model.forward(fake, ids, positions, x)

    (got, aux), gm = _compile(fn, ids, x, dynamic=(0, 1))
    plain = cpu._fake_model(layers, overlap.OFF, aux=(1,))
    want, want_aux = mm.MiniMaxM3Model.forward(plain, ids, positions, x)
    torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(aux[0], want_aux[0], rtol=1e-12, atol=1e-12)
    assert _count(gm, "_c10d_functional.all_reduce") == 8
    # The branch left a guard on the token count (s0 >= 32, s0 % 32 == 0).
    guards = [str(g.expr) for g in gm.trace_shape_env.guards]
    assert any("32" in g for g in guards), guards
