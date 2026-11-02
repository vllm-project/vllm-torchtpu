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
"""Unit tests for the Gemma4 sequence-parallel residual stream.

TP ranks run as threads over a barrier-synchronized collective mock. Each test
drives upstream's own Gemma4 attention, MLP and decoder-layer forwards on
lightweight per-rank stand-ins for the parallel linears, the attention kernel
and the MoE runner, so the reference is upstream's dataflow, not a copy of it.
"""

from __future__ import annotations

import threading
import types
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch.fx.passes.split_module import split_module
from vllm.model_executor.models import gemma4
from vllm.model_executor.models.gemma4 import (
    Gemma4Attention,
    Gemma4DecoderLayer,
    Gemma4MLP,
    Gemma4Model,
)
from vllm.model_executor.models.interfaces import EagleModelMixin

from vllm_torchtpu import envs
from vllm_torchtpu.compilation import shape_variants
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU
from vllm_torchtpu.models.vllm import gemma4_patch

pytestmark = pytest.mark.usefixtures("default_on_knobs_off")

_UPSTREAM_LAYER_FORWARD = Gemma4DecoderLayer.forward


class _TPCollectiveMock:
    """Barrier-synchronized in-process collectives over TP rank threads."""

    def __init__(self, tp_size: int) -> None:
        self.tp_size = tp_size
        self.barrier = threading.Barrier(tp_size)
        self.storage: list[Any] = [None] * tp_size
        self.calls: list[tuple[str, torch.dtype, tuple[int, ...]]] = []

    def _exchange(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        rank = threading.current_thread().rank  # type: ignore[attr-defined]
        self.storage[rank] = tensor
        self.barrier.wait(timeout=30.0)
        tensors = list(self.storage)
        self.barrier.wait(timeout=30.0)
        if rank == 0:
            self.calls.append(("", tensor.dtype, tuple(tensor.shape)))
        return tensors

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        return sum(self._exchange(tensor))  # type: ignore[return-value]

    def reduce_scatter(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        rank = threading.current_thread().rank  # type: ignore[attr-defined]
        total = sum(self._exchange(tensor))
        return total.chunk(self.tp_size, dim=dim)[rank].clone()  # type: ignore[union-attr]

    def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        return torch.cat(self._exchange(tensor), dim=dim)


def _run_ranks(tp: int, fn: Callable[[int], Any]) -> list[Any]:
    results: list[Any] = [None] * tp
    errors: list[BaseException | None] = [None] * tp

    def worker(rank: int) -> None:
        try:
            results[rank] = fn(rank)
        except BaseException as e:  # noqa: BLE001 - re-raised below
            errors[rank] = e

    threads = []
    for rank in range(tp):
        t = threading.Thread(target=worker, args=(rank,))
        t.rank = rank  # type: ignore[attr-defined]
        threads.append(t)
        t.start()
    for t in threads:
        t.join(timeout=60.0)
        assert not t.is_alive()
    for e in errors:
        if e is not None:
            raise e
    return results


@pytest.fixture
def mock_collectives(monkeypatch: pytest.MonkeyPatch):
    def install(tp: int) -> _TPCollectiveMock:
        mock = _TPCollectiveMock(tp)
        for name, fn in (
            ("tensor_model_parallel_all_reduce", mock.all_reduce),
            ("_reduce_scatter_rows", mock.reduce_scatter),
            ("_all_gather_rows", mock.all_gather),
        ):
            monkeypatch.setattr(gemma4_patch, name, fn)
        return mock

    return install


# ---------------------------------------------------------------------------
# Per-rank stand-ins
# ---------------------------------------------------------------------------


class _Norm(torch.nn.Module):
    def __init__(self, dim: int, has_weight: bool = True) -> None:
        super().__init__()
        self.weight = (
            torch.nn.Parameter(1.0 + 0.1 * torch.randn(dim)) if has_weight else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        return y if self.weight is None else y * self.weight


class _Bf16Method:
    """A linear method that cannot take quantized rows."""

    def apply(self, layer: Any, x: torch.Tensor) -> torch.Tensor:
        return x @ layer.weight


class _PerTokenFp8Method(VllmFp8LinearMethodTPU):
    """VllmFp8LinearMethodTPU's per-token split, emulated in plain torch."""

    def __init__(self) -> None:  # no quant config: only the split is used
        pass

    def supports_quantized_input(self, layer: Any, rows: int) -> bool:
        return True

    def quantize_input(
        self, layer: Any, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale = x.abs().amax(-1, keepdim=True) / 448.0
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        return (x / scale).to(torch.float8_e4m3fn), scale

    def apply_quantized_input(
        self, layer: Any, x_q: torch.Tensor, x_scale: torch.Tensor
    ) -> torch.Tensor:
        assert x_q.dtype == torch.float8_e4m3fn
        return (x_q.to(torch.float32) @ layer.weight) * x_scale

    def apply(self, layer: Any, x: torch.Tensor) -> torch.Tensor:
        # As linear_common.quantized_matmul: 2-D in, reshaped back out.
        x_2d = x.reshape(-1, x.shape[-1])
        out = self.apply_quantized_input(layer, *self.quantize_input(layer, x_2d))
        return out.reshape(*x.shape[:-1], out.shape[-1])


class _Linear:
    """A column- or row-parallel linear shard; row-parallel reduces like vLLM."""

    def __init__(
        self,
        weight: torch.Tensor,
        tp_size: int,
        method: _Bf16Method | _PerTokenFp8Method,
        all_reduce: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.weight = weight
        self.tp_size = tp_size
        self.quant_method = method
        self.bias = None
        self.reduce_results = all_reduce is not None
        self._all_reduce = all_reduce

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        out = self.quant_method.apply(self, x)
        if self.reduce_results:
            assert self._all_reduce is not None
            out = self._all_reduce(out)
        return out, None


class _Experts:
    """One expert per rank, weighted by the softmax of the router logits."""

    def __init__(self, rank: int, weight: torch.Tensor, all_reduce: Callable) -> None:
        self.rank = rank
        self.weight = weight
        self.moe_config = SimpleNamespace(
            dp_size=1, is_sequence_parallel=False, skip_final_all_reduce=False
        )
        self._all_reduce = all_reduce

    def __call__(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor:
        gate = torch.softmax(router_logits, dim=-1)[:, self.rank : self.rank + 1]
        out = (hidden_states @ self.weight) * gate
        if not self.moe_config.skip_final_all_reduce:
            out = self._all_reduce(out)
        return out


def _causal_mix(q, k, v, num_heads: int, num_kv_heads: int, head_dim: int):
    """Token-mixing stand-in for the attention kernel (a causal prefix mean)."""
    t = q.shape[0]
    kv = (k + v).view(t, num_kv_heads, head_dim)
    kv = kv.repeat_interleave(num_heads // num_kv_heads, dim=1)
    prefix = kv.cumsum(0) / torch.arange(1, t + 1, dtype=kv.dtype).view(t, 1, 1)
    return (q.view(t, num_heads, head_dim) * prefix).flatten(1)


def _rope(positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor):
    # Viewed by the positions' length, as vLLM's rotary embedding does.
    num_tokens = positions.shape[0]
    angle = positions.to(q.dtype)[:, None] * 0.01
    q = q.view(num_tokens, -1) * torch.cos(angle)
    return q, (k + k * torch.sin(angle)).view(num_tokens, -1)


class _Model:
    """Full-size weights of a small Gemma4 layer stack, sharded per rank."""

    hidden, heads, kv_heads, head_dim, inter = 32, 8, 8, 8, 64

    def __init__(self, tp: int, moe: bool, fp8: bool, num_layers: int = 1) -> None:
        torch.manual_seed(1234)
        self.tp, self.moe, self.num_layers = tp, moe, num_layers
        self.method = _PerTokenFp8Method() if fp8 else _Bf16Method()
        h, hd = self.hidden, self.head_dim

        def w(*shape: int) -> torch.Tensor:
            return torch.randn(*shape) / shape[0] ** 0.5

        self.layer_weights = [
            dict(
                q=w(h, self.heads * hd),
                k=w(h, self.kv_heads * hd),
                v=w(h, self.kv_heads * hd),
                o=w(self.heads * hd, h),
                gate=w(h, self.inter),
                up=w(h, self.inter),
                down=w(self.inter, h),
                router=w(h, tp),
                experts=[w(h, h) for _ in range(tp)],
                norms={
                    name: _Norm(h, has_weight=name != "router")
                    for name in (
                        "input",
                        "post_attention",
                        "pre_feedforward",
                        "post_feedforward",
                        "post_feedforward_1",
                        "pre_feedforward_2",
                        "post_feedforward_2",
                        "router",
                    )
                },
                head_norms=(_Norm(hd), _Norm(hd), _Norm(hd, has_weight=False)),
                layer_scalar=torch.tensor([0.75]),
            )
            for _ in range(num_layers)
        ]
        self.embedding = torch.randn(64, h)
        self.final_norm = _Norm(h)

    def layer(self, index: int, rank: int, all_reduce: Callable) -> Any:
        lw = self.layer_weights[index]
        tp, hd = self.tp, self.head_dim
        heads, kv_heads, inter = self.heads // tp, self.kv_heads // tp, self.inter // tp

        def cols(weight: torch.Tensor, n: int) -> torch.Tensor:
            return weight[:, rank * n : (rank + 1) * n]

        attn = SimpleNamespace(
            is_kv_shared_layer=False,
            num_heads=heads,
            num_kv_heads=kv_heads,
            head_dim=hd,
            q_size=heads * hd,
            kv_size=kv_heads * hd,
            qkv_proj=_Linear(
                torch.cat(
                    [
                        cols(lw["q"], heads * hd),
                        cols(lw["k"], kv_heads * hd),
                        cols(lw["v"], kv_heads * hd),
                    ],
                    dim=1,
                ),
                tp,
                self.method,
            ),
            o_proj=_Linear(
                lw["o"][rank * heads * hd : (rank + 1) * heads * hd],
                tp,
                self.method,
                all_reduce,
            ),
            q_norm=lw["head_norms"][0],
            k_norm=lw["head_norms"][1],
            v_norm=lw["head_norms"][2],
            rotary_emb=_rope,
            attn=lambda q, k, v: _causal_mix(q, k, v, heads, kv_heads, hd),
        )

        def attn_forward(positions, hidden_states, **kwargs):
            return Gemma4Attention.forward(attn, positions, hidden_states, **kwargs)

        mlp = SimpleNamespace(
            gate_up_proj=_Linear(
                torch.cat([cols(lw["gate"], inter), cols(lw["up"], inter)], dim=1),
                tp,
                self.method,
            ),
            down_proj=_Linear(
                lw["down"][rank * inter : (rank + 1) * inter],
                tp,
                self.method,
                all_reduce,
            ),
            act_fn=lambda gu: F.gelu(gu[:, :inter], approximate="tanh") * gu[:, inter:],
        )
        norms = lw["norms"]
        router_norm = norms["router"]
        router_weight = lw["router"]
        layer = _CallableLayer(
            self_attn=_CallableAttention(attn, attn_forward),
            mlp=_CallableMlp(mlp),
            input_layernorm=norms["input"],
            post_attention_layernorm=norms["post_attention"],
            pre_feedforward_layernorm=norms["pre_feedforward"],
            post_feedforward_layernorm=norms["post_feedforward"],
            enable_moe_block=self.moe,
            router=(lambda h: router_norm(h) @ router_weight) if self.moe else None,
            experts=(
                _Experts(rank, lw["experts"][rank], all_reduce) if self.moe else None
            ),
            post_feedforward_layernorm_1=(
                norms["post_feedforward_1"] if self.moe else None
            ),
            pre_feedforward_layernorm_2=(
                norms["pre_feedforward_2"] if self.moe else None
            ),
            post_feedforward_layernorm_2=(
                norms["post_feedforward_2"] if self.moe else None
            ),
            per_layer_input_gate=None,
            layer_scalar=lw["layer_scalar"],
        )
        return layer


class _CallableAttention(SimpleNamespace):
    def __init__(self, attn: SimpleNamespace, forward: Callable) -> None:
        super().__init__(**vars(attn))
        self._forward = forward

    def __call__(self, positions, hidden_states, **kwargs):
        return self._forward(positions, hidden_states, **kwargs)


class _CallableMlp(SimpleNamespace):
    def __init__(self, mlp: SimpleNamespace) -> None:
        super().__init__(**vars(mlp))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return Gemma4MLP.forward(self, x)


class _CallableLayer(SimpleNamespace):
    def __call__(self, *args: Any, **kwargs: Any):
        return gemma4_patch._patched_layer_forward(self, *args, **kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("chunks", [1, 2])
def test_layer_matches_upstream(
    mock_collectives, tp: int, moe: bool, fp8: bool, chunks: int
):
    """Both branches reproduce upstream's layer; SP on this rank's chunk rows."""
    mock = mock_collectives(tp)
    model = _Model(tp, moe, fp8)
    num_tokens = 16
    x = torch.randn(num_tokens, model.hidden)
    positions = torch.arange(num_tokens) + 100

    def rank_fn(rank: int):
        upstream = model.layer(0, rank, mock.all_reduce)
        ours = model.layer(0, rank, mock.all_reduce)
        gemma4_patch._configure_layer(ours)
        up, _ = _UPSTREAM_LAYER_FORWARD(upstream, positions, x.clone(), None)
        ar = gemma4_patch._all_reduce_layer(ours, positions, x.clone())
        rows = [c.chunk(tp)[rank].clone() for c in x.chunk(chunks)]
        sp = gemma4_patch._sequence_parallel_layer(ours, positions, rows)
        return up, ar, sp

    results = _run_ranks(tp, rank_fn)
    upstream = results[0][0]
    # Chunk-major, then rank-major: the token order the model forward restores.
    sp = torch.cat([r[2][c] for c in range(chunks) for r in results])
    for up, ar, _ in results:
        torch.testing.assert_close(up, upstream, rtol=0, atol=0)
        torch.testing.assert_close(ar, upstream, rtol=0, atol=0)
    torch.testing.assert_close(sp, upstream, rtol=1e-5, atol=1e-5)


def test_sequence_parallel_layer_collectives(mock_collectives):
    """FP8 rows travel as int8 plus f32 scales; two reductions are scatters."""
    tp = 4
    mock = mock_collectives(tp)
    model = _Model(tp, moe=False, fp8=True)
    x = torch.randn(16, model.hidden)

    def rank_fn(rank: int):
        layer = model.layer(0, rank, mock.all_reduce)
        gemma4_patch._configure_layer(layer)
        return gemma4_patch._sequence_parallel_layer(
            layer, torch.arange(16), [x.chunk(tp)[rank]]
        )

    _run_ranks(tp, rank_fn)
    h = model.hidden
    assert [c[1:] for c in mock.calls] == [
        (torch.int8, (4, h)),  # qkv input rows
        (torch.float32, (4, 1)),  # their scales
        (torch.float32, (16, h)),  # o_proj partial, reduce-scattered
        (torch.int8, (4, h)),  # gate_up input rows
        (torch.float32, (4, 1)),
        (torch.float32, (16, h)),  # down_proj partial, reduce-scattered
    ]


@pytest.mark.parametrize("chunks", [1, 2])
def test_model_forward_matches_all_reduce(mock_collectives, monkeypatch, chunks):
    """Embedding shard -> SP layers -> final norm -> gather == upstream."""
    tp, num_layers, num_tokens = 4, 3, 32
    mock = mock_collectives(tp)
    model = _Model(tp, moe=True, fp8=True, num_layers=num_layers)
    monkeypatch.setattr(
        gemma4,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    input_ids = torch.randint(0, model.embedding.shape[0], (num_tokens,))
    positions = torch.arange(num_tokens)

    def build(rank: int) -> Any:
        fake = SimpleNamespace(
            layers=[model.layer(i, rank, mock.all_reduce) for i in range(num_layers)],
            start_layer=0,
            end_layer=num_layers,
            norm=model.final_norm,
            aux_hidden_state_layers=(),
            fast_prefill_enabled=False,
            embed_input_ids=lambda ids: model.embedding[ids],
            get_per_layer_inputs=lambda ids: None,
            project_per_layer_inputs=lambda h, p: None,
            config=SimpleNamespace(hidden_size=model.hidden),
            normalizer=torch.tensor(1.0),
        )
        fake._maybe_add_hidden_state = types.MethodType(
            EagleModelMixin._maybe_add_hidden_state, fake
        )
        for layer in fake.layers:
            gemma4_patch._configure_layer(layer)
        return fake

    def forward(rank: int) -> torch.Tensor:
        return gemma4_patch._patched_model_forward(
            build(rank), input_ids, positions, None
        )

    monkeypatch.setattr(
        gemma4_patch, "get_tensor_model_parallel_world_size", lambda: tp
    )
    monkeypatch.setattr(
        gemma4_patch,
        "sequence_parallel_chunk",
        lambda x: x.chunk(tp)[threading.current_thread().rank].clone(),
    )
    shard_bytes = num_tokens // tp * model.hidden * 4
    monkeypatch.setattr(gemma4_patch, "_MIN_SHARD_BYTES", shard_bytes + 1)
    reference = _run_ranks(tp, forward)
    assert {c[2] for c in mock.calls} == {(num_tokens, model.hidden)}
    mock.calls.clear()
    monkeypatch.setattr(gemma4_patch, "_MIN_SHARD_BYTES", shard_bytes // chunks)
    monkeypatch.setattr(gemma4_patch, "_microbatches", chunks)
    sp = _run_ranks(tp, forward)
    shard_rows = num_tokens // tp // chunks
    assert (shard_rows, model.hidden) in {c[2] for c in mock.calls}
    for out in sp:
        torch.testing.assert_close(out, reference[0], rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("moe", [False, True])
def test_branches_trace_the_same_inputs(monkeypatch, moe: bool) -> None:
    """The all-reduce and sequence-parallel traces take the same tensors, in order.

    The shape-variant warmup hands a bucket's executable from one trace to the
    other, and the receiving graph calls it with its own inputs by position.
    """
    tp, rank = 4, 1
    pp = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    for name, fn in (
        ("tensor_model_parallel_all_reduce", lambda x: x * 1),
        ("_reduce_scatter_rows", lambda x: x.chunk(tp)[rank].contiguous()),
        ("_all_gather_rows", lambda x: torch.cat([x] * tp)),
        ("get_tensor_model_parallel_world_size", lambda: tp),
        ("sequence_parallel_chunk", lambda x: x.chunk(tp)[rank].clone()),
    ):
        monkeypatch.setattr(gemma4_patch, name, fn)
    monkeypatch.setattr(gemma4, "get_pp_group", lambda: pp)
    model = _Model(tp, moe=moe, fp8=True, num_layers=2)
    fake = SimpleNamespace(
        layers=[
            model.layer(i, rank, gemma4_patch.tensor_model_parallel_all_reduce)
            for i in range(2)
        ],
        start_layer=0,
        end_layer=2,
        norm=model.final_norm,
        aux_hidden_state_layers=(),
        fast_prefill_enabled=False,
        embed_input_ids=lambda ids: model.embedding[ids],
        get_per_layer_inputs=lambda ids: None,
        project_per_layer_inputs=lambda h, p: None,
        config=SimpleNamespace(hidden_size=model.hidden),
        normalizer=torch.tensor(1.0),
    )
    fake._maybe_add_hidden_state = types.MethodType(
        EagleModelMixin._maybe_add_hidden_state, fake
    )
    for layer in fake.layers:
        gemma4_patch._configure_layer(layer)
    input_ids, positions = torch.randint(0, 64, (64,)), torch.arange(64)

    def traced_inputs(min_shard_bytes: int, chunks: int) -> tuple[tuple, list[int]]:
        """vt's signature of the graph vLLM compiles, and its tensors' storage."""
        monkeypatch.setattr(gemma4_patch, "_MIN_SHARD_BYTES", min_shard_bytes)
        monkeypatch.setattr(gemma4_patch, "_microbatches", chunks)
        seen: list[tuple[Any, list[Any]]] = []

        def backend(gm, example_inputs):
            seen.append((gm, example_inputs))
            return gm.forward

        torch._dynamo.reset()
        compiled = torch.compile(
            lambda ids, pos: gemma4_patch._patched_model_forward(fake, ids, pos, None),
            backend=backend,
            fullgraph=True,
        )
        ids, pos = input_ids.clone(), positions.clone()
        # As vLLM marks them: the token dimension is the only dynamic one.
        torch._dynamo.mark_dynamic(ids, 0)
        torch._dynamo.mark_dynamic(pos, 0)
        with shape_variants.warmup():
            compiled(ids, pos)
        graph, example_inputs = seen[0]
        # vLLM's split orders a subgraph's inputs by first use.
        subgraph = split_module(graph, None, lambda node: 0).submod_0
        skip = {ids.data_ptr(), pos.data_ptr()}
        tensors = [
            x.data_ptr()
            for x in example_inputs
            if isinstance(x, torch.Tensor) and x.data_ptr() not in skip
        ]
        return shape_variants.graph_signature(subgraph), tensors

    signature, tensors = traced_inputs(1 << 60, 1)
    assert "SymInt" in signature and tensors
    assert traced_inputs(0, 1) == (signature, tensors)
    assert traced_inputs(0, 2) == (signature, tensors)


def test_bf16_gather_when_method_cannot_take_quantized_rows(mock_collectives):
    tp = 2
    mock = mock_collectives(tp)
    model = _Model(tp, moe=False, fp8=False)
    x = torch.randn(8, model.hidden)

    def rank_fn(rank: int):
        layer = model.layer(0, rank, mock.all_reduce)
        return gemma4_patch._gathered_projection(
            layer.self_attn.qkv_proj, x.chunk(tp)[rank]
        )

    out = _run_ranks(tp, rank_fn)
    assert mock.calls == [("", torch.float32, (4, model.hidden))]
    assert out[0].shape[0] == 8


def _collectives_worker(rank: int, world: int, port: int) -> None:
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world
    )
    try:
        group = SimpleNamespace(device_group=dist.group.WORLD)
        gemma4_patch.get_tp_group = lambda: group
        rows = torch.arange(world * 4 * 3, dtype=torch.float32).view(world * 4, 3)
        partial = rows + 100 * rank
        expected_sum = sum(rows + 100 * r for r in range(world))
        torch.testing.assert_close(
            gemma4_patch._reduce_scatter_rows(partial),
            expected_sum.chunk(world)[rank],
            rtol=0,
            atol=0,
        )
        shard = rows.chunk(world)[rank]
        torch.testing.assert_close(
            gemma4_patch._all_gather_rows(shard), rows, rtol=0, atol=0
        )
        fp8 = (shard / 16).to(torch.float8_e4m3fn)
        gathered = gemma4_patch._all_gather_rows(fp8.view(torch.int8))
        assert torch.equal(
            gathered.view(torch.float8_e4m3fn).float(),
            (rows / 16).to(torch.float8_e4m3fn).float(),
        )

        # Traced, every wait_tensor must consume a collective, not another wait
        # (torch_tpu's split compiler refuses a wait on a wait).
        graphs = []

        def backend(gm, example_inputs):
            graphs.append(gm)
            return gm.forward

        def step(x):
            return gemma4_patch._all_gather_rows(gemma4_patch._reduce_scatter_rows(x))

        traced = torch.compile(step, backend=backend, fullgraph=True)(partial)
        torch.testing.assert_close(traced, step(partial), rtol=0, atol=0)
        wait = torch.ops._c10d_functional.wait_tensor

        def is_wait(node: Any) -> bool:
            return getattr(node.target, "overloadpacket", node.target) is wait

        waits = [node for node in graphs[0].graph.nodes if is_wait(node)]
        assert len(waits) == 2, graphs[0].code
        assert not any(is_wait(node.args[0]) for node in waits), graphs[0].code
    finally:
        dist.destroy_process_group()


def test_sequence_parallel_collectives_on_a_process_group() -> None:
    """Rank-ordered gathers (FP8 rows as int8 too) and sum-and-chunk scatters."""
    import portpicker
    import torch.multiprocessing as mp

    mp.spawn(_collectives_worker, args=(2, portpicker.pick_unused_port()), nprocs=2)


@pytest.mark.parametrize(
    ("num_tokens", "hidden", "tp", "expected"),
    [
        (16384, 5376, 8, True),  # 31B prefill bucket
        (1024, 5376, 8, True),  # 1.3 MiB per rank
        (512, 5376, 8, False),
        (64, 5376, 8, False),  # decode
        (8192, 2816, 8, True),  # 26B prefill bucket
        (1024, 2816, 8, False),
        (16384, 5376, 2, True),
        (4100, 5376, 8, False),  # rows do not split evenly
    ],
)
def test_use_sequence_parallel(num_tokens, hidden, tp, expected) -> None:
    row_bytes = hidden * 2
    assert gemma4_patch._use_sequence_parallel(num_tokens, row_bytes, tp) is expected


@pytest.mark.parametrize(
    ("setting", "num_tokens", "expected"),
    [
        (1, 16384, 1),
        (2, 16384, 2),
        (2, 1024, 1),  # a half would be 0.66 MiB per rank
        (2, 2048, 2),
        (4, 16384, 4),
    ],
)
def test_num_microbatches(monkeypatch, setting, num_tokens, expected) -> None:
    monkeypatch.setattr(gemma4_patch, "_microbatches", setting)
    assert gemma4_patch._num_microbatches(num_tokens, 5376 * 2, 8) == expected


def test_sequence_parallel_only_in_backbone_warmup_when_compiling(monkeypatch) -> None:
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    assert not gemma4_patch._use_sequence_parallel(16384, 5376 * 2, 8)
    with shape_variants.warmup():
        assert gemma4_patch._use_sequence_parallel(16384, 5376 * 2, 8)


def _layer_for_configure(**overrides: Any) -> SimpleNamespace:
    def linear() -> SimpleNamespace:
        return SimpleNamespace(bias=None, tp_size=8, reduce_results=True)

    layer = SimpleNamespace(
        self_attn=SimpleNamespace(
            is_kv_shared_layer=False, qkv_proj=linear(), o_proj=linear()
        ),
        mlp=SimpleNamespace(gate_up_proj=linear(), down_proj=linear()),
        per_layer_input_gate=None,
        experts=SimpleNamespace(
            moe_config=SimpleNamespace(
                dp_size=1, is_sequence_parallel=False, skip_final_all_reduce=False
            )
        ),
    )
    for key, value in overrides.items():
        obj = layer
        *path, attr = key.split(".")
        for name in path:
            obj = getattr(obj, name)
        setattr(obj, attr, value)
    return layer


def test_configure_layer() -> None:
    layer = _layer_for_configure()
    gemma4_patch._configure_layer(layer)
    assert layer.self_attn.o_proj.reduce_results is False
    assert layer.mlp.down_proj.reduce_results is False
    assert layer.self_attn.qkv_proj.reduce_results is True
    assert layer.experts.moe_config.skip_final_all_reduce is True

    dense = _layer_for_configure(experts=None)
    gemma4_patch._configure_layer(dense)
    assert dense.mlp.down_proj.reduce_results is False


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"self_attn.is_kv_shared_layer": True}, "KV sharing"),
        ({"per_layer_input_gate": object()}, "per-layer embeddings"),
        ({"mlp.down_proj.bias": torch.zeros(1)}, "biases"),
        ({"self_attn.o_proj.tp_size": 1}, "TP > 1"),
        ({"experts.moe_config.dp_size": 2}, "DP > 1"),
        ({"experts.moe_config.is_sequence_parallel": True}, "sequence parallelism"),
    ],
)
def test_configure_layer_refuses(override: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        gemma4_patch._configure_layer(_layer_for_configure(**override))


def test_patched_model_init_refuses_pp_and_fast_prefill(monkeypatch) -> None:
    monkeypatch.setattr(gemma4_patch, "_orig_model_init", lambda self, *a, **k: None)
    monkeypatch.setattr(
        gemma4_patch, "get_pp_group", lambda: SimpleNamespace(world_size=1)
    )
    model = torch.nn.Module()
    model.fast_prefill_enabled = False
    gemma4_patch._patched_model_init(model)
    # Nothing is added to the model: both branches must trace the same inputs.
    assert list(model.state_dict()) == [] and list(model.buffers()) == []

    model.fast_prefill_enabled = True
    with pytest.raises(ValueError, match="kv_sharing_fast_prefill"):
        gemma4_patch._patched_model_init(model)
    monkeypatch.setattr(
        gemma4_patch, "get_pp_group", lambda: SimpleNamespace(world_size=2)
    )
    with pytest.raises(ValueError, match="PP"):
        gemma4_patch._patched_model_init(torch.nn.Module())


def test_maybe_patch_waits_for_gemma4(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "TPU_GEMMA4_SEQUENCE_PARALLEL", False)
    assert gemma4_patch.maybe_patch_gemma4_sequence_parallel(None) is None

    monkeypatch.setattr(envs, "TPU_GEMMA4_SEQUENCE_PARALLEL", True)
    assert gemma4_patch.maybe_patch_gemma4_sequence_parallel(None) is False
    other = SimpleNamespace(hf_text_config=SimpleNamespace(model_type="qwen3"))
    assert gemma4_patch.maybe_patch_gemma4_sequence_parallel(other) is False

    saved = {
        (cls, name): cls.__dict__[name]
        for cls in (Gemma4DecoderLayer, Gemma4Model)
        for name in ("__init__", "forward")
    }
    monkeypatch.setattr(gemma4_patch, "_microbatches", gemma4_patch._microbatches)
    cfg = SimpleNamespace(hf_text_config=SimpleNamespace(model_type="gemma4_text"))
    monkeypatch.setattr(envs, "TPU_PRECOMPILE_PEER_REPLAY", True)
    with pytest.raises(ValueError, match="TPU_PRECOMPILE_PEER_REPLAY"):
        gemma4_patch.maybe_patch_gemma4_sequence_parallel(cfg)
    monkeypatch.setattr(envs, "TPU_PRECOMPILE_PEER_REPLAY", False)
    monkeypatch.setattr(envs, "TPU_GEMMA4_SP_MICROBATCHES", 0)
    with pytest.raises(ValueError, match="TPU_GEMMA4_SP_MICROBATCHES"):
        gemma4_patch.maybe_patch_gemma4_sequence_parallel(cfg)
    monkeypatch.setattr(envs, "TPU_GEMMA4_SP_MICROBATCHES", 2)
    try:
        assert gemma4_patch.maybe_patch_gemma4_sequence_parallel(cfg) is None
        assert Gemma4DecoderLayer.__init__ is gemma4_patch._patched_layer_init
        assert Gemma4DecoderLayer.forward is gemma4_patch._patched_layer_forward
        assert Gemma4Model.__init__ is gemma4_patch._patched_model_init
        assert Gemma4Model.forward is gemma4_patch._patched_model_forward
        assert gemma4_patch._microbatches == 2
    finally:
        for (cls, name), value in saved.items():
            setattr(cls, name, value)
