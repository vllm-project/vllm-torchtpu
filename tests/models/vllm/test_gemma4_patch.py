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

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.fx.passes.split_module import split_module
from vllm.model_executor.models import gemma4
from vllm.model_executor.models.gemma4 import (
    Gemma4Attention,
    Gemma4DecoderLayer,
    Gemma4ForCausalLM,
    Gemma4MLP,
    Gemma4Model,
)
from vllm.model_executor.models.interfaces import EagleModelMixin

from vllm_torchtpu import envs
from vllm_torchtpu.compilation import shape_variants
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU
from vllm_torchtpu.models.vllm import gemma4_patch

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


def _gelu_and_mul(gate_up: torch.Tensor) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    return F.gelu(gate, approximate="tanh") * up


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
            total_num_heads=self.heads,
            total_num_kv_heads=self.kv_heads,
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
            act_fn=_gelu_and_mul,
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
        # A spawned process of its own: the rebinding cannot leak into other tests.
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
    model.vllm_config = SimpleNamespace(speculative_config=None)
    gemma4_patch._patched_model_init(model)
    # Nothing is added to the model: both branches must trace the same inputs.
    assert list(model.state_dict()) == [] and list(model.buffers()) == []

    model.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(method="mtp")
    )
    gemma4_patch._patched_model_init(model)
    model.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(method="eagle3")
    )
    with pytest.raises(ValueError, match="eagle3"):
        gemma4_patch._patched_model_init(model)

    model.vllm_config = SimpleNamespace(speculative_config=None)
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


# ---------------------------------------------------------------------------
# Context parallelism
# ---------------------------------------------------------------------------


class _GqaModel(_Model):
    kv_heads = 4


def _zigzag_plan(world: int, num_tokens: int, page_size: int = 2) -> Any:
    return gemma4_patch.cp_prefill.make_zigzag_plan(num_tokens, world, page_size)


@pytest.mark.parametrize("world", [3, 4, 8])
def test_zigzag_rows_round_trip(world: int) -> None:
    plan = _zigzag_plan(world, 4 * world * 2)
    tokens = torch.arange(plan.num_tokens)
    rank_major = gemma4_patch._rank_major(tokens, plan)
    size = plan.piece_tokens
    for rank in range(world):
        rows = gemma4_patch._rank_rows(tokens, plan, rank)
        assert torch.equal(rank_major.view(world, -1)[rank], rows)
        first, second = plan.rank_pieces(rank)
        assert rows.tolist() == [
            *range(first * size, (first + 1) * size),
            *range(second * size, (second + 1) * size),
        ]
    assert torch.equal(gemma4_patch._token_order(rank_major, plan), tokens)


@pytest.fixture(autouse=True)
def _eager_vocab_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    """vLLM compiles get_masked_input_and_mask for its platform; the TPU
    platform installs an eager one, which this stands in for."""

    def masked(ids, start, end, padding, added_start, added_end):
        in_shard = (ids >= start) & (ids < end)
        in_added = (ids >= added_start) & (ids < added_end)
        offset = start * in_shard + (added_start - (end - start) - padding) * in_added
        keep = in_shard | in_added
        return keep * (ids - offset), ~keep

    monkeypatch.setattr(
        gemma4_patch.vocab_parallel_embedding, "get_masked_input_and_mask", masked
    )


def _rows(plan: Any, rank: int) -> torch.Tensor:
    return gemma4_patch._rank_rows(torch.arange(plan.num_tokens), plan, rank)


def _vocab_shard(embedding: torch.Tensor, rank: int, world: int) -> SimpleNamespace:
    """A VocabParallelEmbedding shard: rows [rank * V/world, (rank + 1) * V/world)."""
    vocab = embedding.shape[0]
    start, end = rank * vocab // world, (rank + 1) * vocab // world
    return SimpleNamespace(
        weight=embedding[start:end],
        shard_indices=SimpleNamespace(
            org_vocab_start_index=start,
            org_vocab_end_index=end,
            num_org_vocab_padding=0,
            added_vocab_start_index=vocab,
            added_vocab_end_index=vocab,
        ),
        quant_method=SimpleNamespace(embedding=lambda layer, ids: layer.weight[ids]),
    )


def test_vocab_shard_embedding_reads_registered_bounds() -> None:
    """Shard bounds come from the TPU platform's buffers where it registers them."""
    embedding = torch.randn(16, 4)
    shard = _vocab_shard(embedding, 1, 4)  # ints say rows [4, 8)
    ids = torch.tensor([0, 5, 9, 15])
    torch.testing.assert_close(
        gemma4_patch._vocab_shard_embedding(shard, ids),
        torch.stack(
            [embedding[0] * 0, embedding[5], embedding[0] * 0, embedding[0] * 0]
        ),
    )
    shard.org_vocab_start_index = torch.tensor(8)  # buffers say rows [8, 12)
    shard.org_vocab_end_index = torch.tensor(12)
    shard.weight = embedding[8:12]
    torch.testing.assert_close(
        gemma4_patch._vocab_shard_embedding(shard, ids),
        torch.stack(
            [embedding[0] * 0, embedding[0] * 0, embedding[9], embedding[0] * 0]
        ),
    )


class _AllExperts:
    """Every expert of _Experts' layout on one rank, nothing reduced."""

    def __init__(self, weights: list[torch.Tensor]) -> None:
        self.weights = weights

    def __call__(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor:
        gates = torch.softmax(router_logits, dim=-1)
        return sum(
            gates[:, e : e + 1] * (hidden_states @ w)
            for e, w in enumerate(self.weights)
        )


def _replica(model: _Model, index: int) -> SimpleNamespace:
    """The projections of one layer, whole, as Gemma4ProjectionReplica holds them."""
    lw = model.layer_weights[index]

    def linear(weight: torch.Tensor) -> _Linear:
        return _Linear(weight, 1, model.method)

    return SimpleNamespace(
        self_attn=SimpleNamespace(
            qkv_proj=linear(torch.cat([lw["q"], lw["k"], lw["v"]], dim=1)),
            o_proj=linear(lw["o"]),
        ),
        mlp=SimpleNamespace(
            gate_up_proj=linear(torch.cat([lw["gate"], lw["up"]], dim=1)),
            down_proj=linear(lw["down"]),
        ),
        experts=_AllExperts(lw["experts"]) if model.moe else None,
    )


def _cp_attention_reference(mock: _TPCollectiveMock, model: _Model, plan: Any):
    """_causal_mix over the whole prompt, for one rank's zigzag rows."""
    num_tokens = plan.num_tokens
    heads, kv_heads, hd = model.heads, model.kv_heads, model.head_dim

    def attention(attn, q, k, v, plan, rank):
        kv = gemma4_patch._token_order(mock.all_gather(k + v), plan)
        kv = kv.view(num_tokens, kv_heads, hd).repeat_interleave(heads // kv_heads, 1)
        prefix = kv.cumsum(0) / torch.arange(1, num_tokens + 1).view(-1, 1, 1)
        rows = _rows(plan, int(rank[0]))
        return (q.view(-1, heads, hd) * prefix[rows]).flatten(1)

    return attention


@pytest.mark.parametrize("world", [3, 4])
@pytest.mark.parametrize("fp8", [False, True])
def test_context_parallel_forward_matches_tp(mock_collectives, world, fp8) -> None:
    """Zigzag rows through whole projections == upstream's TP forward, by token."""
    num_layers, normalizer = 2, torch.tensor(2.0)
    mock = mock_collectives(world)
    model = _GqaModel(1, moe=False, fp8=fp8, num_layers=num_layers)
    plan = _zigzag_plan(world, 2 * world * 4)
    num_tokens = plan.num_tokens
    input_ids = torch.randint(0, model.embedding.shape[0], (num_tokens,))
    positions = torch.arange(num_tokens)

    def reference() -> torch.Tensor:
        x = model.embedding[input_ids] * normalizer
        for i in range(num_layers):
            layer = model.layer(i, 0, lambda t: t)
            x, _ = _UPSTREAM_LAYER_FORWARD(layer, positions, x, None)
        return model.final_norm(x)

    attention = _cp_attention_reference(mock, model, plan)

    def rank_fn(rank: int) -> torch.Tensor:
        fake = SimpleNamespace(
            embed_tokens=_vocab_shard(model.embedding, rank, world),
            normalizer=normalizer,
            layers=[model.layer(i, 0, lambda t: t) for i in range(num_layers)],
            cp_replica=SimpleNamespace(
                layers=[_replica(model, i) for i in range(num_layers)]
            ),
            norm=model.final_norm,
        )
        return gemma4_patch._context_parallel_forward(
            fake,
            input_ids,
            positions,
            plan,
            torch.tensor([rank]),
            _rows(plan, rank),
            attention,
        )

    expected = reference()
    for out in _run_ranks(world, rank_fn):
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
    # One embedding scatter of every row, then only the K/V gathers until the
    # final gather of every rank's rows.
    rows = num_tokens // world
    assert [c[2] for c in mock.calls] == [
        (num_tokens, model.hidden),
        *[(rows, model.kv_heads * model.head_dim)] * num_layers,
        (rows, model.hidden),
    ]


def test_context_parallel_moe_forward_matches_tp(mock_collectives) -> None:
    """MoE layers: dense MLP and every expert local == upstream's TP forward,
    whose ranks each hold one expert and all-reduce."""
    world, num_layers, normalizer = 4, 2, torch.tensor(2.0)
    mock = mock_collectives(world)
    model = _GqaModel(world, moe=True, fp8=False, num_layers=num_layers)
    plan = _zigzag_plan(world, 2 * world * 4)
    input_ids = torch.randint(0, model.embedding.shape[0], (plan.num_tokens,))
    positions = torch.arange(plan.num_tokens)

    def tp_rank(rank: int) -> torch.Tensor:
        x = model.embedding[input_ids] * normalizer
        for i in range(num_layers):
            layer = model.layer(i, rank, mock.all_reduce)
            x, _ = _UPSTREAM_LAYER_FORWARD(layer, positions, x, None)
        return model.final_norm(x)

    expected = _run_ranks(world, tp_rank)[0]
    attention = _cp_attention_reference(mock, model, plan)

    def cp_rank(rank: int) -> torch.Tensor:
        fake = SimpleNamespace(
            embed_tokens=_vocab_shard(model.embedding, rank, world),
            normalizer=normalizer,
            layers=[model.layer(i, rank, mock.all_reduce) for i in range(num_layers)],
            cp_replica=SimpleNamespace(
                layers=[_replica(model, i) for i in range(num_layers)]
            ),
            norm=model.final_norm,
        )
        return gemma4_patch._context_parallel_forward(
            fake,
            input_ids,
            positions,
            plan,
            torch.tensor([rank]),
            _rows(plan, rank),
            attention,
        )

    for out in _run_ranks(world, cp_rank):
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


def _trace_context_parallel_forward(monkeypatch, rank: int) -> Any:
    """Dynamo's graph of `_context_parallel_forward` on rank ``rank`` of 4, with
    collectives and attention that are the same on every rank and the vocab
    shard bounds as the TPU platform's buffers."""
    world = 4
    model = _GqaModel(1, moe=False, fp8=True, num_layers=2)
    plan = _zigzag_plan(world, 2 * world * 4)
    monkeypatch.setattr(
        gemma4_patch, "_reduce_scatter_rows", lambda x: x[: x.shape[0] // world] * 1
    )
    monkeypatch.setattr(gemma4_patch, "_all_gather_rows", lambda x: x.repeat(world, 1))
    embed = _vocab_shard(model.embedding, rank, world)
    for name in ("org_vocab_start_index", "org_vocab_end_index"):
        setattr(embed, name, torch.tensor(getattr(embed.shard_indices, name)))
    fake = SimpleNamespace(
        embed_tokens=embed,
        normalizer=torch.tensor(2.0),
        layers=[model.layer(i, 0, lambda t: t) for i in range(2)],
        cp_replica=SimpleNamespace(layers=[_replica(model, i) for i in range(2)]),
        norm=model.final_norm,
    )
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    def forward(ids, pos, rank_tensor, rows):
        return gemma4_patch._context_parallel_forward(
            fake, ids, pos, plan, rank_tensor, rows, lambda a, q, k, v, p, r: q * 1
        )

    ids = torch.randint(0, model.embedding.shape[0], (plan.num_tokens,))
    pos = torch.arange(plan.num_tokens)
    torch._dynamo.mark_dynamic(ids, 0)
    torch._dynamo.mark_dynamic(pos, 0)
    torch._dynamo.reset()
    out = torch.compile(forward, backend=backend, fullgraph=True)(
        ids, pos, torch.tensor([rank], dtype=torch.int32), _rows(plan, rank)
    )
    assert out.shape == (plan.num_tokens, model.hidden)
    return graphs[0], plan.num_tokens


def test_context_parallel_trace_keeps_the_bucket_symbolic(monkeypatch) -> None:
    """Only the token slices touch the token dimension: no specialization, and
    the trace refuses smaller buckets, as shape_variants reads its guards."""
    graph, num_tokens = _trace_context_parallel_forward(monkeypatch, 1)
    shape_env = shape_variants.trace_shape_env(graph)
    assert shape_env is not None and shape_env.var_to_range
    assert shape_variants.unsupported_reason(shape_env, num_tokens) is None
    assert shape_variants.unsupported_reason(shape_env, num_tokens // 2) is not None


def test_context_parallel_trace_is_the_same_on_every_rank(monkeypatch) -> None:
    """The rank reaches the program only as tensors (collectives compiled into
    the program assume every rank runs the same one)."""
    graphs = [_trace_context_parallel_forward(monkeypatch, r)[0] for r in (0, 1, 3)]
    assert graphs[0].code == graphs[1].code == graphs[2].code


def _cp_config(**overrides: Any) -> SimpleNamespace:
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            data_parallel_size=1,
            tensor_parallel_size=8,
            enable_ep_weight_filter=False,
            enable_eplb=False,
        ),
        lora_config=None,
        speculative_config=None,
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        compilation_config=SimpleNamespace(
            compile_sizes=[16, 256, 4096, 16384, 32768, 65536, 81920]
        ),
    )
    for key, value in overrides.items():
        obj = config
        *path, attr = key.split(".")
        for name in path:
            obj = getattr(obj, name)
        setattr(obj, attr, value)
    return config


def _cp_model(**overrides: Any) -> SimpleNamespace:
    layer = SimpleNamespace(
        enable_moe_block=False,
        per_layer_input_gate=None,
        self_attn=SimpleNamespace(
            is_kv_shared_layer=False, o_proj=SimpleNamespace(bias=None)
        ),
    )
    model = SimpleNamespace(fast_prefill_enabled=False, layers=[layer])
    for key, value in overrides.items():
        obj = model
        *path, attr = key.split(".")
        for name in path:
            obj = obj.layers[0] if name == "layer" else getattr(obj, name)
        setattr(obj, attr, value)
    return model


@pytest.fixture
def cp_envs(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(gemma4_patch, "_cp_min_tokens", 32768)
    monkeypatch.setattr(envs, "VLLM_TPU_SOLO_PREFILL_OVER_TOKENS", 16384)
    monkeypatch.setattr(envs, "TPU_PRECOMPILE_PEER_REPLAY", False)
    monkeypatch.setattr(envs, "VLLM_TPU_MOST_MODEL_LEN", None)
    return monkeypatch


@pytest.mark.parametrize(
    ("solo", "cp_min", "ok"),
    [
        (16384, 32768, True),
        (10000, 20000, True),  # bucket 16384 < 20000 <= 32768
        (16384, 16384, False),  # a regular step's bucket would run CP
        (16384, 65536, False),  # solo steps padded to 32768 would not
        (0, 32768, False),  # no solo-prefill scheduler
        (81920, 100000, False),  # nothing above the solo bucket
    ],
)
def test_check_context_parallel_bucket_rule(cp_envs, solo, cp_min, ok) -> None:
    cp_envs.setattr(envs, "VLLM_TPU_SOLO_PREFILL_OVER_TOKENS", solo)
    cp_envs.setattr(gemma4_patch, "_cp_min_tokens", cp_min)
    if ok:
        gemma4_patch._check_context_parallel(_cp_model(), _cp_config())
    else:
        with pytest.raises(ValueError, match="VLLM_TPU_SOLO_PREFILL_OVER_TOKENS"):
            gemma4_patch._check_context_parallel(_cp_model(), _cp_config())


@pytest.mark.parametrize(
    ("model", "config", "env", "match"),
    [
        ({}, {"parallel_config.pipeline_parallel_size": 2}, {}, "PP"),
        ({}, {"parallel_config.data_parallel_size": 2}, {}, "DP"),
        ({}, {"lora_config": object()}, {}, "LoRA"),
        ({}, {"speculative_config": object()}, {}, "speculative"),
        ({}, {"cache_config.enable_prefix_caching": True}, {}, "prefix caching"),
        ({"fast_prefill_enabled": True}, {}, {}, "kv_sharing_fast_prefill"),
        ({}, {}, {"TPU_PRECOMPILE_PEER_REPLAY": True}, "PEER_REPLAY"),
        ({}, {}, {"VLLM_TPU_MOST_MODEL_LEN": 4096}, "MOST_MODEL_LEN"),
        ({"layer.self_attn.is_kv_shared_layer": True}, {}, {}, "KV sharing"),
        ({"layer.per_layer_input_gate": object()}, {}, {}, "per-layer"),
        ({"layer.self_attn.o_proj.bias": object()}, {}, {}, "biases"),
        ({}, {"parallel_config.tensor_parallel_size": 2}, {}, "TP >= 3"),
        (
            {"layer.enable_moe_block": True},
            {"parallel_config.enable_ep_weight_filter": True},
            {},
            "EP weight filter",
        ),
        (
            {"layer.enable_moe_block": True},
            {"parallel_config.enable_eplb": True},
            {},
            "EPLB",
        ),
    ],
)
def test_check_context_parallel_refuses(cp_envs, model, config, env, match) -> None:
    for name, value in env.items():
        cp_envs.setattr(envs, name, value)
    with pytest.raises(ValueError, match=match):
        gemma4_patch._check_context_parallel(_cp_model(**model), _cp_config(**config))


def test_check_solo_prompt(monkeypatch) -> None:
    def with_layout(query_start_loc, seq_lens) -> None:
        context = SimpleNamespace(
            query_start_loc_cpu=query_start_loc, seq_lens_cpu=seq_lens
        )
        monkeypatch.setattr(
            gemma4_patch, "get_vllm_model_wrapper_context", lambda: context
        )
        gemma4_patch._check_solo_prompt(32768)

    with_layout(None, None)  # warmup
    with_layout(np.array([0, 20000]), np.array([20000]))
    with pytest.raises(RuntimeError, match="one prompt"):
        with_layout(np.array([0, 20000, 20001]), np.array([20000, 70]))
    with pytest.raises(RuntimeError, match="cached"):
        with_layout(np.array([0, 20000]), np.array([20512]))


def test_load_weights_tees_the_projections(monkeypatch) -> None:
    seen: list[str] = []
    monkeypatch.setattr(
        gemma4_patch,
        "_orig_model_load_weights",
        lambda self, weights: {name for name, _ in weights if not seen.append(name)},
    )
    names = [
        "embed_tokens.weight",
        "layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.k_proj.weight_scale",
        "layers.0.self_attn.k_norm.weight",
        "layers.0.self_attn.o_proj.weight",
        "layers.10.mlp.gate_proj.weight",
        "layers.10.mlp.down_proj.weight_scale",
        "layers.10.input_layernorm.weight",
        "layers.3.experts.7.gate_proj.weight",
        "layers.3.experts.7.down_proj.weight_scale",
        "layers.3.router.proj.weight",
        "layers.3.router.per_expert_scale",
        "norm.weight",
    ]
    weights = [(name, torch.zeros(1)) for name in names]
    model = torch.nn.Module()
    gemma4_patch._patched_model_load_weights(model, weights)
    assert seen == names  # no replica, nothing added
    seen.clear()
    model.cp_replica = torch.nn.Module()
    gemma4_patch._patched_model_load_weights(model, weights)
    expected = []
    for name in names:
        expected.append(name)
        if "_proj." in name and ".router." not in name:
            expected.append(f"cp_replica.{name}")
    assert seen == expected


def test_replica_layer_builds_upstreams_experts_unsharded(monkeypatch) -> None:
    """A MoE layer's replica: upstream's FusedMoEFactory arguments, tp=dp=1."""

    class _Recorded(torch.nn.Module):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__()
            self.args, self.kwargs = args, kwargs

    for name in (
        "QKVParallelLinear",
        "RowParallelLinear",
        "MergedColumnParallelLinear",
        "FusedMoEFactory",
    ):
        monkeypatch.setattr(gemma4_patch, name, _Recorded)
    router = SimpleNamespace(routing_function=object())
    layer = SimpleNamespace(
        hidden_size=32,
        enable_moe_block=True,
        router=router,
        self_attn=SimpleNamespace(
            hidden_size=32, head_dim=8, total_num_heads=8, total_num_kv_heads=4
        ),
        mlp=SimpleNamespace(down_proj=SimpleNamespace(input_size=64)),
    )
    config = SimpleNamespace(
        num_experts=16, top_k_experts=4, hidden_size=32, moe_intermediate_size=48
    )
    replica = gemma4_patch._ReplicaLayer(
        layer, config, "qc", "model.cp_replica.layers.3"
    )
    assert replica.experts.kwargs == dict(
        num_experts=16,
        top_k=4,
        hidden_size=32,
        intermediate_size=48,
        renormalize=True,
        quant_config="qc",
        tp_size=1,
        dp_size=1,
        prefix="model.cp_replica.layers.3.experts",
        custom_routing_function=router.routing_function,
        activation="gelu_tanh",
    )
    assert replica.self_attn.qkv_proj.kwargs["disable_tp"] is True
    assert replica.mlp.down_proj.args == (64, 32)
    layer.enable_moe_block = False
    assert not hasattr(gemma4_patch._ReplicaLayer(layer, config, "qc", "p"), "experts")


def test_maybe_patch_context_parallel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "TPU_GEMMA4_CP_MIN_TOKENS", 0)
    assert gemma4_patch.maybe_patch_gemma4_context_parallel(None) is None
    monkeypatch.setattr(envs, "TPU_GEMMA4_CP_MIN_TOKENS", 32768)
    assert gemma4_patch.maybe_patch_gemma4_context_parallel(None) is False
    text = SimpleNamespace(model_type="gemma4_text")
    multimodal = SimpleNamespace(
        hf_text_config=text, architecture="Gemma4ForConditionalGeneration"
    )
    with pytest.raises(ValueError, match="Gemma4ForCausalLM"):
        gemma4_patch.maybe_patch_gemma4_context_parallel(multimodal)

    saved = {
        (cls, name): cls.__dict__[name]
        for cls, name in (
            (Gemma4ForCausalLM, "__init__"),
            (Gemma4ForCausalLM, "forward"),
            (Gemma4Model, "load_weights"),
        )
    }
    monkeypatch.setattr(gemma4_patch, "_cp_min_tokens", 0)
    causal_lm = SimpleNamespace(hf_text_config=text, architecture="Gemma4ForCausalLM")
    try:
        assert gemma4_patch.maybe_patch_gemma4_context_parallel(causal_lm) is None
        assert gemma4_patch._cp_min_tokens == 32768
        assert Gemma4ForCausalLM.__init__ is gemma4_patch._patched_causal_lm_init
        assert Gemma4ForCausalLM.forward is gemma4_patch._patched_causal_lm_forward
        assert Gemma4Model.load_weights is gemma4_patch._patched_model_load_weights
    finally:
        for (cls, name), value in saved.items():
            setattr(cls, name, value)
