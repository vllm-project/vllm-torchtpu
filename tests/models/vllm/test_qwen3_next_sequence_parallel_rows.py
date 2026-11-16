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
"""Equivalence of the row-gated and fp8-gather sequence-parallel forwards.

Runs the patched Qwen3.5 MoE model and decoder-layer forwards on a toy model
with one thread per TP rank and in-process collectives, and compares every
rank's output with a single-device reference that never shards anything.
"""

import threading
from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.models import qwen3_next

import vllm_torchtpu.envs as envs
from vllm_torchtpu.distributed import sequence_parallel_rows
from vllm_torchtpu.distributed import tpu_communicator as comm
from vllm_torchtpu.layers.core.quantization import dequantize_tensor, quantize_tensor
from vllm_torchtpu.models.vllm import qwen3_next_patch as patch_mod

HIDDEN = 16
INTERMEDIATE = 16
NUM_EXPERTS = 4
NUM_LAYERS = 2
VOCAB = 64
MIN_ROWS = 16


class ThreadGroup:
    """Lockstep collectives over one thread per rank, summed in rank order."""

    def __init__(self, world_size: int):
        self.world_size = world_size
        self._slots: list[torch.Tensor | None] = [None] * world_size
        self._barrier = threading.Barrier(world_size, timeout=60)
        self._local = threading.local()
        self.calls: list[str] = []

    @property
    def rank(self) -> int:
        return self._local.rank

    def bind(self, rank: int) -> None:
        self._local.rank = rank

    def _exchange(self, name: str, x: torch.Tensor) -> list[torch.Tensor]:
        if self.rank == 0:
            self.calls.append(name)
        self._slots[self.rank] = x
        self._barrier.wait()
        slots = list(self._slots)
        self._barrier.wait()
        return slots  # type: ignore[return-value]

    def all_gather(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        return torch.cat(self._exchange("all_gather", x), dim=dim)

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        slots = self._exchange("all_reduce", x)
        return torch.stack(slots).sum(0)

    def reduce_scatter(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        slots = self._exchange("reduce_scatter", x)
        return torch.stack(slots).sum(0).chunk(self.world_size, dim=dim)[self.rank]


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


def fp8_projection_input(x: torch.Tensor) -> torch.Tensor:
    """The activation an fp8 W8A8 projection multiplies: per-row fp8 rounded."""
    x_q, scale = quantize_tensor(x, quant_dtype=torch.float8_e4m3fn, axis=-1)
    return dequantize_tensor(x_q, scale, axis=-1, out_dtype=x.dtype)


def make_weights(seed: int = 0) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, generator=gen) / shape[0] ** 0.5

    return {
        "embed": torch.randn(VOCAB, HIDDEN, generator=gen),
        "attn_in": rand(NUM_LAYERS, HIDDEN, INTERMEDIATE),
        "attn_out": rand(NUM_LAYERS, INTERMEDIATE, HIDDEN),
        "router": rand(NUM_LAYERS, HIDDEN, NUM_EXPERTS),
        "expert_in": rand(NUM_LAYERS, NUM_EXPERTS, HIDDEN, INTERMEDIATE),
        "expert_out": rand(NUM_LAYERS, NUM_EXPERTS, INTERMEDIATE, HIDDEN),
        "shared_in": rand(NUM_LAYERS, HIDDEN, INTERMEDIATE),
        "shared_out": rand(NUM_LAYERS, INTERMEDIATE, HIDDEN),
    }


def attention(w, layer_idx, x, columns, quantized_input):
    """Column-parallel in-projection, a token-mixing core, row-parallel out.

    Returns this rank's partial sum; the cumulative sum over tokens makes the
    result depend on every row being present and in order.
    """
    if quantized_input:
        x = fp8_projection_input(x)
    core = torch.tanh(x @ w["attn_in"][layer_idx][:, columns]).cumsum(0) / 4
    return core @ w["attn_out"][layer_idx][columns]


def routed_experts(w, layer_idx, x, logits, experts):
    """Top-1 routed experts, summed over the given experts only."""
    probs = logits.softmax(-1)
    top1 = probs.argmax(-1)
    out = torch.zeros_like(x)
    for e in experts:
        h = torch.tanh(x @ w["expert_in"][layer_idx][e]) @ w["expert_out"][layer_idx][e]
        out += (top1 == e).unsqueeze(-1) * probs[:, e : e + 1] * h
    return out


def shared_expert(w, layer_idx, x):
    return torch.tanh(x @ w["shared_in"][layer_idx]) @ w["shared_out"][layer_idx]


def reference_forward(w, input_ids, quantized_input=False):
    """Single device, no sharding: what every rank must reproduce."""
    hidden = w["embed"][input_ids]
    residual = None
    for i in range(NUM_LAYERS):
        residual = hidden if residual is None else hidden + residual
        normed = rms_norm(residual)
        attn = attention(w, i, normed, slice(None), quantized_input)
        residual = attn + residual
        normed = rms_norm(residual)
        logits = normed @ w["router"][i]
        hidden = routed_experts(
            w, i, normed, logits, range(NUM_EXPERTS)
        ) + shared_expert(w, i, normed)
    return rms_norm(hidden + residual)


class ToyDecoderLayer:
    """Qwen3.5 MoE decoder layer shape: what the patched forward calls."""

    forward = patch_mod.patched_decoder_layer_forward
    layer_type = "linear_attention"
    layer_scale = False
    use_attn_reduce_scatter_for_moe = True

    def __init__(self, w, layer_idx, group, communicator, quantized_input):
        self.w = w
        self.layer_idx = layer_idx
        self.group = group
        self.communicator = communicator
        self.quantized_input = quantized_input

    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def input_layernorm(self, x, residual=None):
        if residual is None:
            return rms_norm(x)
        residual = x + residual
        return rms_norm(residual), residual

    def post_attention_layernorm(self, x, residual):
        residual = x + residual
        return rms_norm(residual), residual

    def linear_attn(self, hidden_states):
        tp, rank = self.group.world_size, self.group.rank
        width = INTERMEDIATE // tp
        columns = slice(rank * width, (rank + 1) * width)
        return attention(
            self.w, self.layer_idx, hidden_states, columns, self.quantized_input
        )

    def mlp(self, x, already_sequence_parallel=False):
        # Upstream's runner order under sequence-parallel MoE: router and
        # shared expert on the rows this rank holds, dispatch, this rank's
        # experts, combine; no final all-reduce.
        assert already_sequence_parallel
        logits = x @ self.w["router"][self.layer_idx]
        shared = shared_expert(self.w, self.layer_idx, x)
        hidden, logits = self.communicator.dispatch_router_logits(
            x, logits, is_sequence_parallel=True
        )
        tp, rank = self.group.world_size, self.group.rank
        own = range(rank, NUM_EXPERTS, tp)
        routed = routed_experts(self.w, self.layer_idx, hidden, logits, own)
        return self.communicator.combine(routed, is_sequence_parallel=True) + shared


def toy_model(w, group, communicator, quantized_input):
    layers = [
        ToyDecoderLayer(w, i, group, communicator, quantized_input)
        for i in range(NUM_LAYERS)
    ]
    return SimpleNamespace(
        use_sequence_parallel=True,
        layers=layers,
        start_layer=0,
        end_layer=NUM_LAYERS,
        embed_input_ids=lambda ids: w["embed"][ids],
        norm=lambda x, residual: (rms_norm(x + residual), None),
        _maybe_add_hidden_state=lambda aux, *args: aux,
    )


@pytest.fixture
def run_sharded(monkeypatch):
    """Run the patched model forward on ``tp`` threads; return each rank's out."""

    def run(tp, input_ids, *, quantized_input=False):
        group = ThreadGroup(tp)
        w = make_weights()
        monkeypatch.setattr(
            patch_mod, "get_tensor_model_parallel_world_size", lambda: tp
        )
        monkeypatch.setattr(patch_mod, "get_tp_group", lambda: group)
        monkeypatch.setattr(
            patch_mod,
            "get_pp_group",
            lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
        )
        monkeypatch.setattr(
            patch_mod,
            "tensor_model_parallel_all_gather",
            lambda x, dim=0: group.all_gather(x, dim),
        )
        monkeypatch.setattr(
            patch_mod, "tensor_model_parallel_all_reduce", group.all_reduce
        )
        monkeypatch.setattr(
            patch_mod,
            "tensor_model_parallel_reduce_scatter",
            lambda x, dim=0: group.reduce_scatter(x, dim),
        )
        monkeypatch.setattr(
            qwen3_next,
            "sequence_parallel_chunk",
            lambda x: x.chunk(tp)[group.rank],
        )
        monkeypatch.setattr(comm, "get_ep_group", lambda: group)
        communicator = object.__new__(comm.TpuDeviceCommunicator)
        positions = torch.arange(input_ids.shape[0])
        outputs: list[torch.Tensor | None] = [None] * tp
        errors: list[BaseException] = []

        def rank_main(rank):
            group.bind(rank)
            model = toy_model(w, group, communicator, quantized_input)
            try:
                outputs[rank] = patch_mod.patched_model_forward(
                    model, input_ids, positions
                )
            except BaseException as e:  # surfaced in the main thread
                errors.append(e)
                group._barrier.abort()

        with sequence_parallel_rows.backbone_step(input_ids.shape[0]):
            threads = [threading.Thread(target=rank_main, args=(r,)) for r in range(tp)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        if errors:
            raise errors[0]
        return outputs, group.calls, w

    return run


def input_ids_for(num_tokens):
    return torch.randint(
        0, VOCAB, (num_tokens,), generator=torch.Generator().manual_seed(1)
    )


@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("min_rows", [0, MIN_ROWS])
@pytest.mark.parametrize("num_tokens", [8, MIN_ROWS, 32])
def test_gated_sequence_parallel_matches_unsharded(
    monkeypatch, run_sharded, tp, min_rows, num_tokens
):
    monkeypatch.setattr(envs, "TPU_SP_MIN_ROWS", min_rows)
    monkeypatch.setattr(envs, "TPU_SP_FP8_GATHER", False)
    input_ids = input_ids_for(num_tokens)
    outputs, calls, w = run_sharded(tp, input_ids)
    expected = reference_forward(w, input_ids)
    for out in outputs:
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    if min_rows and num_tokens < min_rows:
        # Replicated rows: one all-reduce after attention and one in the MoE
        # combine per layer, nothing gathered or scattered.
        assert calls == ["all_reduce"] * (2 * NUM_LAYERS)
    else:
        # Main's form: gather before attention, scatter after, gather and
        # scatter around the experts, gather the final rows.
        layer = ["all_gather", "reduce_scatter", "all_gather", "all_gather"]
        layer += ["reduce_scatter"]
        assert calls == layer * NUM_LAYERS + ["all_gather"]


@pytest.mark.parametrize("tp", [2, 4])
def test_fp8_gather_matches_fp8_projections_without_sharding(
    monkeypatch, run_sharded, tp
):
    """The fp8 gather only moves the rounding the fp8 projections do anyway."""
    monkeypatch.setattr(envs, "TPU_SP_MIN_ROWS", 0)
    monkeypatch.setattr(envs, "TPU_SP_FP8_GATHER", True)
    input_ids = input_ids_for(32)
    outputs, calls, w = run_sharded(tp, input_ids, quantized_input=True)
    expected = reference_forward(w, input_ids, quantized_input=True)
    for out in outputs:
        # Same fp8 values in the projection; only its per-row scale can differ,
        # by a bf16-rounding-sized amount.
        torch.testing.assert_close(out, expected, rtol=2e-3, atol=2e-3)
    # Every layer's gather before attention moves fp8 rows and their scales:
    # two all-gathers where main has one.
    layer = ["all_gather", "all_gather", "reduce_scatter", "all_gather", "all_gather"]
    layer += ["reduce_scatter"]
    assert calls == layer * NUM_LAYERS + ["all_gather"]


def test_fp8_gather_reaches_projection_with_same_fp8_values():
    """Re-quantizing the gathered rows reproduces the rows' own fp8 values."""
    x = torch.randn(64, 256, generator=torch.Generator().manual_seed(2)) * 3
    x = x.to(torch.bfloat16)
    group = SimpleNamespace(all_gather=lambda t, dim=0: t)
    from vllm_torchtpu.layers.adapter.pipelined_fused_moe import quantized_all_gather

    gathered = quantized_all_gather(group, x)
    assert gathered.dtype == x.dtype
    direct_q, direct_scale = quantize_tensor(x, torch.float8_e4m3fn, axis=-1)
    requant_q, requant_scale = quantize_tensor(gathered, torch.float8_e4m3fn, axis=-1)
    assert torch.equal(direct_q.view(torch.int8), requant_q.view(torch.int8))
    torch.testing.assert_close(requant_scale, direct_scale, rtol=2**-8, atol=0)


def test_drafter_layer_call_stays_sequence_parallel(monkeypatch):
    """A decoder layer called without the model's flag keeps main's form.

    The MTP drafter chunks its rows itself and calls the layer directly, so a
    step below TPU_SP_MIN_ROWS must not change what the layer does.
    """
    monkeypatch.setattr(envs, "TPU_SP_MIN_ROWS", 1024)
    calls = []
    monkeypatch.setattr(
        patch_mod,
        "tensor_model_parallel_all_gather",
        lambda x, dim=0: calls.append("all_gather") or torch.cat([x, x], dim),
    )
    monkeypatch.setattr(
        patch_mod,
        "tensor_model_parallel_reduce_scatter",
        lambda x, dim=0: calls.append("reduce_scatter") or x.chunk(2, dim)[0],
    )
    monkeypatch.setattr(patch_mod, "get_tensor_model_parallel_world_size", lambda: 2)
    layer = SimpleNamespace(
        use_attn_reduce_scatter_for_moe=True,
        layer_type="linear_attention",
        layer_scale=False,
        input_layernorm=lambda x, residual=None: x if residual is None else (x, x),
        linear_attn=lambda hidden_states: hidden_states,
        post_attention_layernorm=lambda x, residual: (x, residual),
        mlp=lambda x, already_sequence_parallel=False: x,
    )
    patch_mod.patched_decoder_layer_forward(
        layer, torch.ones(4, 8), None, torch.arange(8)
    )
    assert calls == ["all_gather", "reduce_scatter"]
