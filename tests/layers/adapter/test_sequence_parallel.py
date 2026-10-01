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
"""Row-gated sequence parallelism on the CPU.

The decoder and model forwards run on two simulated TP ranks (threads
exchanging through fake collectives) against a one-device reference; the
MoE block goes through the real communicator.
"""

import functools
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import vllm.distributed as vllm_distributed

from vllm_torchtpu.distributed import tpu_communicator as comm
from vllm_torchtpu.layers.adapter import linear_common
from vllm_torchtpu.layers.adapter import sequence_parallel as sp
from vllm_torchtpu.models.vllm import qwen3_next_sp_patch as patch

pytestmark = pytest.mark.cpu_test

_TP, _H, _MIN_ROWS = 2, 8, 8


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("TPU_SEQUENCE_PARALLEL_PREFILL", "1")
    monkeypatch.setattr(sp, "MIN_ROWS", _MIN_ROWS)
    monkeypatch.setattr(sp, "_active", False)
    monkeypatch.setattr(sp, "_layers_built", 0)


def test_the_step_form_follows_the_row_count(enabled):
    assert sp.enabled()
    assert not sp.step_is_sequence_parallel(_MIN_ROWS - 1)
    assert sp.step_is_sequence_parallel(_MIN_ROWS)
    sp.set_active(True)
    assert sp.rows_are_sliced(True) and not sp.rows_are_replicated(True)
    sp.set_active(False)
    assert sp.rows_are_replicated(True) and not sp.rows_are_sliced(True)
    # A layer not built sequence parallel takes neither branch.
    assert not sp.rows_are_sliced(False) and not sp.rows_are_replicated(False)


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("TPU_SEQUENCE_PARALLEL_PREFILL", raising=False)
    assert not sp.enabled()
    assert not sp.step_is_sequence_parallel(1 << 20)


@pytest.mark.parametrize("flag", ["0", "1"])
def test_only_the_layers_that_implement_it_see_sequence_parallel_moe(monkeypatch, flag):
    from vllm.config.parallel import ParallelConfig

    import vllm_torchtpu

    monkeypatch.setenv("TPU_SEQUENCE_PARALLEL_PREFILL", flag)
    monkeypatch.setattr(ParallelConfig, "_tpu_no_sp_moe_patch", False, raising=False)
    monkeypatch.setattr(
        ParallelConfig,
        "use_sequence_parallel_moe",
        ParallelConfig.use_sequence_parallel_moe,
    )
    vllm_torchtpu._patch_disable_sequence_parallel_moe()
    tp8 = ParallelConfig(tensor_parallel_size=8, enable_expert_parallel=True)
    others = [
        ParallelConfig(data_parallel_size=8, enable_expert_parallel=True),
        ParallelConfig(tensor_parallel_size=8, enable_expert_parallel=False),
        ParallelConfig(
            tensor_parallel_size=4, data_parallel_size=2, enable_expert_parallel=True
        ),
    ]
    assert not tp8.use_sequence_parallel_moe
    with sp.building_layer():
        assert tp8.use_sequence_parallel_moe is (flag == "1")
        assert not any(config.use_sequence_parallel_moe for config in others)
    assert not tp8.use_sequence_parallel_moe


def test_uneven_buckets_are_refused_at_boot(enabled):
    sp.check_buckets([16, 32, 48, 64], _TP)
    with pytest.raises(NotImplementedError, match=r"buckets \[9\]"):
        sp.check_buckets([4, 9, 16], _TP)


def test_a_model_without_row_gated_layers_is_refused(enabled):
    with pytest.raises(NotImplementedError, match="TPU_SEQUENCE_PARALLEL_PREFILL"):
        sp.check_installed(8)
    sp.record_layer()
    sp.check_installed(8)


def test_the_runner_records_the_step_form_before_the_model(enabled):
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    seen = []
    runner = SimpleNamespace(model=lambda **kw: seen.append(sp.is_active()) or kw)
    for rows in (_MIN_ROWS - 1, _MIN_ROWS):
        TPUModelRunner.forward_model(runner, torch.zeros(rows), torch.arange(rows))
    assert seen == [False, True]


class _FakeTP:
    """TP collectives between threads, one thread per rank."""

    def __init__(self, world_size):
        self.world_size = world_size
        self.rank_in_group = None
        self._slots = [None] * world_size
        self._barrier = threading.Barrier(world_size)
        self._local = threading.local()

    @property
    def rank(self):
        return self._local.rank

    def _exchange(self, x):
        self._slots[self.rank] = x
        self._barrier.wait()
        out = list(self._slots)
        self._barrier.wait()
        return out

    def all_gather(self, x, dim=0):
        return torch.cat(self._exchange(x), dim=dim)

    def reduce_scatter(self, x, dim=0):
        return sum(self._exchange(x)).chunk(self.world_size, dim=dim)[self.rank]

    def all_reduce(self, x):
        return sum(self._exchange(x))

    def run(self, fn):
        results, errors = [None] * self.world_size, []

        def worker(rank):
            self._local.rank = rank
            try:
                results[rank] = fn(rank)
            except BaseException as e:  # noqa: BLE001
                errors.append(e)
                self._barrier.abort()

        threads = [
            threading.Thread(target=worker, args=(r,)) for r in range(self.world_size)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]
        return results


class _Norm(torch.nn.Module):
    def forward(self, x, residual=None):
        y = x if residual is None else x + residual
        out = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-6)
        return out if residual is None else (out, y)


class _Attention(torch.nn.Module):
    """Mixes rows (a running sum), so it needs the step's rows whole; `w`
    is this rank's row-parallel share of the output projection."""

    def __init__(self, w):
        super().__init__()
        self.w = w

    def forward(self, hidden_states, positions=None, prequant=None):
        if prequant is not None:
            x_q, x_s = prequant
            assert hidden_states is x_q
            hidden_states = x_q * x_s
        return torch.cumsum(hidden_states, 0) @ self.w


class _Moe(torch.nn.Module):
    """Experts split over the ranks, exchanged by the real communicator."""

    def __init__(self, scale, communicator):
        super().__init__()
        self.scale = scale
        self.communicator = communicator

    def forward(self, hidden_states, already_sequence_parallel=False):
        if self.communicator is None:
            return hidden_states * self.scale
        x, _ = self.communicator.dispatch_router_logits(
            hidden_states, hidden_states[:, :1], is_sequence_parallel=True
        )
        return self.communicator.combine(x * self.scale, is_sequence_parallel=True)


def _layer(w, scale, communicator, rank, layer_type="linear_attention"):
    layer = torch.nn.Module()
    layer.layer_type = layer_type
    layer.layer_scale = False
    layer.input_layernorm = _Norm()
    layer.post_attention_layernorm = _Norm()
    if layer_type == "linear_attention":
        layer.linear_attn = _Attention(w)
    else:
        layer.self_attn = _Attention(w)
    layer.mlp = _Moe(scale, communicator)
    setattr(layer, patch.SP_LAYER_ATTR, communicator is not None)
    setattr(layer, patch.SP_RANK_ATTR, torch.tensor([rank], dtype=torch.int32))
    return layer


def _reference_layer(layer, hidden_states, residual, positions):
    if residual is None:
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
    else:
        hidden_states, residual = layer.input_layernorm(hidden_states, residual)
    if layer.layer_type == "linear_attention":
        hidden_states = layer.linear_attn(hidden_states)
    else:
        hidden_states = layer.self_attn(hidden_states)
    hidden_states, residual = layer.post_attention_layernorm(hidden_states, residual)
    return layer.mlp(hidden_states), residual


class _Model(torch.nn.Module):
    """vLLM 0.29's model forward with the chunk and gather it skips once the
    layers clear `use_attn_reduce_scatter_for_moe`."""

    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.start_layer = 0
        self.norm = _Norm()

    def forward(
        self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None
    ):
        hidden_states, residual = inputs_embeds, None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual, positions)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


@pytest.fixture
def tp(monkeypatch, enabled):
    """Two TP ranks whose TP and EP groups are the fake."""
    group = _FakeTP(_TP)
    for name, fn in dict(
        tensor_model_parallel_all_gather=group.all_gather,
        tensor_model_parallel_reduce_scatter=group.reduce_scatter,
        tensor_model_parallel_all_reduce=group.all_reduce,
        get_tensor_model_parallel_world_size=lambda: _TP,
    ).items():
        monkeypatch.setattr(vllm_distributed, name, fn)
    monkeypatch.setattr(comm, "get_ep_group", lambda: group)

    def init(instance, *args, **kwargs):
        instance.use_all2all = False

    monkeypatch.setattr(comm.DeviceCommunicatorBase, "__init__", init)
    forward = patch._row_gated_forward(lambda *a, **k: pytest.fail("not row-gated"))
    return SimpleNamespace(group=group, forward=forward)


def _shares(total, rank):
    """Rank `rank`'s share of `total` over _TP ranks: unequal, summing back."""
    weights = torch.linspace(0.25, 1.75, _TP)
    return total * weights[rank] / weights.sum()


def _weights(num_layers, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [
        (torch.randn(_H, _H, generator=g) / _H, torch.randn(_H, generator=g))
        for _ in range(num_layers)
    ]


def _bind(module, forward):
    module.forward = forward.__get__(module)
    return module


@pytest.mark.parametrize(
    "rows", [4, _MIN_ROWS, 2 * _MIN_ROWS], ids=lambda r: f"{r}rows"
)
@pytest.mark.parametrize("fp8_gather", [False, True], ids=["bf16", "fp8"])
def test_the_target_model_matches_one_device(tp, monkeypatch, rows, fp8_gather):
    # A stand-in quantization whose scale the projections must apply.
    monkeypatch.setattr(
        linear_common,
        "quantize_rows",
        lambda x: (x * 2, torch.full((x.shape[0], 1), 0.5)),
    )
    forward = patch._row_gated_forward(None, fp8_gather)
    weights = _weights(3)
    layer_types = ["linear_attention", "linear_attention", "full_attention"]
    embeds = torch.randn(rows, _H, generator=torch.Generator().manual_seed(1))
    positions = torch.arange(rows)

    reference = _Model(
        [_layer(w, s, None, 0, t) for (w, s), t in zip(weights, layer_types)]
    )
    for layer in reference.layers:
        _bind(layer, _reference_layer)
    expected = reference(None, positions, inputs_embeds=embeds)

    model_forward = patch._gather_model_rows(_Model.forward)
    sp.set_active(sp.step_is_sequence_parallel(rows))

    def rank_forward(rank):
        communicator = comm.TpuDeviceCommunicator()
        layers = [
            _bind(
                _layer(_shares(w, rank), _shares(s, rank), communicator, rank, t),
                forward,
            )
            for (w, s), t in zip(weights, layer_types)
        ]
        model = _Model(layers)
        return model_forward(model, None, positions, inputs_embeds=embeds)

    for out in tp.group.run(rank_forward):
        assert out.shape == (rows, _H)
        torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("fp8_gather", [False, True], ids=["bf16", "fp8"])
def test_both_forms_lift_the_same_inputs(enabled, monkeypatch, fp8_gather):
    """A bucket compiled by one form's trace is handed to the other's
    (compilation/shape_variants): their graphs must take the same inputs."""
    from vllm_torchtpu.compilation import shape_variants

    identity = dict(
        tensor_model_parallel_all_gather=lambda x, dim=0: x,
        tensor_model_parallel_reduce_scatter=lambda x, dim=0: x,
        tensor_model_parallel_all_reduce=lambda x: x,
        get_tensor_model_parallel_world_size=lambda: 1,
    )
    for name, fn in identity.items():
        monkeypatch.setattr(vllm_distributed, name, fn)
    monkeypatch.setattr(
        linear_common, "quantize_rows", lambda x: (x, x.new_ones(x.shape[0], 1))
    )
    forward = patch._row_gated_forward(None, fp8_gather)
    layers = [
        _bind(_layer(w, s, None, 0, t), forward)
        for (w, s), t in zip(_weights(2), ["linear_attention", "full_attention"])
    ]
    for layer in layers:
        setattr(layer, patch.SP_LAYER_ATTR, True)
    model = _Model(layers)
    traces = []

    def backend(graph, example_inputs):
        traces.append(shape_variants._handover_inputs(graph))
        return graph.forward

    torch._dynamo.reset()
    compiled = torch.compile(
        lambda x, p: model(None, p, inputs_embeds=x), backend=backend, fullgraph=True
    )
    for rows in (4, 2 * _MIN_ROWS):
        x, p = torch.randn(rows, _H), torch.arange(rows)
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(p, 0)
        compiled(x, p)
    torch._dynamo.reset()
    assert len(traces) == 2
    below, above = traces
    assert any(patch.SP_RANK_ATTR in name for name, _ in below)
    assert sorted(below) == sorted(above)


def test_install_patches_the_vllm_classes_once(enabled, monkeypatch):
    import inspect

    from vllm.model_executor.models import (
        qwen3_5,
        qwen3_next,
    )

    layer_cls = qwen3_next.Qwen3NextDecoderLayer
    targets = [
        (layer_cls, "__init__"),
        (layer_cls, "forward"),
        (qwen3_5.Qwen3_5DecoderLayer, "__init__"),
        (qwen3_next.Qwen3NextModel, "forward"),
    ]
    originals = [vars(cls)[name] for cls, name in targets]
    for (cls, name), original in zip(targets, originals):
        monkeypatch.setattr(cls, name, original)
    monkeypatch.setattr(layer_cls, "_tpu_sp_by_rows_patch", False, raising=False)
    patch.install()
    patched = [vars(cls)[name] for cls, name in targets]
    for original, new in zip(originals, patched):
        assert new is not original
        # support_torch_compile binds the forward's arguments by name.
        assert inspect.signature(new) == inspect.signature(original)
    assert qwen3_5.Qwen3_5DecoderLayer.forward is layer_cls.forward
    patch.install()
    assert [vars(cls)[name] for cls, name in targets] == patched


def test_install_is_a_no_op_when_not_selected(monkeypatch):
    from vllm.model_executor.models import qwen3_next

    monkeypatch.delenv("TPU_SEQUENCE_PARALLEL_PREFILL", raising=False)
    forward = qwen3_next.Qwen3NextDecoderLayer.forward
    patch.install()
    assert qwen3_next.Qwen3NextDecoderLayer.forward is forward


@pytest.mark.parametrize("tp_size,dp_size", [(8, 1), (1, 8)])
def test_a_layer_is_built_for_the_row_gated_form(
    enabled, monkeypatch, tp_size, dp_size
):
    """vLLM's layer reads `use_sequence_parallel_moe` while it is built; the
    wrapper then takes the form over from it."""
    monkeypatch.setattr(vllm_distributed, "get_tensor_model_parallel_rank", lambda: 3)
    monkeypatch.setattr(
        vllm_distributed, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    parallel_config = SimpleNamespace(
        tensor_parallel_size=tp_size,
        enable_expert_parallel=True,
        data_parallel_size=dp_size,
        prefill_context_parallel_size=1,
        pipeline_parallel_size=1,
    )
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        compilation_config=SimpleNamespace(compile_sizes=[16, 512, 1024]),
    )

    class Layer(torch.nn.Module):
        def __init__(self, vllm_config, layer_type, prefix=""):
            super().__init__()
            self.use_attn_reduce_scatter_for_moe = sp.use_sequence_parallel_moe(
                vllm_config.parallel_config
            )
            self.input_layernorm = torch.nn.Linear(1, 1)

    Layer.__init__ = patch._build_row_gated(Layer.__init__)
    layer = Layer(vllm_config, "full_attention", prefix="model.layers.0")
    row_gated = tp_size > 1
    assert getattr(layer, patch.SP_LAYER_ATTR) is row_gated
    assert layer.use_attn_reduce_scatter_for_moe is False
    assert sp._layers_built == int(row_gated)
    if row_gated:
        assert getattr(layer, patch.SP_RANK_ATTR).tolist() == [3]
        vllm_config.compilation_config.compile_sizes = [16, 516]
        with pytest.raises(NotImplementedError, match="516"):
            Layer(vllm_config, "full_attention")
    else:
        assert not hasattr(layer, patch.SP_RANK_ATTR)


def test_the_fp8_gather_needs_sequence_parallelism(monkeypatch):
    monkeypatch.delenv("TPU_SEQUENCE_PARALLEL_PREFILL", raising=False)
    monkeypatch.setenv("TPU_SEQUENCE_PARALLEL_FP8_GATHER", "1")
    assert not sp.fp8_gather()
    with pytest.raises(ValueError, match="TPU_SEQUENCE_PARALLEL_PREFILL"):
        patch.install()


def test_prequantized_rows_give_the_quantizing_matmul_exactly():
    """The fp8 gather changes no fp8 projection's result: the rows and scales
    each rank quantizes are the ones the projection would compute."""
    import jax.numpy as jnp
    import numpy as np

    g = torch.Generator().manual_seed(3)
    x = jnp.asarray(torch.randn(16, 256, generator=g).numpy(), jnp.bfloat16)
    w_q = jnp.asarray(
        (0.05 * torch.randn(256, 96, generator=g)).numpy(), jnp.float8_e4m3fn
    )
    w_s = jnp.asarray((0.02 + 0.01 * torch.rand(96, generator=g)).numpy())
    x_q, x_s = linear_common._quantize_rows_jax(x)
    assert x_q.dtype == jnp.float8_e4m3fn and x_s.shape == (16, 1)
    np.testing.assert_array_equal(
        np.asarray(linear_common._prequant_matmul_jax(x_q, x_s, w_q, w_s), np.float32),
        np.asarray(linear_common._quantized_matmul_jax(x, w_q, w_s), np.float32),
    )


def _fp8_linear(n_in, n_out, keep_vllm_layout=False, bias=None):
    from vllm_torchtpu.layers.adapter.quantization.fp8 import (
        VllmFp8Config,
        VllmFp8LinearMethodTPU,
    )

    shape = (n_out, n_in) if keep_vllm_layout else (n_in, n_out)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.zeros(shape, dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(torch.ones(n_out), requires_grad=False)
    layer.bias, layer.skip_bias_add = bias, False
    layer.quant_method = VllmFp8LinearMethodTPU(
        VllmFp8Config.from_config({"activation_scheme": "dynamic"})
    )
    if keep_vllm_layout:
        setattr(layer, linear_common.KEEP_VLLM_LAYOUT_ATTR, True)
    return layer


@pytest.mark.parametrize("keep_vllm_layout", [False, True])
def test_a_per_channel_fp8_projection_takes_the_gathered_rows(
    monkeypatch, keep_vllm_layout
):
    calls = []
    monkeypatch.setattr(
        linear_common,
        "quantized_matmul_prequant",
        lambda x_q, x_s, w, s: calls.append((x_q, x_s, w, s)) or torch.ones(4, 6),
    )
    layer = _fp8_linear(8, 6, keep_vllm_layout, bias=torch.full((6,), 2.0))
    x_q, x_s = torch.zeros(4, 8, dtype=torch.float8_e4m3fn), torch.ones(4, 1)
    out = sp.project(layer, x_q, (x_q, x_s))
    torch.testing.assert_close(out, torch.full((4, 6), 3.0))
    ((got_q, got_s, w, s),) = calls
    assert got_q is x_q and got_s is x_s and s is layer.weight_scale
    assert w.shape == (8, 6)
    assert w.data_ptr() == layer.weight.data_ptr()


def test_another_projection_is_given_the_rows_dequantized():
    """The checkpoint leaves the GDN's b/a projection in bf16."""
    layer = Mock(return_value=(torch.ones(4, 2), None), spec=["quant_method"])
    x_q = torch.full((4, 8), 2.0).to(torch.float8_e4m3fn)
    x_s = torch.full((4, 1), 0.25)
    out = sp.project(layer, x_q, (x_q, x_s))
    torch.testing.assert_close(out, torch.ones(4, 2))
    (rows,), _ = layer.call_args
    assert rows.dtype == torch.bfloat16
    torch.testing.assert_close(rows, torch.full((4, 8), 0.5, dtype=torch.bfloat16))
    # Without gathered rows the projection runs as before.
    x = torch.randn(4, 8)
    sp.project(layer, x, None)
    assert layer.call_args.args == (x,)


def test_the_gdn_in_projections_take_the_gathered_rows(monkeypatch):
    from vllm_torchtpu.layers.adapter.custom_ops import gdn_attention_op

    seen = []

    def project(layer, x, prequant):
        seen.append((layer, x, prequant))
        return torch.zeros(x.shape[0], layer.out)

    monkeypatch.setattr(gdn_attention_op, "project", project)
    attn = gdn_attention_op.VllmGatedDeltaNetAttention.__new__(
        gdn_attention_op.VllmGatedDeltaNetAttention
    )
    torch.nn.Module.__init__(attn)
    attn.in_proj_qkv, attn.in_proj_z, attn.in_proj_ba = (
        SimpleNamespace(out=n) for n in (96, 64, 32)
    )
    attn.head_v_dim, attn.num_v_heads, attn.tp_size = 16, 4, 1
    attn.kv_cache = None
    attn._has_active_pooled_pcp_state = lambda kv_cache: False
    attn._apply_output_projection = lambda core, z, n: core
    x = torch.zeros(4, 64, dtype=torch.float8_e4m3fn)
    prequant = (x, torch.ones(4, 1))
    attn.forward(x, prequant=prequant)
    assert {id(layer) for layer, _, _ in seen} == {
        id(attn.in_proj_qkv),
        id(attn.in_proj_z),
        id(attn.in_proj_ba),
    }
    assert all(got_x is x and got is prequant for _, got_x, got in seen)


def test_the_attention_projects_the_gathered_rows(monkeypatch):
    original = Mock(return_value="upstream")
    forward = patch._attention_forward(original)
    monkeypatch.setattr(sp, "project", lambda layer, x, prequant: prequant[0] + 1)
    attn = SimpleNamespace(
        qkv_proj=object(),
        _project_qkv_gate=lambda qkv, positions: (qkv, qkv, qkv, None),
        attn=lambda q, k, v: q + k + v,
        o_proj=lambda x: (x * 2, None),
    )
    positions, x = torch.arange(4), torch.ones(4, 8)
    assert forward(attn, positions, x) == "upstream"
    original.assert_called_once_with(attn, positions, x)
    torch.testing.assert_close(
        forward(attn, positions, x, prequant=(x, None)), torch.full((4, 8), 12.0)
    )


@pytest.mark.parametrize("draft_tp", [8, 1], ids=["sharded-draft", "replicated-draft"])
def test_the_mtp_draft_is_built_outside_it(enabled, monkeypatch, draft_tp):
    """The draft ("eagle_head" graph) compiles every bucket from one trace,
    which cannot carry both forms, so it runs vLLM's plain TP form. A
    replicated draft (draft_tensor_parallel_size=1) is built under a TP group
    collapsed to one rank while the parallel config still says TP8."""
    from vllm.compilation import backends

    monkeypatch.setattr(vllm_distributed, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        vllm_distributed, "get_tensor_model_parallel_world_size", lambda: draft_tp
    )
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=8,
            enable_expert_parallel=True,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        compilation_config=SimpleNamespace(compile_sizes=[16, 512]),
    )

    class Layer(torch.nn.Module):
        def __init__(self, vllm_config, layer_type, prefix=""):
            super().__init__()
            self.use_attn_reduce_scatter_for_moe = sp.use_sequence_parallel_moe(
                vllm_config.parallel_config
            )
            self.input_layernorm = torch.nn.Linear(1, 1)

    Layer.__init__ = patch._build_row_gated(Layer.__init__)
    with backends.set_model_tag("eagle_head"):
        layer = Layer(vllm_config, "full_attention", prefix="mtp.layers.0")
    assert getattr(layer, patch.SP_LAYER_ATTR) is False
    assert layer.use_attn_reduce_scatter_for_moe is False
    assert sp._layers_built == 0
    if draft_tp > 1:
        # The same layer in the backbone takes the row-gated form.
        backbone = Layer(vllm_config, "full_attention", prefix="model.layers.0")
        assert getattr(backbone, patch.SP_LAYER_ATTR) is True


def test_moe_chunk_pipelining_is_refused(enabled, monkeypatch):
    """Chunk pipelining exchanges over the DP group: it does not serve a step
    whose rows are replicated one step and sliced the next."""
    from vllm.model_executor.models import qwen3_next

    # Were it not refused, stop before patching vLLM for the rest of the session.
    monkeypatch.setattr(
        qwen3_next.Qwen3NextDecoderLayer, "_tpu_sp_by_rows_patch", True, raising=False
    )
    monkeypatch.setenv("TPU_MOE_COLLECTION_CHUNK_SIZE", "4096")
    with pytest.raises(NotImplementedError, match="TPU_MOE_COLLECTION_CHUNK_SIZE"):
        patch.install()


@pytest.mark.parametrize("impl", ["adaptive", "v3"])
def test_both_fused_ep_kernels_that_serve_it_are_accepted(enabled, monkeypatch, impl):
    """The adaptive kernel and fused EP v3 each serve sliced steps and take the
    grouped path on replicated ones, so either may sit behind the selector."""
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_IMPL", impl)
    sp.check_flags()


@pytest.mark.parametrize("row_gated", [False, True], ids=["draft", "backbone"])
def test_only_a_row_gated_layer_leaves_a_row_count_guard(
    enabled, monkeypatch, row_gated
):
    """The draft graph is traced once for every bucket, so a layer built
    outside the row-gated form must leave no guard on the row count: one
    trace serves both sides of MIN_ROWS. The backbone's layer is the control."""
    for name, fn in dict(
        tensor_model_parallel_all_gather=lambda x, dim=0: x,
        tensor_model_parallel_reduce_scatter=lambda x, dim=0: x,
        tensor_model_parallel_all_reduce=lambda x: x,
        get_tensor_model_parallel_world_size=lambda: 1,
    ).items():
        monkeypatch.setattr(vllm_distributed, name, fn)
    ((w, s),) = _weights(1)
    forward = patch._row_gated_forward(_reference_layer)
    layer = _bind(_layer(w, s, None, 0, "full_attention"), forward)
    setattr(layer, patch.SP_LAYER_ATTR, row_gated)
    traces = []

    def backend(graph, example_inputs):
        traces.append(graph)
        return graph.forward

    torch._dynamo.reset()
    compiled = torch.compile(
        lambda x, p: layer(x, None, p), backend=backend, fullgraph=True
    )
    for rows in (4, 2 * _MIN_ROWS):
        x, p = torch.randn(rows, _H), torch.arange(rows)
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(p, 0)
        compiled(x, p)
    torch._dynamo.reset()
    assert len(traces) == (2 if row_gated else 1)


_E, _TOPK = 4, 2


def _moe(x, gating, experts, padded):
    """Softmax top-k over `experts` (None: not this rank's), zero on padded
    rows."""
    weights, ids = torch.softmax(gating, -1).topk(_TOPK, -1)
    weights = weights / weights.sum(-1, keepdim=True)
    out = torch.zeros(x.shape[0], _H)
    for k in range(_TOPK):
        for e, w in enumerate(experts):
            rows = ids[:, k] == e
            if w is not None:
                out[rows] += weights[rows, k, None] * (x[rows] @ w)
    return out.masked_fill(padded.bool()[:, None], 0.0)


def _adaptive_kernel(group, experts):
    """The adaptive kernel's contract on two fake ranks: the members of a
    replica group hold identical rows and each routes one contiguous slice of
    them, every rank computes its own experts' share of every rank's routed
    rows, and the group gathers its rows back (token_parallel.TokenLayout)."""

    def kernel(x, *args, rank, token_replica_groups, padded, **kwargs):
        me = int(rank[0, 0])
        (members,) = [g for g in token_replica_groups if me in g]
        n, width = x.shape[0], len(members)
        per = -(-n // width)
        fill = per * width - n
        x = torch.nn.functional.pad(torch.from_numpy(x).float(), (0, 0, 0, fill))
        gating = torch.from_numpy(args[-1])
        gating = torch.nn.functional.pad(gating, (0, 0, 0, fill))
        mask = torch.from_numpy(padded) if padded is not None else torch.zeros(n)
        mask = torch.nn.functional.pad(mask, (0, fill), value=1)
        mine = slice(members.index(me) * per, (members.index(me) + 1) * per)
        routed = group._exchange((x[mine], gating[mine], mask[mine]))
        local = [w if e * _TP // _E == me else None for e, w in enumerate(experts)]
        shares = group._exchange([_moe(*r, local, m) for *r, m in routed])
        out = [sum(share[r] for share in shares) for r in range(_TP)]
        return torch.cat([out[m] for m in members])[:n].numpy()

    return kernel


@pytest.fixture
def adaptive(monkeypatch, enabled):
    """The adaptive bridge prebuilt for a sequence-parallel layer with the
    padded-row skip armed, over `_adaptive_kernel`."""
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.kernels.experimental import adaptive_fused_moe as package
    from vllm_torchtpu.kernels.experimental.adaptive_fused_moe import (
        vllm_adapter as bridge,
    )

    group = _FakeTP(_TP)
    experts = list(torch.randn(_E, _H, _H, generator=torch.Generator().manual_seed(2)))
    monkeypatch.setattr(package, "adaptive_fused_moe", _adaptive_kernel(group, experts))
    monkeypatch.setattr(bridge, "_OPS", {})

    def jax_op(name, fn, **kwargs):
        def op(*args):
            # The weights only pass through: the kernel holds its experts.
            return torch.from_numpy(fn(*(a.float().numpy() for a in args)))

        op.register_fake = lambda fake: None
        return op

    replicas = {True: ((0,), (1,)), False: ((0, 1),)}
    mesh = Mock(shape={bridge.EP_AXIS_NAME: _TP})
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    for name, value in dict(
        jax_op=jax_op,
        ep_token_replica_groups=lambda is_sequence_parallel: replicas[
            is_sequence_parallel
        ],
        build_ep_mesh=lambda: mesh,
        ep_mesh_index=lambda: 0,
        ep_rank_order=lambda: None,
        _max_node_tokens=lambda *_: 4096,
        _device_info=lambda: info,
        _in_backbone_graph=lambda: True,
    ).items():
        monkeypatch.setattr(bridge, name, value)
    monkeypatch.setattr(bridge.envs, "TPU_MOE_SKIP_PADDED_TOKENS", True)
    monkeypatch.setattr(bridge.envs, "MOE_FUSED_EP_KERNEL_MIN_TOKENS", 1)
    fp8 = torch.float8_e4m3fn
    layer = SimpleNamespace(
        w13_weight=torch.empty((_E // _TP, 512, 512), dtype=fp8),
        w2_weight=torch.empty((_E // _TP, 256, 512), dtype=fp8),
        w13_weight_scale_inv=torch.empty((_E // _TP, 1, 1, 512)),
        w2_weight_scale_inv=torch.empty((_E // _TP, 1, 1, 512)),
        global_num_experts=_E,
        custom_routing_function=None,
        scoring_func="softmax",
        e_score_correction_bias=None,
        use_grouped_topk=False,
        num_expert_group=None,
        topk_group=None,
        routed_scaling_factor=1.0,
        moe_config=SimpleNamespace(
            moe_parallel_config=SimpleNamespace(
                use_ep=True, pcp_size=1, is_sequence_parallel=True
            )
        ),
    )
    op = bridge.prebuild_adaptive_fused_moe(
        layer, topk=_TOPK, renormalize=True, activation="silu"
    )
    # A replicated step's grouped path is exchanged by the real communicator.
    monkeypatch.setattr(comm, "get_ep_group", lambda: group)
    monkeypatch.setattr(
        comm.DeviceCommunicatorBase,
        "__init__",
        lambda instance, *a, **k: setattr(instance, "use_all2all", False),
    )
    import vllm.distributed.parallel_state as parallel_state

    communicator = comm.TpuDeviceCommunicator()
    monkeypatch.setattr(parallel_state, "get_ep_group", lambda: communicator)
    owner = SimpleNamespace(**{bridge.FUSED_MOE_EP_OP_ATTR: op})
    return SimpleNamespace(
        bridge=bridge, owner=owner, group=group, experts=experts, layer=layer
    )


def _rank_layer(layer, rank):
    """`layer` as rank `rank` loads it: its first expert in `experts_start`."""
    ns = SimpleNamespace(**vars(layer))
    ns._experts_start = torch.tensor(rank * (_E // _TP), dtype=torch.int32)
    return ns


def _grouped(experts):
    """The grouped path's contract: this rank's experts over the rows it is
    given, padded rows (this rank's mask) routed nowhere."""
    from vllm_torchtpu.layers.adapter import token_padding

    def grouped(layer, x, router_logits):
        first = int(layer._experts_start)
        local = [
            w if first <= e < first + _E // _TP else None for e, w in enumerate(experts)
        ]
        mask = token_padding.local_padding_mask(x.shape[0])
        return _moe(x, router_logits, local, mask)

    return grouped


def _adaptive_forward(adaptive, x, gating, sliced):
    """Each fake rank's MoE output through `apply_adaptive_fused_moe`: its
    slice of the rows on a sliced step, the rows whole on a replicated one."""
    grouped = _grouped(adaptive.experts)

    def rank_forward(rank):
        rows = x.chunk(_TP)[rank] if sliced else x
        logits = gating.chunk(_TP)[rank] if sliced else gating
        return adaptive.bridge.apply_adaptive_fused_moe(
            adaptive.owner, _rank_layer(adaptive.layer, rank), rows, logits, grouped
        )

    return adaptive.group.run(rank_forward)


@pytest.mark.parametrize(
    "rows,valid",
    [(2 * _MIN_ROWS, 2 * _MIN_ROWS - 5), (2 * _MIN_ROWS, 3), (_MIN_ROWS - 2, 4)],
    ids=["sliced-tail-pad", "sliced-rank1-all-pad", "replicated"],
)
@pytest.mark.parametrize("wrong_form", [False, True], ids=["", "positive-control"])
def test_adaptive_on_a_sequence_parallel_layer_matches_one_device(
    adaptive, monkeypatch, rows, valid, wrong_form
):
    """A sequence-parallel layer's MoE on two fake ranks against a one-device
    reference: a sliced step runs the adaptive op (singleton replica groups,
    the rank's slice of the step's padding mask), a replicated step the
    grouped path between the communicator's dispatch and combine. Taking the
    other form (positive control) must fail the comparison."""
    from vllm_torchtpu.layers.adapter import token_padding

    g = torch.Generator().manual_seed(3)
    x = torch.randn(rows, _H, generator=g)
    gating = torch.randn(rows, _E, generator=g)
    state = token_padding.TokenPaddingState.create(4 * _MIN_ROWS, device="cpu")
    state.update(valid, rows)
    monkeypatch.setattr(token_padding, "_padding_state", state)
    sliced = sp.step_is_sequence_parallel(rows)
    sp.set_active(sliced != wrong_form)
    expected = _moe(x, gating, adaptive.experts, state.local_padding_mask[:rows])

    outs = _adaptive_forward(adaptive, x, gating, sliced)
    actual = torch.cat(outs) if sliced else outs
    if wrong_form:
        if sliced:
            assert not torch.allclose(actual, expected, atol=1e-5)
        else:
            # The adaptive op on replicated rows computes them whole, but its
            # mask is the rank slice of rows it does not hold.
            assert not all(torch.allclose(o, expected, atol=1e-5) for o in outs)
        return
    if sliced:
        torch.testing.assert_close(actual, expected)
    else:
        for out in outs:
            torch.testing.assert_close(out, expected)


@torch.library.custom_op("sp_adaptive_test::fused", mutates_args=())
def _fused_stub(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    gating: torch.Tensor,
    rank: torch.Tensor,
    padded: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(x)


@_fused_stub.register_fake
def _(x, *args):
    return torch.empty_like(x)


def test_both_adaptive_forms_lift_the_same_inputs(adaptive, monkeypatch):
    """The decoder forks on the row count, and a bucket compiled by one side's
    trace is handed to the other's (compilation/shape_variants): the sliced
    step's adaptive op and the replicated step's grouped path (the FP8
    method's own, routing stubbed) must lift the same inputs."""
    from vllm_torchtpu.compilation import shape_variants
    from vllm_torchtpu.layers.adapter import moe_routing, token_padding
    from vllm_torchtpu.layers.adapter.quantization import fp8

    bridge = adaptive.bridge
    monkeypatch.setattr(bridge, "_OPS", {})
    monkeypatch.setattr(bridge, "jax_op", lambda *a, **k: _fused_stub)
    layer = _rank_layer(adaptive.layer, 1)
    layer.moe_config.experts_per_token = _TOPK
    layer.renormalize = True
    owner = SimpleNamespace(
        **{
            bridge.FUSED_MOE_EP_OP_ATTR: bridge.prebuild_adaptive_fused_moe(
                layer, topk=_TOPK, renormalize=True, activation="silu"
            )
        },
        _tpu_activation_str="silu",
    )
    monkeypatch.setattr(
        moe_routing,
        "route",
        lambda layer, x, logits: torch.softmax(logits.float(), -1).topk(_TOPK, -1),
    )
    monkeypatch.setattr(
        fp8,
        "fused_moe_gmm",
        lambda **kw: _gmm_stub(
            kw["hidden_states"],
            kw["w1"],
            kw["w2"],
            kw["w1_scale"],
            kw["w2_scale"],
            kw["topk_weights"],
            kw["topk_ids"],
            kw["experts_start"],
        ),
    )
    monkeypatch.setattr(fp8, "enable_pipelined_collective_and_compute", lambda: False)
    monkeypatch.setattr(fp8.envs, "TPU_MOE_SKIP_PADDED_TOKENS", True)
    comm_ = SimpleNamespace(
        dispatch_router_logits=lambda x, g, sp_: (x, g), combine=lambda x, sp_: x
    )
    import vllm.distributed.parallel_state as parallel_state

    monkeypatch.setattr(parallel_state, "get_ep_group", lambda: comm_)
    dp = SimpleNamespace(world_size=1)
    monkeypatch.setattr(parallel_state, "get_dp_group", lambda: dp)
    state = token_padding.TokenPaddingState.create(4 * _MIN_ROWS, device="cpu")
    monkeypatch.setattr(token_padding, "_padding_state", state)
    for name in ("w13_weight", "w2_weight"):
        setattr(layer, name, torch.zeros(getattr(layer, name).shape))
    grouped = functools.partial(fp8.VllmFp8MoEMethodTPU._apply_grouped, owner)
    traces = []

    def backend(graph, example_inputs):
        traces.append(shape_variants._handover_inputs(graph))
        return graph.forward

    torch._dynamo.reset()
    compiled = torch.compile(
        lambda x, g: bridge.apply_adaptive_fused_moe(owner, layer, x, g, grouped),
        backend=backend,
        fullgraph=True,
    )
    for rows in (4, 2 * _MIN_ROWS):
        sp.set_active(sp.step_is_sequence_parallel(rows))
        x, g = torch.randn(rows, _H), torch.randn(rows, _E)
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(g, 0)
        compiled(x, g)
    torch._dynamo.reset()
    assert len(traces) == 2
    replicated, sliced = traces
    assert any("experts_start" in name for name, _ in sliced)
    assert any("padding_mask" in name for name, _ in replicated)
    assert sorted(replicated) == sorted(sliced)


@torch.library.custom_op("sp_adaptive_test::gmm", mutates_args=())
def _gmm_stub(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    experts_start: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(x)


@_gmm_stub.register_fake
def _(x, *args):
    return torch.empty_like(x)
