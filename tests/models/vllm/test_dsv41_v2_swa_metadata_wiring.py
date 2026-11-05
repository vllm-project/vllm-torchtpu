# SPDX-License-Identifier: Apache-2.0
"""SWA metadata bridge, stage reuse, and per-layer address contracts."""

import functools
import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.compilation.shape_variants import (
    BucketExecutables,
    graph_signature,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention as v2_attn
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import preprocess_ops
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import PagedCacheRows

pytestmark = pytest.mark.cpu_test


def _inputs(ragged):
    counts = (
        np.tile(np.array([0, 1, 6, 3, 5, 7], np.int32), 8)
        if ragged
        else np.full(48, 6, np.int32)
    )
    starts = np.concatenate((np.zeros(1, np.int32), counts.cumsum(dtype=np.int32)))
    bases = np.tile(np.array([0, 1, 63, 64, 127, 128, 130, 195], np.int32), 6)
    positions = np.zeros(288, np.int32)
    for request, count in enumerate(counts):
        positions[starts[request] : starts[request + 1]] = bases[request] + np.arange(
            count, dtype=np.int32
        )
    minimum = np.maximum(bases - 4, 0)
    minimum[::4] = bases[::4] + 2
    return tuple(torch.from_numpy(x) for x in (starts, positions, minimum))


def _interpret_kernel(monkeypatch):
    kernel = preprocess_ops.swa_metadata_kernel.swa_attention_records
    mesh = jax.sharding.AbstractMesh(
        (1,),
        ("device",),
        abstract_device=jax.sharding.AbstractDevice(
            "TPU7x", num_cores=1, platform="tpu"
        ),
    )
    tpu = preprocess_ops.swa_metadata_kernel.pltpu
    with jax.sharding.use_abstract_mesh(mesh):
        target_info = tpu.get_tpu_info()
    monkeypatch.setattr(tpu, "get_tpu_info", lambda: target_info)

    def interpret(*args, **kwargs):
        return kernel(*args, **kwargs, interpret=True)

    monkeypatch.setattr(
        preprocess_ops.swa_metadata_kernel, "swa_attention_records", interpret
    )


def _make_layer(prefix, row_offset, ced):
    attn = object.__new__(v2_attn.DeepseekV41Attention)
    torch.nn.Module.__init__(attn)
    attn.window_size = 128
    attn.compress_ratio = 0
    attn._csa2 = attn._fused_preprocess = attn._fp8_kv = False
    attn._ced_decoder = ced
    attn._local_q_attn = False
    attn._swa_query_group_size = 6
    attn._swa_records_fusion = True
    attn.compressor = attn.indexer = None
    attn.n_local_heads, attn.head_dim, attn.scale = 2, 8, 0.25
    attn.attn_sink = torch.zeros(2)
    attn.wq_b = lambda value: value
    attn._rotate = lambda value, positions: value
    cache = MagicMock(spec=PagedCacheRows)
    cache.numel.return_value = 1024
    cache.shape, cache.page_stride, cache.row_offset = (64, 64, 512), 128, row_offset
    attn.swa_cache_layer = SimpleNamespace(prefix=prefix, kv_cache=cache)
    return attn


@pytest.mark.parametrize("ragged", [False, True])
@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("ced", [False, True])
def test_shared_metadata_reuses_computation_but_keeps_layer_addresses(
    monkeypatch, ragged, bounded, ced
):
    starts, positions, minimum = _inputs(ragged)
    table = torch.arange(48 * 8, dtype=torch.int32).reshape(48, 8)
    table[:3, 0] = -1
    tables = {"a": table, "b": table + 1000, "c": table, "d": table, "e": table}
    metadata = {
        prefix: SimpleNamespace(
            seq_lens=torch.ones(48, dtype=torch.int32),
            query_start_loc=starts,
            block_tables=blocks,
        )
        for prefix, blocks in tables.items()
    }
    additional = {"deepseek_v41_window_starts": minimum} if bounded else {}
    expected_minimum = minimum if bounded else torch.zeros_like(minimum)
    if ced:
        additional["deepseek_v41_replay_starts"] = expected_minimum
        # Decoder replay bounds take precedence over encoder window state.
        additional["deepseek_v41_window_starts"] = torch.full_like(minimum, 10000)
    ctx = SimpleNamespace(attn_metadata=metadata, additional_kwargs=additional)
    monkeypatch.setattr(v2_attn, "get_forward_context", lambda: ctx)
    monkeypatch.setattr(v2_attn, "get_tensor_model_parallel_world_size", lambda: 1)
    _interpret_kernel(monkeypatch)
    invocations = []

    def metadata_op(window, page, stride, offset):
        bridge = preprocess_ops._swa_records_bridge(window, page, stride, offset)

        def call(*arrays):
            invocations.append(tuple(a.shape for a in arrays))
            torch.testing.assert_close(arrays[2], expected_minimum, rtol=0, atol=0)
            result = bridge(*(jnp.asarray(a.numpy()) for a in arrays))
            return tuple(torch.from_numpy(np.array(value)) for value in result)

        return call

    monkeypatch.setattr(preprocess_ops, "swa_records", metadata_op)
    captured, inserted = [], []

    def attend(q, swa, main, records, main_records, sink, scale, query_group_size):
        assert query_group_size == 6
        captured.append(records.clone())
        return torch.zeros_like(q)

    monkeypatch.setattr(v2_attn, "paged_attention", attend)
    monkeypatch.setattr(
        v2_attn,
        "_insert",
        lambda cache, slots, values, valid: inserted.append(
            (slots.clone(), valid.clone())
        ),
    )
    layer_a, layer_b = _make_layer("a", 1024, ced), _make_layer("b", 1024, ced)
    layer_c, layer_d = _make_layer("c", 4096, ced), _make_layer("d", 1024, ced)
    layer_e = _make_layer("e", 1024, ced)
    layer_e.swa_cache_layer.kv_cache.page_stride = 192
    qr, kv = (
        torch.ones(288, 16, dtype=torch.bfloat16),
        torch.ones(288, 8, dtype=torch.bfloat16),
    )
    out = torch.zeros(288, 2, 8, dtype=torch.bfloat16)

    def forward_stage():
        with v2_attn.stage_attention_cache_scope():
            for layer in (layer_a, layer_b, layer_a, layer_c, layer_d, layer_e):
                layer._prepare_tpu(None, qr, kv, None, None, None, positions, out)

    for layer in (layer_a, layer_b, layer_c, layer_d, layer_e):
        layer._swa_records_fusion = False
    forward_stage()
    expected_records, expected_inserts = captured[:], inserted[:]
    assert not invocations
    captured.clear()
    inserted.clear()
    for layer in (layer_a, layer_b, layer_c, layer_d, layer_e):
        layer._swa_records_fusion = True
    forward_stage()
    assert len(invocations) == 4
    for actual, expected in zip(captured, expected_records, strict=True):
        assert actual.dtype == torch.int32
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, check_dtype=False)
    for actual, expected in zip(inserted, expected_inserts, strict=True):
        assert actual[0].dtype == torch.int32 and actual[1].dtype == torch.bool
        for got, ref in zip(actual, expected, strict=True):
            torch.testing.assert_close(got, ref, rtol=0, atol=0, check_dtype=False)
    assert not torch.equal(captured[0], captured[1])
    assert torch.equal(captured[0], captured[2])
    assert torch.equal(captured[0], captured[4])
    assert not torch.equal(captured[0], captured[3])
    assert not torch.equal(captured[0], captured[5])
    assert not inserted[0][1][:6].all()  # Negative current-page sentinel.
    if ragged:
        assert not inserted[0][1][int(starts[-1]) :].any()
        assert (captured[0][int(starts[-1]) :] == -1).all()
    forward_stage()
    assert len(invocations) == 8
    monkeypatch.setattr(v2_attn, "get_tensor_model_parallel_world_size", lambda: 2)
    forward_stage()
    assert len(invocations) == 8


def test_metadata_bridge_fake_and_registration_contract(monkeypatch):
    _interpret_kernel(monkeypatch)
    bridge = preprocess_ops._swa_records_bridge(128, 64, 128, 1024)
    parameters = list(inspect.signature(bridge).parameters.values())
    assert len(parameters) == 4
    assert all(
        p.kind is p.POSITIONAL_OR_KEYWORD
        and p.default is p.empty
        and p.annotation is jax.Array
        for p in parameters
    )
    torch_inputs = (
        *_inputs(True),
        torch.arange(48 * 1544, dtype=torch.int32).reshape(48, 1544),
    )
    inputs = tuple(jnp.asarray(a.numpy()) for a in torch_inputs)
    actual = bridge(*inputs)
    abstract = jax.eval_shape(bridge, *inputs)
    expected = preprocess_ops.swa_metadata_kernel.swa_attention_records(
        *inputs, window_size=128, page_size=64, page_stride=128, row_offset=1024
    )
    fake = preprocess_ops._swa_records_fake(128)(*torch_inputs)
    assert len(actual) == len(expected) == len(fake) == 3
    for value, reference, shape, tensor in zip(
        actual, expected, abstract, fake, strict=True
    ):
        np.testing.assert_array_equal(value, reference)
        assert value.shape == shape.shape == tuple(tensor.shape)
        assert str(value.dtype) == str(tensor.dtype).removeprefix("torch.")
    registrations = []

    def register(name, function, **kwargs):
        registrations.append(name)
        assert len(kwargs["input_partition_specs"]) == 4
        return SimpleNamespace(register_fake=lambda fake: None)

    monkeypatch.setattr(preprocess_ops, "_ops", {})
    monkeypatch.setattr(preprocess_ops, "get_pallas_op_mesh", lambda: None)
    monkeypatch.setattr(preprocess_ops.pallas, "jax_op", register)
    first = preprocess_ops._register_swa_records(128, 64, 128, 1024)
    assert preprocess_ops._register_swa_records(128, 64, 128, 1024) == first
    assert first in preprocess_ops._ops
    assert registrations == ["pallas::dsv41_swa_records_w128_p64_s128_o1024"]


def test_metadata_bridge_accepts_oversized_window_starts(monkeypatch):
    _interpret_kernel(monkeypatch)
    starts, positions, minimum = _inputs(True)
    bridge = preprocess_ops._swa_records_bridge(128, 64, 128, 1024)
    table = torch.arange(48 * 1544, dtype=torch.int32).reshape(48, 1544)
    inputs = tuple(jnp.asarray(a.numpy()) for a in (starts, positions, minimum, table))
    expected = bridge(*inputs)
    oversized = jnp.concatenate((inputs[2], jnp.full(4, 100000, jnp.int32)))
    actual = bridge(inputs[0], inputs[1], oversized, inputs[3])
    for value, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(value, reference)


@pytest.mark.parametrize("fusion_enabled", [False, True])
@pytest.mark.parametrize("extra_window_slots", [0, 4])
@pytest.mark.parametrize("token_order", [(2048, 288), (288, 2048)])
@pytest.mark.parametrize("fused_preprocess", [False, True])
@pytest.mark.parametrize("layer_count", [1, 3])
def test_metadata_prepare_split_preserves_bucket_input_order(
    monkeypatch,
    fusion_enabled,
    extra_window_slots,
    token_order,
    fused_preprocess,
    layer_count,
):
    """Real prefill/decode preparation must hand off split bucket executables."""
    from torch.fx.passes.split_module import split_module

    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)

    def unexpected_accelerator_stream(*args, **kwargs):
        raise AssertionError("CPU tracing test must not initialize a TPU stream")

    monkeypatch.setattr(
        torch.accelerator, "current_stream", unexpected_accelerator_stream
    )

    def records_sentinel(qsl, positions, minimum, table, *, stride, offset):
        token_rows = torch.arange(positions.shape[0], dtype=torch.int32)
        requests = torch.bucketize(
            token_rows, qsl[1:], right=True, out_int32=True
        ).clamp(max=qsl.shape[0] - 2)
        logical = positions[:, None] - 127 + torch.arange(128)[None, :]
        safe = logical.clamp(0, table.shape[1] * 64 - 1)
        current = (positions // 64).clamp(max=table.shape[1] - 1)
        write_slots = (table[requests, current] * 64 + positions % 64).masked_fill(
            token_rows >= qsl[-1], -1
        )
        write_valid = write_slots >= 0
        blocks = table[requests[:, None], (safe // 64).int()]
        allowed = (
            write_valid[:, None] & (logical >= 0) & (logical >= minimum[requests, None])
        )
        records = (offset + blocks * stride + safe % 64).masked_fill(~allowed, -1)
        return write_slots.int(), write_valid, records.int()

    record_ops = {}
    for index in range(layer_count):
        offset = 1024 + index * 1024
        op = torch.library.custom_op(
            f"swa_metadata_test::prepare_records_o{offset}",
            functools.partial(records_sentinel, stride=128, offset=offset),
            mutates_args=(),
            schema="(Tensor qsl, Tensor positions, Tensor minimum, Tensor table) -> "
            "(Tensor, Tensor, Tensor)",
        )
        op.register_fake(preprocess_ops._swa_records_fake(128))
        record_ops[offset] = op
    monkeypatch.setattr(preprocess_ops, "swa_records", lambda w, p, s, o: record_ops[o])
    monkeypatch.setattr(v2_attn, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(v2_attn, "_active_stage_attn_cache", {})
    monkeypatch.setattr(v2_attn, "_insert", lambda *args: None)

    def attention_sentinel(q, records, main_records, sink):
        return q.clone()

    attend_op = torch.library.custom_op(
        "swa_metadata_test::attention",
        attention_sentinel,
        mutates_args=(),
        schema="(Tensor q, Tensor records, Tensor main_records, Tensor sink) -> Tensor",
    )
    attend_op.register_fake(lambda q, records, main_records, sink: torch.empty_like(q))

    def attend(q, swa, main, records, main_records, sink, scale, query_group_size):
        # Keep every real input, especially the zero-width main_records. A
        # Python stub ignoring it conceals the token-symbol first-use mismatch.
        return attend_op(q, records, main_records, sink)

    monkeypatch.setattr(v2_attn, "paged_attention", attend)

    def projection_sentinel(qr, weight, scale, positions, rope_cache):
        return qr[:, :1, None].expand(-1, 2, 512).clone()

    projection_op = torch.library.custom_op(
        "swa_metadata_test::fp8_projection_rope",
        projection_sentinel,
        mutates_args=(),
        schema="(Tensor qr, Tensor weight, Tensor scale, "
        "Tensor positions, Tensor rope_cache) -> Tensor",
    )
    projection_op.register_fake(
        lambda qr, weight, scale, positions, rope_cache: qr.new_empty(
            (qr.shape[0], 2, 512)
        )
    )
    qb_op = torch.library.custom_op(
        "swa_metadata_test::qb_rope",
        lambda qr, weight, scale, rows: qr[:, :1, None].expand(-1, 2, 512).clone(),
        mutates_args=(),
        schema="(Tensor qr, Tensor weight, Tensor scale, Tensor rows) -> Tensor",
    )
    qb_op.register_fake(preprocess_ops._qb_fake)
    monkeypatch.setattr(preprocess_ops, "qb_rope", lambda: qb_op)
    qa_op = torch.library.custom_op(
        "swa_metadata_test::qa_rope",
        lambda x, weight, scale, gamma_qr, gamma_kv, rows: (
            x[:, :1].expand(-1, gamma_qr.shape[0]).clone(),
            x[:, :1].expand(-1, gamma_kv.shape[0]).clone(),
        ),
        mutates_args=(),
        schema="(Tensor x, Tensor weight, Tensor scale, Tensor gamma_qr, "
        "Tensor gamma_kv, Tensor rows) -> (Tensor, Tensor)",
    )
    qa_op.register_fake(preprocess_ops._qa_fake(True))
    monkeypatch.setattr(preprocess_ops, "qa_rope", lambda *args: qa_op)
    rope_op = torch.library.custom_op(
        "swa_metadata_test::rope_rows",
        lambda cache, positions: cache[positions.long()],
        mutates_args=(),
        schema="(Tensor cache, Tensor positions) -> Tensor",
    )
    rope_op.register_fake(
        lambda cache, positions: cache.new_empty((positions.shape[0], cache.shape[1]))
    )
    monkeypatch.setattr(preprocess_ops, "rope_rows", lambda: rope_op)
    input_op = torch.library.custom_op(
        "swa_metadata_test::fp8_input_projection",
        lambda x, weight, scale: x[:, :1].expand(-1, weight.shape[1]).clone(),
        mutates_args=(),
        schema="(Tensor x, Tensor weight, Tensor scale) -> Tensor",
    )
    input_op.register_fake(
        lambda x, weight, scale: x.new_empty((x.shape[0], weight.shape[1]))
    )
    layers = []
    for index in range(layer_count):
        layer = _make_layer(f"swa{index}", 1024 + index * 1024, ced=True)
        layer._fused_preprocess = fused_preprocess
        layer._swa_records_fusion = fusion_enabled
        layer._fp8_kv = True
        layer._sp_sharded_input = False
        layer.padded_heads, layer.head_dim, layer.q_lora_rank = 2, 512, 16
        layer._q_projection_rope_op = projection_op
        layer._tpu_rope_cache = torch.ones(512, 64)
        layer.wq_b = SimpleNamespace(
            weight=torch.zeros(16, 1024).to(torch.float8_e4m3fn),
            weight_scale=torch.ones(1024),
        )
        layer.fused_wqa_wkv = SimpleNamespace(
            weight=torch.zeros(16, 528).to(torch.float8_e4m3fn),
            weight_scale=torch.ones(528),
        )
        layer.q_norm, layer.kv_norm = torch.nn.Identity(), torch.nn.Identity()
        layer.q_norm.weight = torch.ones(16, dtype=torch.bfloat16)
        layer.kv_norm.weight = torch.ones(512, dtype=torch.bfloat16)
        layer.kv_norm.variance_epsilon = 1e-6
        input_weight = layer.fused_wqa_wkv.weight
        input_scale = layer.fused_wqa_wkv.weight_scale
        layer._fused_wqa_wkv_gemm = (
            lambda x, weight=input_weight, scale=input_scale: input_op(x, weight, scale)
        )
        layer._prepare_and_attn_fn = layer._prepare_tpu
        layer._o_proj = lambda q, positions: q[:, 0, :16]
        cache = object.__new__(PagedCacheRows)
        cache.shape = (64, 64, 512)
        cache.page_stride, cache.row_offset = 128, 1024 + index * 1024
        layer.swa_cache_layer.kv_cache = cache
        layers.append(layer)
    ctx = None
    monkeypatch.setattr(v2_attn, "get_forward_context", lambda: ctx)
    graphs = []

    def capture(graph, example_inputs):
        split = split_module(graph, graph, lambda node: 0)
        child = dict(split.named_children())["submod_0"]
        originals = {node.name: node for node in graph.graph.nodes}
        for node in child.graph.nodes:
            if node.op == "placeholder":
                node.meta.update(originals[node.name].meta)
        graphs.append(child)
        return graph.forward

    def forward(positions, hidden):
        # Execute the real attention forward, including its zero-like output
        # allocation, FP8 preprocessing and shared stage metadata. Allocating
        # output with an explicit token shape would hide the symbol regression.
        with v2_attn.stage_attention_cache_scope():
            for layer in layers:
                hidden = layer(positions, hidden)
        return hidden

    for tokens in token_order:
        counts = torch.full((48,), tokens // 48, dtype=torch.int32)
        counts[: tokens % 48] += 1
        starts = torch.cat((torch.zeros(1, dtype=torch.int32), counts.cumsum(0).int()))
        minimum = torch.arange(48 + extra_window_slots, dtype=torch.int32)
        ctx = SimpleNamespace(
            attn_metadata={
                layer.swa_cache_layer.prefix: SimpleNamespace(
                    seq_lens=torch.ones(48, dtype=torch.int32),
                    query_start_loc=starts,
                    block_tables=torch.arange(48 * 1544, dtype=torch.int32).reshape(
                        48, 1544
                    ),
                )
                for layer in layers
            },
            additional_kwargs={"deepseek_v41_replay_starts": minimum},
        )
        torch._dynamo.reset()
        compiled = torch.compile(
            forward, backend=capture, fullgraph=True, dynamic=False
        )
        inputs = (
            torch.arange(tokens, dtype=torch.int32) % 6 + 128,
            torch.ones(tokens, 16, dtype=torch.bfloat16),
        )
        for tensor in inputs:
            torch._dynamo.mark_dynamic(tensor, 0)
        compiled(*inputs)
        compiled(*(tensor.clone() for tensor in inputs))
        record_calls = [
            node
            for node in graphs[-1].graph.nodes
            if node.op == "call_function"
            and str(node.target).startswith("swa_metadata_test.prepare_records_")
        ]
        expected_calls = layer_count if tokens == 288 and fusion_enabled else 0
        assert len(record_calls) == expected_calls
        if tokens == 288 and fusion_enabled:
            assert all(
                node.target is not torch.bucketize for node in graphs[-1].graph.nodes
            )

    assert len(graphs) == 2
    executables = BucketExecutables()
    key = "artifact_compile_range_288_288_subgraph_0"
    runnable = graphs[0].forward
    executables.put("test-cache", key, graphs[0], runnable)
    assert executables.get("test-cache", key, graphs[1]) is runnable
    assert graph_signature(graphs[0]) == graph_signature(graphs[1])


def test_metadata_cold_factory_inside_compiled_forward(monkeypatch):
    """Register after other preprocess ops exposed their cache to Dynamo."""
    # Torch 2.13 initializes SymbolicStreamState for every graph, including
    # CPU-only tests. Hide accelerators here so it cannot claim the real TPU.
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)

    def unexpected_accelerator_stream(*args, **kwargs):
        raise AssertionError("CPU registration test must not initialize a TPU stream")

    monkeypatch.setattr(
        torch.accelerator, "current_stream", unexpected_accelerator_stream
    )
    monkeypatch.setattr(preprocess_ops, "_ops", {})
    monkeypatch.setattr(preprocess_ops, "get_pallas_op_mesh", lambda: None)
    registrations, graphs = [], []

    def metadata_sentinel(qsl, positions, minimum, table):
        # Registration and tracing only; independent tests verify arithmetic.
        return (
            positions + qsl[0] + table[0, 0],
            positions >= minimum[0],
            positions[:, None].expand(-1, 128) + 1,
        )

    def register(name, function, **kwargs):
        registrations.append(name)
        if name == "pallas::dsv41_rope_rows":
            return torch.library.custom_op(
                name,
                lambda cache, pos: cache.index_select(0, pos.long()),
                mutates_args=(),
                schema="(Tensor cache, Tensor pos) -> Tensor",
            )
        # Keep the real PartitionSpec creation and custom-op/fake registration
        # contract; only replace the TPU implementation with a Torch sentinel.
        return torch.library.custom_op(
            name,
            metadata_sentinel,
            mutates_args=(),
            schema="(Tensor qsl, Tensor positions, Tensor minimum, Tensor table) -> "
            "(Tensor, Tensor, Tensor)",
        )

    monkeypatch.setattr(preprocess_ops.pallas, "jax_op", register)
    preprocess_ops.rope_rows()  # Model initialization registers existing ops.

    def first_stage(qsl, positions, minimum, table, cache):
        rows = preprocess_ops.rope_rows()(cache, positions)
        metadata = preprocess_ops.swa_records(128, 64, 128, 1024)(
            qsl, positions, minimum, table
        )
        return (*metadata, rows)

    def second_stage(qsl, positions, minimum, table, cache):
        rows = preprocess_ops.rope_rows()(cache, positions)
        metadata = preprocess_ops.swa_records(128, 64, 128, 2048)(
            qsl, positions, minimum, table
        )
        return (*metadata, rows + 1)

    def capture(graph, example_inputs):
        graphs.append(graph)
        return graph.forward

    assert registrations == ["pallas::dsv41_rope_rows"]
    starts, positions, minimum = _inputs(True)
    cache = torch.arange(512 * 4, dtype=torch.float32).reshape(512, 4)
    for stage, forward in enumerate((first_stage, second_stage)):
        columns = 1544 + stage
        table = torch.zeros((48, columns), dtype=torch.int32)
        compiled = torch.compile(
            forward, backend=capture, fullgraph=True, dynamic=False
        )
        for shift in (0, 9):
            inputs = (starts, positions + shift, minimum)
            actual = compiled(*inputs, table, cache)
            expected = (
                *metadata_sentinel(*inputs, table),
                cache.index_select(0, inputs[1].long()) + stage,
            )
            for value, wanted in zip(actual, expected, strict=True):
                torch.testing.assert_close(value, wanted, rtol=0, atol=0)
    assert registrations == [
        "pallas::dsv41_rope_rows",
        "pallas::dsv41_swa_records_w128_p64_s128_o1024",
        "pallas::dsv41_swa_records_w128_p64_s128_o2048",
    ]
    assert len(graphs) == 2
    for stage, graph in enumerate(graphs):
        calls = [
            str(node.target) for node in graph.graph.nodes if node.op == "call_function"
        ]
        assert "pallas.dsv41_rope_rows.default" in calls
        assert (
            f"pallas.dsv41_swa_records_w128_p64_s128_o{1024 * (stage + 1)}.default"
            in calls
        )
