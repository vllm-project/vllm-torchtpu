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
"""Actual runner/adapter metadata through the native pooled QSA Torch bridge.

The runner fixture supplies scheduler rows, CPU staging owners and physical
page IDs (no checkpoint). _prepare_inputs, AttentionMetadataBuilder.build,
Qwen4ExpAdapter.prepare_inputs, the custom op and every kernel are real.
Projection outputs are deterministic BF16 fixtures, outside this test's seam.
"""

import math
from types import SimpleNamespace as NS

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from runner_input_test_utils import _BlockTable, _make_runner, _scheduler_output

from tests.models.vllm.qwen4_exp_test_utils import q4exp_text_config
from tests.runner.test_qwen4_exp_pool import pool_config
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadataBuilder
from vllm_torchtpu.models.vllm.qwen4_exp.inputs import Qwen4ExpAdapter
from vllm_torchtpu.models.vllm.qwen4_exp.pool_layout import pooled_cache_specs
from vllm_torchtpu.runner.model_adapters import ModelInputSnapshot
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def runner_inputs(
    monkeypatch, *, tokens=256, context=0, fp8=True, local_heads=24, kv_heads=2
):
    monkeypatch.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", "1")
    cfg = pool_config(tp=24 // local_heads, kv="fp8_e4m3" if fp8 else "bfloat16")
    assert (
        cfg.model_config.hf_text_config.num_key_value_heads
        // cfg.parallel_config.tensor_parallel_size
        == kv_heads
    )
    cfg.model_config.max_model_len = max(9216, context + tokens)
    layout, specs = pooled_cache_specs(cfg)
    page = layout.block_size
    runner = _make_runner(
        num_computed_tokens=[context],
        prompt_tokens=[context + tokens],
        scheduled_tokens=[tokens - 3],
        token_paddings=[tokens],
        pcp_size=1,
        block_size=page,
    )
    runner.model_config = NS(hf_text_config=q4exp_text_config())
    runner.cache_config.block_size = page
    columns = math.ceil((context + tokens) / page)
    # Compressed keys share main logical IDs; raw rings own separate IDs.
    tables = [
        torch.arange(1, columns + 1, dtype=torch.int32)[None],
        torch.tensor([[columns + 1]], dtype=torch.int32),
    ]
    runner.input_batch.block_table = [_BlockTable(t) for t in tables]
    names = ("model.layers.3.self_attn", "model.layers.3.self_attn.raw_ring")
    builders = [
        AttentionMetadataBuilder(
            specs[name], [name], cfg, torch.device("cpu"), runner, i
        )
        for i, name in enumerate(names)
    ]

    def build(**kwargs):
        common = NS(num_reqs=kwargs["num_reqs"])
        return {name: b.build(0, common) for name, b in zip(names, builders)}, None

    runner._build_attention_metadata = build
    prepared = TPUModelRunner._prepare_inputs(
        runner, _scheduler_output([tokens - 3]), 0, 0
    )
    runner.test_attention_metadata = prepared[0]
    adapter = Qwen4ExpAdapter(runner)
    kwargs = adapter.prepare_inputs(
        ModelInputSnapshot(
            adapter.begin_step(),
            0,
            1,
            tokens,
            prepared[0],
            runner.input_ids_cpu[:tokens],
        )
    )
    assert kwargs["query_start_loc"].tolist() == [0] + [tokens - 3] * 8
    assert kwargs["context_lens"].tolist() == [context] + [0] * 7
    assert kwargs["state_slots"].tolist() == [0] + [-1] * 7
    assert kwargs["valid_tokens"].tolist() == [True] * (tokens - 3) + [False] * 3
    return runner, kwargs, layout, columns


@pytest.mark.parametrize("fp8", [False, True])
def test_real_runner_preserves_padded_request_capacity(monkeypatch, fp8):
    _, kwargs, layout, _ = runner_inputs(monkeypatch, fp8=fp8)
    assert len(kwargs["query_start_loc"]) == 9
    assert layout.block_size == (1664 if fp8 else 896)


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "fp8,local_heads,kv_heads",
    [(False, 24, 2), (True, 24, 2), (False, 12, 1), (True, 12, 1)],
)
def test_runner_adapter_to_pooled_qsa_bridge_executes_prefill(
    monkeypatch, fp8, local_heads, kv_heads, record_property
):
    from vllm_torchtpu.kernels.qwen4_exp.pooled_qsa import compact_pooled_qsa_step
    from vllm_torchtpu.kernels.qwen4_exp.qsa import produce
    from vllm_torchtpu.kernels.sparse_gqa_prefill.cost_model import select_sc
    from vllm_torchtpu.layers.adapter.custom_ops.qsa_op import (
        build_compact_pooled_qsa_op,
    )

    tokens = 512
    context = 262144 - tokens if fp8 and local_heads == 24 else 37
    runner, md, layout, columns = runner_inputs(
        monkeypatch,
        tokens=tokens,
        context=context,
        fp8=fp8,
        local_heads=local_heads,
        kv_heads=kv_heads,
    )
    dtype, packing = (jnp.float8_e4m3fn, 4) if fp8 else (jnp.bfloat16, 2)
    rng = np.random.default_rng(2018)

    def rand(shape):
        return jnp.asarray(rng.normal(0, 0.3, shape), jnp.bfloat16)

    # Null page, separate main/compressed rows, plus one ring owner.
    ring_id = columns + 1
    pool = rand(
        (
            ring_id + 1,
            layout.block_size,
            math.ceil(2 * kv_heads / packing),
            packing,
            256,
        )
    ).astype(dtype)
    index_pool = rand(
        (ring_id + 1, layout.index_rows // 2, 1, 2, layout.compressed_width)
    )
    index_pool = index_pool.at[jnp.asarray([0, ring_id])].set(0)
    pool = pool.at[jnp.asarray([0, ring_id])].set(0)
    trig = jnp.zeros((context + tokens + 1, 128), jnp.bfloat16).at[:, :32].set(1)
    positions = jnp.asarray(runner.position_ids.numpy(), jnp.int32)
    producers = (
        rand((tokens, 2 * local_heads * 256)),
        rand((tokens, kv_heads * 256)),
        rand((tokens, kv_heads * 256)),
        rand((tokens, 5 * 128)),
        jnp.zeros(256, jnp.bfloat16),
        jnp.zeros(256, jnp.bfloat16),
        jnp.zeros(128, jnp.bfloat16),
        jnp.zeros(128, jnp.bfloat16),
        positions,
        trig,
    )
    producers = tuple(
        x.at[-3:].set(jnp.nan) if i < 4 else x for i, x in enumerate(producers)
    )
    starts, contexts = (
        jnp.asarray(md[n].numpy(), jnp.int32)
        for n in ("query_start_loc", "context_lens")
    )
    rings = jnp.asarray(
        runner.test_attention_metadata[
            "model.layers.3.self_attn.raw_ring"
        ].block_tables.numpy(),
        jnp.int32,
    )
    tables = tuple(
        jnp.asarray(md[n].numpy(), jnp.int32).reshape(-1)
        for n in ("main_block_table", "compressed_block_table")
    )
    args = (*producers, pool, index_pool, starts, contexts, rings, *tables)
    options = dict(compress_ratio=4, token_topk=2048, norm_eps=1e-6, rotary_dim=64)
    import functools

    compiled = (
        jax.jit(functools.partial(compact_pooled_qsa_step, **options))
        .lower(*args)
        .compile()
    )
    hlo = compiled.as_text()
    assert "prefill_paired_pages" in hlo
    # Small context is statically TC; long FP8 context retains the SC branch.
    assert ("sc_compact_grouped_prefill" in hlo) == (fp8 and local_heads == 24)

    # Actual serving bridge, both direct and fullgraph torch.compile.
    def tensor(x):
        if x.dtype in (jnp.bfloat16, jnp.float8_e4m3fn):
            target = torch.bfloat16 if x.dtype == jnp.bfloat16 else torch.float8_e4m3fn
            return (
                torch.from_numpy(np.asarray(x).copy().view(np.uint8))
                .to("tpu")
                .view(target)
                .reshape(x.shape)
            )
        return torch.from_numpy(np.asarray(x).copy()).to("tpu")

    tensors = [tensor(x) for x in args]
    tensors[-1] = tensors[-2]  # Both stores use the same production table object.
    # Each call mutates its pool. Compare direct/compiled execution from the
    # same initial history, rather than replaying an old context on new state.
    compiled_tensors = list(tensors)
    compiled_tensors[10] = tensor(pool)
    compiled_tensors[11] = tensor(index_pool)
    repeat_tensors = list(tensors)
    repeat_tensors[10] = tensor(pool)
    repeat_tensors[11] = tensor(index_pool)
    direct_owner, compiled_owner = tensors[10], compiled_tensors[10]
    direct_index, compiled_index = tensors[11], compiled_tensors[11]
    op = build_compact_pooled_qsa_op()
    with jax.profiler.TraceAnnotation("runner-qsa-direct"):
        output, selected = op(*tensors, **options)
        from torch_tpu._internal import sync

        sync.synchronize([output, selected, tensors[10], tensors[11]], wait=True)
        actual = output.cpu().float().numpy().reshape(tokens, local_heads, 256)
    selected_np = selected.cpu().numpy()
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[-3:], 0)
    fullgraph = torch.compile(op, backend="tpu", fullgraph=True)
    with jax.profiler.TraceAnnotation("runner-qsa-torchcompile"):
        repeated, selected_again = fullgraph(*compiled_tensors, **options)
        sync.synchronize(
            [repeated, selected_again, compiled_tensors[10], compiled_tensors[11]],
            wait=True,
        )
        compiled_actual = (
            repeated.cpu().float().numpy().reshape(tokens, local_heads, 256)
        )
    delta = compiled_actual - actual
    record_property("direct_vs_compiled_max_abs", float(np.max(np.abs(delta))))
    record_property(
        "direct_vs_compiled_relative_l2",
        float(np.linalg.norm(delta) / np.linalg.norm(actual)),
    )
    # Cross-mode fusion can differ by one BF16 step on the unchanged parent
    # too. Each arm must meet the original independent-reference tolerance.
    # Repeated execution of the same compiled graph remains bit-exact.
    with jax.profiler.TraceAnnotation("runner-qsa-torchcompile-repeat"):
        again, selected_repeat = fullgraph(*repeat_tensors, **options)
        sync.synchronize(
            [again, selected_repeat, repeat_tensors[10], repeat_tensors[11]], wait=True
        )
        assert torch.equal(again.cpu(), repeated.cpu())
        assert torch.equal(selected_repeat.cpu(), selected_again.cpu())
    assert torch.equal(
        repeat_tensors[10].cpu().view(torch.uint8),
        compiled_owner.cpu().view(torch.uint8),
    )
    assert tensors[10] is direct_owner and compiled_tensors[10] is compiled_owner
    assert tensors[11] is direct_index and compiled_tensors[11] is compiled_index
    assert torch.equal(
        repeat_tensors[11].cpu().view(torch.uint8),
        compiled_index.cpu().view(torch.uint8),
    )
    assert torch.equal(
        direct_index.cpu().view(torch.uint8)[0],
        torch.zeros_like(direct_index.cpu().view(torch.uint8)[0]),
    )
    query, key, value, gate, _, _ = produce(
        *producers[:7], producers[8], producers[9], 1e-6, 64
    )
    from vllm_torchtpu.kernels.qwen4_exp.qsa_rpa import pack_rows

    main_table = md["main_block_table"][0].numpy()
    logical = context + np.arange(tokens)
    expected_written = np.asarray(
        pack_rows(key[:-3], value[:-3], fp8=fp8, k_scale=1.0, v_scale=1.0)
    )
    inactive_page = main_table[logical[-3:] // layout.block_size]
    inactive_offset = logical[-3:] % layout.block_size
    lengths = context + np.arange(1, tokens + 1, dtype=np.int32)
    lengths[-3:] = 0
    for arm, result, selections, owner in (
        ("direct", actual, selected_np, direct_owner),
        ("compiled", compiled_actual, selected_again.cpu().numpy(), compiled_owner),
    ):
        assert np.isfinite(result).all()
        np.testing.assert_array_equal(result[-3:], 0)
        updated = owner.cpu().view(torch.uint8).numpy().copy()
        cache = updated.view(np.dtype(dtype)).reshape(pool.shape)
        written = cache[
            main_table[logical[:-3] // layout.block_size],
            logical[:-3] % layout.block_size,
        ]
        assert (
            np.linalg.norm(
                written.astype(np.float32) - expected_written.astype(np.float32)
            )
            / np.linalg.norm(expected_written.astype(np.float32))
            < 0.015
        )
        # Values have no RMS/RoPE arithmetic; require their exact native bits.
        np.testing.assert_array_equal(
            written.reshape(tokens - 3, -1, 256)[:, 1 : 2 * kv_heads : 2]
            .copy()
            .view(np.uint8),
            expected_written.reshape(tokens - 3, -1, 256)[:, 1 : 2 * kv_heads : 2]
            .copy()
            .view(np.uint8),
        )
        np.testing.assert_array_equal(
            cache[inactive_page, inactive_offset].copy().view(np.uint8),
            np.asarray(pool)[inactive_page, inactive_offset].copy().view(np.uint8),
        )
        block_ids = selections[:, :2048:4] // 4
        block_ids = np.where(selections[:, :2048:4] >= 0, block_ids, -1)
        if fp8 and local_heads == 24:
            assert bool(
                jax.jit(
                    functools.partial(
                        select_sc,
                        num_tiles=math.ceil(columns * layout.block_size / 256),
                        dtype=dtype,
                    )
                )(jnp.asarray(block_ids), jnp.asarray(lengths))
            )
        errors = []
        for row in (0, 128, tokens - 4):
            ids = selections[row]
            ids = ids[ids >= 0]
            values = (
                cache[main_table[ids // layout.block_size], ids % layout.block_size]
                .reshape(len(ids), -1, 256)
                .astype(np.float64)
            )
            expected = []
            per_head = local_heads // kv_heads
            for h in range(kv_heads):
                score = (
                    np.asarray(
                        query[row, h * per_head : (h + 1) * per_head], np.float64
                    )
                    @ values[:, 2 * h].T
                    / 16
                )
                mass = np.exp(score - score.max(axis=1, keepdims=True))
                expected.append(
                    mass / mass.sum(axis=1, keepdims=True) @ values[:, 2 * h + 1]
                )
            wanted = np.concatenate(expected) / (
                1 + np.exp(-np.asarray(gate[row], np.float64))
            )
            error = np.linalg.norm(result[row] - wanted) / np.linalg.norm(wanted)
            assert error < 0.015
            errors.append(float(error))
        record_property(arm + "_max_reference_relative_l2", max(errors))
        record_property(arm + "_live_cache_rows_checked", tokens - 3)
        record_property(arm + "_inactive_cache_rows_preserved", 3)
