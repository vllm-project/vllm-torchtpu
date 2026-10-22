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
"""Padded runner decode metadata through the real native pooled QSA bridge.

Scheduler rows and projected BF16 inputs are fixtures; runner preparation,
metadata builders, model adapter, Torch bridge and TPU kernels are real.
"""

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


@pytest.mark.tpu_test
@pytest.mark.parametrize("fp8", [False, True])
def test_runner_padded_decode_reaches_native_fused_attention(
    monkeypatch, fp8, record_property
):
    from vllm_torchtpu.kernels.qwen4_exp.pooled_qsa import compact_pooled_qsa_step
    from vllm_torchtpu.kernels.qwen4_exp.qsa import produce
    from vllm_torchtpu.layers.adapter.custom_ops.qsa_op import (
        build_compact_pooled_qsa_op,
    )

    monkeypatch.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", "1")
    cfg = pool_config(tp=1, kv="fp8_e4m3" if fp8 else "bfloat16")
    layout, specs = pooled_cache_specs(cfg)
    page = layout.block_size
    contexts = [37, 1659]
    runner = _make_runner(
        num_computed_tokens=contexts,
        prompt_tokens=[c + 1 for c in contexts],
        scheduled_tokens=[1, 1],
        token_paddings=[8],
        pcp_size=1,
        block_size=page,
    )
    runner.model_config = NS(hf_text_config=q4exp_text_config())
    runner.cache_config.block_size = page
    tables = (
        torch.tensor([[4, 2], [3, 1]], dtype=torch.int32),
        torch.tensor([[9], [10]], dtype=torch.int32),
    )
    runner.input_batch.block_table = [_BlockTable(t) for t in tables]
    names = tuple("model.layers.3.self_attn" + suffix for suffix in ("", ".raw_ring"))
    builders = [
        AttentionMetadataBuilder(specs[n], [n], cfg, torch.device("cpu"), runner, i)
        for i, n in enumerate(names)
    ]

    def build(**kwargs):
        common = NS(num_reqs=kwargs["num_reqs"])
        return {n: b.build(0, common) for n, b in zip(names, builders)}, None

    runner._build_attention_metadata = build
    prepared = TPUModelRunner._prepare_inputs(runner, _scheduler_output([1, 1]), 0, 0)
    adapter = Qwen4ExpAdapter(runner)
    metadata = adapter.prepare_inputs(
        ModelInputSnapshot(
            adapter.begin_step(),
            0,
            2,
            8,
            prepared[0],
            runner.input_ids_cpu[:8],
        )
    )
    assert metadata["query_start_loc"].tolist() == [0, 1] + [2] * 7
    assert metadata["context_lens"].tolist() == contexts + [0] * 6
    assert metadata["valid_tokens"].tolist() == [True, True] + [False] * 6
    gen = torch.Generator(device="cpu").manual_seed(2123)

    def normal(shape):
        return (torch.randn(shape, generator=gen) * 0.3).to(torch.bfloat16)

    dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    packing = 4 if fp8 else 2
    initial = torch.zeros((11, page, 4 // packing, packing, 256), dtype=dtype)
    initial[1:5] = normal(initial[1:5].shape).to(dtype)
    initial_index = torch.zeros(
        (11, layout.index_rows // 2, 1, 2, layout.compressed_width),
        dtype=torch.bfloat16,
    )
    producers = [
        normal((8, 48 * 256)),
        normal((8, 2 * 256)),
        normal((8, 2 * 256)),
        normal((8, 5 * 128)),
        torch.zeros(256, dtype=torch.bfloat16),
        torch.zeros(256, dtype=torch.bfloat16),
        torch.zeros(128, dtype=torch.bfloat16),
        torch.zeros(128, dtype=torch.bfloat16),
        runner.position_ids.to(torch.int32),
        torch.zeros((1662, 128), dtype=torch.bfloat16),
    ]
    producers[-1][:, :32] = 1
    for x in producers[:4]:
        x[2:] = torch.nan
    tail = [metadata[n] for n in ("query_start_loc", "context_lens")]
    tail += [prepared[0][names[1]].block_tables]
    tail += [
        metadata[n].reshape(-1) for n in ("main_block_table", "compressed_block_table")
    ]
    options = dict(compress_ratio=4, token_topk=2048, norm_eps=1e-6, rotary_dim=64)

    def array(x):
        if x.dtype == torch.bfloat16:
            return jnp.asarray(x.float().numpy(), jnp.bfloat16)
        if x.dtype == torch.float8_e4m3fn:
            return jax.lax.bitcast_convert_type(
                jnp.asarray(x.view(torch.uint8).numpy()), jnp.float8_e4m3fn
            )
        return jnp.asarray(x.numpy(), jnp.int32)

    args = [array(x) for x in [*producers, initial, initial_index, *tail]]
    hlo = (
        jax.jit(compact_pooled_qsa_step, static_argnames=tuple(options))
        .lower(*args, **options)
        .compile()
        .as_text()
    )
    assert "fused_qsa_decode" in hlo and "prefill_paired_pages" not in hlo
    query, _, _, gate, _, _ = produce(*args[:7], args[8], args[9], 1e-6, 64)
    tensor_args = [x.to("tpu") for x in producers]
    tensor_tail = [x.to(torch.int32).to("tpu") for x in tail]
    tensor_tail[-1] = tensor_tail[-2]
    op = build_compact_pooled_qsa_op()
    compiled = torch.compile(op, backend="tpu", fullgraph=True)
    for label, function in (("direct", op), ("compiled", compiled)):
        owner = initial.to("tpu")
        index_owner = initial_index.to("tpu")
        with jax.profiler.TraceAnnotation("runner-qsa-decode-" + label):
            out, selected = function(
                *tensor_args, owner, index_owner, *tensor_tail, **options
            )
            from torch_tpu._internal import sync

            sync.synchronize([out, selected, owner, index_owner], wait=True)
            actual = out.cpu().float().numpy().reshape(8, 24, 256)
        assert np.isfinite(actual).all()
        np.testing.assert_array_equal(actual[2:], 0)
        stored = owner.cpu().view(torch.uint8).numpy().copy()
        cache = stored.view(
            np.dtype(jnp.float8_e4m3fn if fp8 else jnp.bfloat16)
        ).reshape(initial.shape)
        # Null and unused page zero stays untouched; the native owner is reused.
        np.testing.assert_array_equal(stored[0], initial.view(torch.uint8).numpy()[0])
        unchanged = stored[1:5].copy()
        before_main = initial[1:5].view(torch.uint8).numpy()
        for row, context in enumerate(contexts):
            physical = int(tables[0][row, context // page])
            offset = context % page
            values = cache[physical, offset].reshape(4, 256)[1::2]
            expected_values = producers[2][row].reshape(2, 256).to(dtype)
            np.testing.assert_array_equal(
                values.copy().view(np.uint8), expected_values.view(torch.uint8).numpy()
            )
            unchanged[physical - 1, offset] = before_main[physical - 1, offset]
        np.testing.assert_array_equal(unchanged, before_main)
        record_property(label + "_live_native_value_rows_checked", 2)
        torch.testing.assert_close(
            index_owner.cpu()[[0, 5, 6, 7, 8]].view(torch.uint8),
            initial_index[[0, 5, 6, 7, 8]].view(torch.uint8),
            rtol=0,
            atol=0,
        )
        errors = []
        for row, context in enumerate(contexts):
            ids = selected[row].cpu().numpy()
            ids = ids[ids >= 0]
            np.testing.assert_array_equal(np.sort(ids), np.arange(context + 1))
            values = (
                cache[tables[0][row].numpy()[ids // page], ids % page]
                .reshape(len(ids), 4, 256)
                .astype(np.float64)
            )
            expected = []
            for head in range(2):
                scores = (
                    np.asarray(query[row, head * 12 : (head + 1) * 12], np.float64)
                    @ values[:, 2 * head].T
                    / 16
                )
                weights = np.exp(scores - scores.max(axis=1, keepdims=True))
                expected.append(
                    weights
                    / weights.sum(axis=1, keepdims=True)
                    @ values[:, 2 * head + 1]
                )
            wanted = np.concatenate(expected) / (
                1 + np.exp(-np.asarray(gate[row], np.float64))
            )
            error = np.linalg.norm(actual[row] - wanted) / np.linalg.norm(wanted)
            assert error < 0.015
            errors.append(float(error))
        record_property(label + "_max_reference_relative_l2", max(errors))
