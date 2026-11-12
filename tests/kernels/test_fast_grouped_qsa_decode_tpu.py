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
"""Unit and TPU regression tests for fast grouped QSA decode and projected op."""

import inspect
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch


def test_can_use_fast_grouped_qsa_decode_guards(monkeypatch):
    from vllm_torchtpu.kernels.qwen4_exp import fast_grouped_qsa_decode as fgq

    monkeypatch.setattr(fgq, "_hardware", lambda: object())
    monkeypatch.setattr(
        fgq.pltpu,
        "get_tpu_info",
        lambda: SimpleNamespace(sparse_core=object()),
    )
    qgate = jnp.zeros((16, 48 * 256), jnp.bfloat16)
    pool_f8 = jnp.zeros((8, 1664, 1, 4, 256), jnp.float8_e4m3fn)
    pool_bf16 = jnp.zeros((8, 896, 2, 2, 256), jnp.bfloat16)
    table = jnp.zeros((16, 64), jnp.int32)
    context = jnp.zeros((16,), jnp.int32)

    assert fgq.can_use_fast_grouped_qsa_decode(
        qgate, pool_f8, table, context, 4, 2048, 256
    )
    assert not fgq.can_use_fast_grouped_qsa_decode(
        qgate, pool_bf16, table, context, 4, 2048, 256
    )
    assert not fgq.can_use_fast_grouped_qsa_decode(
        jnp.zeros((12, 48 * 256), jnp.bfloat16), pool_f8, table, context, 4, 2048, 256
    )
    assert not fgq.can_use_fast_grouped_qsa_decode(
        jnp.zeros((64, 48 * 256), jnp.bfloat16),
        pool_f8,
        jnp.zeros((64, 64), jnp.int32),
        jnp.zeros((64,), jnp.int32),
        4,
        2048,
        256,
    )
    assert not fgq.can_use_fast_grouped_qsa_decode(
        qgate, pool_f8, table, context, 8, 2048, 256
    )


def test_build_compact_pooled_qsa_projected_op_schema():
    from vllm_torchtpu.layers.adapter.custom_ops import qsa_op

    qsa_op._build_compact_pooled_qsa_projected_op.cache_clear()
    op = qsa_op.build_compact_pooled_qsa_projected_op()
    assert op is qsa_op.build_compact_pooled_qsa_projected_op()
    registered = inspect.getclosurevars(op).nonlocals["op"]
    assert tuple(registered._init_fn.donate_argnums) == (11, 12, 13)
    schema = registered._opoverload._schema
    assert [a.name for a in schema.arguments][11:14] == [
        "pool",
        "index_pool",
        "ring_pool",
    ]
    assert len(schema.returns) == 4


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "batch,live,context", [(8, 5, 515), (16, 16, 4203), (32, 27, 2501)]
)
def test_fast_grouped_qsa_projected_kernel_matches_reference(batch, live, context):
    from vllm_torchtpu.kernels.qwen4_exp.fast_grouped_qsa_decode import (
        qsa_fast_grouped_qsa_projected_attention,
        qsa_sc_gather_2level_tc_tiled,
    )
    from vllm_torchtpu.kernels.qwen4_exp.qsa_decode_fused_ops import (
        qsa_prepare_token_indices_packed_meta,
    )

    rng = np.random.default_rng(1000 + batch * 17 + live)
    page_size = 1664
    num_pages = 64
    budget = 512
    ratio = 4
    heads, kv_heads, dim = 24, 2, 256
    hidden_size, branches, tile = 2560, 4, 512

    pool_np = rng.normal(0, 0.4, (num_pages, page_size, 1, 4, dim)).astype(np.float32)
    pool_np[0] = 0.0
    pool = jnp.asarray(pool_np, dtype=jnp.float8_e4m3fn)

    pages_per_req = (context + page_size - 1) // page_size
    table_np = np.zeros((batch, 8), dtype=np.int32)
    next_p = 1
    for r in range(live):
        for c in range(pages_per_req):
            table_np[r, c] = next_p
            next_p = (next_p % (num_pages - 1)) + 1
    table = jnp.asarray(table_np)

    full_groups = context // ratio
    n_sel = min(budget, full_groups)
    blocks_np = np.full((batch, budget), -1, dtype=np.int32)
    lengths_np = np.zeros((batch,), dtype=np.int32)
    for r in range(live):
        lengths_np[r] = context
        chosen = np.sort(rng.choice(full_groups, size=n_sel, replace=False))
        blocks_np[r, :n_sel] = chosen
    blocks = jnp.asarray(blocks_np)
    lengths = jnp.asarray(lengths_np)

    q = jnp.asarray(rng.normal(0, 1.0, (batch, heads, dim)), dtype=jnp.bfloat16)
    gate = jnp.asarray(rng.normal(0, 0.5, (batch, heads, dim)), dtype=jnp.bfloat16)

    def _pack_gqa(x: jax.Array) -> jax.Array:
        x = x.reshape(batch // 8, 8, kv_heads, 12, dim)
        x = jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, 4), (0, 0)))
        return x.transpose(0, 2, 1, 3, 4).reshape(batch // 8, kv_heads, 128, dim)

    qg_combined = jnp.stack([_pack_gqa(q), _pack_gqa(gate)], axis=1)

    w_o = jnp.asarray(
        rng.normal(0, 0.02, (heads * dim, hidden_size)), dtype=jnp.bfloat16
    )
    w_o_tiles = w_o.reshape(heads * dim, hidden_size // tile, tile).transpose(1, 0, 2)
    residual = jnp.asarray(
        rng.normal(0, 0.2, (batch, branches * hidden_size)), dtype=jnp.bfloat16
    )
    hc_gate = jnp.asarray(rng.uniform(0.2, 1.5, (batch, branches)), dtype=jnp.float32)
    k_scale, v_scale = 0.5, 1.25

    def _make_run_fast(sc_chunk: int):
        @jax.jit
        def _fn(qgc, blk, lens, tbl, pl, wot, res, hcg):
            tok_idx, meta_2d = qsa_prepare_token_indices_packed_meta(
                blk,
                lens,
                tbl,
                page_size=page_size,
                ratio=ratio,
                sc_chunk_tokens=sc_chunk,
            )
            gathered = qsa_sc_gather_2level_tc_tiled(
                pl,
                tok_idx,
                None,
                chunk_tokens=sc_chunk,
                skip_negative_chunks=True,
            )
            return qsa_fast_grouped_qsa_projected_attention(
                qgc,
                None,
                gathered,
                pl,
                meta_2d,
                None,
                wot,
                res,
                hcg,
                kv_heads=kv_heads,
                ratio=ratio,
                k_scale=k_scale,
                v_scale=v_scale,
                bf16_pv=True,
                combined_qgate=True,
                unroll_tiles=True,
                native_f8_kv=True,
            )

        return _fn

    actual_512 = np.asarray(
        _make_run_fast(512)(
            qg_combined, blocks, lengths, table, pool, w_o_tiles, residual, hc_gate
        ).astype(jnp.float32)
    )
    actual = np.asarray(
        _make_run_fast(256)(
            qg_combined, blocks, lengths, table, pool, w_o_tiles, residual, hc_gate
        ).astype(jnp.float32)
    )
    np.testing.assert_array_equal(actual, actual_512)
    pool_deq = np.asarray(pool.astype(jnp.float32)).reshape(
        num_pages, page_size, 4, dim
    )
    q_np = np.asarray(q.astype(jnp.float32))
    g_np = np.asarray(gate.astype(jnp.float32))
    res_np = np.asarray(residual.astype(jnp.float32)).reshape(
        batch, branches, hidden_size
    )
    hcg_np = np.asarray(hc_gate)
    wo_np = np.asarray(w_o.astype(jnp.float32))

    # Inactive rows preserve residual bit-exact
    np.testing.assert_array_equal(
        actual[live:], np.asarray(residual.astype(jnp.float32))[live:]
    )

    def _bf16_round(arr: np.ndarray) -> np.ndarray:
        return np.asarray(jnp.asarray(arr, dtype=jnp.bfloat16).astype(jnp.float32))

    for r in range(live):
        sel_groups = blocks_np[r, :n_sel]
        tok_ids = (sel_groups[:, None] * ratio + np.arange(ratio)[None, :]).reshape(-1)
        tail_ids = np.arange(full_groups * ratio, context)
        all_ids = np.concatenate([tok_ids, tail_ids])
        phys_p = table_np[r, all_ids // page_size]
        phys_o = all_ids % page_size
        kv_rows = pool_deq[phys_p, phys_o]  # [N, 4, 256]
        head_outs = []
        for h in range(kv_heads):
            qh = q_np[r, h * 12 : (h + 1) * 12]
            kh = kv_rows[:, 2 * h] * k_scale
            vh = kv_rows[:, 2 * h + 1] * v_scale
            scores = (qh @ kh.T) * (dim**-0.5)
            probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
            probs /= probs.sum(axis=-1, keepdims=True)
            head_outs.append(probs @ vh)
        attn_out = _bf16_round(np.concatenate(head_outs, axis=0))
        sig_g = _bf16_round(1.0 / (1.0 + np.exp(-g_np[r])))
        gated = _bf16_round(attn_out * sig_g)
        proj = _bf16_round(gated.reshape(-1) @ wo_np)
        expected_r = _bf16_round(res_np[r] + hcg_np[r, :, None] * proj[None, :])
        rel_err = np.linalg.norm(actual[r] - expected_r.reshape(-1)) / np.linalg.norm(
            expected_r
        )
        assert rel_err < 0.015, f"row {r} rel_err={rel_err}"


@pytest.mark.tpu_test
@pytest.mark.parametrize("mrope", [False, True])
@torch.inference_mode()
def test_qwen4_exp_qsa_forward_fused_decode_and_mixed_batches(tmp_path, mrope):
    from torch_tpu._internal.compile._backend import TpuBackend
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

    from tests.models.vllm.qwen4_exp_test_utils import q4exp_text_config, qsa_test_pools
    from vllm_torchtpu.layers.adapter.quantization.unquantized import (
        VllmUnquantizedLinearMethod,
    )
    from vllm_torchtpu.models.vllm.qwen4_exp.qsa import QSAMetadata, Qwen4ExpQSA
    from vllm_torchtpu.utils import synchronize_tensors

    cfg_kwargs = {}
    if not mrope:
        cfg_kwargs["rope_parameters"] = {
            "rope_type": "default",
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.25,
        }
    text_cfg = q4exp_text_config(**cfg_kwargs)

    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    resolve_kv_cache_layout(config, [["LBNHC", "LBHNC"]])
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method="file://" + str(tmp_path / f"gloo_{int(mrope)}"),
            backend="gloo",
        )
        initialize_model_parallel(backend="gloo")
        try:
            layer = Qwen4ExpQSA(
                config=text_cfg,
                tp_rank=0,
                tp_size=1,
                max_model_len=8192,
                cache_dtype="fp8_e4m3",
                device="cpu",
            )
            gen = torch.Generator().manual_seed(20261111 + int(mrope))
            weights = {
                name: (torch.randn(p.shape, generator=gen) * 0.025).bfloat16()
                for name, p in layer.named_parameters()
            }
            packed = weights.pop("input_proj.weight")
            for name, value in zip(
                ("k_proj.weight", "v_proj.weight", "indexer.index_qk_proj.weight"),
                packed.split(layer.input_sizes[1:], dim=0),
            ):
                weights[name] = value
            layer.load_weights(weights.items())
            layer.load_weights(
                [("_k_scale", torch.tensor(0.5)), ("_v_scale", torch.tensor(1.5))]
            )
            initial, _ = qsa_test_pools(layer, 24, block_size=128)
            layer.finalize_weights(block_size=initial[0].shape[1])
            for linear in layer.modules():
                if isinstance(linear, LinearBase):
                    linear.quant_method = VllmUnquantizedLinearMethod()
                    linear.quant_method.process_weights_after_loading(linear)
            layer.to("tpu")
            assert layer._fused_projected_op is not None

            def ref_fn(hidden, positions, metadata, residual, gate):
                out, _ = layer.forward_with_indices(
                    hidden,
                    positions,
                    metadata,
                    residual,
                    gate,
                    hc_token_tile=128,
                    hc_min_tokens=1024,
                )
                return out

            def fast_fn(hidden, positions, metadata, residual, gate):
                return layer.forward(
                    hidden,
                    positions,
                    metadata,
                    residual,
                    gate,
                    hc_token_tile=128,
                    hc_min_tokens=1024,
                )

            ref_compiled = torch.compile(
                ref_fn, backend=TpuBackend(), fullgraph=True, dynamic=False
            )
            fast_compiled = torch.compile(
                fast_fn, backend=TpuBackend(), fullgraph=True, dynamic=False
            )

            requests = 16
            columns = 8
            shared_cu = torch.arange(requests + 1, dtype=torch.int32).to("tpu")

            # Stateful 5-step decode crossing compression boundary (context 0..4)
            ref_owners = [x.to("tpu") for x in initial]
            fast_owners = [x.to("tpu") for x in initial]
            for context in range(5):
                tokens, live = 16, 12
                starts = torch.tensor(
                    [*range(live + 1), *([live] * (requests - live))], dtype=torch.int32
                ).to("tpu")
                lengths = torch.zeros(requests, dtype=torch.int32)
                lengths[:live] = context
                lengths = lengths.to("tpu")
                slots = torch.arange(1, requests + 1, dtype=torch.int32)
                slots[live:] = 0
                slots = slots.to("tpu")
                pages = torch.zeros((requests, columns), dtype=torch.int32)
                pages[:live, 0] = torch.arange(1, live + 1)
                pages = pages.to("tpu")
                dist = torch.tensor([live, live, live], dtype=torch.int32).to("tpu")
                md = QSAMetadata(starts, lengths, decode_cu=shared_cu)
                pos_1d = torch.full((tokens,), context, dtype=torch.int64)
                positions = (
                    torch.stack([pos_1d, pos_1d + 10, pos_1d + 20], dim=0).to("tpu")
                    if mrope
                    else pos_1d.to("tpu")
                )
                hidden = (
                    (torch.randn(tokens, 2560, generator=gen) * 0.2)
                    .bfloat16()
                    .to("tpu")
                )
                residual = (
                    (torch.randn(tokens, 10240, generator=gen) * 0.2)
                    .bfloat16()
                    .to("tpu")
                )
                gate = (2 * torch.sigmoid(torch.randn(tokens, 4, generator=gen))).to(
                    "tpu"
                )
                per_layer = {
                    layer.prefix: SimpleNamespace(
                        block_tables=pages,
                        request_distribution=dist,
                        qsa_ring_slots=slots,
                    )
                }
                with set_forward_context(per_layer, config, num_tokens=tokens):
                    layer.bind_pool(*ref_owners)
                    expected = ref_compiled(hidden, positions, md, residual, gate)
                    layer.bind_pool(*fast_owners)
                    actual = fast_compiled(hidden, positions, md, residual, gate)
                synchronize_tensors([actual, expected, *ref_owners, *fast_owners])
                a, b = actual.cpu().float(), expected.cpu().float()
                rel = float((a - b).norm() / b.norm())
                assert rel < 0.015, f"decode step context={context} rel={rel}"
                torch.testing.assert_close(
                    a[live:], residual.cpu().float()[live:], rtol=0, atol=0
                )

            # Non-contiguous active decode slots (e.g. slots 0, 3, 9) leaving
            # interior sequence tiles empty.
            tokens = 16
            active_slots = (0, 3, 9)
            n_act = len(active_slots)
            starts_list = [0]
            cur = 0
            for r_i in range(requests):
                if r_i in active_slots:
                    cur += 1
                starts_list.append(cur)
            starts = torch.tensor(starts_list, dtype=torch.int32).to("tpu")
            lengths = torch.zeros(requests, dtype=torch.int32)
            for r_i in active_slots:
                lengths[r_i] = 5
            lengths = lengths.to("tpu")
            slots = torch.zeros(requests, dtype=torch.int32)
            for r_i in active_slots:
                slots[r_i] = r_i + 1
            slots = slots.to("tpu")
            pages = torch.zeros((requests, columns), dtype=torch.int32)
            for r_i in active_slots:
                pages[r_i, 0] = r_i + 1
            pages = pages.to("tpu")
            dist = torch.tensor([n_act, n_act, requests], dtype=torch.int32).to("tpu")
            md = QSAMetadata(starts, lengths, decode_cu=shared_cu)
            pos_1d = torch.full((tokens,), 5, dtype=torch.int64)
            positions = (
                torch.stack([pos_1d, pos_1d + 10, pos_1d + 20], dim=0).to("tpu")
                if mrope
                else pos_1d.to("tpu")
            )
            hidden = (
                (torch.randn(tokens, 2560, generator=gen) * 0.2).bfloat16().to("tpu")
            )
            residual = (
                (torch.randn(tokens, 10240, generator=gen) * 0.2).bfloat16().to("tpu")
            )
            gate = (2 * torch.sigmoid(torch.randn(tokens, 4, generator=gen))).to("tpu")
            per_layer = {
                layer.prefix: SimpleNamespace(
                    block_tables=pages,
                    request_distribution=dist,
                    qsa_ring_slots=slots,
                )
            }
            with set_forward_context(per_layer, config, num_tokens=tokens):
                layer.bind_pool(*ref_owners)
                expected = ref_compiled(hidden, positions, md, residual, gate)
                layer.bind_pool(*fast_owners)
                actual = fast_compiled(hidden, positions, md, residual, gate)
            synchronize_tensors([actual, expected, *ref_owners, *fast_owners])
            a, b = actual.cpu().float(), expected.cpu().float()
            rel = float((a[:n_act] - b[:n_act]).norm() / b[:n_act].norm())
            assert rel < 0.015, f"non-contiguous decode rel={rel}"
            torch.testing.assert_close(
                a[n_act:], residual.cpu().float()[n_act:], rtol=0, atol=0
            )
            for x, y in zip(ref_owners, fast_owners):
                assert torch.equal(x.cpu().view(torch.uint8), y.cpu().view(torch.uint8))

            # Short prefill and mixed decode+verify+prefill in bucket 16 execute
            # directly on the zero-cond fast QSA path with bit-exact cache updates.
            for context, live, q_lens in (
                (0, 2, [8, 8]),
                (4, 4, [1, 2, 6, 7]),
            ):
                tokens = 16
                ref_owners = [x.to("tpu") for x in initial]
                fast_owners = [x.to("tpu") for x in initial]
                valid_rows = sum(q_lens)
                starts = torch.tensor(
                    [
                        0,
                        *torch.tensor(q_lens).cumsum(0).tolist(),
                        *([valid_rows] * (requests - live)),
                    ],
                    dtype=torch.int32,
                ).to("tpu")
                lengths = torch.zeros(requests, dtype=torch.int32)
                lengths[:live] = context
                lengths = lengths.to("tpu")
                slots = torch.arange(1, requests + 1, dtype=torch.int32)
                slots[live:] = 0
                slots = slots.to("tpu")
                pages = torch.zeros((requests, columns), dtype=torch.int32)
                pages[:live, 0] = torch.arange(1, live + 1)
                pages = pages.to("tpu")
                dec = sum(n == 1 for n in q_lens)
                ver = dec + sum(n == 2 for n in q_lens)
                dist = torch.tensor([dec, ver, live], dtype=torch.int32).to("tpu")
                md = QSAMetadata(starts, lengths, decode_cu=shared_cu)
                pos_1d = torch.full((tokens,), context, dtype=torch.int64)
                pos_1d[:valid_rows] = (
                    torch.cat([torch.arange(n) for n in q_lens]) + context
                )
                positions = (
                    torch.stack([pos_1d, pos_1d + 10, pos_1d + 20], dim=0).to("tpu")
                    if mrope
                    else pos_1d.to("tpu")
                )
                hidden = (
                    (torch.randn(tokens, 2560, generator=gen) * 0.2)
                    .bfloat16()
                    .to("tpu")
                )
                residual = (
                    (torch.randn(tokens, 10240, generator=gen) * 0.2)
                    .bfloat16()
                    .to("tpu")
                )
                gate = (2 * torch.sigmoid(torch.randn(tokens, 4, generator=gen))).to(
                    "tpu"
                )
                per_layer = {
                    layer.prefix: SimpleNamespace(
                        block_tables=pages,
                        request_distribution=dist,
                        qsa_ring_slots=slots,
                    )
                }
                with set_forward_context(per_layer, config, num_tokens=tokens):
                    layer.bind_pool(*ref_owners)
                    expected = ref_compiled(hidden, positions, md, residual, gate)
                    layer.bind_pool(*fast_owners)
                    actual = fast_compiled(hidden, positions, md, residual, gate)
                synchronize_tensors([actual, expected, *ref_owners, *fast_owners])
                a, b = actual.cpu().float(), expected.cpu().float()
                rel = float(
                    (a[:valid_rows] - b[:valid_rows]).norm() / b[:valid_rows].norm()
                )
                assert rel < 0.015, f"mixed q_lens={q_lens} rel={rel}"
                torch.testing.assert_close(
                    a[valid_rows:],
                    residual.cpu().float()[valid_rows:],
                    rtol=0,
                    atol=0,
                )
                for x, y in zip(ref_owners, fast_owners):
                    assert torch.equal(
                        x.cpu().view(torch.uint8), y.cpu().view(torch.uint8)
                    )
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "batch,reqs,width,cache_tags",
    [
        (8, 8, 1, False),
        (8, 8, 1, True),
        (16, 4, 4, False),
        (16, 4, 4, True),
        (32, 8, 4, False),
    ],
)
def test_compact_pooled_qsa_fast_decode_mtp_ring8_and_verify_step_tpu(
    batch: int, reqs: int, width: int, cache_tags: bool
):
    """Verify ring_capacity=8 fast decode for MTP draft and target verify."""
    from vllm_torchtpu.kernels.qwen4_exp.pooled_qsa import compact_pooled_qsa_step
    from vllm_torchtpu.kernels.qwen4_exp.qsa_decode_fused_ops import (
        compact_pooled_qsa_fast_decode_step,
    )
    from vllm_torchtpu.kernels.qwen4_exp.qsa_state import write_ring_and_tags

    rng = np.random.default_rng(20261112 + batch * 17 + reqs * 3 + int(cache_tags))
    num_reqs_padded = 16
    pages_per_req = 4
    num_blocks = reqs * pages_per_req + 4

    def rand_bf16(*shape, scale=0.25):
        return jnp.asarray(rng.normal(0.0, scale, size=shape), dtype=jnp.bfloat16)

    base_lens = [515, 1024, 1663, 1664, 2049, 2558, 3071, 3329][:reqs]
    ctx_list = base_lens + [0] * (num_reqs_padded - reqs)
    context_lens = jnp.asarray(ctx_list, dtype=jnp.int32)

    counts = [width] * reqs + [0] * (num_reqs_padded - reqs)
    starts = jnp.asarray(np.cumsum([0] + counts), dtype=jnp.int32)
    pos_list = []
    for r in range(reqs):
        pos_list.extend(range(base_lens[r], base_lens[r] + width))
    pos_list.extend([0] * (batch - len(pos_list)))
    positions = jnp.asarray(pos_list, dtype=jnp.int32)

    ring_ids = jnp.asarray(
        list(range(1, reqs + 1)) + [0] * (num_reqs_padded - reqs), dtype=jnp.int32
    )
    main_bt = np.zeros((num_reqs_padded, pages_per_req), dtype=np.int32)
    for r in range(reqs):
        for p in range(pages_per_req):
            main_bt[r, p] = 1 + r * pages_per_req + p
    main_bt_j = jnp.asarray(main_bt)
    dist = jnp.asarray([reqs if width == 1 else 0, reqs, reqs], dtype=jnp.int32)

    main_pool = jax.lax.bitcast_convert_type(
        jnp.asarray(
            rng.integers(0, 120, size=(num_blocks, 1664, 1, 4, 256), dtype=np.uint8)
        ),
        jnp.float8_e4m3fn,
    )
    index_pool = rand_bf16(num_blocks, 208, 1, 2, 128, scale=0.2)
    ring_pairs = 8 if cache_tags else 4
    ring_pool = jnp.zeros(
        (num_reqs_padded + 4, ring_pairs, 1, 2, 128), dtype=jnp.bfloat16
    )
    init_ring = rand_bf16(num_reqs_padded, 8, 128)
    if cache_tags:
        init_tags = np.full((num_reqs_padded, 8, 2), -1, dtype=np.int32)
        for r in range(reqs):
            c = base_lens[r]
            for s in range(8):
                pos = c - 1 - ((c - 1 - s) % 8)
                init_tags[r, s, 0] = pos
                init_tags[r, s, 1] = 1 if pos >= 0 else 0
        tags_flat = jnp.asarray(init_tags).reshape(num_reqs_padded, -1)
    else:
        tags_flat = jnp.zeros((num_reqs_padded, 0), dtype=jnp.int32)
    ring_pool = jax.jit(write_ring_and_tags, donate_argnums=(0,))(
        ring_pool,
        init_ring.reshape(num_reqs_padded, -1),
        tags_flat,
        ring_ids,
    )

    hidden = rand_bf16(batch, 2560)
    w_q = rand_bf16(2560, 12288, scale=0.04)
    w_in = rand_bf16(2560, 1664, scale=0.04)
    w_o = rand_bf16(6144, 2560, scale=0.04)
    w_o_tiles = w_o.reshape(6144, 5, 512).transpose(1, 0, 2)
    residual = rand_bf16(batch, 4 * 2560)
    hc_gate = jnp.ones((batch, 4), dtype=jnp.float32)

    q_gate = jnp.dot(hidden, w_q, preferred_element_type=jnp.float32).astype(
        jnp.bfloat16
    )
    packed = jnp.dot(hidden, w_in, preferred_element_type=jnp.float32).astype(
        jnp.bfloat16
    )
    key, value, idx_in = packed[:, :512], packed[:, 512:1024], packed[:, 1024:]

    q_norm = rand_bf16(256, scale=0.1)
    k_norm = rand_bf16(256, scale=0.1)
    iq_norm = rand_bf16(128, scale=0.1) + 1.0
    ik_norm = rand_bf16(128, scale=0.1) + 1.0
    k_norm_mult = 1.0 + k_norm.astype(jnp.float32)
    norm_weights_packed = jnp.concatenate(
        [k_norm_mult, iq_norm.astype(jnp.float32), ik_norm.astype(jnp.float32)],
        axis=0,
    )
    angle = jnp.arange(4096)[:, None] * (jnp.arange(32)[None, :] / 4096.0)
    trig = jnp.pad(
        jnp.concatenate([jnp.cos(angle), jnp.sin(angle)], axis=1), ((0, 0), (0, 64))
    ).astype(jnp.bfloat16)

    ref_fn = jax.jit(
        lambda mp, ip, rp: compact_pooled_qsa_step(
            q_gate,
            key,
            value,
            idx_in,
            q_norm,
            k_norm,
            iq_norm,
            ik_norm,
            positions,
            trig,
            mp,
            ip,
            rp,
            starts,
            context_lens,
            ring_ids,
            main_bt_j,
            main_bt_j,
            dist,
            k_norm_multiplier=k_norm_mult,
            output_weight=w_o_tiles,
            residual=residual,
            hc_gate=hc_gate,
            compress_ratio=4,
            token_topk=2048,
            norm_eps=1e-6,
            rotary_dim=64,
            k_scale=0.5,
            v_scale=0.5,
            ring_capacity=8,
            cache_tags=cache_tags,
        ),
        donate_argnums=(0, 1, 2),
    )
    fast_fn = jax.jit(
        lambda mp, ip, rp: compact_pooled_qsa_fast_decode_step(
            hidden,
            packed,
            w_q,
            q_norm,
            norm_weights_packed,
            positions,
            trig,
            mp,
            ip,
            rp,
            starts,
            context_lens,
            ring_ids,
            main_bt_j,
            main_bt_j,
            w_o_tiles,
            residual,
            hc_gate,
            4,
            2048,
            1e-6,
            64,
            0.5,
            0.5,
            0,
            ring_capacity=8,
            cache_tags=cache_tags,
        ),
        donate_argnums=(0, 1, 2),
    )

    ref_mp, ref_ip, ref_rp, ref_out, _ref_blk = ref_fn(
        jnp.copy(main_pool), jnp.copy(index_pool), jnp.copy(ring_pool)
    )
    fast_mp, fast_ip, fast_rp, fast_out = fast_fn(
        jnp.copy(main_pool), jnp.copy(index_pool), jnp.copy(ring_pool)
    )

    active = reqs * width
    ref_f32 = ref_out[:active].astype(jnp.float32)
    fast_f32 = fast_out[:active].astype(jnp.float32)
    cos_sim = float(
        jnp.sum(ref_f32 * fast_f32)
        / (jnp.linalg.norm(ref_f32) * jnp.linalg.norm(fast_f32))
    )
    assert cos_sim > 0.9999, f"cos_sim={cos_sim}"
    np.testing.assert_array_equal(
        np.asarray(fast_mp).view(np.uint8), np.asarray(ref_mp).view(np.uint8)
    )
    np.testing.assert_allclose(
        np.asarray(fast_ip, dtype=np.float32),
        np.asarray(ref_ip, dtype=np.float32),
        rtol=1e-2,
        atol=1e-2,
    )
    np.testing.assert_allclose(
        np.asarray(fast_rp, dtype=np.float32),
        np.asarray(ref_rp, dtype=np.float32),
        rtol=1e-2,
        atol=1e-2,
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize("prefix", [3, 7, 127, 128, 1663, 1664])
def test_compact_pooled_qsa_fast_decode_mtp_rejection_and_idle_rank_tpu(prefix: int):
    """Fast decode with ring_capacity=8 handles MTP rejection and idle rank."""
    from vllm_torchtpu.kernels.qwen4_exp.qsa_decode_fused_ops import (
        compact_pooled_qsa_fast_decode_step,
    )
    from vllm_torchtpu.kernels.qwen4_exp.qsa_state import write_ring_and_tags

    rng = np.random.default_rng(909 + prefix)
    rows, tokens, k = 8, 8, 3

    def rand_bf16(*shape, scale=0.25):
        return jnp.asarray(rng.normal(0.0, scale, size=shape), dtype=jnp.bfloat16)

    pool = jnp.zeros((7, 1664, 1, 4, 256), jnp.float8_e4m3fn)
    index_pool = jnp.zeros((7, 208, 1, 2, 128), jnp.bfloat16)
    ring_pool = jnp.zeros((7, 8, 1, 2, 128), jnp.bfloat16)
    ring = rand_bf16(rows, 8, 128)
    tags = (
        jnp.zeros((rows, 8, 2), jnp.int32)
        .at[0, :, 0]
        .set(prefix - 1 - (prefix - 1 - jnp.arange(8)) % 8)
    )
    owners = jnp.array([5] + [0] * (rows - 1), jnp.int32)
    ring_pool = jax.jit(write_ring_and_tags, donate_argnums=(0,))(
        ring_pool, ring.reshape(rows, -1), tags.reshape(rows, -1), owners
    )
    original = (
        np.asarray(pool).copy(),
        np.asarray(index_pool).copy(),
        np.asarray(ring_pool).copy(),
    )
    main = jnp.zeros((rows, 2), jnp.int32).at[0].set(jnp.array([1, 2]))
    w_q = rand_bf16(2560, 12288, scale=0.04)
    w_in = rand_bf16(2560, 1664, scale=0.04)
    w_o = rand_bf16(6144, 2560, scale=0.04)
    w_o_tiles = w_o.reshape(6144, 5, 512).transpose(1, 0, 2)
    residual = jnp.zeros((tokens, 4 * 2560), dtype=jnp.bfloat16)
    hc_gate = jnp.ones((tokens, 4), dtype=jnp.float32)

    q_norm = rand_bf16(256, scale=0.1)
    k_norm = rand_bf16(256, scale=0.1)
    iq_norm = rand_bf16(128, scale=0.1) + 1.0
    ik_norm = rand_bf16(128, scale=0.1) + 1.0
    norm_weights_packed = jnp.concatenate(
        [
            1.0 + k_norm.astype(jnp.float32),
            iq_norm.astype(jnp.float32),
            ik_norm.astype(jnp.float32),
        ],
        axis=0,
    )
    angle = jnp.arange(4096)[:, None] * jnp.arange(32)[None, :] / 8192
    trig = jnp.pad(
        jnp.concatenate([jnp.cos(angle), jnp.sin(angle)], 1), ((0, 0), (0, 64))
    ).astype(jnp.bfloat16)

    def _fast_step(h, pk, pos, mp, ip, rp, st, ctx, rids, mbt):
        return compact_pooled_qsa_fast_decode_step(
            h,
            pk,
            w_q,
            q_norm,
            norm_weights_packed,
            pos,
            trig,
            mp,
            ip,
            rp,
            st,
            ctx,
            rids,
            mbt,
            mbt,
            w_o_tiles,
            residual,
            hc_gate,
            4,
            2048,
            1e-6,
            64,
            0.375,
            2.5,
            0,
            ring_capacity=8,
            cache_tags=True,
        )

    run = jax.jit(_fast_step, donate_argnums=(3, 4, 5))
    verify_h = rand_bf16(tokens, 2560)
    next_h = rand_bf16(tokens, 2560)

    def step(owner, h, context, length, ring_ids=owners):
        starts = jnp.array([0] + [length] * rows, jnp.int32)
        contexts = jnp.array([context] + [0] * (rows - 1), jnp.int32)
        positions = jnp.arange(tokens, dtype=jnp.int32) + context
        pk = jnp.dot(h, w_in, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
        return run(h, pk, positions, *owner, starts, contexts, ring_ids, main)

    def copy_owners(values):
        return tuple(jax.device_put(v.copy()) for v in values)

    speculative = step(copy_owners(original), verify_h, prefix, k + 1)
    verified = tuple(np.asarray(o).copy() for o in speculative[:3])
    for accepted in range(k + 1):
        committed = copy_owners(original)
        for off in range(accepted + 1):
            h_step = jnp.zeros_like(verify_h).at[0].set(verify_h[off])
            committed = step(committed, h_step, prefix + off, 1)[:3]
        exp_main, exp_idx, _exp_ring, exp_out = step(
            committed, next_h, prefix + accepted + 1, 1
        )
        act_main, act_idx, _act_ring, act_out = step(
            copy_owners(verified), next_h, prefix + accepted + 1, 1
        )
        np.testing.assert_allclose(
            np.asarray(act_out)[0].astype(np.float32),
            np.asarray(exp_out)[0].astype(np.float32),
            rtol=0.03,
            atol=0.02,
        )
        end = prefix + accepted + 2
        np.testing.assert_array_equal(
            np.asarray(act_main)[[1, 2]].reshape(-1, 1024)[:end].view(np.uint8),
            np.asarray(exp_main)[[1, 2]].reshape(-1, 1024)[:end].view(np.uint8),
        )
        np.testing.assert_allclose(
            np.asarray(act_idx)[[1, 2]].reshape(-1, 128)[: end // 4].astype(np.float32),
            np.asarray(exp_idx)[[1, 2]].reshape(-1, 128)[: end // 4].astype(np.float32),
            rtol=1e-2,
            atol=1e-2,
        )

    idle_starts = jnp.zeros((rows + 1,), jnp.int32)
    idle_ctx = jnp.zeros((rows,), jnp.int32)
    idle_rings = jnp.zeros((rows,), jnp.int32)
    zero_bt = jnp.zeros_like(main)
    next_pk = jnp.dot(next_h, w_in, preferred_element_type=jnp.float32).astype(
        jnp.bfloat16
    )
    idle = run(
        next_h,
        next_pk,
        jnp.zeros((tokens,), jnp.int32),
        *copy_owners(verified),
        idle_starts,
        idle_ctx,
        idle_rings,
        zero_bt,
    )
    for after, before in zip(idle[:3], verified):
        np.testing.assert_array_equal(
            np.asarray(after).view(np.uint8), before.view(np.uint8)
        )
