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
"""Exercise model projection/HC integration and mutable QSA state together."""

import math
from types import SimpleNamespace

import pytest
import torch
from qwen4_exp_test_utils import q4exp_text_config, qsa_test_pools


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "max_model_len,requests,pool_block_size",
    [(67584, 32, 128), (3328, 16, 128), (3328, 16, 3328)],
)
@torch.inference_mode()
def test_projected_decode_preserves_indices_and_cache(
    tmp_path, max_model_len, requests, pool_block_size
):
    from torch_tpu._internal.compile._backend import TpuBackend
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.forward_context import set_forward_context
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

    from vllm_torchtpu.models.vllm.qwen4_exp.qsa import QSAMetadata, Qwen4ExpQSA
    from vllm_torchtpu.utils import synchronize_tensors

    torch._dynamo.reset()
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    resolve_kv_cache_layout(config, [["LBNHC", "LBHNC"]])
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method="file://" + str(tmp_path / "gloo"),
            backend="gloo",
        )
        initialize_model_parallel(backend="gloo")
        try:
            layer = Qwen4ExpQSA(
                config=q4exp_text_config(),
                tp_rank=0,
                tp_size=1,
                # Exercise completed-group RoPE DMA together with the
                # prefetched output projection at the serving table size.
                max_model_len=max_model_len,
                cache_dtype="fp8_e4m3",
                device="cpu",
            )
            gen = torch.Generator().manual_seed(13581)
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
            initial, layout = qsa_test_pools(layer, 20, block_size=pool_block_size)
            layer.finalize_weights(block_size=initial[0].shape[1])
            from vllm_torchtpu.kernels.qwen4_exp.tuned_block_sizes import (
                output_projection_tile,
            )

            weight = layer.o_proj.weight
            tile = output_projection_tile(tuple(weight.shape))
            prefetch = (
                tile is not None
                and layer.tp_size == 1
                and weight.dtype == torch.bfloat16
            )
            if prefetch:
                assert tile is not None
                expected_shape = (
                    math.ceil(weight.shape[1] / tile),
                    weight.shape[0],
                    tile,
                )
                assert layer.output_weight_tiles.shape == expected_shape
                storage = layer.output_weight_tiles.data_ptr()
                layer.finalize_weights(block_size=layout.block_size)
                assert layer.output_weight_tiles.data_ptr() == storage
                torch.testing.assert_close(
                    layer.output_weight_tiles.permute(1, 0, 2).reshape(
                        weight.shape[0], -1
                    )[:, : weight.shape[1]],
                    layer.o_proj.weight,
                    rtol=0,
                    atol=0,
                )
            else:
                assert layer.output_weight_tiles is None
                assert layer._projected_op is None
            layer.load_weights(
                [("_k_scale", torch.tensor(0.375)), ("_v_scale", torch.tensor(2.5))]
            )
            from vllm.model_executor.layers.linear import LinearBase

            from vllm_torchtpu.layers.adapter.quantization.unquantized import (
                VllmUnquantizedLinearMethod,
            )

            for linear in layer.modules():
                if isinstance(linear, LinearBase):
                    linear.quant_method = VllmUnquantizedLinearMethod()
                    linear.quant_method.process_weights_after_loading(linear)
            columns = (max_model_len + initial[0].shape[1] - 1) // initial[0].shape[1]
            base_owners = [x.to("tpu") for x in initial]
            fused_owners = [x.to("tpu") for x in initial]
            layer.to("tpu")
            assert layer.k_norm_multiplier.dtype == torch.float32
            torch.testing.assert_close(
                layer.k_norm_multiplier.cpu(),
                1 + layer.k_norm.weight.detach().cpu().float(),
                rtol=0,
                atol=0,
            )
            shared_cu = torch.arange(17, dtype=torch.int32).to("tpu")

            from vllm_torchtpu.kernels.qwen4_exp.vllm_adapter import (
                build_gated_residual_op,
            )

            prefill_combine = build_gated_residual_op(4, 2560, 128)

            def baseline(hidden, positions, metadata, residual, gate):
                output, selected = layer.forward_with_indices(
                    hidden, positions, metadata
                )
                if hidden.shape[0] >= 1024:
                    return prefill_combine(residual, output, gate), selected
                return torch.cat(
                    [
                        residual[:, i * 2560 : (i + 1) * 2560].float()
                        + output.float() * gate[:, i : i + 1]
                        for i in range(4)
                    ],
                    dim=-1,
                ).bfloat16(), selected

            base = torch.compile(
                baseline, backend=TpuBackend(), fullgraph=True, dynamic=False
            )

            def routed_forward(hidden, positions, metadata, residual, gate):
                if layer.can_prefetch_output(hidden):
                    return layer.forward_with_indices(
                        hidden,
                        positions,
                        metadata,
                        residual,
                        gate,
                        hc_token_tile=128,
                        hc_min_tokens=1024,
                    )
                return baseline(hidden, positions, metadata, residual, gate)

            fused = torch.compile(
                routed_forward,
                backend=TpuBackend(),
                fullgraph=True,
                dynamic=False,
            )
            # Serving marks the token dimension dynamic. An equality guard
            # on bucket 16 must not specialize it back to a constant.
            layer.bind_pool(*base_owners)

            def output_route(hidden):
                return hidden + (1 if layer.can_prefetch_output(hidden) else 0)

            # This checks Dynamo's symbolic guard, before TPU lowering. The
            # standalone TPU backend cannot lower bounded dynamic outputs;
            # serving specializes them through its own shape-variant adaptor.
            routed = torch.compile(output_route, backend="eager", fullgraph=True)
            for rows in (16, 8, 32):
                hidden = torch.zeros(rows, 2560, dtype=torch.bfloat16).to("tpu")
                torch._dynamo.mark_dynamic(hidden, 0)
                actual = routed(hidden)
                synchronize_tensors([actual])
                torch.testing.assert_close(
                    actual.cpu(),
                    torch.full((rows, 2560), int(prefetch), dtype=torch.bfloat16),
                    rtol=0,
                    atol=0,
                )

            records = []
            cases = [
                (16, context, live, 1)
                for context, live in ((0, 8), (1, 8), (2, 8), (3, 8), (4, 16), (5, 0))
            ]
            cases += [
                (8, 0, 8, 1),
                (128, 0, 8, 16),
                (1024, 0, 8, 128),
                (16, 0, 2, 8),
                (16, 4, 4, (1, 2, 6, 7)),
            ]
            for tokens, context, live, query_len in cases:
                if tokens != 16 or not isinstance(query_len, int) or query_len > 1:
                    base_owners = [x.to("tpu") for x in initial]
                    fused_owners = [x.to("tpu") for x in initial]
                query_lengths = (
                    [query_len] * live
                    if isinstance(query_len, int)
                    else list(query_len)
                )
                valid_rows = sum(query_lengths)
                starts = torch.tensor(
                    [
                        0,
                        *torch.tensor(query_lengths).cumsum(0).tolist(),
                        *([valid_rows] * (requests - live)),
                    ],
                    dtype=torch.int32,
                )
                lengths = torch.zeros(requests, dtype=torch.int32)
                lengths[:live] = context
                slots = torch.arange(1, requests + 1, dtype=torch.int32)
                slots[live:] = 0
                pages = torch.zeros((requests, columns), dtype=torch.int32)
                pages[:live, 0] = torch.arange(1, live + 1)
                starts, lengths, slots, pages = [
                    x.to("tpu") for x in (starts, lengths, slots, pages)
                ]
                decode = sum(n == 1 for n in query_lengths)
                verify = decode + sum(n == 2 for n in query_lengths)
                distribution = torch.tensor(
                    [decode, verify, live], dtype=torch.int32
                ).to("tpu")
                metadata = QSAMetadata(starts, lengths, decode_cu=shared_cu)
                positions = torch.full((tokens,), context, dtype=torch.int64)
                if live:
                    positions[:valid_rows] = (
                        torch.cat([torch.arange(n) for n in query_lengths]) + context
                    )
                positions = positions.to("tpu")
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
                        request_distribution=distribution,
                        qsa_ring_slots=slots,
                    ),
                }
                with set_forward_context(per_layer, config, num_tokens=tokens):
                    layer.bind_pool(*base_owners)
                    expected, expected_ids = base(
                        hidden, positions, metadata, residual, gate
                    )
                    layer.bind_pool(*fused_owners)
                    assert layer.can_prefetch_output(hidden) == prefetch
                    actual, actual_ids = fused(
                        hidden,
                        positions,
                        metadata,
                        residual,
                        gate,
                    )
                synchronize_tensors(
                    [
                        actual,
                        expected,
                        actual_ids,
                        expected_ids,
                        *base_owners,
                        *fused_owners,
                    ]
                )
                a, b = actual.cpu().float(), expected.cpu().float()
                error = float((a - b).norm() / b.norm())
                assert torch.isfinite(a).all()
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                torch.testing.assert_close(
                    a[valid_rows:], residual.cpu().float()[valid_rows:], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    actual_ids.cpu(), expected_ids.cpu(), rtol=0, atol=0
                )
                for x, y in zip(base_owners, fused_owners):
                    assert torch.equal(
                        x.cpu().view(torch.uint8), y.cpu().view(torch.uint8)
                    )
                records.append(
                    dict(
                        tokens=tokens,
                        query_len=query_len,
                        context=context,
                        live=live,
                        relative_l2=error,
                        max_abs=float((a - b).abs().max()),
                    )
                )
            return records
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()
