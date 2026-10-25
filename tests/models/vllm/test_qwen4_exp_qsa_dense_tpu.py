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
"""Whole native layer against independent CPU Torch dense gated GQA."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from qwen4_exp_test_utils import q4exp_text_config, qsa_test_pools


def _dense_step(hidden, weights, positions, prior, fp8):
    """CPU-only oracle: no production QSA producer, indexer or attention helper."""
    dim, heads = 128, 2
    qgate = F.linear(hidden, weights["q_proj.weight"]).reshape(-1, heads, 2 * dim)
    key = F.linear(hidden, weights["k_proj.weight"]).reshape(-1, 1, dim)
    value = F.linear(hidden, weights["v_proj.weight"]).reshape(-1, 1, dim)

    def normalize(x, weight):
        x = x.float()
        return (
            x
            * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
            * (1 + weight.float())
        ).to(torch.bfloat16)

    def rotary(x):
        rotary_dim = 32
        angles = positions.float()[:, None] * 10000000.0 ** (
            -torch.arange(0, rotary_dim, 2).float() / rotary_dim
        )
        cos = angles.cos().to(torch.bfloat16).float()[:, None]
        sin = angles.sin().to(torch.bfloat16).float()[:, None]
        a, b = x[..., :16].float(), x[..., 16:32].float()
        return torch.cat(
            (
                (a * cos - b * sin).to(torch.bfloat16),
                (b * cos + a * sin).to(torch.bfloat16),
                x[..., 32:],
            ),
            -1,
        )

    query = rotary(normalize(qgate[..., :dim], weights["q_norm.weight"]))
    key = rotary(normalize(key, weights["k_norm.weight"]))
    if fp8:
        key = (key.float() / 0.375).clamp(-448, 448).to(
            torch.float8_e4m3fn
        ).float() * 0.375
        value = (value.float() / 2.5).clamp(-448, 448).to(
            torch.float8_e4m3fn
        ).float() * 2.5
    else:
        key, value = key.float(), value.float()
    if prior is not None:
        key = torch.cat((prior[0], key))
        value = torch.cat((prior[1], value))
    scores = (
        torch.einsum("thd,shd->hts", query.float(), key.expand(-1, heads, -1))
        / dim**0.5
    )
    future = torch.arange(len(key))[None, :] > positions[:, None]
    probabilities = scores.masked_fill(future[None], -torch.inf).softmax(-1)
    attended = torch.einsum("hts,shd->thd", probabilities, value.expand(-1, heads, -1))
    gate = qgate[..., dim:].float().sigmoid().to(torch.bfloat16)
    gated = (attended.to(torch.bfloat16).float() * gate.float()).to(torch.bfloat16)
    return F.linear(gated.flatten(1), weights["o_proj.weight"]), (key, value)


@pytest.mark.tpu_test
@pytest.mark.parametrize("fp8", [False, True])
@torch.inference_mode()
def test_complete_layer_dense_context_against_plain_torch(fp8, tmp_path):
    from torch_tpu._internal.compile._backend import TpuBackend
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

    from vllm_torchtpu.models.vllm.qwen4_exp.qsa import QSAMetadata, Qwen4ExpQSA
    from vllm_torchtpu.utils import synchronize_tensors

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
            cfg = q4exp_text_config(
                hidden_size=128,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=128,
                indexer_n_heads=2,
                indexer_head_dim=128,
                indexer_budget=2048,
                partial_rotary_factor=0.25,
            )
            model = Qwen4ExpQSA(
                config=cfg,
                tp_rank=0,
                tp_size=1,
                max_model_len=2048,
                cache_dtype="fp8_e4m3" if fp8 else "auto",
                device="cpu",
            )
            generator = torch.Generator().manual_seed(1965)
            weights = {
                name: (torch.randn(p.shape, generator=generator) * 0.025).to(
                    torch.bfloat16
                )
                for name, p in model.named_parameters()
            }
            model.load_weights(weights.items())
            model.finalize_weights()
            if fp8:
                model.load_weights(
                    [("_k_scale", torch.tensor(0.375)), ("_v_scale", torch.tensor(2.5))]
                )
            cpu_owners, _ = qsa_test_pools(model, 20)
            owners = [owner.to("tpu") for owner in cpu_owners]
            model.to("tpu")
            model.bind_pool(*owners)
            identities = [id(x) for x in owners]
            compiled = torch.compile(
                model.forward_with_indices,
                backend=TpuBackend(),
                fullgraph=True,
                dynamic=False,
            )
            previous = None
            for context, count, bucket in [(0, 2047, 2048), (2047, 1, 8)]:
                hidden = (torch.randn((bucket, 128), generator=generator) * 0.2).to(
                    torch.bfloat16
                )
                positions = torch.zeros(bucket, dtype=torch.int64)
                positions[:count] = torch.arange(context, context + count)
                expected, previous = _dense_step(
                    hidden[:count], weights, positions[:count], previous, fp8
                )
                starts = torch.tensor([0] + [count] * 8, dtype=torch.int32)
                lengths = torch.tensor([context] + [0] * 7, dtype=torch.int32)
                slots = torch.tensor([0] + [-1] * 7, dtype=torch.int32)
                pages = torch.zeros((8, 16), dtype=torch.int32)
                pages[0] = torch.arange(1, 17)
                metadata = QSAMetadata(
                    *(
                        x.to("tpu")
                        for x in (starts, lengths, slots, pages, pages.clone())
                    )
                )
                output, selected = compiled(
                    hidden.to("tpu"), positions.to("tpu"), metadata
                )
                synchronize_tensors([output, selected, *owners])
                actual = output.cpu().float().numpy()
                expected_np = expected.float().numpy()
                error = float(
                    np.linalg.norm(actual[:count] - expected_np)
                    / np.linalg.norm(expected_np)
                )
                ids = selected.cpu().numpy()
                out = Path(os.environ.get("QWEN4_REVIEW_OUTPUT", tmp_path))
                out.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    out / f"dense-{fp8}-{context}.npz",
                    actual=actual,
                    expected=expected_np,
                    selected=ids,
                    positions=positions.numpy(),
                    relative_l2=error,
                )
                assert np.isfinite(actual).all() and error < 0.01
                np.testing.assert_array_equal(actual[count:], 0)
                # Budget2048 / ratio4 holds every complete group at this context;
                # partial groups are appended separately. Check that guarantee, too.
                for row in range(count):
                    np.testing.assert_array_equal(
                        np.sort(ids[row][ids[row] >= 0]), np.arange(context + row + 1)
                    )
                np.testing.assert_array_equal(ids[count:], -1)
                assert [id(x) for x in model.cache_owners] == identities
                assert (
                    torch.count_nonzero(owners[0][[0, 17, 18, 19]].cpu().float()) == 0
                )
                # Index owner page0 holds the active ring at independent ID0;
                # unreferenced pages remain untouched in both physical owners.
                assert torch.count_nonzero(owners[1][[17, 18, 19]].cpu()) == 0
                print(
                    f"DENSE_LAYER fp8={fp8} context={context} "
                    f"count={count} relative_l2={error}"
                )
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()
