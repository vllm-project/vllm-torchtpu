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
"""Gemma 4 dense sequence parallelism over real TP collectives (TP=8, one host).

`test_gemma4_patch.py` pins routing and math against threaded stand-ins; this
runs both branches of `gemma4_patch._dense_sequence_parallel_forward` on real
TPU ranks with real serialized per-channel `VllmFp8LinearMethodTPU` linears,
`fp8_linear_op` custom ops, per-rank `_tpu_sp_tp_rank` buffer slicing, and real
`reduce_scatter` / `uint8`+`f32` `all_gather` / `all_reduce` collectives (eager
and `torch.compile`).

Run on one host (4 chips x 2 cores = 8 ranks).
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.multichip, pytest.mark.spawns_tpu_workers]

REPO_ROOT = Path(__file__).resolve().parents[3]

WORLD_SIZE = 8
HIDDEN = 256
HEADS = 16
HEAD_DIM = 32
INTERMEDIATE = 512
NUM_LAYERS = 2
# 1024 // 8 == 128 == dense_sequence_parallel.MIN_TOKENS_PER_RANK (SP branch);
# 64 < 1024 takes the small-bucket upstream all-reduce branch.
SP_TOKENS = 1024
AR_TOKENS = 64
# Under the per-PR multichip lane's pytest --timeout=600.
LAUNCH_TIMEOUT_SECONDS = 480


def _build_layer(layer_idx: int, quant_config):
    import torch
    import torch.nn as nn
    from vllm.model_executor.layers.activation import get_act_and_mul_fn
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.linear import (
        MergedColumnParallelLinear,
        QKVParallelLinear,
        RowParallelLinear,
    )
    from vllm.model_executor.models.gemma4 import (
        Gemma4Attention,
        Gemma4DecoderLayer,
        Gemma4MLP,
    )

    gen = torch.Generator().manual_seed(20261126 + layer_idx * 97)

    def rand_fp8(out_dim: int, in_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Keep pre-RMSNorm activations O(1) so 1-ULP bf16 reduction-order
        # differences are not amplified by tiny norms.
        w = (torch.randn(out_dim, in_dim, generator=gen) * 0.5).to(torch.float8_e4m3fn)
        scale = torch.rand((out_dim, 1), generator=gen, dtype=torch.float32) * 0.4 + 0.3
        return w, scale

    qkv_proj = QKVParallelLinear(
        hidden_size=HIDDEN,
        head_size=HEAD_DIM,
        total_num_heads=HEADS,
        total_num_kv_heads=HEADS,
        bias=False,
        quant_config=quant_config,
        prefix=f"model.layers.{layer_idx}.self_attn.qkv_proj",
        params_dtype=torch.bfloat16,
    )
    o_proj = RowParallelLinear(
        input_size=HEADS * HEAD_DIM,
        output_size=HIDDEN,
        bias=False,
        quant_config=quant_config,
        prefix=f"model.layers.{layer_idx}.self_attn.o_proj",
        params_dtype=torch.bfloat16,
    )
    gate_up_proj = MergedColumnParallelLinear(
        input_size=HIDDEN,
        output_sizes=[INTERMEDIATE, INTERMEDIATE],
        bias=False,
        quant_config=quant_config,
        prefix=f"model.layers.{layer_idx}.mlp.gate_up_proj",
        params_dtype=torch.bfloat16,
    )
    down_proj = RowParallelLinear(
        input_size=INTERMEDIATE,
        output_size=HIDDEN,
        bias=False,
        quant_config=quant_config,
        prefix=f"model.layers.{layer_idx}.mlp.down_proj",
        params_dtype=torch.bfloat16,
    )

    for proj_id in ("q", "k", "v"):
        w_full, s_full = rand_fp8(HEADS * HEAD_DIM, HIDDEN)
        qkv_proj.weight_loader(qkv_proj.weight, w_full, proj_id)
        qkv_proj.weight_loader(qkv_proj.weight_scale, s_full, proj_id)
    o_w, o_s = rand_fp8(HIDDEN, HEADS * HEAD_DIM)
    o_proj.weight_loader(o_proj.weight, o_w)
    o_proj.weight_loader(o_proj.weight_scale, o_s)

    for shard_id in (0, 1):
        w_full, s_full = rand_fp8(INTERMEDIATE, HIDDEN)
        gate_up_proj.weight_loader(gate_up_proj.weight, w_full, shard_id)
        gate_up_proj.weight_loader(gate_up_proj.weight_scale, s_full, shard_id)
    d_w, d_s = rand_fp8(HIDDEN, INTERMEDIATE)
    down_proj.weight_loader(down_proj.weight, d_w)
    down_proj.weight_loader(down_proj.weight_scale, d_s)

    for proj in (qkv_proj, o_proj, gate_up_proj, down_proj):
        proj.to("tpu")
        assert proj.quant_method is not None
        proj.quant_method.process_weights_after_loading(proj)

    attn = Gemma4Attention.__new__(Gemma4Attention)
    nn.Module.__init__(attn)
    attn.config = SimpleNamespace(num_hidden_layers=NUM_LAYERS)
    attn.is_kv_shared_layer = False
    attn.num_heads = attn.num_kv_heads = HEADS // WORLD_SIZE
    attn.head_dim = HEAD_DIM
    attn.q_size = attn.kv_size = attn.num_heads * HEAD_DIM
    attn.qkv_proj = qkv_proj
    attn.o_proj = o_proj
    attn.q_norm = attn.k_norm = attn.v_norm = nn.Identity()
    attn.rotary_emb = lambda positions, q, k: (
        q + positions.unsqueeze(-1).to(q.dtype) * 0.0,
        k,
    )
    # Deterministic pure-TPU attention stand-in (no KV cache manager needed).
    attn.attn = lambda q, k, v: (torch.tanh(q) + (k * v) * 0.05).to(q.dtype)

    mlp = Gemma4MLP.__new__(Gemma4MLP)
    nn.Module.__init__(mlp)
    mlp.gate_up_proj = gate_up_proj
    mlp.down_proj = down_proj
    mlp.act_fn = get_act_and_mul_fn("gelu_pytorch_tanh")

    layer = Gemma4DecoderLayer.__new__(Gemma4DecoderLayer)
    nn.Module.__init__(layer)
    layer.layer_idx = layer_idx
    layer.self_attn = attn
    layer.mlp = mlp
    norms = []
    for _ in range(4):
        norm = RMSNorm(HIDDEN, eps=1e-6).to(torch.bfloat16)
        norm.weight.data.copy_(
            (1.0 + 0.1 * torch.randn(HIDDEN, generator=gen)).to(torch.bfloat16)
        )
        norms.append(norm)
    (
        layer.input_layernorm,
        layer.post_attention_layernorm,
        layer.pre_feedforward_layernorm,
        layer.post_feedforward_layernorm,
    ) = norms
    layer.enable_moe_block = False
    layer.per_layer_input_gate = None
    layer.register_buffer(
        "layer_scalar",
        torch.tensor([0.75 + 0.1 * layer_idx], dtype=torch.bfloat16),
    )
    layer.to("tpu")
    return layer


def _check(rank: int) -> dict:
    import torch
    import torch.nn as nn
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import initialize_model_parallel

    import vllm_torchtpu.layers  # noqa: F401 - registers TPU FP8 linear adapters
    from vllm_torchtpu.compilation import shape_variants
    from vllm_torchtpu.layers.adapter import dense_sequence_parallel
    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8Config
    from vllm_torchtpu.models.vllm import gemma4_patch

    with set_current_vllm_config(VllmConfig()):
        initialize_model_parallel(tensor_model_parallel_size=WORLD_SIZE)
        quant_config = VllmFp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
        )
        quant_config.is_channel_quant = True
        layers = nn.ModuleList(
            _build_layer(idx, quant_config) for idx in range(NUM_LAYERS)
        )
        for layer in layers:
            gemma4_patch._configure_layer(layer, num_hidden_layers=NUM_LAYERS)

        def run_patched(positions: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
            residual = None
            for layer in layers:
                hidden, residual = gemma4_patch._dense_sequence_parallel_forward(
                    layer, positions, hidden, residual
                )
            return hidden

        def run_upstream(positions: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
            residual = None
            for layer in layers:
                hidden, residual = gemma4_patch._orig_layer_forward(
                    layer, positions, hidden, residual
                )
            return hidden

        gen = torch.Generator().manual_seed(12345)
        x_sp = (torch.randn(SP_TOKENS, HIDDEN, generator=gen) * 0.5).to(
            dtype=torch.bfloat16, device="tpu"
        )
        pos_sp = torch.arange(SP_TOKENS, dtype=torch.int32, device="tpu")
        x_ar = (torch.randn(AR_TOKENS, HIDDEN, generator=gen) * 0.5).to(
            dtype=torch.bfloat16, device="tpu"
        )
        pos_ar = torch.arange(AR_TOKENS, dtype=torch.int32, device="tpu")

        out: dict = {}
        # 1. Exact checks on layer-0 token slicing and real uint8+f32 FP8 all-gather.
        l0 = layers[0]
        x_shard = dense_sequence_parallel.slice_token_shard(
            x_sp, SP_TOKENS, l0._tpu_sp_tp_rank, WORLD_SIZE
        )
        shard_size = SP_TOKENS // WORLD_SIZE
        out["slice_max_abs_err"] = float(
            (
                x_shard.cpu().float()
                - x_sp[rank * shard_size : (rank + 1) * shard_size].cpu().float()
            )
            .abs()
            .max()
        )
        qkv_sp = dense_sequence_parallel.quantize_all_gather_linear(
            l0.self_attn.qkv_proj, l0.input_layernorm(x_shard), SP_TOKENS
        )
        qkv_ar, _ = l0.self_attn.qkv_proj(l0.input_layernorm(x_sp))
        out["qkv_fp8_ag_max_abs_err"] = float(
            (qkv_sp.cpu().float() - qkv_ar.cpu().float()).abs().max()
        )

        # 2. Confirm layer 0 shards SP_TOKENS (eager and compiled) and keeps AR_TOKENS.
        out["sp_layer0_rows"] = int(
            gemma4_patch._dense_sequence_parallel_forward(l0, pos_sp, x_sp, None)[
                0
            ].shape[0]
        )
        out["ar_layer0_rows"] = int(
            gemma4_patch._dense_sequence_parallel_forward(l0, pos_ar, x_ar, None)[
                0
            ].shape[0]
        )
        torch._dynamo.reset()
        with shape_variants.warmup():
            fn_l0 = torch.compile(
                lambda p, h: gemma4_patch._dense_sequence_parallel_forward(
                    l0, p, h, None
                )[0],
                backend="tpu",
                fullgraph=True,
                dynamic=False,
            )
            out["compiled_sp_layer0_rows"] = int(fn_l0(pos_sp, x_sp).shape[0])

        # 3. Full 2-layer parity for SP (>= MIN_TOKENS_PER_RANK) and small-bucket AR.
        for how in ("eager", "compiled"):
            try:
                for name, pos, x in (("sp", pos_sp, x_sp), ("ar", pos_ar, x_ar)):
                    if how == "eager":
                        got = run_patched(pos, x).cpu().float()
                        want = run_upstream(pos, x).cpu().float()
                    else:
                        torch._dynamo.reset()
                        with shape_variants.warmup():
                            fn_patched = torch.compile(
                                run_patched,
                                backend="tpu",
                                fullgraph=True,
                                dynamic=False,
                            )
                            got = fn_patched(pos, x).cpu().float()
                        torch._dynamo.reset()
                        fn_upstream = torch.compile(
                            run_upstream, backend="tpu", fullgraph=True, dynamic=False
                        )
                        want = fn_upstream(pos, x).cpu().float()
                    diff = (got - want).abs()
                    cos = float(
                        torch.nn.functional.cosine_similarity(
                            got.reshape(1, -1), want.reshape(1, -1)
                        ).item()
                    )
                    out[f"{how}/{name}"] = {
                        "shape": list(got.shape),
                        "max_abs_err": float(diff.max()),
                        "mean_abs_err": float(diff.mean()),
                        "ref_max_abs": float(want.abs().max()),
                        "cosine_sim": cos,
                        "checksum": float(got.sum()),
                    }
            except Exception as exc:  # noqa: BLE001 - reported to the parent
                traceback.print_exc()
                out[f"{how}/error"] = repr(exc)
        return out


def test_gemma4_dense_sequence_parallel_on_real_collectives(tmp_path) -> None:
    result_dir = tmp_path / "gemma4_sp"
    result_dir.mkdir()
    from tests.distributed.tpu_test_utils import (
        launch_torchrun_workers,
        read_worker_results,
        torchrun_worker_env,
    )

    env = torchrun_worker_env(WORLD_SIZE)
    completed = launch_torchrun_workers(
        Path(__file__).resolve(),
        result_dir,
        world_size=WORLD_SIZE,
        env=env,
        cwd=REPO_ROOT,
        timeout=LAUNCH_TIMEOUT_SECONDS,
    )
    results = read_worker_results(result_dir, WORLD_SIZE, completed)
    for r in results:
        assert r["check"]["slice_max_abs_err"] == 0.0, r
        assert r["check"]["qkv_fp8_ag_max_abs_err"] == 0.0, r
        assert r["check"]["sp_layer0_rows"] == SP_TOKENS // WORLD_SIZE, r
        assert r["check"]["compiled_sp_layer0_rows"] == SP_TOKENS // WORLD_SIZE, r
        assert r["check"]["ar_layer0_rows"] == AR_TOKENS, r
    for how in ("eager", "compiled"):
        for r in results:
            assert f"{how}/error" not in r["check"], (r["rank"], r["check"])
        for name, expected_tokens in (("sp", SP_TOKENS), ("ar", AR_TOKENS)):
            checksums = []
            for r in results:
                case = r["check"][f"{how}/{name}"]
                assert case["shape"] == [expected_tokens, HIDDEN], (r["rank"], case)
                if name == "ar":
                    assert case["max_abs_err"] == 0.0, (r["rank"], how, case)
                else:
                    # Layer 0's token slice and uint8+f32 FP8 all-gather are bit-exact
                    # above. Across 2 full layers (4 dynamic e4m3fn quantizations),
                    # 1-ULP bf16 reduction-order diffs between all-reduce and
                    # reduce-scatter can cross FP8 bin boundaries.
                    assert case["cosine_sim"] >= 0.995, (r["rank"], how, case)
                    assert case["mean_abs_err"] <= 0.02 * max(
                        case["ref_max_abs"], 1.0
                    ), (r["rank"], how, case)
                checksums.append(case["checksum"])
            # All ranks must observe the same gathered output.
            assert max(checksums) - min(checksums) <= 1e-2 * max(
                abs(checksums[0]), 1.0
            ), (how, name, checksums)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.worker:
        sys.path.insert(0, str(REPO_ROOT))
        from tests.distributed.tpu_test_utils import run_torchrun_worker

        run_torchrun_worker(args.result_dir, _check)
