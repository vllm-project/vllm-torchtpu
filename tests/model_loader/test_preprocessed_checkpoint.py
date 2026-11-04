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
"""Unit and end-to-end tests for preprocessed checkpoint format, CLI, and loader."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_torchtpu.model_loader.preprocessed_format import (
    dequantize_block_tensor,
    dtype_byte_size,
    format_raw_leaf_name,
    parse_raw_leaf_name,
    quantize_block_tensor,
    repack_columns_for_tp,
    repack_expert_paired_cols_for_ep_tp,
    repack_expert_rows_for_ep_tp,
    resolve_and_read_shard,
    write_raw_array,
)
from vllm_torchtpu.model_loader.preprocessed_loader import (
    is_preprocessed_checkpoint_dir,
    load_preprocessed_arrays_into_model,
)
from vllm_torchtpu.tools.preprocess_checkpoint import (
    main as cli_main,
)
from vllm_torchtpu.tools.preprocess_checkpoint import (
    read_safetensors_tensor,
    scan_safetensors_headers,
    write_safetensors_file,
)


# ---------------------------------------------------------------------------
# 1. Raw Leaf Filename Parsing and Formatting
# ---------------------------------------------------------------------------
@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "filename,expected_dtype,expected_dims,expected_ep,expected_tp,expected_fmt,expected_ls",
    [
        (
            "f8e4m3fn_512_2560_1280_tp1",
            "f8e4m3fn",
            (512, 2560, 1280),
            None,
            1,
            None,
            None,
        ),
        ("f32_512_1_1280_ep8tp1", "f32", (512, 1, 1280), 8, 1, None, None),
        ("bf16_320001536_160_tp0", "bf16", (320001536, 160), None, 0, None, None),
        ("s4_256_2880_5760_ep4tp2", "s4", (256, 2880, 5760), 4, 2, None, None),
        (
            "f8e4m3fn_512_2560_1280_ep8tp1_tpu7x_ls1",
            "f8e4m3fn",
            (512, 2560, 1280),
            8,
            1,
            "tpu7x",
            1,
        ),
    ],
)
def test_parse_and_format_raw_leaf_name(
    filename: str,
    expected_dtype: str,
    expected_dims: tuple[int, ...],
    expected_ep: int | None,
    expected_tp: int,
    expected_fmt: str | None,
    expected_ls: int | None,
) -> None:
    parsed = parse_raw_leaf_name(filename)
    assert parsed is not None
    assert parsed.dtype_token == expected_dtype
    assert parsed.dims == expected_dims
    assert parsed.ep_size == expected_ep
    assert parsed.tp_size == expected_tp
    assert parsed.tpu_format == expected_fmt
    assert parsed.linear_sublanes == expected_ls

    reformatted = format_raw_leaf_name(
        parsed.dtype_token,
        parsed.dims,
        parsed.tp_spec,
        tpu_format=parsed.tpu_format,
        linear_sublanes=parsed.linear_sublanes,
    )
    assert reformatted == filename


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "bad_name",
    [
        ".f8e4m3fn_4_8_tp1.tmp",
        "unknown_4_8_tp1",
        "bf16_tp1",
        "bf16_0_8_tp1",
        "bf16_4_8_ep0tp1",
        "bf16_4_8_invalid",
    ],
)
def test_parse_raw_leaf_name_rejects_invalid(bad_name: str) -> None:
    assert parse_raw_leaf_name(bad_name) is None


@pytest.mark.cpu_test
def test_dtype_byte_size() -> None:
    assert dtype_byte_size("f8e4m3fn", (512, 2560, 1280)) == 512 * 2560 * 1280
    assert dtype_byte_size("bf16", (10, 20)) == 10 * 20 * 2
    assert dtype_byte_size("f32", (4, 1, 8)) == 4 * 8 * 4
    assert dtype_byte_size("s4", (4, 8, 16)) == (4 * 8 * 16) // 2


# ---------------------------------------------------------------------------
# 2. Repacking & Contiguous Shard Reader
# ---------------------------------------------------------------------------
@pytest.mark.cpu_test
def test_repack_columns_for_tp_and_read(tmp_path: Path) -> None:
    src = torch.arange(64, dtype=torch.bfloat16).reshape(8, 8)
    repacked = repack_columns_for_tp(src, tp=4)
    assert repacked.shape == (4, 8, 2)
    write_raw_array(
        tmp_path,
        "misc",
        "layer.col_weight",
        repacked,
        "tp4",
        logical_dims=(8, 8),
    )
    array_dir = tmp_path / "misc" / "array_layer.col_weight"
    for r in range(4):
        shard = resolve_and_read_shard(array_dir, tp_size=4, tp_rank=r)
        assert torch.equal(shard, src[:, r * 2 : (r + 1) * 2])


@pytest.mark.cpu_test
@pytest.mark.parametrize("ep,tp", [(4, 1), (2, 2), (1, 4)])
def test_repack_expert_paired_cols_and_rows(tmp_path: Path, ep: int, tp: int) -> None:
    num_experts, hidden, inter = 4, 16, 8
    # K-major w13: [E, H, 2 * I]
    w13 = torch.arange(num_experts * hidden * (2 * inter), dtype=torch.float32).reshape(
        num_experts, hidden, 2 * inter
    )
    # K-major w2: [E, I, H]
    w2 = torch.arange(num_experts * inter * hidden, dtype=torch.float32).reshape(
        num_experts, inter, hidden
    )

    w13_repacked, w13_dims = repack_expert_paired_cols_for_ep_tp(w13, ep=ep, tp=tp)
    w2_repacked, w2_dims = repack_expert_rows_for_ep_tp(w2, ep=ep, tp=tp)

    write_raw_array(
        tmp_path,
        "experts",
        "model.layers.0.mlp.experts.w13_weight",
        w13_repacked,
        f"ep{ep}tp{tp}",
        logical_dims=w13_dims,
    )
    write_raw_array(
        tmp_path,
        "experts",
        "model.layers.0.mlp.experts.w2_weight",
        w2_repacked,
        f"ep{ep}tp{tp}",
        logical_dims=w2_dims,
    )

    w13_dir = tmp_path / "experts" / "array_model.layers.0.mlp.experts.w13_weight"
    w2_dir = tmp_path / "experts" / "array_model.layers.0.mlp.experts.w2_weight"

    e_per_ep = num_experts // ep
    i_per_tp = inter // tp
    for ep_r in range(ep):
        e_slice = slice(ep_r * e_per_ep, (ep_r + 1) * e_per_ep)
        for tp_r in range(tp):
            i_slice = slice(tp_r * i_per_tp, (tp_r + 1) * i_per_tp)
            w13_tile = resolve_and_read_shard(
                w13_dir, ep_size=ep, ep_rank=ep_r, tp_size=tp, tp_rank=tp_r
            )
            expected_gate = w13[e_slice, :, i_slice]
            up_start = inter + tp_r * i_per_tp
            up_end = inter + (tp_r + 1) * i_per_tp
            expected_up = w13[e_slice, :, up_start:up_end]
            expected_w13 = torch.cat([expected_gate, expected_up], dim=-1)
            assert torch.equal(w13_tile, expected_w13)

            w2_tile = resolve_and_read_shard(
                w2_dir, ep_size=ep, ep_rank=ep_r, tp_size=tp, tp_rank=tp_r
            )
            expected_w2 = w2[e_slice, i_slice, :]
            assert torch.equal(w2_tile, expected_w2)


@pytest.mark.cpu_test
def test_tp1_can_serve_any_ep_rank_directly(tmp_path: Path) -> None:
    """When tp_size == 1, _tp1 is already contiguous per ep_rank along dim 0."""
    w13 = torch.arange(8 * 4 * 6, dtype=torch.bfloat16).reshape(8, 4, 6)
    write_raw_array(
        tmp_path,
        "experts",
        "model.layers.0.mlp.experts.w13_weight",
        w13,
        "tp1",
    )
    w13_dir = tmp_path / "experts" / "array_model.layers.0.mlp.experts.w13_weight"
    for ep_r in range(4):
        shard = resolve_and_read_shard(
            w13_dir, ep_size=4, ep_rank=ep_r, tp_size=1, tp_rank=0
        )
        assert torch.equal(shard, w13[ep_r * 2 : (ep_r + 1) * 2])


# ---------------------------------------------------------------------------
# 3. End-to-End Qwen/Qwen3.8-Flash-Next-FP8 CLI + Loader Parity Test
# ---------------------------------------------------------------------------
class _StandInRoutedExperts(nn.Module):
    """Stand-in RoutedExperts module matching vLLM's parameter and config contract."""

    def __init__(
        self,
        *,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        ep_size: int = 1,
        ep_rank: int = 0,
        tp_size: int = 1,
        tp_rank: int = 0,
    ) -> None:
        super().__init__()
        e_local = num_experts // ep_size
        i_shard = intermediate_size // tp_size
        self.moe_config = SimpleNamespace(
            moe_parallel_config=SimpleNamespace(
                use_ep=ep_size > 1,
                ep_size=ep_size,
                ep_rank=ep_rank,
                tp_size=tp_size,
                tp_rank=tp_rank,
            )
        )
        self.w13_weight = nn.Parameter(
            torch.empty((e_local, 2 * i_shard, hidden_size), dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        self.w2_weight = nn.Parameter(
            torch.empty((e_local, hidden_size, i_shard), dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        self.w13_weight_scale_inv = nn.Parameter(
            torch.empty(
                (e_local, (2 * i_shard) // 128, hidden_size // 128),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.w2_weight_scale_inv = nn.Parameter(
            torch.empty(
                (e_local, hidden_size // 128, i_shard // 128),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )


class _StandInPLETable(nn.Module):
    """Stand-in Qwen4ExpPLETable matching vLLM's row-sharded PLE embedding table."""

    def __init__(
        self,
        *,
        padded_rows: int,
        org_rows: int,
        head_dim: int,
        tp_size: int,
        tp_rank: int,
    ) -> None:
        super().__init__()
        self.tp_size = tp_size
        rows_per_rank = padded_rows // tp_size
        start = tp_rank * rows_per_rank
        end = min(start + rows_per_rank, org_rows)
        num_org = max(0, end - start)
        self.shard_indices = SimpleNamespace(
            org_vocab_start_index=start,
            org_vocab_end_index=start + rows_per_rank,
            num_org_elements=num_org,
        )
        self.weight = nn.Parameter(
            torch.zeros((rows_per_rank, head_dim), dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        self.weight_scale = nn.Parameter(
            torch.ones((1,), dtype=torch.bfloat16),
            requires_grad=False,
        )


class _StandInNGramEmbedding(nn.Module):
    """Stand-in Qwen4ExpNGramEmbedding holding `.ngram_embedding` and hash buffers."""

    def __init__(
        self,
        *,
        padded_rows: int,
        org_rows: int,
        head_dim: int,
        tp_size: int,
        tp_rank: int,
    ) -> None:
        super().__init__()
        self.register_buffer("layer_multipliers", torch.zeros((3,), dtype=torch.int64))
        self.register_buffer(
            "ngram_heads_vocab_sizes", torch.zeros((4,), dtype=torch.int64)
        )
        self.register_buffer(
            "ngram_heads_offsets", torch.zeros((4,), dtype=torch.int64)
        )
        self.ngram_embedding = _StandInPLETable(
            padded_rows=padded_rows,
            org_rows=org_rows,
            head_dim=head_dim,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )


def _reference_moe_requant_for_rank(
    expert_tensors: dict[str, torch.Tensor],
    *,
    prefix: str,
    num_experts: int,
    hidden_size: int,
    inter_size: int,
    ep_size: int,
    ep_rank: int,
    tp_size: int,
    tp_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the exact runtime K-major tensors produced by
    `VllmFp8MoEMethodTPU.process_weights_after_loading` for `(ep_rank, tp_rank)`.
    """
    e_per_ep = num_experts // ep_size
    i_per_tp = inter_size // tp_size
    b_per_tp = i_per_tp // 128
    i_slice = slice(tp_rank * i_per_tp, (tp_rank + 1) * i_per_tp)
    b_slice = slice(tp_rank * b_per_tp, (tp_rank + 1) * b_per_tp)

    w13_list: list[torch.Tensor] = []
    w2_list: list[torch.Tensor] = []
    for eid in range(ep_rank * e_per_ep, (ep_rank + 1) * e_per_ep):
        g_w = expert_tensors[f"{prefix}.{eid}.gate_proj.weight"][i_slice, :]
        g_s = expert_tensors[f"{prefix}.{eid}.gate_proj.weight_scale_inv"][b_slice, :]
        u_w = expert_tensors[f"{prefix}.{eid}.up_proj.weight"][i_slice, :]
        u_s = expert_tensors[f"{prefix}.{eid}.up_proj.weight_scale_inv"][b_slice, :]
        d_w = expert_tensors[f"{prefix}.{eid}.down_proj.weight"][:, i_slice]
        d_s = expert_tensors[f"{prefix}.{eid}.down_proj.weight_scale_inv"][:, b_slice]

        g_f32 = dequantize_block_tensor(
            g_w.unsqueeze(0), g_s.unsqueeze(0), axis=(1, 2)
        ).squeeze(0)
        u_f32 = dequantize_block_tensor(
            u_w.unsqueeze(0), u_s.unsqueeze(0), axis=(1, 2)
        ).squeeze(0)
        d_f32 = dequantize_block_tensor(
            d_w.unsqueeze(0), d_s.unsqueeze(0), axis=(1, 2)
        ).squeeze(0)

        w13_list.append(torch.cat([g_f32, u_f32], dim=0))
        w2_list.append(d_f32)

    w13_f32 = torch.stack(w13_list, dim=0)
    w2_f32 = torch.stack(w2_list, dim=0)

    # Pad to 128 if i_per_tp is not a multiple of 128
    aligned_half = (i_per_tp + 127) // 128 * 128
    if aligned_half != i_per_tp:
        pad13 = w13_f32.new_zeros(
            w13_f32.shape[0], aligned_half - i_per_tp, hidden_size
        )
        w13_f32 = torch.cat(
            [w13_f32[:, :i_per_tp, :], pad13, w13_f32[:, i_per_tp:, :], pad13], dim=1
        )
        pad2 = w2_f32.new_zeros(w2_f32.shape[0], hidden_size, aligned_half - i_per_tp)
        w2_f32 = torch.cat([w2_f32, pad2], dim=2)

    w13_q, w13_s = quantize_block_tensor(
        w13_f32, quant_dtype=torch.float8_e4m3fn, axis=-1, block_size=None
    )
    w2_q, w2_s = quantize_block_tensor(
        w2_f32, quant_dtype=torch.float8_e4m3fn, axis=-1, block_size=None
    )

    w13_kmajor = w13_q.transpose(1, 2).contiguous()
    w13_s_4d = w13_s.transpose(1, 2).contiguous().to(torch.float32).unsqueeze(2)
    w2_kmajor = w2_q.transpose(1, 2).contiguous()
    w2_s_4d = w2_s.transpose(1, 2).contiguous().to(torch.float32).unsqueeze(2)
    return w13_kmajor, w13_s_4d, w2_kmajor, w2_s_4d


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "ep_size,tp_size",
    [
        (1, 1),
        (2, 1),
        (4, 1),
        (1, 2),
        (2, 2),
    ],
)
def test_qwen38_cli_pipeline_and_loader_bit_exact_parity(
    tmp_path: Path,
    ep_size: int,
    tp_size: int,
) -> None:
    """Create a synthetic Qwen/Qwen3.8-Flash-Next-FP8 checkpoint, run the CLI
    `pipeline` command, and verify bit-exact parity across all EP/TP ranks for:
      1. RoutedExperts (w13_weight, w13_weight_scale_inv, w2_weight,
         w2_weight_scale_inv) and `_tpu_preprocessed_loaded` fast-path flag.
      2. Qwen4ExpNGramEmbedding PLE table (sharded across TP ranks), weight_scale,
         and hash buffers.
      3. Consolidated dense safetensors weights.
    """
    raw_dir = tmp_path / "raw_qwen38_ckpt"
    preprocessed_dir = tmp_path / "preprocessed_qwen38_ckpt"
    raw_dir.mkdir(parents=True)

    num_experts = 4
    hidden_size = 256
    inter_size = 256
    block_h, block_w = 128, 128

    config_json = {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "model_type": "qwen4_exp",
        "text_config": {
            "vocab_size": 32,
            "hidden_size": 16,
            "hc_count": 4,
            "hc_lowrank": 8,
            "num_hidden_layers": 1,
            "layer_types": ["linear_attention"],
            "num_experts": num_experts,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": inter_size,
            "shared_expert_intermediate_size": inter_size,
            "ple_layer_ids": [1],
            "ngram_size": 3,
            "heads_per_ngram": 2,
            "ngram_vocab_size_base": 10,
            "make_ngram_vocab_size_divisible_by": 16,
            "split_ngram_parts": 3,
            "ple_embed_dim": 16,
            "ple_conv_kernel_size": 4,
            "eos_token_id": 3,
        },
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        },
    }
    (raw_dir / "config.json").write_text(json.dumps(config_json), encoding="utf-8")

    gen = torch.Generator().manual_seed(42)
    expert_tensors: dict[str, torch.Tensor] = {}
    prefix = "model.language_model.layers.0.mlp.experts"
    for eid in range(num_experts):
        for proj, shape, scale_shape in (
            (
                "gate_proj",
                (inter_size, hidden_size),
                (inter_size // block_h, hidden_size // block_w),
            ),
            (
                "up_proj",
                (inter_size, hidden_size),
                (inter_size // block_h, hidden_size // block_w),
            ),
            (
                "down_proj",
                (hidden_size, inter_size),
                (hidden_size // block_h, inter_size // block_w),
            ),
        ):
            w_fp32 = torch.randn(shape, generator=gen) * 0.5
            w_fp8 = w_fp32.to(torch.float8_e4m3fn)
            # When testing tp_size > 1, use uniform scale across intermediate
            # blocks so full-row vs per-shard w2 quantization is identical.
            if tp_size > 1 and proj == "down_proj":
                col_s = (
                    torch.rand((scale_shape[0], 1), generator=gen) * 0.25 + 0.05
                ).to(torch.bfloat16)
                s_bf16 = col_s.expand(scale_shape).contiguous()
                # Also make w_fp8 symmetric across intermediate halves so amax matches
                half_i = inter_size // 2
                w_fp8 = torch.cat([w_fp8[:, :half_i], w_fp8[:, :half_i]], dim=1)
            else:
                s_bf16 = (torch.rand(scale_shape, generator=gen) * 0.25 + 0.05).to(
                    torch.bfloat16
                )
            expert_tensors[f"{prefix}.{eid}.{proj}.weight"] = w_fp8
            expert_tensors[f"{prefix}.{eid}.{proj}.weight_scale_inv"] = s_bf16

    # Build PLE shards: 60 org rows, padded to a multiple of 16 -> 64 rows
    table_org_rows = 60
    padded_rows = math.ceil(table_org_rows / 16) * 16
    head_dim = 4
    full_ple_org = (torch.randn(table_org_rows, head_dim, generator=gen)).to(
        torch.float8_e4m3fn
    )
    ple_prefix = "model.language_model.layers.0.ple.ple_embedding"
    ple_tensors: dict[str, torch.Tensor] = {}
    for idx, shard_rows in enumerate(full_ple_org.split(22)):
        ple_tensors[f"{ple_prefix}.ngram_embedding.shard_{idx}.weight"] = shard_rows
    ple_scale = torch.tensor([0.125], dtype=torch.bfloat16)
    ple_tensors[f"{ple_prefix}.ngram_embedding.weight_scale"] = ple_scale
    ple_multipliers = torch.tensor([3, 5, 7], dtype=torch.int64)
    ple_sizes = torch.tensor([11, 13, 17, 19], dtype=torch.int64)
    ple_offsets = torch.tensor([0, 11, 24, 41], dtype=torch.int64)
    ple_tensors[f"{ple_prefix}.layer_multipliers"] = ple_multipliers
    ple_tensors[f"{ple_prefix}.ngram_heads_vocab_sizes"] = ple_sizes
    ple_tensors[f"{ple_prefix}.ngram_heads_offsets"] = ple_offsets

    dense_tensors: dict[str, torch.Tensor] = {
        "model.language_model.embed_tokens.weight": torch.randn(
            32, 16, generator=gen, dtype=torch.bfloat16
        ),
        "lm_head.weight": torch.randn(32, 16, generator=gen, dtype=torch.bfloat16),
    }

    write_safetensors_file(raw_dir / "model-00001-of-00002.safetensors", expert_tensors)
    write_safetensors_file(
        raw_dir / "model-00002-of-00002.safetensors",
        {**ple_tensors, **dense_tensors},
    )
    weight_map = {k: "model-00001-of-00002.safetensors" for k in expert_tensors}
    weight_map.update(
        {k: "model-00002-of-00002.safetensors" for k in (*ple_tensors, *dense_tensors)}
    )
    (raw_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8"
    )

    # Run CLI pipeline!
    rc = cli_main(
        [
            "pipeline",
            "--input-dir",
            str(raw_dir),
            "--output-dir",
            str(preprocessed_dir),
            "--ep",
            str(ep_size),
            "--tp",
            str(tp_size),
            "--verify",
        ]
    )
    assert rc == 0
    assert is_preprocessed_checkpoint_dir(preprocessed_dir)

    # 1. Verify dense index shrunk from 31 tensors down to ONLY the 2 dense tensors
    out_index = json.loads(
        (preprocessed_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    assert set(out_index["weight_map"].keys()) == set(dense_tensors.keys())
    dense_locs = scan_safetensors_headers(preprocessed_dir)
    for k, v in dense_tensors.items():
        assert torch.equal(read_safetensors_tensor(dense_locs[k]), v)

    # 2. Verify RoutedExperts bit-exact parity across every (ep_rank, tp_rank)
    for ep_rank in range(ep_size):
        for tp_rank in range(tp_size):
            ref_w13, ref_w13_s, ref_w2, ref_w2_s = _reference_moe_requant_for_rank(
                expert_tensors,
                prefix=prefix,
                num_experts=num_experts,
                hidden_size=hidden_size,
                inter_size=inter_size,
                ep_size=ep_size,
                ep_rank=ep_rank,
                tp_size=tp_size,
                tp_rank=tp_rank,
            )

            fast_experts = _StandInRoutedExperts(
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size=inter_size,
                ep_size=ep_size,
                ep_rank=ep_rank,
                tp_size=tp_size,
                tp_rank=tp_rank,
            )

            class _DummyModel(nn.Module):
                def __init__(self, exp: nn.Module) -> None:
                    super().__init__()
                    self.model = nn.Module()
                    self.model.layers = nn.ModuleList([nn.Module()])
                    self.model.layers[0].mlp = nn.Module()
                    self.model.layers[0].mlp.experts = exp

            wrapper = _DummyModel(fast_experts)
            loaded_names = load_preprocessed_arrays_into_model(
                wrapper, preprocessed_dir
            )
            assert "model.layers.0.mlp.experts.w13_weight" in loaded_names
            assert getattr(fast_experts, "_tpu_preprocessed_loaded", False) is True

            assert torch.equal(
                fast_experts.w13_weight.view(torch.uint8),
                ref_w13.view(torch.uint8),
            )
            assert torch.equal(
                fast_experts.w2_weight.view(torch.uint8),
                ref_w2.view(torch.uint8),
            )
            assert torch.equal(fast_experts.w13_weight_scale_inv, ref_w13_s)
            assert torch.equal(fast_experts.w2_weight_scale_inv, ref_w2_s)

    # 3. Verify PLE N-gram Embedding Table Parity across TP ranks
    full_ple_padded_u8 = torch.zeros(
        (padded_rows, head_dim), dtype=torch.uint8, device="cpu"
    )
    full_ple_padded_u8[:table_org_rows] = full_ple_org.view(torch.uint8)
    rows_per_rank = padded_rows // tp_size

    for tp_rank in range(tp_size):
        fast_ple = _StandInNGramEmbedding(
            padded_rows=padded_rows,
            org_rows=table_org_rows,
            head_dim=head_dim,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )

        class _DummyPLEModel(nn.Module):
            def __init__(self, ple_emb: nn.Module) -> None:
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([nn.Module()])
                self.model.layers[0].ple = nn.Module()
                self.model.layers[0].ple.ple_embedding = ple_emb

        ple_wrapper = _DummyPLEModel(fast_ple)
        loaded_ple_names = load_preprocessed_arrays_into_model(
            ple_wrapper, preprocessed_dir
        )
        assert (
            "model.layers.0.ple.ple_embedding.ngram_embedding.weight"
            in loaded_ple_names
        )
        expected_slice = full_ple_padded_u8[
            tp_rank * rows_per_rank : (tp_rank + 1) * rows_per_rank
        ]
        assert torch.equal(
            fast_ple.ngram_embedding.weight.view(torch.uint8),
            expected_slice,
        )
        assert torch.equal(fast_ple.ngram_embedding.weight_scale, ple_scale)
        assert torch.equal(fast_ple.layer_multipliers, ple_multipliers)
        assert torch.equal(fast_ple.ngram_heads_vocab_sizes, ple_sizes)
        assert torch.equal(fast_ple.ngram_heads_offsets, ple_offsets)


@pytest.mark.cpu_test
def test_mtp_draft_model_loads_mtp_experts_instead_of_target_layer(
    tmp_path: Path,
) -> None:
    """Verify that a standalone MTP draft model (e.g. Qwen4ExpMTP whose
    `named_modules()` yields `model.layers.0.mlp.experts.routed_experts`)
    resolves `array_mtp.layers.0.mlp.experts.*` instead of colliding with
    target `array_model.layers.0.mlp.experts.*`.
    """
    num_experts, hidden_size, inter_size = 4, 16, 8
    target_w13 = torch.full(
        (num_experts, hidden_size, 2 * inter_size),
        1.0,
        dtype=torch.float8_e4m3fn,
    )
    target_w2 = torch.full(
        (num_experts, inter_size, hidden_size),
        2.0,
        dtype=torch.float8_e4m3fn,
    )
    mtp_w13 = torch.full(
        (num_experts, hidden_size, 2 * inter_size),
        7.0,
        dtype=torch.float8_e4m3fn,
    )
    mtp_w2 = torch.full(
        (num_experts, inter_size, hidden_size),
        9.0,
        dtype=torch.float8_e4m3fn,
    )
    s13 = torch.ones((num_experts, 1, 2 * inter_size), dtype=torch.float32)
    s2 = torch.ones((num_experts, 1, hidden_size), dtype=torch.float32)

    for prefix, w13, w2 in (
        ("model.layers.0.mlp.experts", target_w13, target_w2),
        ("mtp.layers.0.mlp.experts", mtp_w13, mtp_w2),
    ):
        write_raw_array(tmp_path, "experts", f"{prefix}.w13_weight", w13, "tp1")
        write_raw_array(tmp_path, "experts", f"{prefix}.w2_weight", w2, "tp1")
        write_raw_array(
            tmp_path, "experts", f"{prefix}.w13_weight_scale_inv", s13, "tp1"
        )
        write_raw_array(tmp_path, "experts", f"{prefix}.w2_weight_scale_inv", s2, "tp1")

    class Qwen4ExpMTP(nn.Module):
        def __init__(self, exp: nn.Module) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Module()])
            self.model.layers[0].mlp = nn.Module()
            self.model.layers[0].mlp.experts = nn.Module()
            self.model.layers[0].mlp.experts.routed_experts = exp

    mtp_experts = _StandInRoutedExperts(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=inter_size,
    )
    mtp_model = Qwen4ExpMTP(mtp_experts)
    loaded = load_preprocessed_arrays_into_model(mtp_model, tmp_path)
    assert "model.layers.0.mlp.experts.routed_experts.w13_weight" in loaded
    assert torch.equal(
        mtp_experts.w13_weight.view(torch.uint8),
        mtp_w13.view(torch.uint8),
    )
    assert torch.equal(
        mtp_experts.w2_weight.view(torch.uint8),
        mtp_w2.view(torch.uint8),
    )
