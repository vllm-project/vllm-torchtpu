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
"""Unit tests for 32-rank Kimi-K3 weight streaming and fail-closed validation."""

from __future__ import annotations

import contextlib
import json
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm_torchtpu.megakernel_runtime import hooks as hooks_mod
from vllm_torchtpu.models.kimi_k3.megakernel.kernels import Config
from vllm_torchtpu.models.kimi_k3.megakernel.kernels.load import (
    BF16,
    PREFIX,
    Checkpoint,
    selected_local_shapes,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kernels.load import (
    _family_layers as canonical_family_layers,
)
from vllm_torchtpu.models.kimi_k3.megakernel.weight_stream import RankWeightStreamer

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

SYNTH_NUM_LAYERS = 2
SYNTH_TP_RANKS = 32
EXPERTS_PER_EP_GROUP = 112
ETP_RANKS_PER_GROUP = 4
EXPECTED_PARAMETER_FAMILIES = 37


@contextlib.contextmanager
def _assert_raises(exc_type: type[BaseException], match: str) -> Iterator[None]:
    """Assert that the block raises `exc_type` containing substring `match`."""
    caught: BaseException | None = None
    try:
        yield
    except exc_type as exc:
        caught = exc
    if caught is None:
        msg = f"Expected {exc_type.__name__} matching {match!r}"
        raise AssertionError(msg)
    assert match in str(caught), f"Expected {match!r} in {caught!r}"


def _max_bit_diff(got_tensor: torch.Tensor, ref_array: np.ndarray) -> float:
    """Compute max byte-wise diff between device tensor and NumPy reference."""
    squeezed = got_tensor.squeeze(0).cpu().contiguous()
    if squeezed.dtype == torch.bfloat16:
        got_arr = squeezed.view(torch.uint16).numpy().view(BF16)
    else:
        got_arr = squeezed.numpy()
    assert got_arr.shape == ref_array.shape, (got_arr.shape, ref_array.shape)
    got_raw = np.ascontiguousarray(got_arr).view(np.uint8)
    ref_raw = np.ascontiguousarray(ref_array).view(np.uint8)
    diff = np.abs(got_raw.astype(np.int32) - ref_raw.astype(np.int32))
    return float(np.max(diff))


def _tensor_view(
    memmap_buf: np.memmap,
    shape: tuple[int, ...],
    dtype_name: str,
    start_byte: int,
    end_byte: int,
) -> torch.Tensor:
    """Slice `memmap_buf[start_byte:end_byte]` as a zero-copy torch tensor."""
    raw = memmap_buf[start_byte:end_byte]
    if dtype_name == "BF16":
        arr_u16 = raw.view(np.uint16).reshape(shape)
        return torch.from_numpy(arr_u16).view(torch.bfloat16)
    if dtype_name == "F32":
        arr_f32 = raw.view(np.float32).reshape(shape)
        return torch.from_numpy(arr_f32)
    arr_u8 = raw.view(np.uint8).reshape(shape)
    return torch.from_numpy(arr_u8)


def _build_synthetic_entries(config: Config) -> list[tuple[str, tuple[int, ...], str]]:
    """Build checkpoint parameter tensor metadata covering all 37 parameter families."""
    hidden, latent, exp_hidden = config.dim, config.latent, config.expert_hidden
    entries: list[tuple[str, tuple[int, ...], str]] = [
        (f"{PREFIX}embed_tokens.weight", (config.vocab, hidden), "BF16"),
    ]
    for layer_idx in range(SYNTH_NUM_LAYERS):
        layer_prefix = f"{PREFIX}layers.{layer_idx}."
        entries.extend(
            [
                (f"{layer_prefix}input_layernorm.weight", (hidden,), "BF16"),
                (f"{layer_prefix}post_attention_layernorm.weight", (hidden,), "BF16"),
                (f"{layer_prefix}self_attention_res_norm.weight", (hidden,), "BF16"),
                (f"{layer_prefix}self_attention_res_proj.weight", (1, hidden), "F32"),
                (f"{layer_prefix}mlp_res_norm.weight", (hidden,), "BF16"),
                (f"{layer_prefix}mlp_res_proj.weight", (1, hidden), "F32"),
            ]
        )
        if layer_idx == 0:
            entries.extend(
                [
                    (
                        f"{layer_prefix}mlp.gate_proj.weight",
                        (config.dense_hidden, hidden),
                        "BF16",
                    ),
                    (
                        f"{layer_prefix}mlp.up_proj.weight",
                        (config.dense_hidden, hidden),
                        "BF16",
                    ),
                    (
                        f"{layer_prefix}mlp.down_proj.weight",
                        (hidden, config.dense_hidden),
                        "BF16",
                    ),
                ]
            )
        else:
            moe_prefix = f"{layer_prefix}block_sparse_moe."
            entries.extend(
                [
                    (f"{moe_prefix}gate.weight", (config.experts, hidden), "BF16"),
                    (
                        f"{moe_prefix}gate.e_score_correction_bias",
                        (config.experts,),
                        "BF16",
                    ),
                    (f"{moe_prefix}routed_expert_norm.weight", (latent,), "BF16"),
                    (
                        f"{moe_prefix}routed_expert_down_proj.weight",
                        (latent, hidden),
                        "BF16",
                    ),
                    (
                        f"{moe_prefix}routed_expert_up_proj.weight",
                        (hidden, latent),
                        "BF16",
                    ),
                    (
                        f"{moe_prefix}shared_experts.gate_proj.weight",
                        (config.shared_hidden, hidden),
                        "BF16",
                    ),
                    (
                        f"{moe_prefix}shared_experts.up_proj.weight",
                        (config.shared_hidden, hidden),
                        "BF16",
                    ),
                    (
                        f"{moe_prefix}shared_experts.down_proj.weight",
                        (hidden, config.shared_hidden),
                        "BF16",
                    ),
                ]
            )
            for expert_id in range(config.experts):
                exp_prefix = f"{moe_prefix}experts.{expert_id}."
                entries.extend(
                    [
                        (
                            f"{exp_prefix}w1.weight_packed",
                            (exp_hidden, latent // 2),
                            "U8",
                        ),
                        (
                            f"{exp_prefix}w1.weight_scale",
                            (exp_hidden, latent // 32),
                            "U8",
                        ),
                        (
                            f"{exp_prefix}w3.weight_packed",
                            (exp_hidden, latent // 2),
                            "U8",
                        ),
                        (
                            f"{exp_prefix}w3.weight_scale",
                            (exp_hidden, latent // 32),
                            "U8",
                        ),
                        (
                            f"{exp_prefix}w2.weight_packed",
                            (latent, exp_hidden // 2),
                            "U8",
                        ),
                        (
                            f"{exp_prefix}w2.weight_scale",
                            (latent, exp_hidden // 32),
                            "U8",
                        ),
                    ]
                )
        qkv_rows = config.heads * config.head_dim
        entries.extend(
            [
                (f"{layer_prefix}self_attn.q_proj.weight", (qkv_rows, hidden), "BF16"),
                (f"{layer_prefix}self_attn.k_proj.weight", (qkv_rows, hidden), "BF16"),
                (f"{layer_prefix}self_attn.v_proj.weight", (qkv_rows, hidden), "BF16"),
                (f"{layer_prefix}self_attn.g_proj.weight", (qkv_rows, hidden), "BF16"),
                (
                    f"{layer_prefix}self_attn.b_proj.weight",
                    (config.heads, hidden),
                    "BF16",
                ),
                (
                    f"{layer_prefix}self_attn.q_conv1d.weight",
                    (qkv_rows, 1, config.conv_size),
                    "BF16",
                ),
                (
                    f"{layer_prefix}self_attn.k_conv1d.weight",
                    (qkv_rows, 1, config.conv_size),
                    "BF16",
                ),
                (
                    f"{layer_prefix}self_attn.v_conv1d.weight",
                    (qkv_rows, 1, config.conv_size),
                    "BF16",
                ),
                (f"{layer_prefix}self_attn.A_log", (config.heads,), "BF16"),
                (f"{layer_prefix}self_attn.dt_bias", (qkv_rows,), "BF16"),
                (f"{layer_prefix}self_attn.f_a_proj.weight", (128, hidden), "BF16"),
                (
                    f"{layer_prefix}self_attn.f_b_proj.weight",
                    (qkv_rows, 128),
                    "BF16",
                ),
                (f"{layer_prefix}self_attn.o_norm.weight", (config.head_dim,), "BF16"),
                (f"{layer_prefix}self_attn.o_proj.weight", (hidden, qkv_rows), "BF16"),
            ]
        )
    mla_prefix = f"{PREFIX}layers.3.self_attn."
    qkv_rows = config.heads * config.head_dim
    entries.extend(
        [
            (f"{mla_prefix}q_a_proj.weight", (config.q_rank, hidden), "BF16"),
            (
                f"{mla_prefix}kv_a_proj_with_mqa.weight",
                (config.kv_rank + config.qk_extra, hidden),
                "BF16",
            ),
            (
                f"{mla_prefix}q_b_proj.weight",
                (config.heads * 192, config.q_rank),
                "BF16",
            ),
            (
                f"{mla_prefix}kv_b_proj.weight",
                (config.heads * 256, config.kv_rank),
                "BF16",
            ),
            (f"{mla_prefix}g_proj.weight", (qkv_rows, hidden), "BF16"),
            (f"{mla_prefix}o_proj.weight", (hidden, qkv_rows), "BF16"),
            (f"{mla_prefix}q_a_layernorm.weight", (config.q_rank,), "BF16"),
            (f"{mla_prefix}kv_a_layernorm.weight", (config.kv_rank,), "BF16"),
            (f"{PREFIX}output_attn_res_norm.weight", (hidden,), "BF16"),
            (f"{PREFIX}output_attn_res_proj.weight", (1, hidden), "F32"),
            (f"{PREFIX}norm.weight", (hidden,), "BF16"),
            ("language_model.lm_head.weight", (config.vocab, hidden), "BF16"),
        ]
    )
    return entries


def _write_synthetic_checkpoint(
    ckpt_dir: Path,
) -> tuple[list[tuple[str, tuple[int, ...], str, int, int]], int]:
    """Write a 2-layer Kimi-K3 safetensors checkpoint and return span metadata."""
    config = Config()
    entries = _build_synthetic_entries(config)
    hf_json = {
        "text_config": {
            "hidden_size": config.dim,
            "routed_expert_hidden_size": config.latent,
            "moe_intermediate_size": config.expert_hidden,
            "num_shared_experts": config.shared_hidden // config.expert_hidden,
            "intermediate_size": config.dense_hidden,
            "num_experts": config.experts,
            "num_experts_per_token": config.top_k,
            "num_hidden_layers": config.layers,
            "num_attention_heads": config.heads,
            "q_lora_rank": config.q_rank,
            "kv_lora_rank": config.kv_rank,
            "qk_rope_head_dim": config.qk_extra,
            "vocab_size": config.vocab,
            "attn_res_block_size": config.residual_block,
            "rms_norm_eps": config.eps,
            "activation_situ_beta": config.situ_beta,
            "activation_situ_linear_beta": config.situ_linear_beta,
            "mla_use_nope": True,
            "mla_use_output_gate": True,
            "latent_moe_use_norm": True,
            "moe_renormalize": True,
            "moe_router_activation_func": "sigmoid",
            "hidden_act": "situ",
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "num_expert_group": 1,
            "routed_scaling_factor": 1.0,
            "linear_attn_config": {
                "use_full_rank_gate": True,
                "head_dim": config.head_dim,
                "short_conv_kernel_size": config.conv_size,
                "gate_lower_bound": config.gate_lower_bound,
                "full_attn_layers": [idx + 1 for idx in config.full_attention],
            },
        }
    }
    (ckpt_dir / "config.json").write_text(json.dumps(hf_json), encoding="utf-8")

    header: dict[str, object] = {}
    offset = 0
    spans: list[tuple[str, tuple[int, ...], str, int, int]] = []
    dtype_map = {"BF16": ("BF16", 2), "U8": ("U8", 1), "F32": ("F32", 4)}
    for name, shape, dt_tag in entries:
        st_dtype, elem_bytes = dtype_map[dt_tag]
        num_bytes = int(np.prod(shape)) * elem_bytes
        header[name] = {
            "dtype": st_dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + num_bytes],
        }
        spans.append((name, shape, dt_tag, offset, offset + num_bytes))
        offset += num_bytes

    header_bytes = json.dumps(header).encode("utf-8")
    header_len = len(header_bytes)
    data_base = 8 + header_len
    st_path = ckpt_dir / "model.safetensors"
    with st_path.open("wb") as handle:
        handle.write(int(header_len).to_bytes(8, "little"))
        handle.write(header_bytes)
        handle.truncate(data_base + offset)
    mm_init = np.memmap(st_path, mode="r+", dtype=np.uint8, offset=data_base)
    for name, _shape, dt_tag, start_b, end_b in spans:
        byte_len = end_b - start_b
        if "weight_scale" in name or "e8m0" in name:
            mm_init[start_b:end_b] = 128
        elif dt_tag == "BF16":
            sample_elems = min(byte_len // 2, 16384)
            mm_init[start_b : start_b + sample_elems * 2].view(np.uint16)[:] = (
                np.arange(sample_elems, dtype=np.uint16)
            )
    mm_init.flush()

    index_json = {
        "metadata": {"total_size": spans[-1][-1] if spans else 0},
        "weight_map": {name: "model.safetensors" for name, *_ in entries},
    }
    (ckpt_dir / "model.safetensors.index.json").write_text(
        json.dumps(index_json), encoding="utf-8"
    )
    return spans, data_base


def _verify_rank_finalized(
    rank: int,
    finalized: dict[str, torch.Tensor],
    expected_shapes: dict[str, tuple[int, ...]],
    checkpoint: Checkpoint,
    exp_refs: dict[str, np.ndarray],
) -> None:
    """Assert every finalized parameter family matches `ckpt` bit-for-bit."""
    config = Config()
    assert set(finalized) == set(expected_shapes)
    for family, exp_shape in expected_shapes.items():
        got_tensor = finalized[family]
        assert tuple(got_tensor.shape) == (1, *exp_shape)
        if family in exp_refs:
            assert _max_bit_diff(got_tensor, exp_refs[family]) == 0.0
            continue
        layer_ids = canonical_family_layers(config, family, SYNTH_NUM_LAYERS)
        if layer_ids is None:
            ref_arr = checkpoint.dense_selected(family, 0, rank)
        else:
            ref_arr = np.stack(
                [checkpoint.dense_selected(family, l_id, rank) for l_id in layer_ids], 0
            )
        assert _max_bit_diff(got_tensor, ref_arr) == 0.0


def test_stream_weights_with_32rank_checkpoint_matches_bit_exact(
    tmp_path: Path,
) -> None:
    """Verify 32-rank RankWeightStreamer repack matches load.Checkpoint bit-exact."""
    config = Config()
    spans, data_base = _write_synthetic_checkpoint(tmp_path)
    checkpoint = Checkpoint(tmp_path)
    memmap_buf = np.memmap(
        tmp_path / "model.safetensors", mode="r+", dtype=np.uint8, offset=data_base
    )
    expected_shapes = selected_local_shapes(
        SYNTH_NUM_LAYERS, config.dim, vocab=config.vocab
    )
    assert len(expected_shapes) == EXPECTED_PARAMETER_FAMILIES

    non_expert_spans = [
        span for span in spans if ".block_sparse_moe.experts." not in span[0]
    ]
    expert_spans_by_ep: dict[int, list[tuple[str, tuple[int, ...], str, int, int]]] = {
        ep_idx: [] for ep_idx in range(SYNTH_TP_RANKS // ETP_RANKS_PER_GROUP)
    }
    for span in spans:
        if ".block_sparse_moe.experts." in span[0]:
            expert_id = int(span[0].removeprefix(PREFIX).split(".")[4])
            expert_spans_by_ep[expert_id // EXPERTS_PER_EP_GROUP].append(span)

    expert_ref_by_part: dict[int, dict[str, np.ndarray]] = {}
    for part in range(ETP_RANKS_PER_GROUP):
        gate_up_ref, gate_up_scale_ref, down_ref, down_scale_ref = checkpoint.experts(
            1, 0, EXPERTS_PER_EP_GROUP, part=part, parts=ETP_RANKS_PER_GROUP
        )
        expert_ref_by_part[part] = {
            "expert_gu": np.asarray(gate_up_ref, np.uint32)[None, ...],
            "expert_gus": np.asarray(gate_up_scale_ref, np.uint8)[None, ...],
            "expert_down": np.asarray(down_ref, np.uint32)[None, ...],
            "expert_ds": np.asarray(down_scale_ref, np.uint8)[None, ...],
        }

    for rank in range(SYNTH_TP_RANKS):
        streamer = RankWeightStreamer(rank, num_layers=SYNTH_NUM_LAYERS, device="cpu")
        ep_group = rank // ETP_RANKS_PER_GROUP
        other_ep_group = (ep_group + 1) % (SYNTH_TP_RANKS // ETP_RANKS_PER_GROUP)
        all_rank_spans = (
            *non_expert_spans,
            *expert_spans_by_ep[other_ep_group][:6],
            *expert_spans_by_ep[ep_group],
        )
        for key, shape, dt_tag, start_b, end_b in all_rank_spans:
            tensor = _tensor_view(memmap_buf, shape, dt_tag, start_b, end_b)
            assert streamer.consume(key, tensor) is hooks_mod.WEIGHT_CONSUMED

        finalized = streamer.finalize()
        stats = streamer.stats()
        assert stats.active_host_staging_bytes == 0
        assert stats.peak_in_flight_layers == 1
        _verify_rank_finalized(
            rank,
            finalized,
            expected_shapes,
            checkpoint,
            expert_ref_by_part[rank % ETP_RANKS_PER_GROUP],
        )
        streamer.device_tensors.clear()
        finalized.clear()


def test_consume_with_unknown_weight_name_raises_value_error() -> None:
    """Verify consume raises ValueError on unrecognized layer or root weight keys."""
    streamer = RankWeightStreamer(0, num_layers=SYNTH_NUM_LAYERS, device="cpu")
    with _assert_raises(ValueError, "unrecognized"):
        streamer.consume(
            "model.layers.0.self_attn.unknown_proj.weight",
            torch.zeros((16, 16), dtype=torch.bfloat16),
        )
    with _assert_raises(ValueError, "unrecognized"):
        streamer.consume(
            "model.unknown_root_weight",
            torch.zeros((16, 16), dtype=torch.bfloat16),
        )


def test_finalize_with_unconsumed_required_tensors_raises_value_error() -> None:
    """Verify finalize raises ValueError when required families or pairs are missing."""
    config = Config()
    streamer = RankWeightStreamer(0, num_layers=SYNTH_NUM_LAYERS, device="cpu")
    streamer.consume(
        "model.norm.weight",
        torch.ones((config.dim,), dtype=torch.bfloat16),
    )
    with _assert_raises(ValueError, "missing"):
        streamer.finalize()


def test_consume_with_non_divisible_shard_dim_raises_value_error() -> None:
    """Verify consume raises ValueError on non-divisible row or column shard shapes."""
    config = Config()
    streamer = RankWeightStreamer(0, num_layers=SYNTH_NUM_LAYERS, device="cpu")
    with _assert_raises(ValueError, "non-divisible shard shape"):
        streamer.consume(
            "model.embed_tokens.weight",
            torch.zeros((config.vocab - 1, config.dim), dtype=torch.bfloat16),
        )
    with _assert_raises(ValueError, "non-divisible shard shape"):
        streamer.consume(
            "model.layers.0.mlp.down_proj.weight",
            torch.zeros((config.dim, config.dense_hidden - 1), dtype=torch.bfloat16),
        )
    with _assert_raises(ValueError, "non-divisible shard shape"):
        streamer.consume(
            "model.layers.1.block_sparse_moe.experts.0.w2.weight_packed",
            torch.zeros(
                (config.latent, config.expert_hidden // 2 - 1), dtype=torch.uint8
            ),
        )
    with _assert_raises(ValueError, "does not match expected"):
        streamer.consume(
            "model.norm.weight",
            torch.ones((config.dim - 1,), dtype=torch.bfloat16),
        )
