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
import gc
import importlib
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.model_loader_patches import (
    _compute_local_expert_ids as compute_local_expert_ids,
)
from vllm_torchtpu.models.kimi_k3.megakernel import adapter
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
from vllm_torchtpu.models.kimi_k3.megakernel.weight_stream import (
    RankWeightStreamer,
    _to_host_numpy,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    import pytest

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

    try:
        for rank in (0, 5, 10, 15, 31):
            part = rank % ETP_RANKS_PER_GROUP
            gate_up_ref, gate_up_scale_ref, down_ref, down_scale_ref = (
                checkpoint.experts(
                    1, 0, EXPERTS_PER_EP_GROUP, part=part, parts=ETP_RANKS_PER_GROUP
                )
            )
            exp_refs = {
                "expert_gu": np.asarray(gate_up_ref, np.uint32)[None, ...],
                "expert_gus": np.asarray(gate_up_scale_ref, np.uint8)[None, ...],
                "expert_down": np.asarray(down_ref, np.uint32)[None, ...],
                "expert_ds": np.asarray(down_scale_ref, np.uint8)[None, ...],
            }
            streamer = RankWeightStreamer(
                rank, num_layers=SYNTH_NUM_LAYERS, device="cpu"
            )
            ep_group = rank // ETP_RANKS_PER_GROUP
            other_ep_group = (ep_group + 1) % (SYNTH_TP_RANKS // ETP_RANKS_PER_GROUP)
            all_rank_spans = (
                *non_expert_spans,
                *expert_spans_by_ep[other_ep_group][:6],
                *expert_spans_by_ep[ep_group],
            )
            for key, shape, dt_tag, start_b, end_b in all_rank_spans:
                tensor = _tensor_view(memmap_buf, shape, dt_tag, start_b, end_b)
                assert streamer.consume(key, tensor) is None

            finalized = streamer.finalize()
            stats = streamer.stats()
            assert stats.active_host_staging_bytes == 0
            assert stats.peak_in_flight_layers == 1
            _verify_rank_finalized(
                rank,
                finalized,
                expected_shapes,
                checkpoint,
                exp_refs,
            )
            streamer.device_tensors.clear()
            finalized.clear()
            exp_refs.clear()
            gc.collect()
    finally:
        (tmp_path / "model.safetensors").unlink(missing_ok=True)


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


def test_ep8_weight_filter_skips_vision_and_retains_112_experts_for_k3_streamer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify EP8 112-expert filter widening, vision skipping, and CPU staging."""
    monkeypatch.setenv("VLLM_TPU_MEGAKERNEL", "1")
    monkeypatch.setattr(envs, "VLLM_TPU_MEGAKERNEL", "1", raising=False)
    for ep_rank in (0, 3, 4, 15, 31):
        local_ids = compute_local_expert_ids(896, ep_rank, 32)
        assert local_ids is not None
        assert len(local_ids) == EXPERTS_PER_EP_GROUP
        group_idx = ep_rank // ETP_RANKS_PER_GROUP
        assert local_ids == set(
            range(
                group_idx * EXPERTS_PER_EP_GROUP,
                (group_idx + 1) * EXPERTS_PER_EP_GROUP,
            )
        )

    streamer = RankWeightStreamer(0, num_layers=SYNTH_NUM_LAYERS, device="cpu")
    assert (
        streamer.consume(
            "vision_tower.encoder.layers.0.self_attn.q_proj.weight",
            torch.zeros((16, 16), dtype=torch.bfloat16),
        )
        is None
    )
    assert (
        streamer.consume(
            "model.mm_projector.0.weight",
            torch.zeros((16, 16), dtype=torch.bfloat16),
        )
        is None
    )
    assert (
        streamer.consume(
            f"model.layers.{SYNTH_NUM_LAYERS}.input_layernorm.weight",
            torch.zeros((Config().dim,), dtype=torch.bfloat16),
        )
        is None
    )
    assert streamer.stats().uploaded_tensors == 0

    adapter.clear_streamed_weights()
    adapter.clear_observed_weights()
    monkeypatch.setattr(
        adapter.jax,
        "devices",
        lambda: [SimpleNamespace(id=idx) for idx in range(SYNTH_TP_RANKS)],
    )
    monkeypatch.setattr(
        adapter.jax,
        "local_devices",
        lambda: [SimpleNamespace(id=0)],
    )
    cfg = Config()
    res = adapter.load_tensor(
        "model.norm.weight",
        torch.ones((cfg.dim,), dtype=torch.bfloat16),
        {},
    )
    assert res is None
    snap = adapter.streamed_weights_snapshot(0)
    assert "final_norm" in snap
    assert snap["final_norm"].device.type == "cpu"
    adapter.clear_streamed_weights()

    prefix_layers = 4
    prefix_shapes = selected_local_shapes(prefix_layers, cfg.dim, vocab=cfg.vocab)
    unlayered = {
        "embedding",
        "lm_head",
        "final_norm",
        "output_res",
        "dense_gu",
        "dense_down",
    }
    prefix_streamer = RankWeightStreamer(0, device="cpu")
    assert prefix_streamer.num_layers == cfg.layers
    vars(prefix_streamer)["_seen_layers"] = set(range(prefix_layers))
    for fam, shp in prefix_shapes.items():
        if fam in unlayered:
            prefix_streamer.device_tensors[fam] = torch.zeros(
                (1, *shp), dtype=torch.uint8
            )
        else:
            vars(prefix_streamer)["_parts"][fam] = {
                idx: torch.zeros((1, 1, *shp[1:]), dtype=torch.uint8)
                for idx in range(shp[0])
            }
    trimmed = prefix_streamer.finalize()
    assert prefix_streamer.num_layers == prefix_layers
    for fam, shp in prefix_shapes.items():
        assert tuple(trimmed[fam].shape) == (1, *shp)
    trimmed.clear()
    prefix_streamer.device_tensors.clear()

    incomplete_prefix = RankWeightStreamer(0, device="cpu")
    vars(incomplete_prefix)["_seen_layers"] = set(range(prefix_layers))
    for fam, shp in prefix_shapes.items():
        if fam in unlayered:
            incomplete_prefix.device_tensors[fam] = torch.zeros(
                (1, *shp), dtype=torch.uint8
            )
        else:
            slots = shp[0] - 1 if fam == "attn_norm" else shp[0]
            vars(incomplete_prefix)["_parts"][fam] = {
                idx: torch.zeros((1, 1, *shp[1:]), dtype=torch.uint8)
                for idx in range(slots)
            }
    with _assert_raises(ValueError, "missing"):
        incomplete_prefix.finalize()
    incomplete_prefix.device_tensors.clear()
    gc.collect()


def test_to_host_numpy_with_presharded_and_full_tensors() -> None:
    """Verify `_to_host_numpy` accepts pre-sharded tensors and slices full tensors."""
    presharded = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32)
    got_pre = _to_host_numpy(presharded, rank=5, width=3, axis=0)
    assert got_pre.shape == (3,)
    assert np.array_equal(got_pre, np.array([10.0, 20.0, 30.0], dtype=np.float32))

    full_tensor = torch.arange(96, dtype=torch.float32)
    got_full = _to_host_numpy(full_tensor, rank=5, width=3, axis=0)
    assert got_full.shape == (3,)
    assert np.array_equal(got_full, np.arange(15, 18, dtype=np.float32))

    with _assert_raises(ValueError, "non-divisible shard shape"):
        _to_host_numpy(
            torch.zeros((95,), dtype=torch.float32),
            rank=5,
            width=3,
            axis=0,
        )
    with _assert_raises(ValueError, "non-divisible shard shape"):
        _to_host_numpy(
            torch.zeros((12,), dtype=torch.float32),
            rank=5,
            width=3,
            axis=0,
        )

    streamer = RankWeightStreamer(5, num_layers=SYNTH_NUM_LAYERS, device="cpu")
    assert (
        streamer.consume(
            "model.layers.0.self_attn.A_log",
            torch.ones((3,), dtype=torch.bfloat16),
        )
        is None
    )
    streamer.device_tensors.clear()


def test_ep8_weight_filter_retains_both_tp_rank_and_device_position_experts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify EP8 filter retains TP-rank EP32 experts and position `p` EP8 experts."""
    vllm_cfg_mod = importlib.import_module("vllm.config")
    vllm_dist_mod = importlib.import_module("vllm.distributed")
    ep_mesh_mod = importlib.import_module("vllm_torchtpu.distributed.ep_mesh")

    monkeypatch.setenv("VLLM_TPU_MEGAKERNEL", "1")
    monkeypatch.setattr(envs, "VLLM_TPU_MEGAKERNEL", "1", raising=False)
    tpu7x_ids = tuple(h * 8 + d for h in range(4) for d in (0, 1, 4, 5, 6, 7, 2, 3))
    monkeypatch.setattr(ep_mesh_mod, "ep_device_ids", lambda: tpu7x_ids)
    sorted_ids = sorted(tpu7x_ids)
    fake_vllm_cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            is_moe=True,
            get_num_experts=lambda: 896,
        ),
        parallel_config=SimpleNamespace(
            enable_expert_parallel=True,
            enable_ep_weight_filter=True,
            enable_eplb=False,
            data_parallel_size=1,
            tensor_parallel_size=32,
            prefill_context_parallel_size=1,
            expert_placement_strategy="linear",
        ),
    )
    monkeypatch.setattr(vllm_cfg_mod, "get_current_vllm_config", lambda: fake_vllm_cfg)

    missing_explicit: list[tuple[int, int]] = []
    missing_cfg: list[tuple[int, int]] = []
    for q in range(SYNTH_TP_RANKS):
        p = sorted_ids.index(tpu7x_ids[q])
        ep32_experts = set(range(q * 28, (q + 1) * 28))
        streamer_group = p // ETP_RANKS_PER_GROUP
        streamer_experts = set(
            range(
                streamer_group * EXPERTS_PER_EP_GROUP,
                (streamer_group + 1) * EXPERTS_PER_EP_GROUP,
            )
        )
        explicit_ids = compute_local_expert_ids(896, q, 32)
        assert explicit_ids is not None
        assert ep32_experts.issubset(explicit_ids), (q, p)
        if not streamer_experts.issubset(explicit_ids):
            missing_explicit.append((q, p))

        monkeypatch.setattr(
            vllm_dist_mod,
            "get_tensor_model_parallel_rank",
            lambda rank=q: rank,
        )
        cfg_ids = compute_local_expert_ids()
        assert cfg_ids is not None
        assert ep32_experts.issubset(cfg_ids), (q, p)
        if not streamer_experts.issubset(cfg_ids):
            missing_cfg.append((q, p))

    assert not missing_explicit, (
        f"{len(missing_explicit)} of {SYNTH_TP_RANKS} ranks missing "
        f"streamer experts: {missing_explicit}"
    )
    assert not missing_cfg, (
        f"{len(missing_cfg)} of {SYNTH_TP_RANKS} ranks missing "
        f"streamer experts: {missing_cfg}"
    )
