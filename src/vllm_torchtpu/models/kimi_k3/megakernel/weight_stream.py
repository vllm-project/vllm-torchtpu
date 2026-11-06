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
"""Single-copy layer-by-layer Kimi K3 weight streaming into TP32 layout."""

from __future__ import annotations

import dataclasses
import importlib

import numpy as np
import torch

from vllm_torchtpu.megakernel_runtime.hooks import WEIGHT_CONSUMED

from .kernels import Config
from .kernels.load import BF16, PREFIX, pack_expert_bytes, selected_local_shapes

type _Slot = tuple[dict[int, int], int] | None
type _ExpertStage = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, set[str]]

TP_RANKS = 32
EP_GROUPS = 8
ETP_PARTS = 4
FALLBACK_MLA = 3
MAX_E8M0 = 255
PAIR_PARTS = 2
KDA_PROJ_PARTS = 5
KDA_CONV_STREAMS = 3
NUM_EXPERT_PROJS = 6
LAYER_KEY_PARTS = 3
EXPERT_KEY_PARTS = 7
DENSE_PAD = 1152
LATENT_ACTIVE = 28
LATENT_UP_WIDTH = 1024
LATENT_NORM_PAD = 512
KDA_PROJ_COLS = 1664
KDA_NORM_PAD = 7
QB_HEAD_WIDTH = 192
KB_HEAD_WIDTH = 256

KDA_OFFSETS = {"q_proj": 0, "k_proj": 384, "v_proj": 768, "g_proj": 1152}
KDA_CONVS = {"q_conv1d.weight": 0, "k_conv1d.weight": 1, "v_conv1d.weight": 2}
WHOLE_TRANSPOSE = {"gate": "router", "f_a_proj": "k_gate", "q_a_proj": "m_qa"}
ROW_SLICES = {
    "f_b_proj": ("k_fb", 384),
    "kv_b_proj": ("m_kb", 768),
    "g_proj": ("m_gate", 384),
}
ROW_NORMS = {
    "input_layernorm": "attn_norm",
    "post_attention_layernorm": "ffn_norm",
    "q_a_layernorm": "m_qnorm",
    "kv_a_layernorm": "m_knorm",
    "gate.e_score_correction_bias": "router_bias",
}
LM_KEYS = {"language_model.lm_head.weight", "model.lm_head.weight", "lm_head.weight"}
VALID_EXPERT_PROJS = frozenset({"w1", "w2", "w3"})
VALID_EXPERT_KINDS = frozenset({"weight_packed", "weight_scale"})


def _to_host_numpy(
    tensor: torch.Tensor,
    rank: int | None = None,
    width: int = 0,
    *,
    axis: int = 0,
) -> np.ndarray:
    """Slice optional shard `[rank*width:(rank+1)*width]` along `axis` to NumPy."""
    if rank is not None:
        dim_size = tensor.shape[axis]
        if width <= 0 or dim_size % width != 0 or (rank + 1) * width > dim_size:
            msg = (
                f"non-divisible shard shape {tuple(tensor.shape)} "
                f"on axis {axis} for index {rank} width {width}"
            )
            raise ValueError(msg)
        tensor = tensor.narrow(axis, rank * width, width)
    cpu_contig = tensor.detach().cpu().contiguous()
    if cpu_contig.dtype == torch.bfloat16:
        return cpu_contig.view(torch.uint16).numpy().view(BF16)
    return cpu_contig.numpy()


def _upload(host_array: np.ndarray, device_name: str) -> torch.Tensor:
    """Upload a host array to `[1, ...]` on `device_name` (`bridge.upload` on TPU)."""
    if device_name == "tpu":
        bridge_module = importlib.import_module(".runtime_shim.bridge", __package__)
        tpu_tensor: torch.Tensor = bridge_module.upload(host_array)
        return tpu_tensor
    contig = np.ascontiguousarray(host_array)
    if contig.dtype == BF16:
        bf16_tensor = torch.from_numpy(contig.view(np.uint16)).view(torch.bfloat16)
        return bf16_tensor.unsqueeze(0).clone().to(device_name)
    return torch.from_numpy(contig).unsqueeze(0).clone().to(device_name)


@dataclasses.dataclass(frozen=True, slots=True)
class StreamStats:
    """Telemetry snapshot of single-copy uploads and host staging bounds."""

    rank: int
    num_layers: int
    uploaded_tensors: int
    uploaded_bytes: int
    peak_host_staging_bytes: int
    peak_in_flight_layers: int
    active_host_staging_bytes: int


class RankWeightStreamer:
    """Per-rank repacker into `selected_local_shapes` and host staging bounds."""

    def __init__(self, rank: int, num_layers: int = 93, *, device: str = "tpu") -> None:
        """Initialize index maps and bounded host staging state for `rank`."""
        config = Config()
        self.rank = rank
        self.num_layers = num_layers
        self.device = device
        self._cfg = config
        self.expected_shapes = selected_local_shapes(
            num_layers, config.dim, vocab=config.vocab
        )
        full_attn = set(config.full_attention)
        mla_layers = [
            layer_idx for layer_idx in range(num_layers) if layer_idx in full_attn
        ] or [FALLBACK_MLA]
        kda_layers = [
            layer_idx for layer_idx in range(num_layers) if layer_idx not in full_attn
        ]
        self._norm_index = {layer_idx: layer_idx for layer_idx in range(num_layers)}
        self._moe_index = {
            layer_id: slot_idx for slot_idx, layer_id in enumerate(range(1, num_layers))
        }
        self._kda_index = {
            layer_id: slot_idx for slot_idx, layer_id in enumerate(kda_layers)
        }
        self._mla_index = {
            layer_id: slot_idx for slot_idx, layer_id in enumerate(mla_layers)
        }
        self._pairs: dict[tuple[int | None, str], dict[str, np.ndarray]] = {}
        self._multi: dict[tuple[int, str], tuple[np.ndarray, set[str]]] = {}
        self._experts: dict[int, _ExpertStage] = {}
        self._parts: dict[str, dict[int, torch.Tensor]] = {}
        self.device_tensors: dict[str, torch.Tensor] = {}
        self._uploaded_count = 0
        self._uploaded_bytes = 0
        self._peak_bytes = 0
        self._peak_layers = 0

    def _sync_peak(self) -> int:
        """Update peak staging bounds and return current active host bytes."""
        active_bytes = sum(
            int(host_array.nbytes)
            for pair_map in self._pairs.values()
            for host_array in pair_map.values()
        )
        active_bytes += sum(
            int(stage_buf.nbytes) for stage_buf, _ in self._multi.values()
        )
        active_bytes += sum(
            int(sum(stage_arr.nbytes for stage_arr in stage[:4]))
            for stage in self._experts.values()
        )
        pair_layers = {key[0] for key in self._pairs if key[0] is not None}
        active_layers = len(
            pair_layers | {key[0] for key in self._multi} | set(self._experts)
        )
        self._peak_bytes = max(self._peak_bytes, active_bytes)
        self._peak_layers = max(self._peak_layers, active_layers)
        return active_bytes

    def stats(self) -> StreamStats:
        """Return current upload and host staging telemetry."""
        active_bytes = self._sync_peak()
        return StreamStats(
            self.rank,
            self.num_layers,
            self._uploaded_count,
            self._uploaded_bytes,
            self._peak_bytes,
            self._peak_layers,
            active_bytes,
        )

    def verify_complete(self) -> None:
        """Assert stream completeness and clean staging buffers."""
        missing = set(self.expected_shapes) - set(self.device_tensors)
        if missing:
            sample = sorted(missing)[:3]
            msg = f"rank {self.rank}: missing {len(missing)} tensors: {sample}"
            raise ValueError(msg)
        if self._experts or self._multi or self._pairs or self._parts:
            msg = f"rank {self.rank}: unconsumed staging buffers remain"
            raise ValueError(msg)

    def finalize(self) -> dict[str, torch.Tensor]:
        """Verify stream completeness and return finalized per-rank device tensors."""
        self.verify_complete()
        return dict(self.device_tensors)

    def _put(self, family: str, host_array: np.ndarray, slot: _Slot = None) -> None:
        """Upload `host_array` and stack layered slices along `dim=1` when complete."""
        expected_slice = (
            self.expected_shapes[family]
            if slot is None
            else self.expected_shapes[family][1:]
        )
        if host_array.shape != expected_slice:
            msg = (
                f"rank {self.rank} {family}: shard shape {host_array.shape} "
                f"does not match expected {expected_slice}"
            )
            raise ValueError(msg)
        dev_tensor = _upload(
            host_array if slot is None else host_array[None, ...], self.device
        )
        self._uploaded_count += 1
        self._uploaded_bytes += int(dev_tensor.numel() * dev_tensor.element_size())
        if slot is None:
            self.device_tensors[family] = dev_tensor
            return
        index_map, layer_id = slot
        layer_parts = self._parts.setdefault(family, {})
        layer_parts[index_map[layer_id]] = dev_tensor
        if len(layer_parts) == len(index_map):
            self.device_tensors[family] = torch.cat(
                [layer_parts[slot_idx] for slot_idx in range(len(index_map))], 1
            )
            self._parts.pop(family, None)

    def _pair(
        self, key: tuple[int | None, str], part_tag: str, host_array: np.ndarray
    ) -> None:
        """Stage a 2-tensor family (`norm*proj` or `gate|up`) and flush when paired."""
        pair_map = self._pairs.setdefault(key, {})
        pair_map[part_tag] = host_array
        self._sync_peak()
        if len(pair_map) < PAIR_PARTS:
            return
        self._pairs.pop(key, None)
        layer_id, family = key
        index_map = self._moe_index if family == "shared_gu" else self._norm_index
        slot = (
            None if layer_id is None or family == "dense_gu" else (index_map, layer_id)
        )
        if "norm" in pair_map:
            self._put(
                family, (pair_map["norm"] * pair_map["proj"]).reshape(1, -1), slot
            )
        else:
            self._put(
                family, np.concatenate([pair_map["gate"], pair_map["up"]], 1), slot
            )

    def _simple(self, key: str, tensor: torch.Tensor, slot: _Slot) -> bool:
        """Consume direct whole-transpose, row-norm, or row-sliced families."""
        stem = key.removesuffix(".weight")
        if stem in WHOLE_TRANSPOSE and key.endswith(".weight"):
            self._put(WHOLE_TRANSPOSE[stem], _to_host_numpy(tensor).T.copy(), slot)
        elif stem in ROW_NORMS and (
            key.endswith(".weight") or key == "gate.e_score_correction_bias"
        ):
            self._put(ROW_NORMS[stem], _to_host_numpy(tensor).reshape(1, -1), slot)
        elif stem in ROW_SLICES and key.endswith(".weight"):
            if stem == "g_proj" and slot is not None and slot[0] is self._kda_index:
                return False
            family, width = ROW_SLICES[stem]
            self._put(family, _to_host_numpy(tensor, self.rank, width).T.copy(), slot)
        elif key == "o_proj.weight" and slot is not None:
            pair_idx, half_idx = divmod(self.rank, 2)
            half_dim = self._cfg.dim // 2
            pair_cols = (self._cfg.heads * self._cfg.head_dim) // 16
            sliced = tensor[
                half_idx * half_dim : (half_idx + 1) * half_dim,
                pair_idx * pair_cols : (pair_idx + 1) * pair_cols,
            ]
            host_array = _to_host_numpy(sliced)
            family = "k_o" if slot[0] is self._kda_index else "m_o"
            self._put(family, host_array.T.copy(), slot)
        else:
            return False
        return True

    def _parse_expert_key(
        self, layer_id: int, parts: list[str]
    ) -> tuple[int, str, str] | None:
        """Validate expert key parts and return `(expert_id, proj_name, kind)`."""
        if (
            layer_id not in self._moe_index
            or len(parts) != EXPERT_KEY_PARTS
            or not parts[4].isdigit()
        ):
            return None
        expert_id = int(parts[4])
        proj_name, kind = parts[5], parts[6]
        if (
            not 0 <= expert_id < self._cfg.experts
            or proj_name not in VALID_EXPERT_PROJS
            or kind not in VALID_EXPERT_KINDS
        ):
            return None
        return expert_id, proj_name, kind

    def _on_expert(self, layer_id: int, parts: list[str], tensor: torch.Tensor) -> bool:
        """Slice local `EP8/TP4` expert bytes and flush on layer completion."""
        parsed = self._parse_expert_key(layer_id, parts)
        if parsed is None:
            return False
        expert_id, proj_name, kind = parsed
        local_count = self._cfg.experts // EP_GROUPS
        quarter_hidden = self._cfg.expert_hidden // ETP_PARTS
        first_expert = local_count * (self.rank // ETP_PARTS)
        if not first_expert <= expert_id < first_expert + local_count:
            return True
        etp_part = self.rank % ETP_PARTS
        local_idx = expert_id - first_expert
        w2_step = (
            quarter_hidden // 2 if kind == "weight_packed" else quarter_hidden // 32
        )
        step = w2_step if proj_name == "w2" else quarter_hidden
        expert_axis = 1 if proj_name == "w2" else 0
        shard = _to_host_numpy(tensor, etp_part, step, axis=expert_axis)
        if layer_id not in self._experts:
            self._experts[layer_id] = (
                np.empty((local_count, 448, 1536), np.uint32),
                np.empty((local_count, 112, 1536), np.uint8),
                np.empty((local_count, 96, 3584), np.uint32),
                np.empty((local_count, 24, 3584), np.uint8),
                set(),
            )
        gate_up, gate_up_scale, down, down_scale, seen = self._experts[layer_id]
        cols = (
            slice(0, quarter_hidden)
            if proj_name == "w1"
            else slice(quarter_hidden, quarter_hidden * 2)
        )
        if kind == "weight_scale":
            if shard.dtype != np.uint8 or bool(
                np.any((shard == 0) | (shard == MAX_E8M0))
            ):
                msg = f"expert {local_idx} {proj_name}: invalid E8M0 scale"
                raise ValueError(msg)
            if proj_name == "w2":
                down_scale[local_idx] = shard.T
            else:
                gate_up_scale[local_idx, :, cols] = shard.T
        elif proj_name == "w2":
            down[local_idx] = pack_expert_bytes(shard)
        else:
            gate_up[local_idx, :, cols] = pack_expert_bytes(shard)
        seen.add(f"{local_idx}.{proj_name}.{kind}")
        if self._sync_peak() >= 0 and len(seen) == local_count * NUM_EXPERT_PROJS:
            self._experts.pop(layer_id, None)
            for family_name, stage_buf in (
                ("expert_gu", gate_up),
                ("expert_gus", gate_up_scale),
                ("expert_down", down),
                ("expert_ds", down_scale),
            ):
                self._put(family_name, stage_buf, (self._moe_index, layer_id))
        return True

    def _on_moe(self, layer_id: int, tail: str, tensor: torch.Tensor) -> bool:
        """Consume router, latent, and shared-expert tensors on MoE layers."""
        rank, config, slot = self.rank, self._cfg, (self._moe_index, layer_id)
        if self._simple(tail, tensor, slot):
            return True
        shared_slice = config.shared_hidden // TP_RANKS
        latent_slice = config.latent // LATENT_ACTIVE
        ep_block = rank % EP_GROUPS
        if tail in {
            "shared_experts.gate_proj.weight",
            "shared_experts.up_proj.weight",
        }:
            part_tag = "gate" if "gate" in tail else "up"
            self._pair(
                (layer_id, "shared_gu"),
                part_tag,
                _to_host_numpy(tensor, rank, shared_slice).T.copy(),
            )
        elif tail == "routed_expert_norm.weight":
            host_array = _to_host_numpy(tensor).astype(np.float32).reshape(1, -1)
            self._put(
                "latent_norm",
                np.pad(host_array, ((0, 0), (0, LATENT_NORM_PAD))),
                slot,
            )
        elif tail == "routed_expert_down_proj.weight":
            host_array = (
                np.zeros((config.dim, latent_slice), BF16)
                if rank >= LATENT_ACTIVE
                else _to_host_numpy(tensor, rank, latent_slice).T
            )
            self._put("latent_down", host_array.copy(), slot)
        elif tail == "routed_expert_up_proj.weight":
            zeros = np.zeros((config.latent, LATENT_UP_WIDTH), BF16)
            host_array = (
                zeros
                if (ep_block + 1) * LATENT_UP_WIDTH > config.dim
                else _to_host_numpy(tensor, ep_block, LATENT_UP_WIDTH).T
            )
            self._put("latent_up", host_array.copy(), slot)
        elif tail == "shared_experts.down_proj.weight":
            host_array = _to_host_numpy(tensor, rank, shared_slice, axis=1)
            self._put("shared_down", host_array.T.copy(), slot)
        else:
            return False
        return True

    def _on_kda(self, layer_id: int, key: str, tensor: torch.Tensor) -> bool:
        """Consume KDA projections, conv1d banks, and decay parameters."""
        rank, config, slot = self.rank, self._cfg, (self._kda_index, layer_id)
        if self._simple(key, tensor, slot):
            return True
        head_dim = config.head_dim
        heads_per_rank = config.heads // TP_RANKS
        rank_cols = heads_per_rank * head_dim
        stem = key.removesuffix(".weight")
        if key in KDA_CONVS:
            shape = (config.conv_size, KDA_CONV_STREAMS, heads_per_rank, head_dim)
            init_conv: tuple[np.ndarray, set[str]] = (np.empty(shape, BF16), set())
            stage_buf, seen = self._multi.setdefault((layer_id, "k_conv"), init_conv)
            conv_slice = _to_host_numpy(tensor, rank, rank_cols)[:, 0].T
            stage_buf[:, KDA_CONVS[key]] = conv_slice.reshape(
                config.conv_size, heads_per_rank, head_dim
            )
            seen.add(key)
            if self._sync_peak() >= 0 and len(seen) == KDA_CONV_STREAMS:
                self._put("k_conv", self._multi.pop((layer_id, "k_conv"))[0], slot)
        elif (stem in KDA_OFFSETS or stem == "b_proj") and key.endswith(".weight"):
            init_proj: tuple[np.ndarray, set[str]] = (
                np.zeros((config.dim, KDA_PROJ_COLS), BF16),
                set(),
            )
            stage_buf, seen = self._multi.setdefault(
                (layer_id, "k_projection"), init_proj
            )
            col_start, width = (
                (rank_cols * 4, heads_per_rank)
                if stem == "b_proj"
                else (KDA_OFFSETS[stem], rank_cols)
            )
            stage_buf[:, col_start : col_start + width] = _to_host_numpy(
                tensor, rank, width
            ).T
            seen.add(key)
            if self._sync_peak() >= 0 and len(seen) == KDA_PROJ_PARTS:
                self._put(
                    "k_projection",
                    self._multi.pop((layer_id, "k_projection"))[0],
                    slot,
                )
        elif key == "A_log":
            self._put(
                "k_a_log",
                _to_host_numpy(tensor, rank, heads_per_rank).reshape(heads_per_rank, 1),
                slot,
            )
        elif key == "dt_bias":
            self._put(
                "k_dt",
                _to_host_numpy(tensor, rank, rank_cols).reshape(
                    heads_per_rank, head_dim
                ),
                slot,
            )
        elif key == "o_norm.weight":
            padded = np.pad(
                _to_host_numpy(tensor).reshape(1, head_dim),
                ((0, KDA_NORM_PAD), (0, 0)),
            )
            self._put("k_norm", padded, slot)
        else:
            return False
        return True

    def _on_mla(self, layer_id: int, key: str, tensor: torch.Tensor) -> bool:
        """Consume MLA LoRA projections, output gate, and layernorms."""
        rank, config, slot = self.rank, self._cfg, (self._mla_index, layer_id)
        if self._simple(key, tensor, slot):
            return True
        heads_per_rank = config.heads // TP_RANKS
        if key == "kv_a_proj_with_mqa.weight":
            self._put(
                "m_ka",
                np.pad(_to_host_numpy(tensor).T, ((0, 0), (0, config.qk_extra))),
                slot,
            )
        elif key == "q_b_proj.weight":
            unpadded_qb = _to_host_numpy(
                tensor, rank, heads_per_rank * QB_HEAD_WIDTH
            ).T.reshape(-1, heads_per_rank, QB_HEAD_WIDTH)
            padded = np.pad(
                unpadded_qb, ((0, 0), (0, 0), (0, config.qk_extra))
            ).reshape(-1, heads_per_rank * KB_HEAD_WIDTH)
            self._put("m_qb", padded, slot)
        else:
            return False
        return True

    def _on_norms(self, layer_id: int, suffix: str, tensor: torch.Tensor) -> bool:
        """Consume layer-0 dense MLP weights and per-layer norms and residuals."""
        rank = self.rank
        dense_rows = self._cfg.dense_hidden // TP_RANKS
        if layer_id == 0 and suffix == "mlp.down_proj.weight":
            host_array = _to_host_numpy(tensor, rank, dense_rows, axis=1).T
            self._put(
                "dense_down", np.pad(host_array, ((0, DENSE_PAD - dense_rows), (0, 0)))
            )
            return True
        if layer_id == 0 and suffix in {
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
        }:
            padded = np.pad(
                _to_host_numpy(tensor, rank, dense_rows).T,
                ((0, 0), (0, DENSE_PAD - dense_rows)),
            )
            part_tag = "gate" if "gate" in suffix else "up"
            self._pair((0, "dense_gu"), part_tag, padded)
            return True
        if layer_id not in self._norm_index:
            return False
        if self._simple(suffix, tensor, (self._norm_index, layer_id)):
            return True
        if suffix in {
            "self_attention_res_norm.weight",
            "self_attention_res_proj.weight",
            "mlp_res_norm.weight",
            "mlp_res_proj.weight",
        }:
            family = "attn_res" if suffix.startswith("self_") else "mlp_res"
            part_tag = "norm" if suffix.endswith("_norm.weight") else "proj"
            self._pair(
                (layer_id, family),
                part_tag,
                _to_host_numpy(tensor).astype(np.float32),
            )
            return True
        return False

    def _consume_layer_tensor(
        self, layer_id: int, parts: list[str], tensor: torch.Tensor
    ) -> bool:
        """Dispatch a per-layer weight tensor and return True if recognized."""
        suffix = ".".join(parts[2:])
        if suffix.startswith("block_sparse_moe.experts."):
            return self._on_expert(layer_id, parts, tensor)
        if suffix.startswith("self_attn.") and layer_id in self._kda_index:
            return self._on_kda(layer_id, suffix.removeprefix("self_attn."), tensor)
        if suffix.startswith("self_attn.") and layer_id in self._mla_index:
            return self._on_mla(layer_id, suffix.removeprefix("self_attn."), tensor)
        if suffix.startswith("block_sparse_moe.") and layer_id in self._moe_index:
            return self._on_moe(
                layer_id, suffix.removeprefix("block_sparse_moe."), tensor
            )
        return self._on_norms(layer_id, suffix, tensor)

    def consume(self, name: str, tensor: torch.Tensor) -> object:
        """Consume one checkpoint tensor and return `WEIGHT_CONSUMED`."""
        norm_name = (
            "lm_head.weight"
            if name in LM_KEYS
            else name.removeprefix(PREFIX).removeprefix("model.")
        )
        vocab_slice = self._cfg.vocab // TP_RANKS
        if norm_name in {"embed_tokens.weight", "lm_head.weight"}:
            host_array = _to_host_numpy(tensor, self.rank, vocab_slice)
            family = "embedding" if norm_name.startswith("embed") else "lm_head"
            payload = (
                host_array if norm_name.startswith("embed") else host_array.T.copy()
            )
            self._put(family, payload)
            return WEIGHT_CONSUMED
        if norm_name == "norm.weight":
            self._put("final_norm", _to_host_numpy(tensor).reshape(1, -1))
            return WEIGHT_CONSUMED
        if norm_name in {
            "output_attn_res_norm.weight",
            "output_attn_res_proj.weight",
        }:
            part_tag = "norm" if norm_name.endswith("_norm.weight") else "proj"
            self._pair(
                (None, "output_res"),
                part_tag,
                _to_host_numpy(tensor).astype(np.float32),
            )
            return WEIGHT_CONSUMED
        if (
            norm_name.startswith("layers.")
            and len(parts := norm_name.split(".")) >= LAYER_KEY_PARTS
            and parts[1].isdigit()
            and self._consume_layer_tensor(int(parts[1]), parts, tensor)
        ):
            return WEIGHT_CONSUMED
        msg = f"unrecognized Kimi-K3 weight tensor: {name!r}"
        raise ValueError(msg)
