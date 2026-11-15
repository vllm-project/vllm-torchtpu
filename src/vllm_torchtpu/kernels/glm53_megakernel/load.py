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
"""Streaming safetensors loader and EP8xTP1 + TP8 sharding for GLM-5.3-Flash."""

from __future__ import annotations

import functools
import json
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, SingleDeviceSharding
from jax.sharding import PartitionSpec as P

from .config import Glm53MegakernelConfig

BF16 = np.dtype(jnp.bfloat16)
FP8_E4M3 = np.dtype(jnp.float8_e4m3fn)
F32 = np.dtype(np.float32)

_SAFETENSOR_DTYPES: dict[str, np.dtype[Any]] = {
    "F8_E4M3": FP8_E4M3,
    "BF16": BF16,
    "F32": F32,
    "F16": np.dtype(np.float16),
    "U8": np.dtype(np.uint8),
    "I32": np.dtype(np.int32),
    "I64": np.dtype(np.int64),
}


def expected_mtp_rank_specs(
    cfg: Glm53MegakernelConfig, world_size: int = 8
) -> dict[str, tuple[tuple[int, ...], np.dtype[Any]]]:
    """Return per-rank (shape, dtype) for the single NextN MTP layer (`layers.{L}`)."""
    cfg.validate(world_size=world_size)
    h = cfg.hidden_size
    hl = h // world_size
    h_mla_loc = cfg.mla_num_heads // world_size
    q_lora = cfg.q_lora_rank
    kv_lora = cfg.kv_lora_rank
    d_qk = cfg.qk_nope_head_dim
    d_v = cfg.v_head_dim
    p_mla_loc = h_mla_loc * d_qk
    i_sh_loc = (cfg.n_shared_experts * cfg.moe_intermediate_size) // world_size
    e_tot = cfg.n_routed_experts
    e_loc = e_tot // world_size
    i_moe = cfg.moe_intermediate_size
    bh, bw = cfg.weight_block_size
    return {
        "mtp_enorm": ((h,), BF16),
        "mtp_hnorm": ((h,), BF16),
        "mtp_eh_proj": ((h, 2 * hl), BF16),
        "mtp_input_layernorm": ((h,), BF16),
        "mtp_post_attention_layernorm": ((h,), BF16),
        "mtp_shared_head_norm": ((h,), BF16),
        "mtp_mla_q_a_proj_q": ((1, q_lora, h), FP8_E4M3),
        "mtp_mla_q_a_proj_s": ((1, q_lora // bh, h // bw), F32),
        "mtp_mla_q_a_norm": ((1, q_lora), BF16),
        "mtp_mla_q_b_proj_q": ((1, p_mla_loc, q_lora), FP8_E4M3),
        "mtp_mla_q_b_proj_s": ((1, p_mla_loc // bh, q_lora // bw), F32),
        "mtp_mla_kv_a_proj_q": ((1, kv_lora, h), FP8_E4M3),
        "mtp_mla_kv_a_proj_s": ((1, kv_lora // bh, h // bw), F32),
        "mtp_mla_kv_a_norm": ((1, kv_lora), BF16),
        "mtp_mla_w_uk": ((1, h_mla_loc, d_qk, kv_lora), BF16),
        "mtp_mla_w_uv": ((1, h_mla_loc, d_v, kv_lora), BF16),
        "mtp_mla_o_proj_q": ((1, h, h_mla_loc * d_v), FP8_E4M3),
        "mtp_mla_o_proj_s": ((1, h // bh, (h_mla_loc * d_v) // bw), F32),
        "mtp_moe_router_w": ((1, e_tot, h), BF16),
        "mtp_moe_router_bias": ((1, e_tot), F32),
        "mtp_moe_shared_gate_q": ((1, i_sh_loc, h), FP8_E4M3),
        "mtp_moe_shared_gate_s": ((1, i_sh_loc // bh, h // bw), F32),
        "mtp_moe_shared_up_q": ((1, i_sh_loc, h), FP8_E4M3),
        "mtp_moe_shared_up_s": ((1, i_sh_loc // bh, h // bw), F32),
        "mtp_moe_shared_down_q": ((1, h, i_sh_loc), FP8_E4M3),
        "mtp_moe_shared_down_s": ((1, h // bh, i_sh_loc // bw), F32),
        "mtp_moe_expert_gate_up_q": ((1, e_loc, 2 * i_moe, h), FP8_E4M3),
        "mtp_moe_expert_gate_up_s": ((1, e_loc, (2 * i_moe) // bh, h // bw), F32),
        "mtp_moe_expert_down_q": ((1, e_loc, h, i_moe), FP8_E4M3),
        "mtp_moe_expert_down_s": ((1, e_loc, h // bh, i_moe // bw), F32),
    }


def expected_rank_specs(
    cfg: Glm53MegakernelConfig, world_size: int = 8, *, include_mtp: bool = False
) -> dict[str, tuple[tuple[int, ...], np.dtype[Any]]]:
    """Return per-rank (shape, dtype) for every family in the EPxTP layout."""
    cfg.validate(world_size=world_size)
    l_all = cfg.num_hidden_layers
    l_dense = cfg.first_k_dense_replace
    l_moe = l_all - l_dense
    l_kda = len(cfg.kda_layers)
    l_mla = len(cfg.mla_layers)
    h = cfg.hidden_size
    v_loc = cfg.vocab_size // world_size
    mix = (2 + cfg.hc_mult) * cfg.hc_mult
    flat_hc = cfg.hc_mult * h
    h_kda_loc = cfg.kda_num_heads // world_size
    d_kda = cfg.kda_head_dim
    p_kda_loc = h_kda_loc * d_kda
    k_conv = cfg.kda_conv_kernel_size
    h_mla_loc = cfg.mla_num_heads // world_size
    q_lora = cfg.q_lora_rank
    kv_lora = cfg.kv_lora_rank
    d_qk = cfg.qk_nope_head_dim
    d_v = cfg.v_head_dim
    p_mla_loc = h_mla_loc * d_qk
    i_dense_loc = cfg.intermediate_size // world_size
    i_sh_loc = (cfg.n_shared_experts * cfg.moe_intermediate_size) // world_size
    e_tot = cfg.n_routed_experts
    e_loc = e_tot // world_size
    i_moe = cfg.moe_intermediate_size
    bh, bw = cfg.weight_block_size

    specs: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {
        "embed_tokens": ((v_loc, h), BF16),
        "lm_head": ((v_loc, h), BF16),
        "final_norm": ((h,), BF16),
        "hc_attn_fn": ((l_all, mix, flat_hc), BF16),
        "hc_attn_base": ((l_all, mix), F32),
        "hc_attn_scale": ((l_all, 3), F32),
        "input_layernorm": ((l_all, h), BF16),
        "hc_ffn_fn": ((l_all, mix, flat_hc), BF16),
        "hc_ffn_base": ((l_all, mix), F32),
        "hc_ffn_scale": ((l_all, 3), F32),
        "post_attention_layernorm": ((l_all, h), BF16),
        "kda_q_proj": ((l_kda, p_kda_loc, h), BF16),
        "kda_k_proj": ((l_kda, p_kda_loc, h), BF16),
        "kda_v_proj": ((l_kda, p_kda_loc, h), BF16),
        "kda_conv": ((l_kda, k_conv, 3, h_kda_loc, d_kda), BF16),
        "kda_f_a_proj": ((l_kda, d_kda, h), BF16),
        "kda_f_b_proj": ((l_kda, p_kda_loc, d_kda), BF16),
        "kda_g_a_proj": ((l_kda, d_kda, h), BF16),
        "kda_g_b_proj": ((l_kda, p_kda_loc, d_kda), BF16),
        "kda_b_proj": ((l_kda, h_kda_loc, h), BF16),
        "kda_a_log": ((l_kda, h_kda_loc), F32),
        "kda_dt_bias": ((l_kda, h_kda_loc, d_kda), F32),
        "kda_o_norm": ((l_kda, d_kda), BF16),
        "kda_o_proj": ((l_kda, h, p_kda_loc), BF16),
        "mla_q_a_proj_q": ((l_mla, q_lora, h), FP8_E4M3),
        "mla_q_a_proj_s": ((l_mla, q_lora // bh, h // bw), F32),
        "mla_q_a_norm": ((l_mla, q_lora), BF16),
        "mla_q_b_proj_q": ((l_mla, p_mla_loc, q_lora), FP8_E4M3),
        "mla_q_b_proj_s": ((l_mla, p_mla_loc // bh, q_lora // bw), F32),
        "mla_kv_a_proj_q": ((l_mla, kv_lora, h), FP8_E4M3),
        "mla_kv_a_proj_s": ((l_mla, kv_lora // bh, h // bw), F32),
        "mla_kv_a_norm": ((l_mla, kv_lora), BF16),
        "mla_w_uk": ((l_mla, h_mla_loc, d_qk, kv_lora), BF16),
        "mla_w_uv": ((l_mla, h_mla_loc, d_v, kv_lora), BF16),
        "mla_o_proj_q": ((l_mla, h, h_mla_loc * d_v), FP8_E4M3),
        "mla_o_proj_s": ((l_mla, h // bh, (h_mla_loc * d_v) // bw), F32),
        "dense_gate_q": ((l_dense, i_dense_loc, h), FP8_E4M3),
        "dense_gate_s": ((l_dense, i_dense_loc // bh, h // bw), F32),
        "dense_up_q": ((l_dense, i_dense_loc, h), FP8_E4M3),
        "dense_up_s": ((l_dense, i_dense_loc // bh, h // bw), F32),
        "dense_down_q": ((l_dense, h, i_dense_loc), FP8_E4M3),
        "dense_down_s": ((l_dense, h // bh, i_dense_loc // bw), F32),
        "moe_router_w": ((l_moe, e_tot, h), BF16),
        "moe_router_bias": ((l_moe, e_tot), F32),
        "moe_shared_gate_q": ((l_moe, i_sh_loc, h), FP8_E4M3),
        "moe_shared_gate_s": ((l_moe, i_sh_loc // bh, h // bw), F32),
        "moe_shared_up_q": ((l_moe, i_sh_loc, h), FP8_E4M3),
        "moe_shared_up_s": ((l_moe, i_sh_loc // bh, h // bw), F32),
        "moe_shared_down_q": ((l_moe, h, i_sh_loc), FP8_E4M3),
        "moe_shared_down_s": ((l_moe, h // bh, i_sh_loc // bw), F32),
        "moe_expert_gate_up_q": ((l_moe, e_loc, 2 * i_moe, h), FP8_E4M3),
        "moe_expert_gate_up_s": ((l_moe, e_loc, (2 * i_moe) // bh, h // bw), F32),
        "moe_expert_down_q": ((l_moe, e_loc, h, i_moe), FP8_E4M3),
        "moe_expert_down_s": ((l_moe, e_loc, h // bh, i_moe // bw), F32),
    }
    if include_mtp and cfg.num_nextn_predict_layers >= 1:
        specs.update(expected_mtp_rank_specs(cfg, world_size=world_size))
    return specs


class Glm53Checkpoint:
    """Coalesced streaming safetensors reader for GLM-5.3-Flash checkpoints."""

    COALESCE_GAP = 16 << 20
    CHUNK = 16 << 20
    READ_THREADS = 8

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.config = Glm53MegakernelConfig.from_checkpoint(self.path)
        idx_file = self.path / "model.safetensors.index.json"
        if not idx_file.is_file():
            raise FileNotFoundError(f"Missing {idx_file}")
        self.index: dict[str, str] = json.loads(idx_file.read_text())["weight_map"]
        missing = [f for f in set(self.index.values()) if not (self.path / f).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Checkpoint {self.path} missing {len(missing)} shard files"
            )
        self._layouts: dict[str, tuple[dict[str, Any], int]] = {}
        self.prefix = (
            "model.language_model."
            if any(k.startswith("model.language_model.") for k in self.index)
            else "model."
        )

    def _layout(self, filename: str) -> tuple[dict[str, Any], int]:
        if filename not in self._layouts:
            with open(self.path / filename, "rb") as f:
                size = struct.unpack("<Q", f.read(8))[0]
                self._layouts[filename] = (json.loads(f.read(size)), 8 + size)
        return self._layouts[filename]

    def tensor_meta(self, key: str) -> tuple[tuple[int, ...], np.dtype[Any]]:
        """Return (shape, dtype) of a tensor key from its safetensors header."""
        filename = self.index[key]
        header, _ = self._layout(filename)
        meta = header[key]
        return tuple(meta["shape"]), _SAFETENSOR_DTYPES[meta["dtype"]]

    def read(self, key: str, selection: Any = None) -> np.ndarray:
        """Read one tensor or a slice from its safetensors shard."""
        filename = self.index[key]
        header, base = self._layout(filename)
        meta = header[key]
        shape = list(meta["shape"])
        dtype = _SAFETENSOR_DTYPES[meta["dtype"]]
        start, end = meta["data_offsets"]
        rows = slice(None)
        rest: Any = ()
        if selection is not None:
            parts = selection if isinstance(selection, tuple) else (selection,)
            leading, tail = parts[0], parts[1:]
            if isinstance(leading, slice) and all(
                isinstance(r, slice) and r == slice(None) for r in tail
            ):
                rows, rest = leading, ()
            else:
                rest = parts
        r0, r1, step = rows.indices(shape[0]) if shape else (0, 1, 1)
        if step != 1:
            raise ValueError("Only unit-step leading slices are supported")
        row_bytes = (end - start) // shape[0] if shape else (end - start)
        offset = base + start + r0 * row_bytes
        length = (r1 - r0) * row_bytes if shape else (end - start)
        fd = os.open(self.path / filename, os.O_RDONLY)
        try:
            pieces = []
            for cs in range(0, length, self.CHUNK):
                pieces.append(os.pread(fd, min(self.CHUNK, length - cs), offset + cs))
        finally:
            os.close(fd)
        raw_bytes = b"".join(pieces)
        arr = np.frombuffer(raw_bytes, dtype=np.uint8).view(dtype)
        arr = arr.reshape([r1 - r0, *shape[1:]] if shape else []).copy()
        return arr[rest].copy() if rest else arr

    def read_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        """Coalesced parallel read of multiple tensor keys across shards."""
        by_file: dict[str, list[str]] = {}
        for k in keys:
            by_file.setdefault(self.index[k], []).append(k)
        out: dict[str, np.ndarray] = {}
        for filename, fkeys in by_file.items():
            header, base = self._layout(filename)
            spans = sorted(
                (header[k]["data_offsets"][0], header[k]["data_offsets"][1], k)
                for k in fkeys
            )
            ranges: list[list[Any]] = []
            for s, e, k in spans:
                if ranges and s - ranges[-1][1] <= self.COALESCE_GAP:
                    ranges[-1][1] = max(ranges[-1][1], e)
                    ranges[-1][2].append(k)
                else:
                    ranges.append([s, e, [k]])
            shard_path = self.path / filename

            def _stream(
                bounds: list[Any], sp: Path = shard_path, b: int = base
            ) -> bytes:
                s0, e0, _ = bounds
                fd = os.open(sp, os.O_RDONLY)
                try:
                    chunks = []
                    for off in range(s0, e0, self.CHUNK):
                        chunks.append(os.pread(fd, min(self.CHUNK, e0 - off), b + off))
                finally:
                    os.close(fd)
                return b"".join(chunks)

            with ThreadPoolExecutor(min(self.READ_THREADS, len(ranges))) as pool:
                buffers = list(pool.map(_stream, ranges))
            for (s0, _, rkeys), buf in zip(ranges, buffers):
                u8 = np.frombuffer(buf, dtype=np.uint8)
                for k in rkeys:
                    m = header[k]
                    a, b = m["data_offsets"]
                    dt = _SAFETENSOR_DTYPES[m["dtype"]]
                    out[k] = u8[a - s0 : b - s0].view(dt).reshape(m["shape"]).copy()
        return out

    def extract_all_ranks(
        self,
        world_size: int = 8,
        *,
        include_routed_experts: bool = False,
        only_rank: int | None = None,
    ) -> list[dict[str, np.ndarray]]:
        """Extract all rank dictionaries in one pass over the checkpoint.

        `only_rank` builds just that rank's dictionary (returned as a
        one-element list), for a process that holds one TP shard.
        """
        del include_routed_experts
        cfg = self.config
        specs = expected_rank_specs(cfg, world_size=world_size)
        routed_names = {
            "moe_expert_gate_up_q",
            "moe_expert_gate_up_s",
            "moe_expert_down_q",
            "moe_expert_down_s",
        }
        rank_ids = list(range(world_size)) if only_rank is None else [int(only_rank)]
        ranks = {
            r: {
                name: np.empty(shape, dtype=dt)
                for name, (shape, dt) in specs.items()
                if name not in routed_names
            }
            for r in rank_ids
        }
        p = self.prefix
        v_loc = cfg.vocab_size // world_size
        emb = self.read(f"{p}embed_tokens.weight")
        lmh = self.read("lm_head.weight")
        fnorm = self.read(f"{p}norm.weight")
        for r in rank_ids:
            ranks[r]["embed_tokens"][...] = emb[r * v_loc : (r + 1) * v_loc]
            ranks[r]["lm_head"][...] = lmh[r * v_loc : (r + 1) * v_loc]
            ranks[r]["final_norm"][...] = fnorm

        for lid in range(cfg.num_hidden_layers):
            lp = f"{p}layers.{lid}."
            hc_keys = [
                f"{lp}hc_attn_fn",
                f"{lp}hc_attn_base",
                f"{lp}hc_attn_scale",
                f"{lp}input_layernorm.weight",
                f"{lp}hc_ffn_fn",
                f"{lp}hc_ffn_base",
                f"{lp}hc_ffn_scale",
                f"{lp}post_attention_layernorm.weight",
            ]
            t_hc = self.read_many(hc_keys)
            for r in rank_ids:
                ranks[r]["hc_attn_fn"][lid] = t_hc[f"{lp}hc_attn_fn"]
                ranks[r]["hc_attn_base"][lid] = t_hc[f"{lp}hc_attn_base"]
                ranks[r]["hc_attn_scale"][lid] = t_hc[f"{lp}hc_attn_scale"]
                ranks[r]["input_layernorm"][lid] = t_hc[f"{lp}input_layernorm.weight"]
                ranks[r]["hc_ffn_fn"][lid] = t_hc[f"{lp}hc_ffn_fn"]
                ranks[r]["hc_ffn_base"][lid] = t_hc[f"{lp}hc_ffn_base"]
                ranks[r]["hc_ffn_scale"][lid] = t_hc[f"{lp}hc_ffn_scale"]
                ranks[r]["post_attention_layernorm"][lid] = t_hc[
                    f"{lp}post_attention_layernorm.weight"
                ]

        self._extract_attention_layers(ranks, world_size)
        self._extract_ffn_layers(ranks, world_size)
        return [ranks[r] for r in rank_ids]

    def _extract_attention_layers(
        self, ranks: dict[int, dict[str, np.ndarray]], world_size: int
    ) -> None:
        cfg = self.config
        p = self.prefix
        h_kda_loc = cfg.kda_num_heads // world_size
        d_kda = cfg.kda_head_dim
        p_kda_loc = h_kda_loc * d_kda
        k_conv = cfg.kda_conv_kernel_size
        for idx, lid in enumerate(cfg.kda_layers):
            ap = f"{p}layers.{lid}.self_attn."
            keys = [
                f"{ap}q_proj.weight",
                f"{ap}k_proj.weight",
                f"{ap}v_proj.weight",
                f"{ap}q_conv1d.weight",
                f"{ap}k_conv1d.weight",
                f"{ap}v_conv1d.weight",
                f"{ap}f_a_proj.weight",
                f"{ap}f_b_proj.weight",
                f"{ap}g_a_proj.weight",
                f"{ap}g_b_proj.weight",
                f"{ap}b_proj.weight",
                f"{ap}A_log",
                f"{ap}dt_bias",
                f"{ap}o_norm.weight",
                f"{ap}o_proj.weight",
            ]
            t = self.read_many(keys)
            a_log = t[f"{ap}A_log"].reshape(-1)
            dt_bias = t[f"{ap}dt_bias"].reshape(cfg.kda_num_heads, d_kda)
            for r in ranks:
                s_proj = slice(r * p_kda_loc, (r + 1) * p_kda_loc)
                s_head = slice(r * h_kda_loc, (r + 1) * h_kda_loc)
                ranks[r]["kda_q_proj"][idx] = t[f"{ap}q_proj.weight"][s_proj]
                ranks[r]["kda_k_proj"][idx] = t[f"{ap}k_proj.weight"][s_proj]
                ranks[r]["kda_v_proj"][idx] = t[f"{ap}v_proj.weight"][s_proj]
                convs = [
                    t[f"{ap}{c}_conv1d.weight"][s_proj, 0, :].T.reshape(
                        k_conv, h_kda_loc, d_kda
                    )
                    for c in ("q", "k", "v")
                ]
                ranks[r]["kda_conv"][idx] = np.stack(convs, axis=1)
                ranks[r]["kda_f_a_proj"][idx] = t[f"{ap}f_a_proj.weight"]
                ranks[r]["kda_f_b_proj"][idx] = t[f"{ap}f_b_proj.weight"][s_proj]
                ranks[r]["kda_g_a_proj"][idx] = t[f"{ap}g_a_proj.weight"]
                ranks[r]["kda_g_b_proj"][idx] = t[f"{ap}g_b_proj.weight"][s_proj]
                ranks[r]["kda_b_proj"][idx] = t[f"{ap}b_proj.weight"][s_head]
                ranks[r]["kda_a_log"][idx] = a_log[s_head]
                ranks[r]["kda_dt_bias"][idx] = dt_bias[s_head]
                ranks[r]["kda_o_norm"][idx] = t[f"{ap}o_norm.weight"]
                ranks[r]["kda_o_proj"][idx] = t[f"{ap}o_proj.weight"][:, s_proj]

        h_mla_loc = cfg.mla_num_heads // world_size
        d_qk = cfg.qk_nope_head_dim
        d_v = cfg.v_head_dim
        p_mla_loc = h_mla_loc * d_qk
        bh, bw = cfg.weight_block_size
        for idx, lid in enumerate(cfg.mla_layers):
            ap = f"{p}layers.{lid}.self_attn."
            keys = [
                f"{ap}q_a_proj.weight",
                f"{ap}q_a_proj.weight_scale_inv",
                f"{ap}q_a_layernorm.weight",
                f"{ap}q_b_proj.weight",
                f"{ap}q_b_proj.weight_scale_inv",
                f"{ap}kv_a_proj_with_mqa.weight",
                f"{ap}kv_a_proj_with_mqa.weight_scale_inv",
                f"{ap}kv_a_layernorm.weight",
                f"{ap}kv_b_proj.weight",
                f"{ap}o_proj.weight",
                f"{ap}o_proj.weight_scale_inv",
            ]
            t = self.read_many(keys)
            kv_b = t[f"{ap}kv_b_proj.weight"].reshape(
                cfg.mla_num_heads, d_qk + d_v, cfg.kv_lora_rank
            )
            for r in ranks:
                s_q = slice(r * p_mla_loc, (r + 1) * p_mla_loc)
                s_qs = slice(r * (p_mla_loc // bh), (r + 1) * (p_mla_loc // bh))
                s_h = slice(r * h_mla_loc, (r + 1) * h_mla_loc)
                s_o = slice(r * (h_mla_loc * d_v), (r + 1) * (h_mla_loc * d_v))
                s_os = slice(
                    r * ((h_mla_loc * d_v) // bw), (r + 1) * ((h_mla_loc * d_v) // bw)
                )
                ranks[r]["mla_q_a_proj_q"][idx] = t[f"{ap}q_a_proj.weight"]
                ranks[r]["mla_q_a_proj_s"][idx] = t[f"{ap}q_a_proj.weight_scale_inv"]
                ranks[r]["mla_q_a_norm"][idx] = t[f"{ap}q_a_layernorm.weight"]
                ranks[r]["mla_q_b_proj_q"][idx] = t[f"{ap}q_b_proj.weight"][s_q]
                ranks[r]["mla_q_b_proj_s"][idx] = t[f"{ap}q_b_proj.weight_scale_inv"][
                    s_qs
                ]
                ranks[r]["mla_kv_a_proj_q"][idx] = t[f"{ap}kv_a_proj_with_mqa.weight"]
                ranks[r]["mla_kv_a_proj_s"][idx] = t[
                    f"{ap}kv_a_proj_with_mqa.weight_scale_inv"
                ]
                ranks[r]["mla_kv_a_norm"][idx] = t[f"{ap}kv_a_layernorm.weight"]
                ranks[r]["mla_w_uk"][idx] = kv_b[s_h, :d_qk, :]
                ranks[r]["mla_w_uv"][idx] = kv_b[s_h, d_qk:, :]
                ranks[r]["mla_o_proj_q"][idx] = t[f"{ap}o_proj.weight"][:, s_o]
                ranks[r]["mla_o_proj_s"][idx] = t[f"{ap}o_proj.weight_scale_inv"][
                    :, s_os
                ]

    def _extract_ffn_layers(
        self,
        ranks: dict[int, dict[str, np.ndarray]],
        world_size: int,
    ) -> None:
        cfg = self.config
        p = self.prefix
        bh, bw = cfg.weight_block_size
        i_dense_loc = cfg.intermediate_size // world_size
        for lid in range(cfg.first_k_dense_replace):
            mp = f"{p}layers.{lid}.mlp."
            t = self.read_many(
                [
                    f"{mp}gate_proj.weight",
                    f"{mp}gate_proj.weight_scale_inv",
                    f"{mp}up_proj.weight",
                    f"{mp}up_proj.weight_scale_inv",
                    f"{mp}down_proj.weight",
                    f"{mp}down_proj.weight_scale_inv",
                ]
            )
            for r in ranks:
                s_i = slice(r * i_dense_loc, (r + 1) * i_dense_loc)
                s_ib = slice(r * (i_dense_loc // bh), (r + 1) * (i_dense_loc // bh))
                s_iw = slice(r * (i_dense_loc // bw), (r + 1) * (i_dense_loc // bw))
                ranks[r]["dense_gate_q"][lid] = t[f"{mp}gate_proj.weight"][s_i]
                ranks[r]["dense_gate_s"][lid] = t[f"{mp}gate_proj.weight_scale_inv"][
                    s_ib
                ]
                ranks[r]["dense_up_q"][lid] = t[f"{mp}up_proj.weight"][s_i]
                ranks[r]["dense_up_s"][lid] = t[f"{mp}up_proj.weight_scale_inv"][s_ib]
                ranks[r]["dense_down_q"][lid] = t[f"{mp}down_proj.weight"][:, s_i]
                ranks[r]["dense_down_s"][lid] = t[f"{mp}down_proj.weight_scale_inv"][
                    :, s_iw
                ]

        i_sh_loc = (cfg.n_shared_experts * cfg.moe_intermediate_size) // world_size
        for idx, lid in enumerate(
            range(cfg.first_k_dense_replace, cfg.num_hidden_layers)
        ):
            mp = f"{p}layers.{lid}.mlp."
            sh_keys = [
                f"{mp}gate.weight",
                f"{mp}gate.e_score_correction_bias",
                f"{mp}shared_experts.gate_proj.weight",
                f"{mp}shared_experts.gate_proj.weight_scale_inv",
                f"{mp}shared_experts.up_proj.weight",
                f"{mp}shared_experts.up_proj.weight_scale_inv",
                f"{mp}shared_experts.down_proj.weight",
                f"{mp}shared_experts.down_proj.weight_scale_inv",
            ]
            t = self.read_many(sh_keys)
            for r in ranks:
                s_i = slice(r * i_sh_loc, (r + 1) * i_sh_loc)
                s_ib = slice(r * (i_sh_loc // bh), (r + 1) * (i_sh_loc // bh))
                s_iw = slice(r * (i_sh_loc // bw), (r + 1) * (i_sh_loc // bw))
                ranks[r]["moe_router_w"][idx] = t[f"{mp}gate.weight"]
                ranks[r]["moe_router_bias"][idx] = t[
                    f"{mp}gate.e_score_correction_bias"
                ]
                ranks[r]["moe_shared_gate_q"][idx] = t[
                    f"{mp}shared_experts.gate_proj.weight"
                ][s_i]
                ranks[r]["moe_shared_gate_s"][idx] = t[
                    f"{mp}shared_experts.gate_proj.weight_scale_inv"
                ][s_ib]
                ranks[r]["moe_shared_up_q"][idx] = t[
                    f"{mp}shared_experts.up_proj.weight"
                ][s_i]
                ranks[r]["moe_shared_up_s"][idx] = t[
                    f"{mp}shared_experts.up_proj.weight_scale_inv"
                ][s_ib]
                ranks[r]["moe_shared_down_q"][idx] = t[
                    f"{mp}shared_experts.down_proj.weight"
                ][:, s_i]
                ranks[r]["moe_shared_down_s"][idx] = t[
                    f"{mp}shared_experts.down_proj.weight_scale_inv"
                ][:, s_iw]

    def extract_mtp_ranks(
        self,
        world_size: int = 8,
        *,
        include_routed_experts: bool = False,
        only_rank: int | None = None,
    ) -> list[dict[str, np.ndarray]]:
        """Read and shard the single NextN MTP layer (`layers.{L}`) across `world_size`
        ranks (or just `only_rank`).
        """
        del include_routed_experts
        cfg = self.config
        lid = cfg.num_hidden_layers
        lp = f"{self.prefix}layers.{lid}."
        ap = f"{lp}self_attn."
        mp = f"{lp}mlp."
        keys = [
            f"{lp}enorm.weight",
            f"{lp}hnorm.weight",
            f"{lp}eh_proj.weight",
            f"{lp}input_layernorm.weight",
            f"{lp}post_attention_layernorm.weight",
            f"{lp}shared_head.norm.weight",
            f"{ap}q_a_proj.weight",
            f"{ap}q_a_proj.weight_scale_inv",
            f"{ap}q_a_layernorm.weight",
            f"{ap}q_b_proj.weight",
            f"{ap}q_b_proj.weight_scale_inv",
            f"{ap}kv_a_proj_with_mqa.weight",
            f"{ap}kv_a_proj_with_mqa.weight_scale_inv",
            f"{ap}kv_a_layernorm.weight",
            f"{ap}kv_b_proj.weight",
            f"{ap}o_proj.weight",
            f"{ap}o_proj.weight_scale_inv",
            f"{mp}gate.weight",
            f"{mp}gate.e_score_correction_bias",
            f"{mp}shared_experts.gate_proj.weight",
            f"{mp}shared_experts.gate_proj.weight_scale_inv",
            f"{mp}shared_experts.up_proj.weight",
            f"{mp}shared_experts.up_proj.weight_scale_inv",
            f"{mp}shared_experts.down_proj.weight",
            f"{mp}shared_experts.down_proj.weight_scale_inv",
        ]
        raw = self.read_many(keys)
        short = {k[len(lp) :]: v for k, v in raw.items()}
        return shard_mtp_non_routed(
            cfg, short, world_size=world_size, only_rank=only_rank
        )


def shard_mtp_non_routed(
    cfg: Glm53MegakernelConfig,
    short_tensors: dict[str, np.ndarray],
    world_size: int = 8,
    only_rank: int | None = None,
) -> list[dict[str, np.ndarray]]:
    """Shard non-routed MTP layer tensors (`short_name -> ndarray`) across `world_size`
    ranks.
    """
    h = cfg.hidden_size
    hl = h // world_size
    h_mla_loc = cfg.mla_num_heads // world_size
    d_qk = cfg.qk_nope_head_dim
    d_v = cfg.v_head_dim
    p_mla_loc = h_mla_loc * d_qk
    i_sh_loc = (cfg.n_shared_experts * cfg.moe_intermediate_size) // world_size
    bh, bw = cfg.weight_block_size
    eh_w = short_tensors["eh_proj.weight"]
    kv_b = short_tensors["self_attn.kv_b_proj.weight"].reshape(
        cfg.mla_num_heads, d_qk + d_v, cfg.kv_lora_rank
    )
    out: list[dict[str, np.ndarray]] = []
    for r in range(world_size) if only_rank is None else [int(only_rank)]:
        s_hl = slice(r * hl, (r + 1) * hl)
        eh_r = np.concatenate(
            [eh_w[:, s_hl], eh_w[:, h + r * hl : h + (r + 1) * hl]], axis=1
        )
        s_q = slice(r * p_mla_loc, (r + 1) * p_mla_loc)
        s_qs = slice(r * (p_mla_loc // bh), (r + 1) * (p_mla_loc // bh))
        s_h = slice(r * h_mla_loc, (r + 1) * h_mla_loc)
        s_o = slice(r * (h_mla_loc * d_v), (r + 1) * (h_mla_loc * d_v))
        s_os = slice(r * ((h_mla_loc * d_v) // bw), (r + 1) * ((h_mla_loc * d_v) // bw))
        s_i = slice(r * i_sh_loc, (r + 1) * i_sh_loc)
        s_ib = slice(r * (i_sh_loc // bh), (r + 1) * (i_sh_loc // bh))
        s_iw = slice(r * (i_sh_loc // bw), (r + 1) * (i_sh_loc // bw))
        out.append(
            {
                "mtp_enorm": short_tensors["enorm.weight"].copy(),
                "mtp_hnorm": short_tensors["hnorm.weight"].copy(),
                "mtp_eh_proj": eh_r.copy(),
                "mtp_input_layernorm": short_tensors["input_layernorm.weight"].copy(),
                "mtp_post_attention_layernorm": short_tensors[
                    "post_attention_layernorm.weight"
                ].copy(),
                "mtp_shared_head_norm": short_tensors["shared_head.norm.weight"].copy(),
                "mtp_mla_q_a_proj_q": short_tensors["self_attn.q_a_proj.weight"][
                    None
                ].copy(),
                "mtp_mla_q_a_proj_s": short_tensors[
                    "self_attn.q_a_proj.weight_scale_inv"
                ][None].copy(),
                "mtp_mla_q_a_norm": short_tensors["self_attn.q_a_layernorm.weight"][
                    None
                ].copy(),
                "mtp_mla_q_b_proj_q": short_tensors["self_attn.q_b_proj.weight"][s_q][
                    None
                ].copy(),
                "mtp_mla_q_b_proj_s": short_tensors[
                    "self_attn.q_b_proj.weight_scale_inv"
                ][s_qs][None].copy(),
                "mtp_mla_kv_a_proj_q": short_tensors[
                    "self_attn.kv_a_proj_with_mqa.weight"
                ][None].copy(),
                "mtp_mla_kv_a_proj_s": short_tensors[
                    "self_attn.kv_a_proj_with_mqa.weight_scale_inv"
                ][None].copy(),
                "mtp_mla_kv_a_norm": short_tensors["self_attn.kv_a_layernorm.weight"][
                    None
                ].copy(),
                "mtp_mla_w_uk": kv_b[s_h, :d_qk, :][None].copy(),
                "mtp_mla_w_uv": kv_b[s_h, d_qk:, :][None].copy(),
                "mtp_mla_o_proj_q": short_tensors["self_attn.o_proj.weight"][:, s_o][
                    None
                ].copy(),
                "mtp_mla_o_proj_s": short_tensors["self_attn.o_proj.weight_scale_inv"][
                    :, s_os
                ][None].copy(),
                "mtp_moe_router_w": short_tensors["mlp.gate.weight"][None].copy(),
                "mtp_moe_router_bias": short_tensors[
                    "mlp.gate.e_score_correction_bias"
                ][None].copy(),
                "mtp_moe_shared_gate_q": short_tensors[
                    "mlp.shared_experts.gate_proj.weight"
                ][s_i][None].copy(),
                "mtp_moe_shared_gate_s": short_tensors[
                    "mlp.shared_experts.gate_proj.weight_scale_inv"
                ][s_ib][None].copy(),
                "mtp_moe_shared_up_q": short_tensors[
                    "mlp.shared_experts.up_proj.weight"
                ][s_i][None].copy(),
                "mtp_moe_shared_up_s": short_tensors[
                    "mlp.shared_experts.up_proj.weight_scale_inv"
                ][s_ib][None].copy(),
                "mtp_moe_shared_down_q": short_tensors[
                    "mlp.shared_experts.down_proj.weight"
                ][:, s_i][None].copy(),
                "mtp_moe_shared_down_s": short_tensors[
                    "mlp.shared_experts.down_proj.weight_scale_inv"
                ][:, s_iw][None].copy(),
            }
        )
    return out


FP8_PER_CHANNEL_PREFIXES = (
    "mla_q_a_proj",
    "mla_q_b_proj",
    "mla_kv_a_proj",
    "mla_o_proj",
    "dense_gate",
    "dense_up",
    "dense_down",
    "moe_shared_gate",
    "moe_shared_up",
    "moe_shared_down",
    "moe_expert_gate_up",
    "moe_expert_down",
    "mtp_mla_q_a_proj",
    "mtp_mla_q_b_proj",
    "mtp_mla_kv_a_proj",
    "mtp_mla_o_proj",
    "mtp_moe_shared_gate",
    "mtp_moe_shared_up",
    "mtp_moe_shared_down",
    "mtp_moe_expert_gate_up",
    "mtp_moe_expert_down",
)


def requant_fp8_per_channel(
    wq_l: jax.Array, ws_l: jax.Array, *, weight_block_size: tuple[int, int]
) -> tuple[jax.Array, jax.Array]:
    """Block-FP8 [..., N, K] + [..., N/bh, K/bw] scales -> per-channel FP8 + [..., N,
    1].
    """
    bh, bw = weight_block_size
    oshape = wq_l.shape
    n_out, k_in = oshape[-2], oshape[-1]
    lead = oshape[:-2]
    wf = (
        wq_l.reshape(*lead, n_out // bh, bh, k_in // bw, bw).astype(jnp.float32)
        * ws_l[..., :, None, :, None].astype(jnp.float32)
    ).reshape(oshape)
    amax = jnp.max(jnp.abs(wf), axis=-1, keepdims=True)
    fmax_inv = jnp.float32(1.0 / float(jnp.finfo(jnp.float8_e4m3fn).max))
    fmax = jnp.finfo(jnp.float8_e4m3fn).max.astype(jnp.float32)
    fmin = jnp.finfo(jnp.float8_e4m3fn).min.astype(jnp.float32)
    sc = amax * fmax_inv
    sc_inv = jnp.where(sc == 0.0, 0.0, 1.0 / sc)
    q_pc = jnp.clip(wf * sc_inv, fmin, fmax).astype(jnp.float8_e4m3fn)
    return q_pc, jnp.where(sc == 0.0, 1.0, sc)


def to_per_channel_fp8(
    cfg: Glm53MegakernelConfig, weights: dict[str, jax.Array]
) -> dict[str, jax.Array]:
    """Apply load-time per-channel FP8 requantization to an in-memory weight dict."""
    fn = jax.jit(
        functools.partial(
            requant_fp8_per_channel, weight_block_size=cfg.weight_block_size
        )
    )
    out = dict(weights)
    for pref in FP8_PER_CHANNEL_PREFIXES:
        if f"{pref}_q" in weights and f"{pref}_s" in weights:
            out[f"{pref}_q"], out[f"{pref}_s"] = fn(
                weights[f"{pref}_q"], weights[f"{pref}_s"]
            )
    return out


def synthetic_rank_weights(
    cfg: Glm53MegakernelConfig,
    rank: int,
    *,
    world_size: int = 8,
    seed: int = 0,
    include_mtp: bool = False,
) -> dict[str, np.ndarray]:
    """Generate deterministic per-rank weights in the EPxTP layout for testing."""
    specs = expected_rank_specs(cfg, world_size=world_size, include_mtp=include_mtp)
    replicated = {
        "final_norm",
        "hc_attn_fn",
        "hc_attn_base",
        "hc_attn_scale",
        "input_layernorm",
        "hc_ffn_fn",
        "hc_ffn_base",
        "hc_ffn_scale",
        "post_attention_layernorm",
        "kda_f_a_proj",
        "kda_g_a_proj",
        "kda_o_norm",
        "mla_q_a_proj_q",
        "mla_q_a_proj_s",
        "mla_q_a_norm",
        "mla_kv_a_proj_q",
        "mla_kv_a_proj_s",
        "mla_kv_a_norm",
        "moe_router_w",
        "moe_router_bias",
        "mtp_enorm",
        "mtp_hnorm",
        "mtp_input_layernorm",
        "mtp_post_attention_layernorm",
        "mtp_shared_head_norm",
        "mtp_mla_q_a_proj_q",
        "mtp_mla_q_a_proj_s",
        "mtp_mla_q_a_norm",
        "mtp_mla_kv_a_proj_q",
        "mtp_mla_kv_a_proj_s",
        "mtp_mla_kv_a_norm",
        "mtp_moe_router_w",
        "mtp_moe_router_bias",
    }
    out: dict[str, np.ndarray] = {}
    for idx, (name, (shape, dt)) in enumerate(specs.items()):
        r_seed = 0 if name in replicated else rank
        rng = np.random.default_rng((seed, 700 + idx, r_seed))
        vals = rng.standard_normal(shape, dtype=np.float32) * 0.02
        if "norm" in name:
            vals = 1.0 + vals
        elif name.endswith("_s"):
            vals = np.abs(vals) + 0.01
        out[name] = vals.astype(dt)
    return out


def place_rank_weights_on_mesh(
    per_rank: list[dict[str, np.ndarray]],
    mesh: jax.sharding.Mesh,
) -> dict[str, jax.Array]:
    """Upload per-rank numpy dicts onto the 1D ('tp',) mesh as [W, ...] arrays."""
    world_size = len(per_rank)
    devices = list(mesh.devices.flat)
    if len(devices) != world_size:
        raise ValueError(f"Mesh has {len(devices)} devices, got {world_size} ranks")
    sharding = NamedSharding(mesh, P("tp"))
    out: dict[str, jax.Array] = {}
    for name in per_rank[0]:
        dev_arrays = [
            jax.device_put(per_rank[r][name][None], SingleDeviceSharding(devices[r]))
            for r in range(world_size)
        ]
        full_shape = (world_size, *per_rank[0][name].shape)
        out[name] = jax.make_array_from_single_device_arrays(
            full_shape, sharding, dev_arrays
        )
    return out
