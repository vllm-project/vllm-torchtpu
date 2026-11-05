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
type _Exp = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, set[str]]
TP_RANKS, EP_GROUPS, ETP_PARTS, FALLBACK_MLA, MAX_E8M0 = 32, 8, 4, 3, 255
PAIR_PARTS, KDA_PROJ_PARTS, KDA_CONV_STREAMS, NUM_EXPERT_PROJS = 2, 5, 3, 6
LAYER_KEY_PARTS, MIN_EXPERT_PARTS, DENSE_PAD = 3, 7, 1152
LATENT_ACTIVE, LATENT_UP_W, LATENT_NORM_PAD = 28, 1024, 512
KDA_PROJ_COLS, KDA_NORM_PAD, QB_HEAD_W, KB_HEAD_W = 1664, 7, 192, 256
KDA_OFFSETS = {"q_proj": 0, "k_proj": 384, "v_proj": 768, "g_proj": 1152}
KDA_CONVS = {"q_conv1d.weight": 0, "k_conv1d.weight": 1, "v_conv1d.weight": 2}
WHOLE_T = {"gate": "router", "f_a_proj": "k_gate", "q_a_proj": "m_qa"}
ROW_SLICES = {
    "f_b_proj": ("k_fb", 384), "kv_b_proj": ("m_kb", 768), "g_proj": ("m_gate", 384)
}
ROW_NORMS = {
    "input_layernorm": "attn_norm", "post_attention_layernorm": "ffn_norm",
    "q_a_layernorm": "m_qnorm", "kv_a_layernorm": "m_knorm",
    "gate.e_score_correction_bias": "router_bias",
}
LM_KEYS = {"language_model.lm_head.weight", "model.lm_head.weight", "lm_head.weight"}


def _np(t: torch.Tensor, r: int | None = None, w: int = 0) -> np.ndarray:
    """Slice optional row shard `[r*w:(r+1)*w]` and convert to host NumPy."""
    c = (t if r is None else t[r * w : (r + 1) * w]).detach().cpu().contiguous()
    if c.dtype == torch.bfloat16:
        return c.view(torch.uint16).numpy().view(BF16)
    return c.numpy()


def _upload(val: np.ndarray, dev: str) -> torch.Tensor:
    """Upload a host array to `[1, ...]` on `dev` (`bridge.upload` on TPU)."""
    if dev == "tpu":
        mod = importlib.import_module(".runtime_shim.bridge", __package__)
        tpu_t: torch.Tensor = mod.upload(val)
        return tpu_t
    a = np.ascontiguousarray(val)
    if a.dtype == BF16:
        b = torch.from_numpy(a.view(np.uint16)).view(torch.bfloat16)
        return b.unsqueeze(0).clone().to(dev)
    return torch.from_numpy(a).unsqueeze(0).clone().to(dev)


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
        cfg = Config()
        self.rank, self.num_layers, self.device = rank, num_layers, device
        self._cfg, d, v = cfg, cfg.dim, cfg.vocab
        self.expected_shapes = selected_local_shapes(num_layers, d, vocab=v)
        full = set(cfg.full_attention)
        mla = [i for i in range(num_layers) if i in full] or [FALLBACK_MLA]
        kda = [i for i in range(num_layers) if i not in full]
        self._norm_i = {i: i for i in range(num_layers)}
        self._moe_i = {l_id: p for p, l_id in enumerate(range(1, num_layers))}
        self._kda_i = {l_id: p for p, l_id in enumerate(kda)}
        self._mla_i = {l_id: p for p, l_id in enumerate(mla)}
        self._pairs: dict[tuple[int | None, str], dict[str, np.ndarray]] = {}
        self._multi: dict[tuple[int, str], tuple[np.ndarray, set[str]]] = {}
        self._experts: dict[int, _Exp] = {}
        self._parts: dict[str, dict[int, torch.Tensor]] = {}
        self.device_tensors: dict[str, torch.Tensor] = {}
        self._up_n = self._up_b = self._pk_b = self._pk_l = 0

    def _sync_peak(self) -> int:
        """Update peak staging bounds and return current active host bytes."""
        nb = sum(int(a.nbytes) for d in self._pairs.values() for a in d.values())
        nb += sum(int(b.nbytes) for b, _ in self._multi.values())
        nb += sum(int(sum(x.nbytes for x in s[:4])) for s in self._experts.values())
        p_ls = {k[0] for k in self._pairs if k[0] is not None}
        n_l = len(p_ls | {k[0] for k in self._multi} | set(self._experts))
        self._pk_b, self._pk_l = max(self._pk_b, nb), max(self._pk_l, n_l)
        return nb

    def stats(self) -> StreamStats:
        """Return current upload and host staging telemetry."""
        b, r, nl = self._sync_peak(), self.rank, self.num_layers
        return StreamStats(r, nl, self._up_n, self._up_b, self._pk_b, self._pk_l, b)

    def verify_complete(self) -> None:
        """Assert stream completeness and clean staging buffers."""
        missing = set(self.expected_shapes) - set(self.device_tensors)
        if missing:
            msg = f"rank {self.rank}: missing {len(missing)} tensors: {sorted(missing)[:3]}"
            raise ValueError(msg)
        if self._experts or self._multi or self._pairs:
            msg = f"rank {self.rank}: unconsumed staging buffers remain"
            raise ValueError(msg)

    def _put(self, fam: str, arr: np.ndarray, slot: _Slot = None) -> None:
        """Upload `arr` and concatenate layered slices along `dim=1` when complete."""
        dt = _upload(arr if slot is None else arr[None, ...], self.device)
        self._up_n += 1
        self._up_b += int(dt.numel() * dt.element_size())
        if slot is None:
            self.device_tensors[fam] = dt
            return
        idx, l_id = slot
        p = self._parts.setdefault(fam, {})
        p[idx[l_id]] = dt
        if len(p) == len(idx):
            self.device_tensors[fam] = torch.cat([p[i] for i in range(len(idx))], 1)
            del self._parts[fam]

    def _pair(self, key: tuple[int | None, str], tag: str, arr: np.ndarray) -> None:
        """Stage a 2-tensor family (`norm*proj` or `gate|up`) and flush when paired."""
        d = self._pairs.setdefault(key, {})
        d[tag] = arr
        self._sync_peak()
        if len(d) < PAIR_PARTS:
            return
        del self._pairs[key]
        l_id, fam = key
        idx = self._moe_i if fam == "shared_gu" else self._norm_i
        slot = None if l_id is None or fam == "dense_gu" else (idx, l_id)
        if "norm" in d:
            self._put(fam, (d["norm"] * d["proj"]).reshape(1, -1), slot)
        else:
            self._put(fam, np.concatenate([d["gate"], d["up"]], 1), slot)

    def _simple(self, key: str, t: torch.Tensor, slot: _Slot) -> bool:
        """Consume direct whole-transpose, row-norm, or row-sliced families."""
        st = key.removesuffix(".weight")
        if st in WHOLE_T and key.endswith(".weight"):
            self._put(WHOLE_T[st], _np(t).T.copy(), slot)
        elif st in ROW_NORMS:
            self._put(ROW_NORMS[st], _np(t).reshape(1, -1), slot)
        elif st in ROW_SLICES and key.endswith(".weight"):
            if st == "g_proj" and slot is not None and slot[0] is self._kda_i:
                return False
            fam, w = ROW_SLICES[st]
            self._put(fam, _np(t, self.rank, w).T.copy(), slot)
        elif key == "o_proj.weight" and slot is not None:
            pair, half = divmod(self.rank, 2)
            h2, r2 = self._cfg.dim // 2, (self._cfg.heads * self._cfg.head_dim) // 16
            arr = _np(t[half * h2 : (half + 1) * h2, pair * r2 : (pair + 1) * r2])
            self._put("k_o" if slot[0] is self._kda_i else "m_o", arr.T.copy(), slot)
        else:
            return False
        return True

    def _on_expert(self, l_id: int, parts: list[str], t: torch.Tensor) -> None:
        """Slice local `EP8/TP4` expert bytes and flush on layer completion."""
        if l_id not in self._moe_i or len(parts) < MIN_EXPERT_PARTS:
            return
        cnt, qtr = self._cfg.experts // EP_GROUPS, self._cfg.expert_hidden // ETP_PARTS
        eid, first = int(parts[4]), cnt * (self.rank // ETP_PARTS)
        if not first <= eid < first + cnt:
            return
        proj, kind, pt, idx = parts[5], parts[6], self.rank % ETP_PARTS, eid - first
        w2_st = qtr // 2 if kind == "weight_packed" else qtr // 32
        st = w2_st if proj == "w2" else qtr
        s = _np(t[:, pt * st : (pt + 1) * st]) if proj == "w2" else _np(t, pt, st)
        if l_id not in self._experts:
            e = np.empty
            self._experts[l_id] = (
                e((cnt, 448, 1536), np.uint32), e((cnt, 112, 1536), np.uint8),
                e((cnt, 96, 3584), np.uint32), e((cnt, 24, 3584), np.uint8),
                set(),
            )
        gu, gus, dn, ds, seen = self._experts[l_id]
        cols = slice(0, qtr) if proj == "w1" else slice(qtr, qtr * 2)
        if kind == "weight_scale":
            if s.dtype != np.uint8 or bool(np.any((s == 0) | (s == MAX_E8M0))):
                msg = f"expert {idx} {proj}: invalid E8M0 scale"
                raise ValueError(msg)
            if proj == "w2":
                ds[idx] = s.T
            else:
                gus[idx, :, cols] = s.T
        elif proj == "w2":
            dn[idx] = pack_expert_bytes(s)
        else:
            gu[idx, :, cols] = pack_expert_bytes(s)
        seen.add(f"{idx}.{proj}.{kind}")
        if self._sync_peak() >= 0 and len(seen) == cnt * NUM_EXPERT_PROJS:
            del self._experts[l_id]
            for fn, b in (
                ("expert_gu", gu), ("expert_gus", gus),
                ("expert_down", dn), ("expert_ds", ds),
            ):
                self._put(fn, b, (self._moe_i, l_id))

    def _on_moe(self, l_id: int, tail: str, t: torch.Tensor) -> None:
        """Consume router, latent, and shared-expert tensors on MoE layers."""
        r, c, slot = self.rank, self._cfg, (self._moe_i, l_id)
        if self._simple(tail, t, slot):
            return
        sh, ld, blk = (
            c.shared_hidden // TP_RANKS,
            c.latent // LATENT_ACTIVE,
            r % EP_GROUPS,
        )
        if tail in {"shared_experts.gate_proj.weight", "shared_experts.up_proj.weight"}:
            tag = "gate" if "gate" in tail else "up"
            self._pair((l_id, "shared_gu"), tag, _np(t, r, sh).T.copy())
        elif tail == "routed_expert_norm.weight":
            a = _np(t).astype(np.float32).reshape(1, -1)
            self._put("latent_norm", np.pad(a, ((0, 0), (0, LATENT_NORM_PAD))), slot)
        elif tail == "routed_expert_down_proj.weight":
            a = np.zeros((c.dim, ld), BF16) if r >= LATENT_ACTIVE else _np(t, r, ld).T
            self._put("latent_down", a.copy(), slot)
        elif tail == "routed_expert_up_proj.weight":
            z = np.zeros((c.latent, LATENT_UP_W), BF16)
            a = z if (blk + 1) * LATENT_UP_W > c.dim else _np(t, blk, LATENT_UP_W).T
            self._put("latent_up", a.copy(), slot)
        elif tail == "shared_experts.down_proj.weight":
            self._put("shared_down", _np(t[:, r * sh : (r + 1) * sh]).T.copy(), slot)

    def _on_kda(self, l_id: int, key: str, t: torch.Tensor) -> None:
        """Consume KDA projections, conv1d banks, and decay parameters."""
        r, c, slot = self.rank, self._cfg, (self._kda_i, l_id)
        if self._simple(key, t, slot):
            return
        hd, hc = c.head_dim, c.heads // TP_RANKS
        rc, stem = hc * hd, key.removesuffix(".weight")
        if key in KDA_CONVS:
            sh = (c.conv_size, KDA_CONV_STREAMS, hc, hd)
            init = (np.empty(sh, BF16), set())
            buf, seen = self._multi.setdefault((l_id, "k_conv"), init)
            buf[:, KDA_CONVS[key]] = _np(t, r, rc)[:, 0].T.reshape(c.conv_size, hc, hd)
            seen.add(key)
            if self._sync_peak() >= 0 and len(seen) == KDA_CONV_STREAMS:
                self._put("k_conv", self._multi.pop((l_id, "k_conv"))[0], slot)
        elif stem in KDA_OFFSETS or stem == "b_proj":
            init = (np.zeros((c.dim, KDA_PROJ_COLS), BF16), set())
            buf, seen = self._multi.setdefault((l_id, "k_projection"), init)
            c0, w = (rc * 4, hc) if stem == "b_proj" else (KDA_OFFSETS[stem], rc)
            buf[:, c0 : c0 + w] = _np(t, r, w).T
            seen.add(key)
            if self._sync_peak() >= 0 and len(seen) == KDA_PROJ_PARTS:
                self._put(
                    "k_projection", self._multi.pop((l_id, "k_projection"))[0], slot
                )
        elif key == "A_log":
            self._put("k_a_log", _np(t, r, hc).reshape(hc, 1), slot)
        elif key == "dt_bias":
            self._put("k_dt", _np(t, r, rc).reshape(hc, hd), slot)
        elif key == "o_norm.weight":
            pad = np.pad(_np(t).reshape(1, hd), ((0, KDA_NORM_PAD), (0, 0)))
            self._put("k_norm", pad, slot)

    def _on_mla(self, l_id: int, key: str, t: torch.Tensor) -> None:
        """Consume MLA LoRA projections, output gate, and layernorms."""
        r, c, slot = self.rank, self._cfg, (self._mla_i, l_id)
        if self._simple(key, t, slot):
            return
        hc = c.heads // TP_RANKS
        if key == "kv_a_proj_with_mqa.weight":
            self._put("m_ka", np.pad(_np(t).T, ((0, 0), (0, c.qk_extra))), slot)
        elif key == "q_b_proj.weight":
            sub = _np(t, r, hc * QB_HEAD_W).T.reshape(-1, hc, QB_HEAD_W)
            p = np.pad(sub, ((0, 0), (0, 0), (0, c.qk_extra))).reshape(-1, hc * KB_HEAD_W)
            self._put("m_qb", p, slot)

    def _on_norms(self, l_id: int, sfx: str, t: torch.Tensor) -> None:
        """Consume layer-0 dense MLP weights and per-layer norms and residuals."""
        r, dr = self.rank, self._cfg.dense_hidden // TP_RANKS
        if l_id == 0 and sfx == "mlp.down_proj.weight":
            a = _np(t[:, r * dr : (r + 1) * dr]).T
            self._put("dense_down", np.pad(a, ((0, DENSE_PAD - dr), (0, 0))))
        elif l_id == 0 and sfx in {"mlp.gate_proj.weight", "mlp.up_proj.weight"}:
            pad = np.pad(_np(t, r, dr).T, ((0, 0), (0, DENSE_PAD - dr)))
            self._pair((0, "dense_gu"), "gate" if "gate" in sfx else "up", pad)
        elif l_id in self._norm_i and not self._simple(sfx, t, (self._norm_i, l_id)):
            if sfx.startswith(("self_attention_res_", "mlp_res_")) and sfx.endswith(
                ("_norm.weight", "_proj.weight")
            ):
                fam, tag = (
                    ("attn_res" if sfx[0] == "s" else "mlp_res"),
                    ("norm" if "_norm." in sfx else "proj"),
                )
                self._pair((l_id, fam), tag, _np(t).astype(np.float32))

    def consume(self, name: str, tensor: torch.Tensor) -> object:
        """Consume one checkpoint tensor and return `WEIGHT_CONSUMED`."""
        norm = (
            "lm_head.weight"
            if name in LM_KEYS
            else name.removeprefix(PREFIX).removeprefix("model.")
        )
        vs = self._cfg.vocab // TP_RANKS
        if norm in {"embed_tokens.weight", "lm_head.weight"}:
            arr = _np(tensor, self.rank, vs)
            fam = "embedding" if norm.startswith("embed") else "lm_head"
            self._put(fam, arr if norm.startswith("embed") else arr.T.copy())
        elif norm == "norm.weight":
            self._put("final_norm", _np(tensor).reshape(1, -1))
        elif norm in {"output_attn_res_norm.weight", "output_attn_res_proj.weight"}:
            tag = "norm" if norm.endswith("_norm.weight") else "proj"
            self._pair((None, "output_res"), tag, _np(tensor).astype(np.float32))
        elif (
            norm.startswith("layers.")
            and len(parts := norm.split(".")) >= LAYER_KEY_PARTS
            and parts[1].isdigit()
        ):
            l_id, sfx = int(parts[1]), ".".join(parts[2:])
            if sfx.startswith("block_sparse_moe.experts."):
                self._on_expert(l_id, parts, tensor)
            elif sfx.startswith("self_attn.") and l_id in self._kda_i:
                self._on_kda(l_id, sfx[len("self_attn.") :], tensor)
            elif sfx.startswith("self_attn.") and l_id in self._mla_i:
                self._on_mla(l_id, sfx[len("self_attn.") :], tensor)
            elif sfx.startswith("block_sparse_moe.") and l_id in self._moe_i:
                self._on_moe(l_id, sfx[len("block_sparse_moe.") :], tensor)
            else:
                self._on_norms(l_id, sfx, tensor)
        return WEIGHT_CONSUMED
