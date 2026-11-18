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
"""Model configuration loader and structural assertions for GLM-5.3-Flash."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Glm53MegakernelConfig:
    """Validated GLM-5.3-Flash architectural constants read from config.json."""

    num_hidden_layers: int
    first_k_dense_replace: int
    hidden_size: int
    vocab_size: int
    rms_norm_eps: float
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    kda_num_heads: int
    kda_head_dim: int
    kda_conv_kernel_size: int
    kda_gate_lower_bound: float
    kda_layers: tuple[int, ...]
    mla_layers: tuple[int, ...]
    mla_num_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    intermediate_size: int
    n_routed_experts: int
    num_experts_per_tok: int
    n_shared_experts: int
    moe_intermediate_size: int
    routed_scaling_factor: float
    swiglu_limit: float
    norm_topk_prob: bool
    scoring_func: str
    topk_method: str
    num_nextn_predict_layers: int
    weight_block_size: tuple[int, int]
    eos_token_ids: tuple[int, ...]
    pad_token_id: int
    page_size: int = 136

    def validate(self, world_size: int = 8) -> None:
        """Assert divisibility and structural invariants for EP x TP sharding."""
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}")
        if self.num_hidden_layers <= self.first_k_dense_replace:
            raise ValueError(
                f"num_hidden_layers={self.num_hidden_layers} must exceed "
                f"first_k_dense_replace={self.first_k_dense_replace}"
            )
        if len(self.kda_layers) + len(self.mla_layers) != self.num_hidden_layers:
            raise ValueError(
                f"kda_layers ({len(self.kda_layers)}) + mla_layers "
                f"({len(self.mla_layers)}) != {self.num_hidden_layers}"
            )
        if sorted((*self.kda_layers, *self.mla_layers)) != list(
            range(self.num_hidden_layers)
        ):
            raise ValueError("kda_layers and mla_layers must partition 0..L-1")
        if self.qk_rope_head_dim != 0:
            raise ValueError(
                "Absorbed NoPE MLA requires qk_rope_head_dim=0, got "
                f"{self.qk_rope_head_dim}"
            )
        if self.kda_num_heads % world_size != 0:
            raise ValueError(
                f"kda_num_heads={self.kda_num_heads} not divisible by {world_size}"
            )
        if self.mla_num_heads % world_size != 0:
            raise ValueError(
                f"mla_num_heads={self.mla_num_heads} not divisible by {world_size}"
            )
        if self.n_routed_experts % world_size != 0:
            raise ValueError(
                f"n_routed_experts={self.n_routed_experts} not divisible by "
                f"{world_size}"
            )
        if self.intermediate_size % world_size != 0:
            raise ValueError(
                f"intermediate_size={self.intermediate_size} not divisible by "
                f"{world_size}"
            )
        if (self.n_shared_experts * self.moe_intermediate_size) % world_size != 0:
            raise ValueError(
                "shared expert intermediate size not divisible by world_size"
            )
        if self.vocab_size % world_size != 0:
            raise ValueError(
                f"vocab_size={self.vocab_size} not divisible by {world_size}"
            )
        bh, bw = self.weight_block_size
        if bh <= 0 or bw <= 0 or bh != bw:
            raise ValueError(f"Invalid weight_block_size={self.weight_block_size}")
        if self.hc_mult != 4 or self.hc_sinkhorn_iters <= 0:
            raise ValueError(
                "Expected hc_mult=4 and positive sinkhorn_iters, got "
                f"{self.hc_mult}, {self.hc_sinkhorn_iters}"
            )
        if self.scoring_func != "sigmoid" or self.topk_method != "noaux_tc":
            raise ValueError(
                "Expected sigmoid/noaux_tc routing, got "
                f"{self.scoring_func}/{self.topk_method}"
            )

    def assert_production_glm53(self) -> None:
        """Verify exact released zai-org/GLM-5.3-Flash checkpoint geometry."""
        self.validate(world_size=8)
        expected = {
            "num_hidden_layers": 45,
            "first_k_dense_replace": 3,
            "hidden_size": 4096,
            "vocab_size": 154880,
            "hc_mult": 4,
            "hc_sinkhorn_iters": 20,
            "kda_num_heads": 64,
            "kda_head_dim": 128,
            "kda_conv_kernel_size": 4,
            "mla_num_heads": 64,
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 256,
            "qk_rope_head_dim": 0,
            "v_head_dim": 256,
            "intermediate_size": 12288,
            "n_routed_experts": 288,
            "num_experts_per_tok": 8,
            "n_shared_experts": 1,
            "moe_intermediate_size": 2048,
            "weight_block_size": (128, 128),
        }
        for key, want in expected.items():
            got = getattr(self, key)
            if got != want:
                raise ValueError(f"Checkpoint config {key}={got!r}, expected {want!r}")

    @classmethod
    def from_dict(
        cls, raw: dict[str, Any], *, page_size: int = 136
    ) -> Glm53MegakernelConfig:
        """Parse and validate a raw HuggingFace config.json dictionary."""
        tc = raw.get("text_config", raw)
        qc = raw.get("quantization_config", {})
        lc = tc.get("linear_attn_config")
        if not isinstance(lc, dict):
            raise ValueError("Missing text_config.linear_attn_config in config.json")
        num_layers = int(tc["num_hidden_layers"])
        layer_types = tc.get("layer_types")
        if isinstance(layer_types, list) and len(layer_types) == num_layers:
            kda_layers = tuple(
                i for i, t in enumerate(layer_types) if t == "linear_attention"
            )
            mla_layers = tuple(
                i for i, t in enumerate(layer_types) if t != "linear_attention"
            )
        else:
            kda_layers = tuple(int(i) for i in lc["kda_layers"])
            mla_layers = tuple(int(i) for i in lc["full_attn_layers"])
        wbs = qc.get("weight_block_size", [128, 128])
        eos_raw = tc.get("eos_token_id", [154820])
        eos_ids = (
            tuple(int(x) for x in eos_raw)
            if isinstance(eos_raw, list)
            else (int(eos_raw),)
        )
        cfg = cls(
            num_hidden_layers=num_layers,
            first_k_dense_replace=int(tc["first_k_dense_replace"]),
            hidden_size=int(tc["hidden_size"]),
            vocab_size=int(tc["vocab_size"]),
            rms_norm_eps=float(tc["rms_norm_eps"]),
            hc_mult=int(tc["hc_mult"]),
            hc_sinkhorn_iters=int(tc["hc_sinkhorn_iters"]),
            hc_eps=float(tc["hc_eps"]),
            kda_num_heads=int(lc["num_heads"]),
            kda_head_dim=int(lc["head_dim"]),
            kda_conv_kernel_size=int(lc["short_conv_kernel_size"]),
            kda_gate_lower_bound=float(lc["gate_lower_bound"]),
            kda_layers=kda_layers,
            mla_layers=mla_layers,
            mla_num_heads=int(tc["num_attention_heads"]),
            q_lora_rank=int(tc["q_lora_rank"]),
            kv_lora_rank=int(tc["kv_lora_rank"]),
            qk_nope_head_dim=int(tc["qk_nope_head_dim"]),
            qk_rope_head_dim=int(tc.get("qk_rope_head_dim", 0)),
            v_head_dim=int(tc["v_head_dim"]),
            intermediate_size=int(tc["intermediate_size"]),
            n_routed_experts=int(tc["n_routed_experts"]),
            num_experts_per_tok=int(tc["num_experts_per_tok"]),
            n_shared_experts=int(tc.get("n_shared_experts", 1)),
            moe_intermediate_size=int(tc["moe_intermediate_size"]),
            routed_scaling_factor=float(tc["routed_scaling_factor"]),
            swiglu_limit=float(tc["swiglu_limit"]),
            norm_topk_prob=bool(tc.get("norm_topk_prob", True)),
            scoring_func=str(tc.get("scoring_func", "sigmoid")),
            topk_method=str(tc.get("topk_method", "noaux_tc")),
            num_nextn_predict_layers=int(tc.get("num_nextn_predict_layers", 1) or 1),
            weight_block_size=(int(wbs[0]), int(wbs[1])),
            eos_token_ids=eos_ids,
            pad_token_id=int(tc.get("pad_token_id", eos_ids[0])),
            page_size=int(page_size),
        )
        cfg.validate(world_size=1)
        return cfg

    @classmethod
    def from_checkpoint(
        cls, checkpoint_dir: str | Path, *, page_size: int = 136
    ) -> Glm53MegakernelConfig:
        """Load and validate config.json from a checkpoint directory."""
        path = Path(checkpoint_dir)
        cfg_file = path / "config.json"
        if not cfg_file.is_file():
            raise FileNotFoundError(f"Missing config.json in {path}")
        raw = json.loads(cfg_file.read_text())
        return cls.from_dict(raw, page_size=page_size)

    @classmethod
    def small_lane_aligned_for_test(
        cls, *, world_size: int = 2, page_size: int = 8
    ) -> Glm53MegakernelConfig:
        """Smallest config meeting the kernel's 128-lane alignment (CPU tests)."""
        cfg = cls(
            num_hidden_layers=4,
            first_k_dense_replace=1,
            hidden_size=256,
            vocab_size=128 * world_size,
            rms_norm_eps=1e-5,
            hc_mult=4,
            hc_sinkhorn_iters=20,
            hc_eps=1e-6,
            kda_num_heads=world_size,
            kda_head_dim=128,
            kda_conv_kernel_size=4,
            kda_gate_lower_bound=-5.0,
            kda_layers=(0, 1, 2),
            mla_layers=(3,),
            mla_num_heads=world_size,
            q_lora_rank=128,
            kv_lora_rank=128,
            qk_nope_head_dim=128,
            qk_rope_head_dim=0,
            v_head_dim=128,
            intermediate_size=128 * world_size,
            n_routed_experts=4 * world_size,
            num_experts_per_tok=2,
            n_shared_experts=1,
            moe_intermediate_size=128 * world_size,
            routed_scaling_factor=2.5,
            swiglu_limit=10.0,
            norm_topk_prob=True,
            scoring_func="sigmoid",
            topk_method="noaux_tc",
            num_nextn_predict_layers=1,
            weight_block_size=(128, 128),
            eos_token_ids=(1, 2),
            pad_token_id=0,
            page_size=page_size,
        )
        cfg.validate(world_size=world_size)
        return cfg

    @classmethod
    def tiny_for_test(
        cls, *, world_size: int = 1, page_size: int = 8
    ) -> Glm53MegakernelConfig:
        """Small scaled-down config for fast CPU unit and round-trip tests."""
        cfg = cls(
            num_hidden_layers=4,
            first_k_dense_replace=1,
            hidden_size=32,
            vocab_size=64 * world_size,
            rms_norm_eps=1e-5,
            hc_mult=4,
            hc_sinkhorn_iters=20,
            hc_eps=1e-6,
            kda_num_heads=2 * world_size,
            kda_head_dim=8,
            kda_conv_kernel_size=4,
            kda_gate_lower_bound=-5.0,
            kda_layers=(0, 1, 2),
            mla_layers=(3,),
            mla_num_heads=2 * world_size,
            q_lora_rank=16,
            kv_lora_rank=16,
            qk_nope_head_dim=8,
            qk_rope_head_dim=0,
            v_head_dim=8,
            intermediate_size=16 * world_size,
            n_routed_experts=8 * world_size,
            num_experts_per_tok=2,
            n_shared_experts=1,
            moe_intermediate_size=16 * world_size,
            routed_scaling_factor=2.5,
            swiglu_limit=10.0,
            norm_topk_prob=True,
            scoring_func="sigmoid",
            topk_method="noaux_tc",
            num_nextn_predict_layers=1,
            weight_block_size=(8, 8),
            eos_token_ids=(1, 2),
            pad_token_id=0,
            page_size=page_size,
        )
        cfg.validate(world_size=world_size)
        return cfg
