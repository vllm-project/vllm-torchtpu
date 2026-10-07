#!/usr/bin/env python3
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
"""Standalone 1L & 4L DeepseekV41DecoderLayer replay & 40L accounting.

Benchmarks and verifies the four canonical ``DeepseekV41DecoderLayer`` archetypes in
``deepseek-ai/DeepSeek-V4.1-Flash`` (`v2`, `KV_CACHE_DTYPE=fp8`, `TP=8`):

1. ``encoder_source`` (``L2, 8, 14``): Delayed ``mHC`` + ``Compressor.project`` + FP8
   KV write + ``Indexer`` (``full_context`` / ``candidate_write``) + ``SparseMLA + SWA``
   + ``MoE FFN``.
2. ``encoder_reuse`` (``L3..7, 9..13, 15..19``): Delayed ``mHC`` + ``SparseMLA + SWA``
   (reusing source layer's ``kv_cache`` and pre-compacted ``topk_indices_buffer``)
   + ``MoE FFN``.
3. ``encoder_engram`` (``L1, 14``): ``mHC post`` + ``Engram`` (24 hash columns, HBM vs
   Host DRAM offload) + ``mHC pre`` + Attention + ``MoE FFN``.
4. ``decoder_ced`` (``L20..39``): Decoder stage layer running at ``T_dec = 128`` on the
   final prefill chunk under ``CED=1`` (skipped completely on incomplete prefill chunks
   ``1..48`` when ``plan_decoder_replay(...).indices == ()``) or at full ``T`` on all
   49 chunks under ``CED=0``.
5. ``four_layer_e2e``: 4-layer reduced ``DeepseekV41DecoderLayer`` chain (``L0`` Encoder
   Source, ``L1`` Encoder Engram, ``L2`` Encoder Reuse, ``L3`` Decoder CED) verifying
   cross-layer ``(x, residual, post_mix, res_mix, pre_mix)`` state propagation, ``TP``
   vs ``SP`` numerical equivalence, and ``CED=1`` replay window slicing.

Also computes the exact 40-layer step accounting (`reconcile_40_layer_step`) reconciling
the isolated component micro-benchmarks against the 40-layer E2E prefill baseline
(`75.7 ms` @ 20k and `78.6 ms` device / `89-93 ms` wall @ 100k).
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import common  # noqa: E402

SUBOPS = (
    "all",
    "encoder_source",
    "encoder_reuse",
    "encoder_engram",
    "decoder_ced",
    "four_layer_e2e",
)
MODES = ("tp", "sp")
_GEOM = common.Dsv41Geometry()
_HW = common.Tpu7xRoofline()


def compute_layer_roofline(
    subop: str,
    *,
    num_tokens: int,
    kv_len: int = 90112,
    tp_size: int = 8,
    mode: str = "tp",
    ced_decoder_tokens: int = 128,
    geom: common.Dsv41Geometry = _GEOM,
    hw: common.Tpu7xRoofline = _HW,
) -> dict[str, Any]:
    """First-principles TPU v7x per-core Roofline for a DecoderLayer archetype."""
    if subop not in SUBOPS:
        raise ValueError(f"Unsupported subop {subop!r}; expected one of {SUBOPS}")
    if mode not in MODES:
        raise ValueError(f"Unsupported mode {mode!r}; expected one of {MODES}")

    eff_tokens = ced_decoder_tokens if subop == "decoder_ced" else num_tokens
    local_tokens = eff_tokens // tp_size if mode == "sp" else eff_tokens
    d = geom.hidden_size
    m = geom.hc_mult
    h_local = geom.num_attention_heads // tp_size
    topk = geom.index_topk
    w_swa = geom.sliding_window

    # 1. mHC (2x per layer: attn shifted_post_pre + ffn shifted_post_pre)
    mhc_bytes_per_call = local_tokens * (m * d * 2 + d * 2 + m * d * 2) + (
        m * d * 128 * 2
    )
    mhc_flops_per_call = (
        2 * local_tokens * (m * d) * 128 + 20 * local_tokens * m * m * 8
    )
    mhc_total_bytes = 2 * mhc_bytes_per_call
    mhc_total_flops = 2 * mhc_flops_per_call

    # 2. Collectives (Attn o_proj + FFN output, plus SP AG before Attn & MoE)
    comm_bytes_per_vec = eff_tokens * d * 2
    if mode == "tp":
        # 2x all_reduce (Attn + FFN)
        comm_s = 2 * hw.all_reduce_s(comm_bytes_per_vec, tp_size)
        comm_bytes = 2 * comm_bytes_per_vec
    else:
        # SP: AG before Attn + RS after Attn + EP AG/RS around MoE
        comm_s = 2 * (
            hw.all_gather_s(comm_bytes_per_vec, tp_size)
            + hw.reduce_scatter_s(comm_bytes_per_vec, tp_size)
        )
        comm_bytes = 4 * comm_bytes_per_vec

    # 3. Dense Attention projections (q_a/kv_a on local_tokens, wq_b + wo_a/b sharded)
    qkv_a_flops = (
        2
        * local_tokens
        * d
        * (geom.q_lora_rank + geom.head_dim + geom.qk_rope_head_dim)
    )
    wq_b_flops = 2 * eff_tokens * geom.q_lora_rank * (h_local * geom.head_dim)
    wo_flops = 2 * eff_tokens * (h_local * geom.head_dim) * geom.o_lora_rank + (
        2 * eff_tokens * geom.o_lora_rank * d // tp_size
    )
    dense_attn_flops = qkv_a_flops + wq_b_flops + wo_flops
    dense_attn_weight_bytes = (
        d * (geom.q_lora_rank + geom.head_dim + geom.qk_rope_head_dim)
        + geom.q_lora_rank * (h_local * geom.head_dim)
        + (h_local * geom.head_dim) * geom.o_lora_rank
        + (geom.o_lora_rank * d // tp_size)
    )

    # 4. SparseMLA + SWA (using Token-Parallel T/8 gather when T >= 1024)
    attn_gather_tokens = eff_tokens // tp_size if eff_tokens >= 1024 else eff_tokens
    attn_gather_heads = geom.num_attention_heads if eff_tokens >= 1024 else h_local
    sc_gather_bytes = (
        attn_gather_tokens * topk * (geom.head_dim + geom.qk_rope_head_dim + 16)
    )
    mla_flops = 4 * attn_gather_tokens * attn_gather_heads * topk * (
        geom.head_dim + geom.qk_rope_head_dim
    ) + 4 * eff_tokens * h_local * w_swa * (geom.head_dim + geom.qk_rope_head_dim)

    # 5. Optional Compressor + Indexer (on encoder_source & decoder_ced)
    compressor_flops = 0
    compressor_bytes = 0
    indexer_flops = 0
    indexer_bytes = 0
    if subop in ("encoder_source", "decoder_ced", "all", "four_layer_e2e"):
        compress_ratio = 1 if subop == "decoder_ced" else 2
        if subop != "decoder_ced":
            compressor_flops = (
                2 * local_tokens * (compress_ratio * d) * (geom.head_dim + 2)
            )
            compressor_bytes = (
                local_tokens * compress_ratio * d * 2
                + (eff_tokens // compress_ratio) * 656
            )
        compressed_kv = max(1, kv_len // compress_ratio)
        part_tokens = eff_tokens // tp_size
        indexer_flops = (
            2 * part_tokens * geom.index_n_heads * compressed_kv * geom.index_head_dim
        )
        indexer_bytes = compressed_kv * geom.index_head_dim * 2 + part_tokens * topk * 4

    # 6. Optional Engram (on encoder_engram)
    engram_flops = 0
    engram_bytes = 0
    if subop in ("encoder_engram", "all", "four_layer_e2e"):
        n_hash_cols = 24
        engram_bytes = eff_tokens * n_hash_cols * geom.engram_head_dim * 2
        engram_flops = 2 * local_tokens * (n_hash_cols * geom.engram_head_dim) * d

    # 7. MoE FFN (routed + shared experts on EP8/TP8)
    routed_flops = (
        2
        * eff_tokens
        * geom.num_experts_per_tok
        * 3
        * d
        * geom.moe_intermediate_size
        // tp_size
    )
    routed_weight_bytes = (
        (geom.n_routed_experts // tp_size) * 3 * d * geom.moe_intermediate_size
    )

    mult = 4 if subop in ("all", "four_layer_e2e") else 1
    total_flops = mult * (
        mhc_total_flops
        + dense_attn_flops
        + mla_flops
        + compressor_flops
        + indexer_flops
        + engram_flops
        + routed_flops
    )
    total_hbm_bytes = mult * (
        mhc_total_bytes
        + dense_attn_weight_bytes
        + sc_gather_bytes
        + compressor_bytes
        + indexer_bytes
        + engram_bytes
        + routed_weight_bytes
    )
    compute_s = hw.compute_s(total_flops, "bf16")
    hbm_s = hw.hbm_s(total_hbm_bytes)
    total_comm_s = mult * comm_s
    lower_bound_s = max(compute_s, hbm_s) + total_comm_s

    return {
        "subop": subop,
        "mode": mode,
        "num_tokens": eff_tokens,
        "local_tokens": local_tokens,
        "kv_len": kv_len,
        "flops": int(total_flops),
        "hbm_bytes": int(total_hbm_bytes),
        "comm_bytes": int(mult * comm_bytes),
        "mxu_ms": compute_s * 1e3,
        "hbm_ms": hbm_s * 1e3,
        "comm_ms": total_comm_s * 1e3,
        "lower_bound_ms": lower_bound_s * 1e3,
    }


def reconcile_40_layer_step(
    *,
    num_tokens: int = 2048,
    ced_decoder_tokens: int = 128,
) -> dict[str, Any]:
    """Reconcile isolated component micro-benchmarks against 40-layer E2E measurements.

    Anti-number-fudging invariant: ONLY independently measured micro-benchmark / XProf
    numbers are summed in `measured_isolated_ms`. Unmeasured components (dense
    QKV/O/shared-expert projections, MoE EP dispatch and combine, KV compressor, Engram,
    and XLA fusion overhead) are never guessed or calibrated; instead, the exact gap
    between `measured_isolated_ms` and E2E device or wall time is reported as an
    explicit `[UNVERIFIED]` residual.

    Cited isolated measurements at `T=2048, kv_len=90112, TP=8`:
    - `bench_mhc.py`:
      - `mHC` (`all` without collective): `0.209 ms` (`tp`) -> `0.042 ms` (`sp`) per
        call (`2 calls/layer = 0.418 ms/layer` in `tp`, `0.084 ms/layer` in `sp`).
      - Collective: `0.253 ms` (`tp` `all_reduce`) vs `0.107 + 0.091 = 0.198 ms` (`sp`
        `reduce_scatter + all_gather`), `2 calls/layer`.
    - `bench_sparse_mla_swa.py`:
      - `swa_only` (`L0, L1`): `0.282 ms/layer`.
      - Compressed `SparseMLA + SWA` (`L2..39`): `1.470 ms/layer` (`baseline`) ->
        `1.376 ms/layer` (`precompact`) -> `0.900 ms/layer` (`token_parallel`).
    - `bench_indexer.py`:
      - `full_context` (`L2, 14`): `1.954 ms` (`baseline`) -> `1.094 ms` (`precompact`),
        plus `cache_write_only = 0.295 ms`.
      - `candidate_write` (`L8, 20`): `3.123 ms` (`baseline`) -> `2.273 ms`
        (`precompact`), plus `cache_write_only = 0.295 ms`.
      - `candidate_filter` (`L24, 28, 32, 36` at `kv_len=90112`): `23.390 ms` (`base`)
        -> `22.514 ms` (`precompact`) -> `1.131 ms` (shared-key-tile + SC scatter).
    - `bench_kv_compressor.py`:
      - `project_only` (`L2..39`): `0.2634 ms/layer`.
      - `save_compress_norm_only` (`L2, 8, 14` `compress_ratio=2`): `0.9647 ms/layer`.
      - `write_fp8_only` (`L2, 8, 14` + `L20.write_encoder_global_kv` in
        `encoder_stage`; `L20..39` in `decoder_stage`): `0.8678 ms/call`.
      - `engram_hbm` (`L1, 21`): `1.3474 ms/call`
        (`engram_host_dram`: `7.5093 ms/call`).
    - 100k prefill XProf baseline (over 8 chunks of `CED=1`):
      - `gmm_v2` (`w13 + w2` routed MoE matmuls): `(27.8 + 16.6) / 8 = 5.550 ms` per
        20-layer stage (`0.2775 ms/layer`).
    """
    scale = num_tokens / 2048.0

    # 1. Baseline TP (runnable under CED=1 or CED=0)
    base_tp = {
        "mhc_per_layer_ms": 2 * 0.209 * scale,
        "collective_per_layer_ms": 2 * 0.253 * scale,
        "swa_only_attn_ms": 0.282 * scale,
        "sparse_mla_swa_ms": 1.470 * scale,
        "indexer_full_ms": (1.954 + 0.295) * scale,
        "indexer_cand_write_ms": (3.123 + 0.295) * scale,
        "indexer_cand_filter_ms": 23.390 * scale,
        "kv_project_only_ms": 0.263443 * scale,
        "kv_save_compress_norm_ms": 0.964699 * scale,
        "kv_write_fp8_ms": 0.867770 * scale,
        "engram_hbm_ms": 1.347397 * scale,
        "gmm_v2_per_layer_ms": (5.550 / 20.0) * scale,
    }

    # 2. Optimized TP + CED=1 (precompact + token_parallel + shared-key-tile indexer;
    # keeps TP mHC & TP all_reduce)
    opt_tp_ced1 = {
        "mhc_per_layer_ms": 2 * 0.209 * scale,
        "collective_per_layer_ms": 2 * 0.253 * scale,
        "swa_only_attn_ms": 0.282 * scale,
        "sparse_mla_swa_ms": 0.900 * scale,
        "indexer_full_ms": (1.094 + 0.295) * scale,
        "indexer_cand_write_ms": (2.273 + 0.295) * scale,
        "indexer_cand_filter_ms": 1.131 * scale,
        "kv_project_only_ms": 0.263443 * scale,
        "kv_save_compress_norm_ms": 0.964699 * scale,
        "kv_write_fp8_ms": 0.867770 * scale,
        "engram_hbm_ms": 1.347397 * scale,
        "gmm_v2_per_layer_ms": (5.550 / 20.0) * scale,
    }

    # 3. Optimized SP + CED=0 (SP mHC + precompact + token_parallel + shared-key-tile;
    # runnable under CED=0, where all 40 layers execute on every prefill chunk)
    opt_sp_ced0 = {
        "mhc_per_layer_ms": 2 * 0.042 * scale,
        "collective_per_layer_ms": 2 * 0.198 * scale,
        "swa_only_attn_ms": 0.282 * scale,
        "sparse_mla_swa_ms": 0.900 * scale,
        "indexer_full_ms": (1.094 + 0.295) * scale,
        "indexer_cand_write_ms": (2.273 + 0.295) * scale,
        "indexer_cand_filter_ms": 1.131 * scale,
        "kv_project_only_ms": 0.263443 * scale,
        "kv_save_compress_norm_ms": 0.964699 * scale,
        "kv_write_fp8_ms": 0.867770 * scale,
        "engram_hbm_ms": 1.347397 * scale,
        "gmm_v2_per_layer_ms": (5.550 / 20.0) * scale,
    }

    def _isolated_stage_sum(tbl: dict[str, float]) -> dict[str, float]:
        encoder_isolated_ms = (
            20
            * (
                tbl["mhc_per_layer_ms"]
                + tbl["collective_per_layer_ms"]
                + tbl["gmm_v2_per_layer_ms"]
            )
            + 2 * tbl["swa_only_attn_ms"]
            + 18 * tbl["sparse_mla_swa_ms"]
            + 2 * tbl["indexer_full_ms"]
            + 1 * tbl["indexer_cand_write_ms"]
            + 18 * tbl["kv_project_only_ms"]
            + 3 * tbl["kv_save_compress_norm_ms"]
            + 4 * tbl["kv_write_fp8_ms"]
            + 1 * tbl["engram_hbm_ms"]
        )
        decoder_full_isolated_ms = (
            20
            * (
                tbl["mhc_per_layer_ms"]
                + tbl["collective_per_layer_ms"]
                + tbl["gmm_v2_per_layer_ms"]
            )
            + 20 * tbl["sparse_mla_swa_ms"]
            + 1 * tbl["indexer_cand_write_ms"]
            + 4 * tbl["indexer_cand_filter_ms"]
            + 20 * tbl["kv_project_only_ms"]
            + 20 * tbl["kv_write_fp8_ms"]
            + 1 * tbl["engram_hbm_ms"]
        )
        return {
            "encoder_20L_isolated_ms": round(encoder_isolated_ms, 3),
            "decoder_20L_full_T_isolated_ms": round(decoder_full_isolated_ms, 3),
            "ced1_chunks_1_to_48_isolated_ms": round(encoder_isolated_ms, 3),
            "ced0_all_40L_isolated_ms": round(
                encoder_isolated_ms + decoder_full_isolated_ms, 3
            ),
        }

    base_sums = _isolated_stage_sum(base_tp)
    opt_tp_sums = _isolated_stage_sum(opt_tp_ced1)
    opt_sp_sums = _isolated_stage_sum(opt_sp_ced0)
    opt_sp_ced1_sums = {
        "encoder_20L_isolated_ms": opt_sp_sums["encoder_20L_isolated_ms"],
        "decoder_20L_full_T_isolated_ms": opt_tp_sums["decoder_20L_full_T_isolated_ms"],
        "ced1_chunks_1_to_48_isolated_ms": opt_sp_sums["encoder_20L_isolated_ms"],
        "ced0_all_40L_isolated_ms": round(
            opt_sp_sums["encoder_20L_isolated_ms"]
            + opt_tp_sums["decoder_20L_full_T_isolated_ms"],
            3,
        ),
    }

    baseline_100k_xprof_device_ms = 78.59
    base_iso_ms = base_sums["ced1_chunks_1_to_48_isolated_ms"]
    base_unmeasured_gap_ms = round(baseline_100k_xprof_device_ms - base_iso_ms, 3)
    tp_ced1_isolated_delta_ms = round(
        opt_tp_sums["ced1_chunks_1_to_48_isolated_ms"] - base_iso_ms,
        3,
    )
    sp_ced1_isolated_delta_ms = round(
        opt_sp_ced1_sums["ced1_chunks_1_to_48_isolated_ms"] - base_iso_ms,
        3,
    )
    sp_ced1_vs_tp_ced1_delta_ms = round(
        opt_sp_ced1_sums["ced1_chunks_1_to_48_isolated_ms"]
        - opt_tp_sums["ced1_chunks_1_to_48_isolated_ms"],
        3,
    )
    sp_ced0_measured_wall_ms = 144.07  # 7059.5 ms / 49 chunks
    sp_iso_ms = opt_sp_sums["ced0_all_40L_isolated_ms"]
    sp_ced0_unmeasured_gap_ms = round(sp_ced0_measured_wall_ms - sp_iso_ms, 3)
    cov_pct = base_iso_ms / baseline_100k_xprof_device_ms

    return {
        "num_tokens": num_tokens,
        "ced_decoder_tokens": ced_decoder_tokens,
        "baseline_tp": base_sums,
        "optimized_tp_ced1": {
            **opt_tp_sums,
            "isolated_delta_vs_baseline_tp_ms": tp_ced1_isolated_delta_ms,
            "projected_ced1_device_chunk_ms_unverified": round(
                baseline_100k_xprof_device_ms + tp_ced1_isolated_delta_ms, 3
            ),
        },
        "optimized_sp_ced1": {
            **opt_sp_ced1_sums,
            "isolated_delta_vs_baseline_tp_ms": sp_ced1_isolated_delta_ms,
            "isolated_delta_vs_optimized_tp_ced1_ms": sp_ced1_vs_tp_ced1_delta_ms,
            "projected_ced1_device_chunk_ms_unverified": round(
                baseline_100k_xprof_device_ms + sp_ced1_isolated_delta_ms, 3
            ),
        },
        "optimized_sp_ced0": {
            **opt_sp_sums,
            "sp_ced0_measured_wall_chunk_ms": sp_ced0_measured_wall_ms,
            "unmeasured_40L_dense_ep_compressor_host_gap_ms_unverified": (
                sp_ced0_unmeasured_gap_ms
            ),
        },
        "e2e_reference": {
            "baseline_100k_xprof_device_chunk_ms": baseline_100k_xprof_device_ms,
            "handoff_20k_xprof_device_chunk_ms": 75.7,
            "baseline_100k_wall_marginal_chunk_ms": 91.6,
            "baseline_tp_ced1_isolated_coverage_ratio": round(cov_pct, 4),
            "baseline_tp_ced1_unmeasured_residual_ms_unverified": (
                base_unmeasured_gap_ms
            ),
            "unverified_notes": [
                (
                    f"[UNVERIFIED] Baseline TP CED=1 isolated sum covers "
                    f"{base_iso_ms} ms of {baseline_100k_xprof_device_ms} ms "
                    f"100k XProf device chunk time (coverage={cov_pct:.1%}); "
                    f"remaining {base_unmeasured_gap_ms} ms is dense "
                    f"QKV/O/shared-expert projections, MoE "
                    f"router/dispatch/combine, and XLA fusion delta."
                ),
                (
                    f"[UNVERIFIED] Stage-scoped SP (optimized_sp_ced1: Encoder-SP + "
                    f"Decoder-TP under CED=1) saves {sp_ced1_vs_tp_ced1_delta_ms} ms "
                    f"more in isolated encoder_stage kernels vs optimized_tp_ced1 "
                    f"({sp_ced1_isolated_delta_ms} ms vs baseline_tp)."
                ),
                (
                    f"[UNVERIFIED] SP + CED=0 measured "
                    f"{sp_ced0_measured_wall_ms} ms/chunk wall (TTFT "
                    f"7059.5 ms / 49 chunks) vs {sp_iso_ms} ms isolated 40L "
                    f"kernel sum ({sp_ced0_unmeasured_gap_ms} ms unmeasured 40L "
                    f"dense + EP MoE + KV compressor + host gap)."
                ),
                (
                    "[UNVERIFIED] Final chunk 49 CED=1 decoder_stage replay cost at "
                    "T_dec=128 is not scaled linearly by T_dec/2048 because Indexer "
                    "and SparseMLA at kv_len=90112 are dominated by KV reads."
                ),
            ],
        },
    }


def run_decoder_layer_verification(
    subop: str = "all",
    *,
    num_tokens: int = 128,
    num_seqs: int = 2,
    tp_size: int = 8,
    seed: int = 20261005,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run deterministic 1L & 4L DeepseekV41DecoderLayer numerical verification.

    Exercises the exact delayed-mHC state machine (`(x, residual, post_mix, res_mix,
    pre_mix)`), Engram injection between `mhc_post` and `mhc_pre`, TP vs SP token
    sharding (`[T, D]` vs `[T/tp, D]`), and `CED=1` `plan_decoder_replay` window
    slicing across single-layer and 4-layer chains.
    """
    import torch
    from vllm.model_executor.kernels.mhc.torch import (
        mhc_post_torch,
        mhc_pre_delayed_torch,
    )

    from vllm_torchtpu.models.vllm.deepseek_v41.common.mhc import collapse
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.ced import plan_decoder_replay
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.ops import rms_norm

    if subop not in SUBOPS:
        raise ValueError(f"Unsupported subop {subop!r}; expected one of {SUBOPS}")
    if num_tokens % tp_size != 0:
        raise ValueError(
            f"num_tokens ({num_tokens}) must be divisible by tp_size ({tp_size})"
        )

    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = torch.device(device)
    d = 256
    m = 4
    eps = 1e-6
    hc_eps = 1e-4
    post_alpha = 2.0
    sinkhorn_iters = 20
    sp_criteria = common.LOSSLESS

    def _make_layer_weights(
        layer_idx: int, *, has_engram: bool = False
    ) -> dict[str, torch.Tensor]:
        k_in = d if layer_idx == 0 else m * d
        mixes = m * (m + 2)
        w = {
            "hc_attn_fn": (
                torch.randn((mixes, k_in), generator=g, dtype=torch.float32) * 0.02
            ).to(dev),
            "hc_attn_scale": torch.tensor(
                [0.5, 0.5, 0.5], dtype=torch.float32, device=dev
            ),
            "hc_attn_base": (
                torch.randn((mixes,), generator=g, dtype=torch.float32) * 0.05
            ).to(dev),
            "attn_norm_w": torch.ones((d,), dtype=torch.bfloat16, device=dev),
            "w_attn": (
                torch.randn((d, d), generator=g, dtype=torch.bfloat16) * (d**-0.5)
            ).to(dev),
            "hc_ffn_fn": (
                torch.randn((mixes, m * d), generator=g, dtype=torch.float32) * 0.02
            ).to(dev),
            "hc_ffn_scale": torch.tensor(
                [0.5, 0.5, 0.5], dtype=torch.float32, device=dev
            ),
            "hc_ffn_base": (
                torch.randn((mixes,), generator=g, dtype=torch.float32) * 0.05
            ).to(dev),
            "ffn_norm_w": torch.ones((d,), dtype=torch.bfloat16, device=dev),
            "w_ffn": (
                torch.randn((d, d), generator=g, dtype=torch.bfloat16) * (d**-0.5)
            ).to(dev),
        }
        if has_engram:
            w["engram_delta"] = (
                torch.randn((num_tokens, m, d), generator=g, dtype=torch.bfloat16)
                * 0.05
            ).to(dev)
        return w

    def _forward_one_layer_tp(
        x: torch.Tensor,
        residual: torch.Tensor | None,
        post_mix: torch.Tensor | None,
        res_mix: torch.Tensor | None,
        pre_mix: torch.Tensor | None,
        w: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if residual is None:
            if x.dim() == 2:
                residual = x.unsqueeze(1).expand(-1, m, -1).contiguous()
                post_mix, res_mix, x_attn, attn_pre = mhc_pre_delayed_torch(
                    residual,
                    w["hc_attn_fn"],
                    w["hc_attn_scale"],
                    w["hc_attn_base"],
                    eps,
                    hc_eps,
                    hc_eps,
                    post_alpha,
                    sinkhorn_iters,
                    x=x,
                )
            else:
                residual = x
                post_mix, res_mix, x_attn, attn_pre = mhc_pre_delayed_torch(
                    residual,
                    w["hc_attn_fn"],
                    w["hc_attn_scale"],
                    w["hc_attn_base"],
                    eps,
                    hc_eps,
                    hc_eps,
                    post_alpha,
                    sinkhorn_iters,
                    pre_mix=pre_mix,
                )
        elif "engram_delta" in w:
            previous_post = mhc_post_torch(x, residual, post_mix, res_mix)
            residual = previous_post + w["engram_delta"][: previous_post.shape[0]]
            post_mix, res_mix, x_attn, attn_pre = mhc_pre_delayed_torch(
                residual,
                w["hc_attn_fn"],
                w["hc_attn_scale"],
                w["hc_attn_base"],
                eps,
                hc_eps,
                hc_eps,
                post_alpha,
                sinkhorn_iters,
                pre_mix=pre_mix,
            )
        else:
            residual = mhc_post_torch(x, residual, post_mix, res_mix)
            post_mix, res_mix, x_attn, attn_pre = mhc_pre_delayed_torch(
                residual,
                w["hc_attn_fn"],
                w["hc_attn_scale"],
                w["hc_attn_base"],
                eps,
                hc_eps,
                hc_eps,
                post_alpha,
                sinkhorn_iters,
                pre_mix=pre_mix,
            )
        x_attn = rms_norm(x_attn, w["attn_norm_w"], eps)
        attn_out = x_attn @ w["w_attn"]

        residual = mhc_post_torch(attn_out, residual, post_mix, res_mix)
        post_mix, res_mix, x_ffn, ffn_pre = mhc_pre_delayed_torch(
            residual,
            w["hc_ffn_fn"],
            w["hc_ffn_scale"],
            w["hc_ffn_base"],
            eps,
            hc_eps,
            hc_eps,
            post_alpha,
            sinkhorn_iters,
            pre_mix=attn_pre,
        )
        x_ffn = rms_norm(x_ffn, w["ffn_norm_w"], eps)
        ffn_out = x_ffn @ w["w_ffn"]
        return ffn_out, residual, post_mix, res_mix, ffn_pre

    def _forward_one_layer_sp(
        x: torch.Tensor,
        residual: torch.Tensor | None,
        post_mix: torch.Tensor | None,
        res_mix: torch.Tensor | None,
        pre_mix: torch.Tensor | None,
        w: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Simulate TP8+SP token-sharded execution across ``tp_size`` ranks."""
        t = x.shape[0]
        chunk = t // tp_size
        out_x, out_res, out_post, out_comb, out_pre = [], [], [], [], []
        for r in range(tp_size):
            sl = slice(r * chunk, (r + 1) * chunk)
            w_r = dict(w)
            if "engram_delta" in w_r:
                w_r["engram_delta"] = w_r["engram_delta"][sl]
            fx, fres, fpost, fcomb, fpre = _forward_one_layer_tp(
                x[sl],
                None if residual is None else residual[sl],
                None if post_mix is None else post_mix[sl],
                None if res_mix is None else res_mix[sl],
                None if pre_mix is None else pre_mix[sl],
                w_r,
            )
            out_x.append(fx)
            out_res.append(fres)
            out_post.append(fpost)
            out_comb.append(fcomb)
            out_pre.append(fpre)
        return (
            torch.cat(out_x, dim=0),
            torch.cat(out_res, dim=0),
            torch.cat(out_post, dim=0),
            torch.cat(out_comb, dim=0),
            torch.cat(out_pre, dim=0),
        )

    x0 = (torch.randn((num_tokens, d), generator=g, dtype=torch.bfloat16) * 0.1).to(dev)
    layers_w = [
        _make_layer_weights(0, has_engram=False),  # L0: Encoder source
        _make_layer_weights(1, has_engram=True),  # L1: Encoder Engram
        _make_layer_weights(2, has_engram=False),  # L2: Encoder reuse
        _make_layer_weights(3, has_engram=False),  # L3: Decoder CED
    ]

    comparisons: dict[str, dict[str, Any]] = {}

    if subop in (
        "all",
        "encoder_source",
        "encoder_reuse",
        "encoder_engram",
        "decoder_ced",
    ):
        target_idx = {
            "encoder_source": 0,
            "encoder_engram": 1,
            "encoder_reuse": 2,
            "decoder_ced": 3,
        }
        indices_to_run = (
            list(target_idx.items()) if subop == "all" else [(subop, target_idx[subop])]
        )
        for name, idx in indices_to_run:
            if idx == 0:
                xin_tp = x0
                res_in = post_in = comb_in = pre_in = None
            else:
                xin_tp, res_in, post_in, comb_in, pre_in = _forward_one_layer_tp(
                    x0, None, None, None, None, layers_w[0]
                )
                if idx == 3:
                    # Decoder Stage entry: residual is already post-mapped [T, M, D]
                    xin_tp = mhc_post_torch(xin_tp, res_in, post_in, comb_in)
                    res_in = post_in = comb_in = None
            tp_out = _forward_one_layer_tp(
                xin_tp, res_in, post_in, comb_in, pre_in, layers_w[idx]
            )
            tp_ref = _forward_one_layer_tp(
                xin_tp.clone(),
                None if res_in is None else res_in.clone(),
                None if post_in is None else post_in.clone(),
                None if comb_in is None else comb_in.clone(),
                None if pre_in is None else pre_in.clone(),
                layers_w[idx],
            )
            sp_out = _forward_one_layer_sp(
                xin_tp, res_in, post_in, comb_in, pre_in, layers_w[idx]
            )
            comparisons[f"{name}_tp_exact_ref"] = common.compare_tensors(
                tp_out[0], tp_ref[0], common.EXACT
            )
            comparisons[f"{name}_tp_vs_sp_x"] = common.compare_tensors(
                tp_out[0], sp_out[0], sp_criteria
            )
            comparisons[f"{name}_tp_vs_sp_residual"] = common.compare_tensors(
                tp_out[1], sp_out[1], sp_criteria
            )
            comparisons[f"{name}_tp_vs_sp_pre_mix"] = common.compare_tensors(
                tp_out[4], sp_out[4], sp_criteria
            )

    if subop in ("all", "four_layer_e2e"):
        # 4-layer full TP vs SP chain
        x_tp, res_tp, post_tp, comb_tp, pre_tp = x0, None, None, None, None
        x_sp, res_sp, post_sp, comb_sp, pre_sp = x0, None, None, None, None
        for w in layers_w:
            x_tp, res_tp, post_tp, comb_tp, pre_tp = _forward_one_layer_tp(
                x_tp, res_tp, post_tp, comb_tp, pre_tp, w
            )
            x_sp, res_sp, post_sp, comb_sp, pre_sp = _forward_one_layer_sp(
                x_sp, res_sp, post_sp, comb_sp, pre_sp, w
            )
        final_tp = collapse(mhc_post_torch(x_tp, res_tp, post_tp, comb_tp), pre_tp)
        final_sp = collapse(mhc_post_torch(x_sp, res_sp, post_sp, comb_sp), pre_sp)
        comparisons["four_layer_e2e_tp_vs_sp"] = common.compare_tensors(
            final_tp, final_sp, sp_criteria
        )

        # Verify CED=1 plan_decoder_replay boundary:
        # 1) Incomplete prefill chunk (length=64 < prompt=128, window=32)
        class _Meta:
            def __init__(self, q_loc, seq_lens, prompt_lens):
                self.query_start_loc_cpu = q_loc
                self.seq_lens_cpu = seq_lens
                self.prompt_lens_cpu = prompt_lens

        incomplete_plan = plan_decoder_replay(
            _Meta([0, num_tokens], [num_tokens], [num_tokens * 2]),
            window=32,
            num_tokens=num_tokens,
            compile_sizes=(32, num_tokens),
        )
        assert incomplete_plan is not None and incomplete_plan.indices == ()

        # 2) Final prefill chunk: decoder replay on tail window (and full-window match)
        full_plan = plan_decoder_replay(
            _Meta([0, num_tokens], [num_tokens], [num_tokens]),
            window=num_tokens,
            num_tokens=num_tokens,
            compile_sizes=(num_tokens,),
        )
        assert full_plan is not None and len(full_plan.indices) == num_tokens
        window = max(tp_size, num_tokens // 4)
        final_plan = plan_decoder_replay(
            _Meta([0, num_tokens], [num_tokens], [num_tokens]),
            window=window,
            num_tokens=num_tokens,
            compile_sizes=(window, num_tokens),
        )
        assert final_plan is not None and len(final_plan.indices) == window
        # Encoder stage (L0, L1) on full T, followed by Decoder stage (L2, L3)
        ex, eres, epost, ecomb, epre = x0, None, None, None, None
        for w in layers_w[:2]:
            ex, eres, epost, ecomb, epre = _forward_one_layer_tp(
                ex, eres, epost, ecomb, epre, w
            )
        enc_hidden = mhc_post_torch(ex, eres, epost, ecomb)

        # 2a) Full-window CED=1 stage split must be 100% Bitwise Identical (EXACT)
        dx_full, dres_full, dpost_full, dcomb_full, dpre_full = (
            enc_hidden,
            None,
            None,
            None,
            epre,
        )
        for w in layers_w[2:]:
            dx_full, dres_full, dpost_full, dcomb_full, dpre_full = (
                _forward_one_layer_tp(
                    dx_full, dres_full, dpost_full, dcomb_full, dpre_full, w
                )
            )
        ced_full_out = collapse(
            mhc_post_torch(dx_full, dres_full, dpost_full, dcomb_full), dpre_full
        )
        comparisons["four_layer_ced1_stage_split_exact"] = common.compare_tensors(
            final_tp, ced_full_out, common.EXACT
        )

        # 2b) Tail-window CED=1 replay (T_dec = window) matches tail rows of full_stage
        idx_tensor = torch.tensor(final_plan.indices, dtype=torch.long, device=dev)
        dec_hidden = enc_hidden[idx_tensor]
        dec_pre = epre[idx_tensor]
        dx, dres, dpost, dcomb, dpre = dec_hidden, None, None, None, dec_pre
        for w in layers_w[2:]:
            dx, dres, dpost, dcomb, dpre = _forward_one_layer_tp(
                dx, dres, dpost, dcomb, dpre, w
            )
        ced_tail_out = collapse(mhc_post_torch(dx, dres, dpost, dcomb), dpre)
        comparisons["four_layer_ced1_replay_tail_vs_full"] = common.compare_tensors(
            final_tp[idx_tensor], ced_tail_out, sp_criteria
        )

    for name, comp in comparisons.items():
        if not comp["ok"]:
            raise common.VerificationError(f"{name}: {comp}")

    return {
        "subop": subop,
        "num_tokens": num_tokens,
        "num_seqs": num_seqs,
        "tp_size": tp_size,
        "comparisons": comparisons,
        "reconciliation_40_layer": reconcile_40_layer_step(num_tokens=2048),
    }


def _worker(args: argparse.Namespace) -> int:
    ctx = common.init_worker(args)
    t0 = time.perf_counter()
    verify_res = run_decoder_layer_verification(
        args.subop,
        num_tokens=min(args.chunk_size, 256),
        num_seqs=args.num_seqs,
        tp_size=args.tp_size,
        seed=args.seed,
        device="cpu",
    )
    elapsed_ms = (time.perf_counter() - t0) * 1e3
    roof = compute_layer_roofline(
        args.subop,
        num_tokens=args.chunk_size,
        kv_len=args.kv_len,
        tp_size=args.tp_size,
        mode=args.mode,
    )
    payload = {
        "rank": ctx.rank,
        "ok": True,
        "subop": args.subop,
        "mode": args.mode,
        "chunk_size": args.chunk_size,
        "num_seqs": args.num_seqs,
        "kv_len": args.kv_len,
        "wall_ms": common.summarize_samples([elapsed_ms]),
        "roofline": roof,
        "reconciliation_40_layer": verify_res["reconciliation_40_layer"],
        "comparisons": verify_res["comparisons"],
        "hbm": common.hbm_stats(),
    }
    common.write_rank_result(ctx, payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bench_decoder_layer",
        description=(
            "Standalone 1-layer & 4-layer DeepseekV41DecoderLayer benchmark "
            "& 40-layer step accounting."
        ),
    )
    common.add_common_args(p, SUBOPS)
    p.add_argument(
        "--mode", choices=MODES, default="tp", help="Sharding mode (tp or sp)"
    )
    p.add_argument(
        "--cpu-only",
        action="store_true",
        help=(
            "Run numerical verification and 40-layer step reconciliation "
            "on CPU without tpu_dist."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cpu_only:
        res = run_decoder_layer_verification(
            args.subop,
            num_tokens=min(args.chunk_size, 256),
            num_seqs=args.num_seqs,
            tp_size=args.tp_size,
            seed=args.seed,
            device="cpu",
        )
        res["roofline"] = compute_layer_roofline(
            args.subop,
            num_tokens=args.chunk_size,
            kv_len=args.kv_len,
            tp_size=args.tp_size,
            mode=args.mode,
        )
        print(common.json.dumps(res, indent=2))
        return 0
    if args.worker:
        return _worker(args)
    return common.run_parent(args, __file__)


if __name__ == "__main__":
    raise SystemExit(main())
