# SPDX-License-Identifier: Apache-2.0
"""q38 G1LINE: per-rank, warm-cache-safe engagement lines of the G1 D1 GDN decode body.

The G1 D1 body (VLLM_TPU_GDN_G1_BODY) is chosen while `kernels/gdn/v3/wrapper.py` is
traced, per compiled bucket, so a line printed there is silent whenever the compile
cache serves the bucket (a warm start, or a peer's artifact). The warmup ladder
(`TPUModelRunner._precompile_backbone`) runs every bucket on every start and hands the
(rows, request slots) of each dummy run to `report_buckets`, which evaluates the same
shared functions the trace path calls (`kernels/gdn/v3/g1_select.py`) and prints, per
rank and per bucket, with plain logger.info (never a once-per-node form):

  ENGAGED      the bucket's BATCHED kernel is the `_g1` body
  NOT_ENGAGED  it is not, with the reason (e.g. seq_tile_size=4 at 16 rows)
  MISMATCH     this process traced the bucket (a cold or partly cold start) and the
               trace chose differently from the prediction: printed instead of the
               two above, and readers fail on it
  SUMMARY      one per rank after the ladder

`trace=observed` means this process traced the bucket and the trace agrees;
`trace=skipped` means it did not trace it (compile-cache hit or peer replay), so the
line rests on the shared functions and the compile key (which hashes these sources and
the env keys). With the key unset the module reads nothing else and prints nothing.
Log only: nothing here enters a program.
"""

import os
from collections.abc import Iterable, Sequence
from typing import Any

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

ENGAGED = (
    "q38 G1 D1 engaged: rank=%d tp_rank=%d dp_rank=%d pid=%d variant=%s rows=%d "
    "num_seqs=%d window=%d T=%d kernel=%s trace=%s"
)
NOT_ENGAGED = (
    "q38 G1 D1 not engaged: rank=%d tp_rank=%d dp_rank=%d pid=%d variant=%s rows=%d "
    "num_seqs=%d window=%d T=%d kernel=%s trace=%s reason=%s"
)
MISMATCH = (
    "q38 G1 D1 MISMATCH: rank=%d tp_rank=%d dp_rank=%d pid=%d variant=%s rows=%d "
    "num_seqs=%d window=%d predicted=%s traced=%s"
)
SUMMARY = (
    "q38 G1 D1 summary: rank=%d tp_rank=%d dp_rank=%d pid=%d buckets=%d engaged=%d "
    "not_engaged=%d mismatches=%d traced=%d"
)


def rank_fields(runner: Any) -> tuple[int, int, int, int]:
    """(rank, tp_rank, dp_rank, pid) of this worker process; -1 where unknown."""
    rank = tp_rank = dp_rank = -1
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
    except Exception:  # noqa: BLE001 - a log field
        pass
    try:
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

        tp_rank = int(get_tensor_model_parallel_rank())
    except Exception:  # noqa: BLE001 - a log field
        pass
    try:
        dp_rank = int(runner.parallel_config.data_parallel_rank)
    except Exception:  # noqa: BLE001 - a log field
        pass
    return rank, tp_rank, dp_rank, os.getpid()


def gdn_layers(runner: Any) -> list:
    """The pooled-capable GDN layers of the runner's model, in prefix order."""
    from vllm.config import get_layers_from_vllm_config
    from vllm.model_executor.layers.mamba.abstract import MambaBase

    from vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op import (
        VllmGatedDeltaNetAttention,
    )

    layers = get_layers_from_vllm_config(runner.vllm_config, MambaBase)
    return [
        layers[name]
        for name in sorted(layers)
        if isinstance(layers[name], VllmGatedDeltaNetAttention)
    ]


def geometry(layer: Any, mesh: Any, act_dtype: str) -> tuple:
    """(pooled, n_kq, n_v, d_k, d_v, kernel_size, dim, num_spec, act_dtype): what the
    layer's pooled op hands `fused_conv1d_gdn`. `_build_pooled_gdn_op` takes the
    TP-local heads, and `run_jax_gdn_attention_pooled` divides heads (and the shard_map
    divides the qkv columns) by the mesh's "model" axis, which is 1 for a per-worker
    mesh. `pooled`: the layer is bound to the unified block pool (one cache tensor,
    the forward's pooled branch); the split-state branch never runs the window log."""
    from vllm_torchtpu.utils import get_mesh_shape_product

    tp_mesh = get_mesh_shape_product(mesh, "model") if mesh is not None else 1
    geom = layer.tp_head_geometry
    kv_cache = getattr(layer, "kv_cache", None)
    return (
        kv_cache is not None and len(kv_cache) == 1,
        geom.local_num_kq_heads // tp_mesh,
        (layer.num_v_heads // layer.tp_size) // tp_mesh,
        layer.head_k_dim,
        layer.head_v_dim,
        layer.conv_kernel_size,
        geom.local_conv_dim(layer.head_k_dim, layer.head_v_dim) // tp_mesh,
        layer.num_spec,
        act_dtype,
    )


def _act_dtype(runner: Any) -> str:
    return str(runner.model_config.dtype).removeprefix("torch.")


def _qkv_2d(rows: int, num_seqs: int, num_spec: int, dim: int, act_dtype: str) -> bool:
    """`fused_conv1d_gdn`'s 2D bf16 rows: only with VLLM_TPU_GDN_PREFILL_BF16_QKV_2D,
    whose inputs are never pre-compacted (models/vllm/qwen4_exp/gdn.py), and only where
    `bf16_qkv_2d_eligible` holds for the raw rows."""
    if not envs.VLLM_TPU_GDN_PREFILL_BF16_QKV_2D:
        return False
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.kernels.gdn.v3 import g1_select, wrapper

    _, mixed_default = g1_select.wrapper_tile_defaults()
    return wrapper.bf16_qkv_2d_eligible(
        enabled=True,
        shape=(rows, dim),
        dtype=act_dtype,
        decode_capacity=num_seqs * (num_spec + 1),
        mixed_tile_size=min(mixed_default, rows),
        num_lanes=pltpu.get_tpu_info().num_lanes,
    )


def report_buckets(
    runner: Any,
    buckets: Iterable[tuple[int, int, str]],
    layers: Sequence | None = None,
) -> dict[str, int] | None:
    """Print the G1 D1 lines of every (rows, num_seqs, variant) dummy-run bucket for
    this rank. Silent, reading nothing but the key, unless VLLM_TPU_GDN_G1_BODY is set.
    `layers` defaults to the runner model's GDN layers. Returns the summary counts."""
    if not envs.VLLM_TPU_GDN_G1_BODY:
        return None
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.kernels.gdn.v3 import g1_select

    rank, tp_rank, dp_rank, pid = rank_fields(runner)
    head = (rank, tp_rank, dp_rank, pid)
    buckets = list(buckets)
    counts = dict(buckets=len(buckets), engaged=0, not_engaged=0, mismatches=0, traced=0)
    if layers is None:
        layers = gdn_layers(runner)
    act_dtype = _act_dtype(runner) if layers else "none"
    mesh = getattr(runner, "mesh", None)
    geoms = sorted({geometry(layer, mesh, act_dtype) for layer in layers})
    window_log = bool(envs.VLLM_TPU_GDN_WINDOW_LOG)
    factored = bool(envs.VLLM_TPU_GDN_WINDOW_FACTORED)
    num_lanes = pltpu.get_tpu_info().num_lanes if geoms else 0
    for rows, num_seqs, variant in buckets:
        if len(geoms) != 1:
            reason = "no_gdn_layer" if not geoms else "gdn_geometries_differ"
            logger.info(
                NOT_ENGAGED, *head, variant, rows, num_seqs, 0, 0, "none", "skipped", reason
            )
            counts["not_engaged"] += 1
            continue
        pooled, n_kq, n_v, d_k, d_v, kernel_size, dim, num_spec, act = geoms[0]
        if not pooled:
            logger.info(
                NOT_ENGAGED, *head, variant, rows, num_seqs, num_spec + 1, 0, "none",
                "skipped", "not_pooled",
            )
            counts["not_engaged"] += 1
            continue
        p = g1_select.predict(
            rows=rows,
            num_seqs=num_seqs,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            dim=dim,
            num_spec_tokens=num_spec,
            act_dtype=act,
            window_log=window_log,
            factored_window=factored,
            qkv_2d=_qkv_2d(rows, num_seqs, num_spec, dim, act),
            num_lanes=num_lanes,
        )
        rec = g1_select.traced(rows, num_seqs, p.window)
        if rec is not None:
            counts["traced"] += 1
            want = g1_select.as_record(p)
            if rec != want:
                logger.info(
                    MISMATCH, *head, variant, rows, num_seqs, p.window,
                    _fmt(want), _fmt(rec),
                )
                counts["mismatches"] += 1
                continue
        trace = "observed" if rec is not None else "skipped"
        if p.engaged:
            logger.info(
                ENGAGED, *head, variant, rows, num_seqs, p.window, p.seq_tile_size,
                p.kernel, trace,
            )
            counts["engaged"] += 1
        else:
            logger.info(
                NOT_ENGAGED, *head, variant, rows, num_seqs, p.window, p.seq_tile_size,
                p.kernel, trace, p.reason,
            )
            counts["not_engaged"] += 1
    logger.info(
        SUMMARY, *head, counts["buckets"], counts["engaged"], counts["not_engaged"],
        counts["mismatches"], counts["traced"],
    )
    return counts


def _fmt(rec: dict[str, Any]) -> str:
    return ",".join(f"{k}:{rec[k]}" for k in sorted(rec))
