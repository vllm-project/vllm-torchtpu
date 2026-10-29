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
"""Standalone TPU/CPU micro-benchmark and verifier for DeepSeek-V4.1 Indexer.

Decomposes the 8 Indexer layers (`L2, 8, 14, 20, 24, 28, 32, 36`:
3 full_context `L2, 8, 14` + 1 candidate_write `L20` + 4 candidate_filter
`L24, 28, 32, 36`) into atomic sub-operations:
  - `all`: Full 8-rank query-sharded `paged_indexer` (`write_index_keys` +
    `paged_indexer_jax` on `[T/8, ...]` + `all_gather` to `[T, topk]`).
  - `full_context`: `paged_indexer_jax(candidate_write=False, candidate_filter=False)`
    (scans all `kv_len // compress_ratio` compressed keys in 512-key tiles).
  - `candidate_write`: `paged_indexer_jax(candidate_write=True, candidate_filter=False)`
    (full context scan + publishes `candidate_topk_blocks` block IDs).
  - `candidate_filter`: `paged_indexer_jax` (`candidate_filter=True`,
    filters to `candidate_topk_blocks * candidate_block_size` candidate keys/token).
  - `topk_only`: `_visible_top_k` / `_compact_selected` on `float32[T_local, W]`
    scores (testing `+0.0` vs `-0.0` tie-breaking and `-inf` / `lengths` masks).
  - `cache_write_only`: `write_index_keys(pool, write_records, keys)` BF16 scatter.

Modes (`--mode`):
  - `baseline`: `unordered=False` (ordered top-k selection).
  - `precompact`: `unordered=True` with `_compact_selected` ascending prefix.
"""

import argparse
import copy
import os
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

SUBOPS = (
    "all",
    "full_context",
    "candidate_write",
    "candidate_filter",
    "topk_only",
    "cache_write_only",
)
MODES = ("baseline", "precompact")


def _build_indexer_inputs(
    rng_seed: int,
    tokens: int,
    tp_size: int,
    tp_rank: int,
    num_seqs: int,
    kv_len: int,
    compress_ratio: int,
    heads: int,
    head_dim: int,
    topk: int,
    candidate_blocks: int,
    block_size: int,
    *,
    q_len: int = 0,
    query_split: str = "pad16",
) -> dict[str, Any]:
    """Construct deterministic CPU inputs for all Indexer sub-operations."""
    import jax.numpy as jnp
    import torch

    from vllm_torchtpu.kernels.deepseek_v41 import index_key_layout

    rng = np.random.default_rng(rng_seed + 2027)
    width = max(topk, kv_len // compress_ratio)
    # Allocate a compact physical pool of 4096 BF16 rows and map `records`
    # (`[num_seqs, width]`) into it so we don't allocate 45k physical rows.
    phys_rows = 4096
    keys = jnp.asarray(rng.standard_normal((phys_rows, 128)), jnp.bfloat16)
    encoded = index_key_layout.encode_records(keys)
    pool_np = np.asarray(
        (encoded[::2] | (encoded[1::2] << 16)).astype(jnp.int32)
    ).copy()

    records_np = rng.integers(0, phys_rows, size=(num_seqs, width), dtype=np.int32)
    # Inject invalid (-1) tail records on the last 4 columns.
    records_np[:, max(0, width - 4) :] = -1

    # Query split across TP ranks:
    # - `pad16`: `rows = 16 * tp_size` (`v2/attention.py:777-789`).
    # - `ceil`: `part = ceil(tokens / tp_size)` (Decode/Verify split).
    if query_split == "ceil":
        part = max(1, (tokens + tp_size - 1) // tp_size)
        padding = part * tp_size - tokens
    else:
        rows_align = 16 * tp_size
        padding = (tokens // rows_align + 1) * rows_align - tokens
        part = (tokens + padding) // tp_size

    q_full = rng.standard_normal((tokens, heads, head_dim)).astype(np.float32) * 0.25
    weights_full = rng.standard_normal((tokens, heads)).astype(np.float32) * (
        (head_dim**-0.5) * (heads**-0.5)
    )

    # Sequence assignments and causal compressed lengths for each token.
    requests_full = np.zeros((tokens,), dtype=np.int32)
    lengths_full = np.zeros((tokens,), dtype=np.int32)
    base_ctx = max(topk + 64, width - tokens // compress_ratio - 32)
    offset = 0
    if q_len > 0:
        active_tokens = min(tokens, num_seqs * q_len)
        for s in range(num_seqs):
            seg = max(0, min(q_len, active_tokens - offset))
            requests_full[offset : offset + seg] = s
            pos = base_ctx + np.arange(seg, dtype=np.int32)
            lengths_full[offset : offset + seg] = np.clip(pos, 1, width)
            offset += seg
        lengths_full[active_tokens:] = 0
    else:
        active_tokens = tokens - 16 if num_seqs > 1 and tokens >= 64 else tokens
        base_per_seq = active_tokens // num_seqs
        for s in range(num_seqs):
            seg = base_per_seq if s < num_seqs - 1 else (active_tokens - offset)
            requests_full[offset : offset + seg] = s
            pos = base_ctx + np.arange(seg, dtype=np.int32)
            lengths_full[offset : offset + seg] = np.clip(pos, 1, width)
            offset += seg
        # Trailing padded tokens (`offset..tokens-1`) have `active_lengths = 0`.
        lengths_full[active_tokens:] = 0
        # Also test underfull row at token 0 (`length = 8 < topk`) when `num_seqs > 1`.
        if num_seqs > 1:
            lengths_full[0] = 8

    num_total_blocks = (width + block_size - 1) // block_size
    candidates_full = rng.integers(
        0, max(1, num_total_blocks), size=(tokens, candidate_blocks), dtype=np.int32
    )
    # Inject duplicate and -1 candidate blocks to test deduplication and masking.
    if candidate_blocks >= 4:
        candidates_full[:, -1] = -1
        candidates_full[:, -2] = candidates_full[:, 0]
    candidates_full[active_tokens:] = -1

    # Pad to `tokens + padding` and slice rank `tp_rank`'s `part` rows.
    def _pad_and_slice(arr: np.ndarray, fill: float | int) -> np.ndarray:
        pad_shape = (padding, *arr.shape[1:])
        padded = np.concatenate(
            [arr, np.full(pad_shape, fill, dtype=arr.dtype)], axis=0
        )
        start = tp_rank * part
        return np.ascontiguousarray(padded[start : start + part])

    q_part = _pad_and_slice(q_full, 0.0)
    weights_part = _pad_and_slice(weights_full, 0.0)
    lengths_part = _pad_and_slice(lengths_full, 0)
    candidates_part = _pad_and_slice(candidates_full, -1)
    requests_part = _pad_and_slice(requests_full, 0)

    # Write keys for `cache_write_only` and `all` (`compress_ratio=2`).
    # Valid destinations must be unique per `scatter_index_keys` contract.
    if phys_rows >= tokens:
        write_records_np = rng.permutation(phys_rows)[:tokens].astype(np.int32)
    else:
        write_records_np = np.arange(tokens, dtype=np.int32)
        write_records_np[phys_rows:] = -1
    write_records_np[::2] = -1  # Only odd positions write compressed KV/index keys
    write_records_np[active_tokens:] = -1
    write_keys_np = rng.standard_normal((tokens, head_dim)).astype(np.float32) * 0.25

    # Score matrix for `topk_only` with `+0.0` vs `-0.0` ties and `-inf` tail.
    scores_np = np.round(rng.standard_normal((part, width)), 1).astype(np.float32)
    signed_zeros = np.where(
        rng.random((part, width)) < 0.5, np.float32(-0.0), np.float32(0.0)
    )
    scores_np = np.where(np.abs(scores_np) < 0.25, signed_zeros, scores_np)
    col_idx = np.arange(width, dtype=np.int32)[None, :]
    scores_np = np.where(col_idx < lengths_part[:, None], scores_np, -np.inf)

    return {
        "width": width,
        "part": part,
        "padding": padding,
        "active_tokens": active_tokens,
        "batch_max": int(lengths_full.max()),
        "pool_t": torch.from_numpy(pool_np),
        "records_t": torch.from_numpy(records_np),
        "q_part_t": torch.from_numpy(q_part).to(torch.bfloat16),
        "weights_part_t": torch.from_numpy(weights_part),
        "lengths_part_t": torch.from_numpy(lengths_part),
        "candidates_part_t": torch.from_numpy(candidates_part),
        "requests_part_t": torch.from_numpy(requests_part),
        "write_records_t": torch.from_numpy(write_records_np),
        "write_keys_t": torch.from_numpy(write_keys_np).to(torch.bfloat16),
        "scores_t": torch.from_numpy(scores_np),
    }


def _compute_roofline(
    hw: common.Tpu7xRoofline,
    subop: str,
    mode: str,
    tokens: int,
    part: int,
    tp_size: int,
    width: int,
    heads: int,
    head_dim: int,
    topk: int,
    candidate_blocks: int,
    block_size: int,
    measured_ms: float | None,
    *,
    num_seqs: int = 1,
    q_len: int = 0,
) -> dict[str, Any]:
    """First-principles hardware roofline for the selected Indexer `subop`."""
    cand_width = min(width, candidate_blocks * block_size)
    reqs_per_rank = (
        min(num_seqs, max(1, (min(part, num_seqs * q_len) + q_len - 1) // q_len))
        if q_len > 0
        else 1
    )
    # Full-context (`requests` shared-tile loop) reads `width` BF16 keys once
    # per active request on this rank, plus writes/reads `[part, width]` FP32 scores.
    full_flops = 2.0 * part * heads * width * head_dim
    full_hbm_bytes = (
        reqs_per_rank * width * head_dim * 2.0
        + part * heads * head_dim * 2.0
        + 2.0 * part * width * 4.0
    )

    # Candidate-filter gathers `cand_width` (2048) BF16 keys per query token.
    cand_flops = 2.0 * part * heads * cand_width * head_dim
    cand_hbm_bytes = (
        part * cand_width * head_dim * 2.0
        + part * heads * head_dim * 2.0
        + 2.0 * part * cand_width * 4.0
    )

    topk_hbm_bytes = part * width * 4.0 + part * topk * 4.0
    write_hbm_bytes = tokens * head_dim * 2.0 * 2.0
    comm_s = 0.0

    if subop == "full_context":
        flops, hbm_bytes = full_flops, full_hbm_bytes + topk_hbm_bytes
    elif subop == "candidate_write":
        flops, hbm_bytes = (
            full_flops,
            full_hbm_bytes + topk_hbm_bytes + part * width * 4.0,
        )
    elif subop == "candidate_filter":
        flops, hbm_bytes = cand_flops, cand_hbm_bytes + part * cand_width * 4.0
    elif subop == "topk_only":
        flops, hbm_bytes = 0.0, topk_hbm_bytes
    elif subop == "cache_write_only":
        flops, hbm_bytes = 0.0, write_hbm_bytes
    else:  # subop == "all"
        flops = full_flops
        hbm_bytes = full_hbm_bytes + topk_hbm_bytes + write_hbm_bytes
        # All-Gather of `int32[part, topk]` across `tp_size` ranks.
        ag_bytes = tp_size * part * topk * 4
        comm_s = hw.all_gather_s(ag_bytes, tp_size)

    return common.RooflineEstimate(
        name=f"indexer[{subop},{mode}] part={part} W={width}",
        flops=flops,
        hbm_bytes=hbm_bytes,
        comm_s=comm_s,
        dtype="bf16",
    ).report(hw, measured_ms)


def worker(ctx: common.WorkerContext, args: argparse.Namespace) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import torch
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal import pallas
    from vllm.distributed import get_tp_group

    from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention import _gather_query_rows
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels import (
        _pad_compacted_records,
        compact_records,
    )
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.indexer_impl import (
        _compact_selected,
        _visible_top_k,
        initialize_indexer,
        paged_indexer,
        paged_indexer_jax,
        write_index_keys,
    )
    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        get_vllm_model_wrapper_context,
    )

    geo = common.Dsv41Geometry()
    tokens = args.chunk_size
    q_len = int(getattr(args, "q_len", 0))
    query_split = str(getattr(args, "query_split", "pad16"))
    tp_rank = get_tp_group().rank_in_group
    inp = _build_indexer_inputs(
        args.seed,
        tokens,
        ctx.tp_size,
        tp_rank,
        args.num_seqs,
        args.kv_len,
        args.compress_ratio,
        geo.index_n_heads,
        geo.index_head_dim,
        geo.index_topk,
        geo.candidate_topk_blocks,
        geo.candidate_block_size,
        q_len=q_len,
        query_split=query_split,
    )
    width = inp["width"]
    part = inp["part"]
    batch_max_val = inp["batch_max"]
    unordered = args.mode == "precompact"

    pool_init = inp["pool_t"].to("tpu")
    pool_work = inp["pool_t"].clone().to("tpu")
    records = inp["records_t"].to("tpu")
    q_part = inp["q_part_t"].to("tpu")
    weights_part = inp["weights_part_t"].to("tpu")
    lengths_part = inp["lengths_part_t"].to("tpu")
    candidates_part = inp["candidates_part_t"].to("tpu")
    requests_part = inp["requests_part_t"].to("tpu")
    write_records = inp["write_records_t"].to("tpu")
    write_keys = inp["write_keys_t"].to("tpu")
    scores_tpu = inp["scores_t"].to("tpu")
    batch_max_t = torch.tensor([batch_max_val], dtype=torch.int32, device="tpu")

    class _MockIndexCache:
        def __init__(self, raw):
            self.raw = raw

    cache_wrapper = _MockIndexCache(pool_work)

    with common.model_wrapper_context():
        mesh = get_vllm_model_wrapper_context().mesh
        initialize_indexer()

        def _make_indexer_subop(
            cand_write: bool, cand_filter: bool, is_unordered: bool
        ):
            def _fn(
                q_a: jax.Array,
                p_a: jax.Array,
                r_a: jax.Array,
                w_a: jax.Array,
                l_a: jax.Array,
                c_a: jax.Array,
                req_a: jax.Array,
                bmax_a: jax.Array,
            ) -> tuple[jax.Array, jax.Array]:
                return paged_indexer_jax(
                    q_a,
                    p_a,
                    r_a,
                    w_a,
                    l_a,
                    c_a,
                    req_a,
                    topk=geo.index_topk,
                    block_size=geo.candidate_block_size,
                    candidate_write=cand_write,
                    candidate_filter=cand_filter,
                    unordered=is_unordered,
                    batch_max=bmax_a,
                )

            return pallas.jax_op(
                f"pallas::dsv41_bench_idx_{int(cand_write)}_{int(cand_filter)}_{int(is_unordered)}",
                _fn,
                mesh=mesh,
                input_partition_specs=(P(),) * 8,
            )

        full_unord_op = _make_indexer_subop(False, False, True)
        full_ref_op = _make_indexer_subop(False, False, False)
        full_op = full_unord_op if unordered else full_ref_op

        cwrite_unord_op = _make_indexer_subop(True, False, True)
        cwrite_ref_op = _make_indexer_subop(True, False, False)
        cwrite_op = cwrite_unord_op if unordered else cwrite_ref_op

        cfilter_unord_op = _make_indexer_subop(False, True, True)
        cfilter_ref_op = _make_indexer_subop(False, True, False)
        cfilter_op = cfilter_unord_op if unordered else cfilter_ref_op

        def _make_topk_op(is_unord: bool):
            def _t_fn(s_a: jax.Array, l_a: jax.Array, bmax_a: jax.Array) -> jax.Array:
                idx = _visible_top_k(
                    s_a,
                    min(geo.index_topk, width),
                    bmax_a.max(),
                    l_a,
                    unordered=is_unord,
                )
                vals = jnp.take_along_axis(s_a, jnp.maximum(idx, 0), axis=1)
                valid = (idx >= 0) & (vals > -jnp.inf)
                sel = jnp.where(valid, idx, -1).astype(jnp.int32)
                return _compact_selected(sel) if is_unord else sel

            return pallas.jax_op(
                f"pallas::dsv41_bench_topk_{int(is_unord)}",
                _t_fn,
                mesh=mesh,
                input_partition_specs=(P(),) * 3,
            )

        topk_unord_op = _make_topk_op(True)
        topk_ref_op = _make_topk_op(False)
        topk_op = topk_unord_op if unordered else topk_ref_op

        def _write_fn(
            p_a: jax.Array, wrec_a: jax.Array, wkeys_a: jax.Array
        ) -> jax.Array:
            return write_index_keys(p_a, wrec_a, wkeys_a)

        write_op = pallas.jax_op(
            "pallas::dsv41_bench_write_index_keys",
            _write_fn,
            mesh=mesh,
            input_partition_specs=(P(),) * 3,
        )

        def _pad_only_fn(s_a: jax.Array) -> jax.Array:
            return _pad_compacted_records(s_a)

        pad_comp_op = pallas.jax_op(
            "pallas::dsv41_bench_pad_comp",
            _pad_only_fn,
            mesh=mesh,
            input_partition_specs=(P(),),
        )

        def _compact_ref_fn(s_a: jax.Array) -> jax.Array:
            return compact_records(s_a)

        compact_ref_op = pallas.jax_op(
            "pallas::dsv41_bench_compact_ref",
            _compact_ref_fn,
            mesh=mesh,
            input_partition_specs=(P(),),
        )
        normalize_sel_op = pad_comp_op if unordered else compact_ref_op

        if args.subop == "full_context":

            def step():
                sel, pub = full_op(
                    q_part,
                    pool_init,
                    records,
                    weights_part,
                    lengths_part,
                    candidates_part,
                    requests_part,
                    batch_max_t,
                )
                return {"selected": sel, "published": pub}

        elif args.subop == "candidate_write":

            def step():
                sel, pub = cwrite_op(
                    q_part,
                    pool_init,
                    records,
                    weights_part,
                    lengths_part,
                    candidates_part,
                    requests_part,
                    batch_max_t,
                )
                return {"selected": sel, "published": pub}

        elif args.subop == "candidate_filter":

            def step():
                sel, pub = cfilter_op(
                    q_part,
                    pool_init,
                    records,
                    weights_part,
                    lengths_part,
                    candidates_part,
                    requests_part,
                    batch_max_t,
                )
                return {"selected": sel, "published": pub}

        elif args.subop == "topk_only":

            def step():
                return {"selected": topk_op(scores_tpu, lengths_part, batch_max_t)}

        elif args.subop == "cache_write_only":

            def step():
                return {"updated_pool": write_op(pool_init, write_records, write_keys)}

        else:  # subop == "all"

            def step():
                updated, sel, pub = paged_indexer(
                    q_part,
                    cache_wrapper,
                    records,
                    weights_part,
                    lengths_part,
                    candidates_part,
                    write_records,
                    write_keys,
                    requests_part,
                    topk=geo.index_topk,
                    block_size=geo.candidate_block_size,
                    candidate_write=True,
                    candidate_filter=False,
                    unordered=unordered,
                    batch_max=batch_max_t,
                )
                # Materialize all 3 outputs `(updated, sel, pub)` together in a
                # single Execute() call before `_gather_query_rows` invokes
                # eager `c10d.all_gather_into_tensor` on `sel` alone (which
                # would otherwise materialize only `sel`, donate `cache.raw`,
                # and then fail when materializing `pub`/`updated`).
                common.synchronize((updated, sel, pub))
                cache_wrapper.raw = updated
                sel_full = _gather_query_rows(sel)[:tokens]
                pub_full = _gather_query_rows(pub)[:tokens]
                return {
                    "selected": sel_full,
                    "published": pub_full,
                }

        common.synchronize(
            (
                pool_init,
                cache_wrapper.raw,
                records,
                q_part,
                weights_part,
                lengths_part,
                candidates_part,
                requests_part,
                write_records,
                write_keys,
                scores_tpu,
            )
        )
        hbm_before = common.hbm_stats()
        samples, out = common.time_steps(
            ctx, step, warmup=args.warmup, iters=args.iters
        )
        profile_dir = common.profile_steps(
            ctx, args, step, label=f"indexer_{args.subop}_{args.mode}"
        )
        hbm_after = common.hbm_stats()

        # ------------------------------------------------------------------
        # Mandatory Correctness Verification (`--verify`)
        # ------------------------------------------------------------------
        verify_report: dict[str, Any] = {}
        subop_suite: dict[str, Any] = {}
        hw = common.Tpu7xRoofline()
        if args.verify:
            if args.subop == "cache_write_only":
                np_pool_after = write_index_keys(
                    jnp.asarray(inp["pool_t"].numpy()),
                    jnp.asarray(inp["write_records_t"].numpy()),
                    jnp.asarray(inp["write_keys_t"].float().numpy(), jnp.bfloat16),
                )
                exp_t = torch.from_numpy(np.asarray(np_pool_after).copy())
                verify_report["vs_cpu_write_index_keys"] = common.verify_outputs(
                    {"updated_pool": out["updated_pool"].cpu()},
                    {"updated_pool": exp_t},
                    common.EXACT,
                    label="cache_write_only vs CPU write_index_keys",
                )
                criteria = common.EXACT

            elif args.subop == "topk_only":
                s_jnp = jnp.asarray(inp["scores_t"].numpy())
                l_jnp = jnp.asarray(inp["lengths_part_t"].numpy())
                _, ref_idx = jax.lax.top_k(s_jnp, min(geo.index_topk, width))
                ref_idx = jnp.sort(ref_idx, axis=-1)
                ref_vals = jnp.take_along_axis(s_jnp, ref_idx, axis=1)
                ref_valid = (
                    (ref_idx >= 0) & (ref_vals > -jnp.inf) & (l_jnp[:, None] > 0)
                )
                ref_sel = jnp.where(ref_valid, ref_idx, -1).astype(jnp.int32)
                exp_comp = compact_records(ref_sel)
                got_comp = normalize_sel_op(out["selected"]).cpu()
                exp_comp_t = torch.from_numpy(np.asarray(exp_comp).copy())
                verify_report["vs_jax_top_k_compacted"] = common.verify_outputs(
                    {"compacted": got_comp},
                    {"compacted": exp_comp_t},
                    common.EXACT,
                    label="topk_only vs jax.lax.top_k (+0.0/-0.0 & -inf mask)",
                )
                criteria = common.EXACT

            elif args.subop in ("full_context", "candidate_write", "candidate_filter"):
                ref_fn = {
                    "full_context": full_ref_op,
                    "candidate_write": cwrite_ref_op,
                    "candidate_filter": cfilter_ref_op,
                }[args.subop]
                ref_sel, ref_pub = ref_fn(
                    q_part,
                    pool_init,
                    records,
                    weights_part,
                    lengths_part,
                    candidates_part,
                    requests_part,
                    batch_max_t,
                )
                got_norm = normalize_sel_op(out["selected"])
                ref_norm = compact_ref_op(ref_sel)
                common.synchronize((got_norm, ref_norm, ref_pub))
                verify_report["selected_compacted_vs_ordered_ref"] = (
                    common.verify_outputs(
                        {
                            "selected": got_norm.cpu(),
                            "published": out["published"].cpu(),
                        },
                        {"selected": ref_norm.cpu(), "published": ref_pub.cpu()},
                        common.EXACT,
                        label=f"{args.subop}[{args.mode}] vs ordered reference",
                    )
                )
                criteria = common.EXACT

            else:  # subop == "all"

                def _fresh_cache() -> _MockIndexCache:
                    buf = inp["pool_t"].clone().to("tpu")
                    common.synchronize(buf)
                    return _MockIndexCache(buf)

                ref_cache = _fresh_cache()
                ref_upd, ref_sel_p, ref_pub_p = paged_indexer(
                    q_part,
                    ref_cache,
                    records,
                    weights_part,
                    lengths_part,
                    candidates_part,
                    write_records,
                    write_keys,
                    requests_part,
                    topk=geo.index_topk,
                    block_size=geo.candidate_block_size,
                    candidate_write=True,
                    candidate_filter=False,
                    unordered=False,
                    batch_max=batch_max_t,
                )
                common.synchronize((ref_upd, ref_sel_p, ref_pub_p))
                ref_sel_full = _gather_query_rows(ref_sel_p)[:tokens]
                ref_pub_full = _gather_query_rows(ref_pub_p)[:tokens]
                got_norm = normalize_sel_op(out["selected"])
                ref_norm = compact_ref_op(ref_sel_full)
                common.synchronize((ref_upd, got_norm, ref_norm, ref_pub_full))
                verify_report["all_vs_ordered_ref"] = common.verify_outputs(
                    {
                        "updated_pool": cache_wrapper.raw.cpu(),
                        "selected": got_norm.cpu(),
                        "published": out["published"].cpu(),
                    },
                    {
                        "updated_pool": ref_upd.cpu(),
                        "selected": ref_norm.cpu(),
                        "published": ref_pub_full.cpu(),
                    },
                    common.EXACT,
                    label=f"all[{args.mode}] vs ordered reference",
                )

                # Also verify `num_seqs=4` (including underfull row 0 and padded tail)
                inp4 = _build_indexer_inputs(
                    args.seed,
                    tokens,
                    ctx.tp_size,
                    tp_rank,
                    4,
                    args.kv_len,
                    args.compress_ratio,
                    geo.index_n_heads,
                    geo.index_head_dim,
                    geo.index_topk,
                    geo.candidate_topk_blocks,
                    geo.candidate_block_size,
                )
                rec4 = inp4["records_t"].to("tpu")
                qp4 = inp4["q_part_t"].to("tpu")
                wp4 = inp4["weights_part_t"].to("tpu")
                lp4 = inp4["lengths_part_t"].to("tpu")
                cp4 = inp4["candidates_part_t"].to("tpu")
                reqp4 = inp4["requests_part_t"].to("tpu")
                wrec4 = inp4["write_records_t"].to("tpu")
                wkeys4 = inp4["write_keys_t"].to("tpu")
                bmax4 = torch.tensor(
                    [inp4["batch_max"]], dtype=torch.int32, device="tpu"
                )
                common.synchronize(
                    (rec4, qp4, wp4, lp4, cp4, reqp4, wrec4, wkeys4, bmax4)
                )
                cache4_u = _fresh_cache()
                upd4_u, sel4_u, pub4_u = paged_indexer(
                    qp4,
                    cache4_u,
                    rec4,
                    wp4,
                    lp4,
                    cp4,
                    wrec4,
                    wkeys4,
                    reqp4,
                    topk=geo.index_topk,
                    block_size=geo.candidate_block_size,
                    candidate_write=True,
                    candidate_filter=False,
                    unordered=True,
                    batch_max=bmax4,
                )
                common.synchronize((upd4_u, sel4_u, pub4_u))
                sel4_u_full = _gather_query_rows(sel4_u)[:tokens]
                pub4_u_full = _gather_query_rows(pub4_u)[:tokens]
                cache4_r = _fresh_cache()
                upd4_r, sel4_r, pub4_r = paged_indexer(
                    qp4,
                    cache4_r,
                    rec4,
                    wp4,
                    lp4,
                    cp4,
                    wrec4,
                    wkeys4,
                    reqp4,
                    topk=geo.index_topk,
                    block_size=geo.candidate_block_size,
                    candidate_write=True,
                    candidate_filter=False,
                    unordered=False,
                    batch_max=bmax4,
                )
                common.synchronize((upd4_r, sel4_r, pub4_r))
                sel4_r_full = _gather_query_rows(sel4_r)[:tokens]
                pub4_r_full = _gather_query_rows(pub4_r)[:tokens]
                got4_norm = pad_comp_op(sel4_u_full)
                ref4_norm = compact_ref_op(sel4_r_full)
                common.synchronize((upd4_u, upd4_r, got4_norm, ref4_norm))
                verify_report["all_num_seqs_4_precompact_vs_baseline"] = (
                    common.verify_outputs(
                        {
                            "updated_pool": upd4_u.cpu(),
                            "selected": got4_norm.cpu(),
                            "published": pub4_u_full.cpu(),
                        },
                        {
                            "updated_pool": upd4_r.cpu(),
                            "selected": ref4_norm.cpu(),
                            "published": pub4_r_full.cpu(),
                        },
                        common.EXACT,
                        label="all[num_seqs=4,precompact] vs baseline",
                    )
                )

                # Atomic subop suite breakdown & verification in a single 8-rank run
                s_jnp = jnp.asarray(inp["scores_t"].numpy())
                l_jnp = jnp.asarray(inp["lengths_part_t"].numpy())
                _, ref_idx = jax.lax.top_k(s_jnp, min(geo.index_topk, width))
                ref_idx = jnp.sort(ref_idx, axis=-1)
                ref_vals = jnp.take_along_axis(s_jnp, ref_idx, axis=1)
                ref_valid = (
                    (ref_idx >= 0) & (ref_vals > -jnp.inf) & (l_jnp[:, None] > 0)
                )
                ref_sel = jnp.where(ref_valid, ref_idx, -1).astype(jnp.int32)
                exp_topk_t = torch.from_numpy(
                    np.asarray(compact_records(ref_sel)).copy()
                )
                np_pool_after = write_index_keys(
                    jnp.asarray(inp["pool_t"].numpy()),
                    jnp.asarray(inp["write_records_t"].numpy()),
                    jnp.asarray(inp["write_keys_t"].float().numpy(), jnp.bfloat16),
                )
                exp_pool_t = torch.from_numpy(np.asarray(np_pool_after).copy())

                suite_defs = [
                    (
                        "full_context",
                        "precompact",
                        lambda: full_unord_op(
                            q_part,
                            pool_init,
                            records,
                            weights_part,
                            lengths_part,
                            candidates_part,
                            requests_part,
                            batch_max_t,
                        ),
                        full_ref_op,
                    ),
                    (
                        "full_context",
                        "baseline",
                        lambda: full_ref_op(
                            q_part,
                            pool_init,
                            records,
                            weights_part,
                            lengths_part,
                            candidates_part,
                            requests_part,
                            batch_max_t,
                        ),
                        full_ref_op,
                    ),
                    (
                        "candidate_write",
                        "precompact",
                        lambda: cwrite_unord_op(
                            q_part,
                            pool_init,
                            records,
                            weights_part,
                            lengths_part,
                            candidates_part,
                            requests_part,
                            batch_max_t,
                        ),
                        cwrite_ref_op,
                    ),
                    (
                        "candidate_write",
                        "baseline",
                        lambda: cwrite_ref_op(
                            q_part,
                            pool_init,
                            records,
                            weights_part,
                            lengths_part,
                            candidates_part,
                            requests_part,
                            batch_max_t,
                        ),
                        cwrite_ref_op,
                    ),
                    (
                        "candidate_filter",
                        "precompact",
                        lambda: cfilter_unord_op(
                            q_part,
                            pool_init,
                            records,
                            weights_part,
                            lengths_part,
                            candidates_part,
                            requests_part,
                            batch_max_t,
                        ),
                        cfilter_ref_op,
                    ),
                    (
                        "candidate_filter",
                        "baseline",
                        lambda: cfilter_ref_op(
                            q_part,
                            pool_init,
                            records,
                            weights_part,
                            lengths_part,
                            candidates_part,
                            requests_part,
                            batch_max_t,
                        ),
                        cfilter_ref_op,
                    ),
                    (
                        "topk_only",
                        "precompact",
                        lambda: topk_unord_op(scores_tpu, lengths_part, batch_max_t),
                        None,
                    ),
                    (
                        "topk_only",
                        "baseline",
                        lambda: topk_ref_op(scores_tpu, lengths_part, batch_max_t),
                        None,
                    ),
                    (
                        "cache_write_only",
                        "precompact",
                        lambda: write_op(pool_init, write_records, write_keys),
                        None,
                    ),
                ]
                orig_profile = args.profile
                orig_sess = os.environ.get(common._PROFILE_SESSION_ENV)
                try:
                    for s_name, m_name, s_fn, r_op in suite_defs:
                        key = f"{s_name}_{m_name}"
                        s_samples, s_out = common.time_steps(
                            ctx, s_fn, warmup=2, iters=5
                        )
                        s_xprof = None
                        if orig_profile is not None:
                            suite_args = copy.copy(args)
                            suite_args.profile = orig_profile.parent / f"indexer_{key}"
                            os.environ[common._PROFILE_SESSION_ENV] = (
                                f"{orig_sess}_{key}"
                            )
                            s_pdir = common.profile_steps(
                                ctx, suite_args, s_fn, label=f"indexer_{key}"
                            )
                            if s_pdir and ctx.rank == 0:
                                s_xprof = common.summarize_xplanes(Path(s_pdir))
                        check_kind = (
                            "determinism self-check"
                            if m_name == "baseline"
                            else "vs ordered baseline"
                        )
                        if r_op is not None:
                            r_sel, r_pub = r_op(
                                q_part,
                                pool_init,
                                records,
                                weights_part,
                                lengths_part,
                                candidates_part,
                                requests_part,
                                batch_max_t,
                            )
                            g_n = (
                                pad_comp_op(s_out[0])
                                if m_name == "precompact"
                                else compact_ref_op(s_out[0])
                            )
                            r_n = compact_ref_op(r_sel)
                            common.synchronize((g_n, r_n, r_pub))
                            v_res = common.verify_outputs(
                                {"selected": g_n.cpu(), "published": s_out[1].cpu()},
                                {"selected": r_n.cpu(), "published": r_pub.cpu()},
                                common.EXACT,
                                label=f"suite:{key} ({check_kind})",
                            )
                        elif s_name == "topk_only":
                            g_n = (
                                pad_comp_op(s_out)
                                if m_name == "precompact"
                                else compact_ref_op(s_out)
                            )
                            common.synchronize((g_n,))
                            v_res = common.verify_outputs(
                                {"compacted": g_n.cpu()},
                                {"compacted": exp_topk_t},
                                common.EXACT,
                                label=f"suite:{key} (vs jax.lax.top_k)",
                            )
                        else:
                            common.synchronize((s_out,))
                            v_res = common.verify_outputs(
                                {"updated_pool": s_out.cpu()},
                                {"updated_pool": exp_pool_t},
                                common.EXACT,
                                label=f"suite:{key} (vs CPU write_index_keys)",
                            )
                        subop_suite[key] = {
                            "wall_median_ms": statistics.median(s_samples),
                            "wall_min_ms": min(s_samples),
                            "roofline": _compute_roofline(
                                hw,
                                s_name,
                                m_name,
                                tokens,
                                part,
                                ctx.tp_size,
                                width,
                                geo.index_n_heads,
                                geo.index_head_dim,
                                geo.index_topk,
                                geo.candidate_topk_blocks,
                                geo.candidate_block_size,
                                statistics.median(s_samples),
                                num_seqs=args.num_seqs,
                                q_len=q_len,
                            ),
                            "verify": v_res,
                            "xprof": s_xprof,
                        }
                finally:
                    if orig_sess is not None:
                        os.environ[common._PROFILE_SESSION_ENV] = orig_sess
                criteria = common.EXACT
        else:
            criteria = common.EXACT

        case_suffix = (
            f"_Q{q_len}_{query_split}" if (q_len > 0 or query_split != "pad16") else ""
        )
        case_name = (
            f"indexer_{args.subop}_{args.mode}_T{tokens}_S{args.num_seqs}"
            f"_KV{args.kv_len}_CR{args.compress_ratio}{case_suffix}"
        )
        g_path = common.golden_path(args.golden_dir, case_name, ctx.rank)
        if args.update_golden or g_path.exists():
            golden = common.check_against_golden(ctx, args, case_name, out, criteria)
        else:
            golden = {
                "golden": "in_process_reference_verified_only",
                "expected_golden_path": str(g_path),
            }

    roof = _compute_roofline(
        hw,
        args.subop,
        args.mode,
        tokens,
        part,
        ctx.tp_size,
        width,
        geo.index_n_heads,
        geo.index_head_dim,
        geo.index_topk,
        geo.candidate_topk_blocks,
        geo.candidate_block_size,
        statistics.median(samples),
        num_seqs=args.num_seqs,
        q_len=q_len,
    )
    return {
        "subop": args.subop,
        "mode": args.mode,
        "chunk_size": tokens,
        "part_rows": part,
        "width": width,
        "num_seqs": args.num_seqs,
        "q_len": q_len,
        "query_split": query_split,
        "kv_len": args.kv_len,
        "compress_ratio": args.compress_ratio,
        "wall_samples_ms": samples,
        "verify": verify_report,
        "subop_suite": subop_suite,
        "golden": golden,
        "roofline_vs_wall": roof,
        "hbm": {"before": hbm_before, "after": hbm_after},
        "profile_dir": profile_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(parser, SUBOPS)
    parser.add_argument("--mode", choices=list(MODES), default="precompact")
    parser.add_argument("--compress-ratio", type=int, default=2, choices=[1, 2])
    parser.add_argument(
        "--q-len",
        type=int,
        default=0,
        help="Per-request query length (0=prefill, 1=Decode, 6=DSpark Verify).",
    )
    parser.add_argument(
        "--query-split",
        choices=["pad16", "ceil"],
        default="pad16",
        help=(
            "Query split across TP ranks ('pad16' for 16*tp_size, "
            "'ceil' for decode/verify)."
        ),
    )
    common.run_main("indexer", parser, worker)


if __name__ == "__main__":
    main()
