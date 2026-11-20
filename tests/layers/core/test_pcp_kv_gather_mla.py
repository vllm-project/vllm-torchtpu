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
"""PCP manager sparse MLA (`USE_TPU_PCP_MANAGER=1`) vs a numpy reference.

One layer on 8 chips, DCP 8 x PCP {2, 4, 8}:
`HeadTailPcpShardingPlan.prepare_real` plans every PCP rank's step,
`sparse_mla_pcp_write_split` writes the split rows into
the DCP-sharded cache and `sparse_mla_attention_pcp_prefill` attends each
local query's top-k (global positions) through the gathered workspace. The
reference is f32 MLA over the same fp8-rounded latents, request by request,
in global position order, so any slip in the partition, the restore index,
the DCP ownership, the workspace pages or the top-k remap shows up as a
wrong row. Steps run back to back, so later steps read earlier steps' KV
as a cached prefix.

The indexer half runs `mla_attention_op.pcp_prefill_topk` (write the split
rows' K, gather the K workspace, top-k each local query over it) against
`sum_h w_h * relu(q_h . k)` in numpy, accepting any top-k that differs from
the exact one only by near-ties at the cut (the kernel's top-k is
approximate), and against the unsharded `streamindex_topk` over each
request's K in position order, which it must match exactly.
"""

import functools
import math
import zlib
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    streamindex_topk as plain_streamindex_topk,
)
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk import (
    convert_cache_to_seq_along_lane,
)
from vllm_torchtpu.kernels.mla.kv_cache_utils import (
    KVCacheLayout,
    KVCacheType,
    SparseMLAKVCacheSpec,
)
from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla_kernel
from vllm_torchtpu.layers.adapter.pcp_kv_gather import pcp_chunk_pages
from vllm_torchtpu.layers.core import attention_interface
from vllm_torchtpu.runner.pcp.head_tail_sharding_plan import HeadTailPcpShardingPlan

pytestmark = [pytest.mark.tpu_test, pytest.mark.multichip]

DCP = 8
LKV_DIM = 512
ROPE_DIM = 64
HEADS_PER_RANK = 16
PAGE_SIZE = 64
INTERLEAVE = 4
NUM_PAGES = 128  # per DCP rank
MAX_REQS = 8
# The runner's request bucket: two local requests per scheduled one.
LOCAL_REQS = MAX_REQS * 2
BLOCKS_PER_REQ = 12  # virtual pages: up to 12 * 64 * 8 tokens per request
TOPK = 128
K_SCALE = 1.5
SM_SCALE = 1.0 / math.sqrt(LKV_DIM + ROPE_DIM)
TOKEN_PADDINGS = [16, 32, 64, 128, 256, 512, 1024, 2048]
CHUNK_TOKENS = 1024  # 2 workspace pages per gather chunk: exercises the loop
WORKSPACE_TOKENS = 16384


def _mesh():
    devices = jax.devices()
    if devices[0].platform != "tpu" or len(devices) < DCP:
        pytest.skip(f"needs {DCP} TPU devices")
    return jax.sharding.Mesh(np.array(devices[:DCP]), (streamindex_topk.DCP_AXIS_NAME,))


def _specs(layout, packed_rope=False):
    """(nope, rope) specs; `packed_rope` (TPU_SPARSE_MLA_PACKED_ROPE) stores
    the rope cache as 64 bytes per token, eight tokens per row."""
    packing = sparse_mla_kernel.get_dtype_packing(jnp.float8_e4m3fn)
    return tuple(
        SparseMLAKVCacheSpec.create(
            kind,
            layout,
            NUM_PAGES,
            PAGE_SIZE,
            dim,
            packing,
            packed_rope=packed_rope and kind is KVCacheType.ROPE,
        )
        for kind, dim in ((KVCacheType.NOPE, LKV_DIM), (KVCacheType.ROPE, ROPE_DIM))
    )


def _fake_runner(num_computed, num_prompt, q_lens, seq_lens):
    cap = 4 * sum(q_lens) + 4096
    positions = torch.zeros(cap, dtype=torch.int64)
    qsl = np.zeros(LOCAL_REQS + 1, dtype=np.int32)
    qsl[1 : len(q_lens) + 1] = np.cumsum(q_lens)
    qsl[len(q_lens) + 1 :] = qsl[len(q_lens)]
    lens = np.zeros(LOCAL_REQS, dtype=np.int32)
    lens[: len(seq_lens)] = seq_lens
    return SimpleNamespace(
        device=torch.device("cpu"),
        input_batch=SimpleNamespace(
            num_computed_tokens_cpu=np.asarray(num_computed, dtype=np.int32),
            num_prompt_tokens=np.asarray(num_prompt, dtype=np.int32),
        ),
        num_tokens_paddings=TOKEN_PADDINGS,
        input_ids_cpu=torch.zeros(cap, dtype=torch.int32),
        positions_cpu=positions,
        positions_np=positions.numpy(),
        arange_np=np.arange(cap, dtype=np.int64),
        query_start_loc_np=qsl,
        seq_lens_np=lens,
    )


def _plan_step(pcp, num_computed, num_prompt, q_lens, page_size=PAGE_SIZE):
    """Every PCP rank's plan, and its local qsl / seq_lens views."""
    seq_lens = [c + q for c, q in zip(num_computed, q_lens)]
    plans = []
    for rank in range(pcp):
        manager = HeadTailPcpShardingPlan(
            pcp_size=pcp,
            pcp_rank=rank,
            dcp_size=DCP,
            page_size=page_size,
            workspace_tokens=WORKSPACE_TOKENS,
            max_num_batched_tokens=TOKEN_PADDINGS[-1],
            max_num_seqs=MAX_REQS,
        )
        assert MAX_REQS * manager.local_reqs_per_req == LOCAL_REQS
        runner = _fake_runner(num_computed, num_prompt, q_lens, seq_lens)
        manager.prepare_real(
            runner=runner,
            scheduler_output=None,
            start_index=0,
            num_reqs=len(q_lens),
            num_scheduled_tokens_per_req=q_lens,
            total_num_scheduled_tokens=sum(q_lens),
            use_max_model_len=False,
            target_num_reqs=LOCAL_REQS,
            padded_num_reqs=MAX_REQS,
        )
        plans.append(
            SimpleNamespace(
                plan=manager,
                qsl=runner.query_start_loc_np.copy(),
                seq_lens=runner.seq_lens_np.copy(),
                num_ws_pages=manager.num_ws_pages,
            )
        )
    return plans


def _by_dcp_rank(pcp, per_pcp_rank):
    """Stack one array per PCP rank into the DCP-sharded global array (DCP
    rank d is PCP rank d // tp; its TP lanes hold the same rows)."""
    tp = DCP // pcp
    return np.concatenate([per_pcp_rank[d // tp] for d in range(DCP)])


def _reference(q, keys_nope, keys_rope, sel):
    """f32 MLA over the selected positions: q [H, 576] -> [H, 512]."""
    keys = np.concatenate([keys_nope[sel], keys_rope[sel]], -1)
    scores = q @ keys.T * SM_SCALE
    scores -= scores.max(-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(-1, keepdims=True)
    return probs @ keys_nope[sel]


def _topk_row(rng, pos):
    row = np.full(TOPK, -1, np.int32)
    if pos + 1 <= TOPK:
        row[: pos + 1] = np.arange(pos + 1)
    else:
        # Always the diagonal and the first token, the rest spread over the
        # whole context (every DCP rank and workspace page).
        rest = rng.choice(np.arange(1, pos), TOPK - 2, replace=False)
        row[:] = np.concatenate([[0, pos], rest])
    return row


# Each scenario is a list of steps; a step is (num_computed, q_lens) per
# request, prompt length = the request's final length. q_len 1 at
# num_computed == prompt - 1 would be a prefill; decodes are marked by
# num_computed == prompt (they only pad the split metadata here).
SCENARIOS = {
    # One long prefill, then a second chunk reading it as a cached prefix.
    "single_two_chunks": [[(0, 1500)], [(1500, 1000)]],
    # Unaligned lengths, several requests sharing the workspace, a request
    # that starts in the second step.
    "multi_req": [
        [(0, 700), (0, 333), (0, 990)],
        [(700, 900), (333, 41), (990, 517), (0, 260)],
    ],
    # A decode in the batch: replicated rows after the split segments.
    "with_decode": [[(0, 900), (0, 50)], [(900, 611), (50, 1)]],
    # A full batch: 8 prefills (16 local requests per rank), then 6 prefills
    # and 2 decodes (14); more than `MAX_REQS` local requests.
    "many_prefills": [
        [(0, 200), (0, 150), (0, 333), (0, 90), (0, 410), (0, 260), (0, 20), (0, 30)],
        [
            (200, 300),
            (150, 280),
            (333, 200),
            (90, 120),
            (410, 95),
            (260, 170),
            (20, 1),
            (30, 1),
        ],
    ],
}


def _final_lens(steps):
    final_len = [0] * max(len(s) for s in steps)
    for step in steps:
        for r, (c, q) in enumerate(step):
            final_len[r] = max(final_len[r], c + q)
    return final_len


def _plan_steps(pcp, steps, block_table, page_size=PAGE_SIZE):
    """Per step: every PCP rank's plan and the (request, position) of every
    packed row (None for padding)."""
    final_len = _final_lens(steps)
    for index, step in enumerate(steps):
        num_computed = [c for c, _ in step]
        q_lens = [q for _, q in step]
        is_decode = [q == 1 and c == final_len[r] - 1 for r, (c, q) in enumerate(step)]
        # A decode is `num_computed == num_prompt`; prefills see the final
        # length as their prompt.
        num_prompt = [
            c if d else final_len[r]
            for r, (c, d) in enumerate(zip(num_computed, is_decode))
        ]
        ranks = _plan_step(pcp, num_computed, num_prompt, q_lens, page_size)
        plan0 = ranks[0].plan
        global_qsl = np.concatenate([[0], np.cumsum(q_lens)])
        req_of = np.searchsorted(global_qsl, np.arange(sum(q_lens)), side="right") - 1
        # Request-major token of every gathered row (rank-major), -1 for
        # padding.
        local_padded = plan0.local_padded_num_tokens
        packed_to_request = np.full(pcp * local_padded, -1, dtype=np.int64)
        for p, rank in enumerate(ranks):
            local_rows = rank.plan.request_major_to_local
            g = np.flatnonzero(local_rows >= 0)
            packed_to_request[p * local_padded + local_rows[g]] = g
        row_tokens = [
            None
            if g < 0
            else (
                int(req_of[g]),
                num_computed[req_of[g]] + g - int(global_qsl[req_of[g]]),
            )
            for g in packed_to_request.tolist()
        ]
        split_tables = np.zeros((LOCAL_REQS, BLOCKS_PER_REQ), np.int32)
        split_tables[: len(plan0.split_block_rows)] = block_table[
            plan0.split_block_rows
        ]
        yield SimpleNamespace(
            index=index,
            is_decode=is_decode,
            ranks=ranks,
            plan0=plan0,
            local_padded=plan0.local_padded_num_tokens,
            row_tokens=row_tokens,
            split_tables=split_tables,
        )


def _rep(x):
    return jnp.asarray(np.asarray(x, dtype=np.int32))


def _shard(pcp, per_rank):
    return jnp.asarray(_by_dcp_rank(pcp, per_rank).astype(np.int32))


def _block_table(rng):
    """Distinct virtual pages per request (resolved `% NUM_PAGES` per rank)."""
    table = rng.permutation(NUM_PAGES)[: MAX_REQS * BLOCKS_PER_REQ]
    return table.reshape(MAX_REQS, BLOCKS_PER_REQ).astype(np.int32)


def _fp8(rng, shape, scale=1.0):
    return np.asarray(
        jnp.asarray(rng.standard_normal(shape) / scale).astype(jnp.float8_e4m3fn)
    )


@pytest.mark.parametrize("layout", ["sparsecore", "tensorcore"])
@pytest.mark.parametrize("pcp", [2, 4, 8])
@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_pcp_prefill_matches_numpy(monkeypatch, scenario, pcp, layout):
    _pcp_prefill_outputs(monkeypatch, scenario, pcp, layout)


@pytest.mark.parametrize("pcp", [2, 4, 8])
@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_pcp_prefill_packed_rope_matches_unpacked(monkeypatch, scenario, pcp):
    """TPU_PCP_PACKED_ROPE: the PCP write + workspace gather + attention over a
    packed rope cache (8 tokens per row) matches the numpy reference and the
    padded rope cache (4 tokens per row) on every split prefill row."""
    padded = _pcp_prefill_outputs(monkeypatch, scenario, pcp, "sparsecore")
    packed = _pcp_prefill_outputs(
        monkeypatch, scenario, pcp, "sparsecore", packed_rope=True
    )
    assert len(packed) == len(padded)
    for step, ((out_p, mask_p), (out_u, mask_u)) in enumerate(zip(packed, padded)):
        np.testing.assert_array_equal(mask_p, mask_u)
        assert mask_p.any()
        np.testing.assert_allclose(
            out_p[mask_p],
            out_u[mask_u],
            rtol=1e-3,
            atol=1e-3,
            err_msg=f"step {step}: packed rope differs from padded rope",
        )


# (ws_fill, hoist) variants of the split prefill: TPU_PCP_WS_NO_ZERO_FILL
# ("empty"), the test-only NaN workspace that no attended row may come from
# ("nan"), and TPU_PCP_GATHER_HOIST (first chunk gathered by the write op).
WORKSPACE_VARIANTS = [
    ("empty", False),
    ("nan", False),
    ("zeros", True),
    ("empty", True),
    ("nan", True),
]
_DEFAULT_OUTPUTS: dict = {}


def _default_outputs(monkeypatch, scenario, pcp, layout, packed_rope=False):
    """`_pcp_prefill_outputs` of the default path, computed once per input."""
    key = (scenario, pcp, layout, packed_rope)
    if key not in _DEFAULT_OUTPUTS:
        _DEFAULT_OUTPUTS[key] = _pcp_prefill_outputs(
            monkeypatch, scenario, pcp, layout, packed_rope=packed_rope
        )
    return _DEFAULT_OUTPUTS[key]


@pytest.mark.parametrize("ws_fill,hoist", WORKSPACE_VARIANTS)
@pytest.mark.parametrize("layout", ["sparsecore", "tensorcore"])
@pytest.mark.parametrize("pcp", [2, 8])
@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_pcp_prefill_workspace_variants_match_default(
    monkeypatch, scenario, pcp, layout, ws_fill, hoist
):
    """TPU_PCP_WS_NO_ZERO_FILL / TPU_PCP_GATHER_HOIST: every output row is bit
    for bit the default path's, also when the workspace starts as fp8 NaN
    (so unwritten rows reach no output), and the split rows match numpy."""
    base = _default_outputs(monkeypatch, scenario, pcp, layout)
    got = _pcp_prefill_outputs(
        monkeypatch, scenario, pcp, layout, ws_fill=ws_fill, hoist=hoist
    )
    assert len(got) == len(base)
    for step, ((out_g, mask_g), (out_b, mask_b)) in enumerate(zip(got, base)):
        np.testing.assert_array_equal(mask_g, mask_b)
        assert np.isfinite(out_g[mask_g]).all(), f"step {step}: non-finite row"
        np.testing.assert_array_equal(
            out_g[mask_g], out_b[mask_b], err_msg=f"step {step}: split rows"
        )
        np.testing.assert_array_equal(
            out_g, out_b, err_msg=f"step {step}: unchecked rows"
        )


@pytest.mark.parametrize("ws_fill,hoist", [("empty", True)])
@pytest.mark.parametrize("pcp", [2, 8])
@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_pcp_prefill_workspace_variants_packed_rope(
    monkeypatch, scenario, pcp, ws_fill, hoist
):
    """The L2b cache (TPU_PCP_PACKED_ROPE, sparsecore) with both flags."""
    base = _default_outputs(monkeypatch, scenario, pcp, "sparsecore", True)
    got = _pcp_prefill_outputs(
        monkeypatch,
        scenario,
        pcp,
        "sparsecore",
        packed_rope=True,
        ws_fill=ws_fill,
        hoist=hoist,
    )
    for step, ((out_g, _), (out_b, _)) in enumerate(zip(got, base)):
        np.testing.assert_array_equal(out_g, out_b, err_msg=f"step {step}")


def _pcp_prefill_outputs(
    monkeypatch,
    scenario,
    pcp,
    layout,
    packed_rope=False,
    ws_fill="zeros",
    hoist=False,
):
    """Runs every step of `scenario` through the PCP write + prefill attention,
    checks each split prefill row against numpy, and returns per step
    `(output [DCP, local_padded, H, LKV_DIM], checked-row mask [DCP, local_padded])`.
    The random inputs depend on (scenario, pcp, layout) only."""
    mesh = _mesh()
    monkeypatch.setattr(envs, "TPU_PCP_KV_CHUNK_TOKENS", CHUNK_TOKENS)
    nope_spec, rope_spec = _specs(KVCacheLayout(layout), packed_rope)
    assert rope_spec.packed_rope == packed_rope
    chunk_pages = pcp_chunk_pages(CHUNK_TOKENS, PAGE_SIZE, DCP)
    rng = np.random.default_rng(zlib.crc32(f"{scenario}-{pcp}-{layout}".encode()))
    tp = DCP // pcp
    steps = SCENARIOS[scenario]

    # Each request's latents for its whole life, fp8-rounded as cached.
    final_len = _final_lens(steps)
    kv_fp8 = [_fp8(rng, (n, LKV_DIM), K_SCALE) for n in final_len]
    pe_fp8 = [_fp8(rng, (n, ROPE_DIM), K_SCALE) for n in final_len]
    kv_deq = [x.astype(np.float32) * K_SCALE for x in kv_fp8]
    pe_deq = [x.astype(np.float32) * K_SCALE for x in pe_fp8]

    caches = tuple(
        jnp.zeros((DCP * s.shape[0], *s.shape[1:]), s.jax_dtype)
        for s in (nope_spec, rope_spec)
    )
    checked = 0
    results = []
    for st in _plan_steps(pcp, steps, _block_table(rng)):
        plan0, local_padded = st.plan0, st.local_padded
        kv_rows, pe_rows, q_rows, topk_rows, owners = [], [], [], [], []
        for p in range(pcp):
            kv = np.zeros((local_padded, LKV_DIM), kv_fp8[0].dtype)
            pe = np.zeros((local_padded, ROPE_DIM), pe_fp8[0].dtype)
            topk = np.full((local_padded, TOPK), -1, np.int32)
            owner = [None] * local_padded
            for i in range(local_padded):
                tok = st.row_tokens[p * local_padded + i]
                if tok is None:
                    continue
                r, pos = tok
                kv[i], pe[i] = kv_fp8[r][pos], pe_fp8[r][pos]
                if st.is_decode[r]:
                    # Decode rows are not in the split prefill; give them high
                    # cache indices anyway, which the prefill must not read.
                    topk[i] = rng.choice(
                        np.arange(WORKSPACE_TOKENS // 2, WORKSPACE_TOKENS),
                        TOPK,
                        replace=False,
                    )
                else:
                    topk[i] = _topk_row(rng, pos)
                    owner[i] = (r, pos)
            kv_rows.append(kv)
            pe_rows.append(pe)
            topk_rows.append(topk)
            owners.append(owner)
            q_rows.append(
                rng.standard_normal(
                    (local_padded, tp * HEADS_PER_RANK, LKV_DIM + ROPE_DIM)
                ).astype(np.float32)
            )
        q_bf16 = [
            np.asarray(jnp.asarray(q, jnp.bfloat16).astype(jnp.float32)) for q in q_rows
        ]
        # DCP rank d = (p, t) holds PCP rank p's rows and TP lane t's heads.
        h = HEADS_PER_RANK
        q_global = np.concatenate(
            [q_bf16[d // tp][:, (d % tp) * h : (d % tp + 1) * h] for d in range(DCP)]
        )

        written = attention_interface.sparse_mla_pcp_write_split(
            *caches,
            jnp.asarray(_by_dcp_rank(pcp, kv_rows)),
            jnp.asarray(_by_dcp_rank(pcp, pe_rows)),
            _rep(plan0.host_metadata["hidden_restore_idx"]),
            _rep(plan0.host_metadata["split_seq_lens"]),
            _rep(st.split_tables.reshape(-1)),
            _rep(plan0.host_metadata["split_query_start_loc"]),
            mesh,
            nope_spec,
            rope_spec,
            dcp_size=DCP,
            pcp_size=pcp,
            interleave_size=INTERLEAVE,
            prefetch=(
                (
                    _rep(plan0.host_metadata["ws_req"]),
                    _rep(plan0.host_metadata["ws_vpage"]),
                    chunk_pages,
                )
                if hoist
                else None
            ),
        )
        caches, chunk0 = tuple(written[:2]), tuple(written[2:])
        assert len(chunk0) == (2 if hoist else 0)
        out = attention_interface.sparse_mla_attention_pcp_prefill(
            *caches,
            jnp.asarray(q_global[..., :LKV_DIM], jnp.bfloat16),
            jnp.asarray(q_global[..., LKV_DIM:], jnp.bfloat16),
            _shard(pcp, topk_rows),
            _rep(st.split_tables),
            _rep(plan0.host_metadata["ws_req"]),
            _rep(plan0.host_metadata["ws_vpage"]),
            _shard(pcp, [r.plan.host_metadata["ws_page_offset"] for r in st.ranks]),
            _shard(pcp, [r.qsl for r in st.ranks]),
            _shard(
                pcp, [np.array([0, 0, r.plan.local_num_split_reqs]) for r in st.ranks]
            ),
            _rep(plan0.host_metadata["num_split_reqs"]),
            _rep(plan0.host_metadata["ws_used_pages"]),
            mesh,
            nope_spec,
            rope_spec,
            sm_scale=SM_SCALE,
            k_scale=K_SCALE,
            dcp_size=DCP,
            interleave_size=INTERLEAVE,
            chunk_pages=chunk_pages,
            ws_fill=ws_fill,
            chunk0=chunk0 or None,
        )
        out = np.asarray(out.astype(jnp.float32)).reshape(
            DCP, local_padded, HEADS_PER_RANK, LKV_DIM
        )
        mask = np.zeros((DCP, local_padded), bool)

        for d in range(DCP):
            p, t = divmod(d, tp)
            heads = slice(t * HEADS_PER_RANK, (t + 1) * HEADS_PER_RANK)
            for i, tok in enumerate(owners[p]):
                if tok is None:
                    continue
                r, pos = tok
                sel = topk_rows[p][i][topk_rows[p][i] >= 0]
                expected = _reference(q_bf16[p][i, heads], kv_deq[r], pe_deq[r], sel)
                np.testing.assert_allclose(
                    out[d, i],
                    expected,
                    rtol=2e-2,
                    atol=2e-2,
                    err_msg=f"step {st.index} dcp rank {d} row {i} "
                    f"(request {r}, position {pos})",
                )
                mask[d, i] = True
                checked += 1
        results.append((out, mask))
    assert checked > 0
    return results


# --- Indexer top-k ---------------------------------------------------------

IDX_HEADS = 8
IDX_DIM = 128
IDX_WIDTH = 256  # fp8 K + one UE8M0 scale byte, lane-padded
IDX_TOPK = 64
# Workspace buckets (context tokens) for the top-k `lax.switch`: 4 and 8
# virtual pages here, plus the full 32-page workspace.
IDX_BUCKET_TOKENS = (2048, 4096)
MAX_MODEL_LEN = BLOCKS_PER_REQ * PAGE_SIZE * DCP


def _quantize_index_k(x):
    """`[n, D]` f32 -> (`[n, IDX_WIDTH]` packed record, dequantized `[n, D]`):
    fp8 K with one UE8M0 (power-of-two) scale per row, as the indexer packs."""
    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    amax = np.clip(np.abs(x).max(-1, keepdims=True), 1e-4, None)
    scale = np.exp2(np.ceil(np.log2(amax / fp8_max))).astype(np.float32)
    quant = jnp.asarray(x / scale).astype(jnp.float8_e4m3fn)
    scale_e8 = jnp.asarray(scale).astype(jnp.float8_e8m0fnu)
    record = np.concatenate(
        [
            np.asarray(jax.lax.bitcast_convert_type(quant, jnp.uint8)),
            np.asarray(jax.lax.bitcast_convert_type(scale_e8, jnp.uint8)),
        ],
        -1,
    )
    record = np.pad(record, ((0, 0), (0, IDX_WIDTH - record.shape[-1])))
    deq = np.asarray(quant.astype(jnp.float32)) * np.asarray(
        scale_e8.astype(jnp.float32)
    )
    return record, deq


def _index_scores(q, weights, k_deq):
    """`sum_h w_h * relu(q_h . k)` over every position of `k_deq`."""
    return (np.maximum(q @ k_deq.T, 0.0) * weights[:, None]).sum(0)


def _check_topk(got, scores, pos, where):
    """`got` is a top-k of `scores[: pos + 1]`, up to near-ties at the cut."""
    picked = got[got >= 0]
    n = min(IDX_TOPK, pos + 1)
    assert len(picked) == n, f"{where}: {len(picked)} entries, expected {n}"
    assert len(set(picked.tolist())) == n, f"{where}: duplicate positions"
    assert picked.min() >= 0 and picked.max() <= pos, f"{where}: not causal"
    visible = scores[: pos + 1]
    cut = np.sort(visible)[-n]
    # The kernel's own top-k is approximate (unsharded it picks scores up to
    # ~0.3 below the exact cut on these inputs), so this is a sanity bound;
    # exactness is checked against the unsharded kernel.
    tol = 0.02 * np.abs(visible).max()
    assert visible[picked].min() >= cut - tol, f"{where}: picked a low score"
    must = np.flatnonzero(visible > cut + tol)
    missing = np.setdiff1d(must, picked)
    assert missing.size == 0, f"{where}: missed positions {missing[:8]}"


def _plain_topk(
    record,
    seq_len,
    q,
    weights,
    page_size=PAGE_SIZE,
    kv_layout=streamindex_topk.KVLayout.HEAD_ALONG_SUBLANE,
):
    """Unsharded `streamindex_topk` of the last `len(q)` positions of a
    `seq_len`-token request whose K is `record`, on one device."""
    num_q = q.shape[0]
    pages = -(-seq_len // page_size)
    flat = np.zeros((pages * page_size, IDX_WIDTH), np.uint8)
    flat[:seq_len, : record.shape[-1]] = record[:seq_len]
    cache = jnp.asarray(flat.reshape(pages, page_size // 4, 4, IDX_WIDTH))
    if kv_layout == streamindex_topk.KVLayout.SEQ_ALONG_LANE:
        cache = convert_cache_to_seq_along_lane(cache, IDX_DIM)
    padded = -(-num_q // 128) * 128
    q_pad = np.zeros((padded, *q.shape[1:]), q.dtype)
    q_pad[:num_q] = q
    w_pad = np.zeros((padded, IDX_HEADS), np.float32)
    w_pad[:num_q] = weights
    out = plain_streamindex_topk(
        jnp.asarray(q_pad),
        jnp.asarray(w_pad),
        cache,
        jnp.asarray([seq_len], jnp.int32),
        jnp.arange(pages, dtype=jnp.int32),
        jnp.asarray([0, num_q], jnp.int32),
        (0, 0, 1),
        k=IDX_TOPK,
        compression_ratio=1,
        num_kv_pages_per_block=(2, 2, 2),
        num_queries_per_block=(1, 128, 128),
        enable_early_exit=False,
        kv_layout=kv_layout,
    )
    return np.asarray(out)[:num_q]


@pytest.mark.parametrize("pcp", [2, 4, 8])
@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_pcp_indexer_topk_matches_numpy(monkeypatch, scenario, pcp):
    from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op

    mesh = _mesh()
    monkeypatch.setattr(envs, "TPU_PCP_KV_CHUNK_TOKENS", CHUNK_TOKENS)
    monkeypatch.setattr(
        mla_attention_op, "PCP_INDEXER_BUCKET_TOKENS", IDX_BUCKET_TOKENS
    )
    rng = np.random.default_rng(zlib.crc32(f"indexer-{scenario}-{pcp}".encode()))
    steps = SCENARIOS[scenario]
    final_len = _final_lens(steps)
    packed = [_quantize_index_k(rng.standard_normal((n, IDX_DIM))) for n in final_len]

    topk_fn = jax.jit(
        functools.partial(
            mla_attention_op.pcp_prefill_topk,
            mesh=mesh,
            topk=IDX_TOPK,
            dcp_size=DCP,
            pcp_size=pcp,
            interleave_size=INTERLEAVE,
            max_model_len=MAX_MODEL_LEN,
            chunk_tokens=envs.TPU_STREAMIDX_CHUNK_TOKENS,
        )
    )
    cache = jnp.zeros((DCP * NUM_PAGES, PAGE_SIZE // 4, 4, IDX_WIDTH), jnp.uint8)
    checked = 0
    for st in _plan_steps(pcp, steps, _block_table(rng)):
        plan0, local_padded = st.plan0, st.local_padded
        k_rows, q_rows, w_rows = [], [], []
        for p in range(pcp):
            k = np.zeros((local_padded, IDX_WIDTH), np.uint8)
            for i in range(local_padded):
                tok = st.row_tokens[p * local_padded + i]
                if tok is not None:
                    k[i] = packed[tok[0]][0][tok[1]]
            k_rows.append(k)
            q_rows.append(_fp8(rng, (local_padded, IDX_HEADS, IDX_DIM)))
            w_rows.append(
                rng.uniform(-1.5, 1.5, (local_padded, IDX_HEADS)).astype(np.float32)
            )

        cache, topk = topk_fn(
            cache,
            jnp.asarray(_by_dcp_rank(pcp, q_rows)).view(jnp.uint8),
            jnp.asarray(_by_dcp_rank(pcp, w_rows)),
            jnp.asarray(_by_dcp_rank(pcp, k_rows)),
            _rep(plan0.host_metadata["hidden_restore_idx"]),
            _rep(plan0.host_metadata["split_seq_lens"]),
            _rep(st.split_tables.reshape(-1)),
            _rep(plan0.host_metadata["split_query_start_loc"]),
            _rep(plan0.host_metadata["ws_req"]),
            _rep(plan0.host_metadata["ws_vpage"]),
            _shard(pcp, [r.plan.host_metadata["ws_page_offset"] for r in st.ranks]),
            _shard(pcp, [r.seq_lens for r in st.ranks]),
            _shard(pcp, [r.qsl for r in st.ranks]),
            _shard(
                pcp, [np.array([0, 0, r.plan.local_num_split_reqs]) for r in st.ranks]
            ),
            _rep(plan0.host_metadata["num_split_reqs"]),
            _rep(plan0.host_metadata["ws_used_pages"]),
        )
        topk = np.asarray(topk).reshape(DCP, local_padded, IDX_TOPK)

        # Every split request's queries in position order, through the
        # unsharded kernel: PCP must reproduce its top-k exactly.
        rows_of = {}
        for p in range(pcp):
            for i in range(local_padded):
                tok = st.row_tokens[p * local_padded + i]
                if tok is not None and not st.is_decode[tok[0]]:
                    rows_of.setdefault(tok[0], []).append((tok[1], p, i))
        plain = {}
        for r, rows in rows_of.items():
            rows.sort()
            pos0, seq_len = rows[0][0], rows[-1][0] + 1
            assert [x[0] for x in rows] == list(range(pos0, seq_len))
            out = _plain_topk(
                packed[r][0],
                seq_len,
                np.stack([q_rows[p][i] for _, p, i in rows]),
                np.stack([w_rows[p][i] for _, p, i in rows]),
            )
            plain.update({(r, pos): out[j] for j, (pos, _, _) in enumerate(rows)})

        tp = DCP // pcp
        for d in range(DCP):
            p = d // tp
            q = np.asarray(jnp.asarray(q_rows[p]).astype(jnp.float32))
            for i in range(local_padded):
                tok = st.row_tokens[p * local_padded + i]
                if tok is None or st.is_decode[tok[0]]:
                    continue
                r, pos = tok
                scores = _index_scores(q[i], w_rows[p][i], packed[r][1])
                where = (
                    f"step {st.index} dcp rank {d} row {i} (request {r}, "
                    f"position {pos})"
                )
                np.testing.assert_array_equal(
                    np.sort(topk[d, i]),
                    np.sort(plain[(r, pos)]),
                    err_msg=f"{where}: differs from the unsharded kernel",
                )
                _check_topk(topk[d, i], scores, pos, where)
                checked += 1
    assert checked > 0


# --- Indexer top-k, SEQ_ALONG_LANE (TPU_PCP_INDEXER_SAL) --------------------

# SEQ_ALONG_LANE needs a 128-multiple page: 128-token pages, 1024-token
# virtual pages, buckets of 4 and 8 of them plus the full 16-page workspace.
SAL_PAGE_SIZE = 128
SAL_BUCKET_TOKENS = (4096, 8192)
SAL_MAX_MODEL_LEN = BLOCKS_PER_REQ * SAL_PAGE_SIZE * DCP


@pytest.mark.parametrize("compact", [True, False])
@pytest.mark.parametrize("pcp", [2, 8])
@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_pcp_indexer_topk_sal_matches_has(monkeypatch, scenario, pcp, compact):
    """`pcp_prefill_topk` over a SEQ_ALONG_LANE indexer K cache
    (TPU_PCP_INDEXER_SAL) writes the same records as the HEAD_ALONG_SUBLANE
    path (up to the layout) and returns the same top-k positions; both match
    the unsharded SAL kernel and the numpy scores."""
    from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op

    has = streamindex_topk.KVLayout.HEAD_ALONG_SUBLANE
    sal = streamindex_topk.KVLayout.SEQ_ALONG_LANE
    mesh = _mesh()
    monkeypatch.setattr(envs, "TPU_PCP_KV_CHUNK_TOKENS", CHUNK_TOKENS)
    monkeypatch.setattr(
        mla_attention_op, "PCP_INDEXER_BUCKET_TOKENS", SAL_BUCKET_TOKENS
    )
    rng = np.random.default_rng(zlib.crc32(f"indexer-sal-{scenario}-{pcp}".encode()))
    steps = SCENARIOS[scenario]
    final_len = _final_lens(steps)
    packed = [_quantize_index_k(rng.standard_normal((n, IDX_DIM))) for n in final_len]
    # The production record: fp8 K + one scale byte, unpadded.
    rec_width = IDX_DIM + 1

    def topk_fn(kv_layout):
        return jax.jit(
            functools.partial(
                mla_attention_op.pcp_prefill_topk,
                mesh=mesh,
                topk=IDX_TOPK,
                dcp_size=DCP,
                pcp_size=pcp,
                interleave_size=INTERLEAVE,
                max_model_len=SAL_MAX_MODEL_LEN,
                chunk_tokens=envs.TPU_STREAMIDX_CHUNK_TOKENS,
                kv_layout=kv_layout,
                k_write_compact=compact,
            )
        )

    fns = {has: topk_fn(has), sal: topk_fn(sal)}
    caches = {
        layout: jnp.zeros(
            streamindex_topk.get_kv_cache_shape(
                DCP * NUM_PAGES, SAL_PAGE_SIZE, IDX_DIM, layout
            ),
            jnp.uint8,
        )
        for layout in (has, sal)
    }
    assert caches[sal].shape[1:] == (IDX_DIM // 4 + 1, 4, SAL_PAGE_SIZE)
    checked = 0
    plans = _plan_steps(pcp, steps, _block_table(rng), page_size=SAL_PAGE_SIZE)
    for st in plans:
        plan0, local_padded = st.plan0, st.local_padded
        k_rows, q_rows, w_rows = [], [], []
        for p in range(pcp):
            k = np.zeros((local_padded, rec_width), np.uint8)
            for i in range(local_padded):
                tok = st.row_tokens[p * local_padded + i]
                if tok is not None:
                    k[i] = packed[tok[0]][0][tok[1], :rec_width]
            k_rows.append(k)
            q_rows.append(_fp8(rng, (local_padded, IDX_HEADS, IDX_DIM)))
            w_rows.append(
                rng.uniform(-1.5, 1.5, (local_padded, IDX_HEADS)).astype(np.float32)
            )
        args = (
            jnp.asarray(_by_dcp_rank(pcp, q_rows)).view(jnp.uint8),
            jnp.asarray(_by_dcp_rank(pcp, w_rows)),
            jnp.asarray(_by_dcp_rank(pcp, k_rows)),
            _rep(plan0.host_metadata["hidden_restore_idx"]),
            _rep(plan0.host_metadata["split_seq_lens"]),
            _rep(st.split_tables.reshape(-1)),
            _rep(plan0.host_metadata["split_query_start_loc"]),
            _rep(plan0.host_metadata["ws_req"]),
            _rep(plan0.host_metadata["ws_vpage"]),
            _shard(pcp, [r.plan.host_metadata["ws_page_offset"] for r in st.ranks]),
            _shard(pcp, [r.seq_lens for r in st.ranks]),
            _shard(pcp, [r.qsl for r in st.ranks]),
            _shard(
                pcp, [np.array([0, 0, r.plan.local_num_split_reqs]) for r in st.ranks]
            ),
            _rep(plan0.host_metadata["num_split_reqs"]),
            _rep(plan0.host_metadata["ws_used_pages"]),
        )
        topk = {}
        for layout in (has, sal):
            caches[layout], out = fns[layout](caches[layout], *args)
            topk[layout] = np.asarray(out).reshape(DCP, local_padded, IDX_TOPK)

        # The SAL writer put every record where the SAL reader expects it.
        np.testing.assert_array_equal(
            np.asarray(caches[sal]),
            np.asarray(convert_cache_to_seq_along_lane(caches[has], IDX_DIM)),
            err_msg=f"step {st.index}: SAL cache != HAS cache in SAL layout",
        )

        rows_of = {}
        for p in range(pcp):
            for i in range(local_padded):
                tok = st.row_tokens[p * local_padded + i]
                if tok is not None and not st.is_decode[tok[0]]:
                    rows_of.setdefault(tok[0], []).append((tok[1], p, i))
        plain = {}
        for r, rows in rows_of.items():
            rows.sort()
            pos0, seq_len = rows[0][0], rows[-1][0] + 1
            assert [x[0] for x in rows] == list(range(pos0, seq_len))
            out = _plain_topk(
                packed[r][0][:, :rec_width],
                seq_len,
                np.stack([q_rows[p][i] for _, p, i in rows]),
                np.stack([w_rows[p][i] for _, p, i in rows]),
                page_size=SAL_PAGE_SIZE,
                kv_layout=sal,
            )
            plain.update({(r, pos): out[j] for j, (pos, _, _) in enumerate(rows)})

        tp = DCP // pcp
        for d in range(DCP):
            p = d // tp
            q = np.asarray(jnp.asarray(q_rows[p]).astype(jnp.float32))
            for i in range(local_padded):
                tok = st.row_tokens[p * local_padded + i]
                if tok is None or st.is_decode[tok[0]]:
                    continue
                r, pos = tok
                where = (
                    f"step {st.index} dcp rank {d} row {i} (request {r}, "
                    f"position {pos})"
                )
                np.testing.assert_array_equal(
                    np.sort(topk[sal][d, i]),
                    np.sort(topk[has][d, i]),
                    err_msg=f"{where}: SAL top-k differs from HAS",
                )
                np.testing.assert_array_equal(
                    np.sort(topk[sal][d, i]),
                    np.sort(plain[(r, pos)]),
                    err_msg=f"{where}: differs from the unsharded SAL kernel",
                )
                scores = _index_scores(q[i], w_rows[p][i], packed[r][1])
                _check_topk(topk[sal][d, i], scores, pos, where)
                checked += 1
    assert checked > 0


@pytest.mark.parametrize("ws_no_zero_fill", [False, True])
@pytest.mark.parametrize("hoist", [False, True])
def test_pcp_adapter_ops_build_under_flags(monkeypatch, hoist, ws_no_zero_fill):
    """The adapter's PCP write / split prefill torch ops register under
    TPU_PCP_GATHER_HOIST / TPU_PCP_WS_NO_ZERO_FILL: `pallas.jax_op` accepts
    only explicit array parameters and a tuple-of-arrays return annotation."""
    from vllm_torchtpu.layers.adapter import attention as adapter_attention

    mesh = _mesh()
    monkeypatch.setattr(envs, "TPU_PCP_GATHER_HOIST", hoist)
    monkeypatch.setattr(envs, "TPU_PCP_WS_NO_ZERO_FILL", ws_no_zero_fill)
    impl = SimpleNamespace(
        _dcp_mesh=mesh, dcp_size=DCP, pcp_size=2, dcp_interleave_size=INTERLEAVE
    )
    layer = SimpleNamespace(
        layer_name=f"model.layers.0.attn_flags_{int(hoist)}{int(ws_no_zero_fill)}",
        mla_kv_spec=_specs(KVCacheLayout.SPARSECORE, packed_rope=True),
        kv_cache_quantized_dtype=None,
    )
    cls = adapter_attention.PallasMLAttentionBackendImpl
    assert callable(cls._build_pcp_write_split_op(impl, layer))
    assert callable(cls._build_pcp_prefill_op(impl, layer))
