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

from __future__ import annotations

import inspect
import types
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from torch_tpu._internal.pallas import pallas as pallas_impl

from vllm_torchtpu.kernels.mla import dcp_token_parallel as tp
from vllm_torchtpu.kernels.mla.fused_qb_absorb import (
    fused_qb_rope_absorb_split_qpe_ref,
    fused_qb_rope_absorb_split_ref,
)
from vllm_torchtpu.layers.adapter import (
    attention as attention_mod,
)
from vllm_torchtpu.layers.adapter import (
    dcp_nonidx_qb_mla_op as in_branch_mod,
)
from vllm_torchtpu.layers.adapter import (
    dcp_token_parallel_op as tp_op_mod,
)
from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op as mla_op

AXIS = "dcp"
HEADS = 4
Q_LORA = 16
D_NOPE = 8
D_PE = 4
DN = 8
DV = 6
ROPE_PER_ROW = 4
TOPK = 12


def _mesh(dcp: int) -> Mesh:
    if jax.device_count() < dcp:
        pytest.skip(f"requires {dcp} devices; CPU virtual devices are sufficient")
    return Mesh(np.array(jax.devices()[:dcp]), (AXIS,))


def _ref_lead_attend(ql_head, qpe_head, nope, rope, h_topk, bt, sl, qsl, dist):
    del sl, qsl, dist
    num_pages, ps, dn = nope.shape
    local_bt = jnp.mod(bt, jnp.int32(num_pages))
    # Map each [S, TOPK] physical-shard local entry to its cached (nope, rope).
    nope_flat = nope.reshape(num_pages * ps, dn)
    rope_flat = rope.reshape(num_pages * ps, D_PE)
    owns = h_topk >= 0
    safe_idx = jnp.where(owns, h_topk, jnp.int32(0))
    # Keep block-table dependency live so accidental bt wiring bugs are caught.
    bt_bias = (local_bt[0] & jnp.int32(0)).astype(jnp.int32)
    safe_idx = jnp.clip(safe_idx + bt_bias, 0, num_pages * ps - 1)
    k_n = nope_flat[safe_idx]  # [S, TOPK, DN]
    k_r = rope_flat[safe_idx]  # [S, TOPK, D_PE]
    logits = (
        jnp.einsum("shd,skd->shk", ql_head.astype(jnp.float32), k_n.astype(jnp.float32))
        + jnp.einsum(
            "shd,skd->shk", qpe_head.astype(jnp.float32), k_r.astype(jnp.float32)
        )
    ) * (1.0 / np.sqrt(DN + D_PE))
    logits = jnp.where(owns[:, None, :], logits, -jnp.inf)
    has_any = jnp.any(owns, axis=-1)
    safe_logits = jnp.where(has_any[:, None, None], logits, 0.0)
    lse = jax.scipy.special.logsumexp(safe_logits, axis=-1)
    weights = jnp.exp(safe_logits - lse[:, :, None])
    weights = jnp.where(owns[:, None, :], weights, 0.0)
    out = jnp.einsum("shk,skd->shd", weights, k_n.astype(jnp.float32)).astype(
        ql_head.dtype
    )
    lse = jnp.where(has_any[:, None], lse, -jnp.inf)
    return out, lse


def _ref_tail_attend(q, g_nope, g_rope, flat, num_rows, q_pe):
    # g_nope: [cap*dcp, ps, DN]; g_rope: [cap*dcp, ps // ROPE_PER_ROW, ...]
    k_n = g_nope.reshape(-1, DN)
    k_r = g_rope.reshape(-1, D_PE)
    valid = flat >= 0
    safe = jnp.clip(jnp.where(valid, flat, jnp.int32(0)), 0, k_n.shape[0] - 1)
    kn_sel = k_n[safe]
    kr_sel = k_r[safe]
    logits = (
        jnp.einsum("rhd,rkd->rhk", q.astype(jnp.float32), kn_sel.astype(jnp.float32))
        + jnp.einsum(
            "rhd,rkd->rhk", q_pe.astype(jnp.float32), kr_sel.astype(jnp.float32)
        )
    ) * (1.0 / np.sqrt(DN + D_PE))
    logits = jnp.where(valid[:, None, :], logits, -jnp.inf)
    has_any = jnp.any(valid, axis=-1)
    safe_logits = jnp.where(has_any[:, None, None], logits, 0.0)
    lse = jax.scipy.special.logsumexp(safe_logits, axis=-1)
    weights = jnp.where(valid[:, None, :], jnp.exp(safe_logits - lse[:, :, None]), 0.0)
    out = jnp.einsum("rhk,rkd->rhd", weights, kn_sel.astype(jnp.float32))
    row_mask = jnp.arange(q.shape[0]) < num_rows
    return jnp.where(row_mask[:, None, None], out, 0.0).astype(q.dtype)


def _make_step_fixture(
    layout: tp.TokenParallelLayout,
    num_tokens: int,
    reqs: list[tuple[int, int]],
    *,
    flat_topk_cap_strided: bool,
    max_reqs: int = 8,
    seed: int = 198,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    vps, ps, dcp = layout.virtual_page_size, layout.page_size, layout.dcp_size
    num_seqs = len(reqs)
    seq_lens = np.zeros(max_reqs, np.int32)
    qsl = [0]
    for i, (prefix, chunk) in enumerate(reqs):
        seq_lens[i] = prefix + chunk
        qsl.append(qsl[-1] + chunk)
    total = qsl[-1]
    assert total <= num_tokens
    qsl_arr = np.full(max_reqs + 1, total, np.int32)
    qsl_arr[: len(qsl)] = qsl
    dist = np.array([0, 0, num_seqs], np.int32)

    vpages_per_seq = -(-int(seq_lens.max()) // vps) + 1
    num_pages = 4 * max_reqs * vpages_per_seq
    perm = rng.permutation(num_pages)
    bt = np.zeros((max_reqs, vpages_per_seq), np.int32)
    cursor = 0
    for i, s_len in enumerate(seq_lens[:num_seqs]):
        k = -(-int(s_len) // vps)
        bt[i, :k] = perm[cursor : cursor + k] + num_pages * (i % 2)
        cursor += k

    nope_c = rng.standard_normal((dcp, num_pages, ps, DN)).astype(np.float32)
    rope_c = rng.standard_normal(
        (dcp, num_pages, ps // ROPE_PER_ROW, ROPE_PER_ROW * D_PE)
    ).astype(np.float32)

    rows = layout.rows_per_rank(num_tokens)
    s = layout.split_rows
    # Anti-cancellation: non-zero distinct head and tail rows across ranks,
    # with head rows replicated across ranks (as q_rows produces) and tail rows
    # rank-unique.
    qc_head = rng.standard_normal((s, Q_LORA)).astype(np.float32)
    qc_tail = rng.standard_normal((dcp, rows, Q_LORA)).astype(np.float32)
    qc_own = np.concatenate(
        [np.broadcast_to(qc_head[None], (dcp, s, Q_LORA)), qc_tail], axis=1
    ).reshape(dcp * (s + rows), Q_LORA)
    cs_head = rng.standard_normal((s, D_PE)).astype(np.float32)
    cs_tail = rng.standard_normal((dcp, rows, D_PE)).astype(np.float32)
    cos_sin = np.concatenate(
        [np.broadcast_to(cs_head[None], (dcp, s, D_PE)), cs_tail], axis=1
    ).reshape(dcp * (s + rows), D_PE)

    w_qb_f32 = rng.standard_normal((Q_LORA, HEADS * (D_NOPE + D_PE))).astype(np.float32)
    s_qb = rng.uniform(0.25, 1.5, (HEADS * (D_NOPE + D_PE),)).astype(np.float32)
    w_uk_t = rng.standard_normal((HEADS, D_NOPE, DN)).astype(np.float32)
    w_uv = rng.standard_normal((HEADS, DN, DV)).astype(np.float32)

    # Build sparse head_topk with some empty-owner rows (-1 across all slots)
    # to test -inf LSE masking across ranks.
    head_topk = np.full((dcp, s, TOPK), -1, dtype=np.int32)
    for row in range(s):
        active_ranks = [row % dcp, (row + 1) % dcp]
        for rk in active_ranks:
            head_topk[rk, row, : TOPK // 2] = rng.integers(
                0, num_pages * ps, size=TOPK // 2, dtype=np.int32
            )

    pages = -(-seq_lens // vps)
    total_pages = int(pages[:num_seqs].sum())
    cap = next(c for c in layout.gather_caps if c >= total_pages)
    stride_pages = cap if flat_topk_cap_strided else layout.max_pages
    max_flat = min(cap, total_pages) * dcp * ps
    flat_topk = np.full((dcp, rows, TOPK), -1, dtype=np.int32)
    for rk in range(dcp):
        for row in range(rows):
            n_live = int(rng.integers(2, TOPK + 1))
            tok_ids = rng.integers(0, max_flat, size=n_live, dtype=np.int32)
            # Encode in (owner * stride_pages + p) * ps + slot format.
            owner = tok_ids % dcp
            pg = (tok_ids // dcp) % max(1, total_pages)
            slot = (tok_ids // (dcp * max(1, total_pages))) % ps
            flat_topk[rk, row, :n_live] = (owner * stride_pages + pg) * ps + slot

    return {
        "nope_cache": nope_c.reshape(dcp * num_pages, ps, DN),
        "rope_cache": rope_c.reshape(dcp * num_pages, ps // ROPE_PER_ROW, -1),
        "qc_own": qc_own,
        "w_qb": w_qb_f32,
        "s_qb": s_qb,
        "w_uk_t": w_uk_t,
        "cos_sin": cos_sin,
        "head_topk": head_topk.reshape(dcp * s, TOPK),
        "flat_topk": flat_topk.reshape(dcp * rows, TOPK),
        "seq_lens": seq_lens,
        "bt": bt.reshape(-1),
        "qsl": qsl_arr,
        "dist": dist,
        "w_uv": w_uv,
        "rank_ids": np.arange(dcp, dtype=np.int32),
    }


def test_should_match_fused_qb_split_qpe_ref_slices():
    rng = np.random.default_rng(199)
    total_tokens, split_rows = 40, 8
    q_c = jnp.asarray(rng.standard_normal((total_tokens, Q_LORA)), dtype=jnp.bfloat16)
    w_qb = jnp.asarray(
        rng.standard_normal((Q_LORA, HEADS * (D_NOPE + D_PE))),
        dtype=jnp.float8_e4m3fn,
    )
    w_s = jnp.asarray(
        rng.uniform(0.25, 1.5, (HEADS * (D_NOPE + D_PE),)), dtype=jnp.float32
    )
    w_uk_t = jnp.asarray(rng.standard_normal((HEADS, D_NOPE, DN)), dtype=jnp.bfloat16)
    cos_sin = jnp.asarray(rng.standard_normal((total_tokens, D_PE)), dtype=jnp.bfloat16)

    ref_h, ref_t, ref_pe = fused_qb_rope_absorb_split_ref(
        q_c, w_qb, w_s, w_uk_t, cos_sin, split_rows
    )
    got_h, got_t, got_peh, got_pet = fused_qb_rope_absorb_split_qpe_ref(
        q_c, w_qb, w_s, w_uk_t, cos_sin, split_rows
    )
    np.testing.assert_array_equal(
        np.asarray(got_h.view(jnp.uint16)), np.asarray(ref_h.view(jnp.uint16))
    )
    np.testing.assert_array_equal(
        np.asarray(got_t.view(jnp.uint16)), np.asarray(ref_t.view(jnp.uint16))
    )
    np.testing.assert_array_equal(
        np.asarray(got_peh.view(jnp.uint16)),
        np.asarray(ref_pe[:split_rows].view(jnp.uint16)),
    )
    np.testing.assert_array_equal(
        np.asarray(got_pet.view(jnp.uint16)),
        np.asarray(ref_pe[split_rows:].view(jnp.uint16)),
    )


@pytest.mark.parametrize("free_nobar_max_cap", [0, 4, 8])
@pytest.mark.parametrize("flat_topk_cap_strided", [False, True])
def test_should_match_separate_fused_qb_and_lead_on_four_devices(
    free_nobar_max_cap: int, flat_topk_cap_strided: bool
):
    dcp = 4
    mesh = _mesh(dcp)
    layout = tp.TokenParallelLayout(
        dcp, 4, 16, split_rows=8, max_pages=8, gather_caps=(2, 4, 8)
    )
    num_tokens = 48
    reqs = [(80, 20), (48, 22)]
    fx = _make_step_fixture(
        layout, num_tokens, reqs, flat_topk_cap_strided=flat_topk_cap_strided
    )

    shd = NamedSharding(mesh, P(AXIS))
    rep = NamedSharding(mesh, P())
    put = jax.device_put

    nope_c = put(jnp.asarray(fx["nope_cache"], dtype=jnp.bfloat16), shd)
    rope_c = put(jnp.asarray(fx["rope_cache"], dtype=jnp.bfloat16), shd)
    qc_own = put(jnp.asarray(fx["qc_own"], dtype=jnp.bfloat16), shd)
    w_qb = put(jnp.asarray(fx["w_qb"], dtype=jnp.float8_e4m3fn), rep)
    s_qb = put(jnp.asarray(fx["s_qb"], dtype=jnp.float32), rep)
    w_uk_t = put(jnp.asarray(fx["w_uk_t"], dtype=jnp.bfloat16), rep)
    cos_sin = put(jnp.asarray(fx["cos_sin"], dtype=jnp.bfloat16), shd)
    head_topk = put(jnp.asarray(fx["head_topk"], dtype=jnp.int32), shd)
    flat_topk = put(jnp.asarray(fx["flat_topk"], dtype=jnp.int32), shd)
    seq_lens = put(jnp.asarray(fx["seq_lens"], dtype=jnp.int32), rep)
    bt = put(jnp.asarray(fx["bt"], dtype=jnp.int32), rep)
    qsl = put(jnp.asarray(fx["qsl"], dtype=jnp.int32), rep)
    dist = put(jnp.asarray(fx["dist"], dtype=jnp.int32), rep)
    w_uv = put(jnp.asarray(fx["w_uv"], dtype=jnp.bfloat16), rep)
    rank_ids = put(jnp.asarray(fx["rank_ids"], dtype=jnp.int32), shd)

    # Separate baseline: eager fused_qb -> lead_attend -> sparse_mla_with_lead.
    ql_h, ql_t, qpe_h, qpe_t = jax.shard_map(
        lambda qc, wqb, sqb, wuk, cs: fused_qb_rope_absorb_split_qpe_ref(
            qc, wqb, sqb, wuk, cs, layout.split_rows
        ),
        mesh=mesh,
        in_specs=(P(AXIS), P(), P(), P(), P(AXIS)),
        out_specs=(P(AXIS), P(AXIS), P(AXIS), P(AXIS)),
        check_vma=False,
    )(qc_own, w_qb, s_qb, w_uk_t, cos_sin)
    lead_partial, lead_lse = jax.shard_map(
        _ref_lead_attend,
        mesh=mesh,
        in_specs=(
            P(AXIS),
            P(AXIS),
            P(AXIS),
            P(AXIS),
            P(AXIS),
            P(),
            P(),
            P(),
            P(),
        ),
        out_specs=(P(AXIS), P(AXIS)),
        check_vma=False,
    )(ql_h, qpe_h, nope_c, rope_c, head_topk, bt, seq_lens, qsl, dist)
    _, _, ref_own_v, ref_lead_v = tp.sparse_mla_attention_with_lead(
        layout,
        mesh,
        AXIS,
        nope_c,
        rope_c,
        ql_t,
        None,
        None,
        flat_topk,
        seq_lens,
        bt,
        qsl,
        dist,
        lead_partial=lead_partial,
        lead_lse=lead_lse,
        write_kv=None,
        attend=_ref_tail_attend,
        w_uv=w_uv,
        rank_ids=rank_ids,
        q_own_rows=True,
        q_pe=qpe_t,
        w_uv_replicated=True,
        num_tokens=num_tokens,
        flat_topk_cap_strided=flat_topk_cap_strided,
    )

    got_own_v, got_lead_v = tp.sparse_mla_attention_with_in_branch_qb(
        layout,
        mesh,
        AXIS,
        nope_c,
        rope_c,
        qc_own,
        w_qb,
        s_qb,
        w_uk_t,
        cos_sin,
        head_topk,
        flat_topk,
        seq_lens,
        bt,
        qsl,
        dist,
        fused_qb=fused_qb_rope_absorb_split_qpe_ref,
        lead_attend=_ref_lead_attend,
        tail_attend=_ref_tail_attend,
        w_uv=w_uv,
        num_tokens=num_tokens,
        rank_ids=rank_ids,
        w_uv_replicated=True,
        flat_topk_cap_strided=flat_topk_cap_strided,
        free_nobar_max_cap=free_nobar_max_cap,
    )

    np.testing.assert_array_equal(
        np.asarray(got_own_v.view(jnp.uint16)),
        np.asarray(ref_own_v.view(jnp.uint16)),
    )
    np.testing.assert_array_equal(
        np.asarray(got_lead_v.view(jnp.uint16)),
        np.asarray(ref_lead_v.view(jnp.uint16)),
    )

    # Fused single-shard_map o_proj path matches project_sequence_rows_from_v bit-exact.
    if free_nobar_max_cap == 4 and flat_topk_cap_strided:
        rng_o = np.random.default_rng(920)
        k_full = ref_own_v.shape[1] * ref_own_v.shape[2]
        half_k = k_full // 2
        hidden = 16
        w_o_fp8 = jnp.asarray(
            rng_o.standard_normal((k_full, hidden)) * 0.5, jnp.float8_e4m3fn
        )
        w_o_shd = put(
            jnp.concatenate(
                [
                    w_o_fp8[:half_k] if r % 2 == 0 else w_o_fp8[half_k:]
                    for r in range(dcp)
                ],
                axis=0,
            ),
            shd,
        )
        s_o_shd = put(
            jnp.tile(
                jnp.asarray(rng_o.uniform(0.01, 0.05, (hidden,)), jnp.float32), (dcp,)
            ),
            shd,
        )
        ref_proj = tp.project_sequence_rows_from_v(
            layout, mesh, AXIS, ref_lead_v, ref_own_v, rank_ids, w_o_shd, s_o_shd
        )
        got_proj = tp.sparse_mla_attention_with_in_branch_qb_oproj(
            layout,
            mesh,
            AXIS,
            nope_c,
            rope_c,
            qc_own,
            w_qb,
            s_qb,
            w_uk_t,
            cos_sin,
            head_topk,
            flat_topk,
            seq_lens,
            bt,
            qsl,
            dist,
            fused_qb=fused_qb_rope_absorb_split_qpe_ref,
            lead_attend=_ref_lead_attend,
            tail_attend=_ref_tail_attend,
            w_uv=w_uv,
            w_o=w_o_shd,
            w_o_scale=s_o_shd,
            num_tokens=num_tokens,
            rank_ids=rank_ids,
            w_uv_replicated=True,
            flat_topk_cap_strided=flat_topk_cap_strided,
            free_nobar_max_cap=free_nobar_max_cap,
            epilogue_3d=True,
        )
        np.testing.assert_array_equal(
            np.asarray(got_proj).view(np.uint16), np.asarray(ref_proj).view(np.uint16)
        )


def test_should_pass_torch_tpu_pallas_verify_signature_for_in_branch_qb_wrappers(
    monkeypatch,
):
    mesh = _mesh(4)
    layout = tp.TokenParallelLayout(4, 4, 32, split_rows=16, max_pages=8)
    monkeypatch.setattr(in_branch_mod, "glm_dsa_v7x_active", lambda *_a, **_kw: True)
    monkeypatch.setattr(
        in_branch_mod, "get_tensor_model_parallel_world_size", lambda: 4
    )
    monkeypatch.setattr(
        in_branch_mod, "dcp_rank_operand", lambda: torch.zeros((1,), dtype=torch.int32)
    )

    registered: list[tuple[str, Any, Any, tuple[Any, ...], tuple[Any, ...]]] = []

    class _FakeBuiltOp:
        def __init__(self, name: str, fn: Any, in_s: Any, out_s: Any) -> None:
            self.name = name
            self.fn = fn
            self.in_s = in_s
            self.out_s = out_s
            self.fake_fn: Any = None

        def register_fake(self, fake_fn: Any) -> Any:
            self.fake_fn = fake_fn
            registered.append((self.name, self.fn, fake_fn, self.in_s, self.out_s))
            return fake_fn

    def _capture_jax_op(
        name: str,
        fn: Any,
        *,
        donate_argnums: tuple[int, ...],
        mesh: Any,
        input_partition_specs: tuple[Any, ...],
        output_partition_specs: tuple[Any, ...],
    ) -> _FakeBuiltOp:
        del donate_argnums, mesh
        sig = inspect.signature(fn)
        pallas_impl._verify_signature(sig)  # noqa: SLF001 -- real torch_tpu op-registration signature check
        assert pallas_impl._infer_static_argnums(sig) == ()  # noqa: SLF001 -- every operand must bind as jax.Array
        pallas_impl._get_torch_signature(sig)  # noqa: SLF001 -- verify Torch schema translation succeeds
        assert len(sig.parameters) == len(input_partition_specs)
        return _FakeBuiltOp(name, fn, input_partition_specs, output_partition_specs)

    monkeypatch.setattr(in_branch_mod, "pcp_streaming_jax_op", _capture_jax_op)
    monkeypatch.setattr(
        in_branch_mod,
        "get_shared_jax_op",
        lambda kind, key, build: build(f"p::{kind}"),
    )

    class _FakeLayer:
        kv_cache_quantized_dtype = None
        scale = 0.135
        num_heads = 4
        v_head_dim = 6
        use_sparse = True

    layer = _FakeLayer()
    impl = types.SimpleNamespace(
        _dcp_mesh=mesh,
        dcp_size=4,
        dcp_interleave_size=4,
    )
    op = in_branch_mod.build_in_branch_qb_attention_op(
        layer, impl, layout, q_replicated=True
    )
    assert op is not None
    assert op.serves_sharded_kv is True
    assert len(registered) == 2
    assert all(len(in_s) == 18 for _, _, _, in_s, _ in registered)

    # Verify fake-tensor shape formulas for both plain and sharded KV ops.
    nope = torch.empty((8, 32, 8), dtype=torch.bfloat16)
    rope = torch.empty((8, 8, 16), dtype=torch.bfloat16)
    q_c = torch.empty((32, 16), dtype=torch.bfloat16)
    w_qb = torch.empty((16, 48), dtype=torch.float8_e4m3fn)
    s_qb = torch.empty((48,), dtype=torch.float32)
    w_uk = torch.empty((4, 8, 8), dtype=torch.bfloat16)
    cs = torch.empty((32, 4), dtype=torch.bfloat16)
    tk = torch.empty((64, 12), dtype=torch.int32)
    sl = torch.empty((8,), dtype=torch.int32)
    bt = torch.empty((8, 2), dtype=torch.int32)
    qsl = torch.empty((9,), dtype=torch.int32)
    dist = torch.empty((3,), dtype=torch.int32)
    rk = torch.empty((1,), dtype=torch.int32)
    w_uv = torch.empty((4, 8, 6), dtype=torch.bfloat16)
    w_o = torch.empty((24, 32), dtype=torch.float8_e4m3fn)
    s_o = torch.empty((32,), dtype=torch.float32)

    plain_fake = registered[0][2]
    sharded_fake = registered[1][2]
    out_plain = plain_fake(
        nope,
        rope,
        q_c,
        w_qb,
        s_qb,
        w_uk,
        cs,
        torch.empty((64, 8), dtype=torch.bfloat16),
        torch.empty((64, 4), dtype=torch.bfloat16),
        tk,
        sl,
        bt,
        qsl,
        dist,
        rk,
        w_uv,
        w_o,
        s_o,
    )
    assert out_plain[2].shape == (16, 32)
    out_sharded = sharded_fake(
        nope,
        rope,
        q_c,
        w_qb,
        s_qb,
        w_uk,
        cs,
        torch.empty((16, 8), dtype=torch.bfloat16),
        torch.empty((16, 4), dtype=torch.bfloat16),
        tk,
        sl,
        bt,
        qsl,
        dist,
        rk,
        w_uv,
        w_o,
        s_o,
    )
    assert out_sharded[2].shape == (16, 32)


@pytest.mark.parametrize(
    ("glm_active", "q_replicated", "tp_size", "has_uv_scale"),
    [
        (False, True, 4, False),
        (True, False, 4, False),
        (True, True, 8, False),
        (True, True, 4, True),
    ],
)
def test_should_gate_build_on_env_and_layer_contracts(
    monkeypatch,
    glm_active: bool,
    q_replicated: bool,
    tp_size: int,
    has_uv_scale: bool,
):
    layout = tp.TokenParallelLayout(4, 4, 32, split_rows=16, max_pages=8)
    monkeypatch.setattr(
        in_branch_mod, "glm_dsa_v7x_active", lambda *_a, **_kw: glm_active
    )
    monkeypatch.setattr(
        in_branch_mod, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    layer = types.SimpleNamespace()
    if has_uv_scale:
        layer.W_UV_scale = torch.ones((4,))
    assert (
        in_branch_mod.build_in_branch_qb_attention_op(
            layer, types.SimpleNamespace(), layout, q_replicated=q_replicated
        )
        is None
    )


def test_should_dispatch_in_branch_qb_and_preserve_empty_cache_probe(monkeypatch):
    impl = attention_mod.PallasMLAttentionBackendImpl.__new__(
        attention_mod.PallasMLAttentionBackendImpl
    )
    impl.dcp_size = 4

    called: list[dict[str, Any]] = []

    def _fake_in_branch(
        layer,
        kv_cache,
        qb_inputs,
        kv_c_normed,
        k_pe,
        topk_indices,
        attn_metadata,
        *,
        w_o,
        kv_sharded=False,
    ):
        called.append(
            {
                "qb_inputs": qb_inputs,
                "kv_c_shape": tuple(kv_c_normed.shape),
                "k_pe_shape": tuple(k_pe.shape),
                "kv_sharded": kv_sharded,
                "w_o": w_o,
            }
        )
        kv_cache[0].fill_(3.0)
        kv_cache[1].fill_(5.0)
        return torch.full((16, 32), 7.0, dtype=qb_inputs.q_c.dtype)

    monkeypatch.setattr(in_branch_mod, "in_branch_qb_attention", _fake_in_branch)
    monkeypatch.setattr(tp_op_mod, "applies", lambda n: n >= 64)
    monkeypatch.setattr(
        impl, "_get_kv_scales", lambda _layer: (None, None, None), raising=False
    )

    layer = types.SimpleNamespace(
        num_heads=4,
        kv_lora_rank=8,
        qk_rope_head_dim=4,
        v_head_dim=6,
        kv_cache_quantized_dtype=None,
        sparse_mla_tp_op=types.SimpleNamespace(serves_oproj=True),
        W_UK_T=torch.ones((4, 8, 8), dtype=torch.bfloat16),
    )
    qb_in = in_branch_mod.InBranchQbInputs(
        q_c=torch.zeros((24, 16), dtype=torch.bfloat16),
        w_qb=torch.zeros((16, 48), dtype=torch.float8_e4m3fn),
        s_qb=torch.ones((48,), dtype=torch.float32),
        w_uk_t=torch.ones((4, 8, 8), dtype=torch.bfloat16),
        cos_sin=torch.zeros((24, 4), dtype=torch.bfloat16),
    )
    q_dead = (
        torch.zeros((1, 4, 8), dtype=torch.bfloat16),
        torch.zeros((1, 4, 4), dtype=torch.bfloat16),
    )
    sp_o_proj = (
        torch.zeros((24, 32), dtype=torch.float8_e4m3fn),
        torch.ones((32,), dtype=torch.float32),
    )
    kv_c = torch.ones((16, 8), dtype=torch.bfloat16)
    k_pe = torch.ones((16, 1, 4), dtype=torch.bfloat16)
    topk = torch.zeros((64, 12), dtype=torch.int32)
    meta = types.SimpleNamespace()

    # 1) Empty cache run (determine_available_memory): must return ones without
    # invoking in_branch_qb_attention.
    probe_out = impl.forward(
        layer=layer,
        q=q_dead,
        kv_c_normed=kv_c,
        k_pe=k_pe,
        kv_cache=torch.empty(0, dtype=torch.bfloat16),
        attn_metadata=meta,
        topk_indices=topk,
        sp_o_proj=sp_o_proj,
        kv_sharded=True,
        in_branch_qb_inputs=qb_in,
    )
    assert isinstance(probe_out, torch.Tensor)
    assert probe_out.shape == (16, 32)
    assert torch.all(probe_out == 1)
    assert len(called) == 0

    # 2) Live cache: dispatches to in_branch_qb_attention, skips eager bmm/absorb,
    # and propagates in-place KV cache updates.
    nope_cache = torch.zeros((4, 16, 8), dtype=torch.bfloat16)
    rope_cache = torch.zeros((4, 4, 16), dtype=torch.bfloat16)
    out = impl.forward(
        layer=layer,
        q=q_dead,
        kv_c_normed=kv_c,
        k_pe=k_pe,
        kv_cache=(nope_cache, rope_cache),
        attn_metadata=meta,
        topk_indices=topk,
        sp_o_proj=sp_o_proj,
        kv_sharded=True,
        in_branch_qb_inputs=qb_in,
    )
    assert isinstance(out, torch.Tensor)
    assert torch.all(out == 7.0)
    assert torch.all(nope_cache == 3.0)
    assert torch.all(rope_cache == 5.0)
    assert len(called) == 1
    assert called[0]["qb_inputs"] is qb_in
    assert called[0]["kv_c_shape"] == (16, 8)
    assert called[0]["k_pe_shape"] == (16, 4)
    assert called[0]["kv_sharded"] is True

    # 3) Contract guard: missing sp_o_proj must fail fast.
    with pytest.raises(ValueError, match="in_branch_qb_inputs requires"):
        impl.forward(
            layer=layer,
            q=q_dead,
            kv_c_normed=kv_c,
            k_pe=k_pe,
            kv_cache=(nope_cache, rope_cache),
            attn_metadata=meta,
            topk_indices=topk,
            sp_o_proj=None,
            kv_sharded=False,
            in_branch_qb_inputs=qb_in,
        )


def test_should_raise_if_in_branch_qb_op_missing():
    fake_layer = types.SimpleNamespace(sparse_mla_in_branch_qb_op=None)
    qb_in = in_branch_mod.InBranchQbInputs(
        q_c=torch.zeros((8, 16), dtype=torch.bfloat16),
        w_qb=torch.zeros((16, 48), dtype=torch.float8_e4m3fn),
        s_qb=torch.ones((48,), dtype=torch.float32),
        w_uk_t=torch.ones((4, 8, 8), dtype=torch.bfloat16),
        cos_sin=torch.zeros((8, 4), dtype=torch.bfloat16),
    )
    with pytest.raises(RuntimeError, match="sparse_mla_in_branch_qb_op was not built"):
        in_branch_mod.in_branch_qb_attention(
            fake_layer,
            (torch.zeros((1,)), torch.zeros((1,))),
            qb_in,
            torch.zeros((8, 8), dtype=torch.bfloat16),
            torch.zeros((8, 4), dtype=torch.bfloat16),
            torch.zeros((8, 12), dtype=torch.int32),
            types.SimpleNamespace(),
            w_o=(torch.zeros((24, 32)), torch.ones((32,))),
        )


@pytest.mark.parametrize(
    ("skip_topk", "has_indexer", "expect_deferred"),
    [
        (True, True, True),
        (False, False, True),
        (False, True, False),
    ],
)
def test_should_defer_fused_qb_only_on_non_indexer_layers(
    monkeypatch, skip_topk: bool, has_indexer: bool, expect_deferred: bool
):
    wrapper = mla_op.VllmTPUMultiHeadLatentAttentionWrapper.__new__(
        mla_op.VllmTPUMultiHeadLatentAttentionWrapper
    )
    torch.nn.Module.__init__(wrapper)
    wrapper.is_sparse = True
    wrapper.skip_topk = skip_topk

    class _FakeIndexer:
        indexer_op = types.SimpleNamespace(
            topk_op_tp=types.SimpleNamespace(q_own_rows=True),
            has_live_unified_cache=lambda: False,
        )

        def __call__(self, *args, **kwargs):
            del args, kwargs
            return torch.zeros((64, 12), dtype=torch.int32)

    wrapper.indexer = _FakeIndexer() if has_indexer else None
    wrapper.indexer_rope_emb = None
    wrapper.kimi_preprocess = None
    wrapper.q_lora_rank = 16
    wrapper.kv_lora_rank = 8
    wrapper.qk_nope_head_dim = 8
    wrapper.qk_rope_head_dim = 4
    wrapper.qk_head_dim = 12
    wrapper.num_heads = 4
    wrapper.v_head_dim = 6
    wrapper.dcp_group_size = 4
    wrapper.dcp_q_replicate = True
    wrapper.gate_is_fused = False
    wrapper.g_proj = None
    wrapper.rotary_emb = object()
    wrapper.topk_indices_buffer = torch.zeros((64, 12), dtype=torch.int32)
    wrapper.q_a_layernorm = lambda x: x
    wrapper.kv_a_layernorm = lambda x: x
    wrapper.fused_qkv_a_proj = object()
    wrapper.q_b_proj = object()
    wrapper.o_proj = types.SimpleNamespace(
        weight=torch.zeros((24, 32), dtype=torch.float8_e4m3fn),
        weight_scale=torch.ones((32,), dtype=torch.float32),
    )

    captured_mla: list[dict[str, Any]] = []

    class _FakeMlaAttn:
        sparse_mla_tp_op = types.SimpleNamespace(
            q_replicated=True,
            serves_oproj=True,
            rank=torch.zeros((1,), dtype=torch.int32),
        )
        sparse_mla_unified_tp_op = None
        sparse_mla_in_branch_qb_op = types.SimpleNamespace(serves_sharded_kv=True)
        W_UV = torch.zeros((4, 8, 6), dtype=torch.bfloat16)
        W_O_tp_half = None

        def __call__(self, q, kv_c, k_pe, **kw):
            del q, kv_c, k_pe
            captured_mla.append(kw)
            return torch.zeros((16, 32), dtype=torch.bfloat16)

    wrapper.mla_attn = _FakeMlaAttn()

    eager_calls: list[int] = []

    def _fake_eager_qb(q_c, w_qb, s_qb, w_uk_t, cos_sin, split_rows):
        del w_qb, s_qb, w_uk_t, cos_sin
        eager_calls.append(split_rows)
        return (
            torch.zeros((split_rows, 4, 8), dtype=q_c.dtype),
            torch.zeros((q_c.shape[0] - split_rows, 4, 8), dtype=q_c.dtype),
            torch.zeros((q_c.shape[0], 4, 4), dtype=q_c.dtype),
        )

    w_qb = torch.zeros((16, 48), dtype=torch.float8_e4m3fn)
    s_qb = torch.ones((48,), dtype=torch.float32)
    w_uk = torch.ones((4, 8, 8), dtype=torch.bfloat16)
    cos_sin_cache = torch.ones((64, 4), dtype=torch.bfloat16)

    monkeypatch.setattr(mla_op, "_FUSED_QB_ROPE_ABSORB_SPLIT_OP", _fake_eager_qb)
    monkeypatch.setattr(mla_op, "rope_half_split", lambda _r, _p, q_, k_: (q_, k_))
    monkeypatch.setattr(mla_op, "_retain_buffers", lambda x, *_a: x)
    monkeypatch.setattr(tp_op_mod, "applies", lambda n: n >= 64)
    monkeypatch.setattr(tp_op_mod, "serves_sequence_row_oproj", lambda _o: True)
    monkeypatch.setattr(
        tp_op_mod, "q_rows", lambda _op, _n: torch.arange(24, dtype=torch.long)
    )
    monkeypatch.setattr(
        wrapper,
        "_fused_qb_rope_absorb_operands",
        lambda *_a, **_kw: (w_qb, s_qb, w_uk, 8, cos_sin_cache),
    )

    prefill_group = types.SimpleNamespace(
        project_gather=lambda _w, _h, _n: (
            (
                torch.zeros((24, 16), dtype=torch.bfloat16),
                torch.zeros((16, 12), dtype=torch.bfloat16),
                torch.arange(16, dtype=torch.long),
            ),
            None,
        )
    )
    positions = torch.arange(64, dtype=torch.long)
    hidden = torch.zeros((16, 32), dtype=torch.bfloat16)

    out = wrapper.forward(positions, hidden, prefill_group=prefill_group)
    assert out.shape == (16, 32)
    assert len(captured_mla) == 1
    if expect_deferred:
        assert captured_mla[0]["in_branch_qb_inputs"] is not None
        assert captured_mla[0]["q_dcp_replicated"] is None
        assert len(eager_calls) == 0
    else:
        assert captured_mla[0]["in_branch_qb_inputs"] is None
        assert captured_mla[0]["q_dcp_replicated"] is not None
        assert eager_calls == [8]
