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
"""`TorchStepMetaOps`: the library's step-metadata packing (`pack_step_meta`)
over torch tensors matches the JAX ops and a plain-torch reference bit for bit,
and traces under Dynamo `fullgraph=True`."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.megakernel.adapter.step_meta_ops import TorchStepMetaOps
from vllm_torchtpu.megakernel.core.runtime import JaxStepMetaOps, pack_step_meta

_TORCH_OPS = TorchStepMetaOps()


def _pack_torch(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    block_tables: torch.Tensor,
    s0: torch.Tensor,
    c0: torch.Tensor,
    r0: torch.Tensor,
    s1: torch.Tensor,
    c1: torch.Tensor,
    r1: torch.Tensor,
    s2: torch.Tensor,
    c2: torch.Tensor,
    r2: torch.Tensor,
    bucket_t: int,
    tokens_per_seq: int,
    max_pages_per_seq: int,
    has_roff: bool,
    use_ckpt_2d: bool,
) -> tuple[torch.Tensor, ...]:
    """`pack_step_meta` over `TorchStepMetaOps` for three recurrent groups."""
    return pack_step_meta(
        input_ids,
        positions,
        block_tables,
        ((s0, c0, r0), (s1, c1, r1), (s2, c2, r2)),
        bucket_t=bucket_t,
        tokens_per_seq=tokens_per_seq,
        max_pages_per_seq=max_pages_per_seq,
        has_roff=has_roff,
        use_ckpt_2d=use_ckpt_2d,
        ops=_TORCH_OPS,
    ).as_tuple()


def _ref_pack(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    block_tables: torch.Tensor,
    s0: torch.Tensor,
    c0: torch.Tensor,
    r0: torch.Tensor,
    s1: torch.Tensor,
    c1: torch.Tensor,
    r1: torch.Tensor,
    s2: torch.Tensor,
    c2: torch.Tensor,
    r2: torch.Tensor,
    bucket_t: int,
    tokens_per_seq: int,
    max_pages_per_seq: int,
    has_roff: bool,
    use_ckpt_2d: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """A plain-torch reference of the step-metadata packing, for parity tests."""

    def _sp_1d(x: torch.Tensor, target_len: int) -> torch.Tensor:
        x1 = x.reshape(-1)
        n = int(x1.shape[0])
        if n >= target_len:
            return x1[:target_len]
        return torch.nn.functional.pad(x1, (0, target_len - n), value=0)

    def _sp_2d(x: torch.Tensor, target_rows: int, target_cols: int) -> torch.Tensor:
        r, c = int(x.shape[0]), int(x.shape[1])
        out = x[: min(r, target_rows), : min(c, target_cols)]
        pad_r = max(0, target_rows - r)
        pad_c = max(0, target_cols - c)
        if pad_r > 0 or pad_c > 0:
            out = torch.nn.functional.pad(out, (0, pad_c, 0, pad_r), value=0)
        return out

    T = tokens_per_seq
    B = bucket_t
    nseq_bkt = B // T

    bt = block_tables
    if bt.ndim == 1:
        n_rows_bt = int(s0.shape[0])
        if n_rows_bt > 0 and bt.numel() % n_rows_bt == 0:
            bt = bt.reshape(n_rows_bt, bt.numel() // n_rows_bt)
        elif max_pages_per_seq > 0 and bt.numel() % max_pages_per_seq == 0:
            bt = bt.reshape(-1, max_pages_per_seq)
        else:
            raise ValueError("invalid 1-D block_tables")
    pt_b = _sp_2d(bt, nseq_bkt, max_pages_per_seq).to(torch.int32)

    ckpt_rows: list[torch.Tensor] = []
    roff_rows: list[torch.Tensor] = []
    for s_raw, c_raw, r_raw in ((s0, c0, r0), (s1, c1, r1), (s2, c2, r2)):
        s_b = (
            _sp_1d(s_raw[:, 0], nseq_bkt).to(torch.int32)
            if s_raw.ndim == 2
            else _sp_1d(s_raw, nseq_bkt).to(torch.int32)
        )
        if has_roff:
            s_idx = s_b.to(torch.long).clamp(min=0, max=r_raw.numel() - 1)
            roff_g = r_raw.reshape(-1)[s_idx].to(torch.int32)
        else:
            roff_g = torch.zeros_like(s_b)

        if T > 1:
            if use_ckpt_2d and c_raw.ndim == 2 and c_raw.shape[1] >= T:
                ckpt_g = _sp_2d(c_raw, nseq_bkt, T).to(torch.int32)
            else:
                src_1d = c_raw[:, 0] if c_raw.ndim == 2 else c_raw.reshape(-1)
                ckpt_1d = _sp_1d(src_1d, nseq_bkt).to(torch.int32)
                ckpt_g = ckpt_1d.unsqueeze(1).expand(nseq_bkt, T).contiguous()
                roff_g = torch.where(roff_g < 0, roff_g, torch.zeros_like(roff_g))
        else:
            if use_ckpt_2d and has_roff and c_raw.ndim == 2 and c_raw.shape[1] > 1:
                c_b = _sp_2d(c_raw, nseq_bkt, int(c_raw.shape[1])).to(torch.int32)
                r_idx = roff_g.to(torch.long).clamp(min=0, max=c_raw.shape[1] - 1)
                read_col = c_b.gather(1, r_idx.unsqueeze(1))
            else:
                read_col = s_b.unsqueeze(1)
            write_col = s_b.unsqueeze(1)
            ckpt_g = torch.cat([read_col, write_col], dim=1)
            roff_g = torch.where(roff_g < 0, roff_g, torch.zeros_like(roff_g))

        active_g = (s_b > 0) & (ckpt_g[:, 0] > 0)
        ckpt_g = torch.where(active_g.unsqueeze(1), ckpt_g, torch.zeros_like(ckpt_g))
        roff_g = torch.where(active_g, roff_g, torch.zeros_like(roff_g))
        ckpt_rows.append(ckpt_g)
        roff_rows.append(roff_g)

    ckpt_b = torch.stack(ckpt_rows, dim=0)
    roff_b = torch.stack(roff_rows, dim=0)
    active_seq = ckpt_b[0, :, 0] > 0
    active_tok = active_seq.unsqueeze(1).expand(nseq_bkt, T).reshape(B)
    zero_b = torch.zeros((B,), dtype=torch.int32, device=input_ids.device)
    ids_b = torch.where(active_tok, _sp_1d(input_ids, B).to(torch.int32), zero_b)
    pos_1d = positions[0] if positions.ndim == 2 else positions
    pos_b = torch.where(active_tok, _sp_1d(pos_1d, B).to(torch.int32), zero_b)
    return ids_b, pos_b, pt_b, ckpt_b, roff_b


def test_torch_and_jax_step_meta_ops_pack_bitwise_like_the_reference() -> None:
    """`pack_step_meta` over `TorchStepMetaOps` and over `JaxStepMetaOps` is
    bitwise equal to a plain-torch reference across decode / verify buckets,
    1-D and M-RoPE positions, 1-D and 2-D block tables, read offsets and 1-D /
    2-D checkpoint indices; the torch path traces with Dynamo `fullgraph=True`."""
    jax_ops = JaxStepMetaOps()
    max_pages_per_seq = 4
    num_req_rows = 6

    s0 = torch.tensor([1, 0, 5, 8, 0, 12], dtype=torch.int32)
    s1 = torch.tensor([2, 0, 6, 9, 0, 13], dtype=torch.int32)
    s2 = torch.tensor([3, 0, 7, 10, 0, 14], dtype=torch.int32)
    c0_2d = torch.tensor(
        [
            [1, 2, 3, 4],
            [0, 0, 0, 0],
            [5, 6, 7, 8],
            [0, 9, 10, 11],
            [15, 16, 17, 18],
            [12, 13, 14, 15],
        ],
        dtype=torch.int32,
    )
    c1_2d = c0_2d + 20
    c1_2d[1] = 0
    c1_2d[3, 0] = 0
    c2_2d = c0_2d + 40
    c2_2d[1] = 0
    c2_2d[3, 0] = 0

    roff_pos = torch.zeros((32,), dtype=torch.int32)
    roff_pos[1] = 2
    roff_pos[5] = 1
    roff_pos[8] = 3
    roff_pos[12] = 0

    roff_neg = roff_pos.clone()
    roff_neg[1] = -1
    roff_neg[5] = 2
    roff_neg[8] = -2
    roff_neg[12] = -1

    bt_2d = torch.arange(
        10, 10 + num_req_rows * max_pages_per_seq, dtype=torch.int32
    ).reshape(num_req_rows, max_pages_per_seq)
    bt_1d_req_blocks = torch.arange(50, 50 + num_req_rows * 2, dtype=torch.int32)
    bt_1d_max_pages = torch.arange(80, 80 + 5 * max_pages_per_seq, dtype=torch.int32)
    s0_j, s1_j, s2_j = (
        jnp.asarray(s0.numpy()),
        jnp.asarray(s1.numpy()),
        jnp.asarray(s2.numpy()),
    )
    c0_2d_j, c1_2d_j, c2_2d_j = (
        jnp.asarray(c0_2d.numpy()),
        jnp.asarray(c1_2d.numpy()),
        jnp.asarray(c2_2d.numpy()),
    )
    r_zero = torch.zeros((1,), dtype=torch.int32)
    r_zero_j = jnp.zeros((1,), dtype=jnp.int32)
    roff_pos_j = jnp.asarray(roff_pos.numpy())
    roff_neg_j = jnp.asarray(roff_neg.numpy())
    bt_pairs = (
        (bt_2d, jnp.asarray(bt_2d.numpy())),
        (bt_1d_req_blocks, jnp.asarray(bt_1d_req_blocks.numpy())),
        (bt_1d_max_pages, jnp.asarray(bt_1d_max_pages.numpy())),
    )

    for T in (1, 4):
        for bucket_t in (4, 8, 16):
            raw_tok_len = 12
            ids_in = torch.arange(100, 100 + raw_tok_len, dtype=torch.int32)
            ids_j = jnp.asarray(ids_in.numpy())
            pos_1d = torch.arange(200, 200 + raw_tok_len, dtype=torch.int32)
            pos_mrope = torch.stack([pos_1d, pos_1d, pos_1d], dim=0)
            for pos_in, pos_j in (
                (pos_1d, jnp.asarray(pos_1d.numpy())),
                (pos_mrope, jnp.asarray(pos_mrope.numpy())),
            ):
                for bt_in, bt_j in bt_pairs:
                    for has_roff, r_in, r_j in (
                        (False, r_zero, r_zero_j),
                        (True, roff_pos, roff_pos_j),
                        (True, roff_neg, roff_neg_j),
                    ):
                        for use_ckpt_2d, c_set, c_set_j in (
                            (False, (s0, s1, s2), (s0_j, s1_j, s2_j)),
                            (True, (c0_2d, c1_2d, c2_2d), (c0_2d_j, c1_2d_j, c2_2d_j)),
                        ):
                            c0_in, c1_in, c2_in = c_set
                            c0_j, c1_j, c2_j = c_set_j
                            if T > 1 and not use_ckpt_2d:
                                with pytest.raises(
                                    ValueError,
                                    match="requires 2-D ckpt_indices with >= T=",
                                ):
                                    _pack_torch(
                                        ids_in,
                                        pos_in,
                                        bt_in,
                                        s0,
                                        c0_in,
                                        r_in,
                                        s1,
                                        c1_in,
                                        r_in,
                                        s2,
                                        c2_in,
                                        r_in,
                                        bucket_t,
                                        T,
                                        max_pages_per_seq,
                                        has_roff,
                                        use_ckpt_2d,
                                    )
                                continue
                            ref_out = _ref_pack(
                                ids_in,
                                pos_in,
                                bt_in,
                                s0,
                                c0_in,
                                r_in,
                                s1,
                                c1_in,
                                r_in,
                                s2,
                                c2_in,
                                r_in,
                                bucket_t,
                                T,
                                max_pages_per_seq,
                                has_roff,
                                use_ckpt_2d,
                            )
                            head_out = _pack_torch(
                                ids_in,
                                pos_in,
                                bt_in,
                                s0,
                                c0_in,
                                r_in,
                                s1,
                                c1_in,
                                r_in,
                                s2,
                                c2_in,
                                r_in,
                                bucket_t,
                                T,
                                max_pages_per_seq,
                                has_roff,
                                use_ckpt_2d,
                            )
                            for r_t, h_t in zip(ref_out, head_out, strict=True):
                                assert torch.equal(r_t, h_t)
                            if bucket_t == 8:
                                meta_jax = pack_step_meta(
                                    ids_j,
                                    pos_j,
                                    bt_j,
                                    (
                                        (s0_j, c0_j, r_j),
                                        (s1_j, c1_j, r_j),
                                        (s2_j, c2_j, r_j),
                                    ),
                                    bucket_t=bucket_t,
                                    tokens_per_seq=T,
                                    max_pages_per_seq=max_pages_per_seq,
                                    has_roff=has_roff,
                                    use_ckpt_2d=use_ckpt_2d,
                                    ops=jax_ops,
                                ).as_tuple()
                                for r_t, m_j in zip(ref_out, meta_jax, strict=True):
                                    assert np.array_equal(r_t.numpy(), np.asarray(m_j))

    # Verify Dynamo traces _pack_torch with fullgraph=True, dynamic=False
    compiled_fn = torch.compile(
        _pack_torch, backend="eager", fullgraph=True, dynamic=False
    )
    comp_out = compiled_fn(
        torch.arange(10, 18, dtype=torch.int32),
        torch.arange(20, 28, dtype=torch.int32),
        bt_2d,
        s0,
        c0_2d,
        roff_neg,
        s1,
        c1_2d,
        roff_neg,
        s2,
        c2_2d,
        roff_neg,
        8,
        4,
        max_pages_per_seq,
        True,
        True,
    )
    eager_out = _ref_pack(
        torch.arange(10, 18, dtype=torch.int32),
        torch.arange(20, 28, dtype=torch.int32),
        bt_2d,
        s0,
        c0_2d,
        roff_neg,
        s1,
        c1_2d,
        roff_neg,
        s2,
        c2_2d,
        roff_neg,
        8,
        4,
        max_pages_per_seq,
        True,
        True,
    )
    for c_t, e_t in zip(comp_out, eager_out, strict=True):
        assert torch.equal(c_t, e_t)

    # Extra edge cases: int64 inputs, PAD_SLOT_ID = -1 in s/c, slot beyond
    # r_raw.numel()-1, and 2-D ckpt narrower than T (shape=(6, 2) at T=4).
    s_edge = torch.tensor([1, -1, 999, 8, -1, 12], dtype=torch.int64)
    c_narrow = torch.tensor(
        [[1, 2], [-1, -1], [5, 6], [-1, 9], [15, 16], [12, 13]], dtype=torch.int64
    )
    r_edge = roff_neg.to(torch.int64)
    ids_64 = torch.arange(10, 22, dtype=torch.int64)
    pos_64 = torch.arange(30, 42, dtype=torch.int64)
    bt_64 = bt_2d.to(torch.int64)
    for T_edge, bkt_edge in ((1, 4), (4, 8)):
        if T_edge > 1:
            with pytest.raises(
                ValueError, match="requires 2-D ckpt_indices with >= T="
            ):
                _pack_torch(
                    ids_64,
                    pos_64,
                    bt_64,
                    s_edge,
                    c_narrow,
                    r_edge,
                    s_edge,
                    c_narrow,
                    r_edge,
                    s_edge,
                    c_narrow,
                    r_edge,
                    bkt_edge,
                    T_edge,
                    max_pages_per_seq,
                    True,
                    True,
                )
            continue
        ref_edge = _ref_pack(
            ids_64,
            pos_64,
            bt_64,
            s_edge,
            c_narrow,
            r_edge,
            s_edge,
            c_narrow,
            r_edge,
            s_edge,
            c_narrow,
            r_edge,
            bkt_edge,
            T_edge,
            max_pages_per_seq,
            True,
            True,
        )
        head_edge = _pack_torch(
            ids_64,
            pos_64,
            bt_64,
            s_edge,
            c_narrow,
            r_edge,
            s_edge,
            c_narrow,
            r_edge,
            s_edge,
            c_narrow,
            r_edge,
            bkt_edge,
            T_edge,
            max_pages_per_seq,
            True,
            True,
        )
        jax_edge = pack_step_meta(
            jnp.asarray(ids_64.numpy()),
            jnp.asarray(pos_64.numpy()),
            jnp.asarray(bt_64.numpy()),
            tuple(
                (
                    jnp.asarray(s_edge.numpy()),
                    jnp.asarray(c_narrow.numpy()),
                    jnp.asarray(r_edge.numpy()),
                )
                for _ in range(3)
            ),
            bucket_t=bkt_edge,
            tokens_per_seq=T_edge,
            max_pages_per_seq=max_pages_per_seq,
            has_roff=True,
            use_ckpt_2d=True,
            ops=jax_ops,
        ).as_tuple()
        for r_t, h_t, j_t in zip(ref_edge, head_edge, jax_edge, strict=True):
            assert r_t.dtype == torch.int32 and h_t.dtype == torch.int32
            assert torch.equal(r_t, h_t)
            assert np.array_equal(r_t.numpy(), np.asarray(j_t))
