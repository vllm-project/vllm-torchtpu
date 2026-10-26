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
"""streamindex_topk over token windows (TPU_DSV4_INDEXER_TOKEN_SPLIT_ABOVE):
each of tp emulated ranks scores only its contiguous T / tp rows with
`token_window_metadata` + `row_window`; the concatenated windows must equal
the whole-batch top-k (same indices, same order), at the c16 served shapes
(T512, TP4: verify windows + a ~495-token chunk, all chunk-start alignments
mod 4, several chunks). Also times one core: whole batch vs one window.

    python3 tests/kernels/test_dsv4_indexer_token_split.py [--time]
"""

import argparse
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.kernels.test_dsv4_indexer_kv_split import (
    K,
    RATIO,
    VQ,
    _case,
    _kw,
)
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    streamindex_topk,
    token_window_metadata,
)

pytestmark = [pytest.mark.tpu_test]

TP = 4


@jax.jit
def _whole(inp):
    return streamindex_topk(**inp, **_kw())


def _window(inp, rank, tp, extra=()):
    T = inp["q"].shape[0]
    w = T // tp
    off = (rank * w).astype(jnp.int32)
    seq_w, pages_w, cu_w, dist_w = token_window_metadata(
        inp["seq_lens"], inp["page_indices"], inp["cu_q_lens"], inp["distribution"],
        off, window=w, compression_ratio=RATIO,
    )  # fmt: skip
    return streamindex_topk(
        q=jax.lax.dynamic_slice_in_dim(inp["q"], off, w, 0),
        indexer_weights=jax.lax.dynamic_slice_in_dim(inp["indexer_weights"], off, w, 0),
        cache_kv=inp["cache_kv"],
        seq_lens=seq_w,
        page_indices=pages_w,
        cu_q_lens=cu_w,
        distribution=dist_w,
        row_window=(inp["seq_lens"], inp["cu_q_lens"], inp["distribution"], off),
        row_window_tokens=T,
        **{**_kw(), **dict(extra)},
    )


_window_jit = jax.jit(_window, static_argnums=(2, 3))

CASES = {
    # c16 TP4 mnbt 512: 2 verify windows + a ~495-token chunk at 31.7k / 127k;
    # the chunk start moves through every alignment mod 4 (window cuts at rows
    # 128 / 256 / 384 land on every position mod 4).
    "v2_c495@32k": ([(VQ, 40_007, "verify"), (VQ, 90_007, "verify"),
                     (495, 32_263, "chunk")], 512),
    "v2_c495@32k+1": ([(VQ, 40_007, "verify"), (VQ, 90_007, "verify"),
                       (495, 32_264, "chunk")], 512),
    "v2_c495@32k+2": ([(VQ, 40_007, "verify"), (VQ, 90_007, "verify"),
                       (495, 32_265, "chunk")], 512),
    "v2_c495@32k+3": ([(VQ, 40_007, "verify"), (VQ, 90_007, "verify"),
                       (495, 32_266, "chunk")], 512),
    "v2_c495@127k": ([(VQ, 40_007, "verify"), (VQ, 90_007, "verify"),
                      (495, 127_495, "chunk")], 512),
    "v1_c500@100k": ([(VQ, 90_007, "verify"), (500, 100_852, "chunk")], 512),
    # two chunks, the second starting inside rank 1's window
    "v1_c150_c300": ([(VQ, 60_007, "verify"), (150, 22_150, "chunk"),
                      (300, 51_300, "chunk")], 512),
    # a short fresh chunk (row lengths < k -> early-exit rows) + padding rows
    "c200@fresh": ([(200, 200, "chunk")], 512),
    # a window boundary exactly at a sequence boundary (rows 0..127 | 128..)
    "v1_c121_c384": ([(VQ, 30_007, "verify"), (121, 9_121, "chunk"),
                      (384, 70_384, "chunk")], 512),
}


def run_case(name, time_it=False, iters=20, extra=()):
    seqs, T = CASES[name]
    inp = _case(seqs, T)
    ref = np.asarray(_whole(inp))
    got = np.concatenate(
        [np.asarray(_window_jit(inp, jnp.int32(r), TP, extra)) for r in range(TP)]
    )
    n_tok = int(np.asarray(inp["cu_q_lens"])[len(seqs)])
    bad_rows = np.any(ref[:n_tok] != got[:n_tok], axis=1)
    res = {"case": name, "rows": n_tok, "mismatch_rows": int(bad_rows.sum())}
    if bad_rows.any():
        r0 = int(np.argmax(bad_rows))
        res.update(first_bad_row=r0, ref=ref[r0, :8].tolist(), got=got[r0, :8].tolist(),
                   same_set=bool(set(ref[r0].tolist()) == set(got[r0].tolist())),
                   ref_valid=int((ref[r0] >= 0).sum()), got_valid=int((got[r0] >= 0).sum()))
    if time_it:
        def t(f):
            f().block_until_ready()
            xs = []
            for _ in range(3):
                t0 = time.perf_counter()
                outs = [f() for _ in range(iters)]
                outs[-1].block_until_ready()
                xs.append((time.perf_counter() - t0) * 1e3 / iters)
            return round(min(xs), 3)

        res["whole_ms"] = t(lambda: _whole(inp))
        for r in range(TP):
            res[f"win{r}_ms"] = t(lambda r=r: _window_jit(inp, jnp.int32(r), TP))
    print(res, flush=True)
    return res


@pytest.mark.parametrize("name", list(CASES))
def test_token_windows_match_whole(name):
    assert run_case(name)["mismatch_rows"] == 0


# K-IDXM: the same windows cut into pipelined row pieces (annotated or not), and
# the group-max SparseCore prefilter: identical rows, identical column order.
VARIANTS = {
    "pieces4": (("chunk_tokens", 512 // TP // 4),),
    "pieces4_ng": (("chunk_tokens", 512 // TP // 4), ("chunk_sched_groups", False)),
    "group_max": (("group_max", True),),
    "group_max_pieces4": (("group_max", True), ("chunk_tokens", 512 // TP // 4),
                          ("chunk_sched_groups", False)),
}


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("name", list(CASES))
def test_token_windows_variants_match_whole(name, variant):
    assert run_case(name, extra=VARIANTS[variant])["mismatch_rows"] == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--cases", default=",".join(CASES))
    a = ap.parse_args()
    bad = 0
    for c in a.cases.split(","):
        try:
            bad += run_case(c, time_it=a.time)["mismatch_rows"] > 0
        except Exception as e:  # noqa
            print({"case": c, "error": repr(e)[:3000]}, flush=True)
            bad += 1
    sys.exit(1 if bad else 0)
