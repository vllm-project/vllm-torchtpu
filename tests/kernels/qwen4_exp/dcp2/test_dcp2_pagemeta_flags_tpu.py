# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_RPA_SCHEDULE_HBM and VLLM_TPU_QSA_DCP_MERGE_SCORES on TPU
(DCP2_HARNESS_DEVICE=tpu; skipped elsewhere).

The compressed-DCP attend step on every rank of a DCP group of 4, on one set of
select-step outputs, with each flag alone and both, against both off:

- the local merge (VLLM_TPU_QSA_COMPRESSED_DCP_LOCAL_MERGE, the layers whose
  selection nobody reads) and the global merge (the layer the MTP draft
  reads): the same pool bytes, attention output, LSE and ``selected``, bit for
  bit;
- two decode envs: ``union`` routes the verify rows through the union RPA
  (VLLM_TPU_QSA_DECODE_RPA_UNION, the served path), ``rows`` through the
  per-row RPA; the compiled attend step holds the union metadata kernel exactly
  in the ``union`` env.

VLLM_TPU_RPA_SCHEDULE_HBM moves the RPA schedule arrays' buffers and
VLLM_TPU_QSA_DCP_MERGE_SCORES reads the merge boundary from the top-k's
returned bits; neither changes a value. Inputs as test_dcp2_local_merge_tpu.py.

Run it in the served decode env, e.g. through the bench recipe:
``BENCH=script "BENCH_CMD=env DCP2_HARNESS_DEVICE=tpu python -m pytest -q -s
tests/kernels/qwen4_exp/dcp2/test_dcp2_pagemeta_flags_tpu.py"``.
"""

import functools

import dcp2_harness
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import test_dcp2_op
from dcp2_harness import (
    ON_TPU,
    from_raw,
    poison_tail_rows,
    project_rank_pool,
    rank_operand,
    raw,
)
from test_dcp2_local_merge_tpu import (
    CONTEXTS,
    DCP,
    K_SCALE,
    QUERY_LENS,
    RATIO,
    SERVED_DECODE_ENV,
    TOPK,
    V_SCALE,
    _bits,
    _zero_most_keys,
)
from test_dcp2_op import ROTARY, _args, make_step

from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa

pytestmark = pytest.mark.skipif(not ON_TPU, reason="DCP2_HARNESS_DEVICE=tpu only")

UNION_ENV = {
    "VLLM_TPU_QSA_DECODE_RPA_UNION": "640",
    "VLLM_TPU_QSA_DECODE_RPA_BKV": "512",
    "VLLM_TPU_QSA_DECODE_RPA_LANES": "4",
    "VLLM_TPU_QSA_DECODE_COMPACT_SCHEDULE": "1",
    "VLLM_TPU_QSA_DECODE_SLOT_READ": "1",
    "VLLM_TPU_QSA_DECODE_LEAN_FETCH": "1",
    "VLLM_TPU_QSA_DECODE_FAST_FETCH": "1",
}
ENVS = {"union": UNION_ENV, "rows": {"VLLM_TPU_QSA_DECODE_RPA_UNION": "0"}}
FLAGS = {  # VLLM_TPU_RPA_SCHEDULE_HBM, VLLM_TPU_QSA_DCP_MERGE_SCORES
    "off": ("0", "0"),
    "schedule": ("1", "0"),
    "scores": ("0", "1"),
    "both": ("1", "1"),
}
VARIANTS = ("random", "zero_keys", "zero_query")


@pytest.mark.parametrize("env", sorted(ENVS))
@pytest.mark.parametrize("variant", VARIANTS)
def test_tpu_pagemeta_flags_attend_the_same(monkeypatch, variant, env):
    monkeypatch.setattr(dcp2_harness, "D", DCP)
    monkeypatch.setattr(test_dcp2_op, "D", DCP)
    for name, value in {**SERVED_DECODE_ENV, **ENVS[env]}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("VLLM_TPU_QSA_COMPRESSED_DCP_LOCAL_MERGE", "1")
    layout, full, keep, qgate, s, m, seq, starts = make_step(13, QUERY_LENS, CONTEXTS)
    if variant == "zero_keys":
        full = _zero_most_keys(full, m["compressed_table"], 5)
    if variant == "zero_query":
        index_qk = s["index_qk"]
        for seat in (1, 4, 11, 15):
            index_qk = index_qk.at[int(starts[seat]) : int(starts[seat + 1])].set(0)
        s = {**s, "index_qk": index_qk}
    pools = [
        poison_tail_rows(
            project_rank_pool(raw(full), layout, r, keep_pages=keep), layout, r, seq
        )
        for r in range(DCP)
    ]
    pair_qgate = jnp.concatenate(qgate, axis=1)
    opts = dict(
        compress_ratio=RATIO,
        token_topk=TOPK,
        norm_eps=1e-6,
        rotary_dim=ROTARY,
        k_scale=K_SCALE,
        v_scale=V_SCALE,
        main_slot=0,
        max_query_len=4,
    )
    att = dict(
        kv_heads=1,
        compress_ratio=RATIO,
        token_topk=TOPK,
        k_scale=K_SCALE,
        v_scale=V_SCALE,
        main_slot=0,
        max_query_len=4,
        dcp_size=DCP,
    )
    select = jax.jit(
        functools.partial(
            pooled_qsa.pooled_qsa_compressed_dcp_select_step, dcp_size=DCP, **opts
        )
    )
    selects = [
        select(
            *_args(pair_qgate, s, from_raw(pools[r]), m, layout.dcp_table),
            rank_operand(r),
        )
        for r in range(DCP)
    ]
    vals_pair = jnp.stack([x[6] for x in selects])
    pos_pair = jnp.stack([x[7] for x in selects])

    def attend(flags, local_merge):
        monkeypatch.setenv("VLLM_TPU_RPA_SCHEDULE_HBM", FLAGS[flags][0])
        monkeypatch.setenv("VLLM_TPU_QSA_DCP_MERGE_SCORES", FLAGS[flags][1])
        step = jax.jit(
            functools.partial(
                pooled_qsa.pooled_qsa_compressed_dcp_attend_step,
                local_merge=local_merge,
                **att,
            )
        )
        out, text = [], None
        for r in range(DCP):
            pool, q, gate, lengths, req, st = selects[r][:6]
            operands = (
                q,
                gate,
                lengths,
                req,
                st,
                jnp.asarray(layout.dcp_table),
                m["context"],
                pool,
                vals_pair,
                pos_pair,
                rank_operand(r),
            )
            if r == 0:
                text = step.lower(*operands).compile().as_text()
            out.append([np.asarray(x) for x in step(*operands)])
        return out, text

    failures, report = [], {}
    for local_merge in (True, False):
        merge = "local" if local_merge else "global"
        base, text = attend("off", local_merge)
        union = "qsa_union_meta" in text
        report[f"{merge} union kernel"] = union
        if union != (env == "union"):
            failures.append(
                f"{merge}: union metadata kernel compiled {union} in env {env}"
            )
        for flags in ("schedule", "scores", "both"):
            got, _ = attend(flags, local_merge)
            for r in range(DCP):
                for i, name in ((0, "pool"), (1, "out"), (2, "lse"), (3, "selected")):
                    same = np.array_equal(_bits(got[r][i]), _bits(base[r][i]))
                    if not same:
                        failures.append(f"{merge} {flags} rank {r}: {name} differs")
            report[f"{merge} {flags}"] = (
                "bit-equal"
                if not any(f.startswith(f"{merge} {flags} ") for f in failures)
                else "differs"
            )
        for r in range(DCP):
            if np.isnan(base[r][1][: int(starts[-1])]).any():
                failures.append(f"{merge} rank {r}: NaN in the flag-off output")
    print(variant, env, report)
    assert not failures, failures
