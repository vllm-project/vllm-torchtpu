# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_QSA_UNION_LEAN_GLUE, VLLM_TPU_QSA_UNION_CHAINED_CALLS and
VLLM_TPU_QSA_INDEXER_META_SELECT on TPU (DCP2_HARNESS_DEVICE=tpu; skipped
elsewhere).

On every rank of a DCP group of 4, in the served union decode env:

- the compressed-DCP select step with VLLM_TPU_QSA_INDEXER_META_SELECT against
  without: all eight outputs equal in bits;
- the attend step with VLLM_TPU_QSA_UNION_LEAN_GLUE, with
  VLLM_TPU_QSA_UNION_CHAINED_CALLS and with both, against neither, for the
  local merge and the global merge: the same pool bytes, attention output, LSE
  and ``selected``, bit for bit, and no NaN in the flag-off output's rows;
- three steps: 64 rows in 16 request slots (one call over the step's rows, a
  request with no rows, rows past the last request), 16 rows in 32 slots (the
  served 16-row shape, 28 slots with no rows), and 192 rows in 48 slots (two
  calls: sliced, or chained over the step's rows); random keys and mostly zero
  compressed keys.

The compiled attend step holds the union metadata kernel in every case, so the
union path is the one compared. The report line prints, from the compiled
programs, the f32 query buffers handed to the union RPA, the selection
relayouts and the indexer's VMEM-to-HBM s32[1024] copies, flag off and on.

Run it in the served decode env, e.g. through the bench recipe:
``BENCH=script "BENCH_CMD=env DCP2_HARNESS_DEVICE=tpu python -m pytest -q -s
tests/kernels/qwen4_exp/dcp2/test_dcp2_attn_glue_flags_tpu.py"``.
"""

import functools
import re

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
from test_dcp2_pagemeta_flags_tpu import UNION_ENV

from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa

pytestmark = pytest.mark.skipif(not ON_TPU, reason="DCP2_HARNESS_DEVICE=tpu only")

STEPS = {  # query lens, contexts
    "rows64": (QUERY_LENS, CONTEXTS),
    "rows16": ([4] * 4 + [0] * 28, CONTEXTS[1:5] + [37 + 50 * i for i in range(28)]),
    "rows192": ([4] * 48, [37 + 331 * i for i in range(48)]),
}
VARIANTS = ("random", "zero_keys")
ATTEND_FLAGS = {  # VLLM_TPU_QSA_UNION_LEAN_GLUE, VLLM_TPU_QSA_UNION_CHAINED_CALLS
    "off": ("0", "0"),
    "lean": ("1", "0"),
    "chained": ("0", "1"),
    "both": ("1", "1"),
}


def _glue(text):
    """Counts in a compiled program: f32 query buffers of 128 rows given to
    an RPA, selection relayouts, s32[1024] VMEM-to-HBM copies."""
    return dict(
        q128=len(re.findall(r"= f32\[1,128,12,1,256\]\S* (?:copy|fusion)\(", text)),
        sel_relayout=len(
            re.findall(r"= s32\[\d+,5(?:,1)?,128\]\S* (?:copy|reshape|fusion)\(", text)
        ),
        meta_copies=len(
            re.findall(
                r"\(s32\[1024\]\{0:T\(1024\)\}, s32\[1024\]\{0:T\(1024\)S\(1\)\}", text
            )
        ),
    )


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("step_name", sorted(STEPS))
def test_tpu_attn_glue_flags_the_same(monkeypatch, step_name, variant):
    monkeypatch.setattr(dcp2_harness, "D", DCP)
    monkeypatch.setattr(test_dcp2_op, "D", DCP)
    for name, value in {**SERVED_DECODE_ENV, **UNION_ENV}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("VLLM_TPU_QSA_COMPRESSED_DCP_LOCAL_MERGE", "1")
    query_lens, contexts = STEPS[step_name]
    layout, full, keep, qgate, s, m, seq, starts = make_step(17, query_lens, contexts)
    if variant == "zero_keys":
        full = _zero_most_keys(full, m["compressed_table"], 5)
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
    failures, report = [], {}

    def select_all(flag):
        monkeypatch.setenv("VLLM_TPU_QSA_INDEXER_META_SELECT", flag)
        step = jax.jit(
            functools.partial(
                pooled_qsa.pooled_qsa_compressed_dcp_select_step, dcp_size=DCP, **opts
            )
        )
        out, text = [], None
        for r in range(DCP):
            operands = (
                *_args(pair_qgate, s, from_raw(pools[r]), m, layout.dcp_table),
                rank_operand(r),
            )
            if r == 0:
                text = step.lower(*operands).compile().as_text()
            out.append(step(*operands))
        return out, text

    selects, sel_text = select_all("0")
    got, sel_text_on = select_all("1")
    for r in range(DCP):
        for i, (a, b) in enumerate(zip(selects[r], got[r])):
            if not np.array_equal(_bits(np.asarray(a)), _bits(np.asarray(b))):
                failures.append(f"select rank {r}: output {i} differs")
    report["select meta copies off/on"] = (
        _glue(sel_text)["meta_copies"],
        _glue(sel_text_on)["meta_copies"],
    )
    vals_pair = jnp.stack([x[6] for x in selects])
    pos_pair = jnp.stack([x[7] for x in selects])

    def attend(flags, local_merge):
        monkeypatch.setenv("VLLM_TPU_QSA_UNION_LEAN_GLUE", ATTEND_FLAGS[flags][0])
        monkeypatch.setenv("VLLM_TPU_QSA_UNION_CHAINED_CALLS", ATTEND_FLAGS[flags][1])
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

    for local_merge in (True, False):
        merge = "local" if local_merge else "global"
        base, text = attend("off", local_merge)
        if "qsa_union_meta" not in text:
            failures.append(f"{merge}: the union metadata kernel is not compiled")
        for r in range(DCP):
            if np.isnan(base[r][1][: int(starts[-1])]).any():
                failures.append(f"{merge} rank {r}: NaN in the flag-off output")
        glue = {"off": _glue(text)}
        for flags in ("lean", "chained", "both"):
            got, glue_text = attend(flags, local_merge)
            glue[flags] = _glue(glue_text)
            for r in range(DCP):
                for i, name in ((0, "pool"), (1, "out"), (2, "lse"), (3, "selected")):
                    if not np.array_equal(_bits(got[r][i]), _bits(base[r][i])):
                        failures.append(f"{merge} {flags} rank {r}: {name} differs")
            report[f"{merge} {flags}"] = (
                "bit-equal"
                if not any(f.startswith(f"{merge} {flags} ") for f in failures)
                else "differs"
            )
        report[f"{merge} glue"] = glue
    print(step_name, variant, report)
    assert not failures, failures
