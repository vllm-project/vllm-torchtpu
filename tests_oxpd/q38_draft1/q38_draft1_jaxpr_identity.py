# SPDX-License-Identifier: Apache-2.0
"""q38 L4 X-D14-DRAFT-1: flag-off identity of the traced kernels (CDK CPU pod only).

Usage: PYTHONPATH=<tree>/src JAX_PLATFORMS=cpu python3 q38_draft1_jaxpr_identity.py <tree>

Traces (jax.make_jaxpr; real Pallas kernels, nothing runs) every pooled QSA step the
change touches, with VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE's code paths at their defaults,
and prints one line per function: ``JAXPR <name> <sha256[:16]> eqns=<n>``, then
``JAXPR_ALL <sha256[:16]>``. Run on the head tree and on the base tree: equal
lines mean the flag-off traced graphs are identical. File paths, line numbers and object
addresses are stripped before hashing (they move with any edit of the file). The indexer's
select_blocks and the batched RPA are the dcp2 harness stand-ins in both trees (neither is
touched by the change); a function that fails to trace prints ``JAXPR <name> ERROR``.
"""

import hashlib
import os
import re
import sys
import traceback

ROOT = os.path.abspath(sys.argv[1])
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.join(ROOT, "tests", "kernels", "qwen4_exp", "dcp2"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402

_INFO = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
pltpu.get_tpu_info = lambda: _INFO

from conftest import reference_rpa_lse  # noqa: E402
from test_dcp2_op import stub_select  # noqa: E402

from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper  # noqa: E402
from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa  # noqa: E402

assert os.path.abspath(pooled_qsa.__file__).startswith(ROOT), (pooled_qsa.__file__, ROOT)
wrapper.ragged_paged_attention = reference_rpa_lse
pooled_qsa.select_blocks = stub_select

B, DH, H, IDX, IDX_HEADS, RATIO, ROT, TOPK = 512, 256, 3, 128, 4, 4, 64, 256
OPTS = dict(compress_ratio=RATIO, token_topk=TOPK, norm_eps=1e-6, rotary_dim=ROT,
            k_scale=0.375, v_scale=2.5, max_query_len=4)
rng = np.random.default_rng(0)


def bf(*shape):
    return jnp.asarray(rng.normal(0, 0.3, shape), jnp.bfloat16)


T = 8
starts = jnp.asarray([0, 3, 6, 8], jnp.int32)
context = jnp.asarray([37, 509, 1022], jnp.int32)
positions = jnp.asarray(np.concatenate([37 + np.arange(3), 509 + np.arange(3), 1022 + np.arange(2)]),
                        jnp.int32)
trig = jnp.pad(bf(4096, ROT), ((0, 0), (0, 128 - ROT)))
ring = jnp.asarray([9, 10, 11], jnp.int32)
main = jnp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 0]], jnp.int32)
comp = jnp.asarray([[12, 13, 14], [15, 16, 17], [18, 19, 20]], jnp.int32)
pool = jnp.zeros((24, B, 1, 4, DH), jnp.float8_e4m3fn)
qgate1, qgate2 = bf(T, 2 * H * DH), bf(T, 4 * H * DH)
key, value, index_qk = bf(T, DH), bf(T, DH), bf(T, (IDX_HEADS + 1) * IDX)
nq, nk, niq, nik = bf(DH), bf(DH), bf(IDX), bf(IDX)
rows = jnp.asarray([2, 5, 7], jnp.int32)
W = TOPK + RATIO - 1
shared_all = jnp.asarray(rng.integers(-1, 1500, (T, W)), jnp.int32)
shared_rows = shared_all[:3]
rank = jnp.asarray([0], jnp.int32)


def head(qg):
    return (qg, key, value, index_qk, nq, nk, niq, nik, positions, trig)


TAIL = (starts, context, ring, main, comp)


def step(fn, qg, *extra, **kw):
    return lambda p: getattr(pooled_qsa, fn)(*head(qg), p, *TAIL, *extra, **OPTS, **kw)


def writes(**kw):
    def run(p):
        k = bf(T, 1, DH)
        v = bf(T, 1, DH)
        raw = bf(T, IDX)
        return pooled_qsa.qsa_cache_writes(
            p, k, v, raw, positions, nik, trig, starts, context, ring, main, comp,
            compress_ratio=RATIO, norm_eps=1e-6, rotary_dim=ROT, max_query_len=4, **kw)

    return run


CASES = [
    ("pooled_qsa_step", step("pooled_qsa_step", qgate1)),
    ("pooled_qsa_rows_step", step("pooled_qsa_rows_step", qgate1, rows)),
    ("pooled_qsa_shared_step", step("pooled_qsa_shared_step", qgate1, shared_all)),
    ("pooled_qsa_shared_rows_step", step("pooled_qsa_shared_rows_step", qgate1, rows, shared_rows)),
    ("pooled_qsa_shared_rows_step[slot0]",
     step("pooled_qsa_shared_rows_step", qgate1, rows, shared_rows, main_slot=0)),
    ("pooled_qsa_dcp_step", step("pooled_qsa_dcp_step", qgate2, rank)),
    ("pooled_qsa_rows_dcp_step", step("pooled_qsa_rows_dcp_step", qgate2, rows, rank)),
    ("pooled_qsa_shared_dcp_step", step("pooled_qsa_shared_dcp_step", qgate2, shared_all, rank)),
    ("pooled_qsa_shared_rows_dcp_step",
     step("pooled_qsa_shared_rows_dcp_step", qgate2, rows, shared_rows, rank)),
    ("pooled_qsa_shared_rows_dcp_step[compressed_dcp]",
     step("pooled_qsa_shared_rows_dcp_step", qgate2, rows, shared_rows, rank, compressed_dcp=True)),
    ("pooled_qsa_select_step", step("pooled_qsa_select_step", qgate1)),
    ("qsa_cache_writes", writes()),
    ("qsa_cache_writes[dcp]", writes(dcp=(rank, 2))),
    ("qsa_cache_writes[slot0]", writes(main_slot=0)),
]


def normalize(text):
    text = text.replace(ROOT, "<root>")
    text = re.sub(r"[\w./<>-]+\.py:\d+(:\d+)?", "<src>", text)
    return re.sub(r"0x[0-9a-fA-F]+", "0x", text)


digests = []
for name, fn in CASES:
    try:
        jaxpr = jax.make_jaxpr(fn)(pool)
    except Exception:  # noqa: BLE001
        print(f"JAXPR {name} ERROR")
        traceback.print_exc(limit=3)
        digests.append(f"{name}:ERROR")
        continue
    digest = hashlib.sha256(normalize(str(jaxpr)).encode()).hexdigest()[:16]
    digests.append(f"{name}:{digest}")
    print(f"JAXPR {name} {digest} eqns={len(jaxpr.jaxpr.eqns)}")
print(f"JAXPR_ALL {hashlib.sha256(chr(10).join(digests).encode()).hexdigest()[:16]}")
