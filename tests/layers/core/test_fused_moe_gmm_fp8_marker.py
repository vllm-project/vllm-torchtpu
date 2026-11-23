# SPDX-License-Identifier: Apache-2.0
"""The sc_fp8 engagement marker prints once per worker process.

A one-host PP8 prefill runs eight worker processes with local ranks 0-7.
vLLM's info_once defaults to scope="local", which prints only where
is_local_first_rank() is true, so only PP0 printed the marker (Q38 FR9 arm A).
Each case below runs eight fresh processes, one per simulated local rank,
because info_once de-duplicates per process.
"""

import os
import subprocess
import sys

import pytest

FP8_MSG = "MoE dispatch gathers packed fp8 rows (sc_fp8)."
FALLBACK_MSG = "MoE sc_fp8 dispatch not applicable here; sc_padded."
RANKS = range(8)

_CHILD = r"""
import sys
import vllm.distributed.parallel_state as ps

rank = int(sys.argv[1])
ps.is_local_first_rank = lambda: rank == 0   # this process's local rank
from vllm_torchtpu.layers.core import fused_moe_gmm as m
import logging

# Capture at INFO under the "vllm" root, whatever this process's logging config is.
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
logging.getLogger("vllm").addHandler(handler)
logging.getLogger("vllm").setLevel(logging.INFO)

mode = sys.argv[2]
if mode == "applied":
    m.log_nonep_fp8_choice(True)
    m.log_nonep_fp8_choice(True)             # once per process, not per call
elif mode == "fallback":
    m.log_nonep_fp8_choice(False)
elif mode == "default_scope":                # the pre-fix call: info_once's default scope
    m.logger.info_once("MoE dispatch gathers packed fp8 rows (sc_fp8).")
print("CHILD_DONE", flush=True)
"""


def _lines_per_rank(mode: str, msg: str) -> dict[int, int]:
    env = dict(os.environ)
    procs = {
        r: subprocess.Popen(
            [sys.executable, "-c", _CHILD, str(r), mode],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
        )
        for r in RANKS
    }
    counts = {}
    for r, p in procs.items():
        out, _ = p.communicate(timeout=600)
        assert p.returncode == 0 and "CHILD_DONE" in out, out[-2000:]
        counts[r] = sum(1 for line in out.splitlines() if msg in line)
    return counts


@pytest.mark.parametrize("mode,msg", [("applied", FP8_MSG), ("fallback", FALLBACK_MSG)])
def test_marker_prints_once_on_every_rank(mode, msg):
    assert _lines_per_rank(mode, msg) == {r: 1 for r in RANKS}


def test_default_scope_info_once_prints_only_local_rank_0():
    # Documents why the marker needs scope="process": the old call is seen on PP0 only.
    assert _lines_per_rank("default_scope", FP8_MSG) == {r: int(r == 0) for r in RANKS}
