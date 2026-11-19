# SPDX-License-Identifier: Apache-2.0
"""q38 L6 lever adcarry, lever-off identity (CPU job step P6).

Runs the l6_adc_harness engines (the real ``_prepare_inputs`` plus the real
Qwen3.8 adapter, served adapter flags) with every L6 lever unset over a
schedule of batch changes, and prints one digest per shape of every device
input of the forward plus the per-step upload counts. Run once with PYTHONPATH
at the head tree and once at the base tree (which has no adapt_carry module):
the L6ADC_IDENTITY_ALL lines must be equal.

    python3 tests/q38_l6/l6_adc_identity.py
"""

import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _k in list(os.environ):
    if _k.startswith("VLLM_TPU_L6_") or _k in ("VLLM_TPU_HOST_AB_PERIOD_MS", "VLLM_TPU_D1N_AB_PERIOD_MS"):
        del os.environ[_k]
os.environ["VLLM_TPU_POSTFWD_FUSED"] = "1"

import l6_adc_harness as A  # noqa: E402
import vllm_torchtpu  # noqa: E402
from vllm_torchtpu.core import l6_ab  # noqa: E402

SHAPES = {
    "b3": dict(nreq=3, max_num_reqs=8, paddings=(16, 32, 64), mamba=True, comb=True),
    "b5p": dict(nreq=5, max_num_reqs=8, paddings=(16, 32, 64), mamba=False, comb=True, pooled=True),
    "b2s": dict(nreq=2, max_num_reqs=8, paddings=(16, 32, 64), mamba=True, comb=True, prompt=1, ngram=8),
}
EVENTS = {
    5: ("join_short",), 8: ("swap",), 10: ("partial",), 12: ("nonspec",), 14: ("pad",), 16: ("skew",),
    18: ("remap",), 20: ("discard",), 22: ("preempt",), 24: ("idle_begin",), 25: ("resume",), 27: ("leave",),
    29: ("flush",),
}
CONFIGS = {"served": A.adapter_env(), "store": A.adapter_env(window=False),
           "inplace": A.adapter_env(functional=False)}


def main() -> int:
    l6_ab._resolve()
    assert not l6_ab.ACTIVE
    undo = A.install_eager_programs()
    lines = []
    try:
        for cname, cenv in CONFIGS.items():
            os.environ.update(cenv)
            for i, (name, shape) in enumerate(SHAPES.items()):
                digests, sim = A.run(shape, 32, seed=200 + i, events=EVENTS)
                up = hashlib.sha256(repr(sim.uploads).encode()).hexdigest()[:16]
                lines.append(f"{cname} {name} steps={len(digests)} out={A.digest_all(digests)} "
                             f"uploads={sum(sim.uploads)}:{up}")
    finally:
        undo()
    print("L6ADC_IDENTITY_FILE", os.path.realpath(vllm_torchtpu.__file__))
    for line in lines:
        print("L6ADC_IDENTITY", line)
    print("L6ADC_IDENTITY_ALL", hashlib.sha256("\n".join(lines).encode()).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
