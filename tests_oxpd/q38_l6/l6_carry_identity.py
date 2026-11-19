# SPDX-License-Identifier: Apache-2.0
"""q38 L6 lever carry, lever-off identity (CPU job step P6).

Runs the l6_carry_harness engines with VLLM_TPU_L6_DECODE_CARRY unset over a
schedule of batch changes and prints one digest per shape of every output
that leaves _prepare_inputs plus the per-step upload counts. Run once with
PYTHONPATH at the head tree and once at the base tree (which has no
decode_carry module): the L6CARRY_IDENTITY_ALL lines must be equal.

    python3 tests/q38_l6/l6_carry_identity.py
"""

import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _k in list(os.environ):
    if _k.startswith("VLLM_TPU_L6_") or _k in ("VLLM_TPU_HOST_AB_PERIOD_MS", "VLLM_TPU_D1N_AB_PERIOD_MS"):
        del os.environ[_k]
os.environ["VLLM_TPU_POSTFWD_FUSED"] = "1"

import l6_carry_harness as H  # noqa: E402
import vllm_torchtpu  # noqa: E402
from vllm_torchtpu.core import l6_ab  # noqa: E402

SHAPES = {
    "b1": dict(nreq=1, max_num_reqs=8, paddings=(16, 32, 64), mamba=True, comb=True),
    "b3": dict(nreq=3, max_num_reqs=8, paddings=(16, 32, 64), mamba=True, comb=True),
    "b5p": dict(nreq=5, max_num_reqs=8, paddings=(16, 32, 64), mamba=False, comb=True, pooled=True),
    "b8n": dict(nreq=8, max_num_reqs=8, paddings=(32, 64), mamba=True, comb=False),
}
EVENTS = {
    5: ("join",), 8: ("swap",), 10: ("partial",), 12: ("nonspec",), 14: ("pad",),
    16: ("skew",), 18: ("remap",), 20: ("discard",), 23: ("leave",), 26: ("flush",),
}


def main() -> int:
    l6_ab._resolve()
    assert not l6_ab.ACTIVE
    undo = H.install_eager_programs()
    lines = []
    try:
        for i, (name, shape) in enumerate(SHAPES.items()):
            events = {k: v for k, v in EVENTS.items() if not (name == "b8n" and "join" in v)}
            digests, sim = H.run(shape, 30, seed=100 + i, events=events)
            up = hashlib.sha256(repr(sim.uploads).encode()).hexdigest()[:16]
            lines.append(f"{name} steps={len(digests)} out={H.digest_all(digests)} "
                         f"uploads={sum(sim.uploads)}:{up}")
    finally:
        undo()
    print("L6CARRY_IDENTITY_FILE", os.path.realpath(vllm_torchtpu.__file__))
    for line in lines:
        print("L6CARRY_IDENTITY", line)
    print("L6CARRY_IDENTITY_ALL", hashlib.sha256("\n".join(lines).encode()).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
