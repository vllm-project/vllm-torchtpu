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
"""Plan static EPLB: replicate each layer's top-k hottest experts on every EP rank.

Reads routing dumps ``r<rank>_l<layer>_ring.npy`` (``[calls, T, topk]`` logical
expert ids, a circular buffer whose oldest slot is ``ctr % calls`` when
``r<rank>_l<layer>_ctr.npy`` exists) and writes a plan for ``DSV41_EPLB_PLAN``
with ``slots_per_rank = k`` (see ``vllm_torchtpu/layers/adapter/eplb_static.py``).
With the locality-first remap every token's hot experts run on its own chip.

``--steps`` selects the MoE calls to count. Use the steady-state decode window
of the workload you will run (full batch, no request finished): padding rows of
finished requests all route identically and skew the counts.

The routing recorder that produced the dumps is not part of this change.

usage: static_eplb_plan.py --dump DIR --ep 16 --k 6 --out plan.json --steps 21:100
"""

import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--ep", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--experts", type=int, default=384)
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--steps", required=True)
    a = ap.parse_args()
    s0, s1 = (int(x) for x in a.steps.split(":"))
    E, ep, k = a.experts, a.ep, a.k
    G = E // ep
    nranks = len(glob.glob(os.path.join(a.dump, "r*_l000_ring.npy")))
    layers = {}
    report = []
    for L in range(a.layers):
        cnt = np.zeros(E, np.int64)
        for r in range(nranks):
            ring = np.load(os.path.join(a.dump, f"r{r:03d}_l{L:03d}_ring.npy"))
            ctr_p = os.path.join(a.dump, f"r{r:03d}_l{L:03d}_ctr.npy")
            if os.path.exists(ctr_p):  # ring is circular: oldest slot = ctr % len
                ring = np.roll(ring, -(int(np.load(ctr_p)[0]) % ring.shape[0]), axis=0)
            ring = ring[s0:s1]
            ids = ring.reshape(-1)
            ids = ids[(ids >= 0) & (ids < E)]
            cnt += np.bincount(ids, minlength=E)
        hot = [int(e) for e in np.argsort(-cnt)[:k]]
        reps = []
        for e in hot:
            home = e // G
            reps += [[e, r] for r in range(ep) if r != home]
        layers[str(L)] = reps
        tot = max(cnt.sum(), 1)
        report.append((L, hot, [round(float(cnt[e] / tot), 2) for e in hot]))
    plan = {
        "num_experts": E,
        "ep": ep,
        "slots_per_rank": k,
        "layers": layers,
        "note": f"top-{k} hottest experts per layer on every rank; steps {a.steps}",
    }
    with open(a.out, "w") as f:
        json.dump(plan, f)
    for L, hot, sh in report[:5]:
        print(f"layer {L}: hot={hot} token share={sh}")
    print(f"wrote {a.out}: {len(layers)} layers, slots_per_rank={k}")


if __name__ == "__main__":
    main()
