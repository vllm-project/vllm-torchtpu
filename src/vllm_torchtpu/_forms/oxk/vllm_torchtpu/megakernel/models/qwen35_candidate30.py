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
"""`qwen35-candidate30`: tuned Qwen3.5 execution profile.

Select with `VLLM_TPU_MEGAKERNEL="profile=qwen35-candidate30"` or
`additional_config={"megakernel": {"profile": "qwen35-candidate30"}}`.

Enables the Qwen3.5 performance flags (`pipe_ep`, `gdn_head_major`, `gqa_qv`,
`moe_unroll`, `rel_fetch`, `kv_sub=1024`, `route_vec`, `dense_acc_pair`,
`pair_kv`, `kv_ering`, `gq_pack`, `topk_lanes`, `dmm_unroll=8`, `ecopy_opt=1`,
`r2_direct_store`), deeper EP expert rings, and `pool_gather="auto"`.
Fused pool I/O stays off here (experimental); `qwen35-candidate30-fused` is
the same tuning with `fused_pool_io="auto"`, the configuration measured
closest to candidate30 in serving (see `CANDIDATE30_BASELINE`).
"""

from __future__ import annotations

import dataclasses

from ..blocks.gdn import GDNBlock
from ..blocks.gqa import GQABlock
from ..blocks.moe import MoEBlock
from ..exec_spec import Tiles
from ..profile import Baseline, Profile, register_profile
from .qwen35_profile import QWEN35_MODEL_NAME

_GQA, _GDN, _MOE = GQABlock.name, GDNBlock.name, MoEBlock.name
# Opt-in performance flags enabled by this profile (core, then per block).
_C30_CORE_FLAGS = dict(dense_acc_pair=True, dmm_unroll=8, rel_fetch=True)
_C30_BLOCK_FLAGS = {
    _GQA: dict(kv_sub=1024, kv_ering=True, pair_kv=True, gq_pack=True, gqa_qv=True),
    _GDN: dict(gdn_head_major=True),
    _MOE: dict(
        ecopy_opt=1,
        pipe_ep=True,
        r2_direct_store=True,
        route_vec=True,
        topk_lanes=True,
        moe_unroll=True,
    ),
}
_BASE = Tiles(hier=1, mb8=2, **_C30_CORE_FLAGS).with_blocks(
    {**_C30_BLOCK_FLAGS, _GQA: {**_C30_BLOCK_FLAGS[_GQA], "pack": 1}}
)
# candidate30 stages the fused-path conv checkpoint only from this many verify rows
# up (`CONV_STAGE_MIN_B` in its kernel); smaller buckets keep the plain conv path.
CONV_STAGE_MIN_B = 8
_VERIFY = dataclasses.replace(_BASE, rsag=True, hier=2).with_blocks({_GQA: {"nkv": 4}})
_WIDE = dataclasses.replace(_VERIFY, nbd=10, vmem_mib=62, hier=0).with_blocks(
    {_GQA: {"nkv": 3}}
)


def candidate30_tiles(B: int, *, moe_tp: bool = False, R: int = 8) -> Tiles:
    """Per-bucket tiles for the tuned Qwen3.5 schedule (deeper EP rings, hier=2/3)."""
    nbe_ep = 12 if not moe_tp else 10
    nbe_small = 20 if not moe_tp else nbe_ep
    if B > 48:
        t = dataclasses.replace(_WIDE, nbd=8 if B <= 56 else 7, nbe=nbe_ep, hier=2)
    elif B > 32:
        t = dataclasses.replace(_WIDE, nbd=8, nbe=nbe_ep, hier=2)
    elif B > 24:
        t = dataclasses.replace(_WIDE, nbe=nbe_ep, hier=2)
    elif B > 16:
        t = dataclasses.replace(_WIDE, nbd=12, nbe=nbe_ep, hier=2)
    elif B > 8:
        t = dataclasses.replace(_VERIFY, nbe=nbe_ep)
    elif B > 4:
        t = dataclasses.replace(_BASE, nbe=nbe_small, rsag=True, hier=2)
    else:
        t = dataclasses.replace(_BASE, nbe=nbe_small)
    t = t.with_blocks({_GDN: {"conv_stage": B >= CONV_STAGE_MIN_B}})
    if R != 8 and t.hier in (1, 2):
        is_pow2_r = R >= 2 and (R & (R - 1)) == 0
        t = dataclasses.replace(t, hier=3 if (B <= 32 and is_pow2_r) else 0)
    return t


# candidate30 reference run (vLLM serving, v7x-8). The image's vllm-torchtpu
# commit is `8743659cd` (a bit-exact block refactor of the `36beb0816` kernel).
CANDIDATE30_BASELINE = Baseline(
    ref="candidate30",
    commit="8743659cdfcfbb0d77d7fc454f519520d223e785",
    job=(
        "c8 j-07676b1a-25aa-48f3-a600, c16 j-ffda2788-53ab-4d8a-a419, "
        "c32 j-de863aa4-7371-49ac-b57f"
    ),
    config=(
        "image vllm-torchtpu-caojx_google_com:infx-qwen35-agentx-candidate30; "
        "VLLM_TPU_QWEN35_MEGAKERNEL=1 VLLM_TPU_QWEN35_MK_MODE=ep, MTP=3; "
        "kernel: verify buckets 4..64, fused pool I/O for B<=24 or B>=56, "
        "GPR on the 3-stage buckets, e_slots=72"
    ),
    bench=(
        "cdk recipe infx-qwen35-agentx CLIENT_CONCURRENCIES={8,16,32} "
        "MAX_NUM_SEQS={16,16,32} DURATION=3600 SYNTHETIC_ACCEPTANCE_LENGTH=3.39 "
        "MTP=true MTP_NUM_SPECULATIVE_TOKENS=3 (job parameters of the jobs above)"
    ),
)

_DESCRIPTION = (
    "Tuned Qwen3.5 schedule: kv_sub=1024, route_vec, "
    "dense_acc_pair, pair_kv, KVE, QV/GQ, Batcher top-k, unrolled experts "
    "+ PIPE_EP, head-major GDN, deeper EP expert rings, hier=2 for B>16, "
    "conv staging from B>=8, GPR auto. Numerics: approx."
)

PROFILE = register_profile(
    Profile(
        name="qwen35-candidate30",
        model=QWEN35_MODEL_NAME,
        numerics="approx",
        tiles=candidate30_tiles,
        fused_pool_io=False,
        pool_gather=None,
        baseline=CANDIDATE30_BASELINE,
        description=_DESCRIPTION,
    )
)

# The same tiles plus selective 1-launch fused pool I/O (`auto`: B <= 24 or
# B >= 56), as in candidate30's own kernel; measured closest to candidate30 in
# serving. Fused pool I/O stays experimental: it requires pool
# donation (enforced) and the driver IR-checks every fused bucket.
FUSED_PROFILE = register_profile(
    dataclasses.replace(
        PROFILE,
        name="qwen35-candidate30-fused",
        fused_pool_io="auto",
        description=_DESCRIPTION + " Plus experimental fused pool I/O (auto).",
    )
)

__all__ = [
    "CANDIDATE30_BASELINE",
    "CONV_STAGE_MIN_B",
    "FUSED_PROFILE",
    "PROFILE",
    "candidate30_tiles",
]
