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
"""`CommParams` and the all-reduce scratch shapes derived from tiles and rank count."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jax.experimental.pallas import tpu as pltpu

from ..errors import MegakernelConfigError
from ..exec_spec import ExecSpec, Tiles
from ..ops import BF16


@dataclass(frozen=True)
class CommParams:
    """Core all-reduce and inter-rank collective parameters."""

    R: int
    B: int
    M: int
    MR: int
    MB: int
    H: int
    PR: int
    PC: int
    HPC: int
    HIER: int
    RSAG: bool
    WAR: bool
    AR_R: int
    AR_C: int
    CW: int
    HC2: int
    AR_ROUNDS: tuple[int, ...]
    AR_PEERS: int
    N_HIER3: int = 3
    HIER_GROUP: int = 8
    N_HOSTS: int = 1


def comm_params(
    ex: ExecSpec, tiles: Tiles, H: int, PR: int, PC: int, n_ar_slots: int
) -> CommParams:
    """Derive the all-reduce mode, payload layout, rounds, and peers for `ex`.

    `n_ar_slots` is the number of all-reduce slots per layer in the plan.
    """
    R, B, M, MR, MB = ex.R, ex.B, ex.M, ex.MR, ex.MB
    HIER = tiles.hier if R > 1 else 0
    HIER_GROUP = tiles.hier_group if HIER == 4 else R
    N_HOSTS = R // HIER_GROUP if HIER == 4 else 1
    RSAG = tiles.rsag and R > 1 and HIER == 0
    WAR = (
        (
            ex.WIDE
            or MR * H <= PR * PC
            or ((RSAG or HIER == 4) and (PC // HIER_GROUP) % 128 != 0)
        )
        if tiles.war < 0
        else bool(tiles.war)
    )
    AR_R = MR if WAR else PR
    AR_C = H if WAR else PC
    CW = AR_C // HIER_GROUP
    HC2 = AR_C // 2
    N_HIER3 = max(1, R.bit_length() - 1)
    if (RSAG or HIER == 4) and CW % 128 != 0:
        raise MegakernelConfigError(
            f"all-reduce slice width CW={CW} (AR_C={AR_C}, group={HIER_GROUP}) "
            "must be a multiple of 128"
        )
    # Round 1 pairs hosts via (host_idx ^ host_xor), which stays < N_HOSTS
    # only when N_HOSTS is a power of 2.
    valid_mode = HIER in (0, 1, 2, 3, 4) and (
        HIER == 0
        or (HIER in (1, 2) and R == 8)
        or (HIER == 3 and R >= 2 and (R & (R - 1)) == 0)
        or (
            HIER == 4
            and HIER_GROUP >= 2
            and (HIER_GROUP & (HIER_GROUP - 1)) == 0
            and R >= HIER_GROUP
            and R % HIER_GROUP == 0
            and (N_HOSTS & (N_HOSTS - 1)) == 0
        )
    )
    if not valid_mode:
        raise MegakernelConfigError(
            f"unsupported all-reduce mode: hier={HIER}, R={R}, "
            f"hier_group={HIER_GROUP}, num_hosts={N_HOSTS}"
        )
    if HIER == 2 and HC2 % 128 != 0:
        raise MegakernelConfigError(f"hier=2 requires HC2={HC2} divisible by 128")
    # Remote peers push directly into recv[slot], ih_recv[slot], and ag_recv[slot],
    # so consecutive hier=4 all-reduces must alternate across at least 2 slots.
    if HIER == 4 and R > 1 and n_ar_slots < 2:
        raise MegakernelConfigError(
            f"hier=4 needs 2 all-reduce slots per layer, got n_ar_slots={n_ar_slots}"
        )
    AR_ROUNDS = (
        tuple(range(N_HIER3))
        if HIER == 3
        else (
            (0, 1, 2) if HIER in (2, 4) else ((0, 1) if (RSAG or HIER == 1) else (0,))
        )
    )
    AR_PEERS = (
        1
        if HIER == 3
        else (
            max(HIER_GROUP - 1, N_HOSTS - 1)
            if HIER == 4
            else (3 if HIER in (1, 2) else max(R - 1, 1))
        )
    )
    return CommParams(
        R=R,
        B=B,
        M=M,
        MR=MR,
        MB=MB,
        H=H,
        PR=PR,
        PC=PC,
        HPC=H // PC,
        HIER=HIER,
        RSAG=RSAG,
        WAR=WAR,
        AR_R=AR_R,
        AR_C=AR_C,
        CW=CW,
        HC2=HC2,
        AR_ROUNDS=AR_ROUNDS,
        AR_PEERS=AR_PEERS,
        N_HIER3=N_HIER3,
        HIER_GROUP=HIER_GROUP,
        N_HOSTS=N_HOSTS,
    )


def ring_scratch(cp: CommParams) -> dict[str, Any]:
    """Send/recv buffers and DMA semaphores every all-reduce mode allocates."""
    R, HIER, RSAG, AR_R, AR_C = cp.R, cp.HIER, cp.RSAG, cp.AR_R, cp.AR_C
    CW, HC2, N_HIER3, group_size = cp.CW, cp.HC2, cp.N_HIER3, cp.HIER_GROUP
    return {
        "send": pltpu.VMEM(
            (N_HIER3, 2, AR_R, AR_C)
            if HIER == 3
            else (2, 2, AR_R, HC2)
            if HIER == 2
            else (2, group_size, AR_R, CW)
            if HIER == 4
            else (2, R, AR_R, CW)
            if RSAG
            else (2, AR_R, AR_C),
            BF16,
        ),
        "recv": pltpu.VMEM(
            (N_HIER3, 2, 2, AR_R, AR_C)
            if HIER == 3
            else (2, 2, AR_R, HC2)
            if HIER == 2
            else (2, 2, AR_R, AR_C)
            if HIER == 1
            else (2, group_size, AR_R, CW)
            if HIER == 4
            else (2, R, AR_R, CW)
            if RSAG
            else (2, R, AR_R, AR_C),
            BF16,
        ),
        "ar_ssem": pltpu.SemaphoreType.DMA((len(cp.AR_ROUNDS), 2, cp.AR_PEERS)),
        "ar_rsem": pltpu.SemaphoreType.DMA((len(cp.AR_ROUNDS), 2, cp.AR_PEERS)),
    }


def stage_scratch(cp: CommParams) -> dict[str, Any]:
    """Extra all-gather and second-stage buffers of the RS+AG and hier=1/2/4 modes."""
    AR_R, AR_C, CW, HC2 = cp.AR_R, cp.AR_C, cp.CW, cp.HC2
    if cp.RSAG or cp.HIER == 4:
        out = {
            "ag_send": pltpu.VMEM((2, AR_R, CW), BF16),
            "ag_recv": pltpu.VMEM((2, cp.HIER_GROUP, AR_R, CW), BF16),
        }
        if cp.HIER == 4 and cp.N_HOSTS > 1:
            out["ih_recv"] = pltpu.VMEM((2, cp.N_HOSTS, AR_R, CW), BF16)
            out["s2_send"] = pltpu.VMEM((2, AR_R, CW), BF16)
        return out
    if cp.HIER == 1:
        return {
            "ag_send": pltpu.VMEM((2, AR_R, AR_C), BF16),
            "ag_recv": pltpu.VMEM((2, 4, AR_R, AR_C), BF16),
        }
    if cp.HIER == 2:
        return {
            "ag_send": pltpu.VMEM((2, AR_R, HC2), BF16),
            "ag_recv": pltpu.VMEM((2, 4, AR_R, HC2), BF16),
            "s2_send": pltpu.VMEM((2, AR_R, HC2), BF16),
            "s2_recv": pltpu.VMEM((2, 2, AR_R, HC2), BF16),
        }
    return {}
