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
from ..ops import BF16, col_chunk, rup


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


def comm_params(
    ex: ExecSpec,
    tiles: Tiles,
    H: int,
    PR: int | None = None,
    PC: int | None = None,
) -> CommParams:
    """Derive the all-reduce mode, payload layout, rounds, and peers for `ex`."""
    R, B, M, MR, MB = ex.R, ex.B, ex.M, ex.MR, ex.MB
    if PC is None:
        PC = col_chunk(H)
    if H % PC != 0:
        raise MegakernelConfigError(
            f"all-reduce width H={H} must be divisible by column chunk PC={PC}"
        )
    HPC = H // PC
    if PR is None:
        PR = rup(B * HPC, 16)
    HIER = tiles.hier if R > 1 else 0
    RSAG = tiles.rsag and R > 1 and HIER == 0
    WAR = (
        (ex.WIDE or MR * H <= PR * PC or (RSAG and (PC // R) % 128 != 0))
        if tiles.war < 0
        else bool(tiles.war)
    )
    AR_R = MR if WAR else PR
    AR_C = H if WAR else PC
    CW = AR_C // R
    HC2 = AR_C // 2
    N_HIER3 = max(1, R.bit_length() - 1)
    if RSAG and CW % 128 != 0:
        raise MegakernelConfigError(
            f"all-reduce slice width CW={CW} (AR_C={AR_C}, R={R}) "
            "must be a multiple of 128"
        )
    valid_mode = HIER in (0, 1, 2, 3) and (
        HIER == 0
        or (HIER in (1, 2) and R == 8)
        or (HIER == 3 and R >= 2 and (R & (R - 1)) == 0)
    )
    if not valid_mode:
        raise MegakernelConfigError(f"unsupported all-reduce mode: hier={HIER}, R={R}")
    if HIER == 2 and HC2 % 128 != 0:
        raise MegakernelConfigError(f"hier=2 requires HC2={HC2} divisible by 128")
    AR_ROUNDS = (
        tuple(range(N_HIER3))
        if HIER == 3
        else ((0, 1, 2) if HIER == 2 else ((0, 1) if (RSAG or HIER == 1) else (0,)))
    )
    AR_PEERS = 1 if HIER == 3 else (3 if HIER in (1, 2) else max(R - 1, 1))
    return CommParams(
        R=R,
        B=B,
        M=M,
        MR=MR,
        MB=MB,
        H=H,
        PR=PR,
        PC=PC,
        HPC=HPC,
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
    )


def ring_scratch(cp: CommParams) -> dict[str, Any]:
    """Send/recv buffers and DMA semaphores every all-reduce mode allocates."""
    R, HIER, RSAG, AR_R, AR_C = cp.R, cp.HIER, cp.RSAG, cp.AR_R, cp.AR_C
    CW, HC2, N_HIER3 = cp.CW, cp.HC2, cp.N_HIER3
    return {
        "send": pltpu.VMEM(
            (N_HIER3, 2, AR_R, AR_C)
            if HIER == 3
            else (2, 2, AR_R, HC2)
            if HIER == 2
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
            else (2, R, AR_R, CW)
            if RSAG
            else (2, R, AR_R, AR_C),
            BF16,
        ),
        "ar_ssem": pltpu.SemaphoreType.DMA((len(cp.AR_ROUNDS), 2, cp.AR_PEERS)),
        "ar_rsem": pltpu.SemaphoreType.DMA((len(cp.AR_ROUNDS), 2, cp.AR_PEERS)),
    }


def stage_scratch(cp: CommParams) -> dict[str, Any]:
    """Extra all-gather and second-stage buffers of the RS+AG and hier=1/2 modes."""
    R, AR_R, AR_C, CW, HC2 = cp.R, cp.AR_R, cp.AR_C, cp.CW, cp.HC2
    if cp.RSAG:
        return {
            "ag_send": pltpu.VMEM((2, AR_R, CW), BF16),
            "ag_recv": pltpu.VMEM((2, R, AR_R, CW), BF16),
        }
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
