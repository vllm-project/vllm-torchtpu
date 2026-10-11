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
from ..ops import BF16, col_chunk, round_up


@dataclass(frozen=True)
class CommParams:
    """Core all-reduce and inter-rank collective parameters.

    Supported `HIER` modes (`comm_params`):
    - `HIER == 0`: flat ring (`RSAG=False`, 1 round) or 2-round reduce-scatter +
      all-gather (`RSAG=True`), accepted for any power-of-two `R >= 1`.
    - `HIER == 1`: 2-stage 8-rank pair-reduce + 4-way all-gather (`R == 8`);
      the 4-way round pushes to the peers `me ^ off` for `off` in `HIER1_PEERS`.
    - `HIER == 2`: 3-stage 8-rank half-column pair-reduce + 4-way all-gather +
      pair all-gather (`R == 8`, requires `HC2 % 128 == 0`).
    - `HIER == 3`: `N_HIER3 = log2(R)`-round recursive halving/doubling butterfly,
      accepted for any power-of-two `R >= 2`.
    """

    R: int  # Number of tensor-parallel ranks in the mesh.
    B: int  # Active token rows per rank before folding.
    MR: int  # Per-rank padded token rows (`round_up(M, 8*mb8)`).
    H: int  # Full hidden dimension (columns).
    PR: int  # Folded payload rows (`round_up(B * HPC, 16)`) when `WAR=False`.
    PC: int  # Folded payload column chunk width (`H // HPC`) when `WAR=False`.
    HPC: int  # Number of column chunks (`H // PC`) folded into rows when `WAR=False`.
    HIER: int  # Collective topology mode (`0`, `1`, `2`, or `3`).
    RSAG: bool  # True when `HIER == 0` uses 2-round reduce-scatter + all-gather.
    WAR: bool  # True when payload stays wide `[MR, H]` instead of folded `[PR, PC]`.
    AR_R: int  # All-reduce buffer row count (`MR` if `WAR` else `PR`, in rows).
    AR_C: int  # All-reduce buffer column width (`H` if `WAR` else `PC`, in columns).
    CW: int  # Per-rank column slice width (`AR_C // R`, in columns) for `RSAG`.
    HC2: int  # Half-column width (`AR_C // 2`, in columns) for `HIER == 2`.
    AR_ROUNDS: tuple[
        int, ...
    ]  # Active round indices (`(0,)` .. `tuple(range(N_HIER3))`).
    AR_PEERS: int  # Remote peer DMA descriptors per round per slot.
    N_HIER3: int  # Number of butterfly rounds (`max(1, log2(R))`) for `HIER == 3`.
    # `HIER == 1` cross-chip peer offsets (`tiles.hier1_peers`): one core on each
    # of the other three chips of the 2x2x1 mesh (core `me ^ off`).
    HIER1_PEERS: tuple[int, ...] = (2, 4, 6)


def _check_hier1_peers(peers: tuple[int, ...]) -> None:
    """`hier1_peers` must name one core on each other chip (`off >> 1` = 1, 2, 3).

    Both cores of a chip hold the same chip sum after round 0, so any such
    choice delivers the same three chip sums to the same `ag_recv` rows.
    """
    if len(peers) != 3 or sorted(off >> 1 for off in peers) != [1, 2, 3]:
        raise MegakernelConfigError(
            f"hier1_peers={peers} must hold one offset per other chip "
            "(off >> 1 in {1, 2, 3}, e.g. (2, 4, 6) or (3, 5, 7))"
        )


def comm_params(
    ex: ExecSpec,
    tiles: Tiles,
    H: int,
    PR: int | None = None,
    PC: int | None = None,
) -> CommParams:
    """Derive the all-reduce mode, payload layout, rounds, and peers for `ex`."""
    R, B, MR = ex.R, ex.B, ex.MR
    if PC is None:
        PC = col_chunk(H)
    if H % PC != 0:
        raise MegakernelConfigError(
            f"all-reduce width H={H} must be divisible by column chunk PC={PC}"
        )
    HPC = H // PC
    if PR is None:
        PR = round_up(B * HPC, 16)
    elif PR < B * HPC:
        raise MegakernelConfigError(
            f"all-reduce folded rows PR={PR} must be >= B * HPC = {B * HPC} "
            f"(B={B}, HPC={HPC})"
        )
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
    pow2_r = R >= 1 and (R & (R - 1)) == 0
    valid_mode = HIER in (0, 1, 2, 3) and (
        (HIER == 0 and pow2_r)
        or (HIER in (1, 2) and R == 8)
        or (HIER == 3 and R >= 2 and pow2_r)
    )
    if not valid_mode:
        raise MegakernelConfigError(f"unsupported all-reduce mode: hier={HIER}, R={R}")
    if HIER == 2 and HC2 % 128 != 0:
        raise MegakernelConfigError(f"hier=2 requires HC2={HC2} divisible by 128")
    HIER1_PEERS = tuple(int(off) for off in tiles.hier1_peers)
    if HIER == 1:
        _check_hier1_peers(HIER1_PEERS)
    AR_ROUNDS = (
        tuple(range(N_HIER3))
        if HIER == 3
        else ((0, 1, 2) if HIER == 2 else ((0, 1) if (RSAG or HIER == 1) else (0,)))
    )
    AR_PEERS = 1 if HIER == 3 else (3 if HIER in (1, 2) else max(R - 1, 1))
    return CommParams(
        R=R,
        B=B,
        MR=MR,
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
        HIER1_PEERS=HIER1_PEERS,
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
