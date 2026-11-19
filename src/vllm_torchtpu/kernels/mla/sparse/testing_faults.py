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
"""Trace-time fault-injection definitions and hooks for SparseMLA tests."""

from __future__ import annotations

import dataclasses
from enum import StrEnum
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax


class PrologueFault(StrEnum):
    """Test-only faults for in-kernel Q-RoPE, kernel-owned LSE, and row widths."""

    SKIP_Q_ROPE = "skip_q_rope"
    ROPE_SWAP = "rope_swap"
    ROPE_F32 = "rope_f32"
    SKIP_LSE_FILL = "skip_lse_fill"
    ROW_WIDTH_UNDERSIZE = "row_width_undersize"


class SendFault(StrEnum):
    """Test-only faults for the DCP send-layout path."""

    SEND_DEST_ROTATE = "send_dest_rotate"
    SEND_LSE_LO_SKIP = "send_lse_lo_skip"


class SendExchangeFault(StrEnum):
    """Test-only faults and timing arms for the fused small all-to-all exchange."""

    SEND_PEER_MISORDER = "send_peer_misorder"
    SEND_PEER_DROP = "send_peer_drop"
    SEND_XCHG_RM_ROUND = "send_xchg_rm_round"
    SEND_XCHG_ENTRY_WAIT = "send_xchg_entry_wait"
    SEND_MERGE_LO_SKIP = "send_merge_lo_skip"


class LivePrepFault(StrEnum):
    """Test-only positive-control faults for fused live-window preparation."""

    KV = "kv"
    OFF = "off"
    WIN1 = "win1"


class OutPiecesFault(StrEnum):
    """Test-only faults for multi-piece output concatenation."""

    SWAP = "swap"


# Backward-compatible tuple constants mirroring historical kernel.py exports.
PROLOGUE_FAULTS: tuple[str | None, ...] = (
    None,
    *(fault.value for fault in PrologueFault),
)
SEND_FAULTS: tuple[str | None, ...] = (
    None,
    *(fault.value for fault in SendFault),
)
SEND_EXCHANGE_FAULTS: tuple[str, ...] = tuple(
    fault.value for fault in SendExchangeFault
)
LIVE_PREP_FAULTS: tuple[str | None, ...] = (
    None,
    *(fault.value for fault in LivePrepFault),
)
OUT_PIECES_FAULTS: tuple[str | None, ...] = (
    None,
    *(fault.value for fault in OutPiecesFault),
)

_PROLOGUE_AND_SEND_MAP: dict[str, PrologueFault | SendFault] = {
    **{fault.value: fault for fault in PrologueFault},
    **{fault.value: fault for fault in SendFault},
}
_SEND_EXCHANGE_MAP: dict[str, SendExchangeFault] = {
    fault.value: fault for fault in SendExchangeFault
}
_LIVE_PREP_MAP: dict[str, LivePrepFault] = {
    fault.value: fault for fault in LivePrepFault
}
_OUT_PIECES_MAP: dict[str, OutPiecesFault] = {
    fault.value: fault for fault in OutPiecesFault
}


@dataclasses.dataclass(frozen=True)
class FaultHooks:
    """Frozen, hashable trace-time fault-injection hooks for SparseMLA.

    Every hook resolves strictly at Python trace time so `NO_FAULT_HOOKS` emits
    zero extra XLA/Mosaic ops and allocates no extra VMEM.
    """

    prologue: PrologueFault | SendFault | SendExchangeFault | None = None
    live_prep: LivePrepFault | None = None
    out_pieces: OutPiecesFault | None = None
    kv_write: str | None = None

    @classmethod
    def from_names(
        cls,
        *,
        prologue_fault: str | None = None,
        live_prep_fault: str | None = None,
        out_pieces_fault: str | None = None,
        kv_write_fault: str | None = None,
        allow_send_exchange: bool = False,
    ) -> FaultHooks:
        """Build validated hooks from string fault names.

        Args:
            prologue_fault: Prologue, send-layout, or (when
                `allow_send_exchange` is True) fused-exchange fault name.
            live_prep_fault: Fused live-window preparation fault name.
            out_pieces_fault: Output-piece merge fault name.
            kv_write_fault: KV-write fault name forwarded to `native_sc_gather`.
            allow_send_exchange: Whether `SEND_EXCHANGE_FAULTS` are permitted in
                `prologue_fault`.

        Returns:
            A frozen, hashable `FaultHooks` instance (`NO_FAULT_HOOKS` when no
            faults are enabled).

        Raises:
            ValueError: If any fault name is unknown or disallowed.
        """
        parsed_prologue: PrologueFault | SendFault | SendExchangeFault | None = None
        if prologue_fault is not None:
            parsed_prologue = _PROLOGUE_AND_SEND_MAP.get(prologue_fault)
            if parsed_prologue is None and allow_send_exchange:
                parsed_prologue = _SEND_EXCHANGE_MAP.get(prologue_fault)
            if parsed_prologue is None:
                raise ValueError(f"unknown prologue_fault {prologue_fault}")

        parsed_live_prep: LivePrepFault | None = None
        if live_prep_fault is not None:
            parsed_live_prep = _LIVE_PREP_MAP.get(live_prep_fault)
            if parsed_live_prep is None:
                raise ValueError(f"unknown live_prep_fault {live_prep_fault}")

        parsed_out_pieces: OutPiecesFault | None = None
        if out_pieces_fault is not None:
            parsed_out_pieces = _OUT_PIECES_MAP.get(out_pieces_fault)
            if parsed_out_pieces is None:
                raise ValueError(f"unknown out_pieces_fault {out_pieces_fault}")

        if (
            parsed_prologue is None
            and parsed_live_prep is None
            and parsed_out_pieces is None
            and kv_write_fault is None
        ):
            return NO_FAULT_HOOKS

        return cls(
            prologue=parsed_prologue,
            live_prep=parsed_live_prep,
            out_pieces=parsed_out_pieces,
            kv_write=kv_write_fault,
        )

    def validate_for_context(self, *, allow_send_exchange: bool) -> None:
        """Validate that active hooks are permitted in this execution context."""
        if isinstance(self.prologue, SendExchangeFault) and not allow_send_exchange:
            raise ValueError(f"unknown prologue_fault {self.prologue.value}")

    # --- Q-RoPE hooks ---

    def q_rope_roll_shifts(self) -> tuple[int, int]:
        """Return `(lo, hi)` lane roll shifts for NeoX Q-RoPE half pairing."""
        if self.prologue == PrologueFault.ROPE_SWAP:
            return (32, 96)
        return (96, 32)

    def apply_q_rope_math(
        self,
        xf: jax.Array,
        rot: jax.Array,
        c: jax.Array,
        s: jax.Array,
        out_dtype: jax.typing.DTypeLike,
    ) -> jax.Array:
        """Compute the pre-roll Q-RoPE linear combination in float32."""
        if self.prologue == PrologueFault.SKIP_Q_ROPE:
            return xf
        if self.prologue == PrologueFault.ROPE_F32:
            return xf * c + rot * s
        # Round each product to out_dtype before adding to match XLA's op-by-op
        # bf16 evaluation.
        a = (xf * c).astype(out_dtype).astype(jnp.float32)
        b = (rot * s).astype(out_dtype).astype(jnp.float32)
        return a + b

    # --- Row-width and LSE fill hooks ---

    def effective_row_widths(
        self,
        row_widths: tuple[int, ...],
    ) -> tuple[int | None, ...]:
        """Return attended row widths for each bucket in `with_row_width`."""
        if self.prologue == PrologueFault.ROW_WIDTH_UNDERSIZE:
            return (row_widths[0], *row_widths)
        return (*row_widths, None)

    def lse_fill_sentinel(self, default_sentinel: float) -> float:
        """Return the sentinel value written by the kernel-owned LSE fill."""
        if self.prologue == PrologueFault.SKIP_LSE_FILL:
            return 0.0
        return default_sentinel

    # --- Send-layout hooks ---

    def send_dest_rank(self, dest: int, world: int) -> int:
        """Return the destination rank slab index for send-layout writes."""
        if self.prologue == SendFault.SEND_DEST_ROTATE:
            return (dest + 1) % world
        return dest

    def transform_send_lse_lo(self, lo_src: jax.Array) -> jax.Array:
        """Return the low-half LSE words placed into the send word row."""
        if self.prologue == SendFault.SEND_LSE_LO_SKIP:
            return jnp.zeros_like(lo_src)
        return lo_src

    # --- Fused exchange hooks ---

    def poison_merge_recv_scratch(self, recv_v_ref: Any) -> None:
        """Poison merge receive VMEM with bf16 quiet-NaN words before barrier."""
        if (
            self.prologue
            in (
                SendExchangeFault.SEND_PEER_MISORDER,
                SendExchangeFault.SEND_PEER_DROP,
            )
            and recv_v_ref is not None
        ):
            recv_v_ref[...] = jnp.full(recv_v_ref.shape, 0x7FC0, jnp.int16)

    def wait_exchange_barrier_at_entry(self) -> bool:
        """Return whether to wait the exchange barrier at kernel entry."""
        return self.prologue == SendExchangeFault.SEND_XCHG_ENTRY_WAIT

    def wait_exchange_barrier_in_round(self) -> bool:
        """Return whether to wait the exchange barrier before the round."""
        return self.prologue != SendExchangeFault.SEND_XCHG_ENTRY_WAIT

    def exchange_local_only_round(self) -> bool:
        """Return whether to replace peer remote sends with local copies."""
        return self.prologue == SendExchangeFault.SEND_XCHG_RM_ROUND

    def exchange_expect_peers(self, world: int) -> int:
        """Return the count of remote peer slabs expected by the exchange."""
        if self.prologue == SendExchangeFault.SEND_PEER_DROP:
            return world - 2
        return world - 1

    def exchange_dst_slot(self, me: jax.Array, world: int) -> jax.Array:
        """Return the remote destination slot for this rank's outgoing slab."""
        if self.prologue == SendExchangeFault.SEND_PEER_MISORDER:
            return lax.rem(me + 1, world)
        return me

    def exchange_should_skip_peer(
        self,
        peer: int,
        me: jax.Array,
        world: int,
    ) -> jax.Array:
        """Return boolean mask indicating whether to skip sending to `peer`."""
        skip = jnp.int32(peer) == me
        if self.prologue == SendExchangeFault.SEND_PEER_DROP:
            skip = skip | (jnp.int32(peer) == lax.rem(me + 3, world))
        return skip

    def transform_merge_w_lo(self, w_lo: jax.Array) -> jax.Array:
        """Return the low-precision weight residual for fused exchange merge."""
        if self.prologue == SendExchangeFault.SEND_MERGE_LO_SKIP:
            return jnp.zeros_like(w_lo)
        return w_lo

    # --- Live fused prep hooks ---

    def corrupt_live_prep_kv_row(self, kv_row: jax.Array) -> jax.Array:
        """Optionally perturb lane 5 of a fused live-prep `kv_lens` row."""
        if self.live_prep == LivePrepFault.KV:
            return kv_row + (
                lax.broadcasted_iota(jnp.int32, kv_row.shape, 1) == 5
            ).astype(jnp.int32)
        return kv_row

    def corrupt_live_prep_offset(self, off: jax.Array, tb: int) -> jax.Array:
        """Optionally perturb the running live-group offset after group 0."""
        if self.live_prep == LivePrepFault.OFF and tb == 0:
            return off + 1
        return off

    def live_prep_overflow_windows_end(self, num_windows: int) -> int:
        """Return the exclusive upper window bound for overflow window prep."""
        if self.live_prep == LivePrepFault.WIN1:
            return 1
        return num_windows

    # --- Merge pieces hook ---

    def transform_piece_outs(
        self,
        piece_outs: list[jax.Array],
    ) -> list[jax.Array]:
        """Return output pieces, swapping pieces 0 and 1 under `SWAP` fault."""
        if self.out_pieces == OutPiecesFault.SWAP:
            return [piece_outs[1], piece_outs[0], *piece_outs[2:]]
        return piece_outs


NO_FAULT_HOOKS = FaultHooks()
