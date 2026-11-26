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

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

SHARED_INTER_16K = 3072
SHARED_HIDDEN_16K = 7168
SHARED_STAGED_MIDDLE_COLS_16K = 2 * SHARED_INTER_16K
SHARED_DOWN_PANEL_COLS_16K = 512
SHARED_DOWN_NUM_PANELS_16K = SHARED_HIDDEN_16K // SHARED_DOWN_PANEL_COLS_16K


def stage_middle_half_copy(
    src_vmem,
    shared_hbm,
    sem,
    *,
    half: int,
    panel: int,
    panel_width: int = 256,
):
    assert half in (0, 1)
    col_offset = half * SHARED_INTER_16K + panel * panel_width
    assert col_offset + panel_width <= SHARED_STAGED_MIDDLE_COLS_16K
    return pltpu.make_async_copy(
        src_vmem,
        shared_hbm.at[:, pl.ds(col_offset, panel_width)],
        sem,
    )


def _await_partner_scope(ready_sem, peer) -> None:
    pl.semaphore_signal(
        ready_sem, 1, device_id=peer, device_id_type=pl.DeviceIdType.LOGICAL
    )
    pl.semaphore_wait(ready_sem, 1)


def run_shared_down_512_poll2(
    middle_v,
    shared_down_hbm,
    shared_hbm,
    down_ready_sem,
    *,
    rows: int,
    parity,
    peer,
    on_poll: Callable[[], None] | None = None,
) -> None:
    assert rows % 128 == 0
    assert middle_v.shape == (2 * rows, SHARED_INTER_16K)
    assert shared_down_hbm.shape == (SHARED_INTER_16K, SHARED_HIDDEN_16K)
    assert shared_hbm.shape == (rows, SHARED_HIDDEN_16K)
    vmem, dma = pltpu.VMEM, pltpu.SemaphoreType.DMA
    panel_cols = SHARED_DOWN_PANEL_COLS_16K
    n_panels = SHARED_DOWN_NUM_PANELS_16K
    assert n_panels % 2 == 0
    n_pairs = n_panels // 2

    def run_down(ds) -> None:
        _await_partner_scope(down_ready_sem, peer)
        peer_off = (1 - parity) * rows
        own_off = parity * rows

        def down_copy(panel, bank: int):
            col0 = pl.multiple_of(panel * panel_cols, panel_cols)
            return pltpu.make_async_copy(
                shared_down_hbm.at[:, pl.ds(col0, panel_cols)],
                ds["down_buffers"][bank],
                ds["down_sems"][bank],
            )

        def finish_reduction(prev_panel, prev_bank: int, wait_prev_store) -> None:
            own_part = ds["partial"][prev_bank][pl.ds(own_off, rows), :]
            transfer = pltpu.make_async_remote_copy(
                ds["partial"][prev_bank].at[pl.ds(peer_off, rows), :],
                ds["peer_y"][prev_bank],
                ds["reduce_send"][prev_bank],
                ds["reduce_recv"][prev_bank],
                device_id=peer,
                device_id_type=pl.DeviceIdType.LOGICAL,
            )
            transfer.wait_send()
            transfer.wait_recv()
            reduced_panel = (own_part + ds["peer_y"][prev_bank][...]).astype(
                jnp.bfloat16
            )

            @pl.when(wait_prev_store)
            def _wait_prev_store():
                c_old = pl.multiple_of((prev_panel - 2) * panel_cols, panel_cols)
                pltpu.make_async_copy(
                    ds["shared_tiles"][prev_bank],
                    shared_hbm.at[:, pl.ds(c_old, panel_cols)],
                    ds["shared_sems"][prev_bank],
                ).wait()

            ds["shared_tiles"][prev_bank][...] = reduced_panel
            c_cur = pl.multiple_of(prev_panel * panel_cols, panel_cols)
            pltpu.make_async_copy(
                ds["shared_tiles"][prev_bank],
                shared_hbm.at[:, pl.ds(c_cur, panel_cols)],
                ds["shared_sems"][prev_bank],
            ).start()

        def run_panel_half(panel, bank: int, has_next, has_prev, prev_wait_store):
            down_copy(panel, bank).wait()

            @pl.when(has_next)
            def _prefetch_next():
                down_copy(panel + 1, 1 - bank).start()

            p_peer = jnp.matmul(
                middle_v[pl.ds(peer_off, rows), :],
                ds["down_buffers"][bank][...],
                preferred_element_type=jnp.float32,
            )
            ds["partial"][bank][pl.ds(peer_off, rows), :] = p_peer

            @pl.when(has_prev)
            def _finish_prev():
                finish_reduction(panel - 1, 1 - bank, prev_wait_store)

            pltpu.make_async_remote_copy(
                ds["partial"][bank].at[pl.ds(peer_off, rows), :],
                ds["peer_y"][bank],
                ds["reduce_send"][bank],
                ds["reduce_recv"][bank],
                device_id=peer,
                device_id_type=pl.DeviceIdType.LOGICAL,
            ).start()
            if on_poll is not None:
                on_poll()
            p_own = jnp.matmul(
                middle_v[pl.ds(own_off, rows), :],
                ds["down_buffers"][bank][...],
                preferred_element_type=jnp.float32,
            )
            ds["partial"][bank][pl.ds(own_off, rows), :] = p_own
            if on_poll is not None:
                on_poll()

        down_copy(jnp.int32(0), 0).start()

        @pl.loop(0, n_pairs)
        def pair_loop(pair_idx):
            p0 = pair_idx * 2
            p1 = p0 + 1
            run_panel_half(p0, 0, jnp.bool_(True), pair_idx > 0, pair_idx >= 2)
            run_panel_half(
                p1, 1, pair_idx < n_pairs - 1, jnp.bool_(True), pair_idx >= 1
            )

        finish_reduction(jnp.int32(n_panels - 1), 1, jnp.bool_(True))
        for tail_panel in (n_panels - 2, n_panels - 1):
            s_bank = tail_panel % 2
            pltpu.make_async_copy(
                ds["shared_tiles"][s_bank],
                shared_hbm.at[:, pl.ds(tail_panel * panel_cols, panel_cols)],
                ds["shared_sems"][s_bank],
            ).wait()

    pl.run_scoped(
        run_down,
        dict(
            down_buffers=tuple(
                vmem((SHARED_INTER_16K, panel_cols), jnp.bfloat16) for _ in range(2)
            ),
            down_sems=(dma, dma),
            partial=tuple(vmem((2 * rows, panel_cols), jnp.float32) for _ in range(2)),
            peer_y=tuple(vmem((rows, panel_cols), jnp.float32) for _ in range(2)),
            reduce_send=(dma, dma),
            reduce_recv=(dma, dma),
            shared_tiles=tuple(
                vmem((rows, panel_cols), jnp.bfloat16) for _ in range(2)
            ),
            shared_sems=(dma, dma),
        ),
    )


def run_deferred_shared_down_from_hbm(
    shared_hbm,
    shared_down_hbm,
    down_ready_sem,
    *,
    rows: int,
    parity,
    peer,
    on_poll: Callable[[], None] | None = None,
) -> None:
    assert rows % 128 == 0
    assert shared_hbm.shape == (rows, SHARED_HIDDEN_16K)
    assert shared_down_hbm.shape == (SHARED_INTER_16K, SHARED_HIDDEN_16K)
    vmem, dma = pltpu.VMEM, pltpu.SemaphoreType.DMA

    def run_staged(ms) -> None:
        load_h0 = pltpu.make_async_copy(
            shared_hbm.at[:, pl.ds(0, SHARED_INTER_16K)],
            ms["middle_v"].at[pl.ds(0, rows), :],
            ms["load_sems"][0],
        )
        load_h1 = pltpu.make_async_copy(
            shared_hbm.at[:, pl.ds(SHARED_INTER_16K, SHARED_INTER_16K)],
            ms["middle_v"].at[pl.ds(rows, rows), :],
            ms["load_sems"][1],
        )
        load_h0.start()
        load_h1.start()
        if on_poll is not None:
            on_poll()
        load_h0.wait()
        load_h1.wait()
        if on_poll is not None:
            on_poll()
        run_shared_down_512_poll2(
            ms["middle_v"],
            shared_down_hbm,
            shared_hbm,
            down_ready_sem,
            rows=rows,
            parity=parity,
            peer=peer,
            on_poll=on_poll,
        )

    pl.run_scoped(
        run_staged,
        dict(
            middle_v=vmem((2 * rows, SHARED_INTER_16K), jnp.bfloat16),
            load_sems=(dma, dma),
        ),
    )
