# SPDX-License-Identifier: Apache-2.0
"""Host tier check mode: entry checksums compared after a load (smokes only).

Turned on by VLLM_TPU_OFFLOAD_TIER_CHECK=1, with the tier classes. The scheduler names
every entry a class store saved, with its source block, and every key a load restored,
with its destination block (``TieredOffloadingManager.take_check_requests``, shipped in
the step's connector metadata). Each worker reads the rows its class owns of that block
in every region the class registers on this worker, before the step's forward, and
hashes them: at a store the digest is kept by key, at a load it is compared with the
kept one. For the compressed-key class with ``VLLM_TPU_OFFLOAD_TIER_KEY_ROWS`` the
digest covers only a page's leading live bytes, the bytes that class moves, so a load
that writes another row shows as a mismatch.

Timing: a victim store has drained before its metadata ships and the block's new owner
writes it in that step's forward or later; a load is reported in the metadata of the
step before its request runs again. Both reads happen at ``start_load_kv``, ahead of the
forward on the same stream, so they see the saved and the restored bytes. A load the
same step submits is the exception: its DMA is not ordered with the stream and can land
in a victim's source block before the workers read it, so the scheduler sends such
entries as block -1 and the workers forget the key (counted as ``unhashed``).

Rows are read with the PD landing verify's DMA row gather (``pd_staging.
dma_row_gather``), byte for byte, and copied to the host: every store and load waits on
the device. Never on in a measured run.
"""

import hashlib
import math
from collections.abc import Callable, Iterable

import numpy as np
import torch

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Counters a class reports, in order.
CHECK_FIELDS = ("compared", "mismatched", "missing", "errors")

RowReader = Callable[[torch.Tensor, int, int], np.ndarray]


def _row_shape(region: torch.Tensor) -> tuple[int, int]:
    return int(region.shape[-2]), int(region.shape[-1])


def read_rows_fn(interpret: bool = False):
    """``(pool, idx) -> pool rows idx`` as uint8, the pool viewed as ``[rows, *minor]``,
    through the PD landing verify's byte-exact DMA row gather."""
    import functools

    import jax
    import jax.numpy as jnp

    from vllm_torchtpu.models.vllm.qwen4_exp.pd_staging import dma_row_gather

    gather = functools.partial(dma_row_gather, out_dtype=jnp.uint8, interpret=interpret)

    # jax_op reads the argument types from these annotations, which is why this
    # module has no `from __future__ import annotations`.
    def read_fn(pool: jax.Array, idx: jax.Array) -> jax.Array:
        return gather(pool.reshape(-1, pool.shape[-2], pool.shape[-1]), idx)

    return read_fn


def read_rows_op():
    """The row read as a torch op (``pallas.jax_op``)."""
    from torch_tpu._internal import pallas

    op = pallas.jax_op("pallas::qwen4_tier_check_read_rows", read_rows_fn())
    op.register_fake(
        lambda pool, idx: torch.empty(
            (idx.shape[0], pool.shape[-2], pool.shape[-1]),
            dtype=torch.uint8,
            device=idx.device,
        )
    )
    return op


def device_row_reader() -> RowReader:
    """Reads rows ``[first, first + count)`` of a region viewed as ``[rows, *minor]``
    to the host as uint8, through one compiled program per row count."""
    programs: dict[int, Callable] = {}
    built: list[Callable] = []

    def read(region: torch.Tensor, first: int, count: int) -> np.ndarray:
        run = programs.get(count)
        if run is None:
            if not built:
                built.append(read_rows_op())
            op = built[0]

            @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
            def run(pool, idx):
                return op(pool, idx)

            programs[count] = run
        idx = torch.arange(first, first + count, dtype=torch.int32).to(region.device)
        return run(region, idx).cpu().numpy()

    return read


class TierCheck:
    """One worker's digests of class entries and its pending counts."""

    def __init__(
        self,
        regions: dict[str, list[torch.Tensor]],
        live_bytes: dict[str, int],
        rank: int,
        read_rows: RowReader | None = None,
    ):
        # Class -> the raw region tensors ([num_blocks, *page]) it registers here.
        self._regions = {name: list(r) for name, r in regions.items() if r}
        self._live = dict(live_bytes)
        self._rank = rank
        self._read = read_rows if read_rows is not None else device_row_reader()
        self._digests: dict[bytes, bytes] = {}
        self._counts: dict[str, list[int]] = {}
        self._logged_errors = 0

    def digest(self, name: str, block: int) -> bytes:
        """Hash of the bytes class ``name`` moves for ``block`` on this worker."""
        h = hashlib.blake2b(digest_size=16)
        live = self._live[name]
        for region in self._regions[name]:
            sublanes, lanes = _row_shape(region)
            row_bytes = sublanes * lanes * region.element_size()
            page_bytes = math.prod(region.shape[1:]) * region.element_size()
            rows = -(-live // row_bytes)
            data = self._read(region, block * (page_bytes // row_bytes), rows)
            h.update(
                np.ascontiguousarray(data).view(np.uint8).reshape(-1)[:live].tobytes()
            )
        return h.digest()

    def _count(self, name: str, field: str) -> None:
        counts = self._counts.setdefault(name, [0] * len(CHECK_FIELDS))
        counts[CHECK_FIELDS.index(field)] += 1

    def run(
        self,
        stores: Iterable[tuple[str, bytes, int]],
        loads: Iterable[tuple[str, bytes, int]],
    ) -> None:
        """Keeps the digests of saved entries, then compares restored ones."""
        for name, key, block in stores:
            if name not in self._regions:
                continue
            if block < 0:
                # Saved, but its source block could not be read safely
                # (raiden_connector._unhashable_as_forget): drop any digest an
                # earlier save of the key left.
                self._digests.pop(key, None)
                continue
            try:
                self._digests[key] = self.digest(name, block)
            except Exception:  # a diagnostic never fails the step
                self._error(name, "store", key, block)
        for name, key, block in loads:
            if name not in self._regions:
                continue
            kept = self._digests.get(key)
            if kept is None:
                self._count(name, "missing")
                continue
            try:
                got = self.digest(name, block)
            except Exception:  # a diagnostic never fails the step
                self._error(name, "load", key, block)
                continue
            self._count(name, "compared")
            if got != kept:
                self._count(name, "mismatched")
                logger.warning(
                    "tier check: rank %d class %s key %s restored into block %d "
                    "differs from what was saved (%s, saved %s)",
                    self._rank,
                    name,
                    key.hex(),
                    block,
                    got.hex(),
                    kept.hex(),
                )

    def _error(self, name: str, what: str, key: bytes, block: int) -> None:
        self._count(name, "errors")
        if self._logged_errors < 5:
            self._logged_errors += 1
            logger.exception(
                "tier check: rank %d class %s could not read the %s rows of key %s, "
                "block %d",
                self._rank,
                name,
                what,
                key.hex(),
                block,
            )

    def take_counts(self) -> dict[str, list[int]]:
        counts, self._counts = self._counts, {}
        return counts
