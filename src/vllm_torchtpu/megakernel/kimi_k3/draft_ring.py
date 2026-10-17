# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSpark draft's ring, computed by the prefill program.

The decoder's draft (:mod:`draft`) attends a ring of the last 2048 positions'
projected keys and values of the target's residual streams entering the aux
layers (24, 48, 72, 88, 92); a decoder that starts it empty drafts poorly until
it fills. :class:`PrefillRing` computes it window by window from the prefill
stack's residual streams, the pool writer stores it in the draft rows of the
request's KDA state blocks (``pools.write_draft_rings``), and the KV transfer
carries it to ``KdaImporter.import_ring``. Each rank gathers the draft's
``fc``, ``kv``, ``kvnorm`` and ``hidden_norm`` shards (``dspark-<revision>`` in
the pre-shard directory) into replicated weights and projects its own window
rows for every KV head (``kernels.dspark.prefill_ring_update``).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import torch
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.megakernel.bridge import hbm, materialize, register_op, upload
from vllm_torchtpu.megakernel.kimi_k3.program import kernel_modules

logger = init_logger(__name__)

# The residual streams the prefill window ops expose for the aux layers:
# ``blocks`` slot L // 12 holds the stream entering layer L (L a multiple of
# 12), the regular pair 10's middle output enters layer 88, and the prefix
# into the final op enters layer 92.
AUX_LAYERS = (24, 48, 72, 88, 92)
WEIGHT_ARRAYS = ("fc", "kv", "kvnorm", "hidden_norm")


class PrefillRing:
    """The prefill side of the DSpark ring (see the module docstring)."""

    def __init__(
        self, mesh, rank: int, axis: str, checkpoint: Path, presharded: Path
    ) -> None:
        (self.dspark,) = kernel_modules("dspark")
        self.config = self.dspark.DraftConfig.from_checkpoint(checkpoint)
        if tuple(self.config.aux_layers) != AUX_LAYERS:
            raise NotImplementedError(
                f"the prefill ring taps aux layers {AUX_LAYERS}; the draft "
                f"{checkpoint} reads {self.config.aux_layers}"
            )
        self.mesh, self.rank, self.axis = mesh, rank, axis
        self._directory = (
            Path(presharded) / f"dspark-{Path(checkpoint).resolve().name[:8]}"
        )
        self.weights: tuple[torch.Tensor, ...] = ()
        self._ops: dict[Any, Any] = {}
        self._zero: torch.Tensor | None = None

    def load_weights(self) -> None:
        """Read this rank's four arrays of the draft's kernel shard and gather
        the replicated weights."""
        import time

        started = time.perf_counter()
        index = json.loads((self._directory / "index.json").read_text())
        if index.get("format") != self.dspark.KERNEL_SHARD_FORMAT:
            raise ValueError(f"{self._directory}: not a DSpark kernel shard set")
        raw = np.memmap(
            self._directory / f"rank{self.rank:02d}.bin", dtype=np.uint8, mode="r"
        )
        shards = []
        for name in WEIGHT_ARRAYS:
            spec = index["arrays"][name]
            dtype = (
                np.dtype(jnp.bfloat16)
                if spec["dtype"] == "bfloat16"
                else np.dtype(spec["dtype"])
            )
            count = int(np.prod(spec["shape"]))
            value = np.array(
                raw[spec["offset"] : spec["offset"] + count * dtype.itemsize]
            )
            shards.append(upload(value.view(dtype).reshape(spec["shape"])))
        c, axis = self.config, self.axis

        def gather(fc, kv, kvnorm, hidden_norm):  # [1, ...]: this rank's shards
            out = self.dspark.ring_weights_from_shards(
                c, fc[0], kv[0], kvnorm[0], hidden_norm[0], axis
            )
            return tuple(out[name] for name in ("fc", "kv", "k_norm", "hidden_norm"))

        op = register_op(
            "pallas::k3_prefill_draft_ring_weights",
            self.mesh,
            gather,
            (P(axis),) * 4,
            (P(),) * 4,
        )
        self.weights = tuple(op(*shards))
        materialize(self.weights[0])
        if self.rank == 0:
            gib = sum(t.numel() * t.element_size() for t in self.weights) / 2**30
            logger.info(
                "K3 prefill: DSpark ring weights resident (%.2f GiB per rank) in "
                "%.1f s; HBM %s",
                gib,
                time.perf_counter() - started,
                hbm(self.rank),
            )

    def empty(self, start: int) -> tuple[torch.Tensor, torch.Tensor]:
        """A request's ring before any of its rows: ``(cache [1, layers,
        window, 128], meta [2])`` -- it covers ``start`` tokens and holds
        none of them."""
        c = self.config
        cache = torch.zeros(
            (1, c.layers, c.window, 2 * c.head_dim), dtype=torch.bfloat16
        ).to("tpu")
        meta = torch.tensor([start, start], dtype=torch.int32).to("tpu")
        return cache, meta

    def zero(self) -> tuple[torch.Tensor, torch.Tensor]:
        """A shared empty ring for a window's unused sequence slots (never
        donated)."""
        if self._zero is None:
            self._zero = self.empty(0)
        return self._zero

    def update(
        self,
        blocks: torch.Tensor,
        prefix: torch.Tensor,
        middle: torch.Tensor,
        positions: np.ndarray,
        sequences: np.ndarray,
        ends: list[int],
        stops: list[int],
        rings: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[list, list]:
        """One packed window (``slots`` sequences): its residual streams
        (``blocks``, the pair-10 ``middle``, the ``prefix`` into the final
        op) into each sequence's ring. ``positions`` / ``sequences``
        ``[window]`` per window row (``-1``: padding); ``ends`` / ``stops``
        per sequence (``-1``: no checkpoint). Returns the ``(cache, meta)``
        rings after the window and at the checkpoints."""
        slots, window = len(rings), int(positions.shape[0])

        def device(values) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(values, np.int32)).to("tpu")

        out = self._update_op(window, slots)(
            blocks,
            prefix,
            middle,
            device(positions),
            device(sequences),
            device(ends),
            device(stops),
            *self.weights,
            *(cache for cache, _ in rings),
            *(meta for _, meta in rings),
        )
        after = list(zip(out[:slots], out[2 * slots : 3 * slots], strict=True))
        at = list(zip(out[slots : 2 * slots], out[3 * slots :], strict=True))
        return after, at

    def _update_op(self, window: int, slots: int) -> Any:
        """:meth:`update`'s op for a ``window`` of ``slots`` sequences."""
        key = ("update", window, slots)
        if key not in self._ops:
            c, axis, dspark = self.config, self.axis, self.dspark

            def body(blocks, prefix, middle, positions, sequences, ends, stops, *rest):
                weights = dict(
                    zip(("fc", "kv", "k_norm", "hidden_norm"), rest[:4], strict=True)
                )
                caches, metas = rest[4 : 4 + slots], rest[4 + slots :]
                aux = jnp.stack(
                    (blocks[0, 2], blocks[0, 4], blocks[0, 6], middle[0], prefix[0])
                )
                after, at = dspark.prefill_ring_update(
                    c,
                    weights,
                    aux,
                    positions,
                    sequences,
                    jnp.stack([cache[0] for cache in caches]),
                    ends,
                    stops,
                    axis,
                )
                firsts = [meta[1] for meta in metas]
                return (
                    *(after[s][None] for s in range(slots)),
                    *(at[s][None] for s in range(slots)),
                    *(jnp.stack((ends[s], firsts[s])) for s in range(slots)),
                    *(jnp.stack((stops[s], firsts[s])) for s in range(slots)),
                )

            self._ops[key] = register_op(
                f"pallas::k3_prefill_draft_ring_w{window}_s{slots}",
                self.mesh,
                body,
                (P(axis),) * 3
                + (P(),) * 4
                + (P(),) * 4
                + (P(axis),) * slots
                + (P(),) * slots,
                (P(axis),) * (2 * slots) + (P(),) * (2 * slots),
                export_key=repr(
                    ("prefill_draft_ring", window, slots, dataclasses.astuple(c))
                ),
            )
        return self._ops[key]
