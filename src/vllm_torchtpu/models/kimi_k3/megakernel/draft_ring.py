# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

# Source provenance: inferact/k3-megakernel@30c9ba966b
# src/vllm_torchtpu/megakernel/kimi_k3/draft_ring.py
"""The DSpark draft's ring computed by the prefill program.

The decoder's draft (:mod:`draft`) attends a ring of the last 2048
positions' projected keys and values of the target's residual streams
entering the aux layers (24, 48, 72, 88, 92); a decoder that starts it empty
drafts poorly until it fills. :class:`PrefillRing` computes it window by
window from the prefill stack's residual streams, the pool writer stores it
in the draft rows of the request's KDA state blocks
(``pools.write_draft_rings``), and the KV transfer carries it to
``KdaImporter.import_ring``. Each rank gathers the draft's ``fc``, ``kv``,
``kvnorm`` and ``hidden_norm`` shards (``dspark-<revision>`` in the
pre-shard directory) into replicated weights and projects its own window
rows for every KV head (``kernels/dspark.py`` ``prefill_ring_update``).
"""

from __future__ import annotations

import dataclasses
import json
import mmap
import time
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.logger import init_logger

from .common.constants import GIB_BYTES, REV_PREFIX_LEN
from .common.paths import require_local_path
from .program import kernel_modules
from .runtime_shim.bridge import (
    OpRegistrationSpec,
    hbm,
    materialize,
    op_key_suffix,
    register_op,
    upload,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

logger = init_logger(__name__)

# The residual streams the prefill window ops expose for the aux layers: ``blocks`` slot
# L // 12 holds the stream entering layer L (L a multiple of 12), the regular pair 10's
# middle output enters layer 88, and the prefix into the final op enters layer 92.
AUX_LAYERS = (24, 48, 72, 88, 92)
WEIGHT_ARRAYS = ("fc", "kv", "kvnorm", "hidden_norm")
_UPDATE_META_ARRAYS = 4


class PrefillRing:
    """The prefill side of the DSpark ring."""

    def __init__(
        self,
        mesh: Mesh,
        rank: int,
        axis: str,
        checkpoint: Path | str,
        presharded: Path | str,
    ) -> None:
        """Initialize ring projector for ``mesh``, ``rank``, and ``checkpoint``."""
        checkpoint_path = require_local_path(checkpoint, label="checkpoint")
        presharded_path = require_local_path(presharded, label="presharded")
        (self.dspark,) = kernel_modules("dspark")
        self.config = self.dspark.DraftConfig.from_checkpoint(checkpoint_path)
        if tuple(self.config.aux_layers) != AUX_LAYERS:
            msg = (
                f"the prefill ring taps aux layers {AUX_LAYERS}; the draft "
                f"{checkpoint_path} reads {self.config.aux_layers}"
            )
            raise NotImplementedError(msg)
        self.mesh, self.rank, self.axis = mesh, rank, axis
        rev = checkpoint_path.resolve().name[:REV_PREFIX_LEN]
        self._directory = presharded_path / f"dspark-{rev}"
        self.weights: tuple[torch.Tensor, ...] = ()
        self._ops: dict[
            tuple[object, ...], Callable[..., tuple[torch.Tensor, ...]]
        ] = {}
        self._zero: tuple[torch.Tensor, torch.Tensor] | None = None

    def _device(self) -> torch.device | str:
        """Return the resident device of loaded ring weights, or ``"tpu"``."""
        return self.weights[0].device if self.weights else "tpu"

    def load_weights(self) -> None:
        """Read this rank's four draft shard arrays and gather replicated weights."""
        started = time.perf_counter()
        index = json.loads((self._directory / "index.json").read_text(encoding="utf-8"))
        if index.get("format") != self.dspark.KERNEL_SHARD_FORMAT:
            msg = f"{self._directory}: not a DSpark kernel shard set"
            raise ValueError(msg)
        shards: list[torch.Tensor] = []
        with (
            (self._directory / f"rank{self.rank:02d}.bin").open("rb") as handle,
            mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mm,
        ):
            for name in WEIGHT_ARRAYS:
                spec = index["arrays"][name]
                dtype = (
                    np.dtype(jnp.bfloat16)
                    if spec["dtype"] == "bfloat16"
                    else np.dtype(spec["dtype"])
                )
                count = int(np.prod(spec["shape"]))
                offset = int(spec["offset"])
                value = np.frombuffer(
                    mm,
                    dtype=np.uint8,
                    count=count * dtype.itemsize,
                    offset=offset,
                ).copy()
                shards.append(upload(value.view(dtype).reshape(spec["shape"])))
        cfg, axis = self.config, self.axis

        def gather(  # [1, ...]: this rank's shards
            fc: jax.Array,
            kv: jax.Array,
            kvnorm: jax.Array,
            hidden_norm: jax.Array,
        ) -> tuple[jax.Array, ...]:
            """Gather per-rank draft ring weight shards across ``axis``."""
            out = self.dspark.ring_weights_from_shards(
                cfg,
                self.dspark.DsparkRingWeightShards(
                    fc=fc[0],
                    kv=kv[0],
                    kvnorm=kvnorm[0],
                    hidden_norm=hidden_norm[0],
                ),
                axis,
            )
            return tuple(out[name] for name in ("fc", "kv", "k_norm", "hidden_norm"))

        op = register_op(
            "pallas::k3_prefill_draft_ring_weights",
            self.mesh,
            gather,
            OpRegistrationSpec(
                in_specs=(P(axis),) * len(WEIGHT_ARRAYS),
                out_specs=(P(),) * len(WEIGHT_ARRAYS),
            ),
        )
        self.weights = tuple(op(*shards))
        materialize(self.weights[0])
        if self.rank == 0:
            gib = sum(t.numel() * t.element_size() for t in self.weights) / GIB_BYTES
            logger.info(
                "K3 prefill: DSpark ring weights resident (%.2f GiB per rank) in "
                "%.1f s; HBM %s",
                gib,
                time.perf_counter() - started,
                hbm(self.rank),
            )

    def empty(self, start: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return an empty ``(cache, meta)`` ring starting at token ``start``.

        Shapes are ``cache [1, layers, window, 128]`` and ``meta [2]`` covering
        ``start`` tokens while holding none of them.
        """
        cfg = self.config
        dev = self._device()
        cache = torch.zeros(
            (1, cfg.layers, cfg.window, 2 * cfg.head_dim),
            dtype=torch.bfloat16,
            device=dev,
        )
        meta = torch.tensor([start, start], dtype=torch.int32, device=dev)
        return cache, meta

    def zero(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a shared empty ring for unused window sequence slots."""
        if self._zero is None:
            self._zero = self.empty(0)
        return self._zero

    def update(
        self,
        blocks: torch.Tensor,
        prefix: torch.Tensor,
        middle: torch.Tensor,
        positions: np.ndarray | Sequence[int],
        sequences: np.ndarray | Sequence[int],
        ends: Sequence[int],
        stops: Sequence[int],
        rings: Sequence[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[
        list[tuple[torch.Tensor, torch.Tensor]],
        list[tuple[torch.Tensor, torch.Tensor]],
    ]:
        """Update ``rings`` from one packed prefill window's residual streams.

        Projects ``blocks``, pair-10 ``middle``, and final-op ``prefix`` into
        each sequence's ring. ``positions`` and ``sequences`` are
        ``[window]`` per row (``-1``: padding); ``ends`` and ``stops`` are per
        sequence (``-1``: no checkpoint). Returns ``(cache, meta)`` rings after
        the window and at checkpoints.
        """
        positions_arr = np.ascontiguousarray(positions, np.int32)
        sequences_arr = np.ascontiguousarray(sequences, np.int32)
        slots, window = len(rings), int(positions_arr.size)

        dev = blocks.device

        def to_device(values: np.ndarray | Sequence[int]) -> torch.Tensor:
            """Upload int32 array-like ``values`` to ``blocks.device``."""
            return torch.from_numpy(np.ascontiguousarray(values, np.int32)).to(dev)

        out = self._update_op(window, slots)(
            blocks,
            prefix,
            middle,
            to_device(positions_arr),
            to_device(sequences_arr),
            to_device(ends),
            to_device(stops),
            *self.weights,
            *(cache for cache, _ in rings),
            *(meta for _, meta in rings),
        )
        after = list(zip(out[:slots], out[2 * slots : 3 * slots], strict=True))
        at = list(zip(out[slots : 2 * slots], out[3 * slots :], strict=True))
        return after, at

    def update_op(
        self,
        window: int,
        count: int,
    ) -> Callable[..., tuple[torch.Tensor, ...]]:
        """Return the cached ring-update custom op for ``window`` and ``count``."""
        return self._update_op(window, count)

    def _update_op(
        self,
        window: int,
        slots: int,
    ) -> Callable[..., tuple[torch.Tensor, ...]]:
        """Build or return the cached :meth:`update` op for ``window`` and ``slots``."""
        key = ("update", window, slots)
        if key not in self._ops:
            cfg, axis, dspark = self.config, self.axis, self.dspark

            def body(
                blocks: jax.Array,
                *args: jax.Array,
            ) -> tuple[jax.Array, ...]:
                """Project window residual streams into each sequence's draft ring."""
                prefix, middle, positions, sequences, ends, stops = args[:6]
                rest = args[6:]
                num_weights = len(WEIGHT_ARRAYS)
                weights = dict(
                    zip(
                        ("fc", "kv", "k_norm", "hidden_norm"),
                        rest[:num_weights],
                        strict=True,
                    )
                )
                caches = rest[num_weights : num_weights + slots]
                metas = rest[num_weights + slots :]
                aux = jnp.stack(
                    (blocks[0, 2], blocks[0, 4], blocks[0, 6], middle[0], prefix[0])
                )
                after, at = dspark.prefill_ring_update(
                    cfg,
                    weights,
                    dspark.DsparkRingUpdateSpec(
                        aux=aux,
                        positions=positions,
                        sequences=sequences,
                        rings=jnp.stack([cache[0] for cache in caches]),
                        ends=ends,
                        stops=stops,
                    ),
                    axis,
                )
                firsts = [meta[1] for meta in metas]
                return (
                    *(after[s][None] for s in range(slots)),
                    *(at[s][None] for s in range(slots)),
                    *(jnp.stack((ends[s], firsts[s])) for s in range(slots)),
                    *(jnp.stack((stops[s], firsts[s])) for s in range(slots)),
                )

            export_key_tuple = (
                "prefill_draft_ring",
                window,
                slots,
                dataclasses.astuple(cfg),
            )
            self._ops[key] = register_op(
                f"pallas::k3_prefill_draft_ring_w{window}_s{slots}_{op_key_suffix(export_key_tuple)}",
                self.mesh,
                body,
                OpRegistrationSpec(
                    in_specs=(P(axis),) * 3
                    + (P(),) * _UPDATE_META_ARRAYS
                    + (P(),) * len(WEIGHT_ARRAYS)
                    + (P(axis),) * slots
                    + (P(),) * slots,
                    out_specs=(P(axis),) * (2 * slots) + (P(),) * (2 * slots),
                    export_key=repr(export_key_tuple),
                ),
            )
        return self._ops[key]
