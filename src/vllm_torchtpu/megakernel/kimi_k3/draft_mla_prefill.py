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
"""The MLA DSpark draft's context latents from a prefill window.

Every prompt token the prefill runs gets, in each of the draft's five layers,
the latent ``[kv_a_layernorm(kv[:512]) | rope(kv[512:576])]`` of ``kv =
kv_a(context_norm(context_proj(aux)))``, ``aux`` the residual streams
entering the draft's aux layers (K3: 3, 24, 48, 72, 90; K3.1: 76, 80, 84, 88,
92; ``kernels.dspark_mla``). Each
sequence-parallel rank projects its own window rows (replicated weights:
``context_proj`` gathered once from the draft's kernel shards), the ranks'
latents are gathered into window order, and every rank writes the window
pages it owns into the draft's pools -- as the prefill writes the target's MLA
latents -- so the KV transfer carries them to the decoder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.megakernel.bridge import hbm, register_op, upload
from vllm_torchtpu.megakernel.kimi_k3.program import kernel_modules

logger = init_logger(__name__)

ATTN_RES_BLOCK = 12
SNAPSHOTS = 8  # the AttnRes block snapshots the prefill keeps (layers 0, 12, .., 84)


def is_snapshot(layer: int) -> bool:
    """Whether the stream entering ``layer`` is a block snapshot (``blocks``
    slot ``layer // 12``) rather than a tapped stream."""
    return layer % ATTN_RES_BLOCK == 0 and layer // ATTN_RES_BLOCK < SNAPSHOTS


class PrefillDraftLatents:
    """See the module docstring."""

    def __init__(
        self,
        mesh,
        rank: int,
        axis: str,
        checkpoint: Path,
        presharded: Path,
        max_position: int,
        fc_ranks: tuple[int, ...] | None = None,
    ):
        """``fc_ranks``: the TP32 ranks whose context-projection shards this
        rank holds (its own by default; a pipeline stage's core holds the
        ``32 / mesh size`` ranks it stands for)."""
        (self.mla,) = kernel_modules("dspark_mla")
        self.mesh, self.rank, self.axis = mesh, rank, axis
        self.fc_ranks = (rank,) if fc_ranks is None else tuple(fc_ranks)
        self.ranks = self.mesh.size
        self.config = self.mla.MlaDraftConfig.from_checkpoint(Path(checkpoint))
        self.directory = (
            Path(presharded) / f"dspark-{Path(checkpoint).resolve().name[:8]}"
        )
        self.max_position = int(max_position)
        self.weights: dict[str, torch.Tensor] = {}
        self.pools: list[torch.Tensor] = []
        self._ops: dict[Any, Any] = {}

    def load_weights(self) -> None:
        c = self.config
        shards = self.mla.load_rank_shards(
            self.directory, self.fc_ranks[0], names={"qkv_a", "kv_norm", "context_norm"}
        )
        lanes = self.mla.LANES
        self.weights["kv_a"] = upload(
            np.ascontiguousarray(shards["qkv_a"][:, :, c.q_lora : c.q_lora + lanes])
        )
        self.weights["kv_norm"] = upload(np.ascontiguousarray(shards["kv_norm"]))
        self.weights["context_norm"] = upload(
            np.ascontiguousarray(shards["context_norm"])
        )
        self.weights["rope"] = upload(self.mla.rope_table(c, self.max_position + 96))
        fc = upload(
            np.stack(
                [
                    self.mla.load_rank_shards(self.directory, r, names={"fc"})["fc"]
                    for r in self.fc_ranks
                ]
            )
        )  # [1, k, 8960, 896]: TP32 rank 8 h + j: host h's inputs, lane j's outputs
        chunk, fc_in = self.mla.CHUNK, self.mla.FC_IN
        axis, held = self.axis, len(self.fc_ranks)

        def gather(fc):
            import jax

            full = jax.lax.all_gather(fc[0], axis)  # [ranks, k, 8960, 896]
            full = full.reshape(-1, fc_in, chunk)  # [32, 8960, 896] in TP32 rank order
            full = (
                full.reshape(4, 8, fc_in, chunk)
                .transpose(0, 2, 1, 3)
                .reshape(4 * fc_in, 8 * chunk)
            )
            return (full[None],)

        op = register_op(
            f"pallas::k3_dspark_mla_gather_context_proj_r{self.ranks}",
            self.mesh,
            gather,
            (P(axis),),
            (P(axis),),
            export_key=repr(("dspark_mla_gather_fc", fc_in, chunk, self.ranks, held)),
        )
        (self.weights["context_proj"],) = op(fc)
        if self.rank == 0:
            logger.info(
                "K3 prefill: MLA DSpark context weights resident; HBM %s",
                hbm(self.rank),
            )

    def bind(self, pools: list[torch.Tensor]) -> None:
        if len(pools) != self.config.layers:
            raise RuntimeError(
                f"K3 prefill: the MLA DSpark draft needs {self.config.layers} draft KV "
                f"cache layers, found {len(pools)}"
            )
        self.pools = list(pools)

    def copy_blocks(self, pairs: list[tuple[int, int]], per_call: int) -> None:
        """vLLM's copy-on-write block copies (``PoolWriter.copy_blocks``) in
        the draft pools: a prefix hit ending inside a cached block reads the
        cached tokens' draft latents from its private copy too."""
        for first in range(0, len(pairs), per_call):
            chunk = list(pairs[first : first + per_call])
            chunk += [(0, 0)] * (per_call - len(chunk))
            source = torch.tensor([a for a, _ in chunk], dtype=torch.int32).to("tpu")
            destination = torch.tensor([b for _, b in chunk], dtype=torch.int32)
            pools = self._copy_op(per_call)(*self.pools, source, destination.to("tpu"))
            for old, fresh in zip(self.pools, pools, strict=True):
                if fresh is not old:
                    old.set_(fresh)

    def _copy_op(self, per_call: int) -> Any:
        key = ("copy", per_call)
        if key in self._ops:
            return self._ops[key]
        layers = self.config.layers

        def body(*args):
            pools, source, destination = args[:layers], args[layers], args[layers + 1]
            return tuple(pool.at[destination].set(pool[source]) for pool in pools)

        self._ops[key] = register_op(
            f"pallas::k3_dspark_mla_prefill_copy_blocks_{per_call}",
            self.mesh,
            body,
            (P(self.axis),) * layers + (P(), P()),
            (P(self.axis),) * layers,
            donate_argnums=tuple(range(layers)),
            export_key=repr(("dspark_mla_prefill_copy_blocks", layers, per_call)),
        )
        return self._ops[key]

    @property
    def tapped(self) -> tuple[int, ...]:
        """The aux layers whose streams :meth:`latents` takes (the others are
        block snapshots)."""
        return tuple(layer for layer in self.config.aux_layers if not is_snapshot(layer))

    def _latents_op(self, window: int) -> Any:
        key = ("latents", window)
        if key in self._ops:
            return self._ops[key]
        mla, c, axis = self.mla, self.config, self.axis
        layers = c.layers
        tapped = self.tapped

        def body(blocks, *args):
            import jax
            import jax.numpy as jnp

            taps, args = args[: len(tapped)], args[len(tapped) :]
            context_proj, kv_a, kv_norm, context_norm, rope, positions, inverse = args
            rank = jax.lax.axis_index(axis)
            local = blocks.shape[2]
            own = jnp.maximum(
                jax.lax.dynamic_slice_in_dim(positions, rank * local, local), 0
            )
            streams = dict(zip(tapped, (tap[0] for tap in taps), strict=True))
            features = jnp.concatenate(
                [
                    blocks[0, layer // ATTN_RES_BLOCK]
                    if is_snapshot(layer)
                    else streams[layer]
                    for layer in c.aux_layers
                ],
                axis=-1,
            )  # [local, aux layers x hidden]
            context = jnp.dot(
                features, context_proj[0], preferred_element_type=jnp.float32
            ).astype(jnp.bfloat16)
            context = mla.rms_norm(context, context_norm[0][0], c.eps)
            table = rope[0][own]  # [local, 2 rope]: cos2 | sin2
            cos, sin = table[:, : c.rope : 2], table[:, c.rope :: 2]
            rows = []
            for layer in range(layers):
                kv = jnp.dot(
                    context, kv_a[0, layer], preferred_element_type=jnp.float32
                ).astype(jnp.bfloat16)
                c_kv = mla.rms_norm(
                    kv[:, : c.kv_lora], kv_norm[0, layer, 0, : c.kv_lora], c.eps
                )
                k_pe = mla.apply_rope(kv[:, c.kv_lora : c.latent], cos, sin)
                rows.append(jnp.concatenate((c_kv, k_pe), axis=-1))
            latents = jnp.stack(rows)  # [layers, local, 576]
            latents = jax.lax.all_gather(
                latents, axis, axis=1, tiled=True
            )  # [layers, W] ownership order
            return (jnp.take(latents, inverse, axis=1)[None],)  # window order

        self._ops[key] = register_op(
            f"pallas::k3_dspark_mla_prefill_latents_r{self.ranks}_w{window}",
            self.mesh,
            body,
            (P(axis),) * (6 + len(tapped)) + (P(), P()),
            (P(axis),),
            export_key=repr(
                (
                    "dspark_mla_prefill_latents",
                    self.ranks,
                    window,
                    layers,
                    c.aux_layers,
                    self.max_position,
                )
            ),
        )
        return self._ops[key]

    def latents(
        self,
        window: int,
        blocks,
        streams: dict[int, Any],
        positions: np.ndarray,
        order: np.ndarray,
    ):
        """The window's draft latents ``[1, layers, W, 576]`` (window order, every
        rank): ``blocks`` the block snapshots, ``streams`` this rank's rows of
        the stream entering each :attr:`tapped` layer (``[1, W / ranks,
        hidden]``), ``positions`` the window rows' positions in window order
        (-1: padding), ``order`` the ranks' rows (``PrefillProgram.sp_order``)."""
        inverse = np.argsort(order).astype(np.int32)
        (latents,) = self._latents_op(window)(
            blocks,
            *(streams[layer] for layer in self.tapped),
            *(
                self.weights[n]
                for n in ("context_proj", "kv_a", "kv_norm", "context_norm", "rope")
            ),
            torch.from_numpy(
                np.ascontiguousarray(positions[order]).astype(np.int32)
            ).to("tpu"),
            torch.from_numpy(inverse).to("tpu"),
        )
        return latents

    def _write_op(self, window: int, block_shape: tuple[int, ...]) -> Any:
        key = ("write", window, block_shape)
        if key in self._ops:
            return self._ops[key]
        layers = self.config.layers
        pages = window // (block_shape[0] * block_shape[1])

        def body(latents, table, *pools):
            import jax
            import jax.numpy as jnp

            table, latents, pools = table[0], latents[0], list(pools)
            for layer in range(layers):
                rows = (
                    jnp.pad(latents[layer], ((0, 0), (0, 64)))
                    .astype(pools[layer].dtype)
                    .reshape(pages, *block_shape)
                )

                def write_page(p, pool, rows=rows):
                    return jax.lax.dynamic_update_index_in_dim(
                        pool,
                        jax.lax.dynamic_index_in_dim(rows, p, keepdims=False),
                        table[p],
                        axis=0,
                    )

                pools[layer] = jax.lax.fori_loop(0, pages, write_page, pools[layer])
            return tuple(pools)

        self._ops[key] = register_op(
            f"pallas::k3_dspark_mla_prefill_write_w{window}",
            self.mesh,
            body,
            (P(self.axis),) * (2 + layers),
            (P(self.axis),) * layers,
            donate_argnums=tuple(range(2, 2 + layers)),
            export_key=repr(("dspark_mla_prefill_write", window, block_shape, layers)),
        )
        return self._ops[key]

    def write(self, latents, table: torch.Tensor, block_shape: tuple[int, ...]) -> None:
        """``latents`` (:meth:`latents`) into the draft pools: window page ``p``
        into block ``table[0, p]`` (this rank's own table, as the target's)."""
        window = int(latents.shape[2])
        pools = self._write_op(window, tuple(block_shape))(latents, table, *self.pools)
        for old, fresh in zip(self.pools, pools, strict=True):
            if fresh is not old:
                old.set_(fresh)
