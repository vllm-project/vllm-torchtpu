# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 KV-cache allocation: layer classification and cache arrays.

Moved verbatim out of `TPUModelRunner`; see vllm-project/vllm-torchtpu#713.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import torch
from vllm.v1.kv_cache_interface import (KVCacheConfig, KVCacheSpec,
                                        MLAAttentionSpec, SlidingWindowMLASpec)

from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

logger = init_logger(__name__)


class DsV4KVCacheAllocator:
    """Owns the DeepSeek-V4 special cases in KV-cache allocation."""

    _DS_V4_STATE_CACHE_SUFFIX = ".compressor.state_cache"
    _DS_V4_INDEXER_CACHE_SUFFIX = ".indexer.k_cache"
    _DS_V4_ROPE_CACHE_SUFFIX = "_rope"
    _DS_V4_KV_PACKING = 4
    _DS_V4_CSA_COMPRESS_RATIO = 4

    def __init__(self, runner: TPUModelRunner) -> None:
        self.runner = runner

    @staticmethod
    def _ds_v4_compressed_kv_layer_name(state_cache_name: str) -> str:
        """The layer whose compressed-KV records a DSv4 state cache accompanies.

        Mirrors how vLLM wires `DeepseekCompressor.k_cache_prefix` (the
        attention layer itself for the main compressor, the indexer's
        `k_cache` for the indexer compressor):

          `<...>.attn.compressor.state_cache`          -> `<...>.attn`
          `<...>.attn.indexer.compressor.state_cache`  ->
              `<...>.attn.indexer.k_cache`
        """
        suffix = DsV4KVCacheAllocator._DS_V4_STATE_CACHE_SUFFIX
        if not state_cache_name.endswith(suffix):
            raise ValueError(
                "DeepSeek-V4 compressor state cache has an unexpected layer "
                f"name {state_cache_name!r}; expected it to end with "
                f"{suffix!r} so the compressed-KV layer can be derived.")
        base = state_cache_name[:-len(suffix)]
        return base + ".k_cache" if base.endswith(".indexer") else base

    @staticmethod
    def _is_ds_v4_swa_layer(layer_name: str, spec: KVCacheSpec) -> bool:
        """A DSv4 sliding-window cache (not a compressor state cache).

        Both declare `SlidingWindowMLASpec`, so the name separates them.
        """
        return isinstance(spec,
                          SlidingWindowMLASpec) and "swa_cache" in layer_name

    def _classify_ds_v4_layers(
        self,
        kv_cache_config: KVCacheConfig,
        per_layer_spec: Callable[[str], KVCacheSpec],
    ) -> tuple[list[str], list[list[str]], list[str]]:
        """Split DSv4's layers by the kernel that reads their array.

        Iteration is in cache-group order so allocation and overlay
        assignment are deterministic across workers.

        Returns `(mla_layer_names, swa_layer_groups, state_layer_names)`.
        SWA layers stay grouped: layers of one cache group share a block
        table, so each must land on a different array.
        """
        mla_layer_names: list[str] = []
        swa_layer_groups: list[list[str]] = []
        state_layer_names: list[str] = []
        for group in kv_cache_config.kv_cache_groups:
            swa_in_group: list[str] = []
            for layer_name in group.layer_names:
                if layer_name in self.runner.shared_kv_cache_layers:
                    continue
                spec = per_layer_spec(layer_name)
                if isinstance(spec, MLAAttentionSpec):
                    mla_layer_names.append(layer_name)
                elif layer_name.endswith(self._DS_V4_STATE_CACHE_SUFFIX):
                    state_layer_names.append(layer_name)
                elif self._is_ds_v4_swa_layer(layer_name, spec):
                    swa_in_group.append(layer_name)
                else:
                    raise ValueError(
                        "DeepSeek-V4 layer has no known role (expected an "
                        "MLAAttentionSpec cache, a `*.compressor.state_cache` "
                        f"or a `*swa_cache*` layer): layer={layer_name}, "
                        f"spec={spec}")
            if swa_in_group:
                swa_layer_groups.append(swa_in_group)

        if not mla_layer_names:
            raise ValueError(
                "DeepSeek-V4 model has no MLAAttentionSpec layers to anchor "
                "the KV cache overlays. groups="
                f"{[g.layer_names for g in kv_cache_config.kv_cache_groups]}")
        return mla_layer_names, swa_layer_groups, state_layer_names

    def _initialize_ds_v4_kv_cache(
        self,
        kv_cache_config: KVCacheConfig,
        per_layer_spec: Callable[[str], KVCacheSpec],
        kv_caches: dict[str, torch.Tensor],
        num_blocks: int,
    ) -> None:
        """Allocate and alias DeepSeek-V4's KV caches.

        vLLM lays DSv4's cache groups out as one packed byte slab and assigns
        page indices assuming the compressed-KV, SWA and compressor-state
        caches share it. TPU tensors cannot be byte views into a slab, so the
        overlay is rebuilt here from the specs.

        Every array is uint8 and shaped for the kernel that reads it, rather
        than by the generic `head_size` formula. With `T =
        spec.num_states` compressed tokens per page:

            CSA `*.attn`      NoPE `(N, T, 4, 128)`     512B/token
                              RoPE `(N, T/4, 4, 128)`   128B/token, in a
                                                        companion array
            indexer k_cache   `(N, T/4, 4, 256)`        256B/token
            HCA `*.attn`      `(N, T*2, 4, 128)`       1024B/token (raw bf16)

        The overlays, all of which the specs' byte budgets already account
        for:

        - Every SWA cache maps onto a CSA NoPE array by its position in its
          cache group.
        - Every CSA / indexer compressor state cache maps onto the array of
          its own compressed-KV layer: the compressor kernel writes the f32
          state rows and the compressed KV through that one buffer. State and
          full-attention groups never own the same block ID, so the rows
          never collide.
        - The i-th HCA compressor state cache maps onto the i-th CSA NoPE
          array.

        Overlays are planned before allocating: allocating an array and
        aliasing it afterwards still counts against peak HBM.
        """
        mla_layer_names, swa_layer_groups, state_layer_names = (
            self._classify_ds_v4_layers(kv_cache_config, per_layer_spec))

        packing = self._DS_V4_KV_PACKING

        def _create_cache(shape: tuple[int, ...], tag: str) -> torch.Tensor:
            cache = torch.zeros(shape,
                                dtype=torch.uint8).to(self.runner.device)
            logger.debug("DeepSeek-V4 KV array for %s: shape=%s", tag, shape)
            return cache

        # CSA NoPE tensors, in layer order; the SWA caches and the HCA
        # compressor states overlay these.
        csa_nope_hosts: list[torch.Tensor] = []
        # HCA compressed-KV layers, whose own page is far too small to host
        # their state cache.
        hca_layer_names: set[str] = set()

        for layer_name in mla_layer_names:
            spec = per_layer_spec(layer_name)
            page_size = spec.num_states
            if layer_name.endswith(self._DS_V4_INDEXER_CACHE_SUFFIX):
                # Lightning indexer: 128 fp8 values + 1 e8m0 scale per token,
                # padded to a 256B record; 4 tokens per row.
                shape = (num_blocks, page_size // packing, packing, 256)
                kv_caches[layer_name] = _create_cache(shape, layer_name)
            elif spec.tokens_per_state == self._DS_V4_CSA_COMPRESS_RATIO:
                # CSA is split across two arrays, for NoPE and RoPE.
                nope = _create_cache((num_blocks, page_size, packing, 128),
                                     layer_name)
                rope = _create_cache(
                    (num_blocks, page_size // packing, packing, 128),
                    f"{layer_name}{self._DS_V4_ROPE_CACHE_SUFFIX}")
                kv_caches[layer_name] = (nope, rope)
                csa_nope_hosts.append(nope)
            else:
                # HCA keeps raw bf16 latents: 512 values = 1024B per token,
                # i.e. two rows per token.
                shape = (num_blocks, page_size * 2, packing, 128)
                kv_caches[layer_name] = _create_cache(shape, layer_name)
                hca_layer_names.add(layer_name)

        if not csa_nope_hosts:
            raise ValueError(
                "DeepSeek-V4 model has no CSA layer (compress_ratio "
                f"{self._DS_V4_CSA_COMPRESS_RATIO}) to host the SWA and HCA "
                f"state caches. MLA layers={mla_layer_names}")

        def _host_at(hosts: list[torch.Tensor], position: int) -> torch.Tensor:
            """`hosts[position]`, growing the list with standalone arrays.

            More SWA (or HCA) layers than CSA arrays is normal -- DSv4-Flash
            has 43 SWA layers to 21 CSA ones -- and the overflow cannot reuse
            an array already taken by a same-group layer.
            """
            while position >= len(hosts):
                hosts.append(
                    _create_cache(tuple(csa_nope_hosts[0].shape),
                                  f"ds_v4_overflow.{len(hosts)}"))
            return hosts[position]

        # SWA caches overlay the CSA NoPE arrays; so do the HCA states, and
        # both index the same host list so they never collide on one array.
        overlay_hosts = list(csa_nope_hosts)
        for group_swa_layers in swa_layer_groups:
            for position, layer_name in enumerate(group_swa_layers):
                kv_caches[layer_name] = _host_at(overlay_hosts, position)

        hca_state_layers: list[str] = []
        for layer_name in state_layer_names:
            kv_layer_name = self._ds_v4_compressed_kv_layer_name(layer_name)
            if kv_layer_name not in kv_caches:
                raise ValueError(
                    "DeepSeek-V4 compressor state cache has no compressed-KV "
                    "layer (its compressed records are written there): "
                    f"state_cache={layer_name}, expected KV layer "
                    f"{kv_layer_name}, known MLA layers={mla_layer_names}")
            if kv_layer_name in hca_layer_names:
                hca_state_layers.append(layer_name)
            else:
                kv_caches[layer_name] = kv_caches[kv_layer_name]

        for position, layer_name in enumerate(hca_state_layers):
            kv_caches[layer_name] = _host_at(overlay_hosts, position)

        self._validate_ds_v4_overlay(kv_cache_config, kv_caches)

        # SWA and HCA-state overlays are assigned by position *within a cache
        # group*, so the group shape decides how many arrays are needed: two
        # layers of one group must never share one (they share a block table),
        # while layers of different groups may. Log it -- a grouping change
        # silently changes the overlay plan.
        logger.info(
            "DeepSeek-V4 KV cache: %d arrays for %d layers (mla=%d of which "
            "csa=%d, swa=%d in %d group(s) "
            "sized %s, state=%d), num_blocks=%d, largest array=%s",
            len({
                id(t)
                for e in kv_caches.values()
                for t in self._ds_v4_cache_arrays(e)
            }), len(kv_caches), len(mla_layer_names), len(csa_nope_hosts),
            sum(len(g) for g in swa_layer_groups),
            len(swa_layer_groups), [len(g) for g in swa_layer_groups],
            len(state_layer_names), num_blocks,
            max((tuple(t.shape) for e in kv_caches.values()
                 for t in self._ds_v4_cache_arrays(e)),
                key=lambda shp: math.prod(shp),
                default=()))

    @staticmethod
    def _ds_v4_cache_arrays(entry) -> tuple[torch.Tensor, ...]:
        """Every array a bound DSv4 KV entry holds.

        A CSA layer binds a `(nope, rope)` pair -- one layer, two arrays under
        one block table -- while every other role binds a bare tensor. This is
        the producer side of that convention; the two consumers that can see a
        pair (the compressor's `k_cache` and the attention layer's own entry)
        unpack it inline.
        """
        return tuple(entry) if isinstance(entry, tuple) else (entry, )

    @classmethod
    def _validate_ds_v4_overlay(cls, kv_cache_config: KVCacheConfig,
                                kv_caches: dict[str, torch.Tensor]) -> None:
        """No two layers of one cache group may share an array.

        Layers of a group share a block table, so a shared array means they
        would write each other's pages.
        """
        for group in kv_cache_config.kv_cache_groups:
            hosts: dict[int, str] = {}
            for layer_name in group.layer_names:
                cache = kv_caches.get(layer_name)
                if cache is None:
                    continue
                cache = cls._ds_v4_cache_arrays(cache)[0]
                ptr = id(cache)
                if ptr in hosts:
                    raise ValueError(
                        "DeepSeek-V4 KV cache overlay put two layers of one "
                        "cache group on the same array; they share a block "
                        f"table, so they would corrupt each other: "
                        f"{layer_name} and {hosts[ptr]} both map to the array "
                        f"of shape {tuple(cache.shape)}. Group "
                        f"layers={group.layer_names}")
                hosts[ptr] = layer_name
