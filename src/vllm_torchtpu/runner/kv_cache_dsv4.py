# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 KV-cache allocation: layer classification and cache arrays.

Moved verbatim out of `TPUModelRunner`; see vllm-project/vllm-torchtpu#713.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import torch
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    KVCacheConfig,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

logger = init_logger(__name__)


def deepseek_v41_cache_spec(spec, vllm_config, *, compressed=False, indexer=False):
    """Budget separate NoPE/RoPE arrays and the auxiliary NoPE overlays.

    Each physical region must hold a BF16 SWA page or the C2 compressor's
    eight FP32 state rows (KV plus scores). Reserve the companion RoPE array
    explicitly, so splitting the native arrays never exceeds vLLM's budget.
    Each logical page uses its array's full record capacity; the fixed
    compressor ring retains its upstream lifetime and row count.
    """
    if spec is None:
        return None
    block_size = vllm_config.cache_config.block_size
    nope_bytes = max(block_size * 512 * 2, 8 * 1024 * 4)
    page_bytes = nope_bytes * 5 // 4
    changes: dict[str, Any] = {"page_size_padded": page_bytes}
    if hasattr(spec, "alignment"):
        changes["alignment"] = None
    if indexer:
        # Four 128-byte keys share each NoPE word row. Each key's E8M0
        # scale occupies one int32 word in the companion RoPE allocation.
        from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41 import (
            deepseek_v41_cache as cache_ops,
        )

        geometry = cache_ops.IndexPageGeometry(nope_bytes, page_bytes - nope_bytes)
        states = geometry.keys_per_page
        changes.update(
            dtype=torch.uint8,
            state_content_bytes=geometry.state_bytes,
            block_size=int(states * spec.tokens_per_state),
        )
    elif compressed:
        changes.update(
            dtype=torch.uint8,
            state_content_bytes=640,
            block_size=int(nope_bytes // 512 * spec.tokens_per_state),
        )
    elif not isinstance(spec, CircularBufferSpec):
        changes["block_size"] = int(
            nope_bytes // spec.state_content_size_bytes * spec.tokens_per_state
        )
    result = replace(spec, **changes)
    budget = page_bytes if compressed or indexer else nope_bytes
    if result.unpadded_page_size_bytes > budget:
        raise ValueError(f"V4.1 cache page exceeds its native array budget: {result}")
    return result


class DsV4KVCacheAllocator:
    """Owns the DeepSeek-V4 special cases in KV-cache allocation."""

    _DS_V4_STATE_CACHE_SUFFIX = ".compressor.state_cache"
    _DS_V4_INDEXER_CACHE_SUFFIX = ".indexer.k_cache"
    _DS_V4_ROPE_CACHE_SUFFIX = "_rope"
    _DS_V4_KV_PACKING = 4
    _DS_V4_CSA_COMPRESS_RATIO = 4

    def __init__(self, runner: TPUModelRunner) -> None:
        self.runner = runner

    def initialize_v41(self, kv_cache_config, per_layer_spec):
        """Bind V4.1 source caches as V4 `(nope, rope)` arrays.

        Retain vLLM's physical-region assignments, including overlays across
        groups. SWA and compressor state use the NoPE array; indexers use
        NoPE for keys and the companion RoPE allocation for scale words.
        """
        from vllm_torchtpu.kv_cache_materializer import layer_to_pool_index
        from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41 import (
            deepseek_v41_cache as cache_ops,
        )

        regions = layer_to_pool_index(kv_cache_config)
        page_bytes = {
            group.kv_cache_spec.page_size_bytes
            for group in kv_cache_config.kv_cache_groups
        }
        if len(page_bytes) != 1:
            raise ValueError(
                "V4.1 native arrays require a uniform physical page budget"
            )
        page_bytes = page_bytes.pop()
        if page_bytes % 2560:
            raise ValueError("V4.1 page budget must contain complete NoPE/RoPE rows")
        nope_bytes = page_bytes * 4 // 5
        index_geometry = cache_ops.IndexPageGeometry(
            nope_bytes, page_bytes - nope_bytes
        )
        rows = nope_bytes // 512
        num_blocks = kv_cache_config.num_blocks
        pairs = [
            (
                torch.zeros(
                    (num_blocks, rows, 128),
                    dtype=torch.int32,
                    device=self.runner.device,
                ),
                torch.zeros(
                    (num_blocks, rows // 4, 128),
                    dtype=torch.int32,
                    device=self.runner.device,
                ),
            )
            for _ in range(max(regions.values()) + 1)
        ]
        # Every layer must capture the same Tensor object for a shared pool.
        # Capture only its word rows; derive kernel shapes inside the graph.
        # Capturing separate 2D/3D views makes AOT synthesize as_strided_scatter
        # updates of the entire pool for every cache mutation.
        raw_nope = [nope.flatten(0, 1) for nope, _ in pairs]
        result = cache_ops.MaterializedNativeKVCache(
            {}, [t for pair in pairs for t in pair]
        )
        for name, region in regions.items():
            spec = per_layer_spec(name)
            nope, rope = pairs[region]
            raw = raw_nope[region]
            is_indexer = name.endswith(self._DS_V4_INDEXER_CACHE_SUFFIX)
            is_main = isinstance(spec, MLAAttentionSpec) and not is_indexer
            shape = (
                num_blocks,
                spec.num_states,
                index_geometry.key_bytes
                if is_indexer
                else spec.state_content_size_bytes // spec.dtype.itemsize,
            )
            if is_indexer and spec.num_states != index_geometry.keys_per_page:
                raise ValueError("Index spec and native page geometry disagree")
            layout = cache_ops.ByteLayout(shape, spec.dtype, nope_bytes, 0)
            result.native_raw_cache[name] = raw
            result.native_cache_layouts[name] = layout
            if is_main:
                result.kv_caches[name] = (nope, rope)
            elif self._is_ds_v4_swa_layer(name, spec):
                result.kv_caches[name] = nope
            else:
                view = cache_ops.native_cache_view(
                    raw.view(torch.int8).flatten(),
                    (num_blocks, 1, *shape[1:]),
                    spec.dtype,
                    nope_bytes,
                    0,
                )
                result.kv_caches[name] = (view, rope) if is_indexer else view
        self._validate_ds_v4_overlay(kv_cache_config, result.kv_caches)
        allocated_bytes = sum(t.nbytes for t in result.raw_tensors)
        budget_bytes = max(t.size for t in kv_cache_config.kv_cache_tensors)
        if allocated_bytes > budget_bytes:
            raise ValueError(
                f"V4.1 native cache allocation {allocated_bytes} exceeds {budget_bytes}"
            )
        return result

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
                f"{suffix!r} so the compressed-KV layer can be derived."
            )
        base = state_cache_name[: -len(suffix)]
        return base + ".k_cache" if base.endswith(".indexer") else base

    @staticmethod
    def _is_ds_v4_swa_layer(layer_name: str, spec: KVCacheSpec) -> bool:
        """A DSv4 sliding-window cache (not a compressor state cache).

        Both declare `SlidingWindowMLASpec`, so the name separates them.
        """
        return isinstance(spec, SlidingWindowMLASpec) and "swa_cache" in layer_name

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
                        f"spec={spec}"
                    )
            if swa_in_group:
                swa_layer_groups.append(swa_in_group)

        if not mla_layer_names:
            raise ValueError(
                "DeepSeek-V4 model has no MLAAttentionSpec layers to anchor "
                "the KV cache overlays. groups="
                f"{[g.layer_names for g in kv_cache_config.kv_cache_groups]}"
            )
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

        Every array is shaped for the kernel that reads it, rather than by
        the generic `head_size` formula. With `T = spec.num_states`
        compressed tokens per page:

            CSA `*.attn`      NoPE int32 `(N, T, 128)`   512B/token
                              RoPE int32 `(N, T/4, 128)` 128B/token, in a
                                                         companion array
            indexer k_cache   uint8 `(N, T/4, 4, 256)`   256B/token
            HCA `*.attn`      uint8 `(N, T*2, 4, 128)`  1024B/token (raw bf16)

        The CSA arrays are int32 so the SparseCore gather can fetch a token's
        own 128 RoPE bytes; the TensorCore kernels view them as the uint8
        slabs they hold (see `csa_cache_layout`).

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
            self._classify_ds_v4_layers(kv_cache_config, per_layer_spec)
        )

        packing = self._DS_V4_KV_PACKING

        def _create_cache(
            shape: tuple[int, ...], tag: str, dtype: torch.dtype = torch.uint8
        ) -> torch.Tensor:
            cache = torch.zeros(shape, dtype=dtype).to(self.runner.device)
            logger.debug(
                "DeepSeek-V4 KV array for %s: shape=%s dtype=%s", tag, shape, dtype
            )
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
                nope = _create_cache(
                    (num_blocks, page_size, csa_cache_layout.ROW_WORDS),
                    layer_name,
                    torch.int32,
                )
                rope = _create_cache(
                    (
                        num_blocks,
                        page_size // csa_cache_layout.ROPE_TOKENS_PER_ROW,
                        csa_cache_layout.ROW_WORDS,
                    ),
                    f"{layer_name}{self._DS_V4_ROPE_CACHE_SUFFIX}",
                    torch.int32,
                )
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
                f"state caches. MLA layers={mla_layer_names}"
            )

        def _host_at(hosts: list[torch.Tensor], position: int) -> torch.Tensor:
            """`hosts[position]`, growing the list with standalone arrays.

            More SWA (or HCA) layers than CSA arrays is normal -- DSv4-Flash
            has 43 SWA layers to 21 CSA ones -- and the overflow cannot reuse
            an array already taken by a same-group layer.
            """
            while position >= len(hosts):
                hosts.append(
                    _create_cache(
                        tuple(csa_nope_hosts[0].shape),
                        f"ds_v4_overflow.{len(hosts)}",
                        csa_nope_hosts[0].dtype,
                    )
                )
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
                    f"{kv_layer_name}, known MLA layers={mla_layer_names}"
                )
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
            len(
                {id(t) for e in kv_caches.values() for t in self._ds_v4_cache_arrays(e)}
            ),
            len(kv_caches),
            len(mla_layer_names),
            len(csa_nope_hosts),
            sum(len(g) for g in swa_layer_groups),
            len(swa_layer_groups),
            [len(g) for g in swa_layer_groups],
            len(state_layer_names),
            num_blocks,
            max(
                (
                    tuple(t.shape)
                    for e in kv_caches.values()
                    for t in self._ds_v4_cache_arrays(e)
                ),
                key=lambda shp: math.prod(shp),
                default=(),
            ),
        )

    @staticmethod
    def _ds_v4_cache_arrays(entry) -> tuple[torch.Tensor, ...]:
        """Every array a bound DSv4 KV entry holds.

        CSA binds `(nope, rope)` under one block table. V4.1 indexers bind
        `(key_view, scale_pool)`, using the same region-wide companion array
        for scales. SWA and compressor state bind only their NoPE array.
        """
        return tuple(entry) if isinstance(entry, tuple) else (entry,)

    @classmethod
    def _validate_ds_v4_overlay(
        cls, kv_cache_config: KVCacheConfig, kv_caches: dict[str, torch.Tensor]
    ) -> None:
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
                for array in cls._ds_v4_cache_arrays(cache):
                    # Storage identity sees through dtype/shape views without
                    # requesting a host-accessible pointer from a TPU tensor.
                    storage = array.untyped_storage()._cdata
                    if storage in hosts and hosts[storage] != layer_name:
                        raise ValueError(
                            "DeepSeek-V4 KV cache overlay put two layers of one "
                            "cache group on the same array or the same storage; "
                            "they share a block "
                            f"table: {layer_name} and {hosts[storage]}. "
                            f"Group layers={group.layer_names}"
                        )
                    hosts[storage] = layer_name
