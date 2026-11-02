# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1-Flash KV-cache allocation: CSA2/SWA/indexer cache arrays."""

from __future__ import annotations

import math
import os
from collections.abc import Callable

import torch
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.runner.kv_cache_dsv4 import DsV4KVCacheAllocator
from vllm_torchtpu.utils import synchronize_tensors

logger = init_logger(__name__)


def _use_prefill_compact_swa_layout() -> bool:
    """Select the prefill (CED pure 20-layer) KV layout.

    Prefill node (VLLM_DSV41_PURE_20L_PREFILL=1, prefill branch cd4eedb): one compact,
    non-overlapping SWA ring per SWA layer (>= 768 blocks, ids remapped by the
    compact BlockPool patch below) and compressor state aliased onto the CSA arrays.
    Every other config (decode branch 1d02e4a): SWA overlaid on the CSA NoPE hosts in
    a tail partition of the shared block pool, a separate per-layer compressor state
    cache, and the UE8M0 scale init limited to [1, main_end).
    """
    return bool(
        envs.VLLM_DSV41_PURE_20L_PREFILL and not envs.TPU_RAIDEN_DSV4_ADMISSION
    )


def _apply_compact_swa_block_pool_patch() -> None:
    # Prefill layout only: on decode, SWA blocks must stay real tail blocks of the
    # shared pool because they are overlaid on the CSA NoPE hosts.
    if not _use_prefill_compact_swa_layout():
        return
    from vllm.v1.core.block_pool import BlockPool, FreeKVCacheBlockQueue
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

    if getattr(BlockPool, "_tpu_compact_swa_pool_patch", False):
        return

    def _configure_aux_partitions_compact_swa(
        self, num_swa_blocks: int, num_state_blocks: int
    ) -> None:
        if (
            self.swa_free_block_queue is not None
            or self.state_free_block_queue is not None
        ):
            return
        if num_swa_blocks + num_state_blocks <= 0:
            return
        if num_state_blocks >= self.free_block_queue.num_free_blocks:
            return
        if num_state_blocks > 0:
            state_blocks = self.free_block_queue.popright_n(num_state_blocks)
            for b in state_blocks:
                b.pool_partition = "state"
            self.state_free_block_queue = FreeKVCacheBlockQueue(state_blocks)
        _, target_swa = DeepseekV41FlashKVCacheAllocator.aux_partition_sizes(
            getattr(self, "num_gpu_blocks", 0)
        )
        swa_cap = (
            max(num_swa_blocks, target_swa)
            if target_swa > 0
            else min(num_swa_blocks, 128)
        )
        swa_blocks = []
        for bid in range(1, swa_cap + 1):
            blk = KVCacheBlock(block_id=bid)
            blk.pool_partition = "swa"
            swa_blocks.append(blk)
        self.swa_free_block_queue = FreeKVCacheBlockQueue(swa_blocks)

    BlockPool.configure_aux_partitions = _configure_aux_partitions_compact_swa
    BlockPool._tpu_compact_swa_pool_patch = True


try:
    _apply_compact_swa_block_pool_patch()
except Exception:
    pass


class DeepseekV41FlashKVCacheAllocator(DsV4KVCacheAllocator):
    """DeepSeek-V4.1-Flash KV-cache allocator (CSA2 + SEQ_ALONG_LANE indexer)."""

    @staticmethod
    def aux_partition_sizes(
        num_blocks: int, max_num_reqs: int | None = None
    ) -> tuple[int, int]:
        """Tail-block counts reserved for compressor state and SWA caches."""
        if num_blocks >= 4096:
            reqs = (
                max_num_reqs if max_num_reqs is not None else max(64, num_blocks // 107)
            )
            num_state = max(
                128,
                min((9 * reqs) // 4 if reqs > 64 else 2 * reqs, num_blocks // 32),
            )
            num_swa = max(768, min((11 * reqs) // 2, num_blocks // 11))
            return num_state, num_swa
        if num_blocks >= 256 and _use_prefill_compact_swa_layout():
            # Prefill SWA ring fix (cd4eedb): 768 non-overlapping SWA ring blocks per
            # layer so 16K-chunked prefill keeps the previous chunk's SWA tail.
            return max(16, min(160, num_blocks // 32)), 768
        if num_blocks >= 64:
            return max(4, min(64, num_blocks // 32)), max(
                8, min(128, (num_blocks * 3) // 16)
            )
        return 0, 0

    def _initialize_ds_v4_kv_cache(
        self,
        kv_cache_config: KVCacheConfig,
        per_layer_spec: Callable[[str], KVCacheSpec],
        kv_caches: dict[str, torch.Tensor],
        num_blocks: int,
    ) -> None:
        runner = self.runner
        bs = runner.block_size = runner.cache_config.block_size
        cap = min(runner.max_num_reqs, runner.max_num_tokens)
        most_len = runner.most_model_len
        runner.num_blocks_per_most_len_req = runner.num_reqs_most_model_len = None
        if most_len is not None:
            runner.num_blocks_per_most_len_req = cdiv(most_len, bs)
            runner.num_reqs_most_model_len = cap
        runner.num_reqs_max_model_len = cap

        mla_layer_names, swa_layer_groups, state_layer_names = (
            self._classify_ds_v4_layers(kv_cache_config, per_layer_spec)
        )
        # Tag every cache layer with its KV-cache group (decodefix 11e6c02).
        # Attention layers share per-step SWA page indices only within one
        # group: each group has its own block table, and in the decode layout
        # same-position SWA layers of different groups are overlaid on one
        # physical array. (The prefill role ignores the tag; see
        # VllmDeepseekV41MLAAttention._swa_kv_cache_group_id.)
        static_ctx = runner.vllm_config.compilation_config.static_forward_context
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            for layer_name in group.layer_names:
                layer_mod = static_ctx.get(layer_name)
                if layer_mod is not None:
                    layer_mod._dsv41_kv_cache_group_id = gid
        if not _use_prefill_compact_swa_layout():
            # Memoize the per-group SWA layer bounds now, outside any trace, so
            # compiled forwards hit the memo instead of scanning every
            # attention layer (and installing guards on all of them) while
            # tracing (decodefix cfacfa7). The prefill role keeps cd4eedb's
            # lazy bounds.
            for mod in static_ctx.values():
                group_bounds = getattr(mod, "_swa_group_bounds", None)
                if callable(group_bounds) and hasattr(mod, "swa_cache_layer"):
                    for ced in (False, True):
                        group_bounds(ced)
        logger.info(
            "DeepSeek-V4.1-Flash KV cache groups: %s",
            [
                (gid, len(g.layer_names), g.layer_names[0] if g.layer_names else "")
                for gid, g in enumerate(kv_cache_config.kv_cache_groups)
            ],
        )
        packing = self._DS_V4_KV_PACKING

        max_reqs = getattr(runner, "max_num_reqs", None)
        if not isinstance(max_reqs, int):
            max_reqs = getattr(
                getattr(runner, "scheduler_config", None), "max_num_seqs", None
            )
        if not isinstance(max_reqs, int):
            max_reqs = None

        if _use_prefill_compact_swa_layout():
            csa_nope_hosts = self._init_prefill_compact_swa_layout(
                kv_cache_config,
                per_layer_spec,
                kv_caches,
                num_blocks,
                mla_layer_names,
                swa_layer_groups,
                state_layer_names,
                packing,
                max_reqs,
            )
        else:
            csa_nope_hosts = self._init_decode_overlay_layout(
                kv_cache_config,
                per_layer_spec,
                kv_caches,
                num_blocks,
                mla_layer_names,
                swa_layer_groups,
                state_layer_names,
                packing,
                max_reqs,
            )

        model = getattr(runner, "model", None)
        draft_model = getattr(getattr(runner, "drafter", None), "draft_model", None)
        roots = [m for m in (model, draft_model) if m is not None]
        if roots:
            init_caches: list[torch.Tensor] = []
            for root in roots:
                for mod in root.modules():
                    if hasattr(mod, "ensure_cache"):
                        swa = getattr(mod, "swa_cache_module", None)
                        if swa is not None and swa.prefix in kv_caches:
                            swa.kv_cache = kv_caches[swa.prefix]
                        if mod.ensure_cache():
                            init_caches.append(mod._cache)
                    if (
                        hasattr(mod, "state_cache")
                        and getattr(mod, "state_cache", None) is not None
                        and hasattr(mod, "k_cache_prefix")
                    ):
                        _st_e = kv_caches.get(mod.state_cache.prefix)
                        _k_e = kv_caches.get(mod.k_cache_prefix)
                        _st_t = _st_e[0] if isinstance(_st_e, tuple) else _st_e
                        _no_t = _k_e[0] if isinstance(_k_e, tuple) else _k_e
                        if _st_t is not None and _no_t is not None:
                            _al = bool(_st_t.data_ptr() == _no_t.data_ptr())
                            object.__setattr__(mod, "_aliased_state_to_nope", _al)
                            assert _al == (_st_t.data_ptr() == _no_t.data_ptr())
                    if hasattr(mod, "_get_dummy_main_cache"):
                        init_caches.append(mod._get_dummy_main_cache(runner.device))
            if init_caches:
                synchronize_tensors(init_caches)

        arrays = [t for e in kv_caches.values() for t in self._ds_v4_cache_arrays(e)]
        logger.info(
            "DeepSeek-V4.1-Flash KV cache: %d arrays for %d layers (mla=%d of which "
            "csa=%d, swa=%d in %d group(s) sized %s, state=%d), num_blocks=%d, "
            "largest array=%s",
            len({id(t) for t in arrays}),
            len(kv_caches),
            len(mla_layer_names),
            len(csa_nope_hosts),
            sum(len(g) for g in swa_layer_groups),
            len(swa_layer_groups),
            [len(g) for g in swa_layer_groups],
            len(state_layer_names),
            num_blocks,
            max((tuple(t.shape) for t in arrays), key=math.prod, default=()),
        )

    def _init_prefill_compact_swa_layout(
        self,
        kv_cache_config: KVCacheConfig,
        per_layer_spec: Callable[[str], KVCacheSpec],
        kv_caches: dict[str, torch.Tensor],
        num_blocks: int,
        mla_layer_names: list[str],
        swa_layer_groups: list[list[str]],
        state_layer_names: list[str],
        packing: int,
        max_reqs: int | None,
    ) -> list[torch.Tensor]:
        """Prefill-branch (cd4eedb) layout; see _use_prefill_compact_swa_layout."""
        runner = self.runner
        num_state_aux, num_swa_aux = self.aux_partition_sizes(num_blocks, max_reqs)
        main_end = max(1, num_blocks - num_state_aux)

        def _create_cache(
            shape: tuple[int, ...],
            tag: str,
            dtype: torch.dtype = torch.uint8,
            scale_idx: tuple | None = None,
        ) -> torch.Tensor:
            cache = torch.zeros(shape, dtype=dtype, device=runner.device)
            fill_end = min(main_end, shape[0])
            bench_fill = (
                fill_end > 1
                and bool(envs.ENABLE_DECODE_BENCH_CONNECTOR)
                and not os.path.exists("/tmp/disable_decode_bench_fill")
            )
            if bench_fill:
                seed = 2026 + sum((i + 1) * ord(c) for i, c in enumerate(tag))
                torch.manual_seed(seed)
                chunk_blks = 1024
                for b0 in range(1, fill_end, chunk_blks):
                    b1 = min(b0 + chunk_blks, fill_end)
                    nb = b1 - b0
                    if dtype == torch.int32 and scale_idx is not None:
                        # CSA NoPE slab: independent per-block fp8 bytes directly on device.
                        blk_u8 = (
                            torch.randn(
                                (nb, *shape[1:], 4),
                                dtype=torch.bfloat16,
                                device=runner.device,
                            )
                            .mul_(0.25)
                            .clamp_(-2.0, 2.0)
                            .to(torch.float8_e4m3fn)
                            .view(torch.uint8)
                        )
                        blk_i32 = blk_u8.view(torch.int32).squeeze(-1)
                        blk_i32[(slice(None), *scale_idx)] = (
                            blk_i32[(slice(None), *scale_idx)] & 0x00FFFFFF
                        ) | (0x7F << 24)
                        cache[b0:b1] = blk_i32
                        del blk_i32, blk_u8
                    elif dtype == torch.int32:
                        # CSA RoPE / compact SWA overflow cache: packed bf16 pairs on device.
                        blk_bf16 = torch.randn(
                            (nb, *shape[1:], 2),
                            dtype=torch.bfloat16,
                            device=runner.device,
                        ).mul_(0.125)
                        cache[b0:b1] = blk_bf16.view(torch.int32).squeeze(-1)
                        del blk_bf16
                    else:
                        # Indexer cache: independent per-block uint8[nb, 33, 4, page_size] on device.
                        k_rows = shape[1] - 1 if scale_idx is not None else shape[1]
                        cache[b0:b1, :k_rows] = (
                            torch.randn(
                                (nb, k_rows, *shape[2:]),
                                dtype=torch.bfloat16,
                                device=runner.device,
                            )
                            .mul_(0.25)
                            .clamp_(-2.0, 2.0)
                            .to(torch.float8_e4m3fn)
                            .view(torch.uint8)
                        )
                        if scale_idx is not None:
                            cache[(slice(b0, b1), *scale_idx)] = 0x7F
                    # Critical: flush lazy torch_tpu graph per 1024-block (~256 MiB) chunk so
                    # unmaterialized bfloat16 PRNG ops never fuse into an 11.46 GiB HBM graph!
                    synchronize_tensors([cache])
                # On full-size CSA NoPE hosts, blocks [shape[0] - num_swa_aux : shape[0]) host
                # SWA window KV pages (packed bfloat16 pairs, NOT fp8+e8m0); fill them too!
                if (
                    dtype == torch.int32
                    and scale_idx is not None
                    and num_swa_aux > 0
                    and shape[0] >= main_end + num_state_aux + num_swa_aux
                ):
                    swa_b0 = shape[0] - num_swa_aux
                    swa_bf16 = torch.randn(
                        (num_swa_aux, *shape[1:], 2),
                        dtype=torch.bfloat16,
                        device=runner.device,
                    ).mul_(0.125)
                    cache[swa_b0 : shape[0]] = swa_bf16.view(torch.int32).squeeze(-1)
                    del swa_bf16
                    synchronize_tensors([cache])
            elif scale_idx is not None and fill_end > 1:
                cache[(slice(1, fill_end), *scale_idx)] = (
                    (0x7F << 24) if dtype == torch.int32 else 0x7F
                )  # e8m0 = 1.0
            synchronize_tensors([cache])
            logger.debug(
                "DeepSeek-V4.1 KV array for %s: shape=%s dtype=%s", tag, shape, dtype
            )
            return cache

        csa_nope_hosts: list[torch.Tensor] = []
        for layer_name in mla_layer_names:
            spec = per_layer_spec(layer_name)
            page_size = spec.num_states
            if layer_name.endswith(self._DS_V4_INDEXER_CACHE_SUFFIX):
                shape = (num_blocks, spec.head_size // packing, packing, page_size)
                kv_caches[layer_name] = _create_cache(
                    shape, layer_name, scale_idx=(32, 0, slice(None))
                )
            else:
                nope = _create_cache(
                    (num_blocks, page_size, csa_cache_layout.ROW_WORDS),
                    layer_name,
                    torch.int32,
                    (slice(None), slice(64, 128)),
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

        if not csa_nope_hosts:
            raise ValueError(f"No CSA layer in MLA layers: {mla_layer_names}")
        mla_ptrs: dict[int, str] = {}
        for layer_name in mla_layer_names:
            entry = kv_caches[layer_name]
            for idx_t, t in enumerate(entry if isinstance(entry, tuple) else (entry,)):
                if t.numel() > 0:
                    ptr = int(t.data_ptr())
                    tag_t = f"{layer_name}[{idx_t}]"
                    assert ptr not in mla_ptrs, (
                        f"KV cache aliasing detected: {tag_t} aliases {mla_ptrs[ptr]}"
                    )
                    mla_ptrs[ptr] = tag_t

        min_host_page_size = min(h.shape[1] for h in csa_nope_hosts)
        csa_nope_hosts = [h for h in csa_nope_hosts if h.shape[1] == min_host_page_size]

        coord_swa = (
            min(2816 if num_blocks >= 16384 else 640, max(32, num_blocks // 11))
            if num_blocks >= 256
            else 0
        )
        eff_swa_blocks = max(num_swa_aux, coord_swa)
        compact_swa_blocks = eff_swa_blocks + 1 if eff_swa_blocks > 0 else num_blocks
        swa_overflow_shape = (
            compact_swa_blocks,
            csa_nope_hosts[0].shape[1],
            csa_cache_layout.ROW_WORDS,
        )
        if swa_layer_groups and swa_layer_groups[0]:
            swa_rows = per_layer_spec(swa_layer_groups[0][0]).block_size * 2
            swa_overflow_shape = (
                compact_swa_blocks,
                swa_rows,
                csa_cache_layout.ROW_WORDS,
            )

        def _host_at(
            hosts: list[torch.Tensor],
            position: int,
            alloc_shape: tuple[int, ...],
        ) -> torch.Tensor:
            while position >= len(hosts):
                hosts.append(
                    _create_cache(
                        alloc_shape,
                        f"deepseek_v41_flash_overflow.{len(hosts)}",
                        csa_nope_hosts[0].dtype,
                    )
                )
            return hosts[position]

        overlay_hosts: list[torch.Tensor] = []
        for group_swa_layers in swa_layer_groups:
            for layer_name in group_swa_layers:
                kv_caches[layer_name] = _host_at(
                    overlay_hosts, len(overlay_hosts), swa_overflow_shape
                )

        for position, layer_name in enumerate(state_layer_names):
            kv_layer_name = self._ds_v4_compressed_kv_layer_name(layer_name)
            if kv_layer_name not in kv_caches:
                raise ValueError(f"State cache {layer_name} missing KV {kv_layer_name}")
            kv_caches[layer_name] = kv_caches[kv_layer_name]

        self._validate_ds_v4_overlay(kv_cache_config, kv_caches)
        return csa_nope_hosts

    def _init_decode_overlay_layout(
        self,
        kv_cache_config: KVCacheConfig,
        per_layer_spec: Callable[[str], KVCacheSpec],
        kv_caches: dict[str, torch.Tensor],
        num_blocks: int,
        mla_layer_names: list[str],
        swa_layer_groups: list[list[str]],
        state_layer_names: list[str],
        packing: int,
        max_reqs: int | None,
    ) -> list[torch.Tensor]:
        """Decode-branch (1d02e4a) layout; see _use_prefill_compact_swa_layout."""
        runner = self.runner
        mla_bytes_per_block = 0
        csa_host_page_sizes: list[int] = []
        for layer_name in mla_layer_names:
            spec = per_layer_spec(layer_name)
            page_size = spec.num_states
            if layer_name.endswith(self._DS_V4_INDEXER_CACHE_SUFFIX):
                mla_bytes_per_block += (
                    (spec.head_size // packing) * packing * page_size
                )
            else:
                mla_bytes_per_block += (
                    page_size * csa_cache_layout.ROW_WORDS * 4
                )
                mla_bytes_per_block += (
                    (page_size // csa_cache_layout.ROPE_TOKENS_PER_ROW)
                    * csa_cache_layout.ROW_WORDS
                    * 4
                )
                csa_host_page_sizes.append(page_size)
        n_disjoint_swa = max((len(g) for g in swa_layer_groups), default=0)
        n_extra_swa_hosts = max(0, n_disjoint_swa - len(csa_host_page_sizes))
        if csa_host_page_sizes and n_extra_swa_hosts > 0:
            min_csa_page_size = min(csa_host_page_sizes)
            swa_rows_init = (
                per_layer_spec(swa_layer_groups[0][0]).block_size * 2
                if swa_layer_groups and swa_layer_groups[0]
                else min_csa_page_size
            )
            swa_extra_bytes_per_block = (
                n_extra_swa_hosts
                * swa_rows_init
                * csa_cache_layout.ROW_WORDS
                * 4
            )
            total_phys_bytes_per_block = (
                mla_bytes_per_block + swa_extra_bytes_per_block
            )
            pass  # Scaled in tpu_worker.determine_available_memory

        num_state_aux = (
            1152 if num_blocks >= 16384 else (1056 if num_blocks >= 4096 else max(16, num_blocks // 40))
        )
        num_swa_aux = (
            2816 if num_blocks >= 16384 else (800 if num_blocks >= 2048 else max(32, num_blocks // 11))
        )
        main_end = (
            num_blocks
            if envs.VLLM_DSV41_PURE_20L_PREFILL
            else max(1, num_blocks - num_state_aux - num_swa_aux)
        )

        def _create_cache(
            shape: tuple[int, ...],
            tag: str,
            dtype: torch.dtype = torch.uint8,
            scale_idx: tuple | None = None,
        ) -> torch.Tensor:
            if scale_idx is not None and main_end > 1:
                # Populate UE8M0 unit scales on CPU and H2D-transfer so XLA's
                # 4D slice-scatter pass does not leave the TPU layout
                # unmaterialized or pad non-multiple-of-8 num_blocks (P/D
                # Raiden admission requires exact block-aligned on-device
                # bytes and a materialized device layout).
                cache_cpu = torch.zeros(shape, dtype=dtype)
                cache_cpu[(slice(1, main_end), *scale_idx)] = (
                    (0x7F << 24) if dtype == torch.int32 else 0x7F
                )  # e8m0 = 1.0
                cache = cache_cpu.to(runner.device)
            else:
                cache = torch.zeros(shape, dtype=dtype, device=runner.device)
            cache._main_end = main_end
            synchronize_tensors([cache])
            logger.debug(
                "DeepSeek-V4.1 KV array for %s: shape=%s dtype=%s", tag, shape, dtype
            )
            return cache

        csa_nope_hosts: list[torch.Tensor] = []
        for layer_name in mla_layer_names:
            spec = per_layer_spec(layer_name)
            page_size = spec.num_states
            if layer_name.endswith(self._DS_V4_INDEXER_CACHE_SUFFIX):
                shape = (num_blocks, spec.head_size // packing, packing, page_size)
                kv_caches[layer_name] = _create_cache(
                    shape, layer_name, scale_idx=(32, 0, slice(None))
                )
            else:
                nope = _create_cache(
                    (num_blocks, page_size, csa_cache_layout.ROW_WORDS),
                    layer_name,
                    torch.int32,
                    (slice(None), slice(64, 128)),
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

        if not csa_nope_hosts:
            raise ValueError(f"No CSA layer in MLA layers: {mla_layer_names}")

        min_host_page_size = min(h.shape[1] for h in csa_nope_hosts)
        csa_nope_hosts = [h for h in csa_nope_hosts if h.shape[1] == min_host_page_size]

        swa_overflow_shape = tuple(csa_nope_hosts[0].shape)
        if swa_layer_groups and swa_layer_groups[0]:
            swa_rows = per_layer_spec(swa_layer_groups[0][0]).block_size * 2
            swa_overflow_shape = (num_blocks, swa_rows, csa_cache_layout.ROW_WORDS)

        def _host_at(
            hosts: list[torch.Tensor],
            position: int,
            alloc_shape: tuple[int, ...],
        ) -> torch.Tensor:
            while position >= len(hosts):
                hosts.append(
                    _create_cache(
                        alloc_shape,
                        f"deepseek_v41_flash_overflow.{len(hosts)}",
                        csa_nope_hosts[0].dtype,
                    )
                )
            return hosts[position]

        if envs.VLLM_DSV41_PURE_20L_PREFILL:
            overlay_hosts: list[torch.Tensor] = []
            n_csa_hosts = len(csa_nope_hosts)
            csa_host_shape = tuple(csa_nope_hosts[0].shape)
            for group_swa_layers in swa_layer_groups:
                for position, layer_name in enumerate(group_swa_layers):
                    alloc_shape = (
                        csa_host_shape
                        if position < n_csa_hosts
                        else swa_overflow_shape
                    )
                    kv_caches[layer_name] = _host_at(
                        overlay_hosts, len(overlay_hosts), alloc_shape
                    )
        else:
            overlay_hosts = list(csa_nope_hosts)
            n_csa_hosts = len(csa_nope_hosts)
            for group_idx, group_swa_layers in enumerate(swa_layer_groups):
                for position, layer_name in enumerate(group_swa_layers):
                    host_pos = (
                        (position + group_idx) % n_csa_hosts
                        if n_csa_hosts > 1 and position < n_csa_hosts
                        else position
                    )
                    kv_caches[layer_name] = _host_at(
                        overlay_hosts, host_pos, swa_overflow_shape
                    )

        for position, layer_name in enumerate(state_layer_names):
            kv_layer_name = self._ds_v4_compressed_kv_layer_name(layer_name)
            if kv_layer_name not in kv_caches:
                raise ValueError(f"State cache {layer_name} missing KV {kv_layer_name}")
            state_rows = max(32, per_layer_spec(layer_name).block_size * 8)
            kv_caches[layer_name] = _create_cache(
                (num_blocks, state_rows, csa_cache_layout.ROW_WORDS),
                layer_name,
                torch.int32,
            )

        self._validate_ds_v4_overlay(kv_cache_config, kv_caches)
        return csa_nope_hosts
