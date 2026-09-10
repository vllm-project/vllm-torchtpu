# SPDX-License-Identifier: Apache-2.0
import dataclasses
import math

import torch
from vllm.model_executor.layers.attention import Attention, MLAAttention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_torchtpu import utils
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

logger = init_logger(__name__)


def _warn_if_kv_cache_is_padded(
    backend: type[AttentionBackend],
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
) -> None:
    """Warn when a page costs more than its shape and dtype imply.

    The baseline is one key and one value vector per head per token. Layouts
    exceed it by rounding the packing dimension up to a 32-bit word and
    head_size up to a 128-lane register; either wastes HBM.

    Read through `get_kv_cache_shape` so any backend works, but MLA has no
    per-head key and value and so is not described by this baseline.
    """
    page_shape = backend.get_kv_cache_shape(1, block_size, num_kv_heads,
                                            head_size, dtype)
    actual = math.prod(page_shape) * dtype.itemsize
    unpadded = block_size * 2 * num_kv_heads * head_size * dtype.itemsize
    if actual == unpadded:
        return
    logger.warning_once(
        "KV cache pages are %.2fx larger than num_kv_heads=%d at "
        "head_size=%d with dtype %s requires: the %s layout pads head_size to "
        "a whole 128-lane register and the K/V vectors to a whole 32-bit word, "
        "so HBM use exceeds a shape-and-dtype estimate. Where head_size is a "
        "multiple of 128, lowering tensor parallelism (more KV heads per rank) "
        "or enabling attention data parallelism avoids it.", actual / unpadded,
        num_kv_heads, head_size, dtype, backend.get_name())


class KvCacheSpecMixin:
    """KV cache spec derivation and page-size padding."""

    def _update_mamba_page_size_padded(
            self, layers: dict[str, AttentionLayerBase]) -> None:
        """Pad attention and mamba page sizes so vLLM's num_blocks matches
        what the TPU allocates per layer.

        For hybrid attention+mamba models, vLLM groups a tensor's memory so
        that one `KVCacheTensor` is `shared_by` one layer from each kv-cache
        group (e.g., Qwen3.5: 1 full-attn + 3 linear-attn per shared_by).
        vLLM's scheduler assumes these layers share a single physical
        tensor at the byte level — each layer's block_table indexes into
        disjoint slots of the same backing allocation, and device kernels
        reinterpret the bytes as attention KV or mamba state depending on
        which layer is accessing the slot.

        TPU `jax.Array`s are strongly typed, so we cannot overlay an
        attention tensor and a mamba tensor on the same bytes.
        `initialize_kv_cache` therefore allocates one physical array per
        layer in the `shared_by` group, carving the group's byte budget
        into separate per-layer tensors. Without the compensation done
        here, vLLM's block pool would hold `num_shared_layers`× more
        block IDs than each per-layer array has slots — the scheduler
        would hand out block IDs beyond a layer's leading dimension,
        JAX's indexed writes would silently clip them, and multiple
        requests' mamba recurrent states would collapse onto the same
        slot (corrupted state → gibberish generation).

        The fix: set every layer's reported `page_size_padded` equal to the
        full per-`shared_by` footprint — `num_attn_groups × attn_page +
        num_mamba_groups × mamba_unpadded`, where `attn_page` is the
        TPU-actual per-block bytes (from `get_attention_page_size_bytes`,
        which accounts for dtype packing like fp8) and `mamba_unpadded` is
        the natural `prod(shape) × dtype_size`. vLLM then computes a
        smaller `num_blocks` that exactly matches what we allocate per layer
        on the TPU side. HBM usage is unchanged; only the block-ID
        accounting lines up.

        Args:
            layers: A dictionary mapping layer names to their corresponding
                attention module instances (e.g., `MambaBase`, `Attention`).
        """
        attn_modules = [
            m for m in layers.values()
            if isinstance(m, (Attention, MLAAttention))
        ]
        if not attn_modules:
            return

        first_attn_module = attn_modules[0]
        num_kv_heads = first_attn_module.num_kv_heads if isinstance(
            first_attn_module, Attention) else 1
        attention_backend = TpuPlatform._find_non_ssm_backend(self.vllm_config)
        attn_page_size_bytes = attention_backend.get_kv_cache_page_size_bytes(
            self.block_size, num_kv_heads, first_attn_module.head_size,
            self.kv_cache_dtype)

        mamba_modules = [
            m for m in layers.values() if isinstance(m, MambaBase)
        ]
        if not mamba_modules:
            # Not hybrid; set `mamba_page_size_padded` to the attention
            # page size as a no-op default (vLLM's platform interface sets
            # this too when it detects hybrid). No layer duplication will
            # happen without mamba layers, so no block-ID mismatch to fix.
            self.cache_config.mamba_page_size_padded = attn_page_size_bytes
            return

        # Compute the unpadded mamba page size from an actual mamba module's
        # spec (shapes × dtype-size), ignoring any existing padding.
        first_mamba_spec = mamba_modules[0].get_kv_cache_spec(self.vllm_config)
        assert isinstance(first_mamba_spec, MambaSpec)
        unpadded_mamba_page_size = dataclasses.replace(
            first_mamba_spec, page_size_padded=None).page_size_bytes

        # Derive vLLM's kv-cache group layout. vLLM splits each type into
        # equal-sized groups of `group_size` layers, then allocates
        # `group_size` `KVCacheTensor`s, each `shared_by` one layer from
        # every group — so each tensor covers `num_attn_groups +
        # num_mamba_groups` layers.
        #
        # Choosing `group_size` trades off padding vs. number of groups:
        #   * group_size = max_count → fewer groups (often 1 per type),
        #     but the smaller side pads its group up to max_count layers
        #     (wastes space if max ≫ min).
        #   * group_size = min_count → no padding, but the larger side
        #     splits into `ceil(max/min)` groups.
        # vLLM's rule: pick max_count only when counts are close enough
        # that the padding is minor (max < 1.5 × min), else min_count.
        #   e.g. 12 sliding-window + 13 full-attn → max (1 group each)
        #   e.g. 10 full-attn      + 30 mamba     → min (1 attn + 3 mamba)
        #
        # This duplicates the heuristic from
        # `vllm/v1/core/kv_cache_utils.py::_get_kv_cache_groups_uniform_page_size`.
        # We can't call it directly because vLLM's grouping needs a fully
        # populated spec dict, while we need the group layout *before* we
        # can finish creating the specs (padding depends on grouping,
        # spec creation depends on padding). Keep in sync if that
        # heuristic ever changes — it has been stable since the hybrid
        # allocator landed.
        num_attn = len(attn_modules)
        num_mamba = len(mamba_modules)
        min_count = min(num_attn, num_mamba)
        max_count = max(num_attn, num_mamba)

        # Match vLLM exactly: float comparison, no int() truncation (matters
        # at e.g. min=3, max=4, where 4 < 4.5 but 4 < int(4.5)==4 differs).
        if max_count < min_count * 1.5:
            group_size = max_count
        else:
            group_size = min_count

        num_attn_groups = (num_attn + group_size - 1) // group_size
        num_mamba_groups = (num_mamba + group_size - 1) // group_size

        uniform_page_size_bytes = (num_attn_groups * attn_page_size_bytes +
                                   num_mamba_groups * unpadded_mamba_page_size)

        logger.info(
            "Hybrid KV cache: padding every layer spec to %d bytes "
            "(num_attn_groups=%d × attn_page=%d + "
            "num_mamba_groups=%d × mamba_unpadded=%d). This makes vLLM's "
            "num_blocks match per-layer TPU allocation when mamba layers "
            "cannot be truly shared.", uniform_page_size_bytes,
            num_attn_groups, attn_page_size_bytes, num_mamba_groups,
            unpadded_mamba_page_size)

        self._hybrid_uniform_page_size_bytes = int(uniform_page_size_bytes)
        self.cache_config.mamba_page_size_padded = int(uniform_page_size_bytes)

        # Prefer compact-mamba sizing: cap each mamba layer at
        # `max_num_reqs + 1` recurrent slots and give the freed HBM to the
        # attention pool. Mamba state is recurrent — one slot per active
        # request — so the uniform layout wastes `num_blocks - max_num_reqs`
        # mamba slots forever. On success this pins
        # `num_gpu_blocks_override` to the (larger) attention block count and
        # sets `_mamba_num_blocks`.
        self._maybe_set_compact_mamba_num_blocks_override(
            attn_page_size_bytes, int(unpadded_mamba_page_size),
            num_attn_groups, num_mamba_groups, group_size)

        # Fallback (compact sizing skipped, e.g. CPU-only tests or a
        # user-pinned num_gpu_blocks_override): pin vLLM's num_blocks via the two-step
        # flooring that keeps peak HBM within `gpu_memory_utilization ×
        # total_hbm` at high utilization. See `_maybe_set_num_blocks_override`
        # for the formula and rationale; the short version is that vLLM's
        # single-step `floor(avail / (uniform × group_size))` can land one
        # block higher than the two-step value, and that extra block ×
        # group_size × uniform bytes is enough to push past the budget against
        # imprecision in vLLM's `avail` estimate.
        if self._mamba_num_blocks is None:
            self._maybe_set_num_blocks_override(attn_page_size_bytes,
                                                int(uniform_page_size_bytes),
                                                group_size)

    def _available_kv_cache_hbm(self) -> int:
        """KV-cache HBM budget for the block-count override paths.

        Matches `TPUWorker.determine_available_memory()`: the
        `utils.compute_hbm_budget` result (which reserves the
        `gpu_memory_utilization` cap minus
        `utils.estimate_kv_connector_hbm_reserve` (HBM a connector allocates
        after profile_run, such as the offload H2D
        staging pool). Sizing overrides against this budget keeps them from
        filling the connector reserve back up with KV blocks and defeating
        the worker-side subtraction.
        """
        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:
            return kv_cache_memory_bytes
        budget = utils.compute_hbm_budget(
            [self.device], self.cache_config.gpu_memory_utilization)
        return budget.available - utils.estimate_kv_connector_hbm_reserve(
            self.vllm_config)

    def _maybe_set_compact_mamba_num_blocks_override(
            self, attn_page_size_bytes: int,
            unpadded_mamba_page_size_bytes: int, num_attn_groups: int,
            num_mamba_groups: int, group_size: int) -> None:
        """Cap mamba layers at `max_num_reqs + 1` recurrent slots and pin
        `cache_config.num_gpu_blocks_override` so the freed HBM grows the
        attention pool.

        Tradeoff vs. the uniform num_blocks layout
        ------------------------------------------
        Mamba state is recurrent: one slot per *active* request, regardless of
        context length. The uniform layout (set by
        `_maybe_set_num_blocks_override`) gives every layer the same
        `num_blocks`, leaving `num_blocks - max_num_reqs` mamba slots idle
        forever. The compact layout caps mamba at `max_num_reqs + 1` (the `+1`
        is the null/sentinel slot), which is strictly better for any model
        where `num_blocks > max_num_reqs` — i.e. all production hybrid configs
        we run. Cost: the GDN op must index mamba state by per-request slot id
        (`AttentionMetadata.mamba_state_indices`) rather than by
        `block_tables[:, 0]`, since the mamba leading dim is now smaller than
        the attention pool.

        Sizing math
        -----------
        vLLM allocates `group_size` KVCacheTensors, each shared across
        `num_attn_groups` attention layers + `num_mamba_groups` mamba layers.
        With per-tensor budget `B = avail / group_size` and
        `N_mamba = max_num_reqs + 1`,
            N_attn = floor((B - num_mamba_groups × N_mamba × mamba_unpadded)
                            / (num_attn_groups × attn_page)).

        We do NOT round to a sharding divisor: our single-process
        TP layout does not shard the mamba/attention block (leading) dim — TP
        sharding is on the head dims via the model — so any positive block
        count is valid.

        Args:
            attn_page_size_bytes: TPU-actual bytes per block per attention
                layer (accounts for dtype packing like fp8).
            unpadded_mamba_page_size_bytes: bytes per slot per mamba layer
                (`prod(shape) × dtype_size`, no padding).
            num_attn_groups: # attention layers backed by each KVCacheTensor.
            num_mamba_groups: # mamba layers backed by each KVCacheTensor.
            group_size: # KVCacheTensors vLLM allocates (= layers per
                kv-cache group); the same value passed to
                `_maybe_set_num_blocks_override`.

        On success: sets `cache_config.num_gpu_blocks_override` (attention
        block count) and `_mamba_num_blocks`. Raises when the KV budget cannot
        hold the mamba slots; the remaining precondition-fail paths leave both
        unset so the caller falls back to uniform sizing.
        """
        if self._uniform_mamba_layout:
            logger.info("Compact mamba sizing skipped.")
            return
        cache_config = self.cache_config
        if cache_config.num_gpu_blocks_override is not None:
            return
        if group_size <= 0:
            return

        avail = self._available_kv_cache_hbm()
        if avail <= 0:
            return

        # +1 reserves slot 0 as the null/sentinel block (never handed to a
        # request); padded tail positions in `mamba_state_indices` also point
        # here, so their writes can never corrupt an active request's state.
        # With speculative decoding each request owns a *group* of
        # `num_spec + 1` consecutive slots so the GDN kernel can checkpoint
        # the state after every speculative window position (see
        # `_init_mamba_slot_pool` for the rollback scheme).
        mamba_num_blocks = self.max_num_reqs * self._mamba_slot_stride + 1

        avail_per_tensor = avail // group_size
        mamba_per_tensor = (num_mamba_groups * mamba_num_blocks *
                            unpadded_mamba_page_size_bytes)
        # Falling back to the uniform layout here would pad every block to the
        # mamba page and silently shrink the pool by ~50x, which surfaces as a
        # throughput collapse rather than a misconfiguration. Fail loudly.
        attn_per_tensor_avail = avail_per_tensor - mamba_per_tensor
        if attn_per_tensor_avail <= 0:
            raise ValueError(
                f"Compact-mamba KV sizing does not fit: mamba slots alone need "
                f"{mamba_per_tensor} B per KVCacheTensor (mamba_num_blocks="
                f"{mamba_num_blocks} x num_mamba_groups={num_mamba_groups} x "
                f"mamba_unpadded={unpadded_mamba_page_size_bytes}), but the "
                f"per-tensor KV budget is {avail_per_tensor} B. Raise "
                f"`gpu_memory_utilization` or lower `max_num_seqs`.")

        attn_num_blocks = attn_per_tensor_avail // (num_attn_groups *
                                                    attn_page_size_bytes)
        if attn_num_blocks <= 0:
            raise ValueError(
                f"Compact-mamba KV sizing does not fit: no attention blocks "
                f"remain (avail_per_tensor={avail_per_tensor} B, "
                f"mamba_per_tensor={mamba_per_tensor} B). Raise "
                f"`gpu_memory_utilization` or lower `max_num_seqs`.")

        cache_config.num_gpu_blocks_override = int(attn_num_blocks)
        self._mamba_num_blocks = int(mamba_num_blocks)

        # Total HBM = group_size KVCacheTensors, each holding num_attn_groups
        # attention blocks + num_mamba_groups mamba slots.
        attn_bytes = (group_size * num_attn_groups * attn_num_blocks *
                      attn_page_size_bytes)
        mamba_bytes = (group_size * num_mamba_groups * mamba_num_blocks *
                       unpadded_mamba_page_size_bytes)
        logger.info(
            "Compact-mamba KV cache: num_gpu_blocks_override=%d (attn), "
            "_mamba_num_blocks=%d. HBM split: attn=%.2f GiB; "
            "mamba=%.2f GiB; total=%.2f GiB / avail=%.2f GiB.",
            attn_num_blocks, mamba_num_blocks, attn_bytes / (2**30),
            mamba_bytes / (2**30), (attn_bytes + mamba_bytes) / (2**30),
            avail / (2**30))

    def _maybe_set_num_blocks_override(self, attn_page_size_bytes: int,
                                       uniform_page_size_bytes: int,
                                       group_size: int) -> None:
        """Pin `cache_config.num_gpu_blocks_override` to the two-step
        flooring value that keeps peak HBM within the user-set
        `gpu_memory_utilization` budget at high utilization.

        Formula:
          `num_blocks_attn = floor(avail / (attn_page × group_size))`
          `num_blocks_tpu  = floor(attn_page × num_blocks_attn / uniform)`

        The two-step flooring can land 1 block lower than vLLM's
        single-step `floor(avail / (uniform × group_size))`. Since
        `uniform > attn_page`, that 1-block gap costs `group_size × uniform`
        bytes of HBM, which — against the imprecision in vLLM's `avail`
        estimate — is enough to tip high-utilization configurations into
        OOM. Pinning to `num_blocks_tpu` preserves the headroom the
        single-step formula silently removes.

        `avail` comes from `_available_kv_cache_hbm`, so the block count
        pinned here matches the KV-cache budget
        `TPUWorker.determine_available_memory()` hands vLLM.

        Skipped only if the user has explicitly set `num_gpu_blocks_override`.

        Args:
            attn_page_size_bytes: TPU-actual bytes per block for one
                attention layer, from `get_attention_page_size_bytes`
                (accounts for dtype packing like fp8).
            uniform_page_size_bytes: bytes per block for one `KVCacheTensor`
                shared across `num_attn_groups + num_mamba_groups` layers
                (the `_hybrid_uniform_page_size_bytes` value set above).
            group_size: number of layers per vLLM kv-cache group, used by
                vLLM to compute `num_blocks` from the attention tensor size.

        Returns:
            None. Side effect: sets `cache_config.num_gpu_blocks_override`
            if all preconditions hold; otherwise leaves it unset.
        """
        cache_config = self.cache_config
        if cache_config.num_gpu_blocks_override is not None:
            return

        avail = self._available_kv_cache_hbm()
        if avail <= 0:
            return

        naive_vllm_num_blocks = avail // (attn_page_size_bytes * group_size)
        if naive_vllm_num_blocks <= 0:
            return
        naive_tensor_size = attn_page_size_bytes * naive_vllm_num_blocks
        num_blocks_tpu = naive_tensor_size // uniform_page_size_bytes
        if num_blocks_tpu <= 0:
            return

        cache_config.num_gpu_blocks_override = int(num_blocks_tpu)
        logger.info(
            "Hybrid KV cache: setting num_gpu_blocks_override=%d to align "
            "the scheduler's block pool with per-layer TPU allocation "
            "(avail=%d, naive_vllm_num_blocks=%d).", num_blocks_tpu, avail,
            naive_vllm_num_blocks)
