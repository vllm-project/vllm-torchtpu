# SPDX-License-Identifier: Apache-2.0
import torch
from vllm.distributed import get_dcp_group, get_pcp_group
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec

from vllm_torchtpu import envs
from vllm_torchtpu.layers.vllm.custom_ops.mamba_state_copy_op import \
    copy_mamba_state_blocks


# Per-step mamba read-offset operations, consolidated into single cached
# TPU programs (avoiding per-group eager launches). Each body is defined
# once; the compiled variant is derived from it, and the dispatchers pick
# eager on CPU (unit tests, pre-init paths) since the tpu backend cannot
# trace CPU tensors.
def _reset_read_offsets_impl(read_offsets: torch.Tensor,
                             keep_dev: torch.Tensor,
                             stacked_groups: torch.Tensor,
                             null: torch.Tensor) -> None:
    """Scatter reset across all mamba groups in a single program.
    `stacked_groups` is [G, width]; `keep_dev` [width] broadcasts across G.
    Non-new rows collapse onto the null slot, where writing 0 is a no-op."""
    targets = torch.where(keep_dev, stacked_groups, null)
    zero = torch.zeros((),
                       dtype=read_offsets.dtype,
                       device=read_offsets.device)
    read_offsets.index_put_((targets.reshape(-1).long(), ), zero)


def _rollback_offsets_seed_impl(read_offsets: torch.Tensor,
                                dst_t: torch.Tensor) -> None:
    read_offsets.index_put_((dst_t, ),
                            torch.zeros(dst_t.shape[0],
                                        dtype=read_offsets.dtype,
                                        device=read_offsets.device))


def _rollback_offsets_migrate_impl(read_offsets: torch.Tensor,
                                   src_t: torch.Tensor,
                                   dst_t: torch.Tensor) -> None:
    read_offsets.index_put_((dst_t, ), read_offsets[src_t])


_tpu_compile = torch.compile(backend="tpu", fullgraph=True, dynamic=False)

_reset_read_offsets_compiled = _tpu_compile(_reset_read_offsets_impl)

_rollback_offsets_seed_compiled = _tpu_compile(_rollback_offsets_seed_impl)

_rollback_offsets_migrate_compiled = _tpu_compile(
    _rollback_offsets_migrate_impl)


def _reset_read_offsets(read_offsets: torch.Tensor, keep_dev: torch.Tensor,
                        stacked_groups: torch.Tensor,
                        null: torch.Tensor) -> None:
    fn = (_reset_read_offsets_compiled
          if read_offsets.device.type == "tpu" else _reset_read_offsets_impl)
    fn(read_offsets, keep_dev, stacked_groups, null)


def _rollback_offsets_seed(read_offsets: torch.Tensor,
                           dst_t: torch.Tensor) -> None:
    fn = (_rollback_offsets_seed_compiled if read_offsets.device.type == "tpu"
          else _rollback_offsets_seed_impl)
    fn(read_offsets, dst_t)


def _rollback_offsets_migrate(read_offsets: torch.Tensor, src_t: torch.Tensor,
                              dst_t: torch.Tensor) -> None:
    fn = (_rollback_offsets_migrate_compiled if read_offsets.device.type
          == "tpu" else _rollback_offsets_migrate_impl)
    fn(read_offsets, src_t, dst_t)


class MambaStateMixin:
    """Mamba/GDN recurrent state pool and seed copies."""

    def _build_mamba_copy_plan(
        self,
        kv_cache_config: KVCacheConfig,
        raw_tensors: list[torch.Tensor],
    ) -> None:
        """Map each mamba kv-cache group to the raw pool buffers hosting its
        layers, for the per-step state-block seed copies.

        Seed copies only exist in mamba align mode, where the state block
        follows the request's last token through the block table. In any
        other mode each request keeps one fixed state block for its whole
        lifetime, so the plan stays empty and the per-step collector is a
        no-op.
        """
        if self.cache_config.mamba_cache_mode != "align":
            self._mamba_copy_plan = []
            self._mamba_state_block_size = None
            return
        layer_to_raw: dict[str, torch.Tensor] = {}
        for raw, kv_cache_tensor in zip(raw_tensors,
                                        kv_cache_config.kv_cache_tensors):
            for layer_name in kv_cache_tensor.shared_by:
                layer_to_raw[layer_name] = raw
        plan: list[tuple[int, list[torch.Tensor]]] = []
        state_block_size: int | None = None
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            if not isinstance(group.kv_cache_spec, MambaSpec):
                continue
            raws: list[torch.Tensor] = []
            seen: set[int] = set()
            for layer_name in group.layer_names:
                raw = layer_to_raw.get(layer_name)
                if raw is not None and id(raw) not in seen:
                    seen.add(id(raw))
                    raws.append(raw)
            if raws:
                plan.append((gid, raws))
                state_block_size = group.kv_cache_spec.block_size
        self._mamba_state_block_size = state_block_size
        self._mamba_copy_plan = plan
        for raw in raw_tensors:
            if raw.dim() > 1:
                kernel_block_size = (raw.shape[-1]
                                     if envs.USE_BATCHED_RPA_SEQ_ON_LANE else
                                     raw.shape[1])
                self._pool_block_split = (self.cache_config.block_size //
                                          kernel_block_size)
                break

    def _collect_mamba_state_seed_copies(self, scheduler_output,
                                         start_index: int,
                                         num_reqs: int) -> None:
        """Stage this chunk's mamba state-block seed copies (the counterpart
        of upstream vLLM's ``preprocess_mamba``).

        The GDN op reads and writes state at the block holding the request's
        last token of the step (see AttentionMetadataBuilder). When that
        block advances — chunked-prefill boundary, decode crossing, or a
        prefix-cache resume, where the previous position is the cached
        boundary-state block — the new block must be seeded from the
        previous one, for every mamba group, on every pool buffer hosting
        that group's layers.
        """
        if not self._mamba_copy_plan:
            return
        live_req_ids = set(self.input_batch.req_id_to_index.keys())
        for req_id in list(self._mamba_state_pos.keys()):
            if req_id not in live_req_ids:
                del self._mamba_state_pos[req_id]

        # Column stride of the mamba block tables, in logical (whole-sequence)
        # tokens. Three block sizes meet here and only one is correct:
        #   * self.block_size is the attention/scheduler size -- unrelated to
        #     mamba tables, whose width is cdiv(max_model_len, mamba size) --
        #     and a stale __init__-time snapshot besides (the executor
        #     finalizes cache_config.block_size only after the runner is
        #     constructed);
        #   * the mamba groups' kv_cache_spec.block_size (the copy of
        #     cache_config.mamba_block_size that get_kv_cache_spec bakes into
        #     every MambaSpec, which then sizes these tables) is physical and
        #     per-rank, and cannot be stale: specs are created only after the
        #     platform finalizes block sizes;
        #   * the scheduler appends a block id to a mamba row only once per
        #     mamba_block_size * total_cp_world logical tokens, because the
        #     PCP coordinator patch presents specs to the hybrid coordinator
        #     at that granularity (_patch_vllm_hybrid_pcp_block_sizes). Mamba
        #     state is never token-sharded across CP ranks; the cp factor
        #     exists purely to mirror that allocation cadence.
        # Every reader of these tables must divide logical token counts by
        # the allocator's stride. This is the same derivation
        # AttentionMetadataBuilder.target_block_size uses for its state-slot
        # lookup -- spec block size times cp world -- so the two readers
        # cannot diverge. A smaller divisor visits columns the scheduler
        # never filled: beyond the row width it raises IndexError, within it
        # the null block silently swallows the recurrent state.
        try:
            cp_world_size = (get_dcp_group().world_size *
                             get_pcp_group().world_size)
        except Exception:
            cp_world_size = 1
        col_stride = self._mamba_state_block_size * cp_world_size
        num_computed = self.input_batch.num_computed_tokens_cpu
        req_ids = self.input_batch.req_ids
        crossings: list[tuple[int, int, int]] = []
        for i in range(num_reqs):
            row = start_index + i
            req_id = req_ids[row]
            assert req_id is not None
            computed = int(num_computed[row])
            scheduled = scheduler_output.num_scheduled_tokens[req_id]
            curr = (computed + scheduled - 1) // col_stride
            prev = self._mamba_state_pos.get(req_id,
                                             (computed - 1) // col_stride)
            self._mamba_state_pos[req_id] = curr
            if 0 <= prev != curr:
                crossings.append((row, prev, curr))
        if not crossings:
            return

        # Speculative decoding: the checkpoint group slides with the state
        # column. vLLM's MambaManager appends one block and never shifts, so
        # after the crossing what was checkpoint 1 IS the state block,
        # checkpoint 2 is checkpoint 1, and a fresh block appears at the
        # end. Carrying the old state block over would therefore leave the
        # request resuming one checkpoint late (and, at the top of the
        # window, off an uninitialized block). What must move instead is the
        # checkpoint the request is about to resume from — which becomes
        # checkpoint 0 of the new group, so the read offset resets to 0.
        # That index lives in the device-resident read-offset buffer, hence
        # the device-side gather in `_spec_seed_sources`.
        window = self._mamba_ckpt_window
        spec_seed = self.mamba_slot_read_offsets is not None and window > 1

        # Keyed by the set of raw buffers a group lives on; groups sharing
        # the same buffers are seeded by one program.
        per_raw: dict[tuple[int, ...], tuple[list[torch.Tensor], list[int],
                                             list[int]]] = {}
        per_raw_dev: dict[tuple[int, ...],
                          tuple[list[torch.Tensor], list[torch.Tensor],
                                list[torch.Tensor]]] = {}
        # Manager-level (src, dst) block pairs across mamba groups: without
        # spec decoding the per-block read offsets just follow the state to
        # the request's new state block.
        offset_pairs: list[tuple[int, int]] = []
        spec_dsts: list[int] = []
        for gid, raws in self._mamba_copy_plan:
            bt = self.input_batch.block_table[gid].get_cpu_tensor()
            row_width = bt.shape[1]
            pairs: list[tuple[int, int]] = []
            state_srcs: list[int] = []
            ckpt_rows: list[list[int]] = []
            for row, prev, curr in crossings:
                src = int(bt[row, prev])
                dst = int(bt[row, curr])
                if src == dst or src == 0 or dst == 0:
                    continue
                pairs.append((src, dst))
                if spec_seed:
                    state_srcs.append(src)
                    # The pre-crossing group's checkpoint blocks. A column
                    # the manager has not filled reads past the row or lands
                    # on the null block; either way fall back to the state
                    # block, so a short row seeds from live state instead of
                    # from nothing. The offset never names one in practice —
                    # it is bounded by the accepted count.
                    row_ckpts = []
                    for t in range(window):
                        col = prev + t
                        ckpt = int(bt[row, col]) if col < row_width else 0
                        row_ckpts.append(ckpt if ckpt != 0 else src)
                    ckpt_rows.append(row_ckpts)
            if not pairs:
                continue
            if spec_seed:
                dsts_g = [d for _, d in pairs]
                spec_dsts.extend(dsts_g)
                src_t = self._spec_seed_sources(state_srcs, ckpt_rows)
                dst_t = torch.tensor(dsts_g,
                                     dtype=torch.int32).to(self.device,
                                                           non_blocking=True)
                src_t, dst_t = self._expand_pool_split(src_t, dst_t)
                key = tuple(id(raw) for raw in raws)
                entry = per_raw_dev.setdefault(key, (raws, [], []))
                entry[1].append(src_t)
                entry[2].append(dst_t)
                continue
            offset_pairs.extend(pairs)
            if self._pool_block_split > 1:
                # The pool is born at kernel granularity: a manager state
                # block is `split` consecutive pool blocks.
                pairs = [(s * self._pool_block_split + j,
                          d * self._pool_block_split + j) for s, d in pairs
                         for j in range(self._pool_block_split)]
            key = tuple(id(raw) for raw in raws)
            entry = per_raw.setdefault(key, (raws, [], []))
            entry[1].extend(p[0] for p in pairs)
            entry[2].extend(p[1] for p in pairs)

        if spec_seed and spec_dsts:
            # The seeded checkpoint is checkpoint 0 of the new group.
            dst_t = torch.tensor(self._pad_to_bucket(spec_dsts),
                                 dtype=torch.long).to(self.device,
                                                      non_blocking=True)
            assert self.mamba_slot_read_offsets is not None
            _rollback_offsets_seed(self.mamba_slot_read_offsets, dst_t)
        elif self.mamba_slot_read_offsets is not None and offset_pairs:
            # Migrate the read offsets with the state: gather at the old
            # blocks, scatter at the new ones (both device-side, before the
            # forward that reads the buffer). Padding with (0, 0) null-block
            # self-copies keeps the shapes on the seed-copy bucket ladder.
            offset_pairs = self._pad_to_bucket(offset_pairs, pad=(0, 0))
            src_t = torch.tensor([s for s, _ in offset_pairs],
                                 dtype=torch.long).to(self.device,
                                                      non_blocking=True)
            dst_t = torch.tensor([d for _, d in offset_pairs],
                                 dtype=torch.long).to(self.device,
                                                      non_blocking=True)
            _rollback_offsets_migrate(self.mamba_slot_read_offsets, src_t,
                                      dst_t)

        for raws, srcs_dev, dsts_dev in per_raw_dev.values():
            src_t = self._pad_dev_to_bucket(torch.cat(srcs_dev))
            dst_t = self._pad_dev_to_bucket(torch.cat(dsts_dev))
            self._pending_mamba_state_copies.append((raws, src_t, dst_t))

        for raws, srcs, dsts in per_raw.values():
            src_t = torch.tensor(self._pad_to_bucket(srcs),
                                 dtype=torch.int32).to(self.device,
                                                       non_blocking=True)
            dst_t = torch.tensor(self._pad_to_bucket(dsts),
                                 dtype=torch.int32).to(self.device,
                                                       non_blocking=True)
            self._pending_mamba_state_copies.append((raws, src_t, dst_t))

    @staticmethod
    def _bucket_len(n: int) -> int:
        """Smallest ladder length >= `n`, so the copy op sees few shapes."""
        padded = 8
        while padded < n:
            padded *= 4
        return padded

    @classmethod
    def _pad_to_bucket(cls, values: list, pad=0) -> list:
        """Pad a CPU pair/id list up to the seed-copy bucket ladder.

        Pads are the null block, i.e. `(0, 0)` self-copies that move no
        live state.
        """
        return values + [pad] * (cls._bucket_len(len(values)) - len(values))

    def _pad_dev_to_bucket(self, ids: torch.Tensor) -> torch.Tensor:
        """`_pad_to_bucket` for an already-on-device id tensor."""
        n = int(ids.shape[0])
        padded = self._bucket_len(n)
        if padded == n:
            return ids
        return torch.cat([
            ids,
            torch.zeros(padded - n, dtype=ids.dtype, device=ids.device),
        ])

    def _expand_pool_split(
            self, src: torch.Tensor,
            dst: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Device twin of the manager-block -> pool-block expansion.

        The pool is born at kernel granularity: a manager state block is
        `split` consecutive pool blocks.
        """
        split = self._pool_block_split
        if split <= 1:
            return src, dst
        j = torch.arange(split, dtype=src.dtype, device=src.device)
        return ((src.unsqueeze(1) * split + j).reshape(-1),
                (dst.unsqueeze(1) * split + j).reshape(-1))

    def _spec_seed_sources(self, state_srcs: list[int],
                           ckpt_rows: list[list[int]]) -> torch.Tensor:
        """Seed source per crossing: the checkpoint being resumed from.

        `mamba_slot_read_offsets` is device-resident and indexed by block
        id, so which of the window's blocks holds the last accepted token's
        state is only known on device — the choice is a gather there rather
        than a CPU lookup.
        """
        offsets = self.mamba_slot_read_offsets
        assert offsets is not None
        state_t = torch.tensor(state_srcs,
                               dtype=torch.long).to(self.device,
                                                    non_blocking=True)
        ckpt_t = torch.tensor(ckpt_rows,
                              dtype=torch.int32).to(self.device,
                                                    non_blocking=True)
        off = offsets[state_t].long().clamp_(0, ckpt_t.shape[1] - 1)
        return ckpt_t.gather(1, off.unsqueeze(1)).squeeze(1)

    def _flush_mamba_state_seed_copies(self) -> None:
        if not self._pending_mamba_state_copies:
            return
        for raws, src_t, dst_t in self._pending_mamba_state_copies:
            copy_mamba_state_blocks(raws, src_t, dst_t)
        self._pending_mamba_state_copies.clear()
