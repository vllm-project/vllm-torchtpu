# SPDX-License-Identifier: Apache-2.0
import torch
from vllm.distributed import get_dcp_group, get_pcp_group
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec

from vllm_torchtpu import envs
from vllm_torchtpu.layers.vllm.custom_ops.mamba_state_copy_op import \
    copy_mamba_state_blocks
from vllm_torchtpu.runner.tpu_runner_async_output import INVALID_TOKEN_ID
from vllm_torchtpu.utils import synchronize_tensors


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

    def _init_mamba_slot_pool(self, mamba_num_blocks: int) -> None:
        """(Re)initialize the per-request mamba recurrent-slot allocator.

        Slot 0 is the null/sentinel block (used for padded persistent-batch
        positions and never handed to a request). Usable slots are
        `[1, mamba_num_blocks)`, popped from the high end so the first request
        gets slot 1 (stable pool ordering for easier debugging).
        Called once after KV cache allocation, when the true mamba block count
        is known. The mapping itself is keyed on req_id and rebuilt lazily in
        `_build_mamba_state_indices`, so re-init only resets the free pool.

        With speculative decoding each request owns a *group* of
        `num_speculative_tokens + 1` consecutive slots; the tracked slot id is
        the group's base. During a verify step the GDN kernel writes one state
        checkpoint per window position into `base + t`, and the next step
        reads its initial state from `base + (num_accepted - 1)`. Rejected
        drafts are thus rolled back by *selection*, never by copying. Without
        spec decode the stride is 1 (one slot per request), matching the
        original layout.
        """
        stride = self._mamba_slot_stride
        num_groups = (mamba_num_blocks - 1) // stride
        self._mamba_slot_by_req_id = {}
        self._free_mamba_slots = [
            1 + g * stride for g in reversed(range(num_groups))
        ]
        self._has_mamba_state = True

    def _build_mamba_state_indices(self, start_index: int, num_reqs: int,
                                   target_num_reqs: int) -> torch.Tensor:
        """Build the device `mamba_state_indices` for the current chunk.

        Reconciles the req_id->slot map against the live persistent batch:
          * frees slots whose req_id is no longer present (request finished),
          * allocates a fresh unique slot for any new req_id.
        Returns an int32 device tensor of length `target_num_reqs`: positions
        `[0, num_reqs)` hold the slot for `req_ids[start_index + i]`; the
        padded tail `[num_reqs, target_num_reqs)` holds slot 0 (null block) so
        the GDN op (which scans the full length every step) cannot alias an
        active slot — the null-tail invariant.

        `num_reqs` may exceed the requests actually in the persistent batch:
        an idle DP rank replaying a peer's collective trace passes the
        coordinated shape (`num_reqs_max_model_len`) while its own batch is
        empty or shorter. Those positions carry no request, so they keep the
        null slot the tail already uses; only the live prefix is assigned.
        Without this, `run_dp_dummy_draft` walks off the end of `req_ids`
        with `IndexError` on the compact layout, killing the engine.

        Slot assignment follows the request through upstream vLLM's
        condense/swap reordering automatically because the map is keyed on the
        stable req_id, not the moving persistent-batch position.
        """
        live_req_ids = set(self.input_batch.req_id_to_index.keys())
        # Free slots for requests that left the batch.
        for req_id in list(self._mamba_slot_by_req_id.keys()):
            if req_id not in live_req_ids:
                self._free_mamba_slots.append(
                    self._mamba_slot_by_req_id.pop(req_id))

        indices = self.mamba_state_indices_cpu[:target_num_reqs]
        indices.fill_(0)
        req_ids = self.input_batch.req_ids
        num_live = max(0, min(num_reqs, len(req_ids) - start_index))
        for i in range(num_live):
            req_id = req_ids[start_index + i]
            assert req_id is not None
            slot = self._mamba_slot_by_req_id.get(req_id)
            if slot is None:
                # New request: take a fresh unique slot. The pool is sized to
                # max_num_reqs + 1, so it can never be exhausted while the
                # batch holds at most max_num_reqs requests.
                slot = self._free_mamba_slots.pop()
                self._mamba_slot_by_req_id[req_id] = slot
            indices[i] = slot
        return indices.to(self.device, non_blocking=True)

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

    @torch.no_grad()
    def _update_mamba_slot_read_offsets(self,
                                        indices_per_group: list[torch.Tensor]
                                        | None,
                                        next_tokens: torch.Tensor | None,
                                        num_reqs: int) -> None:
        """Scatter this chunk's mamba read offsets into the slot-indexed buffer.

        Each entry holds one mamba group's base slot per batch position for
        the chunk (padded tail = slot 0, the null block, so its writes are
        harmless); all groups get the same offsets, since the buffer is
        indexed by slot/block id and groups allocate distinct ones. The
        compact pool shares one slot tensor across groups and so passes a
        single-element list.

        The offset is `num_accepted - 1` derived from the rejection-sampler
        output for verify chunks, and 0 for non-spec chunks (prefill / plain
        decode) — which also resets a group's offset after prefill. The next
        step's GDN kernel resumes each request from checkpoint `offset` of
        its state block (the checkpoint of its last accepted token).
        """
        if self.mamba_slot_read_offsets is None or not indices_per_group:
            return
        offsets = torch.zeros(indices_per_group[0].shape[0],
                              dtype=torch.int32,
                              device=indices_per_group[0].device)
        if next_tokens is not None:
            # num_valid = accepted drafts + 1 (bonus); the checkpoint of the
            # last accepted token is at offset num_valid - 1.
            num_valid = (next_tokens[:num_reqs]
                         != INVALID_TOKEN_ID).sum(dim=1).to(torch.int32)
            offsets[:num_reqs] = (num_valid - 1).clamp(min=0)
        if len(indices_per_group) == 1:
            self.mamba_slot_read_offsets.index_put_(
                (indices_per_group[0].long(), ), offsets)
        else:
            all_indices = torch.cat(indices_per_group, dim=0).long()
            all_offsets = offsets.repeat(len(indices_per_group))
            self.mamba_slot_read_offsets.index_put_((all_indices, ),
                                                    all_offsets)

    def _mamba_state_index_groups(self) -> list[torch.Tensor] | None:
        """This chunk's mamba state slot/block ids, one tensor per group.

        The compact pool shares one slot tensor across every mamba group, so
        it yields a single-element list; the unified pool allocates a
        distinct state block per group and the group builders collect them
        during the metadata build.
        """
        ctx = self._attn_metadata_builder_ctx
        if self.mamba_slot_read_offsets is None or ctx is None:
            return None
        if ctx.mamba_state_indices is not None:
            return [ctx.mamba_state_indices]
        return ctx.unified_mamba_state_indices

    @torch.no_grad()
    def _reset_read_offsets_for_new_requests(self, start_index: int,
                                             num_reqs: int) -> None:
        """Zero the read offset of every request entering the batch.

        `mamba_slot_read_offsets` is indexed by slot/block id, but a slot
        outlives the request that wrote it. A request that leaves the batch
        mid-verify-window leaves `num_accepted - 1` behind, and both layouts
        hand that storage straight to the next request:

          * compact -- `_build_mamba_state_indices` returns the slot to
            `_free_mamba_slots` and pops it for a new req_id;
          * unified -- vLLM's block manager frees the state block and
            reallocates it.

        Nothing else clears the entry. `_update_mamba_slot_read_offsets`
        only covers requests already in the batch and runs *after* the
        forward, and the seed-copy migration is doubly gated (align mode
        and the unified layout), so it never runs for a compact/`none`
        deployment at all. The new owner's first forward would then resume
        from a checkpoint belonging to the previous conversation — silent
        GDN state corruption, not a crash.

        The scatter is issued UNCONDITIONALLY, every step, whether or not
        any request is new. Under EP-DP lockstep every rank must run the
        same device program, and which requests are new is rank-local: an
        early return on "nothing to reset" makes one rank skip device work
        its peers perform, they fall out of step, and the next collective
        deadlocks. The mask makes an all-old batch a no-op instead —
        already-seen requests and padded positions are redirected to slot 0,
        the null block, whose offset is 0 regardless.
        """
        if self.mamba_slot_read_offsets is None:
            return
        groups = self._mamba_state_index_groups()
        if not groups:
            return
        req_ids = self.input_batch.req_ids
        # Forget requests that have left, so a later req_id reusing the same
        # string is still treated as new.
        self._mamba_offset_seeded &= set(
            self.input_batch.req_id_to_index.keys())

        width = groups[0].shape[0]
        keep, keep_dev, null = self._read_offset_reset_scratch(
            width, groups[0].device)
        keep.zero_()
        for i in range(min(num_reqs, max(0, len(req_ids) - start_index))):
            req_id = req_ids[start_index + i]
            if req_id is None or i >= width:
                continue
            if req_id not in self._mamba_offset_seeded:
                self._mamba_offset_seeded.add(req_id)
                keep[i] = True
        # No `if nothing_new: return` here — see the docstring. The device
        # work below must be issued on every rank on every step.
        keep_dev.copy_(keep, non_blocking=True)
        # Stack groups and reset offsets across all groups in one compiled program.
        stacked = torch.stack(groups) if len(
            groups) > 1 else groups[0].unsqueeze(0)
        _reset_read_offsets(self.mamba_slot_read_offsets, keep_dev, stacked,
                            null)

    def _read_offset_reset_scratch(
        self, width: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reusable `(keep_cpu, keep_dev, null)` of length `width`.

        `_reset_read_offsets_for_new_requests` runs on every step, so its
        working set is allocated once per width rather than per step. Width
        is the padded batch bucket, so this holds one entry per bucket —
        a handful for the life of the process. `null` is a 0-d zero that
        broadcasts in the `torch.where`, replacing a full-width
        `zeros_like` per mamba group per step.
        """
        assert self.mamba_slot_read_offsets is not None
        key = (width, str(device))
        entry = self._read_offset_scratch.get(key)
        if entry is None:
            entry = (
                torch.zeros(width, dtype=torch.bool),
                torch.zeros(width, dtype=torch.bool, device=device),
                torch.zeros((), dtype=torch.int32, device=device),
            )
            self._read_offset_scratch[key] = entry
        return entry

    def _precompile_mamba_state_seed_copies(self) -> None:
        """Warm the seed-copy program for every raw-pool set at every bucket
        length, so the first real block-boundary crossing does not compile
        mid-serving. Runs right after the pool zero-fill sync: the pools must
        be quiescent when their donation is first compiled (see the note in
        initialize_kv_cache). All-zero indices copy the null block onto
        itself; src and dst are distinct tensors (dynamo specializes on
        src-is-dst otherwise).
        """
        if not self._mamba_copy_plan:
            return
        raw_sets: dict[tuple[int, ...], list[torch.Tensor]] = {}
        for _, raws in self._mamba_copy_plan:
            raw_sets.setdefault(tuple(id(r) for r in raws), raws)
        # Serving calls the program under execute_model's no_grad, and dynamo
        # guards on grad mode: compile it the same way.
        with self._precompile_timed(
                "mamba state seed copies"), torch.no_grad():
            limit = self._bucket_len(
                len(self._mamba_copy_plan) * self.max_num_reqs *
                self._pool_block_split)
            n = 8
            while n <= limit:
                src = torch.zeros(n, dtype=torch.int32).to(self.device)
                dst = torch.zeros(n, dtype=torch.int32).to(self.device)
                for raws in raw_sets.values():
                    copy_mamba_state_blocks(raws, src, dst)
                n *= 4
            for raws in raw_sets.values():
                synchronize_tensors(raws)

    def _precompile_mamba_rollback_helpers(self) -> None:
        """Warm the compiled rollback scatters at every bucket length.

        Mamba block-boundary crossings do not occur during the synthetic
        warmup request (unlike the reset scatter, which the real warmup
        path exercises), so without this the first real crossing would
        compile XLA mid-serving while lockstep peers wait at their next
        collective. All-zero indices write the null slot, moving no state.
        Crossings accumulate across every mamba group in a step, so the
        ladder tops out at num_groups * max_num_reqs; src and dst must be
        DISTINCT tensors or dynamo specializes the migrate graph on
        src-is-dst and the real two-tensor call retraces mid-serving.
        """
        offsets = self.mamba_slot_read_offsets
        if offsets is None:
            return
        with self._precompile_timed("mamba rollback helpers"):
            num_groups = max(1, len(self._mamba_copy_plan))
            limit = self._bucket_len(num_groups * self.max_num_reqs)
            n = 8
            while n <= limit:
                src = torch.zeros(n, dtype=torch.long, device=offsets.device)
                dst = torch.zeros(n, dtype=torch.long, device=offsets.device)
                _rollback_offsets_seed_compiled(offsets, dst)
                _rollback_offsets_migrate_compiled(offsets, src, dst)
                n *= 4
