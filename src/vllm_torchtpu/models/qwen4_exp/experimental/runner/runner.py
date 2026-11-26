# Copyright 2026 Google LLC
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
"""Qwen3.8-Flash runner: `TPUModelRunner` with the model's step changes.

Registered for the Qwen4Exp architectures through `register_model_runner`
(see `hooks.register()`); every other model keeps `TPUModelRunner`. The
overrides are copies of the runner methods with these changes applied, so the
served programs and their order are those of the PR code:

- #2605 one compiled program for the post-verify read-offset update
  (`_update_mamba_slot_read_offsets`, `_update_read_offsets_impl`);
- #2722 launch the draft before waiting on the previous D2H copy
  (`sample_tokens`);
- #2607 speculative seed-copy sources selected in one compiled program
  (`_collect_mamba_state_seed_copies`, `_spec_seed_copies`,
  `_precompile_mamba_rollback_helpers`, `_spec_seed_rows_impl`);
- #2609 reuse unchanged sampling-parameter uploads and the num_reqs scalar
  (`_init_speculative_decoding`, `_upload_keyed`, `_num_reqs_tensor`,
  `_build_padded_sampling_params`, `_padded_sampling_params`,
  `_update_mamba_slot_read_offsets`);
- #2650 one compiled program for the one-chunk substitution tail
  (`_assemble_async_spec_substitution`, `_precompile_spec_tail`, run right
  after `_precompile_spec_boundary_copies` as in `capture_model`);
- #2651 token padding: skip the mask upload when its rows are unchanged
  (`SkipUnchangedTokenPaddingState`, swapped in after `load_model`);
- #2723 record stage timing only with a KV connector sink
  (`SinkStageTiming`, swapped in after `__init__`).

Re-sync the copies on rebases; tests/.../runner/test_runner.py checks that
every override still names a `TPUModelRunner` method.
"""

import contextlib
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from torch_tpu._internal.stream.streams import TpuEvent
from vllm.distributed.kv_transfer import has_kv_transfer_group
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.spec_decode.ngram_proposer import NgramProposer

from vllm_torchtpu.layers.adapter import token_padding
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4.deepseek_v4_indexer import (
    static_verify_q_len,
)
from vllm_torchtpu.layers.adapter.sample.rejection_sampler import RejectionSampler
from vllm_torchtpu.layers.adapter.token_padding import TokenPaddingState
from vllm_torchtpu.layers.core.attention_metadata import stage_h2d_upload
from vllm_torchtpu.runner import (
    greedy_top_tokens,
    prompt_logprobs,
    spec_boundary_state,
)
from vllm_torchtpu.runner.stage_timing import RunnerStageTiming, _PendingTiming
from vllm_torchtpu.runner.tpu_pooling import (
    StaticPoolerChunkOutput,
    unpack_static_pool_output,
)
from vllm_torchtpu.runner.tpu_runner import (
    TPUModelRunner,
    _get_padded_num_reqs_with_upper_limit,
    _rollback_offsets_migrate,
    _rollback_offsets_migrate_compiled,
    _rollback_offsets_seed,
    _rollback_offsets_seed_compiled,
    _tpu_compile,
)
from vllm_torchtpu.runner.tpu_runner_async_output import (
    INVALID_TOKEN_ID,
    AsyncPreResults,
    AsyncTPUCopyState,
    AsyncTPUModelRunnerOutput,
    LogprobsChunk,
)
from vllm_torchtpu.spec_decode.dflash import DFlashProposer
from vllm_torchtpu.spec_decode.dspark import DSparkProposer
from vllm_torchtpu.spec_decode.eagle3 import (
    Eagle3Proposer,
    _static_shape_args,
)
from vllm_torchtpu.spec_decode.utils import (
    fit_rows,
    handoff_seed_tokens,
)
from vllm_torchtpu.utils import synchronize_tensors

from ..metadata_access import mamba_ckpt_of, mamba_state_of
from .metadata import is_ping_pong_mamba_spec
from .two_slot import (
    PingPongMambaChunkState,
    compute_boundary_copy_and_interior_compiled,
    compute_boundary_copy_and_interior_impl,
    compute_boundary_copy_no_verify_compiled,
    compute_boundary_copy_no_verify_impl,
    compute_replay_commit_lens,
    expand_split_masked,
    get_or_build_multi_layer_two_slot_replay,
    has_ping_pong_copy_groups,
    is_ping_pong_copy_gid,
    ping_pong_layers_match_group_order,
    spec_boundary_pairs_compiled,
    spec_boundary_pairs_impl,
    two_slot_entries_shape,
    update_ping_pong_read_offsets_compiled,
    update_ping_pong_read_offsets_impl,
)
from .two_slot import (
    SpecBoundaryPlan as TwoSlotBoundaryPlan,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig

    from vllm_torchtpu.block_major_pool import BlockMajorPoolLayout


def _update_read_offsets_impl(
    read_offsets: torch.Tensor,
    groups: list[torch.Tensor],
    next_tokens: torch.Tensor,
    num_reqs: torch.Tensor,
) -> torch.Tensor:
    """Post-verify read-offset scatter for every mamba group in one program.
    Each of `groups` is a [width] slot tensor; `next_tokens` the rejection
    sampler output ([>= num_reqs, K+1], INVALID_TOKEN_ID-padded); `num_reqs`
    a 0-d tensor, so it enters as a value, not a shape. Rows past it take
    offset 0 (their slots are the null block). Returns the offsets, one per
    batch position."""
    width = groups[0].shape[0]
    rows = next_tokens.shape[0]
    if rows >= width:
        next_tokens = next_tokens[:width]
    else:
        next_tokens = torch.nn.functional.pad(
            next_tokens, (0, 0, 0, width - rows), value=INVALID_TOKEN_ID
        )
    num_valid = (next_tokens != INVALID_TOKEN_ID).sum(dim=1)
    valid = torch.arange(width, device=read_offsets.device) < num_reqs
    offsets = torch.where(valid, (num_valid - 1).clamp(min=0), 0).to(read_offsets.dtype)
    indices = torch.cat(groups, dim=0).long()
    read_offsets.index_put_((indices,), offsets.repeat(len(groups)))
    return offsets


def _use_fused_read_offset_update(
    read_offsets: torch.Tensor, next_tokens: torch.Tensor | None
) -> bool:
    """Verify chunks take `_update_read_offsets_compiled`; non-verify chunks
    (`next_tokens is None`, offsets all 0) take the eager scatter. Serving
    buffers are always on TPU; the device check only routes the CPU unit tests,
    which the tpu backend cannot trace, to the eager path."""
    return next_tokens is not None and read_offsets.device.type == "tpu"


def _spec_seed_rows_impl(
    read_offsets: torch.Tensor,
    packed: torch.Tensor,
    split: int,
    out_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Seed-copy (src, dst) pool rows for a step's speculative crossings.

    `packed` is int64 [B, 4 + window], one row per crossing: state source block,
    new-owner flag, valid flag (0 on bucket padding), destination block, then the
    pre-crossing group's checkpoint blocks. The source is the checkpoint the
    request resumes from, picked by its device-resident read offset (checkpoint 0
    for a new owner: a recycled source block can carry another request's offset).
    Each manager block fans out to `split` consecutive pool blocks, sized to
    `out_len` on the seed-copy bucket ladder; padding rows become (0, 0)
    null-block self-copies.
    """
    state, dst, ckpt = packed[:, 0], packed[:, 3], packed[:, 4:]
    off = read_offsets[state].long().clamp(0, ckpt.shape[1] - 1)
    off = torch.where(packed[:, 1] != 0, 0, off)
    src = ckpt.gather(1, off.unsqueeze(1)).squeeze(1)
    valid = packed[:, 2] != 0
    if split > 1:
        j = torch.arange(split, dtype=src.dtype, device=src.device)
        src = src.unsqueeze(1) * split + j
        dst = dst.unsqueeze(1) * split + j
        valid = valid.unsqueeze(1).expand(-1, split)
    src = torch.where(valid, src, 0).reshape(-1).to(torch.int32)
    dst = torch.where(valid, dst, 0).reshape(-1).to(torch.int32)
    n = src.shape[0]
    if n < out_len:
        src = torch.nn.functional.pad(src, (0, out_len - n))
        dst = torch.nn.functional.pad(dst, (0, out_len - n))
    elif n > out_len:
        src = src[:out_len]
        dst = dst[:out_len]
    return src, dst


def _spec_tail(next_tokens, drafts, draft_lengths, rows, num_reqs, bound):
    """One async-spec step's substitution source, padded to `bound`, and its
    per-request rejected count: `assemble_spec_next_tokens`,
    `compute_num_rejected`, the row fits and the source pad in one function."""
    from vllm_torchtpu.runner.tpu_runner_async_output import (
        assemble_spec_next_tokens,
        compute_num_rejected,
    )
    from vllm_torchtpu.spec_decode.utils import fit_rows

    src = assemble_spec_next_tokens(next_tokens, fit_rows(drafts, rows), num_reqs)
    num_rejected = compute_num_rejected(next_tokens, fit_rows(draft_lengths, rows))
    if src.shape[0] < bound:
        src = torch.nn.functional.pad(src, (0, bound - src.shape[0]))
    return src, num_rejected


_update_read_offsets_compiled = _tpu_compile(_update_read_offsets_impl)
_spec_seed_rows_compiled = _tpu_compile(_spec_seed_rows_impl)
_spec_tail_compiled = _static_shape_args(
    torch.compile(_spec_tail, backend="tpu", fullgraph=True, dynamic=False)
)


@dataclass
class SkipUnchangedTokenPaddingState(TokenPaddingState):
    """`TokenPaddingState` whose `update` skips the H2D for unchanged rows
    (#2651)."""

    _last_key: tuple[int, int] | None = None

    def update(self, local_valid_tokens: int, local_padded_tokens: int) -> None:
        """Write this rank's token padding mask."""
        key = (local_valid_tokens, local_padded_tokens)
        # The mask only ever changes here, so the same rows as the last write
        # leave it exact: skip the synchronous H2D.
        if self._last_key == key:
            return
        host = torch.zeros(self.local_padding_mask.shape[0], dtype=torch.bool)
        host[local_valid_tokens:local_padded_tokens] = True
        # Synchronous copy since the host tensor is temporary.
        self.local_padding_mask.copy_(host)
        self._last_key = key


class SinkStageTiming(RunnerStageTiming):
    """`RunnerStageTiming` that records only when a KV connector can report
    the samples (#2723)."""

    def __init__(self, rank: int):
        super().__init__(rank)
        # Whether a KV connector exists to report samples through.
        self._sink = False

    def _has_sink(self) -> bool:
        """Samples only reach the metrics through a KV connector's stats
        (`TPUModelRunner.get_finished_kv_transfers`); without one they are
        discarded every step, so skip recording them -- each device timing
        costs two TpuEvent records on the serving thread."""
        if not self._sink:
            # Re-checked until true: warmup can run before the connector exists.
            self._sink = has_kv_transfer_group()
        return self._sink

    @contextlib.contextmanager
    def host_stage(self, stage: str) -> Iterator[None]:
        if not self._has_sink():
            yield
            return
        cpu_start = time.perf_counter()
        yield
        self._cpu_samples.append((self._rank, stage, time.perf_counter() - cpu_start))

    @contextlib.contextmanager
    def stage(self, stage: str) -> Iterator[None]:
        if not self._has_sink():
            yield
            return
        if len(self._pending) >= self.MAX_IN_FLIGHT_DEVICE_TIMINGS:
            with self.host_stage(stage):
                yield
            return

        start = TpuEvent(enable_timing=True)
        end = TpuEvent(enable_timing=True)
        start.record()
        pending = _PendingTiming(stage, start, end)
        self._pending.append(pending)
        cpu_start = time.perf_counter()
        # Close the timing even when the stage body raises. An active outer
        # context can remain at the queue head while a nested context drains.
        try:
            yield
        finally:
            cpu_elapsed = time.perf_counter() - cpu_start
            end.record()
            pending.end_recorded = True
            self._cpu_samples.append((self._rank, stage, cpu_elapsed))


class Qwen38Runner(TPUModelRunner):
    """`TPUModelRunner` with the Qwen3.8-Flash step changes (module docstring)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # #2723: stage timings are recorded only with a KV connector sink.
        self._stage_timing = SinkStageTiming(int(self._stage_timing._rank))
        self._mamba_copy_group_windows: dict[int, int] = {}
        self._mamba_copy_ping_pong_gids: set[int] = set()
        self._two_slot_in_execute_model: bool = False
        self._two_slot_chunk_records: list[dict[str, Any]] = []

    def _build_mamba_copy_plan(
        self,
        kv_cache_config: "KVCacheConfig",
        layer_to_raw: dict[str, torch.Tensor],
        bm_pool_layout: "BlockMajorPoolLayout | None",
    ) -> None:
        super()._build_mamba_copy_plan(kv_cache_config, layer_to_raw, bm_pool_layout)
        from vllm.v1.kv_cache_interface import MambaSpec

        group_windows: dict[int, int] = {}
        ping_pong_gids: set[int] = set()
        if self.cache_config.mamba_cache_mode == "align":
            plan_gids = {gid for gid, _ in self._mamba_copy_plan}
            for gid, group in enumerate(kv_cache_config.kv_cache_groups):
                if not isinstance(group.kv_cache_spec, MambaSpec):
                    continue
                spec_blocks = int(
                    getattr(group.kv_cache_spec, "num_speculative_blocks", 0) or 0
                )
                if spec_blocks > 0:
                    group_windows[gid] = 1 + spec_blocks
                if gid in plan_gids and is_ping_pong_mamba_spec(
                    group.kv_cache_spec, getattr(self, "vllm_config", None)
                ):
                    ping_pong_gids.add(gid)
        self._mamba_copy_group_windows = group_windows
        self._mamba_copy_ping_pong_gids = ping_pong_gids

    def _mamba_state_index_groups(self) -> list[torch.Tensor] | None:
        ctx = self._attn_metadata_builder_ctx
        if self.mamba_slot_read_offsets is None or ctx is None:
            return None
        if ctx.mamba_state_indices is not None:
            return [ctx.mamba_state_indices]
        combined = list(ctx.unified_mamba_state_indices or ()) + list(
            getattr(ctx, "unified_ping_pong_mamba_state_indices", None) or ()
        )
        return combined or None

    def _get_two_slot_gdn_layers(self) -> tuple[Any, ...]:
        model = getattr(self, "model", None)
        if model is None:
            return ()
        cached = getattr(self, "_two_slot_gdn_layers_cache", None)
        if cached is not None and cached[0] is model:
            return cached[1]
        found: list[Any] = []
        backbone = getattr(model, "model", model)
        for layer in getattr(backbone, "layers", ()):
            lin = getattr(layer, "linear_attn", None)
            if lin is not None and getattr(lin, "_two_slot_mtp", False):
                found.append(lin)
        if not found and hasattr(model, "named_modules"):
            for _, mod in model.named_modules():
                if getattr(mod, "_two_slot_mtp", False) and hasattr(
                    mod, "replay_two_slot_state"
                ):
                    found.append(mod)
        layers_tuple = tuple(found)
        self._two_slot_gdn_layers_cache = (model, layers_tuple)
        return layers_tuple

    def _snapshot_two_slot_gdn_verify_inputs(
        self, max_rows: int
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        snap: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for layer in self._get_two_slot_gdn_layers():
            item = layer.snapshot_verify_inputs(max_rows=max_rows)
            if item is not None:
                snap[layer.prefix] = item
        return snap

    def _plan_two_slot_boundary_copies(
        self,
        scheduler_output: "SchedulerOutput",
        start_index: int,
        num_reqs: int,
        seq_lens: torch.Tensor,
    ) -> TwoSlotBoundaryPlan | None:
        if not (
            has_ping_pong_copy_groups(self)
            and self._mamba_ckpt_window > 1
            and self.mamba_slot_read_offsets is not None
            and bool(self._mamba_copy_plan)
            and (
                self.cache_config.enable_prefix_caching
                or self.vllm_config.kv_transfer_config is not None
            )
        ):
            return None
        block_size = self._mamba_column_stride()
        slack = 0
        if self.scheduler_config.async_scheduling:
            assert self.speculative_config is not None
            slack = self.speculative_config.num_speculative_tokens
        num_computed = self.input_batch.num_computed_tokens_cpu
        if (
            hasattr(self, "query_start_loc_np")
            and len(self.query_start_loc_np) >= num_reqs + 1
        ):
            comp_slice = num_computed[start_index : start_index + num_reqs]
            sched_slice = (
                self.query_start_loc_np[1 : num_reqs + 1]
                - self.query_start_loc_np[:num_reqs]
            )
            first_bnds = (
                np.maximum(comp_slice - slack, 0) // block_size + 1
            ) * block_size
            if bool(np.all(first_bnds > comp_slice + sched_slice)):
                return None
        spec_tokens = scheduler_output.scheduled_spec_decode_tokens or {}
        req_ids = self.input_batch.req_ids
        candidates: list[tuple[int, int, int]] = []
        for i in range(num_reqs):
            req_id = req_ids[start_index + i]
            assert req_id is not None
            computed = int(num_computed[start_index + i])
            scheduled = scheduler_output.num_scheduled_tokens[req_id]
            is_spec_req = bool(spec_tokens.get(req_id))
            if not is_spec_req and not (scheduled == 1 and computed > 0):
                continue
            boundary = (max(computed - slack, 0) // block_size + 1) * block_size
            while boundary <= computed + scheduled:
                candidates.append((i, boundary, scheduled))
                boundary += block_size
        if not candidates:
            return None
        width = self._mamba_ckpt_window + 1
        n = len(candidates)
        host = torch.zeros(two_slot_entries_shape(self, n), dtype=torch.int32)
        host[:n, :3] = torch.tensor(candidates, dtype=torch.int32)
        for g, (gid, _) in enumerate(self._mamba_copy_plan):
            bt = self.input_batch.block_table[gid].get_cpu_tensor()
            row_width = bt.shape[1]
            base = 3 + g * width
            for e, (i, boundary, _) in enumerate(candidates):
                first = boundary // block_size - 1
                last = min(first + width, row_width)
                host[e, base : base + last - first] = bt[start_index + i, first:last]
        entries = stage_h2d_upload(self, host)
        if entries.device.type == "tpu":
            return TwoSlotBoundaryPlan(
                entries=entries,
                rows=None,
                seq_lens=None,
                block_size=block_size,
                host_entries=host,
                chunk_seq_lens=seq_lens,
            )
        rows = entries[:, 0].long()
        return TwoSlotBoundaryPlan(
            entries=entries,
            rows=rows,
            seq_lens=seq_lens.index_select(0, rows).to(torch.int32),
            block_size=block_size,
            host_entries=host,
            chunk_seq_lens=seq_lens,
        )

    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
        start_index: int,
        num_decode_reqs: int,
        num_windowed_reqs: int | None = None,
    ):
        # If a previous chunk executed inside the same `execute_model` step,
        # snapshot its staged verify buffers before this chunk's forward can
        # overwrite them (multi-chunk batch safety; zero cost on 1-chunk steps).
        if self._two_slot_in_execute_model and self._two_slot_chunk_records:
            prev = self._two_slot_chunk_records[-1]
            if prev["staged_verify_inputs"] is None and prev["win_rows"] > 0:
                prev["staged_verify_inputs"] = (
                    self._snapshot_two_slot_gdn_verify_inputs(prev["win_rows"])
                )
        result = super()._prepare_inputs(
            scheduler_output, start_index, num_decode_reqs, num_windowed_reqs
        )
        if self._two_slot_in_execute_model and self.mamba_slot_read_offsets is not None:
            per_layer_attn_metadata = result[0]
            num_reqs = result[3]
            eff_windowed = (
                num_decode_reqs if num_windowed_reqs is None else num_windowed_reqs
            )
            ctx = self._attn_metadata_builder_ctx
            pp_indices = getattr(ctx, "unified_ping_pong_mamba_state_indices", None)
            chunk_num_win = max(0, min(eff_windowed - start_index, num_reqs))
            win_rows = (
                int(self.query_start_loc_np[chunk_num_win])
                if chunk_num_win > 0 and hasattr(self, "query_start_loc_np")
                else 0
            )
            boundary_plan = self._plan_two_slot_boundary_copies(
                scheduler_output, start_index, num_reqs, ctx.seq_lens
            )
            self._two_slot_chunk_records.append(
                {
                    "pp_indices": list(pp_indices) if pp_indices else None,
                    "attn_metadata": per_layer_attn_metadata,
                    "chunk_num_windowed": chunk_num_win,
                    "win_rows": win_rows,
                    "num_reqs": num_reqs,
                    "query_start_loc": ctx.query_start_loc,
                    "seq_lens": ctx.seq_lens,
                    "staged_verify_inputs": None,
                    "staged_ping_pong_stacks": getattr(
                        ctx, "staged_ping_pong_stacks", None
                    ),
                    "staged_std_mamba_stack": getattr(
                        ctx, "staged_std_mamba_stack", None
                    ),
                    "boundary_plan": boundary_plan,
                }
            )
        return result

    @torch.no_grad()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Any = None,
    ) -> ModelRunnerOutput | AsyncTPUModelRunnerOutput | None:
        self._two_slot_in_execute_model = True
        self._two_slot_chunk_records = []
        try:
            out = super().execute_model(scheduler_output, intermediate_tensors)
            state = self.execute_model_state
            if state is not None and self._two_slot_chunk_records:
                pp_chunks: list[PingPongMambaChunkState | None] = []
                boundary_plans: list[TwoSlotBoundaryPlan | None] = []
                for rec in self._two_slot_chunk_records:
                    if rec["pp_indices"]:
                        pp_chunks.append(
                            PingPongMambaChunkState(
                                state_indices=rec["pp_indices"],
                                attn_metadata=rec["attn_metadata"],
                                chunk_num_windowed=rec["chunk_num_windowed"],
                                num_reqs=rec["num_reqs"],
                                query_start_loc=rec["query_start_loc"],
                                seq_lens=rec["seq_lens"],
                                staged_verify_inputs=rec["staged_verify_inputs"],
                                staged_ping_pong_stacks=rec["staged_ping_pong_stacks"],
                                staged_std_mamba_stack=rec["staged_std_mamba_stack"],
                            )
                        )
                    else:
                        pp_chunks.append(None)
                    boundary_plans.append(rec["boundary_plan"])
                state.ping_pong_mamba_chunks = pp_chunks  # type: ignore[attr-defined]
                state.two_slot_boundary_plans = boundary_plans  # type: ignore[attr-defined]
            return out
        finally:
            self._two_slot_in_execute_model = False
            self._two_slot_chunk_records = []

    def _two_slot_dist3(self, w: int, dev: torch.device) -> torch.Tensor:
        cache = self.__dict__.setdefault("_two_slot_dist3_device", {})
        key = (int(w), dev)
        t = cache.get(key)
        if t is None:
            t = torch.tensor([int(w), int(w), int(w)], dtype=torch.int32, device=dev)
            cache[key] = t
        return t

    def _get_compiled_boundary_meta(
        self, layers: tuple[Any, ...], dev: torch.device
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if not self._mamba_copy_plan or self.mamba_slot_read_offsets is None:
            return None
        raw_keys = {tuple(id(raw) for raw in raws) for _, raws in self._mamba_copy_plan}
        if len(raw_keys) != 1:
            return None
        raws0 = self._mamba_copy_plan[0][1]
        cache_key = (
            tuple(gid for gid, _ in self._mamba_copy_plan),
            tuple(id(r) for r in raws0),
            tuple(id(layer) for layer in layers),
            int(self.mamba_slot_read_offsets.shape[0]),
            dev,
        )
        cached = self.__dict__.get("_compiled_boundary_meta_cache")
        if cached is not None and cached[0] == cache_key:
            return cached[1]
        gid_to_pos = {gid: pos for pos, (gid, _) in enumerate(self._mamba_copy_plan)}
        layer_to_gid: dict[str, int] = {}
        for group_list in getattr(self, "attn_groups", ()) or ():
            for group in group_list:
                for lname in group.layer_names:
                    layer_to_gid[lname] = group.kv_cache_group_id
        layer_positions: list[int] = []
        for layer in layers:
            gid = layer_to_gid.get(layer.prefix)
            if (
                gid is None
                or gid not in gid_to_pos
                or not is_ping_pong_copy_gid(self, gid)
            ):
                self._compiled_boundary_meta_cache = (cache_key, None)
                return None
            layer_positions.append(gid_to_pos[gid])
        is_pp_mask = torch.tensor(
            [is_ping_pong_copy_gid(self, gid) for gid, _ in self._mamba_copy_plan],
            dtype=torch.bool,
            device=dev,
        )
        layer_to_plan_idx = torch.tensor(layer_positions, dtype=torch.int64, device=dev)
        zero_offsets = self.__dict__.get("_zero_mamba_slot_read_offsets")
        if (
            zero_offsets is None
            or zero_offsets.shape != self.mamba_slot_read_offsets.shape
            or zero_offsets.device != dev
        ):
            zero_offsets = torch.zeros_like(self.mamba_slot_read_offsets)
            self._zero_mamba_slot_read_offsets = zero_offsets
        res = (raws0, is_pp_mask, layer_to_plan_idx, zero_offsets)
        self._compiled_boundary_meta_cache = (cache_key, res)
        return res

    def _two_slot_win_and_reqs(self, w: int, n: int, dev: torch.device) -> torch.Tensor:
        cache = self.__dict__.setdefault("_two_slot_win_reqs_device", {})
        key = (int(w), int(n), dev)
        t = cache.get(key)
        if t is None:
            t = torch.tensor([int(w), int(n)], dtype=torch.int32, device=dev)
            cache[key] = t
        return t

    @torch.no_grad()
    def _post_verify_two_slot_gdn_chunk(
        self,
        chunk: PingPongMambaChunkState | None,
        md: Any,
        next_tokens: torch.Tensor,
        boundary_plan: Any,
        mamba_indices: list[torch.Tensor] | None = None,
    ) -> bool:
        """Fused post-verify 2-slot GDN replay + boundary export + offset update.

        Returns True when non-ping-pong Mamba groups (`mamba_indices`, e.g. 12
        PLE groups) also had their `slot_read_offsets` updated inside
        `replay_and_flip`, allowing `sample_tokens` to skip the separate
        `_update_mamba_slot_read_offsets` dispatch.
        """
        if (
            chunk is not None
            and chunk.chunk_num_windowed > 0
            and chunk.staged_verify_inputs is None
            and chunk.staged_ping_pong_stacks is not None
            and self.mamba_slot_read_offsets is not None
            and chunk.query_start_loc.device.type == "tpu"
            and bool(md.draft_lengths_cpu[: chunk.num_reqs].any())
        ):
            layers = self._get_two_slot_gdn_layers()
            if layers:
                s_st, c_st = chunk.staged_ping_pong_stacks
                if (
                    s_st.shape[0] in (1, len(layers))
                    and c_st.shape[0] == s_st.shape[0]
                    and ping_pong_layers_match_group_order(
                        self, layers, is_ping_pong_spec=is_ping_pong_mamba_spec
                    )
                ):
                    multi_runner = get_or_build_multi_layer_two_slot_replay(
                        self, layers
                    )
                    if (
                        multi_runner is not None
                        and multi_runner.can_replay_and_flip_fast()
                    ):
                        dev = chunk.query_start_loc.device
                        w = chunk.chunk_num_windowed
                        ple_st = (
                            chunk.staged_std_mamba_stack
                            if (
                                chunk.staged_std_mamba_stack is not None
                                and mamba_indices is not None
                                and chunk.staged_std_mamba_stack.shape[0]
                                == len(mamba_indices)
                            )
                            else None
                        )
                        ple_done = ple_st is not None or not mamba_indices
                        dist = self._two_slot_dist3(w, dev)
                        wr_vec = self._two_slot_win_and_reqs(w, chunk.num_reqs, dev)
                        if boundary_plan is None:
                            multi_runner.replay_and_flip(
                                state_stack=s_st,
                                ckpt_stack=c_st,
                                query_start_loc=chunk.query_start_loc,
                                next_tokens=next_tokens,
                                win_and_reqs=wr_vec,
                                distribution=dist,
                                seq_lens=chunk.seq_lens,
                                slot_read_offsets=self.mamba_slot_read_offsets,
                                ple_state_stack=ple_st,
                            )
                            return ple_done
                        bmeta = self._get_compiled_boundary_meta(layers, dev)
                        if bmeta is not None:
                            raws0, is_pp_mask, layer_to_plan_idx, zero_offsets = bmeta
                            width = self._mamba_ckpt_window + 1
                            split = max(1, self._pool_block_split)
                            out_bucket = self._bucket_len(
                                len(self._mamba_copy_plan)
                                * int(boundary_plan.entries.shape[0])
                                * split
                            )
                            # Compute boundary copy & interior replay targets BEFORE
                            # `replay_and_flip` flips `mamba_slot_read_offsets`.
                            c_src, c_dst, s_int, ckpt_int, cl_int = (
                                compute_boundary_copy_and_interior_compiled(
                                    boundary_plan.entries,
                                    chunk.seq_lens,
                                    next_tokens,
                                    self.mamba_slot_read_offsets,
                                    self._num_reqs_tensor(w),
                                    is_pp_mask,
                                    layer_to_plan_idx,
                                    boundary_plan.block_size,
                                    width,
                                    split,
                                    out_bucket,
                                )
                            )
                            multi_runner.replay_and_flip(
                                state_stack=s_st,
                                ckpt_stack=c_st,
                                query_start_loc=chunk.query_start_loc,
                                next_tokens=next_tokens,
                                win_and_reqs=wr_vec,
                                distribution=dist,
                                seq_lens=chunk.seq_lens,
                                slot_read_offsets=self.mamba_slot_read_offsets,
                                ple_state_stack=ple_st,
                            )
                            self._copy_mamba_state_blocks(raws0, c_src, c_dst)
                            multi_runner(
                                raw_state_indices_stack=s_int,
                                raw_ckpt_indices_stack=ckpt_int,
                                commit_lens_stack=cl_int,
                                query_start_loc=chunk.query_start_loc,
                                distribution=dist,
                                seq_lens=chunk.seq_lens,
                                slot_read_offsets=zero_offsets,
                            )
                            return ple_done

        self._replay_two_slot_gdn_chunk(chunk, md, next_tokens)
        self._copy_two_slot_boundary_states(
            boundary_plan, next_tokens, ping_pong_chunk=chunk
        )
        self._update_ping_pong_mamba_slot_read_offsets(chunk)
        return False

    @torch.no_grad()
    def _replay_two_slot_gdn_chunk(
        self,
        chunk: PingPongMambaChunkState | None,
        md: Any,
        next_tokens: torch.Tensor,
    ) -> None:
        if (
            chunk is None
            or chunk.chunk_num_windowed <= 0
            or self.mamba_slot_read_offsets is None
        ):
            return
        if int(md.draft_lengths_cpu[: chunk.num_reqs].sum()) == 0:
            return
        layers = self._get_two_slot_gdn_layers()
        if not layers:
            return
        commit_lens = compute_replay_commit_lens(
            chunk.query_start_loc, next_tokens, chunk.chunk_num_windowed
        )
        if commit_lens.device.type == "cpu" and not bool((commit_lens > 0).any()):
            return
        dev = chunk.query_start_loc.device
        dist = self._two_slot_dist3(chunk.chunk_num_windowed, dev)
        multi_runner = get_or_build_multi_layer_two_slot_replay(self, layers)
        if multi_runner is not None:
            if chunk.staged_ping_pong_stacks is not None and dev.type == "tpu":
                s_st, c_st = chunk.staged_ping_pong_stacks
                if (
                    s_st.shape[0] in (1, len(layers))
                    and c_st.shape[0] == s_st.shape[0]
                    and ping_pong_layers_match_group_order(
                        self, layers, is_ping_pong_spec=is_ping_pong_mamba_spec
                    )
                ):
                    multi_runner(
                        raw_state_indices_stack=s_st,
                        raw_ckpt_indices_stack=c_st,
                        commit_lens_stack=commit_lens.unsqueeze(0),
                        query_start_loc=chunk.query_start_loc,
                        distribution=dist,
                        seq_lens=chunk.seq_lens,
                        slot_read_offsets=self.mamba_slot_read_offsets,
                        staged_inputs=chunk.staged_verify_inputs,
                    )
                    return
            per_layer_sc: list[tuple[torch.Tensor, torch.Tensor]] = []
            for layer in layers:
                layer_md = chunk.attn_metadata.get(layer.prefix)
                if layer_md is None:
                    per_layer_sc = []
                    break
                s_ids = mamba_state_of(layer_md)
                c_ids = mamba_ckpt_of(layer_md)
                if s_ids is None or c_ids is None:
                    per_layer_sc = []
                    break
                per_layer_sc.append((s_ids, c_ids))
            if len(per_layer_sc) == len(layers):
                s_stack = torch.stack(
                    [s.to(torch.int32) for s, _ in per_layer_sc], dim=0
                )
                c_stack = torch.stack(
                    [c.to(torch.int32) for _, c in per_layer_sc], dim=0
                )
                multi_runner(
                    raw_state_indices_stack=s_stack,
                    raw_ckpt_indices_stack=c_stack,
                    commit_lens_stack=commit_lens.unsqueeze(0),
                    query_start_loc=chunk.query_start_loc,
                    distribution=dist,
                    seq_lens=chunk.seq_lens,
                    slot_read_offsets=self.mamba_slot_read_offsets,
                    staged_inputs=chunk.staged_verify_inputs,
                )
                return
        for layer in layers:
            layer_md = chunk.attn_metadata.get(layer.prefix)
            if layer_md is None:
                continue
            state_ids = mamba_state_of(layer_md)
            ckpt_ids = mamba_ckpt_of(layer_md)
            if state_ids is None or ckpt_ids is None:
                continue
            staged = (
                chunk.staged_verify_inputs.get(layer.prefix)
                if chunk.staged_verify_inputs is not None
                else None
            )
            layer.replay_two_slot_state(
                state_indices=state_ids,
                query_start_loc=chunk.query_start_loc,
                distribution=dist,
                seq_lens=chunk.seq_lens,
                slot_read_offsets=self.mamba_slot_read_offsets,
                ckpt_indices=ckpt_ids,
                commit_lens=commit_lens,
                staged_inputs=staged,
            )

    @torch.no_grad()
    def _export_two_slot_interior_boundaries(
        self,
        s_int: torch.Tensor,
        ckpt_int: torch.Tensor,
        cl_int: torch.Tensor,
        chunk: PingPongMambaChunkState,
        zero_offsets: torch.Tensor,
    ) -> None:
        layers = self._get_two_slot_gdn_layers()
        if not layers:
            return
        dev = chunk.seq_lens.device
        dist = self._two_slot_dist3(chunk.chunk_num_windowed, dev)
        if cl_int.device.type == "cpu" and not bool((cl_int > 0).any()):
            return
        multi_runner = get_or_build_multi_layer_two_slot_replay(self, layers)
        if multi_runner is not None:
            multi_runner(
                raw_state_indices_stack=s_int,
                raw_ckpt_indices_stack=ckpt_int,
                commit_lens_stack=cl_int,
                query_start_loc=chunk.query_start_loc,
                distribution=dist,
                seq_lens=chunk.seq_lens,
                slot_read_offsets=zero_offsets,
                staged_inputs=chunk.staged_verify_inputs,
            )
            return
        for l_idx, layer in enumerate(layers):
            cl_l = cl_int[l_idx]
            if cl_l.device.type == "cpu" and not bool((cl_l > 0).any()):
                continue
            staged = (
                chunk.staged_verify_inputs.get(layer.prefix)
                if chunk.staged_verify_inputs is not None
                else None
            )
            layer.replay_two_slot_state(
                state_indices=s_int[l_idx],
                query_start_loc=chunk.query_start_loc,
                distribution=dist,
                seq_lens=chunk.seq_lens,
                slot_read_offsets=zero_offsets,
                ckpt_indices=ckpt_int[l_idx],
                commit_lens=cl_l,
                staged_inputs=staged,
            )

    @torch.no_grad()
    def _copy_two_slot_boundary_states(
        self,
        plan: Any,
        next_tokens: torch.Tensor | None,
        ping_pong_chunk: PingPongMambaChunkState | None = None,
    ) -> None:
        if plan is None or self.mamba_slot_read_offsets is None:
            return
        width = self._mamba_ckpt_window + 1
        split = max(1, self._pool_block_split)
        entries = plan.entries
        dev = entries.device
        chunk_seq_lens = (
            getattr(plan, "chunk_seq_lens", None)
            if isinstance(plan, TwoSlotBoundaryPlan)
            else plan.seq_lens
        )
        if chunk_seq_lens is None and ping_pong_chunk is not None:
            chunk_seq_lens = ping_pong_chunk.seq_lens
        layers = self._get_two_slot_gdn_layers()
        bmeta = self._get_compiled_boundary_meta(layers, dev) if layers else None
        if next_tokens is None and bmeta is not None and chunk_seq_lens is not None:
            raws0, is_pp_mask, _, _ = bmeta
            out_bucket = self._bucket_len(
                len(self._mamba_copy_plan) * int(entries.shape[0]) * split
            )
            has_win = ping_pong_chunk is not None
            win_val = ping_pong_chunk.chunk_num_windowed if has_win else 0
            fn = (
                compute_boundary_copy_no_verify_compiled
                if dev.type == "tpu"
                else compute_boundary_copy_no_verify_impl
            )
            c_src, c_dst = fn(
                entries,
                chunk_seq_lens,
                self.mamba_slot_read_offsets,
                self._num_reqs_tensor(win_val),
                is_pp_mask,
                has_win,
                plan.block_size,
                width,
                split,
                out_bucket,
            )
            self._copy_mamba_state_blocks(raws0, c_src, c_dst)
            return
        if (
            next_tokens is not None
            and bmeta is not None
            and chunk_seq_lens is not None
            and ping_pong_chunk is not None
        ):
            raws0, is_pp_mask, layer_to_plan_idx, zero_offsets = bmeta
            out_bucket = self._bucket_len(
                len(self._mamba_copy_plan) * int(entries.shape[0]) * split
            )
            fn_v = (
                compute_boundary_copy_and_interior_compiled
                if dev.type == "tpu"
                else compute_boundary_copy_and_interior_impl
            )
            c_src, c_dst, s_int, ckpt_int, cl_int = fn_v(
                entries,
                chunk_seq_lens,
                next_tokens,
                self.mamba_slot_read_offsets,
                self._num_reqs_tensor(ping_pong_chunk.chunk_num_windowed),
                is_pp_mask,
                layer_to_plan_idx,
                plan.block_size,
                width,
                split,
                out_bucket,
            )
            self._copy_mamba_state_blocks(raws0, c_src, c_dst)
            self._export_two_slot_interior_boundaries(
                s_int, ckpt_int, cl_int, ping_pong_chunk, zero_offsets
            )
            return
        # Fallback when groups use distinct raw pools or there are no GDN layers.
        rows = entries[:, 0].long()
        if chunk_seq_lens is None:
            return
        max_seqs = chunk_seq_lens.shape[0]
        valid_seq_row = (rows >= 0) & (rows < max_seqs)
        safe_rows = rows.clamp(0, max(0, max_seqs - 1))
        plan_seq_lens = torch.where(
            valid_seq_row,
            chunk_seq_lens.index_select(0, safe_rows).to(torch.int32),
            torch.zeros(rows.shape[0], dtype=torch.int32, device=dev),
        )
        if next_tokens is not None:
            valid_nt = valid_seq_row & (rows < next_tokens.shape[0])
            safe_nt = rows.clamp(0, max(0, next_tokens.shape[0] - 1))
            num_valid = torch.where(
                valid_nt,
                (next_tokens.index_select(0, safe_nt) != INVALID_TOKEN_ID)
                .sum(dim=1)
                .to(torch.int32),
                torch.zeros_like(plan_seq_lens),
            )
            fn_std = (
                spec_boundary_pairs_compiled
                if dev.type == "tpu"
                else spec_boundary_pairs_impl
            )
            std_src, std_dst = fn_std(
                entries,
                plan_seq_lens,
                num_valid,
                plan.block_size,
                width,
                valid_rows=valid_nt,
            )
        else:
            valid_nt = valid_seq_row
            num_valid = torch.ones_like(plan_seq_lens)
            std_src = std_dst = None
        offsets = self.mamba_slot_read_offsets
        n_off = int(offsets.shape[0])
        boundary = entries[:, 1]
        scheduled = entries[:, 2]
        start = plan_seq_lens - scheduled
        reached = (
            valid_nt
            & (boundary > 0)
            & (scheduled > 0)
            & (start < boundary)
            & (boundary <= start + num_valid)
        )
        if ping_pong_chunk is not None:
            reached = reached & (rows < ping_pong_chunk.chunk_num_windowed)
        exact_hit = reached & (boundary == start + num_valid)
        interior_hit = reached & (boundary < start + num_valid)
        blocks = entries[:, 3:].reshape(entries.shape[0], -1, width).transpose(0, 1)
        raw_col_rel = (plan_seq_lens - 1) // plan.block_size - (
            boundary // plan.block_size - 1
        )
        col_in_range = (raw_col_rel >= 0) & (raw_col_rel + 1 < width)
        col_rel = raw_col_rel.clamp(0, max(0, width - 2))
        idx0 = col_rel.view(1, -1, 1).expand(blocks.shape[0], -1, 1)
        idx1 = (
            (col_rel + 1)
            .clamp(max=width - 1)
            .view(1, -1, 1)
            .expand(blocks.shape[0], -1, 1)
        )
        b0_all = blocks.gather(2, idx0).squeeze(2)
        b1_all = blocks.gather(2, idx1).squeeze(2)
        per_raw: dict[
            tuple[int, ...],
            tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]],
        ] = {}
        interior_gid_pairs: dict[
            int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        for g, (gid, raws) in enumerate(self._mamba_copy_plan):
            if is_ping_pong_copy_gid(self, gid):
                b0 = b0_all[g]
                b1 = b1_all[g]
                dst_b = blocks[g, :, 0]
                valid_pair = (
                    col_in_range
                    & (b0 > 0)
                    & (b0 < n_off)
                    & (b1 > 0)
                    & (b1 < n_off)
                    & (dst_b > 0)
                    & (dst_b < n_off)
                )
                null_g = torch.zeros_like(dst_b)
                c_read = (
                    offsets[b0.clamp(0, n_off - 1).long()].clamp(0, 1)
                    if n_off > 0
                    else null_g
                )
                write_blk = torch.where(c_read == 0, b1, b0)
                read_blk = torch.where(c_read == 0, b0, b1)
                do_copy = exact_hit & valid_pair & (write_blk != dst_b)
                src_g, dst_g = expand_split_masked(
                    torch.where(do_copy, write_blk, null_g),
                    torch.where(do_copy, dst_b, null_g),
                    split,
                )
                entry = per_raw.setdefault(tuple(id(r) for r in raws), (raws, [], []))
                entry[1].append(src_g)
                entry[2].append(dst_g)
                if next_tokens is not None:
                    int_mask = interior_hit & valid_pair
                    if int_mask.device.type != "cpu" or bool(int_mask.any()):
                        interior_gid_pairs[gid] = (
                            torch.where(int_mask, read_blk, null_g),
                            torch.where(int_mask, dst_b, null_g),
                            torch.where(
                                int_mask,
                                (boundary - start).to(torch.int32),
                                null_g,
                            ),
                        )
                continue
            if std_src is None or std_dst is None:
                continue
            src_g, dst_g = expand_split_masked(std_src[g], std_dst[g], split)
            entry = per_raw.setdefault(tuple(id(r) for r in raws), (raws, [], []))
            entry[1].append(src_g)
            entry[2].append(dst_g)
        for raws, srcs, dsts in per_raw.values():
            self._copy_mamba_state_blocks(
                raws,
                self._pad_dev_to_bucket(torch.cat(srcs)),
                self._pad_dev_to_bucket(torch.cat(dsts)),
            )
        if interior_gid_pairs and ping_pong_chunk is not None and layers:
            zero_offsets = torch.zeros_like(offsets)
            dummy_row = torch.full_like(rows, max_seqs)
            layer_to_gid: dict[str, int] = {}
            for group_list in getattr(self, "attn_groups", ()) or ():
                for group in group_list:
                    for lname in group.layer_names:
                        layer_to_gid[lname] = group.kv_cache_group_id
            dist = self._two_slot_dist3(ping_pong_chunk.chunk_num_windowed, dev)
            for layer in layers:
                gid = layer_to_gid.get(layer.prefix)
                if gid is None or gid not in interior_gid_pairs:
                    continue
                read_c, dst_c, k_c = interior_gid_pairs[gid]
                active_c = valid_seq_row & (k_c > 0) & (read_c > 0) & (dst_c > 0)
                if active_c.device.type == "cpu" and not bool(active_c.any()):
                    continue
                tgt = torch.where(active_c, rows.clamp(0, max_seqs), dummy_row)
                r_pad = torch.zeros(max_seqs + 1, dtype=torch.int32, device=dev)
                d_pad = torch.zeros(max_seqs + 1, dtype=torch.int32, device=dev)
                c_pad = torch.zeros(max_seqs + 1, dtype=torch.int32, device=dev)
                r_pad.index_put_((tgt,), read_c.to(torch.int32))
                d_pad.index_put_((tgt,), dst_c.to(torch.int32))
                c_pad.index_put_((tgt,), k_c.to(torch.int32))
                r_seq = r_pad[:max_seqs]
                d_seq = d_pad[:max_seqs]
                c_seq = c_pad[:max_seqs]
                staged = (
                    ping_pong_chunk.staged_verify_inputs.get(layer.prefix)
                    if ping_pong_chunk.staged_verify_inputs is not None
                    else None
                )
                layer.replay_two_slot_state(
                    state_indices=r_seq,
                    query_start_loc=ping_pong_chunk.query_start_loc,
                    distribution=dist,
                    seq_lens=ping_pong_chunk.seq_lens,
                    slot_read_offsets=zero_offsets,
                    ckpt_indices=torch.stack([r_seq, d_seq], dim=1),
                    commit_lens=c_seq,
                    staged_inputs=staged,
                )

    @torch.no_grad()
    def _update_ping_pong_mamba_slot_read_offsets(
        self, chunk: PingPongMambaChunkState | None
    ) -> None:
        if (
            chunk is None
            or not chunk.state_indices
            or self.mamba_slot_read_offsets is None
        ):
            return
        groups = (
            [chunk.staged_ping_pong_stacks[0]]
            if chunk.staged_ping_pong_stacks is not None
            else list(chunk.state_indices)
        )
        fn = (
            update_ping_pong_read_offsets_compiled
            if self.mamba_slot_read_offsets.device.type == "tpu"
            else update_ping_pong_read_offsets_impl
        )
        fn(
            self.mamba_slot_read_offsets,
            groups,
            chunk.query_start_loc,
            self._num_reqs_tensor(chunk.chunk_num_windowed),
            self._num_reqs_tensor(chunk.num_reqs),
        )

    def load_model(self) -> None:
        super().load_model()
        # #2651: the padding mask skips its upload when its rows are unchanged
        # (same mask tensor, so whatever captured it still reads it).
        state = self._token_padding_state
        if state is not None:
            self._token_padding_state = SkipUnchangedTokenPaddingState(
                local_padding_mask=state.local_padding_mask
            )
            token_padding.set_padding_state(self._token_padding_state)

    def _precompile_spec_boundary_copies(self) -> None:
        super()._precompile_spec_boundary_copies()
        # #2650: `capture_model` compiles the one-chunk spec tail right after
        # the boundary copies.
        self._precompile_spec_tail()
        self._precompile_two_slot_runtime()

    def _precompile_two_slot_runtime(self) -> None:
        """Warm 2-slot ping-pong GDN replay, offset flips, and boundary copies.

        Synthetic warmup never rejects drafts or crosses a 512-token block
        boundary, so without precompilation the first served rejection, chunk
        crossing, or boundary export compiles XLA mid-serving while lockstep
        peers wait at their next collective. All-zero indices address the null
        block (0) and `commit_lens == 0` / `dist == [0, 0, 0]` perform no HBM
        state writes.
        """
        dev = getattr(self, "device", None)
        offsets = getattr(self, "mamba_slot_read_offsets", None)
        if dev is None or dev.type != "tpu" or offsets is None:
            return
        layers = self._get_two_slot_gdn_layers()
        has_pp_copy = has_ping_pong_copy_groups(self)
        if not layers and not has_pp_copy:
            return
        num_rows = sorted(
            r
            for r in {
                getattr(self, "num_reqs_max_model_len", None),
                getattr(self, "num_reqs_most_model_len", None),
            }
            if r is not None and r > 0
        )
        if not num_rows:
            num_rows = [int(self.max_num_reqs)]
        pp_group_ids: set[int] = set(getattr(self, "_mamba_copy_ping_pong_gids", ()))
        if not pp_group_ids:
            vllm_cfg = getattr(self, "vllm_config", None)
            for group_list in getattr(self, "attn_groups", ()) or ():
                for group in group_list:
                    if is_ping_pong_mamba_spec(group.kv_cache_spec, vllm_cfg):
                        pp_group_ids.add(group.kv_cache_group_id)
        num_pp_groups = len(pp_group_ids) if pp_group_ids else len(layers)
        with self._precompile_timed("two-slot GDN runtime"), torch.no_grad():
            z0 = self._num_reqs_tensor(0)
            # 1. Warm ping-pong read-offset flips for both the staged `[G, n]`
            # stack and the non-staged list-of-G input shapes.
            if num_pp_groups > 0:
                for rows in num_rows:
                    qsl = torch.zeros(rows + 1, dtype=torch.int32, device=dev)
                    stacked = torch.zeros(
                        (num_pp_groups, rows), dtype=torch.int32, device=dev
                    )
                    update_ping_pong_read_offsets_compiled(
                        offsets, [stacked], qsl, z0, z0
                    )
                    per_group = [
                        torch.zeros(rows, dtype=torch.int32, device=dev)
                        for _ in range(num_pp_groups)
                    ]
                    update_ping_pong_read_offsets_compiled(
                        offsets, per_group, qsl, z0, z0
                    )
            # 2. Warm per-layer replay (`gdn_pooled_replay_op`) on fixed-shape
            # `(win_cap, ...)` staging buffers for every `num_reqs_*_model_len`.
            if layers:
                seen_replay_ops: set[tuple[int, int, int]] = set()
                dist0 = self._two_slot_dist3(0, dev)
                for layer in layers:
                    kv = layer.kv_cache
                    if kv is None or len(kv) != 1 or kv[0].numel() == 0:
                        continue
                    op_key = (
                        id(layer.gdn_pooled_replay_op),
                        int(layer._verify_win_cap),
                        id(kv[0]),
                    )
                    if op_key in seen_replay_ops:
                        continue
                    seen_replay_ops.add(op_key)
                    for rows in num_rows:
                        z_rows = torch.zeros(rows, dtype=torch.int32, device=dev)
                        qsl = torch.zeros(rows + 1, dtype=torch.int32, device=dev)
                        ckpt2 = torch.zeros((rows, 2), dtype=torch.int32, device=dev)
                        layer.replay_two_slot_state(
                            state_indices=z_rows,
                            query_start_loc=qsl,
                            distribution=dist0,
                            seq_lens=z_rows,
                            slot_read_offsets=offsets,
                            ckpt_indices=ckpt2,
                            commit_lens=z_rows,
                        )
            # 3. Warm two-slot boundary copy & interior export programs plus
            # resized `_copy_mamba_state_blocks` buckets.
            if (
                has_pp_copy
                and self._mamba_ckpt_window > 1
                and bool(self._mamba_copy_plan)
                and (
                    self.cache_config.enable_prefix_caching
                    or self.vllm_config.kv_transfer_config is not None
                )
            ):
                width = self._mamba_ckpt_window + 1
                split = max(1, self._pool_block_split)
                block_size = self._mamba_column_stride()
                entries_shapes = sorted(
                    {
                        two_slot_entries_shape(self, n)
                        for n in range(1, self.max_num_reqs + 1)
                    }
                )
                bmeta = (
                    self._get_compiled_boundary_meta(layers, dev) if layers else None
                )
                k_cols = (
                    self.speculative_config.num_speculative_tokens + 1
                    if self.speculative_config is not None
                    else self._mamba_ckpt_window
                )
                warmed_copy_buckets: set[tuple[tuple[int, ...], int]] = set()
                for rows in num_rows:
                    seq_lens = torch.zeros(rows, dtype=torch.int32, device=dev)
                    nt_paddings = sorted(
                        {
                            _get_padded_num_reqs_with_upper_limit(n, self.max_num_reqs)
                            for n in range(1, min(rows, self.max_num_reqs) + 1)
                        }
                    )
                    for shape in entries_shapes:
                        entries = torch.zeros(shape, dtype=torch.int32, device=dev)
                        out_bucket = self._bucket_len(
                            len(self._mamba_copy_plan) * shape[0] * split
                        )
                        if bmeta is not None:
                            raws0, is_pp_mask, layer_to_plan_idx, _ = bmeta
                            raws_key = (tuple(id(r) for r in raws0), out_bucket)
                            for has_win in (False, True):
                                c_src, c_dst = compute_boundary_copy_no_verify_compiled(
                                    entries,
                                    seq_lens,
                                    offsets,
                                    z0,
                                    is_pp_mask,
                                    has_win,
                                    block_size,
                                    width,
                                    split,
                                    out_bucket,
                                )
                                if raws_key not in warmed_copy_buckets:
                                    warmed_copy_buckets.add(raws_key)
                                    self._copy_mamba_state_blocks(raws0, c_src, c_dst)
                            for nt_rows in nt_paddings:
                                next_tokens = torch.full(
                                    (nt_rows, k_cols),
                                    INVALID_TOKEN_ID,
                                    dtype=torch.int32,
                                    device=dev,
                                )
                                c_src_v, c_dst_v, _, _, _ = (
                                    compute_boundary_copy_and_interior_compiled(
                                        entries,
                                        seq_lens,
                                        next_tokens,
                                        offsets,
                                        z0,
                                        is_pp_mask,
                                        layer_to_plan_idx,
                                        block_size,
                                        width,
                                        split,
                                        out_bucket,
                                    )
                                )
                                if raws_key not in warmed_copy_buckets:
                                    warmed_copy_buckets.add(raws_key)
                                    self._copy_mamba_state_blocks(
                                        raws0, c_src_v, c_dst_v
                                    )
                        else:
                            z_cand = torch.zeros(
                                shape[0], dtype=torch.int32, device=dev
                            )
                            z_valid = torch.zeros(
                                shape[0], dtype=torch.bool, device=dev
                            )
                            spec_boundary_pairs_compiled(
                                entries,
                                z_cand,
                                z_cand,
                                block_size,
                                width,
                                valid_rows=z_valid,
                            )
                            per_raw_counts: dict[
                                tuple[int, ...], tuple[list[torch.Tensor], int]
                            ] = {}
                            for _, raws in self._mamba_copy_plan:
                                rk = tuple(id(r) for r in raws)
                                prev_cnt = per_raw_counts.get(rk, (raws, 0))[1]
                                per_raw_counts[rk] = (
                                    raws,
                                    prev_cnt + shape[0] * split,
                                )
                            for rk, (raws, total_pairs) in per_raw_counts.items():
                                b_len = self._bucket_len(total_pairs)
                                if (rk, b_len) not in warmed_copy_buckets:
                                    warmed_copy_buckets.add((rk, b_len))
                                    z_src = torch.zeros(
                                        b_len, dtype=torch.int32, device=dev
                                    )
                                    z_dst = torch.zeros(
                                        b_len, dtype=torch.int32, device=dev
                                    )
                                    self._copy_mamba_state_blocks(raws, z_src, z_dst)
            synchronize_tensors([offsets])
            for _, raws in getattr(self, "_mamba_copy_plan", ()):
                synchronize_tensors(raws)

    def _init_speculative_decoding(self) -> None:
        self.drafter = None
        self.rejection_sampler = None
        # Full speculative verify windows (1 + K tokens) lead the verify
        # segment and their count goes into request_distribution[1]
        # (_reorder_batch_for_rpa, _build_attention_metadata). The length
        # comes from the one helper the DSv4 attention and indexer ops use
        # too, since only ops that implement the static query-length pass may
        # see rows in that segment; see static_verify_q_len for why.
        self._static_verify_q_len: int | None = static_verify_q_len(self.vllm_config)
        self._num_static_verify_reqs = 0
        self._num_reqs_device: dict[int, torch.Tensor] = {}
        # Drafting and verification happen where sampling happens, on the
        # last pipeline stage; the other stages keep the config for the KV
        # cache spec and the scheduler, and nothing else.
        if self.speculative_config and self._pp_is_last:
            self.rejection_sampler = RejectionSampler(
                self.speculative_config, self.device
            )
            if self.speculative_config.method == "ngram":
                self.drafter = NgramProposer(self.vllm_config)
            elif self.speculative_config.method == "dflash":
                self.drafter = DFlashProposer(self, self.vllm_config)
            elif self.speculative_config.method == "dspark":
                self.drafter = DSparkProposer(self, self.vllm_config)
            elif self.speculative_config.use_eagle():
                self.drafter = Eagle3Proposer(self, self.vllm_config)
            else:
                raise NotImplementedError(
                    "Unsupported speculative decoding method: "
                    f"{self.speculative_config.method}"
                )

    @torch.no_grad()
    def _update_mamba_slot_read_offsets(
        self,
        indices_per_group: list[torch.Tensor] | None,
        next_tokens: torch.Tensor | None,
        num_reqs: int,
    ) -> torch.Tensor | None:
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
        its state block (the checkpoint of its last accepted token). Returns
        the offsets, one per batch position.
        """
        if self.mamba_slot_read_offsets is None or not indices_per_group:
            return None
        if _use_fused_read_offset_update(self.mamba_slot_read_offsets, next_tokens):
            assert next_tokens is not None
            return _update_read_offsets_compiled(
                self.mamba_slot_read_offsets,
                list(indices_per_group),
                next_tokens,
                self._num_reqs_tensor(num_reqs),
            )
        offsets = torch.zeros(
            indices_per_group[0].shape[0],
            dtype=torch.int32,
            device=indices_per_group[0].device,
        )
        if next_tokens is not None:
            # num_valid = accepted drafts + 1 (bonus); the checkpoint of the
            # last accepted token is at offset num_valid - 1.
            num_valid = (
                (next_tokens[:num_reqs] != INVALID_TOKEN_ID).sum(dim=1).to(torch.int32)
            )
            offsets[:num_reqs] = (num_valid - 1).clamp(min=0)
        if len(indices_per_group) == 1:
            self.mamba_slot_read_offsets.index_put_(
                (indices_per_group[0].long(),), offsets
            )
        else:
            all_indices = torch.cat(indices_per_group, dim=0).long()
            all_offsets = offsets.repeat(len(indices_per_group))
            self.mamba_slot_read_offsets.index_put_((all_indices,), all_offsets)
        return offsets

    def _num_reqs_tensor(self, num_reqs: int) -> torch.Tensor:
        """A cached 0-d int64 device tensor holding `num_reqs` (read-only), so
        the per-step read-offset update does not upload a fresh scalar."""
        # Lazily created: runners built without __init__ (tests, partial
        # construction) have no cache yet.
        cache = self.__dict__.setdefault("_num_reqs_device", {})
        t = cache.get(num_reqs)
        if t is None:
            t = torch.tensor(num_reqs, dtype=torch.int64, device=self.device)
            cache[num_reqs] = t
        return t

    def _upload_keyed(
        self, slot: str, host: torch.Tensor, values: bytes
    ) -> torch.Tensor:
        """`_upload_unless_unchanged` with the caller's value key (for dtypes
        numpy cannot view)."""
        key = (host.dtype, tuple(host.shape), values)
        prev = self._h2d_reuse.get(slot)
        if prev is not None and prev[0] == key:
            return prev[1]
        dev = host.to(self.device, non_blocking=True)
        self._h2d_reuse[slot] = (key, dev, host)
        return dev

    def _build_padded_sampling_params(
        self,
        cur_start_idx: int,
        cur_end_idx: int,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._padded_sampling_params(
            cur_start_idx, cur_end_idx, logits.shape[0], logits.dtype, logits.device
        )

    def _padded_sampling_params(
        self,
        cur_start_idx: int,
        cur_end_idx: int,
        padded_num_reqs: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Stage [padded_num_reqs] on CPU with neutral padding, then use one
        # fixed-shape H2D copy per tensor to keep decode shape-stable.
        num_active_reqs = cur_end_idx - cur_start_idx
        temps_cpu = torch.ones(padded_num_reqs, dtype=dtype)
        top_k_cpu = torch.zeros(padded_num_reqs, dtype=torch.int32)
        top_p_cpu = torch.ones(padded_num_reqs, dtype=torch.float32)

        temps_cpu[:num_active_reqs].copy_(
            self.input_batch.temperature_cpu_tensor[cur_start_idx:cur_end_idx]
        )
        active_top_k = self.input_batch.top_k_cpu_tensor[cur_start_idx:cur_end_idx]
        top_k_cpu[:num_active_reqs].copy_(
            torch.where(
                active_top_k >= self.vocab_size,
                torch.zeros_like(active_top_k),
                active_top_k,
            )
        )
        top_p_cpu[:num_active_reqs].copy_(
            self.input_batch.top_p_cpu_tensor[cur_start_idx:cur_end_idx]
        )

        if device == self.device:
            # Requests' sampling parameters rarely change between steps: reuse
            # the previous uploads while the values match (keyed in float32,
            # as numpy has no bfloat16).
            return (
                self._upload_keyed(
                    "sampling_temps",
                    temps_cpu.unsqueeze(1),
                    temps_cpu.float().numpy().tobytes(),
                ),
                self._upload_unless_unchanged("sampling_top_k", top_k_cpu.unsqueeze(1)),
                self._upload_unless_unchanged("sampling_top_p", top_p_cpu.unsqueeze(1)),
            )
        return (
            temps_cpu.unsqueeze(1).to(device, non_blocking=True),
            top_k_cpu.unsqueeze(1).to(device, non_blocking=True),
            top_p_cpu.unsqueeze(1).to(device, non_blocking=True),
        )

    def _collect_mamba_state_seed_copies(
        self, scheduler_output, start_index: int, num_reqs: int
    ) -> None:
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
        # The last written column can be ahead of the committed token count
        # after draft rejection. Retain it while the request still owns KV.
        live_req_ids = self.requests.keys()
        for req_id in list(self._mamba_state_pos.keys()):
            if req_id not in live_req_ids:
                del self._mamba_state_pos[req_id]

        col_stride = self._mamba_column_stride()
        num_computed = self.input_batch.num_computed_tokens_cpu
        req_ids = self.input_batch.req_ids
        crossings: list[tuple[int, int, int]] = []
        previous = (
            self._pre_async_results
            if self.scheduler_config.async_scheduling and self.speculative_config
            else None
        )
        sampled_cpu: torch.Tensor | None = None
        for i in range(num_reqs):
            row = start_index + i
            req_id = req_ids[row]
            assert req_id is not None
            computed = int(num_computed[row])
            scheduled = scheduler_output.num_scheduled_tokens[req_id]
            end_position = computed + scheduled - 1
            if previous is not None and previous.num_draft_per_req:
                previous_row = previous.req_id_to_index_copy.get(req_id)
                drafts = (
                    previous.num_draft_per_req.get(previous_row, 0)
                    if previous_row is not None
                    else 0
                )
                # Host positions assume every previous draft was accepted.
                # Only near a block edge can the correction change the seed
                # destination; reuse the pending sampling copy in that case.
                if end_position // col_stride != (end_position - drafts) // col_stride:
                    if sampled_cpu is None:
                        sampled_cpu = previous.wait_for_copy()
                        assert sampled_cpu is not None
                    assert previous_row is not None
                    accepted = int(
                        (sampled_cpu[previous_row] != INVALID_TOKEN_ID).sum().item()
                    )
                    rejected = max(0, drafts + 1 - accepted)
                    computed -= rejected
                    end_position -= rejected
            curr = end_position // col_stride
            prev = self._mamba_state_pos.get(req_id, (computed - 1) // col_stride)
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
        # the device-side gather in `_spec_seed_copies`.
        window = self._mamba_ckpt_window
        spec_seed = self.mamba_slot_read_offsets is not None and window > 1

        # Keyed by the set of raw buffers a group lives on; groups sharing
        # the same buffers are seeded by one program.
        per_raw: dict[
            tuple[int, ...], tuple[list[torch.Tensor], list[int], list[int]]
        ] = {}
        # Speculative crossings of every group on the same buffers, as
        # `_spec_seed_rows_impl` rows: one upload and one program per buffer
        # set.
        per_raw_spec: dict[
            tuple[int, ...], tuple[list[torch.Tensor], list[list[int]]]
        ] = {}
        # Manager-level (src, dst) block pairs across mamba groups: without
        # spec decoding the per-block read offsets just follow the state to
        # the request's new state block.
        offset_pairs: list[tuple[int, int]] = []
        spec_dsts: list[int] = []
        group_windows = getattr(self, "_mamba_copy_group_windows", None) or {}
        for gid, raws in self._mamba_copy_plan:
            gw_raw = group_windows.get(gid, 0)
            gw = min(window, gw_raw) if gw_raw > 1 else window
            bt = self.input_batch.block_table[gid].get_cpu_tensor()
            row_width = bt.shape[1]
            pairs: list[tuple[int, int]] = []
            spec_rows: list[list[int]] = []
            for row, prev, curr in crossings:
                src = int(bt[row, prev])
                dst = int(bt[row, curr])
                if src == dst or src == 0 or dst == 0:
                    continue
                pairs.append((src, dst))
                if spec_seed:
                    new_owner = req_ids[row] not in self._mamba_offset_seeded
                    # The pre-crossing group's checkpoint blocks. Columns past
                    # the group's own checkpoint window (`gw`, e.g. 2 for 2-slot
                    # GDN vs 4 for PLE) fall back to `src`.
                    row_ckpts = []
                    for t in range(window):
                        if t < gw:
                            col = prev + t
                            ckpt = int(bt[row, col]) if col < row_width else 0
                            row_ckpts.append(ckpt if ckpt != 0 else src)
                        else:
                            row_ckpts.append(src)
                    spec_rows.append([src, int(new_owner), 1, dst, *row_ckpts])
            if not pairs:
                continue
            if spec_seed:
                spec_dsts.extend(d for _, d in pairs)
                key = tuple(id(raw) for raw in raws)
                per_raw_spec.setdefault(key, (raws, []))[1].extend(spec_rows)
                continue
            offset_pairs.extend(pairs)
            if self._pool_block_split > 1:
                # The pool is born at kernel granularity: a manager state
                # block is `split` consecutive pool blocks.
                pairs = [
                    (s * self._pool_block_split + j, d * self._pool_block_split + j)
                    for s, d in pairs
                    for j in range(self._pool_block_split)
                ]
            key = tuple(id(raw) for raw in raws)
            entry = per_raw.setdefault(key, (raws, [], []))
            entry[1].extend(p[0] for p in pairs)
            entry[2].extend(p[1] for p in pairs)

        if spec_seed and spec_dsts:
            # The seeded checkpoint is checkpoint 0 of the new group.
            dst_t = stage_h2d_upload(
                self, torch.tensor(self._pad_to_bucket(spec_dsts), dtype=torch.long)
            )
            assert self.mamba_slot_read_offsets is not None
            _rollback_offsets_seed(self.mamba_slot_read_offsets, dst_t)
        elif self.mamba_slot_read_offsets is not None and offset_pairs:
            # Migrate the read offsets with the state: gather at the old
            # blocks, scatter at the new ones (both device-side, before the
            # forward that reads the buffer). Padding with (0, 0) null-block
            # self-copies keeps the shapes on the seed-copy bucket ladder.
            offset_pairs = self._pad_to_bucket(offset_pairs, pad=(0, 0))
            src_t = stage_h2d_upload(
                self, torch.tensor([s for s, _ in offset_pairs], dtype=torch.long)
            )
            dst_t = stage_h2d_upload(
                self, torch.tensor([d for _, d in offset_pairs], dtype=torch.long)
            )
            _rollback_offsets_migrate(self.mamba_slot_read_offsets, src_t, dst_t)

        for raws, rows in per_raw_spec.values():
            src_t, dst_t = self._spec_seed_copies(rows, window)
            self._pending_mamba_state_copies.append((raws, src_t, dst_t))

        for raws, srcs, dsts in per_raw.values():
            src_t = stage_h2d_upload(
                self, torch.tensor(self._pad_to_bucket(srcs), dtype=torch.int32)
            )
            dst_t = stage_h2d_upload(
                self, torch.tensor(self._pad_to_bucket(dsts), dtype=torch.int32)
            )
            self._pending_mamba_state_copies.append((raws, src_t, dst_t))

    def _spec_seed_copies(
        self, rows: list[list[int]], window: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Seed-copy (src, dst) for speculative crossings, one program.

        `mamba_slot_read_offsets` is device-resident and indexed by block
        id, so which of the window's blocks holds the last accepted token's
        state is only known on device — the choice is a gather there rather
        than a CPU lookup (`_spec_seed_rows_impl`). Rows pad up to
        `ceil(out_len / split)` before the upload so `(in_len, out_len)` sits
        1-to-1 on the seed-copy bucket ladder.
        """
        offsets = self.mamba_slot_read_offsets
        assert offsets is not None
        split = self._pool_block_split
        out_len = self._bucket_len(len(rows) * split)
        in_len = (out_len + split - 1) // split
        pad = [0, 1, 0, 0] + [0] * window
        rows = rows + [pad] * (in_len - len(rows))
        packed = stage_h2d_upload(self, torch.tensor(rows, dtype=torch.long))
        fn = (
            _spec_seed_rows_compiled
            if packed.device.type == "tpu"
            else _spec_seed_rows_impl
        )
        return fn(offsets, packed, split, out_len)

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
        with self._precompile_timed("mamba rollback helpers"), torch.no_grad():
            num_groups = max(
                1,
                len(self._mamba_copy_plan),
                len(self.kv_cache_config.kv_cache_groups)
                if self.kv_cache_config is not None
                else 0,
            )
            limit = self._bucket_len(num_groups * self.max_num_reqs)
            n = 8
            while n <= limit:
                src = torch.zeros(n, dtype=torch.long).to(offsets.device)
                dst = torch.zeros(n, dtype=torch.long).to(offsets.device)
                _rollback_offsets_seed_compiled(offsets, dst)
                _rollback_offsets_migrate_compiled(offsets, src, dst)
                n *= 4
            window = getattr(self, "_mamba_ckpt_window", 1)
            if window > 1:
                split = self._pool_block_split
                spec_limit = self._bucket_len(num_groups * self.max_num_reqs * split)
                out_len = 8
                while out_len <= spec_limit:
                    in_len = (out_len + split - 1) // split
                    packed = torch.zeros((in_len, 4 + window), dtype=torch.long).to(
                        offsets.device
                    )
                    _spec_seed_rows_compiled(offsets, packed, split, out_len)
                    out_len *= 4

    def _assemble_async_spec_substitution(self, drafts, next_tokens_per_chunk, state):
        """Async-spec substitution assembler.

        From the already-proposed on-device ``drafts`` (the unified eagle3
        propose call in ``sample_tokens``), builds the per-request
        ``[bonus, draft_1..K]`` substitution source and the stride-``1+K``
        ``next_token_indices``, and computes the per-request rejected count.
        Returns ``(next_tokens_tpu_chunks, next_token_indices,
        spec_decode_num_rejected_tokens, num_draft_per_req)``.
        """
        from vllm_torchtpu.runner.tpu_runner_async_output import (
            assemble_spec_next_tokens,
            compute_num_rejected,
        )

        assert drafts is not None, (
            "async spec substitution requires the eagle3 drafts proposed "
            "earlier in sample_tokens"
        )
        drafts_per_chunk = self._drafts_per_chunk(drafts, state.num_reqs_list)
        # Bucketed drafters keep the sampler's padded rows (an exact-count slice
        # would compile a program per request count); others keep exact rows.
        bucketed = getattr(getattr(self, "drafter", None), "bucketed_drafts", False)
        md0 = (
            state.spec_decode_metadata_list[0]
            if len(next_tokens_per_chunk) == 1
            else None
        )
        if (
            md0 is not None
            and hasattr(self, "max_num_reqs")
            and self.speculative_config is not None
        ):
            # Steady decode (one chunk, drafts verified): build the source and
            # rejected count in one program, with the source already padded to
            # the bound `sample_tokens` would pad it to.
            nt_chunk = next_tokens_per_chunk[0]
            num_reqs = state.num_reqs_list[0]
            spec_tail_fn = (
                _spec_tail_compiled if nt_chunk.device.type == "tpu" else _spec_tail
            )
            src, num_rejected = spec_tail_fn(
                nt_chunk,
                drafts_per_chunk[0],
                md0.draft_lengths,
                nt_chunk.shape[0] if bucketed else num_reqs,
                None if bucketed else num_reqs,
                self.max_num_reqs
                * (1 + self.speculative_config.num_speculative_tokens),
            )
            return (
                [src],
                {r: r for r in range(num_reqs)},
                num_rejected,
                {r: int(md0.draft_lengths_cpu[r]) for r in range(num_reqs)},
            )
        next_tokens_tpu_chunks: list[torch.Tensor] = []
        next_token_indices: dict[int, int] = {}
        num_rejected_chunks: list[torch.Tensor] = []
        num_draft_per_req: dict[int, int] = {}
        req_offset = 0
        padded_req_offset = 0
        # Uniform `1+K` source span per request: req position `p` maps to source
        # `[p*(1+K) .. p*(1+K)+K]`, and `req_id_to_index_copy` stores `p` (the
        # index builder multiplies by the stride). All chunks carry their next
        # proposals, even if they did not verify drafts in the current step.
        for nt_chunk, num_reqs, md, drafts_chunk in zip(
            next_tokens_per_chunk,
            state.num_reqs_list,
            state.spec_decode_metadata_list,
            drafts_per_chunk,
        ):
            chunk_padded_num_reqs = nt_chunk.shape[0] if bucketed else num_reqs
            drafts_chunk = fit_rows(drafts_chunk, chunk_padded_num_reqs)
            src = assemble_spec_next_tokens(
                nt_chunk, drafts_chunk, None if bucketed else num_reqs
            )
            scheduled = self._scheduled_draft_counts(
                req_offset, num_reqs, chunk_padded_num_reqs, md
            )
            if md is not None:
                if scheduled is None:
                    scheduled = fit_rows(md.draft_lengths, chunk_padded_num_reqs)
                num_rejected_chunks.append(compute_num_rejected(nt_chunk, scheduled))
                for r in range(num_reqs):
                    num_draft_per_req[req_offset + r] = int(md.draft_lengths_cpu[r])
            else:
                # No drafts were verified this step, so there is no
                # speculative output-placeholder rollback; drafts trimmed by
                # adaptive verification still roll back as rejected.
                num_rejected_chunks.append(
                    torch.zeros(
                        chunk_padded_num_reqs, dtype=torch.int32, device=nt_chunk.device
                    )
                    if scheduled is None
                    else scheduled
                )
            next_tokens_tpu_chunks.append(src)
            for r in range(num_reqs):
                next_token_indices[req_offset + r] = padded_req_offset + r
            req_offset += num_reqs
            padded_req_offset += chunk_padded_num_reqs
        spec_num_rejected = (
            torch.cat(num_rejected_chunks) if num_rejected_chunks else None
        )
        return (
            next_tokens_tpu_chunks,
            next_token_indices,
            spec_num_rejected,
            num_draft_per_req,
        )

    def _precompile_spec_tail(self) -> None:
        """Compile the one-chunk spec tail for every live request count
        (exact-count drafts or the sampler's padded rows), so serving never
        compiles it."""
        if not (
            self.device.type == "tpu"
            and self.speculative_config
            and self.scheduler_config.async_scheduling
        ):
            return
        k = self.speculative_config.num_speculative_tokens
        bound = self.max_num_reqs * (1 + k)
        bucketed = getattr(getattr(self, "drafter", None), "bucketed_drafts", False)
        with self._precompile_timed("async-spec substitution tail"):
            for num_reqs in range(1, self.max_num_reqs + 1):
                rows = _get_padded_num_reqs_with_upper_limit(
                    num_reqs, self.max_num_reqs
                )
                next_tokens = torch.full((rows, k + 1), -1, dtype=torch.int32).to(
                    self.device
                )
                drafts = torch.zeros(
                    (rows if bucketed else num_reqs, k), dtype=torch.int32
                ).to(self.device)
                draft_lengths = torch.zeros(rows, dtype=torch.int32).to(self.device)
                synchronize_tensors(
                    _spec_tail_compiled(
                        next_tokens,
                        drafts,
                        draft_lengths,
                        rows if bucketed else num_reqs,
                        None if bucketed else num_reqs,
                        bound,
                    )
                )

    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> ModelRunnerOutput | AsyncTPUModelRunnerOutput:
        if self.execute_model_state is None:
            if not self._pp_is_last:
                return self._sample_tokens_pp_intermediate_stage()
            # Nothing to do, output isn't used.
            return None  # type: ignore[return-value]

        state = self.execute_model_state
        scheduler_output = state.scheduler_output
        self.execute_model_state = None
        min_toks = self._min_tokens_batch_state()

        # Used to keep rand draws consistent across TP ranks.
        sampling_generator = self._get_sampling_generator()

        # Hand the per-chunk draft inputs to the drafter.
        if self._is_async_drafter:
            self._async_drafter.draft_chunks = state.draft_chunks

        # Prepare inputs, the requests might be split into multiple
        # executions, combine the result of each execution.

        max_num_logprobs = self.input_batch.max_num_logprobs
        if max_num_logprobs == -1:
            raise NotImplementedError(
                "TPU runner does not support full logprobs (`logprobs=-1`) "
                "with the merged vLLM v1 sampler path yet."
            )
        needs_logprobs = max_num_logprobs is not None
        # step_eligible keeps grammar and logprobs steps on the full logits.
        assert not state.top_tokens or (
            grammar_output is None and not needs_logprobs and not min_toks
        ), "top_tokens chosen with grammar, logprobs or a request below min_tokens"

        # Per-chunk bucketed (padded) tensors plus their real (unpadded)
        # request counts; kept at bucketed shapes so the async D2H copy hits
        # a precompiled program. Host-side trim happens in AsyncTPUCopyState.
        combined_selected_tokens: list[torch.Tensor] = []
        combined_selected_tokens_real_lens: list[int] = []
        next_tokens_tpu_chunks: list[torch.Tensor] = []
        next_token_indices: dict[int, int] = {}
        next_tokens_tpu_offset = 0
        # Per-chunk logprobs, cut to each request's generated tokens once they
        # reach the host (AsyncTPUModelRunnerOutput.get_output).
        logprobs_chunks: list[LogprobsChunk] = []

        # True if the step has drafts to verify (a request carried drafts in
        # from the previous step). Not the same as self.speculative_config, the
        # static "spec configured" flag: on prefill/first-decode steps spec is
        # configured but no drafts exist yet, so is_spec_step is False there.
        is_spec_step = any(md is not None for md in state.spec_decode_metadata_list)
        if state.top_tokens:
            greedy_top_tokens.log_step_path(is_spec_step, self.input_batch.req_ids)
        # Per-chunk mamba slot ids (hybrid + spec decode only; None entries
        # otherwise) for the read-offset scatter after sampling.
        mamba_idx_list = state.mamba_state_indices_list or [None] * len(
            state.logits_list
        )
        boundary_plans = state.mamba_boundary_plans or [None] * len(state.logits_list)
        ping_pong_chunks = getattr(state, "ping_pong_mamba_chunks", None) or [
            None
        ] * len(state.logits_list)
        two_slot_boundary_plans = getattr(state, "two_slot_boundary_plans", None) or [
            None
        ] * len(state.logits_list)
        # Per-chunk device rejection outputs, kept for the async-spec
        # producer below.
        next_tokens_per_chunk: list[torch.Tensor] = []
        if self.is_pooling_model:
            req_ids = cast(
                list[str], self.input_batch.req_ids[: self.input_batch.num_reqs]
            )
            self._update_num_xla_graphs("pooling_step")
        elif is_spec_step:
            cur_start_idx = 0
            all_greedy = self.input_batch.all_greedy
            for (
                logits,
                num_reqs,
                md,
                mamba_indices,
                boundary_plan,
                ping_pong_chunk,
                two_slot_plan,
            ) in zip(
                state.logits_list,
                state.num_reqs_list,
                state.spec_decode_metadata_list,
                mamba_idx_list,
                boundary_plans,
                ping_pong_chunks,
                two_slot_boundary_plans,
            ):
                cur_end_idx = cur_start_idx + num_reqs
                if md is not None:
                    with self._stage_timing.stage("sample"):
                        next_tokens = self._sample_spec_verify_chunk(
                            logits,
                            md,
                            grammar_output,
                            scheduler_output,
                            cur_start_idx,
                            cur_end_idx,
                            all_greedy,
                            sampling_generator,
                            state.top_tokens,
                            min_toks,
                        )
                    combined_selected_tokens.append(next_tokens)
                    combined_selected_tokens_real_lens.append(num_reqs)
                    next_tokens_per_chunk.append(next_tokens)
                    if ping_pong_chunk is not None or has_ping_pong_copy_groups(self):
                        active_plan = (
                            two_slot_plan
                            if two_slot_plan is not None
                            else boundary_plan
                        )
                        ple_handled = self._post_verify_two_slot_gdn_chunk(
                            ping_pong_chunk,
                            md,
                            next_tokens,
                            active_plan,
                            mamba_indices,
                        )
                        if not ple_handled:
                            self._update_mamba_slot_read_offsets(
                                mamba_indices, next_tokens, num_reqs
                            )
                    else:
                        read_offsets = self._update_mamba_slot_read_offsets(
                            mamba_indices, next_tokens, num_reqs
                        )
                        spec_boundary_state.copy_states(
                            self, boundary_plan, read_offsets
                        )
                    if needs_logprobs:
                        # Raw verify logits, one row per draft slot and bonus
                        # slot; each row paired with the token its slot chose.
                        row_tokens = self.spec_logprob_row_tokens(
                            next_tokens,
                            md.bonus_logits_indices,
                            md.draft_lengths,
                            md.final_logits_indices,
                        )
                        logprobs_chunks.append(
                            LogprobsChunk(
                                tensors=self.gather_logprobs(logits, row_tokens),
                                row_starts=np.concatenate(
                                    ([0], np.cumsum(md.draft_lengths_cpu + 1)[:-1])
                                ),
                            )
                        )
                else:
                    selected, logprob_logits = self._sample_plain_chunk(
                        logits,
                        cur_start_idx=cur_start_idx,
                        cur_end_idx=cur_end_idx,
                        top_tokens=state.top_tokens,
                        all_greedy=all_greedy,
                        sampling_generator=sampling_generator,
                        grammar_output=grammar_output,
                        scheduler_output=scheduler_output,
                        is_spec_step=True,
                        min_toks=min_toks,
                    )
                    assert self.speculative_config is not None
                    padded = torch.full(
                        (
                            selected.shape[0],
                            self.speculative_config.num_speculative_tokens + 1,
                        ),
                        INVALID_TOKEN_ID,
                        dtype=selected.dtype,
                        device=selected.device,
                    )
                    padded[:, 0] = selected.view(-1)
                    combined_selected_tokens.append(padded)
                    combined_selected_tokens_real_lens.append(num_reqs)
                    next_tokens_per_chunk.append(padded)
                    if needs_logprobs:
                        logprobs_chunks.append(
                            LogprobsChunk(
                                tensors=self.gather_logprobs(logprob_logits, selected),
                                row_starts=np.arange(num_reqs),
                            )
                        )
                    # Non-verify chunk in a spec step: export any 1-token
                    # ping-pong boundary crossing before updating/resetting offsets.
                    if ping_pong_chunk is not None or two_slot_plan is not None:
                        self._copy_two_slot_boundary_states(
                            two_slot_plan, None, ping_pong_chunk=ping_pong_chunk
                        )
                    self._update_mamba_slot_read_offsets(mamba_indices, None, num_reqs)
                    self._update_ping_pong_mamba_slot_read_offsets(ping_pong_chunk)
                self._update_num_xla_graphs("spec_step")
                cur_start_idx = cur_end_idx
        else:
            cur_start_idx = 0
            req_ids = cast(
                list[str], self.input_batch.req_ids[: self.input_batch.num_reqs]
            )
            all_greedy = self.input_batch.all_greedy
            for (
                logits,
                num_reqs,
                mamba_indices,
                ping_pong_chunk,
                two_slot_plan,
            ) in zip(
                state.logits_list,
                state.num_reqs_list,
                mamba_idx_list,
                ping_pong_chunks,
                two_slot_boundary_plans,
            ):
                cur_end_idx = cur_start_idx + num_reqs
                # Non-spec step (prefill / plain decode): export any 1-token
                # ping-pong boundary crossing before resetting/flipping offsets.
                if ping_pong_chunk is not None or two_slot_plan is not None:
                    self._copy_two_slot_boundary_states(
                        two_slot_plan, None, ping_pong_chunk=ping_pong_chunk
                    )
                self._update_mamba_slot_read_offsets(mamba_indices, None, num_reqs)
                self._update_ping_pong_mamba_slot_read_offsets(ping_pong_chunk)
                selected_token_ids, logprob_logits = self._sample_plain_chunk(
                    logits,
                    cur_start_idx=cur_start_idx,
                    cur_end_idx=cur_end_idx,
                    top_tokens=state.top_tokens,
                    all_greedy=all_greedy,
                    sampling_generator=sampling_generator,
                    grammar_output=grammar_output,
                    scheduler_output=scheduler_output,
                    is_spec_step=False,
                    min_toks=min_toks,
                )
                # NOTE (NickLucche) Use the original logits (before any penalties or
                # temperature scaling) for the top-k logprobs. We can't enforce it
                # due to recompilations outside torch.compiled code, so just make
                # sure `sample_from_logits` does not modify the logits in-place.
                if needs_logprobs:
                    logprobs_chunks.append(
                        LogprobsChunk(
                            tensors=self.gather_logprobs(
                                logprob_logits, selected_token_ids
                            ),
                            row_starts=np.arange(num_reqs),
                        )
                    )

                # Keep the bucketed (padded) tensor; trim happens on the host
                # inside AsyncTPUCopyState to avoid per-`num_reqs` recompiles.
                combined_selected_tokens.append(selected_token_ids)
                combined_selected_tokens_real_lens.append(num_reqs)
                if self.scheduler_config.async_scheduling:
                    next_tokens_tpu_chunks.append(selected_token_ids.view(-1))
                    for req_idx in range(cur_start_idx, cur_end_idx):
                        next_token_indices[req_idx] = (
                            next_tokens_tpu_offset + req_idx - cur_start_idx
                        )
                    next_tokens_tpu_offset += selected_token_ids.shape[0]

                self._update_num_xla_graphs("decoding_step")
                cur_start_idx = cur_end_idx

        # NOTE: current kv load and save get h2d/d2h copies involved.
        # Those copies are blocking. Once they become async., kv_save
        # should be called right after each single forward pass,
        # instead of the forwards of the entire input batch.

        discard_sampled_tokens_req_indices = []
        num_reqs = self.input_batch.num_reqs
        req_ids = cast(list[str], self.input_batch.req_ids[:num_reqs])

        request_seq_lens = []
        for i, req_id in enumerate(req_ids):
            assert req_id is not None
            req_state = self.requests[req_id]
            seq_len = (
                req_state.num_computed_tokens
                + scheduler_output.num_scheduled_tokens[req_id]
            )
            # Ignore the sampled token from the partial request.
            # Rewind the generator state as if the token was not sampled.
            if seq_len < req_state.num_tokens:
                generator = self.input_batch.generators.get(i)
                if generator is not None:
                    # This relies on cuda-specific torch-internal impl details
                    generator.set_offset(generator.get_offset() - 4)
                # Record the index of the request that should not be sampled,
                # so that we could clear the sampled tokens before returning.
                discard_sampled_tokens_req_indices.append(i)
            else:
                request_seq_lens.append((i, req_state, seq_len, req_id))

        next_tokens_tpu = None
        copy_state = None
        if combined_selected_tokens:
            with self._stage_timing.stage("sampled_token_d2h"):
                copy_state = AsyncTPUCopyState.from_device_chunks(
                    combined_selected_tokens,
                    combined_selected_tokens_real_lens,
                    logprobs=logprobs_chunks,
                )

        # The previous step's sampled-token copy completes only behind this
        # step's forward, so waiting on it here holds the draft's host-side
        # preparation and launch until the forward finishes. Nothing the
        # draft reads depends on the rollback it feeds (the draft seeds on
        # device; host tokens are read for partial prefills only, i.e. from
        # the prompt), so wait after the launch instead. With a KV connector
        # keep the early wait: the save below relies on it.
        defer_prev_wait = (
            self.scheduler_config.async_scheduling and not has_kv_transfer_group()
        )
        if self.scheduler_config.async_scheduling and not defer_prev_wait:
            self._modify_prev_results()

        # Unified eagle3 draft propose -- ONE call serving both sync and async
        # (mirrors the tpu-inference reference, where a single
        # propose_draft_token_ids call runs every step and an async bool only
        # picks the return form). Seeds on-device from this step's rejection
        # output (spec step) or the just-sampled tokens (non-spec / prefill
        # step), so neither mode waits on a host copy of the sampled tokens.
        # Async gets the raw [num_reqs, K] device tensor back (consumed by the
        # substitution assemblers below); sync caches the host list in the
        # manager for take_draft_token_ids(). ngram has no device path and
        # keeps proposing from the host-materialized output in the sync block
        # below.
        is_async = bool(self.scheduler_config.async_scheduling)
        eagle3_drafts = None
        if (
            self._is_async_drafter
            and combined_selected_tokens
            and not self._is_dspark_pd_producer
        ):
            if is_spec_step:
                with self._stage_timing.stage("draft_propose"):
                    eagle3_drafts = self.spec_decode_manager.propose_draft_token_ids(
                        sampled_token_ids=None,
                        discard_sampled_tokens_req_indices=discard_sampled_tokens_req_indices,
                        num_rejected_tokens_np=None,
                        scheduler_output=scheduler_output,
                        return_device=is_async,
                        next_tokens_per_chunk=next_tokens_per_chunk,
                    )
            else:
                # Seed = the just-sampled tokens, kept on-device (no D2H):
                # propose takes the sync seed path but reads the seed from
                # `device_seed`.
                device_seed: torch.Tensor | list[torch.Tensor]
                if getattr(self._async_drafter, "bucketed_drafts", False):
                    # One bucketed tensor per chunk; the drafter masks the
                    # padded tail on device (no exact-count slice here).
                    device_seed = [sel.view(-1) for sel in combined_selected_tokens]
                else:
                    flat_seed = torch.cat(
                        [
                            sel.view(-1)[:n]
                            for sel, n in zip(
                                combined_selected_tokens,
                                combined_selected_tokens_real_lens,
                            )
                        ]
                    )
                    if self._mtp_prefill_only:
                        # The producer's sample at the prompt boundary is
                        # thrown away; the draft row there takes the token the
                        # consumer computes, carried by this step's published
                        # send.
                        flat_seed = handoff_seed_tokens(
                            flat_seed, req_ids, scheduler_output.kv_connector_metadata
                        )
                    device_seed = flat_seed
                with self._stage_timing.stage("draft_propose"):
                    eagle3_drafts = self.spec_decode_manager.propose_draft_token_ids(
                        sampled_token_ids=[],
                        discard_sampled_tokens_req_indices=discard_sampled_tokens_req_indices,
                        num_rejected_tokens_np=None,
                        scheduler_output=scheduler_output,
                        return_device=is_async,
                        device_seed=device_seed,
                    )
        if self._adaptive_verification is not None:
            self._record_draft_confidences()

        if defer_prev_wait:
            self._modify_prev_results()

        if self._is_dspark_pd_producer:
            assert isinstance(self.drafter, DFlashProposer)
            self.drafter.prepare_prefill_handoff()
            # Context projections are deferred TPU graphs. A hardware-only
            # barrier does not materialize their writes before Raiden reads.
            synchronize_tensors()

        if copy_state is not None and self._fence_spec_kv_under_token_copy:
            # A Raiden producer registers a request's KV blocks ahead of its
            # finish, so a consumer may pull as soon as the first token is
            # out, while the drafter's KV writes above are still deferred.
            # Enqueue the drafter's graph now through its own output (the
            # draft tokens depend on the draft-layer attention that performs
            # those writes) and move the sampled-token copy's ready event
            # behind it, so the host sees this step's tokens only once every
            # KV write of the step has executed. Rooting the enqueue on the
            # draft tokens keeps the graph to the drafter itself; rooting it
            # on the KV cache tensors was measured to enqueue work the size
            # of the whole forward again.
            if isinstance(eagle3_drafts, torch.Tensor):
                synchronize_tensors([eagle3_drafts], wait=False)
            copy_state.fence_pending_device_work()

        # Save, poll and clear the KV connector only after the drafter. Its
        # forward above writes the draft layers' KV into the blocks the target
        # just filled; a save taken before it, or the save fence
        # TPURaidenOffloadingConnector records in get_transfer_results, would
        # miss those writes. (vLLM likewise defers wait_for_save and the
        # metadata clear past its drafter.) ngram drafts below from host tokens
        # and writes no KV. The wait above also proves the previous forward
        # finished, so a store deferred behind it can be handed to raiden in
        # this step rather than the next. Note that state.kv_save_nan_check
        # snapshots only the target forward chunks from execute_model(), not
        # the drafter's KV writes above.
        self.maybe_wait_for_kv_save(state.kv_save_nan_check)
        kv_connector_output = self._build_kv_connector_output(scheduler_output)

        if self.scheduler_config.async_scheduling:
            # A replicated (draft_tp=1) drafter proposes independently per TP
            # rank and the proposals are not bit-identical; async verifies
            # each rank's OWN drafts (no driver round-trip like sync), so the
            # ranks drift apart and the output corrupts. Pin every rank to
            # rank 0's proposal; sharded drafters skip this.
            eagle3_drafts = self._sync_replicated_drafts_across_tp(eagle3_drafts)

            if (
                eagle3_drafts is not None
                and scheduler_output.has_structured_output_requests
            ):
                # The deferred grammar-bitmask will fetch these drafts via
                # take_draft_token_ids() to replace the async scheduler's -1
                # splaceholders. Stage after the TP pin above so the host
                # copy matches what gets substituted on device.
                self.spec_decode_manager.stage_draft_token_ids_for_host(
                    eagle3_drafts, state.num_reqs_list
                )

            # Build the async substitution source from the drafts proposed
            # above: the [bonus, draft_1..K] source + 1+K next_token_indices,
            # and the per-request rejected count to park.
            spec_num_rejected = None
            num_draft_per_req = None
            if is_spec_step:
                (
                    next_tokens_tpu_chunks,
                    next_token_indices,
                    spec_num_rejected,
                    num_draft_per_req,
                ) = self._assemble_async_spec_substitution(
                    eagle3_drafts, next_tokens_per_chunk, state
                )
            elif self._is_async_drafter and next_tokens_tpu_chunks:
                # Prefill / pure-non-spec eagle3 bootstrap: the non-spec sampling
                # branch parked a stride-1 source (sampled token only). Rebuild
                # it as a stride-(1+K) [bonus, draft_1..K] source carrying real
                # prompt-context drafts so the NEXT step's verify gets real
                # drafts instead of placeholders.
                (
                    next_tokens_tpu_chunks,
                    next_token_indices,
                    spec_num_rejected,
                    num_draft_per_req,
                ) = self._assemble_async_prefill_bootstrap(
                    eagle3_drafts,
                    combined_selected_tokens,
                    combined_selected_tokens_real_lens,
                )
            req_id_to_index_copy = {}
            if not self.is_pooling_model:
                req_id_to_index_copy = self._update_placeholder(
                    discard_sampled_tokens_req_indices,
                    request_seq_lens,
                    next_token_indices,
                    num_draft_per_req,
                )
            if next_tokens_tpu_chunks:
                if len(next_tokens_tpu_chunks) == 1:
                    next_tokens_tpu = next_tokens_tpu_chunks[0]
                else:
                    next_tokens_tpu = torch.cat(next_tokens_tpu_chunks, dim=0)
                # Bound the async-spec source length so the torch.compiled
                # _substitute_placeholder_token (next step) sees a fixed shape
                # instead of recompiling per real num_reqs. Append-pad to the
                # max stride-(1+K) span: the substitution only reads
                # [position*stride .. +n_sched-1] for real reqs (positions <
                # num_reqs), so the appended tail is never indexed -- real
                # positions are unchanged. Spec only (stride 1+K); pure-non-spec
                # async keeps its stride-1 source + existing precompile.
                if spec_num_rejected is not None:
                    assert self.speculative_config is not None
                    rows = self._async_spec_source_rows()
                    bound = rows * (1 + self.speculative_config.num_speculative_tokens)
                    if next_tokens_tpu.shape[0] < bound:
                        next_tokens_tpu = torch.nn.functional.pad(
                            next_tokens_tpu, (0, bound - next_tokens_tpu.shape[0])
                        )
                    elif next_tokens_tpu._base is not None:
                        # A full single chunk parks the flattened view of its
                        # source; the precompile warms a plain tensor, and a
                        # view input compiles another program per bucket.
                        next_tokens_tpu = next_tokens_tpu.clone()
                    # Same for the rejected counts the next step's
                    # _subtract_num_rejected_tokens reads by request
                    # position: one fixed length, however many chunks and
                    # rows this step had. The tail is never indexed.
                    if spec_num_rejected.shape[0] < rows:
                        spec_num_rejected = torch.nn.functional.pad(
                            spec_num_rejected, (0, rows - spec_num_rejected.shape[0])
                        )
                self._pre_async_results = AsyncPreResults(
                    req_ids=req_ids,
                    next_tokens_tpu=next_tokens_tpu,
                    request_seq_lens=request_seq_lens,
                    discard_sampled_tokens_req_indices=discard_sampled_tokens_req_indices,
                    req_id_to_index_copy=req_id_to_index_copy,
                    copy_state=copy_state,
                    spec_decode_num_rejected_tokens=spec_num_rejected,
                    num_draft_per_req=num_draft_per_req,
                )
            else:
                self._pre_async_results = None

        pooler_output: list[torch.Tensor | None] = []
        if self.is_pooling_model:
            for out in state.pooler_output_list:
                if isinstance(out, StaticPoolerChunkOutput):
                    pooler_output.extend(unpack_static_pool_output(out))
                    continue
                # Move to CPU as expected by vLLM
                if isinstance(out, torch.Tensor):
                    pooler_output.extend(list(out.cpu().unbind(dim=0)))
                elif isinstance(out, list):
                    for item in out:
                        if isinstance(item, torch.Tensor):
                            pooler_output.append(item.cpu())
                        else:
                            pooler_output.append(item)
                elif out is not None:
                    pooler_output.extend(out)

        with self._stage_timing.host_stage("output_assembly"):
            model_runner_output = ModelRunnerOutput(
                req_ids=req_ids,
                # Snapshot of the req_id_to_index for the VLLM scheduler.
                req_id_to_index=dict(self.input_batch.req_id_to_index),
                sampled_token_ids=[],  # Filled in AsyncTPUModelRunnerOutput get_output
                logprobs=None,  # Filled in AsyncTPUModelRunnerOutput get_output
                prompt_logprobs_dict=prompt_logprobs.emit(
                    state.prompt_logprobs_chunks,
                    self.requests,
                    self.num_prompt_logprobs,
                ),
                pooler_output=pooler_output,
                kv_connector_output=kv_connector_output,
            )
            # Read by SrptPrefillScheduler.update_from_output.
            # pyrefly: ignore[missing-attribute]
            model_runner_output.other_rank_decoding = self._dp_step_other_decode

            async_output = AsyncTPUModelRunnerOutput(
                model_runner_output=model_runner_output,
                copy_state=copy_state,
                discard_sampled_tokens_req_indices=discard_sampled_tokens_req_indices,
            )

        if not self.scheduler_config.async_scheduling:
            final_output = async_output.get_output()
            if not self.is_pooling_model:
                for i, req_state, seq_len, req_id in request_seq_lens:
                    if i in discard_sampled_tokens_req_indices:
                        continue
                    valid_tokens = final_output.sampled_token_ids[i]
                    if not valid_tokens:
                        continue
                    req_idx = self.input_batch.req_id_to_index[req_id]

                    # Update the persistent batch.
                    start_tok_idx = self.input_batch.num_tokens_no_spec[req_idx]
                    end_tok_idx = start_tok_idx + len(valid_tokens)
                    self.input_batch.token_ids_cpu[
                        req_idx, start_tok_idx:end_tok_idx
                    ] = valid_tokens
                    self.input_batch.num_tokens_no_spec[req_idx] = end_tok_idx

                    req_state.output_token_ids.extend(valid_tokens)

            if self.speculative_config and self.speculative_config.method == "ngram":
                # ngram drafts on the host from the committed token ids, so it
                # must run after materialization; eagle3/dflash were already proposed
                # from device state by the unified call above.
                self.spec_decode_manager.propose_draft_token_ids(
                    sampled_token_ids=final_output.sampled_token_ids,
                    discard_sampled_tokens_req_indices=discard_sampled_tokens_req_indices,
                    num_rejected_tokens_np=None,
                    scheduler_output=scheduler_output,
                )

            return final_output

        return async_output
