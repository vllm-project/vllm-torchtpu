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
"""Persistent decode/verify input buffer tracker and shadow verification.

Supports:
  1. `TPU_STATIC_DECODE_INPUTS`: persistent host/device metadata buffer tracker
     (`StaticDecodeBufferTracker`) for decode-only and uniform speculative-verify
     batches, updating only changed entries per step and skipping redundant
     block-table and attention-metadata rebuilds when no request crosses a KV
     or Mamba block boundary.
  2. `TPU_STATIC_DECODE_INPUTS_VERIFY`: shadow-verification assertions comparing
     every device-consumed tensor produced by the static decode path against a
     freshly computed reference (`build_fresh_verify_reference` and
     `verify_static_decode_PrepareInputs`).
"""

import copy
import types
from typing import Any

import numpy as np
import torch


class StaticDecodeBufferTracker:
    """Persistent host/device buffer and metadata tracker for
    `TPU_STATIC_DECODE_INPUTS=1`.

    Maintains persistent preallocated host arrays for pure-decode (`q_len == 1`)
    and uniform spec-verify (`q_len == 1 + K`) steps, updating only changed
    entries in place (`positions += accepted_count`, `seq_lens += accepted_count`,
    `input_ids` gathered at updated positions, and block tables rebuilt only when
    batch membership changes or a request crosses a KV block boundary).
    """

    def __init__(
        self,
        max_num_reqs: int,
        max_num_tokens: int,
        block_sizes: tuple[int, ...] = (16,),
    ) -> None:
        self.max_num_reqs = max_num_reqs
        self.max_num_tokens = max_num_tokens
        self.block_sizes = block_sizes
        self.positions_np = np.zeros(max_num_tokens, dtype=np.int32)
        self.input_ids_np = np.zeros(max_num_tokens, dtype=np.int32)
        self.seq_lens_np = np.zeros(max_num_reqs, dtype=np.int32)
        self.query_start_loc_np = np.zeros(max_num_reqs + 1, dtype=np.int32)
        self.prev_req_ids: tuple[str, ...] | None = None
        self.prev_query_len: int = 0
        self.prev_num_computed: np.ndarray | None = None
        self.prev_block_counts: tuple[tuple[int, ...], ...] | None = None
        self.cached_attn_metadata: dict[str, Any] | None = None
        self.cached_unified_mamba_state_indices: list[torch.Tensor] | None = None
        self.cached_bt_ver_key: Any = None
        self.cached_target_num_reqs: int = 0
        self.cached_padded_tokens: int = 0
        self.metadata_reused_last_step: bool = False
        self.block_boundary_crossed_last_step: bool = False

    def _compute_block_counts(
        self,
        seq_lens_slice: np.ndarray,
        max_rejected: int = 0,
    ) -> tuple[tuple[tuple[int, ...], ...], bool]:
        seq_arr = np.asarray(seq_lens_slice, dtype=np.int32)
        counts_per_group: list[tuple[int, ...]] = []
        straddles_boundary = False
        rej_bound = max(0, max_rejected)
        if rej_bound > 0 and seq_arr.size > 0:
            high_pos = np.maximum(0, seq_arr - 1)
            low_pos = np.maximum(0, seq_arr - rej_bound - 1)
        else:
            high_pos = None
            low_pos = None
        for block_size in self.block_sizes:
            eff_bs = max(1, block_size)
            num_blocks = (seq_arr + eff_bs - 1) // eff_bs
            counts_per_group.append(tuple(num_blocks.tolist()))
            if (
                not straddles_boundary
                and high_pos is not None
                and low_pos is not None
                and bool(np.any((low_pos // eff_bs) != (high_pos // eff_bs)))
            ):
                straddles_boundary = True
        return tuple(counts_per_group), straddles_boundary

    def update_step_buffers(
        self,
        req_ids: list[str] | tuple[str, ...],
        num_computed_tokens: np.ndarray,
        token_ids_table: np.ndarray,
        query_len: int,
        padded_num_tokens: int | None = None,
        force_block_rebuild: bool = False,
        max_rejected: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, bool]:
        """Update persistent host buffers in place for the current decode/verify
        step.

        Returns `(input_ids_slice, positions_slice, seq_lens_slice,
        query_start_loc_slice, same_batch, block_boundary_crossed)`.
        """
        num_reqs = len(req_ids)
        total_tokens = num_reqs * query_len
        req_ids_tuple = tuple(req_ids)
        computed_slice = np.asarray(num_computed_tokens[:num_reqs], dtype=np.int32)
        stride = int(token_ids_table.shape[1])

        same_batch = (
            self.prev_req_ids == req_ids_tuple
            and self.prev_query_len == query_len
            and self.prev_num_computed is not None
            and len(self.prev_num_computed) == num_reqs
            and bool((computed_slice >= self.prev_num_computed).all())
        )
        if same_batch:
            assert self.prev_num_computed is not None
            accepted_delta = computed_slice - self.prev_num_computed
            self.prev_num_computed[:] = computed_slice
            if query_len == 1:
                self.positions_np[:num_reqs] += accepted_delta
            else:
                delta_per_token = np.repeat(accepted_delta, query_len)
                self.positions_np[:total_tokens] += delta_per_token
            self.seq_lens_np[:num_reqs] += accepted_delta
        else:
            self.prev_req_ids = req_ids_tuple
            self.prev_query_len = query_len
            self.prev_num_computed = computed_slice.copy()
            if query_len == 1:
                np.copyto(self.positions_np[:num_reqs], computed_slice)
            else:
                req_repeats = np.repeat(computed_slice, query_len)
                intra_offsets = np.tile(np.arange(query_len, dtype=np.int32), num_reqs)
                np.add(
                    req_repeats,
                    intra_offsets,
                    out=self.positions_np[:total_tokens],
                )
            self.seq_lens_np[:num_reqs] = computed_slice + query_len
            self.seq_lens_np[num_reqs:] = 0
            self.query_start_loc_np[0] = 0
            self.query_start_loc_np[1 : num_reqs + 1] = (
                np.arange(1, num_reqs + 1, dtype=np.int32) * query_len
            )
            self.query_start_loc_np[num_reqs + 1 :] = total_tokens

        if query_len == 1:
            req_row_offsets = np.arange(num_reqs, dtype=np.int64) * stride
        else:
            req_row_offsets = (
                np.repeat(np.arange(num_reqs, dtype=np.int64), query_len) * stride
            )
        token_flat_indices = (
            self.positions_np[:total_tokens].astype(np.int64) + req_row_offsets
        )
        np.take(
            token_ids_table.ravel(),
            token_flat_indices,
            out=self.input_ids_np[:total_tokens],
        )
        if padded_num_tokens is not None and padded_num_tokens > total_tokens:
            self.input_ids_np[total_tokens:padded_num_tokens] = 0
            self.positions_np[total_tokens:padded_num_tokens] = 0

        new_block_counts, straddles_boundary = self._compute_block_counts(
            self.seq_lens_np[:num_reqs],
            max_rejected=max_rejected,
        )
        block_boundary_crossed = (
            not same_batch
            or force_block_rebuild
            or straddles_boundary
            or self.prev_block_counts != new_block_counts
        )
        self.prev_block_counts = None if straddles_boundary else new_block_counts
        self.block_boundary_crossed_last_step = block_boundary_crossed
        self.metadata_reused_last_step = same_batch and not block_boundary_crossed

        eff_tokens = (
            padded_num_tokens
            if padded_num_tokens is not None and padded_num_tokens >= total_tokens
            else total_tokens
        )
        return (
            self.input_ids_np[:eff_tokens],
            self.positions_np[:eff_tokens],
            self.seq_lens_np[:num_reqs],
            self.query_start_loc_np[: num_reqs + 1],
            same_batch,
            block_boundary_crossed,
        )

    def reset(self) -> None:
        """Reset cached batch metadata so subsequent steps recompute afresh."""
        self.prev_req_ids = None
        self.prev_query_len = 0
        self.prev_num_computed = None
        self.prev_block_counts = None
        self.cached_attn_metadata = None
        self.cached_unified_mamba_state_indices = None
        self.cached_bt_ver_key = None
        self.cached_target_num_reqs = 0
        self.cached_padded_tokens = 0
        self.metadata_reused_last_step = False
        self.block_boundary_crossed_last_step = False


def _corrupt_first_element(tensor_val: torch.Tensor) -> torch.Tensor:
    """Return a cloned tensor with element 0 incremented by 1."""
    cloned = tensor_val.cpu().clone()
    flat_view = cloned.reshape(-1)
    if flat_view.numel() > 0:
        flat_view[0] = flat_view[0] + 1
    return cloned


def _corrupt_first_np_element(arr_val: np.ndarray) -> np.ndarray:
    """Return a copied numpy array with element 0 incremented by 1."""
    copied = arr_val.copy()
    flat_view = copied.reshape(-1)
    if flat_view.size > 0:
        flat_view[0] = flat_view[0] + 1
    return copied


def corrupt_static_decode_field_for_verify(
    corrupt_field: str,
    static_result: tuple[Any, ...],
    static_input_ids: torch.Tensor,
    static_position_ids: torch.Tensor,
    static_unified_mamba_state_indices: list[torch.Tensor] | None = None,
    static_slot_mappings: dict[Any, torch.Tensor] | None = None,
) -> tuple[
    tuple[Any, ...],
    torch.Tensor,
    torch.Tensor,
    list[torch.Tensor] | None,
    dict[Any, torch.Tensor] | None,
]:
    """Deliberately corrupt `corrupt_field` on a shallow copy of the static
    decode outputs so `TPU_STATIC_DECODE_INPUTS_VERIFY` is proven to catch it.
    """
    target = corrupt_field.strip()
    if not target:
        return (
            static_result,
            static_input_ids,
            static_position_ids,
            static_unified_mamba_state_indices,
            static_slot_mappings,
        )
    res_list = list(static_result)
    if target == "input_ids":
        return (
            tuple(res_list),
            _corrupt_first_element(static_input_ids),
            static_position_ids,
            static_unified_mamba_state_indices,
            static_slot_mappings,
        )
    if target in ("positions", "position_ids"):
        return (
            tuple(res_list),
            static_input_ids,
            _corrupt_first_element(static_position_ids),
            static_unified_mamba_state_indices,
            static_slot_mappings,
        )
    if target == "logits_indices":
        res_list[1] = _corrupt_first_element(res_list[1])
        return (
            tuple(res_list),
            static_input_ids,
            static_position_ids,
            static_unified_mamba_state_indices,
            static_slot_mappings,
        )
    if target == "unified_mamba_state_indices":
        if not static_unified_mamba_state_indices:
            raise AssertionError(
                "corrupt_field failed: unified_mamba_state_indices is empty or None"
            )
        corrupted_umi = list(static_unified_mamba_state_indices)
        corrupted_umi[0] = _corrupt_first_element(corrupted_umi[0])
        return (
            tuple(res_list),
            static_input_ids,
            static_position_ids,
            corrupted_umi,
            static_slot_mappings,
        )
    if target in (
        "spec_decode_metadata",
        "draft_token_ids",
        "deferred_draft_token_ids_cpu",
        "draft_lengths",
        "bonus_logits_indices",
        "target_logits_indices",
        "final_logits_indices",
        "segment_ids",
        "group_indices",
    ):
        spec_md = res_list[7]
        if spec_md is None:
            raise AssertionError(
                f"corrupt_field failed: "
                f"spec_decode_metadata is None for target {target!r}"
            )
        spec_copy = copy.copy(spec_md)
        spec_field = (
            "draft_token_ids"
            if (
                target == "spec_decode_metadata"
                and spec_copy.draft_token_ids is not None
            )
            else (
                "final_logits_indices" if target == "spec_decode_metadata" else target
            )
        )
        field_val = getattr(spec_copy, spec_field)
        if isinstance(field_val, torch.Tensor):
            setattr(spec_copy, spec_field, _corrupt_first_element(field_val))
        elif isinstance(field_val, np.ndarray):
            setattr(spec_copy, spec_field, _corrupt_first_np_element(field_val))
        else:
            raise AssertionError(
                f"corrupt_field failed: "
                f"spec_decode_metadata.{spec_field} is not a Tensor or ndarray"
            )
        res_list[7] = spec_copy
        return (
            tuple(res_list),
            static_input_ids,
            static_position_ids,
            static_unified_mamba_state_indices,
            static_slot_mappings,
        )
    if target == "slot_mapping":
        if not static_slot_mappings:
            raise AssertionError(
                "corrupt_field failed: static_slot_mappings is empty or None"
            )
        corrupted_sm = dict(static_slot_mappings)
        first_sm_key = next(iter(corrupted_sm))
        corrupted_sm[first_sm_key] = _corrupt_first_element(corrupted_sm[first_sm_key])
        return (
            tuple(res_list),
            static_input_ids,
            static_position_ids,
            static_unified_mamba_state_indices,
            corrupted_sm,
        )
    if target in (
        "input_positions",
        "seq_lens",
        "query_start_loc",
        "request_distribution",
        "block_tables",
        "mamba_state_indices",
        "mamba_ckpt_indices",
        "mamba_slot_read_offsets",
        "mamba_request_distribution",
    ):
        attn_md = res_list[0]
        if not isinstance(attn_md, dict) or not attn_md:
            raise AssertionError(
                f"corrupt_field failed: attn_metadata is empty for target {target!r}"
            )
        attn_copy = dict(attn_md)
        corrupted_any = False
        for layer_key, md_obj in attn_copy.items():
            field_val = getattr(md_obj, target)
            if isinstance(field_val, torch.Tensor):
                md_copy = copy.copy(md_obj)
                setattr(md_copy, target, _corrupt_first_element(field_val))
                attn_copy[layer_key] = md_copy
                corrupted_any = True
                break
        if not corrupted_any:
            raise AssertionError(
                f"corrupt_field failed: "
                f"no layer in attn_metadata had tensor field {target!r}"
            )
        res_list[0] = attn_copy
        return (
            tuple(res_list),
            static_input_ids,
            static_position_ids,
            static_unified_mamba_state_indices,
            static_slot_mappings,
        )
    raise AssertionError(
        f"corrupt_field failed: unknown corrupt target field {target!r}"
    )


def _format_verify_error(message: str, step_context: dict[str, Any] | None) -> str:
    if not step_context:
        return message
    return f"{message} | step_context={step_context}"


def verify_static_decode_PrepareInputs(
    ref_result: tuple[Any, ...],
    static_result: tuple[Any, ...],
    ref_input_ids: torch.Tensor,
    static_input_ids: torch.Tensor,
    ref_position_ids: torch.Tensor,
    static_position_ids: torch.Tensor,
    *,
    ref_unified_mamba_state_indices: list[torch.Tensor] | None = None,
    static_unified_mamba_state_indices: list[torch.Tensor] | None = None,
    ref_slot_mappings: dict[Any, torch.Tensor] | None = None,
    static_slot_mappings: dict[Any, torch.Tensor] | None = None,
    corrupt_field: str | None = None,
    step_context: dict[str, Any] | None = None,
) -> None:
    """Assert every tensor field produced by static decode input preparation is
    `torch.equal` on host to the fresh reference output."""
    if corrupt_field and corrupt_field.strip():
        (
            static_result,
            static_input_ids,
            static_position_ids,
            static_unified_mamba_state_indices,
            static_slot_mappings,
        ) = corrupt_static_decode_field_for_verify(
            corrupt_field,
            static_result,
            static_input_ids,
            static_position_ids,
            static_unified_mamba_state_indices=static_unified_mamba_state_indices,
            static_slot_mappings=static_slot_mappings,
        )

    if not torch.equal(ref_input_ids.cpu(), static_input_ids.cpu()):
        raise AssertionError(
            _format_verify_error(
                "TPU_STATIC_DECODE_INPUTS_VERIFY failed: input_ids mismatch",
                step_context,
            )
        )
    if not torch.equal(ref_position_ids.cpu(), static_position_ids.cpu()):
        raise AssertionError(
            _format_verify_error(
                "TPU_STATIC_DECODE_INPUTS_VERIFY failed: position_ids mismatch",
                step_context,
            )
        )
    ref_attn_md, ref_logits_idx = ref_result[0], ref_result[1]
    static_attn_md, static_logits_idx = static_result[0], static_result[1]
    if not torch.equal(ref_logits_idx.cpu(), static_logits_idx.cpu()):
        raise AssertionError(
            _format_verify_error(
                "TPU_STATIC_DECODE_INPUTS_VERIFY failed: logits_indices mismatch",
                step_context,
            )
        )
    if ref_slot_mappings is not None or static_slot_mappings is not None:
        ref_sm = ref_slot_mappings or {}
        static_sm = static_slot_mappings or {}
        if set(ref_sm.keys()) != set(static_sm.keys()):
            raise AssertionError(
                _format_verify_error(
                    "TPU_STATIC_DECODE_INPUTS_VERIFY failed: "
                    "slot_mapping keys mismatch",
                    step_context,
                )
            )
        for sm_key, ref_sm_tensor in ref_sm.items():
            static_sm_tensor = static_sm[sm_key]
            if (
                isinstance(ref_sm_tensor, torch.Tensor)
                and isinstance(static_sm_tensor, torch.Tensor)
                and not torch.equal(ref_sm_tensor.cpu(), static_sm_tensor.cpu())
            ):
                raise AssertionError(
                    _format_verify_error(
                        f"TPU_STATIC_DECODE_INPUTS_VERIFY failed: "
                        f"slot_mapping[{sm_key}] mismatch",
                        step_context,
                    )
                )
    if isinstance(ref_attn_md, dict) and isinstance(static_attn_md, dict):
        if set(ref_attn_md.keys()) != set(static_attn_md.keys()):
            raise AssertionError(
                _format_verify_error(
                    "TPU_STATIC_DECODE_INPUTS_VERIFY failed: "
                    "attn_metadata keys mismatch",
                    step_context,
                )
            )
        for layer_key, ref_md in ref_attn_md.items():
            static_md = static_attn_md[layer_key]
            for field_name, ref_val, static_val in (
                ("input_positions", ref_md.input_positions, static_md.input_positions),
                ("seq_lens", ref_md.seq_lens, static_md.seq_lens),
                ("query_start_loc", ref_md.query_start_loc, static_md.query_start_loc),
                (
                    "request_distribution",
                    ref_md.request_distribution,
                    static_md.request_distribution,
                ),
                ("block_tables", ref_md.block_tables, static_md.block_tables),
                (
                    "mamba_state_indices",
                    ref_md.mamba_state_indices,
                    static_md.mamba_state_indices,
                ),
                (
                    "mamba_ckpt_indices",
                    ref_md.mamba_ckpt_indices,
                    static_md.mamba_ckpt_indices,
                ),
                (
                    "mamba_slot_read_offsets",
                    ref_md.mamba_slot_read_offsets,
                    static_md.mamba_slot_read_offsets,
                ),
                (
                    "mamba_request_distribution",
                    ref_md.mamba_request_distribution,
                    static_md.mamba_request_distribution,
                ),
            ):
                if (ref_val is None) != (static_val is None):
                    ref_none = ref_val is None
                    static_none = static_val is None
                    raise AssertionError(
                        _format_verify_error(
                            f"TPU_STATIC_DECODE_INPUTS_VERIFY failed: "
                            f"attn_metadata[{layer_key}].{field_name} "
                            f"presence mismatch (ref_is_none={ref_none}, "
                            f"static_is_none={static_none})",
                            step_context,
                        )
                    )
                if (
                    isinstance(ref_val, torch.Tensor)
                    and isinstance(static_val, torch.Tensor)
                    and not torch.equal(ref_val.cpu(), static_val.cpu())
                ):
                    raise AssertionError(
                        _format_verify_error(
                            f"TPU_STATIC_DECODE_INPUTS_VERIFY failed: "
                            f"attn_metadata[{layer_key}].{field_name} mismatch",
                            step_context,
                        )
                    )
    if (
        ref_unified_mamba_state_indices is not None
        or static_unified_mamba_state_indices is not None
    ):
        if (ref_unified_mamba_state_indices is None) != (
            static_unified_mamba_state_indices is None
        ):
            raise AssertionError(
                _format_verify_error(
                    "TPU_STATIC_DECODE_INPUTS_VERIFY failed: "
                    "unified_mamba_state_indices presence mismatch",
                    step_context,
                )
            )
        if (
            ref_unified_mamba_state_indices is not None
            and static_unified_mamba_state_indices is not None
        ):
            if len(ref_unified_mamba_state_indices) != len(
                static_unified_mamba_state_indices
            ):
                raise AssertionError(
                    _format_verify_error(
                        "TPU_STATIC_DECODE_INPUTS_VERIFY failed: "
                        "unified_mamba_state_indices length mismatch",
                        step_context,
                    )
                )
            for group_idx, (ref_umi, static_umi) in enumerate(
                zip(
                    ref_unified_mamba_state_indices,
                    static_unified_mamba_state_indices,
                )
            ):
                if not torch.equal(ref_umi.cpu(), static_umi.cpu()):
                    raise AssertionError(
                        _format_verify_error(
                            f"TPU_STATIC_DECODE_INPUTS_VERIFY failed: "
                            f"unified_mamba_state_indices[{group_idx}] mismatch",
                            step_context,
                        )
                    )
    ref_spec_md = ref_result[7]
    static_spec_md = static_result[7]
    if (ref_spec_md is None) != (static_spec_md is None):
        raise AssertionError(
            _format_verify_error(
                "TPU_STATIC_DECODE_INPUTS_VERIFY failed: "
                "spec_decode_metadata presence mismatch",
                step_context,
            )
        )
    if ref_spec_md is not None and static_spec_md is not None:
        for field_name, ref_val, static_val in (
            (
                "draft_token_ids",
                ref_spec_md.draft_token_ids,
                static_spec_md.draft_token_ids,
            ),
            (
                "deferred_draft_token_ids_cpu",
                ref_spec_md.deferred_draft_token_ids_cpu,
                static_spec_md.deferred_draft_token_ids_cpu,
            ),
            (
                "draft_lengths",
                ref_spec_md.draft_lengths,
                static_spec_md.draft_lengths,
            ),
            (
                "bonus_logits_indices",
                ref_spec_md.bonus_logits_indices,
                static_spec_md.bonus_logits_indices,
            ),
            (
                "target_logits_indices",
                ref_spec_md.target_logits_indices,
                static_spec_md.target_logits_indices,
            ),
            (
                "final_logits_indices",
                ref_spec_md.final_logits_indices,
                static_spec_md.final_logits_indices,
            ),
            ("segment_ids", ref_spec_md.segment_ids, static_spec_md.segment_ids),
            (
                "group_indices",
                ref_spec_md.group_indices,
                static_spec_md.group_indices,
            ),
        ):
            if (ref_val is None) != (static_val is None):
                raise AssertionError(
                    _format_verify_error(
                        f"TPU_STATIC_DECODE_INPUTS_VERIFY failed: "
                        f"spec_decode_metadata.{field_name} presence mismatch",
                        step_context,
                    )
                )
            if (
                isinstance(ref_val, torch.Tensor)
                and isinstance(static_val, torch.Tensor)
                and not torch.equal(ref_val.cpu(), static_val.cpu())
            ) or (
                isinstance(ref_val, np.ndarray)
                and isinstance(static_val, np.ndarray)
                and not np.array_equal(ref_val, static_val)
            ):
                raise AssertionError(
                    _format_verify_error(
                        f"TPU_STATIC_DECODE_INPUTS_VERIFY failed: "
                        f"spec_decode_metadata.{field_name} mismatch",
                        step_context,
                    )
                )


def build_fresh_verify_reference(
    runner: Any,
    scheduler_output: Any,
    start_index: int,
    num_decode_reqs: int,
    num_windowed_reqs: int | None = None,
    layout_plan: Any = None,
    padded_num_reqs_override: int | None = None,
) -> dict[str, Any]:
    """Compute a complete, cache-free reference snapshot of all device-consumed
    decode/verify inputs directly from `scheduler_output` + `runner.input_batch`
    + `runner.requests` + `runner._pre_async_results` without reading or writing
    any runner caches.
    """
    if num_windowed_reqs is None:
        num_windowed_reqs = num_decode_reqs
    reorder_thresh = int(runner.reorder_batch_threshold or 1)
    rpa_num_decode_reqs = num_windowed_reqs if reorder_thresh > 1 else num_decode_reqs
    total_batch_reqs = int(runner.input_batch.num_reqs)
    most_model_len = runner.most_model_len
    use_max_model_len = most_model_len is None
    num_scheduled_map = scheduler_output.num_scheduled_tokens
    req_ids_all = runner.input_batch.req_ids
    num_reqs_max = int(runner.num_reqs_max_model_len)
    num_reqs_most = runner.num_reqs_most_model_len

    if use_max_model_len:
        end_index = min(total_batch_reqs, start_index + num_reqs_max)
        sched_per_req = [
            int(num_scheduled_map[req_ids_all[idx]])
            for idx in range(start_index, end_index)
        ]
    else:
        sched_per_req = []
        for idx in range(start_index, total_batch_reqs):
            req_id = req_ids_all[idx]
            num_tok = int(num_scheduled_map[req_id])
            if (
                not use_max_model_len
                and most_model_len is not None
                and num_tok > int(most_model_len)
            ):
                use_max_model_len = True
            sched_per_req.append(num_tok)
        cap_reqs = num_reqs_max if use_max_model_len else int(num_reqs_most or 0)
        if len(sched_per_req) > cap_reqs:
            sched_per_req = sched_per_req[:cap_reqs]
            end_index = start_index + cap_reqs
        else:
            end_index = total_batch_reqs

    num_reqs = len(sched_per_req)
    total_tokens = sum(sched_per_req)
    target_num_reqs = num_reqs_max if use_max_model_len else int(num_reqs_most or 0)
    if padded_num_reqs_override is not None:
        padded_num_reqs = int(padded_num_reqs_override)
    else:
        padded_num_reqs = num_reqs
        req_paddings = runner.num_reqs_paddings
        if req_paddings:
            for pad_cand in req_paddings:
                if int(pad_cand) >= num_reqs:
                    padded_num_reqs = min(int(pad_cand), int(runner.max_num_reqs))
                    break

    padded_total_tokens = (
        int(layout_plan.global_padded_num_tokens)
        if layout_plan is not None
        else total_tokens
    )
    local_token_slice = (
        layout_plan.token_slice
        if layout_plan is not None
        else slice(0, padded_total_tokens)
    )

    qsl_np = np.zeros(target_num_reqs + 1, dtype=np.int32)
    np.cumsum(sched_per_req, out=qsl_np[1 : num_reqs + 1])
    qsl_np[num_reqs + 1 :] = total_tokens

    seq_lens_np = np.zeros(target_num_reqs, dtype=np.int32)
    positions_np = np.zeros(padded_total_tokens, dtype=np.int32)
    input_ids_pre_np = np.zeros(padded_total_tokens, dtype=np.int32)

    num_computed_cpu = runner.input_batch.num_computed_tokens_cpu
    token_ids_cpu = runner.input_batch.token_ids_cpu
    for req_idx in range(num_reqs):
        batch_row = start_index + req_idx
        base_pos = int(num_computed_cpu[batch_row])
        n_sched = int(sched_per_req[req_idx])
        seq_lens_np[req_idx] = base_pos + n_sched
        tok_start = int(qsl_np[req_idx])
        for offset in range(n_sched):
            pos_val = base_pos + offset
            positions_np[tok_start + offset] = pos_val
            input_ids_pre_np[tok_start + offset] = int(
                token_ids_cpu[batch_row, pos_val]
            )

    uses_mrope = bool(runner.uses_mrope)
    reqs_dict = runner.requests
    has_mrope_delta = bool(
        uses_mrope
        and any(
            bool(reqs_dict[req_ids_all[idx]].mrope_position_delta)
            if req_ids_all[idx] in reqs_dict
            else False
            for idx in range(start_index, end_index)
        )
    )
    mrope_pos_np: np.ndarray | None = None
    if uses_mrope:
        mrope_buf = runner.mrope_positions
        mrope_dtype = mrope_buf.cpu.numpy().dtype
        mrope_pos_np = np.zeros((3, padded_total_tokens), dtype=mrope_dtype)
        if has_mrope_delta:
            mrope_pos_np[:, :total_tokens] = mrope_buf.np[:, :total_tokens]
        else:
            mrope_pos_np[:, :total_tokens] = positions_np[:total_tokens]

    async_sched = bool(runner.scheduler_config.async_scheduling)
    spec_cfg = runner.speculative_config
    pre_async = runner._pre_async_results
    spec_k = int(spec_cfg.num_speculative_tokens) if spec_cfg is not None else 0

    rejected_per_req: list[int | None] = [None] * num_reqs
    accepted_per_req: list[int | None] = [None] * num_reqs
    first_decode_after_prefill: list[bool] = [False] * num_reqs
    pre_num_draft_map = pre_async.num_draft_per_req if pre_async is not None else None
    req_id_to_idx_all = (
        pre_async.req_id_to_index_copy or {} if pre_async is not None else {}
    )
    for req_idx in range(num_reqs):
        req_id = req_ids_all[start_index + req_idx]
        req_state = reqs_dict.get(req_id) if isinstance(reqs_dict, dict) else None
        prompt_len = req_state.num_prompt_tokens if req_state is not None else None
        num_comp_req = int(num_computed_cpu[start_index + req_idx])
        if prompt_len is not None and num_comp_req == int(prompt_len):
            first_decode_after_prefill[req_idx] = True
        elif pre_async is not None:
            prev_pos = req_id_to_idx_all.get(req_id)
            if prev_pos is None or (
                isinstance(pre_num_draft_map, dict)
                and int(pre_num_draft_map.get(int(prev_pos), 0)) == 0
            ):
                first_decode_after_prefill[req_idx] = True

    if (
        async_sched
        and spec_cfg is not None
        and pre_async is not None
        and pre_async.spec_decode_num_rejected_tokens is not None
    ):
        rej_arr = pre_async.spec_decode_num_rejected_tokens.cpu().numpy().reshape(-1)
        req_id_to_idx = pre_async.req_id_to_index_copy or {}
        for req_idx in range(num_reqs):
            req_id = req_ids_all[start_index + req_idx]
            prev_pos = req_id_to_idx.get(req_id)
            if prev_pos is not None and 0 <= int(prev_pos) < len(rej_arr):
                rej_val = int(rej_arr[int(prev_pos)])
                rejected_per_req[req_idx] = rej_val
                prev_draft_cnt = (
                    int(pre_num_draft_map.get(int(prev_pos), spec_k))
                    if isinstance(pre_num_draft_map, dict)
                    else spec_k
                )
                accepted_per_req[req_idx] = 1 + max(0, prev_draft_cnt - rej_val)
                seq_lens_np[req_idx] -= rej_val
                tok_start = int(qsl_np[req_idx])
                tok_end = int(qsl_np[req_idx + 1])
                positions_np[tok_start:tok_end] -= rej_val
                if mrope_pos_np is not None:
                    mrope_pos_np[:, tok_start:tok_end] -= rej_val

    input_ids_post_np = input_ids_pre_np.copy()
    any_substituted = False
    if async_sched and pre_async is not None and pre_async.next_tokens_tpu is not None:
        req_id_to_idx = pre_async.req_id_to_index_copy or {}
        if req_id_to_idx:
            next_arr = pre_async.next_tokens_tpu.cpu().numpy().reshape(-1)
            stride = (
                1 + int(spec_cfg.num_speculative_tokens)
                if (
                    spec_cfg is not None
                    and pre_async.spec_decode_num_rejected_tokens is not None
                )
                else 1
            )
            for req_idx in range(num_reqs):
                req_id = req_ids_all[start_index + req_idx]
                prev_pos = req_id_to_idx.get(req_id)
                if prev_pos is None:
                    continue
                src_start = int(prev_pos) * stride
                tok_start = int(qsl_np[req_idx])
                n_sched = int(sched_per_req[req_idx])
                for offset in range(n_sched):
                    dst_idx = tok_start + offset
                    if src_start + offset < len(next_arr):
                        input_ids_post_np[dst_idx] = int(next_arr[src_start + offset])
                        any_substituted = True

    ref_input_ids_pre = torch.from_numpy(input_ids_pre_np[local_token_slice].copy())
    ref_input_ids_post = torch.from_numpy(input_ids_post_np[local_token_slice].copy())
    if uses_mrope and mrope_pos_np is not None:
        ref_position_ids = torch.from_numpy(mrope_pos_np[:, local_token_slice].copy())
    else:
        ref_position_ids = torch.from_numpy(positions_np[local_token_slice].copy())
    ref_seq_lens = torch.from_numpy(seq_lens_np[:target_num_reqs].copy())
    ref_query_start_loc = torch.from_numpy(qsl_np[: target_num_reqs + 1].copy())

    ref_spec_md = None
    if spec_cfg is not None:
        sched_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        if sched_spec_tokens:
            req_ids_slice = req_ids_all[start_index : start_index + num_reqs]
            num_draft_list = [
                len(sched_spec_tokens[rid]) if rid in sched_spec_tokens else 0
                for rid in req_ids_slice
            ]
            if any(num_draft_list):
                num_draft_np = np.array(num_draft_list, dtype=np.int32)
                ref_spec_md = runner.spec_decode_manager.get_spec_decode_metadata(
                    num_draft_np,
                    qsl_np[1 : num_reqs + 1],
                    padded_num_reqs,
                    defer_draft_token_ids=bool(async_sched),
                )
                ref_spec_md = copy.copy(ref_spec_md)
                if async_sched and any_substituted:
                    final_arr = ref_spec_md.final_logits_indices.cpu().numpy()
                    target_arr = ref_spec_md.target_logits_indices.cpu().numpy()
                    gather_pos = final_arr[target_arr + 1]
                    post_slice_arr = input_ids_post_np[local_token_slice]
                    ref_spec_md.draft_token_ids = torch.from_numpy(
                        post_slice_arr[gather_pos].copy()
                    )

    if ref_spec_md is not None:
        ref_logits_indices = ref_spec_md.final_logits_indices.cpu().clone()
    elif layout_plan is not None and layout_plan.logits_indices_cpu is not None:
        ref_logits_indices = layout_plan.logits_indices_cpu.cpu().clone()
    else:
        ref_logits_indices = torch.from_numpy(
            (qsl_np[1 : padded_num_reqs + 1] - 1).copy()
        )

    chunk_num_decode = max(0, min(rpa_num_decode_reqs - start_index, num_reqs))
    chunk_num_windowed = max(0, min(num_windowed_reqs - start_index, num_reqs))
    chunk_static_verify_end = max(
        chunk_num_decode,
        min(
            num_decode_reqs + int(runner._num_static_verify_reqs or 0) - start_index,
            num_reqs,
        ),
    )
    has_read_offsets = runner.mamba_slot_read_offsets is not None
    ref_request_distribution = torch.tensor(
        [
            chunk_num_decode,
            chunk_num_decode if has_read_offsets else chunk_static_verify_end,
            num_reqs,
        ],
        dtype=torch.int32,
    )
    ref_mamba_request_distribution = (
        torch.tensor(
            [chunk_num_windowed, chunk_num_windowed, num_reqs],
            dtype=torch.int32,
        )
        if has_read_offsets
        else None
    )

    ref_attn_metadata: dict[str, Any] = {}
    ref_unified_mamba_state_indices: list[torch.Tensor] | None = (
        [] if (bool(runner._unified_kv_layout) and has_read_offsets) else None
    )
    compact_mamba_indices: torch.Tensor | None = None
    if bool(runner._has_mamba_state):
        slot_by_req = runner._mamba_slot_by_req_id or {}
        compact_np = np.zeros(target_num_reqs, dtype=np.int32)
        for req_idx in range(num_reqs):
            req_id = req_ids_all[start_index + req_idx]
            slot_val = slot_by_req.get(req_id)
            if slot_val is not None:
                compact_np[req_idx] = int(slot_val)
        compact_mamba_indices = torch.from_numpy(compact_np)

    attn_groups = runner.attn_groups or ()
    ckpt_window = int(runner._mamba_ckpt_window or 1)
    geom_ctx = types.SimpleNamespace(
        use_max_model_len=use_max_model_len,
        mamba_ckpt_window=ckpt_window,
    )
    for group_list in attn_groups:
        for group in group_list:
            builder = group.metadata_builders[0]
            block_table_obj, target_num_blocks = builder._block_table_geometry(geom_ctx)
            source_bt: Any = block_table_obj.get_cpu_tensor()
            source_np = source_bt.numpy()
            bt_padded_np = np.zeros(
                (target_num_reqs, target_num_blocks), dtype=np.int32
            )
            bt_padded_np[:num_reqs, :target_num_blocks] = source_np[
                start_index : start_index + num_reqs, :target_num_blocks
            ]
            ref_bt_flat = torch.from_numpy(bt_padded_np.reshape(-1).copy())

            mamba_state_idx: torch.Tensor | None = None
            mamba_ckpt_idx: torch.Tensor | None = None
            is_mamba_grp = bool(builder.is_mamba_group)
            if is_mamba_grp:
                if compact_mamba_indices is not None:
                    mamba_state_idx = compact_mamba_indices
                elif bool(runner._unified_kv_layout):
                    target_bs = max(1, int(builder.target_block_size))
                    active_mask = seq_lens_np[:target_num_reqs] > 0
                    state_cols = np.clip(
                        (seq_lens_np[:target_num_reqs] - 1) // target_bs,
                        0,
                        max(0, target_num_blocks - 1),
                    )
                    row_arange = np.arange(target_num_reqs)
                    gathered_state = bt_padded_np[row_arange, state_cols]
                    state_np = np.where(active_mask, gathered_state, 0).astype(np.int32)
                    mamba_state_idx = torch.from_numpy(state_np)
                    if ckpt_window > 1:
                        ckpt_cols = (
                            state_cols[:, None]
                            + np.arange(ckpt_window, dtype=np.int64)[None, :]
                        )
                        in_row = (ckpt_cols < target_num_blocks) & active_mask[:, None]
                        safe_cols = np.where(in_row, ckpt_cols, 0)
                        ckpt_np = (
                            bt_padded_np[row_arange[:, None], safe_cols] * in_row
                        ).astype(np.int32)
                        mamba_ckpt_idx = torch.from_numpy(ckpt_np)
                    if ref_unified_mamba_state_indices is not None:
                        ref_unified_mamba_state_indices.append(mamba_state_idx)

            ref_md_obj = types.SimpleNamespace(
                input_positions=ref_position_ids,
                block_tables=ref_bt_flat,
                seq_lens=ref_seq_lens,
                query_start_loc=ref_query_start_loc,
                request_distribution=ref_request_distribution,
                mamba_state_indices=mamba_state_idx,
                mamba_ckpt_indices=mamba_ckpt_idx,
                mamba_slot_read_offsets=(
                    runner.mamba_slot_read_offsets.cpu().clone()
                    if (is_mamba_grp and has_read_offsets)
                    else None
                ),
                mamba_request_distribution=(
                    ref_mamba_request_distribution if is_mamba_grp else None
                ),
            )
            for layer_name in group.layer_names:
                ref_attn_metadata[layer_name] = ref_md_obj

    step_context: dict[str, Any] = {
        "num_reqs": num_reqs,
        "padded_num_reqs": padded_num_reqs,
        "target_num_reqs": target_num_reqs,
        "total_tokens": total_tokens,
        "req_ids": [str(req_ids_all[start_index + i]) for i in range(num_reqs)],
        "num_scheduled_tokens_per_req": list(sched_per_req),
        "num_computed_tokens": [
            int(num_computed_cpu[start_index + i]) for i in range(num_reqs)
        ],
        "ref_seq_lens": [int(seq_lens_np[i]) for i in range(num_reqs)],
        "ref_positions_head": [
            int(positions_np[i]) for i in range(min(total_tokens, 32))
        ],
        "first_decode_after_prefill": first_decode_after_prefill,
        "any_first_decode_after_prefill": any(first_decode_after_prefill),
        "has_new_prefill_reqs": bool(scheduler_output.scheduled_new_reqs),
        "is_uniform_spec_step": bool(
            spec_k > 0
            and len(sched_per_req) == num_reqs
            and all(int(cnt) == 1 + spec_k for cnt in sched_per_req)
        ),
        "accepted_counts": accepted_per_req,
        "rejected_counts": rejected_per_req,
    }

    ref_tuple = (
        ref_attn_metadata,
        ref_logits_indices,
        padded_num_reqs,
        num_reqs,
        end_index,
        None,
        None,
        ref_spec_md,
        None,
        layout_plan,
    )
    return {
        "ref_result": ref_tuple,
        "ref_input_ids_pre": ref_input_ids_pre,
        "ref_input_ids_post": ref_input_ids_post,
        "ref_position_ids": ref_position_ids,
        "ref_unified_mamba_state_indices": ref_unified_mamba_state_indices,
        "any_substituted": any_substituted,
        "num_reqs": num_reqs,
        "total_tokens": total_tokens,
        "step_context": step_context,
    }
