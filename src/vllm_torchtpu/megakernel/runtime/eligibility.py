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
"""Startup / post-mesh / step eligibility rules (policy-bound).

The framework-free step statistics (`BatchShape`, `inspect_batch_shape`,
`StepMixTracker`, `MKABCalibrator`, ...) live in `stats` and are re-exported
here.
"""

from __future__ import annotations

from collections.abc import Container, Mapping, Sequence

from ..errors import MegakernelConfigError
from ..profiles.default import chip_budget
from .contract import EligibilityResult, StepDecision
from .policy import BucketPolicy
from .stats import (
    BatchShape,
    MKABCalibrator,
    StepMixTracker,
    check_mrope_uniform,
    check_qsl_uniform,
    inspect_batch_shape,
)

_STARTUP_FLAG_RULES = (
    ("unaligned_cache_layout", "requires unified KV and recurrent cache layout"),
    (
        "multimodal_inputs",
        "requires text-only token IDs (multimodal inputs unsupported)",
    ),
    ("lora_enabled", "does not support LoRA adapters"),
    ("pipeline_parallel", "requires single-stage pipeline execution"),
    ("context_parallel", "does not support context parallelism"),
    ("dp_lockstep", "does not support data-parallel lockstep"),
    ("aux_hidden_states", "does not support auxiliary hidden-state capture"),
)


def check_post_mesh_eligibility(
    *,
    tp_size: int,
    mesh_size: int | None = None,
    device_kind: str | None = None,
    bucket_policy: BucketPolicy | None = None,
) -> EligibilityResult:
    """Validate post-mesh device count, TPU chip budget, and bucket policy."""
    if mesh_size is not None and int(mesh_size) != int(tp_size):
        msg = (
            f"Megakernel TP-rank mesh has {mesh_size} devices, "
            f"expected {tp_size}; skipping."
        )
        return EligibilityResult(False, "mesh_size_mismatch", msg)
    if device_kind is not None:
        try:
            chip_budget(str(device_kind))
        except MegakernelConfigError as exc:
            msg = f"Megakernel disabled: {exc}; skipping."
            return EligibilityResult(False, "unsupported_tpu_chip", msg)
    if (
        bucket_policy is not None
        and not bucket_policy.buckets
        and not bucket_policy.verify_buckets
    ):
        msg = "Megakernel disabled (both bucket lists empty)."
        return EligibilityResult(False, "empty_buckets", msg, "info")
    return EligibilityResult(True, "ok", "")


def check_startup_eligibility(
    *,
    enabled: bool = True,
    hf_config_result: EligibilityResult | None = None,
    num_layers: int,
    expected_num_layers: int,
    unified_cache_layout: bool = True,
    multimodal_inputs: bool = False,
    lora_enabled: bool = False,
    pipeline_parallel: bool = False,
    context_parallel: bool = False,
    dp_lockstep: bool = False,
    aux_hidden_states: bool = False,
    tp_size: int | None = None,
    supported_tp_sizes: Sequence[int] = (),
    mesh_size: int | None = None,
    device_kind: str | None = None,
    bucket_policy: BucketPolicy | None = None,
) -> EligibilityResult:
    """Validate framework-neutral startup compatibility for the megakernel runtime.

    `tp_size=None` skips the TP-size check (a runtime that enforces the TP
    sizes itself, e.g. from a spec's `tp_sizes`).
    """
    if not enabled:
        return EligibilityResult(False, "disabled", "", "debug")
    if hf_config_result is not None and not hf_config_result.eligible:
        return hf_config_result
    if int(num_layers) != int(expected_num_layers):
        return EligibilityResult(False, "layer_count_mismatch", "", "debug")
    bad_flags = (
        not unified_cache_layout,
        multimodal_inputs,
        lora_enabled,
        pipeline_parallel,
        context_parallel,
        dp_lockstep,
        aux_hidden_states,
    )
    for is_bad, (code, detail) in zip(bad_flags, _STARTUP_FLAG_RULES, strict=True):
        if bool(is_bad):
            return EligibilityResult(False, code, f"Megakernel {detail}; skipping.")
    if tp_size is None:
        return check_post_mesh_eligibility(
            tp_size=0, device_kind=device_kind, bucket_policy=bucket_policy
        )
    allowed_tp = tuple(int(x) for x in supported_tp_sizes)
    if not allowed_tp:
        msg = (
            "Megakernel model does not declare any supported TPU mesh sizes; skipping."
        )
        return EligibilityResult(False, "unsupported_tp_size", msg)
    if int(tp_size) not in allowed_tp:
        msg = (
            f"Megakernel requires {allowed_tp[0]} TPU devices in mesh, "
            f"got {tp_size}; skipping."
            if len(allowed_tp) == 1
            else f"Megakernel requires TPU mesh size in {allowed_tp}, "
            f"got {tp_size}; skipping."
        )
        return EligibilityResult(False, "unsupported_tp_size", msg)
    return check_post_mesh_eligibility(
        tp_size=tp_size,
        mesh_size=mesh_size,
        device_kind=device_kind,
        bucket_policy=bucket_policy,
    )


def check_step_eligibility(
    batch_shape: BatchShape,
    bucket_policy: BucketPolicy,
    compiled_buckets: Container[tuple[int, int]],
    bucket_out_rows: Mapping[tuple[int, int], int],
    *,
    input_rows: int,
    mrope_ok: bool = True,
) -> StepDecision:
    """Decide whether a forward step with `batch_shape` can run on a bucket."""
    if not batch_shape.eligible or batch_shape.num_reqs <= 0:
        return StepDecision(
            False,
            batch_shape.reason,
            batch_shape.num_reqs,
            batch_shape.tokens_per_seq,
        )
    num_reqs, T = batch_shape.num_reqs, batch_shape.tokens_per_seq
    if not mrope_ok:
        return StepDecision(False, "stock_mrope", num_reqs, T)
    if T not in (1, 4):
        return StepDecision(False, f"stock_q{T}", num_reqs, T)
    b_list = bucket_policy.verify_buckets if T == 4 else bucket_policy.buckets
    max_b = bucket_policy.max_verify_bucket if T == 4 else bucket_policy.max_bucket
    active_rows = num_reqs * T
    if not b_list or active_rows > max_b:
        return StepDecision(False, f"stock_t{T}_n{num_reqs}", num_reqs, T)
    bkt = bucket_policy.select(active_rows, T)
    if (T, bkt) not in compiled_buckets:
        return StepDecision(False, f"stock_t{T}_nobkt", num_reqs, T, bkt)
    if bucket_policy.token_paddings and bucket_out_rows.get(
        (T, bkt), input_rows
    ) != int(input_rows):
        return StepDecision(False, f"stock_t{T}_rows{bkt}", num_reqs, T, bkt)
    return StepDecision(True, f"mk_t{T}_b{bkt}", num_reqs, T, bkt)


__all__ = [
    "BatchShape",
    "MKABCalibrator",
    "StepMixTracker",
    "check_mrope_uniform",
    "check_post_mesh_eligibility",
    "check_qsl_uniform",
    "check_startup_eligibility",
    "check_step_eligibility",
    "inspect_batch_shape",
]
