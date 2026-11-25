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
"""Startup, post-mesh, and per-step eligibility rules.

No torch or vLLM imports. Imports `jax` lazily only through the TPU chip table
(`profiles.default`), executing no device code: the chip check takes
`device_kind` as a string argument. Step statistics (`BatchShape`,
`inspect_batch_shape`, `StepMixTracker`, `MKABCalibrator`) live in `stats` and
are re-exported here.
"""

from __future__ import annotations

from collections.abc import Container, Sequence
from typing import TYPE_CHECKING

from ..errors import MegakernelConfigError
from ..profiles.default import chip_budget
from .constants import STARTUP_FLAG_RULES
from .contract import (
    MESH_SIZE_MISMATCH,
    UNSUPPORTED_TP_SIZE,
    UNSUPPORTED_TPU_CHIP,
    EligibilityResult,
    StepDecision,
)
from .stats import (
    BatchShape,
    MKABCalibrator,
    StepMixTracker,
    check_mrope_uniform,
    check_query_start_loc_uniform,
    inspect_batch_shape,
)

if TYPE_CHECKING:
    from .policy import BucketPolicy


def check_post_mesh_eligibility(
    *,
    tp_size: int,
    mesh_size: int | None = None,
    device_kind: str | None = None,
    bucket_policy: BucketPolicy | None = None,
) -> EligibilityResult:
    """Validate post-mesh device count, TPU chip budget, and bucket policy.

    Args:
        tp_size: Expected tensor-parallel device count.
        mesh_size: Actual JAX mesh device count; `None` skips mesh size check.
        device_kind: TPU device kind string (e.g. `'TPU7x'`); `None` skips chip
            budget lookup.
        bucket_policy: Resolved `BucketPolicy`; `None` skips empty-bucket check.

    Returns:
        `EligibilityResult` indicating whether post-mesh invariants hold.
    """
    if mesh_size is not None and mesh_size != tp_size:
        msg = (
            f"Megakernel TP-rank mesh has {mesh_size} devices, "
            f"expected {tp_size}; skipping."
        )
        return EligibilityResult(False, MESH_SIZE_MISMATCH, msg)
    if device_kind is not None:
        try:
            chip_budget(device_kind)
        except MegakernelConfigError as exc:
            msg = f"Megakernel disabled: {exc}; skipping."
            return EligibilityResult(False, UNSUPPORTED_TPU_CHIP, msg)
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

    Args:
        enabled: Whether megakernel execution is enabled in config.
        hf_config_result: Pre-computed HuggingFace config check; `None` skips.
        num_layers: Actual model layer count.
        expected_num_layers: Layer count required by the megakernel plugin.
        unified_cache_layout: Whether KV and recurrent caches share unified layout.
        multimodal_inputs: Whether multimodal inputs are enabled.
        lora_enabled: Whether LoRA adapters are active.
        pipeline_parallel: Whether multi-stage pipeline parallelism is active.
        context_parallel: Whether prefill context parallelism is active.
        dp_lockstep: Whether data-parallel lockstep mode is active.
        aux_hidden_states: Whether auxiliary hidden-state capture is requested.
        tp_size: Configured TP world size; `None` skips TP-size and mesh-size
            checks (used when a caller enforces mesh topology separately).
        supported_tp_sizes: Allowed TP world sizes when `tp_size` is not `None`.
        mesh_size: Actual JAX mesh size; requires `tp_size` when not `None`.
        device_kind: TPU device kind string; `None` skips chip budget check.
        bucket_policy: Resolved `BucketPolicy`; `None` skips empty-bucket check.

    Returns:
        `EligibilityResult` describing startup eligibility or decline reason.

    Raises:
        ValueError: If `mesh_size` is provided while `tp_size` is `None`.
    """
    if tp_size is None and mesh_size is not None:
        msg = (
            f"check_startup_eligibility: mesh_size={mesh_size} needs tp_size to "
            "check it against (tp_size=None skips the mesh-size check)"
        )
        raise ValueError(msg)
    if not enabled:
        return EligibilityResult(False, "disabled", "", "debug")
    if hf_config_result is not None and not hf_config_result.eligible:
        return hf_config_result
    if num_layers != expected_num_layers:
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
    for is_bad, (code, detail) in zip(bad_flags, STARTUP_FLAG_RULES, strict=True):
        if is_bad:
            return EligibilityResult(False, code, f"Megakernel {detail}; skipping.")
    if tp_size is None:
        return check_post_mesh_eligibility(
            tp_size=0,
            device_kind=device_kind,
            bucket_policy=bucket_policy,
        )
    if not supported_tp_sizes:
        msg = (
            "Megakernel model does not declare any supported TPU mesh sizes; skipping."
        )
        return EligibilityResult(False, UNSUPPORTED_TP_SIZE, msg)
    if tp_size not in supported_tp_sizes:
        msg = (
            f"Megakernel requires {supported_tp_sizes[0]} TPU devices in mesh, "
            f"got {tp_size}; skipping."
            if len(supported_tp_sizes) == 1
            else f"Megakernel requires TPU mesh size in {supported_tp_sizes}, "
            f"got {tp_size}; skipping."
        )
        return EligibilityResult(False, UNSUPPORTED_TP_SIZE, msg)
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
    *,
    input_rows: int,
    mrope_ok: bool = True,
) -> StepDecision:
    """Decide whether a forward step with `batch_shape` can run on a bucket.

    The bucket policy serves `T=1` decode and `T in bucket_policy.verify_tokens`
    verify steps; any other uniform step width declines as `stock_q{T}`.

    Args:
        batch_shape: Uniformity and token-count inspection for the current step.
        bucket_policy: Resolved decode and verify bucket policy.
        compiled_buckets: Set/container of compiled `(tokens_per_seq, bucket)` keys.
        input_rows: Actual padded row count of the runner's input tensor.
        mrope_ok: Whether M-RoPE positions satisfy uniformity invariants.

    Returns:
        `StepDecision` naming the selected bucket or stock fallback reason.
    """
    if not batch_shape.eligible or batch_shape.num_reqs <= 0:
        return StepDecision(
            eligible=False,
            reason=batch_shape.reason,
            num_reqs=batch_shape.num_reqs,
            tokens_per_seq=batch_shape.tokens_per_req,
        )
    num_reqs = batch_shape.num_reqs
    tokens_per_req = batch_shape.tokens_per_req
    if tokens_per_req != 1 and tokens_per_req not in bucket_policy.verify_tokens:
        return StepDecision(
            eligible=False,
            reason=f"stock_q{tokens_per_req}",
            num_reqs=num_reqs,
            tokens_per_seq=tokens_per_req,
        )
    if not mrope_ok:
        return StepDecision(
            eligible=False,
            reason="stock_mrope",
            num_reqs=num_reqs,
            tokens_per_seq=tokens_per_req,
        )
    valid_buckets = bucket_policy.verify_buckets_for(tokens_per_req)
    max_bucket_rows = valid_buckets[-1] if valid_buckets else 0
    active_rows = num_reqs * tokens_per_req
    if not valid_buckets or active_rows > max_bucket_rows:
        return StepDecision(
            eligible=False,
            reason=f"stock_t{tokens_per_req}_n{num_reqs}",
            num_reqs=num_reqs,
            tokens_per_seq=tokens_per_req,
        )
    bucket_rows = bucket_policy.select(active_rows, tokens_per_req)
    if (tokens_per_req, bucket_rows) not in compiled_buckets:
        return StepDecision(
            eligible=False,
            reason=f"stock_t{tokens_per_req}_nobkt",
            num_reqs=num_reqs,
            tokens_per_seq=tokens_per_req,
            bucket_t=bucket_rows,
        )
    if (
        bucket_policy.token_paddings
        and bucket_policy.out_rows(bucket_rows) != input_rows
    ):
        return StepDecision(
            eligible=False,
            reason=f"stock_t{tokens_per_req}_rows{bucket_rows}",
            num_reqs=num_reqs,
            tokens_per_seq=tokens_per_req,
            bucket_t=bucket_rows,
        )
    return StepDecision(
        eligible=True,
        reason=f"mk_t{tokens_per_req}_b{bucket_rows}",
        num_reqs=num_reqs,
        tokens_per_seq=tokens_per_req,
        bucket_t=bucket_rows,
    )


__all__ = [
    "BatchShape",
    "MKABCalibrator",
    "StepMixTracker",
    "check_mrope_uniform",
    "check_post_mesh_eligibility",
    "check_query_start_loc_uniform",
    "check_startup_eligibility",
    "check_step_eligibility",
    "inspect_batch_shape",
]
