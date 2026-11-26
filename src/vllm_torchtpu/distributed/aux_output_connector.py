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
"""vLLM's auxiliary-output worker connector, fed by a capturer of our own."""

import numpy as np
import torch
from vllm.distributed.aux_output_connector.worker import AuxOutputWorkerConnector
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
    bind_routed_experts_capturer,
)


def use_capturer(
    connector: AuxOutputWorkerConnector,
    model: torch.nn.Module,
    capturer: RoutedExpertsCapturer,
) -> None:
    """Points vLLM's worker `connector` and every MoE layer of `model` at
    `capturer`, such as one that keeps every forward of a TPU step, instead
    of the capturer vLLM built and bound.

    `capturer` must have the row shape and dtype of the capturer vLLM built,
    as one from the same `vllm_config` does.
    """
    # AuxOutputWorkerConnector cannot be given a capturer, so replace the one
    # it built. It reads `_capturer` only to size its rows and take snapshots.
    built = connector._capturer
    row_format = (capturer.shape_per_token, capturer.output_dtype_name)
    built_row_format = (built.shape_per_token, built.output_dtype_name)
    if row_format != built_row_format:
        raise ValueError(
            f"capturer row format {row_format} differs from the "
            f"{built_row_format} vLLM's connector was sized for"
        )
    connector._capturer = capturer
    # Rebinds every layer vLLM bound, so no hook keeps vLLM's capturer and its
    # device table alive.
    bind_routed_experts_capturer(model, capturer)


def num_rejected_rows(num_drafts: np.ndarray, num_sampled: np.ndarray) -> np.ndarray:
    """Returns how many of each request's rows came from rejected drafts.

    Every draft gets a row, because experts are captured before drafts are
    verified. vLLM's connector drops the rejected ones given this count,
    which must match the scheduler's rollback
    (`Scheduler.update_from_output`): drafts minus the accepted
    `num_sampled - 1`, and zero without drafts or without a sampled token.

    Args:
        num_drafts: [num_reqs] drafts each request verified this step.
        num_sampled: [num_reqs] tokens each request sampled this step.

    Returns:
        [num_reqs] row counts.
    """
    rejected = num_drafts - (num_sampled - 1)
    return np.where((num_drafts > 0) & (num_sampled > 0), rejected, 0)
