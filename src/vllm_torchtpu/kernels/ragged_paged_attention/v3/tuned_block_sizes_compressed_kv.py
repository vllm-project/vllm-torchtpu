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
"""Tuned VREG dequant `bkv_csz` caps for compressed-KV ragged paged attention."""

from vllm_torchtpu.kernels.pallas_util import align_to, get_tpu_version
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Calibrated `(tpu_version, head_dim_aligned_128, page_size) -> max_bkv_csz`
# entries.
#
# TPU v7 (`tpu7x`) was hardware-profiled on Qwen3.5-397B-A17B (`head_dim=256`,
# `page_size=64`) and `head_dim=128` (`page_size in {16, 32, 64, 128, 256}`):
#   - With `page_size >= 64` and `head_dim <= 256`, dequantizing up to 4 pages
#     (256 tokens) per inner step and running MXU `Q @ K^T` / `P @ V` at
#     `bkv_csz=256` vs `128` reduced mean TTFT by 42.8%, median ITL by 40.5%,
#     and lifted total throughput by 32.0% with zero VREG spills.
#   - With `page_size in (16, 32)` and `head_dim <= 256`, `128` keeps per-step
#     VREG page gather loops compact (`<= 8` pages) without spilling.
TUNED_MAX_BKV_CSZ: dict[tuple[int, int, int], int] = {
    (7, 128, 16): 128,
    (7, 128, 32): 128,
    (7, 128, 64): 256,
    (7, 128, 128): 256,
    (7, 128, 256): 256,
    (7, 256, 16): 128,
    (7, 256, 32): 128,
    (7, 256, 64): 256,
    (7, 256, 128): 256,
    (7, 256, 256): 256,
}


def get_tuned_max_bkv_csz(
    head_dim: int,
    page_size: int,
    *,
    tpu_version: int | None = None,
) -> int:
    """Return the `bkv_csz` cap for VREG page dequantization.

    Looks up `(tpu_version, align_to(head_dim, 128), page_size)` in
    `TUNED_MAX_BKV_CSZ`. On a table miss (e.g. TPU v5/v6, `head_dim > 256`, or
    larger `page_size`), logs once with the lookup key and returns a
    conservative analytic VREG bound (`256` on TPU v7 when `head_dim <= 256` and
    `page_size >= 64`, `128` for `head_dim > 256` when `page_size >= 64` or on
    TPU v5/v6 when `head_dim <= 128 or page_size >= 64`, and `64` otherwise).
    """
    if tpu_version is None:
        tpu_version = get_tpu_version()
    aligned_hd = align_to(head_dim, 128)
    key = (tpu_version, aligned_hd, page_size)
    tuned = TUNED_MAX_BKV_CSZ.get(key)
    if tuned is not None:
        return tuned

    if tpu_version >= 7 and aligned_hd <= 256 and page_size >= 64:
        fallback = 256
    elif tpu_version >= 7:
        fallback = 128 if (aligned_hd <= 256 or page_size >= 64) else 64
    else:
        fallback = 128 if (aligned_hd <= 128 or page_size >= 64) else 64

    logger.warning_once(
        "Uncalibrated compressed-KV max_bkv_csz key "
        "(tpu_version=%d, head_dim=%d, page_size=%d); using analytic cap %d.",
        tpu_version,
        aligned_hd,
        page_size,
        fallback,
    )
    return fallback
