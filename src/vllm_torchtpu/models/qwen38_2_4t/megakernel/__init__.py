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
"""Qwen3.8-2.4T TP32xDCP8 NVFP4 megakernel model folder (`qwen38_2_4t`).

Selected explicitly via `plugin=vllm_torchtpu.models.qwen38_2_4t.megakernel` in
`VLLM_TPU_MEGAKERNEL` or `additional_config["megakernel"]["plugin"]`.
Stdlib-only at import; `.adapter:create` and `.adapter:layout_request` are resolved
lazily by the runtime.
"""

from __future__ import annotations

from vllm_torchtpu.megakernel.core.runtime.contract import (
    DOMAIN_TP,
    PAR_DCP,
    PAR_EP_IN_TP,
    PAR_TP,
    TpuMonoSpec,
)

from .constants import (
    ERR_BACKBONE_NOT_FOUND,
    ERR_EMPTY_BUCKETS,
    ERR_POOL_MISMATCH,
    ERR_POOL_UNKNOWN,
    ERR_UNALIGNED_CACHE,
    ERR_UNSUPPORTED_ARCH,
    ERR_UNSUPPORTED_DCP,
    ERR_UNSUPPORTED_GEOM,
    ERR_UNSUPPORTED_QUANT,
    ERR_UNSUPPORTED_TP,
)

SPEC = TpuMonoSpec(
    name="qwen38_2_4t",
    version="1",
    create=".adapter:create",
    chips=("TPU7x",),
    # TP=32 is checked in `adapter.create` so TP mismatches vote across ranks
    # via `agree_on_decline` and produce a rank-uniform `DeclinedDriver`.
    tp_sizes=(),
    parallelism=frozenset({PAR_TP, PAR_DCP, PAR_EP_IN_TP}),
    features=frozenset(),
    numerics="approx",
    collective_domain=DOMAIN_TP,
    layout_request=".adapter:layout_request",
    decline_codes=(
        ERR_UNSUPPORTED_ARCH,
        ERR_UNSUPPORTED_GEOM,
        ERR_UNSUPPORTED_TP,
        ERR_UNSUPPORTED_DCP,
        ERR_BACKBONE_NOT_FOUND,
        ERR_POOL_UNKNOWN,
        ERR_POOL_MISMATCH,
        ERR_UNALIGNED_CACHE,
        ERR_UNSUPPORTED_QUANT,
        ERR_EMPTY_BUCKETS,
    ),
)

__all__ = ["SPEC"]
