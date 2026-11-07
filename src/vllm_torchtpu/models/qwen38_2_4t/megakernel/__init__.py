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
Stdlib-only at import; `.adapter:create`, `.adapter:layout_request`, and
`.weights:load_tensor` are resolved lazily by the runtime.
"""

from __future__ import annotations

from vllm_torchtpu.megakernel.runtime.contract import (
    DOMAIN_TP_X_DCP,
    PAR_DCP,
    PAR_EP_IN_TP,
    PAR_TP,
    TpuMonoSpec,
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
    collective_domain=DOMAIN_TP_X_DCP,
    layout_request=".adapter:layout_request",
    load_tensor=".weights:load_tensor",
    decline_codes=(
        "unsupported_arch",
        "unsupported_geometry",
        "unsupported_tp_size",
        "unsupported_dcp_size",
        "backbone_not_found",
        "unaligned_cache_layout",
        "unsupported_quant",
        "empty_buckets",
    ),
)

__all__ = ["SPEC"]
