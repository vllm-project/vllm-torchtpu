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
"""Admission rules of the fused MLA DSpark draft kernel."""

from __future__ import annotations

import importlib
from typing import Any

from vllm_torchtpu.models.kimi_k3.megakernel.kernels import dspark_mla_kernel
from vllm_torchtpu.models.kimi_k3.megakernel.kernels.dspark_mla import MlaDraftConfig

pytest: Any = importlib.import_module("pytest")
PAGES = 16


@pytest.mark.parametrize(
    ("geometry", "error"),
    [
        ({"requests": 1}, None),
        ({"requests": 8}, None),
        ({"requests": 16, "rows": 4, "block": 4}, None),
        ({"requests": 1, "rows": 6}, "4 or 8 rows"),
        ({"requests": 1, "block": 1}, "block must be"),
        ({"requests": 9}, "up to 8 requests"),
        ({"requests": 3, "attention_dp": 2}, "attention DP2"),
    ],
)
def test_fused_draft_admission(geometry: dict[str, int], error: str | None) -> None:
    """Supported geometries build a kernel; others raise naming the limit."""
    if error is None:
        kernel = dspark_mla_kernel.fused_draft_local(
            MlaDraftConfig(), pages=PAGES, **geometry
        )
        assert callable(kernel)
    else:
        with pytest.raises(ValueError, match=error):
            dspark_mla_kernel.fused_draft_local(
                MlaDraftConfig(), pages=PAGES, **geometry
            )
