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
"""Reduced-precision attention modes run on the target model only, never on a
speculative draft.

The batched RPA kernel can run its QK and/or PV matmuls in fp8 (``native_fp8``
for the decode kernel, ``native_fp8_prefill`` for the prefill and mixed
kernels) and its decode softmax in base 2 (``decode_exp2``). All three trade
the last bits of precision for speed, and all are off unless a model turns
them on for its attention layers (``PallasAttentionBackendImpl.attention_modes``).
A draft model must run at the precision it ships with, so its attention keeps
the full-precision path whatever the target uses.

The runner marks each draft attention layer's impl
(``PallasAttentionBackendImpl.native_fp8_draft = True``) before it builds that
layer's kernels. The kernel builders pass a layer's modes to the batched and
DCP kernel entries only when the layer is not a draft. The modes are part of
every kernel registry key, so a draft layer never shares a target layer's op.
With every mode off nothing changes: no entry binds anything new and every key
resolves to the op it did before.
"""

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class AttentionModes:
    """Reduced-precision modes of a layer's batched RPA kernels ("" = off)."""

    native_fp8: str = ""
    native_fp8_prefill: str = ""
    decode_exp2: bool = False

    def any(self) -> bool:
        return bool(self.native_fp8 or self.native_fp8_prefill or self.decode_exp2)


OFF = AttentionModes()


def effective(modes: AttentionModes, is_draft: bool) -> AttentionModes:
    """The modes a layer's kernels run: the target's, or off for a draft."""
    return OFF if is_draft else modes


def kernel_kwargs(modes: AttentionModes) -> dict[str, Any]:
    """The batched RPA wrapper's keyword arguments for `modes` (only the on ones)."""
    kwargs: dict[str, Any] = {}
    if modes.native_fp8:
        kwargs["native_fp8"] = modes.native_fp8
    if modes.native_fp8_prefill:
        kwargs["native_fp8_prefill"] = modes.native_fp8_prefill
    if modes.decode_exp2:
        kwargs["decode_exp2"] = True
    return kwargs


def scope_line(modes: AttentionModes, num_target: int, num_draft: int) -> str:
    """The startup line that states the scope, for the compliance log check."""
    return (
        f"Native FP8 attention: target decode={modes.native_fp8 or 'off'} "
        f"prefill={modes.native_fp8_prefill or 'off'} "
        f"exp2={'on' if modes.decode_exp2 else 'off'}, draft=off | "
        f"attention layers: {num_target} target, {num_draft} draft"
    )
