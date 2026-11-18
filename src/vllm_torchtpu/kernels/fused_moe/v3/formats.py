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
"""Weight format descriptors for Fused EP MoE v3."""

import enum
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .rowquant import FP8


class WeightFormat(str, enum.Enum):
    """A weight format's name, as written in logs, cache keys and error
    messages."""

    FP8 = "fp8"
    BF16 = "bf16"

    def __str__(self):
        return self.value

    def __repr__(self):
        return repr(self.value)


class WeightForm(NamedTuple):
    """A weight format: the element type of the expert weights, and the
    layout of the scales the caller supplies: one f32 per output channel
    ("per_channel"), or none for unscaled bf16 weights. Config selects the
    result row format within the supported weight formats."""

    name: WeightFormat
    weight_dtype: jax.typing.DTypeLike
    scale_layout: str

    @property
    def has_scales(self):
        """Whether the weights carry scales at all."""
        return self.scale_layout != "none"


WEIGHT_FORMS = {
    WeightFormat.FP8: WeightForm(WeightFormat.FP8, FP8, "per_channel"),
    WeightFormat.BF16: WeightForm(WeightFormat.BF16, jnp.bfloat16, "none"),
}
WEIGHT_FORMAT_NAMES = tuple(WEIGHT_FORMS)


def weight_form(weight_format):
    """The record for a format. Raises ValueError for one outside the
    table."""
    try:
        return WEIGHT_FORMS[WeightFormat(weight_format)]
    except ValueError:
        raise ValueError(
            f"the fused EP MoE kernel takes weight formats "
            f"{WEIGHT_FORMAT_NAMES}. Got {weight_format!r}"
        ) from None


def weight_format_of_dtype(dtype):
    """The format carrying this weight element type, or None."""
    for weight_format, form in WEIGHT_FORMS.items():
        if jnp.dtype(form.weight_dtype) == jnp.dtype(dtype):
            return weight_format
    return None
