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
"""The weight formats the kernel takes."""

import enum
from typing import NamedTuple

import jax.numpy as jnp
from jax.typing import DTypeLike

from vllm_torchtpu.kernels.fused_moe.v3.layout import FP4_PER_WORD, SUBLANES
from vllm_torchtpu.kernels.fused_moe.v3.rowquant import FP8

# Storage only: there is no four-bit matrix unit, so each block is upcast
# to fp8 in VMEM before it is contracted.
FP4 = jnp.float4_e2m1fn
INT4 = jnp.int4
PACKED_DTYPES = (
    FP4,
    INT4,
)  # four-bit weights: packed in 32-bit words, widened to fp8 in the kernel
# Contraction rows one four-bit weight scale covers, by default.
WEIGHT_BLOCK_DEFAULT = 512


class WeightFormat(str, enum.Enum):
    """A weight format's name, as written in logs, cache keys and error
    messages."""

    FP8 = "fp8"
    FP8_BLOCK = "fp8_block"
    FP4 = "fp4"
    INT4 = "int4"
    BF16 = "bf16"

    def __str__(self):
        return self.value

    def __repr__(self):
        return repr(self.value)


class WeightForm(NamedTuple):
    """A weight format: the element type of the expert weights, and the
    layout of the scales the caller supplies, one f32 per output channel
    ("per_channel"), one per (contraction block, output channel)
    ("per_contraction_block": four-bit weights, fp4 or int4 with one
    scale per block of the contraction, and fp8 weights in the block-scaled
    form the DeepSeek-family checkpoints ship in), or none for unscaled
    bf16 weights. Token rows are fp8 with one f32 scale per
    row on every format, and so are the rows the experts return."""

    name: WeightFormat
    weight_dtype: DTypeLike
    scale_layout: str

    @property
    def block_scaled(self):
        """Whether the scales are per contraction block."""
        return self.scale_layout == "per_contraction_block"

    @property
    def packed(self):
        """Four-bit weights (fp4 or int4): stored packed in 32-bit words
        and widened to fp8 in the kernel (an int4 value is exact in fp8)."""
        return any(jnp.dtype(self.weight_dtype) == jnp.dtype(d) for d in PACKED_DTYPES)

    @property
    def has_scales(self):
        """Whether the weights carry scales at all."""
        return self.scale_layout != "none"


WEIGHT_FORMS = {
    WeightFormat.FP8: WeightForm(WeightFormat.FP8, FP8, "per_channel"),
    WeightFormat.FP8_BLOCK: WeightForm(
        WeightFormat.FP8_BLOCK, FP8, "per_contraction_block"
    ),
    WeightFormat.FP4: WeightForm(WeightFormat.FP4, FP4, "per_contraction_block"),
    WeightFormat.INT4: WeightForm(WeightFormat.INT4, INT4, "per_contraction_block"),
    WeightFormat.BF16: WeightForm(WeightFormat.BF16, jnp.bfloat16, "none"),
}
WEIGHT_FORMAT_NAMES = tuple(WEIGHT_FORMS)
# Rows of packed four-bit weights one addressable block is a whole number
# of: FP4_PER_WORD values to a 32-bit word, and those words tile to
# layout.SUBLANES sublanes.
PACKED_ROW_TILE = FP4_PER_WORD * SUBLANES


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
