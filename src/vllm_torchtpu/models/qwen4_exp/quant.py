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
"""W8A8 FP8 for the Qwen4Exp attention projections, quantized at load time.

The FP8 checkpoint keeps the attention projections in BF16
(``modules_to_not_convert``). On chips whose MXU multiplies FP8 natively, a
projection switched to ``Qwen4ExpFp8LinearMethod`` loads that BF16 weight
unchanged, quantizes it once in ``process_weights_after_loading`` and then
runs every step as one FP8 x FP8 ``dot_general``:

- Weights: per-output-channel ``float8_e4m3fn``, stored in the (k, n) layout
  the contraction reads, so a step does no weight transpose or relayout. A
  per-channel scale factors out of the K sum and is applied once to the f32
  result; K-blocked scales would need one partial product per block.
- Activations: dynamic per-token ``float8_e4m3fn`` (scale = row absmax / 448),
  so both MXU operands are FP8 and the matmul runs at the FP8 rate (the MXU
  rate follows the LHS dtype; a BF16 LHS against an FP8 weight runs at the
  BF16 rate).

Only the FP8 weight is kept. Small token buckets (decode and speculative
verify, at most ``w8a16_max_rows`` rows) are bound by reading the weight, not
by the MXU, so they skip the activation quantization and multiply BF16
activations by the same FP8 weight (W8A16): as fast as W8A8 there, with BF16
activation precision, and still half the weight bytes of BF16. The choice is
static per token bucket.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from jax.experimental.pallas import tpu as pltpu
from torch import nn
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs

from vllm_torchtpu.layers.adapter.linear_common import KEEP_VLLM_LAYOUT_ATTR
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.qwen4_exp.ops.fp8_linear import (
    FP8_MAX,
    FP8_MIN_AMAX,
    _call_fp8_linear,
    _get_fp8_linear_op,
)
from vllm_torchtpu.utils import fresh_buffer, synchronize_tensors

logger = init_logger(__name__)

FP8_DTYPE = torch.float8_e4m3fn

# Chips whose native FP8 MXU path this method was validated on; every other
# chip keeps the BF16 projections.
_FP8_MXU_CHIPS = frozenset(
    {
        pltpu.ChipVersion.TPU_7,
        pltpu.ChipVersion.TPU_7X,
        pltpu.ChipVersion.TPU_8I,
    }
)

# Token buckets of at most this many rows run W8A16; larger ones run W8A8.
# Measured crossover: W8A8 pulls ahead once a bucket has more rows than this.
DEFAULT_W8A16_MAX_ROWS = 64


def _chip_version() -> pltpu.ChipVersion | None:
    """This worker's TPU generation, or None off TPU."""
    try:
        return pltpu.get_tpu_info().chip_version
    except ValueError:  # Non-TPU host.
        return None


def has_native_fp8_matmul() -> bool:
    """Whether this worker's TPU runs FP8 x FP8 matmuls on a validated MXU path.

    Queries the device, so call it from a worker that owns the TPU.
    """
    return _chip_version() in _FP8_MXU_CHIPS


def quantize_weight_per_channel(
    weight: torch.Tensor, transpose: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantizes an [n_out, n_in] weight to FP8 with one scale per output row.

    Returns the FP8 weight, transposed to (k, n) = [n_in, n_out] unless
    ``transpose`` is False, and its f32 scale [n_out]. Rows whose absmax is
    below ``FP8_MIN_AMAX`` (including all-zero rows) use that floor, so every
    scale is a normal float32 and the weight stays finite.
    """
    if weight.dim() != 2:
        raise ValueError(f"Expected a 2-D weight, got shape {tuple(weight.shape)}")
    w = weight.to(torch.float32)
    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=FP8_MIN_AMAX) / FP8_MAX
    w_q = (w / scale).clamp(-FP8_MAX, FP8_MAX)
    if transpose:
        w_q = w_q.t()
    # Transpose before the FP8 cast: the cast is the only FP8 op needed, and
    # as the last op it writes a dense buffer of the stored shape.
    return w_q.contiguous().to(FP8_DTYPE), scale.reshape(-1)


class Qwen4ExpFp8LinearMethod(LinearMethodBase):
    """Loads a BF16 projection and runs it as W8A8 FP8 (see module docstring).

    Weight creation and loading are those of ``UnquantizedLinearMethod``, so
    checkpoint loaders and their BF16 checks see the same parameter.
    """

    def __init__(self, w8a16_max_rows: int = DEFAULT_W8A16_MAX_ROWS) -> None:
        self.w8a16_max_rows = w8a16_max_rows
        # Built at model construction: the forward is traced by torch.compile,
        # which cannot enter the lock that guards building the op.
        self._op = _get_fp8_linear_op()

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        UnquantizedLinearMethod().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        weight = cast(torch.Tensor, layer.weight)
        if weight.dtype == FP8_DTYPE:
            return  # Reload paths re-enter this hook.
        if weight.dtype != torch.bfloat16:
            raise ValueError(
                f"FP8 projection {getattr(layer, 'prefix', '')!r} expects a "
                f"BF16 weight, got {weight.dtype}"
            )
        # A consumer that reads `.weight` raw expects vLLM's [n_out, n_in];
        # keep that layout and flip the view in `apply` instead.
        keep_layout = getattr(layer, KEEP_VLLM_LAYOUT_ATTR, False)
        w_q, w_scale = quantize_weight_per_channel(
            weight.data, transpose=not keep_layout
        )
        # As a requantization source, the BF16 parameter is replaced by one
        # owning the FP8 buffer (keeping `weight_loader`) instead of having
        # the FP8 bytes copied into its twice-as-large storage, which would
        # stay allocated.
        set_weight_attrs(weight, {"requantization_source": True})
        del weight
        # Own-shape buffers keep the step free of strided re-materialization.
        # The FP8 cast already wrote the weight into a fresh dense buffer;
        # the scale is still a view of its [n, 1] reduction.
        replace_parameter(layer, "weight", w_q)
        replace_parameter(layer, "weight_scale", fresh_buffer(w_scale))
        if w_q.device.type == "tpu":
            synchronize_tensors(
                [
                    cast(torch.Tensor, layer.weight),
                    cast(torch.Tensor, layer.weight_scale),
                ]
            )
        logger.info_once(
            "Qwen4Exp FP8 projections: BF16 weights quantized to per-channel "
            "float8_e4m3fn at load (W8A8 above %d rows).",
            self.w8a16_max_rows,
        )

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w_q = cast(torch.Tensor, layer.weight)
        if getattr(layer, KEEP_VLLM_LAYOUT_ATTR, False):
            w_q = w_q.transpose(0, 1)  # stored [n_out, n_in]; the op takes (k, n)
        w_scale = cast(torch.Tensor, layer.weight_scale)
        out = _call_fp8_linear(self._op, x, w_q, w_scale, self.w8a16_max_rows)
        if bias is not None:
            out = out + bias
        return out


def enable_fp8_projection(
    layer: LinearBase, w8a16_max_rows: int = DEFAULT_W8A16_MAX_ROWS
) -> bool:
    """Switches a BF16 projection to W8A8 FP8 when the chip supports it.

    Call after construction and before weight loading; the weight parameter
    is left untouched until ``process_weights_after_loading``. Returns whether
    the projection was switched; without native FP8 matmuls it stays BF16.
    """
    if not isinstance(layer.quant_method, UnquantizedLinearMethod):
        raise ValueError(
            f"FP8 projection {layer.prefix!r} needs an unquantized BF16 layer, "
            f"got {type(layer.quant_method).__name__}"
        )
    if not has_native_fp8_matmul():
        logger.info_once(
            "Qwen4Exp FP8 projections stay BF16: no validated native FP8 MXU "
            "path on %s.",
            _chip_version() or "this non-TPU host",
        )
        return False
    layer.quant_method = Qwen4ExpFp8LinearMethod(w8a16_max_rows)
    return True
