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
"""Two-batch overlap of the TP all-reduces on large MiniMax M3 steps
(TPU_MINIMAX_M3_PREFILL_OVERLAP).

On an 8192-token step each layer all-reduces two bf16 [T, 6144] partials (the
o_proj and the FFN output). XLA offloads them to the SparseCore, and the
TensorCore waits for each one: nothing else in the layer is independent of
it. A step of at least TPU_MINIMAX_M3_PREFILL_OVERLAP_MIN_TOKENS tokens
(a multiple of 32) therefore runs everything after attention on two token
halves A and B (`model.MiniMaxM3DecoderLayer.forward_overlap`):

    attn = attend(all tokens)                      # one MSA call, as before
    oA, oB = o_proj(attn[A]), o_proj(attn[B])      # partial sums
    AR(oA)   | -
    AR(oB)   | norm(A), FFN(A)
    AR(fA)   | norm(B), FFN(B)
    AR(fB)   | next layer: norm(A), qkv_proj(A)    # _QKV (default on)

so each half's all-reduce has the other half's work to hide behind. The
halves stay split across layers; the step's last layer joins them.

The half all-reduces are issued as torch functional all-reduces (AR "xla"):
an XLA all-reduce in the step's own computation, where the latency-hiding
scheduler can place the other half's work between its start and done. The
TP group's all-reduce ("group") is the Pallas all-reduce op when that is
armed; its psum sits in the op's called computation, and the TensorCore then
blocks in it (seen in the cg arm 21 profile), unless libtpu inlines the call.

Not bitwise against the plain path: half-size all-reduces and GEMMs sum in
another order. The MoE token-padding mask is read at the half's own rows
(`token_padding.row_offset`). Steps below the threshold, and every trace
outside the backbone warmup, keep the plain layer.
"""

import dataclasses

import torch
from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce

from vllm_torchtpu import envs
from vllm_torchtpu.compilation import shape_variants

AR_ROUTES = ("xla", "group")
# Half sizes stay a multiple of 16 rows (bf16 sublane tiles).
TOKEN_MULTIPLE = 32


@dataclasses.dataclass(frozen=True)
class OverlapConfig:
    """`min_tokens` > 0: steps of at least that many tokens (a multiple of
    TOKEN_MULTIPLE) overlap; `ar`: "xla" | "group" (module doc); `qkv`: the
    next layer's input norm and qkv_proj run per half too."""

    min_tokens: int = 0
    ar: str = "xla"
    qkv: bool = True

    @property
    def on(self) -> bool:
        return self.min_tokens > 0


OFF = OverlapConfig()


def config_from_env() -> OverlapConfig:
    if not envs.TPU_MINIMAX_M3_PREFILL_OVERLAP:
        return OFF
    min_tokens = envs.TPU_MINIMAX_M3_PREFILL_OVERLAP_MIN_TOKENS
    if min_tokens == 0:
        return OFF
    ar = envs.TPU_MINIMAX_M3_PREFILL_OVERLAP_AR
    if ar not in AR_ROUTES:
        raise ValueError(f"TPU_MINIMAX_M3_PREFILL_OVERLAP_AR={ar!r}: not {AR_ROUTES}")
    return OverlapConfig(min_tokens, ar, envs.TPU_MINIMAX_M3_PREFILL_OVERLAP_QKV)


def step(config: OverlapConfig, num_tokens, tp_size: int) -> bool:
    """Whether a step of `num_tokens` takes the overlap path.

    Only inside a backbone warmup trace (`shape_variants.aot_cache_tag` set
    there), which gives each side of this token-count branch its own trace
    (the comparison leaves a shape guard); eager forwards and any other
    compile keep the plain path.
    """
    return bool(
        config.on
        and tp_size > 1
        and torch.compiler.is_compiling()
        and shape_variants.aot_cache_tag()
        and num_tokens >= config.min_tokens
        and num_tokens % TOKEN_MULTIPLE == 0
    )


def split(x: torch.Tensor, half) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows [0, half) and [half, 2 * half) of `x` (2 * half rows)."""
    return x[:half], x[half:]


def all_reduce(x: torch.Tensor, route: str) -> torch.Tensor:
    """Start the TP all-reduce of `x`. "xla": a functional all-reduce whose
    wait Dynamo puts at the result's first use; "group": the TP group's."""
    if route == "group":
        return tensor_model_parallel_all_reduce(x)
    from torch.distributed import _functional_collectives as funcol

    return funcol.all_reduce(x, "sum", get_tp_group().device_group)


def wait(x: torch.Tensor) -> torch.Tensor:
    from torch.distributed import _functional_collectives as funcol

    if isinstance(x, funcol.AsyncCollectiveTensor):
        return funcol.wait_tensor(x)
    return x
