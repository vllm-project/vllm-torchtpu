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
# ==============================================================================
"""Input row quantization, token-replica ownership, and the shared expert FFN.

Row quantization (`rowquant_fp8`) and token-replica ownership
(`TokenReplicaLayout`) are shared with `fused_moe`; the clamped activations
come from `gmm_v2`. The replicated shared expert FFN is this layer's own.
"""

import jax
import jax.numpy as jnp
from tokamax._src.ops.experimental.fused_moe import host as fused_host
from tokamax._src.ops.experimental.fused_moe import kernel as fused_kernel
from tokamax._src.ops.experimental.fused_moe import token_parallel
from tokamax._src.ops.gmm_v2 import gmm_v2

rowquant_fp8 = fused_kernel.rowquant_fp8
TokenReplicaLayout = token_parallel.TokenReplicaLayout
silu_and_mul_with_clamp = gmm_v2.silu_and_mul_with_clamp
swigluoai = gmm_v2.swigluoai

# Activations the shared expert FFN supports.
SHARED_EXPERT_ACT_FNS = ("silu", "gelu", "swigluoai")


def _apply_shared_expert_act(
    gate: jax.Array, up: jax.Array, act_fn: str
) -> jax.Array:
  """Applies the shared-expert gated activation on f32 gate and up halves."""
  if act_fn == "silu":
    return silu_and_mul_with_clamp(gate, up)
  if act_fn == "gelu":
    return jax.nn.gelu(gate) * up
  if act_fn == "swigluoai":
    return swigluoai(gate, up)
  raise NotImplementedError(
      f"shared expert act_fn must be one of {SHARED_EXPERT_ACT_FNS}; got"
      f" {act_fn!r}"
  )


def shared_expert_ffn(
    x_l: jax.Array,
    ws1: jax.Array,
    ws2: jax.Array,
    *,
    act_fn: str = "silu",
    ws1_scale: jax.Array | None = None,
    ws2_scale: jax.Array | None = None,
    q_l: jax.Array | None = None,
    row_scale_l: jax.Array | None = None,
    shared_expert_fp8: bool = False,
) -> jax.Array:
  """Replicated shared expert FFN on the local tokens.

  ws1 [D, 2 * Fs] holds the concatenated [gate, up] projection weights
  (matching the routed experts' w1 [E, D, 2 * F] layout); ws2 [Fs, D] is the
  down projection. "silu" applies the clamped SiLU-and-multiply form.

  By default (`shared_expert_fp8=False`), weights are cast to bf16 so the
  output is bit-exact with the bf16 shared-expert reference. When opt-in
  `shared_expert_fp8=True` and `ws1`/`ws2` are fp8 (`fused_host.FP8`), runs
  native W8A8 dynamic per-row fp8 matrix multiplications on the fp8 MXU
  (reusing pre-quantized `(q_l, row_scale_l)` from Step 1 when provided, and
  applying optional per-channel `ws1_scale` / `ws2_scale`).

  The result stays f32 on purpose: the caller adds it to the routed
  combine's f32 sum and rounds the total to the output dtype once, as the
  f32 reference does.

  Args:
    x_l: [t, D] local tokens.
    ws1: [D, 2 * Fs] gate and up projection weights (bf16 or fp8).
    ws2: [Fs, D] down projection weights (bf16 or fp8).
    act_fn: One of SHARED_EXPERT_ACT_FNS.
    ws1_scale: Optional [2 * Fs] or [1, 2 * Fs] f32 per-channel scale for ws1.
    ws2_scale: Optional [D] or [1, D] f32 per-channel scale for ws2.
    q_l: Optional pre-quantized [t, D] fp8 local tokens.
    row_scale_l: Optional pre-computed [t, 1] f32 row scales for `q_l`.
    shared_expert_fp8: Opt-in flag (default False). When True and weights are
      fp8, runs native W8A8 fp8 matmuls instead of upcasting weights to bf16.

  Returns:
    [t, D // 128, 128] f32 shared expert output.

  Raises:
    NotImplementedError: If `act_fn` is not supported.
  """
  t_local, hidden = x_l.shape
  fs = ws2.shape[0]
  if act_fn not in SHARED_EXPERT_ACT_FNS:
    raise NotImplementedError(
        f"shared expert act_fn must be one of {SHARED_EXPERT_ACT_FNS}; got"
        f" {act_fn!r}"
    )
  is_fp8 = (
      shared_expert_fp8
      and ws1.dtype == fused_host.FP8
      and ws2.dtype == fused_host.FP8
  )
  if is_fp8:
    if q_l is None or row_scale_l is None:
      if x_l.dtype == fused_host.FP8 and row_scale_l is not None:
        q_in, s_in = x_l.reshape(t_local, hidden), row_scale_l
      else:
        q_in, s_in = rowquant_fp8(x_l.astype(jnp.bfloat16))
    else:
      q_in = q_l.reshape(t_local, hidden)
      s_in = row_scale_l.reshape(t_local, 1).astype(jnp.float32)
    gate_up = (
        jnp.dot(q_in, ws1, preferred_element_type=jnp.float32) * s_in
    )
    if ws1_scale is not None:
      gate_up = gate_up * ws1_scale.astype(jnp.float32).reshape(1, 2 * fs)
    gate, up = gate_up[:, :fs], gate_up[:, fs:]
    mid = _apply_shared_expert_act(gate, up, act_fn)
    mid_q, mid_scale = rowquant_fp8(mid.astype(jnp.bfloat16))
    out = (
        jnp.dot(mid_q, ws2, preferred_element_type=jnp.float32) * mid_scale
    )
    if ws2_scale is not None:
      out = out * ws2_scale.astype(jnp.float32).reshape(1, hidden)
  else:
    gate_up = jnp.dot(
        x_l.astype(jnp.bfloat16),
        ws1.astype(jnp.bfloat16),
        preferred_element_type=jnp.float32,
    )
    if ws1_scale is not None:
      gate_up = gate_up * ws1_scale.astype(jnp.float32).reshape(1, 2 * fs)
    gate, up = gate_up[:, :fs], gate_up[:, fs:]
    mid = _apply_shared_expert_act(gate, up, act_fn)
    out = jnp.dot(
        mid.astype(jnp.bfloat16),
        ws2.astype(jnp.bfloat16),
        preferred_element_type=jnp.float32,
    )
    if ws2_scale is not None:
      out = out * ws2_scale.astype(jnp.float32).reshape(1, hidden)
  lanes = fused_host.HIDDEN_LANE_BLOCK
  return out.reshape(t_local, hidden // lanes, lanes)
