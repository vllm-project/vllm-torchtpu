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
"""e4m3 LM head with one scale per vocab row (``TPU_LM_HEAD_E4M3``).

The LM head is a weight-bound matmul: every logits call (target verify and
the draft) reads this rank's whole vocab shard, 25008 x 6144 bf16 = 307 MB
for a 200064-token vocab at TP 8. Stored as e4m3 with a float32 scale per
row it reads half the bytes. Each row is scaled by its own absolute maximum
over 448, so every row uses the full e4m3 range; the matmul runs in the
activation dtype on the converted weight and the row scale multiplies the
output column. Not exact: logits differ from the bf16 head by the e4m3
rounding of the weight (about 2^-4 relative per element).

With ``TPU_LM_HEAD_E4M3_SPLIT_MAX_ROWS`` calls of at most that many rows
(decode verify and draft logits) also run the matmul in fp8: the activations
are split into two e4m3 terms, ``hi`` and the residual ``lo`` at 16x scale,
and one fp8 x fp8 matmul over ``[hi; lo]`` reads the e4m3 weight once. The
two terms carry about 7 mantissa bits, as bf16 does, so the logits differ
from the upcast-weight path by output rounding only.
"""

import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch_tpu._internal import pallas
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)

import vllm_torchtpu.envs as envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_E4M3_MAX = 448.0
# The low term is quantized at 1/16 of the row scale: the high term's rounding
# error is at most half an e4m3 ulp (<= 16 * s at the top binade), so the
# scaled residual stays inside e4m3's range.
_SPLIT_LO = 16.0


def split_e4m3_logits(
    x: jax.Array, w: jax.Array, row_scale: jax.Array
) -> jax.Array:
    """``(x @ w.T) * row_scale`` for bf16 ``x[T, H]``, e4m3 ``w[V, H]`` and
    float32 ``row_scale[V]``, with ``x`` split into two e4m3 terms and one
    fp8 x fp8 matmul.

    Per row of ``x``, ``s`` is a power of two that puts the row's abs-max at
    or below 448; ``hi = e4m3(x / s)``, ``lo = e4m3((x / s - hi) * 16)``. Both
    terms share one pass over the weight: the matmul runs over ``[hi; lo]``
    (2T rows) and ``out = (y_hi + y_lo / 16) * s * row_scale``, accumulated
    in float32."""
    rows = x.shape[0]
    xf = x.astype(jnp.float32)
    amax = jnp.max(jnp.abs(xf), axis=-1, keepdims=True)
    amax = jnp.where(amax > 0, amax, 1.0)
    # Power-of-two scale: dividing by it is exact.
    s = jnp.exp2(jnp.ceil(jnp.log2(amax / _E4M3_MAX)))
    xs = xf / s
    hi = xs.astype(jnp.float8_e4m3fn)
    lo = ((xs - hi.astype(jnp.float32)) * _SPLIT_LO).astype(jnp.float8_e4m3fn)
    y = jax.lax.dot_general(
        jnp.concatenate([hi, lo], axis=0),
        w,
        (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    out = (y[:rows] + y[rows:] * (1.0 / _SPLIT_LO)) * s
    out = out * row_scale.astype(jnp.float32)[None, :]
    return out.astype(x.dtype)


# Built at import: torch_tpu cannot create a jax_op inside a traced forward.
_split_logits = pallas.jax_op("pallas::lm_head_split_e4m3_logits", split_e4m3_logits)
_split_logits.register_fake(
    lambda x, w, s: torch.empty(
        (x.shape[0], w.shape[0]), dtype=x.dtype, device=x.device
    )
)


def quantize_rows(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``weight`` [rows, hidden] -> (e4m3 weight, float32 scale [rows])."""
    w = weight.float()
    amax = w.abs().amax(dim=1)
    scale = torch.where(amax > 0, amax / _E4M3_MAX, torch.ones_like(amax))
    q = (w / scale[:, None]).clamp(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn)
    return q, scale


class E4m3RowLMHeadMethod(UnquantizedEmbeddingMethod):
    """Logits from an e4m3 weight and a per-row scale."""

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dump = getattr(layer, "_head_dump", None)
        if dump is not None:
            dump.record(x)
        rows = math.prod(x.shape[:-1])
        if (
            rows <= envs.TPU_LM_HEAD_E4M3_SPLIT_MAX_ROWS
            and x.dtype == torch.bfloat16
        ):
            # Rows are static at trace time, so each bucket picks one path.
            out = _split_logits(
                x.reshape(rows, x.shape[-1]),
                layer.weight,
                layer.weight_row_scale,
            ).reshape(*x.shape[:-1], layer.weight.shape[0])
        else:
            out = torch.matmul(x, layer.weight.to(x.dtype).t())
            out = (out.float() * layer.weight_row_scale).to(x.dtype)
        if bias is not None:
            out = out + bias
        return out

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("the e4m3 LM head is not an input embedding")


def _blocker(head: torch.nn.Module, processors: list) -> str | None:
    if type(head.quant_method) is not UnquantizedEmbeddingMethod:
        return f"the LM head uses {type(head.quant_method).__name__}"
    if head.weight.dtype != torch.bfloat16:
        return f"the LM head weight is {head.weight.dtype}"
    for p in processors:
        # LogitsProcessor._apply_head calls quant_method.apply only when
        # head_dtype is unset or equals the activation (= weight) dtype.
        head_dtype = getattr(p, "head_dtype", None) if p is not None else None
        if head_dtype is not None and head_dtype != torch.bfloat16:
            return f"a logits processor sets head_dtype {head_dtype}"
    return None


def arm(model: torch.nn.Module, draft: torch.nn.Module | None) -> bool:
    """Convert the target's LM head (shared with the draft when the draft
    holds the same module) to e4m3 rows. Returns whether it was converted."""
    head = getattr(model, "lm_head", None)
    if head is None:
        logger.warning("TPU_LM_HEAD_E4M3=1 ignored: the model has no lm_head")
        return False
    draft_head = getattr(draft, "lm_head", None) if draft is not None else None
    if draft_head is not None and draft_head is not head:
        logger.warning(
            "TPU_LM_HEAD_E4M3=1 ignored: the draft holds its own LM head"
        )
        return False
    procs = [
        getattr(model, "logits_processor", None),
        getattr(draft, "logits_processor", None) if draft is not None else None,
    ]
    why = _blocker(head, procs)
    if why is not None:
        logger.warning("TPU_LM_HEAD_E4M3=1 ignored: %s", why)
        return False
    if envs.TPU_LM_HEAD_E4M3_DUMP:
        head._bf16_for_dump = head.weight.data
    q, scale = quantize_rows(head.weight.data)
    head.weight = torch.nn.Parameter(q, requires_grad=False)
    head.register_buffer("weight_row_scale", scale, persistent=False)
    head.quant_method = E4m3RowLMHeadMethod()
    logger.info(
        "TPU_LM_HEAD_E4M3=1: LM head shard %s stored as e4m3 with a per-row "
        "scale (shared with the draft: %s); split-e4m3 fp8 matmul for calls "
        "of 1..%d rows (TPU_LM_HEAD_E4M3_SPLIT_MAX_ROWS)",
        tuple(q.shape),
        draft_head is head,
        envs.TPU_LM_HEAD_E4M3_SPLIT_MAX_ROWS,
    )
    return True


def _bf16_bits(t: torch.Tensor) -> np.ndarray:
    return t.detach().to("cpu").contiguous().view(torch.int16).numpy()


class HeadDump:
    """Debug only (``TPU_LM_HEAD_E4M3_DUMP``): captures the LM head inputs of
    decode steps for an offline numerics check. Every rank copies each call's
    input into a device buffer inside the compiled graph (one buffer for
    calls of at most 32 rows, the verify logits, one for larger, the draft
    logits); rank 0 saves the buffers outside the graph (bf16 bits, .npy)
    with its bf16 head shard."""

    _ROWS = 256
    _SMALL = 32

    def __init__(self, head: torch.nn.Module, count: int, out_dir: str):
        dev = head.weight.device
        hidden = head.weight.shape[1]
        self.count, self.out_dir, self.saved, self.steps = count, out_dir, 0, 0
        self.bufs = {
            k: torch.zeros(self._ROWS, hidden, dtype=torch.bfloat16, device=dev)
            for k in ("small", "large")
        }
        # [rows of the last call, calls so far]
        self.meta = {
            k: torch.zeros(2, dtype=torch.int32, device=dev) for k in self.bufs
        }
        self.last_calls = {k: 0 for k in self.bufs}
        head._head_dump = self

    @classmethod
    def maybe_create(cls, head: torch.nn.Module, count: int) -> "HeadDump | None":
        from vllm.distributed import get_tensor_model_parallel_rank

        out_dir = os.path.join(os.environ.get("CDK_OUTPUT_DIR", "/tmp"), "lm_head_dump")
        dump = cls(head, count, out_dir)
        dump.rank0 = get_tensor_model_parallel_rank() == 0
        if dump.rank0:
            os.makedirs(out_dir, exist_ok=True)
        bf16 = getattr(head, "_bf16_for_dump", None)
        if bf16 is not None:
            dump.save_weight(bf16)
            del head._bf16_for_dump
        return dump

    def save_weight(self, weight_bf16: torch.Tensor) -> None:
        if self.rank0:
            np.save(os.path.join(self.out_dir, "weight_bf16.npy"), _bf16_bits(weight_bf16))
            logger.info("TPU_LM_HEAD_E4M3_DUMP: saved head shard %s", tuple(weight_bf16.shape))

    def record(self, x: torch.Tensor) -> None:
        rows = math.prod(x.shape[:-1])
        key = "small" if rows <= self._SMALL else "large"
        n = min(rows, self._ROWS)
        self.bufs[key][:n].copy_(x.reshape(rows, x.shape[-1])[:n])
        self.meta[key][0:1].fill_(rows)
        self.meta[key][1:2].add_(1)

    def save(self) -> None:
        if not self.rank0 or self.saved >= self.count:
            return
        self.steps += 1
        # Skip warm-up and sample every 8th step.
        if self.steps < 64 or self.steps % 8:
            return
        for key, buf in self.bufs.items():
            rows, calls = (int(v) for v in self.meta[key].cpu())
            if calls == self.last_calls[key] or rows == 0:
                continue
            self.last_calls[key] = calls
            n = min(rows, self._ROWS)
            np.save(
                os.path.join(self.out_dir, f"x_{key}_{self.steps:06d}_r{rows}.npy"),
                _bf16_bits(buf[:n]),
            )
        self.saved += 1
