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
"""TPU Dory attention: the nGPT-scaled query enters attention in fp32.

DoryAttention multiplies the normalized query by (sqk * sqrt(head_dim) /
initializer_range)^2, which reaches about 3e4 on the released 4B checkpoint,
then casts it to the model dtype. For a bf16 query, ragged paged attention v3
rounds q.k^T to bf16 before the softmax, and at that scale the rounding
visibly changes the attention weights (the GPU kernels keep fp32 logits). An
fp32 query makes the kernel keep fp32 logits, accumulators and output; the
output is written back in the model dtype for the gate and o_proj.
"""

import torch
from vllm.model_executor.models.dory import DoryAttention, justnorm


class TPUDoryAttention(DoryAttention):
    """DoryAttention whose query reaches the attention backend in fp32."""

    def forward(self, positions, hidden_states, loop_idx=0):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        gate, _ = self.g_proj(hidden_states)

        q, k = self.rotary_emb(positions, q, k)

        t = q.shape[0]
        q = q.view(t, self.num_heads, self.head_dim)
        sqk = (self.sqk.float() * self._sqk_factor).view(
            1, self.num_heads, self.head_dim
        )
        # UPDATE: no cast back to the model dtype, so RPA keeps fp32 logits.
        q = (sqk.square() * justnorm(q).float()).view(t, self.q_size)
        k = k.view(t, self.num_kv_heads, self.head_dim)
        k = justnorm(k).view(t, self.kv_size)

        # UPDATE: the fp32 query's attention output is stored in the model dtype.
        out = self.attn[loop_idx if self.per_loop_kv else 0](
            q, k, v, output_dtype=k.dtype
        )

        sg = (self.s_gate.float() * self._s_gate_factor).square()
        out = (out.float() * torch.sigmoid(gate.float() * sg)).to(out.dtype)
        out = justnorm(out)
        out, _ = self.o_proj(out)
        return out
