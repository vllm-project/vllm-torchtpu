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
"""Dense-row top-p kernel vs `rejection_sampler._recover_tokens` (XLA path).

``interpret=True`` runs the kernel body on the host; it does not gate Mosaic
lowering.
"""

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels.top_p_sample import top_p_rows
from vllm_torchtpu.layers.adapter.sample.rejection_sampler import _recover_tokens

VOCAB = 128 * 45


def _case(rows: int, scale: float, seed: int):
    rng = np.random.default_rng(seed)
    logits = (rng.standard_normal((rows, VOCAB)) * scale).astype(np.float32)
    logits[:, :50] += 6 * scale  # a peaked head
    logits = torch.from_numpy(logits).bfloat16()
    u = torch.from_numpy(rng.random((rows, VOCAB), dtype=np.float32))
    temp = torch.tensor([1.0, 0.7, 0.0, 1.3, 1.0, 1.0, 0.9, 1.0])[:rows]
    top_p = torch.tensor([0.95, 0.5, 0.95, 1.0, 0.1, 0.999, 0.95, 0.8])[:rows]
    # Drop the argmax on some rows, nothing (-1) on others.
    draft = torch.where(torch.arange(rows) % 3 == 0, -1, logits.float().argmax(-1))
    return logits, u, temp, top_p, draft


@pytest.mark.parametrize("rows", [1, 4, 8])
@pytest.mark.parametrize("scale", [1.0, 4.0])
@pytest.mark.parametrize("apply_top_p", [True, False])
def test_matches_recover_tokens(rows: int, scale: float, apply_top_p: bool) -> None:
    logits, u, temp, top_p, draft = _case(rows, scale, rows * 10 + int(scale))
    want = _recover_tokens(
        logits,
        draft,
        temp[:, None],
        torch.zeros(rows, 1, dtype=torch.int32),
        top_p[:, None],
        u,
        topk_candidates=0,
        apply_topk_topp=apply_top_p,
        draft_probs=None,
        apply_top_k=False,
    )
    params = torch.stack([temp, top_p if apply_top_p else torch.ones(rows)])
    got = top_p_rows(
        jnp.asarray(logits.float().numpy(), jnp.bfloat16).reshape(rows, -1, 128),
        jnp.asarray(u.numpy()).reshape(rows, -1, 128),
        jnp.asarray(params.numpy()),
        jnp.asarray(draft.numpy().astype(np.int32)),
        interpret=True,
    )
    np.testing.assert_array_equal(np.asarray(got)[:rows, 0], want.numpy())


def test_torch_op_signature() -> None:
    """The torch op takes the kernel's argument order (shape inference only)."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    from vllm_torchtpu.layers.adapter.sample import top_p_sample

    with FakeTensorMode():
        out = top_p_sample._op(
            torch.empty(3, 45, 128, dtype=torch.bfloat16),
            torch.empty(3, 45, 128),
            torch.empty(2, 3),
            torch.empty(3, dtype=torch.int32),
        )
    assert out.shape == (8, 128) and out.dtype == torch.int32


def test_shards_side_by_side_match_the_dense_slab() -> None:
    """The row split's layout (each TP rank's LM head shard as one row, padded
    to whole lane groups, ids past the vocabulary padding) samples the tokens
    the dense ``[vocab / 128, 128]`` slab does."""
    rows, shards, width, padded = 8, 4, 1500, 1536  # 6000 ids cover VOCAB
    logits, u, temp, top_p, draft = _case(rows, 4.0, 3)
    params = jnp.asarray(torch.stack([temp, top_p]).numpy())
    draft = jnp.asarray(draft.numpy().astype(np.int32))
    want = top_p_rows(
        jnp.asarray(logits.float().numpy(), jnp.bfloat16).reshape(rows, -1, 128),
        jnp.asarray(u.numpy()).reshape(rows, -1, 128),
        params,
        draft,
        interpret=True,
    )

    def side_by_side(x: np.ndarray) -> np.ndarray:
        ids = np.full((rows, shards * width), np.nan, np.float32)
        ids[:, :VOCAB] = x
        out = np.full((rows, shards, padded), np.nan, np.float32)
        out[:, :, :width] = ids.reshape(rows, shards, width)
        return out

    got = top_p_rows(
        jnp.asarray(side_by_side(logits.float().numpy()), jnp.bfloat16),
        jnp.asarray(side_by_side(u.numpy())),
        params,
        draft,
        interpret=True,
        row_width=width,
        vocab=VOCAB,
    )
    np.testing.assert_array_equal(np.asarray(got)[:, 0], np.asarray(want)[:, 0])
