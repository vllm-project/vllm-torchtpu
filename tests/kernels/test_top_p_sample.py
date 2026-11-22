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


def test_keyed_uniforms_match_the_torch_hash() -> None:
    """The kernel's keyed uniforms are bit-for-bit the torch ones the runner
    makes from the same stream keys (`rejection_sampler.keyed_uniforms`)."""
    from vllm_torchtpu.kernels.top_p_sample import uniform_from_key
    from vllm_torchtpu.layers.adapter.sample.rejection_sampler import keyed_uniforms

    rng = np.random.default_rng(0)
    ids = rng.integers(0, 200_064, size=4096, dtype=np.int32)
    for key in (0, 1, -7, 2**31 - 1, -(2**31)):
        want = keyed_uniforms(
            torch.tensor([key], dtype=torch.int32), torch.from_numpy(ids)
        )
        got = uniform_from_key(jnp.asarray(ids), jnp.int32(key))
        np.testing.assert_array_equal(np.asarray(got), want.numpy())
        assert want.min() >= 0.0 and want.max() < 1.0


def test_stream_keys_give_uniform_independent_rows() -> None:
    """Each step's rows get distinct streams; their uniforms look uniform."""
    from vllm_torchtpu.layers.adapter.sample.rejection_sampler import (
        keyed_uniforms,
        stream_keys,
    )

    k0 = stream_keys(torch.tensor([0], dtype=torch.int32), 9, seed=0)
    k1 = stream_keys(torch.tensor([1], dtype=torch.int32), 9, seed=0)
    assert len(set(k0.tolist()) | set(k1.tolist())) == 18
    u = keyed_uniforms(k0[:, None], torch.arange(200_000, dtype=torch.int32)[None])
    assert abs(float(u.mean()) - 0.5) < 2e-3
    hist = torch.histc(u, bins=16, min=0.0, max=1.0) / u.numel()
    assert float((hist - 1 / 16).abs().max()) < 2e-3
    assert abs(float(torch.corrcoef(torch.stack([u[0], u[1]]))[0, 1])) < 1e-2


def test_keyed_kernel_matches_the_drawn_layout() -> None:
    """Stream keys in SMEM sample what the same uniforms in the row split's
    layout do (shard padding and ids past the vocabulary included)."""
    from vllm_torchtpu.layers.adapter.sample.rejection_sampler import keyed_uniforms

    rows, shards, width, padded = 8, 4, 1500, 1536
    logits, _, temp, top_p, draft = _case(rows, 4.0, 5)
    keys = torch.tensor([3, -11, 7, 2**30, -5, 0, 99, 12345], dtype=torch.int32)
    ids = torch.arange(shards, dtype=torch.int32)[:, None] * width + torch.arange(
        padded, dtype=torch.int32
    )
    u = keyed_uniforms(keys[:, None, None], ids[None])  # [rows, shards, padded]
    lg = np.full((rows, shards * width), np.nan, np.float32)
    lg[:, :VOCAB] = logits.float().numpy()
    side = np.full((rows, shards, padded), np.nan, np.float32)
    side[:, :, :width] = lg.reshape(rows, shards, width)
    args = (
        jnp.asarray(torch.stack([temp, top_p]).numpy()),
        jnp.asarray(draft.numpy().astype(np.int32)),
    )
    common = dict(interpret=True, row_width=width, vocab=VOCAB)
    side = jnp.asarray(side, jnp.bfloat16)
    want = top_p_rows(side, jnp.asarray(u.numpy()), *args, **common)
    got = top_p_rows(side, jnp.asarray(keys.numpy()), *args, **common)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
