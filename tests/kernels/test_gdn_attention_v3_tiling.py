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

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.gdn.v3 import compute_gdn, config, wrapper

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _on_cpu():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


@pytest.mark.parametrize("chunk", [1, 3, 8, 15, 16, 32, 80, 128])
def test_triangular_inverse_preserves_every_row(chunk):
    rng = np.random.default_rng(0)
    lower = rng.normal(scale=0.02, size=(2, chunk, chunk)).astype(np.float32)
    matrix = np.tril(lower, -1) + np.eye(chunk, dtype=np.float32)

    actual = compute_gdn.invert_triangular_matrix(jnp.asarray(matrix))

    assert actual.shape == matrix.shape
    np.testing.assert_allclose(actual,
                               np.linalg.inv(matrix),
                               rtol=2e-5,
                               atol=2e-6)


@pytest.mark.parametrize("chunk", [17, 20, 31, 33])
def test_triangular_inverse_handles_non_multiple_blocks(chunk):
    rng = np.random.default_rng(chunk)
    lower = rng.normal(scale=0.02, size=(2, chunk, chunk)).astype(np.float32)
    matrix = np.tril(lower, -1) + np.eye(chunk, dtype=np.float32)

    actual = compute_gdn.invert_triangular_matrix(jnp.asarray(matrix))

    assert actual.shape == matrix.shape
    np.testing.assert_allclose(actual,
                               np.linalg.inv(matrix),
                               rtol=2e-5,
                               atol=2e-6)


@pytest.mark.parametrize(
    ("tokens", "heads", "requested_tile", "decodes", "expected_tile"),
    [
        (3, 8, None, 0, 3),
        (16, 8, None, 0, 16),
        (20, 8, None, 0, 16),
        (20, 8, None, 2, 16),
        (31, 8, None, 0, 16),
        (80, 8, None, 0, 80),
        (144, 8, None, 0, 128),
        (20, 16, None, 0, 16),
        (80, 16, None, 0, 64),
        (144, 16, None, 0, 64),
        (80, 8, 20, 0, 16),
    ],
)
def test_wrapper_tiles_cover_prefill_rows(monkeypatch, tokens, heads,
                                          requested_tile, decodes,
                                          expected_tile):
    """Exercise the wrapper and metadata, replacing the TPU kernel boundary."""
    monkeypatch.setattr(
        wrapper.pltpu, "get_tpu_info",
        lambda: SimpleNamespace(num_lanes=128,
                                vmem_capacity_bytes=128 * 1024 * 1024))
    per_seq_calls = []

    def pallas_call(kernel, *, out_shape, **kwargs):
        cfg = kernel.keywords["cfg"]

        def call(metadata, qkv, b, a, conv, recurrent, out, weights):
            if cfg.mode == config.GDNMode.PER_SEQ:
                per_seq_calls.append((cfg, metadata))
            return jnp.zeros(out_shape[0].shape,
                             out_shape[0].dtype), conv, recurrent

        return call

    monkeypatch.setattr(wrapper.pl, "pallas_call", pallas_call)
    lengths = [1] * decodes + [tokens - decodes]
    num_seqs = len(lengths)
    dim = heads * 3
    # Bypass the outer jit so each call observes the substituted boundary.
    wrapper.fused_conv1d_gdn.__wrapped__(
        qkv=jnp.zeros((tokens, dim)),
        b=jnp.zeros((tokens, heads)),
        a=jnp.zeros((tokens, heads)),
        conv_state=jnp.zeros((num_seqs + 1, 3, dim)),
        recurrent_state=jnp.zeros((num_seqs + 1, heads, 1, 1)),
        conv_weight=jnp.zeros((dim, 1, 4)),
        conv_bias=None,
        a_log=jnp.zeros((heads, )),
        dt_bias=jnp.zeros((heads, )),
        query_start_loc=jnp.asarray(np.cumsum([0] + lengths), jnp.int32),
        state_indices=jnp.arange(1, num_seqs + 1, dtype=jnp.int32),
        distribution=jnp.asarray([decodes, num_seqs, num_seqs], jnp.int32),
        seq_lens=jnp.asarray(lengths, jnp.int32),
        n_kq=heads,
        n_v=heads,
        d_k=1,
        d_v=1,
        kernel_size=4,
        mixed_tile_size=requested_tile,
    )

    assert len(per_seq_calls) == 1
    cfg, metadata = per_seq_calls[0]
    assert cfg.chunk_size == expected_tile
    count = int(metadata.num_tiles)
    starts = np.asarray(metadata.p_id_to_r_base.data)[:count]
    sizes = np.asarray(metadata.p_id_to_r_size.data)[:count]
    assert np.all((sizes > 0) & (sizes <= expected_tile))
    covered = [
        row for start, size in zip(starts, sizes)
        for row in range(int(start), int(start + size))
    ]
    assert covered == list(range(decodes, tokens))
