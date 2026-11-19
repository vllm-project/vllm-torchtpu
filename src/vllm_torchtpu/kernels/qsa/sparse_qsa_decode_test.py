"""Correctness tests for the SC-gather + TC-attention QSA decode on TPU."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.qsa import qsa_ref
from vllm_torchtpu.kernels.qsa import sparse_qsa


def _rel_l2(actual, expected):
  return float(
      np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-12)
  )


# Top-k budget (selected compression groups per request).
_GROUPS = 512

# (batch, context, page, heads, kv_heads); ratio is always `RATIO` (4).
_CASES = [
    (1, 1, 128, 16, 1),
    (3, 4, 128, 16, 1),
    (3, 2051, 128, 16, 1),
    (16, 8193, 256, 16, 1),
    (65, 2051, 128, 16, 1),  # Two decode chunks (64 + 1).
    (3, 0, 128, 16, 1),  # All requests empty.
    (7, 2051, 64, 24, 1),  # Small page / non-power-of-2 head count.
    # Two KV heads, 16 query heads each.
    (1, 1, 128, 32, 2),
    (3, 4, 128, 32, 2),
    (3, 2051, 128, 32, 2),
    (16, 8193, 256, 32, 2),
    (65, 2051, 128, 32, 2),  # Two decode chunks (64 + 1).
    (3, 0, 128, 32, 2),  # All requests empty.
]


class SparseQsaDecodeTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.devices()[0].platform != "tpu":
      self.skipTest("Requires TPU.")

  @parameterized.parameters(*_CASES)
  def test_matches_reference(self, batch, context, page, heads, kv_heads):
    ratio = sparse_qsa.RATIO
    q, gate, blocks, lengths, table, cache = qsa_ref.make_inputs(
        batch,
        context,
        page,
        heads=heads,
        kv_heads=kv_heads,
        ratio=ratio,
        groups=_GROUPS,
    )
    args = (q, gate, blocks, lengths, table, cache)
    opts = dict(ratio=ratio, kv_heads=kv_heads)
    out = sparse_qsa.sparse_qsa(*args, **opts)
    expected = qsa_ref.ref_sparse_qsa_decode(*args, **opts)
    out = np.asarray(out.astype(jnp.float32))
    expected = np.asarray(expected.astype(jnp.float32))
    self.assertEqual(out.shape, (batch, heads, sparse_qsa.HEAD_DIM))
    self.assertTrue(np.isfinite(out).all(), "NaN/inf leaked into the output.")
    self.assertLess(_rel_l2(out, expected), 2e-2)


if __name__ == "__main__":
  absltest.main()
