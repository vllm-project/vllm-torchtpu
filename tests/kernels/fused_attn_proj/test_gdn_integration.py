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

"""Tests that fused_gdn's outputs feed the GDN kernel, fused_conv1d_gdn.

The GDN wrapper takes qkv (T, qkv_cols) and b, a (T, Hv) and pads b, a itself,
so mixed_qkv is reshaped (a bitcast) and b, a are sliced back to Hv lanes.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest

from tests.kernels.fused_attn_proj import models, reference, test_util
from vllm_torchtpu.kernels.fused_attn_proj.gdn import kernel as gdn
from vllm_torchtpu.kernels.gdn.v3 import wrapper as gdn_wrapper

pytestmark = pytest.mark.tpu_test

MODEL = models.QWEN3_5_GDN_TP8
NUM_TOKENS = 1024  # One prefill sequence.
KERNEL_SIZE = 4  # Qwen3.5's conv1d width.
Numerics = reference.Numerics
NAMES = ("out", "conv_state", "recurrent_state")
# vLLM (torch) rounds every bf16 op; XLA must not elide those roundings.
strict_reference = jax.jit(
    reference.gdn_reference,
    static_argnums=0,
    static_argnames=("numerics",),
    compiler_options={"xla_allow_excess_precision": False},
)


def gdn_params() -> dict[str, jax.Array]:
    """Random conv1d / GDN weights and states for one sequence (slot 1)."""
    c = MODEL.config
    dim, nv, d = c.qkv_cols, c.num_v_heads, c.head_dim
    ks = jax.random.split(jax.random.key(7), 6)
    normal = jax.random.normal
    return dict(
        # vLLM's GDN v3 kernel reads the conv state in fp32.
        conv_state=0.5 * normal(ks[0], (2, KERNEL_SIZE - 1, dim), jnp.float32),
        recurrent_state=(0.1 * normal(ks[1], (2, nv, d, d))).astype(jnp.bfloat16),
        conv_weight=(0.5 * normal(ks[2], (dim, 1, KERNEL_SIZE))).astype(jnp.bfloat16),
        conv_bias=(0.1 * normal(ks[3], (dim,))).astype(jnp.bfloat16),
        a_log=0.5 * normal(ks[4], (nv,), jnp.float32),
        dt_bias=normal(ks[5], (nv,)).astype(jnp.bfloat16),
    )


def run_gdn(
    mixed_qkv: jax.Array,  # (T, 1, qkv_cols) f32
    b: jax.Array,  # (T, 1, lanes) f32
    a: jax.Array,  # (T, 1, lanes) f32
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """fused_conv1d_gdn on fused_gdn's outputs.

    The wrapper donates the states, so every call builds fresh params.

    Args:
      mixed_qkv: (T, 1, qkv_cols) f32.
      b: (T, 1, lanes) f32.
      a: (T, 1, lanes) f32.

    Returns:
      (out, conv_state, recurrent_state).
    """
    c = MODEL.config
    t, nv = mixed_qkv.shape[0], c.num_v_heads
    params = gdn_params()
    states = (params.pop("conv_state"), params.pop("recurrent_state"))
    (conv_state, recurrent_state), out = gdn_wrapper.fused_conv1d_gdn(
        qkv=mixed_qkv.reshape(t, c.qkv_cols),
        b=b[:, 0, :nv],
        a=a[:, 0, :nv],
        states=states,
        **params,
        query_start_loc=jnp.array([0, t], jnp.int32),
        state_indices=jnp.array([1], jnp.int32),
        distribution=jnp.array([0, 1, 1], jnp.int32),
        seq_lens=jnp.array([t], jnp.int32),
        n_kq=c.num_k_heads,
        n_v=nv,
        d_k=c.head_dim,
        d_v=c.head_dim,
        kernel_size=KERNEL_SIZE,
    )
    return out, conv_state, recurrent_state


class GdnIntegrationTest(absltest.TestCase):
    def run_both(
        self, exact: bool, numerics: tuple[Numerics, ...]
    ) -> tuple[tuple[jax.Array, ...], dict[Numerics, tuple[jax.Array, ...]]]:
        """GDN on fused_gdn's outputs, and on gdn_reference's for each numerics."""
        config = MODEL.config
        x, residual, weights = models.make_gdn_inputs(MODEL, NUM_TOKENS, exact)
        prepared = models.prepare_gdn_weights(config, weights)
        mixed_qkv, _, b, a, _ = gdn.fused_gdn(
            x, residual, *prepared, **models.gdn_kwargs(config)
        )
        got = run_gdn(mixed_qkv, b, a)
        fed = {}
        for n in numerics:
            ins = strict_reference(config, x, residual, weights, numerics=n)
            fed[n] = run_gdn(ins[0], ins[2], ins[3])
        return got, fed

    def test_matches_kernel_numerics(self) -> None:
        """GDN on fused_gdn's outputs is GDN on Numerics.KERNEL's, bit for bit."""
        got, fed = self.run_both(True, (Numerics.KERNEL,))
        for name, g, w in zip(NAMES, got, fed[Numerics.KERNEL]):
            g, w = np.asarray(g, np.float32), np.asarray(w, np.float32)
            diff = np.abs(g - w)
            print(
                f"{name} vs KERNEL-fed: max_abs_diff={diff.max():.3e} "
                f"mismatches={(diff > 0).sum()}/{diff.size}",
                flush=True,
            )
            self.assertEqual(diff.max(), 0, name)

    def test_accuracy_report(self) -> None:
        """Prints SNR vs. UNQUANTIZED-fed GDN for fused_gdn and vLLM; no asserts."""
        got, fed = self.run_both(False, (Numerics.VLLM, Numerics.UNQUANTIZED))
        truth = fed[Numerics.UNQUANTIZED]
        for name, g, v, e in zip(NAMES, got, fed[Numerics.VLLM], truth):
            print(
                f"{name}: SNR vs UNQUANTIZED-fed fused={test_util.snr_db(g, e):.2f}"
                f" dB vllm={test_util.snr_db(v, e):.2f} dB",
                flush=True,
            )


if __name__ == "__main__":
    absltest.main()
