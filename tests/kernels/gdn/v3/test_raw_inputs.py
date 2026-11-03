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
"""CPU interpret-mode bit-exactness of the GDN v3 raw-input and in-kernel
zero-fill paths (TPU_GDN_RAW_QKV, TPU_GDN_RAW_BA, TPU_GDN_KERNEL_ZERO_OUT)
against the default prepared-input path."""

import itertools
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.gdn.v3 import config, wrapper

N_KQ, N_V, D, KS = 1, 2, 128, 4
DIM = 2 * N_KQ * D + N_V * D
SPEC = 3
FLAGS = ("TPU_GDN_RAW_QKV", "TPU_GDN_RAW_BA", "TPU_GDN_KERNEL_ZERO_OUT")


def _inputs(qlens, num_decode, batch, seed=0):
    num_seqs = len(qlens)
    rngs = iter(jax.random.split(jax.random.key(seed), 12))
    normal = lambda shape, dt=jnp.bfloat16: jax.random.normal(  # noqa: E731
        next(rngs), shape, dtype=dt
    )
    num_slots = num_seqs * (SPEC + 1) + 1
    q_loc = np.concatenate([[0], np.cumsum(qlens)]).astype(np.int32)
    assert q_loc[-1] <= batch
    return dict(
        qkv=normal((batch, DIM)),
        b=normal((batch, N_V)),
        a=normal((batch, N_V)),
        conv_state=normal((num_slots, KS - 1, DIM)),
        recurrent_state=normal((num_slots, N_V, D, D), jnp.float32),
        conv_weight=normal((DIM, 1, KS)),
        conv_bias=normal((DIM,)),
        a_log=normal((N_V,), jnp.float32),
        dt_bias=normal((N_V,), jnp.float32),
        query_start_loc=jnp.asarray(q_loc),
        # Base slot of each request's group of SPEC + 1 checkpoints.
        state_indices=jnp.asarray(1 + np.arange(num_seqs) * (SPEC + 1), jnp.int32),
        distribution=jnp.asarray([num_decode, num_seqs, num_seqs], jnp.int32),
        seq_lens=jnp.asarray(np.asarray(qlens) + 9, jnp.int32),
        read_offsets=jnp.asarray(np.arange(num_seqs) % (SPEC + 1), jnp.int32),
    )


KERNEL_NAMES: list[str] = []


def _run(inputs, flags, **kw):
    KERNEL_NAMES.clear()
    orig_name = config.GDNConfig.get_kernel_name

    def spy_name(cfg):
        KERNEL_NAMES.append(orig_name(cfg))
        return KERNEL_NAMES[-1]

    mesh = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    patches = [
        mock.patch.object(envs, name, name in flags) for name in FLAGS
    ] + [
        mock.patch.object(envs, "TPU_ENABLE_GDN_DYNAMIC_TILING", False),
        mock.patch.object(config.GDNConfig, "get_kernel_name", spy_name),
    ]
    jax.clear_caches()
    with use_abstract_mesh(mesh), pltpu.force_tpu_interpret_mode(
        pltpu.InterpretParams(dma_execution_mode="eager", skip_floating_point_ops=False)
    ):
        for p in patches:
            p.start()
        try:
            # The states are donated: hand each run fresh copies.
            fresh = {k: jnp.array(v, copy=True) for k, v in inputs.items()}
            (conv, rec), out = wrapper.fused_conv1d_gdn(
                **fresh,
                n_kq=N_KQ,
                n_v=N_V,
                d_k=D,
                d_v=D,
                kernel_size=KS,
                num_spec_tokens=SPEC,
                # CPU interpret mode lacks ref.reshape on the C > 1 chunked path.
                mixed_tile_size=1,
                **kw,
            )
            jax.block_until_ready((conv, rec, out))
        finally:
            for p in patches:
                p.stop()
    return [np.asarray(x).view(np.uint8) for x in (out, conv, rec)]


# (qlens, num_decode, batch): verify windows of 1..4 tokens, a mixed step with
# two prefills after the windows, and padded rows past the last token.
SCENARIOS = dict(
    verify=([4, 1, 3, 4, 2, 4, 4, 1], 8, 32),
    verify_aligned=([4] * 8, 8, 32),
    mixed=([4, 2, 4, 1, 5, 3], 4, 32),
)


class GDNRawInputTest(parameterized.TestCase):
    @parameterized.named_parameters(
        (f"{name}_{'_'.join(f[4:].lower() for f in flags) or 'none'}", name, flags)
        for name in SCENARIOS
        for flags in [
            c for r in (1, 3) for c in itertools.combinations(FLAGS, r)
        ]
    )
    def test_bit_exact(self, scenario, flags):
        inputs = _inputs(*SCENARIOS[scenario])
        ref = _run(inputs, (), decode_tile_size=2)
        got = _run(inputs, flags, decode_tile_size=2)
        for name, r, g in zip(("out", "conv", "rec"), ref, got):
            np.testing.assert_array_equal(r, g, err_msg=name)
        # Padded output rows are zero in both paths.
        q_end = int(inputs["query_start_loc"][-1])
        out = got[0].view(np.uint16).reshape(inputs["qkv"].shape[0], -1)
        self.assertFalse(out[q_end:].any())
        # The flagged paths compiled into the kernels.
        names = " ".join(KERNEL_NAMES)
        for flag, suffix in zip(FLAGS, ("_rawqkv", "_rawba", "_zf")):
            self.assertEqual(flag in flags, suffix in names, (flag, names))

    def test_fault_is_caught(self):
        """Raw rows placed one row late make the outputs differ."""
        inputs = _inputs(*SCENARIOS["verify"])
        ref = _run(inputs, (), decode_tile_size=2)
        orig = wrapper._load_raw_rows

        def shifted(*args, **kwargs):
            return jnp.roll(orig(*args, **kwargs), 1, axis=1)

        with mock.patch.object(wrapper, "_load_raw_rows", shifted):
            got = _run(inputs, ("TPU_GDN_RAW_QKV",), decode_tile_size=2)
        self.assertFalse(np.array_equal(ref[0], got[0]))


if __name__ == "__main__":
    absltest.main()
