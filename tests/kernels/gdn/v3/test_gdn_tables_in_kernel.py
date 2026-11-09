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
"""CPU interpret-mode checks that GDN v3 gives the same results with its
routing tables built inside the kernel (`build_tables_in_kernel=True`) as
with the tables built by XLA ops in the wrapper."""

import contextlib
import sys
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src.state import indexing
from jax._src.state import types as state_types
from jax.experimental.pallas import tpu as pltpu
from jax.extend import core as jax_core
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.gdn.v3 import wrapper

N_KQ, N_V, D_K, D_V, KERNEL_SIZE = 1, 2, 128, 128, 4
DIM = 2 * N_KQ * D_K + N_V * D_V
_TPU7X_MESH = AbstractMesh(
    (), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu")
)


@contextlib.contextmanager
def _cpu_interpret_mode():
    """TPU interpret mode on CPU, with the granule-view shims of
    test_gdn_attention_v3.GDNStatePipelineRaceTest: CPU interpret mode
    otherwise assumes NDIndexer-only DMA transforms."""
    mosaic_interpret = sys.modules[pltpu.reset_tpu_interpret_mode_state.__module__]
    interpret_utils = mosaic_interpret.interpret_utils
    shared_memory = sys.modules[f"{mosaic_interpret.__package__}.shared_memory"]
    orig_store = shared_memory.SharedMemory.store_buffer_content

    def _resolve_to_range(transforms):
        ret, viewed = (), False
        for t in transforms:
            if isinstance(t, indexing.NDIndexer):
                idx = tuple(
                    interpret_utils._transform_slice_or_index(i) for i in t.indices
                )
                if viewed:
                    full = tuple(slice(0, d, 1) for d in t.shape[-3:])
                    assert idx[-3:] == full, f"partial-state ref: {idx}"
                    idx = idx[:-3]
                ret = interpret_utils._compose_slice_or_index(ret, idx)
            elif isinstance(
                t, (state_types.BitcastTransform, state_types.ReshapeTransform)
            ):
                if not viewed:
                    ret, viewed = ret[:-3], True
            else:
                raise NotImplementedError(f"unsupported ref transform: {t}")
        return ret

    def _store_buffer_content(buf_self, key, rnge, value, *args, **kwargs):
        buff_dtype = buf_self.mem[key].dtype
        assert value.itemsize == buff_dtype.itemsize, (value.dtype, buff_dtype)
        return orig_store(buf_self, key, rnge, value.view(buff_dtype), *args, **kwargs)

    pltpu.reset_tpu_interpret_mode_state()
    try:
        with (
            use_abstract_mesh(_TPU7X_MESH),
            mock.patch.object(envs, "TPU_ENABLE_GDN_DYNAMIC_TILING", False),
            mock.patch.object(interpret_utils, "to_range", _resolve_to_range),
            mock.patch.object(
                shared_memory.SharedMemory,
                "store_buffer_content",
                _store_buffer_content,
            ),
            pltpu.force_tpu_interpret_mode(
                pltpu.InterpretParams(
                    dma_execution_mode="eager", skip_floating_point_ops=False
                )
            ),
        ):
            yield
    finally:
        pltpu.reset_tpu_interpret_mode_state()


def _make_inputs(query_lens, context_lens, num_padded_seqs=0, seed=0):
    """Inputs for sequences 1..n in state slots 1..n (slot 0 is the null
    block), plus `num_padded_seqs` unused trailing sequence entries."""
    num_seqs = len(query_lens)
    max_seqs = num_seqs + num_padded_seqs
    num_tokens = sum(query_lens)
    num_blocks = max_seqs + 2
    rngs = iter(jax.random.split(jax.random.key(seed), 9))
    q_loc = np.concatenate([[0], np.cumsum(query_lens)])
    q_loc = np.concatenate([q_loc, np.full(num_padded_seqs, q_loc[-1])])
    seq_lens = np.array(
        [q + c for q, c in zip(query_lens, context_lens)] + [0] * num_padded_seqs
    )
    return dict(
        qkv=jax.random.normal(next(rngs), (num_tokens, DIM), dtype=jnp.bfloat16),
        b=jax.random.normal(next(rngs), (num_tokens, N_V), dtype=jnp.bfloat16),
        a=jax.random.normal(next(rngs), (num_tokens, N_V), dtype=jnp.bfloat16),
        conv_weight=jax.random.normal(
            next(rngs), (DIM, 1, KERNEL_SIZE), dtype=jnp.bfloat16
        ),
        conv_bias=jax.random.normal(next(rngs), (DIM,), dtype=jnp.bfloat16),
        a_log=jax.random.normal(next(rngs), (N_V,), dtype=jnp.float32),
        dt_bias=jax.random.normal(next(rngs), (N_V,), dtype=jnp.float32),
        conv_state=jax.random.normal(
            next(rngs), (num_blocks, KERNEL_SIZE - 1, DIM), dtype=jnp.float32
        ),
        recurrent_state=jax.random.normal(
            next(rngs), (num_blocks, N_V, D_K, D_V), dtype=jnp.float32
        ),
        query_start_loc=jnp.asarray(q_loc, dtype=jnp.int32),
        state_indices=jnp.arange(1, max_seqs + 1, dtype=jnp.int32),
        seq_lens=jnp.asarray(seq_lens, dtype=jnp.int32),
    )


def _run(inputs, distribution, build_tables_in_kernel, **kwargs):
    # The wrapper donates the states, so give each run its own copy.
    states = (jnp.copy(inputs["conv_state"]), jnp.copy(inputs["recurrent_state"]))
    with _cpu_interpret_mode():
        new_states, out = wrapper.fused_conv1d_gdn(
            qkv=inputs["qkv"],
            b=inputs["b"],
            a=inputs["a"],
            states=states,
            conv_weight=inputs["conv_weight"],
            conv_bias=inputs["conv_bias"],
            a_log=inputs["a_log"],
            dt_bias=inputs["dt_bias"],
            query_start_loc=inputs["query_start_loc"],
            state_indices=inputs["state_indices"],
            distribution=jnp.asarray(distribution, dtype=jnp.int32),
            seq_lens=inputs["seq_lens"],
            n_kq=N_KQ,
            n_v=N_V,
            d_k=D_K,
            d_v=D_V,
            kernel_size=KERNEL_SIZE,
            # CPU interpret mode does not support the C > 1 chunked prefill
            # path; prefill sequences run one token per tile.
            mixed_tile_size=1,
            build_tables_in_kernel=build_tables_in_kernel,
            **kwargs,
        )
        return jax.block_until_ready((new_states, out))


def _kernel_names(inputs, distribution, build_tables_in_kernel, **kwargs):
    """Names of the Pallas calls the wrapper traces for one step."""
    with use_abstract_mesh(_TPU7X_MESH):
        jaxpr = jax.make_jaxpr(
            lambda: wrapper.fused_conv1d_gdn(
                qkv=inputs["qkv"],
                b=inputs["b"],
                a=inputs["a"],
                states=(inputs["conv_state"], inputs["recurrent_state"]),
                conv_weight=inputs["conv_weight"],
                conv_bias=inputs["conv_bias"],
                a_log=inputs["a_log"],
                dt_bias=inputs["dt_bias"],
                query_start_loc=inputs["query_start_loc"],
                state_indices=inputs["state_indices"],
                distribution=jnp.asarray(distribution, dtype=jnp.int32),
                seq_lens=inputs["seq_lens"],
                n_kq=N_KQ,
                n_v=N_V,
                d_k=D_K,
                d_v=D_V,
                kernel_size=KERNEL_SIZE,
                build_tables_in_kernel=build_tables_in_kernel,
                **kwargs,
            )
        )()
    names = []

    def visit(jaxpr):
        for eqn in jaxpr.eqns:
            if eqn.primitive.name == "pallas_call":
                # The parameter is `name` in recent JAX, `name_and_src_info` before.
                name = eqn.params.get("name", eqn.params.get("name_and_src_info"))
                names.append(getattr(name, "name", name))
            for sub in jax_core.jaxprs_in_params(eqn.params):
                visit(sub)

    visit(jaxpr.jaxpr)
    return names


def _assert_same(test, expected, actual):
    jax.tree.map(
        lambda e, a: np.testing.assert_array_equal(np.asarray(a), np.asarray(e)),
        expected,
        actual,
    )
    del test


class GDNTablesInKernelTest(parameterized.TestCase):
    @parameterized.named_parameters(
        dict(
            # Decode only: the per-sequence pass has no tiles.
            testcase_name="decode_only",
            query_lens=[1] * 6,
            context_lens=[5, 0, 7, 3, 9, 2],
            num_decode=6,
            num_padded_seqs=0,
        ),
        dict(
            # Decodes, then prefills of several one-token tiles each, some
            # resuming a context (chunked prefill).
            testcase_name="decode_and_prefill",
            query_lens=[1, 1, 1, 3, 1, 2],
            context_lens=[4, 0, 6, 0, 5, 8],
            num_decode=3,
            num_padded_seqs=2,
        ),
        dict(
            # Prefill only: the batched pass has no tiles.
            testcase_name="prefill_only",
            query_lens=[3, 2],
            context_lens=[0, 4],
            num_decode=0,
            num_padded_seqs=1,
        ),
    )
    def test_same_results_as_wrapper_tables(
        self, query_lens, context_lens, num_decode, num_padded_seqs
    ):
        inputs = _make_inputs(query_lens, context_lens, num_padded_seqs)
        num_seqs = len(query_lens)
        distribution = [num_decode, num_seqs, num_seqs]
        expected = _run(inputs, distribution, False, decode_tile_size=2)
        actual = _run(inputs, distribution, True, decode_tile_size=2)
        _assert_same(self, expected, actual)

    def test_same_results_with_prefill_checkpoints(self):
        # Sequence 3 (4 tokens) writes its state after 2 query tokens to block
        # 7; sequence 4's offset equals its length, so it has no checkpoint;
        # sequence 5 has no reserved block (-1).
        query_lens, context_lens = [1, 1, 4, 2, 3], [2, 6, 0, 3, 0]
        inputs = _make_inputs(query_lens, context_lens, num_padded_seqs=1)
        distribution = [2, 5, 5]
        prefill_ckpt = dict(
            prefill_ckpt_offsets=jnp.array([0, 0, 2, 2, 1, 0], dtype=jnp.int32),
            prefill_ckpt_indices=jnp.array([-1, -1, 7, 7, -1, -1], dtype=jnp.int32),
        )
        expected = _run(inputs, distribution, False, **prefill_ckpt)
        actual = _run(inputs, distribution, True, **prefill_ckpt)
        _assert_same(self, expected, actual)
        # The checkpoint block was written.
        self.assertFalse(
            np.array_equal(
                np.asarray(actual[0][1][7]), np.asarray(inputs["recurrent_state"][7])
            )
        )

    def test_one_call_when_enabled(self):
        inputs = _make_inputs([1] * 4, [3] * 4)
        distribution = [4, 4, 4]
        self.assertEqual(
            _kernel_names(inputs, distribution, True),
            ["fused_conv1d_gdn_batched_b4_c1_tables_in"],
        )
        self.assertEqual(
            _kernel_names(inputs, distribution, False),
            ["fused_conv1d_gdn_batched_b4_c1", "fused_conv1d_gdn_per_seq_b1_c4"],
        )

    @parameterized.named_parameters(
        dict(testcase_name="speculative", num_spec_tokens=1),
        dict(testcase_name="batched_only", batched_only=True),
        dict(testcase_name="prefill_only", prefill_only=True),
    )
    def test_two_calls_when_not_supported(self, **kwargs):
        inputs = _make_inputs([1] * 4, [3] * 4)
        if kwargs.get("num_spec_tokens"):
            kwargs["read_offsets"] = jnp.zeros((4,), dtype=jnp.int32)
        names = _kernel_names(inputs, [4, 4, 4], True, **kwargs)
        self.assertTrue(names)
        self.assertFalse(any(n.endswith("_tables_in") for n in names), names)

    def test_two_calls_above_max_rows(self):
        inputs = _make_inputs([1] * 260, [3] * 260)
        names = _kernel_names(inputs, [260, 260, 260], True)
        self.assertLen(names, 2)
        self.assertFalse(any(n.endswith("_tables_in") for n in names), names)


if __name__ == "__main__":
    absltest.main()
