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

from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized

from vllm_torchtpu.layers.common import gdn_attention
from vllm_torchtpu.layers.common.gdn_attention import (
    GdnAttentionConfig, RaggedGatedDeltaRuleImpl, run_jax_gdn_attention_local)


class GDNAttentionTest(parameterized.TestCase):

    @parameterized.named_parameters(
        dict(
            testcase_name="prefill",
            max_reqs=1,
            lengths=[8192],
            q_loc=[0, 8192],
            distribution=[0, 0, 3],
            test_config=GdnAttentionConfig(
                ragged_gated_delta_rule_impl=RaggedGatedDeltaRuleImpl.
                CHUNKED_JAX_PD),
            ref_config=GdnAttentionConfig(
                ragged_gated_delta_rule_impl=RaggedGatedDeltaRuleImpl.REF),
        ),
        dict(
            testcase_name="mixed",
            max_reqs=3,
            lengths=[256, 128, 128],
            q_loc=[0, 256, 384, 512],
            distribution=[0, 3, 3],
            test_config=GdnAttentionConfig(
                ragged_gated_delta_rule_impl=RaggedGatedDeltaRuleImpl.
                CHUNKED_JAX_PD),
            ref_config=GdnAttentionConfig(
                ragged_gated_delta_rule_impl=RaggedGatedDeltaRuleImpl.REF),
        ),
        dict(
            testcase_name="decode_only",
            max_reqs=64,
            lengths=[1] * 64,
            q_loc=list(range(65)),
            distribution=[64, 64, 64],
            test_config=GdnAttentionConfig(
                ragged_gated_delta_rule_impl=RaggedGatedDeltaRuleImpl.
                CHUNKED_JAX_PD),
            ref_config=GdnAttentionConfig(
                ragged_gated_delta_rule_impl=RaggedGatedDeltaRuleImpl.REF),
        ),
        dict(
            testcase_name="mixed_prefill_decode",
            max_reqs=11,
            lengths=[1] * 8 + [128, 128, 256],
            q_loc=[0, 1, 2, 3, 4, 5, 6, 7, 8, 136, 264, 520],
            distribution=[8, 11, 11],
            test_config=GdnAttentionConfig(
                ragged_gated_delta_rule_impl=RaggedGatedDeltaRuleImpl.
                CHUNKED_JAX_PD),
            ref_config=GdnAttentionConfig(
                ragged_gated_delta_rule_impl=RaggedGatedDeltaRuleImpl.REF),
        ),
        dict(
            testcase_name="padded_mixed_prefill",
            max_reqs=16,
            lengths=[128, 64, 32, 16, 8],
            q_loc=[0, 128, 192, 224, 240, 248] + [1] * 11,
            distribution=[0, 5, 5],
            test_config=GdnAttentionConfig(
                ragged_gated_delta_rule_impl=RaggedGatedDeltaRuleImpl.
                CHUNKED_JAX_PD),
            ref_config=GdnAttentionConfig(
                ragged_gated_delta_rule_impl=RaggedGatedDeltaRuleImpl.REF),
        ),
        dict(
            testcase_name="padded_decode_only",
            max_reqs=512,
            lengths=[1] * 64,
            q_loc=list(range(65)) + [1] * 448,
            distribution=[64, 64, 64],
            test_config=GdnAttentionConfig(
                ragged_gated_delta_rule_impl=RaggedGatedDeltaRuleImpl.
                CHUNKED_JAX_PD),
            ref_config=GdnAttentionConfig(
                ragged_gated_delta_rule_impl=RaggedGatedDeltaRuleImpl.REF),
        ),
    )
    def test_run_jax_gdn_attention_local(self, max_reqs, lengths, q_loc,
                                         distribution, test_config,
                                         ref_config):
        kq_head_dim = 128
        v_head_dim = 128
        n_kq = 2
        n_v = 8
        kernel_size = 4

        num_tokens = sum(lengths)

        q_loc = jnp.array(q_loc)
        distribution = jnp.array(distribution, dtype=jnp.int32)

        # recurrent_state[0] and conv_state[0] are reserved for null blocks
        # (invalid / padded tokens). so start with index 1
        state_indices = jnp.arange(1, max_reqs + 1)
        num_blocks = max_reqs + 1

        rngs = iter(jax.random.split(jax.random.key(0), 12))

        query = jax.random.normal(next(rngs), (num_tokens, n_kq * kq_head_dim))
        key = jax.random.normal(next(rngs), (num_tokens, n_kq * kq_head_dim))
        value = jax.random.normal(next(rngs), (num_tokens, n_v * v_head_dim))
        b = jax.random.normal(next(rngs), (num_tokens, n_v))
        a = jax.random.normal(next(rngs), (num_tokens, n_v))

        conv_state_q = jnp.zeros(
            (num_blocks, kernel_size - 1, n_kq * kq_head_dim))
        conv_state_k = jnp.zeros(
            (num_blocks, kernel_size - 1, n_kq * kq_head_dim))
        conv_state_v = jnp.zeros(
            (num_blocks, kernel_size - 1, n_v * v_head_dim))
        recurrent_state = jnp.zeros((num_blocks, n_v, kq_head_dim, v_head_dim))

        conv_weight_q = jax.random.normal(next(rngs),
                                          (n_kq * kq_head_dim, 1, kernel_size))
        conv_weight_k = jax.random.normal(next(rngs),
                                          (n_kq * kq_head_dim, 1, kernel_size))
        conv_weight_v = jax.random.normal(next(rngs),
                                          (n_v * v_head_dim, 1, kernel_size))

        conv_bias_q = jax.random.normal(next(rngs), (n_kq * kq_head_dim, ))
        conv_bias_k = jax.random.normal(next(rngs), (n_kq * kq_head_dim, ))
        conv_bias_v = jax.random.normal(next(rngs), (n_v * v_head_dim, ))

        A_log = jax.random.normal(next(rngs), (n_v, ))
        dt_bias = jax.random.normal(jax.random.key(0), (n_v, ))

        mixed_qkv = jnp.concatenate([query, key, value], axis=-1)
        conv_state = jnp.concatenate(
            [conv_state_q, conv_state_k, conv_state_v], axis=-1)
        conv_weight = jnp.concatenate(
            [conv_weight_q, conv_weight_k, conv_weight_v], axis=0)
        conv_bias = jnp.concatenate([conv_bias_q, conv_bias_k, conv_bias_v],
                                    axis=-1)

        # All sequences in this test start from a fresh slot; the existing
        # parametrizations don't exercise prefix-cache-hit / chunked-prefill
        # continuation. ``seq_lens == query_lens`` (context_len = 0)
        # reproduces the prior behavior (zero initial state regardless of
        # slot contents).
        seq_lens = jnp.asarray(q_loc[1:max_reqs + 1] - q_loc[:max_reqs],
                               dtype=jnp.int32)

        run_jax_gdn_attention_local_jitted = jax.jit(
            run_jax_gdn_attention_local,
            static_argnames=[
                "n_kq", "n_v", "d_k", "d_v", "kernel_size", "dp_enabled",
                "config"
            ],
        )

        common_kwargs = dict(
            mixed_qkv=mixed_qkv,
            b=b,
            a=a,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            conv_weight=conv_weight,
            conv_bias=conv_bias,
            A_log=A_log,
            dt_bias=dt_bias,
            query_start_loc=q_loc,
            state_indices=state_indices,
            distribution=distribution,
            seq_lens=seq_lens,
            n_kq=n_kq,
            n_v=n_v,
            d_k=kq_head_dim,
            d_v=v_head_dim,
            kernel_size=kernel_size,
            dp_enabled=False,
        )

        # Run ref
        new_states_ref, output_ref = run_jax_gdn_attention_local_jitted(
            config=ref_config, **common_kwargs)

        # Run chunked
        new_states_chunked, output_chunked = run_jax_gdn_attention_local_jitted(
            config=test_config, **common_kwargs)

        # Compare results
        np.testing.assert_allclose(output_ref,
                                   output_chunked,
                                   rtol=2e-2,
                                   atol=2e-2)
        np.testing.assert_allclose(new_states_ref[0],
                                   new_states_chunked[0],
                                   rtol=2e-2,
                                   atol=2e-2)
        np.testing.assert_allclose(new_states_ref[1],
                                   new_states_chunked[1],
                                   rtol=2e-2,
                                   atol=2e-2)

    def test_chunked_kernel_v3_pd_routes_to_fused_conv1d_gdn(self):
        n_kq = 1
        n_v = 1
        d_k = 4
        d_v = 4
        kernel_size = 4
        max_reqs = 1
        num_tokens = 2
        qkv_dim = 2 * n_kq * d_k + n_v * d_v

        mixed_qkv = jnp.zeros((num_tokens, qkv_dim), dtype=jnp.float32)
        b = jnp.zeros((num_tokens, n_v), dtype=jnp.float32)
        a = jnp.zeros((num_tokens, n_v), dtype=jnp.float32)
        conv_state = jnp.zeros((max_reqs + 1, kernel_size - 1, qkv_dim),
                               dtype=jnp.float32)
        recurrent_state = jnp.zeros((max_reqs + 1, n_v, d_k, d_v),
                                    dtype=jnp.float32)
        conv_weight = jnp.zeros((qkv_dim, 1, kernel_size), dtype=jnp.float32)
        conv_bias = jnp.zeros((qkv_dim, ), dtype=jnp.float32)
        A_log = jnp.zeros((n_v, ), dtype=jnp.float32)
        dt_bias = jnp.zeros((n_v, ), dtype=jnp.float32)
        query_start_loc = jnp.array([0, num_tokens])
        state_indices = jnp.arange(1, max_reqs + 1)
        distribution = jnp.array([0, 1, 1], dtype=jnp.int32)
        seq_lens = jnp.asarray([num_tokens], dtype=jnp.int32)

        expected_conv_state = jnp.ones_like(conv_state)
        expected_recurrent_state = jnp.ones_like(recurrent_state)
        expected_output = jnp.ones((num_tokens, n_v * d_v), dtype=jnp.float32)
        config = GdnAttentionConfig(
            ragged_gated_delta_rule_impl=RaggedGatedDeltaRuleImpl.
            CHUNKED_KERNEL_V3_PD)

        with patch.object(
                gdn_attention.gdn_v3_wrapper,
                "fused_conv1d_gdn",
                return_value=((expected_conv_state, expected_recurrent_state),
                              expected_output),
        ) as mock_fused, patch.object(
                gdn_attention.causal_conv1d,
                "ragged_causal_conv1d",
                side_effect=AssertionError(
                    "V3 config should use fused Conv1D+GDN."),
        ) as mock_v2_conv:
            new_states, output = run_jax_gdn_attention_local(
                mixed_qkv=mixed_qkv,
                b=b,
                a=a,
                conv_state=conv_state,
                recurrent_state=recurrent_state,
                conv_weight=conv_weight,
                conv_bias=conv_bias,
                A_log=A_log,
                dt_bias=dt_bias,
                query_start_loc=query_start_loc,
                state_indices=state_indices,
                distribution=distribution,
                seq_lens=seq_lens,
                n_kq=n_kq,
                n_v=n_v,
                d_k=d_k,
                d_v=d_v,
                kernel_size=kernel_size,
                dp_enabled=False,
                config=config,
            )

        self.assertIs(new_states[0], expected_conv_state)
        self.assertIs(new_states[1], expected_recurrent_state)
        self.assertIs(output, expected_output)
        mock_v2_conv.assert_not_called()
        mock_fused.assert_called_once()
        self.assertIs(mock_fused.call_args.args[0], mixed_qkv)
        self.assertIs(mock_fused.call_args.args[3], conv_state)
        self.assertIs(mock_fused.call_args.args[4], recurrent_state)
        self.assertEqual(mock_fused.call_args.kwargs["n_kq"], n_kq)
        self.assertEqual(mock_fused.call_args.kwargs["n_v"], n_v)
        self.assertEqual(mock_fused.call_args.kwargs["d_k"], d_k)
        self.assertEqual(mock_fused.call_args.kwargs["d_v"], d_v)
        self.assertEqual(mock_fused.call_args.kwargs["kernel_size"],
                         kernel_size)
