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

import os

# Set up VMEM limit for TPU
os.environ["LIBTPU_INIT_ARGS"] = (os.environ.get("LIBTPU_INIT_ARGS", "") +
                                  " --xla_tpu_scoped_vmem_limit_kib=65536")

import traceback

import jax
import jax.numpy as jnp
import numpy as np

import vllm_torchtpu.kernels.deepseek_v4.mla as mla_module

mla_ragged_paged_attention = mla_module.mla_ragged_paged_attention


def main():
    print("Imported mla from:", mla_module.__file__)
    # Initialize JAX TPU
    print("JAX devices:", jax.devices())

    # Hyperparameters (matching a single TPU device in the 8-way TP run)
    num_q_heads = 16
    head_dim = 512
    lkv_dim = 640  # aligned dimension for packed KV (448 nope + 128 rope + padding)
    page_size = 256
    q_dtype = jnp.bfloat16
    kv_dtype = jnp.uint8

    # We test a single sequence with 1 query token, but 0 KV tokens to attend to in main attention
    # (e.g., because kv_len < sliding_window)
    q_len = 1
    kv_len = 0

    # Generate dummy inputs
    rng = np.random.default_rng(1234)

    # q: [total_q_len, num_q_heads, head_dim]
    q = jnp.array(rng.standard_normal(size=(q_len, num_q_heads,
                                            head_dim))).astype(q_dtype)

    # cache_kv: [num_pages, page_size // 4, 4, lkv_dim]
    num_pages = 2
    packing = 4
    cache_kv = jnp.zeros((num_pages, page_size // packing, packing, lkv_dim),
                         dtype=kv_dtype)

    # kv_lens: [num_seqs]
    kv_lens = jnp.array([kv_len], dtype=jnp.int32)

    # kv_lens_to_attend: [total_q_len]
    # For main attention, we attend to 0 tokens
    kv_lens_to_attend = jnp.array([0], dtype=jnp.int32)

    # page_indices: [num_seqs * pages_per_seq]
    page_indices = jnp.array([0], dtype=jnp.int32)

    # cu_q_lens: [num_seqs + 1]
    cu_q_lens = jnp.array([0, q_len], dtype=jnp.int32)

    # distribution: [3] (decode_only, prefill_only, mixed)
    # We treat it as a prefill sequence (since q_len = 1, kv_len = 0, it is the first token)
    distribution = jnp.array([0, 1, 1], dtype=jnp.int32)

    # attention_sinks: [num_q_heads]
    attention_sinks = jnp.full((num_q_heads, ), -jnp.inf, dtype=jnp.float32)

    # Case 1: SWA disabled (initialized to 0/ -inf)
    print("\n--- Running Case 1: SWA Disabled ---")
    swa_accumulation = jnp.zeros((q_len, num_q_heads, head_dim), dtype=q_dtype)
    swa_l = jnp.zeros((q_len, num_q_heads), dtype=jnp.float32)
    swa_m = jnp.full((q_len, num_q_heads), -jnp.inf, dtype=jnp.float32)

    try:
        out = mla_ragged_paged_attention(
            q=q,
            cache_kv=cache_kv,
            kv_lens=kv_lens,
            kv_lens_to_attend=kv_lens_to_attend,
            topk_indices=None,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            attention_sinks=attention_sinks,
            swa_accumulation=swa_accumulation,
            swa_l=swa_l,
            swa_m=swa_m,
            sm_scale=1.0 / (head_dim**0.5),
            num_kv_pages_per_block=1,
            num_queries_per_block=1,
        )
        # Force compilation and execution
        out.block_until_ready()

        print("Output shape:", out.shape)
        has_nan = jnp.isnan(out).any().item()
        print("Output contains NaN:", has_nan)
        # Print output sample
        print("Output sample (first 10 channels of first token/head 0):")
        print(out[0, 0, :10])

    except Exception:
        print("Failed with error:")
        traceback.print_exc()

    # Case 2: SWA enabled (initialized with dummy finite values)
    print("\n--- Running Case 2: SWA Enabled ---")
    # Simulate that SWA has processed some tokens and has finite max/sum
    swa_accumulation = jnp.array(
        rng.standard_normal(size=(q_len, num_q_heads,
                                  head_dim))).astype(q_dtype)
    swa_l = jnp.full((q_len, num_q_heads), 128.0, dtype=jnp.float32)
    swa_m = jnp.full((q_len, num_q_heads),
                     -0.7 * float(jnp.finfo(jnp.dtype("float32")).max),
                     dtype=jnp.float32)

    try:
        out = mla_ragged_paged_attention(
            q=q,
            cache_kv=cache_kv,
            kv_lens=kv_lens,
            kv_lens_to_attend=kv_lens_to_attend,
            topk_indices=None,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            attention_sinks=attention_sinks,
            swa_accumulation=swa_accumulation,
            swa_l=swa_l,
            swa_m=swa_m,
            sm_scale=1.0 / (head_dim**0.5),
            num_kv_pages_per_block=1,
            num_queries_per_block=1,
        )
        out.block_until_ready()

        print("Output shape:", out.shape)
        has_nan = jnp.isnan(out).any().item()
        print("Output contains NaN:", has_nan)
        # Print output sample
        print("Output sample (first 10 channels of first token/head 0):")
        print(out[0, 0, :10])

    except Exception:
        print("Failed with error:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
