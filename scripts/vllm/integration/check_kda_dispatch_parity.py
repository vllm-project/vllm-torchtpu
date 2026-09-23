# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare KDA dispatch with a pinned pre-cleanup wrapper on the same TPU.

KDA_BASELINE_WRAPPER must point to wrapper.py from the base commit. Each
case checks output, every checkpoint, untouched padding and repeated rollback.
The baseline runs the two guarded calls used by the pre-cleanup KDA adapter.
"""

import importlib.util
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.gdn.v3 import config, wrapper


def load_baseline():
    spec = importlib.util.spec_from_file_location(
        "kda_dispatch_baseline", os.environ["KDA_BASELINE_WRAPPER"]
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_case(baseline, num_heads, dtype, lengths, verify_count, spec_tokens):
    window = spec_tokens + 1
    n = len(lengths)
    dim = 3 * num_heads * 128
    rng = np.random.default_rng(42)
    qloc = np.r_[0, np.cumsum(lengths)].astype(np.int32)
    # Include inactive tokens and unowned state slots with nonzero contents.
    # The chunked prefill solve requires whole triangular blocks. Use
    # serving-style padded buckets while keeping inactive tail tokens.
    tokens = ((int(qloc[-1]) + 3 + 63) // 64) * 64
    blocks = n * window + 3

    def rand(shape, dt=jnp.float32):
        return jnp.asarray(rng.normal(0, 0.1, shape), dt)

    common = dict(
        qkv=rand((tokens, dim), dtype),
        b=rand((tokens, num_heads), dtype),
        a=rand((tokens, num_heads * 128), dtype),
        conv_weight=rand((dim, 1, 4), dtype),
        conv_bias=None,
        a_log=rand((num_heads,)),
        dt_bias=rand((num_heads, 128)),
        query_start_loc=jnp.asarray(qloc),
        state_indices=jnp.asarray(1 + np.arange(n) * window, jnp.int32),
        distribution=jnp.asarray([verify_count, n, n], jnp.int32),
        seq_lens=jnp.asarray(lengths, jnp.int32) + 32,
        n_kq=num_heads,
        n_v=num_heads,
        d_k=128,
        d_v=128,
        kernel_size=4,
        attention_mode=config.AttentionMode.KDA,
        gate_lower_bound=-5.0,
        num_spec_tokens=spec_tokens,
    )
    conv = rand((blocks, 3, dim))
    rec = rand((blocks, num_heads, 128, 128))

    def old_dispatch(conv_in, rec_in, offsets):
        args = dict(common, read_offsets=offsets)
        if spec_tokens == 0:
            return baseline.fused_conv1d_gdn(
                **args, conv_state=conv_in, recurrent_state=rec_in
            )
        empty = jnp.zeros((tokens, num_heads * 128), dtype)
        (states, verify_out) = jax.lax.cond(
            args["distribution"][0] > 0,
            lambda: baseline.fused_conv1d_gdn(
                **args, conv_state=conv_in, recurrent_state=rec_in, batched_only=True
            ),
            lambda: ((conv_in, rec_in), empty),
        )
        first = args["query_start_loc"][args["distribution"][0]]
        (states, prefill_out) = jax.lax.cond(
            first < jnp.minimum(args["query_start_loc"][-1], tokens),
            lambda: baseline.fused_conv1d_gdn(
                **args,
                conv_state=states[0],
                recurrent_state=states[1],
                prefill_only=True,
            ),
            lambda: (states, empty),
        )
        return states, jnp.where(
            (jnp.arange(tokens) < first)[:, None], verify_out, prefill_out
        )

    old = jax.jit(old_dispatch)
    new = jax.jit(
        lambda c, r, o: wrapper.fused_conv1d_gdn(
            **common, conv_state=c, recurrent_state=r, read_offsets=o
        )
    )
    old_state = (conv.copy(), rec.copy())
    new_state = (conv.copy(), rec.copy())
    errors = []
    for step in range(4):
        offsets = np.zeros(n, np.int32)
        for i in range(verify_count):
            offsets[i] = step % max(1, lengths[i])
        offsets = jnp.asarray(offsets)
        old_state, old_out = old(*old_state, offsets)
        new_state, new_out = new(*new_state, offsets)
        jax.block_until_ready((old_state, old_out, new_state, new_out))
        for name, actual, expected in zip(
            ["output", "conv_checkpoints", "recurrent_checkpoints"],
            [new_out, *new_state],
            [old_out, *old_state],
        ):
            actual, expected = np.asarray(actual), np.asarray(expected)
            # A dispatch-only refactor must preserve bitwise results, not just
            # fit the broader reference-kernel numerical tolerance.
            np.testing.assert_array_equal(
                actual, expected, err_msg=f"{name}, step={step}"
            )
            errors.append(
                float(np.max(np.abs(actual.astype(float) - expected.astype(float))))
            )
    return dict(
        num_heads=num_heads,
        dtype=str(dtype),
        lengths=lengths,
        verify_count=verify_count,
        spec_tokens=spec_tokens,
        iterations=4,
        max_abs_error=max(errors),
        bitwise_equal=True,
    )


def main():
    assert jax.default_backend() == "tpu", jax.devices()
    baseline = load_baseline()
    output = Path(os.environ["OUT"]) / "benchmark"
    output.mkdir(parents=True, exist_ok=True)
    cases = [
        ([1, 1, 1, 1], 4, 0),
        ([6, 6, 6, 6], 4, 5),
        ([1, 3, 6, 0], 4, 5),
        ([6, 3, 17, 8], 2, 5),
        ([17, 8, 0, 0], 0, 5),
        ([0, 0, 0, 0], 0, 5),
    ]
    with (output / "metrics.jsonl").open("w", buffering=1) as f:
        for heads in [3, 12]:
            for dtype in [jnp.bfloat16, jnp.float32]:
                for lengths, count, k in cases:
                    result = check_case(baseline, heads, dtype, lengths, count, k)
                    f.write(json.dumps(result) + "\n")
                    print("PARITY_PASS", json.dumps(result), flush=True)
        for n in [64, 128]:
            result = check_case(baseline, 3, jnp.bfloat16, [6] * n, n, 5)
            f.write(json.dumps(result) + "\n")
            print("PARITY_PASS", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
