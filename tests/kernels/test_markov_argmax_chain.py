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
"""kernels/collectives/markov_argmax_chain against the gathered greedy Markov
chain (bf16 bias and sum, argmax over the full vocab, first index on ties), on
8 CPU devices in interpret mode with the race detector (a subprocess, since the
device count is fixed when JAX starts)."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

N_RANKS = 8
SLOTS = 7
RANK_DIM = 256
# (rows, vocab shard, kind, last shard's pad columns): decode row buckets, a
# shard that is not a lane multiple, small-integer inputs whose scores tie
# within and across shards, NaN logits (torch.argmax picks the first NaN), and
# a vocab (of whole 16-row embedding tiles) that does not fill the last shard.
CASES = (
    (1, 300, "normal", 0),
    (2, 256, "normal", 0),
    (5, 300, "normal", 0),
    (2, 200, "tied", 0),
    (3, 300, "nan", 0),
    (3, 300, "normal", 48),
)


def _key(rows, vocab_shard, kind, pad):
    return f"{rows}x{vocab_shard}{kind}p{pad}"


def test_max_rows_fits_the_vmem_budget():
    from vllm_torchtpu.kernels.collectives import markov_argmax_chain as chain

    # MiniMax-M3.1-Flash's draft at TP8: 7 slots, rank 256, 25008-id shards.
    budget = 48 << 20  # 3/4 of a tpu7x TensorCore's VMEM
    rows = chain.max_rows(N_RANKS, SLOTS, RANK_DIM, 25008, budget)
    assert rows == chain.MAX_ROWS
    assert chain.vmem_bytes(N_RANKS, rows, SLOTS, RANK_DIM, 25008) <= budget
    assert chain.max_rows(N_RANKS, SLOTS, RANK_DIM, 25008, 20 << 20) == 0
    with pytest.raises(ValueError, match="at most"):
        chain._call(N_RANKS, rows + 1, SLOTS, RANK_DIM, 25008)


def _inputs(rows, vocab_shard, kind, pad, seed=0):
    rng = np.random.default_rng(seed)
    vocab = N_RANKS * vocab_shard - pad
    if kind == "tied":
        logits = rng.integers(-2, 3, (N_RANKS, rows, SLOTS, vocab_shard))
        # A zero bias leaves many maxima in every shard and slot.
        w2 = np.zeros((vocab, RANK_DIM))
        w1 = rng.integers(-1, 2, (vocab, RANK_DIM))
    else:
        logits = rng.normal(0, 4, (N_RANKS, rows, SLOTS, vocab_shard))
        w2 = rng.normal(0, 0.1, (vocab, RANK_DIM))
        w1 = rng.normal(0, 1, (vocab, RANK_DIM))
    if kind == "nan":
        # Row 0: a NaN in one slot on two shards; row 1: NaNs in every slot
        # of one shard; row 2: none.
        logits[3, 0, 2, 17] = logits[5, 0, 2, 4] = np.nan
        logits[6, 1, :, 100] = np.nan
    prev = rng.integers(0, vocab, rows)
    return logits.astype(np.float32), w2.astype(np.float32), w1.astype(np.float32), prev


def _reference(logits, w2, w1, prev):
    """The gathered path: per slot, bf16(logits + bf16(w1[prev] @ w2.T)) over
    the full vocab, first index among the maxima (or of the first NaN), as
    torch.argmax."""
    import jax.numpy as jnp

    vocab = w2.shape[0]
    full = np.concatenate(list(logits), axis=-1)[..., :vocab]
    full = jnp.asarray(full, jnp.bfloat16)
    w2b, w1b = jnp.asarray(w2, jnp.bfloat16), jnp.asarray(w1, jnp.bfloat16)
    ids, prev = [], jnp.asarray(prev)
    for s in range(SLOTS):
        bias = jnp.dot(w1b[prev], w2b.T, preferred_element_type=jnp.float32)
        scores = full[:, s] + bias.astype(jnp.bfloat16)
        prev = jnp.asarray(np.argmax(np.asarray(scores, np.float32), axis=-1))
        ids.append(prev)
    return np.asarray(jnp.stack(ids, axis=1))


def _worker():
    import jax
    import jax.numpy as jnp
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.kernels.collectives import markov_argmax_chain as chain

    mesh = Mesh(np.asarray(sorted(jax.devices("cpu"), key=lambda d: d.id)), ("d",))
    run = jax.jit(
        jax.shard_map(
            lambda lg, w2t, w1, prev, r: chain.markov_argmax_chain(
                lg[0], w2t[0], w1[0], prev[0], r, N_RANKS
            )[None],
            mesh=mesh,
            in_specs=(P("d"),) * 5,
            out_specs=P("d"),
            check_vma=False,
        )
    )
    result = {}
    for rows, vocab_shard, kind, pad in CASES:
        logits, w2, w1, prev = _inputs(rows, vocab_shard, kind, pad)
        lanes = chain.padded_vocab(vocab_shard)
        w2t = np.zeros((N_RANKS, RANK_DIM, lanes), np.float32)
        scalars = np.zeros((N_RANKS, 3), np.int32)
        for d in range(N_RANKS):
            part = w2[d * vocab_shard : (d + 1) * vocab_shard]
            w2t[d, :, : len(part)] = part.T
            scalars[d] = (d, d * vocab_shard, len(part))
        pltpu.reset_tpu_interpret_mode_state()
        out = run(
            jnp.asarray(logits, jnp.bfloat16),
            jnp.asarray(w2t, jnp.bfloat16),
            jnp.asarray(np.broadcast_to(w1, (N_RANKS, *w1.shape)), jnp.bfloat16),
            jnp.asarray(np.broadcast_to(prev, (N_RANKS, rows)), jnp.int32),
            jnp.asarray(scalars),
        )
        out = np.asarray(out)
        want = _reference(logits, w2, w1, prev)
        result[_key(rows, vocab_shard, kind, pad)] = dict(
            exact=all(np.array_equal(out[d], want) for d in range(N_RANKS)),
            races=bool(interpret_pallas_call.races.races_found),
            got=out[0].tolist(),
            want=want.tolist(),
        )
    print("RESULT " + json.dumps(result))


@pytest.fixture(scope="module")
def on_8_devices():
    flags = os.environ.get("XLA_FLAGS", "")
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=f"{flags} --xla_force_host_platform_device_count={N_RANKS}",
    )
    proc = subprocess.run(
        [sys.executable, __file__], env=env, capture_output=True, text=True, timeout=900
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stdout[-3000:] + proc.stderr[-3000:]
    return json.loads(lines[-1][len("RESULT ") :])


@pytest.mark.parametrize("rows, vocab_shard, kind, pad", CASES)
def test_kernel_on_8_devices(on_8_devices, rows, vocab_shard, kind, pad):
    case = on_8_devices[_key(rows, vocab_shard, kind, pad)]
    assert case["exact"] and not case["races"], case


if __name__ == "__main__":
    from jax.experimental.pallas import tpu as pltpu

    with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams(detect_races=True)):
        _worker()
