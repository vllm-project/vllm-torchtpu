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
"""The Qwen3.5 GDN copy's state DMA waits.

A decode tile of T sequences with a verify window of W tokens writes W state
checkpoints per sequence out of a double-buffered window, and the next tile
reuses that buffer once its writes have been waited for. StateInBufferedRef
and StateOutBufferedRef wait for each copy they issued. The single aggregate
wait they used before, over T * W checkpoints, sliced past the window axis
for W > 1 at T >= 2.

Running the kernel does not tell the two apart: with the aggregate waits put
back, CPU interpret mode gives the same outputs and states as with the
per-copy waits under every DMA execution mode and with race detection on,
and so does a TPU7x. So the regression check is structural, on CPU: in the
traced kernel, every state semaphore must be waited on once per copy issued
on it. With the aggregate waits put back each wait site has one wait, and
the check fails.

The TPU test repeats W = 4 windows over many decode tiles (T = 2 and T = 4)
and requires bit-identical outputs and states every time.
"""

import collections

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from vllm_torchtpu.models.qwen3_5.kernels.gdn import wrapper

N_KQ, N_V, D, KERNEL_SIZE = 2, 8, 128, 4
DIM = 2 * N_KQ * D + N_V * D
WINDOW, NUM_SEQS, RUNS = 4, 64, 5
NUM_BLOCKS = (NUM_SEQS + 1) * WINDOW


def _verify_batch(seed=7):
    rngs = iter(jax.random.split(jax.random.key(seed), 9))
    num_tokens = NUM_SEQS * WINDOW
    return dict(
        qkv=jax.random.normal(next(rngs), (num_tokens, DIM), jnp.bfloat16),
        b=jax.random.normal(next(rngs), (num_tokens, N_V), jnp.bfloat16),
        a=jax.random.normal(next(rngs), (num_tokens, N_V), jnp.bfloat16),
        states=(
            jax.random.normal(next(rngs), (NUM_BLOCKS, KERNEL_SIZE - 1, DIM)),
            jax.random.normal(next(rngs), (NUM_BLOCKS, N_V, D, D)),
        ),
        conv_weight=jax.random.normal(next(rngs), (DIM, 1, KERNEL_SIZE), jnp.bfloat16),
        conv_bias=jax.random.normal(next(rngs), (DIM,), jnp.bfloat16),
        a_log=jax.random.normal(next(rngs), (N_V,), jnp.float32),
        dt_bias=jax.random.normal(next(rngs), (N_V,), jnp.float32),
        query_start_loc=jnp.arange(NUM_SEQS + 1, dtype=jnp.int32) * WINDOW,
        state_indices=jnp.arange(1, NUM_SEQS + 1, dtype=jnp.int32) * WINDOW,
        distribution=jnp.array([NUM_SEQS] * 3, jnp.int32),
        seq_lens=jnp.full((NUM_SEQS,), 20, jnp.int32),
        read_offsets=jnp.zeros((NUM_SEQS,), jnp.int32),
    )


def _call(decode_tile_size, bf16_qkv_mode):
    return dict(
        n_kq=N_KQ,
        n_v=N_V,
        d_k=D,
        d_v=D,
        kernel_size=KERNEL_SIZE,
        num_spec_tokens=WINDOW - 1,
        decode_tile_size=decode_tile_size,
        batched_only=True,
        bf16_qkv_mode=bf16_qkv_mode,
    )


# --- CPU: one wait per state copy in the traced kernel -------------------------


def _nested(eqn):
    """(jaxpr, its vars, the outer atoms bound to them) for each jaxpr nested
    in `eqn`. A run_scoped body's invars are the refs it allocates, so they
    stay unbound: each is its own root. Its constvars are the outer refs it
    closes over."""
    params, name = eqn.params, eqn.primitive.name
    if name == "cond":
        return [(b.jaxpr, b.jaxpr.invars, eqn.invars[1:]) for b in params["branches"]]
    if name == "while":
        nc, nb = params["cond_nconsts"], params["body_nconsts"]
        carry = eqn.invars[nc + nb :]
        cond, body = params["cond_jaxpr"].jaxpr, params["body_jaxpr"].jaxpr
        return [
            (cond, cond.invars, [*eqn.invars[:nc], *carry]),
            (body, body.invars, [*eqn.invars[nc : nc + nb], *carry]),
        ]
    if name == "run_scoped":
        body = params["jaxpr"]
        bound = len(body.constvars) == len(eqn.invars)
        return [(body, body.constvars if bound else [], eqn.invars if bound else [])]
    nested = []
    for value in params.values():
        inner = getattr(value, "jaxpr", value)
        if hasattr(inner, "eqns") and len(inner.invars) == len(eqn.invars):
            nested.append((inner, inner.invars, eqn.invars))
    return nested


def _dmas(jaxpr, roots, site, out):
    """Append (primitive, site, [(root ref, atom)]) for every DMA start and
    wait under `jaxpr`. A site is the block of the kernel the DMA sits in;
    `roots` maps a nested block's vars to the refs they stand for."""
    for i, eqn in enumerate(jaxpr.eqns):
        if eqn.primitive.name in ("dma_start", "dma_wait"):
            refs = [(roots.get(a, a), a) for a in eqn.invars if not hasattr(a, "val")]
            out.append((eqn.primitive.name, site, refs))
        for k, (inner, inner_vars, outer) in enumerate(_nested(eqn)):
            bound = {
                v: roots.get(a, a)
                for v, a in zip(inner_vars, outer, strict=True)
                if not hasattr(a, "val")
            }
            _dmas(inner, bound, (*site, i, k), out)


def _is_semaphore(atom):
    aval = getattr(atom.aval, "inner_aval", atom.aval)
    return "sem" in str(getattr(aval, "dtype", ""))


def _is_state_pool(atom):
    return tuple(getattr(atom.aval, "shape", ()))[:1] == (NUM_BLOCKS,)


def _state_dma_counts(decode_tile_size, bf16_qkv_mode):
    """{state semaphore: {"dma_start": [...], "dma_wait": [...]}}: per
    semaphore of a copy to or from the state blocks, the number of DMAs at
    each site of the traced kernel. The pipeline issues a buffer's copies,
    and its waits, from blocks of their own (the first step's and the steady
    state's), so a site's count is what one copy or wait call issues."""
    inputs = _verify_batch()
    call = _call(decode_tile_size, bf16_qkv_mode)
    traced = jax.make_jaxpr(
        lambda kw: wrapper.fused_conv1d_gdn.__wrapped__(**kw, **call)
    )(inputs)
    kernels, stack = [], [traced.jaxpr]
    while stack:
        for eqn in stack.pop().eqns:
            if eqn.primitive.name == "pallas_call":
                kernels.append(eqn.params["jaxpr"])
            else:
                stack.extend(inner for inner, _, _ in _nested(eqn))
    assert kernels
    dmas = []
    for kernel in kernels:
        _dmas(kernel, {}, (), dmas)
    state_sems = {
        root
        for prim, _, refs in dmas
        if prim == "dma_start" and any(_is_state_pool(a) for _, a in refs)
        for root, a in refs
        if _is_semaphore(a)
    }
    per_site = collections.Counter(
        (root, prim, site)
        for prim, site, refs in dmas
        for root, a in refs
        if _is_semaphore(a) and root in state_sems
    )
    counts = {s: {"dma_start": [], "dma_wait": []} for s in state_sems}
    for (root, prim, _), n in per_site.items():
        counts[root][prim].append(n)
    return counts


@pytest.mark.cpu_test
@pytest.mark.parametrize("decode_tile_size", [2, 4])
@pytest.mark.parametrize("bf16_qkv_mode", [0, 2])
def test_every_state_copy_has_its_own_wait(decode_tile_size, bf16_qkv_mode):
    """Each of the four state buffers (conv and recurrent, in and out) waits
    once per copy it issued: each copy site issues T copies (initial states)
    or T * W (checkpoints), and each wait site as many waits."""
    mesh = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(mesh):
        counts = _state_dma_counts(decode_tile_size, bf16_qkv_mode)
    copies = []
    for sem, sites in counts.items():
        starts = set(sites["dma_start"])
        assert len(starts) == 1, (sem, sites)
        assert sites["dma_wait"] and set(sites["dma_wait"]) == starts, (sem, sites)
        copies += starts
    tile, window = decode_tile_size, WINDOW
    assert sorted(copies) == [tile, tile, tile * window, tile * window], counts


@pytest.mark.tpu_test
@pytest.mark.parametrize("decode_tile_size", [2, 4])
@pytest.mark.parametrize("bf16_qkv_mode", [0, 2])
def test_verify_window_checkpoints_are_identical_across_runs_on_tpu(
    decode_tile_size, bf16_qkv_mode
):
    if jax.devices()[0].platform != "tpu":
        pytest.skip("the state DMA waits only matter on a TPU")
    inputs = _verify_batch()
    results = []
    for _ in range(RUNS):
        # fused_conv1d_gdn donates `states`, so each run gets its own copy.
        call_inputs = dict(inputs, states=jax.tree.map(jnp.copy, inputs["states"]))
        results.append(
            jax.block_until_ready(
                wrapper.fused_conv1d_gdn(
                    **call_inputs, **_call(decode_tile_size, bf16_qkv_mode)
                )
            )
        )
    first = [np.asarray(x) for x in jax.tree.leaves(results[0])]
    for run in results[1:]:
        for got, want in zip(jax.tree.leaves(run), first, strict=True):
            np.testing.assert_array_equal(np.asarray(got), want)
