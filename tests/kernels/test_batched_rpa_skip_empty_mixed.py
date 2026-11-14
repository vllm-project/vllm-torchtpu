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
"""Which kernel calls the batched RPA wrapper traces, with and without
``tuned_tiles``.

By default every call runs the DECODE and the MIXED kernel, each after its
schedule kernel, and nothing is conditional: a conditional around MIXED in
every attention layer changes how XLA compiles the whole step. With
``tuned_tiles`` (the Qwen3.5 PD deployments) a call of at most
``decode_query_size`` tokens runs DECODE alone, and any other call puts the
MIXED schedule and kernel inside a ``lax.cond`` whose other branch passes the
output and the KV cache through.
"""

import jax
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper

# Qwen2.5-1.5B-shaped layer: 12 q heads, 2 KV heads, head_dim 128; 8
# sequences of 16 pages of 128 tokens.
_Q_HEADS, _KV_HEADS, _HEAD_DIM = 12, 2, 128
_SEQS, _PAGES_PER_SEQ, _PAGE = 8, 16, 128
_LAYOUT = configs.KVLayout.HEAD_ALONG_SUBLANE


@pytest.fixture(autouse=True)
def tpu7x(monkeypatch):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda *args, **kwargs: info)


def _sub_jaxprs(params):
    for value in params.values():
        for sub in value if isinstance(value, (list, tuple)) else [value]:
            sub = getattr(sub, "jaxpr", sub)  # a ClosedJaxpr's Jaxpr
            if hasattr(sub, "eqns"):
                yield sub


def _kernel_calls(jaxpr):
    """(pallas_calls outside any cond, [pallas_calls per branch of each cond]).
    Kernel bodies are not entered."""
    outside = 0
    conds = []

    def count(j):
        n = 0
        for eqn in j.eqns:
            if eqn.primitive.name == "pallas_call":
                n += 1
            else:
                for sub in _sub_jaxprs(eqn.params):
                    n += count(sub)
        return n

    def walk(j):
        nonlocal outside
        for eqn in j.eqns:
            name = eqn.primitive.name
            if name == "pallas_call":
                outside += 1
            elif name == "cond":
                conds.append([count(b.jaxpr) for b in eqn.params["branches"]])
            else:
                for sub in _sub_jaxprs(eqn.params):
                    walk(sub)

    walk(jaxpr)
    return outside, conds


def _trace(tokens, *, decode_query_size=1, **kwargs):
    """The wrapper's jaxpr for one new sequence of `tokens` tokens (a 10-token
    cache) in a `tokens`-token bucket."""
    rng = np.random.default_rng(0)
    q = jnp.asarray(rng.standard_normal((tokens, _Q_HEADS, _HEAD_DIM)), jnp.bfloat16)
    k = jnp.asarray(rng.standard_normal((tokens, _KV_HEADS, _HEAD_DIM)), jnp.bfloat16)
    v = jnp.asarray(rng.standard_normal((tokens, _KV_HEADS, _HEAD_DIM)), jnp.bfloat16)
    kv_cache = jnp.zeros(
        wrapper.get_kv_cache_shape(
            _SEQS * _PAGES_PER_SEQ + 4,
            _PAGE,
            _KV_HEADS,
            _HEAD_DIM,
            jnp.bfloat16,
            kv_layout=_LAYOUT,
        ),
        jnp.bfloat16,
    )
    kv_lens = np.zeros(_SEQS, np.int32)
    kv_lens[0] = 10 + tokens
    cu_q_lens = np.zeros(_SEQS + 1, np.int32)
    cu_q_lens[1:] = tokens
    page_indices = np.zeros(_SEQS * _PAGES_PER_SEQ, np.int32)
    distribution = np.array([0, 1, 1], np.int32)

    def call(*args):
        return wrapper.ragged_paged_attention(
            *args,
            sm_scale=_HEAD_DIM**-0.5,
            kv_layout=_LAYOUT,
            decode_query_size=decode_query_size,
            **kwargs,
        )

    return jax.make_jaxpr(call)(
        q,
        k,
        v,
        kv_cache,
        jnp.asarray(kv_lens),
        jnp.asarray(page_indices),
        jnp.asarray(cu_q_lens),
        jnp.asarray(distribution),
    ).jaxpr


# (tokens, decode_query_size): a 16-token prefill bucket, and a call no larger
# than decode_query_size (MTP K=3), where tuned_tiles runs DECODE alone.
_CALLS = [(16, 1), (4, 4)]


@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize(("tokens", "decode_query_size"), _CALLS)
def test_by_default_decode_and_mixed_always_run(tokens, decode_query_size, return_lse):
    jaxpr = _trace(tokens, decode_query_size=decode_query_size, return_lse=return_lse)
    # Schedule + DECODE, schedule + MIXED; nothing conditional.
    assert _kernel_calls(jaxpr) == (4, [])


@pytest.mark.parametrize("return_lse", [False, True])
def test_tuned_tiles_skip_an_empty_mixed_call(return_lse):
    jaxpr = _trace(16, tuned_tiles=True, return_lse=return_lse)
    # Schedule + DECODE always; schedule + MIXED in the cond's true branch
    # (branches are ordered false, true), and the false branch passes through.
    assert _kernel_calls(jaxpr) == (2, [[0, 2]])


def test_tuned_tiles_run_decode_alone_on_a_call_of_decode_size():
    jaxpr = _trace(4, decode_query_size=4, tuned_tiles=True)
    assert _kernel_calls(jaxpr) == (2, [])
