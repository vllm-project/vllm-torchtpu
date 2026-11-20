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
"""The batched RPA kernels zero their KV buffer exactly when V is not masked.

P @ V multiplies masked scores' zero weights by every V row of a block, and
0 * NaN = NaN. With `mask_invalid_v` (default for SEQ_ALONG_LANE) every step
zeroes the V rows past each lane's KV end as it loads them, so the kernel
skips zeroing its KV buffer. Without it (default for HEAD_ALONG_SUBLANE) the
kernel zeroes the buffer once per call, so rows a step does not fetch never
hold NaN.
"""

import re

import jax
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper

_Q_HEADS, _KV_HEADS, _HEAD_DIM = 12, 2, 128
_SEQS, _PAGES_PER_SEQ, _PAGE = 8, 16, 128
_HEAD = configs.KVLayout.HEAD_ALONG_SUBLANE
_SEQ = configs.KVLayout.SEQ_ALONG_LANE

_VMEM_BF16 = re.compile(r"Ref<vmem>\{bf16\[([0-9,]+)\]\}")
# A u32 zeros constant, then a store of exactly that constant through the
# window's u32 [rows, 128] view.
_ZERO_STORE = re.compile(
    r"(\w+):u32\[([0-9]+),128\] = broadcast_in_dim 0:u32\[\]\n"
    r"\s*\w+:u32\[\2,128\], \w+\{bitcast\(uint32\)\}\{reshape\(\[\2, 128\]\)\} <- "
    r"\S+\{bitcast\(uint32\)\}\{reshape\(\[\2, 128\]\)\}, \1\n"
)


@pytest.fixture(autouse=True)
def tpu7x(monkeypatch):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda *args, **kwargs: info)


def _rpa_kernels(kv_layout, **kwargs):
    """{kernel name: kernel jaxpr text} of the RPA kernels (not the schedule
    kernels) the wrapper traces for a 16-token bucket with one new sequence."""
    rng = np.random.default_rng(0)
    tokens = 16
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
            kv_layout=kv_layout,
        ),
        jnp.bfloat16,
    )
    kv_lens = np.zeros(_SEQS, np.int32)
    kv_lens[0] = 10 + tokens
    cu_q_lens = np.zeros(_SEQS + 1, np.int32)
    cu_q_lens[1:] = tokens
    page_indices = np.zeros(_SEQS * _PAGES_PER_SEQ, np.int32)

    def call(*args):
        return wrapper.ragged_paged_attention(
            *args, sm_scale=_HEAD_DIM**-0.5, kv_layout=kv_layout, **kwargs
        )

    jaxpr = jax.make_jaxpr(call)(
        q,
        k,
        v,
        kv_cache,
        jnp.asarray(kv_lens),
        jnp.asarray(page_indices),
        jnp.asarray(cu_q_lens),
        jnp.asarray(np.array([0, 1, 1], np.int32)),
    ).jaxpr
    kernels = {}

    def walk(j):
        for eqn in j.eqns:
            if eqn.primitive.name == "pallas_call":
                name = eqn.params["name"]
                if name.startswith("RPA"):
                    kernels[name] = str(eqn.params["jaxpr"])
                continue
            for value in eqn.params.values():
                for sub in value if isinstance(value, (list, tuple)) else [value]:
                    sub = getattr(sub, "jaxpr", sub)
                    if hasattr(sub, "eqns"):
                        walk(sub)

    walk(jaxpr)
    assert len(kernels) == 2, kernels.keys()  # DECODE and MIXED
    return kernels


def _zeroes_kv_buffer(kernel_text):
    """Whether the kernel stores zeros over its whole triple-buffered KV window."""
    windows = {s for s in _VMEM_BF16.findall(kernel_text) if len(s.split(",")) == 6}
    assert len(windows) == 1, windows
    rows = int(np.prod([int(d) for d in windows.pop().split(",")])) * 2 // (4 * 128)
    return rows in {int(r) for _, r in _ZERO_STORE.findall(kernel_text)}


@pytest.mark.parametrize(
    ("kv_layout", "mask_invalid_v", "zeroed"),
    [
        (_HEAD, None, True),  # default: V not masked
        (_HEAD, True, False),
        (_SEQ, None, False),  # default: V masked on load
        (_SEQ, False, True),
    ],
)
def test_kv_buffer_zeroed_iff_v_is_not_masked(kv_layout, mask_invalid_v, zeroed):
    kernels = _rpa_kernels(kv_layout, mask_invalid_v=mask_invalid_v)
    for name, text in kernels.items():
        assert _zeroes_kv_buffer(text) == zeroed, name
