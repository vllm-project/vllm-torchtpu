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
"""Shared QSA decode test cases: CPU stand-in for batched RPA, verify-step
inputs and a TPU7x lowering helper.

Batched RPA does not run in a Pallas interpret mode, so on CPU it is replaced
by a decode-only reference with identical math (``kernels`` fixture). That
checks everything around the kernel: routing, virtual page metadata,
query/gate layouts, K/V scales, paired-page head offsets, idle-row zeroing
and request chunking.
"""

import functools
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.mesh import AbstractDevice, AbstractMesh
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tests.models.qwen4_exp.experimental.kernels.test_paired_main_kv import (
    reference_rpa,
)
from vllm_torchtpu.models.qwen4_exp.experimental.kernels import qsa_rpa

RATIO, WIDTH, Q_HEADS = 4, 256, 3
TPU7X = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)


@pytest.fixture
def kernels(monkeypatch):
    """CPU stand-in for batched RPA; records its calls."""
    from vllm_torchtpu.models.qwen4_exp.experimental.kernels.batched_rpa import (
        wrapper,
    )

    original = pl.pallas_call
    monkeypatch.setattr(
        pl,
        "pallas_call",
        lambda *args, **kwargs: original(*args, **{**kwargs, "interpret": True}),
    )
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: TPU7X)
    calls = []

    def rpa(*args, **kwargs):
        calls.append(("rpa", args[0].shape, args[1].shape))
        return reference_rpa(*args, **kwargs)

    monkeypatch.setattr(wrapper, "ragged_paged_attention", rpa)
    return calls


def _cache(rng, pages, page, fp8=True):
    packing = 4 if fp8 else 2
    dtype = jnp.float8_e4m3fn if fp8 else jnp.bfloat16
    # Paired FP8 rows: layer slot s holds K/V at head slots 2s and 2s + 1.
    return jnp.asarray(
        rng.normal(0, 0.5, (pages, page, 4 // packing, packing, WIDTH)), dtype
    )


def _rows(rng, lengths, columns, groups, pages):
    """Per-token rows: selected groups (top-k within budget), tables, Q, gate."""
    lengths = np.asarray(lengths, np.int32)
    blocks = np.full((len(lengths), groups), -1, np.int32)
    for row, length in enumerate(lengths):
        count = min(length // RATIO, groups)
        if count:
            blocks[row, :count] = rng.choice(length // RATIO, count, replace=False)
    table = rng.integers(0, pages, (len(lengths), columns)).astype(np.int32)
    q = jnp.asarray(rng.normal(0, 0.7, (len(lengths), Q_HEADS, WIDTH)), jnp.bfloat16)
    gate = jnp.asarray(rng.normal(0, 2, q.shape), jnp.bfloat16)
    return q, gate, jnp.asarray(blocks), jnp.asarray(lengths), jnp.asarray(table)


def _verify_step(rng, contexts, tokens_per_request, rows, groups, page, columns):
    """An MTP verify step: K+1 rows per request, idle requests, padding rows."""
    starts, lengths = [0], []
    for context in contexts:
        n = tokens_per_request if context else 0
        lengths += [context + i + 1 for i in range(n)]
        starts.append(starts[-1] + n)
    lengths += [0] * (rows - len(lengths))
    pages = 2 * columns
    q, gate, blocks, lengths, table = _rows(rng, lengths, columns, groups, pages)
    starts = jnp.asarray(starts, jnp.int32)
    return (q, gate, blocks, lengths, table, _cache(rng, pages, page)), starts


def _attend(args, starts, **options):
    return jax.jit(
        functools.partial(
            qsa_rpa.qsa_attention,
            starts=starts,
            kv_heads=1,
            ratio=RATIO,
            k_scale=0.375,
            v_scale=2.5,
            **options,
        )
    )(*args)


def _lower(fn, *specs):
    device = AbstractDevice(device_kind="TPU7x", num_cores=1, platform="tpu")
    with jax.sharding.use_abstract_mesh(AbstractMesh((), (), abstract_device=device)):
        return (
            jax.jit(fn, donate_argnums=(5,))
            .trace(*specs)
            .lower(lowering_platforms=("tpu",))
            .as_text()
        )


def _schedules(text):
    """Backend configs of the RPA schedules, and the RPA kernel names."""
    calls = [line for line in text.splitlines() if "@tpu_custom_call" in line]
    names = [re.search(r'kernel_name = "([^"]+)"', c).group(1) for c in calls]
    return (
        [
            c[c.index("backend_config") :]
            for c, n in zip(calls, names)
            if n == "rpa_metadata_schedule"
        ],
        [n for n in names if n.startswith("RPA")],
    )
