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
"""The array-op decode schedule equals the schedule kernel where it is read.

Every case runs `schedule.generate_rpa_metadata` (natively on a TPU, in the
TPU interpreter elsewhere) and `schedule_xla.generate_rpa_metadata` on the same
inputs; the step count and every entry of the steps the attention kernel reads
must be equal.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.batched_rpa import (
    configs,
    schedule,
    schedule_xla,
)

_SEQ = configs.KVLayout.SEQ_ALONG_LANE
_HEAD = configs.KVLayout.HEAD_ALONG_SUBLANE


def _on_tpu():
    # Asked inside tests only: initializing a backend at collection is refused.
    return jax.default_backend() == "tpu"


@pytest.fixture(autouse=True)
def _tpu_info(monkeypatch):
    if _on_tpu():
        return
    try:
        from jax._src import tpu_info
    except ImportError:
        from jax._src.pallas.mosaic import tpu_info
    info = tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)


def _cfgs(num_seqs, page, pages_per_seq, bkv, batch, query, bq, layout, skip=False):
    serve = configs.ServingConfigs(
        num_seqs=num_seqs,
        page_size=page,
        total_q_tokens=num_seqs * query,
        num_page_indices=num_seqs * pages_per_seq,
        dtype_q=jnp.dtype(jnp.bfloat16),
        dtype_kv=jnp.dtype(jnp.float8_e4m3fn),
        dtype_out=jnp.dtype(jnp.bfloat16),
        kv_layout=layout,
        decode_query_size=query,
        skip_kv_update=skip,
    )
    return configs.RpaConfigs(
        block=configs.BlockSizes(
            bq_sz=bq, bq_c_sz=bq, bkv_sz=bkv, batch_size=batch, n_buffer=2
        ),
        model=configs.ModelConfigs(
            num_q_heads=16, num_kv_heads=1, head_dim=256, mask_value=0.0
        ),
        serve=serve,
        mode=configs.RpaCase.DECODE,
        vmem_limit_bytes=64 << 20,
    )


def _inputs(seed, cfgs, decode_seqs, empty=0, kv_range=None, full_windows=False):
    rng = np.random.default_rng(seed)
    serve = cfgs.serve
    n, query = serve.num_seqs, serve.decode_query_size
    if full_windows:
        q_lens = np.full(n, query, np.int32)
    else:
        q_lens = rng.integers(1, query + 1, n).astype(np.int32)
    if empty:
        q_lens[rng.choice(n, empty, replace=False)] = 0
    lo, hi = kv_range or (1, serve.pages_per_seq * serve.page_size - query)
    kv_lens = np.maximum(rng.integers(lo, hi, n), q_lens).astype(np.int32)
    cu_q_lens = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    cache_lens = kv_lens - q_lens
    distribution = np.array([decode_seqs, n, n], np.int32)
    return tuple(
        jnp.asarray(a)
        for a in (cu_q_lens, cache_lens, cache_lens, q_lens, distribution)
    )


def _assert_same_read_entries(cfgs, args):
    interpret = False if _on_tpu() else pltpu.InterpretParams()
    ref = schedule.generate_rpa_metadata(*args, cfgs, interpret=interpret)
    new = jax.jit(schedule_xla.generate_rpa_metadata, static_argnames="cfgs")(
        *args, cfgs=cfgs
    )
    steps = int(np.asarray(ref.actual_steps)[0])
    assert int(np.asarray(new.actual_steps)[0]) == steps
    total_steps = cfgs.max_steps_ub * cfgs.max_schedule_size_multiplier
    for field in dataclasses.fields(ref):
        if field.name in ("cfgs", "actual_steps", "chunk_steps"):
            continue
        ref_data = np.asarray(getattr(ref, field.name).data)
        new_data = np.asarray(getattr(new, field.name).data)
        assert ref_data.shape == new_data.shape, field.name
        read = steps * (ref_data.shape[0] // total_steps)
        np.testing.assert_array_equal(new_data[:read], ref_data[:read], field.name)
    return steps


@pytest.mark.parametrize(
    "case",
    [
        # seqs, page, pages/seq, bkv, batch, queries, bq, layout, skip, decode, empty
        (8, 16, 64, 64, 4, 1, 1, _SEQ, False, 8, 0),
        (8, 16, 64, 64, 4, 4, 4, _SEQ, False, 8, 0),
        (8, 16, 64, 64, 4, 4, 4, _SEQ, False, 5, 2),
        (8, 16, 64, 64, 4, 2, 4, _SEQ, False, 8, 0),
        (8, 16, 16, 16, 4, 4, 4, _SEQ, False, 8, 0),
        (8, 16, 64, 64, 4, 1, 1, _SEQ, True, 8, 0),
        (8, 16, 64, 64, 4, 1, 1, _HEAD, False, 8, 0),
        (8, 16, 64, 64, 4, 4, 4, _HEAD, False, 6, 1),
        (8, 16, 16, 16, 4, 4, 4, _HEAD, False, 8, 0),
        (6, 128, 40, 384, 8, 4, 4, _SEQ, False, 6, 0),
    ],
)
def test_small_decode_batches(case):
    *shape, decode, empty = case
    cfgs = _cfgs(*shape)
    assert schedule_xla.supported(cfgs)
    for seed in range(2):
        _assert_same_read_entries(cfgs, _inputs(seed, cfgs, decode, empty))


@pytest.mark.parametrize("query", [1, 4])
def test_served_draft_shape_across_the_smem_chunk(query):
    # 24 x ~128K-token sequences at 3072-token KV blocks: 1032 tasks fill one
    # 128-step chunk of 8 lanes and spill into a second.
    cfgs = _cfgs(24, 128, 1664, 3072, 8, query, query, _SEQ)
    args = _inputs(7, cfgs, 24, kv_range=(130_500, 131_072), full_windows=True)
    assert _assert_same_read_entries(cfgs, args) > cfgs.max_steps_ub


def test_unsupported_configs_keep_the_kernel():
    base = _cfgs(8, 16, 64, 64, 4, 4, 4, _SEQ)
    assert schedule_xla.supported(base)
    mixed = dataclasses.replace(base, mode=configs.RpaCase.MIXED)
    split = _cfgs(8, 16, 64, 64, 4, 4, 1, _SEQ)
    streamed = dataclasses.replace(
        base, serve=dataclasses.replace(base.serve, stream_page_indices=True)
    )
    windowed = dataclasses.replace(
        base, model=dataclasses.replace(base.model, sliding_window=128)
    )
    for cfgs in (mixed, split, streamed, windowed):
        assert not schedule_xla.supported(cfgs)
    assert not schedule_xla.supported(base, extra_scalars=(jnp.zeros(1),))
