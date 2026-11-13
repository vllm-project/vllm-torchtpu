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
"""Exactness tests for the fused async draft seed.

`_async_seed_inputs_impl` / `_async_seed_inputs_compiled` (seed, input
shift/scatter and loop-bucket gather pad in one program) against the separate
seed, scatter and pad programs.
"""

import pytest
import torch

from vllm_torchtpu.spec_decode.eagle3 import (
    _async_draft_seed_compiled,
    _async_draft_seed_impl,
    _async_seed_inputs_compiled,
    _async_seed_inputs_impl,
    _maybe_pad_dim0,
)

_tpu_compile = torch.compile(backend="tpu", fullgraph=True, dynamic=False)


def _seed_scatter_impl(input_ids, last_token_indices, next_token_ids):
    shifted = input_ids
    if input_ids.shape[0] > 1:
        shifted = torch.cat((input_ids[1:], input_ids[-1:]), dim=0)
    return shifted.index_put((last_token_indices,), next_token_ids.to(input_ids.dtype))


_seed_scatter = _tpu_compile(_seed_scatter_impl)


def check_seed(device, seeds: int, *, compiled: bool = True) -> int:
    """Fused async draft seed vs the separate seed / scatter / pad programs."""
    k = 3
    fail = 0
    seed_fn = _async_draft_seed_compiled if compiled else _async_draft_seed_impl
    scatter_fn = _seed_scatter if compiled else _seed_scatter_impl
    fused_fn = _async_seed_inputs_compiled if compiled else _async_seed_inputs_impl
    for width, total_tokens, qsl_len in (
        (4, 16, 33),
        (8, 32, 33),
        (16, 64, 33),
        (24, 128, 33),
    ):
        for seed in range(seeds):
            g = torch.Generator().manual_seed(seed)
            n = int(torch.randint(1, width + 1, (1,), generator=g))
            d = torch.randint(0, k + 1, (width,), generator=g)
            d[n:] = 0
            nt = torch.full((width, k + 1), -1, dtype=torch.int32)
            for r in range(n):
                acc = int(torch.randint(0, int(d[r]) + 1, (1,), generator=g))
                nt[r, : acc + 1] = torch.randint(
                    0, 1000, (acc + 1,), generator=g, dtype=torch.int32
                )
            q = torch.cumsum(
                torch.cat([torch.zeros(1, dtype=torch.int64), (d + 1).to(torch.int64)]),
                0,
            )
            qsl = torch.full((qsl_len,), int(q[n]), dtype=torch.int32)
            qsl[: n + 1] = q[: n + 1].to(torch.int32)
            ids = torch.randint(
                0, 1000, (total_tokens,), generator=g, dtype=torch.int32
            )
            num_reqs = torch.tensor(n, dtype=torch.int64)
            p = max(16, 1 << (n - 1).bit_length())
            args = (
                nt.to(device),
                d.to(torch.int32).to(device),
                qsl.to(device),
                num_reqs.to(device),
            )
            lti, nti, nrej = seed_fn(*args)
            ref_ids = scatter_fn(ids.to(device), lti, nti)
            ref_g = _maybe_pad_dim0(lti[:p], p)
            out_ids, out_g, out_nrej = fused_fn(*args, ids.to(device), p)
            ok = (
                torch.equal(ref_ids.cpu(), out_ids.cpu())
                and torch.equal(ref_g.cpu(), out_g.cpu())
                and torch.equal(nrej.cpu(), out_nrej.cpu())
            )
            if not ok:
                fail += 1
                if fail <= 10:
                    print(
                        f"SEED MISMATCH width={width} T={total_tokens} "
                        f"n={n} seed={seed}"
                    )
    return fail


@pytest.fixture
def tpu_device():
    dev = torch.device("tpu")
    probe = torch.tensor([17], dtype=torch.int32, device=dev)
    assert probe.cpu().tolist() == [17]
    return dev


def test_async_seed_inputs_impl_matches_separate_cpu():
    assert check_seed(torch.device("cpu"), seeds=5, compiled=False) == 0


@pytest.mark.tpu_test
def test_draft_seed_fused_compiled_matches_separate_tpu(tpu_device):
    assert check_seed(tpu_device, seeds=4, compiled=True) == 0
