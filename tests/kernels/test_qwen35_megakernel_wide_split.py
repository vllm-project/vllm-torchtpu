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
"""CPU tests for the N-way microbatch split of wide megakernel buckets
(A2 plumbing): the split plan, the bucket parsers to 800 rows, the wide
bucket expansion from max_num_seqs, and the per-microbatch row bookkeeping
of the split step (prep slices, pool chaining, merge) on a fake op."""

from __future__ import annotations

import types

import pytest
import torch

from vllm_torchtpu.kernels.qwen35_megakernel import vllm_adapter as VA

from qwen35_megakernel_test_utils import bare_mk


def test_split_plan_reproduces_the_two_way_split_and_extends_to_800():
    assert VA._split_plan(84, 4) == (80, 4)
    assert VA._split_plan(128, 4) == (80, 48)
    assert VA._split_plan(160, 4) == (80, 80)
    assert VA._split_plan(192, 4) == (80, 80, 32)
    assert VA._split_plan(200, 1) == (80, 80, 40)
    assert VA._split_plan(512, 4) == (80,) * 6 + (32,)
    assert VA._split_plan(800, 4) == (80,) * 10
    for bad in (80, 64, 162):
        with pytest.raises(ValueError):
            VA._split_plan(bad, 4)
    # every default wide bucket's remainder is a built single bucket or the base
    built = set(VA._DEFAULT_VERIFY_BUCKETS_64) | {VA._SPLIT_BASE_B}
    for b in VA._DEFAULT_WIDE_VERIFY_BUCKETS_128 + VA._DEFAULT_WIDE_VERIFY_BUCKETS_800:
        if b > VA._MAX_SINGLE_KERNEL_B:
            assert VA._split_plan(b, 4)[-1] in built, b


def test_parsers_accept_wide_buckets():
    assert VA.parse_mk_verify_buckets("4,8,64,512,800") == (4, 8, 64, 512, 800)
    assert VA.parse_mk_buckets("8,16,32,200,800") == (8, 16, 32, 200, 800)
    with pytest.raises(ValueError):
        VA.parse_mk_verify_buckets("4,804")
    with pytest.raises(ValueError):
        VA.parse_mk_verify_buckets("4,802")
    with pytest.raises(ValueError):
        VA.parse_mk_buckets("801")


@pytest.mark.parametrize(
    "max_reqs,expect_max,expect_has",
    [
        (16, 64, ()),  # unchanged: no wide buckets at or below 16 sequences
        (32, 128, (80, 96, 128)),  # unchanged 2-microbatch range
        (64, 256, (160, 192, 224, 256)),
        (128, 512, (320, 384, 448, 512)),
        (200, 800, (640, 768, 800)),
        (224, 800, (800,)),  # capped at the 800-row plan
    ],
)
def test_wide_verify_expansion_targets_four_rows_per_sequence(
    monkeypatch, max_reqs, expect_max, expect_has
):
    monkeypatch.delenv("VLLM_TPU_QWEN35_MK_VERIFY_BUCKETS", raising=False)
    mk = bare_mk(
        monkeypatch,
        buckets=(8,),
        verify_buckets=VA._DEFAULT_VERIFY_BUCKETS_64,
        max_verify_bucket=64,
        split_wide=True,
    )
    runner = types.SimpleNamespace(
        max_num_reqs=max_reqs, scheduler_config=types.SimpleNamespace(max_num_seqs=max_reqs)
    )
    mk._maybe_expand_wide_verify_buckets(runner)
    assert mk.max_verify_bucket == expect_max
    for b in expect_has:
        assert b in mk.verify_buckets
    assert all(b % 4 == 0 for b in mk.verify_buckets)
    assert mk.verify_buckets == tuple(sorted(set(mk.verify_buckets)))


def test_build_all_buckets_sizes_split_rows_from_the_plan(monkeypatch):
    mk = bare_mk(monkeypatch, buckets=(), verify_buckets=(4, 64, 128, 192, 512))
    mk._runner_token_paddings = ()
    seen = {}

    def fake_single(b, tokens_per_seq=1, out_pad_rows=None):
        mk._bucket_fused[(tokens_per_seq, b)] = False
        return ("single", b)

    def fake_split(b, tokens_per_seq=1, out_pad_rows=None):
        seen[b] = out_pad_rows
        return ("split", b)

    mk._build_bucket_op = fake_single
    mk._build_split_bucket_op = fake_split
    mk._build_all_buckets()
    assert mk._bucket_steps[(4, 64)] == ("single", 64)
    assert mk._bucket_steps[(4, 128)] == ("split", 128)
    # out rows: full microbatches plus the 16-row-aligned remainder
    assert seen == {128: 80 + 48, 192: 160 + 32, 512: 480 + 32}


class _FakeOp:
    """Stands in for a compiled single-bucket op: returns rows tagged with
    the microbatch's first position and passes the pools through with one
    added to a counter so the chaining is observable."""

    def __init__(self, rows: int):
        self.rows = rows
        self.calls = 0

    def __call__(self, pos, h, pt, ckpt, roff, weights, pools):
        self.calls += 1
        assert h.shape[0] == max(16, -(-self.rows // 16) * 16)
        assert pos.shape[0] == self.rows
        assert pt.shape[0] == ckpt.shape[1] == roff.shape[1] == self.rows // 4
        out = torch.full((h.shape[0], h.shape[1]), float(pos[0].item()), dtype=torch.float32)
        return (out, *[p + 1 for p in pools])


def test_split_step_slices_rows_per_microbatch_and_chains_pools(monkeypatch):
    """A 3-microbatch bucket (192 = 80 + 80 + 32 rows at T=4): each
    microbatch sees its own rows, pages and checkpoint slots, the pools flow
    from one microbatch to the next, and the merged output keeps step row
    order and the runner's padded row count."""
    T, B = 4, 192
    mk = bare_mk(monkeypatch, buckets=(), verify_buckets=(4, 32, 64))
    mk.max_pages_per_seq = 3
    mk.raw_pools = [torch.zeros(2)]
    mk.zero_copy = False
    mk.base_cfg = types.SimpleNamespace()
    # base_cfg.M is the kernel's padded row count: dataclasses.replace is used
    # on it, so give it a dataclass-like stand-in.
    import dataclasses

    @dataclasses.dataclass
    class Cfg:
        B: int = 0
        T: int = 1

        @property
        def M(self):
            return max(16, -(-self.B // 16) * 16)

    mk.base_cfg = Cfg()
    mk.embed_tokens = lambda ids: ids.to(torch.float32).unsqueeze(1).expand(-1, 8).contiguous()
    ops = {}

    def fake_build(b, tokens_per_seq=1, out_pad_rows=None):
        mk._flat_callables[(tokens_per_seq, b)] = object()
        ops[b] = mk._bucket_ops[(tokens_per_seq, b)] = _FakeOp(b)
        mk._bucket_fused[(tokens_per_seq, b)] = False

    mk._flat_callables = {}
    mk._build_bucket_op = fake_build
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: fn)
    step = mk._build_split_bucket_op(B, tokens_per_seq=T, out_pad_rows=B)
    assert sorted(ops) == [32, 80]
    S = B // T
    input_ids = torch.arange(B, dtype=torch.int32)
    positions = torch.arange(B, dtype=torch.int32) * 10
    block_tables = torch.arange(S * 3, dtype=torch.int32).reshape(S, 3)
    s0 = torch.arange(1, S + 1, dtype=torch.int32).unsqueeze(1)
    c0 = torch.arange(1, S + 1, dtype=torch.int32).unsqueeze(1).expand(S, T).contiguous()
    roff = torch.zeros(1, dtype=torch.int32)
    pools = [torch.zeros(2)]
    out = step(input_ids, positions, block_tables, s0, c0, s0, c0, s0, c0, roff, [], pools)
    assert out.shape[0] == B
    # row 0 of microbatch i carries position 10 * first row of that microbatch
    assert out[0, 0].item() == 0.0 and out[80, 0].item() == 800.0 and out[160, 0].item() == 1600.0
    # the 80-row op ran twice, the 32-row op once, and the pools passed through all three
    assert ops[80].calls == 2 and ops[32].calls == 1
    assert torch.equal(pools[0], torch.full((2,), 3.0))
