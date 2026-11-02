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
"""TPU_MLA_FUSED_SMALL_A2A: the decode MLA kernel runs the DCP small
all-to-all itself.

CPU: the bucket gate (`fused_small_a2a_applies`) and the kernel's static
geometry.

The bit-identity claim -- the fused kernel's receive buffer equals
`small_all_to_all(send, row0=SEND_SLAB_ROW0)` of the unfused kernel's send
buffer in every row it writes, so the unpack and merge see the same bits --
needs 8 TensorCores with remote DMA and a barrier. It was checked on 2x2x2
(16 ranks = two DCP8 groups) at the served per-rank shapes (T = 4 / 8 and the
T = 16 guard, edge cells with padding tokens and whole ranks at lse -inf),
with NaN-poisoned receive buffers and positive controls (misrouted peer
block, dropped peer block, misrouted destination slab, dropped lse low
halves) that must fail, by the write-tpu-kernels harness of close-gaps trial 6
(kernel_work/t6-mla-fuse, probe 02-m1).
"""

import os
from unittest import mock

import pytest

from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla
from vllm_torchtpu.layers.adapter import cp_mla_attention as cp_mla

GATE_ENV = {
    "TPU_MLA_FUSED_SMALL_A2A": "1",
    "TPU_DCP_MLA_SEND_SMALL_A2A": "1",
    "TPU_DCP_MLA_SEND_SMALL_A2A_MAX_TOKENS": "8",
    "TPU_DCP_INDEXER_SMALL_AG": "1",
    "TPU_SC_TOPK_DCP_MERGE_FLAT": "1",
    "TPU_DCP_RANK_OPERAND": "1",
    "TPU_MLA_DECODE_BODY_LEVERS": "1",
    "TPU_DCP_MLA_TOPK_CAP": "512",
    "TPU_DCP_MLA_FUSED_PREP": "1",
    "TPU_MLA_DECODE_ROW_WIDTHS": "128,256,384",
    "TPU_DCP_MLA_SMALL_CHUNK_TOKENS": "32",
}


def _applies(tokens, **over):
    env = {k: v for k, v in dict(GATE_ENV, **over).items() if v is not None}
    with mock.patch.dict(os.environ, env):
        for k, v in over.items():
            if v is None:
                os.environ.pop(k, None)
        return cp_mla.fused_small_a2a_applies(tokens)


def test_gate_buckets():
    # T <= cap (8) with a one-step grid; never the T = 16 bucket.
    assert [t for t in (1, 2, 4, 8, 16, 32) if _applies(t)] == [1, 2, 4, 8]


@pytest.mark.parametrize(
    "key,off",
    [
        # Explicit off values: each of these is on by default on this branch.
        ("TPU_MLA_FUSED_SMALL_A2A", "0"),
        ("TPU_DCP_MLA_SEND_SMALL_A2A", "0"),
        ("TPU_DCP_RANK_OPERAND", "0"),
        ("TPU_DCP_MLA_FUSED_PREP", "0"),
        ("TPU_MLA_DECODE_ROW_WIDTHS", ""),
        ("TPU_DCP_MLA_TOPK_CAP", "0"),
    ],
)
def test_gate_off_without(key, off):
    assert not _applies(4, **{key: off})


def test_gate_needs_step8():
    # Without "step8" the decode grid walks the small-chunk batch: not one step.
    assert not _applies(8, TPU_MLA_DECODE_BODY_LEVERS="tr,zu,pv512,quad")
    assert _applies(8, TPU_MLA_DECODE_BODY_LEVERS="tr,step8")


def test_gate_cap():
    assert not _applies(8, TPU_DCP_MLA_SEND_SMALL_A2A_MAX_TOKENS="4")
    assert _applies(4, TPU_DCP_MLA_SEND_SMALL_A2A_MAX_TOKENS="4")
    # A cap above the step: buckets up to the step stay fused, the larger
    # ones keep the small all-to-all after the kernel.
    assert _applies(4, TPU_DCP_MLA_SEND_SMALL_A2A_MAX_TOKENS="32")
    assert _applies(8, TPU_DCP_MLA_SEND_SMALL_A2A_MAX_TOKENS="32")
    assert not _applies(16, TPU_DCP_MLA_SEND_SMALL_A2A_MAX_TOKENS="32")
    assert not _applies(4, TPU_DCP_MLA_SMALL_CHUNK_TOKENS="4")


def test_gate_step32():
    # "step32": buckets of up to 32 tokens are one grid step.
    levers = "tr,zu,pv512,noinf,quad,fxskip,lrows,step32"
    env = dict(TPU_DCP_MLA_SEND_SMALL_A2A_MAX_TOKENS="32", TPU_MLA_DECODE_BODY_LEVERS=levers)
    assert [t for t in (4, 8, 16, 32, 64) if _applies(t, **env)] == [4, 8, 16, 32]
    assert [t for t in (4, 8, 16, 32) if _applies(t, TPU_MLA_DECODE_BODY_LEVERS=levers)] == [4, 8]


def test_gate_xchg_max_tokens():
    # TPU_MLA_FUSED_XCHG_MAX_TOKENS: buckets above the step, up to it (within
    # the small all-to-all's cap), run the exchange over the grid steps.
    env = dict(
        TPU_DCP_MLA_SEND_SMALL_A2A_MAX_TOKENS="32", TPU_MLA_FUSED_SMALL_A2A_MERGE="1"
    )
    buckets = (1, 2, 4, 8, 16, 32, 64)
    assert [t for t in buckets if _applies(t, **env)] == [1, 2, 4, 8]
    on = dict(env, TPU_MLA_FUSED_XCHG_MAX_TOKENS="32")
    assert [t for t in buckets if _applies(t, **on)] == [1, 2, 4, 8, 16, 32]
    sixteen = dict(env, TPU_MLA_FUSED_XCHG_MAX_TOKENS="16")
    assert [t for t in buckets if _applies(t, **sixteen)] == [1, 2, 4, 8, 16]
    # Capped by the small all-to-all, and only with the in-kernel merge.
    assert not _applies(16, **dict(on, TPU_DCP_MLA_SEND_SMALL_A2A_MAX_TOKENS="8"))
    assert not _applies(16, **dict(on, TPU_MLA_FUSED_SMALL_A2A_MERGE="0"))
    assert _applies(8, **dict(on, TPU_MLA_FUSED_SMALL_A2A_MERGE="0"))


def test_step_lever_tokens():
    assert sparse_mla.decode_step_tokens(sparse_mla.DECODE_BODY_DEFAULT) == 8
    assert sparse_mla.decode_step_tokens(("tr", "step32")) == 32
    assert sparse_mla.decode_step_tokens(("tr", "quad")) == 0
    with pytest.raises(ValueError):
        sparse_mla.decode_step_tokens(("step8", "step16"))


def test_gate_compares_tokens_once():
    # The bucket size is symbolic under torch.compile: one `T <= cap`
    # comparison and nothing else (no modulo, no lower bound), or the
    # backbone trace specializes on the first bucket.
    seen = []

    class T(int):
        def __le__(self, other):
            seen.append(("<=", other))
            return int(self) <= other

        def __gt__(self, other):
            seen.append((">", other))
            return int(self) > other

        def __mod__(self, other):
            raise AssertionError("modulo on the token count")

        def __lt__(self, other):
            raise AssertionError("< on the token count")

        def __ge__(self, other):
            raise AssertionError(">= on the token count")

    assert _applies(T(4))
    assert seen == [("<=", 8)], seen


def test_slab_geometry():
    # The slab starts on a 16-row int16 tile and its last head row is the lse
    # word row, which lands on row SEND_PAD_ROWS - 1 of the receiving block.
    assert sparse_mla.SEND_SLAB_ROW0 % 8 == 0
    assert sparse_mla.SEND_SLAB_ROW0 + (sparse_mla.SEND_SLAB_ROW0 - 1) == (
        sparse_mla.SEND_PAD_ROWS - 1
    )
    assert sparse_mla.SEND_EXCHANGE_COLLECTIVE_ID == 13


def test_bucket_config():
    from vllm_torchtpu.layers.core import attention_interface as ai

    def cfg(tokens, **env):
        with mock.patch.dict(os.environ, env):
            for k in ("TPU_MLA_FUSED_SMALL_A2A_BUCKETS",):
                if k not in env:
                    os.environ.pop(k, None)
            return ai.fused_small_a2a_bucket_config(tokens)

    # The default bucket spec (trial 6): T = 4 pins only the gathered KV at
    # 16 MiB; T = 8 is not listed and takes the knob defaults.
    assert cfg(4) == (("kv",), 16) and cfg(8) == (True, 32)
    # Knob defaults (M1e, no spec): T < 8 unpinned at 16 MiB, T >= 8 everything
    # pinned at 32.
    no_spec = {"TPU_MLA_FUSED_SMALL_A2A_BUCKETS": ""}
    assert cfg(4, **no_spec) == (False, 16) and cfg(8, **no_spec) == (True, 32)
    spec = "4=kv@16,8=kv+tab@32"
    assert cfg(4, TPU_MLA_FUSED_SMALL_A2A_BUCKETS=spec) == (("kv",), 16)
    assert cfg(8, TPU_MLA_FUSED_SMALL_A2A_BUCKETS=spec) == (("kv", "tab"), 32)
    # A bucket the spec does not list keeps the defaults.
    assert cfg(8, TPU_MLA_FUSED_SMALL_A2A_BUCKETS="4=all@24") == (True, 32)
    assert cfg(4, TPU_MLA_FUSED_SMALL_A2A_BUCKETS="4=all@24") == (True, 24)
    assert cfg(4, TPU_MLA_FUSED_SMALL_A2A_BUCKETS="4=none@20") == (False, 20)
    with pytest.raises(ValueError):
        cfg(4, TPU_MLA_FUSED_SMALL_A2A_BUCKETS="4=weights@16")
    # A multi-step bucket (TPU_MLA_FUSED_XCHG_MAX_TOKENS) the spec does not
    # list pins only the gathered KV at VMEM_MIB; a listed one takes its entry.
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("TPU_MLA_FUSED_SMALL_A2A_BUCKETS", None)
        assert ai.fused_small_a2a_bucket_config(16, steps=True) == (("kv",), 16)
        assert ai.fused_small_a2a_bucket_config(32, steps=True) == (("kv",), 16)
    with mock.patch.dict(os.environ, {"TPU_MLA_FUSED_SMALL_A2A_BUCKETS": "16=none@16"}):
        assert ai.fused_small_a2a_bucket_config(16, steps=True) == (False, 16)


def test_pin_roles_cover_every_operand_class():
    assert set(sparse_mla.SEND_EXCHANGE_PIN_ROLES) == {"kv", "q", "q_pe", "tab", "recv"}


def test_min_tokens_default_fuses_every_gated_bucket():
    from vllm_torchtpu import envs

    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("TPU_MLA_FUSED_SMALL_A2A_MIN_TOKENS", None)
        assert envs.TPU_MLA_FUSED_SMALL_A2A_MIN_TOKENS == 1
    with mock.patch.dict(os.environ, {"TPU_MLA_FUSED_SMALL_A2A_MIN_TOKENS": "8"}):
        assert envs.TPU_MLA_FUSED_SMALL_A2A_MIN_TOKENS == 8


def test_merge_mode_constants():
    from vllm_torchtpu import envs

    assert "send_merge_lo_skip" in sparse_mla.SEND_EXCHANGE_FAULTS
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("TPU_MLA_FUSED_SMALL_A2A_MERGE", None)
        assert envs.TPU_MLA_FUSED_SMALL_A2A_MERGE is True  # default on (trial 6)
    with mock.patch.dict(os.environ, {"TPU_MLA_FUSED_SMALL_A2A_MERGE": "0"}):
        assert envs.TPU_MLA_FUSED_SMALL_A2A_MERGE is False
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("TPU_MLA_FUSED_SMALL_A2A_MERGE_MAX_TOKENS", None)
        assert envs.TPU_MLA_FUSED_SMALL_A2A_MERGE_MAX_TOKENS == 4
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("TPU_MLA_FUSED_XCHG_MAX_TOKENS", None)
        assert envs.TPU_MLA_FUSED_XCHG_MAX_TOKENS == 0  # off by default
