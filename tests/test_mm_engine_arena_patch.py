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
"""Large tensors reach EngineCore through the arena with their bytes intact."""

import gc
import hashlib
import inspect

import pytest
import torch
import vllm
from vllm.distributed.device_communicators.shm_tensor_arena import ShmTensorArena
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from vllm_torchtpu import mm_engine_arena_patch as patch_mod

PINNED_VLLM_VERSION = "0.33.0"
PINNED_SOURCE_HASHES = {
    "MPClient.__init__": (
        "3f70fcf92e30495a51dac0f6f49a99ec8b62ba3abeb1b2d3a7ecca408717f842"
    ),
    "process_input_sockets": (
        "05ccb56b568644c49ac5c0c2e1eecdbdd6f8f93450d9eb1005d47cd2bf4d5ee0"
    ),
}


@pytest.mark.skipif(
    vllm.__version__ != PINNED_VLLM_VERSION,
    reason=f"the patch wraps vLLM {PINNED_VLLM_VERSION}'s engine client and core",
)
@pytest.mark.parametrize(
    ("name", "fn"),
    [
        ("MPClient.__init__", patch_mod.UPSTREAM_CLIENT_INIT),
        ("process_input_sockets", patch_mod.UPSTREAM_PROCESS_INPUT_SOCKETS),
    ],
)
def test_upstream_sources_unchanged(name, fn):
    """Fail when upstream changes where the encoder and decoders are built."""
    digest = hashlib.sha256(inspect.getsource(fn).encode("utf-8")).hexdigest()
    assert digest == PINNED_SOURCE_HASHES[name], (
        f"Upstream {name} has drifted; re-check "
        "src/vllm_torchtpu/mm_engine_arena_patch.py."
    )


def _sender():
    """A sender whose arena has 3 slots of one large tensor each."""
    sender = patch_mod.ArenaTensorSender()
    sender._arena = ShmTensorArena(n_reader=1, slot_bytes=_big().nbytes, n_slots=3)
    return sender


def _big():
    gen = torch.Generator().manual_seed(0)
    rows = patch_mod.MIN_BYTES // 1536 + 1
    return torch.randint(0, 256, (rows, 1536), dtype=torch.uint8, generator=gen)


def test_encoder_round_trip():
    """Through vLLM's encoder and decoder, a large tensor arrives as a view
    of an arena slot and a small one inline, both with the sent bytes."""
    sender = _sender()
    encoder = MsgpackEncoder(oob_tensor_consumer=sender)
    decoder = MsgpackDecoder(
        list[torch.Tensor], oob_tensor_provider=patch_mod.ArenaTensorReceiver()
    )
    big, small = _big(), torch.arange(1000, dtype=torch.int64)
    got_big, got_small = decoder.decode(encoder.encode([big, small]))
    assert torch.equal(got_big, big) and torch.equal(got_small, small)
    assert sender._arena.write_tensor(big) == 1  # the large one took slot 0


def test_slot_frees_after_the_view_dies():
    """A slot stays taken while EngineCore holds its view, and is written
    again once the view is gone."""
    sender, receiver = _sender(), patch_mod.ArenaTensorReceiver()
    big = _big()
    shape = tuple(big.shape)
    meta = sender(big)
    assert meta["slot"] == 0
    view = receiver("uint8", shape, meta)
    assert sender(big)["slot"] == 1  # slot 0 is still read
    del view
    gc.collect()
    receiver("uint8", shape, sender(big))  # the next read retires slot 0
    assert sender(big)["slot"] == 0
