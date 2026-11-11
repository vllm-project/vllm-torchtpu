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
"""The mover's device legs on a TPU through tpu-sync's raw transfer: a slot
of a uint8 and of an fp4-pair (float4_e2m1fn_x2) slot-major tensor, and a
flat f32 scale table moved whole with a slot's rows patched in the host image.

The fp4 tensors come from the model's own Pallas quantizer: torch_tpu backs
them with an F4 array of twice the minor dimension (storage twice nbytes), and
raw transfer addresses them by their on-device slot bytes. A `.view(x2)` of a
uint8 device tensor is a different buffer and is refused. Host bytes are the
device's physical layout, so slots are compared as device images (the same
read on both sides), never against host values. Never print an x2 tensor:
torch_tpu aborts in its repr; run with `pytest --tb=short`."""

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu import expert_mover as mv
from vllm_torchtpu import expert_relayout as rl

pytestmark = pytest.mark.tpu_test

SLOTS, K, N = 6, 1024, 2560


def _bits(seed, *shape):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 255, shape, generator=g, dtype=torch.int32).to(torch.uint8)


def _raw():
    try:
        return mv.RawTransferTransport()
    except ImportError as exc:
        pytest.skip(f"needs tpu-sync with fp4 raw-transfer support: {exc}")


def _fp4(seed, slots, k, n):
    """An fp4-pair weight as the model holds it: [slots, k, n/2] x2, F4-backed."""
    from vllm_torchtpu.layers.adapter.fused_moe import quantize_native_fp4_kmajor
    from vllm_torchtpu.utils import synchronize_tensors

    g = torch.Generator().manual_seed(seed)
    w = torch.randn(slots, k, n, generator=g, dtype=torch.bfloat16).to("tpu")
    q, _scale = quantize_native_fp4_kmajor(w, block=32, pack=False)
    synchronize_tensors([q])
    assert q.dtype == torch.float4_e2m1fn_x2
    assert q.untyped_storage().nbytes() == 2 * q.nbytes
    return q


def _image(transport, t):
    """Every slot's on-device bytes, through the transport's own read."""
    future, host = transport.d2h(t, 0, t.shape[0])
    future.wait()
    image = host.clone()
    transport.release(host)
    return image


def test_a_uint8_slot_round_trips_byte_for_byte():
    transport = _raw()
    src_bits, dst_bits = _bits(1, SLOTS, K, N), _bits(2, SLOTS, K, N)
    src, dst = src_bits.to("tpu"), dst_bits.to("tpu")
    assert transport.slice_bytes(src) == K * N
    future, host = transport.d2h(src, 1, 1)
    future.wait()
    assert host.device.type == "cpu" and tuple(host.shape) == (1, K * N)
    transport.h2d(host, dst, 4).wait()
    got = dst.cpu()
    torch.testing.assert_close(got[4], src_bits[1])
    torch.testing.assert_close(got[:4], dst_bits[:4])
    torch.testing.assert_close(got[5], dst_bits[5])


def test_an_fp4_pair_slot_moves_as_its_on_device_bytes():
    """The fused path's weights: torch packs two fp4 per byte, torch_tpu backs
    the tensor with an F4 array of twice the minor dim, accounted at a byte per
    element, but the device packs fp4, so the slice is the slot's packed bytes:
    half the storage accounting, and a slot lands in another fp4 tensor's slot."""
    transport = _raw()
    src, dst = _fp4(3, SLOTS, K, N), _fp4(4, SLOTS, K, N)
    slot = transport.slice_bytes(src)
    assert slot == K * N // 2
    assert src.untyped_storage().nbytes() == 2 * slot * SLOTS
    src_image, dst_before = _image(transport, src), _image(transport, dst)
    future, host = transport.d2h(src, 1, 1)
    future.wait()
    assert tuple(host.shape) == (1, slot)
    assert torch.equal(host[0], src_image[1])
    transport.h2d(host, dst, 4).wait()
    transport.release(host)
    got = _image(transport, dst)
    assert torch.equal(got[4], src_image[1])
    assert not torch.equal(got[4], dst_before[4])
    assert torch.equal(got[:4], dst_before[:4])
    assert torch.equal(got[5], dst_before[5])


def test_a_uint8_view_of_an_fp4_tensor_is_refused():
    """`.view(float4_e2m1fn_x2)` of a uint8 device tensor keeps the uint8
    buffer, half of what the x2 shape resolves to on the device; raw transfer
    refuses it instead of moving half a slot."""
    transport = _raw()
    v = _bits(5, SLOTS, K, N // 2).to("tpu").view(torch.float4_e2m1fn_x2)
    with pytest.raises(ValueError, match="span"):
        transport.slice_bytes(v)


def test_a_scale_table_is_imaged_whole_and_patched_by_slot_rows():
    transport = _raw()
    rows = 32  # fp4_scale_rows pads to the f32 tile: a slot's rows = 4 tile rows
    a_host = torch.randn(SLOTS * rows, N)
    b_host = torch.randn(SLOTS * rows, N)
    a, b = a_host.to("tpu"), b_host.to("tpu")
    assert transport.slice_bytes(a) == N * 4
    fa, image_a = transport.d2h(a, 0, SLOTS * rows)
    fb, image_b = transport.d2h(b, 0, SLOTS * rows)
    fa.wait()
    fb.wait()
    row_bytes = image_a.shape[1]
    flat_a, flat_b = image_a.reshape(-1), image_b.reshape(-1)
    span = rows * row_bytes
    flat_b[3 * span : 4 * span].copy_(flat_a[1 * span : 2 * span])
    transport.h2d(image_b, b, 0).wait()
    got = b.cpu()
    torch.testing.assert_close(got[3 * rows : 4 * rows], a_host[1 * rows : 2 * rows])
    torch.testing.assert_close(got[: 3 * rows], b_host[: 3 * rows])
    torch.testing.assert_close(got[4 * rows :], b_host[4 * rows :])


def test_the_mover_stages_fp4_weights_and_scale_tables_between_two_fake_ranks():
    """Two modules on one chip stand in for two ranks; the mover moves expert
    slot 1 of rank 0 into staging slot 5 of rank 1 through the LocalExchange,
    with the real device legs."""
    transport = _raw()
    rows = 32

    def module(seed):
        w13 = _fp4(seed, SLOTS, K, N)
        w2 = _fp4(seed + 1, SLOTS, K // 2, N)
        return SimpleNamespace(
            local_num_experts=SLOTS,
            w13_weight=torch.nn.Parameter(w13, requires_grad=False),
            w2_weight=torch.nn.Parameter(w2, requires_grad=False),
            _tpu_fused_w13_scale=torch.randn(SLOTS * rows, N).to("tpu"),
            _tpu_fused_w2_scale=torch.randn(SLOTS * rows // 2, N).to("tpu"),
        )

    a, b = module(10), module(20)
    before = {
        name: _image(transport, getattr(b, name).detach())
        for name in ("w13_weight", "w2_weight")
    }
    before.update(
        {
            name: getattr(b, name).cpu().clone()
            for name in ("_tpu_fused_w13_scale", "_tpu_fused_w2_scale")
        }
    )
    plan = rl.Plan(
        1,
        0,
        {("target", 0): None},
        [rl.Move("target", 0, 7, src_rank=0, src_slot=1, dst_rank=1, dst_slot=5)],
    )
    exchange = mv.LocalExchange()
    movers = [mv.SlabMover(r, transport, exchange.endpoint(r)) for r in range(2)]
    mods = {0: {("target", 0): a}, 1: {("target", 0): b}}
    for r, m in enumerate(movers):
        m.start(plan, mods[r])
    import time

    deadline = time.monotonic() + 60
    while any(m.staged_generation() != 1 for m in movers):
        assert not any(m.failed() for m in movers), "a mover failed"
        assert time.monotonic() < deadline, "the movers did not finish"
        for m in movers:
            m.pump()
        time.sleep(0.01)
    for name in ("w13_weight", "w2_weight"):
        src = _image(transport, getattr(a, name).detach())
        got = _image(transport, getattr(b, name).detach())
        assert torch.equal(got[5], src[1]), name
        assert torch.equal(got[:5], before[name][:5]), name
    for name, r in (("_tpu_fused_w13_scale", rows), ("_tpu_fused_w2_scale", rows // 2)):
        src = getattr(a, name).cpu()
        got = getattr(b, name).cpu()
        torch.testing.assert_close(got[5 * r : 6 * r], src[1 * r : 2 * r])
        torch.testing.assert_close(got[: 5 * r], before[name][: 5 * r])
    assert movers[1].last_report["tables"] == 2
