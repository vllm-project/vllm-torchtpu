# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm_torchtpu.layers.adapter.sample.fused_greedy_head import FusedGreedyHead


def test_fused_greedy_head_cpu_reference(device):
    """Check full-vocabulary CPU64 rounding, including a cross-tile tie."""
    generator = torch.Generator().manual_seed(9126)
    hidden = torch.randn(288, 5120, generator=generator).to(torch.bfloat16)
    weight = torch.randn(129280, 5120, generator=generator).to(torch.bfloat16)
    # A deliberately dominant identical pair straddles the tile boundary.
    hidden[0].zero_()
    hidden[0, 0] = 1
    weight[:, 0] = 0
    weight[1279:1281, 0] = 32
    reference = (hidden[:4].double() @ weight.double().T).to(torch.bfloat16)
    expected = reference.argmax(dim=-1)
    device_weight = weight.to(device)
    head = FusedGreedyHead(device_weight, hidden.shape[0])
    assert head.weight is device_weight
    compiled = torch.compile(head, backend="tpu", fullgraph=True, dynamic=False)
    actual = compiled(hidden.to(device)).cpu()
    torch.testing.assert_close(actual[:4].long(), expected, rtol=0, atol=0)
    assert actual[0].item() == 1279

    # The compiled head must read the shared weight, not a load-time snapshot.
    device_weight[1279, 0] = 0
    updated = compiled(hidden.to(device)).cpu()
    assert updated[0].item() == 1280
