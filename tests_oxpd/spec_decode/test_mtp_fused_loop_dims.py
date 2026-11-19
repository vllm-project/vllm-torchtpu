# SPDX-License-Identifier: Apache-2.0
"""The fused MTP loop compiles with only the draft dims its forward takes."""

from vllm_torchtpu.spec_decode import mtp_fused_loop


def test_draft_only_dynamic_dims_are_dropped(monkeypatch):
    seen = {}

    def fake_support_torch_compile(*, dynamic_arg_dims):
        seen.update(dynamic_arg_dims)
        return lambda cls: (lambda **kwargs: kwargs)

    monkeypatch.setattr(
        mtp_fused_loop, "support_torch_compile", fake_support_torch_compile
    )
    mtp_fused_loop.build_mtp_fused_loop(
        vllm_config=None,
        prefix="p",
        mtp=None,
        num_steps=2,
        dynamic_arg_dims={
            "input_ids": 0,
            "positions": 0,
            "hidden_states": 0,
            "draft_rows": 0,
        },
    )
    assert seen == {"input_ids": 0, "positions": 0, "hidden_states": 0}
