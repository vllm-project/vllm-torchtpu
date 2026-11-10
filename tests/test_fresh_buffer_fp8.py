# SPDX-License-Identifier: Apache-2.0
"""utils.fresh_buffer on a torch_tpu without a float8 `add` kernel.

torch_tpu 0.2.1.dev nightlies raise ``NotImplementedError: add(): not
implemented for float8_e4m3fn`` for ``t + 0`` on a float8 TPU tensor, and
fresh_buffer runs that on every FP8 weight at load. The fake below reports a
TPU device and refuses float8 ``add`` the same way, on CPU storage.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu import utils

_DTYPE_VIEWS = []  # (from, to) of every Tensor.view(dtype) on a fake tensor
_ADDS = {torch.add, torch.Tensor.add, torch.Tensor.__add__, torch.Tensor.__radd__}


class _NightlyTpuTensor(torch.Tensor):
    @property
    def device(self):
        return SimpleNamespace(type="tpu")

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.Tensor.view and len(args) == 2 and isinstance(args[1], torch.dtype):
            _DTYPE_VIEWS.append((args[0].dtype, args[1]))
        if func in _ADDS and args[0].dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            raise NotImplementedError(f"add(): not implemented for {args[0].dtype}")
        return super().__torch_function__(func, types, args, kwargs)


def _fake(t: torch.Tensor) -> torch.Tensor:
    return t.as_subclass(_NightlyTpuTensor)


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_fresh_buffer_survives_a_torch_tpu_without_float8_add(dtype):
    src = torch.randn(4, 6, 8).to(dtype)
    view = _fake(src)[:, 1:5, :]  # a view, the case fresh_buffer exists for
    out = utils.fresh_buffer(view)
    assert out.dtype == dtype
    assert out.shape == view.shape
    assert out.is_contiguous()
    assert out.untyped_storage().data_ptr() != src.untyped_storage().data_ptr()
    expected = src[:, 1:5, :].contiguous().view(torch.uint8)
    assert torch.equal(out.as_subclass(torch.Tensor).view(torch.uint8), expected)


def test_fresh_buffer_still_raises_for_wider_dtypes():
    class _NoAdd(_NightlyTpuTensor):
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            if func in _ADDS:
                raise NotImplementedError("add(): not implemented")
            return super().__torch_function__(func, types, args, kwargs)

    with pytest.raises(NotImplementedError):
        utils.fresh_buffer(torch.randn(4, 4, dtype=torch.bfloat16).as_subclass(_NoAdd))


def _float8_patterns(dtype):
    """Every bit pattern of `dtype` except NaNs, as a [16, 16] tensor."""
    bits = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(dtype)
    keep = ~torch.isnan(bits.float())
    return bits[keep]


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_fresh_buffer_float8_fallback_is_not_a_uint8_view(dtype, monkeypatch):
    """The fallback's result must own a float8 buffer: a uint8 buffer viewed as
    float8 is re-materialized by torch_tpu (`tt_jit_as_strided`) on every
    forward. Bit-exact over every non-NaN pattern."""
    monkeypatch.delenv("TPU_FRESH_BUFFER_FP8_UINT8_VIEW", raising=False)
    src = _float8_patterns(dtype)
    _DTYPE_VIEWS.clear()
    out = utils.fresh_buffer(_fake(src))
    plain = out.as_subclass(torch.Tensor)
    assert out.dtype == dtype and out.shape == src.shape
    assert (dtype, torch.uint8) not in _DTYPE_VIEWS, (
        "fresh_buffer reinterpreted the float8 bytes as uint8 (re-materialized per step)")
    assert torch.equal(plain.view(torch.uint8), src.view(torch.uint8))


def test_fresh_buffer_uint8_view_switch_keeps_the_old_path(monkeypatch):
    """TPU_FRESH_BUFFER_FP8_UINT8_VIEW=1 (the A/B control) is the old uint8 view."""
    monkeypatch.setenv("TPU_FRESH_BUFFER_FP8_UINT8_VIEW", "1")
    src = _float8_patterns(torch.float8_e4m3fn)
    _DTYPE_VIEWS.clear()
    out = utils.fresh_buffer(_fake(src))
    assert (torch.uint8, torch.float8_e4m3fn) in _DTYPE_VIEWS
    assert torch.equal(out.as_subclass(torch.Tensor).view(torch.uint8), src.view(torch.uint8))
