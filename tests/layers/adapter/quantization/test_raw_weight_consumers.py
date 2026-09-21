"""One layer keeps vLLM's `[n_out, n_in]`; everything else moves to (k, n).

The (k, n) migration flips dense linear weights, which is invisible to anything
going through a linear method's `apply` and *not* invisible to code that reads
`layer.weight` directly. Three such consumers existed; two were resolved by
making the consumer consistent rather than by exempting the layer:

  fused_wkv_wgate  the DSV4 compressor used to transpose this itself at load
                   time, to exactly the layout the linear method now produces.
                   Its transpose was deleted -- one flip instead of two.
  kv_b_proj        upstream vLLM's `MLAAttention.process_weights_after_loading`
                   reads it raw and asserts `[n_out, n_in]`. Our MLA op hands
                   it that view for the duration of the upstream call; the
                   layer is cleared immediately after and never runs a forward
                   pass, so nothing else observes the layout.
  in_proj_qkvz     still exempt. `fused_qkvz_projection_pcp_gdn` is a Pallas
                   kernel whose block specs encode `[n_out, n_in]`; teaching it
                   (k, n) means reworking index maps and re-tuning block sizes
                   under a VMEM budget, so the layer keeps the old layout for
                   now. This is the one remaining marker.
"""

import torch

from vllm_torchtpu.layers.adapter.linear_common import (
    KEEP_VLLM_LAYOUT_ATTR,
    WEIGHT_FLIPPED_ATTR,
)
from vllm_torchtpu.layers.adapter.quantization.unquantized import (
    VllmUnquantizedLinearMethod,
)


def _linear(n_in=8, n_out=16):
    """A layer shaped like vLLM's: weight is `[n_out, n_in]`."""
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.randn(n_out, n_in), requires_grad=False)
    return layer


def test_unmarked_layer_is_flipped_to_kmajor():
    layer = _linear()
    VllmUnquantizedLinearMethod().process_weights_after_loading(layer)
    assert layer.weight.shape == (8, 16), "should be (k, n) after loading"
    assert getattr(layer, WEIGHT_FLIPPED_ATTR, False)


def test_marked_layer_keeps_the_vllm_layout():
    layer = _linear()
    setattr(layer, KEEP_VLLM_LAYOUT_ATTR, True)
    VllmUnquantizedLinearMethod().process_weights_after_loading(layer)
    assert layer.weight.shape == (16, 8), (
        "must stay [n_out, n_in] for the consumer that reads it raw"
    )
    assert not getattr(layer, WEIGHT_FLIPPED_ATTR, False)


def test_mark_is_off_by_default():
    assert not getattr(_linear(), KEEP_VLLM_LAYOUT_ATTR, False)


def test_marked_layer_still_computes_the_same_output():
    """The fallback path must agree with the flipped path numerically."""
    torch.manual_seed(0)
    x = torch.randn(4, 8)
    ref = _linear()
    w = ref.weight.detach().clone()

    flipped = _linear()
    flipped.weight = torch.nn.Parameter(w.clone(), requires_grad=False)
    kept = _linear()
    kept.weight = torch.nn.Parameter(w.clone(), requires_grad=False)
    setattr(kept, KEEP_VLLM_LAYOUT_ATTR, True)

    method = VllmUnquantizedLinearMethod()
    method.process_weights_after_loading(flipped)
    method.process_weights_after_loading(kept)

    torch.testing.assert_close(method.apply(flipped, x), method.apply(kept, x))


def test_double_processing_is_idempotent_on_a_square_weight():
    """The guard exists for exactly this case.

    `o_proj` is square (`input_size == output_size`) in most models, so the
    layout cannot be recovered from the shape: a second flip would transpose
    the values back while the shape still looks correct.
    """
    layer = _linear(n_in=16, n_out=16)
    original = layer.weight.detach().clone()
    method = VllmUnquantizedLinearMethod()

    method.process_weights_after_loading(layer)
    once = layer.weight.detach().clone()
    method.process_weights_after_loading(layer)
    twice = layer.weight.detach().clone()

    torch.testing.assert_close(once, original.transpose(0, 1).contiguous())
    torch.testing.assert_close(twice, once), "second call must be a no-op"


def test_marked_layer_is_never_flipped_even_twice():
    layer = _linear()
    setattr(layer, KEEP_VLLM_LAYOUT_ATTR, True)
    original = layer.weight.detach().clone()
    method = VllmUnquantizedLinearMethod()
    method.process_weights_after_loading(layer)
    method.process_weights_after_loading(layer)
    torch.testing.assert_close(layer.weight.detach(), original)
