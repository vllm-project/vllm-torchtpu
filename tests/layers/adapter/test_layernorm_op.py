# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for VllmTPUGemmaRMSNorm custom op."""

import types

import pytest
import torch
import torch.nn as nn
from vllm import ir
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_torchtpu.layers.adapter.custom_ops.layernorm_op import VllmTPUGemmaRMSNorm
from vllm_torchtpu.layers.adapter.quantization.unquantized import (
    VllmUnquantizedGemmaNormWeightFold,
)

HIDDEN_SIZE = 256
EPS = 1e-6

# The sweep reads just these three: `dtype` for attention layers,
# `quantization` for the torchao reload hook, and the tying flag for
# `maybe_retie_word_embeddings`. A real ModelConfig would pull in a model
# download. False says the checkpoint left the embeddings tied, so the retie
# pass returns before it looks for an `lm_head` this bare module lacks.
_MODEL_CONFIG = types.SimpleNamespace(
    dtype=torch.bfloat16,
    quantization=None,
    word_embeddings_untied_by_checkpoint=False,
)


@pytest.fixture
def vllm_cpu_config():
    """Provides a minimal VllmConfig configured for CPU."""
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    with set_current_vllm_config(config):
        yield config


def _load(norm: VllmTPUGemmaRMSNorm, checkpoint_weight: torch.Tensor) -> None:
    """Load a checkpoint weight, then run vLLM's real post-load sweep.

    `BaseModelLoader.load_model` runs the sweep for every loader, right after
    `load_weights`. Going through the upstream function rather than calling
    the fold directly is what makes these tests check that a norm layer is
    actually reachable from the sweep -- the whole premise of attaching a
    `quant_method` to it.
    """
    default_weight_loader(norm.weight, checkpoint_weight)
    model = nn.Module()
    model.norm = norm
    process_weights_after_loading(model, _MODEL_CONFIG, torch.device("cpu"))


def test_sweep_folds_the_implicit_one_and_forward_uses_it(vllm_cpu_config):
    """The sweep stores `1 + w` in fp32, and forward normalizes with it."""
    checkpoint_weight = torch.randn(HIDDEN_SIZE, dtype=torch.bfloat16)
    folded = checkpoint_weight.float() + 1.0

    # Built under the model dtype, as `BaseModelLoader.load_model` does. The
    # parameter is fp32 only because `__init__` asks for it; bf16 is too
    # coarse to hold `1 + w` at the precision the per-step upcast provided.
    with set_default_torch_dtype(torch.bfloat16):
        norm = VllmTPUGemmaRMSNorm(HIDDEN_SIZE, eps=EPS)
    _load(norm, checkpoint_weight)

    assert norm.weight.dtype == torch.float32
    torch.testing.assert_close(norm.weight.data, folded)

    x = torch.randn(4, 16, HIDDEN_SIZE)
    torch.testing.assert_close(norm(x), ir.ops.rms_norm(x, folded, EPS))


def test_reload_refolds_the_new_checkpoint_value(vllm_cpu_config):
    """A reload rewrites the raw weight, so the fold has to run again.

    `_layerwise_process` copies the checkpoint value back into the parameter
    and then calls this hook a second time. A stored "already folded" flag
    would suppress the second fold and leave the norm on `w`, not `1 + w`.
    """
    norm = VllmTPUGemmaRMSNorm(HIDDEN_SIZE)
    _load(norm, torch.randn(HIDDEN_SIZE))

    reloaded_weight = torch.randn(HIDDEN_SIZE)
    _load(norm, reloaded_weight)

    torch.testing.assert_close(norm.weight.data, reloaded_weight.float() + 1.0)


def test_quant_method_refuses_the_linear_entry_points(vllm_cpu_config):
    """`create_weights`/`apply` exist only to satisfy QuantizeMethodBase."""
    norm = VllmTPUGemmaRMSNorm(HIDDEN_SIZE)
    assert isinstance(norm.quant_method, VllmUnquantizedGemmaNormWeightFold)

    with pytest.raises(NotImplementedError):
        norm.quant_method.create_weights(norm)
    with pytest.raises(NotImplementedError):
        norm.quant_method.apply(norm, torch.randn(2, HIDDEN_SIZE))


def test_forward_with_residual_matches_upstream(vllm_cpu_config):
    """Residual path matches stock GemmaRMSNorm on the same weight."""
    checkpoint_weight = torch.randn(HIDDEN_SIZE)

    norm = VllmTPUGemmaRMSNorm(HIDDEN_SIZE, eps=EPS)
    _load(norm, checkpoint_weight)

    x = torch.randn(4, 16, HIDDEN_SIZE)
    residual = torch.randn(4, 16, HIDDEN_SIZE)
    out, out_residual = norm(x, residual)

    expected_out, expected_residual = ir.ops.fused_add_rms_norm(
        x, residual, checkpoint_weight.float() + 1.0, EPS
    )
    assert out.shape == x.shape
    assert out_residual.shape == residual.shape
    torch.testing.assert_close(out, expected_out)
    torch.testing.assert_close(out_residual, expected_residual)
