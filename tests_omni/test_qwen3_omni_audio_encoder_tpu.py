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
"""On-TPU boundary tests for the Qwen3-Omni audio encoder chunk-length patch.

Why these run on TPU: ``Qwen3OmniMoeAudioEncoder.forward`` (vllm-omni) splits
the mel features into 100-frame windows with
``chunk_num = torch.ceil(feature_lens / 100)``. On TPU that float division is a
reciprocal multiply, so every exact window multiple rounds up (800 / 100 ->
8.000001 -> 9 windows) and the following ``split`` raises because the window
lengths no longer add up to the input. CPU computes the same expression
exactly, which is why the CPU unit tests in test_qwen3omni_thinker_patch.py
cannot see the bug. ``maybe_patch_qwen3_omni_thinker`` moves ``feature_lens``
to CPU for that computation.

The tests build the real encoder with the real chunking geometry (128 mel
bins, three stride-2 convolutions, 100-frame windows, 800-frame attention
windows), shrunk to one 256-wide transformer layer, and run it in bf16 on the
TPU:

- exact window multiples with their +-1 neighbours, 99/100/101 and
  799/800/801 mel frames (10 ms per frame), plus 3200 frames, which spans
  several 800-frame attention windows; each is checked for the expected
  number of encoder tokens;
- a mixed batch with two exact multiples;
- numerical parity with a CPU reference built from the same weights;
- a canary that fails once TPU computes the ceil exactly, so the patch can be
  retired.
"""

from __future__ import annotations

from collections.abc import Generator
from types import SimpleNamespace
from typing import NamedTuple

import portpicker
import pytest
import torch
import torch.distributed as dist
from transformers import Qwen3OmniMoeAudioEncoderConfig
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.distributed.parallel_state import model_parallel_is_initialized
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention,
)
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.models.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeAudioEncoder as VllmQwen3OmniMoeAudioEncoder,
)
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeAudioEncoder,
)

from tests_omni.omni_test_utils import assert_parity
from vllm_torchtpu.omni.patches.qwen3omni_thinker import (
    _AUDIO_CHUNK_LENGTHS_PATCHED,
    maybe_patch_qwen3_omni_thinker,
)

pytestmark = [pytest.mark.omni]

# Whole seconds of 16 kHz audio are 100 mel frames each; 800 (8 s) is the
# length the production bug was found with. The +-1 neighbours take the
# partial-window branch of the chunking code on both sides of the boundary;
# 3200 adds a length several attention windows (n_window_infer = 800) long.
BOUNDARY_LENGTHS = [99, 100, 101, 799, 800, 801, 3200]
MIXED_BATCH = [800, 250, 200, 30]
# The exact multiple the patch rewrites and one partial-window neighbour.
PARITY_LENGTHS = [800, 801]

# Measured on TPU v6e (bf16 encoder vs the float32 CPU reference, 799-3200
# frames): relative L2 error 5.9e-3, max abs diff 1.7e-3 on outputs of
# magnitude ~0.19; batched vs alone identical except the 30-frame sample
# (5.4e-3 / 1.5e-3). The gates leave ~5x headroom.
PARITY_MAX_REL_L2 = 3e-2
PARITY_MAX_ABS = 1e-2


def _thinker_model_config() -> SimpleNamespace:
    """The part of a Stage-0 ``ModelConfig`` the patch callback looks at."""
    return SimpleNamespace(architectures=["Qwen3OmniMoeForConditionalGeneration"])


def _tiny_audio_config() -> Qwen3OmniMoeAudioEncoderConfig:
    """Production chunking geometry on a one-layer, 256-wide encoder.

    ``num_mel_bins`` (128), ``n_window`` (50 -> 100-frame windows),
    ``n_window_infer`` (800) and the three stride-2 convolutions keep their
    defaults: they decide how a length is chunked and how many tokens it
    yields. Only the width and depth shrink, so the test needs no checkpoint.
    """
    return Qwen3OmniMoeAudioEncoderConfig(
        d_model=256,
        encoder_layers=1,
        encoder_attention_heads=4,
        encoder_ffn_dim=512,
        output_dim=128,
        downsample_hidden_size=32,
    )


def _cnn_len(n_frames: int) -> int:
    """Frames left after three stride-2, kernel-3, padding-1 convolutions."""
    for _ in range(3):
        n_frames = (n_frames + 1) // 2
    return n_frames


def expected_tokens(n_frames: int, window: int) -> int:
    """Encoder tokens for ``n_frames`` mel frames: 13 per full window + tail."""
    full, tail = divmod(n_frames, window)
    return full * _cnn_len(window) + (_cnn_len(tail) if tail else 0)


def _init_linear_weights(module: torch.nn.Module, seed: int) -> None:
    """Give vLLM's parallel linear layers deterministic weights.

    They allocate with ``torch.empty`` and expect a checkpoint loader to fill
    them; without one the encoder computes on uninitialized memory and returns
    NaN. Convolutions, layer norms and the sinusoidal positions already get
    PyTorch's own initialization.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for submodule in module.modules():
            if isinstance(submodule, LinearBase):
                for param in submodule.parameters(recurse=False):
                    if param.dim() > 1:
                        param.copy_(
                            torch.randn(param.shape, generator=generator) * 0.02
                        )
                    else:
                        param.zero_()


def _reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | None = None,
    sequence_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pure-PyTorch stand-in for ``MMEncoderAttention`` on the CPU reference.

    Same contract: ``[batch, seq, heads, head_dim]`` inputs, full attention
    inside every ``cu_seqlens`` segment, scale ``head_dim ** -0.5``. Runs in
    float64 so the reference carries no precision loss of its own.
    """
    q, k, v = (t.transpose(1, 2).double() for t in (query, key, value))
    bounds = cu_seqlens.tolist() if cu_seqlens is not None else [0, q.shape[2]]
    scale = q.shape[-1] ** -0.5
    out = torch.empty_like(q)
    for start, end in zip(bounds[:-1], bounds[1:]):
        scores = q[:, :, start:end] @ k[:, :, start:end].transpose(-1, -2) * scale
        out[:, :, start:end] = scores.softmax(dim=-1) @ v[:, :, start:end]
    return out.transpose(1, 2).to(query.dtype)


def _route_custom_ops_to_cpu(module: torch.nn.Module) -> None:
    """Make every vLLM ``CustomOp`` in ``module`` runnable on CPU tensors.

    With the TPU platform active, ``CustomOp`` dispatch binds the TPU kernels
    to every instance, including one whose weights live on CPU, and vLLM's own
    SDPA fallback behind ``MMEncoderAttention.forward_native`` is a custom op
    registered for the TPU dispatch key only. The attention therefore gets the
    plain-PyTorch reference above; other custom ops get ``forward_native``.
    """
    for submodule in module.modules():
        if isinstance(submodule, MMEncoderAttention):
            submodule._forward_method = _reference_attention
        elif isinstance(submodule, CustomOp):
            submodule._forward_method = submodule.forward_native


def _encode(
    encoder: Qwen3OmniMoeAudioEncoder,
    features: torch.Tensor,
    lengths: list[int],
    window: int,
) -> torch.Tensor:
    """Run ``encoder`` on the concatenated ``[mel, frames]`` features."""
    param = next(encoder.parameters())
    device, dtype = param.device, param.dtype
    feature_lens = torch.tensor(lengths, dtype=torch.long, device=device)
    aftercnn_lens = torch.tensor(
        [expected_tokens(n, window) for n in lengths], dtype=torch.long, device=device
    )
    with torch.no_grad():
        out = encoder(
            features.to(device=device, dtype=dtype), feature_lens, aftercnn_lens
        )
    out = out[0] if isinstance(out, tuple) else out
    # The patch moves a copy; the caller's tensor must stay where it was.
    assert feature_lens.device == device
    return out


def _random_features(n_frames: int, num_mel_bins: int, seed: int) -> torch.Tensor:
    """Deterministic bf16-representable features, shared by TPU and CPU runs."""
    generator = torch.Generator().manual_seed(seed)
    feats = torch.randn(num_mel_bins, n_frames, generator=generator)
    return feats.to(torch.bfloat16).float()


class Encoders(NamedTuple):
    tpu: Qwen3OmniMoeAudioEncoder
    cpu: Qwen3OmniMoeAudioEncoder
    window: int  # mel frames per chunk, n_window * 2
    num_mel_bins: int

    def features(self, n_frames: int) -> torch.Tensor:
        return _random_features(n_frames, self.num_mel_bins, seed=n_frames)


@pytest.fixture(scope="module")
def tpu_device() -> torch.device:
    # Fail, not skip: the Omni lanes always run on a TPU, so a missing device
    # means the host is broken and must not turn the file into green skips.
    if not torch.accelerator.is_available():
        pytest.fail("Needs a TPU: the chunk-length bug is in TPU float division.")
    return torch.device("tpu")


@pytest.fixture(scope="module")
def dist_ctx(tpu_device: torch.device) -> Generator[None, None, None]:
    """Single-rank world and model-parallel groups for the encoder's layers.

    Mirrors ``dist_ctx`` in tests/conftest.py, which tests_omni/ does not
    inherit: destroys only the groups it created. Depends on ``tpu_device``
    because the groups bind to the TPU device; without one the module fails
    there, before any distributed state is created.
    """
    owns_world = not dist.is_initialized()
    if owns_world:
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{portpicker.pick_unused_port()}",
            backend="gloo",
        )
    owns_model_parallel = not model_parallel_is_initialized()
    # The parallel linear layers consult the current vLLM config when built.
    with set_current_vllm_config(VllmConfig()):
        try:
            ensure_model_parallel_initialized(1, 1)
            yield
        finally:
            if owns_model_parallel:
                destroy_model_parallel()
            if owns_world:
                destroy_distributed_environment()


@pytest.fixture(scope="module")
def encoders(
    dist_ctx: None, tpu_device: torch.device
) -> Generator[Encoders, None, None]:
    """A bf16 TPU encoder and a float32 CPU reference sharing one set of weights.

    The TPU encoder runs the patched forward, as in production. The CPU
    reference runs the same module with native custom ops.
    """
    # The callback rewrites two attributes on the shared class. Register them
    # first so the class is restored for the test modules that run after this
    # one in the same process.
    with pytest.MonkeyPatch.context() as mp:
        for cls in (VllmQwen3OmniMoeAudioEncoder, Qwen3OmniMoeAudioEncoder):
            mp.setattr(cls, "forward", cls.forward)
            mp.setattr(cls, "_get_cnn_output_lengths", cls._get_cnn_output_lengths)
            mp.setattr(
                cls,
                "_tpu_orig_get_cnn_output_lengths",
                getattr(cls, "_tpu_orig_get_cnn_output_lengths", None),
                raising=False,
            )
        # The registry callback with a thinker-stage config: the same path
        # TPUWorker.load_model() takes in production. Idempotent.
        assert maybe_patch_qwen3_omni_thinker(_thinker_model_config()) is True
        assert getattr(
            Qwen3OmniMoeAudioEncoder.forward, _AUDIO_CHUNK_LENGTHS_PATCHED, False
        )

        config = _tiny_audio_config()
        torch.manual_seed(0)
        cpu_encoder = Qwen3OmniMoeAudioEncoder(config).eval()
        _init_linear_weights(cpu_encoder, seed=0)
        assert all(torch.isfinite(p).all() for p in cpu_encoder.parameters())
        _route_custom_ops_to_cpu(cpu_encoder)

        tpu_encoder = Qwen3OmniMoeAudioEncoder(config).eval()
        tpu_encoder.load_state_dict(cpu_encoder.state_dict())
        tpu_encoder = tpu_encoder.to(tpu_device).to(torch.bfloat16)
        yield Encoders(
            tpu_encoder, cpu_encoder, config.n_window * 2, config.num_mel_bins
        )


def test_tpu_float_ceil_still_inexact_for_window_multiples(
    tpu_device: torch.device,
) -> None:
    """Canary for the bug behind ``maybe_patch_qwen3_omni_thinker``.

    Evaluates the upstream ``chunk_num`` expression on TPU for every whole
    window up to 100 s of audio. When this fails, TPU rounds the division
    exactly and the patch, the ``_get_cnn_output_lengths`` device hop and this
    canary can be removed.
    """
    window = _tiny_audio_config().n_window * 2
    lengths = torch.arange(window, 100 * window + 1, window)
    upstream = torch.ceil(lengths.to(tpu_device) / window).long().cpu()
    exact = lengths // window
    if torch.equal(upstream, exact):
        pytest.fail(
            "torch.ceil(feature_lens / 100) is exact on this TPU for every window "
            "multiple up to 10000 frames. Remove maybe_patch_qwen3_omni_thinker "
            "and this canary."
        )
    assert torch.equal(torch.ceil(lengths / window).long(), exact), (
        "CPU math must be exact"
    )


@pytest.mark.parametrize("n_frames", BOUNDARY_LENGTHS)
def test_audio_encoder_boundary_lengths_on_tpu(
    encoders: Encoders, n_frames: int
) -> None:
    """Exact window multiples and their +-1 neighbours encode on TPU."""
    out = _encode(
        encoders.tpu, encoders.features(n_frames), [n_frames], encoders.window
    )
    assert out.device.type == "tpu"
    assert out.shape[0] == expected_tokens(n_frames, encoders.window)
    assert torch.isfinite(out.float().cpu()).all()


def test_audio_encoder_mixed_batch_on_tpu(encoders: Encoders) -> None:
    """A batch mixing exact multiples (800, 200) with partial windows (250, 30).

    Checks the row count of the packed output and that each sample's rows match
    the same sample encoded alone, so no chunk leaks into its neighbour.
    """
    per_sample = [encoders.features(n) for n in MIXED_BATCH]
    out = _encode(
        encoders.tpu, torch.cat(per_sample, dim=1), MIXED_BATCH, encoders.window
    )
    expected = [expected_tokens(n, encoders.window) for n in MIXED_BATCH]
    assert out.shape[0] == sum(expected)  # 104 + 33 + 26 + 4 = 167

    row = 0
    for n_frames, feats, n_tokens in zip(MIXED_BATCH, per_sample, expected):
        alone = _encode(encoders.tpu, feats, [n_frames], encoders.window)
        assert alone.shape[0] == n_tokens
        assert_parity(
            out[row : row + n_tokens],
            alone,
            name=f"audio encoder {n_frames} frames, batched vs alone",
            max_rel_l2=PARITY_MAX_REL_L2,
            max_abs=PARITY_MAX_ABS,
        )
        row += n_tokens


@pytest.mark.parametrize("n_frames", PARITY_LENGTHS)
def test_audio_encoder_matches_cpu_reference(encoders: Encoders, n_frames: int) -> None:
    """The patched TPU forward matches the float32 CPU encoder around 8 s."""
    feats = encoders.features(n_frames)
    reference = _encode(encoders.cpu, feats, [n_frames], encoders.window)
    out = _encode(encoders.tpu, feats, [n_frames], encoders.window)
    assert out.shape == reference.shape
    assert_parity(
        out,
        reference,
        name=f"audio encoder {n_frames} frames, TPU bf16 vs CPU float32",
        max_rel_l2=PARITY_MAX_REL_L2,
        max_abs=PARITY_MAX_ABS,
    )
