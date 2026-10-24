# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Multi-LoRA compiled the way the TPU runner compiles it.

Adapter and mapping changes must reach the compiled programs as data: one
trace serves every token bucket, swaps take effect without recompiling, and
tokens without an adapter stay unaffected.
"""

import dataclasses
import os
from functools import partial

import pytest
import torch
import vllm
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    DeviceConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.forward_context import set_forward_context
from vllm.lora.layers import LoRAMapping

import vllm_torchtpu
import vllm_torchtpu.env_override  # noqa: F401
from tests.distributed.tpu_test_utils import run_in_isolated_process
from vllm_torchtpu.compilation import shape_variants, tpu_compiler
from vllm_torchtpu.lora.torch_punica_tpu import PunicaWrapperTPU

if not hasattr(vllm, "__version__"):  # namespace-package quirk in the CPU image
    import importlib.metadata

    vllm.__version__ = importlib.metadata.version("vllm")

# Each case runs in a spawned child that needs the chip, so this module is
# marked spawns_tpu_workers to run before in-process TPU tests (see
# tests/conftest.py::pytest_collection_modifyitems).
pytestmark = [pytest.mark.tpu_test, pytest.mark.spawns_tpu_workers]

BUCKETS = [16, 64, 256]
HIDDEN, OUT, RANK = 256, 512, 16
# Every slot is filled, so a token without an adapter cannot pass by reading
# an empty slot.
LORA_INDEX_TO_ID = [9, 7, 5]
SLOT_OF_ID = {9: 0, 7: 1, 5: 2, 0: -1}
NUM_SLOTS = len(LORA_INDEX_TO_ID)
TOLERANCE = {"rtol": 2e-2, "atol": 5e-2}


@support_torch_compile(dynamic_arg_dims={"x": 0})
class LoRALinear(nn.Module):
    """A linear layer that applies LoRA the way BaseLinearLayerWithLoRA does."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.weight = nn.Parameter(
            _weight(OUT, HIDDEN, seed=0).to("tpu"), requires_grad=False
        )
        self.lora_a_stacked = (_zeros(NUM_SLOTS, 1, RANK, HIDDEN),)
        self.lora_b_stacked = (_zeros(NUM_SLOTS, 1, OUT, RANK),)
        self.punica_wrapper = PunicaWrapperTPU(BUCKETS[-1], 8, torch.device("tpu"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x @ self.weight.T
        return self.punica_wrapper.add_lora_linear(
            y, x, self.lora_a_stacked, self.lora_b_stacked, 1.0, (OUT,)
        )


def _zeros(*shape):
    return torch.zeros(*shape, dtype=torch.bfloat16, device="tpu")


def _weight(*shape, seed):
    """Keeps outputs and LoRA deltas near unit size, well above the tolerance."""
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=gen) / shape[-1] ** 0.5).bfloat16()


def _inputs(rows, seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(rows, HIDDEN, generator=gen).bfloat16().to("tpu")


def _setup(cache_root):
    """Returns the model, its config, and a list that records every program
    compiled, so that a test can tell when something recompiles."""
    os.environ.update(
        VLLM_CACHE_ROOT=cache_root,
        VLLM_DISABLE_COMPILE_CACHE="1",
        VLLM_USE_AOT_COMPILE="1",
        VLLM_USE_MEGA_AOT_ARTIFACT="0",
    )
    # The cache key hashes a ModelConfig that this test does not build.
    tpu_compiler.compute_tpu_compilation_hash = lambda config: "lora"
    vllm_torchtpu._patch_vllm_aot_compile_cache_key()
    compiles = []
    compile_ = tpu_compiler.TpuCompilerAdaptor.compile

    def counted_compile(self, graph, *args, **kwargs):
        compiles.append(graph)
        return compile_(self, graph, *args, **kwargs)

    tpu_compiler.TpuCompilerAdaptor.compile = counted_compile
    config = _config()
    with set_current_vllm_config(config):
        model = LoRALinear(vllm_config=config)
    return model, config, compiles


def _config() -> VllmConfig:
    from vllm.platforms import current_platform

    # TpuPlatform.check_and_update_config() needs a full ModelConfig; only the
    # compile pipeline is under test.
    platform = type(current_platform)
    hook = platform.check_and_update_config
    platform.check_and_update_config = classmethod(lambda cls, config: None)
    try:
        config = VllmConfig(
            device_config=DeviceConfig(device="tpu"),
            scheduler_config=SchedulerConfig(
                max_num_batched_tokens=BUCKETS[-1],
                max_model_len=BUCKETS[-1],
                is_encoder_decoder=False,
            ),
            compilation_config=CompilationConfig(
                mode=CompilationMode.VLLM_COMPILE,
                backend="",  # the platform's backend
                compile_sizes=list(BUCKETS),
                splitting_ops=[],
                cudagraph_capture_sizes=[],
            ),
        )
    finally:
        platform.check_and_update_config = hook
    pass_config = config.compilation_config.pass_config
    for field in dataclasses.fields(pass_config):
        if isinstance(getattr(pass_config, field.name), bool):
            setattr(pass_config, field.name, False)
    # As on the TPU platform, only the exact buckets are compiled.
    type(config.compilation_config).get_compile_ranges = lambda self: []
    return config


def _set_mapping(model, token_ids):
    """Hands the wrapper a step's mapping, as the LoRA manager does."""
    model.punica_wrapper.update_metadata(
        LoRAMapping(tuple(token_ids), (0,) * 8), LORA_INDEX_TO_ID, NUM_SLOTS + 1, 0
    )


def _token_ids(num_tokens):
    """Each adapter and no adapter, in turn."""
    return [(9, 7, 5, 0)[t % 4] for t in range(num_tokens)]


def _forward(model, config, x):
    with set_forward_context({}, config):
        return model(x).float().cpu()


def _warm(model, config) -> int:
    """Precompiles every bucket as the runner does at startup, before any
    adapter is loaded. Returns how many extra traces that took."""
    retraces = 0
    with set_current_vllm_config(config), shape_variants.warmup() as tracked:
        for num_tokens in BUCKETS:
            _set_mapping(model, [0] * num_tokens)
            if tracked.needs_retrace(num_tokens):
                shape_variants.retrace(model, config)
                retraces += 1
            _forward(model, config, _inputs(num_tokens, seed=0))
    return retraces


def _load_adapters(model, seed):
    """Writes new adapters into every slot in place, as set_lora does."""
    for slot in range(NUM_SLOTS):
        model.lora_a_stacked[0][slot, 0].copy_(_weight(RANK, HIDDEN, seed=seed + slot))
        model.lora_b_stacked[0][slot, 0].copy_(_weight(OUT, RANK, seed=seed - slot))


def _reference(model, x, token_ids):
    """The base layer plus each token's adapter, in fp32 on the host."""
    x = x.float().cpu()
    out = x @ model.weight.float().cpu().T
    lora_a = model.lora_a_stacked[0].float().cpu()
    lora_b = model.lora_b_stacked[0].float().cpu()
    for t, lora_id in enumerate(token_ids):
        slot = SLOT_OF_ID[lora_id]
        if slot >= 0:
            out[t] += lora_b[slot, 0] @ (lora_a[slot, 0] @ x[t])
    return out


def _one_trace_serves_every_bucket(cache_root):
    model, config, compiles = _setup(cache_root)

    assert _warm(model, config) == 0
    assert len(compiles) == len(BUCKETS)
    for num_tokens in BUCKETS:
        x = _inputs(num_tokens, seed=num_tokens)
        _set_mapping(model, [0] * num_tokens)
        torch.testing.assert_close(
            _forward(model, config, x),
            _reference(model, x, [0] * num_tokens),
            **TOLERANCE,
        )


def test_one_trace_serves_every_bucket(tmp_path):
    run_in_isolated_process(partial(_one_trace_serves_every_bucket, str(tmp_path)))


def _swapped_adapters_apply_without_recompiling(cache_root):
    model, config, compiles = _setup(cache_root)
    _warm(model, config)
    num_compiles = len(compiles)

    # The second round replaces adapters that are already in use.
    for seed in (100, 200):
        _load_adapters(model, seed)
        for num_tokens in BUCKETS:
            x = _inputs(num_tokens, seed=num_tokens)
            token_ids = _token_ids(num_tokens)
            _set_mapping(model, token_ids)
            torch.testing.assert_close(
                _forward(model, config, x),
                _reference(model, x, token_ids),
                **TOLERANCE,
            )
    assert len(compiles) == num_compiles


def test_swapped_adapters_apply_without_recompiling(tmp_path):
    run_in_isolated_process(
        partial(_swapped_adapters_apply_without_recompiling, str(tmp_path))
    )


def _rows_without_an_adapter_match_the_base_layer(cache_root):
    model, config, _ = _setup(cache_root)
    _warm(model, config)
    x = _inputs(64, seed=64)
    token_ids = _token_ids(64)
    _set_mapping(model, token_ids)

    # No adapter is loaded yet, so every row's LoRA delta is exactly zero.
    base = _forward(model, config, x)
    _load_adapters(model, 100)
    mixed = _forward(model, config, x)

    no_adapter = torch.tensor(token_ids) == 0
    assert torch.equal(mixed[no_adapter], base[no_adapter])
    assert not torch.equal(mixed[~no_adapter], base[~no_adapter])


def test_rows_without_an_adapter_match_the_base_layer(tmp_path):
    """Same program, inputs and mapping: a row without an adapter comes out
    bit-identical to the base layer while other rows use adapters."""
    run_in_isolated_process(
        partial(_rows_without_an_adapter_match_the_base_layer, str(tmp_path))
    )


def _logits_follow_each_requests_adapter_without_recompiling():
    vocab, padded_vocab = 1000, 1024
    punica = PunicaWrapperTPU(16, 8, torch.device("tpu"))
    lm_head = _weight(padded_vocab, HIDDEN, seed=0).to("tpu")
    lora_a = _zeros(NUM_SLOTS, 1, RANK, HIDDEN)
    lora_b = _zeros(NUM_SLOTS, 1, vocab, RANK)
    compiles = []

    def counted_tpu_backend(graph, example_inputs):
        compiles.append(graph)
        return torch._dynamo.lookup_backend("tpu")(graph, example_inputs)

    @torch.compile(backend=counted_tpu_backend, fullgraph=True, dynamic=False)
    def compute_logits(hidden_states):
        logits = hidden_states @ lm_head.T
        return punica.add_lora_logits(logits, hidden_states, lora_a, lora_b, 1.0)

    for seed in (100, 200):
        for slot in range(NUM_SLOTS):
            lora_a[slot, 0].copy_(_weight(RANK, HIDDEN, seed=seed + slot))
            lora_b[slot, 0].copy_(_weight(vocab, RANK, seed=seed - slot))
        for num_reqs in (8, 16):
            for request_ids in (_token_ids(num_reqs), _token_ids(num_reqs)[::-1]):
                punica.update_metadata(
                    LoRAMapping((0,) * num_reqs, tuple(request_ids)),
                    LORA_INDEX_TO_ID,
                    NUM_SLOTS + 1,
                    vocab,
                )
                h = _inputs(num_reqs, seed=num_reqs)
                got = compute_logits(h).float().cpu()

                want = h.float().cpu() @ lm_head.float().cpu().T
                for row, lora_id in enumerate(request_ids):
                    slot = SLOT_OF_ID[lora_id]
                    if slot >= 0:
                        a = lora_a[slot, 0].float().cpu()
                        b = lora_b[slot, 0].float().cpu()
                        want[row, :vocab] += b @ (a @ h[row].float().cpu())
                torch.testing.assert_close(got, want, **TOLERANCE)
    assert len(compiles) == 2


def test_logits_follow_each_requests_adapter_without_recompiling():
    """The runner compiles the logits with guards kept, so a mapping or adapter
    change must not count as a new input."""
    run_in_isolated_process(_logits_follow_each_requests_adapter_without_recompiling)
