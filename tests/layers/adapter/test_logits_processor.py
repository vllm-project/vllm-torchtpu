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
"""Equivalence and ordering tests for TpuLogitsProcessor."""

from types import SimpleNamespace
from typing import cast

import pytest
import torch
import vllm.lora.layers.logits_processor as lora_lp_mod
import vllm.model_executor.layers.logits_processor as logits_processor_mod
from vllm.config import LoRAConfig
from vllm.lora.layers import LogitsProcessorWithLoRA, LoRAMapping
from vllm.model_executor.custom_op import maybe_get_oot_by_class, op_registry_oot
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)

import vllm_torchtpu.layers  # noqa: F401 - triggers registration side effect
from vllm_torchtpu.layers.adapter.logits_processor import TpuLogitsProcessor
from vllm_torchtpu.lora.torch_punica_tpu import PunicaWrapperTPU


def _make_processor(
    cls: type[LogitsProcessor],
    *,
    org_vocab_size: int,
    vocab_size: int | None = None,
    soft_cap: float | None = 30.0,
    scale: float = 1.0,
    logits_as_input: bool = False,
) -> LogitsProcessor:
    proc = object.__new__(cls)
    torch.nn.Module.__init__(proc)
    proc.vocab_size = vocab_size or org_vocab_size
    proc.org_vocab_size = org_vocab_size
    proc.soft_cap = soft_cap
    proc.scale = scale
    proc.logits_as_input = logits_as_input
    proc.use_all_gather = True
    proc.head_dtype = None
    return proc


def _make_lm_head(tp_size: int, weight_shard: torch.Tensor) -> VocabParallelEmbedding:
    def apply(_layer: object, hidden: torch.Tensor, bias: torch.Tensor | None = None):
        out = hidden @ weight_shard.T
        if bias is not None:
            out = out + bias
        return out

    return cast(
        "VocabParallelEmbedding",
        SimpleNamespace(
            tp_size=tp_size,
            quant_method=SimpleNamespace(apply=apply),
        ),
    )


def _run_tp_group(
    monkeypatch: pytest.MonkeyPatch,
    forward_fn,
    proc: LogitsProcessor,
    w_shards: list[torch.Tensor],
    hidden: torch.Tensor,
    bias_shards: list[torch.Tensor | None],
    *,
    skip_gather: bool = False,
) -> tuple[list[torch.Tensor | None], list[torch.Tensor]]:
    """Simulate a TP all-gather across all ranks in two passes."""
    tp_size = len(w_shards)
    captured_inputs: list[torch.Tensor] = []

    def capture_all_gather(tensor: torch.Tensor) -> torch.Tensor:
        captured_inputs.append(tensor.clone())
        return tensor.repeat(1, tp_size)

    monkeypatch.setattr(
        logits_processor_mod, "tensor_model_parallel_all_gather", capture_all_gather
    )
    for r in range(tp_size):
        head = _make_lm_head(tp_size, w_shards[r])
        forward_fn(proc, head, hidden, bias_shards[r], skip_gather)

    if not captured_inputs:
        outputs = [
            forward_fn(
                proc,
                _make_lm_head(tp_size, w_shards[r]),
                hidden,
                bias_shards[r],
                skip_gather,
            )
            for r in range(tp_size)
        ]
        return outputs, []

    gathered = torch.cat(captured_inputs, dim=-1)
    monkeypatch.setattr(
        logits_processor_mod,
        "tensor_model_parallel_all_gather",
        lambda _tensor: gathered.clone(),
    )
    outputs = [
        forward_fn(
            proc,
            _make_lm_head(tp_size, w_shards[r]),
            hidden,
            bias_shards[r],
            skip_gather,
        )
        for r in range(tp_size)
    ]
    return outputs, captured_inputs


def test_registered_under_the_upstream_class_name() -> None:
    assert op_registry_oot["LogitsProcessor"] is TpuLogitsProcessor
    assert maybe_get_oot_by_class(LogitsProcessor) is TpuLogitsProcessor
    assert isinstance(LogitsProcessor.__new__(LogitsProcessor), TpuLogitsProcessor)
    assert TpuLogitsProcessor.forward is not LogitsProcessor.forward


@pytest.mark.parametrize("tp_size,org_vocab_size", [(2, 13), (4, 16)])
@pytest.mark.parametrize(
    "soft_cap,scale",
    [(30.0, 1.0), (None, 0.5), (30.0, 0.5), (None, 1.0)],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("with_bias", [False, True])
def test_matches_upstream_and_applies_pre_gather(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    org_vocab_size: int,
    soft_cap: float | None,
    scale: float,
    dtype: torch.dtype,
    with_bias: bool,
) -> None:
    rng = torch.Generator().manual_seed(42)
    shard_vocab = (org_vocab_size + tp_size - 1) // tp_size
    padded_vocab = shard_vocab * tp_size
    hidden_dim, tokens = 8, 3

    w_full = (torch.randn(padded_vocab, hidden_dim, generator=rng) * 20.0).to(dtype)
    hidden = torch.randn(tokens, hidden_dim, generator=rng).to(dtype)
    b_full = torch.randn(padded_vocab, generator=rng).to(dtype) if with_bias else None

    w_shards = list(w_full.chunk(tp_size, dim=0))
    b_shards: list[torch.Tensor | None] = (
        list(b_full.chunk(tp_size, dim=0)) if b_full is not None else [None] * tp_size
    )

    tpu_proc = _make_processor(
        TpuLogitsProcessor,
        org_vocab_size=org_vocab_size,
        vocab_size=padded_vocab,
        soft_cap=soft_cap,
        scale=scale,
    )
    ref_proc = _make_processor(
        LogitsProcessor,
        org_vocab_size=org_vocab_size,
        vocab_size=padded_vocab,
        soft_cap=soft_cap,
        scale=scale,
    )

    ours, our_pre_gather = _run_tp_group(
        monkeypatch, TpuLogitsProcessor.forward, tpu_proc, w_shards, hidden, b_shards
    )
    upstream, ref_pre_gather = _run_tp_group(
        monkeypatch, LogitsProcessor.forward, ref_proc, w_shards, hidden, b_shards
    )

    for rank in range(tp_size):
        assert ours[rank] is not None and upstream[rank] is not None
        assert ours[rank].shape == (tokens, org_vocab_size)
        assert torch.equal(ours[rank], upstream[rank])

    if soft_cap is not None or scale != 1.0:
        # Pre-gather verification: each shard fed into all-gather must already
        # carry soft_cap and scale, unlike upstream's raw shard logits.
        assert len(our_pre_gather) == tp_size
        for rank in range(tp_size):
            expected_shard = ref_pre_gather[rank]
            if soft_cap is not None:
                expected_shard = torch.tanh(expected_shard / soft_cap) * soft_cap
            if scale != 1.0:
                expected_shard = expected_shard * scale
            assert torch.equal(our_pre_gather[rank], expected_shard)
            assert not torch.equal(our_pre_gather[rank], ref_pre_gather[rank])


def test_delegates_tp1_skip_gather_and_logits_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = torch.Generator().manual_seed(7)
    w = (torch.randn(8, 4, generator=rng) * 15.0).to(torch.bfloat16)
    hidden = torch.randn(2, 4, generator=rng).to(torch.bfloat16)

    tpu_proc = _make_processor(
        TpuLogitsProcessor, org_vocab_size=6, vocab_size=8, soft_cap=30.0, scale=0.5
    )
    ref_proc = _make_processor(
        LogitsProcessor, org_vocab_size=6, vocab_size=8, soft_cap=30.0, scale=0.5
    )

    # TP=1
    ours_tp1, _ = _run_tp_group(
        monkeypatch, TpuLogitsProcessor.forward, tpu_proc, [w], hidden, [None]
    )
    ref_tp1, _ = _run_tp_group(
        monkeypatch, LogitsProcessor.forward, ref_proc, [w], hidden, [None]
    )
    assert ours_tp1[0] is not None and ref_tp1[0] is not None
    assert torch.equal(ours_tp1[0], ref_tp1[0])

    # skip_gather=True at TP=2
    w_shards = list(w.chunk(2, dim=0))
    ours_skip, calls = _run_tp_group(
        monkeypatch,
        TpuLogitsProcessor.forward,
        tpu_proc,
        w_shards,
        hidden,
        [None, None],
        skip_gather=True,
    )
    ref_skip, _ = _run_tp_group(
        monkeypatch,
        LogitsProcessor.forward,
        ref_proc,
        w_shards,
        hidden,
        [None, None],
        skip_gather=True,
    )
    assert calls == []
    assert torch.equal(ours_skip[0], ref_skip[0])

    # logits_as_input=True
    tpu_proc_in = _make_processor(
        TpuLogitsProcessor,
        org_vocab_size=6,
        soft_cap=30.0,
        scale=0.5,
        logits_as_input=True,
    )
    ref_proc_in = _make_processor(
        LogitsProcessor,
        org_vocab_size=6,
        soft_cap=30.0,
        scale=0.5,
        logits_as_input=True,
    )
    raw_logits = (torch.randn(2, 6, generator=rng) * 25.0).to(torch.bfloat16)
    head_tp2 = _make_lm_head(2, w_shards[0])
    assert torch.equal(
        tpu_proc_in.forward(head_tp2, raw_logits.clone()),
        LogitsProcessor.forward(ref_proc_in, head_tp2, raw_logits.clone()),
    )


def test_real_logits_processor_with_lora_delegates_to_upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real LogitsProcessorWithLoRA calls type(self.base_layer).forward(self, ...)
    unbound and adds its LoRA delta post-gather inside _get_logits before
    soft_cap and scale run."""
    rng = torch.Generator().manual_seed(11)
    # org_vocab_size=6 padded to 8 across TP=2: shard 0 holds [0, 1, 2, pad],
    # shard 1 holds [3, 4, 5, pad], reindexed by sharded_to_full_mapping.
    sharded_to_full = [0, 1, 2, 4, 5, 6, 3, 7]
    w_full = (torch.randn(8, 4, generator=rng) * 10.0).to(torch.bfloat16)
    hidden = torch.randn(2, 4, generator=rng).to(torch.bfloat16)
    w_shards = list(w_full.chunk(2, dim=0))

    monkeypatch.setattr(lora_lp_mod, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(lora_lp_mod, "get_tensor_model_parallel_rank", lambda: 0)
    gathered_base = hidden @ w_full.T
    monkeypatch.setattr(
        logits_processor_mod,
        "tensor_model_parallel_all_gather",
        lambda _tensor: gathered_base.clone(),
    )

    def build_wrapped(
        cls: type[LogitsProcessor], *, active_adapter: bool
    ) -> LogitsProcessorWithLoRA:
        base = _make_processor(
            cls, org_vocab_size=6, vocab_size=8, soft_cap=30.0, scale=0.5
        )
        wrapped = LogitsProcessorWithLoRA(
            base, 4, torch.bfloat16, torch.device("cpu"), sharded_to_full
        )
        wrapped.create_lora_weights(
            2, LoRAConfig(max_lora_rank=8, max_loras=2, lora_dtype=torch.bfloat16)
        )
        wrapped.lora_a_stacked[1, 0, 0, :] = 1.0
        wrapped.lora_b_stacked[1, 0, :6, 0] = 5.0
        punica = PunicaWrapperTPU(8, 2, "cpu")
        prompt_map = (3, 0) if active_adapter else (0, 0)
        punica.update_metadata(LoRAMapping(prompt_map, prompt_map), [None, 3], 3, 8)
        wrapped.set_mapping(punica)
        return wrapped

    head_tp2 = _make_lm_head(2, w_shards[0])
    out_lora = build_wrapped(TpuLogitsProcessor, active_adapter=True)(head_tp2, hidden)
    ref_lora = build_wrapped(LogitsProcessor, active_adapter=True)(head_tp2, hidden)
    out_no_lora = build_wrapped(TpuLogitsProcessor, active_adapter=False)(
        head_tp2, hidden
    )

    assert out_lora is not None and ref_lora is not None and out_no_lora is not None
    assert torch.equal(out_lora, ref_lora)
    # Control: request 0 carries adapter 3's delta; request 1 carries none.
    assert not torch.equal(out_lora[0], out_no_lora[0])
    assert torch.equal(out_lora[1], out_no_lora[1])
