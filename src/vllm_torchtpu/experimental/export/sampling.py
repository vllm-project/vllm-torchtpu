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
"""Model-independent token samplers lowered into every exported artifact."""

from __future__ import annotations

import os
from typing import Any

import torch

from vllm_torchtpu.experimental.export import api, tpu_lowering


def greedy_tokens(logits: torch.Tensor) -> torch.Tensor:
    """Returns the argmax token id i64[R, 1] of each row of logits[R, V]."""
    return torch.argmax(logits, dim=-1, keepdim=True).to(torch.int64)


def stochastic_tokens(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    min_p: torch.Tensor,
    uniform: torch.Tensor,
) -> torch.Tensor:
    """Samples one token id i64[R, 1] per row of logits[R, V].

    Each row is sampled with its own parameters. A row with `temperature <= 1e-5`
    takes the argmax. Otherwise the logits are divided by the temperature, sorted
    by descending probability (ties by ascending token id), and filtered:

    - Top-K keeps the `top_k` most likely tokens (`top_k <= 0` keeps all).
    - Top-P keeps the smallest prefix whose probability reaches `top_p`
      (`top_p >= 1` keeps all).
    - Min-P keeps tokens at least `min_p` times as likely as the most likely
      token (`min_p <= 0` keeps all).

    The most likely token always survives. The token is the first one whose
    cumulative probability over the kept tokens exceeds `uniform`.
    """
    logits = logits.to(torch.float32)
    vocab = int(logits.shape[-1])
    temp_f32 = temperature.to(torch.float32)
    scaled = logits / torch.clamp(temp_f32, min=1e-5).unsqueeze(-1)
    sorted_logits, order = torch.sort(scaled, dim=-1, descending=True, stable=True)
    ranks = torch.arange(vocab, device=logits.device, dtype=torch.int32).unsqueeze(0)
    top_k_i32 = top_k.to(torch.int32)
    vocab_t = torch.full_like(top_k_i32, vocab)
    k = torch.where(top_k_i32 > 0, torch.minimum(top_k_i32, vocab_t), vocab_t)
    keep = ranks < k.unsqueeze(-1)
    neg_inf = torch.full_like(sorted_logits, -float("inf"))
    probs = torch.softmax(torch.where(keep, sorted_logits, neg_inf), dim=-1)
    keep = keep & (
        (torch.cumsum(probs, dim=-1) - probs) < top_p.to(torch.float32).unsqueeze(-1)
    )
    keep = keep & (probs >= min_p.to(torch.float32).unsqueeze(-1) * probs[:, :1])
    keep = keep | (ranks == 0)
    probs = torch.softmax(torch.where(keep, sorted_logits, neg_inf), dim=-1)
    cdf = torch.cumsum(probs, dim=-1)
    threshold = uniform.to(torch.float32).unsqueeze(-1) * cdf[:, -1:]
    index = torch.sum((cdf <= threshold).to(torch.int64), dim=-1)
    max_idx = torch.sum(keep.to(torch.int64), dim=-1) - 1
    index = torch.minimum(index, max_idx)
    sampled = torch.gather(order.to(torch.int64), -1, index.unsqueeze(-1)).squeeze(-1)
    greedy = torch.argmax(logits, dim=-1).to(torch.int64)
    tokens = torch.where(temp_f32 <= 1e-5, greedy, sampled)
    return tokens.unsqueeze(-1)


def penalized_tokens(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    min_p: torch.Tensor,
    uniform: torch.Tensor,
    repetition_penalty: torch.Tensor,
    presence_penalty: torch.Tensor,
    frequency_penalty: torch.Tensor,
    history: torch.Tensor,
    output_start: torch.Tensor,
) -> torch.Tensor:
    """Samples one token id i64[R, 1] per row after applying penalties."""
    logits = logits.to(torch.float32)
    rows, vocab = int(logits.shape[0]), int(logits.shape[1])
    hist_len = int(history.shape[1])
    valid = history >= 0
    ids = torch.where(valid, history, torch.zeros_like(history)).to(torch.int64)
    col_idx = torch.arange(hist_len, device=logits.device, dtype=torch.int32).unsqueeze(
        0
    )
    generated = valid & (col_idx >= output_start.to(torch.int32).unsqueeze(-1))
    seen_acc = torch.zeros((rows, vocab), dtype=torch.float32, device=logits.device)
    seen = seen_acc.scatter_add(1, ids, valid.to(torch.float32)) > 0
    counts_acc = torch.zeros((rows, vocab), dtype=torch.float32, device=logits.device)
    counts = counts_acc.scatter_add(1, ids, generated.to(torch.float32))
    rep = repetition_penalty.to(torch.float32).unsqueeze(-1)
    repeated = torch.where(logits > 0, logits / rep, logits * rep)
    logits = torch.where(seen, repeated, logits)
    logits = (
        logits
        - presence_penalty.to(torch.float32).unsqueeze(-1)
        * (counts > 0).to(torch.float32)
        - frequency_penalty.to(torch.float32).unsqueeze(-1) * counts
    )
    return stochastic_tokens(logits, temperature, top_k, top_p, min_p, uniform)


def dispatch_sample_tokens(
    logits: torch.Tensor,
    *,
    temperature: torch.Tensor | None = None,
    top_k: torch.Tensor | None = None,
    top_p: torch.Tensor | None = None,
    min_p: torch.Tensor | None = None,
    uniform: torch.Tensor | None = None,
    repetition_penalty: torch.Tensor | None = None,
    presence_penalty: torch.Tensor | None = None,
    frequency_penalty: torch.Tensor | None = None,
    history: torch.Tensor | None = None,
    output_start: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dispatches `logits` to the greedy, stochastic or penalized sampler."""
    if temperature is None:
        return greedy_tokens(logits)
    rows = int(logits.shape[0])
    dev = logits.device
    tk = (
        top_k
        if top_k is not None
        else torch.zeros((rows,), dtype=torch.int32, device=dev)
    )
    tp = (
        top_p
        if top_p is not None
        else torch.ones((rows,), dtype=torch.float32, device=dev)
    )
    mp = (
        min_p
        if min_p is not None
        else torch.zeros((rows,), dtype=torch.float32, device=dev)
    )
    unif = (
        uniform
        if uniform is not None
        else torch.zeros((rows,), dtype=torch.float32, device=dev)
    )
    if (
        repetition_penalty is not None
        and presence_penalty is not None
        and frequency_penalty is not None
        and history is not None
        and output_start is not None
    ):
        return penalized_tokens(
            logits,
            temperature,
            tk,
            tp,
            mp,
            unif,
            repetition_penalty,
            presence_penalty,
            frequency_penalty,
            history,
            output_start,
        )
    return stochastic_tokens(logits, temperature, tk, tp, mp, unif)


def export_samplers(
    output_dir: str,
    subgraph_index: int,
    logits_shape: tuple[int, ...],
    logits_dtype: Any,
    logits_dtype_name: str,
    history_len: int,
    *,
    subdir: str = "",
) -> dict[str, dict[str, Any]]:
    """Writes the samplers over `[R, V]` logits and returns their metadata variants."""
    req_size = logits_shape[0]
    logits_spec = api.ShapeDtypeStruct(logits_shape, logits_dtype_name)
    param_inputs = [
        ("temperature", "f32"),
        ("top_k", "i32"),
        ("top_p", "f32"),
        ("min_p", "f32"),
        ("uniform", "f32"),
    ]
    penalty_inputs = [
        ("repetition_penalty", (req_size,), "f32"),
        ("presence_penalty", (req_size,), "f32"),
        ("frequency_penalty", (req_size,), "f32"),
        ("history", (req_size, history_len), "i32"),
        ("output_start", (req_size,), "i32"),
    ]
    param_specs = [
        api.ShapeDtypeStruct((req_size,), dtype_str) for _, dtype_str in param_inputs
    ]
    penalty_specs = [
        api.ShapeDtypeStruct(shape, dtype_str) for _, shape, dtype_str in penalty_inputs
    ]
    greedy_mlir = tpu_lowering.lower_torch_callable_to_mlir(
        greedy_tokens,
        [logits_spec],
        module_name=f"sample_tokens_req{req_size}",
    )
    stochastic_mlir = tpu_lowering.lower_torch_callable_to_mlir(
        stochastic_tokens,
        [logits_spec, *param_specs],
        module_name=f"sample_tokens_stochastic_req{req_size}",
    )
    penalized_mlir = tpu_lowering.lower_torch_callable_to_mlir(
        penalized_tokens,
        [logits_spec, *param_specs, *penalty_specs],
        module_name=f"sample_tokens_penalized_req{req_size}",
    )

    def _tensor_entry(
        arg_index: int,
        name: str,
        shape: tuple[int, ...] | list[int],
        dtype: str,
        argument_type: str,
    ) -> dict[str, Any]:
        return {
            "arg_index": arg_index,
            "index": arg_index,
            "name": name,
            "shape": list(shape),
            "dtype": dtype,
            "argument_type": argument_type,
            "is_weight": False,
            "weight_name": "",
            "buffer_donor": False,
        }

    logits_input = _tensor_entry(
        0, "logits", logits_shape, logits_dtype_name, "intermediate"
    )
    tokens_output = _tensor_entry(0, "tokens", (req_size, 1), "i64", "user_io")
    stochastic_inputs = [logits_input] + [
        _tensor_entry(i, name, (req_size,), dtype_name, "user_io")
        for i, (name, dtype_name) in enumerate(param_inputs, start=1)
    ]
    penalized_inputs = stochastic_inputs + [
        _tensor_entry(i, name, shape, dtype_name, "user_io")
        for i, (name, shape, dtype_name) in enumerate(
            penalty_inputs, start=len(stochastic_inputs)
        )
    ]

    target_dir = os.path.join(output_dir, subdir) if subdir else output_dir
    os.makedirs(target_dir, exist_ok=True)

    variants: dict[str, dict[str, Any]] = {}
    for function, mlir, inputs in [
        ("sample_tokens", greedy_mlir, [logits_input]),
        ("sample_tokens_stochastic", stochastic_mlir, stochastic_inputs),
        ("sample_tokens_penalized", penalized_mlir, penalized_inputs),
    ]:
        file_name = f"{function}_req{req_size}.mlir"
        rel_path = os.path.join(subdir, file_name) if subdir else file_name
        with open(os.path.join(output_dir, rel_path), "w") as f:
            f.write(mlir)
        variants[function] = {
            "file_path": rel_path,
            "subgraph_index": subgraph_index,
            "compile_size": req_size,
            "inputs": inputs,
            "outputs": [tokens_output],
        }
    return variants
