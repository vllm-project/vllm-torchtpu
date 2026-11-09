# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.kernels import fused_greedy_head as kernel
from vllm_torchtpu.layers.adapter.sample.fused_greedy_head import FusedGreedyHead
from vllm_torchtpu.models.vllm.deepseek_v41.v2.model import DeepseekV41ForCausalLM
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def _runner(weight):
    model = DeepseekV41ForCausalLM.__new__(DeepseekV41ForCausalLM)
    torch.nn.Module.__init__(model)
    model.lm_head = torch.nn.Module()
    model.lm_head.weight = torch.nn.Parameter(weight, requires_grad=False)
    return SimpleNamespace(
        model=model,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1
        ),
        speculative_config=SimpleNamespace(method="dspark", num_speculative_tokens=5),
        max_num_reqs=48,
        lora_config=None,
        input_batch=SimpleNamespace(all_greedy=True, max_num_logprobs=None),
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize("backend", [None, "auto", "pallas", "logits"])
def test_greedy_head_backend_selection(monkeypatch, backend):
    monkeypatch.delenv("TPU_GREEDY_HEAD_BACKEND", raising=False)
    if backend is not None:
        monkeypatch.setenv("TPU_GREEDY_HEAD_BACKEND", backend)
    selected = envs.environment_variables["TPU_GREEDY_HEAD_BACKEND"]()
    assert selected == (backend or "auto")
    monkeypatch.setattr(envs, "TPU_GREEDY_HEAD_BACKEND", selected)
    monkeypatch.setattr(kernel, "get_device_name", lambda: "TPU v7")
    runner = _runner(torch.empty(129280, 5120, dtype=torch.bfloat16, device="meta"))
    TPUModelRunner._init_fused_greedy_head(runner)
    head = runner._fused_greedy_head
    if selected == "logits":
        assert head is None
    else:
        assert head.weight is runner.model.lm_head.weight
        assert head.rows == 288


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "unsupported",
    [
        "model",
        "tp",
        "pp",
        "lora",
        "no_spec",
        "method",
        "rows",
        "shape",
        "dtype",
        "device",
    ],
)
def test_auto_greedy_head_falls_back_and_forced_mode_rejects(monkeypatch, unsupported):
    weight = torch.empty(129280, 5120, dtype=torch.bfloat16, device="meta")
    if unsupported == "shape":
        weight = torch.empty(128, 5120, dtype=torch.bfloat16, device="meta")
    elif unsupported == "dtype":
        weight = weight.float()
    runner = _runner(weight)
    monkeypatch.setattr(
        kernel,
        "get_device_name",
        lambda: "TPU v6" if unsupported == "device" else "TPU v7",
    )
    if unsupported == "model":
        runner.model = torch.nn.Module()
    elif unsupported in ("tp", "pp"):
        setattr(
            runner.parallel_config,
            "tensor_parallel_size" if unsupported == "tp" else "pipeline_parallel_size",
            2,
        )
    elif unsupported == "lora":
        runner.lora_config = object()
    elif unsupported == "no_spec":
        runner.speculative_config = None
    elif unsupported == "method":
        runner.speculative_config.method = "eagle"
    elif unsupported == "rows":
        runner.max_num_reqs = 1
    monkeypatch.setattr(envs, "TPU_GREEDY_HEAD_BACKEND", "auto")
    TPUModelRunner._init_fused_greedy_head(runner)
    assert runner._fused_greedy_head is None
    monkeypatch.setattr(envs, "TPU_GREEDY_HEAD_BACKEND", "pallas")
    with pytest.raises(AssertionError, match="Forced fused greedy head"):
        TPUModelRunner._init_fused_greedy_head(runner)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "fallback", [None, "rows", "metadata", "sampling", "grammar", "logprobs"]
)
def test_greedy_head_request_dispatch(monkeypatch, fallback):
    monkeypatch.setattr(envs, "TPU_GREEDY_HEAD_BACKEND", "auto")
    monkeypatch.setattr(kernel, "get_device_name", lambda: "TPU v7")
    runner = _runner(torch.empty(129280, 5120, dtype=torch.bfloat16, device="meta"))
    TPUModelRunner._init_fused_greedy_head(runner)
    scheduler = SimpleNamespace(has_structured_output_requests=fallback == "grammar")
    runner.input_batch.all_greedy = fallback != "sampling"
    runner.input_batch.max_num_logprobs = 1 if fallback == "logprobs" else None
    enabled = TPUModelRunner._can_use_fused_greedy_head(
        runner,
        scheduler,
        None if fallback == "metadata" else object(),
        1 if fallback == "rows" else 288,
    )
    assert enabled is (fallback is None)


def test_fused_greedy_head_cpu_reference(device, monkeypatch):
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
    monkeypatch.delenv("TPU_GREEDY_HEAD_BACKEND", raising=False)
    monkeypatch.setattr(
        envs,
        "TPU_GREEDY_HEAD_BACKEND",
        envs.environment_variables["TPU_GREEDY_HEAD_BACKEND"](),
    )
    runner = _runner(device_weight)
    TPUModelRunner._init_fused_greedy_head(runner)
    head = runner._fused_greedy_head
    assert isinstance(head, FusedGreedyHead)
    assert head.weight is runner.model.lm_head.weight
    compiled = torch.compile(head, backend="tpu", fullgraph=True, dynamic=False)
    actual = compiled(hidden.to(device)).cpu()
    torch.testing.assert_close(actual[:4].long(), expected, rtol=0, atol=0)
    assert actual[0].item() == 1279

    # The compiled head must read the shared weight, not a load-time snapshot.
    device_weight[1279, 0] = 0
    updated = compiled(hidden.to(device)).cpu()
    assert updated[0].item() == 1280


@pytest.mark.cpu_test
def test_greedy_head_uses_multimodal_language_model(monkeypatch):
    monkeypatch.setattr(envs, "TPU_GREEDY_HEAD_BACKEND", "auto")
    monkeypatch.setattr(kernel, "get_device_name", lambda: "TPU v7")
    runner = _runner(torch.empty(129280, 5120, dtype=torch.bfloat16, device="meta"))
    language_model = runner.model
    runner.model = SimpleNamespace(
        supports_multimodal=True, get_language_model=lambda: language_model
    )
    TPUModelRunner._init_fused_greedy_head(runner)
    assert runner._fused_greedy_head.weight is language_model.lm_head.weight


def _reference_first_argmax(logits):
    """Serving semantics: BF16 round, a row with NaN takes its first NaN,
    otherwise the first column holding the maximum (-0 == +0)."""
    import numpy as np

    rounded = np.asarray(
        torch.from_numpy(logits).to(torch.bfloat16).float().numpy(), np.float64
    )
    out = []
    for row in rounded:
        nan = np.flatnonzero(np.isnan(row))
        out.append(nan[0] if nan.size else int(np.flatnonzero(row == row.max())[0]))
    return np.asarray(out)


@pytest.mark.cpu_test
def test_packed_key_argmax_matches_first_index_semantics():
    """The packed int32 key must reproduce the two-pass argmax it replaced:
    a wrong order of keys, a lost tie rule or a missed NaN would pick a
    different token for real greedy decoding."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    rng = np.random.default_rng(0)
    rows, tile, tiles = 16, 128, 3
    logits = rng.standard_normal((rows, tile * tiles)).astype(np.float32)
    logits[0, [5, 300]] = 7.0  # tie across tiles: first tile wins
    logits[1, [40, 41]] = 7.0  # tie inside one tile: first column wins
    logits[2, 200] = np.nan  # NaN beats every finite value
    logits[2, 3] = np.inf
    logits[3, [10, 250]] = np.nan  # earliest NaN across tiles
    logits[4] = -np.inf
    logits[4, 77] = -0.0
    logits[4, 20] = 0.0  # -0 and +0 tie, first column wins
    logits[5] = -1.0
    logits[5, 9] = -3.0e-8  # all negative: largest (closest to 0) wins
    logits[6, 30] = 5.0
    logits[6, 31] = 5.0001  # equal after BF16 rounding: first column wins
    logits[7] = -np.inf  # all -inf: column 0

    blocks = jnp.asarray(logits).reshape(rows, tiles, tile).transpose(1, 0, 2)
    packed = jax.vmap(kernel.packed_tile_argmax)(blocks)
    # Fold tiles in vocab order as the kernel does, from garbage initial state.
    key = jnp.full((rows,), 0x7FFFFFFF, jnp.int32)
    ids = jnp.full((rows,), -1, jnp.int32)
    for n in range(tiles):
        key, ids = kernel.merge_tile(key, ids, packed[n], n * tile, n == 0)
    got = np.asarray(ids)
    np.testing.assert_array_equal(got, _reference_first_argmax(logits))
    assert got[:8].tolist() == [5, 40, 200, 10, 20, 9, 30, 0]
