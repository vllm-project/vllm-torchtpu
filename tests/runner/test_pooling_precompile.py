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
"""CPU tests for the AOT-bucketed static pooler and pooling KV cap (PR2)."""

import inspect
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch._dynamo.testing import CompileCounter
from vllm.model_executor.layers.pooler import DispatchPooler
from vllm.model_executor.layers.pooler.activations import (
    PoolerClassify,
    PoolerNormalize,
)
from vllm.model_executor.layers.pooler.seqwise.heads import (
    ClassifierPoolerHead,
    EmbeddingPoolerHead,
)
from vllm.model_executor.layers.pooler.seqwise.methods import (
    CLSPool,
    LastPool,
    MeanPool,
)
from vllm.model_executor.layers.pooler.seqwise.poolers import SequencePooler
from vllm.pooling_params import PoolingParams
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates

from vllm_torchtpu import envs
from vllm_torchtpu.runner.tpu_pooling import (
    StaticPoolerChunkOutput,
    StaticPoolerSpec,
    _static_pool_impl,
    build_dummy_static_pool_inputs,
    build_static_pool_inputs,
    compute_pooling_kv_cache_cap_bytes,
    compute_pooling_kv_cache_cap_tokens,
    gate_unfinished,
    install_tpu_pooling_patches,
    iter_pooler_precompile_grid,
    resolve_static_pooler_spec,
    static_pool,
    unpack_static_pool_output,
)

pytestmark = pytest.mark.cpu_test

HIDDEN = 32
POOLING_METHODS = {"LAST": LastPool, "CLS": CLSPool, "MEAN": MeanPool}


@pytest.fixture(autouse=True)
def setup_patches():
    # Serving installs these patches; parity must hold with them in place.
    install_tpu_pooling_patches()


def _import_runner():
    """tpu_runner needs the full TPU vLLM stack; skip where it is missing."""
    try:
        import vllm_torchtpu.runner.tpu_runner as runner_mod
    except ImportError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"tpu_runner is not importable in this environment: {exc}")
    return runner_mod


def _params(n, *, dims=None, use_act=None, task="embed"):
    dims = dims or [None] * n
    use_act = use_act or [True] * n
    return [
        PoolingParams(task=task, dimensions=d, use_activation=a)
        for d, a in zip(dims, use_act)
    ]


def _embed_pooler(pooling_type, *, projector=None, head_dtype=torch.float32):
    head = EmbeddingPoolerHead(
        projector=projector, head_dtype=head_dtype, activation=PoolerNormalize()
    )
    return SequencePooler(pooling=POOLING_METHODS[pooling_type](), head=head)


def _model_config(hidden=HIDDEN):
    return SimpleNamespace(get_hidden_size=lambda: hidden, embedding_size=hidden)


def _spec_for(pooler, hidden=HIDDEN):
    model = SimpleNamespace(pooler=DispatchPooler({"embed": pooler}))
    spec, reason = resolve_static_pooler_spec(model, _model_config(hidden))
    assert spec is not None, reason
    return spec


def _reference(pooler, hidden_states, prompt_lens, params):
    """Run the real vLLM pooler on the unpadded batch (full prefill)."""
    lens = np.asarray(prompt_lens, dtype=np.int32)
    meta = PoolingMetadata(
        prompt_lens=torch.from_numpy(lens.copy()),
        prompt_token_ids=None,
        prompt_token_ids_cpu=None,
        pooling_params=params,
        pooling_states=[PoolingStates() for _ in params],
    )
    meta.build_pooling_cursor(
        num_scheduled_tokens_np=lens,
        seq_lens_cpu=torch.from_numpy(lens.copy()),
        device=torch.device("cpu"),
    )
    total = int(lens.sum())
    out = pooler(hidden_states[:total], meta)
    return list(out) if isinstance(out, list) else list(out.unbind(0))


def _padded_hidden(prompt_lens, t_pad, *, dtype=torch.float32, pad_value=0.0):
    torch.manual_seed(0)
    total = sum(prompt_lens)
    hidden = torch.full((t_pad, HIDDEN), pad_value, dtype=dtype)
    hidden[:total] = torch.randn(total, HIDDEN).to(dtype)
    return hidden


def _run_static(spec, hidden, prompt_lens, params, t_pad, r_pad):
    n = len(prompt_lens)
    inputs = build_static_pool_inputs(
        spec,
        params,
        np.asarray(prompt_lens),
        np.zeros(n, dtype=np.int32),
        np.asarray(prompt_lens, dtype=np.int32),
        t_pad,
        r_pad,
        "cpu",
    )
    assert inputs is not None
    with torch.no_grad():
        padded = static_pool(hidden, inputs, spec)
    return inputs, padded


# ---------------------------------------------------------------------------
# Precompile grid
# ---------------------------------------------------------------------------


def test_precompile_pooler_grid_recipe_config():
    tokens = [16, 32, 64, 128, 256, 512]
    grid = list(iter_pooler_precompile_grid(tokens, [8, 16], 16))
    assert grid == [(t, r) for t in tokens for r in (8, 16)]
    assert len(grid) == 12


@pytest.mark.parametrize(
    ("max_num_seqs", "reqs", "expected"),
    [(64, [8, 16, 32, 64], 37), (256, [8, 16, 32, 64, 128, 256], 50)],
)
def test_precompile_pooler_grid_serving_config(max_num_seqs, reqs, expected):
    tokens = [16 << i for i in range(10)]  # 16 .. 8192
    grid = list(iter_pooler_precompile_grid(tokens, reqs, max_num_seqs))
    assert len(grid) == len(set(grid)) == expected
    assert all(r <= max(t, 8) for t, r in grid)

    # The grid is exactly the runtime-reachable set: n requests need at least
    # n tokens, and pad to max(8, next_pow2(n)) capped at max_num_seqs.
    def pad_reqs(n):
        return min(8 if n <= 8 else 1 << (n - 1).bit_length(), max_num_seqs)

    reachable = {
        (t, pad_reqs(n)) for n in range(1, max_num_seqs + 1) for t in tokens if t >= n
    }
    assert set(grid) == reachable


# ---------------------------------------------------------------------------
# Static pool parity with the vLLM pooler
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pooling_type", ["LAST", "CLS", "MEAN"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_static_pool_matches_vllm_pooler(pooling_type, dtype):
    prompt_lens = [3, 5, 1, 4]
    t_pad, r_pad = 16, 8
    pooler = _embed_pooler(pooling_type)
    spec = _spec_for(pooler)
    assert spec.pooling_type == pooling_type
    hidden = _padded_hidden(prompt_lens, t_pad, dtype=dtype)
    params = _params(len(prompt_lens))

    _, padded = _run_static(spec, hidden, prompt_lens, params, t_pad, r_pad)
    ref = torch.stack(_reference(pooler, hidden, prompt_lens, params))

    assert padded.shape == (r_pad, HIDDEN)
    assert padded.dtype == torch.float32
    if pooling_type == "MEAN":
        torch.testing.assert_close(padded[: len(prompt_lens)], ref, atol=1e-6, rtol=0)
    else:
        assert torch.equal(padded[: len(prompt_lens)], ref)


@pytest.mark.parametrize("pooling_type", ["LAST", "CLS", "MEAN"])
def test_static_pool_padding_is_inert(pooling_type):
    prompt_lens = [4, 2, 6]
    t_pad, r_pad = 16, 8
    pooler = _embed_pooler(pooling_type)
    spec = _spec_for(pooler)
    params = _params(len(prompt_lens))
    clean = _padded_hidden(prompt_lens, t_pad)
    dirty = clean.clone()
    total = sum(prompt_lens)
    dirty[total:] = float("nan")
    dirty[-1] = float("inf")

    _, out_clean = _run_static(spec, clean, prompt_lens, params, t_pad, r_pad)
    _, out_dirty = _run_static(spec, dirty, prompt_lens, params, t_pad, r_pad)

    assert torch.isfinite(out_dirty).all()
    assert torch.equal(out_dirty[:3], out_clean[:3])


def test_static_pool_matryoshka_and_mixed_activation():
    prompt_lens = [3, 4, 5]
    t_pad, r_pad = 16, 8
    pooler = _embed_pooler("LAST")
    spec = _spec_for(pooler)
    params = _params(3, dims=[None, 8, 16], use_act=[True, False, True])
    hidden = _padded_hidden(prompt_lens, t_pad)

    inputs, padded = _run_static(spec, hidden, prompt_lens, params, t_pad, r_pad)
    rows = unpack_static_pool_output(
        StaticPoolerChunkOutput(
            padded, inputs.num_reqs, inputs.finished, inputs.out_dims
        )
    )
    ref = _reference(pooler, hidden, prompt_lens, params)

    assert [r.shape[-1] for r in rows] == [HIDDEN, 8, 16]
    for got, want in zip(rows, ref):
        torch.testing.assert_close(got, want, atol=1e-6, rtol=0)


def test_static_pool_with_projector():
    torch.manual_seed(1)
    projector = torch.nn.Linear(HIDDEN, 12)
    prompt_lens = [2, 7]
    t_pad, r_pad = 16, 8
    for pooling_type in ("LAST", "MEAN"):
        pooler = _embed_pooler(pooling_type, projector=projector)
        spec = _spec_for(pooler)
        assert spec.projector is projector and spec.out_dim == 12
        params = _params(2, dims=[None, 5])
        hidden = _padded_hidden(prompt_lens, t_pad)
        inputs, padded = _run_static(spec, hidden, prompt_lens, params, t_pad, r_pad)
        rows = unpack_static_pool_output(
            StaticPoolerChunkOutput(
                padded, inputs.num_reqs, inputs.finished, inputs.out_dims
            )
        )
        with torch.no_grad():
            ref = _reference(pooler, hidden, prompt_lens, params)
        for got, want in zip(rows, ref):
            torch.testing.assert_close(got, want, atol=1e-6, rtol=0)


def test_static_pool_head_dtype_none_keeps_input_dtype():
    pooler = _embed_pooler("LAST", head_dtype=None)
    spec = _spec_for(pooler)
    assert spec.head_dtype is None
    hidden = _padded_hidden([3], 16, dtype=torch.bfloat16)
    _, padded = _run_static(spec, hidden, [3], _params(1), 16, 8)
    assert padded.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# Host-side input builders
# ---------------------------------------------------------------------------


def _spec(pooling_type="LAST", out_dim=HIDDEN):
    return StaticPoolerSpec(
        pooling_type=pooling_type,
        head_dtype=torch.float32,
        projector=None,
        normalize=True,
        out_dim=out_dim,
    )


def _build(spec, prompt, computed, scheduled, *, params=None, t_pad=16, r_pad=8, **kw):
    return build_static_pool_inputs(
        spec,
        params if params is not None else _params(len(prompt)),
        np.asarray(prompt),
        np.asarray(computed, dtype=np.int32),
        np.asarray(scheduled, dtype=np.int32),
        t_pad,
        r_pad,
        "cpu",
        **kw,
    )


def test_build_static_pool_inputs_last_chunked():
    # Request 0 (10 tokens) is on its first 6-token chunk; request 1 is whole.
    inputs = _build(_spec("LAST"), [10, 4], [0, 0], [6, 4])
    assert inputs is not None
    assert inputs.finished.tolist() == [False, True]
    assert inputs.gather_idx.dtype == torch.int32
    assert inputs.gather_idx.tolist() == [5, 9] + [9] * 6  # pad = total - 1
    assert inputs.seg_ids is None and inputs.lens is None

    # Final chunk of request 0 finishes it.
    last = _build(_spec("LAST"), [10], [6], [4])
    assert last.finished.tolist() == [True]
    assert last.gather_idx.tolist()[:1] == [3]


def test_build_static_pool_inputs_reuses_logits_indices():
    logits_indices = torch.tensor([2, 6, 6, 6, 6, 6, 6, 6], dtype=torch.int32)
    inputs = _build(
        _spec("LAST"), [3, 4], [0, 0], [3, 4], last_token_idx=logits_indices
    )
    assert inputs.gather_idx is logits_indices

    wrong_dtype = logits_indices.to(torch.int64)
    inputs = _build(_spec("LAST"), [3, 4], [0, 0], [3, 4], last_token_idx=wrong_dtype)
    assert inputs.gather_idx is not wrong_dtype
    assert inputs.gather_idx.dtype == torch.int32
    assert inputs.gather_idx.tolist() == logits_indices.tolist()

    wrong_shape = logits_indices[:4]
    inputs = _build(_spec("LAST"), [3, 4], [0, 0], [3, 4], last_token_idx=wrong_shape)
    assert inputs.gather_idx.shape == (8,)


def test_build_static_pool_inputs_cls_and_mean():
    cls = _build(_spec("CLS"), [3, 4], [0, 0], [3, 4])
    assert cls.gather_idx.tolist() == [0, 3] + [0] * 6

    mean = _build(_spec("MEAN"), [3, 4], [0, 0], [3, 4])
    assert mean.gather_idx is None
    assert mean.seg_ids.dtype == torch.int32
    assert mean.seg_ids.tolist() == [0] * 3 + [1] * 4 + [8] * 9  # pad -> R_pad
    assert mean.lens.dtype == torch.float32
    assert mean.lens.tolist() == [3.0, 4.0] + [1.0] * 6


@pytest.mark.parametrize("pooling_type", ["CLS", "MEAN"])
def test_build_static_pool_inputs_partial_prefill_ineligible(pooling_type):
    assert _build(_spec(pooling_type), [10, 4], [0, 0], [6, 4]) is None
    assert _build(_spec(pooling_type), [10], [6], [4]) is None


def test_build_static_pool_inputs_ineligible_chunks():
    spec = _spec("LAST")
    mixed = _params(1) + _params(1, task="token_embed")
    assert _build(spec, [3, 4], [0, 0], [3, 4], params=mixed) is None
    assert _build(spec, [3, 4], [0, 0], [3, 4], params=[None, None]) is None
    assert _build(spec, [10, 10], [0, 0], [10, 10], t_pad=16) is None  # > T_pad
    assert _build(spec, [1] * 9, [0] * 9, [1] * 9, r_pad=8) is None  # > R_pad
    assert _build(spec, [], [], [], params=[]) is None


def test_build_static_pool_inputs_dims_and_activation():
    params = _params(3, dims=[None, 10_000, 0], use_act=[True, False, True])
    inputs = _build(
        _spec("LAST", out_dim=HIDDEN), [2, 2, 2], [0] * 3, [2] * 3, params=params
    )
    assert inputs.dims.dtype == torch.int32
    assert inputs.dims.tolist() == [HIDDEN, HIDDEN, 1] + [HIDDEN] * 5
    assert inputs.out_dims == [None, HIDDEN, 1]
    assert inputs.use_act.dtype == torch.bool
    assert inputs.use_act.tolist() == [True, False, True] + [False] * 5


@pytest.mark.parametrize("pooling_type", ["LAST", "CLS", "MEAN"])
def test_dummy_inputs_match_runtime_signature(pooling_type):
    spec = _spec(pooling_type)
    real = _build(spec, [3, 4], [0, 0], [3, 4])
    dummy = build_dummy_static_pool_inputs(spec, 16, 8, "cpu")
    fields = ("gather_idx", "seg_ids", "lens", "dims", "use_act")
    for name in fields:
        a, b = getattr(real, name), getattr(dummy, name)
        assert (a is None) == (b is None), name
        if a is not None:
            assert (a.shape, a.dtype, a.stride()) == (b.shape, b.dtype, b.stride())
    ptrs = [
        getattr(dummy, n).data_ptr() for n in fields if getattr(dummy, n) is not None
    ]
    assert len(ptrs) == len(set(ptrs)), "dummy inputs must not alias"


# ---------------------------------------------------------------------------
# Output unpack and finished-mask gating (§2.6)
# ---------------------------------------------------------------------------


def test_unpack_static_pool_output_gates_and_slices():
    padded = torch.arange(8 * 6, dtype=torch.float32).reshape(8, 6)
    out = StaticPoolerChunkOutput(
        padded=padded,
        num_reqs=3,
        finished=np.array([True, False, True]),
        out_dims=[None, 4, 2],
    )
    rows = unpack_static_pool_output(out)
    assert len(rows) == 3  # padded rows dropped
    assert torch.equal(rows[0], padded[0])
    assert rows[1] is None
    assert torch.equal(rows[2], padded[2, :2])
    assert rows[0].device.type == "cpu"

    none_finished = StaticPoolerChunkOutput(
        None, 2, np.array([False, False]), [None] * 2
    )
    assert unpack_static_pool_output(none_finished) == [None, None]


def test_eager_fallback_gates_unfinished_last_pool():
    """Regression for §2.6: a chunked LAST prefill must not emit early."""
    pooler = _embed_pooler("LAST")
    prompt_lens = np.array([10, 4], dtype=np.int32)
    scheduled = np.array([6, 4], dtype=np.int32)
    computed = np.array([0, 0], dtype=np.int32)
    params = _params(2)
    meta = PoolingMetadata(
        prompt_lens=torch.from_numpy(prompt_lens.copy()),
        prompt_token_ids=None,
        prompt_token_ids_cpu=None,
        pooling_params=params,
        pooling_states=[PoolingStates(), PoolingStates()],
    )
    meta.build_pooling_cursor(
        num_scheduled_tokens_np=scheduled,
        seq_lens_cpu=torch.from_numpy(computed + scheduled),
        device=torch.device("cpu"),
    )
    hidden = torch.randn(int(scheduled.sum()), HIDDEN)
    raw = pooler(hidden, meta)
    # Upstream LastPool emits a row for the unfinished request...
    assert len(raw) == 2 and all(r is not None for r in raw)

    finished = (computed + scheduled) == prompt_lens
    gated = gate_unfinished(raw, finished)
    assert gated[0] is None
    torch.testing.assert_close(gated[1], raw[1])


def test_gate_unfinished_variants():
    t = torch.randn(3, 4)
    assert gate_unfinished(t, np.array([True, True, True])) is t
    gated = gate_unfinished(t, np.array([True, False, True]))
    assert gated[1] is None and torch.equal(gated[2], t[2])
    as_list = gate_unfinished([t[0], None, t[2]], [False, False, True])
    assert as_list[:2] == [None, None]
    assert gate_unfinished(None, np.array([False])) is None
    with pytest.raises(ValueError):
        gate_unfinished(t, np.array([True, False]))


# ---------------------------------------------------------------------------
# Spec resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pooling_type", ["LAST", "CLS", "MEAN"])
def test_resolve_static_pooler_spec_supported(pooling_type):
    pooler = _embed_pooler(pooling_type)
    for wrapped in (pooler, DispatchPooler({"embed": pooler})):
        spec, reason = resolve_static_pooler_spec(
            SimpleNamespace(pooler=wrapped), _model_config()
        )
        assert reason == "ok"
        assert spec == StaticPoolerSpec(pooling_type, torch.float32, None, True, HIDDEN)


def test_resolve_static_pooler_spec_head_variants():
    no_act = SequencePooler(
        pooling=LastPool(), head=EmbeddingPoolerHead(head_dtype="float32")
    )
    spec, _ = resolve_static_pooler_spec(
        SimpleNamespace(pooler=no_act), _model_config()
    )
    assert spec.normalize is False and spec.head_dtype == torch.float32


def test_resolve_static_pooler_spec_unsupported():
    cfg = _model_config()

    def resolve(pooler):
        return resolve_static_pooler_spec(SimpleNamespace(pooler=pooler), cfg)

    assert resolve(None) == (None, "model has no pooler")

    classify = SequencePooler(pooling=LastPool(), head=ClassifierPoolerHead())
    spec, reason = resolve(DispatchPooler({"classify": classify}))
    assert spec is None and "no 'embed'" in reason
    spec, reason = resolve(classify)
    assert spec is None and "ClassifierPoolerHead" in reason

    odd_act = SequencePooler(
        pooling=LastPool(), head=EmbeddingPoolerHead(activation=PoolerClassify())
    )
    spec, reason = resolve(odd_act)
    assert spec is None and "activation" in reason

    class CustomPool(LastPool):
        pass

    spec, reason = resolve(
        SequencePooler(pooling=CustomPool(), head=EmbeddingPoolerHead())
    )
    assert spec is None and "CustomPool" in reason

    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.pooler.tokwise.methods import StepPool

    mock_cfg = MagicMock()
    mock_cfg.scheduler_config.enable_chunked_prefill = False
    with set_current_vllm_config(mock_cfg):
        step_pool = StepPool()
    spec, reason = resolve(step_pool)
    assert spec is None and "StepPool" in reason


# ---------------------------------------------------------------------------
# Dynamo guards: one graph per (T_pad, R_pad), none at replay
# ---------------------------------------------------------------------------


def _call_compiled(fn, spec, hidden, inputs):
    return fn(
        hidden,
        inputs.gather_idx,
        inputs.seg_ids,
        inputs.lens,
        inputs.dims,
        inputs.use_act,
        pooling_type=spec.pooling_type,
        head_dtype=spec.head_dtype,
        projector=spec.projector,
        normalize=spec.normalize,
    )


@pytest.mark.parametrize("pooling_type", ["LAST", "CLS", "MEAN"])
@pytest.mark.parametrize("with_projector", [False, True])
def test_static_pool_dynamo_guards(pooling_type, with_projector):
    torch._dynamo.reset()
    projector = torch.nn.Linear(HIDDEN, 16) if with_projector else None
    spec = StaticPoolerSpec(
        pooling_type, torch.float32, projector, True, 16 if with_projector else HIDDEN
    )
    counter = CompileCounter()
    fn = torch.compile(
        _static_pool_impl, backend=counter, fullgraph=True, dynamic=False
    )
    grid = [(16, 8), (16, 16), (32, 8), (32, 16)]

    with torch.no_grad():
        for t, r in grid:  # precompile
            hidden = torch.zeros(t, HIDDEN)
            _call_compiled(
                fn, spec, hidden, build_dummy_static_pool_inputs(spec, t, r, "cpu")
            )
        assert counter.frame_count == len(grid)

        for t, r in grid:  # serve with real inputs
            n = min(r, 5)
            inputs = _build(spec, [2] * n, [0] * n, [2] * n, t_pad=t, r_pad=r)
            _call_compiled(fn, spec, torch.randn(t, HIDDEN), inputs)
        assert counter.frame_count == len(grid)

    # Grad mode is part of the guard: precompile must run under no_grad.
    _call_compiled(
        fn,
        spec,
        torch.zeros(16, HIDDEN),
        build_dummy_static_pool_inputs(spec, 16, 8, "cpu"),
    )
    assert counter.frame_count == len(grid) + 1
    torch._dynamo.reset()


# ---------------------------------------------------------------------------
# KV cache cap
# ---------------------------------------------------------------------------


def _cap_tokens(**overrides):
    kwargs = dict(
        max_model_len=8192,
        max_num_batched_tokens=8192,
        max_num_seqs=64,
        block_size=128,
        async_scheduling=False,
    )
    kwargs.update(overrides)
    return compute_pooling_kv_cache_cap_tokens(**kwargs)


def test_compute_pooling_kv_cache_cap_tokens():
    # sync: (8192 + 8192) / 128 + 64 tail blocks
    assert _cap_tokens() == (128 + 64) * 128
    # async: two in-flight steps and 2x tail slack
    assert _cap_tokens(async_scheduling=True) == (192 + 128) * 128
    # long_prefill_token_threshold=2048 -> P = min(64, 4) = 4 partial prefills
    assert _cap_tokens(long_prefill_token_threshold=2048) == (320 + 64) * 128
    assert _cap_tokens(long_prefill_token_threshold=None) == _cap_tokens()
    # explicit override, block aligned
    assert _cap_tokens(override_tokens=1000, max_model_len=512) == 8 * 128
    # floored at one max_model_len request
    assert _cap_tokens(override_tokens=100) == 8192
    assert _cap_tokens(max_num_batched_tokens=16, max_num_seqs=1) % 128 == 0
    assert _cap_tokens(max_num_batched_tokens=16, max_num_seqs=1) >= 8192
    with pytest.raises(ValueError):
        _cap_tokens(block_size=0)


def _full_spec(block_size=16, heads=8):
    return FullAttentionSpec(
        block_size=block_size, num_kv_heads=heads, head_size=128, dtype=torch.bfloat16
    )


def test_compute_pooling_kv_cache_cap_bytes():
    specs = {f"layers.{i}.attn": _full_spec() for i in range(4)}
    kw = dict(
        max_model_len=512,
        max_num_batched_tokens=512,
        max_num_seqs=16,
        async_scheduling=False,
    )
    page = specs["layers.0.attn"].page_size_bytes
    tokens = compute_pooling_kv_cache_cap_tokens(block_size=16, **kw)
    assert (
        compute_pooling_kv_cache_cap_bytes(specs, **kw) == (tokens // 16 + 1) * page * 4
    )

    assert compute_pooling_kv_cache_cap_bytes({}, **kw) is None
    assert compute_pooling_kv_cache_cap_bytes(None, **kw) is None
    mixed_pages = dict(specs, extra=_full_spec(heads=4))
    assert compute_pooling_kv_cache_cap_bytes(mixed_pages, **kw) is None
    swa = dict(
        specs,
        swa=SlidingWindowSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.bfloat16,
            sliding_window=128,
        ),
    )
    assert compute_pooling_kv_cache_cap_bytes(swa, **kw) is None


# ---------------------------------------------------------------------------
# Env registration
# ---------------------------------------------------------------------------


def test_pooling_env_defaults(monkeypatch):
    monkeypatch.delenv("VLLM_TPU_STATIC_POOLER", raising=False)
    monkeypatch.delenv("VLLM_TPU_POOLING_KV_CACHE_TOKENS", raising=False)
    assert envs.VLLM_TPU_STATIC_POOLER is True
    assert envs.VLLM_TPU_POOLING_KV_CACHE_TOKENS == -1
    monkeypatch.setenv("VLLM_TPU_STATIC_POOLER", "0")
    monkeypatch.setenv("VLLM_TPU_POOLING_KV_CACHE_TOKENS", "0")
    assert envs.VLLM_TPU_STATIC_POOLER is False
    assert envs.VLLM_TPU_POOLING_KV_CACHE_TOKENS == 0


def test_pooling_env_vars_excluded_from_compile_hash(monkeypatch):
    try:
        from vllm_torchtpu.compilation import tpu_compiler
    except ImportError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"tpu_compiler is not importable in this environment: {exc}")
    for name in ("VLLM_TPU_STATIC_POOLER", "VLLM_TPU_POOLING_KV_CACHE_TOKENS"):
        assert name in tpu_compiler._TPU_COMPILE_ENV_IGNORED
        monkeypatch.setenv(name, "7")
        assert name not in tpu_compiler._tpu_compile_env_factors()


# ---------------------------------------------------------------------------
# Runner wiring (skipped where tpu_runner cannot be imported)
# ---------------------------------------------------------------------------


def _fake_precompile_runner(spec, **overrides):
    runner = SimpleNamespace(
        is_pooling_model=True,
        _pp_is_last=True,
        _static_pooler_spec=spec,
        _static_pooler_reason="ok" if spec else "no spec",
        model_config=_model_config(),
        num_tokens_paddings=[16, 32, 64, 128, 256, 512],
        num_reqs_paddings=[8, 16],
        max_num_reqs=16,
        device="cpu",
        _hidden_states_dtype=torch.float32,
        _precompile_timed=lambda name: nullcontext(),
    )
    for key, value in overrides.items():
        setattr(runner, key, value)
    return runner


def test_runner_precompile_pooler_grid(monkeypatch):
    runner_mod = _import_runner()
    calls = []

    def fake_static_pool(hidden, inputs, spec):
        calls.append((hidden.shape[0], inputs.dims.shape[0], torch.is_grad_enabled()))
        return hidden

    monkeypatch.setattr(runner_mod, "static_pool", fake_static_pool)
    monkeypatch.setattr(runner_mod, "synchronize_tensors", lambda *a, **k: None)
    runner = _fake_precompile_runner(_spec("LAST"))
    runner_mod.TPUModelRunner._precompile_pooler_subgraphs(runner)

    assert [(t, r) for t, r, _ in calls] == [
        (t, r) for t in runner.num_tokens_paddings for r in (8, 16)
    ]
    assert not any(grad for _, _, grad in calls)


@pytest.mark.parametrize(
    "overrides",
    [
        {"is_pooling_model": False},
        {"_pp_is_last": False},
        {"_static_pooler_spec": None},
    ],
)
def test_runner_precompile_pooler_noop_cases(monkeypatch, overrides):
    runner_mod = _import_runner()
    fake = MagicMock()
    monkeypatch.setattr(runner_mod, "static_pool", fake)
    runner = _fake_precompile_runner(_spec("LAST"), **overrides)
    runner_mod.TPUModelRunner._precompile_pooler_subgraphs(runner)
    fake.assert_not_called()


@pytest.mark.parametrize("is_pooling", [True, False])
def test_runner_sampling_precompile_bypassed_for_pooling(monkeypatch, is_pooling):
    runner_mod = _import_runner()
    monkeypatch.setattr(
        runner_mod,
        "_get_sequence_layout_planner_for_runner",
        lambda runner: SimpleNamespace(uses_selected_logits_hidden_states=True),
    )
    names = (
        "_precompile_compute_selected_logits",
        "_precompile_compute_logits_from_hidden_states",
        "_precompile_compute_top_tokens",
        "_precompile_structured_decoding",
        "_precompile_sample_from_logits",
        "_precompile_gather_logprobs",
    )
    runner = SimpleNamespace(is_pooling_model=is_pooling, _pp_is_last=True)
    for name in names:
        setattr(runner, name, MagicMock())
    runner_mod.TPUModelRunner._precompile_sampling_subgraphs(runner)
    for name in names:
        assert getattr(runner, name).called is (not is_pooling), name


def test_runner_capture_and_profile_invoke_pooler_precompile():
    runner_mod = _import_runner()
    cls = runner_mod.TPUModelRunner
    for method in (cls.capture_model, cls.profile_run):
        src = inspect.getsource(method)
        assert "self._precompile_pooler_subgraphs()" in src, method.__name__
        assert src.index("self._precompile_sampling_subgraphs()") < src.index(
            "self._precompile_pooler_subgraphs()"
        )


def _fake_pool_chunk_runner(runner_mod, spec, pooler, prompt_lens, computed):
    req_ids = [f"r{i}" for i in range(len(prompt_lens))]
    requests = {
        rid: SimpleNamespace(
            num_prompt_tokens=int(p),
            pooling_params=PoolingParams(task="embed", use_activation=True),
            pooling_states=PoolingStates(),
            req_id=rid,
            prompt_token_ids=list(range(int(p))),
        )
        for rid, p in zip(req_ids, prompt_lens)
    }
    computed_np = np.asarray(computed, dtype=np.int32)
    batch = SimpleNamespace(
        req_ids=req_ids,
        num_reqs=len(req_ids),
        num_computed_tokens_cpu=computed_np,
        num_computed_tokens_cpu_tensor=torch.from_numpy(computed_np),
    )
    from vllm_torchtpu.runner.tpu_pooling import build_pooling_metadata

    batch.get_pooling_metadata = lambda: build_pooling_metadata(batch, requests)
    return SimpleNamespace(
        input_batch=batch,
        requests=requests,
        _static_pooler_spec=spec,
        model=SimpleNamespace(pooler=pooler),
        device=torch.device("cpu"),
    )


@pytest.mark.parametrize("use_static", [True, False])
def test_runner_pool_chunk_gates_chunked_last(use_static):
    runner_mod = _import_runner()
    pooler = _embed_pooler("LAST")
    spec = _spec_for(pooler) if use_static else None
    prompt_lens, computed, scheduled = [10, 4], [0, 0], [6, 4]
    runner = _fake_pool_chunk_runner(runner_mod, spec, pooler, prompt_lens, computed)
    sched = SimpleNamespace(
        num_scheduled_tokens=dict(zip(runner.input_batch.req_ids, scheduled))
    )
    hidden = _padded_hidden([6, 4], 16)

    out = runner_mod.TPUModelRunner._pool_chunk(
        runner, sched, hidden, 0, 2, 16, 8, None, None
    )
    if use_static:
        assert isinstance(out, StaticPoolerChunkOutput)
        rows = unpack_static_pool_output(out)
    else:
        rows = out
    assert rows[0] is None
    ref = torch.nn.functional.normalize(hidden[9].float(), dim=-1)
    torch.testing.assert_close(rows[1].float().cpu(), ref, atol=1e-6, rtol=0)
