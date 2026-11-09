# SPDX-License-Identifier: Apache-2.0
"""Opt-in forward post-processing model/runner regressions.

Run explicitly with pytest; this module is not selected by default discovery.
"""

from contextlib import nullcontext
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest import mock
from unittest.mock import MagicMock, Mock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.quantized_matmul import buffered as ob
from vllm_torchtpu.layers.adapter.sample.rejection_sampler import RejectionSampler
from vllm_torchtpu.models.vllm.deepseek_v41.common import mhc
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention as v2_attn
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import cache as cache_ops
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import dspark as v2_dspark
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import dspark_sampling
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import model as v2_model_module
from vllm_torchtpu.models.vllm.deepseek_v41.v2.ced import (
    DecoderReplayPlan,
    plan_decoder_replay,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.model import DeepseekV41Model
from vllm_torchtpu.runner import tpu_runner, utils

MULT = 4
WINDOW = 128


def _np(a) -> np.ndarray:
    if isinstance(a, torch.Tensor):
        return a.float().numpy()
    return np.asarray(jnp.asarray(a).astype(jnp.float32))


def _plan_metadata(bounds, lengths, prompts, hits=None):
    return SimpleNamespace(
        query_start_loc_cpu=tuple(bounds),
        seq_lens_cpu=tuple(lengths),
        prompt_lens_cpu=tuple(prompts),
        replay_starts_cpu=None if hits is None else tuple(hits),
    )


class _Attention(torch.nn.Module):
    def __init__(self, table: torch.Tensor):
        super().__init__()
        self.register_buffer("_tpu_rope_cache", table, persistent=False)


def _relayout(model: torch.nn.Module) -> tuple[list[torch.Tensor], bool]:
    """Runs `relayout_rope_tables` with a fake row-major copy.

    Returns the tensors handed to `_to_row_major` and whether the entry
    layout pin was installed.
    """
    seen: list[torch.Tensor] = []

    def fake_row_major(tensor: torch.Tensor) -> torch.Tensor:
        seen.append(tensor)
        return tensor.clone()

    with (
        mock.patch.object(utils, "_to_row_major", side_effect=fake_row_major),
        mock.patch.object(utils, "_install_entry_layout_pin") as pin,
    ):
        utils.relayout_rope_tables(model)
    return seen, pin.called


def _view(owner, offset, *, stride=12, page_rows=4):
    return cache_ops.PagedCacheRows(
        owner, ((4, page_rows, 128), torch.float32, stride * 512, offset * 512)
    )


@dataclass
class _FakeForwardCtx:
    attn_metadata: Any
    dp_metadata: Any = None
    is_padding: Any = None
    additional_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class _FakeAttnMeta:
    input_positions: Any = None
    query_start_loc: Any = None
    seq_lens: Any = None
    block_tables: Any = None


@pytest.mark.parametrize("tokens", [240, 288])
def test_flat_collapse_matches_view_collapse(tokens):
    """The flat op must compute the same weighted stream sum as ``collapse``.

    288 is the target decode bucket (48 requests x 6 positions); 240 is
    DSpark's.
    """
    if jax.default_backend() != "tpu":
        pytest.skip("The reference reduction tree is specific to TPU")
    rng = np.random.default_rng(3)
    dim = 5120
    flat = jnp.asarray(rng.standard_normal((tokens, MULT * dim)) * 4, jnp.bfloat16)
    pre = jnp.asarray(rng.uniform(0.01, 1.5, (tokens, MULT)), jnp.float32)

    def view_collapse(f, p):
        # Same expression as ``collapse``: f32 view, weight, sum, back to BF16.
        view = f.reshape(tokens, MULT, dim).astype(jnp.float32)
        return (view * p[..., None]).sum(1).astype(f.dtype)

    want = jax.jit(view_collapse)(flat, pre)
    got = jax.jit(mhc._flat_collapse_jax)(flat, pre)
    assert got.dtype == jnp.bfloat16
    np.testing.assert_array_equal(_np(got), _np(want))


@pytest.mark.cpu_test
def test_flat_collapse_dispatch(monkeypatch):
    """Only TPU BF16 4-stream inputs reach the op; the rest keep ``collapse``."""
    calls = []

    def fake_op(flat, pre):
        calls.append((tuple(flat.shape), tuple(pre.shape)))
        return flat[:, : flat.shape[1] // pre.shape[1]]

    x = torch.randn(3, MULT, 8, dtype=torch.bfloat16)
    pre = torch.rand(3, MULT, dtype=torch.float32)

    # Unregistered op: fall back even on TPU-shaped inputs.
    monkeypatch.setattr(mhc, "_flat_collapse_op", None)
    torch.testing.assert_close(mhc.flat_collapse(x, pre), mhc.collapse(x, pre))

    # Registered, but CPU tensors: still the fallback.
    monkeypatch.setattr(mhc, "_flat_collapse_op", fake_op)
    torch.testing.assert_close(mhc.flat_collapse(x, pre), mhc.collapse(x, pre))
    assert not calls

    # TPU device: the op receives the flattened [T, 4*D] input.
    tpu_x = MagicMock()
    tpu_x.device.type = "tpu"
    tpu_x.dim.return_value = 3
    tpu_x.shape = (3, MULT, 8)
    tpu_x.dtype = torch.bfloat16
    tpu_x.flatten.return_value = x.flatten(1)
    mhc.flat_collapse(tpu_x, pre)
    tpu_x.flatten.assert_called_once_with(1)
    assert calls == [((3, MULT * 8), (3, MULT))]

    # Wrong stream count or dtype on TPU: fallback, op not called.
    for shape, dtype in (((3, 2, 8), torch.bfloat16), ((3, MULT, 8), torch.float32)):
        calls.clear()
        other = MagicMock()
        other.device.type = "tpu"
        other.dim.return_value = 3
        other.shape = shape
        other.dtype = dtype
        monkeypatch.setattr(mhc, "collapse", lambda h, p: "fallback")
        assert mhc.flat_collapse(other, pre) == "fallback"
        assert not calls


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "tokens,k,expected",
    [(288, 15360, True), (240, 15360, False), (256, 15360, False), (288, 10240, False)],
)
@pytest.mark.parametrize("generation", [6, 7])
def test_buffered_matmul_admits_dspark_main_proj(
    monkeypatch, tokens, k, expected, generation
):
    """DSpark main_proj (3 x 5120 aux hidden -> 5120) takes the buffered kernel
    only at its tuned v7 decode shape (288 target rows)."""
    monkeypatch.setattr(ob, "get_tpu_version", lambda: generation)
    x = jax.ShapeDtypeStruct((tokens, k), jnp.bfloat16)
    w = jax.ShapeDtypeStruct((k, 5120), jnp.float8_e4m3fn)
    s = jax.ShapeDtypeStruct((5120,), jnp.float32)
    assert ob.is_supported(x, w, s) is (expected and generation == 7)


def test_buffered_matmul_main_proj_is_bitwise_equal():
    """The buffered kernel keeps xla_quantized_matmul's row quantization and
    scale order, so the DSpark main_proj output must not change."""
    if jax.default_backend() != "tpu" or ob.get_tpu_version() != 7:
        pytest.skip("Buffered FP8 matmul is tuned for TPU v7")
    from vllm_torchtpu.kernels.quantized_matmul.util import xla_quantized_matmul

    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(5), 3)
    x = (jax.random.normal(k1, (288, 15360), jnp.float32) * 2).astype(jnp.bfloat16)
    w = (jax.random.normal(k2, (15360, 5120), jnp.float32) * 30).astype(
        jnp.float8_e4m3fn
    )
    s = jax.random.uniform(k3, (5120,), jnp.float32, 1e-4, 2e-3)
    assert ob.is_supported(x, w, s)
    got = jax.jit(ob.buffered_matmul)(x, w, s)
    want = jax.jit(xla_quantized_matmul)(x, w, s)
    np.testing.assert_array_equal(
        np.asarray(got.astype(jnp.float32)), np.asarray(want.astype(jnp.float32))
    )


@pytest.mark.parametrize("merged", [False, True])
@pytest.mark.parametrize(
    "changed",
    [None, "block_tables", "query_start_loc", "seq_lens", "page_size", "window"],
)
def test_context_kv_addresses_derived_once_per_cache_group(
    monkeypatch: pytest.MonkeyPatch, changed: str | None, merged: bool
):
    """Reuse only identical address inputs; both write paths preserve every layer."""
    monkeypatch.setattr(envs, "TPU_DSPARK_MERGED_CONTEXT_WRITES", merged)
    monkeypatch.setattr(v2_attn, "_active_stage_attn_cache", None)
    monkeypatch.setattr(v2_dspark, "_fused_context_kv", lambda attns, x, pos: None)
    monkeypatch.setattr(
        v2_dspark, "_context_kv", lambda attn, x, pos: x + attn.layer_id
    )

    layers = [
        SimpleNamespace(
            attn=SimpleNamespace(
                layer_id=i,
                window_size=4,
                swa_cache_layer=SimpleNamespace(kv_cache=torch.zeros((8, 4, 1, 8))),
            )
        )
        for i in range(3)
    ]
    base = SimpleNamespace(
        block_tables=torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        seq_lens=torch.tensor([6, 6], dtype=torch.int32),
    )
    # Separate metadata objects may share the actual tensors that define addresses.
    mds = [SimpleNamespace(**vars(base)) for _ in layers]
    if changed == "block_tables":
        mds[1].block_tables = base.block_tables + 1
    elif changed == "query_start_loc":
        mds[1].query_start_loc = torch.tensor([0, 1, 4], dtype=torch.int32)
    elif changed == "seq_lens":
        mds[1].seq_lens = torch.tensor([4, 6], dtype=torch.int32)
    elif changed == "page_size":
        layers[1].attn.swa_cache_layer.kv_cache = torch.zeros((8, 8, 1, 8))
    elif changed == "window":
        layers[1].attn.window_size = 2
    positions = torch.tensor([1, 2, 3, 5], dtype=torch.long)
    expected = [
        v2_dspark._context_kv_addresses(layer.attn, md, positions)
        for layer, md in zip(layers, mds, strict=True)
    ]
    if changed is not None:
        assert any(
            not torch.equal(a, b) for a, b in zip(expected[0], expected[1], strict=True)
        )
    # Count real address computations, not a stub that ignores parts of the key.
    slots = MagicMock(wraps=v2_dspark._cache_slots)
    insert = MagicMock()
    publish = MagicMock()
    monkeypatch.setattr(v2_dspark, "_cache_slots", slots)
    monkeypatch.setattr(v2_dspark, "_insert", insert)
    monkeypatch.setattr(v2_dspark, "write_cache_layers", publish)
    model = SimpleNamespace(config=SimpleNamespace(hidden_size=8), layers=layers)
    hidden = torch.ones((4, 8))
    result = v2_dspark.DSparkDeepseekV41Model.precompute_and_store_context_kv(
        model, hidden, positions, tuple(mds)
    )

    assert result is hidden
    assert slots.call_count == (1 if changed is None else 2)
    if merged:
        publish.assert_called_once()
        insert.assert_not_called()
        written = list(zip(*publish.call_args.args, strict=True))
    else:
        publish.assert_not_called()
        assert insert.call_count == len(layers)
        written = [call.args for call in insert.call_args_list]
    for i, (cache, actual_slots, kv, valid) in enumerate(written):
        assert cache is layers[i].attn.swa_cache_layer.kv_cache
        assert torch.equal(actual_slots, expected[i][0])
        assert torch.equal(valid, expected[i][1])
        assert torch.equal(kv, hidden + i)


@pytest.mark.parametrize(
    "indices,padded,rows,expected",
    [
        (tuple(range(8)), 8, 8, True),  # every row, in order, no padding
        (tuple(range(8)), 16, 8, False),  # DP raised the bucket: padding rows
        (tuple(range(1, 8)), 8, 8, False),  # a skipped prompt row
        ((0, 1, 3, 2, 4, 5, 6, 7), 8, 8, False),  # reordered
        (tuple(range(6)), 6, 8, False),  # fewer rows than the input bucket
        ((), 0, 0, True),  # empty bucket is trivially in order
    ],
)
def test_identity_replay_detection(indices, padded, rows, expected):
    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import ced

    plan = ced.DecoderReplayPlan(indices, (0, len(indices)), (0,), padded)
    assert ced.is_identity_replay(plan, rows) is expected


def test_identity_replay_none_plan():
    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import ced

    assert ced.is_identity_replay(None, 8) is False


@pytest.mark.parametrize(
    "requests,count,prompt,length,rows,compile_sizes,expected",
    [
        # Steady DSpark decode: 48 requests x 6 positions, all past the window.
        (48, 6, 1024, 2048, 288, [288], True),
        # Partial batch padded up to the 288 bucket: padding rows stay.
        (47, 6, 1024, 2048, 288, [288], False),
        # Prefill: only the last WINDOW prompt rows replay.
        (1, 256, 256, 256, 256, [256], False),
    ],
)
def test_identity_replay_from_planner(
    requests, count, prompt, length, rows, compile_sizes, expected
):
    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import ced

    bounds = [i * count for i in range(requests + 1)]
    metadata = _plan_metadata(bounds, [length] * requests, [prompt] * requests)
    plan = plan_decoder_replay(metadata, WINDOW, rows, compile_sizes)
    assert ced.is_identity_replay(plan, rows) is expected


def test_64_wide_tables_get_shared_paired_copy(monkeypatch):
    monkeypatch.setenv("TPU_ROPE_PAIRED_ROWS", "1")
    shared, narrow = torch.randn(9, 64), torch.randn(8, 4)
    model = torch.nn.ModuleList(
        [_Attention(shared), _Attention(shared), _Attention(narrow)]
    )
    _relayout(model)
    first, second, third = model
    assert first._tpu_rope_pairs is second._tpu_rope_pairs
    pairs = first._tpu_rope_pairs
    assert pairs.shape == (5, 128)
    assert torch.equal(pairs[:4].reshape(8, 64), shared[:8])
    assert torch.equal(pairs[4, :64], shared[8]) and torch.equal(
        pairs[4, 64:], shared[8]
    )
    assert not hasattr(third, "_tpu_rope_pairs")


def test_paired_copy_disabled(monkeypatch):
    monkeypatch.setenv("TPU_ROPE_PAIRED_ROWS", "0")
    model = torch.nn.ModuleList([_Attention(torch.randn(8, 64))])
    _relayout(model)
    assert not hasattr(model[0], "_tpu_rope_pairs")


def test_sampler_option_specializes_op_and_stays_bound(monkeypatch):
    built = []

    def build(name, fn, **kwargs):
        op = Mock()
        built.append((name, fn))
        return op

    monkeypatch.setattr(dspark_sampling.pallas, "jax_op", build)
    sample = Mock(return_value="tokens")
    monkeypatch.setattr(dspark_sampling, "samples", sample)
    dspark_sampling._initialize_samples.cache_clear()
    try:
        selected = []
        for enabled in (False, True, False):
            monkeypatch.setattr(envs, "TPU_DSPARK_PIPELINED_MARKOV", enabled)
            selected.append(dspark_sampling.initialize_samples())
        assert selected[0] is selected[2]
        assert selected[0] is not selected[1]
        assert len(built) == 2 and built[0][0] != built[1][0]
        for enabled, (_, fn) in zip((False, True), built, strict=True):
            assert fn(None, None, None, None) == "tokens"
            assert sample.call_args.kwargs["pipelined"] is enabled
    finally:
        dspark_sampling._initialize_samples.cache_clear()


@pytest.mark.parametrize("pipelined", [False, True])
def test_each_slot_receives_the_registered_option(monkeypatch, pipelined):
    slot = Mock(return_value=jnp.zeros((2,), dtype=jnp.int32))
    monkeypatch.setattr(dspark_sampling, "sample_slot", slot)
    result = dspark_sampling.samples(
        jnp.zeros((10, 4)),
        jnp.zeros((4, 8)),
        jnp.zeros((4, 8)),
        jnp.zeros((2,), dtype=jnp.int32),
        pipelined=pipelined,
    )
    assert result.shape == (2, 5)
    assert [call.kwargs["slot"] for call in slot.call_args_list] == list(range(5))
    assert all(call.kwargs["pipelined"] is pipelined for call in slot.call_args_list)


@pytest.mark.parametrize(
    "offsets,strides,page_rows,merged",
    [
        ((0, 4, 8), (12, 12, 12), (4, 4, 4), True),
        ((8, 0, 4), (12, 12, 12), (4, 4, 4), True),
        ((0, 0), (12, 12), (4, 4), False),
        ((0, 1), (12, 12), (4, 4), False),
        ((0, 4), (12, 16), (4, 4), False),
        ((0, 4), (12, 12), (4, 2), False),
        ((0, 10), (12, 12), (4, 4), False),
    ],
)
def test_merged_writes_match_ordered_layer_publication(
    monkeypatch, offsets, strides, page_rows, merged
):
    # Meta owners exercise dispatch without allocating a TPU; replace only
    # the final hardware write with a CPU row writer.
    owner = torch.empty((64, 128), dtype=torch.int32, device="meta")
    caches = [
        _view(owner, off, stride=stride, page_rows=rows)
        for off, stride, rows in zip(offsets, strides, page_rows, strict=True)
    ]
    actual = torch.zeros((64, 128), dtype=torch.float32)
    expected = torch.zeros_like(actual)
    writes = []

    def publish(_cache, records, values):
        writes.append(records.clone())
        valid = records >= 0
        actual.index_copy_(0, records[valid].long(), values[valid])

    monkeypatch.setattr(cache_ops.PagedCacheRows, "_write_word_records", publish)
    slots = [torch.tensor([0, 1, 4, 5, -1, 7]) for _ in caches]
    valid = [torch.tensor([True, True, True, False, True, True]) for _ in caches]
    values = [torch.full((6, 128), float(i + 1)) for i in range(len(caches))]
    for off, stride, rows, ids, vals, ok in zip(
        offsets, strides, page_rows, slots, values, valid, strict=True
    ):
        for slot, value, keep in zip(ids.tolist(), vals, ok.tolist(), strict=True):
            if keep and slot >= 0:
                expected[off + slot // rows * stride + slot % rows] = value

    cache_ops.write_cache_layers(caches, slots, values, valid)

    assert len(writes) == (1 if merged else len(caches))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_distinct_owners_keep_separate_writes(monkeypatch):
    owners = [
        torch.empty((64, 128), dtype=torch.int32, device="meta") for _ in range(2)
    ]
    caches = [_view(owners[0], 0), _view(owners[1], 4)]
    writes = []
    monkeypatch.setattr(
        cache_ops.PagedCacheRows,
        "_write_word_records",
        lambda cache, records, values: writes.append(cache.raw),
    )
    cache_ops.write_cache_layers(
        caches,
        [torch.tensor([0])] * 2,
        [torch.ones((1, 128))] * 2,
        [torch.tensor([True])] * 2,
    )
    assert len(writes) == 2
    assert writes[0] is owners[0]
    assert writes[1] is owners[1]


@pytest.mark.parametrize("synthetic_mode", [False, True])
def test_head_ids_warmup_keeps_draft_and_head_inputs_independent(
    monkeypatch, synthetic_mode
):
    rows, requests, drafts = 288, 48, 5
    output = torch.zeros(requests, dtype=torch.int32)
    sampler = Mock(return_value=output, synthetic_mode=synthetic_mode)
    sampler.greedy_from_head_ids.return_value = output
    sampler.greedy_from_token_ids.return_value = output
    runner = SimpleNamespace(
        _is_async_drafter=True,
        speculative_config=SimpleNamespace(num_speculative_tokens=drafts),
        rejection_sampler=sampler,
        num_tokens_paddings=[rows],
        num_reqs_paddings=[requests],
        max_num_reqs=requests,
        _fused_greedy_head=SimpleNamespace(rows=rows),
        _hidden_states_dtype=torch.bfloat16,
        device=torch.device("cpu"),
        vocab_size=16,
        model_config=SimpleNamespace(get_hidden_size=lambda: 8),
        _precompile_timed=lambda _name: nullcontext(),
        _spec_verify_bucket_reachable=lambda _rows, _reqs: True,
        compute_selected_greedy_ids=Mock(return_value=output),
        compute_greedy_ids=Mock(return_value=output),
        spec_bonus_and_target_token_ids=Mock(return_value=(output, output)),
        spec_gather_bonus_and_target_logits=Mock(return_value=(output, output)),
        _greedy_sample=Mock(return_value=output),
    )
    monkeypatch.setattr(tpu_runner, "synchronize_tensors", lambda _value: None)

    tpu_runner.TPUModelRunner._precompile_rejection_sampler(runner)

    sampler.greedy_from_head_ids.assert_called_once()
    inputs = sampler.greedy_from_head_ids.call_args.kwargs
    head_ids, draft_ids = inputs["head_token_ids"], inputs["draft_token_ids"]
    # Dynamo guards both object aliasing and storage relationships.
    assert head_ids is not draft_ids
    assert head_ids.data_ptr() != draft_ids.data_ptr()
    assert head_ids.shape == draft_ids.shape == (rows,)
    assert head_ids.dtype == draft_ids.dtype == torch.int32
    assert head_ids.device == draft_ids.device


class TestCEDForward:
    def _create_model(
        self,
        *,
        ced_enabled: bool = True,
        sp_enabled: bool = False,
        cpu_offload: bool = False,
        use_native_mega_moe: bool = False,
    ) -> DeepseekV41Model:
        model = DeepseekV41Model.__new__(DeepseekV41Model)
        model._ced_enabled = ced_enabled
        model._ced_boundary = 1
        model._ced_window = 512
        model._ced_compile_sizes = (4, 8)
        model.start_layer = 0
        model.end_layer = 2
        model.use_sequence_parallel = sp_enabled
        model.use_native_mega_moe = use_native_mega_moe
        model.config = SimpleNamespace(hidden_size=64)
        model._engram_prepared_outside_stage = False
        model._has_cpu_offloaded_engram = MagicMock(return_value=cpu_offload)
        model._prepare_engram = MagicMock(return_value=(None, None))
        return model

    def _run_ced_forward(self, plan, dec_out, rows=4):
        model = self._create_model(ced_enabled=True)
        hidden = torch.ones((rows, 64))
        pre_mix = torch.zeros((rows, 64))
        model.encoder_stage = MagicMock(return_value=(hidden, pre_mix))
        model.decoder_stage = MagicMock(return_value=dec_out)
        meta_entry = _FakeAttnMeta(seq_lens=torch.tensor([rows // 2, rows // 2]))
        mock_ctx = _FakeForwardCtx(attn_metadata={"layer_0": meta_entry})
        with (
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_forward_context",
                return_value=mock_ctx,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.plan_decoder_replay",
                return_value=plan,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model._scatter_decoder_rows",
                wraps=v2_model_module._scatter_decoder_rows,
            ) as scatter,
        ):
            res = model.forward(None, torch.arange(rows))
        return res, scatter, model

    def test_forward_ced_identity_plan_skips_scatter(self) -> None:
        # Steady decode: every row kept, in order, bucket equals the rows.
        plan = DecoderReplayPlan(
            indices=(0, 1, 2, 3),
            padded_num_tokens=4,
            query_start_loc=(0, 2, 4),
            replay_starts=(0, 0),
        )
        dec_hidden = torch.arange(4 * 64, dtype=torch.float32).view(4, 64)
        dec_aux = [dec_hidden + 1, dec_hidden + 2, dec_hidden + 3]
        res, scatter, model = self._run_ced_forward(plan, (dec_hidden, dec_aux))
        scatter.assert_not_called()
        hidden, aux = res
        # Same values the scatter path would produce for an identity mapping.
        expected = v2_model_module._scatter_decoder_rows(dec_hidden, torch.arange(4), 4)
        torch.testing.assert_close(hidden, expected, rtol=0, atol=0)
        assert hidden is dec_hidden
        assert len(aux) == 3 and all(a is b for a, b in zip(aux, dec_aux))
        # The plan still drives the decoder's query bounds.
        model.decoder_stage.assert_called_once()

    @pytest.mark.parametrize(
        "indices,padded",
        [
            ((0, 1, 2, 3), 8),  # DP raised the bucket: padding rows to drop
            ((1, 2, 3), 4),  # a skipped prompt row
        ],
    )
    def test_forward_ced_non_identity_plan_keeps_scatter(self, indices, padded) -> None:
        plan = DecoderReplayPlan(
            indices=indices,
            padded_num_tokens=padded,
            query_start_loc=(0, len(indices)),
            replay_starts=(0, 0),
        )
        dec_out = torch.arange(padded * 64, dtype=torch.float32).view(padded, 64)
        res, scatter, _ = self._run_ced_forward(plan, dec_out)
        scatter.assert_called_once()
        assert res.shape == (4, 64)
        for out_row, src_row in enumerate(indices):
            torch.testing.assert_close(res[src_row], dec_out[out_row])


@pytest.fixture
def _sampler_recompile_limit():
    """The sampler is torch.compiled with dynamic=False, so every distinct
    input shape recompiles. Production raises this limit in the runner
    (tpu_runner.py); the unit test never builds a runner, so the default (8)
    is hit once the parametrized cases span >8 shapes. Mirror the runner here
    and restore afterwards so the bump doesn't leak into other tests."""
    cfg = torch._dynamo.config
    names = [n for n in ("recompile_limit", "cache_size_limit") if hasattr(cfg, n)]
    saved = {n: getattr(cfg, n) for n in names}
    for n in names:
        setattr(cfg, n, 1024)
    yield
    for n, v in saved.items():
        setattr(cfg, n, v)


@pytest.mark.usefixtures("_sampler_recompile_limit")
@pytest.mark.parametrize("synthetic", [False, True])
@pytest.mark.parametrize("edge", ["ragged", "ties", "nonfinite", "zero_drafts"])
def test_head_ids_match_logits_path(device, synthetic, edge):
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    sampler = RejectionSampler()
    if synthetic:
        sampler.synthetic_mode = True
        sampler.synthetic_conditional_rates = torch.tensor(
            [0.8, 0.5, 0.2], device=device
        )
    torch.manual_seed(17)
    logits = torch.randn(12, 257, dtype=torch.bfloat16)
    if edge == "ties":
        logits[:, 7] = 20
        logits[:, 130] = 20
    elif edge == "nonfinite":
        logits[0, 3] = float("nan")
        logits[0, 132] = float("nan")
        logits[1, 12] = float("inf")
        logits[2] = -float("inf")
    logits = logits.to(device)
    target_indices = torch.tensor([0, 1, 2, 6, 7, 0], dtype=torch.int32, device=device)
    bonus_indices = torch.tensor([3, 5, 8], dtype=torch.int32, device=device)
    lengths = [0, 0, 0] if edge == "zero_drafts" else [3, 0, 2]
    num_drafts = torch.tensor(lengths, dtype=torch.int32, device=device)
    segments = torch.tensor([0, 0, 0, 2, 2, 3], dtype=torch.int64, device=device)
    groups = torch.tensor([0, 1, 2, 0, 1, 0], dtype=torch.int32, device=device)
    gathered = torch.index_select(logits, 0, target_indices)
    drafts = torch.argmax(gathered, dim=-1).to(torch.int32)
    drafts[1] = (drafts[1] + 1) % logits.shape[1]
    accept_u = torch.tensor([0.1, 0.7, 0.3, 0.9, 0.2, 0.0], device=device)
    common = dict(
        draft_token_ids=drafts,
        num_draft_tokens=num_drafts,
        segment_ids=segments,
        group_indices=groups,
        max_draft_tokens=3,
        accept_u=accept_u if synthetic else None,
    )
    bonus_reference = torch.argmax(torch.index_select(logits, 0, bonus_indices), dim=-1)
    expected = sampler(
        target_logits=gathered, bonus_token_ids=bonus_reference, **common
    )
    bonus_ids, target_ids = TPUModelRunner.spec_bonus_and_target_token_ids(
        None, logits, bonus_indices, target_indices
    )
    actual = sampler.greedy_from_token_ids(
        target_token_ids=target_ids, bonus_token_ids=bonus_ids, **common
    )
    assert torch.equal(bonus_ids.cpu(), bonus_reference.cpu())
    assert torch.equal(target_ids.cpu(), torch.argmax(gathered, dim=-1).cpu())
    assert torch.equal(actual.cpu(), expected.cpu())

    # Fused greedy head path: per-row head IDs, rows selected in the same
    # program as the validation.
    head_ids = torch.argmax(logits, dim=-1).to(torch.int32)
    fused = sampler.greedy_from_head_ids(
        head_token_ids=head_ids,
        bonus_indices=bonus_indices,
        target_indices=target_indices,
        **common,
    )
    assert torch.equal(fused.cpu(), expected.cpu())
