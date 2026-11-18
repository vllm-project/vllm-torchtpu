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
"""Device decode inputs: equal to what the stock `_prepare_inputs` stages for
the same pure-decode batch, and turned into the same attention metadata by the
stock builders."""

import dataclasses
import types
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from runner_input_test_utils import _make_runner
from vllm import SamplingParams
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
from vllm.v1.worker.block_table import SlotMappingMode
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

from vllm_torchtpu.gdn_pool_layout import PrefillCheckpointUnits
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadataBuilder
from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.device_decode import inputs as inputs_mod
from vllm_torchtpu.runner.device_decode.inputs import (
    DecodeShape,
    WalkGeometry,
    build_decode_attention_metadata,
    precompile_prepare,
    prepare_decode_inputs,
)
from vllm_torchtpu.runner.device_decode.slots import (
    DecodeSlots,
    SlotConfig,
    StockChunkPlan,
    num_reqs_padding,
)
from vllm_torchtpu.runner.kernel_block_table import install_kernel_block_tables
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

MAX_NUM_REQS = 256
TOKEN_PADDINGS = (16, 32, 64, 128, 256)
BLOCK = 16
# Qwen3.5-0.8B's attention row: 19 blocks at max_model_len 10240.
ATTN_WIDTH = 19
MAX_MODEL_LEN = BLOCK * ATTN_WIDTH
VOCAB = 1000


@dataclass(frozen=True)
class Geometry:
    """KV cache groups after the first (full attention) group: mamba groups
    keyed by whether they declare prefill checkpoints, or none."""

    mamba: tuple[bool, ...] = ()
    # "unified" pool (indices from the block table) or "compact" slot pool.
    pool: str = "unified"
    split_attention: bool = False


GEOMETRIES = {
    "attention": Geometry(),
    "split": Geometry(split_attention=True),
    # Qwen3.5-0.8B: one attention group and three one-block GDN groups.
    "unified": Geometry(mamba=(False, False, False)),
    "compact": Geometry(mamba=(False, False, False), pool="compact"),
    "prefill_ckpt": Geometry(mamba=(True, False, True)),
}


def _mamba_width(geometry):
    # Prefill checkpoints need align mode, a block per BLOCK tokens.
    return ATTN_WIDTH if any(geometry.mamba) else 1


def _specs(geometry):
    specs = [
        FullAttentionSpec(
            block_size=BLOCK, num_kv_heads=1, head_size=128, dtype=torch.bfloat16
        )
    ]
    for declares in geometry.mamba:
        specs.append(
            MambaSpec(
                block_size=BLOCK,
                shapes=((4, 128), (8, 64, 32)),
                dtypes=(torch.bfloat16, torch.float32),
                page_size_padded=16384,
                mamba_cache_mode="align" if any(geometry.mamba) else "none",
                num_prefill_checkpoint_blocks=1 if declares else 0,
                prefill_checkpoint_alignment=64 if declares else None,
            )
        )
    return specs


def make_runner(geometry, *, target_rows=MAX_NUM_REQS, async_mode=False):
    """The `_prepare_inputs` stand-in of the runner tests over a real
    InputBatch and the real attention metadata builders."""
    runner = _make_runner(
        num_computed_tokens=[0],
        prompt_tokens=[1],
        scheduled_tokens=[1],
        token_paddings=TOKEN_PADDINGS,
        pcp_size=1,
        max_num_reqs=MAX_NUM_REQS,
        block_size=BLOCK,
    )
    runner.num_reqs_max_model_len = target_rows
    runner.num_reqs_paddings = tpu_runner._get_req_paddings(
        tpu_runner.MIN_NUM_SEQS, MAX_NUM_REQS
    )
    runner.vocab_size = VOCAB
    runner.model_config = SimpleNamespace(seed=0)
    specs = _specs(geometry)
    widths = [ATTN_WIDTH] + [_mamba_width(geometry)] * len(geometry.mamba)
    runner.input_batch = InputBatch(
        max_num_reqs=MAX_NUM_REQS,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=max(TOKEN_PADDINGS),
        device=torch.device("cpu"),
        vocab_size=VOCAB,
        block_sizes=[BLOCK] * len(specs),
        max_num_blocks_per_req=widths,
        slot_mapping_modes=[SlotMappingMode.NONE] * len(specs),
    )
    if geometry.split_attention:
        install_kernel_block_tables(
            runner.input_batch.block_table, [BLOCK // 2] + [BLOCK] * (len(specs) - 1)
        )
    runner.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec) for spec in specs]
    )
    runner.attn_groups = [
        [
            SimpleNamespace(
                metadata_builders=[
                    AttentionMetadataBuilder(
                        kv_cache_spec=spec,
                        layer_names=[f"layer.{gid}"],
                        vllm_config=MagicMock(),
                        device=runner.device,
                        runner=runner,
                        kv_cache_group_id=gid,
                    )
                ],
                layer_names=[f"layer.{gid}"],
            )
        ]
        for gid, spec in enumerate(specs)
    ]
    runner._build_attention_metadata = types.MethodType(
        TPUModelRunner._build_attention_metadata, runner
    )
    runner._unified_kv_layout = bool(geometry.mamba) and geometry.pool == "unified"
    runner.requests = {}
    runner._mamba_slot_by_req_id = {}
    runner._free_mamba_slots = []
    if geometry.mamba and geometry.pool == "compact":
        runner._has_mamba_state = True
        runner._build_mamba_state_indices = types.MethodType(
            TPUModelRunner._build_mamba_state_indices, runner
        )
    if any(geometry.mamba):
        runner._prefill_ckpt_units = PrefillCheckpointUnits(
            hash_block_size=BLOCK,
            cache_hit_alignment_tokens=BLOCK,
            mamba_block_size=BLOCK,
            alignment=64,
            drop_eagle_block=False,
        )
        runner._prefill_ckpt_offsets_cpu = torch.zeros(MAX_NUM_REQS, dtype=torch.int32)
    runner._fast_token_substitution = False
    if async_mode:
        for name in (
            "_prepare_async_token_substitution_indices",
            "_apply_async_token_substitution",
        ):
            setattr(
                runner, name, types.MethodType(getattr(TPUModelRunner, name), runner)
            )
    return runner


def add_decoding_requests(runner, geometry, num_reqs, *, async_mode=False):
    """Adds `num_reqs` requests at a decode step, in a batch order unlike
    their sorted ids. Returns {req_id: input token}: the token sampled last,
    which the stock path reads from the token buffer (sync) or substitutes
    from the previous step's device output (async)."""
    rng = np.random.default_rng(num_reqs)
    mamba_blocks = _mamba_width(geometry) > 1
    next_block = 1
    last_tokens = {}
    for i in range(num_reqs):
        req_id = f"r{(i * 37) % num_reqs:03d}"
        # Sequences end anywhere in a block, up to the full row.
        seq_len = 2 + (i * 53 + 14) % (MAX_MODEL_LEN - 1)
        num_output = min(1 + i % 3, seq_len - 1)
        num_blocks = -(-seq_len // BLOCK)
        block_ids = []
        for count in [num_blocks] + [
            num_blocks if mamba_blocks else 1 for _ in geometry.mamba
        ]:
            block_ids.append(list(range(next_block, next_block + count)))
            next_block += count
        last_tokens[req_id] = int(rng.integers(1, VOCAB))
        outputs = [int(t) for t in rng.integers(1, VOCAB, num_output - 1)]
        outputs.append(0 if async_mode else last_tokens[req_id])
        state = CachedRequestState(
            req_id=req_id,
            prompt_token_ids=[
                int(t) for t in rng.integers(1, VOCAB, seq_len - num_output)
            ],
            mm_features=[],
            sampling_params=SamplingParams(temperature=0.8, top_k=40, top_p=0.95),
            generator=None,
            block_ids=tuple(block_ids),
            num_computed_tokens=seq_len - 1,
            output_token_ids=outputs,
        )
        runner.requests[req_id] = state
        runner.input_batch.add_request(state)
        if runner._has_mamba_state:
            runner._mamba_slot_by_req_id[req_id] = 2 * i + 1
    return last_tokens


def decode_output(req_ids):
    return SimpleNamespace(
        total_num_scheduled_tokens=len(req_ids),
        num_scheduled_tokens={req_id: 1 for req_id in req_ids},
        scheduled_spec_decode_tokens={},
    )


def admitted(req_ids):
    output = SchedulerOutput.make_empty()
    output.scheduled_new_reqs = [
        NewRequestData(
            req_id=req_id,
            prompt_token_ids=None,
            mm_features=[],
            sampling_params=None,
            pooling_params=None,
            block_ids=(),
            num_computed_tokens=0,
            lora_request=None,
        )
        for req_id in req_ids
    ]
    return output


def ready_slots(runner, last_tokens):
    """Slots after the stock step that sampled `last_tokens`, written back in
    that step's (reversed) batch order."""
    slots = DecodeSlots(SlotConfig.from_runner(runner), runner.device)
    req_ids = list(runner.input_batch.req_ids)
    slots.sync_from_runner(runner, admitted(req_ids))
    slots.apply_staged_writes(runner)
    previous = req_ids[::-1]
    num_rows = num_reqs_padding(runner, len(previous))
    pad = num_rows - len(previous)
    seq_lens = [runner.requests[r].num_computed_tokens for r in previous]
    plan = StockChunkPlan(
        num_rows_padded=num_rows,
        slots=np.array(
            [slots.slot_of(r) for r in previous] + [slots.config.null_slot] * pad,
            dtype=np.int32,
        ),
        positions=np.array([n - 1 for n in seq_lens] + [0] * pad, dtype=np.int32),
        produced=np.array([True] * len(previous) + [False] * pad),
        new_num_computed=np.array(seq_lens + [0] * pad, dtype=np.int32),
    )
    sampled = torch.tensor([last_tokens[r] for r in previous] + [0] * pad)
    slots.write_back_stock_chunk(runner, plan, sampled.reshape(num_rows, 1))
    return slots, previous, sampled


def snapshot(value):
    """A deep copy of the tensors in `value`, so later steps cannot change
    them (the stock path stages into recycled host buffers)."""
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {k: snapshot(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(snapshot(v) for v in value)
    if dataclasses.is_dataclass(value):
        return SimpleNamespace(
            **{
                f.name: snapshot(getattr(value, f.name))
                for f in dataclasses.fields(value)
                if f.init
            }
        )
    return value


def run_stock(runner, num_reqs, previous=None, sampled=None):
    """The stock pure-decode `_prepare_inputs`, with the async placeholder
    substitution when `previous` (the last step's batch order) is given."""
    req_ids = runner.input_batch.req_ids[:num_reqs]
    if previous is not None:
        runner._pre_async_results = SimpleNamespace(
            req_id_to_index_copy={r: i for i, r in enumerate(previous)},
            next_tokens_tpu=sampled,
            spec_decode_num_rejected_tokens=None,
        )
    prepared = TPUModelRunner._prepare_inputs(
        runner, decode_output(req_ids), 0, num_reqs
    )
    input_ids = runner.input_ids
    if previous is not None:
        with torch.compiler.set_stance("force_eager"):
            input_ids = runner._apply_async_token_substitution(
                input_ids, prepared[5], prepared[6]
            )
    ctx = runner._attn_metadata_builder_ctx
    return snapshot(
        {
            "metadata": prepared[0],
            "logits_indices": prepared[1],
            "input_ids": input_ids,
            "positions": runner.position_ids,
            "seq_lens": ctx.seq_lens,
            "query_start_loc": ctx.query_start_loc,
            "request_distribution": ctx.request_distribution,
            "staged_block_tables": ctx.staged_block_tables,
            "staged_mamba_products": ctx.staged_mamba_products,
            "mamba_state_indices": ctx.mamba_state_indices,
            "prefill_ckpt_offsets": ctx.mamba_prefill_ckpt_offsets,
        }
    )


def run_device(runner, slots, num_reqs, prefill_ckpt_offsets):
    idx_mapping = torch.from_numpy(
        slots.idx_mapping_host(runner.input_batch.req_ids[:num_reqs])
    )
    shape = DecodeShape.for_batch(runner, num_reqs)
    geometry = WalkGeometry.from_runner(runner)
    inputs = prepare_decode_inputs(
        slots, idx_mapping, shape, geometry, prefill_ckpt_offsets
    )
    metadata = build_decode_attention_metadata(
        runner, inputs, shape, prefill_ckpt_offsets
    )
    return inputs, metadata, shape, geometry, idx_mapping


def assert_same_layout(got, want):
    assert got.dtype == want.dtype and got.shape == want.shape
    assert got.is_contiguous() and want.is_contiguous()


def assert_live_equal(got, want, num_live):
    """Equal on live rows; the device path zeroes the rows the stock path
    leaves stale."""
    assert_same_layout(got, want)
    assert torch.equal(got[:num_live], want[:num_live])
    assert not got[num_live:].any()


def assert_equal(got, want):
    if want is None or got is None:
        assert got is None and want is None
        return
    assert_same_layout(got, want)
    assert torch.equal(got, want)


def check_equivalence(runner, stock, inputs, metadata, num_reqs):
    assert_equal(inputs.input_ids, stock["input_ids"])
    assert_live_equal(inputs.positions, stock["positions"], num_reqs)
    assert_live_equal(inputs.seq_lens, stock["seq_lens"], num_reqs)
    assert_equal(inputs.query_start_loc, stock["query_start_loc"])
    assert_equal(inputs.logits_indices, stock["logits_indices"])
    assert_equal(inputs.request_distribution, stock["request_distribution"])
    assert inputs.staged_block_tables.keys() == stock["staged_block_tables"].keys()
    for gid, table in stock["staged_block_tables"].items():
        assert_equal(inputs.staged_block_tables[gid], table)
    if stock["staged_mamba_products"] is None:
        assert inputs.staged_mamba_products is None
    else:
        assert inputs.staged_mamba_products.keys() == (
            stock["staged_mamba_products"].keys()
        )
        for gid, products in stock["staged_mamba_products"].items():
            for got, want in zip(inputs.staged_mamba_products[gid], products):
                assert_equal(got, want)
    assert_equal(inputs.mamba_state_indices, stock["mamba_state_indices"])
    assert metadata.keys() == stock["metadata"].keys()
    for layer, want in stock["metadata"].items():
        got = metadata[layer]
        assert_live_equal(got.input_positions, want.input_positions, num_reqs)
        assert_live_equal(got.seq_lens, want.seq_lens, num_reqs)
        for field in (
            "block_tables",
            "query_start_loc",
            "request_distribution",
            "mamba_state_indices",
            "mamba_slot_read_offsets",
            "mamba_ckpt_indices",
            "mamba_request_distribution",
            "mamba_prefill_ckpt_offsets",
            "mamba_prefill_ckpt_indices",
        ):
            assert_equal(getattr(got, field), getattr(want, field))
        for field in (
            "pcp_metadata",
            "sequence_layout_kind",
            "sequence_layout_protocol",
            "sequence_layout_version",
        ):
            assert getattr(got, field) == getattr(want, field), field


@pytest.mark.parametrize("geometry_name", list(GEOMETRIES))
@pytest.mark.parametrize("num_reqs", [1, 7, 8, 9, 100, 256])
def test_inputs_equal_the_stock_upload(geometry_name, num_reqs):
    geometry = GEOMETRIES[geometry_name]
    runner = make_runner(geometry)
    last_tokens = add_decoding_requests(runner, geometry, num_reqs)
    slots, _, _ = ready_slots(runner, last_tokens)
    stock = run_stock(runner, num_reqs)
    inputs, metadata, _, _, _ = run_device(
        runner, slots, num_reqs, stock["prefill_ckpt_offsets"]
    )
    check_equivalence(runner, stock, inputs, metadata, num_reqs)


@pytest.mark.parametrize("geometry_name", ["attention", "unified", "compact"])
def test_input_tokens_equal_the_async_substitution(geometry_name):
    """Under async scheduling the stock path substitutes the previous step's
    device tokens for the placeholders; the slots hold the same tokens."""
    geometry = GEOMETRIES[geometry_name]
    runner = make_runner(geometry, async_mode=True)
    last_tokens = add_decoding_requests(runner, geometry, 100, async_mode=True)
    slots, previous, sampled = ready_slots(runner, last_tokens)
    stock = run_stock(runner, 100, previous, sampled)
    expected = [last_tokens[r] for r in runner.input_batch.req_ids[:100]]
    assert stock["input_ids"][:100].tolist() == expected
    inputs, metadata, _, _, _ = run_device(
        runner, slots, 100, stock["prefill_ckpt_offsets"]
    )
    check_equivalence(runner, stock, inputs, metadata, 100)


def test_fewer_metadata_rows_than_logits_rows():
    """num_reqs_max_model_len below max_num_reqs: 100 rows of metadata, a
    128-row logits bucket and a 128-token bucket."""
    geometry = GEOMETRIES["unified"]
    runner = make_runner(geometry, target_rows=100)
    last_tokens = add_decoding_requests(runner, geometry, 100)
    slots, _, _ = ready_slots(runner, last_tokens)
    stock = run_stock(runner, 100)
    inputs, metadata, shape, _, _ = run_device(
        runner, slots, 100, stock["prefill_ckpt_offsets"]
    )
    assert (shape.num_reqs_padded, shape.num_tokens_padded, shape.target_rows) == (
        128,
        128,
        100,
    )
    check_equivalence(runner, stock, inputs, metadata, 100)


@pytest.mark.parametrize("geometry_name", list(GEOMETRIES))
def test_outputs_alias_nothing(geometry_name):
    geometry = GEOMETRIES[geometry_name]
    runner = make_runner(geometry)
    last_tokens = add_decoding_requests(runner, geometry, 9)
    slots, _, _ = ready_slots(runner, last_tokens)
    offsets = (
        torch.zeros(MAX_NUM_REQS, dtype=torch.int32) if any(geometry.mamba) else None
    )
    inputs, _, _, _, idx_mapping = run_device(runner, slots, 9, offsets)
    outputs = [
        inputs.input_ids,
        inputs.positions,
        inputs.seq_lens,
        inputs.query_start_loc,
        inputs.logits_indices,
        inputs.request_distribution,
        *inputs.staged_block_tables.values(),
    ]
    if inputs.mamba_state_indices is not None:
        outputs.append(inputs.mamba_state_indices)
    for products in (inputs.staged_mamba_products or {}).values():
        outputs.extend(t for t in products if t is not None)
    inputs_of_the_program = [
        idx_mapping,
        slots.num_computed,
        slots.last_token,
        slots.mamba_slot,
        *slots.block_tables,
    ]
    if offsets is not None:
        inputs_of_the_program.append(offsets)
    storages = [t.untyped_storage().data_ptr() for t in outputs]
    assert len(set(storages)) == len(storages)
    assert not set(storages) & {
        t.untyped_storage().data_ptr() for t in inputs_of_the_program
    }


@pytest.mark.parametrize("geometry_name", list(GEOMETRIES))
def test_walk_geometry_is_the_serving_walk(geometry_name):
    geometry = GEOMETRIES[geometry_name]
    runner = make_runner(geometry)
    walk = WalkGeometry.from_runner(runner)
    num_groups = 1 + len(geometry.mamba)
    attention_width = 2 * ATTN_WIDTH if geometry.split_attention else ATTN_WIDTH
    assert walk.gids == tuple(range(num_groups))
    assert walk.widths == (attention_width,) + (_mamba_width(geometry),) * len(
        geometry.mamba
    )
    offsets = np.cumsum((0,) + walk.widths[:-1]) * MAX_NUM_REQS
    assert walk.table_geom == tuple(
        (int(off), MAX_NUM_REQS, width) for off, width in zip(offsets, walk.widths)
    )
    unified = bool(geometry.mamba) and geometry.pool == "unified"
    assert walk.mamba_gids == (tuple(range(1, num_groups)) if unified else ())
    assert walk.compact_mamba == (bool(geometry.mamba) and not unified)
    assert walk.single_table == (num_groups == 1)
    assert walk.ckpt_window == 1


def test_decode_shapes_cover_every_single_chunk_batch():
    runner = make_runner(GEOMETRIES["attention"], target_rows=100)
    shapes = DecodeShape.all_decode_shapes(runner)
    assert [(s.num_tokens_padded, s.num_reqs_padded) for s in shapes] == [
        (16, 8),
        (16, 16),
        (32, 32),
        (64, 64),
        (128, 128),
    ]
    keys = {s.program_key for s in shapes}
    for num_reqs in range(1, 101):
        shape = DecodeShape.for_batch(runner, num_reqs)
        assert shape.program_key in keys
        assert shape.num_reqs_padded == (
            tpu_runner._get_padded_num_reqs_with_upper_limit(num_reqs, MAX_NUM_REQS)
        )


@pytest.mark.parametrize("geometry_name", list(GEOMETRIES))
def test_precompile_prepare_runs_every_shape(geometry_name, monkeypatch):
    geometry = GEOMETRIES[geometry_name]
    runner = make_runner(geometry, target_rows=100)
    slots = DecodeSlots(SlotConfig.from_runner(runner), runner.device)
    before = [t.clone() for t in (slots.num_computed, slots.last_token)]
    calls = []
    original = inputs_mod._prepare_decode_impl

    def spy(*args):
        calls.append(args[6:9])
        return original(*args)

    monkeypatch.setattr(inputs_mod, "_prepare_decode_impl", spy)
    shapes = DecodeShape.all_decode_shapes(runner)
    keys = precompile_prepare(runner, slots, WalkGeometry.from_runner(runner), shapes)
    assert keys == [s.program_key for s in shapes]
    assert calls == [
        (s.num_tokens_padded, s.num_reqs_padded, s.target_rows) for s in shapes
    ]
    assert all(
        torch.equal(a, b)
        for a, b in zip(before, (slots.num_computed, slots.last_token))
    )


@pytest.mark.parametrize("geometry_name", list(GEOMETRIES))
def test_prepare_program_traces_as_one_graph(geometry_name):
    """No graph break in the prepare program (backend-independent), and the
    traced program computes what the eager one does."""
    geometry = GEOMETRIES[geometry_name]
    runner = make_runner(geometry)
    last_tokens = add_decoding_requests(runner, geometry, 9)
    slots, _, _ = ready_slots(runner, last_tokens)
    idx_mapping = torch.from_numpy(
        slots.idx_mapping_host(runner.input_batch.req_ids[:9])
    )
    shape = DecodeShape.for_batch(runner, 9)
    walk = WalkGeometry.from_runner(runner)
    offsets = (
        torch.zeros(MAX_NUM_REQS, dtype=torch.int32) if any(geometry.mamba) else None
    )
    args = (
        idx_mapping,
        slots.num_computed,
        slots.last_token,
        slots.mamba_slot if walk.compact_mamba else None,
        offsets,
        tuple(slots.block_tables[gid] for gid in walk.gids),
        shape.num_tokens_padded,
        shape.num_reqs_padded,
        shape.target_rows,
        slots.config.null_slot,
        walk.widths,
        walk.table_geom,
        walk.mamba_geom,
        walk.ckpt_window,
        walk.single_table,
    )
    traced = torch.compile(
        inputs_mod._prepare_decode_impl, backend="eager", fullgraph=True, dynamic=False
    )
    eager_outs = inputs_mod._prepare_decode_impl(*args)
    traced_outs = traced(*args)
    assert len(eager_outs) == len(traced_outs)
    for got, want in zip(traced_outs, eager_outs):
        assert torch.equal(got, want)
