# SPDX-License-Identifier: Apache-2.0
"""A CPU-only TPUModelRunner shell for `_prepare_inputs`' metadata path.

Builds a runner with `object.__new__` and only the state `_prepare_inputs`
reads up to (and including) the attention-metadata walk, over real
`AttentionMetadataBuilder`s (one attention group + mamba groups on the unified
pool) and the real `AllSequenceLayoutPlanner` / `SpeculativeDecodingManager`.
`run_prepare` stops the method right after the walk and returns everything it
produced. Shared by tests/runner/test_steady_decode_metadata.py and the
microbench tests/runner/bench_steady_decode_metadata.py.
"""

from __future__ import annotations

import os
import random
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from vllm_torchtpu.layers.core import attention_metadata as am
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadataBuilder
from vllm_torchtpu.layers.core.sequence_layout import AllSequenceLayoutPlanner
from vllm_torchtpu.runner import steady_decode_metadata as steady_md
from vllm_torchtpu.runner.speculative_decoding_manager import (
    SpeculativeDecodingManager,
)
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

# The served host-path flags the metadata walk runs with (q38f c1/c4 arms).
SERVED_ENV = {
    "VLLM_TPU_BLOCK_TABLE_UPLOAD_REUSE": "1",
    "VLLM_TPU_BLOCK_TABLE_PACK_DELTA": "1",
    "VLLM_TPU_STAGE_ENTRIES_CACHE": "1",
    "VLLM_TPU_STAGED_WALK_CLASS_GATHER": "1",
    "VLLM_TPU_ZERO_PADDED_SEQ_LENS": "1",
}


class _Stop(Exception):
    pass


class FakeBlockTable:
    """The `BlockTable` surface the staged walk reads; `set_row` bumps the
    shared version like the real (patched) mutators do."""

    def __init__(self, max_num_reqs: int, max_num_blocks: int):
        self.max_num_blocks_per_req = max_num_blocks
        self._np = np.zeros((max_num_reqs, max_num_blocks), dtype=np.int32)
        self._t = torch.from_numpy(self._np)
        self._tpu_version = 0

    def get_cpu_tensor(self):
        return self._t

    def get_numpy_array(self):
        return self._np

    def set_row(self, row: int, values) -> None:
        self._np[row] = 0
        self._np[row, : len(values)] = values
        am._BLOCK_TABLE_VERSION[0] += 1
        self._tpu_version = am._BLOCK_TABLE_VERSION[0]


class FakeInputBatch:
    def __init__(self, max_num_reqs: int, max_model_len: int, tables):
        self.req_ids: list[str] = []
        self.num_computed_tokens_cpu = np.zeros(max_num_reqs, dtype=np.int32)
        self.token_ids_cpu = np.zeros((max_num_reqs, max_model_len), dtype=np.int32)
        self.token_ids_cpu_tensor = torch.from_numpy(self.token_ids_cpu)
        self.block_table = tables

    @property
    def num_reqs(self) -> int:
        return len(self.req_ids)


def _spec(kind: str, block_size: int):
    if kind == "attn":
        return FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=2,
            head_size=128,
            dtype=torch.bfloat16,
            page_size_padded=16384,
        )
    return MambaSpec(
        block_size=block_size,
        shapes=[(4, 128), (8, 64, 32)],
        dtypes=[torch.bfloat16, torch.float32],
        page_size_padded=16384,
    )


def make_runner(
    *,
    max_num_reqs: int = 8,
    max_model_len: int = 4096,
    k: int = 3,
    attn_block: int = 64,
    mamba_block: int = 16,
    num_mamba_groups: int = 2,
    max_num_blocks: int | None = None,
) -> TPUModelRunner:
    r = object.__new__(TPUModelRunner)
    if max_num_blocks is None:
        max_num_blocks = -(-max_model_len // min(attn_block, mamba_block)) + k
    r.device = torch.device("cpu")
    r.max_num_reqs = max_num_reqs
    r.max_model_len = max_model_len
    r.max_num_tokens = 512
    r.most_model_len = None
    r.num_reqs_most_model_len = None
    r.num_reqs_max_model_len = max_num_reqs
    r.num_tokens_paddings = [16, 32, 64, 128, 256, 512]
    r._dp_target_bucket = None
    r.reorder_batch_threshold = k + 1
    r.uses_mrope = False
    r._pcp_enabled = False
    r._mtp_enabled = True
    r.lora_config = None
    r.speculative_config = SimpleNamespace(num_speculative_tokens=k)
    r.scheduler_config = SimpleNamespace(async_scheduling=True)
    r.sequence_layout_planner = AllSequenceLayoutPlanner()
    r._attention_schedule_capacity = lambda use_max_model_len: None
    r._input_staging_fence = None
    r._fenced_h2d_sources = []
    r._unfenced_h2d_sources = []
    r._free_h2d_sources = {}
    r._registered_h2d_sources = set()
    r._h2d_reuse = {}
    r.arange_np = np.arange(max(r.max_num_tokens, max_model_len) + 1, dtype=np.int32)
    n = r.max_num_tokens + 1
    r.input_ids_cpu = torch.zeros(n, dtype=torch.int32)
    r.positions_cpu = torch.zeros(n, dtype=torch.int32)
    r.positions_np = r.positions_cpu.numpy()
    r.query_start_loc_cpu = torch.zeros(n, dtype=torch.int32)
    r.query_start_loc_np = r.query_start_loc_cpu.numpy()
    r.seq_lens_cpu = torch.zeros(r.max_num_tokens, dtype=torch.int32)
    r.seq_lens_np = r.seq_lens_cpu.numpy()
    r._request_distribution_cpu = torch.zeros(3, dtype=torch.int32)
    r._combined_request_distribution_cpu = torch.zeros(6, dtype=torch.int32)
    r._fast_token_substitution = False
    r._decode_device_cache_key = None
    r._steady_decode_cache = steady_md.SteadyDecodeMetadataCache()
    r._pre_async_results = None
    r._has_mamba_state = False
    r._unified_kv_layout = True
    r._mamba_tail_fold = False
    r._mamba_ckpt_window = k + 1
    r.mamba_slot_read_offsets = torch.zeros(64, dtype=torch.int32)
    r.empty_slot_mappings = {}
    r._block_table_stage_cpu = None
    r.spec_decode_manager = SpeculativeDecodingManager(r)

    kinds = [("attn", attn_block)] + [("mamba", mamba_block)] * num_mamba_groups
    tables = [FakeBlockTable(max_num_reqs, max_num_blocks) for _ in kinds]
    r.input_batch = FakeInputBatch(max_num_reqs, max_model_len, tables)
    groups = []
    for gid, (kind, bs) in enumerate(kinds):
        builder = AttentionMetadataBuilder(
            kv_cache_spec=_spec(kind, bs),
            layer_names=[f"{kind}.{gid}.{j}" for j in range(3)],
            vllm_config=MagicMock(),
            device=r.device,
            runner=r,
            kv_cache_group_id=gid,
        )
        group = MagicMock()
        group.metadata_builders = [builder]
        group.layer_names = list(builder.layer_names)
        groups.append([group])
    r.attn_groups = groups
    r._fake_kinds = kinds
    return r


@contextmanager
def env(steady: bool, extra: dict | None = None):
    values = dict(SERVED_ENV)
    values["VLLM_TPU_STEADY_DECODE_METADATA"] = "1" if steady else "0"
    values.update(extra or {})
    with patch.dict(os.environ, values):
        yield


def run_prepare(
    runner: TPUModelRunner, scheduler_output, num_decode: int, num_windowed: int
):
    """`_prepare_inputs` up to the metadata walk; returns its products."""
    captured = {}
    build = runner._build_attention_metadata

    def capture_build(*args, **kwargs):
        out = build(*args, **kwargs)
        captured["metadata"] = out[0]
        captured["ctx"] = runner._attn_metadata_builder_ctx
        return out

    def stop(*args, **kwargs):
        raise _Stop

    runner._build_attention_metadata = capture_build
    runner._prepare_async_token_substitution_indices = stop
    try:
        runner._prepare_inputs(scheduler_output, 0, num_decode, num_windowed)
    except _Stop:
        pass
    finally:
        del runner._build_attention_metadata
        del runner._prepare_async_token_substitution_indices
    captured["input_ids"] = runner.input_ids
    captured["position_ids"] = runner.position_ids
    captured["qsl_np"] = runner.query_start_loc_np.copy()
    captured["seq_lens_np"] = runner.seq_lens_np.copy()
    return captured


class Workload:
    """A random decode workload: requests with long contexts, random
    acceptance (0..K drafts), block-boundary crossings that append a block,
    and occasional churn (add / finish / reorder) or a prefill piece.

    `apply(runner)` mirrors the same state into any number of runners;
    `scheduler_output()` describes the step for all of them.
    """

    def __init__(
        self,
        seed: int,
        num_reqs: int,
        k: int = 3,
        context: int = 2000,
        churn: float = 0.05,
        prefill: float = 0.03,
        max_reqs: int = 8,
        max_model_len: int = 4096,
        attn_block: int = 64,
        mamba_block: int = 16,
    ):
        self.rng = random.Random(seed)
        self.k = k
        self.churn = churn
        self.prefill = prefill
        self.max_reqs = max_reqs
        self.max_model_len = max_model_len
        self.attn_block = attn_block
        self.mamba_block = mamba_block
        self.context = context
        self.next_id = 0
        self.reqs: list[dict] = []
        self.new_ids: list[str] = []
        self.finished: set[str] = set()
        self.prefill_id: str | None = None
        for _ in range(num_reqs):
            self._add()
        self.rejected = [0] * max_reqs

    def _add(self):
        rid = f"r{self.next_id}"
        self.next_id += 1
        ctx = self.context + self.rng.randrange(0, 4 * self.attn_block)
        self.reqs.append({"id": rid, "computed": ctx, "blocks": None, "mamba": None})
        self.new_ids.append(rid)

    def step(self):
        """Advance to the next step (acceptance, churn)."""
        self.new_ids = []
        self.finished = set()
        self.prefill_id = None
        for req in self.reqs:
            req["computed"] += 1 + self.rng.randrange(0, self.k + 1)
        if self.rng.random() < self.churn:
            op = self.rng.choice(["add", "finish", "reorder"])
            if op == "add" and len(self.reqs) < self.max_reqs - 1:
                self._add()
            elif op == "finish" and len(self.reqs) > 1:
                gone = self.reqs.pop(self.rng.randrange(len(self.reqs)))
                self.finished.add(gone["id"])
            elif len(self.reqs) > 1:
                i, j = self.rng.sample(range(len(self.reqs)), 2)
                self.reqs[i], self.reqs[j] = self.reqs[j], self.reqs[i]
        if self.rng.random() < self.prefill:
            self.prefill_id = self.rng.choice(self.reqs)["id"]
        self.rejected = [
            self.rng.randrange(0, self.k + 1) for _ in range(self.max_reqs)
        ]

    def num_scheduled(self, req) -> int:
        return 9 if req["id"] == self.prefill_id else self.k + 1

    def apply(self, runner: TPUModelRunner) -> None:
        batch = runner.input_batch
        batch.req_ids = [req["id"] for req in self.reqs]
        for i, req in enumerate(self.reqs):
            batch.num_computed_tokens_cpu[i] = req["computed"]
            start, end = req["computed"], req["computed"] + self.k + 9
            batch.token_ids_cpu[i, start:end] = (
                np.arange(start, end) * 7 + int(req["id"][1:])
            ) % 1000
            # Attention rows: one block per attn_block tokens incl. this step.
            n_attn = -(-(req["computed"] + self.num_scheduled(req)) // self.attn_block)
            base = 1000 * (int(req["id"][1:]) + 1)
            attn_row = list(range(base, base + n_attn))
            # Mamba rows: positional state block + K checkpoint blocks.
            n_m = (
                -(-(req["computed"] + self.num_scheduled(req)) // self.mamba_block)
                + self.k
            )
            for gid, table in enumerate(batch.block_table):
                row = (
                    attn_row
                    if gid == 0
                    else [v + gid * 100000 for v in range(base, base + n_m)]
                )
                cur = table.get_numpy_array()[i, : len(row)]
                if len(row) > table.max_num_blocks_per_req:
                    raise AssertionError("raise max_num_blocks")
                if (
                    not np.array_equal(cur, row)
                    or table.get_numpy_array()[i, len(row) :].any()
                ):
                    table.set_row(i, row)
        for i in range(len(self.reqs), runner.max_num_reqs):
            for table in batch.block_table:
                if table.get_numpy_array()[i].any():
                    table.set_row(i, [])
        index_of = {req["id"]: j for j, req in enumerate(self.reqs)}
        runner._pre_async_results = SimpleNamespace(
            spec_decode_num_rejected_tokens=torch.tensor(
                self.rejected, dtype=torch.int32
            ),
            req_id_to_index_copy={
                rid: j for rid, j in index_of.items() if rid not in self.new_ids
            },
        )

    def scheduler_output(self):
        num_scheduled = {req["id"]: self.num_scheduled(req) for req in self.reqs}
        spec = {
            req["id"]: [0] * self.k for req in self.reqs if req["id"] != self.prefill_id
        }
        return SimpleNamespace(
            total_num_scheduled_tokens=sum(num_scheduled.values()),
            num_scheduled_tokens=num_scheduled,
            scheduled_spec_decode_tokens=spec,
            scheduled_new_reqs=list(self.new_ids),
            finished_req_ids=set(self.finished),
            scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
        )

    def segments(self) -> tuple[int, int]:
        """(num_decode, num_windowed): the batch is [verify][prefill]."""
        if self.prefill_id is None:
            return 0, len(self.reqs)
        # Keep the prefill piece last, as _reorder_batch_for_rpa would.
        i = next(j for j, r in enumerate(self.reqs) if r["id"] == self.prefill_id)
        self.reqs.append(self.reqs.pop(i))
        return 0, len(self.reqs) - 1
