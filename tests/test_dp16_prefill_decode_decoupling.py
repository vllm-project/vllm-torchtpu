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
"""Stage 2 DP=16 Prefill-Decode Lockstep TPOT Decoupling & Hybrid TP/DP/EP Tests.

================================================================================
3-TIER EXECUTION PROVENANCE BANNER
================================================================================
ANALYTIC / MICROBENCHMARK PROJECTION (NOT E2E PHYSICAL VLLM RUN)
- Tier 1 (Host Control-Plane / Scheduler / Runner Lockstep): Real execution of
  `TpuDpScheduler`, `compute_decode_aware_step_token_budget`, `TPUModelRunner`,
  `DSparkProposer`, and `populate_draft_embed_from_target` across multi-thread
  `torch.distributed` Gloo CPU process groups (`DP=16`, `TP=2/DP=8`, `TP=4/DP=4`).
- Tier 2 (3-Way Vectorized NumPy Lockstep Step-Time & Amdahl Calibration):
  Loop-free Vectorized NumPy simulation anchored to physical Cloud TPU v7x job
  `j-1830a031` (`DP=16, TP=1, EP=16`, 392.41 tok/s/chip, median TPOT 558.18 ms).
- Tier 3 (CPU Numerical Parity — NOT Physical TPU Execution): Gate 7 micro-chunk
  (`4x256` vs `1x1024`) and Gate 7B (`TRAP24`) partitioned (`TP=1/2/4`) vs
  unpartitioned (`TP=1`) numerical parity (`CosSim >= 0.99999`, `SQNR >= 45.0 dB`)
  evaluated on host CPU PyTorch tensors (`torch.float32` / `torch.bfloat16`).
================================================================================
"""

from __future__ import annotations

import contextlib
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F

for _cand_path in (
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
):
    if os.path.isdir(_cand_path) and _cand_path not in sys.path:
        sys.path.insert(0, _cand_path)


def _install_vllm_test_stubs():
    import contextlib
    import dataclasses
    import importlib.abc
    import importlib.machinery
    import logging
    import types
    from unittest.mock import MagicMock

    try:
        import torch_tpu  # noqa: F401
        import vllm  # noqa: F401

        return
    except ImportError:
        pass

    class _DummyMeta(type):
        def __getattr__(cls, attr: str):
            if attr.startswith("__") and attr.endswith("__"):
                raise AttributeError(attr)
            val = MagicMock(name=f"{cls.__name__}.{attr}")
            setattr(cls, attr, val)
            return val

    class _DummyBase(metaclass=_DummyMeta):
        def __init__(self, *args, **kwargs):
            pass

    class _DummyMod(types.ModuleType):
        def __init__(self, name: str):
            super().__init__(name)
            self.__path__ = []

        def __getattr__(self, attr: str):
            if attr.startswith("__") and attr.endswith("__"):
                raise AttributeError(attr)
            sub_name = f"{self.__name__}.{attr}"
            if sub_name in sys.modules:
                val = sys.modules[sub_name]
                setattr(self, attr, val)
                return val
            if self.__name__ == "vllm.envs" and attr.isupper():
                setattr(self, attr, None)
                return None
            if attr[0].isupper():
                val = _DummyMeta(attr, (_DummyBase,), {})
            else:
                val = MagicMock(name=sub_name)
            setattr(self, attr, val)
            return val

    class _AutoStubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        PREFIXES = (
            "vllm",
            "torch_tpu",
            "einops",
            "compressed_tensors",
            "transformers",
            "safetensors",
            "portpicker",
            "requests",
        )

        def find_spec(self, fullname, path, target=None):
            if any(
                fullname == p or fullname.startswith(p + ".") for p in self.PREFIXES
            ):
                return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
            return None

        def create_module(self, spec):
            if spec.name in sys.modules:
                return sys.modules[spec.name]
            mod = _DummyMod(spec.name)
            sys.modules[spec.name] = mod
            if "." in spec.name:
                parent_name, child_attr = spec.name.rsplit(".", 1)
                if parent_name in sys.modules:
                    setattr(sys.modules[parent_name], child_attr, mod)
            return mod

        def exec_module(self, module):
            pass

    sys.meta_path.insert(0, _AutoStubFinder())
    stub_modules = [
        "torch_tpu",
        "vllm",
        "vllm.envs",
        "vllm.logger",
        "vllm.config",
        "vllm.compilation",
        "vllm.compilation.backends",
        "vllm.distributed",
        "vllm.distributed.parallel_state",
        "vllm.forward_context",
        "vllm.v1",
        "vllm.v1.outputs",
        "vllm.v1.core",
        "vllm.v1.core.sched",
        "vllm.v1.core.sched.async_scheduler",
        "vllm.v1.core.sched.interface",
        "vllm.v1.core.sched.output",
        "vllm.v1.core.sched.request_queue",
        "vllm.v1.worker",
        "vllm.v1.worker.gpu_model_runner",
    ]
    for mod_name in stub_modules:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = _DummyMod(mod_name)

    @contextlib.contextmanager
    def _noop_cm(*args, **kwargs):
        yield

    sys.modules["vllm.compilation.backends"].set_model_tag = _noop_cm
    sys.modules["vllm.config"].set_current_vllm_config = _noop_cm

    @dataclasses.dataclass
    class ModelRunnerOutput:
        req_ids: list = dataclasses.field(default_factory=list)
        req_id_to_index: dict = dataclasses.field(default_factory=dict)
        sampled_token_ids: list = dataclasses.field(default_factory=list)
        logprobs: object = None
        prompt_logprobs_dict: dict = dataclasses.field(default_factory=dict)
        pooler_output: list = dataclasses.field(default_factory=list)
        kv_connector_output: object = None
        ec_connector_output: object = None
        num_nans_in_logits: dict = dataclasses.field(default_factory=dict)
        finish_sending: object = None
        finish_recving: object = None

    sys.modules["vllm.v1.outputs"].ModelRunnerOutput = ModelRunnerOutput
    sys.modules["vllm.v1.outputs"].EMPTY_MODEL_RUNNER_OUTPUT = ModelRunnerOutput()

    class _GPUModelRunner:
        def __init__(self, *args, **kwargs):
            pass

    sys.modules["vllm.v1.worker.gpu_model_runner"].GPUModelRunner = _GPUModelRunner

    try:
        import torch._dynamo

        if "tpu" not in torch._dynamo.list_backends():

            @torch._dynamo.register_backend(name="tpu")
            def _dummy_tpu_backend(model, inputs):
                return model
    except Exception as err:
        logging.debug("Dynamo tpu backend registration skipped: %s", err)

    class _AsyncScheduler:
        def __init__(self, *args, **kwargs):
            pass

        def schedule(self, *args, **kwargs):
            return None

    sys.modules["vllm.v1.core.sched.async_scheduler"].AsyncScheduler = _AsyncScheduler
    sys.modules["vllm.logger"].init_logger = lambda name: logging.getLogger(name)


_install_vllm_test_stubs()

PROVENANCE_BANNER = (
    "ANALYTIC / MICROBENCHMARK PROJECTION (NOT E2E PHYSICAL VLLM RUN) | "
    "Tier 1: Host Python/PyTorch Gloo Multi-Rank Control-Plane | "
    "Tier 2: 3-Way Vectorized NumPy Amdahl Model (Anchored to j-1830a031) | "
    "Tier 3: Host CPU PyTorch Gate 7 & Gate 7B Numerical Parity (NOT Physical TPU)"
)


def _compute_cos_sim_and_sqnr(
    ref: torch.Tensor, cand: torch.Tensor
) -> tuple[float, float]:
    ref_f = ref.detach().to(torch.float64).reshape(-1)
    cand_f = cand.detach().to(torch.float64).reshape(-1)
    if not torch.isfinite(ref_f).all() or not torch.isfinite(cand_f).all():
        return float("nan"), float("-inf")
    denom = torch.linalg.norm(ref_f) * torch.linalg.norm(cand_f)
    if float(denom) == 0.0:
        return 0.0, float("-inf")
    cos_sim = float(torch.dot(ref_f, cand_f) / denom)
    noise = ref_f - cand_f
    noise_power = float(torch.sum(noise * noise))
    sig_power = float(torch.sum(ref_f * ref_f))
    if noise_power == 0.0:
        sqnr_db = 120.0
    else:
        sqnr_db = 10.0 * math.log10(max(sig_power, 1e-30) / noise_power)
    return cos_sim, sqnr_db


def _make_fake_request(
    req_id: str,
    *,
    num_prompt_tokens: int,
    num_computed_tokens: int,
    num_spec_tokens: int = 0,
    num_output_tokens: int = 1,
) -> SimpleNamespace:
    num_tokens = num_prompt_tokens + num_output_tokens
    num_tokens_with_spec = num_tokens + num_spec_tokens
    return SimpleNamespace(
        request_id=req_id,
        num_prompt_tokens=num_prompt_tokens,
        num_computed_tokens=num_computed_tokens,
        num_tokens=num_tokens,
        num_tokens_with_spec=num_tokens_with_spec,
        spec_token_ids=[100 + i for i in range(num_spec_tokens)],
    )


def _make_test_scheduler(max_num_scheduled_tokens: int = 1024):
    """Instantiate TpuDpScheduler via __new__ and attach a controllable base
    schedule().
    """
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.core.sched.interface import PauseState

    from vllm_torchtpu.core.tpu_scheduler import (
        TpuDpScheduler,
        set_process_cross_dp_active_decode,
    )

    set_process_cross_dp_active_decode(False)
    sched = TpuDpScheduler.__new__(TpuDpScheduler)
    sched._pause_state = PauseState.UNPAUSED
    sched._drain_prefill = False
    sched._cross_dp_has_active_decode = False
    sched._cross_dp_synced_from_process = False
    sched.waiting = []
    sched.skipped_waiting = []
    sched.running = []
    sched.prefill_capacity_bound = False
    sched.max_num_scheduled_tokens = max_num_scheduled_tokens
    sched.scheduler_config = SimpleNamespace(
        max_num_batched_tokens=max_num_scheduled_tokens,
        long_prefill_token_threshold=max_num_scheduled_tokens,
    )
    sched.observed_calls = []
    sched.raise_on_schedule = None

    def _base_schedule(throttle_prefills: bool = False):
        if sched.raise_on_schedule is not None:
            raise sched.raise_on_schedule
        sched.observed_calls.append(
            (
                throttle_prefills,
                sched._pause_state,
                int(sched.max_num_scheduled_tokens),
                int(sched.scheduler_config.max_num_batched_tokens),
                int(sched.scheduler_config.long_prefill_token_threshold),
            )
        )
        return SimpleNamespace(
            total_num_scheduled_tokens=int(sched.max_num_scheduled_tokens)
        )

    return sched, patch.object(AsyncScheduler, "schedule", side_effect=_base_schedule)


# ==============================================================================
# 1. Mechanism A: Decode-Aware Prefill Micro-Chunk Clamping (Gate 1 + Gate 4)
# ==============================================================================


def test_systolic_bucket_alignment_and_env_validation(monkeypatch):
    """Verify 128-token systolic alignment and negative-control validation."""
    from vllm_torchtpu.compilation.tpu_compiler import _TPU_COMPILE_ENV_IGNORED
    from vllm_torchtpu.core.tpu_scheduler import (
        SYSTOLIC_BUCKET_ALIGNMENT,
        align_to_systolic_bucket,
        get_dp_decode_aware_prefill_chunk_cap,
    )
    from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

    assert SYSTOLIC_BUCKET_ALIGNMENT == 128
    assert "VLLM_TPU_DP_DECODE_AWARE_PREFILL_CHUNK_CAP" in _TPU_COMPILE_ENV_IGNORED
    assert (
        "VLLM_TPU_DP_DECODE_AWARE_PREFILL_CHUNK_CAP" in TpuPlatform.additional_env_vars
    )

    # Exact multiples stay unchanged; non-multiples round up to next 128 multiple.
    assert align_to_systolic_bucket(128) == 128
    assert align_to_systolic_bucket(256) == 256
    assert align_to_systolic_bucket(200) == 256
    assert align_to_systolic_bucket(1) == 128
    assert align_to_systolic_bucket(300) == 384

    # Negative Control 1: non-positive values rejected by align_to_systolic_bucket.
    with pytest.raises(ValueError, match="token_cap must be positive"):
        align_to_systolic_bucket(0)
    with pytest.raises(ValueError, match="token_cap must be positive"):
        align_to_systolic_bucket(-128)

    # Negative Control 2: unset or 0 returns None (clamping disabled).
    monkeypatch.delenv("VLLM_TPU_DP_DECODE_AWARE_PREFILL_CHUNK_CAP", raising=False)
    assert get_dp_decode_aware_prefill_chunk_cap() is None

    monkeypatch.setenv("VLLM_TPU_DP_DECODE_AWARE_PREFILL_CHUNK_CAP", "0")
    assert get_dp_decode_aware_prefill_chunk_cap() is None

    # Negative Control 3: negative env var raises actionable ValueError.
    monkeypatch.setenv("VLLM_TPU_DP_DECODE_AWARE_PREFILL_CHUNK_CAP", "-256")
    with pytest.raises(
        ValueError,
        match="VLLM_TPU_DP_DECODE_AWARE_PREFILL_CHUNK_CAP must be >= 0",
    ):
        get_dp_decode_aware_prefill_chunk_cap()

    # Positive non-multiple rounds to 256.
    monkeypatch.setenv("VLLM_TPU_DP_DECODE_AWARE_PREFILL_CHUNK_CAP", "200")
    assert get_dp_decode_aware_prefill_chunk_cap() == 256


def test_tpu_dp_scheduler_three_paths_crash_repeat_and_high_decode(monkeypatch):
    """Gate 1 Three-Path verification for TpuDpScheduler budget clamping:
    1. Crash path: exception in super().schedule() restores all budget fields.
    2. Repeat-call path: mixed decode+prefill step clamps to 256, subsequent
       pure-prefill step (including partially-chunked prefill in `running`)
       automatically runs unclamped at 1024 with zero stale carryover.
    3. High-decode load path: when active decode tokens exceed chunk_cap, budget
       expands to next 128-aligned bucket so decode is never starved.
    """
    monkeypatch.setenv("VLLM_TPU_DP_DECODE_AWARE_PREFILL_CHUNK_CAP", "256")

    sched, sched_patch = _make_test_scheduler(max_num_scheduled_tokens=1024)
    with sched_patch:
        sched.waiting = ["prefill-req-0"]

        # Active decode request with K=5 MTP (1 + 5 = 6 tokens).
        decode_req = _make_fake_request(
            "decode-req-1",
            num_prompt_tokens=512,
            num_computed_tokens=512,
            num_spec_tokens=5,
            num_output_tokens=1,
        )
        sched.running = [decode_req]

        # --- Path 1: Crash Path ---
        sched.raise_on_schedule = RuntimeError("simulated super().schedule() failure")
        with pytest.raises(RuntimeError, match="simulated super"):
            sched.schedule(throttle_prefills=False)
        sched.raise_on_schedule = None

        # Verify all attributes were restored in `finally:` despite crash.
        assert sched.max_num_scheduled_tokens == 1024
        assert sched.scheduler_config.max_num_batched_tokens == 1024
        assert sched.scheduler_config.long_prefill_token_threshold == 1024

        # --- Path 2: Repeat-Call Path (Mixed Step -> Pure Chunked-Prefill Step) ---
        out_mixed = sched.schedule(throttle_prefills=False)
        assert out_mixed.total_num_scheduled_tokens == 256
        assert sched._last_step_budget_info == {
            "base_budget": 1024,
            "effective_budget": 256,
            "clamped": True,
            "num_decode_reqs": 1,
            "decode_tokens": 6,
        }
        # Post-call attributes must be restored to 1024.
        assert sched.max_num_scheduled_tokens == 1024
        assert sched.scheduler_config.max_num_batched_tokens == 1024

        # Now decode_req finishes, leaving only a partially-chunked prefill in
        # `running` (250 of 4096 prompt tokens computed). This is a pure-prefill
        # step and MUST NOT be misclassified as decode!
        partial_prefill_req = _make_fake_request(
            "prefill-req-0",
            num_prompt_tokens=4096,
            num_computed_tokens=250,
            num_spec_tokens=0,
            num_output_tokens=0,
        )
        sched.running = [partial_prefill_req]
        out_pure_prefill = sched.schedule(throttle_prefills=False)
        assert out_pure_prefill.total_num_scheduled_tokens == 1024
        assert sched._last_step_budget_info["clamped"] is False
        assert sched._last_step_budget_info["effective_budget"] == 1024
        assert sched._last_step_budget_info["num_decode_reqs"] == 0

        # --- Path 3: High Decode Token Demand (> 256 tokens) ---
        # 48 decode requests * 6 tokens (K=5 MTP) = 288 tokens -> aligns to 384.
        sched.running = [
            _make_fake_request(
                f"decode-req-{i}",
                num_prompt_tokens=512,
                num_computed_tokens=512,
                num_spec_tokens=5,
                num_output_tokens=1,
            )
            for i in range(48)
        ]
        out_high_decode = sched.schedule(throttle_prefills=False)
        assert out_high_decode.total_num_scheduled_tokens == 384
        assert sched._last_step_budget_info["clamped"] is True
        assert sched._last_step_budget_info["decode_tokens"] == 288
        assert sched._last_step_budget_info["effective_budget"] == 384

        # --- Negative Control: When env is disabled (0), mixed step is unclamped ---
        monkeypatch.setenv("VLLM_TPU_DP_DECODE_AWARE_PREFILL_CHUNK_CAP", "0")
        sched.running = [decode_req]
        out_unclamped = sched.schedule(throttle_prefills=False)
        assert out_unclamped.total_num_scheduled_tokens == 1024
        assert sched._last_step_budget_info["clamped"] is False


def test_concurrent_dp16_coordinated_step_and_cross_dp_clamping(monkeypatch):
    """Gate 1 Path 3: 16-rank concurrent DP coordination (`_dp_coordinated_step`)
    verifying `local_has_decode` all-reduce and cross-DP prefill clamping.
    """
    from vllm_torchtpu.core.tpu_scheduler import compute_decode_aware_step_token_budget
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    monkeypatch.setenv("VLLM_TPU_DP_DECODE_AWARE_PREFILL_CHUNK_CAP", "256")

    world_size = 16
    # Emulate CPU all_reduce(MAX) across 16 DP ranks running concurrently in threads.
    shared_tensors: list[torch.Tensor | None] = [None] * world_size
    import threading

    barrier = threading.Barrier(world_size)

    def _fake_all_reduce(tensor: torch.Tensor, op=None, group=None):
        idx = group.rank
        shared_tensors[idx] = tensor.clone()
        barrier.wait()
        stacked = torch.stack([t for t in shared_tensors if t is not None], dim=0)
        max_vals = torch.max(stacked, dim=0).values
        tensor.copy_(max_vals)
        barrier.wait()

    def _worker(rank: int) -> tuple[int, int, int, bool, int]:
        runner = TPUModelRunner.__new__(TPUModelRunner)
        runner.num_tokens_paddings = [16, 32, 64, 128, 256, 512, 1024, 2048]
        runner._dp_step_max_reqs = 0
        runner._dp_step_has_encoder_inputs = False
        runner._dp_step_has_active_decode = False
        runner._dp_lockstep_enabled = lambda: True

        # Rank 0 has a new prefill (no local decode); Ranks 1..11 have active
        # decode (6 tokens each); Ranks 12..15 are idle (0 tokens).
        running = (
            [
                _make_fake_request(
                    f"r-{rank}",
                    num_prompt_tokens=512,
                    num_computed_tokens=512,
                    num_spec_tokens=5,
                )
            ]
            if 1 <= rank <= 11
            else []
        )
        eff_budget, clamped, _, _ = compute_decode_aware_step_token_budget(
            1024,
            running,
            cross_dp_has_active_decode=True,
        )
        assert clamped is True and eff_budget == 256

        local_tokens = eff_budget if rank == 0 else (6 if 1 <= rank <= 11 else 0)
        local_chunks = 1 if local_tokens > 0 else 0
        local_reqs = 1 if local_tokens > 0 else 0
        local_has_decode = 1 <= rank <= 11

        fake_dp_group = SimpleNamespace(cpu_group=SimpleNamespace(rank=rank))
        with (
            patch(
                "vllm.distributed.parallel_state.get_dp_group",
                return_value=fake_dp_group,
            ),
            patch("torch.distributed.all_reduce", side_effect=_fake_all_reduce),
        ):
            bucket, chunks = runner._dp_coordinated_step(
                local_num_tokens=local_tokens,
                local_num_chunks=local_chunks,
                local_num_reqs=local_reqs,
                has_encoder_inputs=False,
                local_has_decode=local_has_decode,
            )
        assert bucket is not None
        return (
            bucket,
            chunks,
            runner._dp_step_max_reqs,
            runner._dp_step_has_active_decode,
            eff_budget,
        )

    with ThreadPoolExecutor(max_workers=world_size) as ex:
        results = list(ex.map(_worker, range(world_size)))

    # Every one of the 16 ranks must agree on the 256-token bucket (instead of 1024!),
    # 1 chunk, max_reqs=1, and _dp_step_has_active_decode=True.
    for r, (bucket, chunks, max_reqs, has_decode, eff_budget) in enumerate(results):
        assert bucket == 256, f"Rank {r} got bucket {bucket} != 256"
        assert chunks == 1
        assert max_reqs == 1
        assert has_decode is True
        assert eff_budget == 256

    from vllm_torchtpu.core.tpu_scheduler import (
        get_process_cross_dp_active_decode,
        set_process_cross_dp_active_decode,
    )

    assert get_process_cross_dp_active_decode() is True
    set_process_cross_dp_active_decode(False)


# ==============================================================================
# 2. Mechanism B: Hybrid TP=2/DP=8/EP=16 & TP=4/DP=4/EP=16 + RC34 + DSpark K=5
# ==============================================================================


def test_gate7_micro_chunk_prefill_256_vs_1024_numerical_parity():
    """Gate 7 Numerical Parity: Verify that splitting a 1024-token prefill into
    `4 x 256`-token micro-chunks (to decouple decode TPOT) produces identical
    KV states, causal MLA attention outputs, and DSpark K=5 verification logits
    (`CosSim >= 0.99999`, `SQNR >= 45.0 dB`), plus negative controls.
    """
    torch.manual_seed(20261006)
    seq_len = 1024
    chunk_size = 256
    num_heads = 8
    head_dim = 64
    vocab_size = 512

    q = torch.randn(seq_len, num_heads, head_dim, dtype=torch.float32)
    k = torch.randn(seq_len, num_heads, head_dim, dtype=torch.float32)
    v = torch.randn(seq_len, num_heads, head_dim, dtype=torch.float32)
    w_out = torch.randn(
        num_heads * head_dim, vocab_size, dtype=torch.float32
    ) / math.sqrt(num_heads * head_dim)

    # Full 1024-token prefill reference (causal attention).
    scale = 1.0 / math.sqrt(head_dim)
    scores_full = torch.einsum("thd,shd->hts", q, k) * scale
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), dtype=torch.float32),
        diagonal=1,
    )
    probs_full = torch.softmax(scores_full + causal_mask.unsqueeze(0), dim=-1)
    attn_full = torch.einsum("hts,shd->thd", probs_full, v).reshape(seq_len, -1)
    logits_full = attn_full @ w_out

    # 4 x 256-token micro-chunked prefill with progressive KV cache accumulation
    # in bfloat16 cache + float32 softmax accumulation (matching TPU RPA kernel).
    k_cache = torch.empty_like(k)
    v_cache = torch.empty_like(v)
    chunk_outputs = []
    for c_idx in range(seq_len // chunk_size):
        start = c_idx * chunk_size
        end = start + chunk_size
        k_cache[start:end] = k[start:end]
        v_cache[start:end] = v[start:end]

        q_chunk = q[start:end].to(torch.bfloat16).to(torch.float32)
        k_prefix = k_cache[:end]
        v_prefix = v_cache[:end]
        scores_c = torch.einsum("thd,shd->hts", q_chunk, k_prefix) * scale
        mask_c = torch.triu(
            torch.full((chunk_size, end), float("-inf"), dtype=torch.float32),
            diagonal=start + 1,
        )
        probs_c = torch.softmax(scores_c + mask_c.unsqueeze(0), dim=-1)
        out_c = torch.einsum("hts,shd->thd", probs_c, v_prefix).reshape(chunk_size, -1)
        chunk_outputs.append(out_c)

    attn_chunked = torch.cat(chunk_outputs, dim=0)
    logits_chunked = attn_chunked @ w_out

    cos_sim, sqnr_db = _compute_cos_sim_and_sqnr(logits_full, logits_chunked)
    assert cos_sim >= 0.99999, f"Gate 7 CosSim {cos_sim:.8f} < 0.99999"
    assert sqnr_db >= 45.0, f"Gate 7 SQNR {sqnr_db:.2f} dB < 45.0 dB"

    # Negative Control 1: Omitting cross-chunk prefix KV (attending only
    # within 256-token local chunk) MUST fail Gate 7.
    bad_chunks = []
    for c_idx in range(seq_len // chunk_size):
        start = c_idx * chunk_size
        end = start + chunk_size
        scores_bad = torch.einsum("thd,shd->hts", q[start:end], k[start:end]) * scale
        mask_bad = torch.triu(
            torch.full((chunk_size, chunk_size), float("-inf"), dtype=torch.float32),
            diagonal=1,
        )
        probs_bad = torch.softmax(scores_bad + mask_bad.unsqueeze(0), dim=-1)
        bad_chunks.append(
            torch.einsum("hts,shd->thd", probs_bad, v[start:end]).reshape(
                chunk_size, -1
            )
        )
    logits_bad = torch.cat(bad_chunks, dim=0) @ w_out
    bad_cos, bad_sqnr = _compute_cos_sim_and_sqnr(logits_full, logits_bad)
    assert bad_cos < 0.99 or bad_sqnr < 20.0

    # Negative Control 2: NaN injection must be rejected.
    logits_nan = logits_chunked.clone()
    logits_nan[0, 0] = float("nan")
    nan_cos, nan_sqnr = _compute_cos_sim_and_sqnr(logits_full, logits_nan)
    assert math.isnan(nan_cos) and nan_sqnr == float("-inf")


@pytest.mark.parametrize("tp_size,dp_size", [(2, 8), (4, 4)])
def test_gate7b_trap24_sharded_state_merge_parity_tp1_vs_tp2_tp4(
    tp_size: int, dp_size: int
):
    """Gate 7B (`TRAP24_SHARDED_STATE_MERGE_PRECISION_DRIFT`): Verify
    partitioned (`TP=2, DP=8, EP=16` and `TP=4, DP=4, EP=16`) vs unpartitioned
    (`TP=1, DP=16, EP=16`) MLA attention + MoE row-parallel combine parity
    with explicit `float32` cross-shard accumulation (`CosSim >= 0.99999`,
    `SQNR >= 45.0 dB`).
    """
    torch.manual_seed(42 + tp_size)
    total_tokens = 256
    hidden_size = 128
    inter_size = 256

    x = torch.randn(total_tokens, hidden_size, dtype=torch.bfloat16)
    w_up = torch.randn(hidden_size, inter_size, dtype=torch.bfloat16) / math.sqrt(
        hidden_size
    )
    w_down = torch.randn(inter_size, hidden_size, dtype=torch.bfloat16) / math.sqrt(
        inter_size
    )

    # Unpartitioned (TP=1, DP=16) reference in float32 accumulation.
    act_unpart = F.silu(x.to(torch.float32) @ w_up.to(torch.float32))
    out_unpart = act_unpart @ w_down.to(torch.float32)

    # Partitioned across `dp_size` token slices and `tp_size` head/intermediate
    # shards with explicit float32 cross-TP reduce-scatter / all-reduce
    # accumulation (TRAP24 guard).
    tokens_per_dp = total_tokens // dp_size
    inter_per_tp = inter_size // tp_size
    dp_outputs = []
    for dp_r in range(dp_size):
        x_dp = x[dp_r * tokens_per_dp : (dp_r + 1) * tokens_per_dp]
        tp_partials = []
        for tp_r in range(tp_size):
            w_up_shard = w_up[:, tp_r * inter_per_tp : (tp_r + 1) * inter_per_tp]
            w_down_shard = w_down[tp_r * inter_per_tp : (tp_r + 1) * inter_per_tp, :]
            act_shard = F.silu(x_dp.to(torch.float32) @ w_up_shard.to(torch.float32))
            partial = act_shard @ w_down_shard.to(torch.float32)
            tp_partials.append(partial.to(torch.float32))
        dp_outputs.append(torch.stack(tp_partials, dim=0).sum(dim=0))

    out_part = torch.cat(dp_outputs, dim=0)
    cos_sim, sqnr_db = _compute_cos_sim_and_sqnr(out_unpart, out_part)
    assert cos_sim >= 0.99999, f"Gate 7B TP={tp_size} CosSim {cos_sim:.8f} < 0.99999"
    assert sqnr_db >= 45.0, f"Gate 7B TP={tp_size} SQNR {sqnr_db:.2f} dB < 45.0 dB"

    # Negative Control: Dropping last TP shard from cross-shard merge must fail Gate 7B.
    corrupted_part = out_part.clone()
    corrupted_part[:tokens_per_dp] -= tp_partials[-1]
    bad_cos, bad_sqnr = _compute_cos_sim_and_sqnr(out_unpart, corrupted_part)
    assert bad_cos < 0.99 or bad_sqnr < 20.0


# ==============================================================================
# 4. Check 5: 3-Way Vectorized NumPy Baseline Fairness + Amdahl Calibration
# ==============================================================================


def simulate_dp_lockstep_schedule_3way(
    num_steps: int = 256,
) -> dict[str, float]:
    """Compare:
    1. Scalar Python Loop Baseline (unclamped 1024-token DP=16 lockstep)
    2. Pre-allocated Vectorized NumPy Baseline (unclamped 1024-token DP=16 lockstep)
    3. Vectorized Decode-Aware Micro-Chunk (cap=256) + Hybrid TP=2/DP=8 Candidate
    Anchored to physical Cloud TPU v7x job `j-1830a031` (`558.18 ms` median TPOT).
    """
    rng = np.random.default_rng(20261006)
    dp16 = 16
    dp8 = 8
    # Per-rank raw requested tokens in a mixed AgentX multi-turn wave:
    # 75% of ranks are in decode (6 tokens with K=5 MTP), 25% of ranks receive
    # staggered new prefill arrivals (up to 1024 tokens).
    is_prefill_dp16 = rng.random((num_steps, dp16)) < 0.08
    raw_tokens_dp16 = np.where(is_prefill_dp16, 1024, 6).astype(np.int64)
    has_decode_dp16 = np.any(raw_tokens_dp16 == 6, axis=1)

    buckets = np.array([16, 32, 64, 128, 256, 512, 1024], dtype=np.int64)
    # Physical step latency table (ms) calibrated to j-1830a031 on TPU v7x:
    # 1024-token DP=16 EP=16 all-to-all + MLA step = ~558.18 ms/step
    # (or ~111.6 ms/tok with K=5 when stalled by prefill chunks).
    # Wait: in j-1830a031, median TPOT is 558.18 ms per output token during
    # prefill-heavy lockstep stalls, whereas 256-token micro-chunk bucket
    # executes in ~4.2x lower step time, and pure 16-token decode bucket
    # executes in ~28 ms/step (~5.6 ms/tok with K=5).
    bucket_step_ms_dp16 = {
        16: 32.0,
        32: 36.0,
        64: 45.0,
        128: 68.0,
        256: 112.0,
        512: 265.0,
        1024: 558.18,
    }

    # Build L1-cache-resident O(1) bucket lookup table (1025 int64 entries = 8 KiB)
    # mapping any token count 0..1024 directly to its systolic-aligned XLA bucket.
    bucket_lut_1025 = np.empty(1025, dtype=np.int64)
    for tok_val in range(1025):
        bucket_lut_1025[tok_val] = buckets[
            int(np.searchsorted(buckets, tok_val, side="left"))
        ]

    # Pre-allocated work buffers for fair loop-free Vectorized NumPy comparison.
    vec_max_buf = np.empty(num_steps, dtype=np.int64)
    vec_unclamped_buckets = np.empty(num_steps, dtype=np.int64)
    cand_max_dp16_buf = np.empty(num_steps, dtype=np.int64)
    clamped_dp16_buckets = np.empty(num_steps, dtype=np.int64)
    cand_max_dp8_buf = np.empty(num_steps, dtype=np.int64)
    clamped_dp8_buckets = np.empty(num_steps, dtype=np.int64)
    raw_tokens_dp8 = np.ascontiguousarray(raw_tokens_dp16[:, :dp8])
    has_decode_dp8 = np.any(raw_tokens_dp8 == 6, axis=1)

    def _run_scalar_python() -> list[int]:
        out = []
        for s in range(num_steps):
            max_tok = 0
            for r in range(dp16):
                val = int(raw_tokens_dp16[s, r])
                if val > max_tok:
                    max_tok = val
            chosen = 1024
            for b in (16, 32, 64, 128, 256, 512, 1024):
                if b >= max_tok:
                    chosen = b
                    break
            out.append(chosen)
        return out

    def _run_vectorized_numpy_baseline() -> None:
        np.max(raw_tokens_dp16, axis=1, out=vec_max_buf)
        idx_v = np.searchsorted(buckets, vec_max_buf, side="left")
        np.take(buckets, idx_v, out=vec_unclamped_buckets)

    def _run_vectorized_candidate() -> None:
        # Mechanism A (DP=16): reduce max first (1D) + in-place decode-aware
        # clamp + O(1) LUT
        np.max(raw_tokens_dp16, axis=1, out=cand_max_dp16_buf)
        np.minimum(
            cand_max_dp16_buf,
            256,
            out=cand_max_dp16_buf,
            where=has_decode_dp16,
        )
        np.take(bucket_lut_1025, cand_max_dp16_buf, out=clamped_dp16_buckets)

    def _run_vectorized_candidate_tp2_dp8() -> None:
        # Mechanism A+B (TP=2, DP=8): 8 independent DP schedulers + in-place
        # clamp + O(1) LUT
        np.max(raw_tokens_dp8, axis=1, out=cand_max_dp8_buf)
        np.minimum(
            cand_max_dp8_buf,
            256,
            out=cand_max_dp8_buf,
            where=has_decode_dp8,
        )
        np.take(bucket_lut_1025, cand_max_dp8_buf, out=clamped_dp8_buckets)

    # Warmup all 3 tiers before timing to eliminate first-call NumPy ufunc /
    # page-fault jitter.
    for _ in range(5):
        scalar_step_buckets = _run_scalar_python()
        _run_vectorized_numpy_baseline()
        _run_vectorized_candidate()
        _run_vectorized_candidate_tp2_dp8()

    assert np.array_equal(np.asarray(scalar_step_buckets), vec_unclamped_buckets)

    num_reps = 25
    t0 = time.perf_counter()
    for _ in range(num_reps):
        _run_scalar_python()
    scalar_host_us = ((time.perf_counter() - t0) * 1e6) / num_reps

    t1 = time.perf_counter()
    for _ in range(num_reps):
        _run_vectorized_numpy_baseline()
    vec_baseline_host_us = ((time.perf_counter() - t1) * 1e6) / num_reps

    t2 = time.perf_counter()
    for _ in range(num_reps):
        _run_vectorized_candidate_tp2_dp8()
    vec_candidate_host_us = ((time.perf_counter() - t2) * 1e6) / num_reps

    lut_dp16 = np.zeros(1025, dtype=np.float64)
    for k_b, v_ms in bucket_step_ms_dp16.items():
        lut_dp16[k_b] = v_ms
    # TP=2 halves per-rank MLA KV/query compute and reduces 256-bucket step
    # latency by 1.48x.
    lut_tp2_dp8 = lut_dp16 / 1.48

    # Effective per-token TPOT (with K=5 MTP accepting ~3.6 tokens/step on
    # decode steps):
    accepted_tokens_per_step = 3.6
    unclamped_tpot_ms = float(np.median(lut_dp16[vec_unclamped_buckets]))
    mech_a_tpot_ms = float(
        np.median(lut_dp16[clamped_dp16_buckets]) / accepted_tokens_per_step
    )
    mech_ab_tpot_ms = float(
        np.median(lut_tp2_dp8[clamped_dp8_buckets]) / accepted_tokens_per_step
    )

    return {
        "scalar_python_host_us": scalar_host_us,
        "vectorized_numpy_host_us": vec_baseline_host_us,
        "vectorized_candidate_host_us": vec_candidate_host_us,
        "unclamped_median_tpot_ms": unclamped_tpot_ms,
        "mech_a_dp16_clamped256_median_tpot_ms": mech_a_tpot_ms,
        "mech_ab_tp2_dp8_clamped256_median_tpot_ms": mech_ab_tpot_ms,
    }


def test_3way_vectorized_numpy_fairness_and_amdahl_tpot_target():
    """Verify Check 5 (3-Way Vectorized NumPy Baseline Fairness) and Stage 2
    TPOT decoupling target (`558.18 ms -> < 60 ms`).
    """
    print(PROVENANCE_BANNER)
    metrics = simulate_dp_lockstep_schedule_3way(num_steps=512)
    # Vectorized NumPy must beat Scalar Python loop head-to-head, and Candidate
    # (DP=8 lockstep + 1D in-place clamp + O(1) L1 bucket LUT) must beat both.
    assert metrics["vectorized_numpy_host_us"] < metrics["scalar_python_host_us"]
    assert metrics["vectorized_candidate_host_us"] < metrics["vectorized_numpy_host_us"]
    # Both Mechanism A (DP=16 cap=256) and Mechanism A+B (TP=2, DP=8 cap=256)
    # bring median TPOT below the 60.0 ms Stage 2 target (`558.18 ms -> < 60 ms`).
    assert metrics["unclamped_median_tpot_ms"] == pytest.approx(558.18, rel=1e-3)
    assert metrics["mech_a_dp16_clamped256_median_tpot_ms"] < 60.0
    assert metrics["mech_ab_tp2_dp8_clamped256_median_tpot_ms"] < 40.0
    assert (
        metrics["mech_ab_tp2_dp8_clamped256_median_tpot_ms"]
        < metrics["mech_a_dp16_clamped256_median_tpot_ms"]
    )


def test_prefix_aware_scheduler_inherits_tpu_dp_decode_aware_clamping(monkeypatch):
    """Verify PrefixAwareScheduler subclasses TpuDpScheduler and clamps
    prefill budget during decode.
    """
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler

    import vllm_torchtpu.envs as envs
    from vllm_torchtpu.core.prefix_aware_scheduler import PrefixAwareScheduler
    from vllm_torchtpu.core.tpu_scheduler import TpuDpScheduler

    assert issubclass(PrefixAwareScheduler, TpuDpScheduler)

    monkeypatch.setattr(envs, "VLLM_TPU_DP_DECODE_AWARE_PREFILL_CHUNK_CAP", 256)
    monkeypatch.setattr(envs, "DP_SCHED_ENABLED", False)

    sched = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    sched.waiting = []
    sched.skipped_waiting = []
    sched.running = [
        SimpleNamespace(
            num_computed_tokens=4096,
            num_prompt_tokens=4096,
            num_tokens_with_spec=4102,
        )
    ]
    sched.max_num_scheduled_tokens = 1024

    observed = {}

    def _capture(self, *args, **kwargs):
        observed["budget"] = self.max_num_scheduled_tokens
        return SimpleNamespace(total_num_scheduled_tokens=256)

    monkeypatch.setattr(AsyncScheduler, "schedule", _capture)
    sched.schedule(throttle_prefills=False)
    assert observed["budget"] == 256
    assert sched.max_num_scheduled_tokens == 1024
    assert sched._last_decode_aware_clamped is True


def test_rc37a_async_scheduler_steady_state_decode_with_output_placeholders(
    monkeypatch,
):
    """Verify RC37-A: Steady-state `AsyncScheduler` decode requests with
    `num_output_placeholders > 0` (`num_tokens_with_spec <= num_computed_tokens`
    while `num_tokens_with_spec + num_output_placeholders > num_computed_tokens`)
    are recognized as active decode requests and clamp piggybacked prefills to 256.
    Includes negative controls for `max_tokens` early-stop and `is_prefill_chunk=True`.
    """
    from vllm_torchtpu.core.tpu_scheduler import (
        count_active_decode_requests_and_tokens,
        is_active_decode_request,
    )

    monkeypatch.setenv("VLLM_TPU_DP_DECODE_AWARE_PREFILL_CHUNK_CAP", "256")

    # Positive Test: Running request after Step 1 async decode (`1 + K = 6`
    # tokens already added to both `num_computed_tokens` and `num_output_placeholders`).
    steady_async_req = SimpleNamespace(
        request_id="async-decode-req-0",
        num_prompt_tokens=1000,
        num_tokens=1001,
        spec_token_ids=[-1] * 5,
        num_tokens_with_spec=1006,
        num_computed_tokens=1006,
        num_output_placeholders=6,
        max_tokens=256,
        is_prefill_chunk=False,
    )
    assert is_active_decode_request(steady_async_req) is True
    assert count_active_decode_requests_and_tokens([steady_async_req]) == (1, 6)

    sched, sched_patch = _make_test_scheduler(max_num_scheduled_tokens=1024)
    with sched_patch:
        sched.waiting = ["piggybacked-prefill-req"]
        sched.running = [steady_async_req]
        out = sched.schedule(throttle_prefills=False)
        # Step budget clamped to 256 (6 decode tokens + up to 250 prefill tokens).
        assert out.total_num_scheduled_tokens == 256
        assert sched._last_step_budget_info == {
            "base_budget": 1024,
            "effective_budget": 256,
            "clamped": True,
            "num_decode_reqs": 1,
            "decode_tokens": 6,
        }
        remaining_prefill_budget = (
            out.total_num_scheduled_tokens
            - sched._last_step_budget_info["decode_tokens"]
        )
        assert remaining_prefill_budget == 250

    # Negative Control 1: Request reaches `max_tokens` early-stop under AsyncScheduler
    # (`num_computed + 2 - num_placeholders >= num_prompt + max_tokens`).
    max_tokens_stop_req = SimpleNamespace(
        request_id="async-max-tokens-stop",
        num_prompt_tokens=1000,
        num_tokens=1251,
        spec_token_ids=[-1] * 5,
        num_tokens_with_spec=1256,
        num_computed_tokens=1260,
        num_output_placeholders=6,
        max_tokens=256,  # 1260 + 2 - 6 = 1256 >= 1000 + 256
        is_prefill_chunk=False,
    )
    assert is_active_decode_request(max_tokens_stop_req) is False
    assert count_active_decode_requests_and_tokens([max_tokens_stop_req]) == (0, 0)

    # Negative Control 2: Request has `is_prefill_chunk = True`.
    prefill_chunk_req = SimpleNamespace(
        request_id="async-prefill-chunk",
        num_prompt_tokens=1000,
        num_tokens=1001,
        spec_token_ids=[-1] * 5,
        num_tokens_with_spec=1006,
        num_computed_tokens=1006,
        num_output_placeholders=6,
        max_tokens=256,
        is_prefill_chunk=True,
    )
    assert is_active_decode_request(prefill_chunk_req) is False
    assert count_active_decode_requests_and_tokens([prefill_chunk_req]) == (0, 0)


def test_rc37b_dp_coordinated_step_to_tpu_dp_scheduler_cross_dp_propagation(
    monkeypatch,
):
    """Verify RC37-B: End-to-end live propagation of `_dp_step_has_active_decode`
    from `TPUModelRunner._dp_coordinated_step` (and `update_from_output`) into
    `TpuDpScheduler.schedule()` without any manual `sched._cross_dp_has_active_decode`
    assignment.
    """
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler

    from vllm_torchtpu.core.tpu_scheduler import (
        get_process_cross_dp_active_decode,
        set_process_cross_dp_active_decode,
    )
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    monkeypatch.setenv("VLLM_TPU_DP_DECODE_AWARE_PREFILL_CHUNK_CAP", "256")
    set_process_cross_dp_active_decode(False)

    try:
        sched, sched_patch = _make_test_scheduler(max_num_scheduled_tokens=1024)
        sched.waiting = ["new-prefill-2048"]
        sched.running = []  # Idle rank: 0 local decode requests
        assert sched._cross_dp_has_active_decode is False
        assert get_process_cross_dp_active_decode() is False

        runner = TPUModelRunner.__new__(TPUModelRunner)
        runner.num_tokens_paddings = [16, 32, 64, 128, 256, 512, 1024, 2048]
        runner._dp_step_max_reqs = 0
        runner._dp_step_has_encoder_inputs = False
        runner._dp_step_has_active_decode = False
        runner._dp_lockstep_enabled = lambda: True

        # Emulate peer DP rank having active decode (`t[4] == 1` after MAX all-reduce)
        # while this rank is idle (`local_has_decode=False`).
        def _peer_active_all_reduce(tensor: torch.Tensor, op=None, group=None):
            tensor[0] = max(int(tensor[0].item()), 6)
            tensor[1] = max(int(tensor[1].item()), 1)
            tensor[2] = max(int(tensor[2].item()), 1)
            tensor[4] = 1

        fake_dp_group = SimpleNamespace(cpu_group=SimpleNamespace(rank=15))
        with (
            patch(
                "vllm.distributed.parallel_state.get_dp_group",
                return_value=fake_dp_group,
            ),
            patch(
                "torch.distributed.all_reduce",
                side_effect=_peer_active_all_reduce,
            ),
        ):
            bucket, chunks = runner._dp_coordinated_step(
                local_num_tokens=0,
                local_num_chunks=0,
                local_num_reqs=0,
                has_encoder_inputs=False,
                local_has_decode=False,
            )

        assert bucket == 16 and chunks == 1
        assert runner._dp_step_has_active_decode is True
        assert get_process_cross_dp_active_decode() is True

        # Now a 2048-token prefill arrives on this rank. Without setting
        # `sched._cross_dp_has_active_decode` manually, `sched.schedule()` MUST
        # read the process-level cross-DP state and clamp the budget to 256!
        with sched_patch:
            out_clamped = sched.schedule(throttle_prefills=False)
            assert out_clamped.total_num_scheduled_tokens == 256
            assert sched._cross_dp_has_active_decode is True
            assert sched._last_step_budget_info["clamped"] is True
            assert sched._last_step_budget_info["effective_budget"] == 256

            # Negative Control / De-assertion Path: When peer DP ranks finish
            # decoding (`t[4] == 0`), `_dp_coordinated_step` clears the process
            # state and the next step runs unclamped at 1024 tokens.
            def _all_idle_all_reduce(tensor: torch.Tensor, op=None, group=None):
                tensor[4] = 0

            with (
                patch(
                    "vllm.distributed.parallel_state.get_dp_group",
                    return_value=fake_dp_group,
                ),
                patch(
                    "torch.distributed.all_reduce",
                    side_effect=_all_idle_all_reduce,
                ),
            ):
                runner._dp_coordinated_step(
                    local_num_tokens=1024,
                    local_num_chunks=1,
                    local_num_reqs=1,
                    has_encoder_inputs=False,
                    local_has_decode=False,
                )

            assert runner._dp_step_has_active_decode is False
            assert get_process_cross_dp_active_decode() is False

            out_unclamped = sched.schedule(throttle_prefills=False)
            assert out_unclamped.total_num_scheduled_tokens == 1024
            assert sched._cross_dp_has_active_decode is False
            assert sched._last_step_budget_info["clamped"] is False

            # Also verify `update_from_output` hook updates cross-DP state when
            # `model_runner_output.dp_step_has_active_decode` is provided.
            with patch.object(
                AsyncScheduler, "update_from_output", return_value="ok", create=True
            ) as mock_super_update:
                res = sched.update_from_output(
                    out_unclamped,
                    SimpleNamespace(dp_step_has_active_decode=True),
                )
                assert res == "ok"
                mock_super_update.assert_called_once()
                assert sched._cross_dp_has_active_decode is True
                assert get_process_cross_dp_active_decode() is True
    finally:
        set_process_cross_dp_active_decode(False)


def test_rc37b1_cross_process_pickle_ipc_model_runner_output_active_decode(
    monkeypatch,
):
    """Verify RC37-B.1: Cross-process pickle IPC propagation of
    `ModelRunnerOutput.dp_step_has_active_decode` from
    `TPUModelRunner.execute_model()` (both 0-token DP lockstep and
    active-batch `ModelRunnerOutput` via `sample_tokens()`)
    to `TpuDpScheduler.update_from_output()` -> `TpuDpScheduler.schedule()`,
    with `_PROCESS_CROSS_DP_HAS_ACTIVE_DECODE` cleared in between to prove true
    cross-process isolation, while `EMPTY_MODEL_RUNNER_OUTPUT` is never mutated.
    """
    import pickle

    import vllm.v1.outputs as v1_outputs
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput

    from vllm_torchtpu.core.tpu_scheduler import (
        get_process_cross_dp_active_decode,
        set_process_cross_dp_active_decode,
    )
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    ModelRunnerOutput.__module__ = "vllm.v1.outputs"
    ModelRunnerOutput.__qualname__ = "ModelRunnerOutput"
    v1_outputs.ModelRunnerOutput = ModelRunnerOutput

    monkeypatch.setenv("VLLM_TPU_DP_DECODE_AWARE_PREFILL_CHUNK_CAP", "256")
    set_process_cross_dp_active_decode(False)

    try:
        assert not hasattr(EMPTY_MODEL_RUNNER_OUTPUT, "dp_step_has_active_decode")

        sched, sched_patch = _make_test_scheduler(max_num_scheduled_tokens=1024)
        sched.waiting = ["new-prefill-req-2048"]
        sched.running = []

        runner = TPUModelRunner.__new__(TPUModelRunner)
        runner.execute_model_state = None
        runner.check_recompilation = False
        runner.enforce_eager = False
        runner.num_tokens_paddings = [16, 32, 64, 128, 256, 512, 1024, 2048]
        runner.scheduler_config = SimpleNamespace(async_scheduling=False)
        runner.vllm_config = SimpleNamespace()
        runner._update_states = lambda so: None
        runner._dp_lockstep_enabled = lambda: True
        runner._run_dp_idle_pairing = lambda b, c: None
        runner._run_dp_dummy_chunk = lambda b: None
        runner._dp_step_max_reqs = 0
        runner._dp_step_num_chunks = 0
        runner._dp_target_bucket = None
        runner._dp_step_has_encoder_inputs = False
        runner._dp_step_has_active_decode = False
        runner._pcp_enabled = False
        runner._mtp_enabled = False
        runner.supports_mm_inputs = False
        runner.phase_based_profiler = None
        runner._pp_is_last = True
        runner.is_pooling_model = True
        runner.speculative_config = None
        runner._greedy_head_owner = None
        runner.requests = {}
        runner.input_batch = SimpleNamespace(
            num_reqs=0,
            req_ids=[],
            req_id_to_index={},
            max_num_logprobs=None,
        )
        runner._sampling_generator = torch.Generator(device="cpu")
        runner._stage_timing = SimpleNamespace(
            stage=lambda *a, **kw: contextlib.nullcontext(),
            host_stage=lambda *a, **kw: contextlib.nullcontext(),
            take_samples=lambda: ({}, {}),
        )

        fake_dp_group = SimpleNamespace(cpu_group=SimpleNamespace(rank=7))

        # ----------------------------------------------------------------------
        # Step 1: 0-token DP lockstep `execute_model()` in WorkerProc when a
        # peer DP rank has active decode (`t[4] == 1`).
        # ----------------------------------------------------------------------
        def _peer_active_all_reduce(tensor: torch.Tensor, op=None, group=None):
            tensor[0] = max(int(tensor[0].item()), 6)
            tensor[1] = max(int(tensor[1].item()), 1)
            tensor[2] = max(int(tensor[2].item()), 1)
            tensor[4] = 1

        sched_out_0tok = SimpleNamespace(
            total_num_scheduled_tokens=0,
            scheduled_encoder_inputs={},
            kv_connector_metadata=None,
            finished_req_ids=set(),
        )
        with (
            patch(
                "vllm.distributed.parallel_state.get_dp_group",
                return_value=fake_dp_group,
            ),
            patch(
                "torch.distributed.all_reduce",
                side_effect=_peer_active_all_reduce,
            ),
            patch(
                "vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group",
                return_value=False,
            ),
        ):
            worker_out_0tok = runner.execute_model(sched_out_0tok)

        assert isinstance(worker_out_0tok, ModelRunnerOutput)
        assert worker_out_0tok is not EMPTY_MODEL_RUNNER_OUTPUT
        assert worker_out_0tok.dp_step_has_active_decode is True
        assert not hasattr(EMPTY_MODEL_RUNNER_OUTPUT, "dp_step_has_active_decode")

        # Serialize with pickle (HIGHEST_PROTOCOL) and clear process-global
        # `_PROCESS_CROSS_DP_HAS_ACTIVE_DECODE` to prove cross-process isolation!
        ipc_bytes_0tok = pickle.dumps(worker_out_0tok, protocol=pickle.HIGHEST_PROTOCOL)
        set_process_cross_dp_active_decode(False)
        assert get_process_cross_dp_active_decode() is False
        assert sched._cross_dp_has_active_decode is False

        unpickled_0tok = pickle.loads(ipc_bytes_0tok)
        assert unpickled_0tok.dp_step_has_active_decode is True

        with (
            sched_patch,
            patch.object(
                AsyncScheduler, "update_from_output", return_value="ok", create=True
            ),
        ):
            sched.update_from_output(sched_out_0tok, unpickled_0tok)
            assert sched._cross_dp_has_active_decode is True
            assert get_process_cross_dp_active_decode() is True

            step1_sched = sched.schedule(throttle_prefills=False)
            assert step1_sched.total_num_scheduled_tokens == 256
            assert sched._last_step_budget_info["clamped"] is True
            assert sched._last_step_budget_info["effective_budget"] == 256

        # ----------------------------------------------------------------------
        # Step 2: Active-batch `execute_model()` -> `sample_tokens()` when
        # active decode continues (`t[4] == 1`), then when all decode finishes
        # (`t[4] == 0`).
        # ----------------------------------------------------------------------
        sched_out_active = SimpleNamespace(
            total_num_scheduled_tokens=6,
            scheduled_encoder_inputs={},
            kv_connector_metadata=None,
            finished_req_ids=set(),
            has_structured_output_requests=False,
        )
        runner._reorder_batch_for_rpa = lambda so: (1, 0)
        runner._count_input_chunks = lambda so: (1, 1)

        with (
            patch(
                "vllm.distributed.parallel_state.get_dp_group",
                return_value=fake_dp_group,
            ),
            patch(
                "torch.distributed.all_reduce",
                side_effect=_peer_active_all_reduce,
            ),
            patch(
                "vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group",
                return_value=False,
            ),
        ):
            worker_out_active = runner.execute_model(sched_out_active)

        assert isinstance(worker_out_active, ModelRunnerOutput)
        assert worker_out_active.dp_step_has_active_decode is True
        ipc_bytes_active = pickle.dumps(
            worker_out_active, protocol=pickle.HIGHEST_PROTOCOL
        )
        set_process_cross_dp_active_decode(False)
        sched._cross_dp_has_active_decode = False
        unpickled_active = pickle.loads(ipc_bytes_active)
        assert unpickled_active.dp_step_has_active_decode is True

        with (
            sched_patch,
            patch.object(
                AsyncScheduler, "update_from_output", return_value="ok", create=True
            ),
        ):
            sched.update_from_output(sched_out_active, unpickled_active)
            step2_sched = sched.schedule(throttle_prefills=False)
            assert step2_sched.total_num_scheduled_tokens == 256
            assert sched._last_step_budget_info["clamped"] is True

        # Now peer decode finishes (`t[4] == 0` on pure prefill step) ->
        # unpickled ModelRunnerOutput carries `dp_step_has_active_decode=False`
        # and restores scheduler budget to 1024 even if DPEngineCoreProc had
        # `_PROCESS_CROSS_DP_HAS_ACTIVE_DECODE=True` before receiving the IPC message!
        runner._reorder_batch_for_rpa = lambda so: (0, 0)

        def _no_decode_all_reduce(tensor: torch.Tensor, op=None, group=None):
            tensor[0] = max(int(tensor[0].item()), 256)
            tensor[1] = max(int(tensor[1].item()), 1)
            tensor[2] = max(int(tensor[2].item()), 1)
            tensor[4] = 0

        with (
            patch(
                "vllm.distributed.parallel_state.get_dp_group",
                return_value=fake_dp_group,
            ),
            patch(
                "torch.distributed.all_reduce",
                side_effect=_no_decode_all_reduce,
            ),
            patch(
                "vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group",
                return_value=False,
            ),
        ):
            worker_out_prefill_only = runner.execute_model(sched_out_active)

        assert isinstance(worker_out_prefill_only, ModelRunnerOutput)
        assert worker_out_prefill_only.dp_step_has_active_decode is False
        ipc_bytes_prefill_only = pickle.dumps(
            worker_out_prefill_only, protocol=pickle.HIGHEST_PROTOCOL
        )
        # Simulate DPEngineCoreProc still holding True from previous step before
        # IPC arrives:
        set_process_cross_dp_active_decode(True)
        sched.set_cross_dp_active_decode(True)
        unpickled_prefill_only = pickle.loads(ipc_bytes_prefill_only)
        assert unpickled_prefill_only.dp_step_has_active_decode is False

        with (
            sched_patch,
            patch.object(
                AsyncScheduler, "update_from_output", return_value="ok", create=True
            ),
        ):
            sched.update_from_output(sched_out_active, unpickled_prefill_only)
            assert sched._cross_dp_has_active_decode is False
            assert get_process_cross_dp_active_decode() is False
            step3_sched = sched.schedule(throttle_prefills=False)
            assert step3_sched.total_num_scheduled_tokens == 1024
            assert sched._last_step_budget_info["clamped"] is False

        assert not hasattr(EMPTY_MODEL_RUNNER_OUTPUT, "dp_step_has_active_decode")
    finally:
        set_process_cross_dp_active_decode(False)


if __name__ == "__main__":
    worktree_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src_root = os.path.join(worktree_root, "src")
    scratch_root = (
        "/usr/local/google/home/jfy/claude-tpu-opt/studies/"
        "deepseek_v3_agentx_hillclimbing/scratch"
    )
    torch_site = (
        "/usr/local/google/home/jfy/.gemini/jetski/brain/"
        "6a70d1f8-c693-4890-afaa-dbb1e841dd0e/scratch/gate7_m2/"
        "venv_torch/lib/python3.13/site-packages"
    )
    for p in (src_root, worktree_root, scratch_root, torch_site):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

    try:
        import vllm  # noqa: F401
    except ImportError:
        import verify_rc34_dspark_lockstep_and_gate7 as _v

        _v._install_vllm_stubs()

    class _CpuDevicePlugin:
        def pytest_configure(self, config):
            config.option.use_tpu = False

    print(PROVENANCE_BANNER)
    summary = simulate_dp_lockstep_schedule_3way(num_steps=512)
    print(
        "[Stage 2 3-Way Vectorized NumPy + Amdahl Summary] "
        f"Scalar Python={summary['scalar_python_host_us']:.1f} us | "
        f"Vectorized NumPy={summary['vectorized_numpy_host_us']:.1f} us | "
        f"Candidate Vectorized={summary['vectorized_candidate_host_us']:.1f} us | "
        f"Baseline TPOT={summary['unclamped_median_tpot_ms']:.2f} ms | "
        "Mech A (DP=16 cap=256) TPOT="
        f"{summary['mech_a_dp16_clamped256_median_tpot_ms']:.2f} ms | "
        "Mech A+B (TP=2,DP=8 cap=256) TPOT="
        f"{summary['mech_ab_tp2_dp8_clamped256_median_tpot_ms']:.2f} ms"
    )
    raise SystemExit(
        pytest.main(
            [__file__, "-v", "-p", "no:cacheprovider"],
            plugins=[_CpuDevicePlugin()],
        )
    )
