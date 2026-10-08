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
"""The drafter's draft megakernel seam (`Eagle3Proposer.attach_draft_megakernel`):
which programs it builds, with which compile sizes and cache prefixes, how it
undoes them, and which program a fused-loop chunk runs. CPU, with the program
builders and the vLLM contexts stubbed."""

from __future__ import annotations

import contextlib
import re
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import vllm_torchtpu.spec_decode.eagle3 as e3
from vllm_torchtpu.megakernel.adapter.fingerprint import (
    runtime_src_hash,
    source_hash,
    src8,
)
from vllm_torchtpu.models.vllm import qwen3_next_patch
from vllm_torchtpu.spec_decode import mtp_fused_loop, mtp_loop_model
from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

PADDINGS = (4, 8, 16, 24, 32, 48, 64, 96, 128, 256)


class _DraftMk:
    """The draft megakernel as the drafter sees it (`draft.py`'s program)."""

    bound = True
    layer_name = "mtp.layers.0.self_attn.attn"

    def __init__(self, loop=(4, 8, 16), verify=(4, 8, 16, 24, 32, 48, 64)):
        self.loop_buckets, self.verify_buckets = tuple(loop), tuple(verify)
        self.decisions: list[tuple[int, bool]] = []
        self.serve = True

    def padded_rows(self, rows: int) -> int:
        return next((p for p in PADDINGS if p >= rows), rows)

    def use_for_step(self, md: Any, rows: int, *, step0: bool) -> bool:
        self.decisions.append((rows, step0))
        return self.serve

    def note_step(self, key: str, run: bool) -> bool:
        return run


def _null(*_a: Any, **_k: Any) -> Any:
    return contextlib.nullcontext()


def _proposer(monkeypatch: pytest.MonkeyPatch, builds: list[dict]) -> Eagle3Proposer:
    from tests.spec_decode.test_eagle3 import _make_proposer

    proposer = _make_proposer(method="mtp")
    proposer.speculative_config.num_speculative_tokens = 3
    proposer.draft_model = SimpleNamespace(
        _dynamic_arg_dims={"input_ids": 0, "positions": -1, "hidden_states": 0}
    )
    proposer.draft_vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(compile_sizes=list(PADDINGS))
    )
    proposer.runner = SimpleNamespace(
        mesh=object(), max_num_reqs=16, _dp_lockstep_enabled=lambda: False
    )
    proposer._mtp_fused_loop = object()
    proposer._mtp_fused_loop_max_p = 16
    proposer._draft_greedy_head_owner = None
    for name in ("set_model_tag", "set_vllm_model_wrapper_context"):
        monkeypatch.setattr(e3, name, _null)
    monkeypatch.setattr(e3, "set_current_vllm_config", _null)
    monkeypatch.setattr(
        qwen3_next_patch, "is_step0_post_attn_gather_installed", lambda _m: True
    )

    def builder(kind: str):
        def build(**kwargs: Any) -> Any:
            builds.append({"kind": kind, **kwargs})
            return SimpleNamespace(kind=kind)

        return build

    monkeypatch.setattr(mtp_fused_loop, "build_mtp_fused_loop", builder("loop"))
    monkeypatch.setattr(mtp_loop_model, "build_mtp_loop_step_model", builder("step0"))
    return proposer


def _tag() -> str:
    return "_" + src8(runtime_src_hash(), source_hash(sys.modules[__package__]))


def test_attach_builds_both_programs_for_the_megakernels_buckets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builds: list[dict] = []
    proposer = _proposer(monkeypatch, builds)
    mk = _DraftMk()
    assert proposer.attach_draft_megakernel(mk) is True
    assert proposer._mtp_mk is mk
    loop, step0 = builds
    tag = _tag()
    assert re.fullmatch(r"_[0-9a-f]{8}", tag)
    assert loop["kind"] == "loop" and step0["kind"] == "step0"
    assert loop["megakernel"] is True and step0["megakernel"] is True
    assert loop["prefix"] == f"mtp_fused_loop_k3_mk{tag}"
    assert step0["prefix"] == f"mtp_mk_step0{tag}"
    # the loop compiles up to its widest bucket (16 rows); the step-0 program up
    # to the padded rows of 16 requests x 4 (64), within twice the widest (128)
    assert loop["vllm_config"].compilation_config.compile_sizes == [4, 8, 16]
    assert step0["vllm_config"].compilation_config.compile_sizes == [
        4,
        8,
        16,
        24,
        32,
        48,
        64,
    ]
    # the loop's hidden carry is padded by the drafter, not marked dynamic
    assert "hidden_states" not in loop["dynamic_arg_dims"]
    assert "hidden_states" in step0["dynamic_arg_dims"]
    assert loop["num_steps"] == 2 and loop["propose_step0"] is True
    # the draft's own compilation config is untouched
    assert proposer.draft_vllm_config.compilation_config.compile_sizes == list(PADDINGS)

    proposer._mtp_mk_step0_warm = proposer._mtp_fused_loop_mk_warm = True
    proposer.detach_draft_megakernel()
    assert proposer._mtp_mk is None
    assert proposer._mtp_mk_step0 is None and proposer._mtp_fused_loop_mk is None
    assert not proposer._mtp_mk_step0_warm and not proposer._mtp_fused_loop_mk_warm


def test_attach_builds_nothing_it_cannot_serve(monkeypatch: pytest.MonkeyPatch) -> None:
    builds: list[dict] = []
    proposer = _proposer(monkeypatch, builds)
    proposer.draft_model = SimpleNamespace(_dynamic_arg_dims=None)
    assert proposer.attach_draft_megakernel(_DraftMk()) is False

    proposer = _proposer(monkeypatch, builds)
    proposer.draft_vllm_config.compilation_config.compile_sizes = []
    assert proposer.attach_draft_megakernel(_DraftMk()) is False
    assert proposer._mtp_mk is None

    # no fused loop and no step-0 gather: neither program
    proposer = _proposer(monkeypatch, builds)
    proposer._mtp_fused_loop = None
    monkeypatch.setattr(
        qwen3_next_patch, "is_step0_post_attn_gather_installed", lambda _m: False
    )
    assert proposer.attach_draft_megakernel(_DraftMk()) is False
    assert builds == []


def test_attach_failure_keeps_the_xla_programs(monkeypatch: pytest.MonkeyPatch) -> None:
    proposer = _proposer(monkeypatch, [])

    def boom(**_kwargs: Any) -> Any:
        raise RuntimeError("trace failed")

    monkeypatch.setattr(mtp_loop_model, "build_mtp_loop_step_model", boom)
    assert proposer.attach_draft_megakernel(_DraftMk()) is False
    assert proposer._mtp_mk is None
    assert proposer._mtp_mk_step0 is None and proposer._mtp_fused_loop_mk is None


def test_unhashable_megakernel_package_builds_untagged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unhashable(_module: Any) -> str:
        raise ValueError("no source")

    monkeypatch.setattr(
        "vllm_torchtpu.megakernel.adapter.fingerprint.source_hash", unhashable
    )
    assert e3._draft_megakernel_tag(_DraftMk()) == ""


class _Program:
    def __init__(self, name: str, log: list[str]) -> None:
        self.name, self.log = name, log

    def __call__(self, _ids: Any, _pos: Any, hidden: torch.Tensor, seq_lens: Any):
        self.log.append(self.name)
        seq_lens.add_(2)
        return hidden


def test_fused_loop_chunk_runs_the_megakernel_program_when_it_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A chunk the megakernel serves runs the megakernel loop program inside
    the megakernel context (twice on first use, seq_lens put back between), a
    chunk it doesn't serve or with a position override runs the XLA loop."""
    proposer = _proposer(monkeypatch, [])
    log: list[str] = []
    seen_mk: list[Any] = []
    seq_lens = torch.zeros(4, dtype=torch.int32)
    md = {"mtp.layers.0.self_attn.attn": SimpleNamespace(seq_lens=seq_lens)}
    proposer._build_draft_attn_metadata = lambda **_k: md
    proposer._draft_md_cache = {}
    proposer._draft_rows_sliced = lambda *_a: False
    proposer._get_loop_token_padding_state = lambda: None
    proposer._loop_bucket = lambda _n: 4
    proposer.runner._dp_num_tokens_across_dp = lambda n: n
    monkeypatch.setattr(e3, "set_forward_context", _null)
    monkeypatch.setattr(e3._ffc, "enabled", lambda: False)
    proposer._mtp_fused_loop = _Program("xla", log)
    proposer._mtp_fused_loop_mk = _Program("mk", log)
    mk = _DraftMk()
    proposer._mtp_mk = mk

    real_ctx = qwen3_next_patch.mtp_megakernel_context

    @contextlib.contextmanager
    def spy(m: Any):
        seen_mk.append(m)
        with real_ctx(m):
            yield

    monkeypatch.setattr(qwen3_next_patch, "mtp_megakernel_context", spy)

    def run(override: Any = None) -> None:
        chunk = SimpleNamespace(
            attn_ctx=SimpleNamespace(position_ids_override=override)
        )
        proposer._run_mtp_fused_loop(
            chunk,
            torch.zeros(4, dtype=torch.int32),
            torch.zeros((4, 2)),
            None,
            None,
            torch.zeros(5, dtype=torch.int32),
            torch.zeros(4, dtype=torch.int32),
        )

    run()
    assert log == ["mk", "mk"] and seen_mk == [mk]
    assert proposer._mtp_fused_loop_mk_warm and not proposer._mtp_fused_loop_warm
    assert seq_lens.tolist() == [2, 2, 2, 2]  # put back once, advanced once
    assert mk.decisions == [(4, False)]
    mk.serve = False
    run()
    run(override=torch.zeros(4))
    assert log == ["mk", "mk", "xla", "xla", "xla"]
    assert len(seen_mk) == 1
    assert qwen3_next_patch._mtp_megakernel is None
