"""Port checks of MM's VLLM_TPU_FAST_METADATA_BUILD (dongruizi's c2ae6a9a5) onto the lever-2 chain 648f9cdb1 (q38
DEC-FB-3). tests/layers/core/test_attention_metadata.py carries MM's equality test and the bucket sweep
(test_fast_staged_build_bitwise_across_buckets: every AttentionMetadata field bitwise, per-layer sharing, the unified
state list, the None fallback). This adds, through the real TPUModelRunner._build_attention_metadata on a mock runner:
1. the gate: flag off -> one build() per group; flag on with a staged chunk -> no build() call and the same metadata;
   flag on with position_ids_override -> the ordinary walk; flag on where the inline walk returns None -> the
   ordinary walk (the fallback is taken, not an empty dict); a group without a staged table makes the inline
   walk decline;
2. the group plan is rebuilt when runner.attn_groups changes (no stale layer names);
3. switch off: tpu_runner.py is 648f9cdb1's, AST for AST, once the flag-guarded branch and the new method are dropped;
4. the flag defaults off (its compile-key status: tests/compilation/test_tpu_compile_key_registration.py);
5. engagement (log only, q38 DEC-FB-3): one plan line per plan, and the inline / fallback counter line every
   _FAST_METADATA_LOG_EVERY walks counts each kind correctly; nothing is logged or counted with the flag off.
"""
import ast
import copy
import importlib.util
import pathlib
import types

import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

HERE = pathlib.Path(__file__).parent
_spec = importlib.util.spec_from_file_location("q38_am_tests", HERE.parent / "layers" / "core" / "test_attention_metadata.py")
_am = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_am)
H = _am.TestAttentionMetadataBuilderPlumbing()
FLAG = "VLLM_TPU_FAST_METADATA_BUILD"


def _setup(unified, n_mamba=2, override=None):
    runner = H._make_runner_mock(num_groups=1 + n_mamba, max_num_blocks_per_req=6)
    runner._unified_kv_layout = unified
    builders = [H._make_builder(runner, kv_cache_group_id=0)]
    builders += [H._make_mamba_builder(runner, kv_cache_group_id=g) for g in range(1, 1 + n_mamba)]
    groups = H._make_attn_groups(runner, builders)
    for i, (group,) in enumerate(groups):
        group.layer_names = [f"layer.{i}.a", f"layer.{i}.b"]
    calls = []
    for b in builders:
        real = b.build
        b.build = (lambda real: lambda *a, **k: (calls.append(1), real(*a, **k))[1])(real)
    runner._build_attention_metadata_staged = types.MethodType(TPUModelRunner._build_attention_metadata_staged, runner)
    ctx = _am.AttentionMetadataBuilderContext(
        num_reqs=4, start_index=0, use_max_model_len=True,
        seq_lens=torch.tensor([0, 17, 33, 49], dtype=torch.int32),
        query_start_loc=torch.arange(5, dtype=torch.int32),
        request_distribution=torch.tensor([4, 4, 4], dtype=torch.int32),
        mamba_state_indices=None if unified else torch.arange(4),
        mamba_slot_read_offsets=torch.zeros(8, dtype=torch.int32),
        mamba_ckpt_window=3 if unified else 1,
        mamba_request_distribution=torch.tensor([4, 4, 4]),
        unified_mamba_state_indices=[] if unified else None,
        position_ids_override=override,
    )
    runner._attn_metadata_builder_ctx = ctx
    return runner, groups, calls


def _decline(runner):
    """Stage the chunk, then drop the walk program's mamba products: with a unified layout build() must then derive
    each mamba group's state itself, so the inline walk declines (staging makes products even for one mamba group,
    so a one-group layout is not a decline case; FMB proof i1 j-a53d72e1)."""
    ctx = runner._attn_metadata_builder_ctx
    _am.stage_block_table_uploads(runner, ctx, 4)
    ctx.staged_mamba_products = None
    assert TPUModelRunner._build_attention_metadata_staged(runner, ctx) is None


def _walk(runner):
    out, extra = TPUModelRunner._build_attention_metadata(runner, num_tokens=4, num_reqs=4, num_reqs_padded=4)
    assert extra is None
    return out


def _same(a, b):
    assert a.keys() == b.keys()
    for name in a:
        for f in ("input_positions", "block_tables", "seq_lens", "query_start_loc", "request_distribution",
                  "mamba_state_indices", "mamba_slot_read_offsets", "mamba_ckpt_indices", "mamba_request_distribution"):
            x, y = getattr(a[name], f), getattr(b[name], f)
            assert (x is None) == (y is None), (name, f)
            if x is not None:
                assert torch.equal(x, y), (name, f)
        for f in ("sequence_layout_kind", "sequence_layout_protocol", "sequence_layout_version"):
            assert getattr(a[name], f) == getattr(b[name], f), (name, f)


# ---------------------------------------------------------------- 1. the gate


@pytest.mark.parametrize("unified", [False, True])
def test_gate_flag_off_runs_one_build_per_group(monkeypatch, unified):
    monkeypatch.delenv(FLAG, raising=False)
    runner, groups, calls = _setup(unified)
    _walk(runner)
    assert len(calls) == len(groups)


@pytest.mark.parametrize("unified", [False, True])
def test_gate_flag_on_staged_skips_build_and_matches(monkeypatch, unified):
    monkeypatch.delenv(FLAG, raising=False)
    runner, groups, calls = _setup(unified)
    ref = _walk(runner)
    monkeypatch.setenv(FLAG, "1")
    runner2, _, calls2 = _setup(unified)
    fast = _walk(runner2)
    assert calls2 == [], "the inline walk must not call build()"
    _same(fast, ref)


def test_gate_flag_on_with_position_override_runs_the_walk(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    runner, groups, calls = _setup(False, override=torch.zeros((3, 8), dtype=torch.int32))
    _walk(runner)
    assert len(calls) == len(groups)


def test_gate_flag_on_falls_back_when_inline_cannot_build(monkeypatch):
    # unified layout without the walk program's mamba products: build() derives each group's state, the inline
    # walk declines, and the caller must run the ordinary walk (same metadata as the flag-off walk)
    monkeypatch.delenv(FLAG, raising=False)
    runner, groups, calls = _setup(True, n_mamba=2)
    _decline(runner)
    ref = _walk(runner)
    monkeypatch.setenv(FLAG, "1")
    runner2, groups2, calls2 = _setup(True, n_mamba=2)
    _decline(runner2)
    calls2.clear()
    out = _walk(runner2)
    assert len(calls2) == len(groups2), "the fallback must run the ordinary walk"
    _same(out, ref)


def test_inline_returns_none_without_a_staged_table(monkeypatch):
    # a group whose table was not staged needs build()'s own upload: the inline walk declines, the caller walks
    monkeypatch.setenv(FLAG, "1")
    runner, groups, calls = _setup(False)
    ctx = runner._attn_metadata_builder_ctx
    _am.stage_block_table_uploads(runner, ctx, 4)
    del ctx.staged_block_tables[1]
    assert TPUModelRunner._build_attention_metadata_staged(runner, ctx) is None


# ---------------------------------------------------------------- 2. the plan follows attn_groups


def test_group_plan_is_rebuilt_when_attn_groups_change(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    runner, groups, calls = _setup(False)
    first = _walk(runner)
    assert set(first) == {f"layer.{i}.{k}" for i in range(3) for k in "ab"}
    # a different group list (new object, new layer names) must not reuse the cached plan
    new_groups = H._make_attn_groups(runner, [g[0].metadata_builders[0] for g in groups])
    for i, (group,) in enumerate(new_groups):
        group.layer_names = [f"other.{i}"]
    runner._attn_metadata_builder_ctx.staged_block_tables = None
    second = _walk(runner)
    assert set(second) == {"other.0", "other.1", "other.2"}


# ---------------------------------------------------------------- 3. switch off = 648f9cdb1


def _reads_flag(node):
    return any(isinstance(n, ast.Attribute) and n.attr == FLAG for n in ast.walk(node))


class _Off(ast.NodeTransformer):
    # log-only additions (q38 DEC-FB-3 engagement counter) are module-level and dropped with the switch off
    NEW_TOP = {"_count_fast_metadata", "_FAST_METADATA_LOG_EVERY"}

    def visit_Module(self, node):
        self.generic_visit(node)
        node.body = [n for n in node.body if not (
            (isinstance(n, ast.FunctionDef) and n.name in self.NEW_TOP)
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in self.NEW_TOP for t in n.targets)))]
        return node

    def visit_If(self, node):
        self.generic_visit(node)
        return (node.orelse or None) if _reads_flag(node.test) else node

    def visit_ClassDef(self, node):
        self.generic_visit(node)
        if node.name == "TPUModelRunner":
            # the plan line lives inside the new method, which is dropped with it
            node.body = [n for n in node.body
                         if not (isinstance(n, ast.FunctionDef) and n.name == "_build_attention_metadata_staged")]
        return node


def test_switch_off_tpu_runner_is_648f9cdb1():
    new = _Off().visit(copy.deepcopy(ast.parse(pathlib.Path(tpu_runner.__file__).read_text())))
    ref = ast.parse((HERE / "tpu_runner_ref_648f9cdb1.py.txt").read_text())
    a, b = [ast.dump(n) for n in new.body], [ast.dump(n) for n in ref.body]
    assert len(a) == len(b)
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    assert not diff, [getattr(new.body[i], "name", i) for i in diff]


# ---------------------------------------------------------------- 5. engagement lines (log only)


import contextlib
import logging


@contextlib.contextmanager
def _lines():
    """INFO lines from tpu_runner's logger (not caplog: vLLM's logging config may stop propagation)."""
    msgs = []
    handler = logging.Handler(logging.INFO)
    handler.emit = lambda record: msgs.append(record.getMessage())
    log = logging.getLogger(tpu_runner.logger.name)
    old_level, old_disable = log.level, logging.root.manager.disable
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    logging.disable(logging.NOTSET)
    try:
        yield msgs
    finally:
        log.removeHandler(handler)
        log.setLevel(old_level)
        logging.disable(old_disable)


def test_plan_line_once_per_plan(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    runner, groups, calls = _setup(True, n_mamba=2)
    with _lines() as msgs:
        _walk(runner)
        runner._attn_metadata_builder_ctx.staged_block_tables = None
        _walk(runner)
    plan = [m for m in msgs if m.startswith("FAST_METADATA_BUILD: inline metadata walk")]
    assert plan == ["FAST_METADATA_BUILD: inline metadata walk over 3 groups (2 mamba)"], plan


def test_counter_counts_inline_and_fallback(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setattr(tpu_runner, "_FAST_METADATA_LOG_EVERY", 2)
    runner, _, _ = _setup(False)
    with _lines() as msgs:
        _walk(runner)                                   # inline
        runner._attn_metadata_builder_ctx.staged_block_tables = None
        _walk(runner)                                   # inline -> line at 2
        runner._attn_metadata_builder_ctx.position_ids_override = torch.zeros((3, 8), dtype=torch.int32)
        runner._attn_metadata_builder_ctx.staged_block_tables = None
        _walk(runner)                                   # override: fallback
        runner._attn_metadata_builder_ctx.staged_block_tables = None
        _walk(runner)                                   # fallback -> line at 4
    lines = [m for m in msgs if m.startswith("FAST_METADATA_BUILD: inline=")]
    assert lines == ["FAST_METADATA_BUILD: inline=2 fallback=0 steps=2",
                     "FAST_METADATA_BUILD: inline=2 fallback=2 steps=4"], lines


def test_counter_counts_a_declined_inline_walk_as_fallback(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setattr(tpu_runner, "_FAST_METADATA_LOG_EVERY", 1)
    runner, _, _ = _setup(True, n_mamba=2)
    _decline(runner)                                    # unified, no mamba products: the inline walk declines
    with _lines() as msgs:
        _walk(runner)
    assert [m for m in msgs if m.startswith("FAST_METADATA_BUILD: inline=")] == [
        "FAST_METADATA_BUILD: inline=0 fallback=1 steps=1"]


def test_no_counter_with_the_flag_off(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    monkeypatch.setattr(tpu_runner, "_FAST_METADATA_LOG_EVERY", 1)
    runner, _, _ = _setup(False)
    with _lines() as msgs:
        _walk(runner)
    assert not [m for m in msgs if m.startswith("FAST_METADATA_BUILD")]
    assert "_fast_metadata_counts" not in runner.__dict__


# ---------------------------------------------------------------- 4. default


def test_flag_defaults_off(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert getattr(envs, FLAG) is False
