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
"""The batch-1 host trims (``runner.host_trims``): the rule that turns them
on, the fast-call guard memo, the AOT guard skip for draft programs, the
connector predicate cache, the connector stats cadence, the DP-size cache and
the process-wide ``tpu_runner`` settings. CPU only: compiled functions use
the eager backend.
"""

from __future__ import annotations

import builtins
import contextlib
import gc
import sys
import types
import weakref

import pytest
import torch

from vllm_torchtpu import utils as vu
from vllm_torchtpu.compilation import decorators
from vllm_torchtpu.compilation import fast_dispatch as fd
from vllm_torchtpu.distributed.kv_transfer import tpu_connector as tc
from vllm_torchtpu.runner import fast_forward_context as ffc
from vllm_torchtpu.runner import host_trims, runner_options


def _fn_scale(x, n):
    return x * n + 1


def _fn_shape(x):
    # dynamic=False bakes the size into the graph: a wrong cache entry gives
    # a wrong value even on the eager backend.
    return x + x.shape[0]


def _wrapped(fn):
    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="eager", fullgraph=True, dynamic=False)
    return fd.wrap(compiled, fn)


@pytest.fixture
def memo_on(monkeypatch):
    # The memo sits on the direct dispatch, which tpu_compile.fast_compiled_call
    # arms; both are off by default.
    monkeypatch.setitem(fd._CONFIG, "fast", True)
    monkeypatch.setitem(fd._MEMO, "on", True)
    for k in fd._MEMO_STATS:
        monkeypatch.setitem(fd._MEMO_STATS, k, 0)
    yield


def test_memo_key_covers_metadata_scalars_and_identity():
    a, b = torch.zeros(4, 8), torch.zeros(4, 8)
    obj = object()
    k1 = fd._memo_key((a, 3, None, "s", obj, [b, 2]))
    assert k1 == fd._memo_key((torch.ones(4, 8), 3, None, "s", obj, [b, 2]))
    # Tensor values are not part of the key; metadata is.
    assert k1 != fd._memo_key((torch.zeros(4, 9), 3, None, "s", obj, [b, 2]))
    assert k1 != fd._memo_key((torch.zeros(8, 4).t(), 3, None, "s", obj, [b, 2]))
    assert k1 != fd._memo_key((a.to(torch.bfloat16), 3, None, "s", obj, [b, 2]))
    assert k1 != fd._memo_key(
        (torch.zeros(4, 8, requires_grad=True), 3, None, "s", obj, [b, 2])
    )
    assert k1 != fd._memo_key((a, 3, None, "s", object(), [b, 2]))
    assert k1 != fd._memo_key((a, 3, None, "s", obj, [b, 3]))
    assert k1 != fd._memo_key((a, True, None, "s", obj, [b, 2]))  # bool vs int
    assert fd._memo_key(([0] * (fd._MEMO_MAX_SEQ + 1),)) is None


class _SubTensor(torch.Tensor):
    pass


def _fn_type(x):
    return x * 2 + (100 if type(x) is _SubTensor else 0)


def test_memo_key_tells_tensor_types_apart():
    # Dynamo's tensor guard checks the exact type: a subclass and an
    # nn.Parameter each get their own cache entry, so they need their own key.
    t = torch.zeros(4)
    assert fd._memo_key((t,)) != fd._memo_key((t.as_subclass(_SubTensor),))
    param = torch.nn.Parameter(torch.zeros(4), requires_grad=False)
    assert fd._memo_key((t,)) != fd._memo_key((param,))


def test_memo_never_hits_across_tensor_types(memo_on):
    w = _wrapped(_fn_type)
    for _ in range(3):
        x, sub = torch.zeros(4), torch.zeros(4).as_subclass(_SubTensor)
        torch.testing.assert_close(w(x), x * 2)
        torch.testing.assert_close(w(sub).as_subclass(torch.Tensor), x * 2 + 100)
    assert fd._MEMO_STATS["hit"] >= 2


def test_memo_key_tells_aliased_tensor_arguments_apart():
    # Dynamo guards whether two tensor inputs are the same tensor, so the
    # same tensor twice and two equal-metadata tensors need different keys.
    a, b = torch.zeros(4), torch.zeros(4)
    assert fd._memo_key((a, a)) != fd._memo_key((a, b))
    assert fd._memo_key((a, [a])) != fd._memo_key((a, [b]))
    assert fd._memo_key((a, b)) == fd._memo_key((b, a))


class _Owner:
    """An argument object Dynamo guards on through its attributes."""

    def __init__(self, n):
        self.n = n


def _fn_owner(x, owner):
    return x * owner.n


def test_memo_never_hits_for_a_recycled_object_id(memo_on, monkeypatch):
    # A memo keyed by a freed object's id must not dispatch a later object
    # that reuses the id to the freed one's cache entry (whose guard on
    # `owner.n` it never re-checks). Whether CPython hands out the freed
    # address again is up to its allocator, so the reuse is simulated: the
    # memo sees the new object under the first one's id.
    w = _wrapped(_fn_owner)
    x = torch.arange(4.0)
    first = _Owner(2)
    for _ in range(3):
        torch.testing.assert_close(w(x, first), x * 2)
    assert fd._MEMO_STATS["store"] == 1
    (key,) = w._fast_dispatch_state["memo"]
    first_id = id(first)
    del first
    gc.collect()
    second = _Owner(5)
    real_id = builtins.id
    monkeypatch.setattr(
        fd, "id", lambda o: first_id if o is second else real_id(o), raising=False
    )
    assert fd._memo_key((x, second)) == key
    hits = fd._MEMO_STATS["hit"]
    torch.testing.assert_close(w(x, second), x * 5)
    assert fd._MEMO_STATS["hit"] == hits


class _Slotted:
    __slots__ = ("n",)


def test_memo_holds_objects_without_weak_references_alive():
    # An object that takes no weak reference is held strongly, so its id
    # cannot be reused while the entry exists.
    obj = _Slotted()
    with pytest.raises(TypeError):
        weakref.ref(obj)
    (ref,) = fd._memo_refs([obj])
    assert ref() is obj
    assert fd._memo_live((ref,), [obj])
    assert not fd._memo_live((ref,), [_Slotted()])


class _SlottedOwner:
    """An argument object that takes no weak reference."""

    __slots__ = ("n",)

    def __init__(self, n):
        self.n = n


def test_a_full_memo_still_stores_new_entries(memo_on):
    # Entries keyed by objects without weak references hold them strongly and
    # never die, so a call site handed a new such object every step fills the
    # memo. The oldest entries then make room, so later keys still hit.
    w = _wrapped(_fn_owner)
    x = torch.arange(4.0)
    for _ in range(fd._MEMO_MAX_ENTRIES + 6):
        torch.testing.assert_close(w(x, _SlottedOwner(3)), x * 3)
    assert len(w._fast_dispatch_state["memo"]) <= fd._MEMO_MAX_ENTRIES
    steady = _SlottedOwner(3)
    torch.testing.assert_close(w(x, steady), x * 3)
    torch.testing.assert_close(w(x, steady), x * 3)
    hits = fd._MEMO_STATS["hit"]
    torch.testing.assert_close(w(x, steady), x * 3)
    assert fd._MEMO_STATS["hit"] == hits + 1


def test_memo_default_off_leaves_dispatch_unchanged(monkeypatch):
    monkeypatch.setitem(fd._CONFIG, "fast", True)
    monkeypatch.setitem(fd._MEMO, "on", False)
    w = _wrapped(_fn_scale)
    x = torch.arange(6.0)
    for _ in range(4):
        torch.testing.assert_close(w(x, 2), _fn_scale(x, 2))
    assert w._fast_dispatch_state["memo"] == {}


def test_memo_hits_and_results_match_eager(memo_on):
    w = _wrapped(_fn_scale)
    xs = [torch.arange(6.0), torch.arange(6.0) + 3, torch.arange(10.0)]
    for _ in range(3):
        for x in xs:
            for n in (2, 5):
                torch.testing.assert_close(w(x, n), _fn_scale(x, n))
    assert fd._MEMO_STATS["hit"] > 0
    assert fd._MEMO_STATS["store"] >= 2
    # Same shape and n -> one memo entry serves both tensor values.
    keys = w._fast_dispatch_state["memo"]
    assert all(len(k) == 2 for k in keys)


def test_memo_respects_scalar_value_and_shape(memo_on):
    w = _wrapped(_fn_scale)
    x6, x10 = torch.arange(6.0), torch.arange(10.0)
    for _ in range(2):
        torch.testing.assert_close(w(x6, 2), _fn_scale(x6, 2))
        torch.testing.assert_close(w(x6, 7), _fn_scale(x6, 7))
        torch.testing.assert_close(w(x10, 2), _fn_scale(x10, 2))
    # After warm-up every repeat is a memo hit, and still right.
    hits = fd._MEMO_STATS["hit"]
    for n in (2, 7):
        torch.testing.assert_close(w(x6, n), _fn_scale(x6, n))
    torch.testing.assert_close(w(x10, 2), _fn_scale(x10, 2))
    assert fd._MEMO_STATS["hit"] == hits + 3


def test_memo_never_reuses_an_entry_across_shapes(memo_on):
    w = _wrapped(_fn_shape)
    x6, x10 = torch.zeros(6), torch.zeros(10)
    for _ in range(3):
        torch.testing.assert_close(w(x6), x6 + 6)
        torch.testing.assert_close(w(x10), x10 + 10)
    assert fd._MEMO_STATS["hit"] >= 2


def test_memo_cleared_when_cache_entries_change(memo_on):
    w = _wrapped(_fn_scale)
    x = torch.arange(6.0)
    w(x, 2)
    w(x, 2)
    w(x, 2)
    st = w._fast_dispatch_state
    assert st["memo"], "warm signature should be memoized"
    count_before = st["memo_count"]
    # A new specialization recompiles: entry count changes, the memo is reset.
    w(torch.arange(12.0), 3)
    assert st["count"] != count_before
    assert st["memo"] == {}
    torch.testing.assert_close(w(x, 2), _fn_scale(x, 2))


# --- draft forwards that read the forward context stay out of the memo -------

_DRAFT_LAYER = "draft.model.layers.0.self_attn.attn"
_VOCAB, _HIDDEN, _BLOCK = 32, 8, 4


class _ContextAttention(torch.nn.Module):
    """Reads its block table from vLLM's forward context, as vLLM's attention
    layer does. Its output depends on the table's static length, as that of a
    paged-attention program built for the length does."""

    def forward(self, h):
        from vllm.forward_context import get_forward_context

        table = get_forward_context().attn_metadata[_DRAFT_LAYER].block_tables
        return h + table.shape[0] + 0 * table[:1].to(h.dtype)


class _ContextBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(_VOCAB, _HIDDEN)
        self.attn = _ContextAttention()

    def forward(self, input_ids, positions):
        return self.attn(self.embed(input_ids))


class _ContextDraft(torch.nn.Module):
    """What the DFlash and DSpark draft forwards call on their draft model."""

    def __init__(self):
        super().__init__()
        self.model = _ContextBackbone()
        self.lm_head = torch.nn.Linear(_HIDDEN, _VOCAB, bias=False)

    def compute_logits(self, hidden):
        return self.lm_head(hidden)

    def compute_draft_logits(self, hidden):
        return self.lm_head(hidden)

    def markov_embed(self, token_ids):
        return self.model.embed(token_ids)

    def markov_bias(self, markov_embed):
        return 0.1 * self.lm_head(markov_embed)

    def map_draft_to_target(self, draft_ids):
        return draft_ids


def _forward_context_config():
    return types.SimpleNamespace(
        parallel_config=types.SimpleNamespace(
            data_parallel_size=1,
            data_parallel_rank=0,
            use_sequence_parallel_moe=False,
            is_moe_model=False,
        ),
        compilation_config=types.SimpleNamespace(
            static_forward_context={},
            fast_moe_cold_start=False,
            static_all_moe_layers=None,
        ),
    )


@pytest.mark.parametrize("which", ["dflash", "dspark"])
def test_draft_forward_runs_the_program_for_the_current_block_table(
    memo_on, monkeypatch, which
):
    # max_model_len 4096 with VLLM_TPU_MOST_MODEL_LEN 1024 and 64-token
    # blocks: a one-request draft step's block table has 64 blocks when the
    # step uses max_model_len and 16 when it does not, and the draft forward's
    # arguments are the same either way. Precompile compiles both programs,
    # serving then switches between them; a memo hit would run the program
    # compiled for the other length.
    from vllm.forward_context import set_forward_context

    from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
    from vllm_torchtpu.spec_decode.dflash import DFlashProposer
    from vllm_torchtpu.spec_decode.dspark import DSparkProposer

    cls = DFlashProposer if which == "dflash" else DSparkProposer
    decorated = cls._dflash_forward_and_sample
    real_compile = torch.compile

    def eager_backend(model=None, **kwargs):
        return real_compile(model, **{**kwargs, "backend": "eager"})

    monkeypatch.setattr(torch, "compile", eager_backend)
    # The class's own tpu_compile setting, on the eager backend. The entries
    # it compiles share the method's code object, so they are dropped after.
    torch._dynamo.reset()
    forward = decorators.tpu_compile(
        guard_memo=decorated._fast_dispatch_state["guard_memo"]
    )(decorated.__wrapped__)
    proposer = cls.__new__(cls)
    proposer.draft_model = _ContextDraft()
    proposer.use_confidence_head = False
    proposer.sample_from_anchor = True
    positions = torch.arange(_BLOCK, dtype=torch.int32)
    try:
        for blocks in (64, 16, 16, 16, 64, 64, 16):
            md = AttentionMetadata(
                input_positions=positions,
                block_tables=torch.zeros(blocks, dtype=torch.int32),
            )
            input_ids = torch.tensor([blocks % _VOCAB, 1, 2, 3])
            with set_forward_context({_DRAFT_LAYER: md}, _forward_context_config()):
                got = forward(proposer, input_ids, positions, _BLOCK)
                want = decorated.__wrapped__(proposer, input_ids, positions, _BLOCK)
            torch.testing.assert_close(got, want)
        assert fd._MEMO_STATS["hit"] == 0
        assert forward._fast_dispatch_state["memo"] == {}
    finally:
        torch._dynamo.reset()


# --- DFlash's context-KV update keeps its metadata arguments out of the memo -


class _KVImpl:
    """Writes the static length of the block table its program was traced
    with, as a paged-attention program built for that length writes its
    pages."""

    def forward(self, layer, query, key, value, kv_cache, attn_metadata):
        table = attn_metadata.block_tables
        kv_cache.copy_(torch.full_like(kv_cache, float(table.shape[0])) + 0 * key.sum())
        return query


class _KVRotary(torch.nn.Module):
    def forward(self, positions, query, key):
        return query, key


class _KVSelfAttn(torch.nn.Module):
    head_dim, num_kv_heads, num_heads = 4, 1, 2

    def __init__(self):
        super().__init__()
        self.rotary_emb = _KVRotary()
        self.attn = torch.nn.Module()
        self.attn.kv_cache = torch.zeros(4)
        self.attn.impl = _KVImpl()


class _KVLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _KVSelfAttn()


class _KVDraft(torch.nn.Module):
    """What DFlash's context-KV update reads from a draft model that has no
    context-KV hook of its own."""

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_KVLayer()])
        self.model._fused_kv_weight = torch.nn.Parameter(
            torch.ones(2 * 4, _HIDDEN), requires_grad=False
        )
        self.model._fused_kv_bias = None


def test_context_kv_update_runs_the_program_for_the_current_metadata(
    memo_on, monkeypatch
):
    # The update takes its attention metadata as arguments, which the memo
    # keys by identity, and its program depends on the block-table width
    # inside them. With the method out of the memo, swapping block_tables on
    # one metadata object between calls still runs the program for the new
    # width; a memo hit would run the 64-block program for the 16-block calls.
    from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
    from vllm_torchtpu.spec_decode.dflash import DFlashProposer

    decorated = DFlashProposer._tpu_precompute_and_update_kv_cache
    real_compile = torch.compile

    def eager_backend(model=None, **kwargs):
        return real_compile(model, **{**kwargs, "backend": "eager"})

    monkeypatch.setattr(torch, "compile", eager_backend)
    # The class's own tpu_compile setting, on the eager backend; its entries
    # share the method's code object, so they are dropped after.
    torch._dynamo.reset()
    update = decorators.tpu_compile(
        guard_memo=decorated._fast_dispatch_state["guard_memo"]
    )(decorated.__wrapped__)
    proposer = DFlashProposer.__new__(DFlashProposer)
    proposer.draft_model = _KVDraft()
    kv_cache = proposer.draft_model.model.layers[0].self_attn.attn.kv_cache
    hidden = torch.ones(_BLOCK, _HIDDEN)
    positions = torch.arange(_BLOCK, dtype=torch.int32)
    md = AttentionMetadata(input_positions=positions)
    try:
        for blocks in (64, 64, 16, 16, 64):
            md.block_tables = torch.zeros(blocks, dtype=torch.int32)
            update(proposer, hidden, positions, (md,))
            assert kv_cache[0].item() == blocks
        assert fd._MEMO_STATS["hit"] == 0
        assert update._fast_dispatch_state["memo"] == {}
    finally:
        torch._dynamo.reset()


class _AOTFn:
    def __init__(self):
        self._guard_check_enabled = True

    def disable_guard_check(self):
        self._guard_check_enabled = False


def _draft_config(evaluate_guards):
    # Built through vLLM's own constructors, so a renamed field fails here:
    # they refuse an unknown keyword, while assigning an unknown attribute to
    # an instance would just add it.
    from vllm.config.compilation import CompilationConfig, DynamicShapesConfig

    cc = CompilationConfig(
        dynamic_shapes_config=DynamicShapesConfig(evaluate_guards=evaluate_guards)
    )
    return types.SimpleNamespace(compilation_config=cc)


def _compiled_module():
    return types.SimpleNamespace(aot_compiled_fn=_AOTFn())


def test_aot_skip_guards_matches_vllm_load_policy(monkeypatch):
    host_trims._reset_for_tests()
    monkeypatch.setattr(host_trims, "AOT_SKIP_GUARDS", True)
    m = _compiled_module()
    assert host_trims.maybe_skip_aot_guards(m, _draft_config(False)) is True
    assert m.aot_compiled_fn._guard_check_enabled is False
    # Once per module.
    assert host_trims.maybe_skip_aot_guards(m, _draft_config(False)) is False
    keep = _compiled_module()
    assert host_trims.maybe_skip_aot_guards(keep, _draft_config(True)) is False
    assert keep.aot_compiled_fn._guard_check_enabled is True


def test_aot_skip_guards_fails_on_a_config_without_evaluate_guards(monkeypatch):
    # A renamed evaluate_guards must fail, not read as "keep the guards".
    host_trims._reset_for_tests()
    monkeypatch.setattr(host_trims, "AOT_SKIP_GUARDS", True)
    renamed = types.SimpleNamespace(
        compilation_config=types.SimpleNamespace(
            dynamic_shapes_config=types.SimpleNamespace()
        )
    )
    m = _compiled_module()
    with pytest.raises(AttributeError, match="evaluate_guards"):
        host_trims.maybe_skip_aot_guards(m, renamed)
    assert m.aot_compiled_fn._guard_check_enabled is True


def test_aot_skip_guards_off_by_default(monkeypatch):
    host_trims._reset_for_tests()
    monkeypatch.setattr(host_trims, "AOT_SKIP_GUARDS", False)
    m = _compiled_module()
    assert host_trims.maybe_skip_aot_guards(m, _draft_config(False)) is False
    assert m.aot_compiled_fn._guard_check_enabled is True


def test_aot_skip_guards_waits_for_compiled_fn(monkeypatch):
    host_trims._reset_for_tests()
    monkeypatch.setattr(host_trims, "AOT_SKIP_GUARDS", True)
    m = types.SimpleNamespace()
    cfg = _draft_config(False)
    assert host_trims.maybe_skip_aot_guards(m, cfg) is False  # not compiled yet
    m.aot_compiled_fn = _AOTFn()
    assert host_trims.maybe_skip_aot_guards(m, cfg) is True


def test_connector_env_cache(monkeypatch):
    monkeypatch.setattr(tc, "_env_predicate_cache", {})
    monkeypatch.setenv("TPU_KV_RESHARD_TRANSPORT", "raiden")
    monkeypatch.setattr(host_trims, "ENABLED", False)
    assert tc._use_raiden_stage3_transport() is True
    monkeypatch.setenv("TPU_KV_RESHARD_TRANSPORT", "zmq")
    assert tc._use_raiden_stage3_transport() is False  # off: follows env
    monkeypatch.setattr(host_trims, "ENABLED", True)
    monkeypatch.setenv("TPU_KV_RESHARD_TRANSPORT", "raiden")
    assert tc._use_raiden_stage3_transport() is True
    monkeypatch.setenv("TPU_KV_RESHARD_TRANSPORT", "zmq")
    assert tc._use_raiden_stage3_transport() is True  # on: first answer kept
    glm = tc._use_raiden_glm_admission()
    assert tc._use_raiden_glm_admission() is glm
    assert set(tc._env_predicate_cache) == {"stage3", "glm"}


# --- KV connector stats cadence ----------------------------------------------


def _get_finished(monkeypatch, every, samples=((), ())):
    from vllm_torchtpu.runner import tpu_runner as tr

    calls = {"stats": 0}

    class Conn:
        def get_transfer_results(self, finished):
            return types.SimpleNamespace(
                finished_sending=set(), finished_recving=set(), failed_recving=set()
            )

        def get_block_ids_with_load_errors(self):
            return set()

        def build_connector_worker_meta(self):
            return None

        def get_kv_connector_stats(self):
            calls["stats"] += 1
            return {"n": calls["stats"]}

        def clear_connector_metadata(self):
            pass

    conn = Conn()
    monkeypatch.setattr(tr, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(tr, "get_kv_transfer_group", lambda: conn)
    monkeypatch.setattr(host_trims, "CONN_STATS_EVERY", every)
    runner = types.SimpleNamespace(
        _stage_timing=types.SimpleNamespace(take_samples=lambda: samples),
        kv_cache_config=types.SimpleNamespace(kv_cache_groups=[1]),
        _conn_stats_step=0,
    )
    so = types.SimpleNamespace(finished_req_ids=set())
    outs = [
        tr.TPUModelRunner.get_finished_kv_transfers(runner, so)[6] for _ in range(8)
    ]
    return outs, calls["stats"]


def test_conn_stats_every_step_by_default(monkeypatch):
    outs, n = _get_finished(monkeypatch, 1)
    assert n == 8 and all(o is not None for o in outs)


def test_conn_stats_throttled(monkeypatch):
    outs, n = _get_finished(monkeypatch, 4)
    assert n == 2
    assert [o is not None for o in outs] == [False, False, False, True] * 2


def test_conn_stats_polled_when_runner_timings_pending(monkeypatch):
    outs, n = _get_finished(monkeypatch, 4, samples=([1.0], []))
    assert n == 8 and all(o is not None for o in outs)


# --- process-wide tpu_runner settings ------------------------------------------


def test_switch_interval_applied_from_the_first_step(monkeypatch):
    orig = sys.getswitchinterval()
    monkeypatch.setattr(host_trims, "SWITCH_US", 200)
    monkeypatch.setattr(host_trims, "GC_FREEZE", False)
    monkeypatch.setattr(
        host_trims, "_hook", {"n": 0, "applied": False, "frozen": False}
    )
    try:
        host_trims.on_step()
        assert abs(sys.getswitchinterval() - 200e-6) < 1e-9
        host_trims.on_step()
        assert abs(sys.getswitchinterval() - 200e-6) < 1e-9
    finally:
        sys.setswitchinterval(orig)


def test_gc_freeze_after_n_steps(monkeypatch):
    orig = gc.get_threshold()
    monkeypatch.setattr(host_trims, "SWITCH_US", 0)
    monkeypatch.setattr(host_trims, "GC_FREEZE", True)
    monkeypatch.setattr(host_trims, "GC_FREEZE_AFTER_STEPS", 3)
    monkeypatch.setattr(host_trims, "GC_FREEZE_GEN0_THRESHOLD", 12345)
    monkeypatch.setattr(
        host_trims, "_hook", {"n": 0, "applied": False, "frozen": False}
    )
    try:
        host_trims.on_step()
        host_trims.on_step()
        assert gc.get_threshold()[0] == orig[0] and not host_trims._hook["frozen"]
        host_trims.on_step()
        assert host_trims._hook["frozen"] and gc.get_threshold()[0] == 12345
        assert gc.get_freeze_count() > 0
    finally:
        gc.unfreeze()
        gc.set_threshold(*orig)


def test_dp_size_cache(monkeypatch):
    pc1 = types.SimpleNamespace(data_parallel_size=4)
    pc2 = types.SimpleNamespace(data_parallel_size=2)
    monkeypatch.delenv("TORCH_TPU_DP_SIZE", raising=False)
    monkeypatch.setattr(vu, "_dp_size_cache_on", False)
    monkeypatch.setattr(vu, "_dp_size_config", None)
    monkeypatch.setattr(vu, "_dp_size_value", 0)
    assert vu.get_dp_size(pc1) == 4
    vu.cache_dp_size(True)
    assert vu.get_dp_size(pc1) == 4
    pc1.data_parallel_size = 8  # cached for the same config object
    assert vu.get_dp_size(pc1) == 4
    assert vu.get_dp_size(pc2) == 2  # another config object refreshes
    monkeypatch.setenv("TORCH_TPU_DP_SIZE", "3")
    assert vu.get_dp_size(types.SimpleNamespace(data_parallel_size=1)) == 3


# --- the batch-1 rule ------------------------------------------------------------


def _config(max_num_seqs, kv_role=None, runner=None):
    kv = None
    if kv_role is not None:
        kv = types.SimpleNamespace(
            is_kv_producer=kv_role in ("kv_producer", "kv_both"),
            is_kv_consumer=kv_role in ("kv_consumer", "kv_both"),
        )
    additional = {} if runner is None else {"tpu_runner": runner}
    return types.SimpleNamespace(
        scheduler_config=types.SimpleNamespace(max_num_seqs=max_num_seqs),
        kv_transfer_config=kv,
        additional_config=additional,
    )


@pytest.fixture
def restore_trims():
    yield
    host_trims.configure(_config(256))
    host_trims._configured = False


@pytest.mark.parametrize(
    "max_num_seqs, kv_role, expected",
    [
        (1, None, True),
        (1, "kv_consumer", True),
        (1, "kv_both", True),
        (1, "kv_producer", False),
        (2, None, False),
        (2, "kv_consumer", False),
        (256, None, False),
    ],
)
def test_rule_is_batch1_decode(max_num_seqs, kv_role, expected):
    assert host_trims.rule(_config(max_num_seqs, kv_role)) is expected


def test_rule_reads_vllms_config_fields():
    # vLLM's own config objects, so a renamed field fails here.
    from vllm.config import KVTransferConfig, SchedulerConfig

    def cfg(kv_role):
        return types.SimpleNamespace(
            scheduler_config=SchedulerConfig(
                max_num_seqs=1, max_model_len=4096, is_encoder_decoder=False
            ),
            kv_transfer_config=KVTransferConfig(
                kv_connector="TPURaidenConnector", kv_role=kv_role
            ),
        )

    assert host_trims.rule(cfg("kv_consumer")) is True
    assert host_trims.rule(cfg("kv_both")) is True
    assert host_trims.rule(cfg("kv_producer")) is False


def test_rule_fails_on_a_config_without_its_fields():
    # A KV config without is_kv_producer (say, renamed upstream) must not
    # read as "not a producer" and turn the trims on for a prefill engine.
    renamed = _config(1)
    renamed.kv_transfer_config = types.SimpleNamespace(is_kv_consumer=False)
    with pytest.raises(AttributeError, match="is_kv_producer"):
        host_trims.rule(renamed)
    with pytest.raises(AttributeError, match="scheduler_config"):
        host_trims.rule(types.SimpleNamespace(kv_transfer_config=None))


def test_configure_turns_every_trim_on_and_off(restore_trims):
    host_trims.configure(_config(1, "kv_consumer"))
    assert host_trims.ENABLED and host_trims.AOT_SKIP_GUARDS
    assert host_trims.CONN_STATS_EVERY == host_trims.CONN_STATS_EVERY_ON == 16
    assert ffc.enabled() and fd._MEMO["on"] and vu._dp_size_cache_on
    host_trims.configure(_config(2, "kv_consumer"))
    assert not host_trims.ENABLED and not host_trims.AOT_SKIP_GUARDS
    assert host_trims.CONN_STATS_EVERY == 1
    assert not ffc.enabled() and not fd._MEMO["on"] and not vu._dp_size_cache_on


def test_old_environment_switches_are_ignored(monkeypatch, restore_trims):
    for var in (
        "VLLM_TPU_T3_HOST",
        "VLLM_TPU_T3_GUARD_MEMO",
        "VLLM_TPU_T3_AOT_SKIP_GUARDS",
        "VLLM_TPU_T3_ENV_CACHE",
        "VLLM_TPU_T3_CONN_STATS",
        "VLLM_TPU_T3_DP_CACHE",
        "VLLM_TPU_FAST_FORWARD_CONTEXT",
    ):
        monkeypatch.setenv(var, "1")
    monkeypatch.setenv("VLLM_TPU_T3_SWITCH_US", "200")
    monkeypatch.setenv("VLLM_TPU_T3_GC", "freeze")
    host_trims.configure(_config(4))
    assert not host_trims.ENABLED and not ffc.enabled() and not fd._MEMO["on"]
    assert host_trims.CONN_STATS_EVERY == 1 and not host_trims.STEP_HOOK


def test_process_settings_come_from_the_tpu_runner_section(restore_trims):
    host_trims.configure(_config(1))
    assert host_trims.SWITCH_US == 0 and not host_trims.GC_FREEZE
    assert not host_trims.STEP_HOOK
    host_trims.configure(_config(4, runner={"switch_interval_us": 200}))
    assert host_trims.SWITCH_US == 200 and host_trims.STEP_HOOK
    assert not host_trims.ENABLED  # the settings do not depend on the rule
    host_trims.configure(_config(4, runner={"gc_freeze": True}))
    assert host_trims.GC_FREEZE and host_trims.STEP_HOOK


@pytest.mark.parametrize(
    "raw, match",
    [
        ({"switch_interval_us": 200.0}, "must be a int"),
        ({"switch_interval_us": True}, "must be a int"),
        ({"switch_interval_us": -1}, ">= 0"),
        ({"gc_freeze": 1}, "must be a bool"),
    ],
)
def test_process_settings_parse_strictly(raw, match):
    with pytest.raises(ValueError, match=match):
        runner_options.parse(raw)
    # The engine reads the section through get(), which must refuse it too.
    with pytest.raises(ValueError, match=match):
        runner_options.get(types.SimpleNamespace(additional_config={"tpu_runner": raw}))


def test_scheduler_step_configures_its_process_once(monkeypatch, restore_trims):
    calls = []
    monkeypatch.setattr(host_trims, "_configured", False)
    real = host_trims.configure
    monkeypatch.setattr(
        host_trims, "configure", lambda cfg: (calls.append(cfg), real(cfg))
    )
    steps = []
    monkeypatch.setattr(host_trims, "on_step", lambda: steps.append(1))
    cfg = _config(4, runner={"switch_interval_us": 200})
    host_trims.on_scheduler_step(cfg)
    host_trims.on_scheduler_step(cfg)
    assert calls == [cfg] and len(steps) == 2


def test_the_runner_applies_the_rule_when_it_is_built():
    # No CPU test can build a TPUModelRunner (its __init__ needs a TPU), so
    # check the parsed __init__: host_trims.configure(self.vllm_config) must
    # be an unconditional statement of its body (not in a branch, a nested
    # function or a comment), so every runner applies the rule.
    import ast
    import inspect
    import textwrap

    from vllm_torchtpu.runner import tpu_runner as tr

    init = ast.parse(textwrap.dedent(inspect.getsource(tr.TPUModelRunner.__init__)))
    body = init.body[0].body
    calls = [
        stmt.value
        for stmt in body
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
    ]
    assert any(
        ast.unparse(call.func) == "host_trims.configure"
        and [ast.unparse(a) for a in call.args] == ["self.vllm_config"]
        for call in calls
    )


# --- The MTP drafter's calls under the rule -----------------------------------


class _Program:
    """A draft program stand-in that records its calls into `events`."""

    def __init__(self, name, events):
        self.name, self.events = name, events

    def __call__(self, *args, **kwargs):
        self.events.append(("run", self.name))
        return torch.zeros(1)


def _drafter_calls(monkeypatch, rule_on, fused):
    """Run Eagle3Proposer's per-step draft forward (or its fused loop) on a
    stand-in proposer; returns the recorded events in order."""
    from vllm_torchtpu.spec_decode import eagle3

    events = []

    def recording(kind):
        def make(attn_metadata, vllm_config, *, num_tokens, num_tokens_across_dp):
            events.append((kind, vllm_config, num_tokens, num_tokens_across_dp))
            return contextlib.nullcontext()

        return make

    monkeypatch.setattr(eagle3, "set_forward_context", recording("stock"))
    monkeypatch.setattr(ffc, "fast_set_forward_context", recording("cached"))
    monkeypatch.setattr(ffc, "enabled", lambda: rule_on)
    monkeypatch.setattr(host_trims, "AOT_SKIP_GUARDS", rule_on)
    monkeypatch.setattr(
        host_trims,
        "maybe_skip_aot_guards",
        lambda module, vllm_config: events.append(
            ("skip_guards", module.name, vllm_config)
        ),
    )
    monkeypatch.setattr(
        eagle3, "set_vllm_model_wrapper_context", lambda **_: contextlib.nullcontext()
    )
    monkeypatch.setattr(eagle3, "set_model_tag", lambda _tag: contextlib.nullcontext())
    draft_config = object()
    seq_lens = torch.zeros(2, dtype=torch.int32)
    proposer = types.SimpleNamespace(
        draft_vllm_config=draft_config,
        runner=types.SimpleNamespace(
            mesh=None,
            max_num_reqs=8,
            _dp_num_tokens_across_dp=lambda n: ("across_dp", n),
        ),
        speculative_config=types.SimpleNamespace(method="mtp"),
        _build_draft_attn_metadata=lambda **_: {
            "layer": types.SimpleNamespace(seq_lens=seq_lens)
        },
        _draft_moe_ctx=contextlib.nullcontext,
        _unwrap_model_out=lambda out: out,
        _draft_md_cache={},
        _loop_bucket=lambda num_reqs: 8,
    )
    input_ids = torch.zeros(3, dtype=torch.int32)
    positions = torch.arange(3, dtype=torch.int32)
    if fused:
        proposer._mtp_fused_loop = _Program("fused_loop", events)
        proposer._mtp_fused_loop_warm = False
        eagle3.Eagle3Proposer._run_mtp_fused_loop(
            proposer, None, input_ids, positions, None, None, None, None, None
        )
    else:
        step = _Program("loop_step", events)
        proposer._mtp_loop_model = step
        proposer._mtp_loop_model_warm = False
        proposer._draft_model_for_step = lambda _idx: step
        eagle3.Eagle3Proposer._forward_draft(
            proposer,
            None,
            input_ids,
            positions,
            None,
            step_idx=1,
            seq_lens_delta=1,
            num_rejected_np=None,
        )
    return events, draft_config


@pytest.mark.parametrize("fused", [False, True], ids=["draft_step", "fused_loop"])
def test_the_drafter_takes_the_cached_context_and_skips_guards_after_the_run(
    monkeypatch, fused
):
    events, draft_config = _drafter_calls(monkeypatch, rule_on=True, fused=fused)
    name = "fused_loop" if fused else "loop_step"
    # The first call traces and compiles, the repeat runs compiled; the guard
    # skip comes only after both, on the program that ran.
    assert events == [
        ("cached", draft_config, 3, ("across_dp", 3)),
        ("run", name),
        ("run", name),
        ("skip_guards", name, draft_config),
    ]


@pytest.mark.parametrize("fused", [False, True], ids=["draft_step", "fused_loop"])
def test_without_the_rule_the_drafter_keeps_vllms_context_and_guards(
    monkeypatch, fused
):
    events, draft_config = _drafter_calls(monkeypatch, rule_on=False, fused=fused)
    name = "fused_loop" if fused else "loop_step"
    assert events == [
        ("stock", draft_config, 3, ("across_dp", 3)),
        ("run", name),
        ("run", name),
    ]
