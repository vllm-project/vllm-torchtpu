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
"""Tests for the out-of-tree TPU VocabParallelEmbedding.

Pins the three halves of the claim: vLLM builds our class wherever a model
asks for ``VocabParallelEmbedding``, its forward returns what upstream's
returns on every TP rank, and, unlike upstream's, its traced graph does not
depend on the rank.
"""

import ast
import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import vllm
import vllm.model_executor.layers.vocab_parallel_embedding as vpe
from vllm.model_executor.custom_op import (maybe_get_oot_by_class,
                                           op_registry_oot)

pytestmark = pytest.mark.cpu_test

_SRC = Path(__file__).resolve().parents[2] / "src/vllm_torchtpu"
_MODULE_PATH = _SRC / "layers/adapter/vocab_parallel_embedding.py"
_LAYERS_INIT_PATH = _SRC / "layers/__init__.py"
_REGISTRY_PATH = _SRC / "patch_registry.py"

_BOUNDS = ("org_vocab_start_index", "org_vocab_end_index",
           "num_org_vocab_padding", "added_vocab_start_index",
           "added_vocab_end_index")

# 37 original + 5 added tokens, padded to multiples of 8: the original range
# pads to 40 and the whole table to 48. At TP=2 and TP=4 the last rank's
# original shard ends in 3 padding rows, the added shards are ragged, and at
# TP=4 the last rank owns no added token at all.
_ORG_VOCAB = 37
_VOCAB = 42
_PADDING = 8
_DIM = 6
# Marks shard rows that belong to no token; a wrong offset reads one.
_PADDING_ROW = 1e6


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _upstream_objects():
    return (vpe.VocabParallelEmbedding.__dict__["__init__"],
            vpe.VocabParallelEmbedding.__dict__["forward"],
            vpe.get_masked_input_and_mask)


@pytest.fixture(scope="module")
def loaded():
    """Load only the module under test; yields (module, upstream objects
    captured before the load).

    Importing it through the ``vllm_torchtpu.layers`` package would also
    import attention kernels that need TorchTPU. ``register_oot`` asserts a
    name is registered once per process, so a registration left by an earlier
    normal import is set aside for the duration and put back afterwards.
    """
    before = _upstream_objects()
    previous = op_registry_oot.pop("VocabParallelEmbedding", None)
    name = "_test_tpu_vocab_parallel_embedding"
    try:
        yield _load_by_path(name, _MODULE_PATH), before
    finally:
        sys.modules.pop(name, None)
        op_registry_oot.pop("VocabParallelEmbedding", None)
        if previous is not None:
            op_registry_oot["VocabParallelEmbedding"] = previous


def _build(monkeypatch, tp_rank, tp_size, cls=None, **kwargs):
    """Build a layer through upstream's constructor as rank ``tp_rank``.

    The two getters below are the only distributed state the constructor
    reads (vocab_parallel_embedding.py:262-263 at v0.29.0).
    """
    monkeypatch.setattr(vpe, "get_tensor_model_parallel_rank", lambda: tp_rank)
    monkeypatch.setattr(vpe, "get_tensor_model_parallel_world_size",
                        lambda: tp_size)
    cls = cls or vpe.VocabParallelEmbedding
    return cls(_VOCAB,
               _DIM,
               params_dtype=torch.float32,
               org_num_embeddings=_ORG_VOCAB,
               padding_size=_PADDING,
               **kwargs)


def _load_full_weight(layer, full_weight):
    """Write this rank's rows of ``full_weight`` into its shard.

    Shard layout: [original rows | original padding | added rows | added
    padding]. ``weight_loader`` is not used because it only loads the
    original vocabulary.
    """
    idx = layer.shard_indices
    shard = torch.full_like(layer.weight.data, _PADDING_ROW)
    org_ids = slice(idx.org_vocab_start_index, idx.org_vocab_end_index)
    added_ids = slice(idx.added_vocab_start_index, idx.added_vocab_end_index)
    first_added = idx.num_org_elements_padded
    added_rows = slice(first_added, first_added + idx.num_added_elements)
    shard[:idx.num_org_elements] = full_weight[org_ids]
    shard[added_rows] = full_weight[added_ids]
    layer.weight.data.copy_(shard)


def _build_group(monkeypatch, tp_size, full_weight):
    layers = [_build(monkeypatch, rank, tp_size) for rank in range(tp_size)]
    for layer in layers:
        _load_full_weight(layer, full_weight)
        # Upstream would otherwise take the fused CUDA kernel.
        assert not layer.use_fused_embedding
    return layers


def _set_all_reduce(monkeypatch, tpu_mod, all_reduce):
    # Upstream reads all_reduce from vLLM's module globals, ours from ours.
    monkeypatch.setattr(vpe, "tensor_model_parallel_all_reduce", all_reduce)
    monkeypatch.setattr(tpu_mod, "tensor_model_parallel_all_reduce",
                        all_reduce)


def _run_tp_group(monkeypatch, tpu_mod, forward, layers, input_ids):
    """Run ``forward`` on every rank of a simulated TP group.

    A real all-reduce hands every rank the same sum, which takes two passes to
    imitate: capture each rank's pre-reduce output, then replay every rank with
    the all-reduce returning their sum.
    """

    def run_all_ranks(all_reduce):
        _set_all_reduce(monkeypatch, tpu_mod, all_reduce)
        return [forward(layer, input_ids) for layer in layers]

    pre_reduce = []

    def capture(tensor):
        pre_reduce.append(tensor)
        return tensor

    run_all_ranks(capture)
    assert len(pre_reduce) == len(layers)
    reduced = sum(pre_reduce)
    return run_all_ranks(lambda tensor: reduced)


def _use_plain_upstream_mask_helper(monkeypatch):
    """Strip ``@torch.compile`` from upstream's helper, as the TPU platform
    does (``_DYNAMIC_COMPILE_TARGETS`` in platforms/tpu_platform.py)."""
    helper = vpe.get_masked_input_and_mask
    monkeypatch.setattr(vpe, "get_masked_input_and_mask",
                        getattr(helper, "__wrapped__", helper))


def test_registered_under_the_upstream_class_name(loaded, monkeypatch):
    """Loading the module is what makes vLLM build our class."""
    tpu_mod, _ = loaded
    cls = tpu_mod.TpuVocabParallelEmbedding
    assert op_registry_oot["VocabParallelEmbedding"] is cls
    # The lookup LoRA uses (vllm/lora/layers/vocab_parallel_embedding.py:123
    # and vllm/lora/utils.py:269 at v0.29.0).
    assert maybe_get_oot_by_class(vpe.VocabParallelEmbedding) is cls
    # Guard against a vacuous equivalence check below.
    assert cls.forward is not vpe.VocabParallelEmbedding.forward

    # Registration is keyed by class name, so the LM head keeps upstream's
    # class and gets none of the buffers.
    assert maybe_get_oot_by_class(vpe.ParallelLMHead) is vpe.ParallelLMHead
    lm_head = _build(monkeypatch, 1, 2, cls=vpe.ParallelLMHead)
    assert type(lm_head) is vpe.ParallelLMHead
    assert dict(lm_head.named_buffers()) == {}
    with pytest.raises(RuntimeError, match="sampler"):
        lm_head(torch.zeros(1, dtype=torch.int64))


def test_loading_leaves_upstream_untouched(loaded):
    """Nothing in vLLM is reassigned any more, the helper included."""
    _, before = loaded
    for was, now in zip(before, _upstream_objects()):
        assert now is was


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_upstream_constructor_builds_our_class_with_bound_buffers(
        loaded, monkeypatch, tp_size):
    tpu_mod, _ = loaded
    for rank in range(tp_size):
        layer = _build(monkeypatch, rank, tp_size)
        assert type(layer) is tpu_mod.TpuVocabParallelEmbedding

        buffers = dict(layer.named_buffers())
        assert set(buffers) == set(_BOUNDS)
        for name in _BOUNDS:
            assert buffers[name].dtype == torch.int64
            assert buffers[name].shape == ()
            assert buffers[name].item() == getattr(layer.shard_indices, name)
        # Non-persistent: checkpoints neither carry nor expect them.
        assert set(layer.state_dict()) == {"weight"}


def test_buffer_name_collision_with_upstream_fails_loudly(loaded, monkeypatch):
    """If vLLM ever defines one of the names, construction must not silently
    shadow it or fall back to int bounds."""
    monkeypatch.setattr(vpe.VocabParallelEmbedding,
                        "num_org_vocab_padding",
                        0,
                        raising=False)
    with pytest.raises(KeyError, match="num_org_vocab_padding"):
        _build(monkeypatch, 0, 2)


def test_test_vocabulary_exercises_padding_and_added_tokens(
        loaded, monkeypatch):
    """The equivalence test below is only as good as its shard layout."""
    last = _build(monkeypatch, 1, 2).shard_indices
    assert last.num_org_vocab_padding == 3
    assert last.num_added_elements == 1
    empty = _build(monkeypatch, 3, 4).shard_indices
    assert empty.num_org_vocab_padding == 3
    assert empty.num_added_elements == 0


@pytest.mark.parametrize("tp_size", [2, 4])
@pytest.mark.parametrize("two_dim", [False, True])
def test_matches_upstream_and_unsharded_lookup(loaded, monkeypatch, tp_size,
                                               two_dim):
    """Our forward == upstream's == one unsharded lookup, on every rank."""
    tpu_mod, _ = loaded
    rng = torch.Generator().manual_seed(0)
    full_weight = torch.randn(_VOCAB, _DIM, generator=rng)
    # Every token id, so every shard's first and last id, original and added.
    input_ids = torch.arange(_VOCAB)
    if two_dim:
        shuffled = torch.randperm(_VOCAB, generator=rng)
        input_ids = input_ids[shuffled].reshape(6, 7)

    layers = _build_group(monkeypatch, tp_size, full_weight)
    _use_plain_upstream_mask_helper(monkeypatch)

    ours = _run_tp_group(monkeypatch, tpu_mod,
                         tpu_mod.TpuVocabParallelEmbedding.forward, layers,
                         input_ids)
    upstream = _run_tp_group(monkeypatch, tpu_mod,
                             vpe.VocabParallelEmbedding.forward, layers,
                             input_ids)
    expected = torch.nn.functional.embedding(input_ids, full_weight)
    for rank, (mine, theirs) in enumerate(zip(ours, upstream)):
        torch.testing.assert_close(mine,
                                   theirs,
                                   rtol=0,
                                   atol=0,
                                   msg=f"rank {rank} vs upstream")
        torch.testing.assert_close(mine,
                                   expected,
                                   rtol=0,
                                   atol=0,
                                   msg=f"rank {rank} vs unsharded")


def test_delegates_to_upstream_at_tp1(loaded, monkeypatch):
    """At TP=1 nothing is masked or reduced, so upstream's forward is used."""
    tpu_mod, _ = loaded
    rng = torch.Generator().manual_seed(0)
    layer = _build(monkeypatch, 0, 1)
    layer.weight.data.copy_(torch.randn(layer.weight.shape, generator=rng))
    input_ids = torch.arange(_VOCAB)

    def no_all_reduce(tensor):
        raise AssertionError("all-reduce at tp_size == 1")

    _set_all_reduce(monkeypatch, tpu_mod, no_all_reduce)

    upstream_forward = vpe.VocabParallelEmbedding.forward
    calls = []

    def spy(self, input_):
        calls.append(self)
        return upstream_forward(self, input_)

    monkeypatch.setattr(vpe.VocabParallelEmbedding, "forward", spy)

    out = layer(input_ids)
    assert calls == [layer]
    # Upstream does not remap ids at TP=1: a plain lookup in the one shard.
    torch.testing.assert_close(out,
                               torch.nn.functional.embedding(
                                   input_ids, layer.weight),
                               rtol=0,
                               atol=0)


def _traced_code(forward, layer, input_ids):
    """Python source of the single FX graph Dynamo captures for ``forward``."""
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    torch._dynamo.reset()
    try:
        compiled = torch.compile(lambda ids: forward(layer, ids),
                                 backend=backend,
                                 fullgraph=True,
                                 dynamic=False)
        compiled(input_ids)
    finally:
        torch._dynamo.reset()
    assert len(graphs) == 1
    return graphs[0].code


def test_traced_graph_is_the_same_on_every_rank(loaded, monkeypatch):
    """The reason the class exists.

    Upstream's forward bakes this rank's shard bounds into the graph as
    literals, so each TP rank compiles a different program. Ours reads them
    from buffers, which Dynamo lifts to graph inputs.
    """
    tpu_mod, _ = loaded
    tp_size = 4
    full_weight = torch.zeros(_VOCAB, _DIM)
    input_ids = torch.arange(_VOCAB)
    layers = _build_group(monkeypatch, tp_size, full_weight)
    _use_plain_upstream_mask_helper(monkeypatch)
    _set_all_reduce(monkeypatch, tpu_mod, lambda tensor: tensor)

    ours = [
        _traced_code(tpu_mod.TpuVocabParallelEmbedding.forward, layer,
                     input_ids) for layer in layers
    ]
    assert "embedding" in ours[0]  # a real graph, not an empty one
    for rank, code in enumerate(ours):
        assert code == ours[0], f"rank {rank} traces a different graph"

    # Negative control: the same harness does tell ranks apart when the
    # bounds are Python ints, as they are in upstream's forward.
    upstream = [
        _traced_code(vpe.VocabParallelEmbedding.forward, layer, input_ids)
        for layer in layers
    ]
    assert len(set(upstream)) == tp_size


def _direct_subclasses_in_vllm():
    found = set()
    for path in Path(vllm.__file__).parent.rglob("*.py"):
        source = path.read_text(encoding="utf-8", errors="ignore")
        if "VocabParallelEmbedding" not in source:
            continue
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.ClassDef):
                continue
            for base in node.bases:
                name = getattr(base, "id", getattr(base, "attr", None))
                if name == "VocabParallelEmbedding":
                    found.add(node.name)
    return found


def test_known_upstream_subclasses_are_the_only_ones():
    """Name-keyed registration does not reach subclasses of the class."""
    assert _direct_subclasses_in_vllm() == {
        "ParallelLMHead",
        "BartScaledWordEmbedding",
        "_VocabParallelEmbeddingBase",
        "PLEVocabParallelEmbedding",
    }, ("The direct subclasses of VocabParallelEmbedding in vLLM changed. "
        "TpuVocabParallelEmbedding replaces only VocabParallelEmbedding "
        "itself, so a subclass that inherits upstream's forward traces "
        "rank-specific shard bounds into its graph under TP > 1. For each new "
        "subclass, check whether a model served on TPU builds it; if so, it "
        "needs its own out-of-tree registration. Then update this set and "
        "the TpuVocabParallelEmbedding docstring.")


def test_layers_package_imports_the_module():
    """Registration is an import side effect of ``vllm_torchtpu.layers``,
    which every process imports before it builds a model."""
    tree = ast.parse(_LAYERS_INIT_PATH.read_text())
    imported = {
        alias.name
        for node in tree.body if isinstance(node, ast.ImportFrom)
        and node.module == "vllm_torchtpu.layers.adapter"
        for alias in node.names
    }
    assert "vocab_parallel_embedding" in imported


def test_class_patch_is_gone_from_the_registry():
    name = "_test_vocab_embedding_patch_registry"
    try:
        registry = _load_by_path(name, _REGISTRY_PATH)
        targets = [patch.target for patch in registry.PATCHES]
    finally:
        sys.modules.pop(name, None)
    assert targets
    assert not [t for t in targets if "vocab_parallel_embedding" in t]
