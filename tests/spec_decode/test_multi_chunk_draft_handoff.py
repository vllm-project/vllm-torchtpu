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

"""The two-chunk draft hand-off between the runner and the MTP proposer: on an
asynchronous spec step whose two sampled chunks fit one bucket, sample_tokens
sets the drafter's ``_skip_next_multi_chunk_draft_cat`` before its propose
call, and ``Eagle3Proposer.propose`` reads and resets it. Structural, because
sample_tokens needs a TPU-backed runner to execute."""

import ast
import inspect
import textwrap

import pytest

from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

pytestmark = pytest.mark.cpu_test

FLAG = "_skip_next_multi_chunk_draft_cat"


def _tree(fn):
    return ast.parse(textwrap.dedent(inspect.getsource(fn)))


def _calls(node, name):
    return [
        c
        for c in ast.walk(node)
        if isinstance(c, ast.Call)
        and isinstance(c.func, ast.Attribute)
        and c.func.attr == name
    ]


def test_sample_tokens_sets_the_flag_before_the_propose_call():
    tree = _tree(TPUModelRunner.sample_tokens)
    setters = [
        (block, n)
        for block in ast.walk(tree)
        if isinstance(block, ast.If)
        for n in block.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == FLAG for t in n.targets)
    ]
    assert len(setters) == 1
    block, setter = setters[0]
    target = setter.targets[0]
    assert isinstance(target, ast.Attribute)
    assert ast.unparse(target.value) == "drafter"
    assert isinstance(setter.value, ast.Constant) and setter.value.value is True
    condition = ast.unparse(block.test)
    assert "is_async" in condition
    assert "len(next_tokens_per_chunk) == 2" in condition
    assert "self.max_num_reqs" in condition
    proposes = [
        c.lineno
        for c in _calls(tree, "propose_draft_token_ids")
        if c.lineno > setter.lineno
    ]
    assert proposes, "the flag must be set before the drafter proposes"


def test_the_proposer_reads_and_resets_the_flag():
    tree = _tree(Eagle3Proposer.propose)
    reads = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute)
        and n.attr == FLAG
        and isinstance(n.ctx, ast.Load)
    ]
    resets = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == FLAG for t in n.targets)
        and isinstance(n.value, ast.Constant)
        and n.value.value is False
    ]
    assert reads and resets
