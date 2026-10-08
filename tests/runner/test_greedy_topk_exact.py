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
"""Host-only exactness tests for the vocab-parallel top-32 candidate extraction.

``topk_candidates_step_eligible`` admits any request with ``1 <= top_k <= 32``
(Qwen3.5's generation_config default is ``top_k=20``) and samples from the
union of every TP shard's local "top-32" candidates. That is exact only if each
shard's candidates really are its exact local top-32: a single shard can hold
all of the global top-k.

``_extract_local_topk32_torch`` (bf16 fallback head, the target model's path)
and ``_extract_local_topk32_jax`` (W8A8/W4A8 fused head) are extracted from the
source with ``ast`` (the module imports vLLM / torch_tpu) and compared against
an exact ``topk``.
"""

import ast
import pathlib

import numpy as np
import pytest
import torch

SRC = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src/vllm_torchtpu/runner/greedy_top_tokens.py"
)

VOCAB = 248320  # Qwen3.5 padded vocab
TP = 8
SHARD = VOCAB // TP  # 31040 = 485 blocks of 64


def _load():
    text = SRC.read_text()
    tree = ast.parse(text)
    ns = {"torch": torch}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name)
            and t.id
            in (
                "LOCAL_TOPK_CANDIDATES",
                "_LOCAL_TOPK_BLOCK_SIZE",
                "_LOCAL_TOPK_NUM_BLOCKS",
            )
            for t in node.targets
        ):
            exec(compile(ast.Module([node], []), str(SRC), "exec"), ns)
    for node in tree.body:
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "_extract_local_topk32_torch"
        ):
            exec(compile(ast.Module([node], []), str(SRC), "exec"), ns)
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "_ensure_tpu_quant_ops_registered"
        ):
            for inner in node.body:
                if (
                    isinstance(inner, ast.FunctionDef)
                    and inner.name == "_extract_local_topk32_jax"
                ):
                    import jax
                    import jax.numpy as jnp

                    ns["jax"] = jax
                    ns["jnp"] = jnp
                    exec(compile(ast.Module([inner], []), str(SRC), "exec"), ns)
    assert "_extract_local_topk32_torch" in ns
    assert "_extract_local_topk32_jax" in ns
    return ns


NS = _load()
K = NS["LOCAL_TOPK_CANDIDATES"]


def _torch_impl(x: torch.Tensor):
    v, i = NS["_extract_local_topk32_torch"](x)
    return v.float(), i.long()


def _jax_impl(x: torch.Tensor):
    import jax.numpy as jnp

    v, i = NS["_extract_local_topk32_jax"](jnp.asarray(x.float().numpy()))
    return torch.from_numpy(np.asarray(v)).float(), torch.from_numpy(
        np.asarray(i)
    ).long()


IMPLS = {"torch": _torch_impl, "jax": _jax_impl}


def _check_local_exact(impl, shard_logits: torch.Tensor):
    """Candidates must be the exact local top-32 (values) and point at them."""
    vals, ids = impl(shard_logits)
    exact_vals, _ = torch.topk(shard_logits.float(), K, dim=-1)
    # ids must point at the values they claim.
    torch.testing.assert_close(
        shard_logits.float().gather(1, ids), vals, rtol=0, atol=0
    )
    torch.testing.assert_close(
        vals.sort(dim=-1, descending=True).values, exact_vals, rtol=0, atol=0
    )


def _global_topk_from_candidates(impl, logits: torch.Tensor, k: int):
    """Mimic tpu_get_topk_candidates + sampler top-k over the union."""
    cv, ci = [], []
    for r in range(TP):
        v, i = impl(logits[:, r * SHARD : (r + 1) * SHARD])
        cv.append(v)
        ci.append(i + r * SHARD)
    cv = torch.cat(cv, dim=-1)
    ci = torch.cat(ci, dim=-1)
    top_v, top_pos = torch.topk(cv, k, dim=-1)
    return top_v, ci.gather(1, top_pos)


@pytest.mark.parametrize("impl", IMPLS)
def test_top20_spread_over_20_blocks_in_one_shard(impl):
    # Adversarial: shard 0 holds the global top-20, one per 64-wide block.
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(2, VOCAB, generator=g) * 0.1
    hot = torch.arange(20) * 64 + 5  # 20 distinct blocks of shard 0
    logits[:, hot] = 10.0 + torch.arange(20, 0, -1).float()
    top_v, top_ids = _global_topk_from_candidates(IMPLS[impl], logits, 20)
    exact_v, exact_ids = torch.topk(logits, 20, dim=-1)
    assert set(top_ids[0].tolist()) == set(exact_ids[0].tolist())
    torch.testing.assert_close(top_v, exact_v, rtol=0, atol=0)


@pytest.mark.parametrize("impl", IMPLS)
def test_top20_in_one_block_sanity(impl):
    # All of the top-20 in one block: exact even with 8 blocks.
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(2, VOCAB, generator=g) * 0.1
    hot = 3 * SHARD + 640 + torch.arange(20)
    logits[:, hot] = 10.0 + torch.arange(20, 0, -1).float()
    _, top_ids = _global_topk_from_candidates(IMPLS[impl], logits, 20)
    assert set(top_ids[0].tolist()) == set(hot.tolist())


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_random_local_top32_exact(impl, dtype):
    # iid logits: the local top-32 almost surely span > 8 blocks.
    g = torch.Generator().manual_seed(2)
    shard_logits = (torch.randn(16, SHARD, generator=g) * 3.0).to(dtype)
    _check_local_exact(IMPLS[impl], shard_logits)


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("k", [8, 20, 32])
def test_random_skewed_global_topk_exact(impl, k):
    # Low token ids (common tokens) live in shard 0: skew mass there.
    g = torch.Generator().manual_seed(3 + k)
    logits = torch.randn(8, VOCAB, generator=g)
    logits[:, :SHARD] += 2.5
    top_v, _ = _global_topk_from_candidates(IMPLS[impl], logits, k)
    exact_v, _ = torch.topk(logits, k, dim=-1)
    torch.testing.assert_close(top_v, exact_v, rtol=0, atol=0)


@pytest.mark.parametrize("impl", IMPLS)
def test_small_and_ragged_shards(impl):
    g = torch.Generator().manual_seed(4)
    for n in (40, 500, 2047, 2049, 4096 + 13):
        _check_local_exact(IMPLS[impl], torch.randn(4, n, generator=g))
