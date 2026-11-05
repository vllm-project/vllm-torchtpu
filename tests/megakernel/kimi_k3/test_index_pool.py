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
"""Kimi-K3.1's index-key cache in serving: its rows per block, its share of
the KV budget, and the one block-major tensor the decoder and a prefill stage
bind."""

import pytest
import torch

from vllm_torchtpu.megakernel.kimi_k3 import pools

pytestmark = pytest.mark.cpu_test

MLA = [*range(3, 92, 4), 92]
STAGE3 = [75, 79, 83, 87, 91, 92]


def names(layers):
    return [
        f"language_model.model.layers.{i}.self_attn.indexer.k_cache" for i in layers
    ]


@pytest.mark.parametrize(
    "dtype,block,rows",
    [
        ("auto", 256, 288),
        ("fp8", 512, 544),
        ("auto", 1024, 4 * 288),
        ("fp8", 2048, 4 * 544),
    ],
)
def test_index_cache_rows(dtype, block, rows):
    assert pools.index_cache_rows(block, dtype) == rows


def test_index_cache_rows_rejects_other_blocks():
    with pytest.raises(ValueError, match="index"):
        pools.index_cache_rows(1024, "fp8")


def test_index_cache_names_sorts_by_layer():
    shuffled = names(MLA)[::-1] + ["language_model.model.layers.3.self_attn.attn"]
    assert pools.index_cache_names(shuffled) == names(MLA)


def test_index_block_budget_leaves_room_for_the_index():
    avail, page = 30 * 2**30, 327680
    blocks = pools.index_block_budget(avail, 29 * page, 24, 512, "fp8")
    per_block = 29 * page + 24 * 544 * 128
    assert blocks * per_block <= avail < (blocks + 1) * per_block


def test_decoder_index_cache_is_one_tensor_the_decoder_binds():
    caches = pools.allocate_index_cache(8, names(MLA), 512, "fp8", "cpu")
    tensor = caches[names(MLA)[0]]
    assert {id(t) for t in caches.values()} == {id(tensor)}
    assert tensor.shape == (8, 24, 544, 128) and tensor.dtype == torch.uint8
    assert pools.index_raw_cache(caches) is tensor


def test_stage_index_cache_stacks_four_decode_pages():
    caches = pools.allocate_index_cache(8, names(STAGE3), 2048, "fp8", "cpu")
    tensor = pools.stage_index_cache(caches, STAGE3)
    assert tensor.shape == (8, 6, 4 * 544, 128)
    assert pools.stage_index_cache({}, STAGE3) is None
    with pytest.raises(RuntimeError, match="index-key cache"):
        pools.stage_index_cache(dict(list(caches.items())[:5]), STAGE3)
    split = dict(caches)
    split[names(STAGE3)[0]] = tensor.clone()
    with pytest.raises(RuntimeError, match="index-key cache"):
        pools.stage_index_cache(split, STAGE3)


def _stage_caches(index: bool):
    """Stage 3's pools (one per MLA layer, the KDA layers spread over them),
    bf16 with 1024-token blocks, and its index-key cache when ``index``."""
    from vllm_torchtpu.megakernel.kimi_k3.layout import stage_cache_layout

    mla = STAGE3
    kda = [layer for layer in range(72, 93) if layer not in mla]
    shape = stage_cache_layout("auto", tuple(mla), tuple(kda)).block_shape
    raws = [torch.zeros((4, *shape), dtype=torch.bfloat16) for _ in mla]
    caches = {
        f"model.layers.{layer}.self_attn.attn": raws[m] for m, layer in enumerate(mla)
    }
    caches |= {
        f"model.layers.{layer}.linear_attn": raws[k % 6] for k, layer in enumerate(kda)
    }
    if index:
        caches |= pools.allocate_index_cache(4, names(mla), 1024, "auto", "cpu")
    return caches


def _stage(monkeypatch, dsa):
    from types import SimpleNamespace

    import jax.numpy as jnp
    import numpy as np

    from vllm_torchtpu.megakernel.kimi_k3 import stage as stage_module
    from vllm_torchtpu.megakernel.kimi_k3.kernels.prefill_stage import stage_plan

    def as_jax(t):
        return jnp.asarray(
            t.view(torch.int16 if t.element_size() == 2 else t.dtype).numpy()
        )

    def register_op(name, mesh, body, in_specs, out_specs, donate_argnums=()):
        def run(*args):
            outs = body(*(as_jax(a) for a in args))
            return tuple(
                torch.from_numpy(np.array(out)).view(arg.dtype)
                for out, arg in zip(outs, args, strict=False)
            )

        return run

    monkeypatch.setattr(stage_module, "register_op", register_op)
    monkeypatch.setattr(
        stage_module, "_device_i32", lambda v: torch.tensor(v, dtype=torch.int32)
    )
    stage = object.__new__(stage_module.KimiPrefillStage)
    stage.config = SimpleNamespace(kv_cache_dtype="auto", dsa=dsa)
    stage.program = SimpleNamespace(plan=stage_plan(3))
    stage.stage, stage.mesh, stage.draft, stage._loaded = 3, None, None, True
    stage.decode_page = 256
    stage._raw, stage._index, stage._ops = [], None, {}
    return stage


def _groups(caches):
    from vllm_torchtpu.megakernel.program import KVGroup

    layers = [name for name in caches if not name.endswith(pools.INDEX_CACHE)]
    return [
        KVGroup(tuple(n for n in layers if n.endswith(".attn")), 1024, 8),
        KVGroup(tuple(n for n in layers if n.endswith("linear_attn")), 1024, 8),
    ]


def test_stage_binds_its_index_cache_and_copies_it_with_the_pools(monkeypatch):
    caches = _stage_caches(index=True)
    stage = _stage(monkeypatch, dsa=object())
    stage.bind_kv_caches(caches, _groups(caches))
    index = caches[names(STAGE3)[0]]
    assert stage._index is index and len(stage._raw) == 6
    assert index.shape == (4, 6, 4 * 288, 128)
    index[1].random_(0, 256)
    stage._raw[0][1].normal_()
    expected_index, expected_pool = index[1].clone(), stage._raw[0][1].clone()
    stage._copy_blocks([(1, 3)])
    # Adopted in place: vLLM's and the connector's tensors see the copy.
    assert stage._index is index and caches[names(STAGE3)[5]] is index
    assert torch.equal(index[3], expected_index)
    assert torch.equal(stage._raw[0][3], expected_pool)
    assert not index[2].any()


@pytest.mark.parametrize("index,dsa", [(False, object()), (True, None)])
def test_stage_refuses_an_index_cache_that_does_not_match_the_model(
    monkeypatch, index, dsa
):
    caches = _stage_caches(index=index)
    stage = _stage(monkeypatch, dsa=dsa)
    with pytest.raises(RuntimeError, match="index-key cache"):
        stage.bind_kv_caches(caches, _groups(caches))


def test_stage_refuses_an_index_cache_of_another_shape(monkeypatch):
    caches = _stage_caches(index=False)
    wrong = torch.zeros((4, 6, 288, 128), dtype=torch.uint8)
    caches |= dict.fromkeys(names(STAGE3), wrong)
    stage = _stage(monkeypatch, dsa=object())
    with pytest.raises(RuntimeError, match="index-key cache"):
        stage.bind_kv_caches(caches, _groups(caches))


def test_stage_writes_window_keys_into_the_index_pages_beside_the_latents(
    monkeypatch,
):
    import jax.numpy as jnp
    import numpy as np

    from vllm_torchtpu.megakernel.kimi_k3 import stage as stage_module
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_prefill

    def as_jax(t):
        if t.dtype == torch.bfloat16:
            return jnp.asarray(t.view(torch.int16).numpy()).view(jnp.bfloat16)
        return jnp.asarray(t.numpy())

    def as_torch(a):
        a = np.array(a)
        if a.dtype.name == "bfloat16":
            return torch.from_numpy(a.view(np.int16)).view(torch.bfloat16)
        return torch.from_numpy(a)

    def register_op(name, mesh, body, in_specs, out_specs, donate_argnums=()):
        return lambda *args: tuple(map(as_torch, body(*map(as_jax, args))))

    caches = _stage_caches(index=True)
    stage = _stage(monkeypatch, dsa=object())
    monkeypatch.setattr(stage_module, "register_op", register_op)
    stage.bind_kv_caches(caches, _groups(caches))
    stage._kda_raw = ()  # latents and keys only
    index = caches[names(STAGE3)[0]]
    window, layers = 512, len(STAGE3)
    generator = torch.Generator().manual_seed(0)
    latents = torch.randn((1, layers, window, 576), generator=generator)
    key_bytes = torch.randint(1, 256, (1, layers, window, 128), dtype=torch.uint8)
    exponents = torch.randint(100, 140, (1, layers, window), dtype=torch.uint8)
    table = torch.tensor([[5, 14]], dtype=torch.int32)  # decode pages 4 b + i
    stage._write(
        window,
        1,
        latents.to(torch.bfloat16),
        torch.zeros(1),
        torch.zeros(1),
        table,
        np.zeros(1, np.int32),
        keys=(key_bytes, exponents),
    )
    assert stage._index is index
    rows = 256 + 32
    for j in range(layers):
        pages = np.asarray(
            dsa_prefill.index_pages(
                jnp.asarray(key_bytes[0, j].numpy()),
                jnp.asarray(exponents[0, j].numpy()),
                256,
            )
        )
        for p, page in enumerate((5, 14)):
            block, sub = divmod(page, 4)
            got = index[block, j, sub * rows : (sub + 1) * rows].numpy()
            np.testing.assert_array_equal(got, pages[p])
            latent = stage._raw[j][block, sub * 128 : (sub + 1) * 128]
            torch.testing.assert_close(
                latent.reshape(256, 640)[:, :576].float(),
                latents[0, j, 256 * p : 256 * (p + 1)].to(torch.bfloat16).float(),
            )
    index[1, :, rows : 2 * rows] = 0
    index[3, :, 2 * rows : 3 * rows] = 0
    assert not index.any()
