# SPDX-License-Identifier: Apache-2.0
"""The single-layer fused-MoE bench's host side (``bench moe``): one layer's
weights read alone match the prefill loader's."""

import json

import numpy as np
import pytest

from vllm_torchtpu.megakernel.kimi_k3.config import KernelConfig
from vllm_torchtpu.megakernel.kimi_k3.kernels import prefill_megakernel as k
from vllm_torchtpu.megakernel.kimi_k3.kernels import rank_loader as rl

pytestmark = pytest.mark.cpu_test


# ----------------------------------------------------------------- weights

LAYERS = 3  # two MoE layers
EXPERT_SHAPES = {  # per TP4 quarter; gate/up: two interleave chunks per half
    "expert_gu": (2, 112, 2, 512),
    "expert_gus": (2, 112, 1, 512),
    "expert_down": (2, 112, 6, 4),
    "expert_ds": (2, 112, 2, 4),
}
RANGES = [(0, 1), (1, 2)]
CACHE = ("latent_down", "latent_up", "shared_gu_0", "shared_gu_1")
CACHE += ("shared_down_0", "shared_down_1")


def _write(directory, name, arrays):
    """Families stored back to back in one file: their index entries."""
    families, blob, offset = {}, [], 0
    for family, value in arrays.items():
        families[family] = {
            "file": name,
            "offset": offset,
            "shape": list(value.shape),
            "dtype": "float32",
        }
        blob.append(value.tobytes())
        offset += value.nbytes
    (directory / name).write_bytes(b"".join(blob))
    return families


@pytest.fixture(scope="module")
def preshard(tmp_path_factory):
    """A small 32-rank pre-shard whose every layer and rank differ."""
    root = tmp_path_factory.mktemp("preshard")
    rng = np.random.default_rng(0)
    names = rl.PREFILL_IO + rl.PREFILL_COMMON + rl.PREFILL_KDA + rl.PREFILL_MLA
    for rank in range(32):
        directory = root / f"rank{rank:02d}"
        directory.mkdir()
        dense = {
            n: rng.standard_normal((LAYERS, 1, 8)).astype(np.float32) for n in names
        }
        dense.update(
            k_o=rng.standard_normal((LAYERS, 768, 3)).astype(np.float32),
            m_o=rng.standard_normal((LAYERS, 768, 3)).astype(np.float32),
            latent_norm=rng.standard_normal((2, 1, 8)).astype(np.float32),
            router=rng.standard_normal((2, 5, 896)).astype(np.float32),
            router_bias=rng.standard_normal((2, 1, 896)).astype(np.float32),
        )
        families = _write(directory, "dense.bin", dense)
        for name, shape in EXPERT_SHAPES.items():
            files = []
            for index, (lo, hi) in enumerate(RANGES):
                part = rng.standard_normal((hi - lo, *shape[1:])).astype(np.float32)
                files.append(f"{name}_{index}.bin")
                part.tofile(directory / files[-1])
            families[name] = {
                "shape": list(shape),
                "dtype": "float32",
                "files": files,
                "layer_ranges": [list(r) for r in RANGES],
            }
        index = {"rank": rank, "layers": LAYERS, "families": families}
        (directory / "index.json").write_text(json.dumps(index))
    cache = root / "prefill-sp-cache"
    cache.mkdir()
    layout = {}
    for name in CACHE:
        rng.standard_normal((2, 4, 6)).astype(np.float32).tofile(cache / f"{name}.bin")
        layout[name] = {"file": f"{name}.bin", "shape": [2, 4, 6], "dtype": "float32"}
    (cache / "layout.json").write_text(
        json.dumps({"layers": LAYERS, "families": layout})
    )
    order = np.stack([rng.permutation(896) for _ in range(LAYERS - 1)])
    np.savez(root / "plan.npz", order=order.astype(np.int32))
    return root


@pytest.mark.parametrize("plan", [True, False])
@pytest.mark.parametrize("rank", [0, 1, 6, 31])
def test_read_moe_layer_is_a_layer_of_the_prefill_loader(
    monkeypatch, preshard, rank, plan
):
    config = KernelConfig(expert_plan=str(preshard / "plan.npz") if plan else "")
    monkeypatch.setattr(rl, "_config", lambda: config)
    whole = {}
    for group, name, value in rl.iter_prefill_rank(
        preshard, preshard / "prefill-sp-cache", rank, layers=LAYERS
    ):
        family, _, part = name.partition("@")
        whole.setdefault((group, family), []).append((part, value))
    whole = {
        key: np.concatenate([value for _, value in sorted(parts)])
        for key, parts in whole.items()
    }
    for layer in range(LAYERS - 1):
        alone = rl.read_moe_layer(preshard, rank, layer, layers=LAYERS)
        assert set(alone) == set(k.MOE_FAMILIES) | {"mlp_res", "ffn_norm"}
        for name in k.MOE_FAMILIES:
            np.testing.assert_array_equal(
                alone[name], whole[("moe", name)][layer : layer + 1], err_msg=name
            )
        for name in ("mlp_res", "ffn_norm"):
            np.testing.assert_array_equal(
                alone[name], whole[("common", name)][layer + 1]
            )
