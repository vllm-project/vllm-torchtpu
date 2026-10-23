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

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn

from vllm_torchtpu.experimental.export import checkpoint

_TP2 = {"data": 1, "model": 2}


def _spec(
    name: str, shape: list[int], dtype: str, sharding: list[Any]
) -> dict[str, Any]:
    return {"name": name, "shape": shape, "dtype": dtype, "sharding": sharding}


def test_shard_shape_divides_sharded_dims() -> None:
    spec = _spec("w", [8, 6], "bf16", ["model", None])
    assert checkpoint.shard_shape(spec, _TP2) == [4, 6]
    assert checkpoint.shard_shape(_spec("s", [], "i64", []), _TP2) == []


def test_local_tensors_ties_lm_head_to_the_embedding() -> None:
    model = nn.Module()
    model.register_parameter("w", nn.Parameter(torch.zeros(2)))
    model.register_buffer("b", torch.ones(1))
    tensors = checkpoint.local_tensors(model)
    assert set(tensors) == {"w", "b"}

    tied = nn.Module()
    tied.model = nn.Module()
    tied.model.embed_tokens = nn.Embedding(4, 2)
    tensors = checkpoint.local_tensors(tied)
    assert tensors["lm_head.weight"] is tensors["model.embed_tokens.weight"]


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2, torch.int64]
)
def test_to_numpy_keeps_the_bytes(dtype: torch.dtype) -> None:
    t = torch.arange(6, dtype=torch.float32).reshape(2, 3).to(dtype)
    array = checkpoint.to_numpy(t)
    assert array.shape == (2, 3)
    assert array.tobytes() == t.contiguous().view(torch.uint8).numpy().tobytes()


def test_checkpoint_shards_reshapes_per_device_constants() -> None:
    inputs = [
        _spec("w", [8, 6], "bf16", ["model", None]),
        _spec("start", [2], "i64", ["model"]),
        _spec("pad", [], "i64", []),
    ]
    tensors = {
        "w": torch.ones(4, 6, dtype=torch.bfloat16),
        "start": torch.tensor(75968, dtype=torch.int64),
        "pad": torch.tensor(0, dtype=torch.int64),
    }
    shards = checkpoint.checkpoint_shards(inputs, tensors, _TP2)
    assert [s.shape for s in shards] == [(4, 6), (1,), ()]
    assert shards[1].tolist() == [75968]


def test_checkpoint_shards_rejects_mismatches() -> None:
    inputs = [_spec("w", [8, 6], "bf16", ["model", None])]
    with pytest.raises(KeyError, match="not in the loaded model"):
        checkpoint.checkpoint_shards(inputs, {}, _TP2)
    with pytest.raises(ValueError, match="dtype f32"):
        checkpoint.checkpoint_shards(inputs, {"w": torch.ones(4, 6)}, _TP2)
    with pytest.raises(ValueError, match="shard shape"):
        checkpoint.checkpoint_shards(
            inputs, {"w": torch.ones(8, 6, dtype=torch.bfloat16)}, _TP2
        )


def test_abstract_params_inputs_are_in_argument_order(tmp_path: Path) -> None:
    inputs = [
        dict(_spec("b", [2], "f32", [None]), arg_index=1),
        dict(_spec("a", [2], "f32", [None]), arg_index=0),
    ]
    meta = {"functions": {"abstract_params": {"default": {"inputs": inputs}}}}
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    names = [t["name"] for t in checkpoint.abstract_params_inputs(str(tmp_path))]
    assert names == ["a", "b"]


def test_save_checkpoint_writes_zarr_v2_arrays(tmp_path: Path) -> None:
    pytest.importorskip("orbax.checkpoint")
    ts = pytest.importorskip("tensorstore")
    import jax

    x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        _save_and_check(tmp_path, ts)
    finally:
        jax.config.update("jax_enable_x64", x64)


def _save_and_check(tmp_path: Path, ts: Any) -> None:
    single = {"data": 1, "model": 1}
    inputs = [
        _spec("w", [2, 3], "bf16", [None, None]),
        _spec("q", [4], "f8E4M3FN", [None]),
        _spec("i", [], "i64", []),
    ]
    tensors = {
        "w": torch.arange(6, dtype=torch.float32).reshape(2, 3).to(torch.bfloat16),
        "q": torch.tensor([1.0, 2.0, 0.5, -4.0]).to(torch.float8_e4m3fn),
        "i": torch.tensor(151936, dtype=torch.int64),
    }
    shards = checkpoint.checkpoint_shards(inputs, tensors, single)
    data_dir = tmp_path / "serving_checkpoint" / "data"
    checkpoint.save_checkpoint(inputs, shards, single, str(data_dir))

    assert (data_dir / "_CHECKPOINT_METADATA").is_file()
    for i, (spec, shard) in enumerate(zip(inputs, shards, strict=True)):
        array_dir = data_dir / f"array_{i}"
        header = json.loads((array_dir / ".zarray").read_text())
        assert header["zarr_format"] == 2 and header["shape"] == spec["shape"]
        assert header["compressor"]["id"] == "zstd"
        value = (
            ts.open(
                {
                    "driver": "zarr",
                    "kvstore": {"driver": "file", "path": str(array_dir)},
                }
            )
            .result()
            .read()
            .result()
        )
        assert np.asarray(value).tobytes() == shard.tobytes(), spec["name"]
    dtypes = [
        json.loads((data_dir / f"array_{i}" / ".zarray").read_text())["dtype"]
        for i in range(3)
    ]
    assert dtypes == ["bfloat16", "float8_e4m3fn", "<i8"]
