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
"""Host-only tests: temporaries uploaded non_blocking stay alive until the
input-staging fence that covers them has been waited on, and no cached or
per-step upload reads a freed temporary.

The runner module needs vLLM + torch-tpu, so the helpers under test are
extracted from the source with ``ast`` and run against a fake torch.
"""

import ast
import pathlib
import re
import textwrap
from types import SimpleNamespace

SRC = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src/vllm_torchtpu/runner/tpu_runner.py"
)


def _load():
    tree = ast.parse(SRC.read_text())
    funcs = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_h2d_temp":
            funcs[node.name] = ast.get_source_segment(SRC.read_text(), node)
        if isinstance(node, ast.ClassDef) and node.name == "TPUModelRunner":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in (
                    "_record_input_staging_fence",
                    "_wait_input_staging_fence",
                ):
                    funcs[item.name] = textwrap.dedent(
                        ast.get_source_segment(SRC.read_text(), item)
                    )
    return funcs


class _Event:
    instances = []

    def __init__(self):
        self.synced = 0
        _Event.instances.append(self)

    def record(self):
        pass

    def synchronize(self):
        self.synced += 1


class _Host:
    def __init__(self, name):
        self.name = name

    def to(self, device, non_blocking=False):
        return ("dev", self.name, non_blocking)


def _ns():
    funcs = _load()
    assert set(funcs) == {
        "_h2d_temp",
        "_record_input_staging_fence",
        "_wait_input_staging_fence",
    }
    g = {"torch": SimpleNamespace(Tensor=object, tpu=SimpleNamespace(Event=_Event))}
    for src in funcs.values():
        exec(src, g)  # noqa: S102 - trusted repo source
    return g


def test_temporaries_live_until_their_fence_is_waited():
    g = _ns()
    runner = SimpleNamespace(device="tpu", _input_staging_fence=None)
    a, b = _Host("a"), _Host("b")
    assert g["_h2d_temp"](runner, a) == ("dev", "a", True)
    g["_record_input_staging_fence"](runner)  # fence covers a
    g["_h2d_temp"](runner, b)  # issued after the record
    assert runner._h2d_fenced == [[a]] and runner._h2d_keepalive == [b]
    g["_wait_input_staging_fence"](runner)  # next chunk: a's transfer is done
    assert runner._h2d_fenced == [] and runner._h2d_keepalive == [b]
    g["_record_input_staging_fence"](runner)
    assert runner._h2d_fenced == [[b]] and runner._h2d_keepalive == []
    g["_wait_input_staging_fence"](runner)
    assert runner._h2d_fenced == []


def test_second_record_before_wait_keeps_older_batch():
    g = _ns()
    runner = SimpleNamespace(device="tpu", _input_staging_fence=None)
    a, b = _Host("a"), _Host("b")
    g["_h2d_temp"](runner, a)
    g["_record_input_staging_fence"](runner)
    g["_h2d_temp"](runner, b)
    g["_record_input_staging_fence"](runner)  # no wait in between
    assert runner._h2d_fenced == [[a], [b]]
    g["_wait_input_staging_fence"](runner)  # newest fence done => all earlier done
    assert runner._h2d_fenced == []


def test_keepalive_is_bounded_without_fences():
    g = _ns()
    runner = SimpleNamespace(device="tpu", _input_staging_fence=None)
    for i in range(4097):
        g["_h2d_temp"](runner, _Host(i))
    n0 = len(_Event.instances)
    g["_wait_input_staging_fence"](runner)
    assert runner._h2d_keepalive == [] and len(_Event.instances) == n0 + 1
    assert _Event.instances[-1].synced == 1


def test_no_non_blocking_upload_of_a_temporary_remains():
    root = SRC.parents[1]
    src = "".join(
        (root / f).read_text()
        for f in (
            "runner/tpu_runner.py",
            "runner/speculative_decoding_manager.py",
            "layers/core/attention_metadata.py",
        )
    )
    pat = re.compile(
        r"torch\.(?:tensor|from_numpy)\((?:[^()]|\((?:[^()]|\([^()]*\))*\))*\)"
        r"\s*\.to\(\s*self\.device,\s*non_blocking=True",
        re.S,
    )
    assert not pat.findall(src)
    assert "- 1\n                ).to(self.device, non_blocking=True)" not in src
    assert (
        "reshape(-1).to(\n                runner.device, non_blocking=True" not in src
    )
    assert "unsqueeze(1).to(logits.device, non_blocking=True)" not in src


def test_cached_subtract_indices_upload_synchronously():
    import numpy as np

    src = SRC.read_text()
    i = src.index('            sub_cache = getattr(self, "_async_subtract_idx_cache"')
    end = src.index(
        "                count=num_reqs,\n                )", i + 500
    ) + len("                count=num_reqs,\n                )")
    block = src[i:end]
    assert "non_blocking=" not in block
    assert "_h2d_temp(" not in block
    assert "sub_cache.clear()" not in block
    assert "while len(sub_cache) >= 256:" in block
    assert "sub_cache.pop(next(iter(sub_cache)))" in block
    assert "seq_idx_dev = torch.from_numpy(seq_idx_np).to(self.device)" in block
    assert "pos_idx_dev = torch.from_numpy(pos_idx_np).to(self.device)" in block
    assert block.count(".to(self.device)") == 2

    uploads = []

    class _FakeDevTensor:
        def __init__(self, arr):
            self.arr = np.asarray(arr).copy()

        def __getitem__(self, slc):
            return _FakeDevTensor(self.arr[slc])

    class _FakeHostTensor:
        def __init__(self, arr):
            self.arr = np.asarray(arr)

        def to(self, device, non_blocking=False):
            uploads.append((device, non_blocking, self.arr.shape))
            return _FakeDevTensor(self.arr)

    code = textwrap.dedent(block)
    runner = SimpleNamespace(device="tpu")
    for k in range(260):
        sub_key = (4, 8, ((k, 2), (None, 3), (1, 1)))
        ns = {
            "self": runner,
            "sub_key": sub_key,
            "sub_pairs": sub_key[2],
            "num_reqs": len(sub_key[2]),
            "target_num_reqs": 4,
            "padded_total_num_scheduled_tokens": 8,
            "np": np,
            "torch": SimpleNamespace(from_numpy=_FakeHostTensor),
        }
        exec(code, ns)  # noqa: S102
        if k == 0:
            assert ns["seq_idx_dev"].arr.tolist() == [0, -1, 1, -1]
            assert ns["pos_idx_dev"].arr.tolist() == [0, 0, -1, -1, -1, 1, -1, -1]

    # 260 distinct keys: FIFO at 256 keeps the 256 most recent (never .clear()),
    # each miss did 1 synchronous upload, and _h2d_keepalive was never touched.
    assert len(runner._async_subtract_idx_cache) == 256
    assert (4, 8, ((0, 2), (None, 3), (1, 1))) not in runner._async_subtract_idx_cache
    assert (4, 8, ((3, 2), (None, 3), (1, 1))) not in runner._async_subtract_idx_cache
    assert (4, 8, ((4, 2), (None, 3), (1, 1))) in runner._async_subtract_idx_cache
    assert (4, 8, ((259, 2), (None, 3), (1, 1))) in runner._async_subtract_idx_cache
    assert len(uploads) == 520 and all(not nb for _, nb, _ in uploads)
    assert not hasattr(runner, "_h2d_keepalive")


def test_cached_subst_indices_fifo_256_and_synchronous():
    src = SRC.read_text()
    i = src.index('subst_cache = getattr(self, "_async_subst_dev_cache"')
    block = src[i : src.index("subst_cache[subst_key] = substitution_indices", i)]
    assert "non_blocking=" not in block
    assert "_h2d_temp(" not in block
    assert "while len(subst_cache) >= 256:" in block
    assert block.count(".to(self.device)") == 1


def test_eagle3_prepare_draft_inputs_uses_h2d_temp_for_temporaries():
    """Uncached per-step temporaries in Eagle3Proposer._prepare_draft_inputs
    must use _h2d_temp, while _async_pad_req_idx_cache stays synchronous."""
    path = SRC.parents[1] / "spec_decode/eagle3.py"
    text = path.read_text()
    tree = ast.parse(text)
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_prepare_draft_inputs"
    )
    imports = [
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.ImportFrom) and any(a.name == "_h2d_temp" for a in n.names)
    ]
    uses = [
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.Name) and n.id == "_h2d_temp"
    ]
    assert len(imports) == 1 and len(uses) >= 9 and imports[0] < min(uses)
    seg = ast.get_source_segment(text, fn)
    assert seg is not None
    # Only the two _async_pad_req_idx_cache miss paths may call .to(runner.device)
    # directly inside _prepare_draft_inputs.
    assert seg.count("pad_req_idx_np).to(") == 2
    assert seg.count("pad_req_idx_np).to(runner.device)") == seg.count(
        ".to(runner.device)"
    )


def test_uniform_and_heterogeneous_sampling_params_cache():
    """_build_padded_sampling_params reuses one uniform cache entry across
    varying num_active_reqs and uploads synchronously (never _h2d_temp)."""
    import numpy as np

    tree = ast.parse(SRC.read_text())
    fn_node = next(
        item
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TPUModelRunner"
        for item in node.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "_build_padded_sampling_params"
    )
    fn_src = textwrap.dedent(ast.get_source_segment(SRC.read_text(), fn_node))
    assert "_h2d_temp(" not in fn_src
    assert "non_blocking=" not in fn_src

    uploads = []

    class _FakeTensor:
        def __init__(self, arr, dtype="bf16", device_type="cpu"):
            self._arr = np.asarray(arr)
            self.dtype = dtype
            self.device = SimpleNamespace(type=device_type)
            self.shape = self._arr.shape

        def __getitem__(self, idx):
            return _FakeTensor(
                self._arr[idx], dtype=self.dtype, device_type=self.device.type
            )

        def numpy(self):
            return self._arr

        def copy_(self, other):
            self._arr[:] = other._arr
            return self

        def unsqueeze(self, dim):
            return _FakeTensor(
                np.expand_dims(self._arr, dim),
                dtype=self.dtype,
                device_type=self.device.type,
            )

        def to(self, device, non_blocking=False):
            uploads.append((self.dtype, non_blocking))
            dev_type = getattr(device, "type", str(device))
            return _FakeTensor(self._arr.copy(), dtype=self.dtype, device_type=dev_type)

        def __ge__(self, val):
            return self._arr >= val

    fake_torch = SimpleNamespace(
        Tensor=_FakeTensor,
        int32="int32",
        float32="float32",
        ones=lambda n, dtype="float32": _FakeTensor(np.ones(n), dtype=dtype),
        zeros=lambda n, dtype="int32": _FakeTensor(
            np.zeros(n, dtype=np.int32), dtype=dtype
        ),
        full=lambda shape, val, dtype="float32": _FakeTensor(
            np.full(shape, val), dtype=dtype
        ),
        zeros_like=lambda t: _FakeTensor(np.zeros_like(t._arr), dtype=t.dtype),
        where=lambda cond, a, b: _FakeTensor(
            np.where(cond, a._arr, b._arr), dtype=b.dtype
        ),
    )
    ns = {"torch": fake_torch, "np": np}
    exec(fn_src, ns)  # noqa: S102
    build = ns["_build_padded_sampling_params"]

    runner = SimpleNamespace(
        vocab_size=151936,
        input_batch=SimpleNamespace(
            temperature_cpu_tensor=_FakeTensor(np.full(32, 0.6, dtype=np.float32)),
            top_k_cpu_tensor=_FakeTensor(
                np.full(32, 20, dtype=np.int32), dtype="int32"
            ),
            top_p_cpu_tensor=_FakeTensor(np.full(32, 0.95, dtype=np.float32)),
        ),
    )
    logits = _FakeTensor(np.zeros((32, 8)), dtype="bf16", device_type="tpu")

    # Vary num_active_reqs (32 -> 17 -> 1) and start/end offsets: all hit the
    # same single uniform cache entry after the first call.
    out_32 = build(runner, 0, 32, logits, padded_num_reqs=32)
    assert len(uploads) == 3 and all(not nb for _, nb in uploads)
    out_17 = build(runner, 0, 17, logits, padded_num_reqs=32)
    out_slice = build(runner, 5, 13, logits, padded_num_reqs=32)
    out_1 = build(runner, 0, 1, logits, padded_num_reqs=32)
    assert out_17 is out_32 and out_slice is out_32 and out_1 is out_32
    assert len(uploads) == 3
    assert len(runner._sampling_params_dev_cache) == 1
    for n in (1, 8, 17, 32):
        assert np.allclose(out_32[0]._arr[:n, 0], 0.6)
        assert (out_32[1]._arr[:n, 0] == 20).all()
        assert np.allclose(out_32[2]._arr[:n, 0], 0.95)

    # Heterogeneous batch falls back to per-slice key with neutral tail padding.
    runner.input_batch.temperature_cpu_tensor._arr[1] = 0.8
    out_het = build(runner, 0, 4, logits, padded_num_reqs=32)
    assert out_het is not out_32
    assert len(uploads) == 6
    assert np.allclose(out_het[0]._arr[:4, 0], [0.6, 0.8, 0.6, 0.6])
    assert np.allclose(out_het[0]._arr[4:, 0], 1.0)


def test_spec_metadata_binds_h2d_temp_before_every_use():
    """A function-local ``from ... import _h2d_temp`` placed after the cache-hit
    path made that path raise UnboundLocalError."""
    path = SRC.parent / "speculative_decoding_manager.py"
    tree = ast.parse(path.read_text())
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "get_spec_decode_metadata"
    )
    imports = [
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.ImportFrom) and any(a.name == "_h2d_temp" for a in n.names)
    ]
    uses = [
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.Name) and n.id == "_h2d_temp"
    ]
    assert len(imports) == 1 and uses and imports[0] < min(uses)
