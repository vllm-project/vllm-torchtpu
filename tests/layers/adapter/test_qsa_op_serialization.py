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
"""Fresh-process persisted identity for actual QSA native factories."""

import pytest


def _serde_worker(mode, artifact, kind):
    import json

    import torch
    from jax._src import xla_bridge
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch_tpu._internal import pallas
    from vllm.compilation.caching import VllmSerializableFunction

    from tests.kernels.test_qwen4_exp_indexer_routing import ReferenceTopK
    from vllm_torchtpu.layers.adapter.custom_ops import qsa_op

    assert not xla_bridge.backends_are_initialized()
    registered = {}
    original = pallas.jax_op

    def register(name, function, **kwargs):
        op = original(name, function, **kwargs)
        registered[str(op._opoverload)] = (
            function.keywords["topk_backend"].reverse,
            op,
        )
        return op

    # Record real registrations and delegate unchanged to the pinned Pallas API.
    pallas.jax_op = register
    factory = getattr(qsa_op, f"build_{kind}_op")
    order = [False, True] if mode == "write" else [True, False]
    for reverse in order:
        factory(
            topk_backend=ReferenceTopK(reverse=reverse),
            topk_backend_key=f"qsa-test-reference-v1/reverse={str(reverse).lower()}",
        )
    target, (_, op) = next(
        (name, value) for name, value in registered.items() if value[0] is False
    )
    if mode == "write":
        graph = torch.fx.Graph()
        arguments = [
            graph.placeholder(arg.name) for arg in op._opoverload._schema.arguments
        ]
        node = graph.call_function(op._opoverload, tuple(arguments))
        graph.output(node)
        module = torch.fx.GraphModule({}, graph)
        artifact.write_bytes(VllmSerializableFunction.serialize_graph_module(module))
    else:
        module = VllmSerializableFunction.deserialize_graph_module(
            artifact.read_bytes(), FakeTensorMode()
        )
    targets = [
        str(node.target) for node in module.graph.nodes if node.op == "call_function"
    ]
    assert targets == [target]
    assert registered[targets[0]][0] is False
    assert not xla_bridge.backends_are_initialized()
    result = {
        "mode": mode,
        "registration_order": order,
        "target": target,
        "loaded_backend_reverse": registered[targets[0]][0],
        "backend_initialized": False,
        "scope": (
            "Actual native factory and vLLM GraphPickler identity; no kernel execution"
        ),
    }
    artifact.with_suffix(f".{mode}.json").write_text(json.dumps(result, indent=2))


@pytest.mark.parametrize("kind", ["compact_pooled_qsa"])
def test_persisted_backend_identity_survives_reverse_registration(tmp_path, kind):
    import datetime
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    artifact = tmp_path / f"{kind}.fxpk"
    for mode in ["write", "read"]:
        argv = [sys.executable, __file__, mode, str(artifact), kind]
        env = dict(os.environ, JAX_PLATFORMS="cpu", TORCH_DEVICE_BACKEND_AUTOLOAD="0")
        start = datetime.datetime.now(datetime.UTC).isoformat()
        child = subprocess.Popen(
            argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        stat = Path(f"/proc/{child.pid}/stat").read_text()
        fields = stat[stat.rfind(")") + 2 :].split()
        birth = {
            "pid": child.pid,
            "pgid": int(fields[2]),
            "start_ticks": int(fields[19]),
        }
        try:
            stdout, stderr = child.communicate(timeout=90)
        except subprocess.TimeoutExpired:
            child.kill()
            stdout, stderr = child.communicate()
            raise
        finally:
            record = {
                "argv": argv,
                "start_utc": start,
                "end_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                "birth": birth,
                "exit": child.returncode,
                "reaped": child.poll() is not None,
                "stdout": stdout,
                "stderr": stderr,
            }
            artifact.with_suffix(f".{mode}.process.json").write_text(
                json.dumps(record, indent=2)
            )
        assert child.returncode == 0, stdout + stderr
    saved = json.loads(artifact.with_suffix(".write.json").read_text())
    loaded = json.loads(artifact.with_suffix(".read.json").read_text())
    assert saved["target"] == loaded["target"]
    assert loaded["loaded_backend_reverse"] is False


if __name__ == "__main__":
    import sys
    from pathlib import Path

    _serde_worker(sys.argv[1], Path(sys.argv[2]), sys.argv[3])
