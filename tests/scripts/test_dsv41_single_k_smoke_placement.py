# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Smoke input placement stays compatible with pinned JAX without using TPU."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.cpu_test
def test_smoke_host_inputs_execute_on_eight_pmap_replicas():
    # Extract only the fixture and actual input-placement expression. Importing
    # the smoke's model bridge is unnecessary to verify JAX's placement API.
    source = Path(__file__).resolve().parents[2] / (
        "scripts/dsv41_bench/single_k_model_smoke.py"
    )
    program = r"""
import ast
import pathlib
import sys
import jax
import jax.numpy as jnp
import numpy as np

assert jax.__version__ == '0.10.2', jax.__version__
assert len(jax.local_devices()) == 8
source = pathlib.Path(sys.argv[1])
tree = ast.parse(source.read_text())
fixture = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name == 'fixture')
main = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
            and node.name == 'main')
placement = next(node.value for node in ast.walk(main)
                 if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == 'inputs'
                         for target in node.targets))
namespace = dict(jax=jax, jnp=jnp, np=np, devices=jax.local_devices(), seed=73)
code = compile(ast.Module(body=[fixture], type_ignores=[]), str(source), 'exec')
exec(code, namespace)
inputs = eval(compile(ast.Expression(placement), str(source), 'eval'), namespace)

def sample(*values):
    assert values[0].shape == (4096, 128)
    assert values[4].shape == (12, 32, 128)
    assert values[5].shape == (12, 32)
    return jnp.stack([value.reshape(-1)[0].astype(jnp.float32) for value in values])

actual = np.asarray(jax.pmap(sample, devices=namespace['devices'])(*inputs))
expected = np.asarray([np.asarray(value).reshape(-1)[0]
                       for value in namespace['fixture'](73)], dtype=np.float32)
np.testing.assert_array_equal(actual, np.broadcast_to(expected, (8, 7)))
print('PINNED_JAX_8_CPU_PMAP_PLACEMENT_PASS', actual.shape)
"""
    environment = dict(os.environ)
    environment.update(
        JAX_PLATFORMS="cpu",
        JAX_NUM_CPU_DEVICES="8",
        VLLM_TARGET_DEVICE="empty",
        TORCH_DEVICE_BACKEND_AUTOLOAD="0",
    )
    result = subprocess.run(
        [sys.executable, "-c", program, str(source)],
        env=environment,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PINNED_JAX_8_CPU_PMAP_PLACEMENT_PASS" in result.stdout
