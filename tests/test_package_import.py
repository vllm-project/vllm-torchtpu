import subprocess
import sys


def test_package_imports_before_vllm():
    result = subprocess.run(
        [sys.executable, "-c", "import vllm_torchtpu; import vllm"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_submodule_imports_before_vllm_initialization_completes():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import vllm_torchtpu.layers.vllm.quantization.fp8; import vllm",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
