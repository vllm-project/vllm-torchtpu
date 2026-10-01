import json
import os
import subprocess
import sys
from typing import Any, Dict, Optional, Set, Tuple

from resolvers.base_sparsity_resolver import BaseSparsityResolver
from sparsity import prune_gemma4

# experiments/ : the package root that `python -m sparsity.tpu_gemma4` needs.
_EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _prune_in_child(kwargs: Dict[str, Any]) -> None:
    """Run ``prune_gemma4(**kwargs, calibrator="tpu")`` in a child process.

    libtpu admits one process per TPU and the JAX client keeps the chips until
    its process exits; the sweep starts vLLM on the same chips right after
    calibration, so the TPU calibrator must not run in this process.  The
    child streams its output here and a non-zero exit raises.
    """
    kwargs = dict(kwargs)
    sli = kwargs.get("sparse_layer_indices")
    if sli is not None:
        kwargs["sparse_layer_indices"] = sorted(int(i) for i in sli)
    cmd = [
        sys.executable, "-u", "-m", "sparsity.tpu_gemma4", "--json",
        json.dumps(kwargs)
    ]
    print(
        ">>> [SPARSEGPT RESOLVER] TPU calibration in a child process "
        "(frees the TPU on exit)",
        flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    rc = subprocess.call(cmd, cwd=_EXPERIMENTS_DIR, env=env)
    if rc != 0:
        raise RuntimeError(f"TPU SparseGPT calibration failed (exit {rc})")


class SparseGptResolver(BaseSparsityResolver):
    """Dedicated resolver for SparseGPT Hessian-based N:M structured sparsity.

    ``resolver_config["use_tpu_sparsegpt"]`` selects the TPU calibrator
    (``sparsity.tpu_gemma4``), run in a child process; otherwise the CPU
    reference (``sparsity.gemma4``) runs in this process.
    """
    METHOD_NAME: str = "sparsegpt"

    def prune_in_memory(
            self,
            base_model: str,
            save_path: str,
            min_layer: int,
            max_layer: int,
            prunen: int,
            prunem: int,
            resolver_config: Dict[str, Any],
            sparse_layer_indices: Optional[Set[int]] = None) -> str:
        kwargs = dict(
            model_path=base_model,
            dataset=resolver_config.get("dataset", "c4"),
            minlayer=min_layer,
            maxlayer=max_layer,
            prunen=prunen,
            prunem=prunem,
            nsamples=resolver_config.get("nsamples", 64),
            seed=resolver_config.get("seed", 0),
            percdamp=resolver_config.get("percdamp", 0.01),
            blocksize=resolver_config.get("blocksize", 128),
            save_path=save_path,
            sparse_layer_indices=sparse_layer_indices,
        )
        if resolver_config.get("use_tpu_sparsegpt", False):
            _prune_in_child(kwargs)
        else:
            prune_gemma4(**kwargs, calibrator="cpu")
        return save_path


def resolve_model(current_config: Dict[str, Any],
                  resolver_config: Dict[str, Any]) -> Tuple[str, Set[str]]:
    resolver = SparseGptResolver()
    return resolver.resolve(current_config=current_config,
                            resolver_config=resolver_config)
