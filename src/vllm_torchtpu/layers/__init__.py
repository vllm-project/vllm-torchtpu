# Imported for their registration side effects; the out-of-tree
# RowParallelLinear in `linear` only affects layers built after the import, so
# this module must be imported before any model is constructed.
from vllm_torchtpu import envs as _envs
from vllm_torchtpu.layers.adapter import attention as attention
from vllm_torchtpu.layers.adapter import linear as linear
from vllm_torchtpu.models.vllm import register_models
from vllm_torchtpu.tokenizers import register_tokenizers

if _envs.TPU_FUSED_ADD_RMSNORM:
    # Out-of-tree RMSNorm; like `linear`, it must be registered before the
    # model is built. Flag off: the stock class, untouched.
    from vllm_torchtpu.layers.adapter import rmsnorm as rmsnorm  # noqa: F401


def register_layers() -> None:
    register_models()
    register_tokenizers()
