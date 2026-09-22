# Imported for their registration side effects; the out-of-tree
# RowParallelLinear in `linear` and MoERunner in `moe_runner` only affect
# layers built after the import, so this module must be imported before any
# model is constructed.
from vllm_torchtpu.layers.adapter import attention as attention
from vllm_torchtpu.layers.adapter import linear as linear
from vllm_torchtpu.layers.adapter import moe_runner as moe_runner
from vllm_torchtpu.models.vllm import register_models
from vllm_torchtpu.tokenizers import register_tokenizers


def register_layers() -> None:
    register_models()
    register_tokenizers()
