from vllm_torchtpu.layers.vllm import attention as attention
from vllm_torchtpu.models.vllm import register_models


def register_layers() -> None:
    register_models()
