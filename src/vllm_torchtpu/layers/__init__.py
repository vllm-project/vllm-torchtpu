try:
    from vllm_torchtpu.layers.adapter import attention as attention
    from vllm_torchtpu.layers.adapter import linear as linear
    from vllm_torchtpu.models.vllm import register_models
    from vllm_torchtpu.tokenizers import register_tokenizers
except ImportError:  # pragma: no cover - CPU test environment without full TPU stack
    register_models = lambda: None  # type: ignore[assignment]
    register_tokenizers = lambda: None  # type: ignore[assignment]


def register_layers() -> None:
    register_models()
    register_tokenizers()
