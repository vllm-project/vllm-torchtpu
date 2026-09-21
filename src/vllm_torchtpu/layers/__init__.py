# Imported for their registration side effects; the out-of-tree
# RowParallelLinear in `linear` and VocabParallelEmbedding in
# `vocab_parallel_embedding` only affect layers built after the import, so
# this module must be imported before any model is constructed.
from vllm_torchtpu.layers.adapter import attention as attention
from vllm_torchtpu.layers.adapter import linear as linear
from vllm_torchtpu.layers.adapter import \
    vocab_parallel_embedding as vocab_parallel_embedding
from vllm_torchtpu.models.vllm import register_models
from vllm_torchtpu.tokenizers import register_tokenizers


def register_layers() -> None:
    register_models()
    register_tokenizers()
