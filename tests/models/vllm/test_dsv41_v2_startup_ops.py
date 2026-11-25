# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model startup registers the real V4.1 cache and FP8 indexer JAX ops."""

import pytest

pytestmark = pytest.mark.cpu_test


def test_model_startup_registers_real_jax_ops_on_cpu(monkeypatch):
    import jax
    import numpy as np
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal import pallas

    from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41 import (
        deepseek_v41_mhc_op,
    )
    from vllm_torchtpu.models.vllm.deepseek_v41.common.mhc import (
        initialize_delayed_mhc,
        initialize_mhc_post,
    )
    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import (
        attention_kernels,
        cache,
        indexer_fp8,
    )
    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        set_vllm_model_wrapper_context,
    )

    mesh = Mesh(np.asarray(jax.devices("cpu")[:1]), ("registration",))
    with set_vllm_model_wrapper_context(mesh=mesh):
        # The former callback must fail TorchTPU's real registration verifier.
        with pytest.raises(ValueError, match="Missing argument type annotation"):
            pallas.jax_op(
                "pallas::dsv41_test_untyped_read_regression",
                lambda pool, records, elements, fp32=False: (
                    cache.gather_cache_values(pool, records, elements, fp32)
                ),
                mesh=mesh,
                input_partition_specs=(P(), P(), None, None),
            )
        # Match the startup registration calls in v2/model.py, plus the FP8
        # indexer ops that IndexerCache registers. These are real jax_op
        # registrations, with no model weights or TPU execution.
        cache.initialize_cache_ops()
        initialize_delayed_mhc()
        deepseek_v41_mhc_op.initialize_shifted_mhc()
        initialize_mhc_post()
        attention_kernels.initialize_paged_attention()
        indexer_fp8.initialize_fp8_indexer()
        assert all(
            operation is not None
            for operation in (
                cache._read_values_op,
                cache._read_values_2bank_op,
                indexer_fp8._query_op,
                indexer_fp8._write_op,
                indexer_fp8._read_op,
                indexer_fp8._select_op,
                indexer_fp8._prepare_op,
                indexer_fp8._prepared_select_op,
            )
        )
        # Re-entering initialization must retain the already registered ops.
        read = cache._read_values_op
        select = indexer_fp8._select_op
        cache.initialize_cache_ops()
        indexer_fp8.initialize_fp8_indexer()
        assert cache._read_values_op is read
        assert indexer_fp8._select_op is select
