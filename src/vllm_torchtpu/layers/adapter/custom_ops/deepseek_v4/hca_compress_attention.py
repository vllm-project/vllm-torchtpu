# SPDX-License-Identifier: Apache-2.0
"""HCA attention plus its compressor with a guarded dense-query composition."""

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import hca_decode_compressor
from vllm_torchtpu.kernels.deepseek_v4.core_attention import (
    hca_compress_decode,
    swa_hca_decode,
)
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4.deepseek_v4_attention_op import (  # noqa: E501
    _attention_hca,
)
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4.deepseek_v4_compressor import (  # noqa: E501
    _compressor_jax_hca,
)


def attention_compressor(
    q: jax.Array,
    kv: jax.Array,
    sw: jax.Array,
    sl: jax.Array,
    sp: jax.Array,
    cu: jax.Array,
    dist: jax.Array,
    main: jax.Array,
    ml: jax.Array,
    attend: jax.Array,
    mp: jax.Array,
    mcu: jax.Array,
    mdist: jax.Array,
    sinks: jax.Array,
    pos: jax.Array,
    cs: jax.Array,
    hidden: jax.Array,
    weight: jax.Array,
    ape: jax.Array,
    norm: jax.Array,
    state: jax.Array,
    st: jax.Array,
    ccu: jax.Array,
    kt: jax.Array,
    cdist: jax.Array,
    *,
    eps: float,
    sm_scale: float,
    sliding_window: int,
    logical_page_size: int,
    swa_only: bool,
    two_caches_same_buffer: bool,
    non_causal_block: bool = False,
    state_block_size: int,
    compress_eps: float,
    quant_block: int,
    shared_state: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    options = dict(
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        logical_page_size=logical_page_size,
        swa_only=swa_only,
        two_caches_same_buffer=two_caches_same_buffer,
        non_causal_block=non_causal_block,
    )

    def separate(_):
        updated, history = _compressor_jax_hca(
            hidden,
            weight,
            ape,
            norm,
            cs,
            pos,
            st,
            ccu,
            kt,
            cdist,
            main,
            state,
            state_block_size=state_block_size,
            head_dim=512,
            compress_ratio=128,
            overlap=False,
            rms_eps=compress_eps,
            quant_block=quant_block,
        )
        output, sliding = _attention_hca(
            q,
            kv,
            history if shared_state else sw,
            sl,
            sp,
            cu,
            dist,
            updated,
            ml,
            attend,
            mp,
            mcu,
            mdist,
            sinks,
            pos,
            cs,
            eps=eps,
            **options,
        )
        return output, sliding if shared_state else history, updated, sliding

    supported = (
        pos.shape == (1,)
        and pos.dtype == jnp.int32
        and cs.ndim == 2
        and cs.shape[1] == 64
        and cs.dtype == jnp.float32
        and swa_hca_decode.get_block_pages(
            q,
            kv,
            sw,
            main,
            sliding_window=sliding_window,
            logical_page_size=logical_page_size,
            swa_only=swa_only,
            two_caches_same_buffer=two_caches_same_buffer,
            non_causal_block=non_causal_block,
        )
        == 32
        and hca_decode_compressor.is_supported(
            hidden,
            weight,
            main,
            state,
            ccu,
            st,
            kt,
            head_dim=512,
            compress_ratio=128,
            overlap=False,
            state_block_size=state_block_size,
        )
    )
    if not supported:
        return separate(None)
    dense = (
        jnp.all(dist == 1)
        & jnp.all(mdist == 1)
        & jnp.all(cdist == 1)
        & (cu[0] == 0)
        & (cu[1] == 1)
        & (mcu[0] == 0)
        & (mcu[1] == 1)
        & jnp.all(ccu == jnp.minimum(jnp.arange(ccu.size), 1))
        & (pos[0] >= 0)
        & (pos[0] // 32 < st.size // (ccu.size - 1))
        & (pos[0] // 1024 < kt.size // (ccu.size - 1))
    )
    b = (
        q,
        kv,
        sl,
        sp.flatten(),
        cu,
        dist,
        main,
        ml,
        attend,
        mp.flatten(),
        sinks,
        pos,
        cs,
    )
    return jax.lax.cond(
        dense,
        lambda _: hca_compress_decode.run(
            b,
            hidden,
            weight,
            ape,
            norm,
            state,
            st.flatten(),
            sw,
            ccu,
            kt.flatten(),
            eps=eps,
            rms_eps=compress_eps,
            sm_scale=sm_scale,
            shared_state=shared_state,
        ),
        separate,
        None,
    )


def attention_compressor_shared(
    q: jax.Array,
    kv: jax.Array,
    sw: jax.Array,
    sl: jax.Array,
    sp: jax.Array,
    cu: jax.Array,
    dist: jax.Array,
    main: jax.Array,
    ml: jax.Array,
    attend: jax.Array,
    mp: jax.Array,
    mcu: jax.Array,
    mdist: jax.Array,
    sinks: jax.Array,
    pos: jax.Array,
    cs: jax.Array,
    hidden: jax.Array,
    weight: jax.Array,
    ape: jax.Array,
    norm: jax.Array,
    st: jax.Array,
    ccu: jax.Array,
    kt: jax.Array,
    cdist: jax.Array,
    *,
    eps: float,
    sm_scale: float,
    sliding_window: int,
    logical_page_size: int,
    swa_only: bool,
    two_caches_same_buffer: bool,
    non_causal_block: bool = False,
    state_block_size: int,
    compress_eps: float,
    quant_block: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Donate one cache when HCA state and SWA overlay the same allocation."""
    output, _, updated_main, updated_shared = attention_compressor(
        q,
        kv,
        sw,
        sl,
        sp,
        cu,
        dist,
        main,
        ml,
        attend,
        mp,
        mcu,
        mdist,
        sinks,
        pos,
        cs,
        hidden,
        weight,
        ape,
        norm,
        sw,
        st,
        ccu,
        kt,
        cdist,
        shared_state=True,
        eps=eps,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        logical_page_size=logical_page_size,
        swa_only=swa_only,
        two_caches_same_buffer=two_caches_same_buffer,
        non_causal_block=non_causal_block,
        state_block_size=state_block_size,
        compress_eps=compress_eps,
        quant_block=quant_block,
    )
    return output, updated_main, updated_shared
