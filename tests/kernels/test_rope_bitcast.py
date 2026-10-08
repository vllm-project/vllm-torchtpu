import jax
import jax.numpy as jnp
import numpy as np


def test_rope_bitcast_correctness():
    """Verify that 3D bitcast slicing correctly recovers bfloat16 RoPE values."""
    # 1. Configuration
    bkv_sz = 1
    # RoPE slice is 128 bytes (576 - 448 = 128 bytes)
    rope_byte_len = 128
    num_rope_elements = 64  # 128 bytes represents 64 bfloat16 values

    # 2. Initialize mock bkv tensor of shape [bkv_sz, 640]
    # We populate the RoPE slice [448:576] with byte values corresponding to
    # known sequential bfloat16 values: 1.0, 2.0, 3.0, ..., 64.0.
    gt_values = jnp.arange(1.0, num_rope_elements + 1, dtype=jnp.bfloat16)
    gt_bytes = jax.lax.bitcast_convert_type(gt_values, jnp.uint8).reshape(
        1, rope_byte_len
    )

    # Embed it into a mock [bkv_sz, 640] bkv array
    bkv = jnp.zeros((bkv_sz, 640), dtype=jnp.uint8)
    bkv = bkv.at[:, 448:576].set(gt_bytes)

    # Extract the RoPE bytes slice
    rope_bytes = bkv[:, 448:576]
    rope_bytes_reshaped = rope_bytes.reshape(bkv_sz, num_rope_elements, 2)

    # -------------------------------------------------------------------------
    # Method 1: Establish Ground-Truth (GT)
    # Directly combine byte pairs to uint16 and bitcast to bfloat16
    # -------------------------------------------------------------------------
    byte0_u16 = rope_bytes_reshaped[:, :, 0].astype(jnp.uint16)
    byte1_u16 = rope_bytes_reshaped[:, :, 1].astype(jnp.uint16)
    rope_uint16 = byte0_u16 | (byte1_u16 << 8)
    rope_gt = jax.lax.bitcast_convert_type(rope_uint16, jnp.bfloat16)

    # -------------------------------------------------------------------------
    # Method 2: Buggy Manual Reshape Version
    # Reconstructs 32-bit integers, bitcasts, and performs row-major reshape
    # -------------------------------------------------------------------------
    byte0_u32 = rope_bytes_reshaped[:, :, 0].astype(jnp.uint32)
    byte1_u32 = rope_bytes_reshaped[:, :, 1].astype(jnp.uint32)
    rope_uint32 = byte0_u32 | (byte1_u32 << 8)

    # Bitcasting uint32 to bfloat16 yields shape [bkv_sz, 128]
    rope_bitcast_u32 = jax.lax.bitcast_convert_type(rope_uint32, jnp.bfloat16)
    # Reshaped to [bkv_sz, 64, 2] to match Mosaic's 3D bitcast register representation
    rope_bitcast_3d = rope_bitcast_u32.reshape(bkv_sz, num_rope_elements, 2)

    # Buggy Reshape and Slice:
    rope_buggy = rope_bitcast_3d.reshape(bkv_sz, 2, num_rope_elements)[:, 0]

    # -------------------------------------------------------------------------
    # Method 3: Correct Direct Slicing Version
    # Directly indexes the first element of the 3D bitcasted array without
    # transposing/reshaping
    # -------------------------------------------------------------------------
    rope_correct = rope_bitcast_3d[:, :, 0]

    # Assertions
    np.testing.assert_array_equal(np.array(rope_correct), np.array(rope_gt))
    assert not np.array_equal(np.array(rope_buggy), np.array(rope_gt))
