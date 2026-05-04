import torch
import triton
import triton.language as tl


@triton.jit
def _flat_qkv_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    q_scale_ptr,
    q_offset_ptr,
    k_scale_ptr,
    k_offset_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    seq_len: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_c: tl.constexpr,
):
    token_block = tl.program_id(0)
    channel_block = tl.program_id(1)
    offs_m = token_block * block_m + tl.arange(0, block_m)
    offs_c = channel_block * block_c + tl.arange(0, block_c)
    hidden = num_heads * head_dim
    kv_hidden = num_kv_heads * head_dim
    mask = (offs_m[:, None] < seq_len) & (offs_c[None, :] < hidden)

    head_idx = offs_c // head_dim
    dim_idx = offs_c - head_idx * head_dim
    kv_group = num_heads // num_kv_heads
    kv_head_idx = head_idx // kv_group
    kv_c = kv_head_idx * head_dim + dim_idx

    q_offsets = offs_m[:, None] * hidden + offs_c[None, :]
    kv_offsets = offs_m[:, None] * kv_hidden + kv_c[None, :]
    q = tl.load(q_ptr + q_offsets, mask=mask, other=0.0).to(tl.float32)
    k = tl.load(k_ptr + kv_offsets, mask=mask, other=0.0).to(tl.float32)
    v = tl.load(v_ptr + kv_offsets, mask=mask, other=0.0)

    q_scale = tl.load(q_scale_ptr + offs_c, mask=offs_c < hidden, other=1.0).to(
        tl.float32
    )
    q_offset = tl.load(q_offset_ptr + offs_c, mask=offs_c < hidden, other=0.0).to(
        tl.float32
    )
    k_scale = tl.load(k_scale_ptr + offs_c, mask=offs_c < hidden, other=1.0).to(
        tl.float32
    )
    k_offset = tl.load(k_offset_ptr + offs_c, mask=offs_c < hidden, other=0.0).to(
        tl.float32
    )

    out_offsets = offs_m[:, None] * hidden + offs_c[None, :]
    q = (q * q_scale[None, :]).to(tl.bfloat16).to(tl.float32) + q_offset[None, :]
    k = (k * k_scale[None, :]).to(tl.bfloat16).to(tl.float32) + k_offset[None, :]
    tl.store(q_out_ptr + out_offsets, q, mask=mask)
    tl.store(k_out_ptr + out_offsets, k, mask=mask)
    tl.store(v_out_ptr + out_offsets, v, mask=mask)


def fused_flat_qkv(
    q_normed: torch.Tensor,
    k_normed: torch.Tensor,
    v: torch.Tensor,
    q_scale: torch.Tensor,
    q_offset: torch.Tensor,
    k_scale: torch.Tensor,
    k_offset: torch.Tensor,
):
    batch_size, seq_len, num_heads, head_dim = q_normed.shape
    assert batch_size == 1
    num_kv_heads = k_normed.shape[2]
    hidden = num_heads * head_dim
    out_shape = (1, seq_len, hidden)
    q_out = torch.empty(out_shape, device=q_normed.device, dtype=q_normed.dtype)
    k_out = torch.empty(out_shape, device=k_normed.device, dtype=k_normed.dtype)
    v_out = torch.empty(out_shape, device=v.device, dtype=v.dtype)
    block_m = 16
    block_c = 512
    _flat_qkv_kernel[(triton.cdiv(seq_len, block_m), triton.cdiv(hidden, block_c))](
        q_normed,
        k_normed,
        v,
        q_scale,
        q_offset,
        k_scale,
        k_offset,
        q_out,
        k_out,
        v_out,
        seq_len,
        num_heads,
        num_kv_heads,
        head_dim,
        block_m,
        block_c,
        num_warps=8,
    )
    return q_out, k_out, v_out
