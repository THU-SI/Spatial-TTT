import torch
import triton
import triton.language as tl


@triton.jit
def _silu(x):
    return x / (1.0 + tl.exp(-x))


@triton.jit
def _prepare_qk_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    out_q_ptr,
    out_k_ptr,
    seq_len: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_d: tl.constexpr,
):
    head_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    offs_m = block_idx * block_m + tl.arange(0, block_m)
    offs_d = tl.arange(0, block_d)
    mask = (offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim)

    in_offsets = (
        offs_m[:, None] * (num_heads * head_dim)
        + head_idx * head_dim
        + offs_d[None, :]
    )
    q = tl.load(q_ptr + in_offsets, mask=mask, other=0.0).to(tl.float32)
    k = tl.load(k_ptr + in_offsets, mask=mask, other=0.0).to(tl.float32)

    q = _silu(q).to(tl.bfloat16).to(tl.float32)
    k = _silu(k).to(tl.bfloat16).to(tl.float32)

    q_norm = tl.sqrt(tl.sum(q * q, axis=1))[:, None]
    k_norm = tl.sqrt(tl.sum(k * k, axis=1))[:, None]
    q = (q / (q_norm + 1.0e-5)).to(tl.bfloat16).to(tl.float32)
    k = (k / (k_norm + 1.0e-5)).to(tl.bfloat16).to(tl.float32)

    half_rope = rope_dim // 2
    shifted_d = tl.where(offs_d < half_rope, offs_d + half_rope, offs_d - half_rope)
    shifted_offsets = (
        offs_m[:, None] * (num_heads * head_dim)
        + head_idx * head_dim
        + shifted_d[None, :]
    )
    q_shift = tl.load(
        q_ptr + shifted_offsets,
        mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < rope_dim),
        other=0.0,
    ).to(tl.float32)
    k_shift = tl.load(
        k_ptr + shifted_offsets,
        mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < rope_dim),
        other=0.0,
    ).to(tl.float32)
    q_shift = (_silu(q_shift).to(tl.bfloat16).to(tl.float32) / (q_norm + 1.0e-5)).to(
        tl.bfloat16
    ).to(tl.float32)
    k_shift = (_silu(k_shift).to(tl.bfloat16).to(tl.float32) / (k_norm + 1.0e-5)).to(
        tl.bfloat16
    ).to(tl.float32)

    rot_sign = tl.where(offs_d < half_rope, -1.0, 1.0)
    cos_sin_offsets = offs_m[:, None] * rope_dim + offs_d[None, :]
    cos = tl.load(
        cos_ptr + cos_sin_offsets,
        mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < rope_dim),
        other=1.0,
    ).to(tl.float32)
    sin = tl.load(
        sin_ptr + cos_sin_offsets,
        mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < rope_dim),
        other=0.0,
    ).to(tl.float32)
    q_rope = q * cos + (q_shift * rot_sign[None, :]) * sin
    k_rope = k * cos + (k_shift * rot_sign[None, :]) * sin
    q = tl.where(offs_d[None, :] < rope_dim, q_rope, q)
    k = tl.where(offs_d[None, :] < rope_dim, k_rope, k)

    out_offsets = (
        head_idx * seq_len * head_dim
        + offs_m[:, None] * head_dim
        + offs_d[None, :]
    )
    tl.store(out_q_ptr + out_offsets, q, mask=mask)
    tl.store(out_k_ptr + out_offsets, k, mask=mask)


def fused_prepare_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
):
    batch_size, seq_len, hidden_size = q.shape
    assert batch_size == 1
    head_dim = hidden_size // num_heads
    rope_dim = cos.shape[-1]
    out_shape = (num_heads, seq_len, head_dim)
    out_q = torch.empty(out_shape, device=q.device, dtype=q.dtype)
    out_k = torch.empty(out_shape, device=k.device, dtype=k.dtype)
    block_m = 16
    _prepare_qk_kernel[(num_heads, triton.cdiv(seq_len, block_m))](
        q,
        k,
        cos,
        sin,
        out_q,
        out_k,
        seq_len,
        num_heads,
        head_dim,
        rope_dim,
        block_m,
        triton.next_power_of_2(head_dim),
        num_warps=8,
    )
    return out_q, out_k
