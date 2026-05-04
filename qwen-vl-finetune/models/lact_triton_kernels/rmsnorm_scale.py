import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_scale_rearrange_kernel(
    x_ptr,
    weight_ptr,
    scale_ptr,
    out_ptr,
    seq_len: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    eps: tl.constexpr,
    block_m: tl.constexpr,
    block_d: tl.constexpr,
):
    head_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    offs_m = block_idx * block_m + tl.arange(0, block_m)
    offs_d = tl.arange(0, block_d)
    mask = (offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim)

    x_offsets = (
        head_idx * seq_len * head_dim
        + offs_m[:, None] * head_dim
        + offs_d[None, :]
    )
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=1)[:, None] / head_dim
    x = x * tl.rsqrt(variance + eps)

    weight = tl.load(weight_ptr + offs_d, mask=offs_d < head_dim, other=0.0).to(
        tl.float32
    )
    scale = tl.load(
        scale_ptr + head_idx * seq_len + offs_m,
        mask=offs_m < seq_len,
        other=0.0,
    ).to(tl.float32)
    x = x * weight[None, :] * scale[:, None]

    out_offsets = (
        offs_m[:, None] * (num_heads * head_dim)
        + head_idx * head_dim
        + offs_d[None, :]
    )
    tl.store(out_ptr + out_offsets, x, mask=mask)


def fused_rmsnorm_scale_rearrange(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
):
    num_heads, seq_len, head_dim = x.shape
    out = torch.empty((1, seq_len, num_heads * head_dim), device=x.device, dtype=x.dtype)
    block_m = 16
    _rmsnorm_scale_rearrange_kernel[(num_heads, triton.cdiv(seq_len, block_m))](
        x,
        weight,
        scale,
        out,
        seq_len,
        num_heads,
        head_dim,
        eps,
        block_m,
        triton.next_power_of_2(head_dim),
        num_warps=8,
    )
    return out
