# Adapted from https://github.com/a1600012888/LaCT/blob/main/lact_llm/lact_model/layer_lact_swiglu.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .ttt_operation import (
    block_causal_lact_swiglu,
    l2_norm,
    prenorm_block_causal_lact_swiglu,
    silu_backprop,
    zeropower_via_newtonschulz5,
)

from .ttt_operation_fused_kernel import (
    postnorm_block_causal_lact_swiglu_fused_kernel_triton,
    prenorm_block_causal_lact_swiglu_fused_kernel_triton,
)


def _find_video_segments(mask):
    segments = []
    mask_np = mask.cpu().numpy()

    in_segment = False
    start = 0

    for i, is_video in enumerate(mask_np):
        if is_video and not in_segment:
            start = i
            in_segment = True
        elif not is_video and in_segment:
            segments.append((start, i))
            in_segment = False

    if in_segment:
        segments.append((start, len(mask_np)))

    return segments


def inv_softplus(x):
    if isinstance(x, torch.Tensor):
        y = x + torch.log(-torch.expm1(-x))
    else:
        y = x + math.log(-math.expm1(-x))
    return y


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_partial_rotary_pos_emb(q, k, cos, sin, rope_dim):
    fw_head_num = q.shape[0] // cos.shape[0]
    q = rearrange(q, "(b h) ... -> b h ...", h=fw_head_num)
    k = rearrange(k, "(b h) ... -> b h ...", h=fw_head_num)
    q_rope = q[..., :rope_dim]
    q_nope = q[..., rope_dim:]
    k_rope = k[..., :rope_dim]
    k_nope = k[..., rope_dim:]

    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    q_rope_embed = (q_rope * cos) + (rotate_half(q_rope) * sin)
    k_rope_embed = (k_rope * cos) + (rotate_half(k_rope) * sin)

    q_embed = torch.cat([q_rope_embed, q_nope], dim=-1)
    k_embed = torch.cat([k_rope_embed, k_nope], dim=-1)

    q_embed = rearrange(q_embed, "b h s d -> (b h) s d")
    k_embed = rearrange(k_embed, "b h s d -> (b h) s d")

    return q_embed, k_embed


@dataclass
class LaCTLayerState:
    w0: torch.Tensor
    w1: torch.Tensor
    w2: torch.Tensor

    w0_norm: torch.Tensor
    w1_norm: torch.Tensor
    w2_norm: torch.Tensor

    dw0_momentum: Optional[torch.Tensor] = None
    dw1_momentum: Optional[torch.Tensor] = None
    dw2_momentum: Optional[torch.Tensor] = None

    pending_k: Optional[torch.Tensor] = None  # [num_fw_heads, pending_len, d]
    pending_v: Optional[torch.Tensor] = None
    pending_lr0: Optional[torch.Tensor] = None
    pending_lr1: Optional[torch.Tensor] = None
    pending_lr2: Optional[torch.Tensor] = None
    pending_momentum: Optional[torch.Tensor] = None


class LaCTCache:
    def __init__(self):
        self._layer_states: Dict[int, LaCTLayerState] = {}

    def has_layer(self, layer_idx: int) -> bool:
        return layer_idx in self._layer_states

    def get_layer_state(self, layer_idx: int) -> LaCTLayerState:
        return self._layer_states[layer_idx]

    def set_layer_state(self, layer_idx: int, state: LaCTLayerState):
        self._layer_states[layer_idx] = state

    def get_pending_length(self, layer_idx: int) -> int:
        if layer_idx not in self._layer_states:
            return 0
        state = self._layer_states[layer_idx]
        if state.pending_k is None:
            return 0
        return state.pending_k.shape[1]

    def reset(self):
        self._layer_states.clear()


def _get_kv_cache_layer(past_key_values, layer_idx: int):
    if past_key_values is None:
        return None, None

    if hasattr(past_key_values, "layers"):
        layers = past_key_values.layers
        if layer_idx >= len(layers):
            return None, None
        layer = layers[layer_idx]
        return getattr(layer, "keys", None), getattr(layer, "values", None)

    key_cache = getattr(past_key_values, "key_cache", None)
    value_cache = getattr(past_key_values, "value_cache", None)
    if key_cache is None or value_cache is None or layer_idx >= len(key_cache):
        return None, None
    return key_cache[layer_idx], value_cache[layer_idx]


def _set_kv_cache_layer(past_key_values, layer_idx: int, key_states, value_states):
    if past_key_values is None:
        return

    if hasattr(past_key_values, "layers"):
        layer = past_key_values.layers[layer_idx]
        layer.keys = key_states
        layer.values = value_states
        return

    key_cache = getattr(past_key_values, "key_cache", None)
    value_cache = getattr(past_key_values, "value_cache", None)
    if key_cache is not None and value_cache is not None:
        while len(key_cache) < layer_idx:
            key_cache.append(torch.tensor([]))
            value_cache.append(torch.tensor([]))
        if len(key_cache) == layer_idx:
            key_cache.append(key_states)
            value_cache.append(value_states)
        else:
            key_cache[layer_idx] = key_states
            value_cache[layer_idx] = value_states


def _truncate_kv_cache_layer(
    past_key_values,
    layer_idx: int,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    window_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if key_states.shape[2] <= window_size:
        return key_states, value_states

    key_states = key_states[:, :, -window_size:, :]
    value_states = value_states[:, :, -window_size:, :]
    _set_kv_cache_layer(past_key_values, layer_idx, key_states, value_states)
    return key_states, value_states


class LowRankFastWeight(nn.Module):
    """
    Low rank fast weight parameterization.

    This is a compromise to keep the number of parameters low when comparing against baselines.
    Ideally, low-rank parameterization always hurts the performance.

    W = W_left @ W_right + I * 0.5 (if add_identity is True)
    """

    def __init__(
        self,
        num_heads: int,
        out_features: int,
        in_features: int,
        rank: int = 32,
        init_gain: float = 0.5,
        add_identity: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.in_features = in_features
        self.rank = rank
        self.add_identity = add_identity
        self.init_gain = init_gain

        # Initialize with proper scaling
        self.w_left = nn.Parameter(torch.empty(num_heads, out_features, rank))
        self.w_right = nn.Parameter(torch.empty(num_heads, rank, in_features))
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.w_left, std=1.0 / math.sqrt(self.rank) * self.init_gain)
        nn.init.normal_(
            self.w_right, std=1.0 / math.sqrt(self.in_features) * self.init_gain
        )

    def forward(self):
        W = self.w_left @ self.w_right
        if self.add_identity:
            W = W + (
                torch.eye(
                    self.out_features, self.in_features, device=W.device, dtype=W.dtype
                ).unsqueeze(0)
                * 0.5
            )
        return W


class Qwen3VLLaCTSWIGLULayer(nn.Module):
    def __init__(
        self,
        attn_layer,
        num_lact_heads: int,
        inter_multi: float = 1.0,
        window_size: int = 2560,
        lact_chunk_size: int = 2560,
        qkv_silu: bool = True,
        no_v_silu: bool = False,
        lr_dim: int = 1,
        use_muon: bool = False,
        lr_parameterization: str = "mamba",
        learnable_ttt_scale: bool = True,
        ttt_prenorm: bool = False,
        ttt_nope: bool = False,
        w0_w2_low_rank: int = -1,
        use_momentum: bool = True,
        fw_init_gain: float = 0.5,
        use_fused_kernel: bool = True,
        fp32_states: bool = True,
        use_conv_layer: bool = False,
    ):
        super().__init__()
        self.attn_layer = attn_layer

        self.config = attn_layer.config
        self.hidden_size = self.config.hidden_size
        self.num_attn_heads = self.config.num_attention_heads
        self.num_kv_heads = self.config.num_key_value_heads
        self.head_dim = attn_layer.head_dim
        self.layer_idx = attn_layer.layer_idx
        self.scaling = attn_layer.scaling

        self.window_size = window_size
        self.lact_chunk_size = lact_chunk_size
        self.num_fw_heads = num_lact_heads
        self.fw_head_dim = self.hidden_size // self.num_fw_heads
        self.inter_multi = inter_multi
        self.qkv_silu = qkv_silu
        self.no_v_silu = no_v_silu
        self.ttt_prenorm = ttt_prenorm
        self.ttt_nope = ttt_nope
        self.use_muon = use_muon
        self.use_momentum = use_momentum
        self.use_fused_kernel = use_fused_kernel
        self.fp32_states = fp32_states
        self.use_conv_layer = use_conv_layer

        d_in = self.fw_head_dim
        d_out = self.fw_head_dim
        d_h = int(d_in * inter_multi)
        self.d_h = d_h
        self.d_in = d_in
        self.d_out = d_out
        self.w0_w2_low_rank = w0_w2_low_rank
        self.fw_init_gain = fw_init_gain

        if self.w0_w2_low_rank > 0:
            self.w0 = LowRankFastWeight(
                self.num_fw_heads,
                d_h,
                d_in,
                self.w0_w2_low_rank,
                init_gain=self.fw_init_gain,
                add_identity=True,
            )
            self.w2 = LowRankFastWeight(
                self.num_fw_heads,
                d_h,
                d_in,
                self.w0_w2_low_rank,
                init_gain=self.fw_init_gain,
                add_identity=True,
            )
        else:
            self.w0 = nn.Parameter(
                torch.randn(self.num_fw_heads, d_h, d_in) / math.sqrt(d_in)
            )
            self.w2 = nn.Parameter(
                torch.randn(self.num_fw_heads, d_h, d_in) / math.sqrt(d_in)
            )
        self.w1 = nn.Parameter(
            torch.randn(self.num_fw_heads, d_out, d_h) / math.sqrt(d_h)
        )

        self.lr_dim = int(lr_dim * 3 * self.num_fw_heads)
        self.lr_proj = nn.Linear(self.hidden_size, self.lr_dim)
        base_lr = 0.001
        self.lr_parameterization = lr_parameterization
        if lr_parameterization.lower() == "mamba":
            self.base_lr_inv = inv_softplus(base_lr)

        self.q_dim = self.num_attn_heads * self.head_dim
        self.k_dim = self.num_attn_heads * self.head_dim  # after expansion
        self.q_scale = nn.Parameter(torch.ones(self.q_dim))
        self.q_offset = nn.Parameter(torch.zeros(self.q_dim))
        self.k_scale = nn.Parameter(torch.ones(self.k_dim))
        self.k_offset = nn.Parameter(torch.zeros(self.k_dim))

        self.learnable_ttt_scale = learnable_ttt_scale
        if self.learnable_ttt_scale:
            self.ttt_scale_proj = nn.Linear(self.hidden_size, self.num_fw_heads)
            # init ttt scale as 0
            nn.init.zeros_(self.ttt_scale_proj.weight)
            nn.init.zeros_(self.ttt_scale_proj.bias)

        self.ttt_norm = nn.RMSNorm(self.fw_head_dim, eps=1e-5, elementwise_affine=True)

        if self.use_momentum:
            self.momentum_proj = nn.Sequential(
                nn.Linear(self.hidden_size, self.num_fw_heads),
                nn.Sigmoid(),
            )

        if self.use_conv_layer:
            # self.conv_layer = nn.Sequential(
            #     nn.Conv3d(self.hidden_size, self.hidden_size // 16, 1, bias=False),
            #     nn.GELU(),
            #     nn.Conv3d(
            #         self.hidden_size // 16,
            #         self.hidden_size // 16,
            #         3,
            #         padding=1,
            #         padding_mode="replicate",
            #         bias=False,
            #     ),
            #     nn.GELU(),
            #     nn.Conv3d(self.hidden_size // 16, self.hidden_size, 1, bias=False),
            # )
            # nn.init.zeros_(self.conv_layer[-1].weight)
            self.conv_q = nn.Conv3d(
                self.hidden_size,
                self.hidden_size,
                kernel_size=3,
                padding=1,
                groups=self.hidden_size,
                padding_mode="replicate",
                bias=False,
            )
            self.conv_k = nn.Conv3d(
                self.hidden_size,
                self.hidden_size,
                kernel_size=3,
                padding=1,
                groups=self.hidden_size,
                padding_mode="replicate",
                bias=False,
            )
            self.conv_v = nn.Conv3d(
                self.hidden_size,
                self.hidden_size,
                kernel_size=3,
                padding=1,
                groups=self.hidden_size,
                padding_mode="replicate",
                bias=False,
            )
            nn.init.dirac_(self.conv_q.weight, groups=self.hidden_size)
            nn.init.dirac_(self.conv_k.weight, groups=self.hidden_size)
            nn.init.dirac_(self.conv_v.weight, groups=self.hidden_size)

    def _rescale_qk(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = q * self.q_scale.view(1, 1, -1) + self.q_offset.view(1, 1, -1)
        k = k * self.k_scale.view(1, 1, -1) + self.k_offset.view(1, 1, -1)
        return q, k

    def _run_ttt_kernel(
        self,
        fw_w0: torch.Tensor,
        fw_w1: torch.Tensor,
        fw_w2: torch.Tensor,
        fast_q: torch.Tensor,
        fast_k: torch.Tensor,
        fast_v: torch.Tensor,
        fw_lr1: torch.Tensor,
        fw_lr2: torch.Tensor,
        fw_lr3: torch.Tensor,
        momentum: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.ttt_prenorm:
            if self.use_fused_kernel:
                return prenorm_block_causal_lact_swiglu_fused_kernel_triton(
                    fw_w0,
                    fw_w1,
                    fw_w2,
                    fast_q,
                    fast_k,
                    fast_v,
                    fw_lr1,
                    fw_lr2,
                    fw_lr3,
                    chunk_size=self.lact_chunk_size,
                    use_muon=self.use_muon,
                    momentum=momentum,
                )
            return prenorm_block_causal_lact_swiglu(
                fw_w0,
                fw_w1,
                fw_w2,
                fast_q,
                fast_k,
                fast_v,
                fw_lr1,
                fw_lr2,
                fw_lr3,
                chunk_size=self.lact_chunk_size,
                use_muon=self.use_muon,
                momentum=momentum,
            )
        else:
            if self.use_fused_kernel:
                return postnorm_block_causal_lact_swiglu_fused_kernel_triton(
                    fw_w0,
                    fw_w1,
                    fw_w2,
                    fast_q,
                    fast_k,
                    fast_v,
                    fw_lr1,
                    fw_lr2,
                    fw_lr3,
                    chunk_size=self.lact_chunk_size,
                    use_muon=self.use_muon,
                    momentum=momentum,
                )
            return block_causal_lact_swiglu(
                fw_w0,
                fw_w1,
                fw_w2,
                fast_q,
                fast_k,
                fast_v,
                fw_lr1,
                fw_lr2,
                fw_lr3,
                chunk_size=self.lact_chunk_size,
                use_muon=self.use_muon,
                momentum=momentum,
            )

    def _compute_ttt_output(
        self,
        hidden_states: torch.Tensor,
        fast_q: torch.Tensor,
        fast_k: torch.Tensor,
        fast_v: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.size()

        fast_q = rearrange(fast_q, "b s (h d) -> (b h) s d", h=self.num_fw_heads)
        fast_k = rearrange(fast_k, "b s (h d) -> (b h) s d", h=self.num_fw_heads)
        fast_v = rearrange(fast_v, "b s (h d) -> (b h) s d", h=self.num_fw_heads)

        if self.qkv_silu:
            fast_q = F.silu(fast_q)
            fast_k = F.silu(fast_k)
            if not self.no_v_silu:
                fast_v = F.silu(fast_v)

        fast_q = l2_norm(fast_q)
        fast_k = l2_norm(fast_k)

        if not self.ttt_nope:
            cos, sin = position_embeddings
            fast_q, fast_k = apply_partial_rotary_pos_emb(
                fast_q, fast_k, cos, sin, rope_dim=self.head_dim
            )

        lr = self.lr_proj(hidden_states)
        if self.lr_parameterization == "mamba":
            lr = F.softplus(lr.float() + self.base_lr_inv)
        else:
            raise NotImplementedError(
                f"LR parameterization {self.lr_parameterization} not implemented"
            )

        fw_lr = rearrange(lr, "b s (h d) -> (b h) s d", h=self.num_fw_heads)
        fw_lr1, fw_lr2, fw_lr3 = fw_lr.chunk(3, dim=-1)

        if self.use_momentum:
            momentum = self.momentum_proj(hidden_states).float()
            momentum = rearrange(
                momentum, "b s (h d) -> (b h) s d", h=self.num_fw_heads
            )
        else:
            momentum = None

        if cu_seqlens is not None and batch_size == 1:
            num_seqs = cu_seqlens.shape[0] - 1

            if self.w0_w2_low_rank > 0:
                base_w0 = self.w0()  # [num_fw_heads, d_h, d_in]
                base_w2 = self.w2()
            else:
                base_w0 = self.w0
                base_w2 = self.w2
            base_w1 = self.w1

            if self.fp32_states:
                base_w0 = base_w0.to(torch.float32)
                base_w1 = base_w1.to(torch.float32)
                base_w2 = base_w2.to(torch.float32)

            fw_x_list = []
            for i in range(num_seqs):
                start_idx = cu_seqlens[i].item()
                end_idx = cu_seqlens[i + 1].item()

                fast_q_i = fast_q[:, start_idx:end_idx, :]
                fast_k_i = fast_k[:, start_idx:end_idx, :]
                fast_v_i = fast_v[:, start_idx:end_idx, :]
                fw_lr1_i = fw_lr1[:, start_idx:end_idx, :]
                fw_lr2_i = fw_lr2[:, start_idx:end_idx, :]
                fw_lr3_i = fw_lr3[:, start_idx:end_idx, :]
                momentum_i = (
                    momentum[:, start_idx:end_idx, :] if momentum is not None else None
                )

                fw_w0_i = base_w0.clone()
                fw_w1_i = base_w1.clone()
                fw_w2_i = base_w2.clone()

                fw_x_i = self._run_ttt_kernel(
                    fw_w0_i,
                    fw_w1_i,
                    fw_w2_i,
                    fast_q_i,
                    fast_k_i,
                    fast_v_i,
                    fw_lr1_i,
                    fw_lr2_i,
                    fw_lr3_i,
                    momentum_i,
                )
                fw_x_list.append(fw_x_i)

            fw_x = torch.cat(fw_x_list, dim=1)
        else:
            if self.w0_w2_low_rank > 0:
                fw_w0 = self.w0().repeat(batch_size, 1, 1)
                fw_w2 = self.w2().repeat(batch_size, 1, 1)
            else:
                fw_w0 = self.w0.repeat(batch_size, 1, 1)
                fw_w2 = self.w2.repeat(batch_size, 1, 1)
            fw_w1 = self.w1.repeat(batch_size, 1, 1)

            if self.fp32_states:
                fw_w0 = fw_w0.to(torch.float32)
                fw_w1 = fw_w1.to(torch.float32)
                fw_w2 = fw_w2.to(torch.float32)

            fw_x = self._run_ttt_kernel(
                fw_w0,
                fw_w1,
                fw_w2,
                fast_q,
                fast_k,
                fast_v,
                fw_lr1,
                fw_lr2,
                fw_lr3,
                momentum,
            )

        ttt_x_normed = self.ttt_norm(fw_x)

        if self.learnable_ttt_scale:
            ttt_scale = F.silu(self.ttt_scale_proj(hidden_states), inplace=False)
            ttt_scale = rearrange(
                ttt_scale, "b s (n d) -> (b n) s d", n=self.num_fw_heads
            )
            ttt_x_normed = ttt_x_normed * ttt_scale

        ttt_output = rearrange(
            ttt_x_normed,
            "(b n) s d -> b s (n d)",
            n=self.num_fw_heads,
            b=batch_size,
        )

        return ttt_output

    def _init_cache_from_prefill(
        self,
        hidden_states: torch.Tensor,
        fast_q: torch.Tensor,
        fast_k: torch.Tensor,
        fast_v: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        lact_cache: LaCTCache,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.size()
        assert batch_size == 1, "Decode cache only supports batch_size=1"

        fast_q = rearrange(fast_q, "b s (h d) -> (b h) s d", h=self.num_fw_heads)
        fast_k = rearrange(fast_k, "b s (h d) -> (b h) s d", h=self.num_fw_heads)
        fast_v = rearrange(fast_v, "b s (h d) -> (b h) s d", h=self.num_fw_heads)

        if self.qkv_silu:
            fast_q = F.silu(fast_q)
            fast_k = F.silu(fast_k)
            if not self.no_v_silu:
                fast_v = F.silu(fast_v)

        fast_q = l2_norm(fast_q)
        fast_k = l2_norm(fast_k)

        if not self.ttt_nope:
            cos, sin = position_embeddings
            fast_q, fast_k = apply_partial_rotary_pos_emb(
                fast_q, fast_k, cos, sin, rope_dim=self.head_dim
            )

        lr = self.lr_proj(hidden_states)
        if self.lr_parameterization == "mamba":
            lr = F.softplus(lr.float() + self.base_lr_inv)
        fw_lr = rearrange(lr, "b s (h d) -> (b h) s d", h=self.num_fw_heads)
        fw_lr1, fw_lr2, fw_lr3 = fw_lr.chunk(3, dim=-1)

        if self.use_momentum:
            momentum = self.momentum_proj(hidden_states).float()
            momentum = rearrange(
                momentum, "b s (h d) -> (b h) s d", h=self.num_fw_heads
            )
        else:
            momentum = None

        if self.w0_w2_low_rank > 0:
            fw_w0 = self.w0()  # [num_fw_heads, d_h, d_in]
            fw_w2 = self.w2()
        else:
            fw_w0 = self.w0.clone()
            fw_w2 = self.w2.clone()
        fw_w1 = self.w1.clone()

        if self.fp32_states:
            fw_w0 = fw_w0.to(torch.float32)
            fw_w1 = fw_w1.to(torch.float32)
            fw_w2 = fw_w2.to(torch.float32)

        w0_norm = fw_w0.norm(dim=2, keepdim=True)
        w1_norm = fw_w1.norm(dim=2, keepdim=True)
        w2_norm = fw_w2.norm(dim=2, keepdim=True)

        if momentum is not None:
            dw0_momentum = torch.zeros_like(fw_w0)
            dw1_momentum = torch.zeros_like(fw_w1)
            dw2_momentum = torch.zeros_like(fw_w2)
        else:
            dw0_momentum = None
            dw1_momentum = None
            dw2_momentum = None

        q_t = fast_q.transpose(1, 2)
        v_t = fast_v.transpose(1, 2)
        output = torch.zeros_like(v_t)

        if self.fp32_states:
            q_t = q_t.float()
            v_t = v_t.float()
            fast_k = fast_k.float()
            output = output.float()

        chunk_size = self.lact_chunk_size
        e_index = 0

        for i in range(0, seq_len - chunk_size, chunk_size):
            s_index = i
            e_index = s_index + chunk_size

            ki = fast_k[:, s_index:e_index, :]
            vi = v_t[:, :, s_index:e_index]
            qi = q_t[:, :, s_index:e_index]
            lr0i = fw_lr1[:, s_index:e_index, :]
            lr1i = fw_lr2[:, s_index:e_index, :]
            lr2i = fw_lr3[:, s_index:e_index, :]

            h = torch.bmm(fw_w2, qi)
            gate = F.silu(torch.bmm(fw_w0, qi), inplace=True)
            output[:, :, s_index:e_index] = torch.bmm(fw_w1, gate * h)

            gate_before_act = torch.bmm(fw_w0, ki.transpose(1, 2))
            hidden_before_mul = torch.bmm(fw_w2, ki.transpose(1, 2))
            hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul
            dhidden = torch.bmm(fw_w1.transpose(1, 2), vi)
            dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
            dgate = dhidden * hidden_before_mul
            dgate_before_act = silu_backprop(dgate, gate_before_act)

            dw1 = torch.bmm(vi, (hidden.transpose(1, 2) * lr1i).type_as(vi))
            dw0 = torch.bmm(dgate_before_act, (ki * lr0i).type_as(dgate_before_act))
            dw2 = torch.bmm(dhidden_before_mul, (ki * lr2i).type_as(dhidden_before_mul))

            if momentum is not None:
                m_i = momentum[:, s_index:e_index, :].mean(dim=1, keepdim=True)
                dw0 = dw0 + dw0_momentum * m_i
                dw1 = dw1 + dw1_momentum * m_i
                dw2 = dw2 + dw2_momentum * m_i
                dw0_momentum = dw0
                dw1_momentum = dw1
                dw2_momentum = dw2

            if self.use_muon:
                dw0 = zeropower_via_newtonschulz5(dw0)
                dw1 = zeropower_via_newtonschulz5(dw1)
                dw2 = zeropower_via_newtonschulz5(dw2)

            fw_w0 = fw_w0 + dw0
            fw_w1 = fw_w1 + dw1
            fw_w2 = fw_w2 + dw2

            fw_w0 = fw_w0 / (fw_w0.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
            fw_w1 = fw_w1 / (fw_w1.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
            fw_w2 = fw_w2 / (fw_w2.norm(dim=2, keepdim=True) + 1e-5) * w2_norm

        # process remaining tokens (last chunk)
        s_index = e_index
        e_index = seq_len
        remaining_len = e_index - s_index

        if remaining_len > 0:
            qi = q_t[:, :, s_index:e_index]
            h = torch.bmm(fw_w2, qi)
            gate = F.silu(torch.bmm(fw_w0, qi), inplace=True)
            output[:, :, s_index:e_index] = torch.bmm(fw_w1, gate * h)

        fw_x = output.transpose(1, 2)

        if remaining_len > 0:
            pending_k = fast_k[:, s_index:e_index, :]
            pending_v = fast_v[:, s_index:e_index, :]
            pending_lr0 = fw_lr1[:, s_index:e_index, :]
            pending_lr1 = fw_lr2[:, s_index:e_index, :]
            pending_lr2 = fw_lr3[:, s_index:e_index, :]
            pending_momentum = (
                momentum[:, s_index:e_index, :] if momentum is not None else None
            )
        else:
            pending_k = None
            pending_v = None
            pending_lr0 = None
            pending_lr1 = None
            pending_lr2 = None
            pending_momentum = None

        state = LaCTLayerState(
            w0=fw_w0,
            w1=fw_w1,
            w2=fw_w2,
            w0_norm=w0_norm,
            w1_norm=w1_norm,
            w2_norm=w2_norm,
            dw0_momentum=dw0_momentum,
            dw1_momentum=dw1_momentum,
            dw2_momentum=dw2_momentum,
            pending_k=pending_k,
            pending_v=pending_v,
            pending_lr0=pending_lr0,
            pending_lr1=pending_lr1,
            pending_lr2=pending_lr2,
            pending_momentum=pending_momentum,
        )
        lact_cache.set_layer_state(self.layer_idx, state)

        ttt_x_normed = self.ttt_norm(fw_x)
        if self.learnable_ttt_scale:
            ttt_scale = F.silu(self.ttt_scale_proj(hidden_states), inplace=False)
            ttt_scale = rearrange(
                ttt_scale, "b s (n_h d) -> (b n_h) s d", n_h=self.num_fw_heads
            )
            ttt_x_normed = ttt_x_normed * ttt_scale

        ttt_output = rearrange(
            ttt_x_normed,
            "(b n_h) s d -> b s (n_h d)",
            n_h=self.num_fw_heads,
            b=batch_size,
        )
        return ttt_output.type_as(hidden_states)

    def _compute_ttt_output_decode(
        self,
        hidden_states: torch.Tensor,
        fast_q: torch.Tensor,
        fast_k: torch.Tensor,
        fast_v: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        lact_cache: LaCTCache,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.size()
        assert batch_size == 1, "Decode cache only supports batch_size=1"
        assert seq_len == 1, "Decode should process one token at a time"

        state = lact_cache.get_layer_state(self.layer_idx)

        fast_q = rearrange(fast_q, "b s (h d) -> (b h) s d", h=self.num_fw_heads)
        fast_k = rearrange(fast_k, "b s (h d) -> (b h) s d", h=self.num_fw_heads)
        fast_v = rearrange(fast_v, "b s (h d) -> (b h) s d", h=self.num_fw_heads)

        if self.qkv_silu:
            fast_q = F.silu(fast_q)
            fast_k = F.silu(fast_k)
            if not self.no_v_silu:
                fast_v = F.silu(fast_v)

        fast_q = l2_norm(fast_q)
        fast_k = l2_norm(fast_k)

        if not self.ttt_nope:
            cos, sin = position_embeddings
            fast_q, fast_k = apply_partial_rotary_pos_emb(
                fast_q, fast_k, cos, sin, rope_dim=self.head_dim
            )

        lr = self.lr_proj(hidden_states)
        if self.lr_parameterization == "mamba":
            lr = F.softplus(lr.float() + self.base_lr_inv)
        fw_lr = rearrange(lr, "b s (h d) -> (b h) s d", h=self.num_fw_heads)
        fw_lr1, fw_lr2, fw_lr3 = fw_lr.chunk(3, dim=-1)

        if self.use_momentum:
            momentum = self.momentum_proj(hidden_states).float()
            momentum = rearrange(
                momentum, "b s (h d) -> (b h) s d", h=self.num_fw_heads
            )
        else:
            momentum = None

        fw_w0 = state.w0
        fw_w1 = state.w1
        fw_w2 = state.w2

        q_t = fast_q.transpose(1, 2)  # [num_fw_heads, d, 1]
        if self.fp32_states:
            q_t = q_t.float()
            fast_k = fast_k.float()
            fast_v = fast_v.float()
        h = torch.bmm(fw_w2, q_t)
        gate = F.silu(torch.bmm(fw_w0, q_t), inplace=True)
        output = torch.bmm(fw_w1, gate * h)  # [num_fw_heads, d, 1]
        fw_x = output.transpose(1, 2)  # [num_fw_heads, 1, d]

        if state.pending_k is None:
            state.pending_k = fast_k
            state.pending_v = fast_v
            state.pending_lr0 = fw_lr1
            state.pending_lr1 = fw_lr2
            state.pending_lr2 = fw_lr3
            state.pending_momentum = momentum
        else:
            assert state.pending_v is not None
            assert state.pending_lr0 is not None
            assert state.pending_lr1 is not None
            assert state.pending_lr2 is not None
            state.pending_k = torch.cat([state.pending_k, fast_k], dim=1)
            state.pending_v = torch.cat([state.pending_v, fast_v], dim=1)
            state.pending_lr0 = torch.cat([state.pending_lr0, fw_lr1], dim=1)
            state.pending_lr1 = torch.cat([state.pending_lr1, fw_lr2], dim=1)
            state.pending_lr2 = torch.cat([state.pending_lr2, fw_lr3], dim=1)
            if momentum is not None and state.pending_momentum is not None:
                state.pending_momentum = torch.cat(
                    [state.pending_momentum, momentum], dim=1
                )

        # check if need to update fast weights
        pending_len = state.pending_k.shape[1]
        if pending_len >= self.lact_chunk_size:
            ki = state.pending_k[:, : self.lact_chunk_size, :]
            vi = state.pending_v[:, : self.lact_chunk_size, :].transpose(1, 2)
            lr0i = state.pending_lr0[:, : self.lact_chunk_size, :]
            lr1i = state.pending_lr1[:, : self.lact_chunk_size, :]
            lr2i = state.pending_lr2[:, : self.lact_chunk_size, :]

            gate_before_act = torch.bmm(fw_w0, ki.transpose(1, 2))
            hidden_before_mul = torch.bmm(fw_w2, ki.transpose(1, 2))
            hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul
            dhidden = torch.bmm(fw_w1.transpose(1, 2), vi)
            dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
            dgate = dhidden * hidden_before_mul
            dgate_before_act = silu_backprop(dgate, gate_before_act)

            dw1 = torch.bmm(vi, (hidden.transpose(1, 2) * lr1i).type_as(vi))
            dw0 = torch.bmm(dgate_before_act, (ki * lr0i).type_as(dgate_before_act))
            dw2 = torch.bmm(dhidden_before_mul, (ki * lr2i).type_as(dhidden_before_mul))

            if state.pending_momentum is not None and state.dw0_momentum is not None:
                assert state.dw1_momentum is not None
                assert state.dw2_momentum is not None
                m_i = state.pending_momentum[:, : self.lact_chunk_size, :].mean(
                    dim=1, keepdim=True
                )
                dw0 = dw0 + state.dw0_momentum * m_i
                dw1 = dw1 + state.dw1_momentum * m_i
                dw2 = dw2 + state.dw2_momentum * m_i
                state.dw0_momentum = dw0
                state.dw1_momentum = dw1
                state.dw2_momentum = dw2

            if self.use_muon:
                dw0 = zeropower_via_newtonschulz5(dw0)
                dw1 = zeropower_via_newtonschulz5(dw1)
                dw2 = zeropower_via_newtonschulz5(dw2)

            state.w0 = fw_w0 + dw0
            state.w1 = fw_w1 + dw1
            state.w2 = fw_w2 + dw2

            state.w0 = (
                state.w0 / (state.w0.norm(dim=2, keepdim=True) + 1e-5) * state.w0_norm
            )
            state.w1 = (
                state.w1 / (state.w1.norm(dim=2, keepdim=True) + 1e-5) * state.w1_norm
            )
            state.w2 = (
                state.w2 / (state.w2.norm(dim=2, keepdim=True) + 1e-5) * state.w2_norm
            )

            if pending_len > self.lact_chunk_size:
                state.pending_k = state.pending_k[:, self.lact_chunk_size :, :]
                state.pending_v = state.pending_v[:, self.lact_chunk_size :, :]
                state.pending_lr0 = state.pending_lr0[:, self.lact_chunk_size :, :]
                state.pending_lr1 = state.pending_lr1[:, self.lact_chunk_size :, :]
                state.pending_lr2 = state.pending_lr2[:, self.lact_chunk_size :, :]
                if state.pending_momentum is not None:
                    state.pending_momentum = state.pending_momentum[
                        :, self.lact_chunk_size :, :
                    ]
            else:
                state.pending_k = None
                state.pending_v = None
                state.pending_lr0 = None
                state.pending_lr1 = None
                state.pending_lr2 = None
                state.pending_momentum = None

        ttt_x_normed = self.ttt_norm(fw_x)
        if self.learnable_ttt_scale:
            ttt_scale = F.silu(self.ttt_scale_proj(hidden_states), inplace=False)
            ttt_scale = rearrange(
                ttt_scale, "b s (h d) -> (b h) s d", h=self.num_fw_heads
            )
            ttt_x_normed = ttt_x_normed * ttt_scale

        ttt_output = rearrange(
            ttt_x_normed,
            "(b h) s d -> b s (h d)",
            h=self.num_fw_heads,
        )
        return ttt_output.type_as(hidden_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        cache_position: Optional[torch.LongTensor] = None,
        lact_cache: Optional[LaCTCache] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.size()

        q = self.attn_layer.q_proj(hidden_states)
        k = self.attn_layer.k_proj(hidden_states)
        v = self.attn_layer.v_proj(hidden_states)

        q_normed = self.attn_layer.q_norm(
            rearrange(q, "... (h d) -> ... h d", d=self.head_dim)
        )
        k_normed = self.attn_layer.k_norm(
            rearrange(k, "... (h d) -> ... h d", d=self.head_dim)
        )
        q_flat = rearrange(q_normed, "... h d -> ... (h d)")
        k_flat = rearrange(k_normed, "... h d -> ... (h d)")

        # for GQA: expand K and V to match Q's dimension
        n_rep = self.num_attn_heads // self.num_kv_heads
        if n_rep > 1:
            # k_flat: (batch, seq, num_kv_heads * head_dim) -> (batch, seq, num_attn_heads * head_dim)
            k_flat = k_flat.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            k_flat = k_flat.unsqueeze(3).expand(-1, -1, -1, n_rep, -1)
            k_flat = k_flat.reshape(batch_size, seq_len, -1)
            # v: (batch, seq, num_kv_heads * head_dim) -> (batch, seq, num_attn_heads * head_dim)
            v_expanded = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            v_expanded = v_expanded.unsqueeze(3).expand(-1, -1, -1, n_rep, -1)
            v_expanded = v_expanded.reshape(batch_size, seq_len, -1)
        else:
            v_expanded = v

        q_flat, k_flat = self._rescale_qk(q_flat, k_flat)

        if self.use_conv_layer:
            video_mask = kwargs["video_mask"]
            t, h, w = kwargs["video_grid_thw"][0]
            h, w = h // 2, w // 2

            def short_conv(x, conv_layer, inference):
                if inference:
                    x_clone = x
                else:
                    x_clone = x.clone()
                video_tokens = x[video_mask]
                num_videos = video_tokens.size(0) // (t * h * w)
                video_reshaped = rearrange(
                    video_tokens,
                    "(n t h w) c -> n c t h w",
                    n=num_videos,
                    t=t,
                    h=h,
                    w=w,
                )
                video_conv = conv_layer(video_reshaped)
                x_clone[video_mask] = rearrange(video_conv, "n c t h w -> (n t h w) c")
                return x_clone

            # prefill
            if video_mask is not None:
                q_flat = short_conv(q_flat, self.conv_q, lact_cache is not None)
                k_flat = short_conv(k_flat, self.conv_k, lact_cache is not None)
                v_expanded = short_conv(v_expanded, self.conv_v, lact_cache is not None)

        if lact_cache is not None:
            if not lact_cache.has_layer(self.layer_idx):
                # prefill: initialize cache and compute TTT output
                ttt_output = self._init_cache_from_prefill(
                    hidden_states,
                    q_flat,
                    k_flat,
                    v_expanded,
                    position_embeddings,
                    lact_cache,
                )
            else:
                # decode: use cached fast weights
                ttt_output = self._compute_ttt_output_decode(
                    hidden_states,
                    q_flat,
                    k_flat,
                    v_expanded,
                    position_embeddings,
                    lact_cache,
                )
        else:
            # training mode: no cache
            ttt_output = self._compute_ttt_output(
                hidden_states,
                q_flat,
                k_flat,
                v_expanded,
                position_embeddings,
                cu_seqlens=kwargs.get("cu_seq_lens_q", None),
            )

        query_states = q_normed.transpose(1, 2)
        key_states = k_normed.transpose(1, 2)
        value_states = rearrange(v, "b s (h d) -> b h s d", d=self.head_dim)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, unsqueeze_dim=1
        )

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
            # only truncate KV cache during decode phase (seq_len == 1)
            # during prefill, sliding window attention handles long sequences
            if seq_len == 1:
                cache_len = key_states.shape[2]
                if cache_len > self.window_size:
                    key_states, value_states = _truncate_kv_cache_layer(
                        past_key_values,
                        self.layer_idx,
                        key_states,
                        value_states,
                        self.window_size,
                    )

        use_varlen_attn = self.config._attn_implementation == "flash_attention_2"

        if attention_mask is None:
            cu_seqlens_q = torch.tensor(
                [0, query_states.shape[2]],
                device=hidden_states.device,
                dtype=torch.int32,
            )
            cu_seqlens_k = torch.tensor(
                [0, key_states.shape[2]],
                device=hidden_states.device,
                dtype=torch.int32,
            )
        else:
            cu_seqlens_q = attention_mask
            cu_seqlens_k = attention_mask

        if use_varlen_attn:
            query_fa = query_states.transpose(1, 2).squeeze(0)  # (seq_len, head, dim)
            key_fa = key_states.transpose(1, 2).squeeze(0)
            value_fa = value_states.transpose(1, 2).squeeze(0)

            with torch.no_grad():
                max_seqlen_q = max(
                    [
                        cu_seqlens_q[idx + 1] - cu_seqlens_q[idx]
                        for idx in range(cu_seqlens_q.size(0) - 1)
                    ]
                ).item()
                max_seqlen_k = max(
                    [
                        cu_seqlens_k[idx + 1] - cu_seqlens_k[idx]
                        for idx in range(cu_seqlens_k.size(0) - 1)
                    ]
                ).item()

            attn_output = flash_attn_varlen_func(
                query_fa,
                key_fa,
                value_fa,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                window_size=(self.window_size - 1, 0),
                causal=True,
            )
            attn_output = attn_output.unsqueeze(0)  # (1, seq_len, head, dim)
            attn_weights = None
        else:
            # fallback
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

            attn_kwargs = dict(kwargs)
            attn_kwargs["sliding_window"] = self.window_size

            attn_output, attn_weights = attention_interface(
                self.attn_layer,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attn_layer.attention_dropout,
                scaling=self.scaling,
                **attn_kwargs,
            )

        if past_key_values is not None:
            # double check for prefill stage
            cache_len = key_states.shape[2]
            if cache_len > self.window_size:
                key_states, value_states = _truncate_kv_cache_layer(
                    past_key_values,
                    self.layer_idx,
                    key_states,
                    value_states,
                    self.window_size,
                )

        attn_output = attn_output.reshape(batch_size, seq_len, -1).contiguous()
        output = attn_output + ttt_output
        output = self.attn_layer.o_proj(output)

        return output, attn_weights
