"""Transformer building blocks: RMSNorm, RoPE, GQA attention with QK-norm, SwiGLU.

Adapted from tiny-vision/vlm/layers.py.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# "auto": manual matmul+softmax for short sequences on MPS (faster there), fused SDPA otherwise.
ATTN_IMPL = "auto"


def rope_cache(max_pos, head_dim, base, device):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_pos, device=device).float()
    f = torch.outer(t, inv)
    return torch.cos(f), torch.sin(f)


def apply_rope(x, cos, sin):
    # x: (B, H, L, Dh); cos/sin: (B or 1, L, Dh/2)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos.unsqueeze(1), sin.unsqueeze(1)
    out = torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1)
    return out.flatten(-2).type_as(x)


def attend(q, k, v, mask=None, causal=False):
    """q: (B,H,Lq,Dh), k/v: (B,Hkv,Lk,Dh). mask: bool, True = may attend, broadcastable to (B,1,Lq,Lk)."""
    if k.size(1) != q.size(1):
        rep = q.size(1) // k.size(1)
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    Lq, Lk = q.size(2), k.size(2)
    impl = ATTN_IMPL
    if impl == "auto":
        impl = "manual" if (q.device.type == "mps" and Lq * Lk <= 1024 * 1024) else "sdpa"
    if impl == "manual":
        att = (q @ k.transpose(-2, -1)) * (q.size(-1) ** -0.5)
        if causal:
            cm = torch.ones(Lq, Lk, dtype=torch.bool, device=q.device).tril(Lk - Lq)
            att = att.masked_fill(~cm, float("-inf"))
        if mask is not None:
            att = att.masked_fill(~mask, float("-inf"))
        return att.softmax(-1) @ v
    if causal and mask is None and Lq == Lk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    if causal:
        cm = torch.ones(Lq, Lk, dtype=torch.bool, device=q.device).tril(Lk - Lq)
        mask = cm if mask is None else (mask & cm)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


class Attention(nn.Module):
    def __init__(self, dim, heads, kv_heads=None):
        super().__init__()
        kv_heads = kv_heads or heads
        assert dim % heads == 0 and heads % kv_heads == 0
        self.heads, self.kv_heads, self.hd = heads, kv_heads, dim // heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wkv = nn.Linear(dim, 2 * kv_heads * self.hd, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.qn = nn.RMSNorm(self.hd)
        self.kn = nn.RMSNorm(self.hd)

    def qkv(self, x, rope=None):
        B, L, _ = x.shape
        q = self.qn(self.wq(x).view(B, L, self.heads, self.hd)).transpose(1, 2)
        k, v = self.wkv(x).view(B, L, 2, self.kv_heads, self.hd).unbind(2)
        k = self.kn(k).transpose(1, 2)
        v = v.transpose(1, 2)
        if rope is not None:
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        return q, k, v

    def out(self, o):
        B, H, L, Dh = o.shape
        return self.proj(o.transpose(1, 2).reshape(B, L, H * Dh))

    def forward(self, x, rope=None, mask=None, causal=False):
        q, k, v = self.qkv(x, rope)
        return self.out(attend(q, k, v, mask, causal))


class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden = 64 * ((int(8 * dim / 3) + 63) // 64)
        self.w12 = nn.Linear(dim, 2 * hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)


class Block(nn.Module):
    """Pre-norm block. `forward` is the plain (single-sequence) path; the generator drives
    the attention sub-steps itself for its two-stream layout."""

    def __init__(self, dim, heads, kv_heads=None):
        super().__init__()
        self.n1 = nn.RMSNorm(dim)
        self.attn = Attention(dim, heads, kv_heads)
        self.n2 = nn.RMSNorm(dim)
        self.mlp = SwiGLU(dim)

    def forward(self, x, rope=None, mask=None, causal=False):
        x = x + self.attn(self.n1(x), rope, mask, causal)
        return x + self.mlp(self.n2(x))


def init_weights(module, depth=None):
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=0.02)


def scale_residual_proj(model, depth):
    """GPT-2 style: shrink output projections by 1/sqrt(2*depth) for stable deep training."""
    for name, p in model.named_parameters():
        if name.endswith("attn.proj.weight") or name.endswith("mlp.w3.weight"):
            with torch.no_grad():
                p.mul_(1.0 / math.sqrt(2 * depth))
