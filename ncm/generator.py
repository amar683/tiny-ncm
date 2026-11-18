"""NCM generator (stage 2): block-wise, coarse-to-fine, all codes of a level in parallel.

Two streams share the same transformer weights:
  * context stream: the text tokens, causal attention, RoPE positions 0..T-1;
  * level stream:   for every block b and level k, l_k slots whose input is the frozen VQ's
                    cumulative reconstruction of levels < k, resized to l_k. A slot of block b
                    attends to the `level_window` context tokens just before block b (all earlier
                    text when level_window == 0) and to the other slots of its own level.
                    Long-range information reaches the slots through the context stream, whose
                    tokens attend to the full history.
Training predicts every level of every block in one pass. Generation runs 5 level passes per
block against a KV cache of the context, decodes the block with the VQ decoder, and appends it.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Block, attend, init_weights, rope_cache, scale_residual_proj
from .vq import down


class NCMGenerator(nn.Module):
    def __init__(self, cfg, vq):
        super().__init__()
        self.cfg = cfg
        self.vq = vq.eval()
        for p in self.vq.parameters():
            p.requires_grad_(False)
        D, nl = cfg.dim, len(cfg.levels)
        self.tok = nn.Embedding(cfg.vocab_size, D)
        self.lvl_in = nn.Linear(cfg.code_dim, D, bias=False)
        self.lvl_emb = nn.Embedding(nl, D)
        self.slot_emb = nn.Embedding(cfg.n_codes, D)
        self.blocks = nn.ModuleList([Block(D, cfg.heads, cfg.kv_heads) for _ in range(cfg.depth)])
        self.norm = nn.RMSNorm(D)
        self.heads = nn.ModuleList([nn.Linear(D, cfg.codebook, bias=False) for _ in range(nl)])
        for m in [self.tok, self.lvl_in, self.lvl_emb, self.slot_emb, self.blocks, self.heads]:
            m.apply(init_weights)
        scale_residual_proj(self.blocks, cfg.depth)
        offs, lid = [0], []
        for k, l in enumerate(cfg.levels):
            offs.append(offs[-1] + l)
            lid += [k] * l
        self.offsets = offs
        self.register_buffer("level_id", torch.tensor(lid), persistent=False)
        self._rope = None
        self._masks = {}
        self.register_buffer("own_mask", self.level_id[:, None] == self.level_id[None, :], persistent=False)

    def train(self, mode=True):
        super().train(mode)
        self.vq.eval()  # always frozen
        return self

    def n_params(self, trainable=True):
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable)

    def rope(self, device):
        if self._rope is None or self._rope[0].device != device:
            hd = self.cfg.dim // self.cfg.heads
            self._rope = rope_cache(self.cfg.max_ctx + self.cfg.block, hd, self.cfg.rope_base, device)
        return self._rope

    # ---- level-stream inputs ------------------------------------------------------------
    def level_inputs(self, codes, corrupt=False):
        """codes: (N, n_codes) long (N = batch*blocks) -> (N, n_codes, D) slot inputs."""
        vq, cfg = self.vq, self.cfg
        parts = vq.split(codes)
        if corrupt:
            parts = [torch.where(torch.rand_like(p, dtype=torch.float) < pr,
                                 torch.randint_like(p, cfg.codebook), p) if pr > 0 else p
                     for p, pr in zip(parts, cfg.corrupt)]
        zhat = vq.embed.new_zeros(codes.size(0), cfg.block, cfg.code_dim)
        ins = []
        with torch.no_grad():
            for k, l in enumerate(cfg.levels):
                ins.append(down(zhat, l))
                if k + 1 < len(cfg.levels):
                    zhat = zhat + vq.contrib(k, vq.lookup(k, parts[k]))
        x = torch.cat(ins, 1).to(self.lvl_in.weight.dtype)
        return self.lvl_in(x) + self.lvl_emb(self.level_id) + self.slot_emb.weight

    def slot_inputs(self, zhat, k):
        """Generation: inputs for level k's slots from the running reconstruction (N, block, C)."""
        a, b = self.offsets[k], self.offsets[k + 1]
        x = down(zhat, self.cfg.levels[k]).to(self.lvl_in.weight.dtype)
        return self.lvl_in(x) + self.lvl_emb.weight[k] + self.slot_emb.weight[a:b]

    def _level_mask(self, b0, G, T, device):
        """Mask for the level slots of blocks b0..b0+G-1 against [context[:Tk], those slots]."""
        key = (b0, G, T, str(device))
        if key not in self._masks:
            n, L = self.cfg.n_codes, self.cfg.block
            Tk = min(T, (b0 + G) * L)
            blk = torch.arange(G, device=device).repeat_interleave(n) + b0       # block of each slot
            lv = self.level_id.to(device).repeat(G)
            ctx_ok = torch.arange(Tk, device=device)[None, :] < (blk * L)[:, None]
            own = (blk[:, None] == blk[None, :]) & (lv[:, None] == lv[None, :])
            self._masks[key] = torch.cat([ctx_ok, own], 1)
            if len(self._masks) > 256:
                self._masks.pop(next(iter(self._masks)))
        return self._masks[key]

    def _full_level_attend(self, ql, kl, vl, kc, vc, chunk):
        """Level slots of block b see all context tokens < block*b (level_window == 0)."""
        L, n = self.cfg.block, self.cfg.n_codes
        T, nb = kc.size(2), ql.size(2) // n
        outs = []
        for b0 in range(0, nb, chunk):
            G = min(chunk, nb - b0)
            Tk = min(T, (b0 + G) * L)
            s0, s1 = b0 * n, (b0 + G) * n
            m = self._level_mask(b0, G, T, ql.device)
            k_ = torch.cat([kc[:, :, :Tk], kl[:, :, s0:s1]], 2)
            v_ = torch.cat([vc[:, :, :Tk], vl[:, :, s0:s1]], 2)
            outs.append(attend(ql[:, :, s0:s1], k_, v_, mask=m))
        return torch.cat(outs, 2)

    def _window_attend(self, ql, kl, vl, kc, vc):
        """Level slots of block b see context tokens [block*b - W, block*b) plus their own level.
        One batched softmax over (window keys, own-block slot keys); no Python loop."""
        L, n, W = self.cfg.block, self.cfg.n_codes, self.cfg.level_window
        B, H, S, hd = ql.shape
        nb = S // n
        r = H // kc.size(1)
        if r > 1:
            kc, vc, kl, vl = (t.repeat_interleave(r, 1) for t in (kc, vc, kl, vl))
        # windows: padded index [L*b, L*b + W) == original positions [L*b - W, L*b)
        kw = F.pad(kc, (0, 0, W, 0)).unfold(2, W, L)[:, :, :nb]                  # (B,H,nb,hd,W)
        vw = F.pad(vc, (0, 0, W, 0)).unfold(2, W, L)[:, :, :nb].transpose(-2, -1)  # (B,H,nb,W,hd)
        q5 = ql.reshape(B, H, nb, n, hd)
        k5 = kl.reshape(B, H, nb, n, hd)
        v5 = vl.reshape(B, H, nb, n, hd)
        key = ("win", nb, str(ql.device))
        if key not in self._masks:
            pos = torch.arange(nb, device=ql.device)[:, None] * L - W + torch.arange(W, device=ql.device)[None]
            self._masks[key] = (pos >= 0)[None, None, :, None, :]                   # (1,1,nb,1,W)
        s1 = (q5 @ kw).masked_fill(~self._masks[key], float("-inf"))
        s2 = (q5 @ k5.transpose(-2, -1)).masked_fill(~self.own_mask, float("-inf"))
        p = (torch.cat([s1, s2], -1).float() * hd ** -0.5).softmax(-1).to(q5.dtype)
        o = p[..., :W] @ vw + p[..., W:] @ v5
        return o.reshape(B, H, S, hd)

    # ---- training forward ---------------------------------------------------------------
    def forward(self, tokens, codes, block_mask=None, corrupt=True, chunk=8, return_logits=False):
        """tokens: (B, T); codes: (B, T/block, n_codes); block_mask: (B, T/block) float/bool
        (1 = this block contributes loss). Returns (loss, stats)."""
        cfg = self.cfg
        B, T = tokens.shape
        L, n = cfg.block, cfg.n_codes
        nb = T // L
        cos, sin = self.rope(tokens.device)
        rc = (cos[:T][None], sin[:T][None])
        x = self.tok(tokens)
        y = self.level_inputs(codes.reshape(B * nb, n), corrupt=corrupt and self.training)
        y = y.view(B, nb * n, -1).to(x.dtype)
        pos_l = (torch.arange(nb, device=tokens.device) * L).repeat_interleave(n)
        rl = (cos[pos_l][None], sin[pos_l][None])
        for i, blk in enumerate(self.blocks):
            at = blk.attn
            h, g = blk.n1(x), blk.n1(y)
            qc, kc, vc = at.qkv(h, rc)
            ql, kl, vl = at.qkv(g, rl)
            if cfg.level_window:
                o = self._window_attend(ql, kl, vl, kc, vc)
            else:
                o = self._full_level_attend(ql, kl, vl, kc, vc, chunk)
            y = y + at.out(o)
            y = y + blk.mlp(blk.n2(y))
            if i + 1 < len(self.blocks):  # the last layer's context output is never read
                x = x + at.out(attend(qc, kc, vc, causal=True))
                x = x + blk.mlp(blk.n2(x))
        y = self.norm(y).view(B, nb, n, -1)
        tgt = codes.view(B, nb, n)
        bm = torch.ones(B, nb, device=tokens.device) if block_mask is None else block_mask.float()
        tot, wsum, zl = 0.0, 0.0, 0.0
        stats = {}
        all_logits = []
        for k, l in enumerate(cfg.levels):
            a, b = self.offsets[k], self.offsets[k + 1]
            logits = self.heads[k](y[:, :, a:b]).float()            # (B, nb, l, K)
            if return_logits:
                all_logits.append(logits)
            ce = F.cross_entropy(logits.flatten(0, 2), tgt[:, :, a:b].flatten(), reduction="none")
            ce = ce.view(B, nb, l) * bm[..., None]
            w = l ** (-cfg.alpha)
            cnt = bm.sum() * l
            tot = tot + w * ce.sum()
            wsum = wsum + w * cnt
            if cfg.zloss:
                zl = zl + (torch.logsumexp(logits, -1).pow(2) * bm[..., None]).sum() / cnt.clamp(min=1)
            stats[f"l{k}"] = (ce.sum() / cnt.clamp(min=1)).detach()
        loss = tot / wsum.clamp(min=1) + cfg.zloss * zl / len(cfg.levels)
        if return_logits:
            return loss, stats, all_logits
        return loss, stats

    # ---- generation pieces ---------------------------------------------------------------
    @torch.no_grad()
    def extend(self, cache, tokens, start):
        """Run the context stream on `tokens` (B, t) at positions start.. and append to the cache.
        cache: list per layer of (k, v) or None."""
        cos, sin = self.rope(tokens.device)
        t = tokens.size(1)
        rc = (cos[start:start + t][None], sin[start:start + t][None])
        x = self.tok(tokens)
        new = []
        for i, blk in enumerate(self.blocks):
            at = blk.attn
            q, k, v = at.qkv(blk.n1(x), rc)
            if cache[i] is not None:
                k = torch.cat([cache[i][0], k], 2)
                v = torch.cat([cache[i][1], v], 2)
            new.append((k, v))
            if i + 1 < len(self.blocks):
                x = x + at.out(attend(q, k, v, causal=True))
                x = x + blk.mlp(blk.n2(x))
        return new

    @torch.no_grad()
    def level_logits(self, cache, zhat, k, block_idx):
        """Logits (B, l_k, codebook) for level k of block `block_idx`, given the context cache
        (which must hold exactly block_idx*block tokens) and the running reconstruction."""
        cos, sin = self.rope(zhat.device)
        l = self.cfg.levels[k]
        p = block_idx * self.cfg.block
        rl = (cos[p:p + 1].expand(l, -1)[None], sin[p:p + 1].expand(l, -1)[None])
        y = self.slot_inputs(zhat, k).to(self.tok.weight.dtype)
        for i, blk in enumerate(self.blocks):
            at = blk.attn
            q, kk, vv = at.qkv(blk.n1(y), rl)
            if cache[i] is not None:
                ck, cv = cache[i]
                if self.cfg.level_window:
                    ck, cv = ck[:, :, -self.cfg.level_window:], cv[:, :, -self.cfg.level_window:]
                kk = torch.cat([ck, kk], 2)
                vv = torch.cat([cv, vv], 2)
            y = y + at.out(attend(q, kk, vv))
            y = y + blk.mlp(blk.n2(y))
        return self.heads[k](self.norm(y)).float()
