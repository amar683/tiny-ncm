"""Multiscale residual VQ-VAE over one block of text (stage 1 of NCM).

A block of `block` tokens is encoded to `block` latent vectors z. Level k (coarse -> fine)
downsamples the current residual to l_k vectors, snaps each to its nearest entry in level k's
codebook, maps the chosen vectors back to block length with R_k (repeat + learned linear), and
adds them to the cumulative reconstruction z_hat. The decoder maps z_hat back to tokens.
Codebooks are updated by EMA, with dead-code reset; nested dropout trains partial prefixes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Block, init_weights


def down(x, l):
    """(N, block, C) -> (N, l, C) by averaging contiguous groups."""
    N, L, C = x.shape
    return x.view(N, l, L // l, C).mean(2)


class MultiscaleVQ(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        L, D, C, K = cfg.block, cfg.vq_dim, cfg.code_dim, cfg.codebook
        nl = len(cfg.levels)
        assert cfg.levels[-1] == L and all(L % l == 0 for l in cfg.levels)
        self.tok = nn.Embedding(cfg.vocab_size, D)
        self.enc_pos = nn.Embedding(L, D)
        self.enc = nn.ModuleList([Block(D, cfg.vq_heads) for _ in range(cfg.vq_depth)])
        self.enc_norm = nn.RMSNorm(D)
        self.to_code = nn.Linear(D, C, bias=False)
        self.up = nn.ModuleList([nn.Linear(C, C, bias=False) for _ in range(nl)])
        self.from_code = nn.Linear(C, D, bias=False)
        self.dec_pos = nn.Embedding(L, D)
        self.dec = nn.ModuleList([Block(D, cfg.vq_heads) for _ in range(cfg.vq_depth)])
        self.dec_norm = nn.RMSNorm(D)
        self.head = nn.Linear(D, cfg.vocab_size, bias=False)
        self.apply(init_weights)
        for u in self.up:
            nn.init.eye_(u.weight)
        self.register_buffer("embed", torch.randn(nl, K, C) * 0.1)
        self.register_buffer("ema_count", torch.ones(nl, K))
        self.register_buffer("ema_sum", self.embed.clone())
        self.register_buffer("initted", torch.zeros((), dtype=torch.bool))
        self._last = [None] * nl  # recent pre-quantization vectors, for dead-code reset

    # ---- pieces -------------------------------------------------------------------------
    def encode_latent(self, tokens):
        pos = torch.arange(tokens.size(1), device=tokens.device)
        h = self.tok(tokens) + self.enc_pos(pos)
        for b in self.enc:
            h = b(h)
        return self.to_code(self.enc_norm(h))

    def decode_latent(self, zhat):
        pos = torch.arange(zhat.size(1), device=zhat.device)
        h = self.from_code(zhat) + self.dec_pos(pos)
        for b in self.dec:
            h = b(h)
        return self.head(self.dec_norm(h))

    def contrib(self, k, q):
        """R_k: (N, l_k, C) code vectors -> (N, block, C) contribution."""
        rep = self.cfg.block // q.size(1)
        return self.up[k](q.repeat_interleave(rep, dim=1))

    def lookup(self, k, idx):
        return F.embedding(idx, self.embed[k])

    def nearest(self, k, x):
        x = x.float()
        e = self.embed[k]
        d = x.pow(2).sum(-1, keepdim=True) - 2 * x @ e.t() + e.pow(2).sum(-1)
        return d.argmin(-1)

    # ---- EMA codebook ------------------------------------------------------------------
    @torch.no_grad()
    def _init_codebooks(self, k, x):
        flat = x.reshape(-1, x.size(-1)).float()
        K = self.cfg.codebook
        sel = flat[torch.randint(0, flat.size(0), (K,), device=flat.device)]
        self.embed[k] = sel + 1e-3 * torch.randn_like(sel)
        self.ema_sum[k] = self.embed[k].clone()
        self.ema_count[k] = 1.0

    @torch.no_grad()
    def _ema(self, k, x, idx):
        flat, idx = x.reshape(-1, x.size(-1)).float(), idx.reshape(-1)
        K, decay = self.cfg.codebook, self.cfg.vq_ema
        counts = torch.zeros(K, device=flat.device).index_add_(0, idx, torch.ones_like(idx, dtype=torch.float))
        sums = torch.zeros(K, flat.size(1), device=flat.device).index_add_(0, idx, flat)
        self.ema_count[k].mul_(decay).add_(counts, alpha=1 - decay)
        self.ema_sum[k].mul_(decay).add_(sums, alpha=1 - decay)
        n = self.ema_count[k].sum()
        cs = (self.ema_count[k] + 1e-5) / (n + K * 1e-5) * n
        self.embed[k] = self.ema_sum[k] / cs.unsqueeze(1)
        self._last[k] = flat[torch.randint(0, flat.size(0), (min(4096, flat.size(0)),), device=flat.device)]

    @torch.no_grad()
    def reset_dead(self, threshold=0.03):
        """Re-seed codes whose EMA usage fell below `threshold` with recent encoder outputs."""
        n_reset = 0
        for k in range(len(self.cfg.levels)):
            dead = self.ema_count[k] < threshold
            nd = int(dead.sum())
            if nd == 0 or self._last[k] is None:
                continue
            src = self._last[k][torch.randint(0, self._last[k].size(0), (nd,), device=self.embed.device)]
            src = src + 1e-3 * torch.randn_like(src)
            self.embed[k][dead] = src
            self.ema_sum[k][dead] = src
            self.ema_count[k][dead] = 1.0
            n_reset += nd
        return n_reset

    # ---- full passes -------------------------------------------------------------------
    def forward(self, tokens, cutoff=None, beta=0.25):
        """Training pass. tokens: (N, block). Returns logits, commit loss, list of code tensors."""
        z = self.encode_latent(tokens)
        cutoff = cutoff or len(self.cfg.levels)
        if self.training and not bool(self.initted):
            with torch.no_grad():
                zh = torch.zeros_like(z)
                for k, l in enumerate(self.cfg.levels):
                    d = down(z - zh, l)
                    self._init_codebooks(k, d)
                    zh = zh + self.contrib(k, self.lookup(k, self.nearest(k, d)).to(z.dtype))
                self.initted.fill_(True)
        zhat = torch.zeros_like(z)
        commit = z.new_zeros(())
        codes = []
        for k, l in enumerate(self.cfg.levels[:cutoff]):
            d = down(z - zhat, l)
            idx = self.nearest(k, d.detach())
            q = self.lookup(k, idx).to(d.dtype)
            if self.training:
                self._ema(k, d.detach(), idx)
            commit = commit + F.mse_loss(d.float(), q.detach().float())
            zhat = zhat + self.contrib(k, d + (q - d).detach())
            codes.append(idx)
        return self.decode_latent(zhat), beta * commit, codes

    @torch.no_grad()
    def encode(self, tokens):
        """(N, block) tokens -> (N, n_codes) long codes, levels concatenated coarse -> fine."""
        z = self.encode_latent(tokens)
        zhat = torch.zeros_like(z)
        out = []
        for k, l in enumerate(self.cfg.levels):
            idx = self.nearest(k, down(z - zhat, l))
            zhat = zhat + self.contrib(k, self.lookup(k, idx).to(z.dtype))
            out.append(idx)
        return torch.cat(out, 1)

    def split(self, codes):
        """(N, n_codes) -> list of (N, l_k)."""
        return list(codes.split(list(self.cfg.levels), dim=1))

    def reconstruct(self, codes, upto=None):
        """Cumulative reconstruction z_hat from the first `upto` levels of concatenated codes."""
        parts = self.split(codes)[: upto or len(self.cfg.levels)]
        N = codes.size(0)
        zhat = self.embed.new_zeros(N, self.cfg.block, self.cfg.code_dim)
        for k, idx in enumerate(parts):
            zhat = zhat + self.contrib(k, self.lookup(k, idx))
        return zhat

    @torch.no_grad()
    def decode(self, codes, upto=None):
        """codes -> token ids (argmax), optionally using only the first `upto` levels."""
        return self.decode_latent(self.reconstruct(codes, upto)).argmax(-1)

    @torch.no_grad()
    def usage(self, threshold=0.03):
        return (self.ema_count > threshold).float().mean(1)  # per level
