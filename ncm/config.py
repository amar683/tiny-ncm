from dataclasses import dataclass, asdict, replace, field


@dataclass
class Config:
    # text
    vocab_size: int = 16384
    block: int = 32                      # tokens per NCM block
    levels: tuple = (2, 4, 8, 16, 32)    # codes per level, coarse -> fine; last == block
    # VQ-VAE (works on one block at a time)
    vq_dim: int = 256
    vq_depth: int = 4
    vq_heads: int = 4
    code_dim: int = 64
    codebook: int = 1024
    vq_ema: float = 0.99
    # generator
    dim: int = 384
    depth: int = 8
    heads: int = 3
    kv_heads: int = 1
    ctx: int = 1024                      # training context (tokens); raised to 4096 in task phase
    max_ctx: int = 4096
    rope_base: float = 100_000.0
    level_window: int = 128              # level slots see this many preceding context tokens (0 = all)
    alpha: float = 0.5                   # per-level loss weighting (0: per code, 1: per level)
    corrupt: tuple = (0.0, 0.05, 0.1, 0.1, 0.1)  # conditioning-corruption prob per level
    zloss: float = 1e-4

    def to_dict(self):
        d = asdict(self)
        d["levels"] = list(self.levels)
        d["corrupt"] = list(self.corrupt)
        return d

    @property
    def n_codes(self):
        return sum(self.levels)


PRESETS = {
    # pipeline debugging, seconds per step
    "nano": Config(vocab_size=2048, vq_dim=64, vq_depth=2, vq_heads=2, code_dim=16, codebook=64,
                   dim=64, depth=2, heads=2, kv_heads=1, ctx=128, max_ctx=256, level_window=64),
    "small": Config(dim=256, depth=8, heads=4, kv_heads=2),
    "base": Config(),
}


def get_config(name, **overrides):
    overrides = {k: (tuple(v) if isinstance(v, list) else v) for k, v in overrides.items()}
    return replace(PRESETS[name], **overrides)


def from_dict(d):
    d = dict(d)
    for k in ("levels", "corrupt"):
        if k in d:
            d[k] = tuple(d[k])
    return Config(**d)
