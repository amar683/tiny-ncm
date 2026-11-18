"""Block-wise coarse-to-fine generation (reference implementation; the apps port this loop)."""
import torch


def sample_codes(logits, temperature=0.8, top_k=50, generator=None):
    if temperature <= 0:
        return logits.argmax(-1)
    logits = logits / temperature
    if top_k:
        v, _ = logits.topk(min(top_k, logits.size(-1)), -1)
        logits = logits.masked_fill(logits < v[..., -1:], float("-inf"))
    p = logits.softmax(-1)
    return torch.multinomial(p.flatten(0, -2), 1, generator=generator).view(p.shape[:-1])


@torch.no_grad()
def generate(gen, prompt, max_blocks=8, temperature=0.8, top_k=50, stop_ids=(), on_level=None,
             pad_id=0, seed=None):
    """prompt: list[int] (padded to a block multiple by the caller or here with pad_id).
    Returns the list of generated token ids (without the prompt), cut after the first stop id.
    on_level(block_idx, level, tokens) is called with the partial decode after every level."""
    cfg, vq = gen.cfg, gen.vq
    L = cfg.block
    dev = gen.tok.weight.device
    prompt = list(prompt)
    if len(prompt) % L:
        prompt = [pad_id] * (L - len(prompt) % L) + prompt  # left-pad keeps text contiguous
    g = None
    if seed is not None:
        g = torch.Generator(device="cpu").manual_seed(seed)
    cache = [None] * len(gen.blocks)
    cache = gen.extend(cache, torch.tensor([prompt], device=dev), 0)
    pos = len(prompt)
    out = []
    stop = set(stop_ids)
    for _ in range(max_blocks):
        if pos + L > cfg.max_ctx:
            break
        b = pos // L
        zhat = vq.embed.new_zeros(1, L, cfg.code_dim)
        for k in range(len(cfg.levels)):
            logits = gen.level_logits(cache, zhat, k, b)
            idx = sample_codes(logits.cpu() if g is not None else logits, temperature, top_k, g).to(dev)
            zhat = zhat + vq.contrib(k, vq.lookup(k, idx))
            if on_level is not None:
                on_level(b, k, vq.decode_latent(zhat).argmax(-1)[0].tolist())
        toks = vq.decode_latent(zhat).argmax(-1)  # (1, L)
        tl = toks[0].tolist()
        hit = next((i for i, t in enumerate(tl) if t in stop), None)
        if hit is not None:
            out += tl[: hit + 1]
            break
        out += tl
        cache = gen.extend(cache, toks, pos)
        pos += L
    return out
