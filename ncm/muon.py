"""Muon (momentum + Newton-Schulz orthogonalization) for 2-D hidden weights, AdamW for the rest."""
import torch


def newton_schulz(G, steps=5, eps=1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    X = X / (X.norm() + eps)
    tr = X.size(0) > X.size(1)
    if tr:
        X = X.t()
    for _ in range(steps):
        A = X @ X.t()
        X = a * X + (b * A + c * A @ A) @ X
    return X.t() if tr else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.0, nesterov=True):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p.grad)
                buf = st["buf"]
                buf.mul_(g["momentum"]).add_(p.grad)
                upd = p.grad.add(buf, alpha=g["momentum"]) if g["nesterov"] else buf
                o = newton_schulz(upd)
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                if g["weight_decay"]:
                    p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(o.to(p.dtype), alpha=-g["lr"] * scale)


def build_optimizers(model, lr_muon=0.02, lr_adam=3e-3, wd=0.0):
    """Muon for hidden 2-D matrices inside transformer blocks; AdamW for embeddings, heads,
    norms and input projections."""
    muon, adam = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and ".blocks." in f".{name}" or (p.ndim == 2 and (".enc." in f".{name}" or ".dec." in f".{name}")):
            muon.append(p)
        else:
            adam.append(p)
    opts = [Muon(muon, lr=lr_muon, weight_decay=wd)]
    if adam:
        opts.append(torch.optim.AdamW(adam, lr=lr_adam, betas=(0.9, 0.95), weight_decay=0.0, eps=1e-10))
    for o in opts:
        for g in o.param_groups:
            g["base_lr"] = g["lr"]
    return opts


def set_lr(opts, mult):
    for o in opts:
        for g in o.param_groups:
            g["lr"] = g["base_lr"] * mult


def wsd(step, total, warmup=200, decay_frac=0.2, final=0.0):
    """Warmup-stable-decay schedule multiplier."""
    if step < warmup:
        return (step + 1) / warmup
    d0 = int(total * (1 - decay_frac))
    if step < d0:
        return 1.0
    t = (step - d0) / max(1, total - d0)
    return final + (1 - final) * (1 - t)
