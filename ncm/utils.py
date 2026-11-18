import contextlib
import json
import time
from pathlib import Path

import torch

from .config import from_dict


def pick_device(name="auto"):  # from tiny-vision/train.py
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def autocast(device, enabled=True):
    if not enabled or device.type == "cpu":
        return contextlib.nullcontext()
    return torch.autocast(device.type, dtype=torch.bfloat16)


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def save_ckpt(path, model, cfg, step, opts=None, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {"cfg": cfg.to_dict(), "model": model.state_dict(), "step": step, "extra": extra or {}}
    if opts is not None:
        obj["opts"] = [o.state_dict() for o in opts]
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def load_ckpt(path, map_location="cpu"):
    obj = torch.load(path, map_location=map_location, weights_only=False)
    obj["cfg"] = from_dict(obj["cfg"])
    return obj


def load_vq(path, device):
    from .vq import MultiscaleVQ
    ck = load_ckpt(path)
    vq = MultiscaleVQ(ck["cfg"])
    vq.load_state_dict(ck["model"])
    return vq.to(device).eval(), ck["cfg"]


def load_generator(path, device):
    from .generator import NCMGenerator
    from .vq import MultiscaleVQ
    ck = load_ckpt(path)
    cfg = ck["cfg"]
    gen = NCMGenerator(cfg, MultiscaleVQ(cfg))
    gen.load_state_dict(ck["model"])
    return gen.to(device).eval(), cfg


class Logger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()

    def __call__(self, step, **kv):
        kv = {k: (round(float(v), 5) if isinstance(v, (float, int, torch.Tensor)) else v) for k, v in kv.items()}
        rec = {"step": step, "min": round((time.time() - self.t0) / 60, 2), **kv}
        print(" ".join(f"{k}={v}" for k, v in rec.items()), flush=True)
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
