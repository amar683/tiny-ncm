"""Encode every block of the prepared corpus with the frozen VQ (cached .codes files).

  python encode_codes.py --data data --vq runs/vq/best.pt
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from ncm.data import read_u16
from ncm.utils import load_vq, pick_device


@torch.no_grad()
def encode_array(vq, blocks, dev, bs=4096):
    out = np.empty((len(blocks), vq.cfg.n_codes), dtype=np.uint16)
    for i in range(0, len(blocks), bs):
        x = torch.from_numpy(np.asarray(blocks[i:i + bs]).astype(np.int64)).to(dev)
        out[i:i + bs] = vq.encode(x).cpu().numpy().astype(np.uint16)
        if (i // bs) % 200 == 0:
            print(f"  {i}/{len(blocks)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--vq", default="runs/vq/best.pt")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    dev = pick_device(args.device)
    vq, cfg = load_vq(args.vq, dev)
    L = cfg.block
    root = Path(args.data)
    import json
    R = json.loads((root / "meta.json").read_text())["R"]
    for name in ["pretrain", "pretrain_val", "task", "task_val"]:
        p = root / f"{name}.bin"
        if not p.exists():
            continue
        tok = read_u16(p)
        nb = len(tok) // L
        print(f"encoding {name}: {nb} blocks")
        codes = encode_array(vq, tok[: nb * L].reshape(nb, L), dev)
        codes.tofile(root / f"{name}.codes")
    print("done")


if __name__ == "__main__":
    main()
