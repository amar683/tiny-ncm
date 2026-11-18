"""Token storage and batch loaders.

Files under a data dir:
  pretrain.bin / pretrain_val.bin   uint16 token streams (documents joined by <|eos|>)
  task.bin, task_bmask.bin          (rows, R) uint16 tokens / (rows, R/block) uint8 loss mask
  task_val.bin, task_val_bmask.bin
  *.codes                           uint16 VQ codes, (n_blocks, n_codes), written by encode_codes.py
  meta.json                         row length R, block, n_codes
"""
import json
from pathlib import Path

import numpy as np
import torch


def write_u16(path, arr):
    np.asarray(arr, dtype=np.uint16).tofile(path)


def read_u16(path, shape=None):
    m = np.memmap(path, dtype=np.uint16, mode="r")
    return m.reshape(shape) if shape is not None else m


def pack_rows(convs, R, pad, block):
    """Greedy-pack block-aligned conversations (tokens, loss) into rows of length R."""
    rows, masks = [], []
    cur, curl = [], []
    for toks, loss in convs:
        if len(toks) > R:
            continue
        if len(cur) + len(toks) > R:
            rows.append(cur + [pad] * (R - len(cur)))
            masks.append(curl + [0] * (R - len(curl)))
            cur, curl = [], []
        cur += toks
        curl += loss
    if cur:
        rows.append(cur + [pad] * (R - len(cur)))
        masks.append(curl + [0] * (R - len(curl)))
    t = np.array(rows, dtype=np.uint16)
    lm = np.array(masks, dtype=np.uint8).reshape(len(rows), R // block, block).max(-1)
    return t, lm


class Data:
    def __init__(self, root, cfg, need_codes=True):
        self.root = Path(root)
        self.cfg = cfg
        meta = json.loads((self.root / "meta.json").read_text())
        self.R = meta["R"]
        L, n = cfg.block, cfg.n_codes
        self.streams, self.tasks = {}, {}
        for split, suf in (("train", ""), ("val", "_val")):
            p = self.root / f"pretrain{suf}.bin"
            if p.exists():
                tok = read_u16(p)
                nb = len(tok) // L
                codes = None
                if need_codes and (self.root / f"pretrain{suf}.codes").exists():
                    codes = read_u16(self.root / f"pretrain{suf}.codes", (-1, n))[:nb]
                self.streams[split] = (tok[: nb * L], codes)
            p = self.root / f"task{suf}.bin"
            if p.exists():
                tok = read_u16(p, (-1, self.R))
                bm = np.fromfile(self.root / f"task{suf}_bmask.bin", dtype=np.uint8).reshape(len(tok), self.R // L)
                codes = None
                if need_codes and (self.root / f"task{suf}.codes").exists():
                    codes = read_u16(self.root / f"task{suf}.codes", (len(tok), self.R // L, n))
                self.tasks[split] = (tok, bm, codes)

    def vq_blocks(self, B, split="train", rng=np.random, task_frac=0.3):
        """Random (B, block) token blocks for VQ training; non-padding task blocks mixed in."""
        L = self.cfg.block
        out = []
        tok, _ = self.streams[split]
        nb = len(tok) // L
        n_task = int(B * task_frac) if split in self.tasks else 0
        if B - n_task > 0:
            idx = rng.randint(0, nb, B - n_task)
            out.append(np.stack([tok[i * L:(i + 1) * L] for i in idx]))
        if n_task:
            ttok, bm, _ = self.tasks[split]
            r = rng.randint(0, len(ttok), n_task * 2)
            c = rng.randint(0, self.R // L, n_task * 2)
            blks = ttok.reshape(len(ttok), -1, L)[r, c]
            keep = blks[(blks != 0).any(1)][:n_task]  # drop all-pad blocks
            if len(keep):
                out.append(keep)
        return torch.from_numpy(np.concatenate(out).astype(np.int64))

    def gen_batch(self, B, T, source="pretrain", split="train", rng=np.random):
        """-> tokens (B,T), codes (B,T/L,n), bmask (B,T/L) or None."""
        L, n = self.cfg.block, self.cfg.n_codes
        nb = T // L
        if source == "pretrain":
            tok, codes = self.streams[split]
            tot = len(tok) // L
            st = rng.randint(0, tot - nb, B)
            t = np.stack([tok[s * L:(s + nb) * L] for s in st])
            c = np.stack([codes[s:s + nb] for s in st])
            bm = None
        else:
            tok, bmk, codes = self.tasks[split]
            rows = rng.randint(0, len(tok), B)
            off = rng.randint(0, self.R // L - nb + 1, B)
            t = np.stack([tok[r, o * L:(o + nb) * L] for r, o in zip(rows, off)])
            c = np.stack([codes[r, o:o + nb] for r, o in zip(rows, off)])
            bm = torch.from_numpy(np.stack([bmk[r, o:o + nb] for r, o in zip(rows, off)]).astype(np.float32))
        return (torch.from_numpy(t.astype(np.int64)), torch.from_numpy(c.astype(np.int64)), bm)
