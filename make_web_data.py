"""Turn collected web answers into training conversations, mixed with the existing chat/tool data.

python make_web_data.py --harvest data_web/harvest.jsonl --old data_copy2 --out data_web [--phrasings 6] [--web-share 0.3]

Each conversation:  system  ->  user question  ->  <|tool|> web_search  ->  <|result|> the picked answer
                    ->  <|assistant|> the answer with its source
so the model learns to (1) send this kind of question to web search and (2) copy the answer from the result.
Only topics marked "train" are used; your own saved questions (~/.tiny-ncm/web_answers.jsonl) are added 3 times,
except any that match a held-out test topic.
"""
import argparse
import datetime as dt
import json
import random
import re
from pathlib import Path

import numpy as np

from assistant.grounding import template_reply
from assistant.websearch import FLUFF, VERBS, _unnumber
from harvest_web import question
from make_copy_data import link
from ncm.bpe import BPE
from ncm.chat_format import format_conv, load_registry, system_prompt, tool_call_text
from ncm.data import pack_rows, read_u16

USER_LOG = Path.home() / ".tiny-ncm" / "web_answers.jsonl"


def clean(rec):
    """A usable answer: 3-8 short steps or 1-3 sentences, no page filler."""
    legal = re.compile(r"\b(?:disclosure|terms and conditions|advertis|lender|affiliate|sponsored|privacy policy)", re.I)
    if rec.get("kind") == "steps":
        st = [s for s in rec.get("steps", []) if 10 <= len(s) <= 220 and not FLUFF.search(s) and not legal.search(s)]
        acts = sum(re.sub(r"\W", "", _unnumber(s).split()[0].lower()) in VERBS for s in st) / (len(st) or 1)
        return ({"kind": "steps", "steps": st[:8], "source": rec["source"]}
                if len(st) >= 3 and acts >= 0.5 and rec.get("source") else None)
    if rec.get("kind") == "answer":
        a = rec.get("answer", "")
        return ({"kind": "answer", "answer": a, "source": rec["source"]}
                if 40 <= len(a) <= 480 and not FLUFF.search(a) and not legal.search(a) and rec.get("source") else None)
    return None


def conv(q, res, tools, r):
    now = dt.datetime(r.choice([2025, 2026, 2027]), r.randint(1, 12), r.randint(1, 28), r.randint(6, 22), r.randrange(60))
    plat = r.choice(["mac", "windows", "linux", "android", "ios"])
    ts = [t for t in tools if plat in t["platforms"]]
    query = q.strip().rstrip("?")
    result = {"kind": res["kind"], **({"steps": res["steps"]} if res["kind"] == "steps" else {"answer": res["answer"]}),
              "source": {"title": res["source"]["title"][:80], "url": res["source"]["url"]}}
    return [{"role": "system", "content": system_prompt(now, ts)}, {"role": "user", "content": q},
            {"role": "tool", "content": tool_call_text("web_search", {"query": query})},
            {"role": "result", "content": json.dumps(result, ensure_ascii=False, separators=(",", ":"))},
            {"role": "assistant", "content": template_reply("web_search", {"query": query}, result)}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest", default="data_web/harvest.jsonl")
    ap.add_argument("--old", default="data_copy2", help="existing chat/tool data to mix in (rows + codes)")
    ap.add_argument("--src", default="data", help="tokenizer, pretrain and validation files")
    ap.add_argument("--out", default="data_web")
    ap.add_argument("--phrasings", type=int, default=6)
    ap.add_argument("--web-share", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--vq", default="runs/vq/best.pt")
    a = ap.parse_args()
    r = random.Random(a.seed)
    src, old, out = Path(a.src), Path(a.old), Path(a.out)
    meta = json.loads((src / "meta.json").read_text())
    R, L = meta["R"], meta["block"]
    bpe, tools = BPE.load(src / "bpe.json"), load_registry()

    recs = [json.loads(l) for l in Path(a.harvest).read_text().splitlines() if l.strip()]
    test_bases = {x["base"].lower() for x in recs if x["split"] == "test"}
    train = [(x, clean(x)) for x in recs if x["split"] == "train"]
    train = [(x, c) for x, c in train if c]
    convs = []
    for x, c in train:
        qs = {x["question"]} | {question(x["qkind"], x["base"], r) for _ in range(a.phrasings * 2)}
        for q in list(qs)[: a.phrasings]:
            convs.append(conv(q, c, tools, r))
    mine = 0
    if USER_LOG.exists():
        for l in USER_LOG.read_text().splitlines():
            x = json.loads(l)
            c = clean(x)
            if c and not any(b in x["question"].lower() for b in test_bases):
                convs += [conv(x["question"], c, tools, r) for _ in range(3)]
                mine += 1
    r.shuffle(convs)
    fm = [f for f in (format_conv(bpe, cv, L, max_len=R, max_segment=R - 4 * L) for cv in convs) if f is not None]
    rows, bm = pack_rows(fm, R, bpe.pad, L)
    print(f"{len(train)} training topics with a clean answer (+{mine} of your own questions) -> {len(convs)} conversations "
          f"-> {len(rows)} rows ({bm.mean():.0%} trained blocks)", flush=True)

    old_tok = read_u16(old / "task.bin", (-1, R))
    old_bm = np.fromfile(old / "task_bmask.bin", dtype=np.uint8).reshape(len(old_tok), R // L)
    n_old = min(len(old_tok), int(len(rows) * (1 - a.web_share) / a.web_share))
    pick = np.sort(np.random.RandomState(a.seed).choice(len(old_tok), n_old, replace=False))
    out.mkdir(parents=True, exist_ok=True)
    np.concatenate([rows, old_tok[pick]]).tofile(out / "task.bin")
    np.concatenate([bm, old_bm[pick]]).tofile(out / "task_bmask.bin")
    (out / "new_rows.txt").write_text(str(len(rows)))
    for name in ("bpe.json", "meta.json", "pretrain.bin", "pretrain_val.bin", "pretrain.codes", "pretrain_val.codes",
                 "task_val.bin", "task_val_bmask.bin", "task_val.codes"):
        link(src / name, out / name)

    from encode_codes import encode_array
    from ncm.utils import load_vq, pick_device
    dev = pick_device("auto")
    vq, vcfg = load_vq(a.vq, dev)
    n = vcfg.n_codes
    new_codes = encode_array(vq, rows.reshape(-1, L), dev).reshape(len(rows), R // L, n)
    old_codes = read_u16(old / "task.codes", (len(old_tok), R // L, n))[pick]
    np.concatenate([new_codes, old_codes]).astype(np.uint16).tofile(out / "task.codes")
    print(f"rows: {len(rows)} web + {len(pick)} existing chat/tool = {len(rows) + len(pick)}; codes written", flush=True)
    for cv in convs[:2]:
        print("sample:", " | ".join(f"{m['role']}: {m['content'][:90]!r}" for m in cv[1:]))
    print("done")


if __name__ == "__main__":
    main()
