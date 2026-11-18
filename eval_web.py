"""Does the model copy web answers into its own reply? Held-out topics only (never collected for training).

python eval_web.py --ckpt runs/web1/best.pt [--harvest data_web/harvest.jsonl] [--per-topic 4]

The search result is replayed from the harvest file, so every model sees exactly the same pages.
  routed     - the question went to web_search
  own reply  - the model's own reply, before the app's check, compared with the answer it should copy:
               exact (same text), and word recall (share of the answer's words it reproduced, in order)
"""
import argparse
import json
import random
import re
from pathlib import Path

from assistant.grounding import template_reply
from assistant.runtime import Assistant
from eval_chat import NOW, FakeGmail, TestTools
from harvest_web import question
from make_web_data import clean
from ncm.bpe import BPE
from ncm.chat_format import load_registry
from ncm.utils import load_generator, pick_device


def lcs_recall(ref, hyp):
    a, b = re.findall(r"\w+", ref.lower()), re.findall(r"\w+", hyp.lower())
    if not a:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1] / len(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/web1/best.pt")
    ap.add_argument("--harvest", default="data_web/harvest.jsonl")
    ap.add_argument("--per-topic", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    recs = [json.loads(l) for l in Path(a.harvest).read_text().splitlines() if l.strip()]
    tests = [(x, clean(x)) for x in recs if x["split"] == "test"]
    tests = [(x, c) for x, c in tests if c]
    r = random.Random(5)
    cases = []
    for x, c in tests:
        qs = list(dict.fromkeys([x["question"]] + [question(x["qkind"], x["base"], r) for _ in range(12)]))[: a.per_topic]
        cases += [(q, c) for q in qs]

    gen, _ = load_generator(a.ckpt, pick_device("auto"))
    bpe = BPE.load("tokenizer/bpe.json")
    tools = TestTools(now=lambda: NOW, dry_run=True, gmail=FakeGmail())
    tools.offline = True
    bot = Assistant(gen, bpe, tools, load_registry(), confirm=lambda c: False, now=lambda: NOW)
    rows = []
    for q, c in cases:
        res = {"kind": c["kind"], **({"steps": c["steps"]} if c["kind"] == "steps" else {"answer": c["answer"]}),
               "source": {"title": c["source"]["title"][:80], "url": c["source"]["url"]}}
        tools.web_search = lambda query, res=res: res
        ev = {"tool": None, "model_reply": None}
        bot.reset()
        bot.ask(q, on_event=lambda k, p: ev.update(tool=p[0]["name"] if p[0] else None) if k == "tool" and ev["tool"] is None
                else ev.update(model_reply=p) if k == "model_reply" else None)
        want = template_reply("web_search", {"query": q}, res)
        own = ev["model_reply"] or ""
        rows.append(dict(q=q, kind=c["kind"], routed=ev["tool"] == "web_search", exact=" ".join(own.split()) == " ".join(want.split()),
                         recall=lcs_recall(want, own) if ev["tool"] == "web_search" else 0.0, own=own, want=want))
        x = rows[-1]
        print(f"{'ok  ' if x['exact'] else format(x['recall'], '.2f')} [{c['kind']}] {q} -> {own[:100]!r}", flush=True)
    n = len(rows)
    agg = lambda rs, k: sum(x[k] for x in rs) / (len(rs) or 1)
    L = [f"# Web answers on {len(tests)} held-out topics ({n} questions): {a.ckpt}", "",
         "| | all | steps | short answers |", "|---|---|---|---|"]
    st, an = [x for x in rows if x["kind"] == "steps"], [x for x in rows if x["kind"] == "answer"]
    L.append(f"| questions | {n} | {len(st)} | {len(an)} |")
    L.append(f"| sent to web search | {agg(rows, 'routed'):.0%} | {agg(st, 'routed'):.0%} | {agg(an, 'routed'):.0%} |")
    L.append(f"| own reply exactly right | {agg(rows, 'exact'):.0%} | {agg(st, 'exact'):.0%} | {agg(an, 'exact'):.0%} |")
    L.append(f"| own reply word recall | {agg(rows, 'recall'):.0%} | {agg(st, 'recall'):.0%} | {agg(an, 'recall'):.0%} |")
    L += ["", "## Samples", ""]
    for x in rows[:: max(1, n // 25)]:
        L.append(f"- **{x['q']}**\n  - model: {x['own'][:300]!r}\n  - should be: {x['want'][:300]!r}")
    out = a.out or f"runs/eval_web_{Path(a.ckpt).parent.name}.md"
    Path(out).write_text("\n".join(L) + "\n")
    print("\n".join(L[:8]))


if __name__ == "__main__":
    main()
