"""Score tool calls end to end: model output alone vs model + app-side check (assistant/grounding.py).

python eval_tools.py --ckpt runs/task6/best.pt [--set dev|new] [-v]
  dev = the 27 requests used while building the check; new = 30 requests it never saw.
A call counts as exact only if the tool and every expected argument match (case-insensitive).
"""
import argparse
import datetime as dt
import json
import re

import torch

from assistant.grounding import check_call, parse_call
from ncm.bpe import BPE
from ncm.chat_format import load_registry, prompt_tokens, system_prompt
from ncm.sampling import generate
from ncm.utils import load_generator, pick_device

DEV = [
    ("set an alarm for 6:30", "set_alarm", {"time": "6:30"}),
    ("wake me up at 7 tomorrow", "set_alarm", {"time": "7 tomorrow"}),
    ("alarm at 5am please", "set_alarm", {"time": "5am"}),
    ("remind me to pay rent tomorrow at 9", "set_reminder", {"text": "pay rent", "time": "tomorrow at 9"}),
    ("remind me to buy milk at 5pm", "set_reminder", {"text": "buy milk", "time": "5pm"}),
    ("remind me to call dad in 2 hours", "set_reminder", {"text": "call dad", "time": "in 2 hours"}),
    ("set a timer for 10 minutes", "set_timer", {"duration": "10 minutes"}),
    ("start a 25 minute timer for focus", "set_timer", {"duration": "25 minute", "label": "focus"}),
    ("what's on my calendar tomorrow?", "calendar_list", {"day": "tomorrow"}),
    ("add dentist appointment on friday at 3pm", "calendar_add", {"title": "dentist appointment", "time": "friday at 3pm"}),
    ("show my unread emails", "list_emails", {"query": "is:unread"}),
    ("any emails from Sarah?", "list_emails", {"query": "from:Sarah"}),
    ("summarize my last 3 emails", "list_emails", {"n": 3}),
    ("who was Ada Lovelace?", "lookup", {"topic": "Ada Lovelace"}),
    ("what is photosynthesis?", "lookup", {"topic": "photosynthesis"}),
    ("tell me about the Eiffel Tower", "lookup", {"topic": "the Eiffel Tower"}),
    ("search the web for best pizza near me", "web_search", {"query": "best pizza near me"}),
    ("what's the weather in London tomorrow?", "weather", {"location": "London", "when": "tomorrow"}),
    ("will it rain today?", "weather", {"when": "today"}),
    ("note that the wifi password is sunflower", "notes_add", {"text": "the wifi password is sunflower"}),
    ("what is 23 times 47?", "calculator", {"expression": "23*47"}),
    ("what time is it in Tokyo?", "get_time", {"place": "Tokyo"}),
    ("open spotify", "open_app", {"app": "Spotify"}),
    ("write an email to Priya about the meeting moving to Monday", "draft_email", {"to": "Priya", "subject": "the meeting moving to Monday"}),
    ("hi! how are you?", None, {}), ("thanks, that's all", None, {}), ("good morning!", None, {}),
]

NEW = [  # never looked at while writing the rules
    ("set an alarm for 8:15 tomorrow", "set_alarm", {"time": "8:15 tomorrow"}),
    ("wake me up at 6am on saturday", "set_alarm", {"time": "6am on saturday"}),
    ("can you set an alarm for 11pm", "set_alarm", {"time": "11pm"}),
    ("remind me to water the plants tonight", "set_reminder", {"text": "water the plants", "time": "tonight"}),
    ("remind me to email Jordan at 4pm", "set_reminder", {"text": "email Jordan", "time": "4pm"}),
    ("remind me to take my medicine in 30 minutes", "set_reminder", {"text": "take my medicine", "time": "in 30 minutes"}),
    ("remind me to pick up Lena from school tomorrow at 3:30", "set_reminder", {"text": "pick up Lena from school", "time": "tomorrow at 3:30"}),
    ("set a timer for 45 seconds", "set_timer", {"duration": "45 seconds"}),
    ("timer for 3 minutes for the tea", "set_timer", {"duration": "3 minutes", "label": "tea"}),
    ("put lunch with Marco on thursday at 1pm", "calendar_add", {"title": "lunch with Marco", "time": "thursday at 1pm"}),
    ("schedule a team call tomorrow at 10am", "calendar_add", {"title": "team call", "time": "tomorrow at 10am"}),
    ("what do I have on monday?", "calendar_list", {"day": "monday"}),
    ("check my emails from Daniel", "list_emails", {"query": "from:Daniel"}),
    ("read me my latest 5 emails", "list_emails", {"n": 5}),
    ("who is Marie Curie?", "lookup", {"topic": "Marie Curie"}),
    ("what is a black hole?", "lookup", {"topic": "black hole"}),
    ("tell me about Mount Kilimanjaro", "lookup", {"topic": "Mount Kilimanjaro"}),
    ("search for cheap flights to Lisbon", "web_search", {"query": "cheap flights to Lisbon"}),
    ("what's the weather like in Berlin?", "weather", {"location": "Berlin"}),
    ("is it going to be cold in Toronto tomorrow?", "weather", {"location": "Toronto", "when": "tomorrow"}),
    ("note that the gate code is 4471", "notes_add", {"text": "the gate code is 4471"}),
    ("what is 128 divided by 4?", "calculator", {"expression": "128/4"}),
    ("what's 19 plus 256", "calculator", {"expression": "19+256"}),
    ("what time is it in Sydney?", "get_time", {"place": "Sydney"}),
    ("open the calculator app", "open_app", {"app": "Calculator"}),
    ("launch whatsapp", "open_app", {"app": "Whatsapp"}),
    ("draft an email to Omar about the invoice", "draft_email", {"to": "Omar", "subject": "the invoice"}),
    ("hey there", None, {}), ("you're awesome, thank you", None, {}), ("good night", None, {}),
]


def same(a, b):
    f = lambda x: re.sub(r"^(?:the|a|an) ", "", " ".join(str(x).lower().split()))     # "the tea" = "tea"
    return f(a) == f(b)


def exact(call, tool, want):
    if tool is None:
        return call is None
    return call is not None and call.get("name") == tool and all(same(call.get("args", {}).get(k), v) for k, v in want.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/task6/best.pt")
    ap.add_argument("--set", choices=["dev", "new", "both"], default="both")
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    dev = pick_device("auto")
    g = load_generator(a.ckpt, dev)
    gen = g[0] if isinstance(g, tuple) else g
    gen.eval()
    bpe, reg = BPE.load("tokenizer/bpe.json"), load_registry()
    now = dt.datetime(2026, 11, 18, 9, 14)
    sets = {"dev": DEV, "new": NEW}
    for nm in (["dev", "new"] if a.set == "both" else [a.set]):
        cases = sets[nm]
        right_tool = raw_ok = fixed_ok = 0
        for q, tool, want in cases:
            msgs = [{"role": "system", "content": system_prompt(now, reg)}, {"role": "user", "content": q}]
            with torch.no_grad():
                out = generate(gen, prompt_tokens(bpe, msgs, gen.cfg.block), max_blocks=4, temperature=0.0,
                               stop_ids=(bpe.end,), pad_id=bpe.pad)
            s = bpe.decode(out, skip_special=False).replace("<|pad|>", "").replace("<|end|>", "")
            raw = fixed = None
            if s.startswith("<|tool|>"):
                p, ok = parse_call(s)
                raw = p if ok else None
                fixed, notes = check_call(s, q, reg)
            got_tool = (raw or fixed or {}).get("name") if s.startswith("<|tool|>") else None
            right_tool += got_tool == tool
            r, f = exact(raw, tool, want), exact(fixed, tool, want)
            raw_ok += r
            fixed_ok += f
            if a.v or not f:
                print(f"  {'ok ' if f else 'BAD'} [{q}]\n       model: {s[:150]!r}\n       fixed: {fixed and fixed['args']}")
        n = len(cases)
        print(f"{nm}: {n} requests | right tool {right_tool}/{n} | exact: model alone {raw_ok}/{n} -> with check {fixed_ok}/{n}\n")


if __name__ == "__main__":
    main()
