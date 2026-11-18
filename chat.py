"""Terminal chat with tiny-ncm and real Mac tools.

  python chat.py                         # chat (model: runs/task6/best.pt)
  python chat.py --dry-run               # tools that change things only say what they would do
  python chat.py --show-levels           # watch each 32-token block form coarse -> fine (NCM levels)
  python chat.py --connect-gmail         # one-time Gmail sign-in (read-only); see assistant/gmail.py
  python chat.py -q "set a timer for 1 minute"   # one message, then exit

In the chat: /reset clears the conversation, /quit exits.
"""
import argparse
import json
import sys
from pathlib import Path

from assistant import gmail as gmail_mod
from assistant.runtime import Assistant
from assistant.tools_mac import MacTools
from ncm.bpe import BPE
from ncm.chat_format import load_registry
from ncm.utils import load_generator, pick_device

DIM, BOLD, CYAN, YEL, RST = "\033[2m", "\033[1m", "\033[36m", "\033[33m", "\033[0m"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=next((p for p in ("weights/tiny-ncm.pt", "runs/task6/best.pt") if Path(p).exists()), "weights/tiny-ncm.pt"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--show-levels", action="store_true")
    ap.add_argument("--connect-gmail", action="store_true")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--keep-turns", type=int, default=0, help="earlier turns the model sees (it was trained on single requests)")
    ap.add_argument("-q", "--query", action="append", help="send this message (repeatable) and exit")
    a = ap.parse_args()

    gm = None
    try:
        gm = gmail_mod.connect(interactive=a.connect_gmail)
    except SystemExit as e:
        print(e)
    except Exception as e:
        print(f"{DIM}(Gmail not available: {str(e)[:100]}){RST}")

    gen, _ = load_generator(a.ckpt, pick_device("auto"))
    bpe = BPE.load("tokenizer/bpe.json")
    tools = MacTools(dry_run=a.dry_run, gmail=gm)

    def confirm(call):
        print(f"{YEL}  {call['name']} {json.dumps(call['args'], ensure_ascii=False)}{RST}")
        try:
            return input(f"{YEL}  Do it? [y/N] {RST}").strip().lower() in ("y", "yes")
        except EOFError:
            return False

    bot = Assistant(gen, bpe, tools, load_registry(), confirm=confirm, temperature=a.temp, keep_turns=a.keep_turns)

    def on_event(kind, payload):
        if kind == "tool":
            call, notes = payload
            if call:
                print(f"{CYAN}  → {call['name']}({json.dumps(call['args'], ensure_ascii=False)}){RST}")
            for n in notes:
                print(f"{DIM}    fixed: {n}{RST}")
        elif kind == "note":
            print(f"{DIM}    fixed {payload}{RST}")
        elif kind == "result":
            print(f"{DIM}  ← {json.dumps(payload, ensure_ascii=False)[:220]}{RST}")
        elif kind == "answer":
            if a.show_levels:
                print()
            print(f"{BOLD}tiny-ncm:{RST} {payload}")

    def on_level(b, k, toks):
        text = bpe.decode([t for t in toks if t != bpe.pad], skip_special=False).replace("\n", " ")
        print(f"\r\033[K{DIM}  block {b} level {k + 1}/5: {text[:110]}{RST}", end="", flush=True)

    flags = [x for x, on in (("dry run", a.dry_run), ("Gmail", gm is not None)) if on]
    if not a.query:
        print(f"tiny-ncm ({a.ckpt}{', ' + ', '.join(flags) if flags else ''}). /reset, /quit")
    msgs = a.query or iter(lambda: input(f"{BOLD}you:{RST} "), None)
    try:
        for msg in msgs:
            msg = msg.strip()
            if a.query:
                print(f"{BOLD}you:{RST} {msg}")
            if not msg:
                continue
            if msg in ("/quit", "/exit"):
                break
            if msg == "/reset":
                bot.reset()
                print(f"{DIM}(conversation cleared){RST}")
                continue
            bot.ask(msg, on_event=on_event, on_level=on_level if a.show_levels else None)
    except (EOFError, KeyboardInterrupt):
        print()
    if tools.timers and any(t.is_alive() for t in tools.timers) and not a.query:
        print(f"{DIM}(alarms and timers only ring while chat is open){RST}")


if __name__ == "__main__":
    sys.exit(main())
