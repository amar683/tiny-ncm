"""100 everyday requests through the whole chat app (model + checks + tools), in dry-run mode.

python eval_chat.py [--ckpt runs/task6/best.pt] [--offline]

Gmail and Calendar are faked (a small made-up inbox and calendar) so nothing personal is read.
Wikipedia, web search, weather and world time are real network calls unless --offline.
Writes a report to runs/eval_chat.md.

Scores per request:
  tool      - right tool (or no tool for small talk)
  args      - every expected argument exact, after the app's check   (model alone in brackets)
  ran       - the tool ran without error
  reply     - the final reply is clean (the model's own words passed the check, or the result was used)
"""
import argparse
import datetime as dt
import json
import re
import time

from assistant.grounding import parse_call
from assistant.runtime import Assistant
from assistant.tools_mac import MacTools
from ncm.bpe import BPE
from ncm.chat_format import load_registry
from ncm.utils import load_generator, pick_device

NOW = dt.datetime(2026, 11, 18, 9, 14)        # a Wednesday morning

INBOX = [
    {"id": "m1", "from": "Sarah", "subject": "Lunch on Friday?", "snippet": "Hi! Are you free for lunch on Friday at 1pm? There's a new Thai place near the office."},
    {"id": "m2", "from": "Amazon", "subject": "Your order has shipped", "snippet": "Your order of a desk lamp has shipped and will arrive on Thursday."},
    {"id": "m3", "from": "Daniel", "subject": "Budget review moved", "snippet": "The budget review is moved to Monday at 10am. Please bring the Q3 numbers."},
    {"id": "m4", "from": "HR", "subject": "Holiday calendar", "snippet": "The office will be closed from December 24 to January 2."},
    {"id": "m5", "from": "Sarah", "subject": "Photos from the trip", "snippet": "Here are the photos from our hiking trip last weekend. The lake one is my favourite."},
    {"id": "m6", "from": "Bank", "subject": "Statement ready", "snippet": "Your monthly statement for October is ready to view in the app."},
]
EVENTS = {"today": [{"title": "standup", "time": "Wed 18 Nov 09:30"}, {"title": "dentist", "time": "Wed 18 Nov 16:00"}],
          "tomorrow": [{"title": "project review", "time": "Thu 19 Nov 11:00"}], "friday": [], "monday": [{"title": "budget review", "time": "Mon 23 Nov 10:00"}]}


class FakeGmail:
    def list(self, query="", n=5):
        items = INBOX
        for part in query.split():
            if part.startswith("from:"):
                items = [e for e in items if e["from"].lower() == part[5:].lower()]
        return items[: n or 5]

    def read(self, id):
        e = next((e for e in INBOX if e["id"] == id), INBOX[0])
        return {"from": e["from"], "subject": e["subject"], "body": e["snippet"]}


class TestTools(MacTools):
    def calendar_list(self, day="today"):
        return {"events": EVENTS.get(day.lower(), [])}

    def weather(self, location=None, when=None):
        if self.offline or not location:
            return {"location": location or "your location", "when": when or "now", "summary": "partly cloudy",
                    "temp_c": 14, "high_c": 16, "low_c": 9}
        return super().weather(location, when)

    def lookup(self, topic):
        if self.offline:
            return {"title": topic.title(), "extract": f"{topic.title()} is a well-known subject. It has a long history."}
        return super().lookup(topic)

    def web_search(self, query):
        if self.offline:
            return {"results": [{"title": query.title(), "snippet": f"Results about {query}."}]}
        return super().web_search(query)

    def get_time(self, place):
        if self.offline:
            return {"place": place, "time": "17:14"}
        return super().get_time(place)


C = []          # (category, message, tool or None, expected args)


def add(cat, tool, *cases):
    for c in cases:
        msg, want = (c, {}) if isinstance(c, str) else c
        C.append((cat, msg, tool, want))


add("alarm", "set_alarm", ("set an alarm for 7:45", {"time": "7:45"}), ("wake me up at 6 tomorrow", {"time": "6 tomorrow"}),
    ("alarm for 5:30am", {"time": "5:30am"}), ("set my alarm to 8am on saturday", {"time": "8am on saturday"}),
    ("wake me at 9:15", {"time": "9:15"}), ("can you set an alarm at 10pm", {"time": "10pm"}),
    ("please wake me up at 7am", {"time": "7am"}), ("alarm tomorrow at 6:20", {"time": "tomorrow at 6:20"}),
    ("set an alarm for noon", {"time": "noon"}))
add("reminder", "set_reminder", ("remind me to call the bank at 2pm", {"text": "call the bank", "time": "2pm"}),
    ("remind me to buy flowers for Anna tomorrow", {"text": "buy flowers for Anna", "time": "tomorrow"}),
    ("remind me to stretch in 45 minutes", {"text": "stretch", "time": "in 45 minutes"}),
    ("remind me to send the invoice on friday at 10am", {"text": "send the invoice", "time": "friday at 10am"}),
    ("remind me to feed the cat tonight", {"text": "feed the cat", "time": "tonight"}),
    ("remind me to check the oven in 20 minutes", {"text": "check the oven", "time": "in 20 minutes"}),
    ("remind me to renew my passport next monday", {"text": "renew my passport", "time": "next monday"}),
    ("remind me to text Kofi at 6:30pm", {"text": "text Kofi", "time": "6:30pm"}),
    ("remind me to drink water in 1 hour", {"text": "drink water", "time": "in 1 hour"}),
    ("remind me to book the hotel tomorrow at 8", {"text": "book the hotel", "time": "tomorrow at 8"}),
    ("remind me about the parcel at 4pm", {"text": "the parcel", "time": "4pm"}))
add("timer", "set_timer", ("set a timer for 5 minutes", {"duration": "5 minutes"}), ("timer 15 minutes", {"duration": "15 minutes"}),
    ("start a 2 hour timer", {"duration": "2 hour"}), ("set a 30 second timer", {"duration": "30 second"}),
    ("set a timer for 12 minutes for the pasta", {"duration": "12 minutes", "label": "the pasta"}),
    ("countdown 40 minutes", {"duration": "40 minutes"}), ("set a timer for 1 hour", {"duration": "1 hour"}))
add("calendar_add", "calendar_add", ("add gym on thursday at 7am", {"title": "gym", "time": "thursday at 7am"}),
    ("schedule a call with Rosa tomorrow at 2pm", {"title": "call with Rosa", "time": "tomorrow at 2pm"}),
    ("put team dinner on my calendar friday at 8pm", {"title": "team dinner", "time": "friday at 8pm"}),
    ("add haircut to my calendar on saturday at 11am", {"title": "haircut", "time": "saturday at 11am"}),
    ("schedule piano lesson monday at 5pm", {"title": "piano lesson", "time": "monday at 5pm"}),
    ("create an event standup review tomorrow at 9:30", {"title": "standup review", "time": "tomorrow at 9:30"}))
add("calendar_list", "calendar_list", ("what's on my calendar today?", {"day": "today"}), ("what do I have tomorrow?", {"day": "tomorrow"}),
    ("any meetings on monday?", {"day": "monday"}), ("show my schedule for friday", {"day": "friday"}),
    ("what's my day look like today", {"day": "today"}))
add("emails", "list_emails", ("check my email", {}), ("any new emails from Daniel?", {"query": "from:Daniel"}),
    ("show me my unread emails", {"query": "is:unread"}), ("summarize my last 2 emails", {"n": 2}),
    ("did Sarah email me?", {"query": "from:Sarah"}), ("read my latest 4 emails", {"n": 4}), ("what's in my inbox?", {}))
add("draft", "draft_email", ("write an email to Leo about the party on saturday", {"to": "Leo", "subject": "the party on saturday"}),
    ("draft an email to Mia about the new schedule", {"to": "Mia", "subject": "the new schedule"}),
    ("email Ken about the late delivery", {"to": "Ken", "subject": "the late delivery"}),
    ("write an email to Grace about tomorrow's meeting", {"to": "Grace", "subject": "tomorrow's meeting"}))
add("lookup", "lookup", ("who was Nikola Tesla?", {"topic": "Nikola Tesla"}), ("what is a volcano?", {"topic": "volcano"}),
    ("tell me about the Great Wall of China", {"topic": "the Great Wall of China"}), ("who is Serena Williams?", {"topic": "Serena Williams"}),
    ("what is DNA?", {"topic": "DNA"}), ("who was Frida Kahlo?", {"topic": "Frida Kahlo"}), ("what are black holes?", {"topic": "black holes"}),
    ("tell me about the Amazon rainforest", {"topic": "the Amazon rainforest"}), ("who was Alan Turing?", {"topic": "Alan Turing"}),
    ("what is inflation?", {"topic": "inflation"}))
add("web", "web_search", ("search the web for vegan lasagna recipe", {"query": "vegan lasagna recipe"}),
    ("google cheap flights to Rome", {"query": "cheap flights to Rome"}), ("search for best running shoes", {"query": "best running shoes"}),
    ("look up train times to Leeds online", {"query": "train times to Leeds"}), ("search python tutorials for kids", {"query": "python tutorials for kids"}))
add("weather", "weather", ("what's the weather in Paris?", {"location": "Paris"}), ("weather in Mumbai tomorrow", {"location": "Mumbai", "when": "tomorrow"}),
    ("is it cold in Oslo today?", {"location": "Oslo", "when": "today"}), ("will it rain tomorrow?", {"when": "tomorrow"}),
    ("how hot is it in Dubai?", {"location": "Dubai"}), ("do i need an umbrella today", {"when": "today"}),
    ("what's the weather like in Chicago this weekend", {"location": "Chicago"}))
add("notes_add", "notes_add", ("note that the door code is 2580", {"text": "the door code is 2580"}),
    ("remember that Jake likes green tea", {"text": "Jake likes green tea"}),
    ("make a note that the car is in level 3", {"text": "the car is in level 3"}),
    ("note down buy batteries and tape", {"text": "buy batteries and tape"}))
add("notes_search", "notes_search", ("what did I note about the door code?", {}), ("search my notes for green tea", {"query": "green tea"}))
add("calc", "calculator", ("what is 12 times 8?", {"expression": "12*8"}), ("what's 144 divided by 12", {"expression": "144/12"}),
    ("calculate 250 plus 375", {"expression": "250+375"}), ("what is 1000 minus 237?", {"expression": "1000-237"}),
    ("what's 20 percent of 150", {"expression": "20/100*150"}), ("how much is 7 times 49", {"expression": "7*49"}),
    ("what is 3.5 times 4?", {"expression": "3.5*4"}))
add("time", "get_time", ("what time is it in New York?", {"place": "New York"}), ("time in Dubai", {"place": "Dubai"}),
    ("what's the time in Seoul right now?", {"place": "Seoul"}), ("current time in Mexico City", {"place": "Mexico City"}))
add("app", "open_app", ("open whatsapp", {"app": "Whatsapp"}), ("launch Safari", {"app": "Safari"}), ("open the maps app", {"app": "Maps"}),
    ("start Notes", {"app": "Notes"}))
add("smalltalk", None, "hello!", "how are you doing?", "thank you so much", "good evening", "what's up?", "you're really helpful",
    "bye for now", "nice to meet you")
assert len(C) == 100, len(C)

# 53 more, written after the fixes above and never used to change the app (the honest check)
FRESH = []
_C, C = C, FRESH
add("alarm", "set_alarm", ("set alarm 6:10", {"time": "6:10"}), ("wake me up tomorrow at 5:45am", {"time": "tomorrow at 5:45am"}),
    ("i need an alarm at 7:30 on sunday", {"time": "7:30 on sunday"}), ("alarm at 9 please", {"time": "9"}))
add("reminder", "set_reminder", ("remind me to pay the electricity bill on thursday", {"text": "pay the electricity bill", "time": "thursday"}),
    ("remind me to call grandma at 11am", {"text": "call grandma", "time": "11am"}),
    ("remind me to move the car in 15 minutes", {"text": "move the car", "time": "in 15 minutes"}),
    ("remind me to bring the charger tomorrow morning", {"text": "bring the charger", "time": "tomorrow morning"}),
    ("can you remind me to reply to Hana at 3pm", {"text": "reply to Hana", "time": "3pm"}))
add("timer", "set_timer", ("timer for 8 minutes", {"duration": "8 minutes"}), ("set a 90 second timer", {"duration": "90 second"}),
    ("start a timer for 3 hours", {"duration": "3 hours"}), ("set a timer for 20 minutes for the laundry", {"duration": "20 minutes", "label": "the laundry"}))
add("calendar_add", "calendar_add", ("add yoga on tuesday at 6pm", {"title": "yoga", "time": "tuesday at 6pm"}),
    ("schedule a meeting with Ivan friday at 4pm", {"title": "meeting with Ivan", "time": "friday at 4pm"}),
    ("put dinner with parents on my calendar sunday at 7pm", {"title": "dinner with parents", "time": "sunday at 7pm"}))
add("calendar_list", "calendar_list", ("what's on my calendar for monday", {"day": "monday"}), ("do i have anything tomorrow", {"day": "tomorrow"}),
    ("show my agenda for today", {"day": "today"}))
add("emails", "list_emails", ("any emails from HR?", {"query": "from:HR"}), ("summarize my last 5 emails", {"n": 5}),
    ("has Daniel written me?", {"query": "from:Daniel"}), ("show my newest 3 emails", {"n": 3}))
add("draft", "draft_email", ("write an email to Nora about the rent", {"to": "Nora", "subject": "the rent"}),
    ("draft an email to Sam about next week's trip", {"to": "Sam", "subject": "next week's trip"}))
add("lookup", "lookup", ("who was Isaac Newton?", {"topic": "Isaac Newton"}), ("what is climate change?", {"topic": "climate change"}),
    ("tell me about the Roman Empire", {"topic": "the Roman Empire"}), ("who is Taylor Swift?", {"topic": "Taylor Swift"}),
    ("what is a hurricane?", {"topic": "hurricane"}), ("who was Mahatma Gandhi?", {"topic": "Mahatma Gandhi"}))
add("web", "web_search", ("search the web for easy pancake recipe", {"query": "easy pancake recipe"}),
    ("google hotels in Kyoto", {"query": "hotels in Kyoto"}), ("search for used bikes near me", {"query": "used bikes near me"}))
add("weather", "weather", ("what's the weather in Madrid?", {"location": "Madrid"}), ("weather in Cairo tomorrow", {"location": "Cairo", "when": "tomorrow"}),
    ("is it raining in Seattle?", {"location": "Seattle"}), ("how cold is it today", {"when": "today"}))
add("notes_add", "notes_add", ("note that the plumber comes on friday", {"text": "the plumber comes on friday"}),
    ("remember that my locker is number 42", {"text": "my locker is number 42"}))
add("calc", "calculator", ("what is 15 times 15?", {"expression": "15*15"}), ("what's 900 divided by 30", {"expression": "900/30"}),
    ("what is 48 plus 52?", {"expression": "48+52"}), ("what's 10 percent of 85", {"expression": "10/100*85"}))
add("time", "get_time", ("what time is it in London?", {"place": "London"}), ("time in Singapore right now", {"place": "Singapore"}))
add("app", "open_app", ("open youtube", {"app": "Youtube"}), ("launch the camera app", {"app": "Camera"}))
add("smalltalk", None, "hey!", "thanks a lot", "good afternoon", "see you later", "how's it going?")
C = _C
assert len(FRESH) == 53, len(FRESH)


def same(a, b):
    f = lambda x: re.sub(r"^(?:the|a|an) ", "", " ".join(str(x).lower().split()))
    return f(a) == f(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/task6/best.pt")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", choices=["main", "fresh"], default="main")
    a = ap.parse_args()
    gen, _ = load_generator(a.ckpt, pick_device("auto"))
    bpe = BPE.load("tokenizer/bpe.json")
    tools = TestTools(now=lambda: NOW, dry_run=True, gmail=FakeGmail())
    tools.offline = a.offline
    bot = Assistant(gen, bpe, tools, load_registry(), confirm=lambda c: True, now=lambda: NOW)

    a.out = a.out or ("runs/eval_chat.md" if a.set == "main" else "runs/eval_chat_fresh.md")
    cases = C if a.set == "main" else FRESH
    rows, t0 = [], time.time()
    for i, (cat, msg, tool, want) in enumerate(cases):
        ev = {"raw": [], "raw_routed": [], "tool": [], "result": [], "model_reply": None, "note": None}

        def on_event(kind, payload):
            if kind in ("raw", "tool", "result", "raw_routed"):
                ev[kind].append(payload)
            else:
                ev[kind] = payload
        reply = bot.ask(msg, on_event=on_event)
        bot.reset()
        call = ev["tool"][0][0] if ev["tool"] else None
        raw = parse_call(ev["raw"][0])[0] if ev["raw"] else None
        got_tool = call["name"] if call else None
        r = {"cat": cat, "msg": msg, "want_tool": tool, "got_tool": got_tool, "reply": reply,
             "tool_ok": got_tool == tool,
             "args_ok": tool is None and got_tool is None or (call is not None and got_tool == tool and
                                                            all(same(call["args"].get(k), v) for k, v in want.items())),
             "raw_args_ok": tool is None and got_tool is None or (raw is not None and raw.get("name") == tool and
                                                                all(same((raw.get("args") or {}).get(k), v) for k, v in want.items())),
             "ran": tool is None or bool(ev["result"]) and ev["result"][0].get("ok", True),
             "model_words": ev["model_reply"] is not None and reply == ev["model_reply"], "args": call and call["args"], "raw": ev["raw"][0][:160] if ev["raw"] else "",
             "result": ev["result"][0] if ev["result"] else None, "model_reply": ev["model_reply"]}
        rows.append(r)
        flag = "ok " if r["tool_ok"] and r["args_ok"] and r["ran"] else "BAD"
        print(f"{i + 1:3d} {flag} [{cat}] {msg}\n      -> {got_tool} {json.dumps(r['args'], ensure_ascii=False) if r['args'] else ''}"
              f"\n      reply ({'model' if r['model_words'] else 'from result'}): {reply[:140]!r}", flush=True)
    report(rows, a, time.time() - t0)


def report(rows, a, secs):
    n = len(rows)
    tot = lambda k: sum(r[k] for r in rows)
    cats = list(dict.fromkeys(r["cat"] for r in rows))
    lines = [f"# {n} everyday requests ({a.set} set): {a.ckpt}", "",
             f"Dry run, fixed time {NOW:%a %d %b %Y %H:%M}, fake inbox and calendar, {'offline' if a.offline else 'real'} Wikipedia/web/weather/time. {secs / 60:.1f} min.", "",
             "| | score |", "|---|---|",
             f"| Right tool (or no tool for small talk) | **{tot('tool_ok')}/{n}** |",
             f"| Every argument exact, with the app's check | **{tot('args_ok')}/{n}** |",
             f"| Every argument exact, model alone | {tot('raw_args_ok')}/{n} |",
             f"| Tool ran without error | {tot('ran')}/{n} |",
             f"| Reply in the model's own words (passed the reply check) | {tot('model_words')}/{n} |", "",
             "| Category | n | tool | args (check) | args (model alone) | own-words reply |", "|---|---|---|---|---|---|"]
    for c in cats:
        rs = [r for r in rows if r["cat"] == c]
        s = lambda k: sum(r[k] for r in rs)
        lines.append(f"| {c} | {len(rs)} | {s('tool_ok')} | {s('args_ok')} | {s('raw_args_ok')} | {s('model_words')} |")
    lines += ["", "## Failures (wrong tool, wrong argument or tool error)", ""]
    for r in rows:
        if not (r["tool_ok"] and r["args_ok"] and r["ran"]):
            lines.append(f"- **{r['msg']}** → `{r['got_tool']}` `{json.dumps(r['args'], ensure_ascii=False)}`"
                         f" (wanted `{r['want_tool']}`); model wrote `{r['raw']}`; result `{json.dumps(r['result'], ensure_ascii=False)[:120]}`")
    lines += ["", "## All replies", ""]
    for r in rows:
        lines.append(f"- {r['msg']} → {r['reply'][:200]!r}" + ("" if r["model_words"] else f"  *(model said: {str(r['model_reply'])[:80]!r})*"))
    open(a.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:20]))
    print(f"\nreport: {a.out}")


if __name__ == "__main__":
    main()
