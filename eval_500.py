"""500 new requests through the whole chat app (model + checks + tools), dry run, offline tool stubs.

  400 generated from templates with NEW names, tasks, places, topics and some new phrasings
      (everyday tools 300, small talk 30, math 70), scored exactly like eval_chat.py / eval_mathcode.py
  100 hard, written by hand: typos, indirect requests, numbers in words, tricky times, two requests in one,
      multi-step math, things no tool can do, long rambling messages, sending/calling

python eval_500.py [--ckpt runs/task6/best.pt] [--seed 500]   -> runs/eval_500.md
"""
import argparse
import json
import random
import re
import time

from eval_chat import NOW, FakeGmail, TestTools
from assistant.runtime import Assistant
from ncm.bpe import BPE
from ncm.chat_format import load_registry
from ncm.utils import load_generator, pick_device

NAMES = ["Tomasz", "Ingrid", "Yusuf", "Beatriz", "Kenji", "Olu", "Mateo", "Freya", "Arjun", "Leila", "Dmitri", "Siobhan",
         "Kwame", "Elif", "Rafael", "Noor", "Henrik", "Amara", "Joaquin", "Zara", "Bodhi", "Ines", "Taro", "Ayla"]
TASKS = ["return the library books", "call the plumber", "order printer ink", "pick up the dry cleaning", "email the landlord",
         "walk the dog", "defrost the chicken", "charge my laptop", "cancel the gym trial", "sign the permission slip",
         "buy a birthday card", "take out the recycling", "back up my phone", "pay the water bill", "book a haircut",
         "water the tomatoes", "renew the car insurance", "send the photos to Freya", "check the mailbox", "submit my timesheet"]
CITIES = ["Lisbon", "Nairobi", "Hanoi", "Montreal", "Krakow", "Lima", "Auckland", "Reykjavik", "Casablanca", "Helsinki",
          "Bogota", "Perth", "Valencia", "Osaka", "Accra", "Edinburgh", "Santiago", "Jakarta"]
TOPICS = ["the Great Barrier Reef", "Marie Antoinette", "plate tectonics", "the printing press", "Leonardo da Vinci", "the Silk Road",
          "quantum computing", "Nelson Mandela", "the water cycle", "the Hubble telescope", "Cleopatra", "the Renaissance",
          "honeybees", "the Sahara desert", "Ludwig van Beethoven", "the Moon landing", "volcanoes", "Ernest Hemingway"]
QUERIES = ["vegetarian chili recipe", "cheap hotels in Lisbon", "how to fix a squeaky door", "best hiking boots",
           "train tickets to Krakow", "yoga for beginners", "how to grow basil", "second hand bikes", "kids birthday party ideas",
           "how to remove coffee stains"]
EVENTS = ["dentist checkup", "coffee with Ingrid", "book club", "parent teacher meeting", "car service", "guitar lesson",
          "sprint planning", "vet appointment", "birthday dinner", "tennis with Kenji", "team offsite", "piano recital"]
APPS = ["Slack", "Zoom", "Spotify", "Photos", "Calendar", "Mail", "Messages", "Netflix", "Discord", "Figma", "Safari", "Notes"]
SUBJECTS = ["the quarterly report", "the weekend plans", "the broken heater", "the new contract", "Friday's dinner",
            "the team photos", "the invoice", "the school trip", "the rent increase", "the project deadline"]
FACTS = ["the wifi password is mango77", "the spare key is under the blue pot", "my bike lock code is 4419",
         "Kenji's birthday is in March", "the car is parked on level 2", "the plumber's number is 555 0199",
         "I owe Freya 20 dollars", "the meeting room is B12"]
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def clock(r):
    h, m = r.choice(range(5, 23)), r.choice([0, 0, 15, 30, 45, 10, 20, 40, 5, 50])
    h12, ap = h % 12 or 12, "am" if h < 12 else "pm"
    return r.choice([f"{h12}:{m:02d}", f"{h12}:{m:02d}{ap}", f"{h12}:{m:02d} {ap}"] if m else [f"{h12}{ap}", f"{h12} {ap}", f"{h12}:00"])


def when(r, relative=True):
    """A time as people say it -> (phrase in the sentence, expected arg)."""
    k = r.random()
    if relative and k < 0.2:
        n = r.choice([5, 10, 15, 20, 25, 30, 40, 45, 90])
        p = r.choice([f"in {n} minutes", f"in {r.choice([1, 2, 3])} hours"])
        return p, p
    c = clock(r)
    if k < 0.45:
        return f"at {c}", c
    if k < 0.65:
        return f"tomorrow at {c}", f"tomorrow at {c}"
    if k < 0.85:
        d = r.choice(DAYS)
        p = r.choice([f"on {d} at {c}", f"{d} at {c}", f"next {d} at {c}"])
        return p, re.sub(r"^on ", "", p)
    return f"at {c} tomorrow", f"{c} tomorrow"


def gen_everyday(r):
    C = []

    def add(cat, tool, msg, want):
        C.append(dict(cat=cat, msg=msg, tool=tool, want=want, kind="exact"))
    for _ in range(30):
        p, e = when(r, relative=False)
        msg = r.choice(["set an alarm {p}", "wake me up {p}", "alarm {p}", "i need an alarm {p}", "can you set an alarm {p}",
                        "please wake me {p}", "set my alarm {p}"]).format(p=p)
        add("alarm", "set_alarm", msg, {"time": e})
    for _ in range(40):
        p, e = when(r)
        x = r.choice(TASKS)
        msg = r.choice(["remind me to {x} {p}", "can you remind me to {x} {p}", "set a reminder to {x} {p}", "remind me {p} to {x}",
                        "please remind me to {x} {p}"]).format(x=x, p=p)
        add("reminder", "set_reminder", msg, {"text": x, "time": e})
    for _ in range(25):
        n, u = r.choice([2, 3, 5, 7, 10, 12, 15, 20, 25, 30, 45, 90]), r.choice(["minutes", "minutes", "seconds", "hours"])
        if u == "hours":
            n = r.choice([1, 2, 3])
            u = "hour" if n == 1 else "hours"
        d = f"{n} {u}"
        lab = r.choice([None, None, "the rice", "the laundry", "the oven", "my nap", "the tea"])
        if lab:
            msg, want = f"set a timer for {d} for {lab}", {"duration": d, "label": lab}
        else:
            msg, want = r.choice(["set a timer for {d}", "timer for {d}", "start a timer for {d}", "{d} timer please",
                                  "count down {d}"]).format(d=d), {"duration": d}
        add("timer", "set_timer", msg, want)
    for _ in range(25):
        ev = r.choice(EVENTS)
        p, e = when(r, relative=False)
        if p.startswith("at ") and r.random() < .7:
            p, e = f"on {r.choice(DAYS)} {p}", None
            e = p[3:]
        msg = r.choice(["add {ev} {p}", "schedule {ev} {p}", "put {ev} on my calendar {p}", "add {ev} to my calendar {p}",
                        "create an event {ev} {p}"]).format(ev=ev, p=p)
        add("calendar_add", "calendar_add", msg, {"title": ev, "time": e})
    for _ in range(15):
        d = r.choice(["today", "tomorrow"] + DAYS)
        msg = r.choice(["what's on my calendar {d}", "what do i have {d}", "any meetings {d}", "show my schedule for {d}",
                        "what's on my calendar for {d}?"]).format(d=d)
        add("calendar_list", "calendar_list", msg, {"day": d})
    for _ in range(25):
        k = r.random()
        nm = r.choice(NAMES)
        if k < 0.4:
            msg, want = r.choice(["any emails from {n}?", "did {n} email me?", "show emails from {n}", "has {n} emailed me"]).format(n=nm), {"query": f"from:{nm}"}
        elif k < 0.6:
            msg, want = r.choice(["show my unread emails", "any unread emails?", "read my unread mail"]), {"query": "is:unread"}
        elif k < 0.85:
            n = r.choice([2, 3, 4, 5, 6])
            msg, want = r.choice(["summarize my last {n} emails", "read my latest {n} emails", "show me my newest {n} emails"]).format(n=n), {"n": n}
        else:
            msg, want = r.choice(["check my email", "what's in my inbox?", "check my inbox"]), {}
        add("emails", "list_emails", msg, want)
    for _ in range(15):
        nm, sub = r.choice(NAMES), r.choice(SUBJECTS)
        msg = r.choice(["write an email to {n} about {s}", "draft an email to {n} about {s}", "email {n} about {s}"]).format(n=nm, s=sub)
        add("draft", "draft_email", msg, {"to": nm, "subject": sub})
    for _ in range(30):
        t = r.choice(TOPICS)
        bare = re.sub(r"^the ", "", t)
        msg = r.choice([f"who was {t}?" if t[0].isupper() and not t.startswith("the") else f"what is {t}?", f"tell me about {t}",
                        f"what is {t}?", f"tell me about {t} please"])
        add("lookup", "lookup", msg, {"topic": bare})
    for _ in range(15):
        q = r.choice(QUERIES)
        msg = r.choice(["search the web for {q}", "google {q}", "search for {q}", "look up {q} online"]).format(q=q)
        add("web", "web_search", msg, {"query": q})
    for _ in range(25):
        c, w = r.choice(CITIES), r.choice([None, "today", "tomorrow"])
        msg = r.choice(["what's the weather in {c}{w}", "weather in {c}{w}", "is it going to rain in {c}{w}", "how hot is it in {c}{w}",
                        "will it be cold in {c}{w}"]).format(c=c, w=f" {w}" if w else "")
        add("weather", "weather", msg, {"location": c, **({"when": w} if w else {})})
    for _ in range(15):
        f = r.choice(FACTS)
        msg = r.choice(["note that {f}", "remember that {f}", "make a note that {f}", "note down {f}"]).format(f=f)
        add("notes_add", "notes_add", msg, {"text": f})
    for _ in range(5):
        k = r.choice(["wifi password", "spare key", "bike lock", "plumber"])
        add("notes_search", "notes_search", r.choice(["what did i note about the {k}?", "search my notes for {k}"]).format(k=k), {})
    for _ in range(15):
        c = r.choice(CITIES)
        add("time", "get_time", r.choice(["what time is it in {c}?", "time in {c}", "what's the time in {c} right now"]).format(c=c), {"place": c})
    for _ in range(20):
        a = r.choice(APPS)
        add("app", "open_app", r.choice(["open {a}", "launch {a}", "start {a}", "open the {a} app"]).format(a=a), {"app": a})
    return C


SMALLTALK = ["hi", "hello there", "hey, good morning", "good night!", "thanks!", "thank you, that helps", "bye", "see you tomorrow",
             "how are you?", "how's your day going?", "you're awesome", "nice one", "ok cool", "great, thanks a lot", "good evening!",
             "what's up", "cheers", "talk to you later", "hey!", "good afternoon", "thanks for the help", "hi, how are you doing?",
             "morning!", "have a nice day", "goodbye", "thank you so much!", "hello!", "you're the best", "sounds good", "perfect, thanks"]


def gen_math(r):
    C = []

    def add(cat, msg, x):
        C.append(dict(cat=cat, msg=msg, tool="calculator", want=x, kind="number"))
    for _ in range(25):
        a, b = r.randint(2, 99), r.randint(2, 99)
        op = r.choice(["*", "+", "-", "/"])
        if op == "/":
            a = a * b
        x = eval(f"{a}{op}{b}")
        w = {"*": ["times", "multiplied by"], "+": ["plus"], "-": ["minus"], "/": ["divided by"]}[op]
        msg = r.choice([f"what is {a} {r.choice(w)} {b}?", f"what's {a} {r.choice(w)} {b}", f"how much is {a} {r.choice(w)} {b}",
                        f"calculate {a} {r.choice(w)} {b}", f"{a}{op}{b}", f"what is {a}{op}{b}?"])
        if op == "+" and r.random() < .3:
            msg = f"add {a} and {b}"
        if op == "-" and r.random() < .3:
            msg, x = f"subtract {b} from {a}", a - b
        add("arithmetic", msg, x)
    for _ in range(10):
        p, a = r.choice([5, 10, 15, 20, 25, 30, 40, 50, 75]), r.choice([40, 60, 80, 120, 200, 240, 300, 500, 1000])
        add("percent", r.choice([f"what's {p} percent of {a}", f"what is {p}% of {a}?", f"{p} percent of {a}"]), p * a / 100)
    for _ in range(15):
        k = r.randrange(8)
        n = r.randint(2, 20)
        if k == 0:
            add("other", f"what is the square root of {n * n}?", n)
        elif k == 1:
            add("other", f"what is half of {2 * n * 3}?", n * 3)
        elif k == 2:
            add("other", f"what is double {n * 7}?", n * 14)
        elif k == 3:
            add("other", f"what is {n} squared?", n * n)
        elif k == 4:
            m = r.randint(2, 6)
            add("other", f"what is {m} cubed?", m ** 3)
        elif k == 5:
            xs = [r.randint(1, 30) for _ in range(3)]
            add("other", f"what is the average of {xs[0]}, {xs[1]} and {xs[2]}?", sum(xs) / 3)
        elif k == 6:
            h = r.randint(2, 9)
            add("other", f"how many minutes are in {h} hours?", h * 60)
        else:
            km = r.randint(2, 9)
            add("other", f"convert {km} km to meters", km * 1000)
    for _ in range(20):
        k = r.randrange(6)
        a, b = r.randint(10, 60), r.randint(2, 9)
        nm, item = r.choice(NAMES), r.choice(["stickers", "marbles", "cookies", "apples", "books"])
        if k == 0:
            add("story", f"{nm} has {a} {item} and gives away {b}. How many are left?", a - b)
        elif k == 1:
            add("story", f"I had {a} dollars and spent {b}, how much do I have now?", a - b)
        elif k == 2:
            add("story", f"{nm} has {a} {item} and gets {b} more. How many now?", a + b)
        elif k == 3:
            add("story", f"one ticket costs {b} dollars, how much do {a} tickets cost?", a * b)
        elif k == 4:
            add("story", f"there are {a * b} chairs and {b} rows, how many chairs per row?", a)
        else:
            add("story", f"I run {b} km every day. How far do I run in a week?", b * 7)
    return C


# Hard cases: (category, message, check). check = dict with
#   tools: acceptable first tools (None = no tool), all: tools that must all be called, args: {arg: substring}, num: number in reply,
#   never: tools that must not be called
HARD = [
    # typos
    ("typo", "remnd me to by milk at 5pm", {"tools": ["set_reminder"], "args": {"time": "5pm"}}),
    ("typo", "set an alrm for 6:30", {"tools": ["set_alarm"], "args": {"time": "6:30"}}),
    ("typo", "whats the wether in paris", {"tools": ["weather"], "args": {"location": "paris"}}),
    ("typo", "opne spotify", {"tools": ["open_app"], "args": {"app": "spotify"}}),
    ("typo", "set a timr for 10 mins", {"tools": ["set_timer"], "args": {"duration": "10"}}),
    ("typo", "who was albert einstien", {"tools": ["lookup"], "args": {"topic": "einst"}}),
    ("typo", "remind me to cal mom tmrw at 9", {"tools": ["set_reminder"], "args": {"time": "9"}}),
    ("typo", "wat time is it in tokyo", {"tools": ["get_time"], "args": {"place": "tokyo"}}),
    ("typo", "chek my emails", {"tools": ["list_emails"]}),
    ("typo", "wake me up at 7 tommorow", {"tools": ["set_alarm"], "args": {"time": "7"}}),
    ("typo", "what is 25 tims 4", {"tools": ["calculator"], "num": 100}),
    ("typo", "serch the web for pasta recipes", {"tools": ["web_search"], "args": {"query": "pasta"}}),
    # indirect
    ("indirect", "I need to be up by 6:30 tomorrow", {"tools": ["set_alarm"], "args": {"time": "6:30"}}),
    ("indirect", "don't let me forget to call the vet at 4pm", {"tools": ["set_reminder"], "args": {"time": "4pm"}}),
    ("indirect", "is it umbrella weather in London today?", {"tools": ["weather"], "args": {"location": "london"}}),
    ("indirect", "I'm hungry, find pizza places near me", {"tools": ["web_search"]}),
    ("indirect", "my eggs need 9 minutes", {"tools": ["set_timer"], "args": {"duration": "9"}}),
    ("indirect", "I keep forgetting the gate code, it's 7781", {"tools": ["notes_add"], "args": {"text": "7781"}}),
    ("indirect", "am I busy on friday?", {"tools": ["calendar_list"], "args": {"day": "friday"}}),
    ("indirect", "has anyone emailed me today?", {"tools": ["list_emails"]}),
    ("indirect", "I want to listen to some music", {"tools": ["open_app", None]}),
    ("indirect", "should I wear a jacket in Oslo tomorrow?", {"tools": ["weather"], "args": {"location": "oslo"}}),
    ("indirect", "I wonder who invented the telephone", {"tools": ["lookup", "web_search"]}),
    ("indirect", "ping me in 20 minutes to check the oven", {"tools": ["set_reminder", "set_timer"]}),
    # numbers in words
    ("number words", "set an alarm for seven thirty", {"tools": ["set_alarm"], "args": {"time": ["seven", "7:30"]}}),
    ("number words", "remind me in ten minutes to stretch", {"tools": ["set_reminder"], "args": {"time": ["ten", "10"]}}),
    ("number words", "set a timer for five minutes", {"tools": ["set_timer"], "args": {"duration": ["five", "5"]}}),
    ("number words", "what is twelve times eight", {"tools": ["calculator"], "num": 96}),
    ("number words", "wake me at six am", {"tools": ["set_alarm"], "args": {"time": ["six", "6"]}}),
    ("number words", "summarize my last three emails", {"tools": ["list_emails"]}),
    ("number words", "what's fifty plus twenty five", {"tools": ["calculator"], "num": 75}),
    ("number words", "timer for half an hour", {"tools": ["set_timer"], "args": {"duration": ["half", "30"]}}),
    # tricky times
    ("tricky time", "remind me the day after tomorrow at 10am to pay rent", {"tools": ["set_reminder"], "args": {"time": ["day after tomorrow", "friday"]}}),
    ("tricky time", "set an alarm for quarter past 7", {"tools": ["set_alarm"], "args": {"time": ["quarter past 7", "7:15"]}}),
    ("tricky time", "wake me at half past six", {"tools": ["set_alarm"], "args": {"time": ["half past six", "6:30"]}}),
    ("tricky time", "remind me in an hour and a half to leave", {"tools": ["set_reminder"], "args": {"time": ["hour and a half", "90"]}}),
    ("tricky time", "add dentist next week on wednesday at 3pm", {"tools": ["calendar_add"], "args": {"time": "wednesday"}}),
    ("tricky time", "remind me at noon to eat lunch", {"tools": ["set_reminder"], "args": {"time": "noon"}}),
    ("tricky time", "set an alarm for 0630", {"tools": ["set_alarm"], "args": {"time": ["0630", "6:30"]}}),
    ("tricky time", "remind me this evening to call Olu", {"tools": ["set_reminder"], "args": {"time": "evening"}}),
    ("tricky time", "set an alarm for 18:45", {"tools": ["set_alarm"], "args": {"time": "18:45"}}),
    ("tricky time", "remind me on the 25th to pay rent", {"tools": ["set_reminder"], "args": {"time": "25"}}),
    # two requests in one
    ("two in one", "set an alarm for 7 and remind me to call mom at 9", {"all": ["set_alarm", "set_reminder"]}),
    ("two in one", "what's the weather in Rome and what time is it there", {"all": ["weather", "get_time"]}),
    ("two in one", "open Spotify and set a timer for 30 minutes", {"all": ["open_app", "set_timer"]}),
    ("two in one", "check my emails and then tell me the weather in Paris", {"all": ["list_emails", "weather"]}),
    ("two in one", "note that the code is 1234 and remind me tomorrow to change it", {"all": ["notes_add", "set_reminder"]}),
    ("two in one", "add lunch with Kenji on friday at 1pm and also remind me thursday to book a table", {"all": ["calendar_add", "set_reminder"]}),
    ("two in one", "what's 15 times 4 and what's 100 minus 37", {"all": ["calculator"]}),
    ("two in one", "who was Ada Lovelace and who was Charles Babbage", {"all": ["lookup"]}),
    ("two in one", "set timers for 5 and 10 minutes", {"all": ["set_timer"]}),
    ("two in one", "wake me at 6 and open the news", {"all": ["set_alarm", "open_app"]}),
    # multi-step math
    ("multi-step math", "a pizza has 8 slices and 3 friends eat 2 each, how many slices are left?", {"num": 2}),
    ("multi-step math", "I buy 3 shirts at 12 dollars each and pay with 50, how much change do I get?", {"num": 14}),
    ("multi-step math", "a recipe needs 2 eggs per cake. How many eggs for 3 cakes and 1 extra egg?", {"num": 7}),
    ("multi-step math", "I earn 15 dollars an hour and work 8 hours a day for 5 days. How much do I earn?", {"num": 600}),
    ("multi-step math", "there are 4 boxes with 6 apples each and I eat 5. How many apples are left?", {"num": 19}),
    ("multi-step math", "what is 10 plus 5 times 2?", {"num": 20}),
    ("multi-step math", "a train leaves at 3pm and the trip takes 2 and a half hours. When does it arrive?", {"num": 5.5, "alt": "5:30"}),
    ("multi-step math", "if I save 20 dollars a week, how many weeks to save 300?", {"num": 15}),
    ("multi-step math", "what is 20 percent of 50 plus 10?", {"num": 20}),
    ("multi-step math", "split a 90 dollar bill between 3 people and add 10 percent tip each", {"num": 33}),
    # no tool can do it (no refusal: it should just try something sensible, never send or call)
    ("no tool", "book me a flight to Paris", {"tools": ["web_search", None], "never": ["send_email", "sms", "call"]}),
    ("no tool", "turn off the living room lights", {"tools": [None, "open_app", "web_search"], "never": ["send_email", "sms", "call"]}),
    ("no tool", "order a pizza", {"tools": ["web_search", None, "open_app"], "never": ["send_email", "sms", "call"]}),
    ("no tool", "play my workout playlist", {"tools": ["open_app", None], "never": ["send_email", "sms", "call"]}),
    ("no tool", "translate good morning into Spanish", {"tools": [None, "web_search", "lookup"]}),
    ("no tool", "tell me a joke", {"tools": [None]}),
    ("no tool", "write a short poem about the sea", {"tools": [None]}),
    ("no tool", "what should I cook tonight?", {"tools": [None, "web_search"]}),
    ("no tool", "how do I make pancakes?", {"tools": [None, "web_search", "lookup"]}),
    ("no tool", "what's your name?", {"tools": [None]}),
    # long and rambling
    ("long", "hey so I was thinking, could you please set a reminder for me to pick up the kids from soccer at 5:30pm, thanks a lot!",
     {"tools": ["set_reminder"], "args": {"text": "pick up the kids", "time": "5:30pm"}}),
    ("long", "ok this is a weird one but I keep sleeping through my alarms, so can you set one for 6:15 tomorrow morning",
     {"tools": ["set_alarm"], "args": {"time": "6:15"}}),
    ("long", "I'm cooking rice right now and the package says it needs about 18 minutes, can you time that for me",
     {"tools": ["set_timer"], "args": {"duration": "18"}}),
    ("long", "my friend keeps talking about Frida Kahlo and honestly I have no idea who she is, can you tell me?",
     {"tools": ["lookup"], "args": {"topic": "frida kahlo"}}),
    ("long", "we are flying to Lisbon on saturday and I want to know if it will be warm there tomorrow",
     {"tools": ["weather"], "args": {"location": "lisbon"}}),
    ("long", "please put my mum's birthday dinner on the calendar for sunday at 7pm, it's at the Italian place",
     {"tools": ["calendar_add"], "args": {"time": "sunday at 7pm"}}),
    ("long", "I just parked the car on level 4 near the elevators, please make a note of that so I don't forget",
     {"tools": ["notes_add"], "args": {"text": "level 4"}}),
    ("long", "can you check whether Daniel sent me anything about the budget review? he said he would",
     {"tools": ["list_emails"]}),
    ("long", "before I forget, what's 18 percent of 240? I need it for the tip",
     {"tools": ["calculator"], "num": 43.2}),
    ("long", "so the meeting got moved and I need to write an email to Priya about the new meeting time, can you do that",
     {"tools": ["draft_email"], "args": {"to": "priya"}}),
    # sending and calling: must never happen without the confirm tap (the eval answers "no" to every confirm)
    ("sending", "send an email to Bob saying I'll be late", {"tools": ["draft_email", "send_email"], "never_ran": ["send_email"]}),
    ("sending", "text Sam that I'm on my way", {"tools": [None, "sms", "draft_email"], "never_ran": ["sms"]}),
    ("sending", "call mom", {"tools": [None, "call"], "never_ran": ["call"]}),
    ("sending", "email my boss that I quit", {"tools": ["draft_email", "send_email", None], "never_ran": ["send_email"]}),
    ("sending", "send Kenji the photos from last weekend", {"tools": [None, "draft_email", "send_email", "sms"], "never_ran": ["send_email", "sms"]}),
    # odd inputs
    ("odd", "", {"tools": [None]}),
    ("odd", "???", {"tools": [None]}),
    ("odd", "alarm", {"tools": ["set_alarm", None]}),
    ("odd", "REMIND ME TO CALL DAD AT 6PM", {"tools": ["set_reminder"], "args": {"time": "6pm"}}),
    ("odd", "set an alarm for 25:99", {"tools": ["set_alarm", None]}),
    ("odd", "remind me", {"tools": ["set_reminder", None]}),
    ("odd", "what is 1 divided by 0?", {"tools": ["calculator", None]}),
    ("odd", "weather", {"tools": ["weather"]}),
    ("odd", "🙂", {"tools": [None]}),
    ("odd", "asdfghjkl", {"tools": [None, "lookup", "web_search"]}),
    ("indirect", "it's getting late, wake me at 5:50 so I don't miss the train", {"tools": ["set_alarm"], "args": {"time": "5:50"}}),
    ("typo", "add yoga to my calender on tuesday at 6pm", {"tools": ["calendar_add"], "args": {"time": "tuesday at 6pm"}}),
    ("long", "honestly I can never remember, how many grams are in 3 kilograms?", {"tools": ["calculator"], "num": 3000}),
]


# 60 more hard cases, written BEFORE fixing the failures above and never used to change the app (the honest check)
HARD_FRESH = [
    ("typo", "remidn me to water the plants at 8pm", {"tools": ["set_reminder"], "args": {"time": "8pm"}}),
    ("typo", "set alarn for 5:45", {"tools": ["set_alarm"], "args": {"time": "5:45"}}),
    ("typo", "wheather in berlin tomorow", {"tools": ["weather"], "args": {"location": "berlin"}}),
    ("typo", "lauch zoom", {"tools": ["open_app"], "args": {"app": "zoom"}}),
    ("typo", "set a timmer for 15 minutes", {"tools": ["set_timer"], "args": {"duration": "15"}}),
    ("typo", "wat is 9 tims 7", {"tools": ["calculator"], "num": 63}),
    ("typo", "any emials from Kenji?", {"tools": ["list_emails"], "args": {"query": "kenji"}}),
    ("typo", "whats on my calender tomorrow", {"tools": ["calendar_list"], "args": {"day": "tomorrow"}}),
    ("indirect", "I have to leave the house by 7:15 tomorrow, wake me up", {"tools": ["set_alarm"], "args": {"time": "7:15"}}),
    ("indirect", "make sure I don't forget to take my pills at 10pm", {"tools": ["set_reminder"], "args": {"time": "10pm"}}),
    ("indirect", "is it warm enough for shorts in Madrid today?", {"tools": ["weather"], "args": {"location": "madrid"}}),
    ("indirect", "the cake needs 35 minutes in the oven", {"tools": ["set_timer"], "args": {"duration": "35"}}),
    ("indirect", "am I free on thursday?", {"tools": ["calendar_list"], "args": {"day": "thursday"}}),
    ("indirect", "did anyone write to me?", {"tools": ["list_emails"]}),
    ("indirect", "find sushi restaurants near me", {"tools": ["web_search"], "args": {"query": "sushi"}}),
    ("indirect", "I keep losing my parking spot, it's row F", {"tools": ["notes_add"], "args": {"text": "row f"}}),
    ("number words", "set an alarm for eight fifteen", {"tools": ["set_alarm"], "args": {"time": "8:15"}}),
    ("number words", "remind me in twenty minutes to call Leila", {"tools": ["set_reminder"], "args": {"time": "20"}}),
    ("number words", "timer for three minutes", {"tools": ["set_timer"], "args": {"duration": "3"}}),
    ("number words", "what is six times nine", {"tools": ["calculator"], "num": 54}),
    ("number words", "what's forty minus twelve", {"tools": ["calculator"], "num": 28}),
    ("number words", "wake me up at five thirty am", {"tools": ["set_alarm"], "args": {"time": "5:30"}}),
    ("tricky time", "wake me at quarter to 7", {"tools": ["set_alarm"], "args": {"time": "6:45"}}),
    ("tricky time", "remind me at half past 4 to call the bank", {"tools": ["set_reminder"], "args": {"time": "4:30"}}),
    ("tricky time", "set an alarm for 0545", {"tools": ["set_alarm"], "args": {"time": "5:45"}}),
    ("tricky time", "timer for an hour and a half", {"tools": ["set_timer"], "args": {"duration": "90"}}),
    ("tricky time", "remind me the day after tomorrow to renew my passport", {"tools": ["set_reminder"], "args": {"text": "renew my passport"}}),
    ("tricky time", "set an alarm for 21:30", {"tools": ["set_alarm"], "args": {"time": "21:30"}}),
    ("two in one", "set an alarm for 6 and a timer for 20 minutes", {"all": ["set_alarm", "set_timer"]}),
    ("two in one", "what's the weather in Tokyo and what time is it there", {"all": ["weather", "get_time"]}),
    ("two in one", "open Zoom and remind me at 3pm to join the call", {"all": ["open_app", "set_reminder"]}),
    ("two in one", "note that the wifi is guest123 and then open Safari", {"all": ["notes_add", "open_app"]}),
    ("two in one", "who was Nikola Tesla and who was Thomas Edison", {"all": ["lookup"]}),
    ("two in one", "what's 8 times 7 and what's 90 divided by 3", {"all": ["calculator"]}),
    ("two in one", "check my inbox and tell me the weather in London", {"all": ["list_emails", "weather"]}),
    ("two in one", "add gym on monday at 6am and remind me sunday night to pack my bag", {"all": ["calendar_add", "set_reminder"]}),
    ("multi-step math", "I buy 4 coffees at 3 dollars each and pay with 20, how much change?", {"num": 8}),
    ("multi-step math", "a class has 6 rows of 5 desks and 4 are broken. How many work?", {"num": 26}),
    ("multi-step math", "what is 6 plus 4 times 3?", {"num": 18}),
    ("multi-step math", "if I read 15 pages a day, how many days to finish 300 pages?", {"num": 20}),
    ("no tool", "book a table for two at an italian restaurant", {"tools": ["web_search", None], "never": ["send_email", "sms", "call"]}),
    ("no tool", "turn up the volume", {"tools": [None, "open_app"], "never": ["send_email", "sms", "call"]}),
    ("no tool", "who are you?", {"tools": [None]}),
    ("no tool", "what can you do?", {"tools": [None]}),
    ("no tool", "how do I change a tire?", {"tools": [None, "web_search", "lookup"]}),
    ("no tool", "buy me some new shoes", {"tools": [None, "web_search"], "never": ["send_email", "sms", "call"]}),
    ("long", "so tomorrow is a big day and I really can't be late, please set an alarm for 5:30 tomorrow",
     {"tools": ["set_alarm"], "args": {"time": "5:30"}}),
    ("long", "I'm making pasta and the water just started boiling, the box says 11 minutes, could you time it",
     {"tools": ["set_timer"], "args": {"duration": "11"}}),
    ("long", "I always forget, could you remind me to call the insurance company tomorrow at 9am please",
     {"tools": ["set_reminder"], "args": {"text": "call the insurance company", "time": "9am"}}),
    ("long", "my daughter asked me about the Eiffel Tower and I didn't know what to say, what is it?",
     {"tools": ["lookup", "web_search"]}),
    ("long", "we're thinking of a picnic in Valencia tomorrow, will it rain there?",
     {"tools": ["weather"], "args": {"location": "valencia"}}),
    ("long", "I just put my passport in the top drawer of the desk, make a note of that please",
     {"tools": ["notes_add"], "args": {"text": "top drawer"}}),
    ("sending", "send an email to Leila saying the meeting is cancelled", {"tools": ["draft_email", "send_email"], "never_ran": ["send_email"]}),
    ("sending", "text Kenji happy birthday", {"tools": [None, "sms", "draft_email"], "never_ran": ["sms"]}),
    ("sending", "call the dentist", {"tools": [None, "call", "web_search"], "never_ran": ["call"]}),
    ("odd", "hmm", {"tools": [None]}),
    ("odd", "SET A TIMER FOR 4 MINUTES", {"tools": ["set_timer"], "args": {"duration": "4"}}),
    ("odd", "remind me to", {"tools": ["set_reminder", None]}),
    ("odd", "timer", {"tools": ["set_timer", None]}),
    ("odd", "🎉🎉", {"tools": [None]}),
]


def nums(text):
    return [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)]


def same(a, b):
    f = lambda x: re.sub(r"^(?:the|a|an) ", "", " ".join(str(x).lower().split()))
    return f(a) == f(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/task6/best.pt")
    ap.add_argument("--seed", type=int, default=500)
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", choices=["main", "hard-fresh"], default="main")
    a = ap.parse_args()
    r = random.Random(a.seed)
    cases = gen_everyday(r) + [dict(cat="smalltalk", msg=m, tool=None, want={}, kind="exact") for m in SMALLTALK] + gen_math(r)
    cases += [dict(cat=c, msg=m, check=k, kind="hard") for c, m, k in HARD]
    assert len(cases) == 500, len(cases)
    if a.set == "hard-fresh":
        cases = [dict(cat=c, msg=m, check=k, kind="hard") for c, m, k in HARD_FRESH]
    a.out = a.out or ("runs/eval_500.md" if a.set == "main" else "runs/eval_hard_fresh.md")

    gen, _ = load_generator(a.ckpt, pick_device("auto"))
    bpe = BPE.load("tokenizer/bpe.json")
    tools = TestTools(now=lambda: NOW, dry_run=True, gmail=FakeGmail())
    tools.offline = True
    ran = []
    real_run = tools.run
    tools.run = lambda name, args: (ran.append(name), real_run(name, args))[1]
    bot = Assistant(gen, bpe, tools, load_registry(), confirm=lambda c: False, now=lambda: NOW)

    t0 = time.time()
    for i, c in enumerate(cases):
        calls, raw = [], []
        ran.clear()
        bot.reset()
        reply = bot.ask(c["msg"], on_event=lambda k, p: calls.append(p[0]) if k == "tool" and p[0] else raw.append(p) if k == "raw" else None) \
            if c["msg"] else "Sorry, I didn't get that."
        names = [x["name"] for x in calls]
        first = calls[0] if calls else None
        if c["kind"] == "exact":
            ok = (first is None if c["tool"] is None else
                  first is not None and first["name"] == c["tool"] and all(same(first["args"].get(k), v) for k, v in c["want"].items()))
            if c["tool"] is None and re.search(r"\d", reply):
                ok = False
        elif c["kind"] == "number":
            ok = any(abs(n - c["want"]) < 1e-6 for n in nums(reply))
        else:
            k = c["check"]
            ok = True
            if "tools" in k:
                ok &= (first["name"] if first else None) in k["tools"]
            if "all" in k:
                ok &= all(names.count(t) >= (2 if k["all"].count(t) == 1 and len(set(k["all"])) == 1 else 1) for t in k["all"])
            if "args" in k and first:
                ok &= all(any(x.lower() in str(first["args"].get(arg, "")).lower() for x in (v if isinstance(v, list) else [v]))
                          for arg, v in k["args"].items())   # a list = any of these counts (e.g. "seven" or "7:30")
            if "num" in k:
                ok &= any(abs(n - k["num"]) < 1e-6 for n in nums(reply)) or ("alt" in k and k["alt"] in reply)
            if "never" in k:
                ok &= not set(names) & set(k["never"])
            if "never_ran" in k:
                ok &= not set(ran) & set(k["never_ran"])
        c.update(ok=ok, reply=reply, calls=[(x["name"], x["args"]) for x in calls], ran=list(ran), raw=raw[0][:120] if raw else "")
        print(f"{i + 1:3d} {'ok ' if ok else 'BAD'} [{c['cat']}] {c['msg'][:70]} -> {c['calls'][:2]} | {reply[:80]!r}", flush=True)
    report(cases, a, time.time() - t0)


def report(cases, a, secs):
    groups = [("Everyday tools (generated)", [c for c in cases if c["kind"] == "exact" and c["cat"] != "smalltalk"]),
              ("Small talk", [c for c in cases if c["cat"] == "smalltalk"]),
              ("Math (generated)", [c for c in cases if c["kind"] == "number"]),
              ("Hard (hand-written)", [c for c in cases if c["kind"] == "hard"])]
    tot = sum(c["ok"] for c in cases)
    L = [f"# {len(cases)} new requests ({a.set}): {a.ckpt}", "",
         f"Dry run, offline tool stubs, fake inbox/calendar, seed {a.seed}, {secs / 60:.1f} min. **Total: {tot}/{len(cases)}.**", "",
         "| Group | n | correct |", "|---|---|---|"]
    L += [f"| {g} | {len(cs)} | **{sum(c['ok'] for c in cs)}** ({100 * sum(c['ok'] for c in cs) / len(cs):.0f}%) |" for g, cs in groups if cs]
    L += ["", "| Category | n | correct |", "|---|---|---|"]
    for cat in dict.fromkeys(c["cat"] for c in cases):
        cs = [c for c in cases if c["cat"] == cat]
        L.append(f"| {cat} | {len(cs)} | {sum(c['ok'] for c in cs)} |")
    L += ["", "## Failures", ""]
    for c in cases:
        if not c["ok"]:
            want = c.get("want") if c["kind"] != "hard" else c["check"]
            L.append(f"- [{c['cat']}] **{c['msg'] or '(empty)'}** → `{json.dumps(c['calls'], ensure_ascii=False)[:150]}` → {c['reply'][:100]!r}"
                     f"  (wanted `{json.dumps(want, ensure_ascii=False)[:100]}`)")
    L += ["", "## Hard questions, every answer", ""]
    for c in cases:
        if c["kind"] == "hard":
            L.append(f"- {'✅' if c['ok'] else '❌'} [{c['cat']}] {c['msg'] or '(empty)'} → `{json.dumps(c['calls'], ensure_ascii=False)[:140]}` → {c['reply'][:120]!r}")
    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L[:12 + len({c['cat'] for c in cases})]))


if __name__ == "__main__":
    main()
