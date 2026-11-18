"""Template generator for tool-use, summarization and lookup conversations.

Each generator returns a list of messages (see chat_format). Tool results use the same JSON
shapes the real runtimes return (assistant/tools_*.py), so training matches deployment.
"""
import datetime as dt
import json
import random

from .chat_format import load_registry, system_prompt, tool_call_text

NAMES = ["Alex", "Sam", "Priya", "Rahul", "Maria", "John", "Emma", "Liam", "Aisha", "Wei", "Carlos",
         "Sofia", "Ravi", "Neha", "David", "Sarah", "Tom", "Anna", "Omar", "Mei", "mom", "dad",
         "Grandma", "my boss", "Jessica", "Arjun", "Kevin", "Laura", "Ben", "Chloe"]
CITIES = ["London", "Delhi", "Mumbai", "New York", "Paris", "Tokyo", "Berlin", "Sydney", "Toronto",
          "Dubai", "Singapore", "San Francisco", "Bangalore", "Chicago", "Madrid", "Seoul", "Lagos"]
TASKS = ["call mom", "pay rent", "buy milk", "take my medicine", "water the plants", "submit the report",
         "book a dentist appointment", "pick up the kids", "renew my passport", "send the invoice",
         "go to the gym", "charge my phone", "reply to Alex", "check the oven", "feed the cat",
         "buy groceries", "call the bank", "prepare slides for the meeting", "pay the electricity bill",
         "walk the dog", "drink water", "stretch", "back up my laptop", "order a birthday cake",
         "return the library books", "take out the trash", "start the washing machine"]
EVENTS = ["team meeting", "lunch with Priya", "doctor appointment", "yoga class", "project review",
          "dinner with Sam", "call with the client", "parent-teacher meeting", "haircut", "standup",
          "interview", "movie night", "flight to Delhi", "birthday party"]
APPS = ["Spotify", "WhatsApp", "Camera", "Maps", "Calendar", "Notes", "Settings", "YouTube", "Chrome",
        "Gmail", "Photos", "Calculator", "Slack", "Instagram"]
WEATHER = ["sunny", "partly cloudy", "cloudy", "light rain", "heavy rain", "thunderstorms", "windy",
           "foggy", "snow", "clear skies"]
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
TIMER_UNITS = [("minute", 60), ("minutes", 60), ("min", 60), ("hour", 3600), ("hours", 3600),
               ("seconds", 1)]

ACKS = ["Done!", "All set.", "Okay!", "Sure.", "Got it.", "Done.", "No problem.", "Okay, done."]
THANKS = ["thanks", "thank you", "thanks!", "great, thanks", "cool", "perfect", "ok thanks"]
WELCOME = ["You're welcome!", "Anytime!", "Happy to help.", "No problem!", "Glad I could help."]


def fmt_dt(t):
    return t.strftime("%a %d %b %H:%M")


def fmt_clock(t, rng):
    h, m = t.hour, t.minute
    h12 = h % 12 or 12
    ap = "am" if h < 12 else "pm"
    if m == 0:
        return rng.choice([f"{h12} {ap}", f"{h12}{ap}", f"{h12}:00 {ap}", f"{h12} o'clock" if rng.random() < .3 else f"{h12}{ap}"])
    return rng.choice([f"{h12}:{m:02d} {ap}", f"{h12}:{m:02d}{ap}", f"{h12}:{m:02d}"])


def sample_when(now, rng):
    """-> (phrase as the user says it, resolved datetime)."""
    r = rng.random()
    if r < 0.2:  # relative
        n = rng.choice([5, 10, 15, 20, 25, 30, 40, 45, 90])
        if rng.random() < 0.3:
            h = rng.choice([1, 2, 3, 4])
            return rng.choice([f"in {h} hour{'s' if h > 1 else ''}", f"in {h}h"]), now + dt.timedelta(hours=h)
        return rng.choice([f"in {n} minutes", f"in {n} mins", f"{n} minutes from now"]), now + dt.timedelta(minutes=n)
    hour = rng.choice(list(range(5, 23)))
    minute = rng.choice([0, 0, 0, 15, 30, 30, 45, 10, 20, 40, 50, 5])
    day_off, dayword = 0, ""
    r2 = rng.random()
    if r2 < 0.35:
        day_off, dayword = 1, "tomorrow"
    elif r2 < 0.5:
        wd = rng.randrange(7)
        day_off = (wd - now.weekday()) % 7 or 7
        dayword = rng.choice(["on ", "", "next " if day_off > 1 else "on "]) + DAYS[wd]
    t = (now + dt.timedelta(days=day_off)).replace(hour=hour, minute=minute, second=0, microsecond=0)
    if day_off == 0 and t <= now:
        t += dt.timedelta(days=1)
        if rng.random() < 0.5:
            dayword = "tomorrow"
    clock = fmt_clock(t, rng)
    if hour == 12 and minute == 0 and rng.random() < 0.5:
        clock = "noon"
    at = rng.choice(["at ", "at ", "for ", ""]) + clock
    if dayword:
        phrase = rng.choice([f"{dayword} {at}", f"{at} {dayword}"])
    elif hour >= 19 and rng.random() < 0.3:
        phrase = f"tonight {at}"
    else:
        phrase = at
    return phrase.strip(), t


def clean_time_arg(phrase):
    for p in ("at ", "for "):
        phrase = phrase.replace(p, "") if phrase.startswith(p) else phrase
    return phrase


class Synth:
    def __init__(self, seed=0, emails=None, articles=None, wiki=None, chats=None):
        self.rng = random.Random(seed)
        self.tools = load_registry()
        self.emails = emails or []      # [{"body", "subject"}]
        self.articles = articles or []  # [{"article", "highlights"}]
        self.wiki = wiki or []          # [{"title", "text"}]
        self.chats = chats or []        # [[{"role", "content"}...]] small talk
        self.gens = [
            (self.alarm, 8), (self.reminder, 10), (self.timer, 5), (self.cal_add, 5), (self.cal_list, 4),
            (self.list_emails, 9), (self.read_email, 3), (self.draft_email, 4), (self.send_email, 3),
            (self.summarize_pasted_email, 8), (self.summarize_article, 5), (self.lookup, 10),
            (self.web_search, 3), (self.weather, 5), (self.notes, 4), (self.calc, 4), (self.time_q, 3),
            (self.open_app, 2), (self.call_sms, 4), (self.multi, 4), (self.smalltalk, 10),
        ]

    # ---- helpers ------------------------------------------------------------------------
    def now(self):
        r = self.rng
        return dt.datetime(r.choice([2025, 2026, 2027]), r.randint(1, 12), r.randint(1, 28),
                           r.randint(6, 22), r.randrange(60))

    def system(self, now, mobile=None):
        r = self.rng
        mobile = r.random() < 0.5 if mobile is None else mobile
        plat = r.choice(["android", "ios"]) if mobile else r.choice(["mac", "windows", "linux"])
        tools = [t for t in self.tools if plat in t["platforms"]]
        return {"role": "system", "content": system_prompt(now, tools)}

    def call(self, name, args):
        return {"role": "tool", "content": tool_call_text(name, args)}

    def result(self, obj):
        return {"role": "result", "content": json.dumps(obj, ensure_ascii=False, separators=(",", ":"))}

    def say(self, text):
        return {"role": "assistant", "content": text}

    def user(self, text):
        r = self.rng
        if r.random() < 0.15:
            text = text.lower()
        if r.random() < 0.2:
            text = r.choice(["hey, ", "hi, ", "please ", "can you ", "could you ", "", "yo "]) + text
        if r.random() < 0.15:
            text += r.choice([" please", " thanks", "?", "!", " pls"])
        return {"role": "user", "content": text}

    def maybe_thanks(self, conv):
        if self.rng.random() < 0.15:
            conv += [self.user(self.rng.choice(THANKS)), self.say(self.rng.choice(WELCOME))]
        return conv

    def sample(self):
        gens, w = zip(*self.gens)
        while True:
            conv = self.rng.choices(gens, weights=w)[0]()
            if conv:
                return conv

    # ---- task generators ----------------------------------------------------------------
    def alarm(self):
        r, now = self.rng, self.now()
        phrase, t = sample_when(now, r)
        label = r.choice([None, None, None, "gym", "work", "meeting", "medicine", "flight"])
        u = r.choice(["wake me up {p}", "set an alarm {p}", "alarm {p}", "set alarm for {p}",
                      "i need to get up {p}", "please set an alarm {p}", "wake me {p}"]).format(p=phrase)
        if label:
            u += f" for {label}"
        args = {"time": clean_time_arg(phrase)}
        if label:
            args["label"] = label
        reply = r.choice(["{a} Alarm set for {t}.", "{a} I'll wake you at {t}.", "Your alarm is set for {t}.",
                          "{a} Alarm for {t} is on."]).format(a=r.choice(ACKS), t=fmt_dt(t))
        return self.maybe_thanks([self.system(now), self.user(u), self.call("set_alarm", args),
                                  self.result({"ok": True, "time": fmt_dt(t)}), self.say(reply)])

    def reminder(self):
        r, now = self.rng, self.now()
        task = r.choice(TASKS)
        if r.random() < 0.12:  # no time given
            u = r.choice(["remind me to {k}", "add a reminder to {k}", "don't let me forget to {k}"]).format(k=task)
            return [self.system(now), self.user(u), self.call("set_reminder", {"text": task}),
                    self.result({"ok": True, "reminder": task}),
                    self.say(r.choice(["{a} I added a reminder to {k}.", "Reminder saved: {k}."]).format(a=r.choice(ACKS), k=task))]
        phrase, t = sample_when(now, r)
        u = r.choice(["remind me to {k} {p}", "remind me {p} to {k}", "set a reminder to {k} {p}",
                      "{p} remind me to {k}", "can you remind me to {k} {p}", "reminder: {k} {p}"]).format(k=task, p=phrase)
        conv = [self.system(now), self.user(u), self.call("set_reminder", {"text": task, "time": clean_time_arg(phrase)})]
        if r.random() < 0.05:
            conv += [self.result({"ok": False, "error": "permission denied for reminders"}),
                     self.say("I couldn't create the reminder because the app doesn't have permission to use Reminders. Please allow access in settings and try again.")]
            return conv
        conv += [self.result({"ok": True, "reminder": task, "time": fmt_dt(t)}),
                 self.say(r.choice(["{a} I'll remind you to {k} on {t}.", "Reminder set: {k}, {t}.",
                                    "{a} You'll get a reminder to {k} at {t}."]).format(a=r.choice(ACKS), k=task, t=fmt_dt(t)))]
        return self.maybe_thanks(conv)

    def timer(self):
        r, now = self.rng, self.now()
        unit, sec = r.choice(TIMER_UNITS)
        n = r.choice([1, 2, 3, 5, 10, 15, 20, 25, 30, 45]) if sec != 3600 else r.choice([1, 2, 3])
        if sec == 1:
            n = r.choice([30, 45, 90])
        if n == 1 and unit.endswith("s"):
            unit = unit[:-1]
        dur = f"{n} {unit}"
        label = r.choice([None, None, "pasta", "eggs", "laundry", "break", "tea"])
        u = r.choice(["set a timer for {d}", "timer {d}", "start a {d} timer", "countdown {d}"]).format(d=dur)
        if label:
            u += f" for the {label}"
        args = {"duration": dur}
        if label:
            args["label"] = label
        end = now + dt.timedelta(seconds=n * sec)
        return [self.system(now), self.user(u), self.call("set_timer", args),
                self.result({"ok": True, "ends": end.strftime("%H:%M:%S")}),
                self.say(r.choice(["{a} Timer set for {d}.", "{d} timer started.", "{a} I'll let you know in {d}."]).format(a=r.choice(ACKS), d=dur))]

    def cal_add(self):
        r, now = self.rng, self.now()
        ev = r.choice(EVENTS)
        phrase, t = sample_when(now, r)
        u = r.choice(["add {e} to my calendar {p}", "schedule {e} {p}", "put {e} on my calendar {p}",
                      "create an event {e} {p}"]).format(e=ev, p=phrase)
        return self.maybe_thanks([self.system(now), self.user(u), self.call("calendar_add", {"title": ev, "time": clean_time_arg(phrase)}),
                                  self.result({"ok": True, "event": ev, "time": fmt_dt(t)}),
                                  self.say(r.choice(["{a} Added {e} on {t}.", "{e} is on your calendar for {t}."]).format(a=r.choice(ACKS), e=ev.capitalize() if r.random() < .5 else ev, t=fmt_dt(t)))])

    def cal_list(self):
        r, now = self.rng, self.now()
        day = r.choice(["today", "tomorrow", "friday", "monday", "this week"])
        u = r.choice(["what's on my calendar {d}", "what do i have {d}", "any meetings {d}", "show my schedule for {d}"]).format(d=day)
        n = r.choice([0, 1, 2, 3, 4])
        evs = []
        for _ in range(n):
            _, t = sample_when(now, r)
            evs.append({"title": r.choice(EVENTS), "time": fmt_dt(t)})
        evs.sort(key=lambda e: e["time"])
        if not evs:
            reply = r.choice([f"Your calendar is clear {day}.", f"Nothing scheduled {day}."])
        else:
            lines = "\n".join(f"- {e['time']}: {e['title']}" for e in evs)
            reply = f"You have {n} event{'s' if n > 1 else ''} {day}:\n{lines}"
        return [self.system(now), self.user(u), self.call("calendar_list", {"day": day}), self.result({"events": evs}), self.say(reply)]

    def _email_items(self, n, sender=None):
        r = self.rng
        items = []
        for i in range(n):
            e = r.choice(self.emails)
            body = " ".join(e["body"].split())[:500]
            items.append({"id": f"m{r.randrange(10**6)}", "from": sender or r.choice(NAMES[:20] + ["Amazon", "Bank", "HR", "Newsletter"]),
                          "subject": e["subject"], "snippet": body})
        return items

    def list_emails(self):
        if not self.emails:
            return None
        r, now = self.rng, self.now()
        n = r.choice([1, 2, 3, 3, 4, 5, 6, 8])
        sender = r.choice(NAMES[:20]) if r.random() < 0.3 else None
        if sender:
            u = r.choice(["any emails from {s}?", "summarize emails from {s}", "what did {s} email me about"]).format(s=sender)
            args = {"query": f"from:{sender}", "n": n}
        else:
            u = r.choice(["summarize my last {n} emails", "check my email", "what's in my inbox", "read my latest emails",
                          "summarize my unread emails", "any new emails?", "give me a summary of my last {n} emails"]).format(n=n)
            args = {"query": "is:unread" if "unread" in u or "new" in u else "", "n": n}
        items = self._email_items(n, sender)
        if r.random() < 0.08:
            items = []
        res = self.result({"emails": items})
        if not items:
            reply = r.choice(["You have no new emails.", "Nothing new in your inbox.", f"No emails from {sender}." if sender else "Your inbox is empty."])
        else:
            lines = "\n".join(f"- {e['from']}: {e['subject']}" for e in items)
            head = r.choice([f"You have {len(items)} email{'s' if len(items) > 1 else ''}:", "Here's a summary:", "Here are your latest emails:"])
            reply = f"{head}\n{lines}"
        return self.maybe_thanks([self.system(now), self.user(u), self.call("list_emails", args), res, self.say(reply)])

    def read_email(self):
        if not self.emails:
            return None
        r, now = self.rng, self.now()
        items = self._email_items(r.choice([2, 3, 4]))
        pick = r.randrange(len(items))
        e = items[pick]
        conv = [self.system(now), self.user("check my email"), self.call("list_emails", {"query": "", "n": len(items)}),
                self.result({"emails": items}),
                self.say("Here are your latest emails:\n" + "\n".join(f"- {x['from']}: {x['subject']}" for x in items)),
                self.user(r.choice(["open the one from {s}", "what does {s}'s email say", "read the email from {s}"]).format(s=e["from"])),
                self.call("read_email", {"id": e["id"]}),
                self.result({"from": e["from"], "subject": e["subject"], "body": e["snippet"]}),
                self.say(f"{e['from']} wrote about \"{e['subject']}\". In short: {e['subject'].rstrip('.')}.")]
        return conv

    def draft_email(self, send=False):
        r, now = self.rng, self.now()
        to = r.choice(NAMES[:20])
        topic, body = r.choice([
            ("running late", "I'm running about 15 minutes late, sorry for the delay."),
            ("meeting tomorrow", "Just confirming our meeting tomorrow. Let me know if the time still works."),
            ("thank you", "Thank you so much for your help this week. I really appreciate it."),
            ("report", "Please find the report attached. Let me know if you have any questions."),
            ("sick leave", "I'm not feeling well today and will be taking a sick day."),
            ("invoice", "Could you please send me the invoice for last month?"),
            ("weekend plans", "Are you free this weekend? It would be great to catch up."),
            ("project update", "The project is on track and we should finish by Friday."),
        ])
        verb = "send" if send else r.choice(["write", "draft"])
        u = f"{verb} an email to {to} about {topic}"
        text = f"Hi {to},\n\n{body}\n\nBest regards"
        args = {"to": to, "subject": topic.capitalize(), "body": text}
        if send:
            return [self.system(now), self.user(u), self.call("send_email", args), self.result({"ok": True}),
                    self.say(r.choice([f"Sent your email to {to}.", f"{r.choice(ACKS)} Email sent to {to}."]))]
        return [self.system(now), self.user(u), self.call("draft_email", args), self.result({"ok": True, "draft_id": f"d{r.randrange(10**5)}"}),
                self.say(f"I saved a draft to {to}:\n\n{text}")]

    def send_email(self):
        return self.draft_email(send=True)

    def summarize_pasted_email(self):
        if not self.emails:
            return None
        r, now = self.rng, self.now()
        e = r.choice(self.emails)
        u = r.choice(["summarize this email:\n\n{b}", "tl;dr:\n{b}", "what is this email about?\n\n{b}",
                      "{b}\n\nsummarize that in one line", "give me a one line summary:\n{b}"]).format(b=e["body"].strip()[:6000])
        return [self.system(now), {"role": "user", "content": u}, self.say(e["subject"].strip())]

    def summarize_article(self):
        if not self.articles:
            return None
        r, now = self.rng, self.now()
        a = r.choice(self.articles)
        u = r.choice(["summarize this article:\n\n{b}", "{b}\n\nsummarize", "key points of this:\n{b}",
                      "tl;dr\n\n{b}"]).format(b=a["article"].strip()[:9000])
        hl = [h.strip() for h in a["highlights"].split("\n") if h.strip()]
        reply = "\n".join(f"- {h}" for h in hl) if r.random() < 0.6 else " ".join(hl)
        return [self.system(now), {"role": "user", "content": u}, self.say(reply)]

    def lookup(self):
        if not self.wiki:
            return None
        r, now = self.rng, self.now()
        w = r.choice(self.wiki)
        title = w["title"]
        u = r.choice(["who was {t}?", "who is {t}", "what is {t}?", "tell me about {t}", "what's {t}",
                      "explain {t}", "give me info on {t}", "{t}?"]).format(t=title)
        sents = [s.strip() for s in w["text"].replace("\n", " ").split(". ") if s.strip()]
        extract = ". ".join(sents[:5])[:1200]
        ans = ". ".join(sents[: r.choice([1, 2, 2, 3])]).rstrip(".") + "."
        return self.maybe_thanks([self.system(now), self.user(u), self.call("lookup", {"topic": title}),
                                  self.result({"title": title, "extract": extract}), self.say(ans)])

    def web_search(self):
        if not self.wiki:
            return None
        r, now = self.rng, self.now()
        ws = r.sample(self.wiki, 3)
        q = ws[0]["title"]
        u = r.choice(["search the web for {q}", "google {q}", "look up {q} online", "search {q}"]).format(q=q)
        res = [{"title": w["title"], "snippet": w["text"].replace("\n", " ")[:220]} for w in ws]
        first = res[0]["snippet"].split(". ")[0].rstrip(".") + "."
        return [self.system(now), self.user(u), self.call("web_search", {"query": q}), self.result({"results": res}),
                self.say(f"Top result: {res[0]['title']}. {first}")]

    def weather(self):
        r, now = self.rng, self.now()
        city = r.choice(CITIES) if r.random() < 0.7 else None
        when = r.choice(["today", "tomorrow", "now", "this weekend", None])
        u = r.choice(["what's the weather{c}{w}", "weather{c}{w}", "will it rain{c}{w}", "how hot is it{c}{w}",
                      "do i need an umbrella{w}{c}"]).format(c=f" in {city}" if city else "", w=f" {when}" if when else "")
        args = {}
        if city:
            args["location"] = city
        if when:
            args["when"] = when
        cond = r.choice(WEATHER)
        temp = r.randint(-5, 40)
        res = {"location": city or "your location", "when": when or "now", "summary": cond, "temp_c": temp,
               "high_c": temp + r.randint(0, 6), "low_c": temp - r.randint(0, 8)}
        rain = "rain" in cond or "storm" in cond
        extra = (" Take an umbrella." if rain else " No rain expected.") if ("rain" in u or "umbrella" in u) else ""
        reply = f"{res['location'].capitalize() if res['location'] == 'your location' else res['location']}, {res['when']}: {cond}, {temp}°C (high {res['high_c']}°C, low {res['low_c']}°C).{extra}"
        return [self.system(now), self.user(u), self.call("weather", args), self.result(res), self.say(reply)]

    def notes(self):
        r, now = self.rng, self.now()
        if r.random() < 0.6:
            txt = r.choice(["wifi password is sunflower42", "parking spot B12", "gift idea for Sam: headphones",
                            "buy eggs, bread and coffee", "the plumber's number is 555-0142", "book idea: a story about a lighthouse",
                            "meeting notes: launch moved to next month"])
            return [self.system(now), self.user(r.choice(["note: {t}", "save a note {t}", "remember that {t}", "write down {t}"]).format(t=txt)),
                    self.call("notes_add", {"text": txt}), self.result({"ok": True}), self.say(r.choice(["Saved.", "Noted!", "Got it, note saved."]))]
        q = r.choice(["wifi", "parking", "plumber", "gift"])
        hits = [{"text": t, "date": fmt_dt(self.now())} for t in {"wifi": ["wifi password is sunflower42"], "parking": ["parking spot B12"],
                                                                 "plumber": ["the plumber's number is 555-0142"], "gift": ["gift idea for Sam: headphones"]}[q]]
        return [self.system(now), self.user(r.choice(["what did i note about {q}", "find my note about {q}", "search notes for {q}"]).format(q=q)),
                self.call("notes_search", {"query": q}), self.result({"notes": hits}), self.say(f"Your note says: {hits[0]['text']}.")]

    def calc(self):
        r, now = self.rng, self.now()
        a, b = r.randint(2, 999), r.randint(2, 99)
        op, sym = r.choice([("+", "plus"), ("-", "minus"), ("*", "times"), ("/", "divided by")])
        expr = f"{a}{op}{b}"
        val = eval(expr)
        val = round(val, 4) if isinstance(val, float) else val
        if r.random() < 0.25:
            p = r.choice([5, 10, 15, 18, 20, 25])
            expr, val = f"{a}*{p}/100", round(a * p / 100, 4)
            u = r.choice([f"what's {p}% of {a}", f"{p} percent of {a}", f"tip of {p}% on {a}"])
        else:
            u = r.choice([f"what is {a} {sym} {b}", f"{a}{op}{b}", f"calculate {a} {op} {b}", f"how much is {a} {sym} {b}"])
        return [self.system(now), self.user(u), self.call("calculator", {"expression": expr}), self.result({"result": val}),
                self.say(r.choice([f"It's {val}.", f"{val}", f"That comes to {val}."]))]

    def time_q(self):
        r, now = self.rng, self.now()
        if r.random() < 0.6:
            u = r.choice(["what time is it", "what's the date today", "what day is it", "time?"])
            if "date" in u or "day" in u:
                reply = f"Today is {now.strftime('%A, %d %B %Y')}."
            else:
                reply = f"It's {now.strftime('%H:%M')}."
            return [self.system(now), self.user(u), self.say(reply)]
        city = r.choice(CITIES)
        t = (now + dt.timedelta(hours=r.randint(-10, 10))).strftime("%H:%M")
        return [self.system(now), self.user(f"what time is it in {city}"), self.call("get_time", {"place": city}),
                self.result({"place": city, "time": t}), self.say(f"It's {t} in {city}.")]

    def open_app(self):
        r, now = self.rng, self.now()
        app = r.choice(APPS)
        return [self.system(now), self.user(r.choice(["open {a}", "launch {a}", "start {a}"]).format(a=app)),
                self.call("open_app", {"app": app}), self.result({"ok": True}), self.say(r.choice([f"Opening {app}.", f"{app} is open."]))]

    def call_sms(self):
        r, now = self.rng, self.now()
        who = r.choice(NAMES)
        if r.random() < 0.4:
            return [self.system(now, mobile=True), self.user(r.choice(["call {w}", "phone {w}", "ring {w}", "give {w} a call"]).format(w=who)),
                    self.call("call", {"contact": who}), self.result({"ok": True}), self.say(f"Calling {who}.")]
        msg = r.choice(["I'm running late", "I'll be home by 7", "can you pick up milk?", "happy birthday!",
                        "on my way", "call me when you're free", "the meeting is moved to 3pm"])
        return [self.system(now, mobile=True),
                self.user(r.choice(["text {w} that {m}", "send a message to {w}: {m}", "tell {w} {m}", "sms {w} {m}"]).format(w=who, m=msg)),
                self.call("sms", {"to": who, "message": msg}), self.result({"ok": True}), self.say(f"Sent to {who}: \"{msg}\"")]

    def multi(self):
        r, now = self.rng, self.now()
        if self.emails and r.random() < 0.5:
            s = r.choice(NAMES[:20])
            items = self._email_items(r.choice([1, 2]), s)
            phrase, t = sample_when(now, r)
            task = f"reply to {s}"
            return [self.system(now), self.user(f"check if {s} emailed me and remind me to reply {phrase}"),
                    self.call("list_emails", {"query": f"from:{s}", "n": 5}), self.result({"emails": items}),
                    self.call("set_reminder", {"text": task, "time": clean_time_arg(phrase)}),
                    self.result({"ok": True, "reminder": task, "time": fmt_dt(t)}),
                    self.say(f"{s} sent you {len(items)} email{'s' if len(items) > 1 else ''}:\n" + "\n".join(f"- {e['subject']}" for e in items)
                             + f"\nI'll remind you to reply on {fmt_dt(t)}.")]
        city = r.choice(CITIES)
        cond = r.choice(WEATHER)
        temp = r.randint(0, 35)
        phrase, t = sample_when(now, r)
        rain = "rain" in cond or "storm" in cond
        conv = [self.system(now), self.user(f"what's the weather in {city} tomorrow? if it rains remind me to take an umbrella {phrase}"),
                self.call("weather", {"location": city, "when": "tomorrow"}),
                self.result({"location": city, "when": "tomorrow", "summary": cond, "temp_c": temp, "high_c": temp + 3, "low_c": temp - 4})]
        if rain:
            conv += [self.call("set_reminder", {"text": "take an umbrella", "time": clean_time_arg(phrase)}),
                     self.result({"ok": True, "reminder": "take an umbrella", "time": fmt_dt(t)}),
                     self.say(f"Tomorrow in {city}: {cond}, {temp}°C. I set a reminder to take an umbrella on {fmt_dt(t)}.")]
        else:
            conv += [self.say(f"Tomorrow in {city}: {cond}, {temp}°C. No rain expected, so I didn't set a reminder.")]
        return conv

    def smalltalk(self):
        if not self.chats:
            return None
        now = self.now()
        c = self.rng.choice(self.chats)
        return [self.system(now)] + c
