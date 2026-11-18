"""Tests for assistant/grounding.py using real (garbled) outputs from runs/task6. Run: python test_grounding.py"""
from assistant.grounding import check_call, time_span, dedupe, parse_call
from ncm.chat_format import load_registry

REG = load_registry()

# (user message, what the model wrote, expected args after the check)
CASES = [
    ("set an alarm for 6:30", '{"name":"set_alarm","args":{"time":"6:30"}}', {"time": "6:30"}),
    ("wake me up at 7 tomorrow", '{"name":"set_alarm","args":{"time":"7 tomorrow tomorrow"}}', {"time": "7 tomorrow"}),
    ("alarm at 5am please", '{"name":"set_alarm","args":{"time":"5am mins"}}', {"time": "5am"}),
    ("remind me to pay rent tomorrow at 9", '{"name":"set_reminder","args":{"text":"pay rent","time":"tomorrow at at 9"}}',
     {"text": "pay rent", "time": "tomorrow at 9"}),
    ("remind me to buy milk at 5pm", '{"name":"set_reminder","args":{"text":"buy milk","time":"5pmpm"}}', {"text": "buy milk", "time": "5pm"}),
    ("remind me to call dad in 2 hours", '{"name":"set_reminder","args":{"text":"go Sam","time":"in 2 hours"}}',
     {"text": "call dad", "time": "in 2 hours"}),
    ("set a timer for 10 minutes", '{"name":"set_timer","args":{"duration":"10 minutes"}}', {"duration": "10 minutes"}),
    ("start a 25 minute timer for focus", '{"name":"set_timer","args":{"duration":"25 minute","label":"eggs"}}',
     {"duration": "25 minute", "label": "focus"}),
    ("what's on my calendar tomorrow?", '{"name":"calendar_list","args":{"day":"tomorrow"}}', {"day": "tomorrow"}),
    ("add dentist appointment on friday at 3pm", '{"name":"calendar_add","args":{"title":"dent the birthday","time":"tomorrow at 3pm"}}',
     {"title": "dentist appointment", "time": "friday at 3pm"}),
    ("show my unread emails", '{"name":"list_emails","args":{"query":"is:unread","n":3}}', {"query": "is:unread"}),
    ("any emails from Sarah?", '{"name":"list_emails","args":{"query":"from:Sarah","n":1}}', {"query": "from:Sarah"}),
    ("summarize my last 3 emails", '{"name":"list_emails","args":{"query":"","n":3}}', {"query": "", "n": 3}),
    ("who was Ada Lovelace?", '{"name":"lookup","args":{"topic":"Pha Lovelace"}}', {"topic": "Ada Lovelace"}),
    ("what is photosynthesis?", '{"name":"lookup","args":{"topic":"Wunive"}}', {"topic": "photosynthesis"}),
    ("tell me about the Eiffel Tower", '{"name":"lookup","args":{"topic":"The Eiffel Tower"}}', {"topic": "The Eiffel Tower"}),
    ("search the web for best pizza near me", '{"name":"web_search","args":{"query":"Good pizza runningtown"}}', {"query": "best pizza near me"}),
    ("what's the weather in London tomorrow?", '{"name":"weather","args":{"location":"Singapore","when":"tomorrow"}}',
     {"location": "London", "when": "tomorrow"}),
    ("will it rain today?", '{"name":"weather","args":{"when":"today"}}', {"when": "today"}),
    ("note that the wifi password is sunflower", '{"name":"notes_add","args":{"text":"wifi password is a42"}}',
     {"text": "the wifi password is sunflower"}),
    ("what is 23 times 47?", '{"name":"calculator","args":{"expression":"23*10"}}', {"expression": "23*47"}),
    ("what time is it in Tokyo?", '{"name":"get_time","args":{"place":"laydney"}}', {"place": "Tokyo"}),
    ("open spotify", '{"name":"open_app","args":{"app":"Spotify"}}', {"app": "Spotify"}),
    ("write an email to Priya about the meeting moving to Monday",
     '{"name":"draft_email","args":{"to":"Priya","subject":"InThe meeting next brought","email',
     {"to": "Priya", "subject": "the meeting moving to Monday", "body": "Hi Priya,\n\nI wanted to get in touch about the meeting moving to Monday.\n\nBest regards"}),
    # a few extra phrasings the model never saw in this test
    ("call mom", '{"name":"call","args":{"contact":"Tom"}}', {"contact": "Mom"}),
    ("what's 15 percent of 80", '{"name":"calculator","args":{"expression":"15*8"}}', {"expression": "15/100*80"}),
    ("wake me at 6:45 next monday", '{"name":"set_alarm","args":{"time":"6:45 monday"}}', {"time": "6:45 next monday"}),
    ("weather in new york tomorrow", '{"name":"weather","args":{"location":"New Yrk","when":"tomorrow"}}',
     {"location": "New York", "when": "tomorrow"}),
]


def main():
    fails = 0
    for user, model, want in CASES:
        call, notes = check_call(model, user, REG)
        got = call["args"] if call else None
        ok = got == want
        fails += not ok
        print(("ok  " if ok else "FAIL"), f"[{user}]", got, "" if ok else f"\n      want {want}\n      notes {notes}")
    # small unit checks
    assert dedupe("5pmpm") == "5pm" and dedupe("tomorrow at at 9") == "tomorrow at 9"
    assert time_span("remind me to pay rent tomorrow at 9")[0] == "tomorrow at 9"
    assert time_span("call dad in 2 hours")[0] == "in 2 hours"
    assert parse_call('{"name":"lookup","args":{"topic":"x"}}')[1] is True
    assert check_call('{"name":"send_email","args":{"to":"Al","subject":"hi","body":"b"}}', "email Al about hi", REG)[0]["confirm"] is True
    c = check_call('{"name":"set_email","args":{}}', "did Sarah email me?", REG)[0]
    assert c is None or not c["confirm"]                       # a garbled name is never turned into a sending tool
    assert check_call('{"name":"fly_to_moon","args":{}}', "fly me to the moon", REG)[0] is None
    print(f"\n{len(CASES) - fails}/{len(CASES)} cases fixed correctly")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
