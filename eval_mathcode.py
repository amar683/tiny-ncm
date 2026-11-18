"""100 basic math and coding requests through the whole chat app (dry run). The model was NOT trained on these;
this measures what happens anyway.

python eval_mathcode.py [--ckpt runs/task6/best.pt]   -> runs/eval_mathcode.md

Math (60):  correct if the expected number appears in the final reply.
Coding (40):
  write   - a function is pulled from the reply and run against small tests (only if it uses no imports/files/system calls)
  explain - correct if the reply contains one of the key words
  output  - correct if the expected output appears in the reply
"""
import argparse
import ast
import json
import re
import subprocess
import sys
import time

from eval_chat import NOW, FakeGmail, TestTools
from assistant.runtime import Assistant
from ncm.bpe import BPE
from ncm.chat_format import load_registry
from ncm.utils import load_generator, pick_device

MATH = [  # (category, question, expected number)
    # arithmetic in words
    ("words", "what is 17 times 23?", 391), ("words", "what's 456 plus 789", 1245), ("words", "what is 1000 minus 1?", 999),
    ("words", "what is 81 divided by 9?", 9), ("words", "how much is 6 times 7", 42), ("words", "what is 250 times 4?", 1000),
    ("words", "calculate 99 plus 1", 100), ("words", "what is 64 divided by 8", 8), ("words", "what's 13 times 13", 169),
    ("words", "what is 500 minus 275?", 225), ("words", "what is 2 plus 2?", 4), ("words", "what's 9 times 9", 81),
    ("words", "what is 3 multiplied by 14?", 42), ("words", "what is 120 divided by 5?", 24), ("words", "add 38 and 47", 85),
    ("words", "subtract 19 from 50", 31), ("words", "what is 7 times 8?", 56), ("words", "what is 1500 plus 2500", 4000),
    ("words", "what is 11 times 12?", 132), ("words", "what is 0.5 times 30?", 15),
    # symbols
    ("symbols", "what is 45+67?", 112), ("symbols", "12*12", 144), ("symbols", "what's 2^10", 1024), ("symbols", "100/8", 12.5),
    ("symbols", "what is 15-28?", -13), ("symbols", "calculate (3+4)*5", 35), ("symbols", "what is 7*6-2?", 40),
    ("symbols", "what is 3.2+4.8", 8), ("symbols", "1234+4321", 5555), ("symbols", "what is 10*10*10", 1000),
    # percentages
    ("percent", "what's 25 percent of 200", 50), ("percent", "what is 10% of 450?", 45), ("percent", "what's 50 percent of 36", 18),
    ("percent", "what is 15 percent of 60?", 9), ("percent", "what's 75% of 80", 60), ("percent", "what is 5 percent of 1000", 50),
    ("percent", "what is 20% of 15?", 3), ("percent", "what's 1 percent of 700", 7),
    # word problems
    ("story", "I have 12 apples and eat 5. How many are left?", 7),
    ("story", "a pizza has 8 slices and 3 friends eat 2 each, how many slices are left?", 2),
    ("story", "if a book costs 15 dollars, how much do 4 books cost?", 60),
    ("story", "I walk 3 km every day. How far do I walk in a week?", 21),
    ("story", "there are 24 students and 6 groups, how many students per group?", 4),
    ("story", "I had 50 dollars and spent 18, how much is left?", 32),
    ("story", "a train goes 60 km per hour for 3 hours, how far does it go?", 180),
    ("story", "Tom is 9 and his sister is 4 years older. How old is she?", 13),
    ("story", "a box has 6 eggs. How many eggs in 5 boxes?", 30),
    ("story", "I sleep 8 hours a night. How many hours in 7 nights?", 56),
    ("story", "there are 100 people and 37 leave, how many stay?", 63),
    ("story", "a shirt costs 40 dollars with 25 percent off, what is the new price?", 30),
    # other
    ("other", "what is the square root of 144?", 12), ("other", "what is 5 squared?", 25), ("other", "what is half of 90?", 45),
    ("other", "what is 1/4 as a decimal?", 0.25), ("other", "what is the average of 4, 8 and 12?", 8),
    ("other", "how many minutes are in 3 hours?", 180), ("other", "how many days are in 2 weeks?", 14),
    ("other", "what is double 35?", 70), ("other", "what is 3 cubed?", 27), ("other", "how many seconds in 5 minutes?", 300),
]

# written BEFORE the new calculator phrasings, never used to change them (the honest check)
MATH_FRESH = [
    ("words", "what is 14 times 6?", 84), ("words", "add 120 and 75", 195), ("words", "subtract 8 from 30", 22),
    ("words", "what is 72 divided by 6?", 12), ("words", "what's 33 plus 67", 100),
    ("symbols", "what is (10-4)*3?", 18), ("symbols", "5*5+5", 30), ("symbols", "what's 3^4", 81),
    ("percent", "what's 30 percent of 90", 27), ("percent", "what is 12% of 50?", 6),
    ("other", "what is the square root of 81?", 9), ("other", "what is half of 64?", 32), ("other", "what is 6 squared?", 36),
    ("other", "what is double 120?", 240), ("other", "what is 4 cubed?", 64), ("other", "what is the average of 10, 20 and 30?", 20),
    ("other", "how many minutes are in 2 hours?", 120), ("other", "how many hours are in 3 days?", 72),
    ("other", "how many seconds in 2 minutes?", 120), ("other", "convert 3 km to meters", 3000),
    ("other", "what is 3/4 as a decimal?", 0.75), ("other", "what is triple 15?", 45),
    ("story", "I had 20 dollars and spent 7, how much do I have left?", 13),
    ("story", "a pen costs 3 dollars, how much do 9 pens cost?", 27),
    ("story", "there are 30 kids and 5 teams, how many kids per team?", 6),
    ("story", "Mia is 12 and her brother is 3 years older. How old is he?", 15),
    ("story", "I read 20 pages a day. How many pages in 5 days?", 100),
    ("story", "a jacket costs 80 dollars with 50 percent off, what is the new price?", 40),
    ("story", "I have 9 cookies and get 4 more. How many do I have?", 13),
    ("story", "a car drives 50 km per hour for 4 hours, how far does it go?", 200),
]

CODE_WRITE = [  # (question, function name, tests as (args, expected))
    ("write a python function add(a, b) that returns the sum", "add", [((2, 3), 5), ((-1, 1), 0)]),
    ("write a python function reverse(s) that reverses a string", "reverse", [(("abc",), "cba"), (("",), "")]),
    ("write a python function is_even(n) that returns True if n is even", "is_even", [((4,), True), ((7,), False)]),
    ("write a python function factorial(n)", "factorial", [((5,), 120), ((0,), 1)]),
    ("write a python function biggest(xs) that returns the largest number in a list", "biggest", [(([3, 9, 2],), 9)]),
    ("write a python function count_vowels(s)", "count_vowels", [(("hello",), 2), (("sky",), 0)]),
    ("write a python function square(x) that returns x squared", "square", [((4,), 16), ((-3,), 9)]),
    ("write a python function c_to_f(c) that converts celsius to fahrenheit", "c_to_f", [((0,), 32), ((100,), 212)]),
    ("write a python function total(xs) that sums a list", "total", [(([1, 2, 3],), 6), (([],), 0)]),
    ("write a python function is_palindrome(s)", "is_palindrome", [(("level",), True), (("hello",), False)]),
    ("write a python function first(xs) that returns the first item of a list", "first", [(([7, 8],), 7)]),
    ("write a python function double(x) that returns twice x", "double", [((21,), 42)]),
    ("write a python function greet(name) that returns 'Hello, ' plus the name", "greet", [(("Ana",), "Hello, Ana")]),
    ("write a python function maximum(a, b) that returns the bigger number", "maximum", [((3, 8), 8), ((9, 2), 9)]),
    ("write a python function length(s) that returns the length of a string", "length", [(("four",), 4)]),
    ("write a python function average(xs) that returns the mean of a list", "average", [(([2, 4, 6],), 4)]),
    ("write a python function fizzbuzz(n) that returns 'Fizz', 'Buzz', 'FizzBuzz' or the number as a string", "fizzbuzz",
     [((3,), "Fizz"), ((5,), "Buzz"), ((15,), "FizzBuzz"), ((7,), "7")]),
    ("write a python function upper(s) that makes a string uppercase", "upper", [(("hi",), "HI")]),
    ("write a python function last(xs) that returns the last item of a list", "last", [(([1, 2, 3],), 3)]),
    ("write a python function is_positive(n)", "is_positive", [((5,), True), ((-2,), False)]),
]
CODE_EXPLAIN = [  # (question, key words: any one counts)
    ("what is a variable in python?", ["store", "value", "name"]), ("what does len do in python?", ["length", "number of items", "count"]),
    ("what is a for loop?", ["repeat", "each", "iterat"]), ("what is a function in programming?", ["reusable", "block", "call", "input"]),
    ("what is a list in python?", ["order", "collection", "items", "sequence"]), ("what does print do in python?", ["output", "display", "screen", "show"]),
    ("what is an if statement?", ["condition", "true", "decide"]), ("what is a bug in code?", ["error", "mistake", "wrong", "fault"]),
    ("what is python?", ["programming language"]), ("what is a string in programming?", ["text", "characters", "sequence"]),
]
CODE_OUTPUT = [  # (question, expected output text)
    ("what does print(2+3) output?", "5"), ("what is the output of len('hello')?", "5"), ("what does 'ab' * 3 give in python?", "ababab"),
    ("what is 10 // 3 in python?", "3"), ("what is 10 % 3 in python?", "1"), ("what does print('hi'.upper()) show?", "HI"),
    ("in python what is 2 ** 5?", "32"), ("what is the result of max(4, 9, 2) in python?", "9"), ("what does sorted([3,1,2]) return?", "[1, 2, 3]"),
    ("what is bool(0) in python?", "False"),
]

BANNED = re.compile(r"\b(?:import|open|exec|eval|compile|__\w+__|globals|locals|getattr|setattr|input|os|sys|subprocess)\b")


def nums(text):
    return [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)]


def run_function(reply, fname, tests):
    """Pull 'def fname' out of the reply and run the tests in a separate, isolated Python with a time limit."""
    m = re.search(rf"def\s+{fname}\s*\(.*", reply, re.S)
    if not m:
        return False, "no function"
    code = m.group().split("```")[0]
    lines = code.splitlines()
    keep = [lines[0]] + [ln for ln in lines[1:] if ln.startswith((" ", "\t")) or not ln.strip()]
    code = "\n".join(keep[: 1 + next((i for i, ln in enumerate(keep[1:]) if not ln.strip() and i > 0), len(keep))])
    try:
        ast.parse(code)
    except SyntaxError:
        return False, "not valid Python"
    if BANNED.search(code):
        return False, "uses blocked names (not run)"
    prog = code + "\nimport json\nprint(json.dumps([" + ", ".join(f"{fname}(*{a!r}) == {e!r}" for a, e in tests) + "]))"
    try:
        p = subprocess.run([sys.executable, "-I", "-c", prog], capture_output=True, text=True, timeout=5)
        ok = p.returncode == 0 and all(json.loads(p.stdout.strip().splitlines()[-1]))
        return ok, "passed" if ok else (p.stderr.strip().splitlines() or ["wrong answer"])[-1][:80]
    except Exception as e:
        return False, type(e).__name__


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/task6/best.pt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", choices=["main", "fresh"], default="main")
    a = ap.parse_args()
    gen, _ = load_generator(a.ckpt, pick_device("auto"))
    bpe = BPE.load("tokenizer/bpe.json")
    tools = TestTools(now=lambda: NOW, dry_run=True, gmail=FakeGmail())
    tools.offline = False
    bot = Assistant(gen, bpe, tools, load_registry(), confirm=lambda c: False, now=lambda: NOW)

    a.out = a.out or ("runs/eval_mathcode.md" if a.set == "main" else "runs/eval_math_fresh.md")
    if a.set == "fresh":
        cases = [("math", c, q, x) for c, q, x in MATH_FRESH]
    else:
        cases = ([("math", c, q, x) for c, q, x in MATH] + [("code", "write", q, (f, t)) for q, f, t in CODE_WRITE] +
                 [("code", "explain", q, k) for q, k in CODE_EXPLAIN] + [("code", "output", q, o) for q, o in CODE_OUTPUT])
        assert len(cases) == 100, len(cases)
    rows, t0 = [], time.time()
    for i, (area, cat, q, want) in enumerate(cases):
        ev = {"tool": [], "model_reply": None}
        bot.reset()
        reply = bot.ask(q, on_event=lambda k, p: ev["tool"].append(p[0]) if k == "tool" else ev.update(model_reply=p) if k == "model_reply" else None)
        tool = ev["tool"][0]["name"] if ev["tool"] and ev["tool"][0] else None
        args = ev["tool"][0]["args"] if ev["tool"] and ev["tool"][0] else None
        if area == "math":
            ok, why = any(abs(n - want) < 1e-6 for n in nums(reply)), ""
        elif cat == "write":
            ok, why = run_function(reply, *want)
        elif cat == "explain":
            ok, why = any(k in reply.lower() for k in want), ""
        else:
            ok, why = want.lower() in reply.lower(), ""
        rows.append(dict(area=area, cat=cat, q=q, want=want if cat != "write" else want[0], ok=ok, why=why, tool=tool, args=args,
                         reply=reply, own=ev["model_reply"] is not None and reply == ev["model_reply"]))
        print(f"{i + 1:3d} {'ok ' if ok else 'BAD'} [{area}/{cat}] {q}\n      tool: {tool} {json.dumps(args) if args else ''}\n      reply: {reply[:150]!r} {why}", flush=True)

    n_ok = lambda rs: sum(r["ok"] for r in rs)
    L = [f"# {len(rows)} basic math{'' if a.set == 'fresh' else ' and coding'} requests ({a.set} set): {a.ckpt}", "",
         "The model was never trained on math or coding. Dry run, through the whole app (model, checks, tools).", f"{(time.time() - t0) / 60:.1f} min.", "",
         "| Area | Kind | n | correct | used a tool |", "|---|---|---|---|---|"]
    for area, cat in dict.fromkeys((r["area"], r["cat"]) for r in rows):
        rs = [r for r in rows if r["area"] == area and r["cat"] == cat]
        tools_used = {}
        for r in rs:
            tools_used[r["tool"] or "none"] = tools_used.get(r["tool"] or "none", 0) + 1
        L.append(f"| {area} | {cat} | {len(rs)} | **{n_ok(rs)}** | {', '.join(f'{k} {v}' for k, v in sorted(tools_used.items(), key=lambda x: -x[1]))} |")
    m, c = [r for r in rows if r["area"] == "math"], [r for r in rows if r["area"] == "code"]
    L += [f"| **math** | all | {len(m)} | **{n_ok(m)}** | |", f"| **code** | all | {len(c)} | **{n_ok(c)}** | |", "", "## Every request", ""]
    for r in rows:
        L.append(f"- {'✅' if r['ok'] else '❌'} [{r['cat']}] **{r['q']}** (want `{r['want']}`) → tool `{r['tool']}` "
                 f"`{json.dumps(r['args']) if r['args'] else ''}` → {r['reply'][:160]!r} {r['why']}")
    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L[:20]))


if __name__ == "__main__":
    main()
