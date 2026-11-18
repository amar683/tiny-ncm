"""The agent loop: model <-> tools.

Each turn:
  [system: date/time + tools] [history...] [user]  -> the model writes the next segment.
  It starts with <|tool|>     -> check the call against your words (grounding.py), ask before
                                 confirm-tools, run it, add the call + <|result|>, and let the model continue.
  It starts with <|assistant|> -> that's the answer; the turn ends.
At most 5 tool calls per turn. Old turns are dropped when the conversation nears 4096 tokens.
"""
import datetime as dt
import json
import re

import torch

from ncm.chat_format import prompt_tokens, system_prompt, tool_call_text
from ncm.sampling import generate

from .grounding import (ASK_FOR, CAN_DO, _place, chat_reply, check_answer, check_call, check_chat, plain_reply_ok, question_like, route,
                        template_reply, tidy)
from .normalize import normalize

MAX_TOOL_CALLS = 5
ANSWER_BLOCKS = 12           # up to 384 tokens per segment


def split_requests(text):
    """'set an alarm for 7 and remind me to call mom at 9' -> two requests, but only if each half is clearly a request
    on its own ('remind me to buy eggs and milk' stays one)."""
    parts = [p.strip(" ,.") for p in re.split(r"\s*,?\s+(?:and then|and also|and|then|also)\s+", text) if p.strip(" ,.")]
    for i in range(1, len(parts)):          # "what time is it there" -> "... in Rome"
        if re.search(r"\bthere\b", parts[i]) and not _place(parts[i]):
            place = _place(parts[i - 1])
            if place:
                parts[i] = re.sub(r"\bthere\b", f"in {place}", parts[i])
    if len(parts) < 2 or any(route(p) in (None, "chat") for p in parts):
        return [text]
    return parts


class Assistant:
    def __init__(self, gen, bpe, tools, registry, platform="mac", confirm=None, temperature=0.0, now=None, keep_turns=0):
        self.gen, self.bpe, self.tools = gen, bpe, tools
        self.registry = [t for t in registry if platform in t["platforms"]]
        self.confirm = confirm or (lambda call: False)
        self.temperature = temperature
        self.now = now or dt.datetime.now
        self.history = []          # the current turn
        self.past = []             # earlier turns, as lists of messages
        self.keep_turns = keep_turns   # earlier turns shown to the model (0: it was trained on single requests)
        self.block = gen.cfg.block

    def reset(self):
        self.history, self.past = [], []

    def _prompt(self, msgs):
        sys = {"role": "system", "content": system_prompt(self.now(), self.registry)}
        room = self.gen.cfg.max_ctx - ANSWER_BLOCKS * self.block
        while True:
            toks = prompt_tokens(self.bpe, [sys] + msgs, self.block)
            if len(toks) <= room or len(msgs) <= 1:
                return toks[-room:]
            msgs = msgs[1:]            # drop the oldest message

    def _next_segment(self, msgs, on_level=None):
        toks = self._prompt(msgs)
        with torch.no_grad():
            out = generate(self.gen, toks, max_blocks=ANSWER_BLOCKS, temperature=self.temperature,
                           stop_ids=(self.bpe.end,), pad_id=self.bpe.pad, on_level=on_level)
        return self.bpe.decode(out, skip_special=False).replace("<|pad|>", "").replace("<|end|>", "").strip()

    def ask(self, user_text, on_event=None, on_level=None):
        """Runs one turn. on_event(kind, payload) gets: ("tool", (call, notes)), ("result", dict), ("answer", text), ("note", str).
        Your text is cleaned up first (typos, number words, spoken times); two clear requests joined by "and" are done one by one."""
        emit = on_event or (lambda *a: None)
        text = normalize(user_text, self.now())
        if text != user_text:
            emit("note", f"read as: {text!r}")
        parts = split_requests(text)
        if len(parts) > 1:
            emit("note", f"two requests: {parts}")
            replies = [self._ask_one(p, lambda k, v: emit(k, v) if k != "answer" else None, on_level) for p in parts]
            emit("answer", "\n".join(replies))
            return "\n".join(replies)
        return self._ask_one(text, emit, on_level)

    def _ask_one(self, user_text, emit, on_level=None):
        if not user_text.strip():
            return self._finish("What can I do for you?", emit)
        self.history = [{"role": "user", "content": user_text}]
        earlier = [m for turn in self.past[-self.keep_turns:] for m in turn] if self.keep_turns else []
        last = None                                  # (name, args, result) of the latest tool call
        for _ in range(MAX_TOOL_CALLS + 1):
            seg = self._next_segment(earlier + self.history, on_level)
            want = route(user_text) if last is None else None       # only for the first step of a turn
            if want == "chat" and seg.startswith("<|tool|>"):
                emit("raw", seg)
                emit("note", "small talk: ignored the tool call, no tool needed")
                return self._finish(chat_reply(user_text), emit)
            if last is not None and seg.startswith("<|tool|>") and not re.search(r"\b(?:and|then|also)\b", user_text, re.I):
                emit("raw", seg)
                emit("note", "ignored a second tool call (you asked for one thing)")
                return self._finish(template_reply(*last), emit)
            if want not in (None, "chat") and not seg.startswith('<|tool|>{"name":"%s"' % want):
                emit("raw", seg)
                emit("note", f"tool: your words ask for {want}")
                seg = '<|tool|>{"name":"%s","args":{}}' % want   # arguments come from your words
                emit("raw_routed", seg)
            elif not seg.startswith("<|tool|>"):
                text = seg.replace("<|assistant|>", "", 1).strip() or "Sorry, I didn't get that."
                emit("model_reply", text)
                note = None
                if last:
                    text, note = check_answer(text, *last, user_text)
                elif want == "chat":
                    text, note = check_chat(text, user_text)
                elif not plain_reply_ok(text, user_text):
                    if len(re.findall(r"\d+", user_text)) >= 2:
                        text, note = ("I can't work that one out yet. Try one step at a time, like \"3 times 12\".",
                                      "reply: garbled answer to a math question")
                    elif question_like(user_text):
                        emit("note", "reply: garbled answer; searching the web instead")
                        seg = '<|tool|>{"name":"web_search","args":{"query":%s}}' % json.dumps(tidy(user_text))
                        want = "web_search"
                        text = None
                    else:
                        text, note = "Sorry, I didn't get that. Could you say it another way?", "reply: garbled answer"
                if text is not None:
                    if note:
                        emit("note", note)
                    return self._finish(text, emit)
            if not seg.startswith('<|tool|>{"name":"%s","args":{}}' % want):
                emit("raw", seg)
            call, notes = check_call(seg, user_text, self.registry)
            if call is not None and last is None and want != call["name"] and (call["unsure"] or call["missing"]):
                # the call doesn't match your words: the model is guessing
                emit("raw", seg) if want is None else None
                if question_like(user_text):
                    emit("note", f"not sure {call['name']} fits; searching the web instead")
                    call, notes = check_call('{"name":"web_search","args":{"query":%s}}' % json.dumps(tidy(user_text)), user_text, self.registry)
                else:
                    emit("note", f"not sure {call['name']} fits your words; not running it")
                    return self._finish(CAN_DO, emit)
            if call is not None and call["missing"] and want == call["name"]:
                arg = call["missing"][0]
                emit("note", f"{call['name']}: missing {arg}")
                return self._finish(ASK_FOR.get((call["name"], arg), f"What {arg} should I use?"), emit)
            emit("tool", (call, notes))
            if call is None:
                result = {"ok": False, "error": "unknown tool"}
                self.history.append({"role": "tool", "content": seg.replace("<|tool|>", "", 1)})
            else:
                self.history.append({"role": "tool", "content": tool_call_text(call["name"], call["args"])})
                if call.get("missing"):            # never run a tool without its required details
                    result = {"ok": False, "error": f"I need more details ({', '.join(call['missing'])}) to use {call['name']}"}
                elif call["confirm"] and not self.confirm(call):
                    result = {"ok": False, "error": "cancelled by the user"}
                else:
                    result = self.tools.run(call["name"], call["args"])
                last = (call["name"], call["args"], result)
            emit("result", result)
            self.history.append({"role": "result", "content": json.dumps(result, ensure_ascii=False, separators=(",", ":"))})
        return self._finish("I tried several tools but couldn't finish that.", emit)

    def _finish(self, text, emit):
        self.history.append({"role": "assistant", "content": text})
        self.past.append(self.history)
        self.past = self.past[-20:]
        emit("answer", text)
        return text
