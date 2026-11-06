"""HR analysis agent.

Two interchangeable backends behind ``HRAgent.ask``:

* ``llm``  - real LLM function calling via the Anthropic Messages API (needs ``pip install anthropic``
  and ``ANTHROPIC_API_KEY``). The model must call ``analyze_hr_segment``; its answer is guard-railed.
* ``rule`` - deterministic offline backend: parses the request with regexes, calls the same tool, and
  explains the result from templates. Works with no network or API key.

``backend="auto"`` picks ``llm`` when both the SDK and an API key are present, otherwise ``rule``.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from passos_wearable.agent.prompts import (DISCLAIMER, FORBIDDEN_CLAIMS, MEDICAL_PATTERNS, MEDICAL_REFUSAL,
                                           SYSTEM_PROMPT, explain_followup, explain_result)
from passos_wearable.config import load_config
from passos_wearable.estimation.heart_rate import METHODS
from passos_wearable.tools.analysis_tool import TOOL_SCHEMA, analyze_hr_segment

log = logging.getLogger(__name__)


@dataclass
class AgentReply:
    text: str
    backend: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    result: dict | None = None
    refused: bool = False


_RANGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-|to|–)\s*(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?\b", re.I)


def _to_seconds(value: float, unit: str | None) -> float:
    return value * 60.0 if unit and unit.lower().startswith("m") and not unit.lower().startswith("ms") else value


def parse_request(text: str, default_method: str = "fusion") -> dict | None:
    """Extract segment parameters from free text. Returns None if no segment is specified."""
    start = dur = None
    m = _RANGE_RE.search(text)
    if m and float(m.group(2)) > float(m.group(1)):
        start, dur = float(m.group(1)), float(m.group(2)) - float(m.group(1))
    else:
        ms = re.search(r"(?:from|at|start(?:ing)?(?:\s+at)?|beginning(?:\s+at)?)\s*(\d+(?:\.\d+)?)\s*(min\w*|m\b|s\w*)?", text, re.I)
        md = re.search(r"(?:for|duration(?:\s+of)?|lasting|over)\s*(\d+(?:\.\d+)?)\s*(min\w*|m\b|s\w*)?", text, re.I)
        if ms:
            start = _to_seconds(float(ms.group(1)), ms.group(2))
        if md:
            dur = _to_seconds(float(md.group(1)), md.group(2))
        if start is None and dur is not None and re.search(r"first|beginning", text, re.I):
            start = 0.0
    if start is None or dur is None:
        return None
    method = next((k for k in sorted(METHODS, key=len, reverse=True) if k in text.lower().replace(" ", "_")), default_method)
    return {"subject": "S1", "start_s": start, "duration_s": dur, "method": method,
            "include_reference": not re.search(r"no reference|without reference|blind", text, re.I)}


class HRAgent:
    def __init__(self, config: dict | None = None, backend: str | None = None, data_dir: str | None = None,
                 synthetic: bool = False, client: Any = None, tool_fn: Callable[..., dict] | None = None,
                 recording=None):
        self.cfg = config or load_config()
        a = self.cfg["agent"]
        self.data_dir, self.synthetic, self.recording = data_dir, synthetic, recording
        self._tool = tool_fn or analyze_hr_segment
        self.client = client
        want = backend or a["backend"]
        if want == "auto":
            want = "llm" if (client is not None or self._llm_available()) else "rule"
        if want == "llm" and self.client is None:
            self.client = self._make_client()
        self.backend = want
        self.last_result: dict | None = None
        self.history: list[dict] = []

    # ------------------------------------------------------------------ backends
    @staticmethod
    def _llm_available() -> bool:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return False
        try:
            import anthropic  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _make_client():
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("LLM backend needs `pip install anthropic`") from e
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set (see .env.example)")
        return anthropic.Anthropic()

    def run_tool(self, **kwargs) -> dict:
        kwargs = {k: v for k, v in kwargs.items() if k in ("subject", "start_s", "duration_s", "method", "include_reference")}
        res = self._tool(data_dir=self.data_dir, synthetic=self.synthetic, config=self.cfg,
                         recording=self.recording, **kwargs)
        self.last_result = res
        return res

    # ------------------------------------------------------------------ public API
    def ask(self, message: str) -> AgentReply:
        message = (message or "").strip()
        if not message:
            return AgentReply("Please enter a question, e.g. 'analyse S1 from 120 s for 30 s'.", self.backend)
        if len(message) > 2000:
            return AgentReply("Message too long (max 2000 characters).", self.backend)
        if MEDICAL_PATTERNS.search(message):
            return AgentReply(MEDICAL_REFUSAL, self.backend, refused=True)
        reply = self._ask_llm(message) if self.backend == "llm" else self._ask_rule(message)
        self.history.append({"user": message, "assistant": reply.text})
        return reply

    # ------------------------------------------------------------------ rule backend
    def _ask_rule(self, message: str) -> AgentReply:
        params = parse_request(message, self.cfg["estimation"]["default_method"])
        if params:
            res = self.run_tool(**params)
            return AgentReply(explain_result(res), "rule", [{"name": TOOL_SCHEMA["name"], "input": params}], res)
        if self.last_result is not None:
            return AgentReply(explain_followup(self.last_result, message), "rule", [], self.last_result)
        return AgentReply("Tell me which segment to analyse, e.g. 'analyse S1 from 120 s for 30 s with fusion'. "
                          f"Methods: {', '.join(METHODS)}.", "rule")

    # ------------------------------------------------------------------ llm backend
    def _ask_llm(self, message: str) -> AgentReply:
        a = self.cfg["agent"]
        tools = [TOOL_SCHEMA]
        msgs = [{"role": "user", "content": message}]
        calls, last = [], None
        if self.history:  # lightweight memory of previous turns
            msgs = [m for h in self.history[-4:] for m in ({"role": "user", "content": h["user"]},
                                                           {"role": "assistant", "content": h["assistant"]})] + msgs
        text = ""
        for _ in range(a["max_tool_rounds"] + 1):
            resp = self.client.messages.create(model=a["model"], max_tokens=a["max_tokens"],
                                               system=SYSTEM_PROMPT, tools=tools, messages=msgs)
            blocks = list(resp.content)
            text = "".join(getattr(b, "text", "") for b in blocks if getattr(b, "type", "") == "text")
            uses = [b for b in blocks if getattr(b, "type", "") == "tool_use"]
            if not uses:
                break
            msgs.append({"role": "assistant", "content": blocks})
            results = []
            for u in uses:
                args = dict(u.input or {})
                calls.append({"name": u.name, "input": args})
                if u.name != TOOL_SCHEMA["name"]:
                    out = {"status": "error", "error_message": f"unknown tool {u.name}"}
                else:
                    out = self.run_tool(**args)
                    last = out
                results.append({"type": "tool_result", "tool_use_id": u.id, "content": json.dumps(out)})
            msgs.append({"role": "user", "content": results})
        else:
            text = text or "I could not finish the analysis within the allowed number of tool calls."
        return AgentReply(self._guard(text, last or self.last_result), "llm", calls, last or self.last_result)

    @staticmethod
    def _guard(text: str, result: dict | None) -> str:
        """Post-hoc safety net: flag diagnostic wording and bpm figures not found in the tool result."""
        notes = []
        if FORBIDDEN_CLAIMS.search(text):
            notes.append("diagnostic-sounding wording was detected; this tool cannot diagnose anything")
        if result is not None:
            allowed = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", json.dumps(result))]
            for fig in re.findall(r"(\d+(?:\.\d+)?)\s*bpm", text, re.I):
                if not any(abs(float(fig) - v) <= 0.06 for v in allowed):
                    notes.append(f"figure '{fig} bpm' does not appear in the tool result")
        elif re.search(r"\d\s*bpm", text, re.I):
            notes.append("HR figures were given without a tool result")
        if notes:
            text += "\n[Guardrail note: " + "; ".join(notes) + ". Rely on the structured tool result.]"
        if DISCLAIMER not in text:
            text += "\n" + DISCLAIMER
        return text
