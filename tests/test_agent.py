from types import SimpleNamespace as NS

from passos_wearable.agent.agent import HRAgent, parse_request


def test_parse_request_variants():
    assert parse_request("analyse S1 from 120 s for 30 s") == {"subject": "S1", "start_s": 120.0, "duration_s": 30.0, "method": "fusion", "include_reference": True}
    p = parse_request("start at 2 minutes for 45 seconds using spectral motion")
    assert p["start_s"] == 120 and p["duration_s"] == 45 and p["method"] == "spectral_motion"
    assert parse_request("what about 100-130 s with peaks")["duration_s"] == 30
    assert parse_request("hello") is None


def test_rule_agent_flow(rec, cfg):
    a = HRAgent(cfg, backend="rule", recording=rec)
    r = a.ask("analyse S1 from 100 s for 24 s")
    assert r.result["status"] == "ok" and "bpm" in r.text and "SYNTHETIC" in r.text and "not a medical device" in r.text
    follow = a.ask("why this confidence?")
    assert "Confidence is" in follow.text and follow.tool_calls == []
    assert "No reference" not in a.ask("how accurate is it compared to the reference?").text


def test_rule_agent_needs_segment_first(rec, cfg):
    assert "Tell me which segment" in HRAgent(cfg, backend="rule", recording=rec).ask("what is the heart rate?").text


def test_medical_questions_refused_without_tool_call(rec, cfg):
    a = HRAgent(cfg, backend="rule", recording=rec)
    for q in ("Do I have arrhythmia?", "is my heart rate normal?", "analyse from 10 s for 20 s and diagnose me"):
        r = a.ask(q)
        assert r.refused and r.result is None and "can't interpret" in r.text
    assert a.last_result is None


def test_error_is_relayed(rec, cfg):
    r = HRAgent(cfg, backend="rule", recording=rec).ask("analyse from 5000 s for 30 s")
    assert "couldn't run" in r.text


def test_input_limits(rec, cfg):
    a = HRAgent(cfg, backend="rule", recording=rec)
    assert "enter a question" in a.ask("  ").text and "too long" in a.ask("x" * 3000).text


class FakeClient:
    """Mimics the Anthropic Messages API: first turn requests the tool, second turn writes the answer."""
    def __init__(self, final_text):
        self.calls, self.final_text = [], final_text
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        if len(self.calls) == 1:
            return NS(content=[NS(type="tool_use", id="t1", name="analyze_hr_segment",
                                  input={"start_s": 100, "duration_s": 24, "method": "spectral"})])
        return NS(content=[NS(type="text", text=self.final_text)])


def test_llm_function_calling_loop_with_mock_client(rec, cfg):
    fc = FakeClient("HR is 999.0 bpm. This indicates tachycardia.")
    a = HRAgent(cfg, backend="llm", client=fc, recording=rec)
    r = a.ask("analyse S1 from 100 s for 24 s")
    assert r.backend == "llm" and r.tool_calls[0]["name"] == "analyze_hr_segment" and r.result["status"] == "ok"
    assert fc.calls[0]["tools"][0]["name"] == "analyze_hr_segment" and "NOT a medical device" in fc.calls[0]["system"]
    tool_msg = fc.calls[1]["messages"][-1]["content"][0]
    assert tool_msg["type"] == "tool_result" and '"status": "ok"' in tool_msg["content"]
    assert "Guardrail note" in r.text and "999.0 bpm" in r.text and "diagnostic-sounding" in r.text


def test_llm_grounded_answer_passes_guard(rec, cfg):
    from passos_wearable.tools.analysis_tool import analyze_hr_segment
    hr = analyze_hr_segment(start_s=100, duration_s=24, method="spectral", recording=rec, config=cfg)["estimate"]["hr_bpm"]
    r = HRAgent(cfg, backend="llm", client=FakeClient(f"Estimated {hr} bpm."), recording=rec).ask("analyse from 100 s for 24 s")
    assert "Guardrail note" not in r.text


def test_auto_backend_without_key_uses_rule(monkeypatch, cfg):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert HRAgent(cfg, backend="auto").backend == "rule"
