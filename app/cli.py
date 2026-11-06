#!/usr/bin/env python3
"""Command-line interface: analyse segments, chat with the agent, explore and evaluate S1.

Examples (from the repo root):
  python app/cli.py info
  python app/cli.py analyze --start 120 --duration 30 --method fusion
  python app/cli.py chat
  python app/cli.py ask "analyse S1 from 300 s for 40 s"
  python app/cli.py explore --out outputs/real
  python app/cli.py evaluate --out outputs/real
Add --synthetic to any command to use generated TEST data (never real results).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from passos_wearable.agent.agent import HRAgent  # noqa: E402
from passos_wearable.agent.prompts import explain_result  # noqa: E402
from passos_wearable.config import load_config, load_dotenv  # noqa: E402
from passos_wearable.data.loader import (DatasetNotFoundError, DataError, describe_recording,  # noqa: E402
                                         load_subject)
from passos_wearable.data.synthetic import make_synthetic_recording  # noqa: E402
from passos_wearable.estimation.heart_rate import METHODS  # noqa: E402
from passos_wearable.tools.analysis_tool import analyze_hr_segment  # noqa: E402


def _recording(args, cfg):
    if args.synthetic:
        print("*** SYNTHETIC test data - not PPG-DaLiA ***", file=sys.stderr)
        return make_synthetic_recording(config=cfg, duration_s=args.synthetic_seconds)
    return load_subject(args.subject, args.data_dir, cfg)


def cmd_info(args, cfg):
    rec = _recording(args, cfg)
    print(json.dumps(describe_recording(rec), indent=2))


def cmd_analyze(args, cfg):
    rec = _recording(args, cfg)
    res = analyze_hr_segment(args.subject, args.start, args.duration, args.method,
                             not args.no_reference, config=cfg, recording=rec)
    print(json.dumps(res, indent=2) if args.json else explain_result(res))
    return 0 if res["status"] == "ok" else 2


def cmd_ask(args, cfg):
    rec = _recording(args, cfg)
    agent = HRAgent(cfg, backend=args.backend, recording=rec)
    r = agent.ask(args.question)
    print(f"[backend: {r.backend}]\n{r.text}")


def cmd_chat(args, cfg):
    rec = _recording(args, cfg)
    agent = HRAgent(cfg, backend=args.backend, recording=rec)
    print(f"HR analysis assistant (backend: {agent.backend}; data: {rec.source_tag}). Ctrl-D or 'quit' to exit.")

    def prompt(label, default):
        v = input(f"{label} [{default}]: ").strip()
        return v or str(default)

    try:
        while True:
            subject = prompt("Participant", "S1")
            try:
                start = float(prompt("Start time (s)", 0))
                dur = float(prompt("Duration (s)", 30))
            except ValueError:
                print("Please enter numbers."); continue
            method = prompt(f"Method ({'/'.join(METHODS)})", cfg["estimation"]["default_method"])
            res = agent.run_tool(subject=subject, start_s=start, duration_s=dur, method=method)
            print("\n" + explain_result(res) + "\n")
            while True:
                q = input("Ask the agent about this result ('new' = new segment, 'quit' = exit): ").strip()
                if q.lower() in ("quit", "exit", "q"):
                    return
                if q.lower() in ("new", "n"):
                    break
                if q:
                    print("\n" + agent.ask(q).text + "\n")
    except (EOFError, KeyboardInterrupt):
        print()


def cmd_explore(args, cfg):
    from passos_wearable.evaluation.experiments import attach_reference, run_estimation
    from passos_wearable.visualization import plots as P
    rec = _recording(args, cfg)
    out = Path(args.out)
    print(json.dumps(describe_recording(rec), indent=2))
    P.plot_overview(rec, out / "overview.png")
    seg_t = min(args.start, max(0, rec.signals.duration_s - 30))
    P.plot_segment(rec, seg_t, 30, cfg, out / "segment_example.png")
    df = attach_reference(run_estimation(rec, cfg, max_windows=args.max_windows), rec, cfg)
    P.plot_activity_comparison(df, rec, out_path=out / "activity_comparison.png")
    print(f"Plots written to {out}")


def cmd_evaluate(args, cfg):
    from passos_wearable.evaluation import experiments as X
    from passos_wearable.visualization import plots as P
    rec = _recording(args, cfg)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = X.attach_reference(X.run_estimation(rec, cfg, max_windows=args.max_windows), rec, cfg)
    tol = cfg["evaluation"]["within_tolerance_bpm"]
    overall = X.summarize(df, tol=tol)
    by_act = X.summarize_by(df, "activity", tol=tol)
    by_mot = X.summarize_by(df, "motion_label", tol=tol)
    gate = X.sqi_gating(df, tol=tol)
    worst = X.worst_windows(df, "fusion")
    df.to_csv(out / "window_estimates.csv", index=False)
    overall.to_csv(out / "metrics_overall.csv"); by_act.to_csv(out / "metrics_by_activity.csv", index=False)
    by_mot.to_csv(out / "metrics_by_motion.csv", index=False); gate.to_csv(out / "sqi_gating.csv", index=False)
    worst.to_csv(out / "worst_windows.csv", index=False)
    ecg = X.summarize(df, ref_col="ref_ecg_hr", tol=tol) if df["ref_ecg_hr"].notna().any() else None
    if ecg is not None:
        ecg.to_csv(out / "metrics_overall_vs_ecg.csv")
    P.plot_overview(rec, out / "overview.png")
    P.plot_hr_comparison(df, rec, ("peaks", "spectral", "spectral_motion", "fusion"), out_path=out / "hr_comparison.png")
    P.plot_scatter_bland(df, rec, "fusion", out_path=out / "scatter_bland_altman.png")
    P.plot_error_by_activity(by_act, rec, out_path=out / "error_by_activity.png")
    P.plot_activity_comparison(df, rec, out_path=out / "activity_comparison.png")
    P.plot_sqi_vs_error(df, rec, "fusion", out_path=out / "sqi_vs_error.png")
    P.plot_motion_vs_error(df, rec, "fusion", out_path=out / "motion_vs_error.png")
    (out / "SOURCE.txt").write_text(f"data source: {rec.source_tag}\nwindows: {len(df)}\n")
    import pandas as pd
    pd.set_option("display.width", 220, "display.max_columns", None, "display.float_format", "{:.2f}".format)
    print(f"Data source: {rec.source_tag}; windows: {len(df)}\n\nOverall vs dataset labels:\n{overall}\n\nSQI gating (fusion):\n{gate}")
    print(f"\nArtifacts in {out}")


def main(argv=None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--config", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--subject", default="S1")
        sp.add_argument("--data-dir", default=None, help="folder containing S1/S1.pkl")
        sp.add_argument("--synthetic", action="store_true", help="use generated TEST data (not real results)")
        sp.add_argument("--synthetic-seconds", type=float, default=600.0)

    sp = sub.add_parser("info", help="dataset summary"); common(sp); sp.set_defaults(fn=cmd_info)
    sp = sub.add_parser("analyze", help="analyse one segment"); common(sp)
    sp.add_argument("--start", type=float, default=0.0); sp.add_argument("--duration", type=float, default=30.0)
    sp.add_argument("--method", default=None, choices=METHODS); sp.add_argument("--no-reference", action="store_true")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(fn=cmd_analyze)
    sp = sub.add_parser("ask", help="one question to the agent"); common(sp)
    sp.add_argument("question"); sp.add_argument("--backend", choices=["auto", "llm", "rule"], default=None)
    sp.set_defaults(fn=cmd_ask)
    sp = sub.add_parser("chat", help="interactive session"); common(sp)
    sp.add_argument("--backend", choices=["auto", "llm", "rule"], default=None); sp.set_defaults(fn=cmd_chat)
    sp = sub.add_parser("explore", help="EDA plots"); common(sp); sp.add_argument("--out", default="outputs/explore")
    sp.add_argument("--start", type=float, default=120.0); sp.add_argument("--max-windows", type=int, default=None)
    sp.set_defaults(fn=cmd_explore)
    sp = sub.add_parser("evaluate", help="full-recording experiments + plots"); common(sp)
    sp.add_argument("--out", default="outputs/evaluate"); sp.add_argument("--max-windows", type=int, default=None)
    sp.set_defaults(fn=cmd_evaluate)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    try:
        return args.fn(args, cfg) or 0
    except DatasetNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr); return 3
    except (DataError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr); return 2


if __name__ == "__main__":
    sys.exit(main())
