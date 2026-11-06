"""Matplotlib figures. Every title carries the data-source tag so synthetic output cannot be mistaken
for PPG-DaLiA results."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from passos_wearable.config import load_config  # noqa: E402
from passos_wearable.data.loader import ACTIVITY_NAMES, Recording  # noqa: E402
from passos_wearable.evaluation.metrics import bland_altman, error_metrics  # noqa: E402
from passos_wearable.preprocessing.filters import dynamic_acc_magnitude, preprocess_ppg  # noqa: E402
from passos_wearable.signal_processing.frequency_analysis import power_spectrum  # noqa: E402
from passos_wearable.signal_processing.peak_detection import detect_ppg_peaks  # noqa: E402

C = {"est": "#1f77b4", "ref": "#222222", "bad": "#d62728", "alt": "#ff7f0e", "ok": "#2ca02c"}
METHOD_COLORS = {"peaks": "#1f77b4", "spectral": "#ff7f0e", "spectral_motion": "#2ca02c",
                 "fusion": "#9467bd", "fusion_continuity": "#8c564b"}


def _save(fig, out_path) -> str | None:
    if out_path is None:
        return None
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def _tag(rec_or_tag) -> str:
    return rec_or_tag.source_tag if hasattr(rec_or_tag, "source_tag") else str(rec_or_tag)


def plot_overview(rec: Recording, out_path=None, t0: float = 0.0, t1: float | None = None):
    """Wrist PPG, ACC magnitude, activity annotation and reference HR over time."""
    s, r = rec.signals, rec.reference
    t1 = min(t1 or s.duration_s, s.duration_s)
    fig, ax = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    tb = np.arange(len(s.bvp)) / s.fs_bvp; mb = (tb >= t0) & (tb <= t1)
    ax[0].plot(tb[mb], s.bvp[mb], lw=0.4, color=C["est"]); ax[0].set_ylabel("wrist BVP (a.u.)")
    ta = np.arange(len(s.acc)) / s.fs_acc; ma = (ta >= t0) & (ta <= t1)
    ax[1].plot(ta[ma], np.linalg.norm(s.acc, axis=1)[ma], lw=0.4, color=C["alt"]); ax[1].set_ylabel("|ACC| (g)")
    if s.activity is not None:
        tc = np.arange(len(s.activity)) / s.fs_activity; mc = (tc >= t0) & (tc <= t1)
        ax[2].step(tc[mc], s.activity[mc], where="post", color=C["ok"])
        ax[2].set_yticks(sorted(ACTIVITY_NAMES)); ax[2].set_yticklabels([ACTIVITY_NAMES[k] for k in sorted(ACTIVITY_NAMES)], fontsize=7)
    ax[2].set_ylabel("activity")
    tc = r.label_centers_s; mc = (tc >= t0) & (tc <= t1)
    ax[3].plot(tc[mc], r.hr_labels[mc], color=C["ref"]); ax[3].set_ylabel("reference HR (bpm)")
    ax[3].set_xlabel("time (s)")
    fig.suptitle(f"{rec.subject} overview [{_tag(rec)}]")
    for a in ax:
        a.grid(alpha=0.25)
    return _save(fig, out_path) or fig


def plot_segment(rec: Recording, start_s: float, duration_s: float, config: dict | None = None,
                 out_path=None):
    """Raw vs filtered PPG with detected peaks, ACC, and the PPG/ACC spectra for one segment."""
    cfg = config or load_config()
    seg = rec.signals.segment(start_s, duration_s)
    filt = preprocess_ppg(seg.bvp, seg.fs_bvp, cfg)
    pk = detect_ppg_peaks(filt, seg.fs_bvp, tuple(cfg["estimation"]["hr_range_bpm"]),
                          cfg["estimation"]["peak_prominence_std"])
    t = np.arange(len(filt)) / seg.fs_bvp + start_s
    fig, ax = plt.subplots(2, 2, figsize=(13, 7))
    ax[0, 0].plot(t, seg.bvp - seg.bvp.mean(), lw=0.7, color="#888"); ax[0, 0].set_title("raw BVP (mean removed)")
    ax[0, 1].plot(t, filt, lw=0.9, color=C["est"])
    ax[0, 1].plot(t[np.round(pk).astype(int).clip(0, len(t) - 1)], np.interp(pk, np.arange(len(filt)), filt), "v", color=C["bad"])
    ax[0, 1].set_title(f"band-passed BVP + {len(pk)} detected peaks")
    ta = np.arange(len(seg.acc)) / seg.fs_acc + start_s
    for c, lab in enumerate("xyz"):
        ax[1, 0].plot(ta, seg.acc[:, c] - seg.acc[:, c].mean(), lw=0.6, label=lab)
    ax[1, 0].plot(ta, dynamic_acc_magnitude(seg.acc, seg.fs_acc), color="k", lw=0.8, label="|dyn|")
    ax[1, 0].legend(ncol=4, fontsize=7); ax[1, 0].set_title("accelerometer (mean removed)")
    f, p = power_spectrum(filt, seg.fs_bvp); fa, pa = power_spectrum(seg.acc[:, 0], seg.fs_acc, 4096)
    ax[1, 1].plot(f * 60, p / p.max(), color=C["est"], label="PPG")
    ax[1, 1].plot(fa * 60, pa / max(pa.max(), 1e-12), color=C["alt"], label="ACC x")
    ax[1, 1].set_xlim(30, 240); ax[1, 1].set_xlabel("bpm"); ax[1, 1].legend(); ax[1, 1].set_title("normalised spectra")
    for a in ax.ravel():
        a.grid(alpha=0.25)
    fig.suptitle(f"{rec.subject} segment {start_s:.0f}-{start_s + duration_s:.0f}s [{_tag(rec)}]")
    return _save(fig, out_path) or fig


def plot_hr_comparison(df, rec_tag, methods=("peaks", "spectral", "fusion"), ref_col="ref_label_hr", out_path=None):
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(df["center_s"], df[ref_col], color=C["ref"], lw=2, label="reference")
    for m in methods:
        ax.plot(df["center_s"], df[f"hr_{m}"], lw=0.9, color=METHOD_COLORS.get(m), label=m, alpha=0.9)
    ax.set_xlabel("time (s)"); ax.set_ylabel("HR (bpm)"); ax.legend(ncol=len(methods) + 1, fontsize=8)
    ax.set_title(f"Estimated vs reference HR [{_tag(rec_tag)}]"); ax.grid(alpha=0.25)
    return _save(fig, out_path) or fig


def plot_scatter_bland(df, rec_tag, method="fusion", ref_col="ref_label_hr", out_path=None):
    est, ref = df[f"hr_{method}"].to_numpy(), df[ref_col].to_numpy()
    ok = np.isfinite(est) & np.isfinite(ref)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].scatter(ref[ok], est[ok], s=6, alpha=0.5, color=C["est"])
    lim = [30, 210]; ax[0].plot(lim, lim, "k--", lw=0.8); ax[0].set_xlim(lim); ax[0].set_ylim(lim)
    ax[0].set_xlabel("reference HR (bpm)"); ax[0].set_ylabel(f"{method} HR (bpm)"); ax[0].set_title("agreement")
    ba = bland_altman(est, ref)
    ax[1].scatter((est[ok] + ref[ok]) / 2, est[ok] - ref[ok], s=6, alpha=0.5, color=C["est"])
    for y, ls in [(ba["mean_diff"], "-"), (ba["loa_low"], "--"), (ba["loa_high"], "--")]:
        ax[1].axhline(y, color=C["bad"], ls=ls, lw=0.9)
    ax[1].set_xlabel("mean of est & ref (bpm)"); ax[1].set_ylabel("est - ref (bpm)")
    ax[1].set_title(f"Bland-Altman (bias {ba['mean_diff']:.1f}, LoA {ba['loa_low']:.1f}..{ba['loa_high']:.1f})")
    for a in ax:
        a.grid(alpha=0.25)
    fig.suptitle(f"[{_tag(rec_tag)}]")
    return _save(fig, out_path) or fig


def plot_error_by_activity(by_df, rec_tag, group_col="activity", metric="mae", out_path=None):
    """``by_df`` from ``evaluation.experiments.summarize_by``."""
    groups = [g for g in by_df[group_col].unique()]
    methods = list(by_df["method"].unique())
    w = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for k, m in enumerate(methods):
        sub = by_df[by_df["method"] == m].set_index(group_col)
        ax.bar(np.arange(len(groups)) + k * w, [sub.loc[g, metric] if g in sub.index else np.nan for g in groups],
               w, label=m, color=METHOD_COLORS.get(m))
    ax.set_xticks(np.arange(len(groups)) + 0.4 - w / 2); ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_ylabel(f"{metric.upper()} (bpm)"); ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")
    ax.set_title(f"{metric.upper()} by {group_col} [{_tag(rec_tag)}]")
    return _save(fig, out_path) or fig


def plot_activity_comparison(df, rec_tag, ref_col="ref_label_hr", out_path=None):
    """Reference HR, motion level and SQI per activity (box plots)."""
    acts = [a for a in sorted(df["activity"].unique()) if a != "unknown"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))
    for a_, col, lab in zip(ax, [ref_col, "motion_g", "sqi"], ["reference HR (bpm)", "motion level (g, std)", "SQI score"]):
        a_.boxplot([df.loc[df["activity"] == a, col].dropna() for a in acts], showfliers=False)
        a_.set_xticks(range(1, len(acts) + 1)); a_.set_xticklabels(acts)  # version-proof (labels= was renamed)
        a_.set_ylabel(lab); a_.tick_params(axis="x", rotation=40); a_.grid(alpha=0.25, axis="y")
    fig.suptitle(f"Movement conditions [{_tag(rec_tag)}]")
    return _save(fig, out_path) or fig


def plot_sqi_vs_error(df, rec_tag, method="fusion", ref_col="ref_label_hr", out_path=None):
    err = (df[f"hr_{method}"] - df[ref_col]).abs()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ok = err.notna() & df["sqi"].notna()
    ax[0].scatter(df.loc[ok, "sqi"], err[ok], s=6, alpha=0.5, color=C["est"])
    ax[0].set_xlabel("SQI score"); ax[0].set_ylabel("|error| (bpm)"); ax[0].set_title("SQI vs error")
    bins = np.linspace(0, 1, 6); idx = np.digitize(df.loc[ok, "sqi"], bins) - 1
    mae = [err[ok][idx == b].mean() if (idx == b).any() else np.nan for b in range(5)]
    ax[1].bar(range(5), mae, color=C["est"]); ax[1].set_xticks(range(5))
    ax[1].set_xticklabels([f"{bins[b]:.1f}-{bins[b + 1]:.1f}" for b in range(5)])
    ax[1].set_xlabel("SQI bin"); ax[1].set_ylabel("MAE (bpm)"); ax[1].set_title("MAE per SQI bin")
    for a in ax:
        a.grid(alpha=0.25)
    fig.suptitle(f"{method} [{_tag(rec_tag)}]")
    return _save(fig, out_path) or fig


def plot_motion_vs_error(df, rec_tag, method="fusion", ref_col="ref_label_hr", out_path=None):
    err = (df[f"hr_{method}"] - df[ref_col]).abs()
    ok = err.notna() & df["motion_g"].notna()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.scatter(df.loc[ok, "motion_g"], err[ok], s=6, alpha=0.5, color=C["alt"])
    ax.set_xlabel("motion level (g)"); ax.set_ylabel("|error| (bpm)"); ax.grid(alpha=0.25)
    ax.set_title(f"Motion vs error, {method} [{_tag(rec_tag)}]")
    return _save(fig, out_path) or fig
