"""Synthetic wrist-PPG recordings in the PPG-DaLiA layout.

ONLY for unit tests, smoke tests and the ``--synthetic`` demo. Results on this data say nothing about
real PPG-DaLiA performance; every output derived from it is tagged ``SYNTHETIC``.
"""
from __future__ import annotations

import numpy as np

from passos_wearable.config import load_config
from passos_wearable.data.loader import Recording, build_recording

# (activity code, duration s, target HR bpm, motion freq Hz, ACC amplitude g)
_SCHEDULE = [(1, 70, 68, 0.3, 0.01), (7, 105, 105, 1.8, 0.5), (8, 75, 78, 0.5, 0.05),
             (4, 125, 118, 1.4, 0.3), (1, 70, 70, 0.3, 0.01), (2, 130, 128, 2.0, 0.7)]


def make_synthetic_recording(duration_s: float = 600.0, seed: int = 0, subject: str = "S1",
                             config: dict | None = None, motion_scale: float = 1.0) -> Recording:
    cfg = config or load_config()
    d = cfg["data"]
    fs_b, fs_a, fs_act, fs_ecg = d["fs_ppg"], d["fs_acc"], d["fs_activity"], d["fs_ecg"]
    rng = np.random.default_rng(seed)
    n_b = int(duration_s * fs_b)
    t = np.arange(n_b) / fs_b

    # piecewise schedule -> target HR / motion parameters per sample
    hr_target = np.zeros(n_b); fm = np.zeros(n_b); amp = np.zeros(n_b); act = np.zeros(n_b, int)
    pos, k = 0, 0
    while pos < n_b:
        code, dur, hr, f, a = _SCHEDULE[k % len(_SCHEDULE)]
        end = min(n_b, pos + int(dur * fs_b))
        hr_target[pos:end], fm[pos:end], amp[pos:end], act[pos:end] = hr, f, a * motion_scale, code
        pos, k = end, k + 1
    # first-order smoothing (tau ~ 15 s) + slow random drift
    alpha = 1.0 - np.exp(-1.0 / (15.0 * fs_b))
    hr = np.empty(n_b); hr[0] = hr_target[0]
    for i in range(1, n_b):
        hr[i] = hr[i - 1] + alpha * (hr_target[i] - hr[i - 1])
    hr += np.convolve(rng.normal(0, 0.6, n_b), np.ones(int(5 * fs_b)) / (5 * fs_b), mode="same")

    phase = 2 * np.pi * np.cumsum(hr / 60.0) / fs_b
    pulse = np.sin(phase) + 0.5 * np.sin(2 * phase - 1.0) + 0.2 * np.sin(3 * phase - 2.0)
    mphase = 2 * np.pi * np.cumsum(fm) / fs_b
    artifact = 2.0 * amp * (np.sin(mphase + 0.7) + 0.5 * np.sin(2 * mphase + 0.2))
    wander = 0.4 * np.sin(2 * np.pi * 0.1 * t + rng.uniform(0, 6))
    bvp = 50.0 * (pulse + artifact + wander + rng.normal(0, 0.15, n_b))

    n_a = int(duration_s * fs_a)
    ta = np.arange(n_a) / fs_a
    amp_a = np.interp(ta, t, amp); mph_a = np.interp(ta, t, mphase)
    acc = np.zeros((n_a, 3))
    for c, (gain, ph) in enumerate([(1.0, 0.0), (0.7, 1.1), (0.9, 2.3)]):
        acc[:, c] = amp_a * gain * (np.sin(mph_a + ph) + 0.3 * np.sin(2 * mph_a + ph)) + rng.normal(0, 0.01, n_a)
    acc[:, 2] += 1.0  # gravity

    n_lab = int((duration_s - d["label_window_s"]) // d["label_step_s"]) + 1
    labels = np.array([hr[int(i * d["label_step_s"] * fs_b): int((i * d["label_step_s"] + d["label_window_s"]) * fs_b)].mean()
                       for i in range(n_lab)])
    cycles = phase / (2 * np.pi)
    beat_idx = np.arange(1, int(cycles[-1]))
    rpeaks = np.round(np.interp(beat_idx, cycles, t) * fs_ecg).astype(int)
    activity = act[:: int(fs_b / fs_act)]
    return build_recording(subject, "synthetic", bvp, acc, activity, labels, rpeaks, cfg, meta={"seed": seed})
