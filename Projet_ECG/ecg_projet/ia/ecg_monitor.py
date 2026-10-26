"""Real-time ECG monitor: the same pipeline as Notebook 6, as a module the AI server can import.

Signal pieces go in (any sampling rate), beat decisions and staff alerts come out.
The model is the Random Forest saved by Notebook 4 (rf_model.joblib).
"""
from collections import deque

import joblib
import numpy as np
import neurokit2 as nk
import pandas as pd
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt

LOW_CUT_HZ, HIGH_CUT_HZ = 0.5, 40.0


def load_model(path):
    """Load rf_model.joblib (saved by Notebook 4): a dict with the model, its threshold and settings."""
    return joblib.load(path)


def clean_ecg(x, fs, powerline_hz):
    """Band-pass 0.5-40 Hz + notch at the powerline frequency (same as Notebook 2)."""
    high = min(HIGH_CUT_HZ, 0.45 * fs)       # a very low sampling rate cannot keep up to 40 Hz
    sos = butter(4, [LOW_CUT_HZ, high], btype="bandpass", fs=fs, output="sos")
    x = sosfiltfilt(sos, x)
    if powerline_hz < fs / 2:                # the notch only exists below half the sampling rate
        b, a = iirnotch(powerline_hz, Q=30, fs=fs)
        x = filtfilt(b, a, x)
    return x


def detect_r_peaks(ecg, fs):
    """R-peak positions found by neurokit2 (same as Notebook 2)."""
    _, info = nk.ecg_peaks(ecg, sampling_rate=fs, method="neurokit")
    return np.asarray(info["ECG_R_Peaks"])


def beat_features(beats, templates, rr_pre, rr_post, rr_local, fs=360, r=90):
    """The 15 hand-made features of Notebook 4, for beats given as 234-point windows at 360 Hz."""
    half_qrs = int(0.10 * fs)
    qrs = beats[:, r - half_qrs: r + half_qrs]
    before_qrs = beats[:, : r - half_qrs]
    after_qrs = beats[:, r + half_qrs:]

    slope = np.abs(np.diff(qrs, axis=1))
    max_slope = slope.max(axis=1)
    steep = slope > 0.3 * max_slope[:, None]
    first = steep.argmax(axis=1)
    last = steep.shape[1] - 1 - steep[:, ::-1].argmax(axis=1)

    a = beats - beats.mean(axis=1, keepdims=True)
    b = templates - templates.mean(axis=1, keepdims=True)
    corr = (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
    diff = np.sqrt(((beats - templates) ** 2).mean(axis=1))

    rr_pre, rr_post, rr_local = (np.asarray(v, dtype=float) for v in (rr_pre, rr_post, rr_local))
    return pd.DataFrame({
        "rr_pre": rr_pre, "rr_post": rr_post, "rr_local": rr_local,
        "prematurity": rr_pre / rr_local, "pause_after": rr_post / rr_local,
        "qrs_width_ms": (last - first + 1) / fs * 1000,
        "r_amplitude": beats[:, r], "qrs_max": qrs.max(axis=1), "qrs_min": qrs.min(axis=1),
        "max_slope": max_slope, "p_area_max": before_qrs.max(axis=1),
        "t_max": after_qrs.max(axis=1), "t_min": after_qrs.min(axis=1),
        "corr_template": corr, "diff_template": diff,
    })


class ECGMonitor:
    """Receives the ECG piece by piece, detects and classifies beats, and raises alerts."""

    def __init__(self, saved, fs, powerline_hz,
                 buffer_seconds=10, safety_seconds=1.0,
                 alert_min_beats=3, alert_window_seconds=10,
                 hr_high=150, hr_low=40, no_beat_seconds=3, cooldown_seconds=30):
        self.model, self.threshold, self.feature_names = saved["model"], saved["threshold"], saved["features"]
        self.model.n_jobs = 1      # one beat at a time: a single processor core is faster than starting many
        self.model_fs, self.before, self.after = saved["fs"], saved["before"], saved["after"]
        self.recent_windows = deque(maxlen=saved["template_beats"])   # for the patient's typical beat
        self.fs, self.powerline_hz = fs, powerline_hz
        self.buffer_size = int(buffer_seconds * fs)
        self.safety = int(safety_seconds * fs)
        self.refractory = int(0.2 * fs)    # two beats can't be closer than 200 ms
        self.alert_min_beats, self.alert_window = alert_min_beats, alert_window_seconds
        self.hr_high, self.hr_low = hr_high, hr_low
        self.no_beat_seconds, self.cooldown = no_beat_seconds, cooldown_seconds

        self.buffer = np.zeros(0)
        self.buffer_start = 0       # position (in the whole stream) of the first point of the buffer
        self.received = 0           # number of points received so far
        self.peaks = []             # confirmed R peaks, positions in the whole stream
        self.next_beat = 1          # index (in self.peaks) of the next beat to classify
        self.rr_history = []        # RR intervals seen so far, to compute rr_local
        self.beats = []             # one dictionary per classified beat
        self.alerts = []            # one dictionary per alert
        self.last_alert = {}        # time of the last alert of each type

    def now(self):
        return self.received / self.fs     # current time, in seconds since the start

    def push(self, chunk):
        """Receive a new piece of signal. Returns the alerts raised by this piece."""
        self.received += len(chunk)
        self.buffer = np.concatenate([self.buffer, np.asarray(chunk, dtype=float)])
        if len(self.buffer) > self.buffer_size:                     # forget the oldest part
            excess = len(self.buffer) - self.buffer_size
            self.buffer = self.buffer[excess:]
            self.buffer_start += excess
        if len(self.buffer) < 3 * self.fs:                         # wait for at least 3 seconds of signal
            return []

        ecg = clean_ecg(self.buffer, self.fs, self.powerline_hz)
        ecg = (ecg - ecg.mean()) / (ecg.std() + 1e-9)
        try:
            found = detect_r_peaks(ecg, self.fs)
        except Exception:      # e.g. flat signal (electrode off): no beat found
            found = []

        # Confirm the new peaks that are far enough from both edges of the buffer
        for p in found:
            if p < self.fs // 2 or p >= len(self.buffer) - self.safety:
                continue
            position = self.buffer_start + int(p)
            if not self.peaks or position > self.peaks[-1] + self.refractory:
                self.peaks.append(position)

        new_alerts = []
        # Classify every beat that now has a confirmed next beat
        while self.next_beat < len(self.peaks) - 1:
            beat = self.classify(self.next_beat, ecg)
            self.next_beat += 1
            if beat is not None:
                new_alerts += self.check_beat_alerts(beat)
        new_alerts += self.check_no_beat()
        return new_alerts

    def classify(self, i, ecg):
        previous, p, following = self.peaks[i - 1], self.peaks[i], self.peaks[i + 1]
        rr_pre = (p - previous) / self.fs
        rr_post = (following - p) / self.fs
        self.rr_history.append(rr_pre)
        rr_local = np.mean(self.rr_history[-10:])     # same as Notebook 2: average of the last 10 RR

        # Cut the window: 0.25 s before to 0.40 s after the R peak, as 234 points at 360 Hz
        local = p - self.buffer_start                  # position inside the buffer
        offsets = np.arange(-self.before, self.after) / self.model_fs * self.fs
        if local + offsets[0] < 0 or local + offsets[-1] > len(ecg) - 1:
            return None                                # window not fully in the buffer: skip
        window = np.interp(local + offsets, np.arange(len(ecg)), ecg)

        # The patient's typical beat: median of the previous beats (same as Notebook 4)
        template = np.median(np.array(self.recent_windows), axis=0) if self.recent_windows else window
        self.recent_windows.append(window)

        features = beat_features(window[None, :], template[None, :], [rr_pre], [rr_post], [rr_local],
                                 fs=self.model_fs, r=self.before)
        probability = float(self.model.predict_proba(features[self.feature_names])[0, 1])
        beat = {"time": p / self.fs, "position": p, "probability": probability,
                "abnormal": probability >= self.threshold, "heart_rate": 60 / rr_local,
                "decided_at": self.now(),
                # kept for the report (ecg_report.py): timing and shape of the beat
                "rr_pre": rr_pre, "rr_post": rr_post, "prematurity": rr_pre / rr_local,
                "corr_template": float(features["corr_template"][0]),
                "qrs_width_ms": float(features["qrs_width_ms"][0])}
        self.beats.append(beat)
        return beat

    def raise_alert(self, kind, message):
        t = self.now()
        if t - self.last_alert.get(kind, -np.inf) < self.cooldown:
            return []          # same type of alert too recently: stay quiet
        self.last_alert[kind] = t
        alert = {"time": t, "type": kind, "message": message}
        self.alerts.append(alert)
        return [alert]

    def check_beat_alerts(self, beat):
        alerts = []
        n_abnormal = 0
        for b in reversed(self.beats):      # only the last seconds: stays fast after hours of signal
            if b["time"] <= beat["time"] - self.alert_window:
                break
            n_abnormal += b["abnormal"]
        if n_abnormal >= self.alert_min_beats:
            alerts += self.raise_alert("abnormal beats",
                                       f"{n_abnormal} abnormal beats in the last {self.alert_window} s")
        if len(self.rr_history) >= 10:
            hr = beat["heart_rate"]
            if hr > self.hr_high:
                alerts += self.raise_alert("heart rate", f"fast heart rate: {hr:.0f} beats/min")
            elif hr < self.hr_low:
                alerts += self.raise_alert("heart rate", f"slow heart rate: {hr:.0f} beats/min")
        return alerts

    def check_no_beat(self):
        last_seen = self.peaks[-1] / self.fs if self.peaks else 0.0
        # the last second is never confirmed yet (safety delay), so we allow for it
        if self.now() - last_seen > self.no_beat_seconds + self.safety / self.fs and self.now() > 5:
            return self.raise_alert("no beat", f"no beat detected for {self.now() - last_seen:.0f} s "
                                               "(check the patient and the electrodes)")
        return []
