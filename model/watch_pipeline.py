"""Watch ECG recommendation pipeline, shared by the training notebook (ml.ipynb) and the API (main.py).

Input: user info + two single-lead (lead I) watch recordings, older first.
Output: step 1 (median-beat group + profile), step 2 (health / lifestyle actions), step 3 (diagnosis-risk scores),
merged into one prioritised list of actions. Trained values live in a bundle dict saved as watch_model.joblib.
"""
import joblib
import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

FS = 100  # model sampling rate (PTB-XL records100)
SNAP = int(0.05 * FS)  # +-50 ms
BEAT_WINDOW_S = (0.25, 0.45)  # before / after the QRS peak
LEAD_II = 1  # lead order: I, II, III, AVR, AVL, AVF, V1-V6
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
SUPERCLASS_NAMES = {"NORM": "normal ECG", "MI": "myocardial infarction", "STTC": "ST/T change",
                    "CD": "conduction disturbance", "HYP": "hypertrophy"}
RHYTHM_COLS = ["heart_rate", "rmssd_ms", "sdnn_ms", "nrmssd", "irregularity"]
INFO_COLS = ["age", "sex", "height_cm", "weight_kg", "bmi"]
LEVELS = {1: "seek care promptly", 2: "see a clinician", 3: "re-record", 4: "lifestyle", 5: "routine"}
DISCLAIMER = "Similarity to PTB-XL patients, not a diagnosis."

B_BP, A_BP = butter(2, [0.5, 40], btype="band", fs=FS)  # remove baseline drift + noise


# ---------- signal processing ----------

def beat_template(sig, detect_on="rss", before=BEAT_WINDOW_S[0], after=BEAT_WINDOW_S[1]):
    """Median beat + quality info from a filtered (n, leads) signal at FS.
    detect_on="lead_ii" (old) or "rss" = root-sum-square of all leads, which peaks at the QRS whatever its polarity."""
    pre, post = int(before * FS), int(after * FS)
    det = sig[:, LEAD_II] if detect_on == "lead_ii" else np.sqrt((sig ** 2).sum(axis=1))
    try:
        _, info = nk.ecg_peaks(det, sampling_rate=FS, correct_artifacts=True)
    except Exception:
        return None
    peaks = np.asarray(info["ECG_R_Peaks"])
    if detect_on == "rss":  # snap each beat to its QRS energy peak
        peaks = np.array([max(p - SNAP, 0) + np.argmax(det[max(p - SNAP, 0):p + SNAP + 1]) for p in peaks])
    peaks = [p for p in peaks if p - pre >= 0 and p + post <= len(sig)]
    if len(peaks) < 2:
        return None
    beats_ = np.stack([sig[p - pre:p + post] for p in peaks])
    med = np.median(beats_, axis=0)
    return {
        "beat": med,
        "consistency": np.median([np.corrcoef(b.ravel(), med.ravel())[0, 1] for b in beats_]),  # beat vs median
        "qrs_offset_ms": (np.sqrt((med ** 2).sum(axis=1)).argmax() - pre) * 1000 / FS,  # 0 = aligned on QRS
        "heart_rate": 60 * FS / np.median(np.diff(peaks)),
    }


def watch_beat(ecg, fs):
    """Single-lead ECG of any length / sampling rate -> median beat at FS + quality (same cleaning as 12-lead)."""
    ecg = np.asarray(ecg, dtype=float)
    if fs != FS:
        ecg = nk.signal_resample(ecg, sampling_rate=fs, desired_sampling_rate=FS)
    sig = filtfilt(B_BP, A_BP, ecg)[:, None]
    t = beat_template(sig, detect_on="rss")  # root-sum-square of one lead = |signal|: works for either QRS polarity
    if t is None:
        return None
    t["usable"] = bool(t["consistency"] >= 0.8 and np.abs(sig).max() <= 10)
    return t


def watch_rr(ecg, fs):
    """RR intervals (ms) at the recording's own sampling rate (no resampling: keeps HRV precision on a 512 Hz watch)."""
    ecg = np.asarray(ecg, dtype=float)
    b_, a_ = butter(2, [0.5, 40], btype="band", fs=fs)
    det = np.abs(filtfilt(b_, a_, ecg))  # |lead I|: works for either QRS polarity
    try:
        _, info = nk.ecg_peaks(det, sampling_rate=fs, correct_artifacts=False)  # no correction: keep real irregularity
    except Exception:
        return np.array([])
    snap = int(0.05 * fs)
    peaks = np.unique([max(p - snap, 0) + np.argmax(det[max(p - snap, 0):p + snap + 1]) for p in info["ECG_R_Peaks"]])
    return np.diff(peaks) * 1000 / fs


def rhythm_features(rr):
    if len(rr) < 4:
        return None
    d = np.diff(rr)
    return {
        "heart_rate": 60000 / np.median(rr),
        "rmssd_ms": np.sqrt(np.mean(d ** 2)),
        "sdnn_ms": rr.std(ddof=1),
        "nrmssd": np.sqrt(np.mean(d ** 2)) / rr.mean(),  # irregularity, sensitive to every irregular beat
        "irregularity": np.median(np.abs(d)) / np.median(rr),  # irregularity, robust to a single extra beat
    }


def pearson(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))


def shift_pearson(a, b, max_shift=SNAP):
    """Best Pearson over shifts of +-50 ms: compares beat shape, ignores small alignment differences."""
    return max(pearson(a[s:], b[:len(b) - s]) if s >= 0 else pearson(a[:s], b[-s:]) for s in range(-max_shift, max_shift + 1))


def to_unit(a):
    """Rows -> centred, unit length (dot product = Pearson r)."""
    a = a.reshape(len(a), -1)
    a = a - a.mean(axis=1, keepdims=True)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def align(beats_1d, ref):
    """Shift each (n, 70) beat by +-50 ms to best match ref, crop to the common window -> (n, 60)."""
    n_samples = beats_1d.shape[1]
    windows = np.stack([beats_1d[:, SNAP + s:n_samples - SNAP + s] for s in range(-SNAP, SNAP + 1)])
    best = np.argmax([to_unit(w) @ ref for w in windows], axis=0)
    return windows[best, np.arange(len(beats_1d))]


def plain(obj):
    """numpy scalars -> plain Python numbers, NaN -> None (readable output, valid JSON)."""
    if isinstance(obj, dict):
        return {k: plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and obj != obj:
        return None
    return obj


def clean_user(user):
    """Missing age -> NaN, so comparisons work; everything else as given."""
    user = dict(user)
    if user.get("age") is None:
        user["age"] = np.nan
    return user
