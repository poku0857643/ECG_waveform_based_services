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


# ---------- the model ----------

class WatchModel:
    """Steps 1-3 on top of a bundle dict. The notebook fills the bundle as training proceeds (values are read at
    call time); the API loads the finished bundle with WatchModel.load("watch_model.joblib")."""

    def __init__(self, bundle):
        self.b = bundle

    @classmethod
    def load(cls, path):
        return cls(joblib.load(path))

    # --- step 1: median-beat group + profile ---
    def assign_group(self, beat_1d):
        """(70,) median beat -> (group 1..k, aligned unit vector)."""
        s1 = self.b["step1"]
        v = to_unit(align(beat_1d[None], s1["reference_beat"]))
        return s1["relabel"][s1["kmeans"].predict(v)[0]], v[0]

    def recommend(self, user, rec_1, rec_2):
        """user: {"age", "sex": "female"/"male", "height_cm", "weight_kg", "diagnosed_label": NORM/MI/STTC/CD/HYP or None}
        rec_1, rec_2: {"signal": 1-D array (single lead, mV), "fs": sampling rate in Hz}; rec_1 is the older recording."""
        s1 = self.b["step1"]
        user = clean_user(user)
        out = {"recordings": [], "recommendations": []}
        beats_ = []
        for name, rec in [("recording 1", rec_1), ("recording 2", rec_2)]:
            t = watch_beat(rec["signal"], rec["fs"])
            if t is None or not t["usable"]:
                out["recordings"].append({"recording": name, "usable": False})
                out["recommendations"].append(f"{name}: signal quality too low - re-record sitting still, arm resting, "
                                              "finger on the crown/electrode for the full recording.")
                continue
            group, _ = self.assign_group(t["beat"][:, 0])
            beats_.append(t["beat"])
            out["recordings"].append({"recording": name, "usable": True, "group": group,
                                      "heart_rate_bpm": round(t["heart_rate"], 1),
                                      "beat_consistency": round(t["consistency"], 3)})
        if not beats_:
            return plain(out)

        latest = [r for r in out["recordings"] if r["usable"]][-1]
        g = latest["group"]
        prof = s1["profile"].loc[g]
        top = prof[[f"{sc}_pct" for sc in SUPERCLASSES if sc != "NORM"]].sort_values(ascending=False)
        top_text = ", ".join(f"{k.removesuffix('_pct')} {v:.0f}%" for k, v in top.head(2).items())
        if prof.NORM_pct >= 60:
            advice = "beat shape typical of mostly-normal ECGs - routine monitoring."
        elif prof.NORM_pct >= 40:
            advice = f"mixed group ({prof.NORM_pct:.0f}% normal; {top_text}) - keep recording regularly and compare over time."
        else:
            advice = (f"beat shape common in people with abnormal ECG findings ({top_text}; only {prof.NORM_pct:.0f}% normal) "
                      "- consider discussing a 12-lead ECG with a clinician.")
        out["group"] = {"group": g, "NORM_pct": round(prof.NORM_pct, 1), **{k: round(v, 1) for k, v in top.items()}}
        out["recommendations"].append(f"Group {g}: {advice}")

        if len(beats_) == 2:  # change between the two recordings
            threshold = s1["change_threshold"]
            r = shift_pearson(beats_[0], beats_[1])
            changed = r < threshold
            g1, g2 = out["recordings"][0]["group"], out["recordings"][1]["group"]
            out["change"] = {"similarity": round(r, 3), "threshold": round(threshold, 3), "changed": bool(changed),
                             "group_changed": g1 != g2}
            if changed:
                out["recommendations"].append(f"Beat shape changed more than in 95% of repeat recordings of the same person "
                                              f"(r = {r:.2f} < {threshold:.2f}) - re-record; if it persists, consult a clinician.")
            elif g1 != g2:  # ~1 in 4 repeat recordings of the same person switch group: the beat sits near a group boundary
                out["recommendations"].append(f"Beat shape is consistent (r = {r:.2f}) but sits between group {g1} and group {g2}; "
                                              f"the latest recording (group {g2}) is used - a third recording would confirm it.")
            else:
                out["recommendations"].append(f"Beat shape is consistent between the two recordings (r = {r:.2f}).")

        # peers: training patients in the same group, same sex, age +-5 (fall back to the whole group)
        sex = {"male": 0, "female": 1}.get(str(user.get("sex", "")).lower())
        m = self.b["peers"]
        peers = m[m.group == g]
        narrowed = peers[(peers.sex == sex) & ((peers.age - user["age"]).abs() <= 5)]
        if len(narrowed) >= 30:
            peers = narrowed
        peer_info = {"peers": len(peers), "matched_on": "group + sex + age +-5" if peers is narrowed else "group only",
                     **{f"{sc}_pct": round(peers.diagnostic_superclass.apply(lambda x: sc in x).mean() * 100, 1)
                        for sc in SUPERCLASSES}}
        label = user.get("diagnosed_label")
        if label in SUPERCLASSES:
            overall = m.diagnostic_superclass.apply(lambda x: label in x).mean() * 100
            peer_info[f"diagnosed_label_{label}_pct_in_peers"] = peer_info[f"{label}_pct"]
            peer_info[f"diagnosed_label_{label}_pct_overall"] = round(overall, 1)
            if label != "NORM" and peer_info[f"{label}_pct"] < overall:
                out["recommendations"].append(f"Your diagnosed label ({label}) is less common in this group than overall "
                                              f"({peer_info[f'{label}_pct']:.0f}% vs {overall:.0f}%) - a single-lead watch may "
                                              "not show it; keep your clinical follow-up.")
        out["peers"] = peer_info
        if user.get("height_cm") and user.get("weight_kg"):
            out["bmi"] = round(user["weight_kg"] / (user["height_cm"] / 100) ** 2, 1)  # reported only: too sparse in PTB-XL
        out["disclaimer"] = DISCLAIMER
        return plain(out)
