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

    # --- step 2: health / lifestyle actions ---
    def low_rmssd_cutoff(self, age):
        s2 = self.b["step2"]
        if age != age:  # unknown age (e.g. > 89, masked in PTB-XL)
            return s2["rmssd_low_overall"]
        return s2["hrv_norms"].loc[pd.cut([age], s2["age_bands"], right=False)[0], "10%"]

    def health_actions(self, user, rec_1, rec_2):
        """Step 2: user info + two watch recordings (older first) -> prioritised actions (+ the step-1 result)."""
        s2 = self.b["step2"]
        rules, af_feature, af_threshold = s2["rules"], s2["af_feature"], s2["af_threshold"]
        user = clean_user(user)
        step1 = self.recommend(user, rec_1, rec_2)
        acts = []

        def add(level, rule, action, reason):
            acts.append({"level": level, "category": LEVELS[level], "rule": rule, "action": action, "reason": reason})

        usable = [r["usable"] for r in step1["recordings"]]
        rhythm = [rhythm_features(watch_rr(rec["signal"], rec["fs"])) if ok else None for rec, ok in zip([rec_1, rec_2], usable)]
        if not all(r is not None for r in rhythm):  # R1
            add(3, "R1", "Re-record: sit still for 1 minute first, rest your arm on a table, keep your finger on the electrode "
                         "for the whole recording.", f"{sum(r is None for r in rhythm)} of 2 recordings too noisy to read")
        if all(r is None for r in rhythm):
            return plain({"top_action": acts[0], "actions": acts, "rhythm": rhythm, "step1": step1})

        latest = next(r for r in reversed(rhythm) if r is not None)
        hr = latest["heart_rate"]
        irregular = [r is not None and r[af_feature] > af_threshold for r in rhythm]
        label = user.get("diagnosed_label")
        known_diagnosis = label in SUPERCLASSES and label != "NORM"
        group = step1.get("group", {}).get("group")
        age = user["age"]

        if hr < rules["hr_urgent_low"] or hr > rules["hr_urgent_high"]:  # R2
            add(1, "R2", "Seek medical advice promptly - urgently if you feel dizzy, faint, short of breath or have chest pain.",
                f"resting heart rate {hr:.0f} bpm")
        elif hr > rules["hr_high"]:  # R3
            add(3, "R3", "Rest for 5 minutes and re-record; if it stays above 100 bpm at rest, see a clinician.",
                f"resting heart rate {hr:.0f} bpm (> {rules['hr_high']})")
        elif hr < rules["hr_low"]:  # R4
            add(4, "R4", "A slow heart rate is common if you are physically fit; if you feel dizzy or unusually tired, "
                         "see a clinician.", f"resting heart rate {hr:.0f} bpm (< {rules['hr_low']})")

        if all(irregular):  # R5
            add(2, "R5", "Irregular rhythm in both recordings (possible atrial fibrillation) - see a clinician and share "
                         "these recordings.", f"{af_feature} above {af_threshold:.2f} in both")
        elif irregular[-1]:  # R6
            add(3, "R6", "Irregular rhythm in the latest recording - re-record at rest; if it is irregular again, see a clinician.",
                f"{af_feature} {rhythm[-1][af_feature]:.2f} > {af_threshold:.2f}")

        if step1.get("change", {}).get("changed"):  # R7
            add(2, "R7", "Your heartbeat shape changed between recordings - re-record; if the change persists, see a clinician.",
                f"beat similarity r = {step1['change']['similarity']:.2f} < {step1['change']['threshold']:.2f}")
        if group == 4 and not known_diagnosis:  # R8
            add(2, "R8", "Your heartbeat shape resembles a group where almost all PTB-XL patients had abnormal 12-lead "
                         "findings - ask a clinician about a 12-lead ECG.", "step-1 group 4")
        if known_diagnosis:  # R9
            add(4, "R9", "Follow your clinician's plan and bring these recordings to your next check-up.",
                f"diagnosed label {label}")

        if all(r is not None for r in rhythm) and abs(rhythm[1]["heart_rate"] - rhythm[0]["heart_rate"]) > rules["hr_change"]:  # R10
            add(3, "R10", "Record at rest and at the same time of day, so your recordings can be compared.",
                f"heart rate {rhythm[0]['heart_rate']:.0f} -> {rhythm[1]['heart_rate']:.0f} bpm")

        low_hrv = (not irregular[-1] and rules["hr_low"] <= hr <= rules["hr_high"]
                   and latest["rmssd_ms"] < self.low_rmssd_cutoff(age))
        if low_hrv:  # R11
            add(4, "R11", "Heart rate variability is low for your age - make today a recovery day: sleep, hydration, "
                          "lower stress.", f"RMSSD {latest['rmssd_ms']:.0f} ms < {self.low_rmssd_cutoff(age):.0f} ms "
                                           "(10th percentile for age)")

        if acts and min(a["level"] for a in acts) <= 2:  # R12
            add(4, "R12", "Avoid strenuous exercise until a clinician has reviewed this.", "a level 1-2 action above")
        elif group in (3, 4) or age >= rules["senior_age"]:
            add(4, "R12", "Moderate activity (brisk walking, cycling); build up gradually.", f"group {group}, age {age:.0f}")
        elif low_hrv:
            add(4, "R12", "Light activity only today.", "low HRV")
        else:
            add(5, "R12", "Normal activity: aim for at least 150 minutes of moderate exercise a week.", "no flags")

        if user.get("height_cm") and user.get("weight_kg"):  # R13
            bmi = user["weight_kg"] / (user["height_cm"] / 100) ** 2
            if bmi >= rules["bmi_high"]:
                add(4, "R13", "Weight management helps heart health - discuss a plan with a clinician or dietitian.",
                    f"BMI {bmi:.1f}")
            elif bmi < rules["bmi_low"]:
                add(4, "R13", "Your weight is low for your height - discuss with a clinician or dietitian.", f"BMI {bmi:.1f}")

        if min(a["level"] for a in acts) >= 4:  # R14
            add(5, "R14", "Keep recording once a week, at rest and at the same time of day, to track changes.",
                "no level 1-3 actions")

        acts.sort(key=lambda a: a["level"])
        return plain({"top_action": acts[0], "actions": acts,
                      "rhythm": [None if r is None else {k: round(v, 3) for k, v in r.items()} for r in rhythm],
                      "step1": step1})

    # --- step 3: diagnosis-risk scores ---
    def feature_row(self, beat_t, rhythm, user):
        """One recording (watch_beat output + rhythm_features) + user info -> model features."""
        s1 = self.b["step1"]
        centroids = to_unit(s1["kmeans"].cluster_centers_)
        group_order = sorted(s1["relabel"], key=s1["relabel"].get)  # raw k-means labels in group 1..k order
        aligned = align(beat_t["beat"][:, 0][None], s1["reference_beat"])  # (1, 60) in mV: amplitude matters
        v = to_unit(aligned)[0]
        row = {f"beat_{i}": x for i, x in enumerate(aligned[0])}
        row |= {f"sim_group_{g + 1}": v @ centroids[raw] for g, raw in enumerate(group_order)}
        row["consistency"] = beat_t["consistency"]
        row |= {k: rhythm[k] if rhythm else np.nan for k in RHYTHM_COLS}
        h, w = user.get("height_cm"), user.get("weight_kg")
        age = user.get("age")
        row |= {"age": np.nan if age is None else age,
                "sex": {"male": 0, "female": 1}.get(str(user.get("sex", "")).lower(), np.nan),
                "height_cm": h or np.nan, "weight_kg": w or np.nan, "bmi": w / (h / 100) ** 2 if h and w else np.nan}
        return row

    def risk_scores(self, user, rec_1, rec_2):
        """Per-class probability from two watch recordings (combined as chosen in training) + user info."""
        s3 = self.b["step3"]
        probs = []
        for rec in (rec_1, rec_2):
            t = watch_beat(rec["signal"], rec["fs"])
            if t is None or not t["usable"]:
                probs.append(None)
                continue
            F = pd.DataFrame([self.feature_row(t, rhythm_features(watch_rr(rec["signal"], rec["fs"])), user)])[s3["feature_cols"]]
            probs.append(np.array([s3["models"][sc].predict_proba(F)[0, 1] for sc in SUPERCLASSES]))
        usable = [p for p in probs if p is not None]
        if not usable:
            return None
        p = np.mean(usable, axis=0) if s3["combine"] == "average of both" else usable[-1]
        prevalence, high = s3["prevalence"], s3["risk_high"]
        return {sc: {"probability": round(float(v), 3), "prevalence": round(float(prevalence[sc]), 3),
                     "level": "high" if v >= high else "above average" if v >= prevalence[sc] else "below average"}
                for sc, v in zip(SUPERCLASSES, p)}

    # --- steps 1-3 together ---
    def recommend_all(self, user, rec_1, rec_2):
        """Group + profile, health / lifestyle actions, diagnosis-risk scores -> one prioritised list."""
        step2 = self.health_actions(user, rec_1, rec_2)
        risks = self.risk_scores(user, rec_1, rec_2)
        acts = list(step2["actions"])
        if risks:
            label = user.get("diagnosed_label")
            known = label in SUPERCLASSES and label != "NORM"
            high = [sc for sc in SUPERCLASSES if sc != "NORM" and risks[sc]["level"] == "high"]
            for sc in high:
                if sc == label:
                    continue  # already diagnosed: covered by R9
                acts.append({"level": 2, "category": LEVELS[2], "rule": "S3",
                             "action": f"High {SUPERCLASS_NAMES[sc]} score - ask a clinician about a 12-lead ECG.",
                             "reason": f"{sc} probability {risks[sc]['probability']:.0%} (average {risks[sc]['prevalence']:.0%})"})
            if known and risks[label]["level"] == "below average":
                acts.append({"level": 4, "category": LEVELS[4], "rule": "S3",
                             "action": f"Your diagnosed {SUPERCLASS_NAMES[label]} is not visible on the watch ECG - a single "
                                       "lead can miss it, so keep your clinical follow-up.",
                             "reason": f"{label} probability {risks[label]['probability']:.0%}"})
            if not high and risks["NORM"]["level"] == "high" and not known:
                acts.append({"level": 5, "category": LEVELS[5], "rule": "S3",
                             "action": "Your watch ECG looks like typical normal ECGs (a single lead can't rule everything out).",
                             "reason": f"NORM probability {risks['NORM']['probability']:.0%}"})
        if any(a["rule"] == "S3" and a["level"] <= 2 for a in acts):  # keep step-2 advice consistent with a new level-2 action
            acts = [a for a in acts if a["rule"] not in ("R12", "R14")]
            acts.append({"level": 4, "category": LEVELS[4], "rule": "R12",
                         "action": "Avoid strenuous exercise until a clinician has reviewed this.",
                         "reason": "a level 1-2 action above"})
        acts.sort(key=lambda a: a["level"])
        step1 = step2["step1"]
        return plain({"top_action": acts[0], "actions": acts, "risk_scores": risks,
                      "group": step1.get("group"), "change": step1.get("change"), "recordings": step1["recordings"],
                      "rhythm": step2["rhythm"], "peers": step1.get("peers"), "bmi": step1.get("bmi"),
                      "disclaimer": DISCLAIMER})
