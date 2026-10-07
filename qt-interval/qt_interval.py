"""QT interval from a median heartbeat (12-lead or single-lead), and the usual heart-rate corrections (QTc).

QRS onset: where the QRS magnitude (root-sum-square of the leads) rises above 15% of its peak, searching back from the R-peak.
T end: tangent method - the steepest slope after the T peak, extended to the isoelectric (PR) level; median over the
leads with a clear T wave.
"""
import neurokit2 as nk
import numpy as np
from scipy.signal import butter, filtfilt

BEFORE_S, AFTER_S = 0.3, 0.7  # median-beat window around the R-peak: long enough for a prolonged QT
LEAD_I = 0
MIN_T_MV = 0.05  # T waves flatter than this give no reliable T end


def median_beat_long(sig, fs):
    """(n, leads) raw ECG -> (median beat, R index in the beat, median RR in s), or None. Beats found on the
    root-sum-square of all given leads (works for any QRS polarity and for a single lead)."""
    b, a = butter(2, [0.5, 40], btype="band", fs=fs)
    sig = filtfilt(b, a, np.asarray(sig, dtype=float), axis=0)
    if sig.ndim == 1:
        sig = sig[:, None]
    det = np.sqrt((sig ** 2).sum(axis=1))
    try:
        _, info = nk.ecg_peaks(det, sampling_rate=fs, correct_artifacts=True)
    except Exception:
        return None
    snap = int(0.05 * fs)
    peaks = np.array([max(p - snap, 0) + np.argmax(det[max(p - snap, 0):p + snap + 1]) for p in info["ECG_R_Peaks"]])
    if len(peaks) < 4:
        return None
    rr = np.median(np.diff(peaks)) / fs
    pre, post = int(BEFORE_S * fs), int(AFTER_S * fs)
    inside = [p for p in peaks if p - pre >= 0 and p + post <= len(sig)]
    if len(inside) < 3:
        return None
    return np.median([sig[p - pre:p + post] for p in inside], axis=0), pre, rr


def qrs_onset(beat, r, fs, frac=0.15):
    """Last sample before the R-peak (within 150 ms) where the QRS magnitude is below frac of its rise."""
    mag = np.sqrt((beat ** 2).sum(axis=1))
    lo = r - int(0.15 * fs)
    base = mag[lo:r].min()
    below = np.where(mag[lo:r + 1] <= base + frac * (mag[r] - base))[0]
    return lo + below[-1] if len(below) else None


def t_end_tangent(x, onset, r, rr, fs):
    """Tangent-method T end (sample index) in one lead, or None if the T wave is too flat / not found."""
    iso = x[max(onset - int(0.02 * fs), 0):max(onset - int(0.004 * fs), 1)].mean()  # PR segment level
    start = max(r + int(0.12 * fs), onset + int(0.2 * fs))  # after the QRS, also when it is wide
    end = min(len(x) - 2, r + int((rr - 0.25) * fs))  # stop before the next P wave
    if end - start < int(0.1 * fs):
        return None
    seg = x[start:end] - iso
    tp = start + np.argmax(np.abs(seg))
    amp = x[tp] - iso
    if abs(amp) < MIN_T_MV:
        return None
    d = np.gradient(x)
    after = d[tp:end]
    if len(after) < 3:
        return None
    ts = tp + (np.argmin(after) if amp > 0 else np.argmax(after))  # steepest slope back towards the baseline
    if d[ts] == 0 or np.sign(d[ts]) == np.sign(amp):
        return None
    te = ts - (x[ts] - iso) / d[ts]
    return te if tp < te <= end + int(0.05 * fs) else None


T_LEADS_12 = (0, 1, 7, 8, 9, 10, 11)  # I, II, V2-V6: leads where the T wave is usually clear


def measure_qt(sig, fs, t_leads=T_LEADS_12):
    """QT (ms), RR (s), heart rate and QRS onset from a raw (n, leads) ECG. T end = median of the tangent-method
    T ends over the t_leads with a clear T wave (robust to one bad lead). For a single-lead watch ECG pass sig as
    (n,) or (n, 1) and t_leads=(0,)."""
    mb = median_beat_long(sig, fs)
    if mb is None:
        return None
    beat, r, rr = mb
    onset = qrs_onset(beat, r, fs)
    if onset is None:
        return None
    ends = [te for lead in t_leads if (te := t_end_tangent(beat[:, lead], onset, r, rr, fs)) is not None]
    qt = (np.median(ends) - onset) * 1000 / fs if ends else np.nan
    return {"qt_ms": qt, "rr_s": rr, "heart_rate": 60 / rr, "t_leads_used": len(ends),
            "qrs_onset_ms": (onset - r) * 1000 / fs, **qtc(qt, rr)}


def qtc(qt_ms, rr_s):
    """Heart-rate corrected QT (ms) by the four common formulas."""
    return {
        "qtc_bazett": qt_ms / np.sqrt(rr_s),
        "qtc_fridericia": qt_ms / np.cbrt(rr_s),
        "qtc_framingham": qt_ms + 154 * (1 - rr_s),
        "qtc_hodges": qt_ms + 1.75 * (60 / rr_s - 60),
    }
