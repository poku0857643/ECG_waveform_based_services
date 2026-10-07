"""API tests for /watch/*. Tests marked needs_data read PTB-XL 500 Hz records (lead I as a stand-in for a watch);
set PTBXL_PATH to the dataset folder, otherwise they are skipped. Run from the project root: python -m pytest"""
import os

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from main import app

PTBXL_PATH = os.environ.get(
    "PTBXL_PATH",
    "/Users/eshan/Desktop/ECGprocess/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3",
)
needs_data = pytest.mark.skipif(not os.path.exists(os.path.join(PTBXL_PATH, "ptbxl_database.csv")),
                                reason="PTB-XL not found (set PTBXL_PATH)")
client = TestClient(app)
USER = {"age": 60, "sex": "male", "height_cm": 175, "weight_kg": 80, "diagnosed_label": None}


def noise_recording(seconds=10, fs=500):
    return {"signal": np.random.default_rng(0).normal(0, 0.3, seconds * fs).tolist(), "fs": fs}


@pytest.fixture(scope="module")
def watch_pair():
    """Two recordings (older first) of a test-fold patient: lead I at 500 Hz, like a watch."""
    import wfdb

    db = pd.read_csv(os.path.join(PTBXL_PATH, "ptbxl_database.csv"), index_col="ecg_id")
    per_patient = db.groupby("patient_id").size()
    patient = db[(db.strat_fold == 10) & db.patient_id.isin(per_patient[per_patient == 2].index)].patient_id.iloc[0]
    ids = db[db.patient_id == patient].sort_values("recording_date").index
    read = lambda eid: wfdb.rdsamp(os.path.join(PTBXL_PATH, db.filename_hr[eid]), channels=[0])[0][:, 0].tolist()
    return {"signal": read(ids[0]), "fs": 500}, {"signal": read(ids[1]), "fs": 500}


def test_model_info():
    r = client.get("/watch/model-info")
    assert r.status_code == 200
    assert set(r.json()["test_fold10"]) == {"NORM", "MI", "STTC", "CD", "HYP"}


def test_noise_only_asks_to_rerecord():
    r = client.post("/watch/recommendation", json={"user": USER, "recording_1": noise_recording(),
                                                   "recording_2": noise_recording()})
    assert r.status_code == 200
    assert r.json()["top_action"]["rule"] == "R1"
    assert r.json()["risk_scores"] is None


def test_too_short_recording_is_rejected():
    r = client.post("/watch/recommendation", json={"user": USER, "recording_1": {"signal": [0.0] * 100, "fs": 500},
                                                   "recording_2": noise_recording()})
    assert r.status_code == 422
    assert "at least 5 s" in r.json()["detail"][0]["msg"]


def test_invalid_user_info_is_rejected():
    r = client.post("/watch/recommendation", json={"user": {"sex": "other"}, "recording_1": noise_recording(),
                                                   "recording_2": noise_recording()})
    assert r.status_code == 422


@needs_data
def test_real_recordings(watch_pair):
    r = client.post("/watch/recommendation", json={"user": USER, "recording_1": watch_pair[0], "recording_2": watch_pair[1]})
    assert r.status_code == 200
    out = r.json()
    assert {"top_action", "actions", "risk_scores", "group", "change", "recordings", "rhythm", "peers"} <= set(out)
    assert all(rec["usable"] for rec in out["recordings"])
    assert all(0 <= s["probability"] <= 1 for s in out["risk_scores"].values())
    assert [a["level"] for a in out["actions"]] == sorted(a["level"] for a in out["actions"])


@needs_data
def test_missing_optional_user_info(watch_pair):
    r = client.post("/watch/recommendation", json={"user": {"sex": "female"}, "recording_1": watch_pair[0],
                                                   "recording_2": watch_pair[1]})
    assert r.status_code == 200


@needs_data
def test_one_noisy_recording_still_scores_the_other(watch_pair):
    r = client.post("/watch/recommendation", json={"user": USER, "recording_1": noise_recording(),
                                                   "recording_2": watch_pair[1]})
    assert r.status_code == 200
    rules = [a["rule"] for a in r.json()["actions"]]
    assert "R1" in rules and r.json()["risk_scores"] is not None
