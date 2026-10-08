from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field, model_validator

from model.watch_pipeline import WatchModel

MODEL_PATH = Path(__file__).parent / "model" / "watch_model.joblib"
MIN_SECONDS = 5  # shorter recordings have too few beats for a median beat / rhythm features

app = FastAPI()


@lru_cache
def get_model() -> WatchModel:
    return WatchModel.load(MODEL_PATH)


class UserInfo(BaseModel):
    age: float | None = Field(None, ge=0, le=120)
    sex: Literal["female", "male"] | None = None
    height_cm: float | None = Field(None, gt=0, le=250)
    weight_kg: float | None = Field(None, gt=0, le=400)
    diagnosed_label: Literal["NORM", "MI", "STTC", "CD", "HYP"] | None = None


class Recording(BaseModel):
    signal: list[float] = Field(description="single-lead (lead I) watch ECG in mV")
    fs: float = Field(gt=50, le=2000, description="sampling rate in Hz")

    @model_validator(mode="after")
    def long_enough(self):
        if len(self.signal) < MIN_SECONDS * self.fs:
            raise ValueError(f"recording is {len(self.signal) / self.fs:.1f} s; at least {MIN_SECONDS} s needed")
        return self


class WatchRequest(BaseModel):
    user: UserInfo
    recording_1: Recording = Field(description="the earlier recording")
    recording_2: Recording = Field(description="the latest recording")


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.get("/watch/model-info")
def watch_model_info():
    b = get_model().b
    return {"version": b["version"], "test_fold10": b["step3"]["test_fold10"], "rules": b["step2"]["rules"]}


@app.post("/watch/recommendation")
def watch_recommendation(req: WatchRequest):
    # plain def: the pipeline is CPU-bound, so FastAPI runs it in a worker thread
    return get_model().recommend_all(req.user.model_dump(), req.recording_1.model_dump(), req.recording_2.model_dump())
