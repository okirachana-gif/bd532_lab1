import json
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from features import FEATURES  # import features ด้วย เพื่อให้ joblib รู้จัก OrdinalMapper ตอนโหลดโมเดล

# โหลด pipeline และข้อมูลกำกับครั้งเดียวตอนเริ่ม server
pipe = joblib.load("model/pipe.joblib")
with open("model/metadata.json") as f:
    metadata = json.load(f)

THRESHOLD = 0.5

app = FastAPI(
    title="Job Change Prediction API",
    description="ทำนายว่าผู้สมัครกำลังมองหางานใหม่หรือไม่ (BD532 Lab 1.2)",
    version=metadata["model_version"],
)


class Candidate(BaseModel):
    """ข้อมูลผู้สมัคร 1 คน: field ที่เป็น None ได้คือ field ที่ข้อมูลจริงมีค่าว่าง pipeline จะเติมค่าให้เอง"""

    city_development_index: float = Field(ge=0, le=1)
    relevent_experience: Literal["Has relevent experience", "No relevent experience"]
    experience: str | None = Field(default=None, pattern=r"^(<1|[1-9]|1[0-9]|20|>20)$")
    last_new_job: str | None = Field(default=None, pattern=r"^(never|[1-4]|>4)$")
    training_hours: int = Field(ge=0)
    gender: Literal["Male", "Female", "Other"] | None = None
    enrolled_university: Literal["no_enrollment", "Full time course", "Part time course"] | None = None
    education_level: Literal["Primary School", "High School", "Graduate", "Masters", "Phd"] | None = None
    major_discipline: Literal["STEM", "Business Degree", "Arts", "Humanities", "No Major", "Other"] | None = None
    company_size: Literal["<10", "10/49", "50-99", "100-500", "500-999", "1000-4999", "5000-9999", "10000+"] | None = None
    company_type: Literal["Pvt Ltd", "Funded Startup", "Early Stage Startup", "Public Sector", "NGO", "Other"] | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "city_development_index": 0.624,
                "relevent_experience": "No relevent experience",
                "experience": "5",
                "last_new_job": "never",
                "training_hours": 83,
                "gender": "Male",
                "enrolled_university": "Full time course",
                "education_level": "Graduate",
                "major_discipline": "STEM",
                "company_size": None,
                "company_type": None,
            }]
        }
    }


class Prediction(BaseModel):
    will_change_job: bool
    probability: float
    threshold: float
    model_version: str


@app.get("/")
def root():
    return {"service": "Job Change Prediction API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return metadata


@app.post("/predict", response_model=Prediction)
def predict(candidate: Candidate):
    df = pd.DataFrame([candidate.model_dump()], columns=FEATURES)
    # แปลง None จาก JSON ให้เป็น NaN แบบเดียวกับค่าว่างในไฟล์ CSV ตอนเทรน
    # ถ้าไม่แปลง SimpleImputer จะไม่นับ None เป็นค่าว่าง และคำทำนายจะผิดโดยไม่มี error
    df = df.astype(object).where(df.notna(), np.nan)
    probability = float(pipe.predict_proba(df)[0, 1])
    return Prediction(
        will_change_job=probability >= THRESHOLD,
        probability=round(probability, 4),
        threshold=THRESHOLD,
        model_version=metadata["model_version"],
    )