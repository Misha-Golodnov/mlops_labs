"""
Веб-приложение FastAPI: предсказание charges по признакам, как в конвейере lab3.
"""

from __future__ import annotations

import json
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

CATEGORICAL_COLUMNS = ["sex", "smoker", "region"]


def lab3_root() -> Path:
    return Path(__file__).resolve().parents[1]


PROCESSED_DIR = lab3_root() / "data" / "processed"
MODEL_PATH = PROCESSED_DIR / "model.pkl"
IMPORTANT_FEATURES_PATH = PROCESSED_DIR / "important_features.txt"
METRICS_PATH = PROCESSED_DIR / "model_metrics.json"


def load_important_features() -> list[str]:
    text = IMPORTANT_FEATURES_PATH.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [line.strip() for line in text.splitlines() if line.strip()]


def encode_and_select_features(
    age: float,
    sex: str,
    bmi: float,
    children: int,
    smoker: str,
    region: str,
    important_features: list[str],
) -> pd.DataFrame:
    row = {
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
    }
    raw = pd.DataFrame([row])
    encoded = pd.get_dummies(raw, columns=CATEGORICAL_COLUMNS, dtype=int)
    vector = {
        col: int(encoded[col].iloc[0]) if col in encoded.columns else 0
        for col in important_features
    }
    return pd.DataFrame([vector], columns=important_features)


class InsuranceFeatures(BaseModel):
    age: float = Field(..., ge=0, le=120, description="Возраст")
    sex: Literal["female", "male"] = Field(..., description="Пол")
    bmi: float = Field(..., gt=0, le=100, description="Индекс массы тела")
    children: int = Field(0, ge=0, le=30, description="Число детей")
    smoker: Literal["yes", "no"] = Field(..., description="Курение")
    region: Literal["southwest", "southeast", "northwest", "northeast"] = Field(
        ...,
        description="Регион",
    )


class PredictionResponse(BaseModel):
    predicted_charges: float
    features_used: list[str]


_model: Any = None
_important_features: list[str] = []


def startup_load_model() -> None:
    global _model, _important_features
    if not MODEL_PATH.is_file():
        raise RuntimeError(
            f"Модель не найдена: {MODEL_PATH}. Выполните: python scripts/train_model.py"
        )
    _important_features = load_important_features()
    if not _important_features:
        raise RuntimeError(f"Пустой список признаков: {IMPORTANT_FEATURES_PATH}")
    with open(MODEL_PATH, "rb") as f:
        _model = pickle.load(f)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    startup_load_model()
    yield


app = FastAPI(
    title="Insurance charges predictor",
    description="Предсказание страховых выплат (lab3, GradientBoostingRegressor).",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    if not METRICS_PATH.is_file():
        raise HTTPException(status_code=404, detail="model_metrics.json не найден")
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@app.post("/predict", response_model=PredictionResponse)
def predict(body: InsuranceFeatures) -> PredictionResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    X = encode_and_select_features(
        age=body.age,
        sex=body.sex,
        bmi=body.bmi,
        children=body.children,
        smoker=body.smoker,
        region=body.region,
        important_features=_important_features,
    )
    pred = float(_model.predict(X)[0])
    return PredictionResponse(
        predicted_charges=round(pred, 2),
        features_used=list(X.columns),
    )


INDEX_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Insurance — предсказание выплат</title>
  <style>
    :root { font-family: system-ui, sans-serif; background: #0f1419; color: #e6edf3; }
    body { max-width: 32rem; margin: 2rem auto; padding: 0 1rem; }
    h1 { font-size: 1.25rem; font-weight: 600; }
    label { display: block; margin-top: 0.75rem; font-size: 0.875rem; color: #8b949e; }
    input, select { width: 100%; margin-top: 0.25rem; padding: 0.5rem; border-radius: 6px;
      border: 1px solid #30363d; background: #161b22; color: #e6edf3; box-sizing: border-box; }
    button { margin-top: 1rem; padding: 0.6rem 1rem; border: none; border-radius: 6px;
      background: #238636; color: #fff; font-weight: 600; cursor: pointer; width: 100%; }
    button:hover { background: #2ea043; }
    #out { margin-top: 1rem; padding: 1rem; border-radius: 8px; background: #161b22;
      border: 1px solid #30363d; min-height: 2rem; white-space: pre-wrap; }
    a { color: #58a6ff; }
  </style>
</head>
<body>
  <h1>Предсказание страховых выплат (lab3)</h1>
  <p>Модель: GradientBoostingRegressor. <a href="/docs">OpenAPI (/docs)</a></p>
  <form id="f">
    <label>Возраст <input name="age" type="number" step="1" value="35" required /></label>
    <label>Пол
      <select name="sex"><option>female</option><option>male</option></select>
    </label>
    <label>BMI <input name="bmi" type="number" step="0.01" value="27.5" required /></label>
    <label>Дети <input name="children" type="number" min="0" value="0" required /></label>
    <label>Курение
      <select name="smoker"><option>no</option><option>yes</option></select>
    </label>
    <label>Регион
      <select name="region">
        <option>southwest</option><option>southeast</option>
        <option>northwest</option><option>northeast</option>
      </select>
    </label>
    <button type="submit">Предсказать</button>
  </form>
  <div id="out"></div>
  <script>
    document.getElementById('f').onsubmit = async (e) => {
      e.preventDefault();
      const fd = new FormData(e.target);
      const body = {
        age: parseFloat(fd.get('age')),
        sex: fd.get('sex'),
        bmi: parseFloat(fd.get('bmi')),
        children: parseInt(fd.get('children'), 10),
        smoker: fd.get('smoker'),
        region: fd.get('region'),
      };
      const out = document.getElementById('out');
      out.textContent = '…';
      try {
        const r = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        const j = await r.json();
        if (!r.ok) { out.textContent = JSON.stringify(j, null, 2); return; }
        out.textContent = 'Прогноз charges: ' + j.predicted_charges + '\\nПризнаки: ' + j.features_used.join(', ');
      } catch (err) {
        out.textContent = String(err);
      }
    };
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML
