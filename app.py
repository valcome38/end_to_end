from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
import json
from prophet import Prophet
from catboost import CatBoostRegressor
from typing import List
from datetime import datetime

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello from Render!"}

request_count = 0

# Загрузка CatBoost модели и списка фичей
cat_model = CatBoostRegressor()
cat_model.load_model("models/catboost_residual_model.cbm")

with open("models/features.json") as f:
    FEATURES = json.load(f)

# Модель входных данных
class DataPoint(BaseModel):
    datetime: str
    zone_id: str

class PredictionRequest(BaseModel):
    data: List[DataPoint]

@app.get("/status")
def status():
    return {"status": "online"}

@app.get("/stats")
def stats():
    return {"total_requests": request_count}

@app.post("/predict")
def predict(request: PredictionRequest):
    global request_count
    request_count += 1

    # Преобразуем входные данные в DataFrame
    df = pd.DataFrame([d.dict() for d in request.data])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["ds"] = df["datetime"]
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    prophet_preds = []
    for zone in df['zone_id'].unique():
        df_zone = df[df['zone_id'] == zone].copy()
        with open(f"models/prophet_zones/prophet_zone_{zone}.pkl", "rb") as f:
            model = pickle.load(f)
        future = model.make_future_dataframe(periods=0, freq='H')
        forecast = model.predict(future)
        df_zone = df_zone.merge(forecast[['ds', 'yhat']], on='ds', how='left')
        prophet_preds.append(df_zone)

    df_all = pd.concat(prophet_preds, ignore_index=True)
    df_all['residual'] = 0  # временно, можно сюда вставить предыдущие остатки, если есть

    df_all['lag_1'] = df_all.groupby('zone_id')['residual'].shift(1)
    df_all['lag_24'] = df_all.groupby('zone_id')['residual'].shift(24)
    df_all['rolling_mean_24'] = (
        df_all.groupby('zone_id')['residual']
        .shift(1).rolling(24).mean().reset_index(0, drop=True)
    )
    df_model = df_all.dropna()

    X = df_model[FEATURES]
    residual_pred = cat_model.predict(X)
    df_model['final_forecast'] = df_model['yhat'] + residual_pred

    return df_model[['datetime', 'zone_id', 'final_forecast']].to_dict(orient="records")

