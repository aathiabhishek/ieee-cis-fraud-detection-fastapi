import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, Literal, Union

app = FastAPI(
    title="IEEE-CIS Fraud Detection API (Production Pipeline)",
    description="Leakage-free E-Commerce Fraud Detection API serving XGBoost, LightGBM, and RandomForest",
    version="2.0.0",
)

MODEL_DIR = "model_results"
loaded_models = {}
comparison_data = []

# Load all 3 model artifacts at startup
for name in ["xgboost", "lightgbm", "randomforest"]:
    path = os.path.join(MODEL_DIR, f"{name}_artifact.joblib")
    if os.path.exists(path):
        loaded_models[name] = joblib.load(path)

best_path = os.path.join(MODEL_DIR, "best_model_artifact.joblib")
if os.path.exists(best_path):
    best_artifact = joblib.load(best_path)
    comparison_data = best_artifact.get("all_models_comparison", [])


class TransactionInput(BaseModel):
    """
    Pydantic schema validating incoming transaction requests.
    Supports human-readable email domains (e.g., 'gmail.com') OR pre-encoded floats.
    """
    TransactionAmt: float = Field(..., gt=0, examples=[450.00], description="Transaction amount in USD")
    card1: Optional[float] = Field(None, examples=[9500.0])
    card2: Optional[float] = Field(None, examples=[555.0])
    card3: Optional[float] = Field(None, examples=[185.0])
    card5: Optional[float] = Field(None, examples=[138.0])
    addr1: Optional[float] = Field(None, examples=[-999.0])
    C1: Optional[float] = Field(None, examples=[145.0])
    C6: Optional[float] = Field(None, examples=[80.0])
    C11: Optional[float] = Field(None, examples=[110.0])
    C13: Optional[float] = Field(None, examples=[0.0])
    C14: Optional[float] = Field(None, examples=[0.0])
    D2: Optional[float] = Field(None, examples=[0.0])
    D15: Optional[float] = Field(None, examples=[0.0])
    P_emaildomain: Optional[Union[str, float]] = Field("gmail.com", examples=["gmail.com"])
    R_emaildomain: Optional[Union[str, float]] = Field("gmail.com", examples=["gmail.com"])
    Transaction_Hour: Optional[int] = Field(14, ge=0, le=23, examples=[3])


@app.get("/")
def home():
    return {
        "message": "IEEE-CIS Fraud Detection API v2.0 is running",
        "available_models": list(loaded_models.keys()),
        "docs_url": "/docs",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "models_loaded": len(loaded_models)}


@app.get("/model-info")
def model_info():
    if not loaded_models:
        raise HTTPException(status_code=500, detail="Models not found. Run train_and_compare.py first.")

    primary = loaded_models.get("xgboost") or list(loaded_models.values())[0]
    return {
        "winning_model": primary["model_name"],
        "optimal_f1_threshold": round(primary.get("optimal_threshold", 0.5), 4),
        "winning_metrics": primary["metrics"],
        "top_15_features": primary["features"],
        "all_models_leaderboard": comparison_data,
    }


@app.post("/predict")
def predict(
    transaction: TransactionInput,
    model_choice: Literal["xgboost", "lightgbm", "randomforest"] = Query(
        "xgboost", description="Choose which model to run inference with"
    ),
):
    if model_choice not in loaded_models:
        raise HTTPException(status_code=404, detail=f"Model '{model_choice}' not found.")

    artifact = loaded_models[model_choice]
    model = artifact["model"]
    features = artifact["features"]
    medians = artifact["medians"]
    freq_encoders = artifact.get("freq_encoders", {})
    card1_means = artifact.get("card1_means", {})
    global_amt_mean = artifact.get("global_amt_mean", 135.0)
    opt_thresh = artifact.get("optimal_threshold", 0.50)

    input_dict = transaction.model_dump()

    # 1. Compute dynamic engineered feature (Amt_to_Mean_Card1) if required by model
    card1_val = input_dict.get("card1") or medians.get("card1", 10000)
    card_mean = card1_means.get(card1_val, global_amt_mean)
    input_dict["Amt_to_Mean_Card1"] = input_dict["TransactionAmt"] / max(card_mean, 1.0)

    # 2. Build ordered feature vector with frequency encoding & true median fallbacks
    row_data = {}
    for feat in features:
        val = input_dict.get(feat)

        # If categorical string (e.g. "gmail.com"), transform using saved training frequency map
        if feat in freq_encoders and isinstance(val, str):
            val = float(freq_encoders[feat].get(val, 0.0))

        # Fallback to true training median if None
        row_data[feat] = float(val) if val is not None else float(medians.get(feat, -999.0))

    X_input = pd.DataFrame([row_data])[features]

    # 3. Predict fraud probability
    fraud_prob = float(model.predict_proba(X_input)[0, 1])

    # 4. Tiered Risk Decisioning using the model's tuned F1 threshold
    block_threshold = min(max(opt_thresh + 0.15, 0.80), 0.95)
    if fraud_prob >= block_threshold:
        decision = "BLOCK"
    elif fraud_prob >= opt_thresh:
        decision = "MANUAL_REVIEW"
    else:
        decision = "APPROVE"

    return {
        "model_used": artifact["model_name"],
        "TransactionAmt": transaction.TransactionAmt,
        "fraud_probability": round(fraud_prob, 4),
        "model_optimal_threshold": round(opt_thresh, 4),
        "decision": decision,
    }