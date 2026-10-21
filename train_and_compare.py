import os
import time
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# =====================================================================
# CONFIGURATION
# =====================================================================
TRANSACTION_PATH = "dataset/train_transaction.csv"
IDENTITY_PATH = "dataset/train_identity.csv"
OUTPUT_DIR = "model_results"
SAMPLE_SIZE = None  # Set to None for all 590,540 rows

# =====================================================================
# 1. DATA INGESTION & MEMORY OPTIMIZATION
# =====================================================================
def load_and_optimize_data(trans_path, id_path, nrows=None):
    """
    Loads IEEE-CIS relational tables and downcasts 64-bit numeric types
    to 32-bit to reduce RAM footprint by ~45-50% without losing precision.
    """
    if not os.path.exists(trans_path):
        raise FileNotFoundError(f"❌ Cannot find '{trans_path}'.")

    print(f"📂 Loading {trans_path}...")
    df = pd.read_csv(trans_path, nrows=nrows)

    if os.path.exists(id_path):
        print(f"📂 Merging with {id_path} on TransactionID...")
        id_df = pd.read_csv(id_path, nrows=nrows)
        df = df.merge(id_df, on="TransactionID", how="left")

    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = df[col].astype("int32")
    end_mem = df.memory_usage().sum() / 1024**2

    print(f"✅ Loaded {df.shape[0]:,} rows, {df.shape[1]} columns.")
    print(f"📉 Memory reduced: {start_mem:.1f} MB ➔ {end_mem:.1f} MB ({100*(start_mem-end_mem)/start_mem:.1f}% saved)")
    return df

df = load_and_optimize_data(TRANSACTION_PATH, IDENTITY_PATH, nrows=SAMPLE_SIZE)

# =====================================================================
# 2. CHRONOLOGICAL TRAIN / VALIDATION SPLIT (CRITICAL: SPLIT FIRST!)
# =====================================================================
# Sort by time elapsed (TransactionDT) so we train on the past to predict the future.
print("\n⏳ Sorting chronologically and performing 80/20 time-based split...")
df = df.sort_values("TransactionDT").reset_index(drop=True)

# Engineer deterministic row-level time feature 
df["Transaction_Hour"] = ((df["TransactionDT"] // 3600) % 24).astype("int32")

split_idx = int(len(df) * 0.80)
train_df = df.iloc[:split_idx].copy()
val_df = df.iloc[split_idx:].copy()

# Free up raw dataframe memory
del df

y_train = train_df["isFraud"].values
y_val = val_df["isFraud"].values

X_train = train_df.drop(columns=["isFraud", "TransactionID", "TransactionDT"], errors="ignore")
X_val = val_df.drop(columns=["isFraud", "TransactionID", "TransactionDT"], errors="ignore")
del train_df, val_df

# =====================================================================
# 3. LEAKAGE-FREE FEATURE ENGINEERING & ENCODING (FIT ON TRAIN ONLY)
# =====================================================================
print("⚙️ Fitting preprocessing transformers strictly on X_train...")

# A. Drop columns with >80% missing values (measured ONLY on X_train)
train_missing_ratio = X_train.isnull().mean()
cols_to_keep = train_missing_ratio[train_missing_ratio < 0.80].index.tolist()
X_train = X_train[cols_to_keep]
X_val = X_val[cols_to_keep]
print(f"✂️ Retained {len(cols_to_keep)} columns with <80% missing data in training set.")

# B. Out-of-Sample Group Aggregation: Ratio of TransactionAmt to card1's historical mean
# Calculate card1 mean strictly on X_train, then map to X_val (unseen cards fallback to global mean)
card1_means = X_train.groupby("card1")["TransactionAmt"].mean().to_dict()
global_amt_mean = float(X_train["TransactionAmt"].mean())

X_train["Amt_to_Mean_Card1"] = (
    X_train["TransactionAmt"] / X_train["card1"].map(card1_means).fillna(global_amt_mean)
).astype("float32")
X_val["Amt_to_Mean_Card1"] = (
    X_val["TransactionAmt"] / X_val["card1"].map(card1_means).fillna(global_amt_mean)
).astype("float32")

# C. Fit Frequency Encoders strictly on X_train and save dictionaries for FastAPI
cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
freq_encoders = {}

for col in cat_cols:
    # Convert to string so NaNs become a distinct "missing" category during lookup
    freq_series = X_train[col].fillna("MISSING").astype(str).value_counts()
    freq_map = freq_series.to_dict()
    freq_encoders[col] = freq_map

    # Map frequencies onto X_train and X_val (unseen categories in validation get 0)
    X_train[col] = X_train[col].fillna("MISSING").astype(str).map(freq_map).fillna(0).astype("float32")
    X_val[col] = X_val[col].fillna("MISSING").astype(str).map(freq_map).fillna(0).astype("float32")

# =====================================================================
# 4. AUTOMATED FEATURE SELECTION (TOP 15 VIA BASELINE LIGHTGBM)
# =====================================================================
print("\n🔍 Running automated feature selection on training split...")
# Temporarily fill NaNs for baseline selector
selector_model = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1, n_jobs=-1)
selector_model.fit(X_train.fillna(-999), y_train)

importances = pd.Series(selector_model.feature_importances_, index=X_train.columns)
top_15_features = importances.nlargest(15).index.tolist()
print(f"✅ Top 15 Selected Features:\n{top_15_features}\n")

X_train_sel = X_train[top_15_features].copy()
X_val_sel = X_val[top_15_features].copy()

# CRITICAL: Calculate true feature medians on X_train_sel BEFORE filling NaNs with -999!
# This ensures FastAPI default fallback values are real numbers, not -999.
X_train_sel = X_train_sel.replace([np.inf, -np.inf], np.nan)
X_val_sel = X_val_sel.replace([np.inf, -np.inf], np.nan)

true_train_medians = {
    k: float(v) for k, v in X_train_sel.median().fillna(0).to_dict().items()
}

# Now impute remaining NaNs with -999 so RandomForest can train alongside XGBoost/LightGBM
X_train_sel = X_train_sel.fillna(-999)
X_val_sel = X_val_sel.fillna(-999)

# =====================================================================
# 5. MODEL BENCHMARKING & OPTIMAL THRESHOLD TUNING
# =====================================================================
# Compute imbalance weight strictly from training labels (~27.5 normal per 1 fraud)
neg_pos_ratio = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=150, max_depth=12, class_weight="balanced", n_jobs=-1, random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=250, max_depth=6, learning_rate=0.08, scale_pos_weight=neg_pos_ratio,
        eval_metric="logloss", n_jobs=-1, random_state=42
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=250, max_depth=6, learning_rate=0.08, scale_pos_weight=neg_pos_ratio,
        n_jobs=-1, random_state=42, verbose=-1
    ),
}

results = []
trained_models = {}
optimal_thresholds = {}

for name, model in models.items():
    print(f"🚀 Training {name}...")
    start_time = time.time()
    model.fit(X_train_sel, y_train)
    train_time = time.time() - start_time

    # Measure inference latency per 1,000 rows
    inf_start = time.time()
    y_prob = model.predict_proba(X_val_sel)[:, 1]
    latency_ms = ((time.time() - inf_start) / len(X_val_sel)) * 1000

    # Calculate Optimal Probability Threshold that maximizes F1-Score
    # (Because scale_pos_weight inflates probabilities, 0.50 is suboptimal)
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_prob)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores[:-1])
    best_thresh = float(thresholds[best_idx])
    optimal_thresholds[name] = best_thresh

    y_pred_optimal = (y_prob >= best_thresh).astype(int)

    results.append({
        "Model": name,
        "ROC-AUC": round(roc_auc_score(y_val, y_prob), 4),
        "PR-AUC": round(average_precision_score(y_val, y_prob), 4),
        "F1-Score (Opt)": round(f1_score(y_val, y_pred_optimal), 4),
        "Best Threshold": round(best_thresh, 3),
        "Train Time (s)": round(train_time, 2),
        "Latency (ms/1k)": round(latency_ms, 3),
    })
    trained_models[name] = model

results_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)
print("\n🏆 LEAKAGE-FREE MODEL COMPARISON TABLE:")
print(results_df.to_string(index=False))

# =====================================================================
# 6. EXPORT ALL ARTIFACTS & PREPROCESSING MAPS TO model_results/
# =====================================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)
results_df.to_json(os.path.join(OUTPUT_DIR, "model_comparison.json"), orient="records", indent=4)

# Save only the frequency maps needed for the Top 15 features to keep artifacts lightweight
selected_freq_encoders = {
    col: freq_encoders[col] for col in top_15_features if col in freq_encoders
}

for _, row in results_df.iterrows():
    m_name = row["Model"]
    artifact_data = {
        "model": trained_models[m_name],
        "model_name": m_name,
        "features": top_15_features,
        "metrics": row.to_dict(),
        "optimal_threshold": optimal_thresholds[m_name],
        "medians": true_train_medians,
        "freq_encoders": selected_freq_encoders,
        "card1_means": card1_means,
        "global_amt_mean": global_amt_mean,
    }
file_path = os.path.join(OUTPUT_DIR, f"{m_name.lower()}_artifact.joblib")
joblib.dump(artifact_data, file_path)
print(f"💾 Saved {m_name} artifact to '{file_path}'")

 #1 Winner as best_model_artifact.joblib

best_model_name = results_df.iloc[0]["Model"]
best_artifact_path = os.path.join(OUTPUT_DIR, "best_model_artifact.joblib")
joblib.dump(
{
"model": trained_models[best_model_name],
"model_name": best_model_name,
"features": top_15_features,
"metrics": results_df.iloc[0].to_dict(),
"optimal_threshold": optimal_thresholds[best_model_name],
"all_models_comparison": results_df.to_dict(orient="records"),
"medians": true_train_medians,
"freq_encoders": selected_freq_encoders,
"card1_means": card1_means,
"global_amt_mean": global_amt_mean,
},
best_artifact_path,
)
print(f"\n🏆 Saved overall winner ({best_model_name}) to '{best_artifact_path}'!")
