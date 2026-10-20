# 🛡️ Real-Time E-Commerce Fraud Detection Engine (IEEE-CIS)

> An end-to-end, leakage-free machine learning pipeline and low-latency **FastAPI** inference microservice that scores e-commerce transactions for fraud in real time.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-inference-009688)
![XGBoost](https://img.shields.io/badge/XGBoost-production%20model-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-benchmarked-brightgreen)
![scikit--learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-F7931E)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Key Results](#2-key-results)
3. [Dataset](#3-dataset)
4. [Methodology](#4-methodology)
   - [4.1 Leakage-free chronological split](#41-leakage-free-chronological-split)
   - [4.2 Feature selection (Top 15)](#42-feature-selection-top-15)
   - [4.3 Preprocessing and imputation](#43-preprocessing-and-imputation)
   - [4.4 Cost-sensitive training](#44-cost-sensitive-training)
   - [4.5 Precision-recall threshold tuning](#45-precision-recall-threshold-tuning)
5. [Top 15 Feature Dictionary](#5-top-15-feature-dictionary)
6. [Repository Structure](#6-repository-structure)
7. [Quickstart](#7-quickstart)
8. [API Reference](#8-api-reference)
9. [Decisioning Logic](#9-decisioning-logic)
10. [Model Artifact Contents](#10-model-artifact-contents)
11. [Limitations and Future Work](#11-limitations-and-future-work)
12. [Acknowledgements](#12-acknowledgements)

---

## 1. Overview

Card-not-present fraud is a needle-in-a-haystack problem: only about **3.5%** of transactions in this dataset are fraudulent, and fraud patterns drift over time. This project addresses both challenges:

- **Severe class imbalance** is handled with cost-sensitive learning plus explicit decision-threshold tuning on the precision-recall curve, rather than a naive 0.50 cutoff.
- **Temporal drift** is respected by using a strictly chronological train/test split, so the evaluation mimics deploying a model on future, unseen transactions.
- **Production readiness** is addressed by packaging the winning model, encoders, and imputation statistics into a single artifact served by a FastAPI microservice with a three-tier decision rule.

Three algorithms are benchmarked on identical data and features: **XGBoost**, **LightGBM**, and **Random Forest**.

---

## 2. Key Results

All models were trained on the first **80%** of transactions in chronological order (**472,432 rows**) using the **Top 15** selected features, and evaluated out-of-time on the final **20%** (**118,108 rows**).

| Model | ROC-AUC | PR-AUC | F1 (optimal threshold) | Best Threshold | Train Time (s) | Latency (ms / 1k rows) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (primary production)** | **0.8899** | **0.4697** | 0.4745 | `0.854` | **1.41** | _TODO_ |
| LightGBM | _TODO_ | _TODO_ | 0.4797 | `0.851` | _TODO_ | _TODO_ |
| Random Forest | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ |

> Full numbers are written to `model_results/model_comparison.csv` and `model_comparison.json` by the training script.

**Highlights**

- **XGBoost** is the production model: best ROC-AUC and PR-AUC, and the fastest to train.
- **Threshold tuning lifted XGBoost F1 by about 48%** (0.3215 → 0.4745) with no retraining.
- A compact 15-feature model keeps inference fast and the input contract small enough to be practical for an upstream checkout service.

---

## 3. Dataset

- **Source:** Vesta Corporation's [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/c/ieee-fraud-detection) on Kaggle.
- **Size:** 590,540 transactions, 434 features (after joining transaction and identity tables).
- **Target:** `isFraud` (binary), with a fraud rate of roughly **3.5%**.
- **Time axis:** `TransactionDT` (seconds offset from a reference timestamp), used to order transactions chronologically.

**Required files**

| File | Required | Notes |
| :--- | :---: | :--- |
| `train_transaction.csv` | Yes | Core transaction table. |
| `train_identity.csv` | Optional | Device and network identity table. |

> The dataset is not redistributed in this repository. Download it from Kaggle (account and competition rules acceptance required) and place the CSVs in the project root.

---

## 4. Methodology

### 4.1 Leakage-free chronological split

Random splits leak future information into training in fraud data, because fraudsters reuse cards, emails, and addresses across bursts of activity. To avoid this:

1. Transactions are sorted by `TransactionDT`.
2. The **first 80%** form the training pool; the **last 20%** are held out as the out-of-time test set.
3. All fitted objects (encoders, medians, selected features) are fit **only on the training portion**.
4. The decision threshold is tuned on a validation slice (`X_val`) taken from the training period, then applied unchanged to the held-out test set.

### 4.2 Feature selection (Top 15)

The raw feature space (434 columns) is pruned to **15 features** to cut noise, latency, and the API input surface. Pruning is based on tree-model importance, with attention to the following observations:

- **Redundancy among `V` features:** the anonymized `V` columns are heavily inter-correlated (pairwise correlation above `0.90` for many pairs), so correlated features split importance among themselves and each looks weak individually.
- **Data density and coverage:** many `V` columns are 40% to 75% missing, whereas the selected features (`card1`, `TransactionAmt`, `C1` to `C14`, `D2`, `D15`) have near-100% coverage.
- **Serving practicality:** the selected features are available at checkout time without joining to slow external sources.

### 4.3 Preprocessing and imputation

| Step | Treatment | Rationale |
| :--- | :--- | :--- |
| Email domains | Frequency encoding (`P_emaildomain`, `R_emaildomain`) | Handles high cardinality without exploding dimensions; the saved mapping lets the API accept human-readable strings. |
| Missing values | Fill with `-999` | `RandomForestClassifier` cannot handle `NaN` natively; `-999` pushes missingness to the far-left tail so trees can isolate it. |
| True medians | Computed and saved as `true_train_medians` **before** `-999` imputation | API fallback values for omitted fields reflect realistic customer behavior, not the `-999` sentinel. |
| `Transaction_Hour` | Derived from `TransactionDT` | Captures time-of-day fraud patterns (for example, late-night spikes). |

> **Order of operations matters:** medians must be computed before `-999` fill, otherwise they collapse to the sentinel value and corrupt API defaults.

### 4.4 Cost-sensitive training

| Parameter | XGBoost / LightGBM | Random Forest |
| :--- | :--- | :--- |
| Class weighting | `scale_pos_weight ≈ 27.5` | `class_weight = "balanced"` |
| `max_depth` | `6` | `12` |
| `learning_rate` | `0.08` | n/a |

- **Class weighting:** `scale_pos_weight` is derived directly from the training class ratio (about 96.5% normal to 3.5% fraud). It penalizes a missed fraud roughly **27.5x** more than a false alarm during gradient updates.
- **Depth choice:** sequential boosted trees use shallower depth (`6`) with shrinkage (`0.08`) to avoid fitting noise. Independent bagged trees in Random Forest need deeper splits (`12`) to capture feature interactions.

### 4.5 Precision-recall threshold tuning

Class weighting shifts predicted probabilities upward. Evaluating at a naive `0.50` cutoff therefore produces too many false positives (**F1 = 0.3215**).

Instead, the precision-recall curve is evaluated across all candidate thresholds on `X_val`, and the threshold maximizing F1 is selected:

$$
\text{F1} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

| Model | Naive 0.50 F1 | Optimal threshold | Tuned F1 |
| :--- | :---: | :---: | :---: |
| XGBoost | 0.3215 | `0.854` | **0.4745** (+48%) |
| LightGBM | _TODO_ | `0.851` | **0.4797** |

The chosen threshold is stored in the model artifact and exposed by the API so that decisioning always uses the value the model was tuned with.

---

## 5. Top 15 Feature Dictionary

> **Note:** IEEE-CIS features are anonymized by Vesta. The descriptions below are widely used community interpretations, not official definitions, and should be treated as informed approximations.

| Feature | Category | Description and fraud signal |
| :--- | :--- | :--- |
| `card1` | Card BIN | Primary payment card identifier / BIN sub-segment. |
| `card2` | Card issuer | Secondary issuing-institution identifier. |
| `card3` | Card country | Issuer country code (for example, `150.0` commonly read as domestic US, `185.0` as international). |
| `card5` | Card tier | Payment network category (consumer debit, credit, prepaid, and so on). |
| `TransactionAmt` | Financial | Transaction value in USD. |
| `addr1` | Geography | Billing region / zip zone (`-999` often indicates digital goods with no billing address). |
| `P_emaildomain` | Identity | Purchaser email domain, frequency-encoded. |
| `R_emaildomain` | Identity | Recipient email domain, frequency-encoded. |
| `C1` | Entity count | Count of payment cards associated with the billing address / email cluster. |
| `C6` | Entity count | Count of addresses or devices linked to the card. |
| `C11` | Entity count | Count of distinct email addresses linked to the identity cluster. |
| `C13` | Entity count | Count of historical profiles tied to the card (high `C1` with zero `C13` can indicate bot-driven attacks). |
| `C14` | Entity count | Count of unique cards / transactions matching the core identity. |
| `D2` | Velocity delta | Days since the last transaction on this card (`0` suggests rapid same-day velocity). |
| `D15` | Account age | Days since the address / email was first seen with this card (`0` means a brand-new link). |

Additionally, the API accepts `Transaction_Hour` (0 to 23) as a derived temporal signal.

---

## 6. Repository Structure

```text
FastAPI/
├── model_results/
│   ├── best_model_artifact.joblib      # Winning XGBoost model + encoders + medians + metrics
│   ├── xgboost_artifact.joblib         # Standalone XGBoost artifact
│   ├── lightgbm_artifact.joblib        # Standalone LightGBM artifact
│   ├── randomforest_artifact.joblib    # Standalone Random Forest artifact
│   ├── model_comparison.csv            # Benchmark leaderboard (CSV)
│   └── model_comparison.json           # Benchmark leaderboard (JSON)
├── train_and_compare.py                # Ingestion, leakage-free split, pruning, training, benchmarking
├── main.py                             # FastAPI inference microservice
├── requirements.txt                    # Pinned Python dependencies
└── README.md                           # Project documentation
```

---

## 7. Quickstart

### 7.1 Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install "fastapi[standard]" pandas numpy scikit-learn xgboost lightgbm joblib
```

> **Apple Silicon:** if OpenMP is missing, run `brew install libomp`.

### 7.2 Train and benchmark

Place `train_transaction.csv` (and optionally `train_identity.csv`) in the project root, then run:

```bash
python train_and_compare.py
```

This will:

1. Load and join the data, then sort chronologically.
2. Split 80/20 out-of-time.
3. Select the Top 15 features and fit encoders and medians.
4. Train XGBoost, LightGBM, and Random Forest.
5. Tune each model's decision threshold on the validation slice.
6. Write artifacts and the leaderboard to `model_results/`.

### 7.3 Start the API

```bash
python -m fastapi dev main.py
```

| Endpoint | URL |
| :--- | :--- |
| Interactive Swagger UI | <http://127.0.0.1:8000/docs> |
| Model metadata | <http://127.0.0.1:8000/model-info> |
| Health check | <http://127.0.0.1:8000/health> |

For production, run without the dev reloader, for example `fastapi run main.py` or `uvicorn main:app --workers 4`.

---

## 8. API Reference

### `POST /predict`

Scores a single transaction. Email fields accept human-readable strings, which are mapped through the saved frequency encoders. Omitted numeric fields are imputed with the true training medians.

**Request**

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionAmt": 450.00,
    "card1": 9500,
    "card2": 555,
    "card3": 185,
    "card5": 138,
    "addr1": -999,
    "C1": 145,
    "C6": 80,
    "C11": 110,
    "C13": 0,
    "C14": 0,
    "D2": 0,
    "D15": 0,
    "P_emaildomain": "gmail.com",
    "R_emaildomain": "gmail.com",
    "Transaction_Hour": 3
  }'
```

**Response (`200 OK`)**

```json
{
  "model_used": "XGBoost",
  "TransactionAmt": 450.0,
  "fraud_probability": 0.8612,
  "model_optimal_threshold": 0.854,
  "decision": "MANUAL_REVIEW"
}
```

**Response fields**

| Field | Type | Description |
| :--- | :--- | :--- |
| `model_used` | string | Name of the loaded model. |
| `TransactionAmt` | float | Echo of the submitted amount. |
| `fraud_probability` | float | Model-estimated probability of fraud (0 to 1). |
| `model_optimal_threshold` | float | F1-optimal threshold stored with the model. |
| `decision` | string | One of `APPROVE`, `MANUAL_REVIEW`, `BLOCK`. |

### `GET /model-info`

Returns metadata about the loaded artifact (model name, selected features, threshold, benchmark metrics).

### `GET /health`

Lightweight liveness check for load balancers and orchestrators.

---

## 9. Decisioning Logic

The API converts a probability into one of three business actions, anchored to the model's optimal threshold `t` (for XGBoost, `t = 0.854`):

| Decision | Condition | Intended action |
| :--- | :--- | :--- |
| `APPROVE` | _TODO: lower band, for example `p < a × t`_ | Let the transaction through. |
| `MANUAL_REVIEW` | _TODO: middle band_ | Route to an analyst or step-up authentication (3-D Secure, OTP). |
| `BLOCK` | _TODO: upper band, for example `p ≥ b`_ | Decline automatically. |

> **Action needed:** fill in the exact band boundaries from `main.py`. In the sample response above, `fraud_probability` (0.8612) sits just above the threshold (0.854) and returns `MANUAL_REVIEW`, so the `BLOCK` boundary evidently lies above the F1-optimal threshold. Documenting this explicitly avoids confusion.

**Why three tiers?** The F1-optimal threshold balances precision and recall, but real operations have asymmetric costs. A middle band lets the business absorb uncertain cases through review or step-up authentication instead of forcing a hard approve or decline.

---

## 10. Model Artifact Contents

Each `*_artifact.joblib` bundles everything needed for consistent inference:

| Key | Purpose |
| :--- | :--- |
| Trained model | The fitted XGBoost / LightGBM / Random Forest estimator. |
| Feature list | The ordered Top 15 features the model expects. |
| Frequency encoders | Email-domain mappings used at train time. |
| `true_train_medians` | Realistic default values for omitted fields. |
| Optimal threshold | F1-optimal decision boundary. |
| Metrics | Benchmark results for `/model-info`. |

Bundling these prevents train/serve skew: the API cannot accidentally use different encoders or defaults than the model was trained with.

---

## 11. Limitations and Future Work

**Limitations**

- Results come from a single chronological split; confidence intervals or rolling-window validation would give a better view of stability.
- Anonymized features limit interpretability, and descriptions are community-derived.
- F1-optimal thresholds do not reflect actual dollar costs of fraud versus false declines.
- The model is static; real fraud drifts and needs scheduled retraining.

**Roadmap**

- [ ] Cost-weighted threshold selection using transaction amount and chargeback fees.
- [ ] Rolling-window (walk-forward) validation.
- [ ] SHAP explanations returned alongside each decision for analyst review.
- [ ] Probability calibration (isotonic or Platt) for more trustworthy scores.
- [ ] Batch scoring endpoint (`POST /predict/batch`).
- [ ] Docker image and CI pipeline.
- [ ] Drift monitoring and automated retraining.

---

## 12. Acknowledgements

- [Vesta Corporation](https://www.vesta.io/) and the IEEE Computational Intelligence Society for the dataset.
- The Kaggle community for shared analysis of the anonymized feature space.
- The maintainers of FastAPI, XGBoost, LightGBM, and scikit-learn.
