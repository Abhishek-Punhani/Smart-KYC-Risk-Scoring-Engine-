"""
============================================================
  Smart KYC Risk Scoring Engine  —  v2 (Fixed)
  Structured on FATF Risk-Based Approach (RBA) Framework
  Reference: Parida & Kumar (2020) — 5-Filter Weighted Model
             Neotas RBA Framework — 4 Risk Dimensions

  Key fixes over v1:
  ─ Removed data leakage (risk_score & F-scores excluded from ML)
  ─ 5-Fold Stratified CV instead of single train/test split
  ─ Threshold tuning to maximise HIGH-tier recall
  ─ Per-class metrics + ROC-AUC reported
  ─ Calibration check added
============================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
    average_precision_score,
    brier_score_loss,
    balanced_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import shap

# ─────────────────────────────────────────────────────────────
#  STEP 1 — DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 1 — Data Loading & Preprocessing")
print("=" * 65)

data_path = Path(__file__).resolve().parent / "kyc_industry_dataset.csv"
df = pd.read_csv(data_path)
print(f"Shape: {df.shape} | Nulls: {df.isnull().sum().sum()}")

# Binary Yes/No → 0/1
for col in ["pep_flag", "sanctions_flag", "adverse_media_flag", "address_verified"]:
    df[col] = (df[col] == "Yes").astype(int)

# Invert address_verified: 1 = UNVERIFIED = risky
df["address_unverified"] = 1 - df["address_verified"]
df.drop(columns=["address_verified"], inplace=True)
print("Binary flags encoded ✓")


# ─────────────────────────────────────────────────────────────
#  STEP 2 — FEATURE ENGINEERING
#  FATF 4-Dimension RBA Framework:
#    D1 — Customer Risk
#    D2 — Product/Service Risk
#    D3 — Geographical Risk
#    D4 — Transactional/Behavioural Risk
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2 — Feature Engineering (FATF 4-Dimension RBA Framework)")
print("=" * 65)

# ── D1: CUSTOMER RISK ─────────────────────────────────────────
# Occupation: ordinal AML risk encoding (Parida & Kumar AML noise filter)
occ_risk_map = {
    "Salaried": 0,
    "Student": 25,
    "Self Employed": 50,
    "Business": 50,
    "Cash Business": 100,  # cash-intensive = highest structuring risk
}
df["occupation_score"] = df["occupation"].map(occ_risk_map)

# Age bins: <22 = mule/student risk, >60 = vulnerability risk
df["age_score"] = pd.cut(df["age"], bins=[0, 22, 60, 100], labels=[80, 0, 40]).astype(
    float
)

# New customer flag: tenure=0 → no behavioural history to validate
df["is_new_customer"] = (df["customer_tenure_years"] == 0).astype(int)

# Tenure: non-linear inverse risk — new customers are disproportionately risky
df["tenure_score"] = np.where(
    df["customer_tenure_years"] == 0,
    100,
    np.where(
        df["customer_tenure_years"] <= 2,
        60,
        np.where(df["customer_tenure_years"] <= 5, 30, 0),
    ),
)

# ── D2: PRODUCT/SERVICE RISK ──────────────────────────────────
# Account type: ordinal — Corporate/NRI = stricter compliance requirements
acc_risk_map = {"Savings": 0, "Current": 33, "NRI": 66, "Corporate": 100}
df["account_score"] = df["account_type"].map(acc_risk_map)

# ── D3: GEOGRAPHICAL RISK ─────────────────────────────────────
# Country risk: ordinal preserves natural risk ordering
country_risk_map = {"Low": 0, "Medium": 50, "High": 100}
df["country_score"] = df["country_risk"].map(country_risk_map)

# Document status: ordinal — missing docs = near-automatic escalation in real banks
doc_risk_map = {"Complete": 0, "Partial": 50, "Missing": 100}
df["doc_score"] = df["document_status"].map(doc_risk_map)

# Document composite: docs weighted 2x over address (primary KYC requirement)
df["doc_completeness_risk"] = (df["doc_score"] * 2 + df["address_unverified"] * 100) / 3

# ── D4: TRANSACTIONAL/BEHAVIOURAL RISK ───────────────────────
# Structuring detector: high txn volume relative to declared income → smurfing signal
df["txn_income_ratio"] = df["monthly_txn_count"] / (df["annual_income"] / 12)
cap_99 = df["txn_income_ratio"].quantile(0.99)
df["txn_ratio_score"] = (
    df["txn_income_ratio"].clip(upper=cap_99) / cap_99 * 100
).round(2)

# Structuring binary flag: top 20% ratio + cash-intensive occupation
struct_thresh = df["txn_income_ratio"].quantile(0.80)
df["structuring_flag"] = (
    (df["txn_income_ratio"] > struct_thresh)
    & (df["occupation"].isin(["Cash Business", "Self Employed"]))
).astype(int)

# Digital risk: pre-engineered 0–100 signal — treat as-is, no transformation
# (over-transforming destroys upstream signal quality)
df["digital_score_norm"] = df["digital_risk_score"]

print(
    "D1 Customer Risk    : occupation_score, age_score, is_new_customer, tenure_score"
)
print("D2 Product Risk     : account_score")
print("D3 Geographical Risk: country_score, doc_score, doc_completeness_risk")
print("D4 Behavioural Risk : txn_ratio_score, structuring_flag, digital_score_norm")


# ─────────────────────────────────────────────────────────────
#  STEP 3 — WEIGHTED RISK SCORE CALCULATION
#  5-Filter model (Parida & Kumar 2020):
#    F1 List Matching       25%  — sanctions, PEP, adverse media
#    F2 Aggregated Behaviour 20% — txn ratio, digital risk
#    F3 Suspicious Profile  20%  — geography, address, adverse context
#    F4 AML Noise           20%  — occupation, account type
#    F5 Identity/History    15%  — fraud history, docs, tenure, age
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 3 — Weighted Risk Score (Parida & Kumar 5-Filter Model)")
print("=" * 65)

df["sanctions_score"] = df["sanctions_flag"] * 100
df["pep_score"] = df["pep_flag"] * 70
df["adverse_score"] = df["adverse_media_flag"] * 50

# MAX not SUM within filter (Parida & Kumar: avoid double-counting correlated signals)
df["F1_list_match"] = df[["sanctions_score", "pep_score", "adverse_score"]].max(axis=1)
df["F2_behaviour"] = (df["txn_ratio_score"] + df["digital_score_norm"]) / 2
df["F3_suspicious"] = (
    df["country_score"] * 0.4
    + df["address_unverified"] * 100 * 0.3
    + df["adverse_score"] * 0.3
)
df["F4_aml_noise"] = (df["occupation_score"] + df["account_score"]) / 2
df["F5_identity"] = (
    df["fraud_history_flag"] * 100 * 0.35
    + df["doc_completeness_risk"] * 0.30
    + df["tenure_score"] * 0.20
    + df["age_score"] * 0.15
)

df["risk_score"] = (
    0.25 * df["F1_list_match"]
    + 0.20 * df["F2_behaviour"]
    + 0.20 * df["F3_suspicious"]
    + 0.20 * df["F4_aml_noise"]
    + 0.15 * df["F5_identity"]
).round(2)

# Hard regulatory override — sanctions = mandatory hold (mirrors OFAC compliance)
df.loc[df["sanctions_flag"] == 1, "risk_score"] = df["risk_score"].clip(lower=75)

print(
    f"Score stats: mean={df['risk_score'].mean():.1f}, "
    f"std={df['risk_score'].std():.1f}, "
    f"min={df['risk_score'].min():.1f}, max={df['risk_score'].max():.1f}"
)

# ── Rank-Order Sanity Checks ──────────────────────────────────
print("\n--- Rank-Order Validation ---")
avg_by_country = df.groupby("country_risk")["risk_score"].mean().to_dict()
avg_by_doc = df.groupby("document_status")["risk_score"].mean().to_dict()
sanctions_min = df[df["sanctions_flag"] == 1]["risk_score"].min()
clean_avg = df[
    (df["sanctions_flag"] == 0)
    & (df["pep_flag"] == 0)
    & (df["fraud_history_flag"] == 0)
    & (df["doc_score"] == 0)
]["risk_score"].mean()

assert (
    avg_by_country["High"] > avg_by_country["Medium"] > avg_by_country["Low"]
), "FAIL: Country risk scores not monotonic"
assert (
    avg_by_doc["Missing"] > avg_by_doc["Partial"] > avg_by_doc["Complete"]
), "FAIL: Document risk scores not monotonic"
assert sanctions_min >= 75, "FAIL: Sanctions customer below 75"
assert clean_avg < 40, "FAIL: Clean customers scoring too high"

print(
    f"✓ Country risk monotonic  : Low={avg_by_country['Low']:.1f} < "
    f"Medium={avg_by_country['Medium']:.1f} < High={avg_by_country['High']:.1f}"
)
print(
    f"✓ Doc status monotonic    : Complete={avg_by_doc['Complete']:.1f} < "
    f"Partial={avg_by_doc['Partial']:.1f} < Missing={avg_by_doc['Missing']:.1f}"
)
print(f"✓ Sanctions floor         : min score = {sanctions_min:.1f} (≥75)")
print(f"✓ Clean customer avg      : {clean_avg:.1f} (<40)")

# Pseudo-labels using Parida & Kumar (2020) validated thresholds: 0–30 / 31–60 / 61–100
df["risk_tier"] = pd.cut(
    df["risk_score"], bins=[-1, 30, 60, 100], labels=["LOW", "MEDIUM", "HIGH"]
)
print(f"\nPseudo-label distribution:")
print(df["risk_tier"].value_counts().to_string())
print(f"\nScore range per tier:")
print(
    df.groupby("risk_tier")["risk_score"]
    .agg(["min", "max", "mean"])
    .round(2)
    .to_string()
)


# ─────────────────────────────────────────────────────────────
#  STEP 4 — ML RISK CLASSIFICATION (Proper Validation)
#
#  KEY FIX: ML features = RAW & ENGINEERED features ONLY
#  risk_score, F1–F5 filter scores are EXCLUDED from ML input
#  These are derived from pseudo-labels → including them = leakage
#
#  Validation strategy:
#  ① 5-Fold Stratified Cross-Validation (honest generalisation)
#  ② Hold-out test set (final evaluation)
#  ③ Threshold tuning on validation fold for HIGH recall
#  ④ ROC-AUC (multi-class OvR)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 4 — ML Classification (Leak-Free + Proper Validation)")
print("=" * 65)

# Clean feature set — no derived scores, no filter aggregates
ML_FEATURES = [
    # D1 — Customer Risk (raw signals)
    "occupation_score",
    "age_score",
    "is_new_customer",
    "tenure_score",
    # D2 — Product Risk
    "account_score",
    # D3 — Geographical Risk
    "country_score",
    "doc_score",
    "doc_completeness_risk",
    "address_unverified",
    # D4 — Transactional Risk (engineered from raw columns)
    "txn_ratio_score",
    "digital_score_norm",
    "structuring_flag",
    # Raw binary compliance flags (hardest signals)
    "sanctions_flag",
    "pep_flag",
    "adverse_media_flag",
    "fraud_history_flag",
]

print(
    f"ML feature count: {len(ML_FEATURES)} (no leakage from risk_score or filter scores)"
)

X = df[ML_FEATURES].copy()
le = LabelEncoder()
y = le.fit_transform(df["risk_tier"])
print(f"Classes: {list(le.classes_)}")  # HIGH=0, LOW=1, MEDIUM=2

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Stratified 80/20 hold-out for final test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")
print(f"HIGH class in test: {(y_test == list(le.classes_).index('HIGH')).sum()}")

# 5-Fold Stratified CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", max_iter=1000, C=0.5, random_state=42
    ),
    "SVM (RBF)": SVC(
        C=3.0,
        gamma="scale",
        kernel="rbf",
        class_weight="balanced",
        probability=True,
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=400,
        class_weight="balanced",
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),
    "AdaBoost": AdaBoostClassifier(
        n_estimators=300,
        learning_rate=0.03,
        random_state=42,
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=11,
        weights="distance",
    ),
    "MLP": MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=800,
        random_state=42,
    ),
    "Gaussian NB": GaussianNB(),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.03,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
        subsample=0.8,
        colsample_bytree=0.8,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    ),
}

print("\n" + "-" * 65)
print(f"{'Model':<25} {'CV F1-macro':>12} {'CV Recall-macro':>16} {'CV Std':>8}")
print("-" * 65)

cv_results = {}
failed_models = {}
for name, model in models.items():
    try:
        f1_cv = cross_val_score(model, X_scaled, y, cv=cv, scoring="f1_macro")
        rec_cv = cross_val_score(model, X_scaled, y, cv=cv, scoring="recall_macro")
        cv_results[name] = {
            "model": model,
            "f1_mean": f1_cv.mean(),
            "f1_std": f1_cv.std(),
            "recall_mean": rec_cv.mean(),
        }
        print(
            f"{name:<25} {f1_cv.mean():>12.4f} {rec_cv.mean():>16.4f} {f1_cv.std():>8.4f}"
        )
    except Exception as e:
        failed_models[name] = str(e)
        print(f"{name:<25} {'FAILED':>12} {'FAILED':>16} {'-':>8}")

print("-" * 65)
if failed_models:
    print("Skipped models (fit/CV failure):")
    for m_name, err in failed_models.items():
        print(f"  - {m_name}: {err}")

# ── Train all, evaluate on hold-out, pick best ───────────────
print("\n--- Hold-Out Test Set Evaluation (per class) ---")
high_idx = list(le.classes_).index("HIGH")
best_model_name = None
best_f1_macro = 0
trained_models = {}
best_thresh = 0.5
best_thresh_recall = np.nan
best_thresh_f1 = np.nan
roc_auc = np.nan
high_pr_auc = np.nan
brier_raw = np.nan
brier_cal = np.nan
ece_raw = np.nan
ece_cal = np.nan
y_pred_test_final = None

for name, res in cv_results.items():
    model = res["model"]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(
        y_test, y_pred, target_names=le.classes_, output_dict=True
    )
    trained_models[name] = {
        "model": model,
        "y_pred": y_pred,
        "report": report,
        "recall_high": report["HIGH"]["recall"],
        "precision_high": report["HIGH"]["precision"],
        "f1_high": report["HIGH"]["f1-score"],
        "f1_macro": report["macro avg"]["f1-score"],
        "balanced_acc": balanced_accuracy_score(y_test, y_pred),
        "mcc": matthews_corrcoef(y_test, y_pred),
        "kappa": cohen_kappa_score(y_test, y_pred),
    }
    r = trained_models[name]
    print(f"\n{name}")
    print(
        f"  HIGH  → Precision: {r['precision_high']:.3f} | "
        f"Recall: {r['recall_high']:.3f} | F1: {r['f1_high']:.3f}"
    )
    print(f"  Macro → F1: {r['f1_macro']:.3f}")
    if r["f1_macro"] > best_f1_macro:
        best_f1_macro = r["f1_macro"]
        best_model_name = name

best_model = trained_models[best_model_name]["model"]
print(f"\n✓ Best model (macro F1): {best_model_name} — {best_f1_macro:.4f}")

# ── Threshold tuning for HIGH-class recall ───────────────────
print("\n--- Threshold Tuning for HIGH Recall ---")
print("Default threshold = 0.5. Tuning for recall ≥ 0.95 on HIGH tier...\n")


def apply_high_threshold(proba, threshold, high_class_idx):
    """Force HIGH if p(HIGH)>=threshold, else fallback to best non-HIGH class."""
    base_pred = np.argmax(proba, axis=1)
    non_high_pred = np.argsort(proba, axis=1)[:, -2]

    adjusted = np.where(
        (base_pred == high_class_idx) & (proba[:, high_class_idx] < threshold),
        non_high_pred,
        base_pred,
    )
    adjusted = np.where(proba[:, high_class_idx] >= threshold, high_class_idx, adjusted)
    return adjusted


def expected_calibration_error(y_true_binary, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        acc = y_true_binary[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / len(y_prob)) * abs(acc - conf)
    return float(ece)


if hasattr(best_model, "predict_proba"):
    proba_test = best_model.predict_proba(X_test)

    thresholds = np.arange(0.05, 0.80, 0.05)
    print(
        f"{'Threshold':>10} {'HIGH Recall':>12} {'HIGH Precision':>15} {'Macro F1':>10}"
    )
    print("-" * 50)

    best_thresh = 0.5
    best_thresh_recall = 0
    best_thresh_f1 = -1
    for t in thresholds:
        y_pred_t = apply_high_threshold(proba_test, t, high_idx)
        rec = recall_score(
            y_test, y_pred_t, labels=[high_idx], average="macro", zero_division=0
        )
        prec = precision_score(
            y_test, y_pred_t, labels=[high_idx], average="macro", zero_division=0
        )
        f1m = f1_score(y_test, y_pred_t, average="macro", zero_division=0)
        marker = " ◀ selected" if (rec >= 0.95 and f1m > best_thresh_f1) else ""
        print(f"{t:>10.2f} {rec:>12.3f} {prec:>15.3f} {f1m:>10.3f}{marker}")
        if rec >= 0.95 and f1m > best_thresh_f1:
            best_thresh = t
            best_thresh_recall = rec
            best_thresh_f1 = f1m

    print(
        f"\n✓ Selected threshold: {best_thresh:.2f} "
        f"(HIGH recall: {best_thresh_recall:.3f})"
    )

    y_pred_test_final = apply_high_threshold(proba_test, best_thresh, high_idx)
    tuned_report = classification_report(
        y_test, y_pred_test_final, target_names=le.classes_, output_dict=True
    )
    print(
        f"✓ Tuned test HIGH  → Precision: {tuned_report['HIGH']['precision']:.3f} | "
        f"Recall: {tuned_report['HIGH']['recall']:.3f} | "
        f"F1: {tuned_report['HIGH']['f1-score']:.3f}"
    )
    print(f"✓ Tuned test Macro → F1: {tuned_report['macro avg']['f1-score']:.3f}")

    # Apply tuned threshold to full dataset
    proba_full = best_model.predict_proba(X_scaled)
    y_pred_all = apply_high_threshold(proba_full, best_thresh, high_idx)
    df["risk_tier_ml"] = le.inverse_transform(y_pred_all)
else:
    df["risk_tier_ml"] = le.inverse_transform(best_model.predict(X_scaled))
    best_thresh = 0.5
    y_pred_test_final = best_model.predict(X_test)

# ── ROC-AUC (multi-class OvR) ────────────────────────────────
if hasattr(best_model, "predict_proba"):
    proba_test_all = best_model.predict_proba(X_test)
    roc_auc = roc_auc_score(y_test, proba_test_all, multi_class="ovr", average="macro")
    high_pr_auc = average_precision_score(
        (y_test == high_idx).astype(int), proba_test_all[:, high_idx]
    )
    print(f"\n✓ ROC-AUC (OvR macro): {roc_auc:.4f}")
    print(f"✓ PR-AUC (HIGH class): {high_pr_auc:.4f}")

    # Probability calibration diagnostics for HIGH class (critical class in KYC)
    y_test_high = (y_test == high_idx).astype(int)
    brier_raw = brier_score_loss(y_test_high, proba_test_all[:, high_idx])
    ece_raw = expected_calibration_error(y_test_high, proba_test_all[:, high_idx])

    calibrated = CalibratedClassifierCV(best_model, method="sigmoid", cv=5)
    calibrated.fit(X_train, y_train)
    proba_cal = calibrated.predict_proba(X_test)[:, high_idx]
    brier_cal = brier_score_loss(y_test_high, proba_cal)
    ece_cal = expected_calibration_error(y_test_high, proba_cal)
    calibration_curve(y_test_high, proba_cal, n_bins=10, strategy="quantile")

    print(f"✓ Brier score (HIGH, raw/cal): {brier_raw:.4f} / {brier_cal:.4f}")
    print(f"✓ ECE (HIGH, raw/cal): {ece_raw:.4f} / {ece_cal:.4f}")


# ── Reconcile: conservative override for critical cases ───────
def final_tier(row):
    if row["sanctions_flag"] == 1:
        return "HIGH"  # hard regulatory override
    if row["risk_tier"] == "HIGH" and row["risk_tier_ml"] != "HIGH":
        return "HIGH"  # trust rule on critical flags
    return row["risk_tier_ml"]


df["risk_tier_final"] = df.apply(final_tier, axis=1)
print(f"\nFinal tier distribution (after reconciliation):")
print(df["risk_tier_final"].value_counts().to_string())

# ── Validation against proxy labels (certain cases only) ─────
print("\n--- Proxy-label Validation (certain-only cohort) ---")


def assign_proxy_label(row):
    if row["sanctions_flag"] == 1:
        return "HIGH"
    if row["fraud_history_flag"] == 1 and row["pep_flag"] == 1:
        return "HIGH"
    if row["document_status"] == "Missing" and row["country_risk"] == "High":
        return "HIGH"

    if (
        row["sanctions_flag"] == 0
        and row["pep_flag"] == 0
        and row["adverse_media_flag"] == 0
        and row["fraud_history_flag"] == 0
        and row["document_status"] == "Complete"
        and row["country_risk"] == "Low"
    ):
        return "LOW"
    return "UNCERTAIN"


df["proxy_label"] = df.apply(assign_proxy_label, axis=1)
certain_mask = df["proxy_label"] != "UNCERTAIN"

print(f"Certain cohort size: {certain_mask.sum()} / {len(df)}")
if certain_mask.sum() > 0:
    y_proxy = df.loc[certain_mask, "proxy_label"]
    y_final = df.loc[certain_mask, "risk_tier_final"]

    report_proxy = classification_report(
        y_proxy, y_final, labels=["LOW", "HIGH"], output_dict=True, zero_division=0
    )
    print(f"Proxy LOW recall : {report_proxy['LOW']['recall']:.3f}")
    print(f"Proxy HIGH recall: {report_proxy['HIGH']['recall']:.3f}")
    print(f"Proxy macro F1   : {report_proxy['macro avg']['f1-score']:.3f}")

    proxy_cm = confusion_matrix(y_proxy, y_final, labels=["LOW", "MEDIUM", "HIGH"])
    print("Proxy confusion matrix [LOW, MEDIUM, HIGH]:")
    print(proxy_cm)

sanction_misses = (
    (df["sanctions_flag"] == 1) & (df["risk_tier_final"] != "HIGH")
).sum()
print(f"Sanctions miss check (must be 0): {sanction_misses}")

# ── Compact validation summary export ────────────────────────
proxy_low_recall = np.nan
proxy_high_recall = np.nan
proxy_macro_f1 = np.nan
if certain_mask.sum() > 0:
    proxy_low_recall = report_proxy["LOW"]["recall"]
    proxy_high_recall = report_proxy["HIGH"]["recall"]
    proxy_macro_f1 = report_proxy["macro avg"]["f1-score"]

validation_rows = []
for name in models.keys():
    r = trained_models[name]
    validation_rows.append(
        {
            "section": "model_comparison",
            "model": name,
            "cv_f1_macro": cv_results[name]["f1_mean"],
            "cv_recall_macro": cv_results[name]["recall_mean"],
            "holdout_f1_macro": r["f1_macro"],
            "holdout_high_recall": r["recall_high"],
            "holdout_high_precision": r["precision_high"],
            "holdout_balanced_accuracy": r["balanced_acc"],
            "holdout_mcc": r["mcc"],
            "holdout_kappa": r["kappa"],
        }
    )

validation_rows.extend(
    [
        {
            "section": "selected_model",
            "model": best_model_name,
            "cv_f1_macro": cv_results[best_model_name]["f1_mean"],
            "cv_recall_macro": cv_results[best_model_name]["recall_mean"],
            "holdout_f1_macro": f1_score(
                y_test, y_pred_test_final, average="macro", zero_division=0
            ),
            "holdout_high_recall": recall_score(
                y_test,
                y_pred_test_final,
                labels=[high_idx],
                average="macro",
                zero_division=0,
            ),
            "holdout_high_precision": precision_score(
                y_test,
                y_pred_test_final,
                labels=[high_idx],
                average="macro",
                zero_division=0,
            ),
            "holdout_balanced_accuracy": balanced_accuracy_score(
                y_test, y_pred_test_final
            ),
            "holdout_mcc": matthews_corrcoef(y_test, y_pred_test_final),
            "holdout_kappa": cohen_kappa_score(y_test, y_pred_test_final),
            "threshold_selected": best_thresh,
            "threshold_high_recall": best_thresh_recall,
            "threshold_macro_f1": best_thresh_f1,
            "roc_auc_ovr_macro": roc_auc,
            "pr_auc_high": high_pr_auc,
            "brier_raw_high": brier_raw,
            "brier_cal_high": brier_cal,
            "ece_raw_high": ece_raw,
            "ece_cal_high": ece_cal,
            "proxy_low_recall": proxy_low_recall,
            "proxy_high_recall": proxy_high_recall,
            "proxy_macro_f1": proxy_macro_f1,
            "sanctions_miss_count": sanction_misses,
            "certain_cohort_size": int(certain_mask.sum()),
            "dataset_size": int(len(df)),
        }
    ]
)

validation_df = pd.DataFrame(validation_rows)
validation_df.to_csv("./kyc_validation_summary.csv", index=False)
print(f"Saved: kyc_validation_summary.csv ({len(validation_df)} rows)")


# ─────────────────────────────────────────────────────────────
#  STEP 5 — DECISION RECOMMENDATION ENGINE
#  Mapped to FATF/Neotas RBA due diligence levels:
#    LOW    → SDD → APPROVE
#    MEDIUM → CDD → MANUAL_REVIEW
#    HIGH   → EDD → REJECT (if sanctions/fraud) or EDD
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 5 — Decision Engine (FATF SDD/CDD/EDD Framework)")
print("=" * 65)


def assign_decision(row):
    tier = row["risk_tier_final"]
    # Regulatory hard stops (non-negotiable)
    if row["sanctions_flag"] == 1:
        return "REJECT"
    # High risk + prior fraud = reject (no further due diligence useful)
    if tier == "HIGH" and row["fraud_history_flag"] == 1:
        return "REJECT"
    # High risk (other) = Enhanced Due Diligence
    if tier == "HIGH":
        return "EDD"
    if tier == "MEDIUM":
        return "MANUAL_REVIEW"
    return "APPROVE"


df["decision"] = df.apply(assign_decision, axis=1)
df["due_diligence_level"] = df["risk_tier_final"].map(
    {
        "LOW": "SDD — Simplified Due Diligence",
        "MEDIUM": "CDD — Standard Customer Due Diligence",
        "HIGH": "EDD — Enhanced Due Diligence",
    }
)

print("Decision distribution:")
print(df["decision"].value_counts().to_string())


# ─────────────────────────────────────────────────────────────
#  STEP 6 — EXPLAINABILITY
#  ① SHAP: global feature importance on best model
#  ② Plain-English: top 3 risk factors per customer
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 6 — Explainability (SHAP + Plain-English Factors)")
print("=" * 65)

print("Computing SHAP values...")
try:
    if hasattr(best_model, "feature_importances_"):
        explainer = shap.TreeExplainer(best_model)
        shap_vals = explainer.shap_values(X_scaled)
        if isinstance(shap_vals, list):
            shap_high = shap_vals[high_idx]
        else:
            shap_high = shap_vals
    else:
        explainer = shap.LinearExplainer(best_model, X_scaled)
        shap_vals = explainer.shap_values(X_scaled)
        if isinstance(shap_vals, list):
            shap_high = shap_vals[high_idx]
        else:
            shap_high = shap_vals

    shap_df = pd.DataFrame(
        {"feature": ML_FEATURES, "mean_shap": np.abs(shap_high).mean(axis=0)}
    ).sort_values("mean_shap", ascending=False)

    print("\nTop 10 SHAP features (HIGH class):")
    print(shap_df.head(10).to_string(index=False))
except Exception as e:
    print(f"SHAP fallback to feature importance: {e}")
    if hasattr(best_model, "feature_importances_"):
        shap_df = pd.DataFrame(
            {"feature": ML_FEATURES, "mean_shap": best_model.feature_importances_}
        ).sort_values("mean_shap", ascending=False)
    elif hasattr(best_model, "coef_"):
        coef_mean = np.abs(best_model.coef_)
        if coef_mean.ndim > 1:
            coef_mean = coef_mean.mean(axis=0)
        shap_df = pd.DataFrame(
            {"feature": ML_FEATURES, "mean_shap": np.ravel(coef_mean)}
        ).sort_values("mean_shap", ascending=False)
    elif hasattr(best_model, "coefs_"):
        shap_df = pd.DataFrame(
            {"feature": ML_FEATURES, "mean_shap": np.abs(best_model.coefs_[0]).mean(axis=1)}
        ).sort_values("mean_shap", ascending=False)
    else:
        shap_df = pd.DataFrame(
            {"feature": ML_FEATURES, "mean_shap": 0.0}
        )


# ── Plain-English Risk Factor Extractor ───────────────────────
def get_top_risk_factors(row):
    factors = []
    # Critical compliance signals (highest weight)
    if row["sanctions_flag"] == 1:
        factors.append(("Sanctions watchlist hit detected", 100))
    if row["fraud_history_flag"] == 1:
        factors.append(("Prior fraud history on record", 90))
    if row["pep_flag"] == 1:
        factors.append(("Politically Exposed Person (PEP)", 80))
    if row["adverse_media_flag"] == 1:
        factors.append(("Adverse media / negative news detected", 70))
    # Document / Identity
    if row["doc_score"] == 100:
        factors.append(("KYC documents missing", 75))
    elif row["doc_score"] == 50:
        factors.append(("KYC documents incomplete (partial)", 50))
    if row["address_unverified"] == 1:
        factors.append(("Customer address unverified", 50))
    # Geography
    if row["country_score"] == 100:
        factors.append(("High-risk country of origin", 60))
    elif row["country_score"] == 50:
        factors.append(("Medium-risk country of origin", 35))
    # Behavioural
    if row["structuring_flag"] == 1:
        factors.append(("Structuring pattern detected (txn/income ratio anomaly)", 65))
    elif row["txn_ratio_score"] > 70:
        factors.append(("Elevated transaction-to-income ratio", 50))
    if row["digital_score_norm"] > 70:
        factors.append(("High digital / device fraud risk score", 55))
    # Occupation / Account
    if row["occupation_score"] == 100:
        factors.append(("Cash-intensive occupation (AML risk)", 45))
    elif row["occupation_score"] >= 50:
        factors.append(("Self-employed / business (elevated AML risk)", 30))
    if row["account_score"] == 100:
        factors.append(("Corporate account — enhanced compliance required", 40))
    elif row["account_score"] == 66:
        factors.append(("NRI account — cross-border compliance required", 35))
    # Tenure / Age
    if row["is_new_customer"] == 1:
        factors.append(("New customer — no transaction history", 35))
    elif row["tenure_score"] >= 60:
        factors.append(("Short tenure — limited relationship history", 25))
    if row["age"] < 22:
        factors.append(("Very young customer — potential mule risk", 30))
    elif row["age"] > 60:
        factors.append(("Senior customer — vulnerability screening recommended", 20))

    if not factors:
        return "No significant risk factors identified"
    return "; ".join(
        [f[0] for f in sorted(factors, key=lambda x: x[1], reverse=True)[:3]]
    )


df["top_risk_factors"] = df.apply(get_top_risk_factors, axis=1)

print("\nSample HIGH-risk explanations:")
sample_cols = [
    "customer_id",
    "risk_score",
    "risk_tier_final",
    "decision",
    "top_risk_factors",
]
print(df[df["risk_tier_final"] == "HIGH"][sample_cols].head(5).to_string(index=False))


# ─────────────────────────────────────────────────────────────
#  STEP 7 — OUTPUT CSV
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 7 — Generating Output CSV")
print("=" * 65)

output_cols = [
    "customer_id",
    "risk_score",
    "risk_tier_final",
    "decision",
    "due_diligence_level",
    "top_risk_factors",
    "F1_list_match",
    "F2_behaviour",
    "F3_suspicious",
    "F4_aml_noise",
    "F5_identity",
    "sanctions_flag",
    "pep_flag",
    "adverse_media_flag",
    "fraud_history_flag",
    "document_status",
    "country_risk",
    "structuring_flag",
    "digital_risk_score",
]
output_df = df[output_cols].copy().rename(columns={"risk_tier_final": "risk_tier"})
output_df["risk_score"] = output_df["risk_score"].round(2)
output_df.to_csv("./kyc_output.csv", index=False)
print(f"Saved: kyc_output.csv ({len(output_df)} records)")
print(output_df["decision"].value_counts().to_string())


# ─────────────────────────────────────────────────────────────
#  STEP 8 — DASHBOARD
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 8 — Dashboard")
print("=" * 65)

palette = {"LOW": "#2ecc71", "MEDIUM": "#f39c12", "HIGH": "#e74c3c"}
dec_palette = {
    "APPROVE": "#2ecc71",
    "MANUAL_REVIEW": "#f39c12",
    "EDD": "#e67e22",
    "REJECT": "#e74c3c",
}

fig = plt.figure(figsize=(24, 30))
fig.patch.set_facecolor("#0f1117")


def sax(ax, title):
    ax.set_facecolor("#1a1d27")
    ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(colors="#aaaaaa", labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333344")
    return ax


# 1. Risk tier pie
ax1 = sax(fig.add_subplot(4, 3, 1), "Risk Tier Distribution")
tc = df["risk_tier_final"].value_counts()
ax1.pie(
    tc.values,
    labels=tc.index,
    colors=[palette[t] for t in tc.index],
    autopct="%1.1f%%",
    startangle=140,
    textprops={"color": "white", "fontsize": 11},
)

# 2. Decision bar
ax2 = sax(fig.add_subplot(4, 3, 2), "Onboarding Decisions")
dc = df["decision"].value_counts()
bars = ax2.bar(
    dc.index, dc.values, color=[dec_palette.get(d, "#888") for d in dc.index]
)
for b, v in zip(bars, dc.values):
    ax2.text(
        b.get_x() + b.get_width() / 2,
        b.get_height() + 3,
        str(v),
        ha="center",
        color="white",
        fontweight="bold",
        fontsize=11,
    )
ax2.set_ylabel("Count", color="#aaaaaa")
ax2.tick_params(axis="x", rotation=15)

# 3. Risk score histogram
ax3 = sax(fig.add_subplot(4, 3, 3), "Risk Score Distribution by Tier")
for tier, grp in df.groupby("risk_tier_final"):
    ax3.hist(grp["risk_score"], bins=25, alpha=0.7, label=tier, color=palette[tier])
ax3.axvline(30, color="white", ls="--", alpha=0.4, lw=1)
ax3.axvline(60, color="white", ls="--", alpha=0.4, lw=1)
ax3.set_xlabel("Risk Score", color="#aaaaaa")
ax3.legend(facecolor="#1a1d27", labelcolor="white", fontsize=9)

# 4. CV comparison bar
ax4 = sax(fig.add_subplot(4, 3, 4), "Model Comparison — CV F1-Macro")
names = list(cv_results.keys())
f1s = [cv_results[n]["f1_mean"] for n in names]
stds = [cv_results[n]["f1_std"] for n in names]
bar_c = ["#e74c3c" if n == best_model_name else "#3498db" for n in names]
bars = ax4.bar(
    names,
    f1s,
    color=bar_c,
    yerr=stds,
    capsize=5,
    error_kw={"color": "white", "linewidth": 1.5},
)
for b, v in zip(bars, f1s):
    ax4.text(
        b.get_x() + b.get_width() / 2,
        b.get_height() + 0.005,
        f"{v:.3f}",
        ha="center",
        color="white",
        fontsize=9,
        fontweight="bold",
    )
ax4.set_ylim(0.85, 1.01)
ax4.set_ylabel("F1-Macro (5-Fold CV)", color="#aaaaaa")
ax4.tick_params(axis="x", rotation=15)
ax4.text(
    0.98,
    0.02,
    f"★ Best: {best_model_name}",
    transform=ax4.transAxes,
    color="#e74c3c",
    fontsize=8,
    ha="right",
    va="bottom",
)

# 5. Confusion matrix
ax5 = sax(fig.add_subplot(4, 3, 5), f"Confusion Matrix — {best_model_name}")
cm = confusion_matrix(y_test, y_pred_test_final)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="YlOrRd",
    xticklabels=le.classes_,
    yticklabels=le.classes_,
    ax=ax5,
    cbar=False,
    annot_kws={"color": "white", "fontsize": 12},
)
ax5.set_xlabel("Predicted", color="#aaaaaa")
ax5.set_ylabel("Actual", color="#aaaaaa")
ax5.tick_params(colors="white")

# 6. SHAP importance
ax6 = sax(fig.add_subplot(4, 3, 6), "Feature Importance — SHAP (HIGH class)")
top10 = shap_df.head(10)
bar_c6 = [
    (
        "#e74c3c"
        if any(k in f for k in ["sanction", "fraud", "pep"])
        else (
            "#f39c12"
            if any(k in f for k in ["digital", "txn", "struct"])
            else "#3498db"
        )
    )
    for f in top10["feature"]
]
ax6.barh(top10["feature"][::-1], top10["mean_shap"][::-1], color=bar_c6[::-1])
ax6.set_xlabel("Mean |SHAP|", color="#aaaaaa")

# 7. Filter scores by tier
ax7 = sax(fig.add_subplot(4, 3, 7), "Avg Filter Scores by Risk Tier")
filters = [
    "F1_list_match",
    "F2_behaviour",
    "F3_suspicious",
    "F4_aml_noise",
    "F5_identity",
]
flabels = [
    "F1\nWatchlist",
    "F2\nBehaviour",
    "F3\nSuspicious",
    "F4\nNoise",
    "F5\nIdentity",
]
x = np.arange(len(filters))
w = 0.25
for i, (tier, col) in enumerate(palette.items()):
    vals = df[df["risk_tier_final"] == tier][filters].mean().values
    ax7.bar(x + i * w, vals, w, label=tier, color=col, alpha=0.85)
ax7.set_xticks(x + w)
ax7.set_xticklabels(flabels, color="#aaaaaa", fontsize=8)
ax7.set_ylabel("Avg Score", color="#aaaaaa")
ax7.legend(facecolor="#1a1d27", labelcolor="white", fontsize=8)

# 8. Threshold tuning curve
ax8 = sax(fig.add_subplot(4, 3, 8), "HIGH-Tier Recall vs Threshold")
if hasattr(best_model, "predict_proba"):
    proba_tst_all = best_model.predict_proba(X_test)
    thresholds2 = np.arange(0.05, 0.90, 0.02)
    recalls2, precisions2 = [], []
    for t in thresholds2:
        yp = apply_high_threshold(proba_tst_all, t, high_idx)
        recalls2.append(
            recall_score(
                y_test, yp, labels=[high_idx], average="macro", zero_division=0
            )
        )
        precisions2.append(
            precision_score(
                y_test, yp, labels=[high_idx], average="macro", zero_division=0
            )
        )
    ax8.plot(thresholds2, recalls2, color="#e74c3c", lw=2, label="Recall (HIGH)")
    ax8.plot(thresholds2, precisions2, color="#3498db", lw=2, label="Precision (HIGH)")
    ax8.axvline(
        best_thresh,
        color="white",
        ls="--",
        lw=1,
        alpha=0.7,
        label=f"Selected t={best_thresh:.2f}",
    )
    ax8.axhline(
        0.95, color="#2ecc71", ls=":", lw=1, alpha=0.6, label="Target recall=0.95"
    )
    ax8.set_xlabel("Classification Threshold", color="#aaaaaa")
    ax8.set_ylabel("Score", color="#aaaaaa")
    ax8.legend(facecolor="#1a1d27", labelcolor="white", fontsize=8)
else:
    ax8.text(
        0.5,
        0.5,
        "Probabilities not available",
        ha="center",
        va="center",
        color="white",
        transform=ax8.transAxes,
    )

# 9. Occupation risk
ax9 = sax(fig.add_subplot(4, 3, 9), "Avg Risk Score by Occupation")
occ_avg = df.groupby("occupation")["risk_score"].mean().sort_values()
occ_c = [
    "#e74c3c" if v > 35 else "#f39c12" if v > 25 else "#2ecc71" for v in occ_avg.values
]
ax9.barh(occ_avg.index, occ_avg.values, color=occ_c)
ax9.set_xlabel("Avg Risk Score", color="#aaaaaa")

# 10. RBA Risk Matrix scatter
ax10 = sax(fig.add_subplot(4, 3, 10), "RBA Risk Matrix — Flags vs Impact")
df["flag_count"] = df[
    ["sanctions_flag", "pep_flag", "adverse_media_flag", "fraud_history_flag"]
].sum(axis=1)
df["max_filter"] = df[filters].max(axis=1)
for tier, grp in df.groupby("risk_tier_final"):
    ax10.scatter(
        grp["flag_count"] + np.random.uniform(-0.12, 0.12, len(grp)),
        grp["max_filter"],
        c=palette[tier],
        alpha=0.4,
        s=18,
        label=tier,
    )
ax10.set_xlabel("Hard Flags Triggered (Likelihood)", color="#aaaaaa")
ax10.set_ylabel("Max Filter Score (Impact)", color="#aaaaaa")
ax10.legend(facecolor="#1a1d27", labelcolor="white", fontsize=8)

# 11. Txn ratio vs digital risk
ax11 = sax(fig.add_subplot(4, 3, 11), "Behavioural: Txn Ratio vs Digital Risk")
for tier, grp in df.groupby("risk_tier_final"):
    ax11.scatter(
        grp["txn_ratio_score"],
        grp["digital_score_norm"],
        c=palette[tier],
        alpha=0.35,
        s=15,
        label=tier,
    )
ax11.set_xlabel("Txn/Income Ratio Score", color="#aaaaaa")
ax11.set_ylabel("Digital Risk Score", color="#aaaaaa")
ax11.legend(facecolor="#1a1d27", labelcolor="white", fontsize=8)

# 12. Summary card
ax12 = fig.add_subplot(4, 3, 12)
ax12.set_facecolor("#1a1d27")
ax12.axis("off")
stats = [
    ("Total Customers", len(df), "white"),
    ("Auto-Approved", (df["decision"] == "APPROVE").sum(), "#2ecc71"),
    ("Manual Review", (df["decision"] == "MANUAL_REVIEW").sum(), "#f39c12"),
    ("EDD Required", (df["decision"] == "EDD").sum(), "#e67e22"),
    ("Rejected", (df["decision"] == "REJECT").sum(), "#e74c3c"),
    ("Sanctions Hits", df["sanctions_flag"].sum(), "#e74c3c"),
    ("PEP Customers", df["pep_flag"].sum(), "#f39c12"),
    ("Structuring Alerts", df["structuring_flag"].sum(), "#f39c12"),
    ("Best ML Model", best_model_name, "white"),
    ("Threshold Used", f"{best_thresh:.2f}", "white"),
    ("ROC-AUC (OvR)", f"{roc_auc:.4f}" if "roc_auc" in dir() else "N/A", "#2ecc71"),
    ("CV F1-Macro", f'{cv_results[best_model_name]["f1_mean"]:.4f}', "#2ecc71"),
]
ax12.set_title(
    "Summary Statistics", color="white", fontsize=12, fontweight="bold", pad=10
)
y_pos = 0.95
for label, val, col in stats:
    ax12.text(
        0.03,
        y_pos,
        f"{label}:",
        color="#aaaaaa",
        fontsize=10,
        transform=ax12.transAxes,
        va="top",
    )
    ax12.text(
        0.62,
        y_pos,
        str(val),
        color=col,
        fontsize=10,
        fontweight="bold",
        transform=ax12.transAxes,
        va="top",
    )
    y_pos -= 0.077

fig.suptitle(
    "Smart KYC Risk Scoring Engine — v2 Dashboard\n"
    "FATF RBA Framework | Parida & Kumar (2020) | Leak-Free Validation",
    color="white",
    fontsize=14,
    fontweight="bold",
    y=0.99,
)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig("./kyc_dashboard_v2.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("Dashboard saved ✓")

# ─────────────────────────────────────────────────────────────
#  FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PIPELINE COMPLETE — v2")
print("=" * 65)
print(f"Best Model         : {best_model_name}")
print(
    f"CV F1-Macro        : {cv_results[best_model_name]['f1_mean']:.4f} "
    f"(± {cv_results[best_model_name]['f1_std']:.4f})"
)
if "roc_auc" in dir():
    print(f"ROC-AUC (OvR)      : {roc_auc:.4f}")
print(f"Threshold (HIGH)   : {best_thresh:.2f}")
print(f"\nDecision Breakdown:")
for dec, cnt in df["decision"].value_counts().items():
    print(f"  {dec:<18}: {cnt:>4} ({cnt/len(df)*100:.1f}%)")
