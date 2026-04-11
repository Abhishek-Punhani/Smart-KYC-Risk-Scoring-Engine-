from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVC


ML_FEATURES = [
    "occupation_score",
    "age_score",
    "is_new_customer",
    "tenure_score",
    "account_score",
    "country_score",
    "doc_score",
    "doc_completeness_risk",
    "address_unverified",
    "txn_ratio_score",
    "digital_score_norm",
    "structuring_flag",
    "sanctions_flag",
    "pep_flag",
    "adverse_media_flag",
    "fraud_history_flag",
]


def _expected_calibration_error(
    y_true_binary: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
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


def _apply_high_threshold(
    proba: np.ndarray, threshold: float, high_class_idx: int
) -> np.ndarray:
    base_pred = np.argmax(proba, axis=1)
    non_high_pred = np.argsort(proba, axis=1)[:, -2]
    adjusted = np.where(
        (base_pred == high_class_idx) & (proba[:, high_class_idx] < threshold),
        non_high_pred,
        base_pred,
    )
    adjusted = np.where(proba[:, high_class_idx] >= threshold, high_class_idx, adjusted)
    return adjusted


def _assign_proxy_label(row: pd.Series) -> str:
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


def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # Binary Yes/No -> 0/1
    for col in ["pep_flag", "sanctions_flag", "adverse_media_flag", "address_verified"]:
        if df[col].dtype == object:
            df[col] = (df[col] == "Yes").astype(int)

    df["address_unverified"] = 1 - df["address_verified"]
    df.drop(columns=["address_verified"], inplace=True)

    occ_risk_map = {
        "Salaried": 0,
        "Student": 25,
        "Self Employed": 50,
        "Business": 50,
        "Cash Business": 100,
    }
    df["occupation_score"] = df["occupation"].map(occ_risk_map)

    df["age_score"] = pd.cut(
        df["age"], bins=[0, 22, 60, 100], labels=[80, 0, 40]
    ).astype(float)
    df["is_new_customer"] = (df["customer_tenure_years"] == 0).astype(int)
    df["tenure_score"] = np.where(
        df["customer_tenure_years"] == 0,
        100,
        np.where(
            df["customer_tenure_years"] <= 2,
            60,
            np.where(df["customer_tenure_years"] <= 5, 30, 0),
        ),
    )

    df["account_score"] = df["account_type"].map(
        {"Savings": 0, "Current": 33, "NRI": 66, "Corporate": 100}
    )
    df["country_score"] = df["country_risk"].map({"Low": 0, "Medium": 50, "High": 100})
    df["doc_score"] = df["document_status"].map(
        {"Complete": 0, "Partial": 50, "Missing": 100}
    )
    df["doc_completeness_risk"] = (
        df["doc_score"] * 2 + df["address_unverified"] * 100
    ) / 3

    df["txn_income_ratio"] = df["monthly_txn_count"] / (df["annual_income"] / 12)
    cap_99 = df["txn_income_ratio"].quantile(0.99)
    df["txn_ratio_score"] = (
        df["txn_income_ratio"].clip(upper=cap_99) / cap_99 * 100
    ).round(2)
    struct_thresh = df["txn_income_ratio"].quantile(0.80)
    df["structuring_flag"] = (
        (df["txn_income_ratio"] > struct_thresh)
        & (df["occupation"].isin(["Cash Business", "Self Employed"]))
    ).astype(int)
    df["digital_score_norm"] = df["digital_risk_score"]

    return df


def _rule_score(df: pd.DataFrame) -> pd.DataFrame:
    df["sanctions_score"] = df["sanctions_flag"] * 100
    df["pep_score"] = df["pep_flag"] * 70
    df["adverse_score"] = df["adverse_media_flag"] * 50

    df["F1_list_match"] = df[["sanctions_score", "pep_score", "adverse_score"]].max(
        axis=1
    )
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

    df.loc[df["sanctions_flag"] == 1, "risk_score"] = df["risk_score"].clip(lower=75)
    df["risk_tier"] = pd.cut(
        df["risk_score"], bins=[-1, 30, 60, 100], labels=["LOW", "MEDIUM", "HIGH"]
    )
    return df


def _assign_decision(row: pd.Series) -> str:
    tier = row["risk_tier_final"]
    if row["sanctions_flag"] == 1:
        return "REJECT"
    if tier == "HIGH" and row["fraud_history_flag"] == 1:
        return "REJECT"
    if tier == "HIGH":
        return "EDD"
    if tier == "MEDIUM":
        return "MANUAL_REVIEW"
    return "APPROVE"


def _get_top_risk_factors(row: pd.Series) -> str:
    factors = []
    if row["sanctions_flag"] == 1:
        factors.append(("Sanctions watchlist hit detected", 100))
    if row["fraud_history_flag"] == 1:
        factors.append(("Prior fraud history on record", 90))
    if row["pep_flag"] == 1:
        factors.append(("Politically Exposed Person (PEP)", 80))
    if row["adverse_media_flag"] == 1:
        factors.append(("Adverse media / negative news detected", 70))
    if row["doc_score"] == 100:
        factors.append(("KYC documents missing", 75))
    elif row["doc_score"] == 50:
        factors.append(("KYC documents incomplete (partial)", 50))
    if row["address_unverified"] == 1:
        factors.append(("Customer address unverified", 50))
    if row["country_score"] == 100:
        factors.append(("High-risk country of origin", 60))
    if row["structuring_flag"] == 1:
        factors.append(("Structuring pattern detected (txn/income ratio anomaly)", 65))
    if row["digital_score_norm"] > 70:
        factors.append(("High digital / device fraud risk score", 55))

    if not factors:
        return "No significant risk factors identified"
    return "; ".join(
        [f[0] for f in sorted(factors, key=lambda x: x[1], reverse=True)[:3]]
    )


def _build_models() -> dict[str, Any]:
    return {
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
            n_estimators=300, learning_rate=0.03, random_state=42
        ),
        "KNN": KNeighborsClassifier(n_neighbors=11, weights="distance"),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=800,
            random_state=42,
        ),
        "Gaussian NB": GaussianNB(),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        ),
    }


def run_pipeline(
    dataset_path: Path, out_dir: Path, config: dict[str, Any] | None = None
) -> dict[str, Any]:
    config = config or {}
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path)
    df = _feature_engineering(df)
    df = _rule_score(df)

    X = df[ML_FEATURES].copy()
    le = LabelEncoder()
    y = le.fit_transform(df["risk_tier"])
    high_idx = list(le.classes_).index("HIGH")

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    test_size = float(config.get("test_size", 0.2))
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42, stratify=y
    )

    cv_folds = int(config.get("cv_folds", 5))
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    models = _build_models()
    cv_results: dict[str, dict[str, Any]] = {}
    failed_models: dict[str, str] = {}
    for name, model in models.items():
        try:
            f1_cv = cross_val_score(model, X_scaled, y, cv=cv, scoring="f1_macro")
            rec_cv = cross_val_score(model, X_scaled, y, cv=cv, scoring="recall_macro")
            cv_results[name] = {
                "model": model,
                "f1_mean": float(f1_cv.mean()),
                "f1_std": float(f1_cv.std()),
                "recall_mean": float(rec_cv.mean()),
            }
        except Exception as e:
            failed_models[name] = str(e)

    trained_models: dict[str, dict[str, Any]] = {}
    best_model_name = None
    best_f1_macro = -1.0

    for name, res in cv_results.items():
        model = res["model"]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(
            y_test, y_pred, target_names=le.classes_, output_dict=True
        )
        trained_models[name] = {
            "model": model,
            "report": report,
            "f1_macro": float(report["macro avg"]["f1-score"]),
            "precision_high": float(report["HIGH"]["precision"]),
            "recall_high": float(report["HIGH"]["recall"]),
            "balanced_acc": float(balanced_accuracy_score(y_test, y_pred)),
            "mcc": float(matthews_corrcoef(y_test, y_pred)),
            "kappa": float(cohen_kappa_score(y_test, y_pred)),
        }
        if trained_models[name]["f1_macro"] > best_f1_macro:
            best_f1_macro = trained_models[name]["f1_macro"]
            best_model_name = name

    if best_model_name is None:
        raise RuntimeError("No model completed successfully.")

    best_model = trained_models[best_model_name]["model"]
    best_thresh = 0.5
    best_thresh_recall = np.nan
    best_thresh_f1 = np.nan

    if hasattr(best_model, "predict_proba"):
        proba_test = best_model.predict_proba(X_test)
        threshold_target = float(config.get("threshold_target_recall", 0.95))

        best_candidate_f1 = -1.0
        for t in np.arange(0.05, 0.80, 0.05):
            y_pred_t = _apply_high_threshold(proba_test, float(t), high_idx)
            rec = recall_score(
                y_test, y_pred_t, labels=[high_idx], average="macro", zero_division=0
            )
            f1m = f1_score(y_test, y_pred_t, average="macro", zero_division=0)
            if rec >= threshold_target and f1m > best_candidate_f1:
                best_candidate_f1 = float(f1m)
                best_thresh = float(t)
                best_thresh_recall = float(rec)
                best_thresh_f1 = float(f1m)

        y_pred_test_final = _apply_high_threshold(proba_test, best_thresh, high_idx)
        proba_full = best_model.predict_proba(X_scaled)
        y_pred_all = _apply_high_threshold(proba_full, best_thresh, high_idx)
    else:
        y_pred_test_final = best_model.predict(X_test)
        y_pred_all = best_model.predict(X_scaled)

    df["risk_tier_ml"] = le.inverse_transform(y_pred_all)

    def final_tier(row: pd.Series) -> str:
        if row["sanctions_flag"] == 1:
            return "HIGH"
        if row["risk_tier"] == "HIGH" and row["risk_tier_ml"] != "HIGH":
            return "HIGH"
        return row["risk_tier_ml"]

    df["risk_tier_final"] = df.apply(final_tier, axis=1)
    df["decision"] = df.apply(_assign_decision, axis=1)
    df["top_risk_factors"] = df.apply(_get_top_risk_factors, axis=1)

    # Metrics + calibration
    roc_auc = np.nan
    high_pr_auc = np.nan
    brier_raw = np.nan
    brier_cal = np.nan
    ece_raw = np.nan
    ece_cal = np.nan

    if hasattr(best_model, "predict_proba"):
        proba_test_all = best_model.predict_proba(X_test)
        roc_auc = float(
            roc_auc_score(y_test, proba_test_all, multi_class="ovr", average="macro")
        )
        high_pr_auc = float(
            average_precision_score(
                (y_test == high_idx).astype(int), proba_test_all[:, high_idx]
            )
        )

        y_test_high = (y_test == high_idx).astype(int)
        brier_raw = float(brier_score_loss(y_test_high, proba_test_all[:, high_idx]))
        ece_raw = float(
            _expected_calibration_error(y_test_high, proba_test_all[:, high_idx])
        )

        calibrated = CalibratedClassifierCV(best_model, method="sigmoid", cv=5)
        calibrated.fit(X_train, y_train)
        proba_cal = calibrated.predict_proba(X_test)[:, high_idx]
        brier_cal = float(brier_score_loss(y_test_high, proba_cal))
        ece_cal = float(_expected_calibration_error(y_test_high, proba_cal))

    # Proxy validation
    df["proxy_label"] = df.apply(_assign_proxy_label, axis=1)
    certain_mask = df["proxy_label"] != "UNCERTAIN"
    proxy_low_recall = np.nan
    proxy_high_recall = np.nan
    proxy_macro_f1 = np.nan
    if certain_mask.sum() > 0:
        report_proxy = classification_report(
            df.loc[certain_mask, "proxy_label"],
            df.loc[certain_mask, "risk_tier_final"],
            labels=["LOW", "HIGH"],
            output_dict=True,
            zero_division=0,
        )
        proxy_low_recall = float(report_proxy["LOW"]["recall"])
        proxy_high_recall = float(report_proxy["HIGH"]["recall"])
        proxy_macro_f1 = float(report_proxy["macro avg"]["f1-score"])

    sanction_misses = int(
        ((df["sanctions_flag"] == 1) & (df["risk_tier_final"] != "HIGH")).sum()
    )

    # Artifacts
    output_cols = [
        "customer_id",
        "risk_score",
        "risk_tier_final",
        "decision",
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
    output_csv = out_dir / "kyc_output.csv"
    output_df.to_csv(output_csv, index=False)

    validation_rows = []
    for name in cv_results.keys():
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

    validation_rows.append(
        {
            "section": "selected_model",
            "model": best_model_name,
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
            "failed_models": (
                "; ".join([f"{k}: {v}" for k, v in failed_models.items()])
                if failed_models
                else ""
            ),
        }
    )

    validation_df = pd.DataFrame(validation_rows)
    validation_csv = out_dir / "kyc_validation_summary.csv"
    validation_df.to_csv(validation_csv, index=False)

    # Basic dashboard artifact
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    df["risk_tier_final"].value_counts().plot(
        kind="bar", ax=axes[0], title="Risk Tiers"
    )
    df["decision"].value_counts().plot(kind="bar", ax=axes[1], title="Decisions")
    axes[2].hist(df["risk_score"], bins=25)
    axes[2].set_title("Risk Score Distribution")
    dashboard_png = out_dir / "kyc_dashboard_v2.png"
    fig.tight_layout()
    fig.savefig(dashboard_png, dpi=120)
    plt.close(fig)

    return {
        "dataset_rows": int(len(df)),
        "best_model": best_model_name,
        "cv_f1_macro": float(cv_results[best_model_name]["f1_mean"]),
        "cv_f1_std": float(cv_results[best_model_name]["f1_std"]),
        "cv_recall_macro": float(cv_results[best_model_name]["recall_mean"]),
        "holdout_f1_macro": float(
            f1_score(y_test, y_pred_test_final, average="macro", zero_division=0)
        ),
        "holdout_high_recall": float(
            recall_score(
                y_test,
                y_pred_test_final,
                labels=[high_idx],
                average="macro",
                zero_division=0,
            )
        ),
        "holdout_high_precision": float(
            precision_score(
                y_test,
                y_pred_test_final,
                labels=[high_idx],
                average="macro",
                zero_division=0,
            )
        ),
        "threshold": float(best_thresh),
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
        "failed_models": failed_models,
        "artifacts": {
            "output_csv": output_csv.name,
            "validation_csv": validation_csv.name,
            "dashboard_png": dashboard_png.name,
        },
    }


def score_single_customer(payload: dict[str, Any]) -> dict[str, Any]:
    df = pd.DataFrame([payload])
    df = _feature_engineering(df)
    df = _rule_score(df)

    row = df.iloc[0].copy()
    tier = str(row["risk_tier"])
    if row["sanctions_flag"] == 1:
        tier = "HIGH"

    row["risk_tier_final"] = tier
    decision = _assign_decision(row)
    factors = _get_top_risk_factors(row)

    return {
        "risk_score": float(row["risk_score"]),
        "risk_tier": tier,
        "decision": decision,
        "top_risk_factors": factors,
        "signals": {
            "F1_list_match": float(row["F1_list_match"]),
            "F2_behaviour": float(row["F2_behaviour"]),
            "F3_suspicious": float(row["F3_suspicious"]),
            "F4_aml_noise": float(row["F4_aml_noise"]),
            "F5_identity": float(row["F5_identity"]),
        },
    }
