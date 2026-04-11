# Smart KYC Risk Scoring Engine — Workdown

---

## Problem Summary

Build an AI-powered KYC (Know Your Customer) Risk Scoring Engine that automates the customer
onboarding decision process for a financial institution. The system must ingest customer data,
compute a risk score, classify customers into risk tiers, recommend onboarding decisions, and
explain the reasoning behind high-risk flags.

---

## Dataset

15 columns per customer record, modelled on real AML (Anti-Money Laundering) onboarding systems.

| Column | Description | Risk Relevance |
|--------|-------------|----------------|
| `customer_id` | Unique customer identifier | Tracking only |
| `age` | Customer age | Very young / very old may need review |
| `country_risk` | Country risk category | Geography fraud indicator |
| `occupation` | Customer profession | Cash-intensive businesses carry higher AML risk |
| `annual_income` | Declared annual income | Detects unusual financial behaviour |
| `account_type` | Type of account (e.g. savings, corporate, NRI) | Corporate / NRI accounts have higher compliance requirements |
| `document_status` | KYC document completeness | Missing or expired documents = high risk |
| `address_verified` | Whether address has been validated | Unverified address is a fraud indicator |
| `pep_flag` | Politically Exposed Person flag | High regulatory compliance risk |
| `sanctions_flag` | Appears on a sanctions list | Critical AML flag — must be caught |
| `adverse_media_flag` | Negative news presence | Reputation and fraud risk |
| `customer_tenure_years` | Length of relationship with the institution | New customers are statistically riskier |
| `digital_risk_score` | Device / network fraud score (0–100) | Digital fraud detection |
| `fraud_history_flag` | Past fraud involvement | Strong predictor of future risk |
| `monthly_txn_count` | Expected monthly transaction volume | Behavioural risk signal |

---

## Deliverables

| Deliverable | Required |
|-------------|----------|
| Code (Python notebook or script) | ✅ Must-have |
| Output CSV report | ✅ Must-have |
| Dashboard | Optional (bonus) |
| Presentation slides | Optional (bonus) |

---

## Workdown

### Step 1 — Data Preprocessing
- Load the dataset and inspect for nulls, data types, and outliers
- Handle missing values appropriately (imputation or flagging)
- Encode categorical columns (`country_risk`, `occupation`, `account_type`, `document_status`, etc.)
- Normalise / scale numerical columns (`age`, `annual_income`, `digital_risk_score`, `monthly_txn_count`, etc.)
- Ensure binary flags (`pep_flag`, `sanctions_flag`, `adverse_media_flag`, `fraud_history_flag`, `address_verified`) are correctly typed

---

### Step 2 — Feature Engineering
- Create composite risk indicators where applicable, for example:
  - Combine `pep_flag` + `sanctions_flag` + `adverse_media_flag` into a **compliance risk composite**
  - Combine `document_status` + `address_verified` into a **document completeness score**
- Derive a **behavioural signal** from `monthly_txn_count` vs `annual_income` ratio
- Flag customers with `customer_tenure_years` = 0 (brand new) as a separate feature
- Consider binning `age` into risk-meaningful groups (e.g. under 20, 20–60, over 60)

---

### Step 3 — Risk Score Calculation
- Assign weights to each feature based on its risk relevance (refer to the dataset table above)
- Compute a **weighted risk score** per customer (0–100 scale recommended)
- Suggested weight hierarchy (teams may define their own, but must justify):
  - **Critical:** `sanctions_flag`, `fraud_history_flag`
  - **High:** `pep_flag`, `adverse_media_flag`, `document_status`
  - **Medium:** `country_risk`, `digital_risk_score`, `address_verified`
  - **Low:** `age`, `occupation`, `account_type`, `customer_tenure_years`, `monthly_txn_count`, `annual_income`

---

### Step 4 — Risk Classification (ML Model)
- Train a classification model to predict risk tier using the engineered features and score
- Suggested models to try: Logistic Regression, Random Forest, Gradient Boosting (XGBoost/LightGBM)
- Target classes:
  - **Low Risk** → Auto-approve
  - **Medium Risk** → Send for manual review
  - **High Risk** → Reject or escalate for Enhanced Due Diligence (EDD)
- Evaluate using precision, recall, and F1-score — false negatives (missed high-risk) are more costly than false positives

---

### Step 5 — Decision Recommendation Engine
- Map each risk classification to an onboarding decision:
  - Low Risk → `APPROVE`
  - Medium Risk → `MANUAL_REVIEW`
  - High Risk → `REJECT` or `EDD`
- Output the decision alongside the risk score for each customer in the output CSV

---

### Step 6 — Explainability (Explainable AI)
- For each customer flagged as Medium or High risk, provide a human-readable explanation of the top contributing factors
- Recommended approaches:
  - **SHAP values** for feature-level contribution
  - **LIME** for local explanations
  - At minimum, list the top 3 risk factors per customer in plain English (e.g. *"Sanctions flag detected, unverified address, high-risk country"*)

---

### Step 7 — Output CSV Report
Produce a final `kyc_output.csv` with at least the following columns:

| Column | Description |
|--------|-------------|
| `customer_id` | |
| `risk_score` | Computed score (0–100) |
| `risk_tier` | `LOW` / `MEDIUM` / `HIGH` |
| `decision` | `APPROVE` / `MANUAL_REVIEW` / `REJECT` / `EDD` |
| `top_risk_factors` | Comma-separated list of the main flags driving the score |

---

### Step 8 — Dashboard (Bonus)
Build a simple visual dashboard showing:
- Distribution of customers across risk tiers (pie/bar chart)
- Top risk factors across the full dataset
- A searchable/filterable table of individual customer decisions
- Summary statistics: total approved, flagged, rejected

---

## Evaluation Criteria

| Criterion | What Judges Look For |
|-----------|----------------------|
| Correct risk scoring | Scores are computed logically, weights are justified |
| Risk classification logic | Tier thresholds are clearly defined and consistently applied |
| AI / ML usage | A model is trained, not just a rule-based lookup |
| Code quality | Clean, readable, modular code with comments |
| Output clarity | CSV is complete, columns are clear, decisions are unambiguous |
| Visualisation | Charts and/or dashboard are informative and well-labelled |
| Innovation | Creative feature engineering, novel explainability, or additional insights |

---

## Grading Tiers

| Tier | What's Expected |
|------|----------------|
| **Minimum Pass** | Correct scoring + classification output |
| **Good Solution** | AI model + explainability for risk factors |
| **Excellent Solution** | Dashboard + explainable AI |
| **Winning Solution** | AI + visualisation + explainability + business insights |

---

## Notes

- Teams must justify their weight assignments for the risk score — there is no single correct answer, but the logic must be sound
- `sanctions_flag` = 1 should be treated as a near-automatic high-risk flag regardless of other scores
- The explainability component (Step 6) is what separates a good solution from an excellent one — invest time here
