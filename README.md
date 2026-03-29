# Fraud Risk Decision Workflow

This repository is a small fintech-style fraud screening project built around interpretable decision logic rather than machine learning. It takes anonymized transaction data, engineers readable risk signals, and assigns operational decisions of `APPROVE`, `REVIEW`, or `REJECT`.

The project has two parts:

- an offline workflow analysis in `fraud_risk_workflow.py`
- a local Flask API prototype in `fraud_decision_api.py`

The goal is to show how raw transaction data can be turned into explainable risk decisions with measurable trade-offs.

## What The Project Does

The workflow evaluates transactions using two decision approaches:

- `Rule-based workflow` for direct, hierarchical policy logic
- `Score-based workflow` for additive risk scoring with threshold-based actions

Both workflows are designed to stay easy to explain in an interview or portfolio review:

- business-readable features
- explicit thresholds
- clear decision reasons
- simple performance summaries

## Dataset

The analysis uses the anonymized `creditcard.csv` dataset.

- `284,807` transactions
- `492` fraud cases
- fraud rate of `0.1727%`
- approximately `2` days of transaction activity

The dataset is used for offline workflow evaluation only. Fraud labels are not used to make decisions inside the workflow itself.

## Feature Engineering

To avoid relying directly on anonymized PCA variables (`V1` to `V28`), the project converts them into compact behavioral signals that are easier to interpret.

### Transaction Features

- `hour_of_day`
- `day_index`
- `is_off_hours`
- `amount_log`
- `amount_bucket`
- `high_amount_flag`
- `very_high_amount_flag`

### Behavioral Risk Signals

- `behavioral_outlier_count`
- `severe_outlier_count`
- `behavioral_peak_abs`
- `extreme_behavior_flag`

This mirrors a realistic fraud workflow pattern where upstream model or anomaly signals are summarized into decision-friendly inputs.

## Decision Design

### Rule-Based Workflow

The rule workflow applies transparent checks in order of severity.

- `REJECT` for very strong anomaly stacks
- `REVIEW` for meaningful but less extreme patterns
- `APPROVE` when no material risk flags are present

Example rule-style reasons:

- `Very high anomaly intensity`
- `Strong anomaly pattern requires analyst review`
- `High-value off-hours transaction`

### Score-Based Workflow

The score workflow assigns additive points based on:

- anomaly intensity
- severe anomaly concentration
- extreme behavior spikes
- off-hours timing
- transaction amount
- high-value off-hours combinations

The final score maps to:

- `APPROVE` when score is below `16`
- `REVIEW` when score is `16` to `49`
- `REJECT` when score is `50+`

Each score-based decision also returns:

- `decision_reason`
- `reason_detail`

## Results

| Workflow | Review Rate | Reject Rate | Fraud Capture Rate | Frauds Captured |
| --- | ---: | ---: | ---: | ---: |
| Rule-based | 2.03% | 0.24% | 72.56% | 357 / 492 |
| Score-based | 2.13% | 0.88% | 79.27% | 390 / 492 |

### Takeaway

- The rule workflow is simpler to audit and explain.
- The score workflow captures more fraud while keeping review volume relatively close.
- The project demonstrates the operational trade-off between fraud control, review workload, and customer friction.

## Outputs

Running the workflow script writes files to `outputs/`:

- `decision_bucket_summary.csv` for decision distribution and fraud rate by bucket
- `decision_reason_summary.csv` for reason-level breakdowns
- `workflow_sample.csv` for example transactions with decisions and explanations
- `project_metrics.txt` for compact headline metrics

## Project Structure

- `fraud_risk_workflow.py` - main offline workflow and output generation
- `fraud_decision_api.py` - local Flask API for single-transaction score decisions
- `outputs/` - generated summaries and samples
- `README.md` - project documentation
- `creditcard.csv` - local dataset file, not committed to GitHub

## Local Setup

Use Python 3.10+ and install the required packages:

```bash
pip install pandas numpy flask
```

## Run The Workflow Analysis

Place `creditcard.csv` in the repository root, then run:

```bash
python fraud_risk_workflow.py
```

This generates the summary files in `outputs/`.

## API Extension

`fraud_decision_api.py` exposes a simple local API that simulates a fraud decision service for one transaction-like request at a time.

### Endpoint

`POST /decision`

### Request Fields

- `Amount`
- `hour_of_day`
- `behavioral_outlier_count`
- `severe_outlier_count`
- `extreme_behavior_flag`

The API derives `is_off_hours` internally from `hour_of_day`, reconstructs the score inputs, and returns the same score-based decision logic used in the main workflow.

The API does not load the full dataset for requests.

### Validation

The API returns `400` errors for:

- missing fields
- invalid numeric types
- `hour_of_day` outside `0` to `23`
- negative values for `Amount`
- negative values for `behavioral_outlier_count`
- negative values for `severe_outlier_count`

### Example Request

```json
{
  "Amount": 875.5,
  "hour_of_day": 2,
  "behavioral_outlier_count": 8,
  "severe_outlier_count": 3,
  "extreme_behavior_flag": 1
}
```

### Example Response

```json
{
  "decision": "REJECT",
  "risk_score": 77,
  "decision_reason": "Extreme behavior spike pushed score above reject threshold",
  "reason_detail": "strong anomaly count, extreme feature spike, cluster of severe anomalies",
  "input_echo": {
    "Amount": 875.5,
    "hour_of_day": 2,
    "behavioral_outlier_count": 8,
    "severe_outlier_count": 3,
    "extreme_behavior_flag": 1,
    "is_off_hours": 1
  }
}
```

### Run The API

Start the local server:

```bash
python fraud_decision_api.py
```

Example request:

```bash
curl -X POST http://127.0.0.1:5000/decision \
  -H "Content-Type: application/json" \
  -d "{\"Amount\":875.5,\"hour_of_day\":2,\"behavioral_outlier_count\":8,\"severe_outlier_count\":3,\"extreme_behavior_flag\":1}"
```

## Why This Is Useful For A Portfolio

This project demonstrates:

- translating raw transaction data into risk decisions
- building explainable fraud logic without black-box modeling
- comparing policy-based and score-based decisioning
- summarizing outcomes with operational metrics
- exposing scoring logic through a minimal local API

## Assumptions And Limitations

- The dataset is anonymized, so the workflow is a policy prototype rather than a production fraud system.
- Thresholds are hand-tuned for interpretability, not globally optimized.
- The API is a local prototype only and does not include authentication, persistence, or deployment setup.
- Real fraud systems would also use customer history, merchant features, device signals, and feedback loops.

## CV-Ready Summary

- Built a Python fraud screening workflow on `284k+` transactions with interpretable `APPROVE` / `REVIEW` / `REJECT` decisions
- Engineered business-readable transaction and anomaly signals from anonymized data
- Compared rule-based and score-based decision strategies and measured fraud capture versus operational review load
- Extended the workflow with a lightweight Flask API for single-transaction scoring
