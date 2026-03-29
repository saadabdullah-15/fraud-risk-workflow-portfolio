# Fraud Risk Workflow Portfolio Project

This project turns the anonymized `creditcard.csv` dataset into a small fintech operations case study. The goal is not to train a model yet. The goal is to show how a student can structure a fraud-screening workflow, define review and reject rules, measure operational tradeoffs, and communicate the outcome in a business-ready way.

It is designed as a portfolio project for a Working Student role focused on Technical Business Analytics and Workflow Development in fintech.

## Project Objective

Build two simple transaction-screening workflows that classify each transaction into:

- `APPROVE`
- `REVIEW`
- `REJECT`

The workflows must stay interpretable, avoid machine learning, and use the fraud label only for offline evaluation.

## Business Framing

In a real fintech risk workflow, fraud controls are not only about detection quality. They are also about:

- keeping manual review queues manageable
- reducing unnecessary false declines
- escalating only the riskiest transactions
- documenting why a transaction was approved, reviewed, or rejected

This project reflects that logic by comparing:

1. a rule-based decision workflow
2. a simple additive risk-scoring workflow

`REVIEW` represents a manual fraud screening queue. `REJECT` represents a hard-stop recommendation in this demo. In production, some firms would replace part of that reject bucket with step-up authentication or another customer verification step.

## Data and Feature Design

The dataset contains `284,807` transactions, of which `492` are labeled as fraud (`0.1727%` fraud rate). It spans roughly `2` days and has no missing values.

To keep the workflow interpretable, the script engineers readable features from `Time` and `Amount`:

- `hour_of_day`
- `day_index`
- `is_off_hours`
- `amount_log`
- `amount_bucket`
- `high_amount_flag`
- `very_high_amount_flag`

Because the source dataset uses anonymized `V1` to `V28` variables, the script does not build opaque rules on individual PCA columns. Instead, it summarizes them into a few explainable anomaly indicators:

- `behavioral_outlier_count`
- `severe_outlier_count`
- `behavioral_peak_abs`
- `extreme_behavior_flag`

That keeps the workflow more realistic for a policy prototype: business-readable timing and amount signals, supported by a compact anomaly summary from upstream risk features.

## Workflow Logic

### 1. Rule-Based Workflow

The rule-based workflow uses a clear hierarchy:

- `REJECT` for very high anomaly intensity or stacked severe anomaly signals
- `REVIEW` for strong anomaly patterns, off-hours risk, or elevated amount plus moderate anomaly evidence
- `APPROVE` when no material policy flags are triggered

Each transaction receives a decision reason such as:

- `Very high anomaly intensity`
- `Strong anomaly pattern requires analyst review`
- `Elevated amount paired with moderate anomaly pattern`

### 2. Risk-Scoring Workflow

The scoring workflow assigns points to a small set of interpretable signals:

- anomaly intensity
- severe anomaly concentration
- extreme behavior spikes
- off-hours timing
- elevated transaction amount
- high-value off-hours combinations

The final score is mapped to:

- `APPROVE` for low scores
- `REVIEW` for medium scores
- `REJECT` for high scores

This version is slightly more sensitive than the rule-based policy and captures more fraud, while still providing a primary decision reason and a short detail field.

## Results

### Headline Metrics

| Workflow | Review Rate | Reject Rate | Fraud Capture Rate | Frauds Captured |
| --- | ---: | ---: | ---: | ---: |
| Rule-based workflow | 2.03% | 0.24% | 72.56% | 357 / 492 |
| Score-based workflow | 2.13% | 0.88% | 79.27% | 390 / 492 |

### Interpretation

- The rule-based workflow is more conservative on hard rejects and easier to explain line by line.
- The score-based workflow captures more fraud with only a small increase in review load.
- Both workflows reduce fraud exposure substantially compared with approving everything, while keeping the policy logic understandable.

## Output Files

Running the script creates the following files in `outputs/`:

- `decision_bucket_summary.csv`
- `decision_reason_summary.csv`
- `workflow_sample.csv`
- `project_metrics.txt`

These files are intended to be easy to reference in an application, interview, or CV discussion.

## How To Run

Place `creditcard.csv` in the repository root before running the script. The dataset is kept out of Git because the raw file is too large for a standard GitHub commit.

```bash
python fraud_risk_workflow.py
```

## Repository Structure

- `fraud_risk_workflow.py`: main analysis and export script
- `creditcard.csv`: raw input dataset kept locally, not committed to GitHub
- `outputs/decision_bucket_summary.csv`: grouped summary by decision bucket
- `outputs/decision_reason_summary.csv`: grouped summary by decision reason
- `outputs/workflow_sample.csv`: recruiter-friendly sample output
- `outputs/project_metrics.txt`: compact metrics snapshot

## Assumptions and Limitations

- `Class` is used only for evaluation, never inside the decision rules.
- The dataset is anonymized, so this is a workflow prototype rather than a production fraud policy.
- Thresholds are chosen for clarity and operational credibility, not for maximum possible precision.
- A production workflow would usually include customer, merchant, channel, and historical behavioral features that are not available here.

## Why This Is Relevant For A Fintech Working Student Role

This project is positioned around workflow thinking, not just analysis:

- translating raw transaction data into decision logic
- documenting decision reasons for operations teams
- measuring the tradeoff between manual review load and fraud capture
- packaging outputs clearly for stakeholders

That aligns well with roles that sit between analytics, risk operations, and workflow development.

## CV-ready project summary

- Built a Python-based fraud screening case study on `284k+` credit card transactions, designing both a rule-based workflow and a simple risk-scoring workflow with `APPROVE`, `REVIEW`, and `REJECT` outcomes.
- Engineered interpretable transaction features from time and amount data, added decision-reason fields, and evaluated workflow performance using review rate and fraud capture rate rather than machine learning accuracy.
- Delivered recruiter-friendly outputs including decision-bucket summaries, reason summaries, and sample screening results; the score-based workflow captured `79.27%` of fraud while keeping manual review to `2.13%` of transactions.

## Suggested Next Step

The natural next step is to keep this policy baseline and add a supervised model on top of it. That would allow comparison between:

- pure policy rules
- policy plus score
- policy plus machine learning

This sequencing is more realistic than jumping directly into a model without first defining an operational decision process.
