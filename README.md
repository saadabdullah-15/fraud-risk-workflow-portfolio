# Fraud Risk Decision Workflow (FinTech Case Study)

This project builds a simplified transaction screening workflow using the anonymized `creditcard.csv` dataset. The focus is not on training a machine learning model, but on designing and evaluating interpretable decision workflows for fraud prevention in a fintech context.

The project demonstrates how transaction data can be translated into `APPROVE` / `REVIEW` / `REJECT` decisions, with clear reasoning and measurable operational trade-offs.

## Objective

Design and evaluate two fraud-screening workflows that classify transactions into:

- `APPROVE`
- `REVIEW`
- `REJECT`

Key constraints:

- Maintain fully interpretable logic
- Avoid machine learning in the decision process
- Use fraud labels (`Class`) only for offline evaluation

## Business Context

In real-world payment systems, fraud prevention is not only about detection accuracy. It requires balancing:

- fraud capture
- manual review capacity
- customer experience (false declines)
- explainability of decisions

This project reflects that trade-off by comparing:

- Rule-based decision workflow
- Risk-scoring workflow

`REVIEW` represents transactions routed to manual investigation.  
`REJECT` represents a hard-stop decision (in practice, often replaced by step-up authentication).

## Dataset and Feature Design

- `284,807` transactions
- `492` fraud cases (`0.17%`)
- No missing values
- `~2` days of activity

To keep the workflow interpretable, features are engineered from:

### Transaction attributes

- `hour_of_day`
- `day_index`
- `is_off_hours`
- `amount_log`
- `amount_bucket`
- `high_amount_flag`
- `very_high_amount_flag`

### Behavioral anomaly signals (derived from PCA features)

Instead of using raw anonymized variables (`V1-V28`), they are aggregated into:

- `behavioral_outlier_count`
- `severe_outlier_count`
- `behavioral_peak_abs`
- `extreme_behavior_flag`

This keeps the workflow closer to how real systems combine business signals + upstream risk indicators.

## Workflow Design

### 1. Rule-Based Workflow

A hierarchical policy assigns decisions based on clear conditions:

- `REJECT` -> extreme anomaly signals or strong stacked risk indicators
- `REVIEW` -> moderate anomaly patterns, off-hours risk, or elevated transaction values
- `APPROVE` -> no significant risk signals

Each decision includes an explicit reason, e.g.:

- `Very high anomaly intensity`
- `Strong anomaly pattern requires analyst review`
- `Elevated amount with moderate anomaly pattern`

### 2. Risk-Scoring Workflow

A simple additive scoring system assigns points based on:

- anomaly intensity
- severe anomaly concentration
- extreme behavioral spikes
- off-hours activity
- transaction amount
- high-value off-hours combinations

Scores are mapped to:

- low -> `APPROVE`
- medium -> `REVIEW`
- high -> `REJECT`

This approach increases flexibility while maintaining interpretability.

## Results

### Key Metrics

| Workflow | Review Rate | Reject Rate | Fraud Capture Rate | Frauds Captured |
| --- | ---: | ---: | ---: | ---: |
| Rule-based | 2.03% | 0.24% | 72.56% | 357 / 492 |
| Score-based | 2.13% | 0.88% | 79.27% | 390 / 492 |

### Interpretation

- The rule-based workflow is simpler and easier to audit.
- The score-based workflow improves fraud capture with minimal increase in review load.
- Both approaches demonstrate a realistic trade-off between risk control and operational efficiency.

## Outputs

Running the script generates:

- `decision_bucket_summary.csv` -> distribution and fraud rates by decision
- `decision_reason_summary.csv` -> breakdown by decision reason
- `workflow_sample.csv` -> sample transactions with decisions and explanations
- `project_metrics.txt` -> compact performance metrics

## How to Run

Place `creditcard.csv` in the root directory and run:

```bash
python fraud_risk_workflow.py
```

The raw dataset is kept out of GitHub because the source file is too large for a standard repository commit.

## Project Structure

- `fraud_risk_workflow.py` — main workflow and analysis script
- `outputs/` — generated summaries and samples
- `README.md` — project documentation
- `creditcard.csv` — local input dataset, not committed to GitHub

## Assumptions and Limitations

- Fraud labels are used only for evaluation
- Dataset is anonymized -> workflow is a policy prototype, not production-ready
- Thresholds are chosen for interpretability, not optimization
- Real systems would include additional features such as customer history, merchant data, and device/channel signals

## Relevance for FinTech / Risk Roles

This project demonstrates:

- translating data into decision workflows
- designing explainable risk logic
- evaluating review vs fraud trade-offs
- structuring outputs for operational use

It reflects the type of work done in roles combining:

- business analysis
- risk/fraud analytics
- workflow development

## CV-Ready Project Summary

- Built a Python-based fraud screening workflow on 284k+ transactions, designing both rule-based and score-based decision systems with `APPROVE` / `REVIEW` / `REJECT` outcomes
- Engineered interpretable transaction and behavioral risk signals and implemented explainable decision logic with reason tracking
- Evaluated workflow performance using review rate and fraud capture rate; improved fraud capture from 72.6% to 79.3% with minimal increase in review volume

## Next Steps

A realistic extension would be to layer a supervised model on top of this policy baseline and compare:

- rule-based workflow
- score-based workflow
- model-assisted decisioning

This reflects how modern risk systems evolve in practice.
