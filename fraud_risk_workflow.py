from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "creditcard.csv"
OUTPUT_DIR = ROOT_DIR / "outputs"

DECISION_ORDER = {"APPROVE": 0, "REVIEW": 1, "REJECT": 2}
SAMPLE_ROWS_PER_DECISION = 20

SECONDS_PER_DAY = 86_400
SECONDS_PER_HOUR = 3_600

MICRO_AMOUNT_THRESHOLD = 25
ELEVATED_AMOUNT_THRESHOLD = 100
HIGH_AMOUNT_THRESHOLD = 500
VERY_HIGH_AMOUNT_THRESHOLD = 1_000

RULE_REJECT_OUTLIER_COUNT = 13
RULE_REJECT_EXTREME_OUTLIER_COUNT = 10
RULE_REVIEW_OUTLIER_COUNT = 8
RULE_REVIEW_CONCENTRATED_OUTLIER_COUNT = 7
RULE_REVIEW_CONCENTRATED_SEVERE_COUNT = 4
RULE_REVIEW_OFF_HOURS_OUTLIER_COUNT = 5
RULE_REVIEW_AMOUNT_OUTLIER_COUNT = 4

OUTLIER_FLAG_THRESHOLD = 3
SEVERE_OUTLIER_FLAG_THRESHOLD = 5
EXTREME_BEHAVIOR_THRESHOLD = 7

SCORE_OUTLIER_HIGH_THRESHOLD = 10
SCORE_OUTLIER_MEDIUM_THRESHOLD = 7
SCORE_OUTLIER_LOW_THRESHOLD = 4
SCORE_SEVERE_HIGH_THRESHOLD = 5
SCORE_SEVERE_MEDIUM_THRESHOLD = 3
SCORE_SEVERE_LOW_THRESHOLD = 1
SCORE_REVIEW_THRESHOLD = 16
SCORE_REJECT_THRESHOLD = 50


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the credit card dataset and validate the expected fields."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    required_columns = {"Time", "Amount", "Class"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing_list}")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create business-readable features and a compact anomaly summary."""
    working_df = df.copy()
    v_columns = [column for column in working_df.columns if column.startswith("V")]
    absolute_v = working_df[v_columns].abs()

    working_df["hour_of_day"] = ((working_df["Time"] % SECONDS_PER_DAY) // SECONDS_PER_HOUR).astype(int)
    working_df["day_index"] = (working_df["Time"] // SECONDS_PER_DAY).astype(int)
    working_df["is_off_hours"] = working_df["hour_of_day"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    working_df["amount_log"] = np.log1p(working_df["Amount"])
    working_df["amount_bucket"] = pd.cut(
        working_df["Amount"],
        bins=[-0.01, MICRO_AMOUNT_THRESHOLD, ELEVATED_AMOUNT_THRESHOLD, HIGH_AMOUNT_THRESHOLD, VERY_HIGH_AMOUNT_THRESHOLD, np.inf],
        labels=["micro", "low", "medium", "high", "very_high"],
        include_lowest=True,
    )
    working_df["high_amount_flag"] = (working_df["Amount"] >= HIGH_AMOUNT_THRESHOLD).astype(int)
    working_df["very_high_amount_flag"] = (working_df["Amount"] >= VERY_HIGH_AMOUNT_THRESHOLD).astype(int)

    # The original V-columns are anonymized. We keep them usable by summarizing how
    # many of those inputs look unusually extreme, which is easier to explain.
    working_df["behavioral_outlier_count"] = (absolute_v > OUTLIER_FLAG_THRESHOLD).sum(axis=1)
    working_df["severe_outlier_count"] = (absolute_v > SEVERE_OUTLIER_FLAG_THRESHOLD).sum(axis=1)
    working_df["behavioral_peak_abs"] = absolute_v.max(axis=1)
    working_df["extreme_behavior_flag"] = (absolute_v > EXTREME_BEHAVIOR_THRESHOLD).any(axis=1).astype(int)

    return working_df


def assign_rule_decision(row: object) -> tuple[str, str]:
    """Apply a transparent, hierarchical rule workflow."""
    if row.behavioral_outlier_count >= RULE_REJECT_OUTLIER_COUNT:
        return "REJECT", "Very high anomaly intensity"

    if row.extreme_behavior_flag and row.behavioral_outlier_count >= RULE_REJECT_EXTREME_OUTLIER_COUNT:
        return "REJECT", "Extreme anomaly stack with multiple severe signals"

    # Hierarchical policy: reject first, then increasingly broader review checks.
    concentrated_anomaly_flag = (
        row.behavioral_outlier_count >= RULE_REVIEW_CONCENTRATED_OUTLIER_COUNT
        and row.severe_outlier_count >= RULE_REVIEW_CONCENTRATED_SEVERE_COUNT
        and row.Amount >= HIGH_AMOUNT_THRESHOLD
    )
    off_hours_anomaly_flag = (
        row.behavioral_outlier_count >= RULE_REVIEW_OFF_HOURS_OUTLIER_COUNT and row.is_off_hours
    )
    moderate_amount_anomaly_flag = (
        row.behavioral_outlier_count >= RULE_REVIEW_AMOUNT_OUTLIER_COUNT
        and row.Amount >= ELEVATED_AMOUNT_THRESHOLD
    )
    high_value_off_hours_flag = row.Amount >= HIGH_AMOUNT_THRESHOLD and row.is_off_hours

    if (
        concentrated_anomaly_flag
        and row.behavioral_outlier_count < RULE_REVIEW_OUTLIER_COUNT
        and not row.extreme_behavior_flag
    ):
        return "REVIEW", "High-value transaction with concentrated anomaly signals"

    if (
        off_hours_anomaly_flag
        and row.behavioral_outlier_count < RULE_REVIEW_OUTLIER_COUNT
        and not row.extreme_behavior_flag
        and not concentrated_anomaly_flag
    ):
        return "REVIEW", "Off-hours transaction with elevated anomaly pattern"

    if (
        moderate_amount_anomaly_flag
        and row.behavioral_outlier_count < RULE_REVIEW_OUTLIER_COUNT
        and not row.extreme_behavior_flag
        and not concentrated_anomaly_flag
        and not off_hours_anomaly_flag
    ):
        return "REVIEW", "Elevated amount paired with moderate anomaly pattern"

    if row.behavioral_outlier_count >= RULE_REVIEW_OUTLIER_COUNT:
        return "REVIEW", "Strong anomaly pattern requires analyst review"

    if row.extreme_behavior_flag:
        return "REVIEW", "Extreme feature movement requires analyst review"

    if (
        high_value_off_hours_flag
        and not row.extreme_behavior_flag
        and row.behavioral_outlier_count < RULE_REVIEW_AMOUNT_OUTLIER_COUNT
    ):
        return "REVIEW", "High-value off-hours transaction"

    return "APPROVE", "No material risk flags under current policy"


def apply_rule_workflow(df: pd.DataFrame) -> pd.DataFrame:
    """Add rule-based decision outcomes and reasons."""
    decisions: list[str] = []
    reasons: list[str] = []

    for row in df.itertuples(index=False):
        decision, reason = assign_rule_decision(row)
        decisions.append(decision)
        reasons.append(reason)

    result = df.copy()
    result["rule_decision"] = decisions
    result["rule_decision_reason"] = reasons
    return result


def build_score_components(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate additive score components for a lightweight screening score."""
    components = pd.DataFrame(index=df.index)
    # Score thresholds are manually calibrated for an interpretable prototype.
    components["score_behavioral_outlier_points"] = np.select(
        [
            df["behavioral_outlier_count"] >= SCORE_OUTLIER_HIGH_THRESHOLD,
            df["behavioral_outlier_count"] >= SCORE_OUTLIER_MEDIUM_THRESHOLD,
            df["behavioral_outlier_count"] >= SCORE_OUTLIER_LOW_THRESHOLD,
        ],
        [35, 25, 12],
        default=0,
    )
    components["score_severe_outlier_points"] = np.select(
        [
            df["severe_outlier_count"] >= SCORE_SEVERE_HIGH_THRESHOLD,
            df["severe_outlier_count"] >= SCORE_SEVERE_MEDIUM_THRESHOLD,
            df["severe_outlier_count"] >= SCORE_SEVERE_LOW_THRESHOLD,
        ],
        [20, 12, 5],
        default=0,
    )
    components["score_extreme_behavior_points"] = np.where(df["extreme_behavior_flag"] == 1, 18, 0)
    components["score_off_hours_points"] = np.where(df["is_off_hours"] == 1, 8, 0)
    components["score_amount_points"] = np.select(
        [
            df["Amount"] >= VERY_HIGH_AMOUNT_THRESHOLD,
            df["Amount"] >= HIGH_AMOUNT_THRESHOLD,
            df["Amount"] >= ELEVATED_AMOUNT_THRESHOLD,
        ],
        [12, 8, 4],
        default=0,
    )
    components["score_high_amount_off_hours_points"] = np.where(
        (df["Amount"] >= HIGH_AMOUNT_THRESHOLD) & (df["is_off_hours"] == 1),
        6,
        0,
    )
    components["risk_score"] = components.sum(axis=1)

    return components


def determine_score_decision(score: int) -> str:
    """Map the total score to an operational action."""
    if score >= SCORE_REJECT_THRESHOLD:
        return "REJECT"
    if score >= SCORE_REVIEW_THRESHOLD:
        return "REVIEW"
    return "APPROVE"


def build_score_reason(row: object) -> tuple[str, str]:
    """Provide a primary reason and a short explanation for the score outcome."""
    decision = determine_score_decision(int(row.risk_score))
    contributor_labels: list[tuple[int, str]] = []

    if row.score_behavioral_outlier_points > 0:
        if row.behavioral_outlier_count >= SCORE_OUTLIER_HIGH_THRESHOLD:
            label = "very high anomaly count"
        elif row.behavioral_outlier_count >= SCORE_OUTLIER_MEDIUM_THRESHOLD:
            label = "strong anomaly count"
        else:
            label = "moderate anomaly count"
        contributor_labels.append((int(row.score_behavioral_outlier_points), label))

    if row.score_severe_outlier_points > 0:
        if row.severe_outlier_count >= SCORE_SEVERE_HIGH_THRESHOLD:
            label = "multiple severe anomalies"
        elif row.severe_outlier_count >= SCORE_SEVERE_MEDIUM_THRESHOLD:
            label = "cluster of severe anomalies"
        else:
            label = "single severe anomaly"
        contributor_labels.append((int(row.score_severe_outlier_points), label))

    if row.score_extreme_behavior_points > 0:
        contributor_labels.append((int(row.score_extreme_behavior_points), "extreme feature spike"))

    if row.score_amount_points > 0:
        if row.Amount >= VERY_HIGH_AMOUNT_THRESHOLD:
            label = "very high transaction amount"
        elif row.Amount >= HIGH_AMOUNT_THRESHOLD:
            label = "high transaction amount"
        else:
            label = "elevated transaction amount"
        contributor_labels.append((int(row.score_amount_points), label))

    if row.score_off_hours_points > 0:
        contributor_labels.append((int(row.score_off_hours_points), "off-hours timing"))

    if row.score_high_amount_off_hours_points > 0:
        contributor_labels.append(
            (int(row.score_high_amount_off_hours_points), "high-value off-hours combination")
        )

    contributor_labels.sort(key=lambda item: (-item[0], item[1]))
    detail = ", ".join(label for _, label in contributor_labels[:3])

    if decision == "REJECT":
        if (
            row.behavioral_outlier_count >= SCORE_OUTLIER_HIGH_THRESHOLD
            and row.severe_outlier_count >= SCORE_SEVERE_HIGH_THRESHOLD
        ):
            primary_reason = "Severe anomaly stack pushed score above reject threshold"
        elif row.extreme_behavior_flag == 1:
            primary_reason = "Extreme behavior spike pushed score above reject threshold"
        else:
            primary_reason = "Risk score exceeded reject threshold"
    elif decision == "REVIEW":
        if row.behavioral_outlier_count >= SCORE_OUTLIER_MEDIUM_THRESHOLD:
            primary_reason = "Strong anomaly stack requires analyst review"
        elif row.Amount >= HIGH_AMOUNT_THRESHOLD and row.is_off_hours == 1:
            primary_reason = "High-value off-hours pattern requires review"
        else:
            primary_reason = "Risk score crossed review threshold"
    else:
        primary_reason = "Score remained below review threshold"
        detail = "limited anomaly evidence"

    return primary_reason, detail


def apply_score_workflow(df: pd.DataFrame) -> pd.DataFrame:
    """Add the score-based decision workflow and explanation fields."""
    result = df.copy()
    score_components = build_score_components(result)
    result = pd.concat([result, score_components], axis=1)
    result["score_decision"] = result["risk_score"].apply(determine_score_decision)

    primary_reasons: list[str] = []
    detail_reasons: list[str] = []
    for row in result.itertuples(index=False):
        primary_reason, detail_reason = build_score_reason(row)
        primary_reasons.append(primary_reason)
        detail_reasons.append(detail_reason)

    result["score_decision_reason"] = primary_reasons
    result["score_reason_detail"] = detail_reasons
    return result


def calculate_workflow_metrics(df: pd.DataFrame, decision_column: str) -> dict[str, float]:
    """Compute the headline metrics used in the README and metrics file."""
    total_transactions = len(df)
    total_frauds = int(df["Class"].sum())
    review_count = int((df[decision_column] == "REVIEW").sum())
    reject_count = int((df[decision_column] == "REJECT").sum())
    approve_count = int((df[decision_column] == "APPROVE").sum())
    captured_fraud_count = int(
        ((df["Class"] == 1) & (df[decision_column].isin(["REVIEW", "REJECT"]))).sum()
    )

    return {
        "transactions": total_transactions,
        "fraud_transactions": total_frauds,
        "approve_count": approve_count,
        "review_count": review_count,
        "reject_count": reject_count,
        "review_rate": review_count / total_transactions,
        "reject_rate": reject_count / total_transactions,
        "fraud_capture_rate": captured_fraud_count / total_frauds,
        "captured_fraud_count": captured_fraud_count,
    }


def build_decision_bucket_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize performance by decision bucket for both workflows."""
    summary_frames: list[pd.DataFrame] = []
    workflow_specs = [
        ("rule_workflow", "rule_decision"),
        ("score_workflow", "score_decision"),
    ]

    for workflow_name, decision_column in workflow_specs:
        summary = (
            df.groupby(decision_column, observed=False)
            .agg(
                transaction_count=("Class", "size"),
                total_amount=("Amount", "sum"),
                avg_amount=("Amount", "mean"),
                median_amount=("Amount", "median"),
                fraud_count=("Class", "sum"),
            )
            .reset_index()
            .rename(columns={decision_column: "decision"})
        )
        summary["fraud_rate"] = summary["fraud_count"] / summary["transaction_count"]
        summary["transaction_share"] = summary["transaction_count"] / len(df)
        summary["workflow"] = workflow_name
        summary["decision_order"] = summary["decision"].map(DECISION_ORDER)
        summary_frames.append(summary)

    combined_summary = pd.concat(summary_frames, ignore_index=True)
    combined_summary = combined_summary[
        [
            "workflow",
            "decision",
            "transaction_count",
            "transaction_share",
            "total_amount",
            "avg_amount",
            "median_amount",
            "fraud_count",
            "fraud_rate",
            "decision_order",
        ]
    ]
    combined_summary = combined_summary.sort_values(["workflow", "decision_order"]).drop(
        columns="decision_order"
    )
    return combined_summary


def build_decision_reason_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize how often each decision reason appears and how risky it was."""
    summary_frames: list[pd.DataFrame] = []
    workflow_specs = [
        ("rule_workflow", "rule_decision", "rule_decision_reason"),
        ("score_workflow", "score_decision", "score_decision_reason"),
    ]

    for workflow_name, decision_column, reason_column in workflow_specs:
        summary = (
            df.groupby([decision_column, reason_column], observed=False)
            .agg(
                transaction_count=("Class", "size"),
                fraud_count=("Class", "sum"),
            )
            .reset_index()
            .rename(columns={decision_column: "decision", reason_column: "decision_reason"})
        )
        summary["fraud_rate"] = summary["fraud_count"] / summary["transaction_count"]
        summary["workflow"] = workflow_name
        summary["decision_order"] = summary["decision"].map(DECISION_ORDER)
        summary_frames.append(summary)

    combined_summary = pd.concat(summary_frames, ignore_index=True)
    combined_summary = combined_summary[
        [
            "workflow",
            "decision",
            "decision_reason",
            "transaction_count",
            "fraud_count",
            "fraud_rate",
            "decision_order",
        ]
    ]
    combined_summary = combined_summary.sort_values(
        ["workflow", "decision_order", "fraud_count", "transaction_count"],
        ascending=[True, True, False, False],
    ).drop(columns="decision_order")
    return combined_summary


def build_sample_export(df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact sample file with examples from each rule bucket."""
    sample_frames: list[pd.DataFrame] = []

    for decision in ["REJECT", "REVIEW", "APPROVE"]:
        bucket = df[df["rule_decision"] == decision].copy()
        bucket = bucket.sort_values(["risk_score", "behavioral_outlier_count", "Amount"], ascending=False)
        sample_frames.append(bucket.head(SAMPLE_ROWS_PER_DECISION))

    sample_df = pd.concat(sample_frames, ignore_index=True)
    sample_df = sample_df[
        [
            "Time",
            "Amount",
            "Class",
            "hour_of_day",
            "day_index",
            "is_off_hours",
            "amount_bucket",
            "behavioral_outlier_count",
            "severe_outlier_count",
            "behavioral_peak_abs",
            "extreme_behavior_flag",
            "rule_decision",
            "rule_decision_reason",
            "risk_score",
            "score_decision",
            "score_decision_reason",
            "score_reason_detail",
        ]
    ]
    return sample_df


def build_dataset_overview(df: pd.DataFrame) -> dict[str, float]:
    """Collect a few descriptive statistics for documentation and logging."""
    overview = {
        "transactions": len(df),
        "fraud_transactions": int(df["Class"].sum()),
        "fraud_rate": float(df["Class"].mean()),
        "amount_median": float(df["Amount"].median()),
        "amount_p90": float(df["Amount"].quantile(0.90)),
        "amount_p99": float(df["Amount"].quantile(0.99)),
        "time_span_days": float(df["Time"].max() / SECONDS_PER_DAY),
        "missing_values": int(df.isna().sum().sum()),
    }
    return overview


def write_metrics_file(
    output_path: Path,
    overview: dict[str, float],
    rule_metrics: dict[str, float],
    score_metrics: dict[str, float],
) -> None:
    """Persist a plain-text summary that is easy to reference in the README or a CV."""
    lines = [
        "Fraud Risk Workflow Metrics",
        "===========================",
        "",
        "Dataset overview",
        f"- Transactions: {overview['transactions']:,}",
        f"- Fraud transactions: {overview['fraud_transactions']:,}",
        f"- Fraud rate: {overview['fraud_rate']:.4%}",
        f"- Median amount: {overview['amount_median']:.2f}",
        f"- 90th percentile amount: {overview['amount_p90']:.2f}",
        f"- 99th percentile amount: {overview['amount_p99']:.2f}",
        f"- Time span covered: {overview['time_span_days']:.2f} days",
        f"- Missing values: {overview['missing_values']:,}",
        "",
        "Rule-based workflow",
        f"- Approve count: {rule_metrics['approve_count']:,}",
        f"- Review count: {rule_metrics['review_count']:,}",
        f"- Reject count: {rule_metrics['reject_count']:,}",
        f"- Review rate: {rule_metrics['review_rate']:.2%}",
        f"- Reject rate: {rule_metrics['reject_rate']:.2%}",
        f"- Fraud capture rate: {rule_metrics['fraud_capture_rate']:.2%}",
        f"- Frauds captured in review/reject: {rule_metrics['captured_fraud_count']:,}",
        "",
        "Score-based workflow",
        f"- Approve count: {score_metrics['approve_count']:,}",
        f"- Review count: {score_metrics['review_count']:,}",
        f"- Reject count: {score_metrics['reject_count']:,}",
        f"- Review rate: {score_metrics['review_rate']:.2%}",
        f"- Reject rate: {score_metrics['reject_rate']:.2%}",
        f"- Fraud capture rate: {score_metrics['fraud_capture_rate']:.2%}",
        f"- Frauds captured in review/reject: {score_metrics['captured_fraud_count']:,}",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def export_outputs(df: pd.DataFrame, overview: dict[str, float]) -> dict[str, dict[str, float]]:
    """Write all requested output files and return workflow metrics for console logging."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    rule_metrics = calculate_workflow_metrics(df, "rule_decision")
    score_metrics = calculate_workflow_metrics(df, "score_decision")

    decision_bucket_summary = build_decision_bucket_summary(df)
    decision_reason_summary = build_decision_reason_summary(df)
    sample_export = build_sample_export(df)

    decision_bucket_summary.to_csv(OUTPUT_DIR / "decision_bucket_summary.csv", index=False)
    decision_reason_summary.to_csv(OUTPUT_DIR / "decision_reason_summary.csv", index=False)
    sample_export.to_csv(OUTPUT_DIR / "workflow_sample.csv", index=False)
    write_metrics_file(OUTPUT_DIR / "project_metrics.txt", overview, rule_metrics, score_metrics)

    return {
        "rule_workflow": rule_metrics,
        "score_workflow": score_metrics,
    }


def print_console_summary(overview: dict[str, float], metrics: dict[str, dict[str, float]]) -> None:
    """Print a concise run summary after all files have been written."""
    print("Fraud risk workflow analysis completed.")
    print(f"Transactions: {overview['transactions']:,}")
    print(f"Fraud transactions: {overview['fraud_transactions']:,} ({overview['fraud_rate']:.4%})")
    print(f"Median amount: {overview['amount_median']:.2f}")
    print("")

    for workflow_name, workflow_metrics in metrics.items():
        readable_name = workflow_name.replace("_", " ").title()
        print(readable_name)
        print(f"  Review rate: {workflow_metrics['review_rate']:.2%}")
        print(f"  Reject rate: {workflow_metrics['reject_rate']:.2%}")
        print(f"  Fraud capture rate: {workflow_metrics['fraud_capture_rate']:.2%}")
        print(f"  Fraud captured: {workflow_metrics['captured_fraud_count']:,}")
        print("")

    print(f"Outputs written to: {OUTPUT_DIR}")


def main() -> None:
    dataset = load_dataset(DATA_PATH)
    overview = build_dataset_overview(dataset)
    featured_dataset = engineer_features(dataset)
    with_rule_workflow = apply_rule_workflow(featured_dataset)
    final_dataset = apply_score_workflow(with_rule_workflow)
    metrics = export_outputs(final_dataset, overview)
    print_console_summary(overview, metrics)


if __name__ == "__main__":
    main()
