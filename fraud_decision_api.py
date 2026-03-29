from __future__ import annotations

from typing import Any

import pandas as pd
from flask import Flask, jsonify, request

from fraud_risk_workflow import build_score_components, build_score_reason, determine_score_decision


REQUIRED_FIELDS = {
    "Amount",
    "hour_of_day",
    "behavioral_outlier_count",
    "severe_outlier_count",
    "extreme_behavior_flag",
}
OFF_HOURS = {0, 1, 2, 3, 4, 5}

app = Flask(__name__)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_payload(payload: Any) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not isinstance(payload, dict):
        return None, {"error": "Request body must be a JSON object."}

    missing_fields = sorted(REQUIRED_FIELDS.difference(payload))
    if missing_fields:
        return None, {
            "error": "Missing required fields.",
            "missing_fields": missing_fields,
        }

    numeric_fields = [
        "Amount",
        "hour_of_day",
        "behavioral_outlier_count",
        "severe_outlier_count",
        "extreme_behavior_flag",
    ]
    invalid_numeric_fields = [field for field in numeric_fields if not _is_number(payload[field])]
    if invalid_numeric_fields:
        return None, {
            "error": "Invalid numeric field types.",
            "invalid_fields": invalid_numeric_fields,
        }

    if int(payload["hour_of_day"]) != float(payload["hour_of_day"]):
        return None, {"error": "hour_of_day must be an integer between 0 and 23."}

    if int(payload["behavioral_outlier_count"]) != float(payload["behavioral_outlier_count"]):
        return None, {"error": "behavioral_outlier_count must be an integer."}

    if int(payload["severe_outlier_count"]) != float(payload["severe_outlier_count"]):
        return None, {"error": "severe_outlier_count must be an integer."}

    if int(payload["extreme_behavior_flag"]) != float(payload["extreme_behavior_flag"]):
        return None, {"error": "extreme_behavior_flag must be 0 or 1."}

    hour_of_day = int(payload["hour_of_day"])
    if hour_of_day < 0 or hour_of_day > 23:
        return None, {"error": "hour_of_day must be between 0 and 23."}

    amount = float(payload["Amount"])
    if amount < 0:
        return None, {"error": "Amount must be greater than or equal to 0."}

    behavioral_outlier_count = int(payload["behavioral_outlier_count"])
    if behavioral_outlier_count < 0:
        return None, {"error": "behavioral_outlier_count must be greater than or equal to 0."}

    severe_outlier_count = int(payload["severe_outlier_count"])
    if severe_outlier_count < 0:
        return None, {"error": "severe_outlier_count must be greater than or equal to 0."}

    extreme_behavior_flag = int(payload["extreme_behavior_flag"])
    if extreme_behavior_flag not in {0, 1}:
        return None, {"error": "extreme_behavior_flag must be 0 or 1."}

    validated_payload = {
        "Amount": amount,
        "hour_of_day": hour_of_day,
        "behavioral_outlier_count": behavioral_outlier_count,
        "severe_outlier_count": severe_outlier_count,
        "extreme_behavior_flag": extreme_behavior_flag,
    }
    return validated_payload, None


def score_transaction(payload: dict[str, Any]) -> dict[str, Any]:
    request_row = {
        **payload,
        "is_off_hours": int(payload["hour_of_day"] in OFF_HOURS),
    }
    transaction_df = pd.DataFrame([request_row])
    scored_df = pd.concat([transaction_df, build_score_components(transaction_df)], axis=1)
    scored_row = next(scored_df.itertuples(index=False))

    risk_score = int(scored_row.risk_score)
    decision = determine_score_decision(risk_score)
    decision_reason, reason_detail = build_score_reason(scored_row)

    return {
        "decision": decision,
        "risk_score": risk_score,
        "decision_reason": decision_reason,
        "reason_detail": reason_detail,
        "input_echo": request_row,
    }


@app.post("/decision")
def decision() -> Any:
    payload = request.get_json(silent=True)
    validated_payload, error_response = validate_payload(payload)
    if error_response is not None:
        return jsonify(error_response), 400

    return jsonify(score_transaction(validated_payload)), 200


if __name__ == "__main__":
    app.run(debug=True)
