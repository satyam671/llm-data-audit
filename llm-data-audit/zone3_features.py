"""
zone3_features.py
-----------------
Zone 3 audit checks for real-time feature reliability.

Two checks:
    1. check_feature_null_rates         — per-feature null/missing rate in
                                          inference-time payloads
    2. check_feature_distribution_drift — z-score comparison of feature means
                                          against training-time statistics

Each function returns a finding dict with keys:
    check       str   — machine-readable check name
    flag        bool  — True if any feature is above threshold
    severity    str   — "OK" | "MEDIUM" | "HIGH"
    ...         various check-specific metrics

Production note on distribution drift:
    The drift check here uses a simplified z-score approach, which is
    appropriate for an initial one-off diagnostic. For continuous production
    monitoring, replace this with Population Stability Index (PSI).
    PSI > 0.1 is a warning. PSI > 0.2 indicates significant drift requiring
    investigation. The z-score check is a fast, zero-dependency first pass.
"""

from typing import Dict, List


def check_feature_null_rates(
    feature_data: Dict[str, List],
    null_threshold: float = 0.05,
) -> Dict:
    """
    Measures null/missing rates per feature in real-time feature payloads.

    A feature that was 2% null during training but is 18% null in production
    means the model receives a materially different input distribution for a
    significant fraction of requests. Because nulls don't raise exceptions,
    this failure is completely invisible without explicit measurement.

    Handles three null-like values:
        None        — Python null
        ""          — empty string
        v != v      — float NaN (the only Python object not equal to itself)

    Args:
        feature_data:   Dict of {feature_name: [list of values at inference time]}.
                        Collect these from a sample of recent inference requests.
        null_threshold: Null rate above which a feature is flagged MEDIUM.
                        Defaults to 0.05 (5%).

    Returns:
        Dict with keys: check, features_checked, features_flagged,
        flagged_features, feature_details (per-feature breakdown), flag, severity.
    """
    feature_report = {}
    flagged_features = []

    for feature_name, values in feature_data.items():
        total = len(values)
        if total == 0:
            continue

        null_count = sum(
            1 for v in values
            if v is None or v == "" or v != v  # v != v catches float NaN
        )
        null_rate = null_count / total
        is_flagged = null_rate > null_threshold

        feature_report[feature_name] = {
            "null_rate": round(null_rate, 4),
            "flag": is_flagged,
            "severity": (
                "HIGH" if null_rate > 0.20
                else "MEDIUM" if is_flagged
                else "OK"
            ),
        }
        if is_flagged:
            flagged_features.append(feature_name)

    n_features = len(feature_data)
    return {
        "check": "feature_null_rates",
        "features_checked": n_features,
        "features_flagged": len(flagged_features),
        "flagged_features": flagged_features,
        "feature_details": feature_report,
        "flag": len(flagged_features) > 0,
        "severity": (
            "HIGH" if len(flagged_features) > n_features * 0.2
            else "MEDIUM" if flagged_features
            else "OK"
        ),
    }


def check_feature_distribution_drift(
    training_stats: Dict[str, Dict],
    current_stats: Dict[str, Dict],
    z_score_threshold: float = 3.0,
) -> Dict:
    """
    Compares mean and std of numeric features at inference time against
    training-time statistics using a population z-score.

    A z-score above z_score_threshold indicates the feature mean has
    shifted significantly from what the model saw during training.
    Common causes: user population shift, upstream schema change, or a
    broken normalization step in the serving pipeline.

    Limitations:
        This check compares only the mean, not the full distribution shape.
        It will miss cases where the mean is stable but variance changed,
        or where the distribution became bimodal. For those cases, use PSI.
        Features with training std == 0 are skipped (no meaningful z-score).

    Args:
        training_stats:    Dict of {feature_name: {"mean": float, "std": float}}
                           computed at training data collection time.
        current_stats:     Same structure, measured from recent inference payloads.
        z_score_threshold: Z-score above which a feature is flagged MEDIUM.
                           Defaults to 3.0. Tighten to 2.0 for sensitive products.

    Returns:
        Dict with keys: check, features_checked, features_drifted,
        drifted_features, feature_details (per-feature z-scores), flag, severity.
    """
    drift_report = {}
    drifted = []

    for feat, train_stat in training_stats.items():
        if feat not in current_stats:
            continue

        train_std = train_stat.get("std", 0)
        if train_std == 0:
            # Cannot compute z-score for a constant feature. Skip silently.
            continue

        curr_mean = current_stats[feat]["mean"]
        train_mean = train_stat["mean"]
        z = abs(curr_mean - train_mean) / train_std

        is_drifted = z > z_score_threshold
        drift_report[feat] = {
            "train_mean": train_mean,
            "current_mean": curr_mean,
            "z_score": round(z, 3),
            "flag": is_drifted,
            "severity": (
                "HIGH" if z > 5.0
                else "MEDIUM" if is_drifted
                else "OK"
            ),
        }
        if is_drifted:
            drifted.append(feat)

    return {
        "check": "distribution_drift",
        "features_checked": len(training_stats),
        "features_drifted": len(drifted),
        "drifted_features": drifted,
        "feature_details": drift_report,
        "flag": len(drifted) > 0,
        "severity": "HIGH" if drifted else "OK",
    }
