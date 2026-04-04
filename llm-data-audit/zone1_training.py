"""
zone1_training.py
-----------------
Zone 1 audit checks for training data quality.

Three checks:
    1. check_duplication      — exact-duplicate rate across training samples
    2. check_temporal_shift   — fraction of samples older than max_age_days
    3. check_label_noise      — out-of-vocabulary labels in fine-tuning data

Each function returns a finding dict with keys:
    check       str   — machine-readable check name
    flag        bool  — True if the check is above threshold
    severity    str   — "OK" | "MEDIUM" | "HIGH"
    ...         various check-specific metrics
"""

import hashlib
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional


def check_duplication(
    samples: List[str],
    threshold: float = 0.10,
) -> Dict:
    """
    Measures the exact-duplicate rate in a list of training samples.

    Computes an MD5 hash for each sample (lowercased, stripped) and
    compares the count of unique hashes against the total. Anything
    above `threshold` is flagged. Above 20% is HIGH severity.

    Duplicated text biases the model toward overrepresented patterns
    without any deliberate weighting decision being made. A corpus
    with 20% duplicates effectively up-weights those examples by ~1.25x.

    Args:
        samples:   Raw text samples from the training corpus.
        threshold: Duplication rate above which the check flags MEDIUM.
                   Defaults to 0.10 (10%).

    Returns:
        Dict with keys: check, total_samples, unique_samples,
        duplication_rate, flag, severity.
    """
    total = len(samples)
    hashes = [
        hashlib.md5(s.strip().lower().encode()).hexdigest()
        for s in samples
    ]
    unique = len(set(hashes))
    dup_rate = (total - unique) / total if total > 0 else 0.0

    return {
        "check": "duplication_rate",
        "total_samples": total,
        "unique_samples": unique,
        "duplication_rate": round(dup_rate, 4),
        "flag": dup_rate > threshold,
        "severity": (
            "HIGH" if dup_rate > 0.20
            else "MEDIUM" if dup_rate > threshold
            else "OK"
        ),
    }


def check_temporal_shift(
    sample_dates: List[datetime],
    eval_date: Optional[datetime] = None,
    max_age_days: int = 365,
) -> Dict:
    """
    Measures what fraction of training samples are older than max_age_days
    relative to the evaluation or deployment date.

    Longpre et al. (2023) showed that temporal shift between pretraining
    and evaluation data causes performance degradation that fine-tuning
    alone does not fully fix. Stale training data means stale parametric
    knowledge baked into the model weights.

    Args:
        sample_dates: Creation or collection datetime per training sample.
        eval_date:    Reference date for measuring staleness.
                      Defaults to datetime.now().
        max_age_days: Age threshold in days. 365 is a reasonable starting
                      point. Tighten to 90 for fast-moving domains
                      (regulatory, software, pricing).

    Returns:
        Dict with keys: check, total_samples, avg_age_days, stale_samples,
        stale_rate, max_age_days, flag, severity.
    """
    eval_date = eval_date or datetime.now()
    ages = [(eval_date - d).days for d in sample_dates]
    stale_count = sum(1 for age in ages if age > max_age_days)
    stale_rate = stale_count / len(ages) if ages else 0.0
    avg_age = sum(ages) / len(ages) if ages else 0.0

    return {
        "check": "temporal_shift",
        "total_samples": len(ages),
        "avg_age_days": round(avg_age, 1),
        "stale_samples": stale_count,
        "stale_rate": round(stale_rate, 4),
        "max_age_days": max_age_days,
        "flag": stale_rate > 0.30,
        "severity": (
            "HIGH" if stale_rate > 0.50
            else "MEDIUM" if stale_rate > 0.30
            else "OK"
        ),
    }


def check_label_noise(
    labels: List[str],
    valid_labels: List[str],
) -> Dict:
    """
    Checks for out-of-vocabulary or malformed labels in fine-tuning data.

    Any label not in valid_labels suggests a labeling pipeline issue:
    casing inconsistency, a broken enum, or accidental free-text values
    slipping through validation.

    In a pre-training corpus, label noise might get averaged away across
    billions of tokens. In a fine-tuning dataset of 10,000 samples, a
    1-5% noise rate has a visible effect on the gradient signal.

    Args:
        labels:       All label values from the fine-tuning dataset.
        valid_labels: The expected label vocabulary (exhaustive list).

    Returns:
        Dict with keys: check, total_labels, invalid_count, noise_rate,
        top_invalid_labels, flag, severity.
    """
    valid_set = set(valid_labels)
    invalid = [l for l in labels if l not in valid_set]
    noise_rate = len(invalid) / len(labels) if labels else 0.0
    invalid_counts = Counter(invalid)

    return {
        "check": "label_noise",
        "total_labels": len(labels),
        "invalid_count": len(invalid),
        "noise_rate": round(noise_rate, 4),
        "top_invalid_labels": dict(invalid_counts.most_common(5)),
        "flag": noise_rate > 0.01,
        "severity": (
            "HIGH" if noise_rate > 0.05
            else "MEDIUM" if noise_rate > 0.01
            else "OK"
        ),
    }
