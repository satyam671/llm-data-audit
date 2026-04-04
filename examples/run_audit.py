"""
run_audit.py
------------
End-to-end example demonstrating all three audit zones with synthetic data.

Run from the project root:
    python examples/run_audit.py

This writes audit_report.json to the current working directory and prints
a summary to stdout. Replace the synthetic data with real data from your
own pipeline to run a genuine diagnostic.

What this example simulates:
    Zone 1 — A training corpus with 20% duplicate samples, 38% of documents
              older than 365 days, and a fine-tuning label set with 12% noise
              from casing errors and free-text labels.

    Zone 2 — A retrieval corpus where 61% of documents are outside the 30-day
              freshness window, and the vector index covers only 85% of the
              source corpus (150 documents missing from the index).

    Zone 3 — A feature payload where last_query_embedding_norm has an 18% null
              rate, and user_tenure_days has drifted dramatically from its
              training-time distribution (mean dropped from 180 to 42 days,
              indicating a wave of new-user onboarding since the last training run).

Expected output: overall status RED (2 HIGH, 3 MEDIUM findings).
"""

import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_data_audit import LLMDataAuditor


def main():
    random.seed(42)
    auditor = LLMDataAuditor(report_path="audit_report.json")

    # ------------------------------------------------------------------
    # Zone 1: Training Data Quality
    # ------------------------------------------------------------------

    # Simulate a corpus where 20% of samples are exact duplicates.
    # A common cause: multiple export jobs pulling from the same source
    # without deduplication, or appending to an existing dataset rather
    # than replacing it.
    samples = (
        ["The product requires a valid license key."] * 200
        + [f"Training sample number {i}" for i in range(800)]
    )
    auditor.check_training_data_duplication(samples)

    # Simulate a corpus where 38% of samples are older than 365 days.
    # Replace these with real document creation dates from your data store.
    training_dates = [
        datetime.now() - timedelta(days=random.randint(0, 600))
        for _ in range(1000)
    ]
    auditor.check_training_data_temporal_shift(
        training_dates, max_age_days=365
    )

    # Simulate a fine-tuning label set with 12% noise from casing errors
    # and free-text values that bypassed enum validation.
    all_labels = (
        ["positive", "negative", "neutral"] * 300
        + ["POSITIVE", "NEG", "unknown", "pos"] * 50
    )
    auditor.check_label_noise(
        all_labels, valid_labels=["positive", "negative", "neutral"]
    )

    # ------------------------------------------------------------------
    # Zone 2: Retrieval Corpus Freshness
    # ------------------------------------------------------------------

    # Simulate a retrieval corpus where 61% of documents are older than
    # the 30-day freshness window. Replace with real index timestamps
    # from your vector store (Pinecone, Weaviate, pgvector, etc).
    doc_timestamps = [
        datetime.now() - timedelta(days=random.randint(0, 120))
        for _ in range(500)
    ]
    auditor.check_corpus_staleness(doc_timestamps, freshness_window_days=30)

    # Simulate an index that covers only 85% of the source corpus.
    # 150 documents exist in the knowledge base but were never indexed,
    # typically because a batch sync job failed silently.
    source_ids = [f"doc_{i}" for i in range(1000)]
    indexed_ids = [f"doc_{i}" for i in range(850)]
    auditor.check_index_coverage(indexed_ids, source_ids)

    # ------------------------------------------------------------------
    # Zone 3: Real-Time Feature Reliability
    # ------------------------------------------------------------------

    # Simulate a feature payload where last_query_embedding_norm has an
    # 18% null rate, indicating the embedding step in the serving pipeline
    # is failing for roughly 1 in 5 requests.
    feature_data = {
        "user_tenure_days": [
            random.randint(0, 1000) if random.random() > 0.03 else None
            for _ in range(500)
        ],
        "last_query_embedding_norm": [
            random.uniform(0.9, 1.1) if random.random() > 0.18 else None
            for _ in range(500)
        ],
        "account_tier": [
            random.choice(["free", "pro", "enterprise"])
            if random.random() > 0.02
            else None
            for _ in range(500)
        ],
    }
    auditor.check_feature_null_rates(feature_data)

    # Simulate user_tenure_days drift: mean dropped from 180 to 42 days
    # after a large new-user onboarding campaign. The model was trained on
    # a mostly-tenured user base and now serves a mostly-new one.
    # z = |180 - 42| / 30 = 4.6 → flags MEDIUM (above 3.0 threshold).
    training_stats = {
        "user_tenure_days": {"mean": 180.0, "std": 30.0}
    }
    current_stats = {
        "user_tenure_days": {"mean": 42.0, "std": 38.0}
    }
    auditor.check_feature_distribution_drift(training_stats, current_stats)

    # ------------------------------------------------------------------
    # Generate report
    # ------------------------------------------------------------------
    auditor.generate_report()


if __name__ == "__main__":
    main()
