"""
auditor.py
----------
LLMDataAuditor: the main class that composes all three zone checks
and generates the final JSON diagnostic report.

This class is a thin orchestration layer. The actual check logic lives
in zone1_training.py, zone2_retrieval.py, and zone3_features.py.
Import and call those functions directly if you only need individual checks.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional

from .zone1_training import (
    check_duplication,
    check_temporal_shift,
    check_label_noise,
)
from .zone2_retrieval import (
    check_corpus_staleness,
    check_index_coverage,
)
from .zone3_features import (
    check_feature_null_rates,
    check_feature_distribution_drift,
)


class LLMDataAuditor:
    """
    Three-zone diagnostic framework for the data layer beneath an LLM product.

    Instantiate once per audit run. Call any combination of check methods
    across the three zones, then call generate_report() to collate findings
    and write audit_report.json.

    Args:
        report_path: File path for the JSON output report.
                     Defaults to "audit_report.json" in the working directory.

    Example:
        auditor = LLMDataAuditor()
        auditor.check_training_data_duplication(samples)
        auditor.check_corpus_staleness(doc_timestamps)
        auditor.check_feature_null_rates(feature_data)
        report = auditor.generate_report()
    """

    def __init__(self, report_path: str = "audit_report.json"):
        self.report_path = report_path
        self.findings: Dict = {
            "training_data": [],
            "retrieval_corpus": [],
            "feature_reliability": [],
            "summary": {},
        }

    # ---- Zone 1: Training Data ----

    def check_training_data_duplication(
        self,
        samples: List[str],
        threshold: float = 0.10,
    ) -> Dict:
        """See zone1_training.check_duplication for full documentation."""
        finding = check_duplication(samples, threshold)
        self.findings["training_data"].append(finding)
        return finding

    def check_training_data_temporal_shift(
        self,
        sample_dates: List[datetime],
        eval_date: Optional[datetime] = None,
        max_age_days: int = 365,
    ) -> Dict:
        """See zone1_training.check_temporal_shift for full documentation."""
        finding = check_temporal_shift(sample_dates, eval_date, max_age_days)
        self.findings["training_data"].append(finding)
        return finding

    def check_label_noise(
        self,
        labels: List[str],
        valid_labels: List[str],
    ) -> Dict:
        """See zone1_training.check_label_noise for full documentation."""
        finding = check_label_noise(labels, valid_labels)
        self.findings["training_data"].append(finding)
        return finding

    # ---- Zone 2: Retrieval Corpus ----

    def check_corpus_staleness(
        self,
        doc_timestamps: List[datetime],
        freshness_window_days: int = 30,
    ) -> Dict:
        """See zone2_retrieval.check_corpus_staleness for full documentation."""
        finding = check_corpus_staleness(doc_timestamps, freshness_window_days)
        self.findings["retrieval_corpus"].append(finding)
        return finding

    def check_index_coverage(
        self,
        indexed_doc_ids: List[str],
        source_doc_ids: List[str],
    ) -> Dict:
        """See zone2_retrieval.check_index_coverage for full documentation."""
        finding = check_index_coverage(indexed_doc_ids, source_doc_ids)
        self.findings["retrieval_corpus"].append(finding)
        return finding

    # ---- Zone 3: Feature Reliability ----

    def check_feature_null_rates(
        self,
        feature_data: Dict[str, List],
        null_threshold: float = 0.05,
    ) -> Dict:
        """See zone3_features.check_feature_null_rates for full documentation."""
        finding = check_feature_null_rates(feature_data, null_threshold)
        self.findings["feature_reliability"].append(finding)
        return finding

    def check_feature_distribution_drift(
        self,
        training_stats: Dict[str, Dict],
        current_stats: Dict[str, Dict],
        z_score_threshold: float = 3.0,
    ) -> Dict:
        """See zone3_features.check_feature_distribution_drift for full docs."""
        finding = check_feature_distribution_drift(
            training_stats, current_stats, z_score_threshold
        )
        self.findings["feature_reliability"].append(finding)
        return finding

    # ---- Report ----

    def generate_report(self) -> Dict:
        """
        Collates all findings, computes an overall status, and writes
        the diagnostic report to self.report_path.

        Overall status:
            GREEN  — no checks flagged
            AMBER  — at least one MEDIUM severity finding (monitor closely)
            RED    — at least one HIGH severity finding (act before next deploy)

        Returns:
            The complete findings dict (same content as the JSON file).
        """
        all_findings = (
            self.findings["training_data"]
            + self.findings["retrieval_corpus"]
            + self.findings["feature_reliability"]
        )

        high_count = sum(
            1 for f in all_findings if f.get("severity") == "HIGH"
        )
        medium_count = sum(
            1 for f in all_findings if f.get("severity") == "MEDIUM"
        )

        self.findings["summary"] = {
            "generated_at": datetime.now().isoformat(),
            "total_checks": len(all_findings),
            "high_severity": high_count,
            "medium_severity": medium_count,
            "overall_status": (
                "RED" if high_count > 0
                else "AMBER" if medium_count > 0
                else "GREEN"
            ),
        }

        with open(self.report_path, "w") as f:
            json.dump(self.findings, f, indent=2, default=str)

        status = self.findings["summary"]["overall_status"]
        print(f"\n{'=' * 52}")
        print(f"  LLM Data Layer Audit")
        print(f"  Status     : {status}")
        print(f"  HIGH issues: {high_count}")
        print(f"  MEDIUM     : {medium_count}")
        print(f"  Report     : {self.report_path}")
        print(f"{'=' * 52}\n")

        return self.findings
