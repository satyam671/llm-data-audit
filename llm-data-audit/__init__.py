"""
llm_data_audit
==============
Three-zone diagnostic framework for auditing the data layer
beneath an LLM product.

    Zone 1 — Training data quality  (zone1_training.py)
    Zone 2 — Retrieval corpus       (zone2_retrieval.py)
    Zone 3 — Feature reliability    (zone3_features.py)

The LLMDataAuditor class (auditor.py) composes all three zones
and generates the final JSON diagnostic report.
"""

from .auditor import LLMDataAuditor

__all__ = ["LLMDataAuditor"]
