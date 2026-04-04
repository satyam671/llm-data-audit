"""
zone2_retrieval.py
------------------
Zone 2 audit checks for retrieval corpus freshness.

Two checks:
    1. check_corpus_staleness  — age distribution of indexed documents
    2. check_index_coverage    — gap between source docs and indexed docs

Each function returns a finding dict with keys:
    check       str   — machine-readable check name
    flag        bool  — True if the check is above/below threshold
    severity    str   — "OK" | "MEDIUM" | "HIGH"
    ...         various check-specific metrics

Context:
    Research on ETL-driven RAG systems shows that LLMs relying purely on
    parametric memory top out at ~34% accuracy on temporally sensitive queries.
    Basic RAG with a stale corpus improves this to ~44%. Optimized retrieval
    with a fresh corpus can reach ~63%. The gap between "we have RAG" and
    "we have good RAG" is largely a corpus freshness and coverage problem.
"""

from datetime import datetime
from typing import Dict, List


def check_corpus_staleness(
    doc_timestamps: List[datetime],
    freshness_window_days: int = 30,
) -> Dict:
    """
    Checks the age distribution of documents in the RAG retrieval corpus.

    Computes what fraction of indexed documents fall outside the freshness
    window. A corpus where more than 30% of documents are stale will cause
    the model to retrieve outdated context for a meaningful fraction of
    queries — producing confident answers grounded in superseded information.

    Freshness window guidance by domain:
        7-14  days  — financial market data, breaking news, release notes
        30    days  — product docs, regulatory guidance, support KB articles
        60-90 days  — stable technical reference, research summaries

    Args:
        doc_timestamps:       Timestamp of last update per indexed document.
        freshness_window_days: Maximum acceptable document age in days.

    Returns:
        Dict with keys: check, total_docs, median_age_days, stale_docs,
        stale_rate, freshness_window_days, flag, severity.
    """
    now = datetime.now()
    ages = [(now - ts).days for ts in doc_timestamps]
    stale_docs = [a for a in ages if a > freshness_window_days]
    stale_rate = len(stale_docs) / len(ages) if ages else 0.0
    sorted_ages = sorted(ages)
    median_age = sorted_ages[len(sorted_ages) // 2] if sorted_ages else 0

    return {
        "check": "corpus_staleness",
        "total_docs": len(ages),
        "median_age_days": median_age,
        "stale_docs": len(stale_docs),
        "stale_rate": round(stale_rate, 4),
        "freshness_window_days": freshness_window_days,
        "flag": stale_rate > 0.30,
        "severity": (
            "HIGH" if stale_rate > 0.50
            else "MEDIUM" if stale_rate > 0.30
            else "OK"
        ),
    }


def check_index_coverage(
    indexed_doc_ids: List[str],
    source_doc_ids: List[str],
) -> Dict:
    """
    Compares document IDs currently in the vector index against the
    authoritative source corpus.

    Two failure modes:
        missing   — source documents not yet indexed. The retriever
                    silently fails to surface these; the model falls
                    back to parametric memory and often hallucinates.
        orphaned  — documents in the index that no longer exist in
                    the source. Wastes token budget and may surface
                    content that has been deliberately removed.

    Coverage below 95% is worth investigating. Below 80% is a HIGH
    severity issue that actively degrades retrieval quality.

    Args:
        indexed_doc_ids: Document IDs currently present in the vector store.
        source_doc_ids:  Document IDs in the authoritative source system.

    Returns:
        Dict with keys: check, source_docs, indexed_docs,
        missing_from_index, orphaned_in_index, coverage_rate, flag, severity.
    """
    indexed_set = set(indexed_doc_ids)
    source_set = set(source_doc_ids)
    missing = source_set - indexed_set
    orphaned = indexed_set - source_set
    coverage_rate = (
        len(indexed_set & source_set) / len(source_set)
        if source_set else 0.0
    )

    return {
        "check": "index_coverage",
        "source_docs": len(source_set),
        "indexed_docs": len(indexed_set),
        "missing_from_index": len(missing),
        "orphaned_in_index": len(orphaned),
        "coverage_rate": round(coverage_rate, 4),
        "flag": coverage_rate < 0.95,
        "severity": (
            "HIGH" if coverage_rate < 0.80
            else "MEDIUM" if coverage_rate < 0.95
            else "OK"
        ),
    }
