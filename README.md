# llm-data-audit

A structured, three-zone diagnostic framework for auditing the data layer beneath an LLM product — before you blame the model.

Companion code for the article [**"Garbage In, Hallucination Out: The Data Quality Problems Sitting Upstream of Every LLM Failure"**](https://medium.com/ai-advances/garbage-in-hallucination-out-the-data-quality-problems-sitting-upstream-of-every-llm-failure-08f0377ceb71) published on [AI Advances](https://ai.gopubby.com).

---

## The Three Zones

| Zone | What it audits | Failure it catches |
|---|---|---|
| Zone 1 | Training data quality | Duplication, temporal shift, label noise |
| Zone 2 | Retrieval corpus freshness | Stale documents, index gaps, coverage drift |
| Zone 3 | Real-time feature reliability | Null spikes, distribution shift |

---

## Repo Structure

```
llm-data-audit/
│
├── llm_data_audit/
│   ├── __init__.py           # Package exports
│   ├── auditor.py            # LLMDataAuditor: main class and report generation
│   ├── zone1_training.py     # Zone 1 checks: duplication, temporal shift, label noise
│   ├── zone2_retrieval.py    # Zone 2 checks: corpus staleness, index coverage
│   └── zone3_features.py     # Zone 3 checks: null rates, distribution drift
│
├── examples/
│   └── run_audit.py          # End-to-end example with synthetic data for all three zones
│
├── docs/
│   └── thresholds.md         # Reference card: what each threshold means and when to tighten it
│
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/llm-data-audit.git
cd llm-data-audit
pip install -r requirements.txt
python examples/run_audit.py
```

This runs all three zones against synthetic data and writes `audit_report.json` to the project root.

---

## Using It Against Your Own Data

```python
from llm_data_audit import LLMDataAuditor

auditor = LLMDataAuditor(report_path="audit_report.json")

# Zone 1 — Training data
auditor.check_training_data_duplication(your_training_samples)
auditor.check_training_data_temporal_shift(your_sample_dates, max_age_days=365)
auditor.check_label_noise(your_labels, valid_labels=["positive", "negative", "neutral"])

# Zone 2 — Retrieval corpus
auditor.check_corpus_staleness(your_doc_timestamps, freshness_window_days=30)
auditor.check_index_coverage(your_indexed_ids, your_source_ids)

# Zone 3 — Features
auditor.check_feature_null_rates(your_feature_payload)
auditor.check_feature_distribution_drift(training_stats, current_stats)

report = auditor.generate_report()
```

The report JSON includes per-check severity (`OK`, `MEDIUM`, `HIGH`) and an overall status (`GREEN`, `AMBER`, `RED`).

---

## Severity Thresholds (defaults)

| Check | MEDIUM | HIGH |
|---|---|---|
| Duplication rate | > 10% | > 20% |
| Temporal shift (stale rate) | > 30% | > 50% |
| Label noise | > 1% | > 5% |
| Corpus staleness | > 30% | > 50% |
| Index coverage | < 95% | < 80% |
| Feature null rate | > 5% | > 20% |
| Distribution drift (z-score) | > 3.0 | > 5.0 |

All thresholds are configurable via function arguments. See `docs/thresholds.md` for guidance on tightening them for specific domains.

---

## Notes on Production Use

The distribution drift check uses a simplified z-score approach. For production monitoring, replace it with **Population Stability Index (PSI)**. A PSI above 0.2 on any feature indicates drift worth acting on. The z-score check is appropriate for an initial one-off diagnostic audit; PSI is appropriate for continuous monitoring.

---

## Reference

- Longpre et al. (2023). *A Pretrainer's Guide to Training Data: Measuring the Effects of Data Age, Domain Coverage, Quality, and Toxicity.* arXiv:2305.13169
- Bommasani et al. (2021). *On the Opportunities and Risks of Foundation Models.* Stanford CRFM. arXiv:2108.07258

---

## License

MIT
