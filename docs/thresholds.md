# Thresholds Reference Card

Default thresholds in `llm_data_audit` are starting points, not fixed rules.
Every domain has a different tolerance for staleness, duplication, and drift.
This document explains what each threshold means and when to tighten it.

---

## Zone 1 — Training Data

### Duplication rate
| Severity | Default | When to tighten |
|---|---|---|
| MEDIUM | > 10% | Tighten to 5% for fine-tuning datasets under 50K samples |
| HIGH   | > 20% | Keep at 20% — anything above this is a pipeline problem |

A 10% duplication rate in a 1M-sample pre-training corpus is tolerable.
In a 5,000-sample fine-tuning dataset, even 5% duplicates are worth removing.
The smaller the dataset, the more each duplicated example distorts gradients.

### Temporal shift (stale rate)
| Severity | Default | Fast-moving domains | Stable domains |
|---|---|---|---|
| MEDIUM | > 30% stale | > 10% stale | > 50% stale |
| HIGH   | > 50% stale | > 25% stale | > 70% stale |

`max_age_days` defaults to 365. Adjust by domain:
- Financial data, release notes, regulatory guidance: 30–90 days
- Product documentation, support articles: 90–180 days
- Academic reference, classical literature: 365–730 days

### Label noise
| Severity | Default | Notes |
|---|---|---|
| MEDIUM | > 1% | Flag for review |
| HIGH   | > 5% | Halt fine-tuning run until source is investigated |

Label noise above 5% in a fine-tuning dataset means roughly 1 in 20 gradient
updates is anchored to a wrong signal. At 10K samples, that's 500 corrupted
examples. The effect compounds over epochs.

---

## Zone 2 — Retrieval Corpus

### Corpus staleness
| Severity | Default | When to tighten |
|---|---|---|
| MEDIUM | > 30% outside freshness window | Tighten to 15% for compliance/legal domains |
| HIGH   | > 50%                           | Keep — above 50% stale is almost always a sync problem |

`freshness_window_days` defaults to 30. Domain guidance:
- Breaking news, financial ticks: 1–7 days
- Product changelog, release notes: 7–14 days
- Policy documents, support KB: 30–60 days
- Historical reference: no staleness concern

### Index coverage
| Severity | Default | Notes |
|---|---|---|
| MEDIUM | < 95% coverage | Below 95%, users will silently miss relevant documents |
| HIGH   | < 80% coverage | Below 80%, retrieval quality is systemically degraded |

Orphaned documents (in index but not in source) don't trigger a flag by default,
but are reported in the output. Orphaned count > 5% of indexed docs is worth
investigating — it suggests the deletion/sync pipeline is broken.

---

## Zone 3 — Feature Reliability

### Feature null rates
| Severity | Default | Notes |
|---|---|---|
| MEDIUM | > 5% null per feature  | Check the upstream feature pipeline for that feature |
| HIGH   | > 20% null per feature | Feature is effectively unavailable at inference time |

Features that were never null in training but show any null rate in production
are worth flagging regardless of the threshold, because any null means the
model is receiving a different input distribution than it was trained on.

Consider adding a hard check: `assert null_rate == 0` for features your model
treats as required inputs. If the feature is truly required and it's null,
the serving pipeline should reject the request, not silently pass None.

### Distribution drift (z-score)
| Severity | Default | When to tighten |
|---|---|---|
| MEDIUM | z > 3.0 | Tighten to z > 2.0 for high-stakes products |
| HIGH   | z > 5.0 | Keep — z > 5 is almost always a real shift |

The z-score check compares means only. It will miss:
- Variance changes (mean stable, distribution wider/narrower)
- Shape changes (distribution becoming bimodal)
- Tail behavior changes

For production monitoring, replace the z-score check with **Population
Stability Index (PSI)**:
- PSI < 0.1 — negligible shift, no action
- PSI 0.1–0.2 — moderate shift, investigate
- PSI > 0.2 — significant shift, retrain or recalibrate

A simple PSI implementation requires `numpy`. The z-score check here is
intentionally dependency-free for portability.

---

## Audit Cadence

| Zone | Check | Recommended cadence |
|---|---|---|
| Training | Duplication | Before each training or fine-tuning run |
| Training | Temporal shift | Before each training run, and quarterly |
| Training | Label noise | Before each fine-tuning run |
| Retrieval | Corpus staleness | Daily (fast-moving domains), weekly (stable) |
| Retrieval | Index coverage | Daily, or triggered on source corpus update |
| Features | Null rates | Continuously in production (sample every N requests) |
| Features | Distribution drift | Weekly, or triggered on upstream pipeline changes |
