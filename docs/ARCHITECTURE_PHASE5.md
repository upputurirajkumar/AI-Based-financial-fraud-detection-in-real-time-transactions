# Phase 5: Transaction Intelligence, Alerts, Investigation & Analytics Architecture

## 1. End-to-End Lifecycle Architecture

```text
       Raw Transaction (API / Batch CSV / Core Banking)
                             │
                             ▼
                    [ Transaction Domain ]
               (Canonical Entity: TXN-XXXXXX)
                             │
                             ▼
                    [ ML Inference Engine ]
               (Production Random Forest v1.0.0)
                             │
                             ▼
                   [ Prediction Record ]
           (Immutable Snapshot: PRD-XXXXXX, τ=0.23)
                             │
                             ▼
                  [ Risk Intelligence Layer ]
        (Score 0–100, Tiers: LOW, MEDIUM, HIGH, CRITICAL)
                             │
                             ▼
               [ Alert Generation & Dedup ]
     (Deduplication Hash: sha256(tx_id:model_ver:rule_name))
                             │
                             ▼
                   [ Fraud Alert Queue ]
          (NEW -> UNDER_REVIEW -> RESOLVED / DISMISSED)
                             │
                             ▼
                [ Case Management (Human XAI) ]
                 (Assigned Compliance Analyst)
                             │
                             ▼
                 [ Ground-Truth Resolution ]
   (CONFIRMED_FRAUD / FALSE_POSITIVE / SUSPICIOUS_ACTIVITY)
                             │
                             ▼
                [ Analytics & Reporting Engine ]
   (Strict Separation: Predicted Fraud Rate vs Confirmed Fraud Rate)
```

---

## 2. Lifecycles & State Machines

### Alert Lifecycle
- **NEW:** Alert automatically generated from deterministic rule or critical risk score.
- **ACKNOWLEDGED:** Analyst acknowledged receipt of the alert.
- **UNDER_REVIEW:** Alert linked to active investigation case.
- **ESCALATED:** Escalated to senior compliance officer or fraud operations supervisor.
- **RESOLVED:** Investigation concluded with formal disposition.
- **DISMISSED:** Determined to be a benign edge case or operational exception.

### Investigation Case Lifecycle
- **OPEN:** Case initiated by analyst or automatically generated from high-risk batch.
- **UNDER_REVIEW:** Evidence analysis, customer contact, and account ledger review underway.
- **ESCALATED:** Complex cross-account mule layering referred to financial crimes unit.
- **RESOLVED:** Formal ground-truth determination recorded.
- **CLOSED:** Administrative closure.

### Ground-Truth Resolution Categories
1. `CONFIRMED_FRAUD`: Verified unauthorized transaction / account takeover / synthetic identity.
2. `FALSE_POSITIVE`: Verified legitimate transaction conducted by authorized account holder.
3. `SUSPICIOUS_ACTIVITY`: Suspicious activity warranting Suspicious Activity Report (SAR) filing without definitive customer dispute.
4. `NO_ISSUE_FOUND`: Benign anomaly explained by legitimate customer profile change.
5. `INCONCLUSIVE`: Insufficient evidence to establish definitive determination.

---

## 3. Mathematical Metric Definitions

1. **Predicted Fraud Rate (%):**
   $$\text{Predicted Fraud Rate} = \frac{\text{Count of Transactions where } p_{\text{fraud}} \ge \tau \text{ or Risk Tier is CRITICAL}}{\text{Total Evaluated Transactions}} \times 100$$
2. **Confirmed Fraud Rate (%):**
   $$\text{Confirmed Fraud Rate} = \frac{\text{Count of Investigation Cases with Resolution } \texttt{CONFIRMED\_FRAUD}}{\text{Total Resolved Investigation Cases}} \times 100$$
3. **False Positive Rate (%):**
   $$\text{False Positive Rate} = \frac{\text{Count of Investigation Cases with Resolution } \texttt{FALSE\_POSITIVE}}{\text{Total Resolved Investigation Cases}} \times 100$$

*Strict Principle:* Machine learning model predictions represent statistical likelihoods and decision-support signals. Only human review or verified chargeback ground truth can produce a confirmed fraud outcome.

---

## 4. RESTful API Contract Reference

| Endpoint | Method | Role Required | Description |
|---|---|---|---|
| `/api/transactions/` | GET | USER+ | Search & filter transactions with pagination |
| `/api/transactions/<tx_id>/` | GET | USER+ | Full transaction intelligence object |
| `/api/transactions/ingest/` | POST | USER+ | Ingest & evaluate single transaction |
| `/api/alerts/` | GET | ANALYST, ADMIN | Review queue with severity & status filters |
| `/api/alerts/<id>/transition/` | PATCH | ANALYST, ADMIN | Transition alert state machine |
| `/api/cases/` | GET, POST | ANALYST, ADMIN | List or open investigation cases |
| `/api/cases/<id>/` | GET | ANALYST, ADMIN | Case details with notes and evidence |
| `/api/cases/<id>/notes/` | POST | ANALYST, ADMIN | Append permanent investigation note |
| `/api/cases/<id>/resolve/` | PATCH | ANALYST, ADMIN | Ground-truth case resolution |
| `/api/analytics/overview/` | GET | USER+ | Volume, predicted vs confirmed fraud rate |
| `/api/analytics/risk-distribution/` | GET | USER+ | Risk tier counts & percentages |
| `/api/analytics/time-series/` | GET | USER+ | Hourly/daily time series metrics |
| `/api/analytics/models/` | GET | USER+ | Multi-model version analytics |
| `/api/reports/summary/` | GET | USER+ | Comprehensive summary report |
| `/api/reports/export/` | GET | ANALYST, ADMIN | Role-governed CSV data export |
