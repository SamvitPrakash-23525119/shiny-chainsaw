# MetaForensics — AI-Assisted Metadata Analysis for Digital Forensics

**COS783 Digital Forensics · University of Pretoria · 2026 Final Assignment**

> **Assignment Option 2 — Metadata Analysis:**  
> *"AI techniques can assist in analyzing large volumes of metadata by automating pattern recognition, anomaly detection, and correlation analysis."*

---

## Overview

MetaForensics is an insider-threat detection platform that demonstrates how AI can be applied to **metadata analysis** in digital forensics investigations. The system ingests raw behavioural metadata records for employees, aggregates them into per-user profiles, and applies two complementary AI techniques:

| Technique | Method | Purpose |
|---|---|---|
| **Anomaly Detection** | Isolation Forest (sklearn, 200 trees) | Identify users whose behaviour deviates from the norm |
| **Correlation Analysis** | Pearson r · Spearman ρ · p-values | Reveal statistical relationships between metadata features and anomalous behaviour |
| **Collusion Detection** | Cross-user Pearson correlation | Find pairs of users with suspiciously similar behavioural profiles |

A **Flask web application** with interactive Plotly charts serves all results through a modern forensic-analysis dashboard.

---

## Project Structure

```
shiny-chainsaw/
│
├── app.py                          ← Flask entry point (run this)
│
├── src/
│   ├── CorrelationAnalysis/        ← NEW: correlation analysis module
│   │   ├── __init__.py
│   │   └── correlation_analyzer.py ← Core AI analysis engine
│   │
│   ├── IsolationForest/            ← Custom Isolation Forest (from scratch)
│   │   ├── config/                 ← File path configuration
│   │   ├── data/                   ← Dataset splitting utilities
│   │   ├── detection/              ← Custom IsolationForest + IsolationTree
│   │   ├── evaluation/             ← Precision, recall, F1, confusion matrix
│   │   ├── models/                 ← Dataclasses (Observation, UserEntity, etc.)
│   │   ├── pipeline/               ← ThreatDetectionPipeline orchestrator
│   │   ├── preprocessing/          ← Feature aggregation & vectorisation
│   │   ├── training/               ← Hyperparameter search (100 iterations)
│   │   └── utils/                  ← CSV loader, pickler, c-factor
│   │
│   ├── RandomForest/               ← Alternative sklearn RF baseline
│   │   └── insider_threat_rf.py    ← 200-tree RF, 95.25% accuracy
│   │
│   └── main.py                     ← CLI entry point (manual testing)
│
├── templates/                      ← NEW: Jinja2 HTML templates
│   ├── base.html                   ← Sidebar layout, Bootstrap 5, Plotly.js
│   ├── index.html                  ← Dashboard (stats, top anomalous users)
│   ├── correlation.html            ← Interactive Pearson/Spearman heatmap
│   ├── anomaly.html                ← Feature→anomaly correlation + scatter
│   └── colluders.html              ← Cross-user correlation (collusion detection)
│
├── static/
│   └── css/
│       └── style.css               ← NEW: Custom UI stylesheet
│
├── dataset/
│   ├── insider_threat_clean_dataset.csv  ← 118,614 observations, 22 features
│   ├── training_data.csv                 ← 70% split
│   └── testing_data.csv                  ← 30% split
│
├── models/                         ← Pre-trained Isolation Forest pickles
│   ├── last_trained_model.pkl
│   ├── balanced_model.pkl
│   ├── high_f1_model.pkl
│   ├── high_precision_model.pkl
│   └── high_recall_model.pkl
│
├── logs/
│   └── training_log.jsonl          ← Hyperparameter search log (100 runs)
│
├── requirements.txt
└── makefile
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the web application
```bash
python app.py
```

Then open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser.

> The first page load takes a few seconds — the system processes 118,614 raw observations and trains the IsolationForest model at startup.

### 3. Alternative CLI entry point
```bash
python -m src.main
```

---

## Dataset

| Property | Value |
|---|---|
| Source | Synthetic insider-threat behavioural dataset |
| Raw observations | 118,614 rows |
| Unique user profiles | 1,654 |
| Malicious actors | 162 (9.8%) |
| Features per observation | 22 columns |
| Aggregated features per user | 17 |

### Raw columns (22)
`employee_department`, `employee_campus`, `employee_position`, `employee_seniority_years`,
`is_contractor`, `employee_classification`, `has_foreign_citizenship`, `has_criminal_record`,
`has_medical_history`, `employee_origin_country`, `total_printed_pages`,
`num_printed_pages_off_hours`, `total_files_burned`, `burned_from_other`, `is_abroad`,
`trip_day_number`, `hostility_country_level`, `num_entries`, `num_unique_campus`,
`late_exit_flag`, `entry_during_weekend`, `is_malicious`

---

## Data Preprocessing

Raw observations are **aggregated per unique user identity** into 17 behavioural features:

| Feature | Computation |
|---|---|
| `employee_seniority_years` | Static per user |
| `is_contractor` | Static per user |
| `has_foreign_citizenship` | Static per user |
| `has_criminal_record` | Static per user |
| `has_medical_history` | Static per user |
| `avg_total_printed_pages` | Mean across observations |
| `avg_off_hours_print_ratio` | Total off-hours pages ÷ total pages |
| `avg_files_burned` | Mean `total_files_burned` |
| `avg_burned_from_other` | Mean `burned_from_other` |
| `avg_num_entries` | Mean `num_entries` |
| `avg_unique_campus_ratio` | Mean of `num_unique_campus ÷ num_entries` per observation |
| `late_exit_ratio` | Fraction of observations with late exit flag |
| `weekend_entry_ratio` | Fraction of observations with weekend entry |
| `avg_trip_duration` | Mean `trip_day_number` |
| `avg_hostility_country_level` | Mean `hostility_country_level` |
| `abroad_ratio` | Fraction of observations where user was abroad |
| `risk_score` | Weighted heuristic (see below) |

### Risk score formula

```
risk_score = 1.5 × has_criminal_record
           + 0.5 × is_contractor
           + 1.0 × (has_foreign_citizenship AND avg_hostility_country_level > 0)
           + 2.0 × avg_off_hours_print_ratio
           + log1p(avg_files_burned)
```

---

## AI Capability: Anomaly Detection

**Algorithm:** Isolation Forest (scikit-learn, 200 estimators)

The Isolation Forest isolates anomalies by randomly selecting a feature and a split value. Anomalous observations require fewer splits to isolate (shorter average path length through the trees). The anomaly score is the negated decision function — higher values indicate more anomalous users.

**Training configuration:**
- `n_estimators = 200`
- `contamination = 0.05` (5% of users expected to be anomalous)
- `max_samples = 'auto'`
- Features standardised with `StandardScaler` before fitting

**Output:**
- Anomaly score per user (range approximately −0.22 to +0.15 on this dataset)
- Users above the contamination threshold are flagged as potential threats

---

## AI Capability: Correlation Analysis

Three complementary correlation analyses reveal which metadata features matter most for forensic investigators.

### 1. Feature–Feature Correlation Matrix

**Pearson r** (linear relationships) and **Spearman ρ** (rank-order / non-linear) are computed between all pairs of the 17 aggregated features, producing a 17×17 matrix.

Example finding: `avg_trip_duration` and `abroad_ratio` have r = 0.966 — strongly correlated, as users who travel further tend to spend more time abroad.

### 2. Feature–Anomaly Score Correlation

For each of the 17 features, Pearson r and Spearman ρ are computed against the Isolation Forest anomaly score. P-values are calculated and features with p < 0.05 are flagged as statistically significant predictors of anomalous behaviour.

Top predictors on this dataset:
| Rank | Feature | Pearson r | Significant |
|---|---|---|---|
| 1 | Files Burned (Others) | 0.5642 | ✓ |
| 2 | Weekend Entry Ratio | 0.4579 | ✓ |
| 3 | Off-hours Print Ratio | 0.3764 | ✓ |
| 4 | Avg Access Entries | 0.3524 | ✓ |
| 5 | Abroad Ratio | 0.3415 | ✓ |

### 3. Cross-User Behavioural Correlation (Collusion Detection)

For every pair of users, Pearson r is computed between their 17-feature profile vectors using `numpy.corrcoef` (vectorised, handles thousands of users efficiently). Pairs above a configurable threshold are flagged as potentially coordinating (colluding) insiders.

A two-tailed t-test with df = 15 (n_features − 2) provides p-values for each pair.

Example finding at r ≥ 0.97: 1,673 flagged pairs, top pair has r = 0.9978.

---

## Web Application

The Flask app exposes four pages and five JSON API endpoints.

### Pages

| URL | Page | Description |
|---|---|---|
| `/` | Dashboard | Stat cards, top-5 anomalous users, anomaly score histogram |
| `/correlation` | Feature Correlation | Interactive Pearson/Spearman heatmap + strongest pairs table |
| `/anomaly` | Anomaly Analysis | Feature→anomaly bar chart, risk vs anomaly scatter, significance table |
| `/colluders` | Collusion Detection | Threshold slider, correlated user-pair table, distribution histogram |

### API endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/correlation-matrix?method=pearson` | GET | 17×17 correlation matrix as JSON |
| `/api/anomaly-correlation` | GET | Feature→anomaly correlations (sorted by \|r\|) |
| `/api/anomaly-scores` | GET | Per-user anomaly scores, risk scores, labels |
| `/api/colluders?threshold=0.85` | GET | Correlated user pairs above threshold |

---

## Existing Modules

### Custom Isolation Forest (`src/IsolationForest/`)

A complete from-scratch implementation including:
- `IsolationTree` — random feature selection and split, path-length computation
- `IsolationForest` — ensemble of trees, c-factor normalisation, anomaly scoring
- `ThreatDetectionPipeline` — end-to-end orchestration from raw CSV to predictions
- `Trainer` — 100-iteration hyperparameter search (n_trees, sampling_size, threshold)
- `Evaluator` — precision, recall, F1, confusion matrix

Pre-trained models are stored as pickle files in `models/`.

### Random Forest Baseline (`src/RandomForest/insider_threat_rf.py`)

- 200-tree sklearn Random Forest, class-weight balanced
- 95.25% accuracy, 91.04% precision, 82.43% recall, 0.9896 ROC-AUC
- Includes feature importance analysis and risk indicator flags

---

## How to Extend the Project

1. Create a directory inside `src/`:
```
src/
└── YourModule/
    ├── __init__.py
    └── your_code.py
```

2. Import in `app.py` or `src/main.py`:
```python
from src.YourModule.your_code import YourClass
```

3. Add a new route to `app.py` and a template to `templates/` following the existing patterns.

---

## Key Results

| Metric | Value |
|---|---|
| Total users profiled | 1,654 |
| Malicious actors detected | 162 (9.8%) |
| Raw observations processed | 118,614 |
| Top anomaly predictor | Files Burned from Other Users (r = 0.564) |
| Strongest feature correlation | Trip Duration ↔ Abroad Ratio (r = 0.966) |
| Correlated pairs at r ≥ 0.97 | 1,673 |

---

## Dependencies

Key packages (see `requirements.txt` for pinned versions):

| Package | Version | Purpose |
|---|---|---|
| `flask` | 3.1.1 | Web framework |
| `scikit-learn` | 1.8.0 | IsolationForest, StandardScaler |
| `scipy` | 1.17.1 | Pearson r, Spearman ρ, t-distribution p-values |
| `pandas` | 3.0.3 | Data loading and aggregation |
| `numpy` | 2.4.5 | Matrix operations, corrcoef |
| `plotly` | 6.7.0 | Interactive charts (server-side JSON) |

---

## Running with Make

```bash
make install   # pip install -r requirements.txt
make run       # python -m src.main  (CLI entry point)
```

To run the Flask app:
```bash
python app.py
```
