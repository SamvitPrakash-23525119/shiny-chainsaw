# MetaForensics — AI-Assisted Metadata Analysis for Digital Forensics

**COS783 Digital Forensics · University of Pretoria · 2026 Final Assignment**

> **Assignment Option 2 — Metadata Analysis:**  
> *"AI techniques can assist in analyzing large volumes of metadata by automating pattern recognition, anomaly detection, and correlation analysis."*

---

## Overview

MetaForensics is an insider-threat detection platform that demonstrates how AI can be applied to **metadata analysis** in digital forensics investigations. The system ingests raw behavioural metadata records for employees, aggregates them into per-user profiles, and applies three complementary AI techniques:

| Technique | Method | Purpose |
|---|---|---|
| **Anomaly Detection** | Isolation Forest (sklearn, 200 trees) | Identify users whose behaviour deviates statistically from the norm |
| **Pattern Recognition** | Logistic Regression (sklearn, balanced) | Learn explicit behavioural signatures of malicious insiders; expose interpretable feature weights |
| **Correlation Analysis** | Pearson r · Spearman ρ · p-values | Reveal statistical relationships between metadata features and threat scores |
| **Collusion Detection** | Cross-user Pearson correlation (numpy) | Find pairs of users with suspiciously similar behavioural profiles |

A **Flask web application** with interactive Plotly.js charts serves all results through a modern forensic-analysis dashboard.

---

## Project Structure

```
shiny-chainsaw/
│
├── app.py                          ← Flask entry point (all routes + API)
├── run.bat                         ← Windows launcher (always runs from project root)
│
├── src/
│   ├── CorrelationAnalysis/        ← Core AI analysis engine (Flask-facing)
│   │   ├── __init__.py
│   │   └── correlation_analyzer.py ← Aggregation, IsolationForest, LR, correlation
│   │
│   ├── IsolationForest/            ← Custom Isolation Forest implementation (from scratch)
│   │   ├── config/                 ← File path configuration
│   │   ├── data/                   ← Dataset splitting utilities
│   │   ├── detection/              ← IsolationForest + IsolationTree
│   │   ├── evaluation/             ← Precision, recall, F1, confusion matrix
│   │   ├── models/                 ← Dataclasses (Observation, UserEntity, etc.)
│   │   ├── pipeline/               ← ThreatDetectionPipeline orchestrator
│   │   ├── preprocessing/          ← Feature aggregation & vectorisation
│   │   ├── training/               ← 100-iteration hyperparameter search
│   │   └── utils/                  ← CSV loader, pickler, c-factor
│   │
│   ├── LogisticRegression/         ← Logistic Regression implementation
│   │   ├── config/                 ← File path configuration
│   │   ├── detection/              ← LogisticRegressionClassifier (fit/predict/weights)
│   │   ├── evaluation/             ← Evaluator
│   │   ├── models/                 ← Training & evaluation result dataclasses
│   │   ├── pipeline/               ← ThreatDetectionPipeline
│   │   └── training/               ← Trainer + training logger
│   │
│   ├── RandomForest/               ← Alternative sklearn RF baseline
│   │   └── insider_threat_rf.py    ← 200-tree RF, 95.25% accuracy
│   │
│   └── main.py                     ← CLI entry point (manual testing)
│
├── templates/                      ← Jinja2 HTML templates
│   ├── base.html                   ← Sidebar layout, dataset banner, Bootstrap 5
│   ├── index.html                  ← Dashboard (stat cards, top anomalous users)
│   ├── correlation.html            ← Interactive Pearson/Spearman heatmap
│   ├── anomaly.html                ← Feature→anomaly correlation + scatter
│   ├── colluders.html              ← Cross-user correlation (collusion detection)
│   ├── pattern.html                ← LR feature weights + probability charts
│   ├── dataset.html                ← CSV upload, schema guide, feature distributions
│   └── model.html                  ← Model config, pre-trained files, training log
│
├── static/
│   └── css/
│       └── style.css               ← Custom UI stylesheet (sidebar, cards, charts)
│
├── dataset/
│   ├── insider_threat_clean_dataset.csv  ← 118,614 observations, 22 features
│   ├── training_data.csv                 ← 70% split
│   └── testing_data.csv                  ← 30% split
│
├── uploads/                        ← Custom CSVs uploaded through the UI (auto-created)
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

**Windows (recommended):**
```
Double-click run.bat
```

**Any terminal:**
```bash
cd shiny-chainsaw
python app.py
```

Then open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser.

> The first page load takes a few seconds — the system processes all raw observations, trains the Isolation Forest, and fits the Logistic Regression model at startup.

> **Important:** always launch from the project root directory (or use `run.bat`). Running `python` from a different directory causes Python to find a stale cached module and silently omit routes.

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
`late_exit_flag`, `entry_during_weekend`, `is_malicious` *(optional)*

### Custom CSV upload

The Dataset page accepts any CSV that contains the 21 required columns above (`is_malicious` is optional). Once uploaded:
- All six analysis pages automatically switch to the custom dataset
- A persistent banner on every page shows the active dataset name with a one-click reset
- Uploading runs the full preprocessing + Isolation Forest + Logistic Regression pipeline on the new data (typically 2–5 seconds)

---

## Data Preprocessing

Raw observations are **aggregated per unique user identity** (10 identity columns) into 17 behavioural features:

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

## AI Capability 1 — Anomaly Detection

**Algorithm:** Isolation Forest (scikit-learn, 200 estimators)

The Isolation Forest isolates anomalies by randomly selecting a feature and split value. Anomalous observations require fewer splits to isolate (shorter average path length through the trees). The anomaly score is the negated decision function — higher values indicate more anomalous users.

**Training configuration:**
- `n_estimators = 200`
- `contamination = 0.05` (5% of users expected to be anomalous)
- `max_samples = 'auto'`
- Features standardised with `StandardScaler` before fitting

**Output:**
- Anomaly score per user (continuous, higher = more anomalous)
- Used as input to the Anomaly Analysis page and as a reference signal in the Pattern Recognition page

---

## AI Capability 2 — Pattern Recognition

**Algorithm:** Logistic Regression (scikit-learn, `class_weight='balanced'`, `max_iter=1000`)

Where Isolation Forest is unsupervised (no labels required), Logistic Regression is a **supervised classifier** that learns an explicit mapping from the 17 aggregated features to the probability that a user is a malicious insider. Because LR is a linear model, every feature has a named coefficient that directly explains the decision — making results fully interpretable for forensic investigators.

**Training:**
- Features standardised with `StandardScaler`
- When ground-truth `is_malicious` labels exist: trains on real labels
- When dataset is unlabelled (custom CSV without `is_malicious`): top 5% of Isolation Forest anomaly scores are used as pseudo-labels so feature weights remain meaningful
- Solver: `lbfgs`, `random_state=42`

**Output per user:**
- Malicious probability (0–1 continuous)
- Binary prediction (threshold 0.5)
- Metrics: accuracy, precision, recall, F1

**Output for investigation:**
- Feature weights ranked by |coefficient| — tells investigators which metadata attributes are the strongest learned signatures of insider threat behaviour
- Positive weight = risk factor (pushes toward malicious), negative = protective

---

## AI Capability 3 — Correlation Analysis

Three complementary analyses reveal which metadata features matter most for forensic investigators.

### 3a. Feature–Feature Correlation Matrix

**Pearson r** (linear) and **Spearman ρ** (rank-order / non-linear) are computed between all pairs of the 17 aggregated features, producing a 17×17 matrix. Both methods are offered via a toggle on the UI.

Example finding: `avg_trip_duration` ↔ `abroad_ratio` — r = 0.966, strongly correlated since travelling users spend more time abroad.

### 3b. Feature–Anomaly Score Correlation

For each of the 17 features, Pearson r and Spearman ρ are computed against the Isolation Forest anomaly score. P-values are calculated and features with p < 0.05 are flagged as statistically significant predictors.

Top predictors on the default dataset:

| Rank | Feature | Pearson r | Significant |
|---|---|---|---|
| 1 | Files Burned (Others) | 0.564 | ✓ |
| 2 | Weekend Entry Ratio | 0.458 | ✓ |
| 3 | Off-hours Print Ratio | 0.376 | ✓ |
| 4 | Avg Access Entries | 0.352 | ✓ |
| 5 | Abroad Ratio | 0.342 | ✓ |

### 3c. Cross-User Behavioural Correlation (Collusion Detection)

For every pair of users, Pearson r is computed between their 17-feature profile vectors using `numpy.corrcoef` (vectorised, handles thousands of users efficiently). Pairs above a configurable threshold are flagged as potentially coordinating (colluding) insiders. A two-tailed t-test with df = n_features − 2 provides p-values.

Example finding at r ≥ 0.97: 1,673 flagged pairs, top pair r = 0.9978.

---

## Web Application

### Pages

| URL | Page | Description |
|---|---|---|
| `/` | Dashboard | Stat cards, top-5 anomalous users, anomaly score histogram |
| `/correlation` | Feature Correlation | Interactive Pearson/Spearman heatmap + strongest pairs table |
| `/anomaly` | Anomaly Analysis | Feature→anomaly bar chart, risk vs anomaly scatter, significance table |
| `/colluders` | Collusion Detection | Threshold slider, correlated user-pair table, distribution histogram |
| `/pattern` | Pattern Recognition | LR feature weights, LR prob vs IF scatter, top flagged users, metrics cards |
| `/dataset` | Dataset | CSV upload (drag-and-drop), schema guide, column schema, feature distributions, sample observations |
| `/model` | Model | Sklearn config, pre-trained pickle files, training log (last 20 runs), anomaly score distribution |

### API endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/correlation-matrix?method=pearson` | GET | 17×17 correlation matrix as JSON |
| `/api/anomaly-correlation` | GET | Feature→anomaly correlations sorted by \|r\| |
| `/api/anomaly-scores` | GET | Per-user anomaly scores, risk scores, labels, departments |
| `/api/colluders?threshold=0.85` | GET | Correlated user pairs above threshold |
| `/api/pattern-weights` | GET | LR feature coefficients ranked by \|weight\| |
| `/api/pattern-predictions` | GET | Per-user LR probabilities, IF scores, labels, metrics |
| `/api/dataset-stats` | GET | Per-feature mean/std/min/max + value distributions by label |
| `/api/model-info` | GET | Sklearn config, anomaly score stats, best training run from log |
| `/api/upload-csv` | POST | Upload a custom CSV; validates schema, runs full pipeline, returns stats |
| `/api/reset-dataset` | POST | Revert to the default dataset |

All endpoints automatically serve data from the currently active dataset (default or uploaded custom).

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

### Logistic Regression (`src/LogisticRegression/`)

A structured LR implementation mirroring the IsolationForest module:
- `LogisticRegressionClassifier` — wraps sklearn LR with `get_pattern_report()` (feature weights ranked by |coefficient|) and `explain_prediction()` (per-observation feature contributions)
- `ThreatDetectionPipeline` — reuses IsolationForest preprocessing and adds LR training/inference
- `Trainer` — hyperparameter search over max_iter, class_weight, random_state

The classifier is integrated into `CorrelationAnalyzer` so the Flask app can use it without the broken import chain in the pipeline.

### Random Forest Baseline (`src/RandomForest/insider_threat_rf.py`)

- 200-tree sklearn Random Forest, class-weight balanced
- 95.25% accuracy, 91.04% precision, 82.43% recall, 0.9896 ROC-AUC
- Includes feature importance analysis and risk indicator flags

---

## Key Results (default dataset)

| Metric | Value |
|---|---|
| Total users profiled | 1,654 |
| Malicious actors | 162 (9.8%) |
| Raw observations processed | 118,614 |
| Top IF anomaly predictor | Files Burned from Other Users (r = 0.564) |
| Top LR pattern weight | Files Burned from Other Users |
| Strongest feature–feature correlation | Trip Duration ↔ Abroad Ratio (r = 0.966) |
| Correlated pairs at r ≥ 0.97 | 1,673 |

---

## Dependencies

Key packages (see `requirements.txt` for pinned versions):

| Package | Purpose |
|---|---|
| `flask` | Web framework |
| `scikit-learn` | IsolationForest, LogisticRegression, StandardScaler, metrics |
| `scipy` | Pearson r, Spearman ρ, t-distribution p-values |
| `pandas` | Data loading and groupby aggregation |
| `numpy` | Vectorised matrix operations, corrcoef |

---

## Running with Make

```bash
make install   # pip install -r requirements.txt
make run       # python -m src.main  (CLI entry point)
```

To run the Flask app:
```bash
python app.py
# or on Windows: double-click run.bat
```
