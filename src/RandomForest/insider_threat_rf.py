"""
InsightGuard – Random Forest Insider Threat Detection Model
===========================================================
COS720 Computer Information & Security I | University of Pretoria | 2026

This file contains the complete Random Forest model implementation:
  - Full explanation of how the model works
  - Training code (to retrain from scratch)
  - Inference code (to use the pre-trained rf_model.pkl)
  - Integration examples for other use cases

Requirements:
    pip install scikit-learn pandas numpy joblib

Usage (pre-trained model):
    python insider_threat_rf.py --predict --input your_data.csv

Usage (retrain from scratch):
    python insider_threat_rf.py --train --data your_dataset.csv
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score,
    classification_report
)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE — HOW THE RANDOM FOREST WORKS
# ─────────────────────────────────────────────────────────────────────────────
#
# A Random Forest is an ensemble of decision trees. Here is exactly what
# happens when you call rf.predict() or rf.predict_proba():
#
# TRAINING (already done — stored in rf_model.pkl):
#
#   1. Bootstrap sampling:
#      For each of the 200 trees, a random sample of ~63% of the training
#      records is drawn WITH replacement (bootstrap sample). The remaining
#      ~37% are the "out-of-bag" samples used for internal validation.
#
#   2. Tree construction:
#      Each tree is grown by recursively splitting the data. At each node,
#      the algorithm considers a random subset of features (sqrt(10) ≈ 3
#      features at each split) and selects the split that maximises the
#      reduction in Gini impurity:
#
#        Gini(node) = 1 - Σ p(class_k)²
#
#      A pure node (all one class) has Gini = 0.
#      A maximally impure node (50/50 split) has Gini = 0.5.
#
#      Splitting continues until min_samples_leaf=2 is reached (no leaf
#      can contain fewer than 2 samples), preventing overfitting.
#
#   3. class_weight="balanced":
#      Because only ~18.5% of records are malicious, the model would
#      otherwise be biased toward predicting Normal for everything (still
#      81.5% accurate while being useless). class_weight="balanced"
#      automatically sets sample weights as:
#
#        weight(class) = n_total / (n_classes × n_samples_in_class)
#
#      So malicious records get weight ≈ 2.7× higher than normal records,
#      forcing the model to pay more attention to minority class errors.
#
# INFERENCE (what happens when you call predict on new data):
#
#   1. The input record is passed down all 200 trees simultaneously.
#
#   2. Each tree votes: each leaf node stores the class probability
#      observed in its training samples. For a new record, the tree
#      returns the probability vector [P(Normal), P(Malicious)] of the
#      leaf it lands in.
#
#   3. Averaging: the 200 probability vectors are averaged:
#        P_final(Malicious) = (1/200) × Σ P_tree_i(Malicious)
#
#   4. Decision: if P_final(Malicious) >= 0.5 → predict Malicious (1)
#                if P_final(Malicious) <  0.5 → predict Normal (0)
#
# FEATURE IMPORTANCES (from the trained model):
#
#   privilege_escalation : 0.2489  ← strongest predictor
#   file_access_count    : 0.1503
#   login_hour           : 0.1407
#   data_transfer_mb     : 0.1279
#   failed_logins        : 0.1036
#   email_external_count : 0.0994
#   access_frequency     : 0.0622
#   usb_usage            : 0.0236
#   remote_access        : 0.0221
#   department_encoded   : 0.0213
#
# PERFORMANCE (on 400-record held-out test set):
#
#   Accuracy  : 95.25%
#   Precision : 91.04%   (of those flagged malicious, 91% actually were)
#   Recall    : 82.43%   (of all actual malicious, 82% were caught)
#   F1-Score  : 86.52%
#   ROC-AUC   : 0.9896   (near-perfect class discrimination)
#   CV F1     : 84.04% ± 4.78% (5-fold, confirms no overfitting)
#
#   Confusion matrix (test set, n=400):
#     True  Negatives (correctly cleared normal)  : 320
#     False Positives (normal flagged as threat)  :   6
#     False Negatives (missed malicious insiders) :  13
#     True  Positives (correctly caught threats)  :  61
#
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    'login_hour',           # Hour of day employee logged in (0–23)
    'file_access_count',    # Number of files accessed in session
    'privilege_escalation', # Whether privilege escalation occurred (0/1)
    'failed_logins',        # Number of failed login attempts
    'data_transfer_mb',     # Volume of data transferred (megabytes)
    'usb_usage',            # Whether a USB device was used (0/1)
    'email_external_count', # Number of external emails sent
    'access_frequency',     # Number of system access events
    'remote_access',        # Whether session was remote (0/1)
    'department_encoded',   # Department as integer (see DEPT_MAP below)
]

DEPT_MAP = {
    'Finance':    0,
    'HR':         1,
    'IT':         2,
    'Legal':      3,
    'Operations': 4,
    'Sales':      5,
}

# Thresholds above which a feature value is considered abnormal
ABNORMAL_THRESHOLDS = {
    'login_hour':            (0, 6),   # hours outside 7–18 are abnormal
    'file_access_count':     80,
    'failed_logins':         5,
    'data_transfer_mb':      500,
    'email_external_count':  30,
    'access_frequency':      100,
}

# Hyperparameters used to train the saved model
RF_PARAMS = {
    'n_estimators':   200,
    'min_samples_leaf': 2,
    'class_weight':   'balanced',
    'random_state':   42,
    'n_jobs':         -1,
}


# ─────────────────────────────────────────────────────────────────────────────
# LOADING THE PRE-TRAINED MODEL
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path='rf_model.pkl'):
    """
    Load the pre-trained Random Forest from disk.

    The .pkl file contains a fully trained sklearn RandomForestClassifier
    object — 200 decision trees, all learned split thresholds, leaf
    probabilities, and feature importances. Loading it is instantaneous
    and does not require the original training data.

    Args:
        model_path (str): Path to rf_model.pkl

    Returns:
        sklearn.ensemble.RandomForestClassifier: ready to call predict()
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Either provide the rf_model.pkl file or run with --train to retrain."
        )
    rf = joblib.load(model_path)
    print(f"[OK] Loaded Random Forest: {rf.n_estimators} trees, "
          f"{rf.n_features_in_} features, classes={rf.classes_}")
    return rf


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(df):
    """
    Prepare a raw DataFrame for model input.

    Handles:
      - Department string encoding (Finance→0, HR→1, etc.)
      - Missing value imputation (median for numerics)
      - Column alignment to FEATURE_COLS order

    Args:
        df (pd.DataFrame): Raw input with behavioural columns.
                           Must have either 'department' (string) or
                           'department_encoded' (int) column.

    Returns:
        pd.DataFrame: Cleaned, encoded DataFrame with exactly FEATURE_COLS.
    """
    df = df.copy()

    # Encode department string to integer if needed
    if 'department' in df.columns and 'department_encoded' not in df.columns:
        df['department_encoded'] = df['department'].map(DEPT_MAP)
        unknown = df['department_encoded'].isna()
        if unknown.any():
            print(f"[WARN] Unknown department values set to 0 (Finance): "
                  f"{df.loc[unknown, 'department'].unique().tolist()}")
            df['department_encoded'] = df['department_encoded'].fillna(0)

    # Impute missing numeric values with column median
    for col in FEATURE_COLS:
        if col in df.columns and df[col].isna().any():
            median = df[col].median()
            df[col] = df[col].fillna(median)
            print(f"[INFO] Imputed {df[col].isna().sum()} missing values "
                  f"in '{col}' with median={median:.2f}")

    # Validate all required columns are present
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input is missing required columns: {missing}\n"
            f"Required: {FEATURE_COLS}"
        )

    return df[FEATURE_COLS].astype(float)


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def predict(rf, df_raw, threshold=0.5):
    """
    Classify one or more employee behaviour records.

    Args:
        rf: Loaded RandomForestClassifier
        df_raw (pd.DataFrame): Raw input records (see preprocess() for schema)
        threshold (float): Decision threshold. Default 0.5. Lower to 0.35
                           for high-risk departments to improve recall.

    Returns:
        pd.DataFrame with columns:
            - All original input columns
            - 'threat_probability': P(Malicious) from the RF (0.0–1.0)
            - 'classification': 'Malicious' or 'Normal'
            - 'confidence_pct': human-readable confidence percentage
            - 'risk_level': 'High' / 'Medium' / 'Low'
            - 'risk_indicators': list of triggered abnormal behaviour flags
    """
    X = preprocess(df_raw)

    # Core RF prediction:
    # predict_proba returns [[P(Normal), P(Malicious)], ...]
    # We take column index 1 = P(Malicious)
    proba = rf.predict_proba(X)[:, 1]

    # Apply threshold (default 0.5, can lower for higher-risk contexts)
    predictions = (proba >= threshold).astype(int)

    results = df_raw.copy()
    results['threat_probability'] = proba.round(4)
    results['classification'] = ['Malicious' if p == 1 else 'Normal'
                                 for p in predictions]
    results['confidence_pct'] = [
        f"{int(proba[i]*100)}%" if predictions[i] == 1
        else f"{int((1-proba[i])*100)}% Normal"
        for i in range(len(predictions))
    ]
    results['risk_level'] = pd.cut(
        proba,
        bins=[0, 0.3, 0.6, 1.0],
        labels=['Low', 'Medium', 'High'],
        include_lowest=True
    )
    results['risk_indicators'] = [
        _get_risk_indicators(X.iloc[i]) for i in range(len(X))
    ]

    return results


def _get_risk_indicators(row):
    """
    Return a list of human-readable strings describing abnormal
    feature values in a single record.
    """
    flags = []
    lh = int(row['login_hour'])
    if lh < 7 or lh > 18:
        flags.append(f"After-hours login (hour {lh})")
    if row['file_access_count'] > ABNORMAL_THRESHOLDS['file_access_count']:
        flags.append(f"High file access ({int(row['file_access_count'])} files)")
    if row['privilege_escalation'] == 1:
        flags.append("Privilege escalation detected")
    if row['failed_logins'] > ABNORMAL_THRESHOLDS['failed_logins']:
        flags.append(f"Excessive failed logins ({int(row['failed_logins'])})")
    if row['data_transfer_mb'] > ABNORMAL_THRESHOLDS['data_transfer_mb']:
        flags.append(f"High data transfer ({row['data_transfer_mb']:.0f} MB)")
    if row['usb_usage'] == 1:
        flags.append("USB device used")
    if row['email_external_count'] > ABNORMAL_THRESHOLDS['email_external_count']:
        flags.append(f"High external emails ({int(row['email_external_count'])})")
    if row['access_frequency'] > ABNORMAL_THRESHOLDS['access_frequency']:
        flags.append(f"Abnormal access frequency ({int(row['access_frequency'])})")
    if row['remote_access'] == 1:
        flags.append("Remote access session")
    return flags if flags else ["No abnormal indicators detected"]


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING FROM SCRATCH
# ─────────────────────────────────────────────────────────────────────────────

def train(data_path, save_path='rf_model.pkl'):
    """
    Train the Random Forest from scratch on a new dataset.

    This replicates the exact training procedure used to produce
    the saved rf_model.pkl. Use this if you have new data or want
    to retrain with different hyperparameters.

    Args:
        data_path (str): Path to CSV file with FEATURE_COLS + 'threat_label'
        save_path (str): Where to save the retrained model

    Returns:
        tuple: (trained RandomForestClassifier, metrics dict)
    """
    print(f"[INFO] Loading dataset: {data_path}")
    df = pd.read_csv(data_path)

    # Encode department if present as string
    if 'department' in df.columns and 'department_encoded' not in df.columns:
        le = LabelEncoder()
        df['department_encoded'] = le.fit_transform(df['department'])
        joblib.dump(le, 'label_encoder.pkl')
        print(f"[INFO] Saved label_encoder.pkl")

    X = df[FEATURE_COLS]
    y = df['threat_label']

    print(f"[INFO] Dataset: {len(df)} records | "
          f"Malicious: {y.mean():.1%} | Normal: {(1-y.mean()):.1%}")

    # Stratified 80/20 split — preserves class ratio in both partitions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[INFO] Train: {len(X_train)} | Test: {len(X_test)}")

    # ── Train Random Forest ───────────────────────────────────────────
    # n_estimators=200: 200 trees. More trees = more stable but slower.
    # min_samples_leaf=2: prevents single-sample leaves (overfitting).
    # class_weight="balanced": compensates for 81.5%/18.5% imbalance.
    # random_state=42: reproducibility.
    # n_jobs=-1: use all CPU cores for parallel tree building.
    print("[INFO] Training Random Forest (200 trees)...")
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train, y_train)

    # ── Evaluate ─────────────────────────────────────────────────────
    rf_pred  = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, rf_pred)
    prec = precision_score(y_test, rf_pred)
    rec  = recall_score(y_test, rf_pred)
    f1   = f1_score(y_test, rf_pred)
    auc  = roc_auc_score(y_test, rf_proba)
    cm   = confusion_matrix(y_test, rf_pred)
    tn, fp, fn, tp = cm.ravel()
    cv   = cross_val_score(rf, X, y, cv=5, scoring='f1')

    print(f"\n{'='*50}")
    print(f"RANDOM FOREST EVALUATION")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"  CV F1    : {cv.mean():.4f} ± {cv.std():.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={tn}  FP={fp}")
    print(f"    FN={fn}  TP={tp}")
    print(f"\n{classification_report(y_test, rf_pred, target_names=['Normal','Malicious'])}")

    metrics = {
        'accuracy': round(acc, 4), 'precision': round(prec, 4),
        'recall': round(rec, 4), 'f1': round(f1, 4), 'roc_auc': round(auc, 4),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'cv_f1_mean': round(cv.mean(), 4), 'cv_f1_std': round(cv.std(), 4),
        'feature_importances': dict(zip(
            FEATURE_COLS, rf.feature_importances_.round(4).tolist()
        ))
    }

    # Save model and metrics
    joblib.dump(rf, save_path)
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[OK] Model saved to: {save_path}")
    print(f"[OK] Metrics saved to: metrics.json")
    return rf, metrics


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION EXAMPLE — USE IN ANOTHER PROJECT
# ─────────────────────────────────────────────────────────────────────────────
#
# Minimal example to use this model in any other Python project:
#
#   from insider_threat_rf import load_model, predict
#   import pandas as pd
#
#   rf = load_model('rf_model.pkl')
#
#   # Single record
#   record = pd.DataFrame([{
#       'login_hour': 2,
#       'file_access_count': 250,
#       'privilege_escalation': 1,
#       'failed_logins': 2,
#       'data_transfer_mb': 4200.0,
#       'usb_usage': 1,
#       'email_external_count': 8,
#       'access_frequency': 45,
#       'remote_access': 1,
#       'department': 'Finance'   # string — will be encoded automatically
#   }])
#   results = predict(rf, record)
#   print(results[['classification','threat_probability','risk_indicators']])
#
#   # Batch from CSV
#   batch = pd.read_csv('employees.csv')
#   results = predict(rf, batch)
#   high_risk = results[results['risk_level'] == 'High']
#   results.to_csv('threat_results.csv', index=False)
#
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='InsightGuard Random Forest – Insider Threat Detection'
    )
    parser.add_argument('--train', action='store_true',
                        help='Train model from scratch')
    parser.add_argument('--predict', action='store_true',
                        help='Run inference using pre-trained model')
    parser.add_argument('--data',  default='insider_threat_dataset.csv',
                        help='Path to dataset CSV (for --train)')
    parser.add_argument('--input', default=None,
                        help='Path to input CSV (for --predict)')
    parser.add_argument('--model', default='rf_model.pkl',
                        help='Path to model file')
    parser.add_argument('--output', default='predictions.csv',
                        help='Path to save prediction results')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold (default 0.5, try 0.35 for high recall)')
    args = parser.parse_args()

    if args.train:
        train(args.data, save_path=args.model)

    elif args.predict:
        rf = load_model(args.model)
        if args.input:
            df = pd.read_csv(args.input)
        else:
            # Demo mode: create one normal and one malicious sample
            print("[INFO] No --input provided. Running demo with two sample records.")
            df = pd.DataFrame([
                {   # Clearly malicious
                    'login_hour': 2, 'file_access_count': 250,
                    'privilege_escalation': 1, 'failed_logins': 8,
                    'data_transfer_mb': 4200.0, 'usb_usage': 1,
                    'email_external_count': 55, 'access_frequency': 320,
                    'remote_access': 1, 'department': 'Finance'
                },
                {   # Clearly normal
                    'login_hour': 9, 'file_access_count': 15,
                    'privilege_escalation': 0, 'failed_logins': 0,
                    'data_transfer_mb': 45.0, 'usb_usage': 0,
                    'email_external_count': 3, 'access_frequency': 22,
                    'remote_access': 0, 'department': 'HR'
                }
            ])

        results = predict(rf, df, threshold=args.threshold)
        print("\n" + "="*60)
        print("PREDICTION RESULTS")
        print("="*60)
        for i, row in results.iterrows():
            print(f"\nRecord {i+1}:")
            print(f"  Classification   : {row['classification']}")
            print(f"  Threat Probability: {row['threat_probability']:.1%}")
            print(f"  Risk Level       : {row['risk_level']}")
            print(f"  Risk Indicators  : {row['risk_indicators']}")

        results.to_csv(args.output, index=False)
        print(f"\n[OK] Results saved to: {args.output}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
