import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

IDENTITY_COLS = [
    'employee_department', 'employee_campus', 'employee_position',
    'employee_seniority_years', 'is_contractor', 'employee_classification',
    'has_foreign_citizenship', 'has_criminal_record', 'has_medical_history',
    'employee_origin_country',
]

FEATURE_NAMES = [
    'employee_seniority_years', 'is_contractor', 'has_foreign_citizenship',
    'has_criminal_record', 'has_medical_history', 'avg_total_printed_pages',
    'avg_off_hours_print_ratio', 'avg_files_burned', 'avg_burned_from_other',
    'avg_num_entries', 'avg_unique_campus_ratio', 'late_exit_ratio',
    'weekend_entry_ratio', 'avg_trip_duration', 'avg_hostility_country_level',
    'abroad_ratio', 'risk_score',
]

FEATURE_LABELS = {
    'employee_seniority_years':   'Seniority (yrs)',
    'is_contractor':              'Is Contractor',
    'has_foreign_citizenship':    'Foreign Citizenship',
    'has_criminal_record':        'Criminal Record',
    'has_medical_history':        'Medical History',
    'avg_total_printed_pages':    'Avg Pages Printed',
    'avg_off_hours_print_ratio':  'Off-hours Print Ratio',
    'avg_files_burned':           'Avg Files Burned',
    'avg_burned_from_other':      'Files Burned (Others)',
    'avg_num_entries':            'Avg Access Entries',
    'avg_unique_campus_ratio':    'Unique Campus Ratio',
    'late_exit_ratio':            'Late Exit Ratio',
    'weekend_entry_ratio':        'Weekend Entry Ratio',
    'avg_trip_duration':          'Avg Trip Duration',
    'avg_hostility_country_level':'Hostility Level',
    'abroad_ratio':               'Abroad Ratio',
    'risk_score':                 'Risk Score',
}

# Weights from the original feature_aggregator.py
_CRIMINAL_W   = 1.5
_CONTRACTOR_W = 0.5
_FOREIGN_W    = 1.0
_OFFPRINT_W   = 2.0


class CorrelationAnalyzer:
    """Aggregates raw observations per user and runs correlation analysis.

    Mirrors the preprocessing logic in IsolationForest/preprocessing/ so the
    Flask layer can operate independently of that module's broken import chain.
    Anomaly scores are produced by scikit-learn's IsolationForest so no
    pre-trained pickle is required at runtime.
    """

    def __init__(self, data_path: str):
        self.data_path = str(data_path)
        self.df_raw = pd.read_csv(data_path)
        self.df_users = self._aggregate_users()
        self.anomaly_scores = self._compute_anomaly_scores()

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _aggregate_users(self) -> pd.DataFrame:
        """Group raw observations by identity and compute per-user features."""
        df = self.df_raw.copy()
        groups = df.groupby(IDENTITY_COLS, sort=False)

        records = []
        for key, group in groups:
            if not isinstance(key, tuple):
                key = (key,)

            rec = dict(zip(IDENTITY_COLS, key))

            total_pages    = group['total_printed_pages'].sum()
            offhours_pages = group['num_printed_pages_off_hours'].sum()

            rec['avg_total_printed_pages']   = float(group['total_printed_pages'].mean())
            rec['avg_off_hours_print_ratio']  = float(offhours_pages / total_pages) if total_pages > 0 else 0.0
            rec['avg_files_burned']           = float(group['total_files_burned'].mean())
            rec['avg_burned_from_other']      = float(group['burned_from_other'].mean())
            rec['avg_num_entries']            = float(group['num_entries'].mean())

            campus_ratios = [
                row['num_unique_campus'] / row['num_entries']
                if row['num_entries'] > 0 else 0.0
                for _, row in group.iterrows()
            ]
            rec['avg_unique_campus_ratio']    = float(np.mean(campus_ratios))
            rec['late_exit_ratio']            = float(group['late_exit_flag'].mean())
            rec['weekend_entry_ratio']        = float(group['entry_during_weekend'].mean())
            rec['avg_trip_duration']          = float(group['trip_day_number'].mean())
            rec['avg_hostility_country_level']= float(group['hostility_country_level'].mean())
            rec['abroad_ratio']               = float(group['is_abroad'].mean())

            # Risk score — mirrors FeatureAggregator weights exactly
            risk = 0.0
            if bool(int(group['has_criminal_record'].iloc[0])):
                risk += _CRIMINAL_W
            if bool(int(group['is_contractor'].iloc[0])):
                risk += _CONTRACTOR_W
            if bool(int(group['has_foreign_citizenship'].iloc[0])) and rec['avg_hostility_country_level'] > 0:
                risk += _FOREIGN_W
            risk += rec['avg_off_hours_print_ratio'] * _OFFPRINT_W
            risk += float(np.log1p(rec['avg_files_burned']))
            rec['risk_score'] = risk

            rec['is_malicious']    = int(group['is_malicious'].max())
            rec['n_observations']  = len(group)

            records.append(rec)

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Anomaly scoring
    # ------------------------------------------------------------------

    def _compute_anomaly_scores(self) -> np.ndarray:
        """Score each user with sklearn IsolationForest (higher = more anomalous)."""
        X = self.df_users[FEATURE_NAMES].values.astype(float)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = IsolationForest(
            n_estimators=200,
            contamination=0.05,
            random_state=42,
            max_samples='auto',
        )
        clf.fit(X_scaled)

        # Negate decision_function: normal = positive → anomalous = larger value
        return -clf.decision_function(X_scaled)

    # ------------------------------------------------------------------
    # Correlation analysis methods
    # ------------------------------------------------------------------

    def feature_correlation_matrix(self, method: str = 'pearson') -> pd.DataFrame:
        """Return the (17 × 17) feature correlation matrix (NaN → 0 for zero-var features)."""
        return self.df_users[FEATURE_NAMES].astype(float).corr(method=method).fillna(0.0)

    def feature_anomaly_correlation(self) -> list:
        """Correlate each feature against the anomaly score.

        Returns a list of dicts sorted by |Pearson r| descending, each with
        pearson_r, pearson_p, spearman_rho, spearman_p, and a significance flag.
        """
        features = self.df_users[FEATURE_NAMES].astype(float)
        scores   = self.anomaly_scores

        results = []
        for col in FEATURE_NAMES:
            vals = features[col].values
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                r,   p_r   = stats.pearsonr(vals, scores)
                rho, p_rho = stats.spearmanr(vals, scores)
            # Zero-variance features produce NaN — treat as no correlation
            r   = float(r)   if not np.isnan(r)   else 0.0
            p_r = float(p_r) if not np.isnan(p_r) else 1.0
            rho = float(rho) if not np.isnan(rho) else 0.0
            p_rho = float(p_rho) if not np.isnan(p_rho) else 1.0
            results.append({
                'feature':      col,
                'label':        FEATURE_LABELS[col],
                'pearson_r':    round(r,    4),
                'pearson_p':    round(p_r,  4),
                'spearman_rho': round(rho,  4),
                'spearman_p':   round(p_rho,4),
                'significant':  bool(p_r < 0.05),
            })

        return sorted(results, key=lambda x: abs(x['pearson_r']), reverse=True)

    def find_correlated_users(self, threshold: float = 0.85) -> list:
        """Find user pairs whose 17-feature profiles are highly correlated.

        Computes all pairwise Pearson correlations using numpy's corrcoef
        (O(N² × F) but vectorised) then filters by threshold.  P-values use
        a t-distribution with df = n_features − 2.

        Args:
            threshold: Minimum |r| to include a pair (default 0.85).

        Returns:
            List of pair dicts sorted by |correlation| descending.
        """
        X = self.df_users[FEATURE_NAMES].values.astype(float)
        scaler  = StandardScaler()
        X_norm  = scaler.fit_transform(X)
        n_users, n_feats = X_norm.shape

        # Full (n_users × n_users) correlation matrix — fast numpy path
        corr_matrix = np.corrcoef(X_norm)

        df_t = n_feats - 2  # degrees of freedom for p-value computation
        pairs = []

        for i in range(n_users):
            for j in range(i + 1, n_users):
                r = float(corr_matrix[i, j])
                if abs(r) < threshold:
                    continue

                # Compute two-tailed p-value via t-distribution
                r_clamped = min(max(r, -0.9999999), 0.9999999)
                t_stat = r_clamped * np.sqrt(df_t / (1.0 - r_clamped ** 2))
                p_val  = float(2.0 * stats.t.sf(abs(t_stat), df_t))

                row_a = self.df_users.iloc[i]
                row_b = self.df_users.iloc[j]

                pairs.append({
                    'user_a':           f"{row_a['employee_position']} @ {row_a['employee_department']}",
                    'user_b':           f"{row_b['employee_position']} @ {row_b['employee_department']}",
                    'user_a_dept':      str(row_a['employee_department']),
                    'user_b_dept':      str(row_b['employee_department']),
                    'user_a_anomaly':   round(float(self.anomaly_scores[i]), 4),
                    'user_b_anomaly':   round(float(self.anomaly_scores[j]), 4),
                    'user_a_malicious': int(row_a['is_malicious']),
                    'user_b_malicious': int(row_b['is_malicious']),
                    'correlation':      round(r, 4),
                    'p_value':          round(p_val, 6),
                    'risk_flag':        bool(row_a['is_malicious'] or row_b['is_malicious']),
                })

        return sorted(pairs, key=lambda x: abs(x['correlation']), reverse=True)

    def get_summary_stats(self) -> dict:
        """Return headline statistics for the dashboard."""
        n_users    = len(self.df_users)
        n_malicious = int(self.df_users['is_malicious'].sum())
        n_normal   = n_users - n_malicious

        # Top 5 most anomalous users
        top_idxs = np.argsort(self.anomaly_scores)[-5:][::-1]
        top_users = []
        for idx in top_idxs:
            row = self.df_users.iloc[idx]
            top_users.append({
                'department':   str(row['employee_department']),
                'position':     str(row['employee_position']),
                'anomaly_score':round(float(self.anomaly_scores[idx]), 4),
                'is_malicious': int(row['is_malicious']),
                'risk_score':   round(float(row['risk_score']), 4),
            })

        # Strongest off-diagonal feature-feature correlation (ignore NaN / zero-var)
        corr   = self.feature_correlation_matrix('pearson')
        vals   = corr.values.copy()
        np.fill_diagonal(vals, 0.0)
        vals   = np.nan_to_num(vals, nan=0.0)
        ia, ib = np.unravel_index(np.argmax(np.abs(vals)), vals.shape)
        top_pair = {
            'feature_a':   FEATURE_LABELS[FEATURE_NAMES[ia]],
            'feature_b':   FEATURE_LABELS[FEATURE_NAMES[ib]],
            'correlation': round(float(vals[ia, ib]), 4),
        }

        return {
            'n_users':            n_users,
            'n_malicious':        n_malicious,
            'n_normal':           n_normal,
            'malicious_rate':     round(n_malicious / n_users * 100, 1) if n_users else 0,
            'avg_anomaly_score':  round(float(self.anomaly_scores.mean()), 4),
            'top_users':          top_users,
            'top_correlation_pair': top_pair,
            'n_raw_observations': len(self.df_raw),
        }
