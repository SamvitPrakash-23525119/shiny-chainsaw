from flask import Flask, render_template, jsonify, request
import numpy as np
from pathlib import Path
from src.CorrelationAnalysis.correlation_analyzer import (
    CorrelationAnalyzer,
    FEATURE_NAMES,
    FEATURE_LABELS,
)

app = Flask(__name__)

DATA_PATH = Path(__file__).parent / 'dataset' / 'insider_threat_clean_dataset.csv'

_analyzer: CorrelationAnalyzer | None = None


def get_analyzer() -> CorrelationAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = CorrelationAnalyzer(DATA_PATH)
    return _analyzer


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    az    = get_analyzer()
    stats = az.get_summary_stats()
    return render_template('index.html', stats=stats, active='dashboard')


@app.route('/correlation')
def correlation():
    return render_template('correlation.html', active='correlation')


@app.route('/anomaly')
def anomaly():
    return render_template('anomaly.html', active='anomaly')


@app.route('/colluders')
def colluders():
    return render_template('colluders.html', active='colluders')


# ---------------------------------------------------------------------------
# JSON APIs (consumed by Plotly.js in the browser)
# ---------------------------------------------------------------------------

@app.route('/api/correlation-matrix')
def api_correlation_matrix():
    az     = get_analyzer()
    method = request.args.get('method', 'pearson')
    corr   = az.feature_correlation_matrix(method=method)

    labels = [FEATURE_LABELS[c] for c in FEATURE_NAMES]
    z      = [[round(v, 4) for v in row] for row in corr.values.tolist()]

    return jsonify({'z': z, 'labels': labels, 'method': method})


@app.route('/api/anomaly-correlation')
def api_anomaly_correlation():
    az = get_analyzer()
    return jsonify(az.feature_anomaly_correlation())


@app.route('/api/anomaly-scores')
def api_anomaly_scores():
    az = get_analyzer()
    return jsonify({
        'scores':       [round(float(s), 4) for s in az.anomaly_scores],
        'is_malicious': az.df_users['is_malicious'].tolist(),
        'departments':  az.df_users['employee_department'].tolist(),
        'positions':    az.df_users['employee_position'].tolist(),
        'risk_scores':  [round(float(r), 4) for r in az.df_users['risk_score']],
    })


@app.route('/api/colluders')
def api_colluders():
    az        = get_analyzer()
    threshold = float(request.args.get('threshold', 0.85))
    pairs     = az.find_correlated_users(threshold)
    return jsonify({'pairs': pairs[:300], 'total': len(pairs)})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
