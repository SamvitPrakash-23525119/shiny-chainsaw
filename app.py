import json
import logging
import time
from pathlib import Path

from flask import Flask, render_template, jsonify, request

from src.CorrelationAnalysis.correlation_analyzer import (
    CorrelationAnalyzer,
    FEATURE_NAMES,
    FEATURE_LABELS,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)
logging.getLogger('werkzeug').setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = Path(__file__).parent
DATA_PATH   = BASE_DIR / 'dataset' / 'insider_threat_clean_dataset.csv'
MODELS_DIR  = BASE_DIR / 'models'
LOG_PATH    = BASE_DIR / 'logs' / 'training_log.jsonl'
UPLOAD_DIR  = BASE_DIR / 'uploads'
UPLOAD_DIR.mkdir(exist_ok=True)

# The 21 columns an uploaded CSV must contain (is_malicious is optional)
REQUIRED_COLS = [
    'employee_department', 'employee_campus', 'employee_position',
    'employee_seniority_years', 'is_contractor', 'employee_classification',
    'has_foreign_citizenship', 'has_criminal_record', 'has_medical_history',
    'employee_origin_country', 'total_printed_pages', 'num_printed_pages_off_hours',
    'total_files_burned', 'burned_from_other', 'is_abroad', 'trip_day_number',
    'hostility_country_level', 'num_entries', 'num_unique_campus',
    'late_exit_flag', 'entry_during_weekend',
]

# ---------------------------------------------------------------------------
# Flask app + analyzer state
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = 'metaforensics-cos783'

_default_analyzer: CorrelationAnalyzer | None = None
_custom_analyzer:  CorrelationAnalyzer | None = None
_custom_filename:  str | None                 = None


def get_default_analyzer() -> CorrelationAnalyzer:
    global _default_analyzer
    if _default_analyzer is None:
        logger.info('Loading default dataset…')
        _default_analyzer = CorrelationAnalyzer(DATA_PATH)
        logger.info('Default dataset ready: %d users, %d observations',
                    len(_default_analyzer.df_users), len(_default_analyzer.df_raw))
    return _default_analyzer


def get_analyzer() -> CorrelationAnalyzer:
    """Return the custom analyzer if one has been uploaded, else the default."""
    return _custom_analyzer if _custom_analyzer is not None else get_default_analyzer()


@app.context_processor
def inject_dataset_banner():
    """Inject dataset-state variables into every template automatically."""
    is_custom = _custom_analyzer is not None
    name = _custom_filename if is_custom else DATA_PATH.name
    return {'is_custom_dataset': is_custom, 'active_dataset_name': name}


@app.errorhandler(Exception)
def handle_unhandled_exception(e):
    """Return JSON (not HTML) for any uncaught exception on /api/* routes."""
    from werkzeug.exceptions import HTTPException
    # Let 404 / 405 / etc. pass through as normal HTTP responses.
    if isinstance(e, HTTPException):
        return e
    if request.path.startswith('/api/'):
        logger.error('Unhandled exception on %s: %s', request.path, e, exc_info=True)
        return jsonify({'error': str(e)}), 500
    # Non-API page error — return a plain response (never re-raise inside a handler).
    return f'<h1>500 — Internal Server Error</h1><pre>{e}</pre>', 500


# ---------------------------------------------------------------------------
# Request logging
# ---------------------------------------------------------------------------
@app.before_request
def _start_timer():
    request._t0 = time.perf_counter()


@app.after_request
def _log_request(response):
    ms = (time.perf_counter() - request._t0) * 1000
    logger.info('%-6s %3d  %-45s  %.0fms',
                request.method, response.status_code, request.path, ms)
    return response


# ===========================================================================
# Page routes
# ===========================================================================

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


@app.route('/dataset')
def dataset():
    az = get_analyzer()
    df = az.df_raw
    columns = [
        {'name': c, 'dtype': str(df[c].dtype), 'nulls': int(df[c].isna().sum())}
        for c in df.columns
    ]
    return render_template(
        'dataset.html',
        active='dataset',
        n_rows=len(df),
        n_cols=len(df.columns),
        n_users=len(az.df_users),
        n_malicious=int(az.df_users['is_malicious'].sum()),
        columns=columns,
        sample=df.head(4).to_dict('records'),
        required_cols=REQUIRED_COLS,
    )


@app.route('/model')
def model():
    model_files = []
    if MODELS_DIR.exists():
        for f in sorted(MODELS_DIR.glob('*.pkl')):
            model_files.append({
                'name':    f.stem,
                'file':    f.name,
                'size_kb': round(f.stat().st_size / 1024, 1),
            })

    log_entries = []
    if LOG_PATH.exists():
        with open(LOG_PATH, 'r') as fh:
            lines = fh.readlines()
        for line in lines[-20:]:
            try:
                log_entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    return render_template('model.html', active='model',
                           model_files=model_files, log_entries=log_entries)


# ===========================================================================
# Upload / reset
# ===========================================================================

@app.route('/api/upload-csv', methods=['POST'])
def api_upload_csv():
    global _custom_analyzer, _custom_filename

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400

    f = request.files['file']
    if not f.filename:
        return jsonify({'error': 'No file selected.'}), 400
    if not f.filename.lower().endswith('.csv'):
        return jsonify({'error': 'Only CSV files are accepted.'}), 400

    # ── 1. Save to disk ──────────────────────────────────────────────────────
    save_path = UPLOAD_DIR / f.filename
    try:
        f.save(str(save_path))
        logger.info('Uploaded file saved: %s', save_path)
    except Exception as e:
        return jsonify({'error': f'Could not save uploaded file: {e}'}), 500

    # ── 2. Validate columns ──────────────────────────────────────────────────
    import pandas as pd
    try:
        df_check = pd.read_csv(save_path, nrows=1)
    except Exception as e:
        save_path.unlink(missing_ok=True)
        return jsonify({'error': f'Could not parse CSV: {e}'}), 400

    missing = [c for c in REQUIRED_COLS if c not in df_check.columns]
    if missing:
        save_path.unlink(missing_ok=True)
        return jsonify({
            'error': 'Missing required columns',
            'missing_columns': missing,
        }), 400

    # ── 3. Build analyzer + compute stats (all in one try block) ─────────────
    try:
        logger.info('Building CorrelationAnalyzer from uploaded file…')
        new_az = CorrelationAnalyzer(save_path)

        n_users = len(new_az.df_users)
        if n_users < 5:
            save_path.unlink(missing_ok=True)
            return jsonify({
                'error': f'Dataset too small: only {n_users} unique user(s) found. '
                         f'At least 5 are required for meaningful analysis.'
            }), 400

        logger.info('Upload analyzer ready: %d users, %d observations',
                    n_users, len(new_az.df_raw))
        stats = new_az.get_summary_stats()          # also inside the try
    except Exception as e:
        save_path.unlink(missing_ok=True)
        logger.error('Failed to build analyzer: %s', e, exc_info=True)
        return jsonify({'error': f'Analysis failed: {e}'}), 500

    # ── 4. Commit ────────────────────────────────────────────────────────────
    _custom_analyzer = new_az
    _custom_filename  = f.filename

    return jsonify({
        'success':        True,
        'filename':       f.filename,
        'n_users':        stats['n_users'],
        'n_malicious':    stats['n_malicious'],
        'n_obs':          stats['n_raw_observations'],
        'malicious_rate': stats['malicious_rate'],
    })


@app.route('/api/reset-dataset', methods=['POST'])
def api_reset_dataset():
    global _custom_analyzer, _custom_filename
    if _custom_filename:
        p = UPLOAD_DIR / _custom_filename
        p.unlink(missing_ok=True)
    _custom_analyzer = None
    _custom_filename  = None
    logger.info('Dataset reset to default.')
    return jsonify({'success': True})


# ===========================================================================
# JSON APIs — all use get_analyzer() so they automatically reflect uploads
# ===========================================================================

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
    return jsonify(get_analyzer().feature_anomaly_correlation())


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


@app.route('/api/dataset-stats')
def api_dataset_stats():
    az  = get_analyzer()
    df  = az.df_users[FEATURE_NAMES].astype(float)
    out = {}
    for col in FEATURE_NAMES:
        out[col] = {
            'label': FEATURE_LABELS[col],
            'mean':  round(float(df[col].mean()), 4),
            'std':   round(float(df[col].std()),  4),
            'min':   round(float(df[col].min()),  4),
            'max':   round(float(df[col].max()),  4),
            'values_normal':    [round(float(v), 4) for v in
                                 df[col][az.df_users['is_malicious'] == 0].tolist()],
            'values_malicious': [round(float(v), 4) for v in
                                 df[col][az.df_users['is_malicious'] == 1].tolist()],
        }
    return jsonify(out)


@app.route('/api/model-info')
def api_model_info():
    az = get_analyzer()
    best, all_entries = None, []
    if LOG_PATH.exists():
        with open(LOG_PATH, 'r') as fh:
            for line in fh:
                try:
                    entry = json.loads(line)
                    all_entries.append(entry)
                    em = entry.get('evaluation_metrics', {})
                    if best is None or em.get('recall', 0) > \
                            best.get('evaluation_metrics', {}).get('recall', 0):
                        best = entry
                except json.JSONDecodeError:
                    pass

    return jsonify({
        'sklearn_config': {
            'n_estimators': 200, 'contamination': 0.05,
            'max_samples': 'auto', 'random_state': 42,
        },
        'score_stats': {
            'min':  round(float(az.anomaly_scores.min()),  4),
            'max':  round(float(az.anomaly_scores.max()),  4),
            'mean': round(float(az.anomaly_scores.mean()), 4),
            'std':  round(float(az.anomaly_scores.std()),  4),
        },
        'n_users':         len(az.df_users),
        'n_malicious':     int(az.df_users['is_malicious'].sum()),
        'best_custom_run': best,
        'total_log_runs':  len(all_entries),
    })


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == '__main__':
    logger.info('=' * 62)
    logger.info('  MetaForensics — Digital Forensics Analysis Platform')
    logger.info('  http://127.0.0.1:5000')
    logger.info('  Every HTTP request is logged below with method, status, path, ms')
    logger.info('=' * 62)
    app.run(debug=True, port=5000, use_reloader=False)
