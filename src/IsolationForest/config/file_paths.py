from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Datasets
DATA_DIR = PROJECT_ROOT / "dataset"
CLEANED_DATA_PATH = DATA_DIR / "insider_threat_clean_dataset.csv"
TRAINING_DATA_PATH = DATA_DIR / "training_data.csv" 
TESTING_DATA_PATH = DATA_DIR / "testing_data.csv"

# Pickles
MODEL_DIR = PROJECT_ROOT / "models"
CURRENT_BEST_MODEL_PATH = MODEL_DIR / "current_best_model.pkl"

BEST_MODEL_PATH = MODEL_DIR / "last_trained_model.pkl"
BALANCED_MODEL_PATH = MODEL_DIR / "balanced_model.pkl"
HIGH_F1_MODEL_PATH = MODEL_DIR / "high_f1_model.pkl"
HIGH_PRECISION_MODEL_PATH = MODEL_DIR / "high_precision_model.pkl"
HIGH_RECALL_MODEL_PATH = MODEL_DIR / "high_recall_model.pkl"

# Training Logs
TRAINING_LOG_DIR = PROJECT_ROOT / "logs"
TRAINING_LOG_PATH = TRAINING_LOG_DIR / "training_log.jsonl"

# CSV Training Storage
TRAINING_CSV_DIR = PROJECT_ROOT / "src" / "api" / "storage" / "training"    

# CSV Detection Storage
DETECTION_CSV_DIR = PROJECT_ROOT / "src" / "api" / "storage" / "detection"

def model_finder(model_name):
    """Return the path for a named model.

    Args:
        model_name (str): Logical name of the model. Expected values include
            "custom", "Balanced", "High_F1", "High_Precision", and
            "High_Recall".

    Returns:
        pathlib.Path | None: Path to the model file if the name is known,
            otherwise ``None``.
    """
    model_paths = {
        "custom": BEST_MODEL_PATH,
        "Balanced": BALANCED_MODEL_PATH,
        "High_F1": HIGH_F1_MODEL_PATH,
        "High_Precision": HIGH_PRECISION_MODEL_PATH,
        "High_Recall": HIGH_RECALL_MODEL_PATH
    }
    return model_paths.get(model_name, None)