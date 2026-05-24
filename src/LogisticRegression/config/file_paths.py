from pathlib import Path

# Project Root (4 parents up from src/LogisticRegression/config/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Datasets
DATA_DIR = PROJECT_ROOT / "dataset"
CLEANED_DATA_PATH = DATA_DIR / "insider_threat_clean_dataset.csv"
TRAINING_DATA_PATH = DATA_DIR / "training_data.csv"
TESTING_DATA_PATH = DATA_DIR / "testing_data.csv"

# Pickles
MODEL_DIR = PROJECT_ROOT / "models"
LR_MODEL_PATH = MODEL_DIR / "logistic_regression_model.pkl"

# Training Logs
TRAINING_LOG_DIR = PROJECT_ROOT / "logs"
LR_TRAINING_LOG_PATH = TRAINING_LOG_DIR / "lr_training_log.jsonl"
