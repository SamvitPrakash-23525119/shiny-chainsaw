# shiny-chainsaw
Digital Forensics Final Project 

# File Structure
```bash
.
├── dataset
│   ├── insider_threat_clean_dataset.csv
│   ├── testing_data.csv
│   └── training_data.csv
├── logs
│   └── training_log.jsonl
├── makefile
├── models
│   ├── balanced_model.pkl
│   ├── high_f1_model.pkl
│   ├── high_precision_model.pkl
│   ├── high_recall_model.pkl
│   └── last_trained_model.pkl
├── README.md
├── requirements.txt
└── src
    ├── __init__.py
    ├── IsolationForest
    │   ├── config
    │   │   ├── file_paths.py
    │   │   └── __init__.py
    │   ├── data
    │   │   ├── __init__.py
    │   │   └── split_dataset.py
    │   ├── detection
    │   │   ├── __init__.py
    │   │   ├── isolation_forest.py
    │   │   ├── isolation_tree_node.py
    │   │   └── isolation_tree.py
    │   ├── evaluation
    │   │   ├── evaluator.py
    │   │   └── __init__.py
    │   ├── __init__.py
    │   ├── models
    │   │   ├── aggregated_features.py
    │   │   ├── evaluation_result.py
    │   │   ├── identity_signature.py
    │   │   ├── __init__.py
    │   │   ├── observation.py
    │   │   ├── prediction.py
    │   │   ├── training_result.py
    │   │   └── user_entity.py
    │   ├── pipeline
    │   │   ├── __init__.py
    │   │   └── threat_detection_pipeline.py
    │   ├── preprocessing
    │   │   ├── feature_aggregator.py
    │   │   ├── identification_resolver.py
    │   │   ├── __init__.py
    │   │   ├── observation_builder.py
    │   │   └── vectorizer.py
    │   ├── training
    │   │   ├── __init__.py
    │   │   ├── trainer.py
    │   │   └── training_logger.py
    │   └── utils
    │       ├── c_factor.py
    │       ├── csv_loader.py
    │       ├── __init__.py
    │       ├── metrics.py
    │       └── pickler.py
    └── main.py
```

# How to add your code

1. Make a directory inside src, and insert your code there.
```bash
└── src
    ├── __init__.py
    ├── IsolationForest
    ├── <Your New Directory Name>
    └── main.py
```

2. Inside your new directory create a "__init__.py" file. If you have any sub-directories please create a new "__init__.py" file at the root for every sub-directory with executable code.
```bash
└── <Your New Directory Name>
    ├── __init__.py
    ├── <Sub-Directory>
    │   ├── __init__.py
    │   └── ...
    └── <Sub-Directory>
        ├── __init__.py
        └── <Sub-Directory>
            └── __init__.py
    
```

3. Import and test your code works inside main. Imports work by listing the file path to your code.
```python
from src.IsolationForest.pipeline import threat_detection_pipeline

def main():
    print("Your test code can go here.")


if __name__ == "__main__":
    main()
```

# How to run your code
Within the root directory of the project, you can run
```bash
make run

    or

python -m src.main
```
